#!/usr/bin/env python3
"""Build a label-blind MoLFormer feature matrix for the frozen ODTI drugs.

The output is aligned exactly to ``drug_feature_index``.  Molecules that cannot
be parsed or exceed the model's audited 202-token pretraining envelope are
explicitly masked instead of being silently truncated.  No activity labels or
benchmark split columns are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DRUG_INDEX = (
    ROOT
    / "outputs/old_drug_target_sota_v1/feature_store_v1/DRUG_FEATURE_INDEX_V1.csv.gz"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_pretrained_features_v1/molformer_xl_both_10pct"
MODEL_ID = "ibm/MoLFormer-XL-both-10pct"
MODEL_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_nonisomeric_smiles(value: str) -> tuple[str | None, str]:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None, "RDKIT_PARSE_FAILURE"
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False), "AVAILABLE"


def validate_dense_index(index: pd.DataFrame) -> None:
    required = {"drug_feature_index", "model_ligand_smiles"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"drug index is missing columns: {sorted(missing)}")
    actual = index["drug_feature_index"].to_numpy(dtype=np.int64)
    expected = np.arange(len(index), dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError("drug_feature_index must be dense and sorted")


def build(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.drug_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / args.feature_name
    index_path = output_dir / args.index_name
    manifest_path = output_dir / args.manifest_name
    for path in (feature_path, index_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"artifact already exists: {path}; use --force")
    if not source.is_file():
        raise FileNotFoundError(source)

    index = pd.read_csv(source, low_memory=False).sort_values(
        "drug_feature_index", kind="stable"
    ).reset_index(drop=True)
    validate_dense_index(index)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    model_id = str(args.model_id)
    revision = str(args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )

    rows: list[dict[str, object]] = []
    candidates: list[tuple[int, str, int]] = []
    base_available = (
        index["feature_available"].fillna(False).astype(bool).to_numpy()
        if "feature_available" in index.columns
        else np.ones(len(index), dtype=bool)
    )
    for row in index.itertuples(index=False):
        feature_index = int(row.drug_feature_index)
        raw_smiles = str(row.model_ligand_smiles)
        canonical, status = canonical_nonisomeric_smiles(raw_smiles)
        token_length = 0
        if not base_available[feature_index]:
            status = "BASE_FEATURE_QUARANTINED"
            canonical = None
        elif canonical is not None:
            token_length = len(
                tokenizer(
                    canonical,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            if token_length > int(args.max_token_length):
                status = "OUTSIDE_PRETRAINING_TOKEN_ENVELOPE"
            else:
                candidates.append((feature_index, canonical, token_length))
        rows.append(
            {
                "drug_feature_index": feature_index,
                "model_ligand_smiles": raw_smiles,
                "molformer_input_smiles": canonical or "",
                "molformer_token_length": token_length,
                "feature_available": status == "AVAILABLE",
                "feature_route": status,
                "embedding_l2_norm": 0.0,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        deterministic_eval=True,
        trust_remote_code=True,
    ).to(device).eval()
    hidden_size = int(getattr(model.config, "hidden_size", args.feature_dim))
    if hidden_size != int(args.feature_dim):
        raise RuntimeError(f"model hidden size {hidden_size} != expected {args.feature_dim}")

    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(index), hidden_size),
    )
    features[:] = 0.0
    # Length bucketing reduces padding without changing feature-index alignment.
    candidates.sort(key=lambda item: (item[2], item[0]))
    completed = 0
    for start in range(0, len(candidates), int(args.batch_size)):
        batch = candidates[start : start + int(args.batch_size)]
        batch_indices = [item[0] for item in batch]
        batch_smiles = [item[1] for item in batch]
        inputs = tokenizer(
            batch_smiles,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if int(inputs["input_ids"].shape[1]) > int(args.max_token_length):
            raise RuntimeError("length-bucketed batch exceeded the audited token envelope")
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        pooled = output.pooler_output.float().cpu().numpy()
        if pooled.shape != (len(batch), hidden_size) or not np.isfinite(pooled).all():
            raise RuntimeError(f"invalid MoLFormer output shape/values: {pooled.shape}")
        features[np.asarray(batch_indices, dtype=np.int64)] = pooled
        norms = np.linalg.norm(pooled, axis=1)
        for feature_index, norm in zip(batch_indices, norms, strict=True):
            rows[feature_index]["embedding_l2_norm"] = float(norm)
        completed += len(batch)
        if args.progress_every > 0 and (
            completed == len(candidates) or completed // args.progress_every
            != (completed - len(batch)) // args.progress_every
        ):
            print(f"encoded {completed}/{len(candidates)} available drugs", flush=True)
    features.flush()

    feature_index = pd.DataFrame(rows).sort_values("drug_feature_index", kind="stable")
    available = feature_index["feature_available"].to_numpy(dtype=bool)
    finite = bool(np.isfinite(np.asarray(features)).all())
    checks = {
        "source_index_dense": feature_index["drug_feature_index"].tolist()
        == list(range(len(feature_index))),
        "shape_exact": list(features.shape) == [len(index), hidden_size],
        "all_values_finite": finite,
        "available_vectors_nonzero": bool(
            (feature_index.loc[available, "embedding_l2_norm"] > 0).all()
        ),
        "unavailable_vectors_exact_zero": bool(
            np.count_nonzero(np.asarray(features[~available])) == 0
        ),
        "no_silent_truncation": bool(
            (
                feature_index.loc[available, "molformer_token_length"]
                <= int(args.max_token_length)
            ).all()
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    feature_index.to_csv(index_path, index=False, compression="gzip")
    try:
        from huggingface_hub import hf_hub_download

        weight_path = Path(
            hf_hub_download(model_id, "model.safetensors", revision=revision)
        )
        weight_sha256 = sha256(weight_path)
    except Exception:
        weight_path = Path("")
        weight_sha256 = None
    route_counts = {
        str(key): int(value)
        for key, value in feature_index["feature_route"].value_counts().items()
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "label-blind frozen ODTI drug entities",
        "model": {
            "id": model_id,
            "revision": revision,
            "architecture": type(model).__name__,
            "pooling": "model_pooler_output",
            "input_normalization": "RDKit canonical non-isomeric SMILES",
            "max_token_length": int(args.max_token_length),
            "overlength_policy": "explicit unavailable mask; never truncate",
            "weight_path": str(weight_path) if weight_sha256 else None,
            "weight_sha256": weight_sha256,
        },
        "counts": {
            "drugs": int(len(index)),
            "available": int(available.sum()),
            "unavailable": int((~available).sum()),
            "routes": route_counts,
        },
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "checks": checks,
        "inputs": {str(source): sha256(source)},
        "outputs": {
            str(feature_path): sha256(feature_path),
            str(index_path): sha256(index_path),
        },
        "label_dependency": "NONE",
        "device": str(device),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if manifest["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-index", default=str(DRUG_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-name", default="MOLFORMER_XL_768_FLOAT32_V1.npy")
    parser.add_argument("--index-name", default="MOLFORMER_XL_DRUG_INDEX_V1.csv.gz")
    parser.add_argument("--manifest-name", default="MOLFORMER_XL_FEATURE_MANIFEST_V1.json")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--max-token-length", type=int, default=202)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()

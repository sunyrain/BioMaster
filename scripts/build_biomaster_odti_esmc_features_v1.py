#!/usr/bin/env python3
"""Build pooled ESM-C 600M features for the frozen ODTI target index.

Long proteins are encoded in overlapping windows under ESM-C's 2048-token
context limit.  Overlapping residue representations are averaged before a
length-uniform whole-protein mean, so overlap does not up-weight residues.
No interaction labels, assay fields, or benchmark roles are read.

This script is intended to run in the isolated Biohub ESM environment created
for the pinned model implementation; see the manifest for exact revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TARGET_INDEX = (
    ROOT
    / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_pretrained_features_v1/esmc_600m"
MODEL_ID = "biohub/ESMC-600M"
MODEL_REVISION = "a7e82012c83126b9eedb055fea9fa84b6c02f094"
ESM_IMPLEMENTATION_REVISION = "26b0bc2b771e3e419ea74f445a5f35cc094a1509"
TRANSFORMERS_IMPLEMENTATION_REVISION = "ef32577f55da19a4989cd7b22e004dc43a4998cb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_bounds(length: int, max_length: int, overlap: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if overlap < 0 or overlap >= max_length:
        raise ValueError("overlap must satisfy 0 <= overlap < max_length")
    if length <= max_length:
        return [(0, length)]
    step = max_length - overlap
    starts = list(range(0, length - max_length + 1, step))
    final_start = length - max_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(length, start + max_length)) for start in starts]


def direct_url(package: str) -> dict[str, object] | None:
    try:
        text = importlib.metadata.distribution(package).read_text("direct_url.json")
        return json.loads(text) if text else None
    except Exception:
        return None


def validate_dense_index(index: pd.DataFrame) -> None:
    required = {"target_feature_index", "sequence_key", "protein_sequence"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"target index is missing columns: {sorted(missing)}")
    actual = index["target_feature_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, np.arange(len(index), dtype=np.int64)):
        raise ValueError("target_feature_index must be dense and sorted")


def build(args: argparse.Namespace) -> dict[str, object]:
    source = Path(args.target_index)
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
        "target_feature_index", kind="stable"
    ).reset_index(drop=True)
    validate_dense_index(index)
    sequences = index["protein_sequence"].astype(str).str.upper().tolist()
    if any(not sequence for sequence in sequences):
        raise ValueError("target index contains an empty protein sequence")
    if "sequence_length" in index.columns and not np.array_equal(
        index["sequence_length"].to_numpy(dtype=np.int64),
        np.asarray([len(sequence) for sequence in sequences], dtype=np.int64),
    ):
        raise ValueError("protein_sequence and sequence_length disagree")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
    )
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()

    probe = tokenizer("ACDE", add_special_tokens=True, return_tensors="pt")
    prefix_special_tokens = 1
    suffix_special_tokens = int(probe["input_ids"].shape[1]) - 4 - prefix_special_tokens
    if prefix_special_tokens != 1 or suffix_special_tokens < 1:
        raise RuntimeError("unexpected ESM-C special-token layout")
    feature_dim = int(args.feature_dim)
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(index), feature_dim),
    )
    rows: list[dict[str, object]] = []
    unknown_token_id = tokenizer.unk_token_id
    for target_row, sequence in enumerate(sequences):
        bounds = window_bounds(
            len(sequence), int(args.max_window_length), int(args.window_overlap)
        )
        residue_sum = np.zeros((len(sequence), feature_dim), dtype=np.float32)
        residue_count = np.zeros(len(sequence), dtype=np.float32)
        unknown_residues = 0
        for start in range(0, len(bounds), int(args.batch_size)):
            batch_bounds = bounds[start : start + int(args.batch_size)]
            batch_sequences = [sequence[left:right] for left, right in batch_bounds]
            inputs = tokenizer(
                batch_sequences,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            if int(inputs["input_ids"].shape[1]) > int(args.max_context_tokens):
                raise RuntimeError("ESM-C batch exceeded the model context limit")
            if unknown_token_id is not None:
                unknown_residues += int((inputs["input_ids"] == unknown_token_id).sum())
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            hidden = output.last_hidden_state.float().cpu().numpy()
            if hidden.shape[2] != feature_dim or not np.isfinite(hidden).all():
                raise RuntimeError(f"invalid ESM-C hidden states: {hidden.shape}")
            for local, (left, right) in enumerate(batch_bounds):
                token_count = right - left
                values = hidden[
                    local,
                    prefix_special_tokens : prefix_special_tokens + token_count,
                ]
                if values.shape != (token_count, feature_dim):
                    raise RuntimeError(
                        f"unexpected ESM-C residue shape for target {target_row}: {values.shape}"
                    )
                residue_sum[left:right] += values
                residue_count[left:right] += 1.0
        if not np.all(residue_count > 0):
            raise RuntimeError(f"incomplete ESM-C window coverage for target {target_row}")
        stitched = residue_sum / residue_count[:, None]
        pooled = stitched.mean(axis=0, dtype=np.float64).astype(np.float32)
        if pooled.shape != (feature_dim,) or not np.isfinite(pooled).all():
            raise RuntimeError(f"invalid pooled ESM-C feature for target {target_row}")
        features[target_row] = pooled
        rows.append(
            {
                "target_feature_index": target_row,
                "sequence_key": str(index.iloc[target_row]["sequence_key"]),
                "sequence_length": len(sequence),
                "window_count": len(bounds),
                "unknown_token_count": unknown_residues,
                "feature_available": True,
                "feature_route": "ESMC_600M_OVERLAP_STITCHED_RESIDUE_MEAN",
                "embedding_l2_norm": float(np.linalg.norm(pooled)),
            }
        )
        if args.progress_every > 0 and (
            (target_row + 1) % int(args.progress_every) == 0
            or target_row + 1 == len(sequences)
        ):
            print(f"encoded {target_row + 1}/{len(sequences)} targets", flush=True)
    features.flush()

    feature_index = pd.DataFrame(rows)
    checks = {
        "target_index_dense": feature_index["target_feature_index"].tolist()
        == list(range(len(index))),
        "shape_exact": list(features.shape) == [len(index), feature_dim],
        "all_values_finite": bool(np.isfinite(np.asarray(features)).all()),
        "all_vectors_nonzero": bool((feature_index["embedding_l2_norm"] > 0).all()),
        "all_residues_tokenized_without_unk": bool(
            (feature_index["unknown_token_count"] == 0).all()
        ),
        "context_limit_respected": bool(
            int(args.max_window_length) + prefix_special_tokens + suffix_special_tokens
            <= int(args.max_context_tokens)
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    feature_index.to_csv(index_path, index=False, compression="gzip")
    try:
        from huggingface_hub import hf_hub_download

        weight_path = Path(
            hf_hub_download(
                args.model_id,
                "model.safetensors",
                revision=args.model_revision,
            )
        )
        weight_sha256 = sha256(weight_path)
    except Exception:
        weight_path = Path("")
        weight_sha256 = None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "label-blind frozen ODTI target entities",
        "model": {
            "id": str(args.model_id),
            "revision": str(args.model_revision),
            "architecture": type(model).__name__,
            "feature_dim": feature_dim,
            "pooling": "overlap-stitched residue-uniform whole-protein mean",
            "max_context_tokens": int(args.max_context_tokens),
            "max_window_length": int(args.max_window_length),
            "window_overlap": int(args.window_overlap),
            "weight_path": str(weight_path) if weight_sha256 else None,
            "weight_sha256": weight_sha256,
        },
        "implementation": {
            "esm_expected_git_revision": ESM_IMPLEMENTATION_REVISION,
            "transformers_expected_git_revision": TRANSFORMERS_IMPLEMENTATION_REVISION,
            "esm_version": importlib.metadata.version("esm"),
            "transformers_version": importlib.metadata.version("transformers"),
            "esm_direct_url": direct_url("esm"),
            "transformers_direct_url": direct_url("transformers"),
        },
        "counts": {
            "targets": int(len(index)),
            "total_residues": int(sum(map(len, sequences))),
            "total_windows": int(feature_index["window_count"].sum()),
            "unknown_tokens": int(feature_index["unknown_token_count"].sum()),
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
        "inference_dtype": str(dtype),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if manifest["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", default=str(TARGET_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-name", default="ESMC_600M_1152_FLOAT32_V1.npy")
    parser.add_argument("--index-name", default="ESMC_600M_TARGET_INDEX_V1.csv.gz")
    parser.add_argument("--manifest-name", default="ESMC_600M_FEATURE_MANIFEST_V1.json")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--feature-dim", type=int, default=1152)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-window-length", type=int, default=2046)
    parser.add_argument("--window-overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()

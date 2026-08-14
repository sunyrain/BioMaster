#!/usr/bin/env python3
"""Extract official DTIAM BerMol + ESM2-650M representations.

The feature definitions follow DTIAM's official ``extract_feature.py``:
BerMol pooled encoder output (768) for compounds and the final-layer mean
ESM2-t33-650M representation (1280) for proteins truncated to 1022 residues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIR_STORE = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DRUG_INDEX = BASE / "feature_store_v1/DRUG_FEATURE_INDEX_V1.csv.gz"
TARGET_INDEX = BASE / "feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
BERMOL = ROOT / "third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl"
BERMOL_CODE = ROOT / "third_party/sota_dti_2026/DTIAM/code/BerMol"
ESM_CHECKPOINT = Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")
ESM_CONTACT = Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt")
OUT = BASE / "public_retrained_v1/dtiam_official_feature_store_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_bermol(device: torch.device, force: bool) -> tuple[np.ndarray, pd.DataFrame, dict]:
    output = OUT / "DTIAM_BERMOL768_FLOAT32_V1.npy"
    index_path = OUT / "DTIAM_BERMOL_DRUG_INDEX_V1.csv.gz"
    if output.is_file() and index_path.is_file() and not force:
        index = pd.read_csv(index_path)
        values = np.load(output, mmap_mode="r")
        if values.shape == (len(index), 768) and len(index) == 62477:
            return values, index, {"route": "REUSED_PASSED_SHAPE_ARTIFACT"}

    sys.path.insert(0, str(BERMOL_CODE))
    from bermol.tokenizer import BerMolTokenizer  # noqa: E402

    index = pd.read_csv(DRUG_INDEX).sort_values("drug_feature_index").reset_index(drop=True)
    if len(index) != 62477 or index["drug_feature_index"].tolist() != list(range(len(index))):
        raise RuntimeError("Frozen compound feature index changed")
    with BERMOL.open("rb") as handle:
        predictor = dill.load(handle)
    predictor.model.to(device).eval()
    tokenizer = BerMolTokenizer(predictor.vocab)
    token_rows: list[torch.Tensor | None] = []
    failures = []
    for row in index.itertuples(index=False):
        try:
            token_rows.append(tokenizer.encode(str(row.model_ligand_smiles)).squeeze(0))
        except Exception as error:
            token_rows.append(None)
            failures.append({
                "drug_feature_index": int(row.drug_feature_index),
                "model_ligand_smiles": str(row.model_ligand_smiles),
                "reason": f"{type(error).__name__}: {str(error)[:160]}",
            })
    available = [(index, token) for index, token in enumerate(token_rows) if token is not None]
    available.sort(key=lambda item: len(item[1]))
    features = np.zeros((len(index), 768), dtype=np.float32)
    cursor = 0
    batches = 0
    max_attention_elements = 4_000_000
    with torch.inference_mode():
        while cursor < len(available):
            maximum_length = len(available[cursor][1])
            batch_size = min(512, max(1, max_attention_elements // max(1, maximum_length**2)))
            end = min(len(available), cursor + batch_size)
            # Rows are sorted ascending, so account for the longest selected row.
            maximum_length = len(available[end - 1][1])
            batch_size = min(end - cursor, max(1, max_attention_elements // max(1, maximum_length**2)))
            end = cursor + batch_size
            selected = available[cursor:end]
            maximum_length = len(selected[-1][1])
            token_ids = torch.zeros((len(selected), maximum_length), dtype=torch.long)
            attention_mask = torch.full(
                (len(selected), maximum_length, maximum_length), -10000.0, dtype=torch.float32
            )
            for position, (_, tokens) in enumerate(selected):
                length = len(tokens)
                token_ids[position, :length] = tokens
                attention_mask[position, :, :length] = 0.0
            _, pooled = predictor.model.encoder(
                token_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
            )
            positions = np.asarray([item[0] for item in selected], dtype=np.int64)
            features[positions] = pooled.float().cpu().numpy()
            cursor = end
            batches += 1
            if cursor % 5000 < batch_size:
                print(json.dumps({"bermol_completed": cursor, "total": len(available)}), flush=True)
    del predictor
    torch.cuda.empty_cache()
    index["dtiam_bermol_available"] = [token is not None for token in token_rows]
    index["dtiam_bermol_token_count"] = [len(token) if token is not None else 0 for token in token_rows]
    np.save(output, features, allow_pickle=False)
    index.to_csv(index_path, index=False)
    return features, index, {
        "route": "OFFICIAL_BERMOL_BASE_POOLED_ENCODER_BATCHED",
        "batches": batches,
        "failures": failures,
    }


def extract_esm(device: torch.device, force: bool) -> tuple[np.ndarray, pd.DataFrame, dict]:
    output = OUT / "DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
    index_path = OUT / "DTIAM_ESM2_TARGET_INDEX_V1.csv.gz"
    if output.is_file() and index_path.is_file() and not force:
        index = pd.read_csv(index_path)
        values = np.load(output, mmap_mode="r")
        if values.shape == (428, 1280) and len(index) == 428:
            return values, index, {"route": "REUSED_PASSED_SHAPE_ARTIFACT"}

    # Reuse the exact local official checkpoint and avoid any hidden network fetch.
    os.environ["TORCH_HOME"] = "/root/autodl-tmp/.cache/torch"
    import esm  # noqa: E402

    index = pd.read_csv(TARGET_INDEX).sort_values("target_feature_index").reset_index(drop=True)
    if len(index) != 428 or index["target_feature_index"].tolist() != list(range(428)):
        raise RuntimeError("Frozen target feature index changed")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device).eval()
    converter = alphabet.get_batch_converter()
    representation_layer = model.num_layers
    features = np.empty((428, 1280), dtype=np.float32)
    lengths = []
    with torch.inference_mode():
        for position, row in enumerate(index.itertuples(index=False)):
            sequence = str(row.protein_sequence)[:1022]
            _, _, batch_tokens = converter([(str(row.sequence_key), sequence)])
            result = model(
                batch_tokens.to(device, non_blocking=True),
                repr_layers=[representation_layer],
                return_contacts=False,
            )
            # Exact DTIAM rule: positions [1:] include the terminal EOS token.
            representation = result["representations"][representation_layer][0, 1:].mean(0)
            features[position] = representation.float().cpu().numpy()
            lengths.append(len(sequence))
            if (position + 1) % 25 == 0:
                print(json.dumps({"esm2_completed": position + 1, "total": 428}), flush=True)
    del model
    torch.cuda.empty_cache()
    index["dtiam_esm_input_length"] = lengths
    index["dtiam_esm_truncated_to_1022"] = index["protein_sequence"].str.len().gt(1022)
    np.save(output, features, allow_pickle=False)
    index.to_csv(index_path, index=False)
    return features, index, {
        "route": "OFFICIAL_ESM2_T33_650M_FINAL_LAYER_MEAN_INCLUDING_EOS",
        "targets_truncated_to_1022": int(index["dtiam_esm_truncated_to_1022"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=["all", "bermol", "esm"], default="all")
    args = parser.parse_args()
    required = [PAIR_STORE, DRUG_INDEX, TARGET_INDEX, BERMOL, ESM_CHECKPOINT, ESM_CONTACT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Official 650M feature extraction requires CUDA for this execution")
    metadata: dict[str, object] = {}
    drug_features = drug_index = target_features = target_index = None
    if args.only in {"all", "bermol"}:
        drug_features, drug_index, metadata["bermol"] = extract_bermol(device, args.force)
    if args.only in {"all", "esm"}:
        target_features, target_index, metadata["esm2"] = extract_esm(device, args.force)
    if args.only != "all":
        print(json.dumps({"status": "PARTIAL_PASS", "completed": args.only, **metadata}, indent=2))
        return
    assert drug_features is not None and drug_index is not None
    assert target_features is not None and target_index is not None
    pairs = pd.read_csv(PAIR_STORE, usecols=[
        "calibration_pair_id", "drug_feature_index", "target_feature_index", "drug_feature_available",
    ])
    pairs["dtiam_bermol_available"] = pairs["drug_feature_index"].map(
        drug_index.set_index("drug_feature_index")["dtiam_bermol_available"]
    ).astype(bool)
    pair_index_path = OUT / "DTIAM_86674_PAIR_FEATURE_INDEX_V1.csv.gz"
    pairs.to_csv(pair_index_path, index=False)
    checks = {
        "official_bermol_weight_hash": sha256(BERMOL) == "afd1929fdbef6da110b5f4d0688a0f4a5f88fc3fb660bfcc1fb56e9624de38b1",
        "bermol_exact_shape": tuple(drug_features.shape) == (62477, 768),
        "esm2_exact_shape": tuple(target_features.shape) == (428, 1280),
        "features_finite": np.isfinite(drug_features).all() and np.isfinite(target_features).all(),
        "pair_index_exact_86674": len(pairs) == 86674 and pairs["calibration_pair_id"].is_unique,
        "same_single_quarantine_boundary": int((~pairs["dtiam_bermol_available"]).sum()) == 1,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparator": "DTIAM_OFFICIAL_REPRESENTATION_COMPATIBLE_RETRAIN_V1",
        "official_representation_definition": {
            "drug": "BerMolModel_base pooled encoder output, 768 dimensions",
            "protein": "ESM2 t33 650M final-layer mean over sequence positions [1:], 1280 dimensions, first 1022 residues",
        },
        "metadata": metadata,
        "counts": {
            "unique_model_smiles": 62477,
            "targets": 428,
            "pairs": 86674,
            "bermol_unavailable_pairs": int((~pairs["dtiam_bermol_available"]).sum()),
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "compatibility_boundary": "Representations and feature concatenation follow official DTIAM. Downstream AutoGluon is version 1.4.0 on Python 3.10 rather than paper version 0.5.2 on Python 3.7 and will be labeled compatible retraining, not bitwise paper reproduction.",
        "inputs": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path) for path in required},
        "outputs": {
            "bermol_sha256": sha256(OUT / "DTIAM_BERMOL768_FLOAT32_V1.npy"),
            "esm2_sha256": sha256(OUT / "DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"),
            "drug_index_sha256": sha256(OUT / "DTIAM_BERMOL_DRUG_INDEX_V1.csv.gz"),
            "target_index_sha256": sha256(OUT / "DTIAM_ESM2_TARGET_INDEX_V1.csv.gz"),
            "pair_index_sha256": sha256(pair_index_path),
        },
    }
    summary_path = OUT / "DTIAM_OFFICIAL_FEATURE_STORE_AUDIT_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

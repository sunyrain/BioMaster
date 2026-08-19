#!/usr/bin/env python3
"""Build checkpoint-compatible protein features for the frozen recent-target panel.

Run with the frozen ConPLex environment so ProtBERT pooling is bitwise aligned:

    PYTHONPATH=third_party/ConPLex .venvs/conplex/bin/python \
      scripts/build_biomaster_recent_novel_target_features_v1.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/biomaster_recent_novel_target_external_freeze_20260819.json"
OUT = ROOT / "outputs/biomaster_recent_novel_target_external_v1"
TRAIN_INDEX = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
TRAIN_PROTBERT = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/PROTBERT1024_FLOAT32_V1.npy"
TRAIN_ESM2 = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
ESM_CHECKPOINT = Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")
ESM_CONTACT = Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt")
PROTBERT_OUT = OUT / "RECENT_NOVEL_TARGET_PROTBERT1024_FLOAT32_V1.npy"
ESM2_OUT = OUT / "RECENT_NOVEL_TARGET_ESM2_T33_650M_1280_FLOAT32_V1.npy"
INDEX_OUT = OUT / "RECENT_NOVEL_TARGET_INDEX_V1.csv"
AUDIT_OUT = OUT / "RECENT_NOVEL_TARGET_FEATURE_AUDIT_V1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def validate_freeze(payload: dict[str, object]) -> list[dict[str, object]]:
    if payload.get("status") != "FROZEN_BEFORE_MODEL_SCORING":
        raise RuntimeError("external panel is not frozen before scoring")
    targets = list(payload.get("targets", []))
    if len(targets) != 2:
        raise RuntimeError("frozen recent-target panel must contain exactly two targets")
    for target in targets:
        sequence = str(target["sequence"])
        if len(sequence) != int(target["sequence_length"]):
            raise RuntimeError(f"sequence length mismatch for {target['uniprot_accession']}")
        if sequence_sha256(sequence) != target["sequence_sha256"]:
            raise RuntimeError(f"sequence hash mismatch for {target['uniprot_accession']}")
    return targets


def extract_protbert(
    targets: list[dict[str, object]],
    reference_sequence: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    import transformers
    from conplex_dti.featurizer.protein import ProtBertFeaturizer

    if transformers.__version__ != "4.46.3":
        raise RuntimeError(
            "ProtBERT must use the frozen ConPLex environment with transformers 4.46.3; "
            f"found {transformers.__version__}"
        )
    featurizer = ProtBertFeaturizer(save_dir=OUT)
    featurizer.cuda(device)
    rows = [reference_sequence] + [str(target["sequence"]) for target in targets]
    values = []
    with torch.inference_mode():
        for position, sequence in enumerate(rows):
            feature = featurizer._transform(sequence).detach().cpu().numpy().astype(np.float32)
            if feature.shape != (1024,) or not np.isfinite(feature).all():
                raise RuntimeError(f"invalid ProtBERT feature at row {position}: {feature.shape}")
            values.append(feature)
    reference = np.asarray(np.load(TRAIN_PROTBERT, mmap_mode="r")[0], dtype=np.float32)
    max_abs = float(np.max(np.abs(values[0] - reference)))
    reference_cosine = cosine(values[0], reference)
    if max_abs > 1e-6:
        raise RuntimeError(f"ProtBERT reference reproduction failed: max_abs={max_abs}")
    del featurizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.stack(values[1:]), {
        "route": "CONPLEX_PROTBERT_PER_RESIDUE_MEAN_EXCLUDING_SPECIAL_TOKENS",
        "transformers_version": transformers.__version__,
        "reference_target_feature_index": 0,
        "reference_max_abs_difference": max_abs,
        "reference_cosine": reference_cosine,
        "reference_bitwise_exact": bool(max_abs == 0.0),
    }


def extract_esm2(
    targets: list[dict[str, object]],
    reference_sequence: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    os.environ["TORCH_HOME"] = "/root/autodl-tmp/.cache/torch"
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device).eval()
    converter = alphabet.get_batch_converter()
    layer = model.num_layers
    rows = [reference_sequence] + [str(target["sequence"]) for target in targets]
    values = []
    lengths = []
    with torch.inference_mode():
        for position, sequence in enumerate(rows):
            truncated = sequence[:1022]
            _, _, tokens = converter([(f"recent_{position}", truncated)])
            result = model(
                tokens.to(device, non_blocking=True),
                repr_layers=[layer],
                return_contacts=False,
            )
            # Exact frozen DTIAM auxiliary rule: [1:] includes terminal EOS.
            feature = result["representations"][layer][0, 1:].mean(0)
            feature = feature.float().cpu().numpy().astype(np.float32)
            if feature.shape != (1280,) or not np.isfinite(feature).all():
                raise RuntimeError(f"invalid ESM2 feature at row {position}: {feature.shape}")
            values.append(feature)
            lengths.append(len(truncated))
    reference = np.asarray(np.load(TRAIN_ESM2, mmap_mode="r")[0], dtype=np.float32)
    max_abs = float(np.max(np.abs(values[0] - reference)))
    reference_cosine = cosine(values[0], reference)
    if max_abs > 1e-5:
        raise RuntimeError(f"ESM2 reference reproduction failed: max_abs={max_abs}")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.stack(values[1:]), {
        "route": "DTIAM_ESM2_T33_650M_FINAL_LAYER_MEAN_POSITIONS_1_THROUGH_EOS",
        "esm_version": getattr(esm, "__version__", "unknown"),
        "reference_target_feature_index": 0,
        "reference_max_abs_difference": max_abs,
        "reference_cosine": reference_cosine,
        "reference_bitwise_exact": bool(max_abs == 0.0),
        "input_lengths": lengths[1:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    required = [FREEZE, TRAIN_INDEX, TRAIN_PROTBERT, TRAIN_ESM2, ESM_CHECKPOINT, ESM_CONTACT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.force and all(path.is_file() for path in [PROTBERT_OUT, ESM2_OUT, INDEX_OUT, AUDIT_OUT]):
        previous = json.loads(AUDIT_OUT.read_text())
        if previous.get("status") == "PASS":
            print(json.dumps(previous, ensure_ascii=False, indent=2))
            return
    payload = json.loads(FREEZE.read_text())
    targets = validate_freeze(payload)
    training_index = pd.read_csv(TRAIN_INDEX).sort_values("target_feature_index").reset_index(drop=True)
    if len(training_index) != 428 or int(training_index.iloc[0]["target_feature_index"]) != 0:
        raise RuntimeError("frozen 428-target reference index changed")
    reference_sequence = str(training_index.iloc[0]["protein_sequence"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("feature extraction requires CUDA")
    protbert, protbert_audit = extract_protbert(targets, reference_sequence, device)
    esm2, esm2_audit = extract_esm2(targets, reference_sequence, device)
    if protbert.shape != (2, 1024) or esm2.shape != (2, 1280):
        raise RuntimeError(f"unexpected feature shapes: {protbert.shape}, {esm2.shape}")
    np.save(PROTBERT_OUT, protbert, allow_pickle=False)
    np.save(ESM2_OUT, esm2, allow_pickle=False)
    index = pd.DataFrame(
        [
            {
                "target_feature_index": position,
                "panel_role": target["panel_role"],
                "uniprot_accession": target["uniprot_accession"],
                "gene_symbol": target["gene_symbol"],
                "protein_name": target["protein_name"],
                "target_assay_family": target["fixed_target_assay_family"],
                "sequence_length": target["sequence_length"],
                "sequence_sha256": target["sequence_sha256"],
                "protbert_l2_norm": float(np.linalg.norm(protbert[position])),
                "esm2_l2_norm": float(np.linalg.norm(esm2[position])),
            }
            for position, target in enumerate(targets)
        ]
    )
    index.to_csv(INDEX_OUT, index=False)
    checks = {
        "freeze_valid": True,
        "exact_two_targets": len(index) == 2,
        "protbert_shape_2x1024": protbert.shape == (2, 1024),
        "esm2_shape_2x1280": esm2.shape == (2, 1280),
        "all_features_finite": bool(np.isfinite(protbert).all() and np.isfinite(esm2).all()),
        "protbert_reference_reproduced": protbert_audit["reference_max_abs_difference"] <= 1e-6,
        "esm2_reference_reproduced": esm2_audit["reference_max_abs_difference"] <= 1e-5,
    }
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "freeze_id": payload["freeze_id"],
        "device": str(device),
        "checks": checks,
        "protbert": protbert_audit,
        "esm2": esm2_audit,
        "inputs": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path) for path in required},
        "outputs": {
            "protbert": str(PROTBERT_OUT.relative_to(ROOT)),
            "protbert_sha256": sha256(PROTBERT_OUT),
            "esm2": str(ESM2_OUT.relative_to(ROOT)),
            "esm2_sha256": sha256(ESM2_OUT),
            "index": str(INDEX_OUT.relative_to(ROOT)),
            "index_sha256": sha256(INDEX_OUT),
        },
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

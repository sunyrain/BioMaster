#!/usr/bin/env python3
"""Build label-free features for the frozen Davis external candidate.

The resulting feature store is for external scoring only.  No Davis affinity
value is read while generating Morgan, ProtBERT or ESM2 representations.  The
source pair manifest is already frozen; this script only adds entity features
and a provenance manifest.  It intentionally does not train or promote a
model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST = ROOT / (
    "outputs/biomaster_odti_local_external_candidates_v1/"
    "DAVIS_ENTITY_COLD_PAIR_MANIFEST_V1.csv.gz"
)
OUT_DIR = ROOT / "outputs/biomaster_odti_davis_feature_store_v1"
HF_CACHE = Path("/root/autodl-tmp/hf-cache")
TORCH_CACHE = Path("/root/autodl-tmp/.cache/torch")
PROTBERT_REPO = "Rostlab/prot_bert"
PROTBERT_SNAPSHOT = HF_CACHE / (
    "hub/models--Rostlab--prot_bert/snapshots/"
    "7a894481acdc12202f0a415dd567f6cfdb698908"
)
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_smiles(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def set_deterministic(seed: int = 20260817) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_morgan(ligands: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    values = np.zeros((len(ligands), 2048), dtype=np.uint8)
    for index, smiles in enumerate(ligands):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"Davis ligand cannot be parsed: {smiles}")
        values[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
    return values


def build_protbert(sequences: list[str], batch_size: int = 2) -> tuple[np.ndarray, list[int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Davis ProtBERT feature extraction")
    os.environ["HF_HOME"] = str(HF_CACHE)
    from transformers import BertModel, BertTokenizer  # noqa: PLC0415

    if not (PROTBERT_SNAPSHOT / "pytorch_model.bin").is_file():
        raise FileNotFoundError(PROTBERT_SNAPSHOT / "pytorch_model.bin")
    tokenizer = BertTokenizer(
        vocab_file=str(PROTBERT_SNAPSHOT / "vocab.txt"), do_lower_case=False
    )
    model = BertModel.from_pretrained(str(PROTBERT_SNAPSHOT), local_files_only=True)
    device = torch.device("cuda")
    model.to(device).eval()
    outputs = np.zeros((len(sequences), 1024), dtype=np.float32)
    used_lengths: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            prepared = [" ".join(list(sequence[:1022])).replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X") for sequence in batch]
            encoded = tokenizer(
                prepared,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            values = pooled.float().cpu().numpy()
            outputs[start : start + len(batch)] = values
            used_lengths.extend([min(len(sequence), 1022) for sequence in batch])
    del model
    torch.cuda.empty_cache()
    return outputs, used_lengths


def build_esm2(sequences: list[str], batch_size: int = 2) -> tuple[np.ndarray, list[int]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Davis ESM2 feature extraction")
    os.environ["TORCH_HOME"] = str(TORCH_CACHE)
    sys.path.insert(0, str(ROOT))
    import esm  # noqa: PLC0415

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda")
    model.to(device).eval()
    converter = alphabet.get_batch_converter()
    layer = model.num_layers
    outputs = np.zeros((len(sequences), 1280), dtype=np.float32)
    used_lengths: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = [sequence[:1022] for sequence in sequences[start : start + batch_size]]
            labels_and_sequences = [(str(start + offset), sequence) for offset, sequence in enumerate(batch)]
            _, _, tokens = converter(labels_and_sequences)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(tokens.to(device, non_blocking=True), repr_layers=[layer], return_contacts=False)
            representation = result["representations"][layer]
            for offset, sequence in enumerate(batch):
                outputs[start + offset] = (
                    representation[offset, 1 : len(sequence) + 1].float().mean(dim=0).cpu().numpy()
                )
            used_lengths.extend([len(sequence) for sequence in batch])
    del model
    torch.cuda.empty_cache()
    return outputs, used_lengths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_MANIFEST.is_file():
        raise FileNotFoundError(INPUT_MANIFEST)
    set_deterministic()
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    manifest = pd.read_csv(INPUT_MANIFEST, low_memory=False)
    required = {"prospective_external_pair_id", "canonical_smiles", "protein_sequence", "exact_frozen_pair_overlap"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(f"Davis manifest missing columns: {missing}")
    manifest = manifest.loc[~manifest["exact_frozen_pair_overlap"].astype(bool)].copy()
    if manifest.empty:
        raise RuntimeError("No non-overlap Davis pairs remain")
    ligands = sorted(manifest["canonical_smiles"].dropna().astype(str).unique())
    sequences = sorted(manifest["protein_sequence"].dropna().astype(str).unique())
    drug_index = pd.DataFrame({"drug_feature_index": np.arange(len(ligands), dtype=np.int32), "model_ligand_smiles": ligands})
    target_index = pd.DataFrame({"target_feature_index": np.arange(len(sequences), dtype=np.int32), "protein_sequence": sequences})
    drug_features = build_morgan(ligands)
    target_protbert, protbert_lengths = build_protbert(sequences)
    target_esm2, esm2_lengths = build_esm2(sequences)
    drug_lookup = dict(zip(ligands, drug_index["drug_feature_index"], strict=True))
    target_lookup = dict(zip(sequences, target_index["target_feature_index"], strict=True))
    manifest["drug_feature_index"] = manifest["canonical_smiles"].map(drug_lookup).astype(np.int32)
    manifest["target_feature_index"] = manifest["protein_sequence"].map(target_lookup).astype(np.int32)
    manifest["feature_source"] = "DAVIS_LABEL_BLIND_PROTBERT_ESM2_MORGAN_V1"
    manifest_path = OUT_DIR / "DAVIS_ENTITY_COLD_FEATURE_INDEXED_PAIR_MANIFEST_V1.csv.gz"
    manifest.to_csv(
        manifest_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    drug_index["feature_available"] = True
    drug_index["on_bits"] = drug_features.sum(axis=1).astype(np.int32)
    target_index["sequence_length"] = target_index["protein_sequence"].str.len().astype(np.int32)
    target_index["protbert_used_length"] = protbert_lengths
    target_index["esm2_used_length"] = esm2_lengths
    target_index["protbert_embedding_l2_norm"] = np.linalg.norm(target_protbert, axis=1)
    target_index["esm2_embedding_l2_norm"] = np.linalg.norm(target_esm2, axis=1)
    drug_index.to_csv(
        OUT_DIR / "DAVIS_DRUG_FEATURE_INDEX_V1.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    target_index.to_csv(
        OUT_DIR / "DAVIS_TARGET_FEATURE_INDEX_V1.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    np.save(OUT_DIR / "DAVIS_MORGAN2048_UINT8_V1.npy", drug_features, allow_pickle=False)
    np.save(OUT_DIR / "DAVIS_PROTBERT1024_FLOAT32_V1.npy", target_protbert, allow_pickle=False)
    np.save(OUT_DIR / "DAVIS_ESM2_T33_650M_1280_FLOAT32_V1.npy", target_esm2, allow_pickle=False)
    report = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_ODTI_DAVIS_LABEL_BLIND_FEATURE_STORE_V1",
        "claim_status": "FEATURES_ONLY; NO_EXTERNAL_SCORE_OR_MODEL_PROMOTION",
        "input_manifest": {"path": str(INPUT_MANIFEST.relative_to(ROOT)), "sha256": sha256(INPUT_MANIFEST)},
        "counts": {
            "nonoverlap_pairs": int(len(manifest)),
            "unique_ligands": int(len(ligands)),
            "unique_targets": int(len(sequences)),
            "both_unseen_pairs": int(manifest["entity_overlap_class"].eq("both_unseen").sum()),
        },
        "features": {
            "morgan_shape": list(drug_features.shape),
            "protbert_shape": list(target_protbert.shape),
            "esm2_shape": list(target_esm2.shape),
            "protbert_max_residues": 1022,
            "esm2_max_residues": 1022,
            "protbert_model": PROTBERT_REPO,
            "esm2_model": ESM2_MODEL_NAME,
            "label_free": True,
        },
        "provenance": {
            "protbert_cache_root": str(PROTBERT_SNAPSHOT),
            "esm2_cache_root": str(TORCH_CACHE),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "outputs": {},
    }
    output_paths = [
        OUT_DIR / "DAVIS_ENTITY_COLD_FEATURE_INDEXED_PAIR_MANIFEST_V1.csv.gz",
        OUT_DIR / "DAVIS_DRUG_FEATURE_INDEX_V1.csv.gz",
        OUT_DIR / "DAVIS_TARGET_FEATURE_INDEX_V1.csv.gz",
        OUT_DIR / "DAVIS_MORGAN2048_UINT8_V1.npy",
        OUT_DIR / "DAVIS_PROTBERT1024_FLOAT32_V1.npy",
        OUT_DIR / "DAVIS_ESM2_T33_650M_1280_FLOAT32_V1.npy",
    ]
    report["outputs"] = {str(path.relative_to(ROOT)): sha256(path) for path in output_paths}
    report_path = OUT_DIR / "DAVIS_FEATURE_STORE_MANIFEST_V1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

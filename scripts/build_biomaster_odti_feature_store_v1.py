#!/usr/bin/env python3
"""Build an audited feature store for the frozen BioMaster-ODTI benchmark.

The historical ConPLex HDF5 cache is useful but its datasets are keyed by the
raw strings used when each batch was written.  Consequently, a cache miss is
not evidence that a molecule is invalid.  This builder uses an explicit,
auditable route for every unique model SMILES:

1. reuse the exact raw-string cache entry when it exists;
2. otherwise recompute a Morgan radius-2, 2048-bit fingerprint with RDKit;
3. quarantine (never silently zero-fill) any molecule that still fails.

Protein embeddings are mapped from the exact sequences in the frozen ConPLex
input file, avoiding dependence on subsequently rebuilt target-universe keys.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
SPLITS = BASE / "benchmark_splits_v1/CHEMBL37_86674_FROZEN_SPLIT_ASSIGNMENTS_V1.csv.gz"
CAL = ROOT / "outputs/current_production_package_v2/conplex_target_calibration_v5_official"
CONPLEX_INPUT = CAL / "CHEMBL37_CONPLEX_CALIBRATION_INPUT_V5.tsv"
SCORED = CAL / "evaluation/CONPLEX_CALIBRATION_PAIRS_SCORED_V5.csv.gz"
MORGAN_H5 = CAL / "conplex_cache/Morgan_features.h5"
PROTBERT_H5 = CAL / "conplex_cache/ProtBert_features.h5"
OUT = BASE / "feature_store_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h5_raw_key(value: str) -> str:
    """Match the sanitizer used by the historical ConPLex featurizer."""
    return value.replace("/", "|")


def main() -> None:
    required = [SPLITS, CONPLEX_INPUT, SCORED, MORGAN_H5, PROTBERT_H5]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(SPLITS, low_memory=False)
    if len(pairs) != 86674 or not pairs["calibration_pair_id"].is_unique:
        raise RuntimeError("Frozen benchmark population changed")

    # The model input is the authoritative sequence source for these SEQ000...
    # keys.  It has no header and contains target_key, pair_id, sequence, SMILES.
    input_table = pd.read_csv(
        CONPLEX_INPUT,
        sep="\t",
        header=None,
        names=["sequence_key", "calibration_pair_id", "protein_sequence", "input_smiles"],
        dtype=str,
    )
    if len(input_table) != len(pairs):
        raise RuntimeError("ConPLex input row count differs from frozen labels")
    target_sequences = input_table[["sequence_key", "protein_sequence"]].drop_duplicates()
    if target_sequences["sequence_key"].duplicated().any() or len(target_sequences) != 428:
        raise RuntimeError("Expected one exact protein sequence for each of 428 targets")

    drug_table = (
        pairs[["model_ligand_smiles"]]
        .drop_duplicates()
        .sort_values("model_ligand_smiles", kind="stable")
        .reset_index(drop=True)
    )
    drug_table.insert(0, "drug_feature_index", np.arange(len(drug_table), dtype=np.int32))
    drug_features = np.empty((len(drug_table), 2048), dtype=np.uint8)
    route = np.empty(len(drug_table), dtype=object)
    failure_reason = np.full(len(drug_table), "", dtype=object)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    with h5py.File(MORGAN_H5, "r") as cache:
        for row in drug_table.itertuples(index=False):
            index = int(row.drug_feature_index)
            smiles = str(row.model_ligand_smiles)
            cache_key = h5_raw_key(smiles)
            if cache_key in cache:
                value = np.asarray(cache[cache_key], dtype=np.float32)
                if value.shape != (2048,) or not np.isfinite(value).all():
                    raise RuntimeError(f"Invalid cached Morgan vector for {smiles}")
                rounded = np.rint(value)
                if not np.allclose(value, rounded) or not np.isin(rounded, [0, 1]).all():
                    raise RuntimeError(f"Cached Morgan vector is not binary for {smiles}")
                drug_features[index] = rounded.astype(np.uint8)
                route[index] = "HISTORICAL_H5_EXACT_RAW_KEY"
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                drug_features[index] = 0
                route[index] = "QUARANTINED_PARSE_FAILURE"
                failure_reason[index] = "RDKit MolFromSmiles returned None and no exact cache key exists"
                continue
            drug_features[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
            route[index] = "RDKIT_2026_RECOMPUTED_CACHE_MISS"

    drug_table["feature_route"] = route
    drug_table["feature_available"] = drug_table["feature_route"].ne("QUARANTINED_PARSE_FAILURE")
    drug_table["failure_reason"] = failure_reason
    drug_table["on_bits"] = drug_features.sum(axis=1).astype(np.int32)

    target_table = target_sequences.sort_values("sequence_key", kind="stable").reset_index(drop=True)
    target_table.insert(0, "target_feature_index", np.arange(len(target_table), dtype=np.int16))
    target_features = np.empty((len(target_table), 1024), dtype=np.float32)
    target_route: list[str] = []
    with h5py.File(PROTBERT_H5, "r") as cache:
        for row in target_table.itertuples(index=False):
            sequence = str(row.protein_sequence)
            if sequence not in cache:
                raise RuntimeError(f"Missing exact ProtBert cache sequence for {row.sequence_key}")
            value = np.asarray(cache[sequence], dtype=np.float32)
            if value.shape != (1024,) or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid ProtBert vector for {row.sequence_key}")
            target_features[int(row.target_feature_index)] = value
            target_route.append("HISTORICAL_H5_EXACT_SEQUENCE")
    target_table["feature_route"] = target_route
    target_table["sequence_length"] = target_table["protein_sequence"].str.len().astype(np.int32)
    target_table["embedding_l2_norm"] = np.linalg.norm(target_features, axis=1)

    # Add stable feature indices and the frozen external ConPLex score to each
    # benchmark pair.  No target labels are used while creating representations.
    drug_lookup = drug_table.set_index("model_ligand_smiles")["drug_feature_index"]
    target_lookup = target_table.set_index("sequence_key")["target_feature_index"]
    pairs["drug_feature_index"] = pairs["model_ligand_smiles"].map(drug_lookup).astype(np.int32)
    pairs["target_feature_index"] = pairs["sequence_key"].map(target_lookup).astype(np.int16)
    pairs["drug_feature_available"] = pairs["drug_feature_index"].map(
        drug_table.set_index("drug_feature_index")["feature_available"]
    ).astype(bool)
    scored = pd.read_csv(SCORED, usecols=["calibration_pair_id", "conplex_score"])
    if len(scored) != len(pairs) or not scored["calibration_pair_id"].is_unique:
        raise RuntimeError("ConPLex scored-pair population changed")
    pairs = pairs.merge(scored, on="calibration_pair_id", how="left", validate="one_to_one")

    drug_npy = OUT / "MORGAN2048_UINT8_V1.npy"
    target_npy = OUT / "PROTBERT1024_FLOAT32_V1.npy"
    drug_csv = OUT / "DRUG_FEATURE_INDEX_V1.csv.gz"
    target_csv = OUT / "TARGET_FEATURE_INDEX_V1.csv.gz"
    pair_csv = OUT / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
    np.save(drug_npy, drug_features, allow_pickle=False)
    np.save(target_npy, target_features, allow_pickle=False)
    drug_table.to_csv(drug_csv, index=False)
    target_table.to_csv(target_csv, index=False)
    pairs.to_csv(pair_csv, index=False)

    failed_drugs = drug_table[~drug_table["feature_available"]]
    failed_pair_count = int((~pairs["drug_feature_available"]).sum())
    checks = {
        "frozen_pair_population_exact": len(pairs) == 86674 and pairs["calibration_pair_id"].is_unique,
        "drug_index_is_dense_and_unique": drug_table["drug_feature_index"].tolist() == list(range(len(drug_table))),
        "target_index_is_dense_and_unique": target_table["target_feature_index"].tolist() == list(range(428)),
        "all_428_exact_protein_embeddings_available": len(target_table) == 428 and np.isfinite(target_features).all(),
        "morgan_vectors_binary": np.isin(drug_features, [0, 1]).all(),
        "available_morgan_vectors_nonempty": (drug_table.loc[drug_table["feature_available"], "on_bits"] > 0).all(),
        "pair_feature_indices_complete": pairs[["drug_feature_index", "target_feature_index"]].notna().all().all(),
        "conplex_score_complete_and_finite": pairs["conplex_score"].notna().all() and np.isfinite(pairs["conplex_score"]).all(),
        "quarantine_is_explicit_not_silent": failed_pair_count == int((~pairs["drug_feature_available"]).sum()),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Frozen ChEMBL37 BioMaster-ODTI benchmark feature layer; label-free representations only",
        "counts": {
            "pairs": int(len(pairs)),
            "unique_model_smiles": int(len(drug_table)),
            "targets": int(len(target_table)),
            "morgan_historical_cache_exact": int((drug_table["feature_route"] == "HISTORICAL_H5_EXACT_RAW_KEY").sum()),
            "morgan_rdkit_recomputed": int((drug_table["feature_route"] == "RDKIT_2026_RECOMPUTED_CACHE_MISS").sum()),
            "morgan_quarantined_unique_smiles": int(len(failed_drugs)),
            "morgan_quarantined_pairs": failed_pair_count,
            "protbert_exact_sequence_cache": int(len(target_table)),
        },
        "feature_shapes": {
            "morgan": list(drug_features.shape),
            "protbert": list(target_features.shape),
        },
        "dtypes": {"morgan": str(drug_features.dtype), "protbert": str(target_features.dtype)},
        "checks": checks,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [drug_npy, target_npy, drug_csv, target_csv, pair_csv]
        },
        "quarantined_smiles": failed_drugs[
            ["drug_feature_index", "model_ligand_smiles", "failure_reason"]
        ].to_dict("records"),
    }
    summary_path = OUT / "FEATURE_STORE_AUDIT_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps({
        "status": summary["status"],
        "counts": summary["counts"],
        "feature_shapes": summary["feature_shapes"],
        "summary": str(summary_path.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

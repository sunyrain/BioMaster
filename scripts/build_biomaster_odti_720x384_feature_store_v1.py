#!/usr/bin/env python3
"""Build audited BioMaster-ODTI deployment features for 720 old drugs x 384 targets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIR_CORE = (
    ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v18"
    / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz"
)
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_SET_SEQUENCE_DTA_ALL_V2.csv"
DRUG_H5 = ROOT / "outputs/strict_dta_720x338_v1/conplex_cache/Morgan_features.h5"
TARGET_STRICT_H5 = ROOT / "outputs/strict_dta_720x338_v1/conplex_cache/ProtBert_features.h5"
TARGET_RECOVERED_H5 = ROOT / "outputs/recovered_dta_720x46_v1/conplex_cache/ProtBert_features.h5"
CALIBRATION_PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
TRAINING_PROTBERT_H5 = (
    ROOT / "outputs/current_production_package_v2/conplex_target_calibration_v5_official"
    / "conplex_cache/ProtBert_features.h5"
)
OUT = BASE / "deployment_720x384_feature_store_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        PAIR_CORE, UNIVERSE, DRUG_H5, TARGET_STRICT_H5, TARGET_RECOVERED_H5,
        CALIBRATION_PAIRS, TRAINING_PROTBERT_H5,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "pairId", "ligand_inchikey", "ligand_smiles", "drug_names",
        "target_chembl_id", "gene_symbol", "assay_lane", "active_target_branch",
        "conplex_score", "is_any_frozen_known_relationship", "local_chembl37_unreported_pair",
        "pair_novelty_class_384", "final_discovery_status_v7", "v10_pair_priority",
        "old_drug_leakage_safe_score_v10", "old_drug_target_deployment_branch_v10",
        "is_v8_mutation_application_pair", "v8_mutation_active_measurements",
        "v8_mutation_best_case_tier", "v8_mutation_best_active_rank",
        "is_v8_prospective_unvalidated_case", "prospective_case_rank_v8",
        "is_v8_database_gap_rediscovery_control", "innovation_validation_role_v7",
    ]
    pairs = pd.read_csv(PAIR_CORE, usecols=columns, low_memory=False)
    if len(pairs) != 276480 or not pairs["pairId"].is_unique:
        raise RuntimeError("Frozen 720x384 core changed")
    drug_table = (
        pairs[["ligand_inchikey", "ligand_smiles", "drug_names"]]
        .drop_duplicates("ligand_inchikey")
        .sort_values("ligand_inchikey")
        .reset_index(drop=True)
    )
    if len(drug_table) != 720 or drug_table["ligand_smiles"].duplicated().any():
        raise RuntimeError("Expected 720 unique old-drug entities and model SMILES")
    drug_table.insert(0, "drug_feature_index", np.arange(len(drug_table), dtype=np.int16))
    drug_features = np.empty((len(drug_table), 2048), dtype=np.uint8)
    with h5py.File(DRUG_H5, "r") as cache:
        for row in drug_table.itertuples(index=False):
            key = str(row.ligand_smiles).replace("/", "|")
            if key not in cache:
                raise RuntimeError(f"Missing exact old-drug Morgan cache: {row.ligand_inchikey}")
            value = np.asarray(cache[key], dtype=np.float32)
            rounded = np.rint(value)
            if value.shape != (2048,) or not np.allclose(value, rounded) or not np.isin(rounded, [0, 1]).all():
                raise RuntimeError(f"Invalid Morgan vector: {row.ligand_inchikey}")
            drug_features[int(row.drug_feature_index)] = rounded.astype(np.uint8)
    drug_table["on_bits"] = drug_features.sum(axis=1).astype(np.int16)

    target_ids = set(pairs["target_chembl_id"])
    universe = pd.read_csv(
        UNIVERSE,
        usecols=[
            "target_chembl_id", "gene_symbol", "uniprot_accession", "sequence",
            "ot_project_assay_family", "assay_lane", "evidence_class",
        ],
        low_memory=False,
    ).drop_duplicates("target_chembl_id")
    target_table = universe[universe["target_chembl_id"].isin(target_ids)].copy()
    if len(target_table) != 384:
        raise RuntimeError(f"Expected 384 target-universe mappings, found {len(target_table)}")
    family_map = {
        "enzyme": "enzyme",
        "kinase": "kinase",
        "ion_channel": "ion_channel",
        "nuclear_epigenetic_transcription": "nuclear_epigenetic",
        "transporter": "transporter",
        "excluded_or_review": "transporter",  # CACNA2D1/2 use transporter functional lane.
    }
    target_table["project_target_assay_family"] = target_table["ot_project_assay_family"].map(family_map)
    if target_table["project_target_assay_family"].isna().any():
        raise RuntimeError("Unmapped deployment target family")
    calibration_target = pd.read_csv(
        CALIBRATION_PAIRS,
        usecols=["target_chembl_id", "target_assay_family", "query_accession"],
        low_memory=False,
    ).drop_duplicates("target_chembl_id")
    calibration_family = calibration_target.set_index("target_chembl_id")["target_assay_family"]
    target_table["target_assay_family"] = target_table["target_chembl_id"].map(calibration_family).fillna(
        target_table["project_target_assay_family"]
    )
    target_table["model_family_route"] = np.where(
        target_table["target_chembl_id"].isin(set(calibration_target["target_chembl_id"])),
        "EXACT_TRAINING_TARGET_FAMILY",
        "PROJECT_FAMILY_FOR_TRAINING_UNSEEN_TARGET",
    )
    pocket_branch = pairs[["target_chembl_id", "active_target_branch"]].drop_duplicates().set_index(
        "target_chembl_id"
    )["active_target_branch"]
    target_table["active_target_branch"] = target_table["target_chembl_id"].map(pocket_branch)
    target_table = target_table.sort_values("target_chembl_id").reset_index(drop=True)
    target_table.insert(0, "target_feature_index", np.arange(len(target_table), dtype=np.int16))
    target_features = np.empty((len(target_table), 1024), dtype=np.float32)
    routes = []
    with (
        h5py.File(TRAINING_PROTBERT_H5, "r") as training,
        h5py.File(TARGET_STRICT_H5, "r") as strict,
        h5py.File(TARGET_RECOVERED_H5, "r") as recovered,
    ):
        for row in target_table.itertuples(index=False):
            sequence = str(row.sequence)
            if sequence in training:
                value = np.asarray(training[sequence], dtype=np.float32)
                route = "EXACT_TRAINING_PROTBERT_CACHE"
            elif sequence in strict:
                value = np.asarray(strict[sequence], dtype=np.float32)
                route = "TRAINING_CACHE_MISS_STRICT_PROJECT_FALLBACK"
            elif sequence in recovered:
                value = np.asarray(recovered[sequence], dtype=np.float32)
                route = "TRAINING_CACHE_MISS_RECOVERED_PROJECT_FALLBACK"
            else:
                raise RuntimeError(f"Missing ProtBert embedding: {row.target_chembl_id}")
            if value.shape != (1024,) or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid ProtBert vector: {row.target_chembl_id}")
            target_features[int(row.target_feature_index)] = value
            routes.append(route)
    target_table["feature_route"] = routes
    target_table["embedding_l2_norm"] = np.linalg.norm(target_features, axis=1)

    drug_lookup = drug_table.set_index("ligand_inchikey")["drug_feature_index"]
    target_lookup = target_table.set_index("target_chembl_id")["target_feature_index"]
    family_lookup = target_table.set_index("target_chembl_id")["target_assay_family"]
    accession_lookup = target_table.set_index("target_chembl_id")["uniprot_accession"]
    pairs["drug_feature_index"] = pairs["ligand_inchikey"].map(drug_lookup).astype(np.int16)
    pairs["target_feature_index"] = pairs["target_chembl_id"].map(target_lookup).astype(np.int16)
    pairs["target_assay_family"] = pairs["target_chembl_id"].map(family_lookup)
    pairs["query_accession"] = pairs["target_chembl_id"].map(accession_lookup)

    drug_npy = OUT / "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
    target_npy = OUT / "PROJECT384_PROTBERT1024_FLOAT32_V1.npy"
    drug_csv = OUT / "OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz"
    target_csv = OUT / "PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz"
    pair_csv = OUT / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
    np.save(drug_npy, drug_features, allow_pickle=False)
    np.save(target_npy, target_features, allow_pickle=False)
    drug_table.to_csv(drug_csv, index=False)
    target_table.to_csv(target_csv, index=False)
    pairs.to_csv(pair_csv, index=False)
    feature_counts = target_table["feature_route"].value_counts().to_dict()
    pocket_counts = target_table["active_target_branch"].value_counts().to_dict()
    family_route_counts = target_table["model_family_route"].value_counts().to_dict()
    checks = {
        "exact_720x384_unique_pairs": len(pairs) == 720 * 384 and pairs["pairId"].is_unique,
        "all_720_exact_morgan_features": len(drug_table) == 720 and (drug_table["on_bits"] > 0).all(),
        "all_384_exact_protbert_features": len(target_table) == 384 and np.isfinite(target_features).all(),
        "strict338_plus_recovered46_exact": pocket_counts.get("STRICT_EXPERIMENTAL_POCKET_MAINLINE_338", 0) == 338 and pocket_counts.get("RECOVERED_NO_EXPERIMENTAL_POCKET_46", 0) == 46,
        "training_protbert_exact382_project_fallback2": feature_counts.get("EXACT_TRAINING_PROTBERT_CACHE", 0) == 382 and sum(value for key, value in feature_counts.items() if "PROJECT_FALLBACK" in key) == 2,
        "training_family_exact357_unseen_fallback27": family_route_counts.get("EXACT_TRAINING_TARGET_FAMILY", 0) == 357 and family_route_counts.get("PROJECT_FAMILY_FOR_TRAINING_UNSEEN_TARGET", 0) == 27,
        "all_feature_indices_complete": pairs[["drug_feature_index", "target_feature_index"]].notna().all().all(),
        "all_six_training_families_mapped": set(target_table["target_assay_family"]) <= {"enzyme", "kinase", "ion_channel", "nuclear_epigenetic", "transporter", "other_assayable"},
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Only the frozen 720 old drugs x active 384 targets; the 480 hard-gate-excluded targets remain excluded",
        "counts": {
            "pairs": int(len(pairs)), "old_drugs": int(len(drug_table)), "targets": int(len(target_table)),
            "strict_experimental_pocket_targets": pocket_counts.get("STRICT_EXPERIMENTAL_POCKET_MAINLINE_338", 0),
            "recovered_only_no_experimental_pocket_targets": pocket_counts.get("RECOVERED_NO_EXPERIMENTAL_POCKET_46", 0),
            "exact_training_protbert_cache_targets": feature_counts.get("EXACT_TRAINING_PROTBERT_CACHE", 0),
            "project_embedding_fallback_targets": sum(value for key, value in feature_counts.items() if "PROJECT_FALLBACK" in key),
            "exact_training_family_targets": family_route_counts.get("EXACT_TRAINING_TARGET_FAMILY", 0),
            "training_unseen_family_fallback_targets": family_route_counts.get("PROJECT_FAMILY_FOR_TRAINING_UNSEEN_TARGET", 0),
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [drug_npy, target_npy, drug_csv, target_csv, pair_csv]
        },
    }
    summary_path = OUT / "DEPLOYMENT_FEATURE_STORE_AUDIT_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

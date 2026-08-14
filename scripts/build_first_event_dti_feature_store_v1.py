#!/usr/bin/env python3
"""Build the label-audited OFER-DTI event manifest and packed feature store."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs/biomaster_first_event_dti_phase_a_freeze_20260814.json"
FULL = ROOT / "outputs/current_production_package_v2/chembl37_target_calibration_v5/PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz"
TARGET_INDEX = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
TARGET_EMBEDDINGS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/PROTBERT1024_FLOAT32_V1.npy"
OLD_DRUGS = ROOT / "configs/project_drugs_v4.csv"
OUT = ROOT / "outputs/old_drug_target_sota_v1/first_event_dti_feature_store_v1"

EXPECTED_FREEZE_SHA256 = "b11f437ebb20e285e4e3ccafb612dff297ffa86cbb5df900b7ea898b29a4dc13"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def morgan_packed(smiles: str) -> tuple[str, np.ndarray] | tuple[None, None]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    raw = np.frombuffer(DataStructs.BitVectToBinaryText(fingerprint), dtype=np.uint8).copy()
    if raw.shape != (256,):
        raise ValueError(f"Unexpected packed Morgan shape {raw.shape}")
    return canonical, raw


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise ValueError("Frozen OFER-DTI config hash changed")
    OUT.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGET_INDEX)
    target_map = dict(zip(targets["sequence_key"].astype(str), targets["target_feature_index"].astype(int)))
    target_embeddings = np.load(TARGET_EMBEDDINGS, mmap_mode="r")
    if target_embeddings.shape != (428, 1024) or len(target_map) != 428:
        raise ValueError("Frozen target feature store does not contain exactly 428 embeddings")

    usecols = [
        "sequence_key",
        "primary_gene",
        "target_chembl_id",
        "parent_molecule_chembl_id",
        "parent_standard_inchi_key",
        "parent_canonical_smiles",
        "min_document_year",
        "max_document_year",
        "calibration_label",
        "numeric_positive_negative_conflict",
        "explicit_inactive_positive_conflict",
    ]
    frame = pd.read_csv(FULL, usecols=usecols, low_memory=False)
    frame = frame[frame["sequence_key"].astype(str).isin(target_map)].copy()
    frame["first_year"] = pd.to_numeric(frame["min_document_year"], errors="coerce")
    label_ok = frame["calibration_label"].isin(["positive", "negative_or_inactive"])
    no_conflict = ~(
        frame["numeric_positive_negative_conflict"].fillna(False).astype(bool)
        | frame["explicit_inactive_positive_conflict"].fillna(False).astype(bool)
    )
    valid_year = frame["first_year"].between(1900, 2025, inclusive="both")
    nonmissing_chemistry = frame["parent_standard_inchi_key"].notna() & frame["parent_canonical_smiles"].notna()
    frame = frame[label_ok & no_conflict & valid_year & nonmissing_chemistry].copy()
    duplicate_pair = frame.duplicated(["sequence_key", "parent_standard_inchi_key"], keep=False)
    duplicate_rows_excluded = int(duplicate_pair.sum())
    frame = frame[~duplicate_pair].copy()
    frame["first_year"] = frame["first_year"].astype(int)
    frame["event_type"] = np.where(frame["calibration_label"].eq("positive"), "ACTIVE", "WEAK_OR_INACTIVE")
    frame["target_feature_index"] = frame["sequence_key"].map(target_map).astype(int)

    old = pd.read_csv(OLD_DRUGS, usecols=["base_chembl_id", "drug_name"]).drop_duplicates("base_chembl_id")
    old_name = dict(zip(old["base_chembl_id"].astype(str), old["drug_name"].astype(str)))
    frame["is_deployment_old_drug"] = frame["parent_molecule_chembl_id"].astype(str).isin(old_name)

    compounds = (
        frame[["parent_standard_inchi_key", "parent_canonical_smiles"]]
        .sort_values(["parent_standard_inchi_key", "parent_canonical_smiles"], kind="mergesort")
        .drop_duplicates("parent_standard_inchi_key")
        .reset_index(drop=True)
    )
    packed_rows: list[np.ndarray] = []
    canonical_rows: list[str | None] = []
    valid_rows: list[bool] = []
    for index, smiles in enumerate(compounds["parent_canonical_smiles"].astype(str), start=1):
        canonical, packed = morgan_packed(smiles)
        valid = packed is not None
        valid_rows.append(valid)
        canonical_rows.append(canonical)
        packed_rows.append(packed if valid else np.zeros(256, dtype=np.uint8))
        if index % 25000 == 0:
            print(f"fingerprints={index}/{len(compounds)}", flush=True)
    compounds["model_ligand_smiles"] = canonical_rows
    compounds["rdkit_parse_ok"] = valid_rows
    compounds["compound_feature_index"] = np.arange(len(compounds), dtype=np.int32)
    fingerprints = np.stack(packed_rows).astype(np.uint8, copy=False)

    compound_map = dict(
        zip(compounds["parent_standard_inchi_key"].astype(str), compounds["compound_feature_index"].astype(int))
    )
    frame["compound_feature_index"] = frame["parent_standard_inchi_key"].astype(str).map(compound_map).astype(int)
    frame["rdkit_parse_ok"] = frame["compound_feature_index"].map(
        compounds.set_index("compound_feature_index")["rdkit_parse_ok"]
    )
    invalid_event_rows = int((~frame["rdkit_parse_ok"]).sum())
    frame = frame[frame["rdkit_parse_ok"]].copy()

    # These are raw, globally indexed events only. Fold-specific entry years and
    # risk sets must be recomputed from the allowed historical prefix.
    frame = frame.sort_values(
        ["first_year", "parent_standard_inchi_key", "sequence_key"], kind="mergesort"
    ).reset_index(drop=True)
    frame["event_index"] = np.arange(len(frame), dtype=np.int64)
    frame["event_pair_id"] = (
        "OFER37_" + frame["sequence_key"].astype(str) + "_" + frame["parent_standard_inchi_key"].astype(str)
    )
    frame["deployment_old_drug_name"] = frame["parent_molecule_chembl_id"].astype(str).map(old_name).fillna("")

    event_columns = [
        "event_index",
        "event_pair_id",
        "sequence_key",
        "primary_gene",
        "target_chembl_id",
        "target_feature_index",
        "parent_molecule_chembl_id",
        "parent_standard_inchi_key",
        "compound_feature_index",
        "first_year",
        "max_document_year",
        "event_type",
        "is_deployment_old_drug",
        "deployment_old_drug_name",
    ]
    events_path = OUT / "OFER_DTI_FIRST_EVENT_MANIFEST_V1.csv.gz"
    frame[event_columns].to_csv(events_path, index=False, compression="gzip")

    compound_first_year = frame.groupby("compound_feature_index")["first_year"].min()
    compound_event_count = frame.groupby("compound_feature_index").size()
    compounds["first_observed_year"] = compounds["compound_feature_index"].map(compound_first_year)
    compounds["event_count"] = compounds["compound_feature_index"].map(compound_event_count).fillna(0).astype(int)
    compound_ids = (
        frame.sort_values(["compound_feature_index", "first_year"], kind="mergesort")
        .drop_duplicates("compound_feature_index")
        .set_index("compound_feature_index")
    )
    compounds["representative_molecule_chembl_id"] = compounds["compound_feature_index"].map(
        compound_ids["parent_molecule_chembl_id"]
    )
    compounds["is_deployment_old_drug"] = compounds["representative_molecule_chembl_id"].astype(str).isin(old_name)
    compounds["deployment_old_drug_name"] = compounds["representative_molecule_chembl_id"].astype(str).map(old_name).fillna("")
    compounds_path = OUT / "OFER_DTI_COMPOUND_FEATURE_INDEX_V1.csv.gz"
    compounds.to_csv(compounds_path, index=False, compression="gzip")
    fingerprints_path = OUT / "OFER_DTI_MORGAN2048_PACKED_UINT8_V1.npy"
    np.save(fingerprints_path, fingerprints, allow_pickle=False)

    year_counts = (
        frame.groupby(["first_year", "event_type"], as_index=False)
        .agg(events=("event_index", "size"), compounds=("compound_feature_index", "nunique"), targets=("target_feature_index", "nunique"))
    )
    year_counts_path = OUT / "OFER_DTI_EVENT_COUNTS_BY_YEAR_V1.csv"
    year_counts.to_csv(year_counts_path, index=False)

    summary = {
        "schema_version": "OFER_DTI_FEATURE_STORE_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": {
            "events": int(len(frame)),
            "active_events": int(frame["event_type"].eq("ACTIVE").sum()),
            "weak_or_inactive_events": int(frame["event_type"].eq("WEAK_OR_INACTIVE").sum()),
            "compounds_total_indexed": int(len(compounds)),
            "compounds_with_valid_fingerprint": int(compounds["rdkit_parse_ok"].sum()),
            "targets": int(frame["target_feature_index"].nunique()),
            "deployment_old_drugs_with_events": int(
                frame.loc[frame["is_deployment_old_drug"], "parent_molecule_chembl_id"].nunique()
            ),
            "duplicate_event_rows_excluded": duplicate_rows_excluded,
            "invalid_chemistry_event_rows_excluded": invalid_event_rows,
        },
        "feature_shapes": {
            "packed_morgan": list(fingerprints.shape),
            "unpacked_morgan_bits": 2048,
            "target_protbert": list(target_embeddings.shape),
        },
        "integrity": {
            "freeze_hash_matches": True,
            "event_pair_ids_unique": bool(frame["event_pair_id"].is_unique),
            "feature_indices_in_range": bool(
                frame["compound_feature_index"].between(0, len(compounds) - 1).all()
                and frame["target_feature_index"].between(0, 427).all()
            ),
            "both_event_types_present": frame["event_type"].nunique() == 2,
            "unobserved_pairs_labeled_inactive": False,
            "fold_specific_risk_sets_precomputed_globally": False,
        },
        "inputs": {
            str(FREEZE.relative_to(ROOT)): sha256(FREEZE),
            str(FULL.relative_to(ROOT)): sha256(FULL),
            str(TARGET_INDEX.relative_to(ROOT)): sha256(TARGET_INDEX),
            str(TARGET_EMBEDDINGS.relative_to(ROOT)): sha256(TARGET_EMBEDDINGS),
            str(OLD_DRUGS.relative_to(ROOT)): sha256(OLD_DRUGS),
        },
        "artifacts": {
            str(events_path.relative_to(ROOT)): sha256(events_path),
            str(compounds_path.relative_to(ROOT)): sha256(compounds_path),
            str(fingerprints_path.relative_to(ROOT)): sha256(fingerprints_path),
            str(year_counts_path.relative_to(ROOT)): sha256(year_counts_path),
        },
    }
    summary_path = OUT / "OFER_DTI_FEATURE_STORE_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the uncapped ChEMBL37 training package used by deployment FULL_FIT.

The historical 86,674-row table is a scaffold-balanced ConPLex calibration
panel, not a complete relation graph.  This builder starts from the audited
509,172-row ChEMBL37 aggregate, retains every feature-resolved positive or
negative relation, and then appends the existing affinity-only BindingDB and
deployment-recovery rows.  No per-target or per-label cap is applied.

Molecules are standardized with the same largest-fragment/uncharging contract
as the frozen calibration builder.  Invalid or structure-less molecules are
quarantined explicitly instead of receiving silent zero features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "outputs/current_production_package_v2/chembl37_target_calibration_v5/"
    "PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz"
)
BASE_TARGET_INDEX = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
BINDINGDB = ROOT / (
    "outputs/biomaster_bindingdb_affinity_feature_package_v1/"
    "BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz"
)
RECOVERED = ROOT / "outputs/biomaster_deployment_augmentation_v1/RECOVERED_CHEMBL37_RELATIONS_V1.csv.gz"
DEPLOYMENT_PAIRS = ROOT / (
    "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1/"
    "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_comprehensive_training_v1"


_FP_GENERATOR = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standardize_and_fingerprint(smiles: str) -> tuple[str, str, bytes, int, str]:
    """Return active-moiety SMILES, scaffold, packed Morgan bits, count, error."""

    global _FP_GENERATOR
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError("RDKit MolFromSmiles returned None")
        fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
        parent = max(fragments, key=lambda value: value.GetNumHeavyAtoms()) if fragments else molecule
        try:
            parent = rdMolStandardize.Uncharger().uncharge(parent)
        except Exception:
            pass
        active = Chem.MolToSmiles(parent, isomericSmiles=True)
        if not active:
            raise ValueError("empty active-moiety SMILES")
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
        except Exception:
            scaffold = ""
        if not scaffold:
            scaffold = active
        if _FP_GENERATOR is None:
            _FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprint = _FP_GENERATOR.GetFingerprintAsNumPy(parent).astype(np.uint8, copy=False)
        packed = np.packbits(fingerprint).tobytes()
        return active, scaffold, packed, int(fingerprint.sum()), ""
    except Exception as error:
        return "", "", b"", 0, f"{type(error).__name__}: {error}"


def standardize_inputs(values: list[str], workers: int) -> pd.DataFrame:
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    rows: list[tuple[str, str, bytes, int, str]] = []
    if workers <= 1:
        for number, value in enumerate(values, start=1):
            rows.append(_standardize_and_fingerprint(value))
            if number % 10000 == 0:
                print(json.dumps({"stage": "molecule_features", "completed": number, "total": len(values)}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_standardize_and_fingerprint, values, chunksize=256)
            for number, row in enumerate(iterator, start=1):
                rows.append(row)
                if number % 10000 == 0:
                    print(json.dumps({"stage": "molecule_features", "completed": number, "total": len(values)}), flush=True)
    result = pd.DataFrame(
        rows,
        columns=["model_ligand_smiles", "murcko_scaffold", "packed_fingerprint", "on_bits", "failure_reason"],
    )
    result.insert(0, "input_smiles", values)
    result["feature_available"] = result["model_ligand_smiles"].ne("")
    return result


def add_exact_key(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["exact_pair_key"] = (
        result["target_chembl_id"].fillna("").astype(str)
        + "__"
        + result["parent_standard_inchi_key"].fillna("").astype(str)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    required = [SOURCE, BASE_TARGET_INDEX, BINDINGDB, RECOVERED, DEPLOYMENT_PAIRS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    source_all = pd.read_csv(SOURCE, low_memory=False)
    source = source_all.loc[
        source_all["calibration_label"].isin(["positive", "negative_or_inactive"])
    ].copy()
    source["binary_label"] = source["calibration_label"].eq("positive").astype(np.int8)
    source["binary_observed"] = 1
    source["source_kind"] = "chembl37_comprehensive"
    source["augmentation_role"] = "comprehensive"
    source["calibration_pair_id"] = (
        "FULL37_" + source["sequence_key"].astype(str) + "_" + source["parent_molecule_chembl_id"].astype(str)
    )
    if source["calibration_pair_id"].duplicated().any():
        raise RuntimeError("comprehensive ChEMBL pair identifiers are not unique")

    targets = pd.read_csv(BASE_TARGET_INDEX, low_memory=False)
    if len(targets) != 428 or targets["sequence_key"].duplicated().any():
        raise RuntimeError("base target index contract changed")
    targets["sequence_sha256"] = targets["protein_sequence"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    source = source.merge(
        targets[["sequence_key", "target_feature_index", "sequence_sha256"]],
        on="sequence_key",
        how="left",
        validate="many_to_one",
    )
    if source[["target_feature_index", "sequence_sha256"]].isna().any().any():
        raise RuntimeError("comprehensive ChEMBL rows failed target-index mapping")
    source["target_sequence_hash"] = source["sequence_sha256"].astype(str)

    bindingdb = pd.read_csv(BINDINGDB, low_memory=False)
    bindingdb["source_kind"] = "bindingdb_affinity_only"
    bindingdb["augmentation_role"] = "bindingdb"
    bindingdb["binary_observed"] = 0
    if "target_sequence_hash" not in bindingdb:
        bindingdb["target_sequence_hash"] = bindingdb["external_target_hash"].astype(str)

    recovered = pd.read_csv(RECOVERED, low_memory=False)
    recovered["source_kind"] = "chembl37_recovered"
    recovered["binary_observed"] = 1
    recovered["target_sequence_hash"] = recovered["sequence_sha256"].astype(str)

    source_input = source["parent_canonical_smiles"].fillna("").astype(str)
    bindingdb_input = bindingdb["model_ligand_smiles"].fillna("").astype(str)
    recovered_input = recovered["model_ligand_smiles"].fillna("").astype(str)
    unique_inputs = sorted((set(source_input) | set(bindingdb_input) | set(recovered_input)) - {""})
    print(json.dumps({
        "stage": "molecule_features_start",
        "unique_input_smiles": len(unique_inputs),
        "workers": args.workers,
    }), flush=True)
    standardized = standardize_inputs(unique_inputs, args.workers)
    lookup = standardized.set_index("input_smiles")["model_ligand_smiles"]
    source["model_ligand_smiles"] = source_input.map(lookup).fillna("")
    bindingdb["model_ligand_smiles"] = bindingdb_input.map(lookup).fillna("")
    recovered["model_ligand_smiles"] = recovered_input.map(lookup).fillna("")

    input_failures = standardized.loc[~standardized["feature_available"]].drop(
        columns=["packed_fingerprint"]
    )
    source_feature_failure = source["model_ligand_smiles"].eq("")
    bindingdb_feature_failure = bindingdb["model_ligand_smiles"].eq("")
    recovered_feature_failure = recovered["model_ligand_smiles"].eq("")
    source = source.loc[~source_feature_failure].copy()
    bindingdb = bindingdb.loc[~bindingdb_feature_failure].copy()
    recovered = recovered.loc[~recovered_feature_failure].copy()

    feature_rows = standardized.loc[standardized["feature_available"]].drop_duplicates(
        "model_ligand_smiles", keep="first"
    ).sort_values("model_ligand_smiles", kind="stable").reset_index(drop=True)
    feature_rows.insert(0, "drug_feature_index", np.arange(len(feature_rows), dtype=np.int32))
    feature_path = out / "MORGAN2048_UINT8_COMPREHENSIVE_V1.npy"
    feature_matrix = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.uint8, shape=(len(feature_rows), 2048)
    )
    for start in range(0, len(feature_rows), 10000):
        stop = min(start + 10000, len(feature_rows))
        packed = feature_rows.iloc[start:stop]["packed_fingerprint"]
        for offset, value in enumerate(packed, start=start):
            feature_matrix[offset] = np.unpackbits(
                np.frombuffer(value, dtype=np.uint8), count=2048
            )
        feature_matrix.flush()
        print(json.dumps({"stage": "matrix_write", "completed": stop, "total": len(feature_rows)}), flush=True)
    del feature_matrix
    feature_lookup = feature_rows.set_index("model_ligand_smiles")
    drug_lookup = feature_lookup["drug_feature_index"]
    scaffold_lookup = feature_lookup["murcko_scaffold"]
    for frame in [source, bindingdb, recovered]:
        frame["drug_feature_index"] = frame["model_ligand_smiles"].map(drug_lookup).astype(np.int32)
        # The relation table is the sampler's authoritative input.  Keeping the
        # standardized scaffold only in DRUG_FEATURE_INDEX makes an uncapped
        # relation table silently fall back to row-frequency sampling.
        frame["murcko_scaffold"] = frame["model_ligand_smiles"].map(scaffold_lookup).fillna("")
        frame["drug_feature_available"] = True
        frame["conplex_score"] = 0.0

    source = add_exact_key(source)
    bindingdb = add_exact_key(bindingdb)
    recovered = add_exact_key(recovered)
    comprehensive_keys = set(source["exact_pair_key"])
    recovered["duplicate_of_comprehensive"] = recovered["exact_pair_key"].isin(comprehensive_keys)
    source["duplicate_of_comprehensive"] = False
    bindingdb["duplicate_of_comprehensive"] = False

    # Retain a uniform schema while preserving every source-specific audit column.
    data = pd.concat([source, bindingdb, recovered], ignore_index=True, sort=False)
    data["target_feature_index"] = data["target_feature_index"].astype(np.int32)
    data["binary_label"] = data["binary_label"].astype(np.int8)
    data["binary_observed"] = data["binary_observed"].astype(np.int8)
    data_path = out / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
    data.to_csv(data_path, index=False, compression="gzip")

    feature_index_path = out / "DRUG_FEATURE_INDEX_COMPREHENSIVE_V1.csv.gz"
    feature_rows.drop(columns=["packed_fingerprint"]).to_csv(
        feature_index_path, index=False, compression="gzip"
    )
    failure_path = out / "MOLECULE_FEATURE_FAILURES_V1.csv.gz"
    input_failures.to_csv(failure_path, index=False, compression="gzip")

    deployment = pd.read_csv(
        DEPLOYMENT_PAIRS, usecols=["ligand_inchikey", "drug_names"], low_memory=False
    ).drop_duplicates("ligand_inchikey")
    deployment_keys = set(deployment["ligand_inchikey"].astype(str))
    source_720 = source[source["parent_standard_inchi_key"].astype(str).isin(deployment_keys)]
    full_binary = pd.concat([source, recovered.loc[~recovered["duplicate_of_comprehensive"]]], ignore_index=True)
    positive_warm_720 = set(
        full_binary.loc[full_binary["binary_label"].eq(1), "parent_standard_inchi_key"].astype(str)
    ) & deployment_keys
    any_warm_720 = set(full_binary["parent_standard_inchi_key"].astype(str)) & deployment_keys
    tepotinib_key = "AHYMHWXQRWRBKT-UHFFFAOYSA-N"
    tepotinib_met = source.loc[
        source["parent_standard_inchi_key"].eq(tepotinib_key)
        & source["target_chembl_id"].eq("CHEMBL3717")
        & source["binary_label"].eq(1)
    ]
    checks = {
        "no_per_target_label_cap": bool(
            source.groupby(["sequence_key", "binary_label"]).size().max() > 150
        ),
        "all_428_base_targets_present": source["sequence_key"].nunique() == 428,
        "comprehensive_source_rows_above_426k": len(source) > 426000,
        "comprehensive_positive_rows_above_330k": int(source["binary_label"].sum()) > 330000,
        "tepotinib_met_positive_present": len(tepotinib_met) == 1,
        "all_feature_indices_in_bounds": int(data["drug_feature_index"].max()) < len(feature_rows),
        "all_relations_have_standardized_scaffold": bool(data["murcko_scaffold"].astype(str).ne("").all()),
        "calibration_pair_ids_unique_within_source": source["calibration_pair_id"].is_unique,
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_utc": now(),
        "protocol": "UNCAPPED_CHEMBL37_COMPREHENSIVE_TRAINING_V1",
        "counts": {
            "source_rows_before_label_filter": len(source_all),
            "source_eligible_rows_before_feature_filter": int(
                source_all["calibration_label"].isin(["positive", "negative_or_inactive"]).sum()
            ),
            "source_rows_feature_resolved": len(source),
            "source_positive_rows": int(source["binary_label"].sum()),
            "source_negative_rows": int((source["binary_label"] == 0).sum()),
            "source_feature_failures": int(source_feature_failure.sum()),
            "bindingdb_rows": len(bindingdb),
            "recovered_rows": len(recovered),
            "recovered_duplicate_of_comprehensive": int(recovered["duplicate_of_comprehensive"].sum()),
            "combined_rows_with_audit_duplicates": len(data),
            "unique_model_smiles": len(feature_rows),
            "source_720_rows": len(source_720),
            "source_720_positive_rows": int(source_720["binary_label"].sum()),
            "any_warm_720_after_comprehensive": len(any_warm_720),
            "positive_warm_720_after_comprehensive": len(positive_warm_720),
        },
        "checks": checks,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "artifacts": {
            "relations": str(data_path.relative_to(ROOT)),
            "morgan": str(feature_path.relative_to(ROOT)),
            "drug_index": str(feature_index_path.relative_to(ROOT)),
            "failures": str(failure_path.relative_to(ROOT)),
        },
    }
    summary_path = out / "COMPREHENSIVE_TRAINING_MANIFEST_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

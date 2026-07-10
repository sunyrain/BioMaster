#!/usr/bin/env python3
"""Freeze the exact v4 drug and target entity manifests used by the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/current_pipeline_v4.yaml")
    parser.add_argument("--drug-output", default="configs/project_drugs_v4.csv")
    parser.add_argument("--target-output", default="configs/project_targets_v4.csv")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    old_pool = pd.read_csv(ROOT / config["inputs"]["enhanced_discovery_pool"], low_memory=False).fillna("")
    drug_library = pd.read_csv(ROOT / config["inputs"]["drug_library"], low_memory=False).fillna("")
    target_extension = pd.read_csv(ROOT / config["inputs"]["target_scope_extension"], low_memory=False).fillna("")
    extra_drugs = set(config.get("scope", {}).get("extra_direct_action_drug_ids", []))

    legacy_drugs = set(old_pool["drug_chembl_id"].astype(str))
    drug_ids = sorted(legacy_drugs | extra_drugs)
    drug_meta = drug_library.drop_duplicates("drug_id").set_index("drug_id")
    drugs = pd.DataFrame({"drug_chembl_id": drug_ids})
    for source, target in [
        ("drug_name", "drug_name"),
        ("chembl_id", "base_chembl_id"),
        ("model_ligand_smiles", "model_ligand_smiles"),
        ("therapeutic_area", "therapeutic_area"),
        ("route", "route"),
    ]:
        drugs[target] = drugs["drug_chembl_id"].map(drug_meta[source].to_dict()).fillna("")
    drugs["scope_source"] = drugs["drug_chembl_id"].map(
        lambda value: "v4_explicit_scope_extension" if value in extra_drugs else "legacy_direct_action_project"
    )
    drugs["inclusion_reason"] = drugs["scope_source"].map(
        {
            "v4_explicit_scope_extension": "valid_small_or_macrocyclic_direct_action_boundary_restore",
            "legacy_direct_action_project": "passed_direct_action_drug_feasibility_rules",
        }
    )

    legacy_target_cols = [
        "sequence_key",
        "primary_gene",
        "representative_protein_id",
        "target_assay_family",
        "structure_bin",
        "pdb_path",
    ]
    targets = old_pool[[column for column in legacy_target_cols if column in old_pool.columns]].drop_duplicates(
        "sequence_key"
    )
    targets["scope_source"] = "legacy_direct_action_project"
    targets["inclusion_reason"] = "passed_direct_small_molecule_target_and_structure_rules"
    extension = target_extension.copy()
    extension["scope_source"] = "v4_explicit_scope_extension"
    extension["inclusion_reason"] = "restored_after_independent_pocket_consensus_audit"
    for column in targets.columns:
        if column not in extension.columns:
            extension[column] = ""
    targets = pd.concat([targets, extension[targets.columns]], ignore_index=True).drop_duplicates(
        "sequence_key", keep="last"
    )
    targets = targets.sort_values(["sequence_key", "primary_gene"], kind="mergesort").reset_index(drop=True)

    if len(drugs) != 750 or drugs["drug_chembl_id"].duplicated().any():
        raise ValueError(f"Drug manifest contract failed: {len(drugs)} rows")
    if len(targets) != 463 or targets["sequence_key"].duplicated().any():
        raise ValueError(f"Target manifest contract failed: {len(targets)} rows")
    if drugs["model_ligand_smiles"].astype(str).eq("").any():
        raise ValueError("Frozen drug manifest contains missing model ligand SMILES")
    if targets["primary_gene"].astype(str).eq("").any():
        raise ValueError("Frozen target manifest contains missing genes")

    drug_output = ROOT / args.drug_output
    target_output = ROOT / args.target_output
    drugs.to_csv(drug_output, index=False)
    targets.to_csv(target_output, index=False)
    summary = {
        "drug_rows": len(drugs),
        "target_rows": len(targets),
        "drug_manifest_sha256": sha256(drug_output),
        "target_manifest_sha256": sha256(target_output),
        "explicit_drug_extensions": sorted(extra_drugs),
        "explicit_target_extensions": sorted(set(target_extension["sequence_key"].astype(str))),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

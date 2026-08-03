#!/usr/bin/env python3
"""Build the physical-pair universe and remote DTA inference manifests.

This stage does not predict affinity and does not rank candidates by chemical
similarity. Similarity to ChEMBL actives is used only as a rediscovery veto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from annotate_compound_assay_liability import molecule_annotations


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "affinity_first_remote_discovery_v1.yaml"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def join_unique(values: pd.Series) -> str:
    return ";".join(sorted({clean(value) for value in values if clean(value)}))


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def make_physical_pair_id(smiles: str, sequence_key: str, gene: str) -> str:
    ligand_hash = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:12]
    safe_gene = "".join(character if character.isalnum() else "_" for character in gene)
    return f"PHY_{ligand_hash}_{safe_gene}_{sequence_key}"


def load_target_metadata(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    requested = [
        "sequence_key",
        "representative_protein_id",
        "pdb_path",
        "top_pocket_rank",
        "top_pocket_score",
        "top_pocket_probability",
        "top_pocket_center_x",
        "top_pocket_center_y",
        "top_pocket_center_z",
        "top_pocket_residue_ids",
        "p2rank26_top_pocket_volume",
        "p2rank26_top_pocket_num_residues",
        "p2rank_pocketability_tier",
        "puresnet_tier",
        "structure_consensus_tier",
        "strict_structure_tier",
        "p2rank_puresnet_overlap_fraction",
        "p2rank_puresnet_jaccard",
        "receptor_residue_count",
        "experimental_feasibility_score",
        "anchor_canonical_uniprot",
        "anchor_target_class_labels",
        "anchor_availability_tier",
        "anchor_sm_structure_with_ligand",
        "anchor_sm_high_quality_ligand",
        "anchor_sm_high_quality_pocket",
        "anchor_project_standard_direct_sm",
    ]
    usecols = [column for column in requested if column in header]
    target = pd.read_csv(path, usecols=usecols, low_memory=False).fillna("")
    return target.sort_values("sequence_key", kind="mergesort").drop_duplicates("sequence_key")


def collapse_qsar(path: Path) -> pd.DataFrame:
    qsar = pd.read_csv(path, low_memory=False)
    required = {"model_ligand_smiles", "sequence_key"}
    if not required.issubset(qsar.columns):
        raise ValueError(f"QSAR table lacks {sorted(required - set(qsar.columns))}")
    grouped = qsar.groupby(["model_ligand_smiles", "sequence_key"], sort=False, dropna=False)
    result = grouped.agg(
        target_qsar_probability_v5=("target_qsar_probability_v5", "max"),
        max_known_active_similarity_v5=("max_known_active_similarity_v5", "max"),
        max_known_compound_similarity_v5=("max_known_compound_similarity_v5", "max"),
        same_known_active_scaffold_v5=("same_known_active_scaffold_v5", "max"),
        exact_known_active_smiles_v5=("exact_known_active_smiles_v5", "max"),
        target_ligand_model_status_v5=("target_ligand_model_status_v5", join_unique),
        temporal_validation_status_v5=("temporal_validation_status_v5", join_unique),
        ligand_applicability_v5=("ligand_applicability_v5", join_unique),
    ).reset_index()
    return result


def build(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {name: ROOT / value for name, value in config["inputs"].items()}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    source = pd.read_csv(paths["full_project_universe"], low_memory=False).fillna("")
    expected_source = int(config["contracts"]["source_rows"])
    if len(source) != expected_source:
        raise ValueError(f"Source row contract failed: {len(source)} != {expected_source}")
    if source.duplicated(["drug_chembl_id", "sequence_key"]).any():
        raise ValueError("Source table has duplicate drug ID x target rows")
    if source["model_ligand_smiles"].astype(str).eq("").any():
        raise ValueError("Source table contains missing model ligand structures")

    source = source.sort_values(
        ["model_ligand_smiles", "sequence_key", "drug_chembl_id"], kind="mergesort"
    ).reset_index(drop=True)
    group_columns = ["model_ligand_smiles", "sequence_key"]
    grouped = source.groupby(group_columns, sort=False, dropna=False)
    score_spread = grouped["conplex_score"].agg(lambda values: float(values.max()) - float(values.min()))
    if (score_spread > 1e-8).any():
        raise ValueError("Equivalent model ligands have inconsistent ConPLEx scores")

    representative = source.drop_duplicates(group_columns, keep="first").copy()
    group_annotations = grouped.agg(
        source_drug_ids=("drug_chembl_id", join_unique),
        source_drug_names=("drug_names", join_unique),
        source_drug_count=("drug_chembl_id", "nunique"),
        source_therapeutic_areas=("fda_therapeutic_area", join_unique),
        source_indications=("fda_indication", join_unique),
        source_label_targets=("fda_target_names", join_unique),
        source_action_types=("fda_action_type", join_unique),
        any_known_fda_target_pair=("is_known_fda_target_pair", "max"),
        any_family_or_rediscovery_risk_v2=("family_or_rediscovery_risk_v2", "max"),
        grouped_risk_notes_v2=("risk_notes_v2", join_unique),
    ).reset_index()
    physical = representative.drop(
        columns=[
            "is_known_fda_target_pair",
            "family_or_rediscovery_risk_v2",
            "risk_notes_v2",
        ],
        errors="ignore",
    ).merge(group_annotations, on=group_columns, how="left", validate="one_to_one")
    physical["physical_pair_id"] = [
        make_physical_pair_id(smiles, sequence, gene)
        for smiles, sequence, gene in zip(
            physical["model_ligand_smiles"], physical["sequence_key"], physical["primary_gene"]
        )
    ]
    physical["rank_within_active_moiety"] = physical.groupby("model_ligand_smiles")["conplex_score"].rank(
        method="average", ascending=False
    )
    physical["rank_within_target_active_collapsed"] = physical.groupby("sequence_key")["conplex_score"].rank(
        method="average", ascending=False
    )

    target_scope = pd.read_csv(paths["project_targets"], low_memory=False).fillna("")
    target_integrity = pd.read_csv(paths["target_integrity"], low_memory=False).fillna("")
    sequences = pd.read_csv(paths["protein_sequences"], low_memory=False).fillna("")
    sequence_meta = sequences[["sequence_key", "length", "sequence"]].drop_duplicates("sequence_key")
    target_metadata = load_target_metadata(paths["target_metadata_pool"])
    target = target_scope.merge(target_metadata, on="sequence_key", how="left", suffixes=("", "_legacy"))
    target = target.merge(
        target_integrity.drop(columns=["primary_gene", "representative_protein_id", "pdb_path"], errors="ignore"),
        on="sequence_key",
        how="left",
        validate="one_to_one",
    ).merge(sequence_meta, on="sequence_key", how="left", validate="one_to_one")
    physical = physical.merge(
        target.drop(columns=["primary_gene"], errors="ignore"),
        on="sequence_key",
        how="left",
        suffixes=("", "_target"),
        validate="many_to_one",
    )

    qsar = collapse_qsar(paths["target_qsar_predictions"])
    physical = physical.merge(qsar, on=group_columns, how="left", validate="one_to_one")
    physical["has_target_ligand_reference"] = physical["max_known_active_similarity_v5"].notna()
    physical["same_known_active_scaffold_v5"] = as_bool(physical["same_known_active_scaffold_v5"])
    physical["exact_known_active_smiles_v5"] = as_bool(physical["exact_known_active_smiles_v5"])
    physical["any_known_fda_target_pair"] = as_bool(physical["any_known_fda_target_pair"])
    physical["any_family_or_rediscovery_risk_v2"] = as_bool(
        physical["any_family_or_rediscovery_risk_v2"]
    )

    ligand_metadata = physical[["model_ligand_smiles"]].drop_duplicates().copy()
    annotations = pd.DataFrame(
        [molecule_annotations(smiles) for smiles in ligand_metadata["model_ligand_smiles"]],
        index=ligand_metadata.index,
    )
    ligand_metadata = pd.concat([ligand_metadata, annotations], axis=1)
    physical = physical.merge(ligand_metadata, on="model_ligand_smiles", how="left", validate="many_to_one")

    similarity_max = float(config["gates"]["remote_active_similarity_max"])
    known_pair = physical["any_known_fda_target_pair"] | physical["exact_known_active_smiles_v5"]
    close_known_chemistry = (
        physical["has_target_ligand_reference"]
        & (
            (pd.to_numeric(physical["max_known_active_similarity_v5"], errors="coerce") >= similarity_max)
            | physical["same_known_active_scaffold_v5"]
        )
    )
    family_risk = physical["any_family_or_rediscovery_risk_v2"]
    severe_liability = as_bool(physical["severe_compound_liability"])
    physical["remote_novelty_status_v1"] = "remote_no_target_ligand_reference"
    physical.loc[
        physical["has_target_ligand_reference"] & ~close_known_chemistry,
        "remote_novelty_status_v1",
    ] = "remote_below_similarity_veto"
    physical.loc[close_known_chemistry, "remote_novelty_status_v1"] = "exclude_close_known_target_chemistry"
    physical.loc[family_risk, "remote_novelty_status_v1"] = "exclude_known_target_family_or_rediscovery"
    physical.loc[known_pair, "remote_novelty_status_v1"] = "known_positive_or_rediscovery_control"

    strict_bins = set(config["gates"]["strict_structure_bins"])
    review_bins = set(config["gates"]["review_structure_bins"])
    physical["strict_structure_ready_v1"] = physical["structure_bin"].isin(strict_bins)
    physical["review_structure_ready_v1"] = physical["structure_bin"].isin(review_bins)
    remote_pair = ~known_pair & ~close_known_chemistry & ~family_risk & ~severe_liability
    physical["remote_pair_eligible_v1"] = remote_pair
    physical["dta_stage1_strict_eligible_v1"] = remote_pair & physical["strict_structure_ready_v1"]
    physical["dta_stage1_review_eligible_v1"] = remote_pair & physical["review_structure_ready_v1"]
    physical["dta_queue_v1"] = "hold_structure_or_novelty"
    physical.loc[remote_pair, "dta_queue_v1"] = "remote_sequence_dta_only"
    physical.loc[
        physical["dta_stage1_review_eligible_v1"], "dta_queue_v1"
    ] = "remote_structure_review"
    physical.loc[
        physical["dta_stage1_strict_eligible_v1"], "dta_queue_v1"
    ] = "remote_structure_strict"
    physical.loc[known_pair, "dta_queue_v1"] = "known_positive_calibration"
    physical.loc[close_known_chemistry & ~known_pair, "dta_queue_v1"] = "similarity_rediscovery_control"
    physical.loc[family_risk & ~known_pair, "dta_queue_v1"] = "family_extension_control"

    expected_physical = int(config["contracts"]["physical_pair_rows"])
    if len(physical) != expected_physical:
        raise ValueError(f"Physical pair contract failed: {len(physical)} != {expected_physical}")
    if physical["physical_pair_id"].duplicated().any():
        raise ValueError("Physical pair IDs are not unique")
    if physical["model_ligand_smiles"].nunique() != int(config["contracts"]["model_ligands"]):
        raise ValueError("Model ligand contract failed")
    if physical["sequence_key"].nunique() != int(config["contracts"]["targets"]):
        raise ValueError("Target contract failed")

    out_dir = ROOT / config["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "PHYSICAL_PAIR_UNIVERSE_334749_V1.csv.gz"
    strict_path = out_dir / "DTA_STAGE1_REMOTE_STRICT_STRUCTURE_V1.csv.gz"
    review_path = out_dir / "DTA_STAGE1_REMOTE_REVIEW_STRUCTURE_V1.csv.gz"
    target_path = out_dir / "TARGET_MODEL_READINESS_463_V1.csv"
    ligand_path = out_dir / "LIGAND_MODEL_READINESS_723_V1.csv"
    physical.to_csv(master_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    physical.loc[physical["dta_stage1_strict_eligible_v1"]].to_csv(
        strict_path, index=False, compression={"method": "gzip", "compresslevel": 5}
    )
    physical.loc[physical["dta_stage1_review_eligible_v1"]].to_csv(
        review_path, index=False, compression={"method": "gzip", "compresslevel": 5}
    )
    target.to_csv(target_path, index=False)
    ligand_metadata.to_csv(ligand_path, index=False)

    summary = {
        "status": "passed",
        "created_utc": now_utc(),
        "source_rows": int(len(source)),
        "physical_pair_rows": int(len(physical)),
        "model_ligands": int(physical["model_ligand_smiles"].nunique()),
        "targets": int(physical["sequence_key"].nunique()),
        "known_or_exact_active_controls": int(known_pair.sum()),
        "close_known_target_chemistry_rows": int((close_known_chemistry & ~known_pair).sum()),
        "family_or_rediscovery_control_rows": int((family_risk & ~known_pair).sum()),
        "severe_liability_rows": int(severe_liability.sum()),
        "target_ligand_reference_coverage_rows": int(physical["has_target_ligand_reference"].sum()),
        "target_ligand_reference_coverage_targets": int(
            physical.loc[physical["has_target_ligand_reference"], "sequence_key"].nunique()
        ),
        "remote_pair_eligible_rows": int(remote_pair.sum()),
        "dta_stage1_strict_rows": int(physical["dta_stage1_strict_eligible_v1"].sum()),
        "dta_stage1_review_rows": int(physical["dta_stage1_review_eligible_v1"].sum()),
        "queue_counts": physical["dta_queue_v1"].value_counts().to_dict(),
        "structure_target_counts": target["structure_bin"].value_counts().to_dict(),
        "policy": {
            "similarity_role": "veto_only_not_ranking",
            "conplex_role": "existing_sequence_DTA_channel_not_hard_gate",
            "disease_role": "not_used_in_affinity_stage",
            "pending_before_final_selection": [
                "expanded_known-target homology audit",
                "experimental holo structure and binding-site mapping",
                "DrugCLIP pocket retrieval",
                "target-calibrated GNINA/Vina docking",
                "Boltz-2 full-protocol refinement",
                "pose geometry and interaction audit",
            ],
        },
        "outputs": {
            "master": str(master_path.relative_to(ROOT)),
            "strict_dta_manifest": str(strict_path.relative_to(ROOT)),
            "review_dta_manifest": str(review_path.relative_to(ROOT)),
            "targets": str(target_path.relative_to(ROOT)),
            "ligands": str(ligand_path.relative_to(ROOT)),
        },
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    summary_path = out_dir / "AFFINITY_FIRST_REMOTE_DISCOVERY_V1_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    build(Path(args.config).resolve())


if __name__ == "__main__":
    main()

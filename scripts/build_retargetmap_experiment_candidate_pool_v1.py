#!/usr/bin/env python3
"""Build the frozen candidate pool for the first ReTargetMap wet-lab comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/retargetmap_experiment_20260901"
POOL_OUT = OUT_DIR / "RETARGETMAP_EXPERIMENT_PRELIMINARY_POOL_V1.csv.gz"
TARGET_ROSTER_OUT = OUT_DIR / "RETARGETMAP_TARGET_ASSAY_FEASIBILITY_ROSTER_V1.csv"
SUMMARY_OUT = OUT_DIR / "RETARGETMAP_EXPERIMENT_PRELIMINARY_POOL_SUMMARY_V1.json"

RANK_PATH = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
SCORE_PATH = ROOT / "outputs/retrain_20260901/bidirectional_720x745_full_fit_rows4/BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_SCORES_V1.csv.gz"
TARGET_PATH = ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_PRIMARY_DIRECT_SM_ASSAYABLE_450_V3.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_bucket(row: pd.Series) -> str:
    r_rank = int(row["s5_anchored_rank_within_drug_384"])
    d_rank = int(row["dtiam_rank_within_drug_384"])
    b_rank = int(row["current_default_borda_rank_within_drug_384"])
    f_rank = int(row["full_fit_directional_rank_within_drug_384"])
    if r_rank <= 10 and d_rank <= 10:
        return "S5_ANCHORED_DTIAM_CONSENSUS_TOP10"
    if r_rank <= 10 and d_rank > 30:
        return "S5_ANCHORED_ONLY_TOP10"
    if d_rank <= 10 and r_rank > 30:
        return "DTIAM_ONLY_TOP10"
    if b_rank <= 10 and r_rank > 30 and d_rank > 30:
        return "CURRENT_DEFAULT_BORDA_ONLY_TOP10"
    if f_rank <= 10 and r_rank > 30 and d_rank > 30:
        return "FULL_FIT_EXPLORATORY_ONLY_TOP10"
    if r_rank > 200 and d_rank > 200 and b_rank > 200 and f_rank > 200:
        return "FOUR_SCORE_LOW_BACKGROUND"
    return "INTERMEDIATE"


def main() -> None:
    rank = pd.read_csv(RANK_PATH)
    score_columns = [
        "ligand_inchikey",
        "ligand_smiles",
        "target_chembl_id",
        "uniprot_accession",
        "assay_lane",
        "production_route",
        "target_model_warmth",
        "structure_evidence_route",
        "current_score_status",
        "binary_observed",
        "binary_label",
        "mean_pchembl",
        "structure_mask",
    ]
    score = pd.read_csv(SCORE_PATH, usecols=score_columns)
    target_columns = [
        "target_chembl_id",
        "target_name",
        "target_class_l1",
        "target_calibration_tier",
        "calibration_8x8",
        "structure_ready_permissive",
        "structure_ready_strict",
        "p2rank_tier",
        "positive_compounds",
        "negative_compounds",
    ]
    target = pd.read_csv(TARGET_PATH, usecols=target_columns)

    merged = rank.merge(
        score,
        on=["ligand_inchikey", "target_chembl_id"],
        how="left",
        validate="one_to_one",
    ).merge(target, on="target_chembl_id", how="left", validate="many_to_one")
    if len(merged) != 720 * 384:
        raise RuntimeError(f"Expected 276,480 deployment rows, found {len(merged):,}")

    merged["dtiam_rank_within_drug_384"] = (
        merged.groupby("ligand_inchikey", sort=False)["dtiam_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    merged["s5_anchored_rank_within_drug_384"] = (
        merged.groupby("ligand_inchikey", sort=False)["independent_validation_rank_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    merged = merged.rename(
        columns={
            "independent_validation_rank_score": "s5_anchored_independent_score",
            "biomaster_independent_borda_score": "current_default_borda_score",
            "biomaster_independent_borda_score_rank_within_drug_384": "current_default_borda_rank_within_drug_384",
            "ensemble_drug_to_target_logit": "full_fit_directional_score",
            "ensemble_drug_to_target_logit_rank_within_drug_384": "full_fit_directional_rank_within_drug_384",
        }
    )

    eligible = merged[
        (merged["current_score_status"] == "SCORED_IN_FROZEN_384_CORE")
        & (merged["production_route"] == "PRIMARY_BIOCHEMICAL_DIRECT_SM")
        & merged["assay_lane"].isin(
            ["ENZYME_BIOCHEMICAL", "KINASE_BIOCHEMICAL", "NUCLEAR_EPIGENETIC_DOMAIN"]
        )
        & (merged["target_model_warmth"] == "TARGET_WARM_IN_SUPERVISED_RELATIONS")
        & merged["calibration_8x8"].fillna(False)
        & (merged["binary_observed"].fillna(0).astype(int) == 0)
    ].copy()
    eligible["model_selection_bucket"] = eligible.apply(model_bucket, axis=1)
    eligible["exact_pair_training_status"] = "UNREPORTED_IN_FROZEN_TRAINING_RELATIONS"
    eligible["external_novelty_status"] = "REQUIRES_CURRENT_DATABASE_AND_LITERATURE_AUDIT"
    eligible["assay_procurement_status"] = "REQUIRES_TARGET_SPECIFIC_FEASIBILITY_AUDIT"
    eligible["compound_qc_status"] = "REQUIRES_IDENTITY_PURITY_SOLUBILITY_QC"
    eligible["final_experiment_status"] = "PRELIMINARY_POOL_NOT_YET_SELECTED"

    keep = [
        "pairId",
        "ligand_inchikey",
        "ligand_smiles",
        "drug_names",
        "target_chembl_id",
        "gene_symbol",
        "target_name",
        "uniprot_accession",
        "target_class_l1",
        "assay_lane",
        "target_calibration_tier",
        "positive_compounds",
        "negative_compounds",
        "structure_evidence_route",
        "structure_ready_permissive",
        "structure_ready_strict",
        "p2rank_tier",
        "s5_anchored_independent_score",
        "s5_anchored_rank_within_drug_384",
        "dtiam_probability",
        "dtiam_rank_within_drug_384",
        "current_default_borda_score",
        "current_default_borda_rank_within_drug_384",
        "full_fit_directional_score",
        "full_fit_directional_rank_within_drug_384",
        "model_selection_bucket",
        "exact_pair_training_status",
        "external_novelty_status",
        "assay_procurement_status",
        "compound_qc_status",
        "final_experiment_status",
    ]
    eligible = eligible[keep].sort_values(
        ["model_selection_bucket", "s5_anchored_rank_within_drug_384", "drug_names", "gene_symbol"],
        kind="mergesort",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(POOL_OUT, index=False, compression="gzip")
    target_identity = [
        "target_chembl_id",
        "gene_symbol",
        "target_name",
        "uniprot_accession",
        "target_class_l1",
        "assay_lane",
        "target_calibration_tier",
        "positive_compounds",
        "negative_compounds",
        "structure_evidence_route",
        "structure_ready_permissive",
        "structure_ready_strict",
        "p2rank_tier",
    ]
    target_roster = eligible.groupby("target_chembl_id", as_index=False)[target_identity[1:]].first()
    target_roster = target_roster.merge(
        eligible.groupby("target_chembl_id")["ligand_inchikey"]
        .nunique()
        .rename("eligible_drugs")
        .reset_index(),
        on="target_chembl_id",
        validate="one_to_one",
    )
    bucket_by_target = pd.crosstab(
        eligible["target_chembl_id"], eligible["model_selection_bucket"]
    ).add_prefix("pair_count__").reset_index()
    target_roster = target_roster.merge(
        bucket_by_target, on="target_chembl_id", how="left", validate="one_to_one"
    )
    for column in (
        "assay_available",
        "assay_provider_or_internal_platform",
        "assay_format_and_endpoint",
        "protein_or_service_catalogue",
        "reference_positive_control",
        "orthogonal_binding_method",
        "estimated_setup_cost",
        "estimated_per_plate_cost",
        "lead_time",
        "lab_priority",
        "lab_decision",
        "lab_notes",
    ):
        target_roster[column] = "REQUIRES_LAB_INPUT"
    target_roster = target_roster.sort_values(
        ["assay_lane", "target_chembl_id"], kind="mergesort"
    )
    target_roster.to_csv(TARGET_ROSTER_OUT, index=False)
    bucket_counts = {
        str(key): int(value)
        for key, value in eligible["model_selection_bucket"].value_counts().sort_index().items()
    }
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "RETARGETMAP_FIRST_WETLAB_PRELIMINARY_POOL_V1",
        "selection_scope": {
            "deployment_denominator": 384,
            "direction": "old_drug_to_target",
            "assay_routes": [
                "ENZYME_BIOCHEMICAL",
                "KINASE_BIOCHEMICAL",
                "NUCLEAR_EPIGENETIC_DOMAIN",
            ],
            "target_warm_only": True,
            "local_calibration_8x8_required": True,
            "exact_pair_observed_in_training": False,
            "unknown_pair_policy": "unknown_not_negative",
        },
        "counts": {
            "rows": int(len(eligible)),
            "drugs": int(eligible["ligand_inchikey"].nunique()),
            "targets": int(eligible["target_chembl_id"].nunique()),
            "model_selection_buckets": bucket_counts,
        },
        "model_roles": {
            "primary_recommender": "s5_anchored_independent_score / rank_within_drug_384",
            "current_contract_default_challenger": "current_default_borda_score / rank_within_drug_384",
            "external_comparator": "dtiam_probability / derived rank_within_drug_384",
            "exploratory_support": "full_fit_directional_score / rank_within_drug_384",
            "not_used_as_independent_recommender": "system score containing DTIAM",
        },
        "claim_boundary": (
            "All rows are unreported in the frozen training relations, not proven novel. "
            "Current database/literature audit, assay feasibility, and compound QC are required "
            "before any pair enters the blinded experimental matrix."
        ),
        "artifacts": {
            "pool": str(POOL_OUT.relative_to(ROOT)),
            "target_assay_feasibility_roster": str(TARGET_ROSTER_OUT.relative_to(ROOT)),
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (RANK_PATH, SCORE_PATH, TARGET_PATH)
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

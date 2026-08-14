#!/usr/bin/env python3
"""Compute bidirectional, cross-model, physical-coverage and independence diagnostics for all 30 candidates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V17 = RUN / "final_evidence_routing_v17"
OUT = RUN / "full_candidate_portfolio_v18"
PROTOCOL = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.json"
STAMP = OUT / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.sha256"
PAIR = V17 / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz"
CASEBOOK = V17 / "PROSPECTIVE_INTEGRATED_CASEBOOK_V10.csv"
MASTER = V17 / "MASTER_VALIDATION_QUEUE_63_ROWS_V17.csv"


VIEWS = {
    "CONPLEX": "conplex_score",
    "DRUGCLIP": "drugclip_cosine_mean",
    "DTA_CONSENSUS": "dta_cross_target_consensus_score",
    "V10_BRANCH_PRIMARY": "branch_primary_score_v10",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def rank_record(
    pair: pd.DataFrame,
    candidate: pd.Series,
    direction: str,
    view: str,
    score_column: str,
) -> dict[str, object]:
    if direction == "TARGET_CENTERED_ACROSS_DRUGS":
        group = pair[pair["target_chembl_id"].eq(candidate["target_chembl_id"])]
        stored_rank_columns = {"CONPLEX": "conplex_rank_within_target", "DRUGCLIP": "drugclip_rank_within_target"}
        stored_percentile_columns = {
            "CONPLEX": "conplex_percentile_within_target", "DRUGCLIP": "drugclip_percentile_within_target",
        }
        comparison_scope = "same target across frozen 720 old drugs"
    elif direction == "DRUG_CENTERED_ACROSS_TARGETS":
        group = pair[pair["ligand_inchikey"].eq(candidate["ligand_inchikey"])]
        if view == "V10_BRANCH_PRIMARY":
            group = group[group["old_drug_target_deployment_branch_v10"].eq(
                candidate["old_drug_target_deployment_branch_v10"]
            )]
            comparison_scope = "same drug within frozen target deployment branch"
        else:
            comparison_scope = "same drug across frozen active targets with finite score"
        stored_rank_columns = {
            "CONPLEX": "conplex_rank_within_ligand_384", "DRUGCLIP": "drugclip_rank_within_ligand_382",
            "DTA_CONSENSUS": "dta_consensus_rank_within_drug_384_v7",
            "V10_BRANCH_PRIMARY": "branch_primary_rank_within_drug_v10",
        }
        stored_percentile_columns = {
            "CONPLEX": "conplex_percentile_within_ligand_384",
            "DRUGCLIP": "drugclip_percentile_within_ligand_382",
        }
    else:
        raise ValueError(direction)
    finite = group[pd.to_numeric(group[score_column], errors="coerce").notna()].copy()
    finite[score_column] = pd.to_numeric(finite[score_column], errors="coerce")
    candidate_mask = finite["ligand_inchikey"].eq(candidate["ligand_inchikey"]) & finite[
        "target_chembl_id"
    ].eq(candidate["target_chembl_id"])
    if candidate_mask.sum() != 1:
        raise RuntimeError(f"Candidate missing or duplicated for {direction} {view}")
    score = float(finite.loc[candidate_mask, score_column].iloc[0])
    ranks = finite[score_column].rank(method="average", ascending=False)
    min_tie_ranks = finite[score_column].rank(method="min", ascending=False)
    max_tie_ranks = finite[score_column].rank(method="max", ascending=False)
    rank = float(ranks.loc[candidate_mask].iloc[0])
    min_tie_rank = float(min_tie_ranks.loc[candidate_mask].iloc[0])
    max_tie_rank = float(max_tie_ranks.loc[candidate_mask].iloc[0])
    n = len(finite)
    percentile = float((n - rank) / (n - 1)) if n > 1 else np.nan
    median = float(finite[score_column].median())
    mad = float((finite[score_column] - median).abs().median())
    robust_z = float((score - median) / (1.4826 * mad)) if n >= 50 and mad > 0 else np.nan
    stored_rank_column = stored_rank_columns.get(view)
    stored_percentile_column = stored_percentile_columns.get(view)
    stored_rank = candidate[stored_rank_column] if stored_rank_column else np.nan
    stored_percentile = candidate[stored_percentile_column] if stored_percentile_column else np.nan
    if not stored_rank_column:
        stored_rank_status = "NOT_PREVIOUSLY_STORED"
    elif np.isclose(float(stored_rank), rank, rtol=0, atol=1e-9):
        stored_rank_status = "MATCH"
    elif np.isclose(float(stored_rank), min_tie_rank, rtol=0, atol=1e-9):
        stored_rank_status = "MATCH_HISTORICAL_MIN_TIE_WHILE_V18_REPORTS_AVERAGE_TIE"
    else:
        stored_rank_status = "MISMATCH"
    if stored_percentile_column:
        difference = abs(float(stored_percentile) - percentile)
        stored_percentile_status = "MATCH_WITHIN_1E-7_STORED_PRECISION" if difference <= 1e-7 else "MISMATCH_GT_1E-7"
    else:
        difference = np.nan
        stored_percentile_status = "NOT_PREVIOUSLY_STORED"
    return {
        "candidate_rank": int(candidate["v10_integrated_case_rank"]),
        "execution_wave": candidate["execution_wave"],
        "drug_name": candidate["drug_names"],
        "ligand_inchikey": candidate["ligand_inchikey"],
        "gene_symbol": candidate["gene_symbol"],
        "target_chembl_id": candidate["target_chembl_id"],
        "deployment_branch": candidate["old_drug_target_deployment_branch_v10"],
        "direction": direction,
        "model_view": view,
        "score_column": score_column,
        "candidate_score": score,
        "score_direction": "HIGHER_IS_BETTER",
        "frozen_comparison_universe_size": len(group),
        "finite_score_denominator": n,
        "missing_score_count": len(group) - n,
        "recomputed_average_tie_rank": rank,
        "recomputed_min_tie_rank": min_tie_rank,
        "recomputed_max_tie_rank": max_tie_rank,
        "ties_at_candidate_score": int(np.isclose(finite[score_column].to_numpy(float), score, rtol=0, atol=1e-15).sum()),
        "empirical_upper_tail_percentile": percentile,
        "comparison_median": median,
        "comparison_mad": mad,
        "robust_mad_z": robust_z,
        "robust_mad_z_status": "DEFINED" if pd.notna(robust_z) else (
            "UNDEFINED_MAD_ZERO" if mad == 0 else "UNDEFINED_FINITE_N_LT_50"
        ),
        "top_10_percent": rank <= math.ceil(0.10 * n),
        "top_20_percent": rank <= math.ceil(0.20 * n),
        "directionally_supportive_ge_80th_percentile": percentile >= 0.80,
        "stored_rank_column": stored_rank_column or "NONE",
        "stored_rank": stored_rank,
        "stored_rank_recalculation_status": stored_rank_status,
        "stored_percentile_column": stored_percentile_column or "NONE",
        "stored_percentile": stored_percentile,
        "stored_percentile_absolute_difference": difference,
        "stored_percentile_recalculation_status": stored_percentile_status,
        "comparison_scope": comparison_scope,
        "claim_boundary": "Internal selected-data sensitivity metric; not an independent validation or calibrated p-value.",
    }


def physical_rank(group: pd.DataFrame, candidate: pd.Series, column: str, ascending: bool) -> tuple[float, int, int]:
    finite = group[pd.to_numeric(group[column], errors="coerce").notna()].copy()
    finite[column] = pd.to_numeric(finite[column], errors="coerce")
    candidate_mask = finite["ligand_inchikey"].eq(candidate["ligand_inchikey"]) & finite[
        "target_chembl_id"
    ].eq(candidate["target_chembl_id"])
    if candidate_mask.sum() != 1:
        return np.nan, len(finite), 0
    score = float(finite.loc[candidate_mask, column].iloc[0])
    ranks = finite[column].rank(method="average", ascending=ascending)
    return (
        float(ranks.loc[candidate_mask].iloc[0]), len(finite),
        int(np.isclose(finite[column].to_numpy(float), score, rtol=0, atol=1e-15).sum()),
    )


def main() -> None:
    if sha256(PROTOCOL) != STAMP.read_text().split()[0]:
        raise RuntimeError("V18 protocol stamp mismatch")
    protocol = json.loads(PROTOCOL.read_text())
    for relative, expected in protocol["frozen_dependencies"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen dependency changed: {relative}")
    columns = [
        "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol", "assay_lane", "active_target_branch",
        "old_drug_target_deployment_branch_v10", "conplex_score", "conplex_rank_within_target",
        "conplex_percentile_within_target", "conplex_rank_within_ligand_384",
        "conplex_percentile_within_ligand_384", "drugclip_cosine_mean", "drugclip_rank_within_target",
        "drugclip_percentile_within_target", "drugclip_rank_within_ligand_382",
        "drugclip_percentile_within_ligand_382", "dta_cross_target_consensus_score",
        "dta_consensus_rank_within_drug_384_v7", "branch_primary_score_v10",
        "branch_primary_rank_within_drug_v10", "robust_coverage_score_v9",
        "robust_coverage_rank_within_drug_384_v9", "five_model_top20_support_count_v9",
        "main_boltz_boltz_completed_effective", "main_boltz_boltzAffinityProbabilityBinary",
        "main_boltz_boltzAffinityPredValue", "main_boltz_boltzConfidenceScore",
        "v3_boltz_support_count_3seed", "v3_multiseed_final_decision", "v3_receptor_ensemble_decision",
        "gnina_primary_completed_v5", "main_gnina_primary_cnn_affinity",
        "main_gnina_primary_pose_vina_affinity", "completed_in_all_target_qualified_models",
        "completed_in_all_remote_qualified_models_v5", "any_physical_pair_calculation_completed_v5",
        "final_pair_evidence_layer_v5", "chembl37_pair_record_class", "any_activity_rows",
        "strict_numeric_rows", "bindingdb_evidence_rows", "bindingdb_exact_pair_found_v9",
        "direct_pair_report_found", "external_rediscovery_control", "is_database_gap_rediscovery_control_v7",
        "manual_literature_class_v7", "external_relation_class_v7", "final_discovery_status_v7",
        "v10_target_lane_evidence_status", "v10_claim_boundary",
    ]
    pair = pd.read_csv(PAIR, usecols=columns, low_memory=False)
    casebook = pd.read_csv(CASEBOOK, low_memory=False)
    master = pd.read_csv(MASTER, low_memory=False)
    wave = master[master["validation_layer"].eq("L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS")][[
        "candidate_rank_context", "execution_wave", "molecule_inchikey", "target_chembl_id"
    ]].rename(columns={"candidate_rank_context": "v10_integrated_case_rank"})
    candidates = casebook[casebook["v10_integrated_case_rank"].between(1, 30)][[
        "ligand_inchikey", "target_chembl_id", "v10_integrated_case_rank", "external_exact_record_count",
        "known_target_component_count_v7", "target_family_relation_v7", "ligand_species_chembl_ids_v7",
    ]].merge(pair, on=["ligand_inchikey", "target_chembl_id"], validate="one_to_one").merge(
        wave[["v10_integrated_case_rank", "execution_wave"]], on="v10_integrated_case_rank", validate="one_to_one"
    ).sort_values("v10_integrated_case_rank")

    rank_rows = []
    for _, candidate in candidates.iterrows():
        for direction in ["TARGET_CENTERED_ACROSS_DRUGS", "DRUG_CENTERED_ACROSS_TARGETS"]:
            for view, column in VIEWS.items():
                rank_rows.append(rank_record(pair, candidate, direction, view, column))
    ranks = pd.DataFrame(rank_rows).sort_values(["candidate_rank", "direction", "model_view"])

    summary_rows = []
    for _, candidate in candidates.iterrows():
        rank = int(candidate["v10_integrated_case_rank"])
        subset = ranks[ranks["candidate_rank"].eq(rank)]
        target_subset = subset[subset["direction"].eq("TARGET_CENTERED_ACROSS_DRUGS")]
        drug_subset = subset[subset["direction"].eq("DRUG_CENTERED_ACROSS_TARGETS")]
        row = {
            "candidate_rank": rank, "execution_wave": candidate["execution_wave"],
            "drug_name": candidate["drug_names"], "gene_symbol": candidate["gene_symbol"],
            "target_chembl_id": candidate["target_chembl_id"],
            "deployment_branch": candidate["old_drug_target_deployment_branch_v10"],
            "target_lane_evidence_status": candidate["v10_target_lane_evidence_status"],
            "known_target_component_count": int(candidate["known_target_component_count_v7"]),
            "target_family_relation": candidate["target_family_relation_v7"],
            "directional_views_evaluated": 8,
            "top10_view_count_of_8": int(subset["top_10_percent"].sum()),
            "top20_view_count_of_8": int(subset["top_20_percent"].sum()),
            "supportive_ge80pct_view_count_of_8": int(subset["directionally_supportive_ge_80th_percentile"].sum()),
            "target_centered_supportive_count_of_4": int(target_subset["directionally_supportive_ge_80th_percentile"].sum()),
            "drug_centered_supportive_count_of_4": int(drug_subset["directionally_supportive_ge_80th_percentile"].sum()),
            "minimum_directional_percentile": float(subset["empirical_upper_tail_percentile"].min()),
            "maximum_directional_percentile": float(subset["empirical_upper_tail_percentile"].max()),
            "directional_percentile_range": float(subset["empirical_upper_tail_percentile"].max() - subset["empirical_upper_tail_percentile"].min()),
            "selection_bias_status": "INTERNAL_EVIDENCE_OVERLAPS_WITH_SELECTION_NOT_INDEPENDENT_VALIDATION",
            "rank_rule": "Retain frozen rank 1-30; no V18 reranking.",
        }
        row["rank_discordance_flag"] = "HIGH_RANGE_GE_0.50" if row["directional_percentile_range"] >= 0.50 else "LOWER_RANGE_LT_0.50"
        for direction, prefix in [
            ("TARGET_CENTERED_ACROSS_DRUGS", "target"), ("DRUG_CENTERED_ACROSS_TARGETS", "drug")
        ]:
            for view in VIEWS:
                match = subset[subset["direction"].eq(direction) & subset["model_view"].eq(view)].iloc[0]
                key = view.lower()
                row[f"{prefix}_{key}_rank"] = match["recomputed_average_tie_rank"]
                row[f"{prefix}_{key}_denominator"] = match["finite_score_denominator"]
                row[f"{prefix}_{key}_upper_tail_percentile"] = match["empirical_upper_tail_percentile"]
        summary_rows.append(row)
    candidate_summary = pd.DataFrame(summary_rows).sort_values("candidate_rank")

    concordance_rows = []
    target_lookup = ranks[ranks["direction"].eq("TARGET_CENTERED_ACROSS_DRUGS")].set_index(
        ["candidate_rank", "model_view"]
    )
    for _, candidate in candidates.iterrows():
        group = pair[pair["target_chembl_id"].eq(candidate["target_chembl_id"])]
        rank = int(candidate["v10_integrated_case_rank"])
        for left_view, right_view in itertools.combinations(VIEWS, 2):
            left_column, right_column = VIEWS[left_view], VIEWS[right_view]
            finite = group[[left_column, right_column]].apply(pd.to_numeric, errors="coerce").dropna()
            left = target_lookup.loc[(rank, left_view)]
            right = target_lookup.loc[(rank, right_view)]
            concordance_rows.append({
                "candidate_rank": rank, "execution_wave": candidate["execution_wave"],
                "drug_name": candidate["drug_names"], "gene_symbol": candidate["gene_symbol"],
                "target_chembl_id": candidate["target_chembl_id"],
                "deployment_branch": candidate["old_drug_target_deployment_branch_v10"],
                "left_model_view": left_view, "right_model_view": right_view,
                "finite_pairwise_drug_count": len(finite),
                "spearman_rho_across_drugs": float(finite[left_column].corr(finite[right_column], method="spearman")),
                "candidate_left_upper_tail_percentile": left["empirical_upper_tail_percentile"],
                "candidate_right_upper_tail_percentile": right["empirical_upper_tail_percentile"],
                "candidate_min_of_two_percentiles": min(
                    left["empirical_upper_tail_percentile"], right["empirical_upper_tail_percentile"]
                ),
                "candidate_both_top20": bool(left["top_20_percent"] and right["top_20_percent"]),
                "interpretation": "Selected target-lane model-dependence diagnostic; no independence or p-value claim.",
            })
    concordance = pd.DataFrame(concordance_rows).sort_values(
        ["candidate_rank", "left_model_view", "right_model_view"]
    )

    physical_rows = []
    for _, candidate in candidates.iterrows():
        group = pair[pair["target_chembl_id"].eq(candidate["target_chembl_id"])]
        boltz_rank, boltz_n, boltz_ties = physical_rank(
            group, candidate, "main_boltz_boltzAffinityProbabilityBinary", False
        )
        cnn_rank, cnn_n, cnn_ties = physical_rank(group, candidate, "main_gnina_primary_cnn_affinity", False)
        vina_rank, vina_n, vina_ties = physical_rank(group, candidate, "main_gnina_primary_pose_vina_affinity", True)
        main_boltz_completed = as_bool(candidate["main_boltz_boltz_completed_effective"])
        main_boltz_value_present = pd.notna(candidate["main_boltz_boltzAffinityProbabilityBinary"])
        multiseed_count = candidate["v3_boltz_support_count_3seed"]
        metadata_flag = "CONSISTENT_MAIN_AND_MULTISEED_METADATA"
        if not main_boltz_completed and pd.notna(multiseed_count) and float(multiseed_count) > 0:
            metadata_flag = "MULTISEED_SUPPORT_METADATA_WITHOUT_MAIN_RESULT_DO_NOT_IMPUTE"
        physical_rows.append({
            "candidate_rank": int(candidate["v10_integrated_case_rank"]),
            "execution_wave": candidate["execution_wave"], "drug_name": candidate["drug_names"],
            "gene_symbol": candidate["gene_symbol"], "target_chembl_id": candidate["target_chembl_id"],
            "assay_lane": candidate["assay_lane"],
            "deployment_branch": candidate["old_drug_target_deployment_branch_v10"],
            "boltz_primary_completed": main_boltz_completed,
            "boltz_primary_value_present": main_boltz_value_present,
            "boltz_affinity_probability_binary": candidate["main_boltz_boltzAffinityProbabilityBinary"],
            "boltz_affinity_pred_value": candidate["main_boltz_boltzAffinityPredValue"],
            "boltz_confidence_score": candidate["main_boltz_boltzConfidenceScore"],
            "boltz_support_count_3seed": multiseed_count,
            "boltz_multiseed_final_decision": candidate["v3_multiseed_final_decision"],
            "physical_metadata_consistency_flag": metadata_flag,
            "boltz_rank_within_selectively_computed_target_subset": boltz_rank,
            "boltz_target_subset_denominator": boltz_n, "boltz_ties_at_candidate_score": boltz_ties,
            "gnina_primary_completed": as_bool(candidate["gnina_primary_completed_v5"]),
            "gnina_primary_cnn_affinity": candidate["main_gnina_primary_cnn_affinity"],
            "gnina_primary_pose_vina_affinity": candidate["main_gnina_primary_pose_vina_affinity"],
            "gnina_cnn_rank_within_selectively_computed_target_subset": cnn_rank,
            "gnina_vina_rank_within_selectively_computed_target_subset": vina_rank,
            "gnina_target_subset_denominator": cnn_n, "gnina_vina_target_subset_denominator": vina_n,
            "gnina_cnn_ties_at_candidate_score": cnn_ties, "gnina_vina_ties_at_candidate_score": vina_ties,
            "receptor_ensemble_decision": candidate["v3_receptor_ensemble_decision"] if pd.notna(
                candidate["v3_receptor_ensemble_decision"]
            ) else "NOT_RUN_OR_NOT_APPLICABLE",
            "completed_in_all_target_qualified_models": as_bool(candidate["completed_in_all_target_qualified_models"]),
            "completed_in_all_remote_qualified_models_v5": as_bool(candidate["completed_in_all_remote_qualified_models_v5"]),
            "final_pair_evidence_layer_v5": candidate["final_pair_evidence_layer_v5"],
            "comparison_bias": "SELECTIVELY_COMPUTED_SUBSET_NOT_RANDOM_NEGATIVE_CONTROL_PANEL",
            "claim_boundary": "Physical rank is diagnostic only and does not establish binding or assay validation.",
        })
    physical = pd.DataFrame(physical_rows).sort_values("candidate_rank")

    independence_rows = []
    for _, candidate in candidates.iterrows():
        exact_external = (
            int(candidate["external_exact_record_count"]) > 0
            or as_bool(candidate["bindingdb_exact_pair_found_v9"])
            or as_bool(candidate["direct_pair_report_found"])
            or int(candidate["any_activity_rows"]) > 0
        )
        independence_rows.append({
            "candidate_rank": int(candidate["v10_integrated_case_rank"]),
            "execution_wave": candidate["execution_wave"], "drug_name": candidate["drug_names"],
            "gene_symbol": candidate["gene_symbol"], "target_chembl_id": candidate["target_chembl_id"],
            "deployment_branch": candidate["old_drug_target_deployment_branch_v10"],
            "target_lane_calibration_status_not_pair_evidence": candidate["v10_target_lane_evidence_status"],
            "chembl37_pair_record_class": candidate["chembl37_pair_record_class"],
            "chembl37_activity_rows": int(candidate["any_activity_rows"]),
            "chembl37_strict_numeric_rows": int(candidate["strict_numeric_rows"]),
            "bindingdb_exact_pair_found": as_bool(candidate["bindingdb_exact_pair_found_v9"]),
            "bindingdb_evidence_rows": int(candidate["bindingdb_evidence_rows"])
            if pd.notna(candidate["bindingdb_evidence_rows"]) else 0,
            "direct_pair_report_found": as_bool(candidate["direct_pair_report_found"]),
            "external_exact_record_count_casebook": int(candidate["external_exact_record_count"]),
            "manual_literature_class": candidate["manual_literature_class_v7"],
            "external_relation_class": candidate["external_relation_class_v7"],
            "exact_pair_external_evidence_present": exact_external,
            "independent_exact_pair_validation_status": (
                "EXACT_PAIR_EXTERNAL_EVIDENCE_PRESENT_REVIEW_REQUIRED"
                if exact_external else "NONE_IN_FROZEN_CHEMBL_BINDINGDB_AND_LITERATURE_AUDITS"
            ),
            "internal_selection_overlap": "YES_INTERNAL_DTA_OR_PHYSICAL_EVIDENCE_OVERLAPS_WITH_SELECTION",
            "interpretation": "Target-lane calibration is not exact-pair confirmation; ranks are internal sensitivity only.",
        })
    independence = pd.DataFrame(independence_rows).sort_values("candidate_rank")

    paths = {
        "ranks": OUT / "FULL30_BIDIRECTIONAL_MODEL_RANKS_240_V18.csv",
        "summary": OUT / "FULL30_BIDIRECTIONAL_ROBUSTNESS_SUMMARY_30_V18.csv",
        "concordance": OUT / "FULL30_TARGET_MODEL_CONCORDANCE_180_V18.csv",
        "physical": OUT / "FULL30_PHYSICAL_MODEL_COVERAGE_AND_RANKS_30_V18.csv",
        "independence": OUT / "FULL30_SELECTION_BIAS_AND_EVIDENCE_INDEPENDENCE_30_V18.csv",
    }
    ranks.to_csv(paths["ranks"], index=False)
    candidate_summary.to_csv(paths["summary"], index=False)
    concordance.to_csv(paths["concordance"], index=False)
    physical.to_csv(paths["physical"], index=False)
    independence.to_csv(paths["independence"], index=False)

    v16_ranks = pd.read_csv(V17 / "W1_BIDIRECTIONAL_MODEL_RANKS_64_V16.csv", low_memory=False).rename(
        columns={"w1_candidate_rank": "candidate_rank"}
    )
    v16_physical = pd.read_csv(V17 / "W1_PHYSICAL_MODEL_COVERAGE_AND_RANKS_8_V16.csv", low_memory=False).rename(
        columns={"w1_candidate_rank": "candidate_rank"}
    )
    v18_w1_ranks = ranks[ranks["candidate_rank"].between(1, 8)]
    rank_common = [column for column in v16_ranks.columns if column in v18_w1_ranks.columns]
    v18_w1_physical = physical[physical["candidate_rank"].between(1, 8)]
    physical_common = [column for column in v16_physical.columns if column in v18_w1_physical.columns]
    applicable_ranks = ranks[ranks["stored_rank_column"].ne("NONE")]
    applicable_percentiles = ranks[ranks["stored_percentile_column"].ne("NONE")]
    checks = {
        "protocol_stamp_and_dependencies_verified": True,
        "pair_core_exact_276480_720_by_384": (
            len(pair) == 276480 and pair["ligand_inchikey"].nunique() == 720
            and pair["target_chembl_id"].nunique() == 384
        ),
        "exact_30_frozen_candidate_pairs_ranks_1_to_30": (
            len(candidates) == 30 and candidates["v10_integrated_case_rank"].astype(int).tolist() == list(range(1, 31))
        ),
        "candidate_population_exact_28_entities_16_targets": (
            candidates["ligand_inchikey"].nunique() == 28 and candidates["target_chembl_id"].nunique() == 16
        ),
        "wave_partition_exact_8_21_1": candidates["execution_wave"].value_counts().to_dict() == {
            "W2_CONTINGENT_ONLY": 21, "W1_BLINDED_CANDIDATE_PILOT": 8, "VETO_NOT_AUTHORIZED": 1,
        },
        "deployment_partition_exact_19_seeded_11_unseeded": candidates[
            "old_drug_target_deployment_branch_v10"
        ].value_counts().to_dict() == {"SEEDED_KNOWN_GRAPH_185": 19, "UNSEEDED_TARGET_DTA_199": 11},
        "bidirectional_rank_table_exact_30_by_2_by_4": len(ranks) == 240,
        "all_target_centered_comparison_universes_exact_720": ranks.loc[
            ranks["direction"].eq("TARGET_CENTERED_ACROSS_DRUGS"), "frozen_comparison_universe_size"
        ].eq(720).all(),
        "drug_centered_denominators_exact_384_382_and_branch_185_199": (
            set(ranks.loc[
                (ranks["direction"].eq("DRUG_CENTERED_ACROSS_TARGETS")) & ranks["model_view"].eq("CONPLEX"),
                "finite_score_denominator",
            ]) == {384}
            and set(ranks.loc[
                (ranks["direction"].eq("DRUG_CENTERED_ACROSS_TARGETS")) & ranks["model_view"].eq("DRUGCLIP"),
                "finite_score_denominator",
            ]) == {382}
            and set(ranks.loc[
                (ranks["direction"].eq("DRUG_CENTERED_ACROSS_TARGETS")) & ranks["model_view"].eq("V10_BRANCH_PRIMARY"),
                "finite_score_denominator",
            ]) == {185, 199}
        ),
        "all_stored_ranks_and_percentiles_recomputed_match_declared_tie_rule": (
            applicable_ranks["stored_rank_recalculation_status"].str.startswith("MATCH").all()
            and applicable_percentiles["stored_percentile_recalculation_status"].str.startswith("MATCH").all()
        ),
        "exact_3_historical_min_tie_rank_rows_transparently_identified": (
            ranks["stored_rank_recalculation_status"].eq(
                "MATCH_HISTORICAL_MIN_TIE_WHILE_V18_REPORTS_AVERAGE_TIE"
            ).sum() == 3
            and set(ranks.loc[
                ranks["stored_rank_recalculation_status"].eq(
                    "MATCH_HISTORICAL_MIN_TIE_WHILE_V18_REPORTS_AVERAGE_TIE"
                ), "candidate_rank"
            ]) == {12, 19, 23}
        ),
        "w1_rank_rows_reproduce_v16_common_fields_exact": all(
            (
                np.allclose(
                    pd.to_numeric(v16_ranks[column], errors="coerce"),
                    pd.to_numeric(v18_w1_ranks[column], errors="coerce"),
                    rtol=0, atol=1e-12, equal_nan=True,
                )
                if pd.api.types.is_numeric_dtype(v16_ranks[column])
                and pd.api.types.is_numeric_dtype(v18_w1_ranks[column])
                else v16_ranks[column].fillna("").astype(str).reset_index(drop=True).equals(
                    v18_w1_ranks[column].fillna("").astype(str).reset_index(drop=True)
                )
            ) for column in rank_common
        ),
        "candidate_summary_exact_30_no_new_order": (
            len(candidate_summary) == 30 and candidate_summary["candidate_rank"].tolist() == list(range(1, 31))
            and candidate_summary["rank_rule"].str.contains("no V18 reranking", case=False).all()
        ),
        "target_model_concordance_exact_30_by_6": (
            len(concordance) == 180 and concordance.groupby("candidate_rank").size().eq(6).all()
        ),
        "physical_table_exact_30_with_29_boltz_and_10_gnina": (
            len(physical) == 30 and physical["boltz_primary_completed"].sum() == 29
            and physical["boltz_primary_value_present"].sum() == 29 and physical["gnina_primary_completed"].sum() == 10
        ),
        "ketoconazole_labeled_rank21_missing_main_physical_not_imputed": (
            physical.loc[physical["candidate_rank"].eq(21), "physical_metadata_consistency_flag"].tolist()
            == ["MULTISEED_SUPPORT_METADATA_WITHOUT_MAIN_RESULT_DO_NOT_IMPUTE"]
            and physical.loc[physical["candidate_rank"].eq(21), "boltz_affinity_probability_binary"].isna().all()
            and physical.loc[physical["candidate_rank"].eq(21), "gnina_primary_cnn_affinity"].isna().all()
        ),
        "w1_physical_rows_reproduce_v16_common_fields_exact": v16_physical[physical_common].reset_index(drop=True).equals(
            v18_w1_physical[physical_common].reset_index(drop=True)
        ),
        "independence_table_exact_30_zero_external_exact_pair_validations": (
            len(independence) == 30 and ~independence["exact_pair_external_evidence_present"].any()
        ),
        "all_30_internal_selection_overlap_explicit": independence["internal_selection_overlap"].eq(
            "YES_INTERNAL_DTA_OR_PHYSICAL_EVIDENCE_OVERLAPS_WITH_SELECTION"
        ).all(),
        "claim_boundaries_forbid_independent_binding_or_w2_authorization_overstatement": (
            any("not independent validation" in item for item in protocol["selection_bias_and_claim_boundaries"])
            and any("establishes binding" in item for item in protocol["selection_bias_and_claim_boundaries"])
            and any("do not authorize procurement" in item for item in protocol["selection_bias_and_claim_boundaries"])
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "candidate_pairs": 30, "unique_entities": 28, "unique_targets": 16,
            "w1": 8, "w2": 21, "veto": 1, "seeded": 19, "unseeded": 11,
            "bidirectional_rank_rows": len(ranks), "target_model_concordance_rows": len(concordance),
            "boltz_primary_completed": int(physical["boltz_primary_completed"].sum()),
            "gnina_primary_completed": int(physical["gnina_primary_completed"].sum()),
            "high_rank_discordance_candidates": int(candidate_summary["rank_discordance_flag"].eq("HIGH_RANGE_GE_0.50").sum()),
            "robust_mad_z_defined": int(ranks["robust_mad_z_status"].eq("DEFINED").sum()),
            "robust_mad_z_undefined_mad_zero": int(ranks["robust_mad_z_status"].eq("UNDEFINED_MAD_ZERO").sum()),
            "external_exact_pair_validations": 0,
        },
        "claim_boundaries": protocol["selection_bias_and_claim_boundaries"],
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [PROTOCOL, PAIR, CASEBOOK, MASTER]},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
    }
    summary_path = OUT / "FULL30_COMPUTATIONAL_ROBUSTNESS_SUMMARY_V18.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": summary["status"], "checks_passed": sum(checks.values()), "checks_total": len(checks),
        **summary["counts"], "summary_sha256": sha256(summary_path),
    }, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        print(json.dumps({key: value for key, value in checks.items() if not value}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

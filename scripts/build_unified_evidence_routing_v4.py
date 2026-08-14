#!/usr/bin/env python3
"""Build a complete 720 x 384 pair-evidence routing table.

V3 intentionally contained only the physical-evidence union and its adaptive
follow-up.  V4 starts from every pair in the frozen 384-target universe, then
adds both successful and unsuccessful physical calculations.  This prevents
"not in V3" from being misread as "not computed".
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UNIFIED = ROOT / "outputs/unified_pair_program_720x384_v1"
EXEC = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
OUT = EXEC / "final_evidence_routing_v4"

PAIR_MATRIX = UNIFIED / "UNIFIED_DTA_720_X_384_PAIR_MATRIX_V1.csv.gz"
TARGET_MASTER = UNIFIED / "UNIFIED_ACTIVE_TARGET_MASTER_384_V1.csv"
MAIN_TARGET_ROUTES = EXEC / "final_evidence_routing_v2/TARGET_EVIDENCE_LAYER_ROUTING_338_V2.csv"
RECOVERED_TARGETS = (
    ROOT
    / "outputs/recovered_target_program_integrated_v1"
    / "RECOVERED_46_INTEGRATED_TARGET_OUTCOMES_V1.csv"
)
MAIN_BOLTZ = (
    ROOT
    / "outputs/current_results_classified_20260806_v1"
    / "CURRENT_RESULTS_MASTER_14488_CLASSIFIED_V1.csv.gz"
)
MAIN_GNINA = (
    ROOT
    / "outputs/gnina_discovery_7511_v1/evaluation"
    / "GNINA_DISCOVERY_7511_TARGET_CALIBRATED_EVIDENCE_V1.csv.gz"
)
V3 = EXEC / "final_evidence_routing_v3/PAIR_EVIDENCE_LAYER_ROUTING_CURRENT_V3.csv.gz"
RECOVERED_GNINA = (
    ROOT
    / "outputs/recovered_gnina_candidate_docking_v1"
    / "RECOVERED_GNINA_CANDIDATE_EVIDENCE_V1.csv"
)
RECOVERED_BOLTZ = (
    ROOT
    / "outputs/recovered_boltz2_loxl2_candidates_v1"
    / "RECOVERED_BOLTZ2_LOXL2_PAIR_SUMMARY_V1.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"true", "1", "1.0", "yes"}
    )


def merge_prefixed(
    base: pd.DataFrame,
    source: pd.DataFrame,
    columns: list[str],
    prefix: str,
    key: list[str],
) -> pd.DataFrame:
    selected = key + [column for column in columns if column in source.columns]
    annotation = source[selected].drop_duplicates(key).rename(
        columns={column: f"{prefix}{column}" for column in selected if column not in key}
    )
    return base.merge(annotation, on=key, how="left", validate="one_to_one")


def make_target_routing(target_master: pd.DataFrame) -> pd.DataFrame:
    main = pd.read_csv(MAIN_TARGET_ROUTES, low_memory=False)
    main_cols = [
        "target_chembl_id",
        "current_target_route",
        "current_target_route_zh",
        "current_qualified_models",
        "current_allowed_scope_zh",
        "current_target_qualification_strength",
        "current_target_state",
        "target_evidence_layer",
        "modality_compute_branch",
        "next_compute_action_zh_v2",
        "gnina_target_qualification",
        "boltz_target_qualification",
    ]
    main = main[[column for column in main_cols if column in main.columns]].copy()
    main = main.rename(
        columns={
            "current_target_route": "unified_target_route",
            "current_target_route_zh": "unified_target_route_zh",
            "current_qualified_models": "qualified_physical_models",
            "current_allowed_scope_zh": "allowed_pair_scope_zh",
            "next_compute_action_zh_v2": "target_next_compute_action_zh",
        }
    )
    main["target_route_family"] = "EXPERIMENTAL_POCKET_MAINLINE"
    main["predicted_pocket_qualification"] = "NOT_APPLICABLE_EXPERIMENTAL_POCKET"
    main["discovery_authorization"] = "AS_DEFINED_BY_E1_TO_E6_ROUTE"
    main["target_gate_pass_for_discovery"] = main["unified_target_route"].isin(
        {
            "E1_DUAL_REMOTE_QUALIFIED",
            "E2_SINGLE_MODEL_REMOTE_QUALIFIED",
            "E3_LOCAL_ONLY_QUALIFIED",
        }
    )
    main["remote_qualified_physical_models"] = main.apply(
        lambda row: "+".join(
            model
            for model, passed in [
                ("Boltz-2", str(row.get("boltz_target_qualification", "")) == "BOLTZ_REMOTE_QUALIFIED"),
                ("GNINA", str(row.get("gnina_target_qualification", "")) == "REMOTE_STRONG"),
            ]
            if passed
        )
        or "NONE",
        axis=1,
    )

    recovered = pd.read_csv(RECOVERED_TARGETS, low_memory=False)
    recovered_cols = [
        "target_chembl_id",
        "current_program_status",
        "predicted_pocket_qualification",
        "discovery_authorization",
        "authorization_scope",
        "next_action_zh",
        "formal_computational_triage_pair_count",
        "diagnostic_rescue_pair_count",
        "boltz_pairs_completed",
        "boltz_md_authorized_pairs",
        "primary_gate_strong",
        "primary_gate_pass",
    ]
    recovered = recovered[[column for column in recovered_cols if column in recovered.columns]].copy()
    recovered = recovered.rename(
        columns={
            "current_program_status": "unified_target_route",
            "authorization_scope": "allowed_pair_scope_zh",
            "next_action_zh": "target_next_compute_action_zh",
        }
    )
    recovered["unified_target_route_zh"] = recovered["unified_target_route"]
    recovered["target_route_family"] = "PREDICTED_POCKET_RECOVERY"
    recovered["current_target_qualification_strength"] = recovered[
        "predicted_pocket_qualification"
    ].fillna("NOT_EVALUABLE_MINIMUM_12X12_NOT_MET")
    recovered["current_target_state"] = recovered["unified_target_route"]
    recovered["target_evidence_layer"] = "PREDICTED_POCKET_RECOVERY_BRANCH"
    recovered["modality_compute_branch"] = "CLASS_SPECIFIC_PREDICTED_POCKET_PROTOCOL"
    recovered["gnina_target_qualification"] = recovered[
        "predicted_pocket_qualification"
    ].fillna("NOT_EVALUABLE_MINIMUM_12X12_NOT_MET")
    recovered["boltz_target_qualification"] = np.where(
        recovered["target_chembl_id"].eq("CHEMBL3714029"),
        "UNCALIBRATED_STRUCTURAL_STRESS_ONLY",
        "NOT_FORMALLY_QUALIFIED",
    )
    strong = recovered["predicted_pocket_qualification"].eq(
        "QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION"
    )
    recovered["qualified_physical_models"] = np.where(
        strong,
        "GNINA_PREDICTED_POCKET_STRONG",
        "NONE",
    )
    recovered.loc[
        recovered["target_chembl_id"].eq("CHEMBL3714029"),
        "qualified_physical_models",
    ] = "GNINA_PREDICTED_POCKET_STRONG+Boltz-2_UNCALIBRATED_STRESS"
    recovered["target_gate_pass_for_discovery"] = strong
    recovered["remote_qualified_physical_models"] = np.where(
        strong, "GNINA_PREDICTED_POCKET_STRONG", "NONE"
    )

    routes = pd.concat([main, recovered], ignore_index=True, sort=False)
    routes = target_master.merge(routes, on="target_chembl_id", how="left", validate="one_to_one")
    if len(routes) != 384 or routes["target_chembl_id"].duplicated().any():
        raise RuntimeError("Unified target routing must contain 384 unique targets")
    if routes["unified_target_route"].isna().any():
        missing = routes.loc[routes["unified_target_route"].isna(), "target_chembl_id"].tolist()
        raise RuntimeError(f"Target routes missing: {missing}")
    return routes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        PAIR_MATRIX,
        TARGET_MASTER,
        MAIN_TARGET_ROUTES,
        RECOVERED_TARGETS,
        MAIN_BOLTZ,
        MAIN_GNINA,
        V3,
        RECOVERED_GNINA,
        RECOVERED_BOLTZ,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative inputs: {missing}")

    pairs = pd.read_csv(PAIR_MATRIX, low_memory=False)
    target_master = pd.read_csv(TARGET_MASTER, low_memory=False)
    target_routes = make_target_routing(target_master)
    target_route_columns = [
        "target_chembl_id",
        "target_route_family",
        "unified_target_route",
        "unified_target_route_zh",
        "qualified_physical_models",
        "remote_qualified_physical_models",
        "allowed_pair_scope_zh",
        "current_target_qualification_strength",
        "current_target_state",
        "target_evidence_layer",
        "modality_compute_branch",
        "target_next_compute_action_zh",
        "gnina_target_qualification",
        "boltz_target_qualification",
        "predicted_pocket_qualification",
        "discovery_authorization",
        "target_gate_pass_for_discovery",
    ]
    pairs = pairs.merge(
        target_routes[target_route_columns],
        on="target_chembl_id",
        how="left",
        validate="many_to_one",
    )
    pairs["pairId"] = (
        pairs["ligand_inchikey"].astype(str) + "__" + pairs["target_chembl_id"].astype(str)
    )
    key = ["ligand_inchikey", "target_chembl_id"]

    main_boltz = pd.read_csv(MAIN_BOLTZ, low_memory=False)
    pairs = merge_prefixed(
        pairs,
        main_boltz,
        [
            "boltzCompleted",
            "boltz_completed_effective",
            "boltzAffinityProbabilityBinary",
            "boltzAffinityPredValue",
            "boltzConfidenceScore",
            "boltzLigandIptm",
            "boltzComplexIplddt",
            "boltz_evidence_tier",
            "may_enter_binding_priority",
            "result_category",
            "result_category_zh",
            "novelty_class",
            "max_tanimoto_to_target_positive",
        ],
        "main_boltz_",
        key,
    )

    main_gnina = pd.read_csv(MAIN_GNINA, low_memory=False)
    pairs = merge_prefixed(
        pairs,
        main_gnina,
        [
            "gnina_structure_evidence_tier",
            "gnina_target_qualification",
            "primary_cnn_affinity",
            "primary_pose_vina_affinity",
            "max_tanimoto_all_known_positive",
            "novelty_lane",
        ],
        "main_gnina_",
        key,
    )

    v3 = pd.read_csv(V3, low_memory=False)
    pairs = merge_prefixed(
        pairs,
        v3,
        [
            "final_pair_evidence_layer",
            "final_pair_evidence_layer_zh",
            "final_pair_action_zh",
            "final_pair_next_action_zh",
            "pair_evidence_source",
            "pair_evidence_lane",
            "pair_evidence_lane_zh",
            "unified_novelty_class",
            "unified_max_tanimoto_to_known_positive",
            "boltz_support",
            "gnina_support",
            "boltz_support_count_3seed",
            "multiseed_final_decision",
            "receptor_ensemble_decision",
            "adaptive_pilot_computed",
            "adaptive_gnina_local_support",
            "adaptive_boltzCompleted",
            "adaptive_boltz_support_count_3seed",
            "adaptive_multiseed_final_decision",
        ],
        "v3_",
        key,
    )

    recovered_gnina = pd.read_csv(RECOVERED_GNINA, low_memory=False)
    pairs = merge_prefixed(
        pairs,
        recovered_gnina,
        [
            "candidate_pair_id",
            "candidate_route",
            "candidate_rank_within_target_route",
            "primary_cnn_affinity",
            "primary_pose_vina_affinity",
            "cnn_affinity_percentile_vs_all_controls",
            "vina_percentile_vs_all_controls",
            "gnina_control_calibrated_support",
            "candidate_triage_status",
            "predicted_pocket_qualification",
        ],
        "recovered_gnina_",
        key,
    )

    recovered_boltz = pd.read_csv(RECOVERED_BOLTZ, low_memory=False)
    pairs = merge_prefixed(
        pairs,
        recovered_boltz,
        [
            "pair_id",
            "models_completed",
            "median_ligand_iptm",
            "median_complex_iplddt",
            "median_pocket_aligned_ligand_rmsd_A",
            "median_contact_jaccard",
            "affinity_probability_binary",
            "cofactor_complete",
            "md_authorized",
            "boltz_interpretation",
            "next_action",
        ],
        "recovered_boltz_",
        key,
    )

    pairs["conplex_completed"] = pairs["conplex_score"].notna()
    pairs["drugclip_completed"] = pairs["drugclip_cosine_mean"].notna()
    pairs["mainline_boltz_attempted"] = pairs["main_boltz_boltzCompleted"].notna()
    pairs["mainline_boltz_completed"] = truth(pairs["main_boltz_boltzCompleted"])
    pairs["mainline_gnina_completed"] = pairs["main_gnina_primary_cnn_affinity"].notna()
    pairs["recovered_gnina_completed"] = pairs["recovered_gnina_candidate_triage_status"].notna()
    pairs["recovered_boltz_completed"] = pairs["recovered_boltz_models_completed"].fillna(0).gt(0)
    pairs["gnina_primary_completed"] = (
        pairs["mainline_gnina_completed"] | pairs["recovered_gnina_completed"]
    )
    pairs["boltz_primary_completed"] = (
        pairs["mainline_boltz_completed"] | pairs["recovered_boltz_completed"]
    )
    pairs["boltz_multiseed_or_multimodel_completed"] = (
        pairs["v3_multiseed_final_decision"].notna()
        | pairs["v3_adaptive_multiseed_final_decision"].notna()
        | pairs["recovered_boltz_models_completed"].fillna(0).ge(3)
    )
    pairs["gnina_receptor_ensemble_completed"] = pairs[
        "v3_receptor_ensemble_decision"
    ].notna()
    pairs["any_physical_pair_calculation_completed"] = (
        pairs["gnina_primary_completed"] | pairs["boltz_primary_completed"]
    )

    mainline = pairs["target_route_family"].eq("EXPERIMENTAL_POCKET_MAINLINE")
    recovery = ~mainline
    needs_boltz = pairs["qualified_physical_models"].str.contains("Boltz-2", na=False)
    needs_gnina = pairs["qualified_physical_models"].str.contains("GNINA", na=False)
    pairs["completed_in_all_target_qualified_models"] = (
        (~needs_boltz | pairs["boltz_primary_completed"])
        & (~needs_gnina | pairs["gnina_primary_completed"])
        & (needs_boltz | needs_gnina)
    )
    needs_remote_boltz = (
        mainline
        & pairs["boltz_target_qualification"].eq("BOLTZ_REMOTE_QUALIFIED")
    )
    needs_remote_gnina = mainline & pairs["gnina_target_qualification"].eq("REMOTE_STRONG")
    pairs["completed_in_all_remote_qualified_models"] = (
        (~needs_remote_boltz | pairs["boltz_primary_completed"])
        & (~needs_remote_gnina | pairs["gnina_primary_completed"])
        & (needs_remote_boltz | needs_remote_gnina)
    )

    pairs["structural_novelty_max_tanimoto"] = pairs[
        [
            "main_boltz_max_tanimoto_to_target_positive",
            "main_gnina_max_tanimoto_all_known_positive",
            "v3_unified_max_tanimoto_to_known_positive",
        ]
    ].max(axis=1, skipna=True)
    structural_score = pairs["structural_novelty_max_tanimoto"]
    pairs["structural_novelty_class"] = np.select(
        [structural_score.gt(0.60), structural_score.gt(0.40), structural_score.notna()],
        ["N1_LOCAL_ANALOG", "N2_SCAFFOLD_HOP", "N3_REMOTE"],
        default="NOT_YET_EVALUATED",
    )

    pairs["final_pair_evidence_layer_v4"] = "L5_DTA_ONLY_NOT_PHYSICALLY_TESTED"
    pairs["final_pair_evidence_layer_zh_v4"] = "仅DTA，尚未进行pair级物理计算"
    physical = pairs["any_physical_pair_calculation_completed"]
    pairs.loc[physical, "final_pair_evidence_layer_v4"] = (
        "L4_COMPUTED_NO_PROMOTABLE_PHYSICAL_SUPPORT"
    )
    pairs.loc[physical, "final_pair_evidence_layer_zh_v4"] = "已完成物理计算但未取得可晋级支持"

    no_gate = ~pairs["target_gate_pass_for_discovery"].fillna(False)
    pairs.loc[no_gate & ~physical, "final_pair_evidence_layer_v4"] = "L5_TARGET_GATE_HOLD"
    pairs.loc[no_gate & ~physical, "final_pair_evidence_layer_zh_v4"] = (
        "靶点门控未通过，不自动送算发现pair"
    )

    known = truth(pairs["is_any_frozen_known_relationship"])
    pairs.loc[known, "final_pair_evidence_layer_v4"] = "C0_KNOWN_RELATIONSHIP_CONTROL"
    pairs.loc[known, "final_pair_evidence_layer_zh_v4"] = "已知关系对照，不作为新靶点发现"

    in_v3 = pairs["v3_final_pair_evidence_layer"].notna()
    pairs.loc[in_v3, "final_pair_evidence_layer_v4"] = pairs.loc[
        in_v3, "v3_final_pair_evidence_layer"
    ]
    pairs.loc[in_v3, "final_pair_evidence_layer_zh_v4"] = pairs.loc[
        in_v3, "v3_final_pair_evidence_layer_zh"
    ]

    recovered_pass = pairs["recovered_gnina_candidate_triage_status"].eq(
        "COMPUTATIONAL_TRIAGE_PASS_REQUIRES_EXPERIMENT"
    )
    pairs.loc[recovered_pass, "final_pair_evidence_layer_v4"] = (
        "L3_PREDICTED_POCKET_COMPUTATIONAL_TRIAGE"
    )
    pairs.loc[recovered_pass, "final_pair_evidence_layer_zh_v4"] = (
        "预测口袋计算分诊通过，必须实验验证"
    )
    recovered_diagnostic = recovery & pairs["recovered_gnina_candidate_triage_status"].notna() & ~recovered_pass
    pairs.loc[recovered_diagnostic, "final_pair_evidence_layer_v4"] = (
        "L4_PREDICTED_POCKET_DIAGNOSTIC_OR_UNSUPPORTED"
    )
    pairs.loc[recovered_diagnostic, "final_pair_evidence_layer_zh_v4"] = (
        "预测口袋诊断信号或未获对照校准支持"
    )

    pairs["pair_next_action_v4"] = "NO_AUTOMATIC_DEEPENING"
    pairs.loc[known, "pair_next_action_v4"] = "USE_AS_CONTROL_ONLY"
    pairs.loc[~known & ~physical & ~no_gate, "pair_next_action_v4"] = (
        "ELIGIBILITY_AND_NOVELTY_REVIEW_BEFORE_PHYSICAL_QUEUE"
    )
    pairs.loc[in_v3, "pair_next_action_v4"] = pairs.loc[
        in_v3, "v3_final_pair_action_zh"
    ].fillna(
        pairs.loc[in_v3, "v3_final_pair_next_action_zh"]
    )
    pairs.loc[recovered_pass, "pair_next_action_v4"] = (
        "WET_LAB_BINDING_OR_ACTIVITY_CONFIRMATION_NO_BINDING_CLAIM"
    )
    pairs.loc[recovered_diagnostic, "pair_next_action_v4"] = (
        "HOLD_DIAGNOSTIC_ONLY_OR_REBUILD_POCKET_PROTOCOL"
    )

    # The 46-target branch has no uncomputed target-top-10 pair left after its
    # frozen target gate: all LOXL2/NOX1 eligible pairs were already docked and
    # SPHK2 has no two-model target-top-10 pair.
    recovery_formal_rank = recovery & truth(pairs["dta_target_top10pct_concordant"])
    pairs["recovered_gate_eligible_target_top10_pair"] = (
        recovery_formal_rank
        & pairs["predicted_pocket_qualification"].eq(
            "QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION"
        )
        & pairs["target_chembl_id"].isin({"CHEMBL3714029", "CHEMBL1287628"})
    )
    pairs["recovered_eligible_pair_compute_gap"] = (
        pairs["recovered_gate_eligible_target_top10_pair"]
        & ~pairs["recovered_gnina_completed"]
    )

    if len(pairs) != 720 * 384 or pairs[key].duplicated().any():
        raise RuntimeError("Pair routing lost the complete 720 x 384 Cartesian universe")
    if not pairs["conplex_completed"].all():
        raise RuntimeError("ConPLEx must be complete for all 276,480 pairs")
    if int(pairs["drugclip_completed"].sum()) != 720 * 382:
        raise RuntimeError("DrugCLIP completion must be 720 x 382")
    if int(pairs["recovered_eligible_pair_compute_gap"].sum()) != 0:
        raise RuntimeError("Recovered qualified target-top-10 pair compute gap is non-zero")

    all_path = OUT / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V4.csv.gz"
    pairs.to_csv(all_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    target_path = OUT / "TARGET_EVIDENCE_LAYER_ROUTING_384_V4.csv"
    target_routes.to_csv(target_path, index=False)

    review_mask = (
        in_v3
        | pairs["recovered_gnina_completed"]
        | truth(pairs["dta_bidirectional_top10pct_concordant_384"])
        | known
    )
    review = pairs[review_mask].copy()
    layer_order = {
        "L1_MULTI_METHOD_STATE_STABLE": 1,
        "L2_BOLTZ_REPRODUCED_GNINA_PRIMARY": 2,
        "L2_GNINA_RECEPTOR_STATE_STABLE": 3,
        "L2_BOLTZ_REPRODUCED": 4,
        "L3_PREDICTED_POCKET_COMPUTATIONAL_TRIAGE": 5,
        "L3_STRUCTURE_OR_STATE_REVIEW": 6,
        "L3_SINGLE_STATE_GNINA_ONLY": 7,
        "L4_PREDICTED_POCKET_DIAGNOSTIC_OR_UNSUPPORTED": 8,
        "L4_COMPUTED_NO_PROMOTABLE_PHYSICAL_SUPPORT": 9,
        "L4_HOLD_NOT_REPRODUCED_OR_UNQUALIFIED": 10,
        "L5_DTA_ONLY_NOT_PHYSICALLY_TESTED": 11,
        "L5_TARGET_GATE_HOLD": 12,
        "C0_KNOWN_RELATIONSHIP_CONTROL": 20,
    }
    review["evidence_layer_order_v4"] = review["final_pair_evidence_layer_v4"].map(
        layer_order
    ).fillna(99)
    review = review.sort_values(
        ["evidence_layer_order_v4", "target_chembl_id", "pairId"], kind="mergesort"
    )
    review_columns = [
        "evidence_layer_order_v4",
        "pairId",
        "drug_names",
        "gene_symbol",
        "target_chembl_id",
        "active_target_branch",
        "pair_pocket_evidence_source",
        "unified_target_route",
        "qualified_physical_models",
        "remote_qualified_physical_models",
        "pair_novelty_class_384",
        "structural_novelty_class",
        "dta_target_top10pct_concordant",
        "dta_drug_centric_top10pct_concordant_384",
        "dta_bidirectional_top10pct_concordant_384",
        "conplex_completed",
        "drugclip_completed",
        "gnina_primary_completed",
        "boltz_primary_completed",
        "boltz_multiseed_or_multimodel_completed",
        "gnina_receptor_ensemble_completed",
        "completed_in_all_target_qualified_models",
        "completed_in_all_remote_qualified_models",
        "final_pair_evidence_layer_v4",
        "final_pair_evidence_layer_zh_v4",
        "pair_next_action_v4",
        "main_boltz_result_category",
        "main_gnina_gnina_structure_evidence_tier",
        "v3_multiseed_final_decision",
        "v3_receptor_ensemble_decision",
        "recovered_gnina_candidate_triage_status",
        "recovered_boltz_boltz_interpretation",
        "recovered_boltz_md_authorized",
    ]
    review_path = OUT / "PAIR_EVIDENCE_LAYER_PRIORITY_REVIEW_V4.csv.gz"
    review[[column for column in review_columns if column in review.columns]].to_csv(
        review_path, index=False, compression={"method": "gzip", "compresslevel": 5}
    )

    summary = {
        "package_name": "UNIFIED_EVIDENCE_LAYER_ROUTING_720_X_384_V4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "universe": {
            "pairs": int(len(pairs)),
            "ligands": int(pairs["ligand_inchikey"].nunique()),
            "targets": int(pairs["target_chembl_id"].nunique()),
            "mainline_targets": int(mainline.groupby(pairs["target_chembl_id"]).any().sum()),
            "recovered_targets": int(recovery.groupby(pairs["target_chembl_id"]).any().sum()),
        },
        "pair_compute_coverage": {
            "conplex_pairs": int(pairs["conplex_completed"].sum()),
            "drugclip_pairs": int(pairs["drugclip_completed"].sum()),
            "gnina_primary_pairs": int(pairs["gnina_primary_completed"].sum()),
            "boltz_primary_pairs": int(pairs["boltz_primary_completed"].sum()),
            "any_physical_pairs": int(pairs["any_physical_pair_calculation_completed"].sum()),
            "boltz_multiseed_or_multimodel_pairs": int(
                pairs["boltz_multiseed_or_multimodel_completed"].sum()
            ),
            "gnina_receptor_ensemble_pairs": int(
                pairs["gnina_receptor_ensemble_completed"].sum()
            ),
        },
        "recovered_46": {
            "dta_pairs": int(recovery.sum()),
            "drugclip_pairs": int((recovery & pairs["drugclip_completed"]).sum()),
            "gnina_candidate_pairs": int(pairs["recovered_gnina_completed"].sum()),
            "boltz_candidate_pairs": int(pairs["recovered_boltz_completed"].sum()),
            "qualified_target_top10_pairs": int(
                pairs["recovered_gate_eligible_target_top10_pair"].sum()
            ),
            "qualified_target_top10_compute_gap": int(
                pairs["recovered_eligible_pair_compute_gap"].sum()
            ),
        },
        "target_route_counts": target_routes["unified_target_route"].value_counts().to_dict(),
        "pair_evidence_layer_counts": pairs[
            "final_pair_evidence_layer_v4"
        ].value_counts().to_dict(),
        "priority_review_pairs": int(len(review)),
        "boundaries": [
            "Absence from V3 is not interpreted as absence of computation; raw GNINA and Boltz completion are tracked separately.",
            "Predicted-pocket DrugCLIP ranks remain exploratory and are not treated as experimental-pocket-equivalent evidence.",
            "The two LOXL2 pairs are computational triage only; Boltz low complex confidence, pose inconsistency, and missing Cu/LTQ prohibit MD and binding claims.",
            "No cross-target total score or binding probability is produced.",
            "N1_UNRECORDED means absent only from the frozen project controls and local ChEMBL37 MoA mapping, not absent from all databases or literature.",
        ],
        "outputs": {
            "all_pairs": str(all_path),
            "targets": str(target_path),
            "priority_review": str(review_path),
        },
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    summary_path = OUT / "EVIDENCE_LAYER_ROUTING_SUMMARY_V4.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "PASS",
        "created_utc": summary["created_utc"],
        "authoritative_outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [all_path, target_path, review_path, summary_path]
        },
    }
    (OUT / "EVIDENCE_LAYER_ROUTING_MANIFEST_V4.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

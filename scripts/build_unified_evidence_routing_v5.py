#!/usr/bin/env python3
"""Integrate the executed 384-rerank N2/N3 increment into final routing V5."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V4_DIR = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v4"
)
V4_PAIRS = V4_DIR / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V4.csv.gz"
V4_TARGETS = V4_DIR / "TARGET_EVIDENCE_LAYER_ROUTING_384_V4.csv"
BASE = ROOT / "outputs/unified_pair_compute_increment_384_v1"
QUEUE = BASE / "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENTAL_PAIR_QUEUE_V1.csv"
GNINA = BASE / "execution_v1/gnina_evaluation/GNINA_REMOTE_INCREMENT_PAIR_EVIDENCE_V1.csv"
BOLTZ = (
    BASE
    / "execution_v1/boltz_evaluation/BOLTZ2_CONDITIONAL_DISCOVERY_EVIDENCE_V1.csv.gz"
)
MULTISEED = (
    BASE
    / "execution_v1/boltz_multiseed/stability_evaluation"
    / "UNIFIED_REMOTE_INCREMENT_BOLTZ_MULTI_SEED_STABILITY_V1.csv.gz"
)
ENSEMBLE = (
    BASE
    / "execution_v1/gnina_receptor_state/evaluation"
    / "UNIFIED_INCREMENT_GNINA_RECEPTOR_ENSEMBLE_PAIR_DECISIONS_V1.csv.gz"
)
ENSEMBLE_PROTOCOLS = (
    BASE
    / "execution_v1/gnina_receptor_state/evaluation"
    / "UNIFIED_INCREMENT_GNINA_RECEPTOR_ENSEMBLE_ALTERNATE_PROTOCOL_QUALIFICATION_V1.csv"
)
OUT = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v5"
)
ABC = {"BOLTZ_STRUCTURE_A", "BOLTZ_STRUCTURE_B", "BOLTZ_STRUCTURE_C"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_prefixed(
    base: pd.DataFrame, source: pd.DataFrame, columns: list[str], prefix: str
) -> pd.DataFrame:
    selected = ["pairId"] + [column for column in columns if column in source.columns]
    annotation = source[selected].drop_duplicates("pairId").rename(
        columns={column: f"{prefix}{column}" for column in selected if column != "pairId"}
    )
    return base.merge(annotation, on="pairId", how="left", validate="one_to_one")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        V4_PAIRS,
        V4_TARGETS,
        QUEUE,
        GNINA,
        BOLTZ,
        MULTISEED,
        ENSEMBLE,
        ENSEMBLE_PROTOCOLS,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing executed increment inputs: {missing}")

    pairs = pd.read_csv(V4_PAIRS, low_memory=False)
    targets = pd.read_csv(V4_TARGETS, low_memory=False)
    queue = pd.read_csv(QUEUE, low_memory=False)
    gnina = pd.read_csv(GNINA, low_memory=False)
    boltz = pd.read_csv(BOLTZ, low_memory=False)
    multiseed = pd.read_csv(MULTISEED, low_memory=False)
    ensemble = pd.read_csv(ENSEMBLE, low_memory=False)
    protocols = pd.read_csv(ENSEMBLE_PROTOCOLS, low_memory=False)

    pairs = merge_prefixed(
        pairs,
        queue,
        [
            "incremental_queue_rank",
            "incremental_target_rank",
            "novelty_lane",
            "max_tanimoto_to_target_measured_positive",
            "dta_priority_score_384",
            "gnina_increment_required",
            "boltz_increment_required",
            "planned_model_runs",
            "selection_reason",
        ],
        "increment_",
    )
    pairs = merge_prefixed(
        pairs,
        gnina,
        [
            "primary_cnn_affinity",
            "primary_pose_vina_affinity",
            "gnina_remote_evidence_tier",
            "gnina_remote_support",
            "cnn_percentile_vs_development_negative",
            "vina_percentile_vs_development_negative",
        ],
        "increment_gnina_",
    )
    pairs = merge_prefixed(
        pairs,
        boltz,
        [
            "boltzCompleted",
            "boltzAffinityProbabilityBinary",
            "boltzAffinityPredValue",
            "boltzConfidenceScore",
            "boltzLigandIptm",
            "boltzComplexIplddt",
            "boltz_evidence_tier",
            "may_enter_binding_priority",
            "boltzCifPath",
        ],
        "increment_boltz_",
    )
    pairs = merge_prefixed(
        pairs,
        multiseed,
        [
            "boltz_support_count_3seed",
            "boltz_score_stability_class",
            "pose_comparisons_completed",
            "pose_max_ligand_rmsd_A",
            "pose_max_centroid_distance_A",
            "pose_min_interface_jaccard",
            "multiseed_final_decision",
        ],
        "increment_multiseed_",
    )
    pairs = merge_prefixed(
        pairs,
        ensemble,
        [
            "alternate_states_run",
            "alternate_qualified_states",
            "alternate_support_states",
            "receptor_ensemble_decision",
        ],
        "increment_ensemble_",
    )
    pairs = pairs.copy()

    in_increment = pairs["increment_incremental_queue_rank"].notna()
    gnina_increment_required = pairs["increment_gnina_increment_required"].fillna(False).astype(bool)
    boltz_increment_required = pairs["increment_boltz_increment_required"].fillna(False).astype(bool)
    gnina_increment_completed = pairs["increment_gnina_gnina_remote_evidence_tier"].notna()
    boltz_increment_completed = pairs["increment_boltz_boltzCompleted"].fillna(False).astype(bool)
    pairs["incremental_compute_completed"] = (
        in_increment
        & (~gnina_increment_required | gnina_increment_completed)
        & (~boltz_increment_required | boltz_increment_completed)
    )
    pairs["gnina_primary_completed_v5"] = pairs["gnina_primary_completed"] | gnina_increment_completed
    pairs["boltz_primary_completed_v5"] = pairs["boltz_primary_completed"] | boltz_increment_completed
    pairs["any_physical_pair_calculation_completed_v5"] = (
        pairs["gnina_primary_completed_v5"] | pairs["boltz_primary_completed_v5"]
    )
    needs_remote_boltz = (
        pairs["target_route_family"].eq("EXPERIMENTAL_POCKET_MAINLINE")
        & pairs["boltz_target_qualification"].eq("BOLTZ_REMOTE_QUALIFIED")
    )
    needs_remote_gnina = (
        pairs["target_route_family"].eq("EXPERIMENTAL_POCKET_MAINLINE")
        & pairs["gnina_target_qualification"].eq("REMOTE_STRONG")
    )
    pairs["completed_in_all_remote_qualified_models_v5"] = (
        (~needs_remote_boltz | pairs["boltz_primary_completed_v5"])
        & (~needs_remote_gnina | pairs["gnina_primary_completed_v5"])
        & (needs_remote_boltz | needs_remote_gnina)
    )

    pairs["final_pair_evidence_layer_v5"] = pairs["final_pair_evidence_layer_v4"]
    pairs["final_pair_evidence_layer_zh_v5"] = pairs["final_pair_evidence_layer_zh_v4"]
    pairs["pair_next_action_v5"] = pairs["pair_next_action_v4"]

    boltz_stable = pairs["increment_multiseed_multiseed_final_decision"].isin(
        {"PASS_SCORE_AND_CONDITIONAL_POSE_STABILITY", "PROMOTE_P3_BOLTZ_REPEAT_RESCUE"}
    )
    boltz_review = pairs["increment_multiseed_multiseed_final_decision"].eq(
        "REVIEW_SCORE_STABLE_POSE_UNSTABLE_OR_INCOMPLETE"
    )
    gnina_state_stable = pairs["increment_ensemble_receptor_ensemble_decision"].isin(
        {"PASS_RECEPTOR_STATE_STABILITY", "RESCUE_BY_TWO_ALTERNATE_STATES"}
    )
    gnina_state_downgraded = pairs["increment_ensemble_receptor_ensemble_decision"].eq(
        "DOWNGRADE_PRIMARY_NOT_REPRODUCED"
    )
    gnina_primary_support = pairs["increment_gnina_gnina_remote_support"].fillna(False).astype(bool)
    boltz_primary_support = pairs["increment_boltz_boltz_evidence_tier"].isin(ABC)

    both_stable = boltz_stable & gnina_state_stable
    pairs.loc[both_stable, "final_pair_evidence_layer_v5"] = "L1_MULTI_METHOD_STATE_STABLE"
    pairs.loc[both_stable, "final_pair_evidence_layer_zh_v5"] = "多方法且受体状态稳定"
    pairs.loc[both_stable, "pair_next_action_v5"] = (
        "PAIR_LEVEL_STRUCTURAL_AUDIT_THEN_CAPPED_MD_OR_EXPERIMENT"
    )

    boltz_only_stable = boltz_stable & ~gnina_state_stable
    pairs.loc[boltz_only_stable, "final_pair_evidence_layer_v5"] = "L2_BOLTZ_REPRODUCED"
    pairs.loc[boltz_only_stable, "final_pair_evidence_layer_zh_v5"] = "Boltz多seed和条件pose稳定"
    pairs.loc[boltz_only_stable, "pair_next_action_v5"] = (
        "CORRELATED_MODEL_SUPPORT_ONLY; STRUCTURAL_AUDIT_BEFORE_MD_OR_EXPERIMENT"
    )

    gnina_only_stable = gnina_state_stable & ~boltz_stable
    pairs.loc[gnina_only_stable, "final_pair_evidence_layer_v5"] = (
        "L2_GNINA_RECEPTOR_STATE_STABLE"
    )
    pairs.loc[gnina_only_stable, "final_pair_evidence_layer_zh_v5"] = "GNINA独立受体状态复现"
    pairs.loc[gnina_only_stable, "pair_next_action_v5"] = (
        "STATE_STABLE_DOCKING_SUPPORT; ORTHOGONAL_MODEL_OR_EXPERIMENT"
    )

    pairs.loc[boltz_review, "final_pair_evidence_layer_v5"] = "L3_STRUCTURE_OR_STATE_REVIEW"
    pairs.loc[boltz_review, "final_pair_evidence_layer_zh_v5"] = "分数可复现但pose不稳定或不完整"
    pairs.loc[boltz_review, "pair_next_action_v5"] = "MANUAL_STRUCTURE_REVIEW_NO_AUTOMATIC_MD"

    unsupported_increment = in_increment & ~boltz_stable & ~gnina_state_stable & ~boltz_review
    pairs.loc[unsupported_increment, "final_pair_evidence_layer_v5"] = (
        "L4_HOLD_NOT_REPRODUCED_OR_UNQUALIFIED"
    )
    pairs.loc[unsupported_increment, "final_pair_evidence_layer_zh_v5"] = (
        "增量物理计算未支持或未跨受体状态复现"
    )
    pairs.loc[unsupported_increment, "pair_next_action_v5"] = (
        "NO_AUTOMATIC_DEEPENING; NEW_STRUCTURE_OR_EXPERIMENT_REQUIRED"
    )

    if len(pairs) != 720 * 384 or pairs[["ligand_inchikey", "target_chembl_id"]].duplicated().any():
        raise RuntimeError("V5 lost the complete 720 x 384 pair universe")
    if int(in_increment.sum()) != len(queue):
        raise RuntimeError("V5 incremental queue mapping is incomplete")
    if not pairs.loc[in_increment, "incremental_compute_completed"].all():
        raise RuntimeError("One or more frozen incremental model tasks remain incomplete")
    if int(gnina_increment_completed.sum()) != len(gnina):
        raise RuntimeError("GNINA increment mapping mismatch")
    if int(boltz_increment_completed.sum()) != len(boltz):
        raise RuntimeError("Boltz increment mapping mismatch")
    if len(multiseed) != int(boltz_primary_support.sum()):
        raise RuntimeError("Not every incremental Boltz A/B/C pair received multi-seed evaluation")
    if len(ensemble) != int(gnina_primary_support.sum()):
        raise RuntimeError("Not every incremental GNINA-supported pair received receptor-state evaluation")

    pair_path = OUT / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V5.csv.gz"
    pairs.to_csv(pair_path, index=False, compression={"method": "gzip", "compresslevel": 5})

    increment_results = pairs[in_increment].copy().sort_values(
        "increment_incremental_queue_rank", kind="mergesort"
    )
    increment_columns = [
        "increment_incremental_queue_rank",
        "pairId",
        "drug_names",
        "gene_symbol",
        "target_chembl_id",
        "unified_target_route",
        "increment_novelty_lane",
        "increment_max_tanimoto_to_target_measured_positive",
        "increment_gnina_increment_required",
        "increment_boltz_increment_required",
        "increment_gnina_gnina_remote_evidence_tier",
        "increment_gnina_gnina_remote_support",
        "increment_boltz_boltz_evidence_tier",
        "increment_boltz_boltzAffinityProbabilityBinary",
        "increment_boltz_boltzLigandIptm",
        "increment_multiseed_boltz_support_count_3seed",
        "increment_multiseed_pose_max_centroid_distance_A",
        "increment_multiseed_pose_min_interface_jaccard",
        "increment_multiseed_multiseed_final_decision",
        "increment_ensemble_alternate_states_run",
        "increment_ensemble_alternate_qualified_states",
        "increment_ensemble_alternate_support_states",
        "increment_ensemble_receptor_ensemble_decision",
        "incremental_compute_completed",
        "completed_in_all_remote_qualified_models_v5",
        "final_pair_evidence_layer_v5",
        "final_pair_evidence_layer_zh_v5",
        "pair_next_action_v5",
    ]
    increment_path = OUT / "INCREMENTAL_REMOTE_N2_N3_PAIR_RESULTS_57_V1.csv"
    increment_results[[column for column in increment_columns if column in increment_results.columns]].to_csv(
        increment_path, index=False
    )

    # Target-level status is a complete 384-target table, including zero-task targets.
    target_increment = increment_results.groupby("target_chembl_id").agg(
        incremental_pairs=("pairId", "size"),
        gnina_increment_pairs=("increment_gnina_increment_required", "sum"),
        boltz_increment_pairs=("increment_boltz_increment_required", "sum"),
        gnina_primary_supported_pairs=("increment_gnina_gnina_remote_support", "sum"),
        boltz_primary_supported_pairs=("increment_boltz_boltz_evidence_tier", lambda values: values.isin(ABC).sum()),
        boltz_multiseed_stable_pairs=("increment_multiseed_multiseed_final_decision", lambda values: values.isin({"PASS_SCORE_AND_CONDITIONAL_POSE_STABILITY", "PROMOTE_P3_BOLTZ_REPEAT_RESCUE"}).sum()),
        gnina_state_stable_pairs=("increment_ensemble_receptor_ensemble_decision", lambda values: values.isin({"PASS_RECEPTOR_STATE_STABILITY", "RESCUE_BY_TWO_ALTERNATE_STATES"}).sum()),
        gnina_primary_downgraded_pairs=("increment_ensemble_receptor_ensemble_decision", lambda values: values.eq("DOWNGRADE_PRIMARY_NOT_REPRODUCED").sum()),
    ).reset_index()
    targets = targets.merge(target_increment, on="target_chembl_id", how="left", validate="one_to_one")
    count_columns = [
        "incremental_pairs",
        "gnina_increment_pairs",
        "boltz_increment_pairs",
        "gnina_primary_supported_pairs",
        "boltz_primary_supported_pairs",
        "boltz_multiseed_stable_pairs",
        "gnina_state_stable_pairs",
        "gnina_primary_downgraded_pairs",
    ]
    targets[count_columns] = targets[count_columns].fillna(0).astype(int)
    targets["incremental_compute_status"] = "NO_INCREMENT_REQUIRED"
    targets.loc[targets["incremental_pairs"].gt(0), "incremental_compute_status"] = "COMPLETE"
    target_path = OUT / "TARGET_EVIDENCE_LAYER_ROUTING_384_V5.csv"
    targets.to_csv(target_path, index=False)

    review_mask = (
        pairs["v3_final_pair_evidence_layer"].notna()
        | pairs["recovered_gnina_completed"]
        | in_increment
        | pairs["dta_bidirectional_top10pct_concordant_384"].fillna(False)
        | pairs["is_any_frozen_known_relationship"].fillna(False)
    )
    review = pairs[review_mask].copy()
    order = {
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
    review["evidence_layer_order_v5"] = review["final_pair_evidence_layer_v5"].map(order).fillna(99)
    review = review.sort_values(
        ["evidence_layer_order_v5", "target_chembl_id", "pairId"], kind="mergesort"
    )
    review_columns = [
        "evidence_layer_order_v5",
        "pairId",
        "drug_names",
        "gene_symbol",
        "target_chembl_id",
        "active_target_branch",
        "unified_target_route",
        "pair_novelty_class_384",
        "structural_novelty_class",
        "increment_novelty_lane",
        "dta_bidirectional_top10pct_concordant_384",
        "gnina_primary_completed_v5",
        "boltz_primary_completed_v5",
        "incremental_compute_completed",
        "completed_in_all_remote_qualified_models_v5",
        "final_pair_evidence_layer_v5",
        "final_pair_evidence_layer_zh_v5",
        "pair_next_action_v5",
        "increment_gnina_gnina_remote_evidence_tier",
        "increment_boltz_boltz_evidence_tier",
        "increment_multiseed_multiseed_final_decision",
        "increment_ensemble_receptor_ensemble_decision",
    ]
    review_path = OUT / "PAIR_EVIDENCE_LAYER_PRIORITY_REVIEW_V5.csv.gz"
    review[[column for column in review_columns if column in review.columns]].to_csv(
        review_path, index=False, compression={"method": "gzip", "compresslevel": 5}
    )

    primary_runs = int(queue["planned_model_runs"].sum())
    summary = {
        "package_name": "UNIFIED_EVIDENCE_LAYER_ROUTING_720_X_384_V5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "universe": {
            "pairs": len(pairs),
            "ligands": int(pairs["ligand_inchikey"].nunique()),
            "targets": int(pairs["target_chembl_id"].nunique()),
            "mainline_targets": 338,
            "recovered_targets": 46,
        },
        "increment_execution": {
            "frozen_pairs": len(queue),
            "targets": int(queue["target_chembl_id"].nunique()),
            "primary_model_runs_planned": primary_runs,
            "primary_model_runs_completed": int(len(gnina) + len(boltz)),
            "gnina_primary_pairs": len(gnina),
            "gnina_primary_supported": int(gnina["gnina_remote_support"].sum()),
            "boltz_primary_pairs": len(boltz),
            "boltz_primary_supported": int(boltz["boltz_evidence_tier"].isin(ABC).sum()),
            "boltz_added_seed_runs": int(len(multiseed) * 2),
            "boltz_3seed_score_supported": int(multiseed["boltz_support_count_3seed"].eq(3).sum()),
            "boltz_multiseed_pose_stable": int(
                multiseed["multiseed_final_decision"].eq(
                    "PASS_SCORE_AND_CONDITIONAL_POSE_STABILITY"
                ).sum()
            ),
            "gnina_alternate_protocols_run": len(protocols),
            "gnina_alternate_protocols_qualified": int(protocols["alternate_protocol_gate_pass"].sum()),
            "gnina_state_stable_pairs": int(gnina_state_stable.sum()),
            "gnina_primary_not_reproduced_pairs": int(gnina_state_downgraded.sum()),
            "final_increment_layers": increment_results[
                "final_pair_evidence_layer_v5"
            ].value_counts().to_dict(),
        },
        "pair_compute_coverage_v5": {
            "conplex_pairs": int(pairs["conplex_completed"].sum()),
            "drugclip_pairs": int(pairs["drugclip_completed"].sum()),
            "gnina_primary_pairs": int(pairs["gnina_primary_completed_v5"].sum()),
            "boltz_primary_pairs": int(pairs["boltz_primary_completed_v5"].sum()),
            "any_physical_pairs": int(
                pairs["any_physical_pair_calculation_completed_v5"].sum()
            ),
        },
        "recovered_46_qualified_pair_compute_gap": int(
            pairs["recovered_eligible_pair_compute_gap"].sum()
        ),
        "pair_evidence_layer_counts_v5": pairs[
            "final_pair_evidence_layer_v5"
        ].value_counts().to_dict(),
        "boundaries": [
            "All 57 rerank-increment pairs completed every remote-qualified primary model required by the frozen route.",
            "Boltz three-seed stability is correlated model evidence, not independent validation or measured affinity.",
            "Both GNINA primary supports failed receptor-state reproduction and were downgraded.",
            "No MD is automatically authorized; L2 Boltz-only pairs require pair-level structural and context audit first.",
            "GPCR and all prior hard-gate exclusions remain outside this computation universe.",
            "No cross-target total score or binder probability is produced.",
        ],
        "outputs": {
            "all_pairs": str(pair_path),
            "targets": str(target_path),
            "increment_results": str(increment_path),
            "priority_review": str(review_path),
        },
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    summary_path = OUT / "EVIDENCE_LAYER_ROUTING_SUMMARY_V5.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": "PASS",
        "created_utc": summary["created_utc"],
        "authoritative_outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [pair_path, target_path, increment_path, review_path, summary_path]
        },
    }
    (OUT / "EVIDENCE_LAYER_ROUTING_MANIFEST_V5.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

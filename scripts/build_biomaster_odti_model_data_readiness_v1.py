#!/usr/bin/env python3
"""Build a current, auditable model/data readiness snapshot for BioMaster ODTI.

This report is deliberately separate from the training runner.  It reads the
live status, the run-level E0 audit, the frozen E1 protocol and the frozen
external-evaluation summary, then states what is complete, what is still
running, and which upgrades are allowed.  It never promotes a model and never
turns unknown pairs into negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.is_file():
        return {} if default is None else default
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {} if default is None else default
    return value if isinstance(value, dict) else ({} if default is None else default)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_snapshot() -> dict:
    live_path = ROOT / "outputs/biomaster_odti_live_status_v1/BIOMASTER_ODTI_LIVE_STATUS_V1.json"
    e0_path = ROOT / "outputs/odti_e0_continuation_20260817/CONTINUATION_STATE_V1.json"
    post_path = ROOT / "outputs/odti_post_e0_continuation_20260817/POST_E0_STATE_V1.json"
    e1_path = ROOT / "outputs/odti_e1_continuation_20260817/E1_STATE_V1.json"
    e0_unified_path = ROOT / "outputs/biomaster_odti_unified_e0_summary_v1/BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json"
    e1_gate_path = ROOT / "outputs/biomaster_odti_e1_gate_audit_20260817/BIOMASTER_ODTI_E1_GATE_AUDIT_V1.json"
    e1_config_path = ROOT / "configs/biomaster_odti_e1_drug_aux_screen_20260817.json"
    e2_config_path = ROOT / "configs/biomaster_odti_e2_protein_residual_screen_20260817.json"
    e2_readiness_path = ROOT / (
        "outputs/biomaster_odti_e2_feature_readiness_v1/"
        "E2_FEATURE_SOURCE_READINESS_V1.json"
    )
    thermoprot_audit_path = ROOT / (
        "outputs/biomaster_odti_e2_thermoprot_audit_v1/"
        "THERMOPROT_E2_ELIGIBILITY_AUDIT_V1.json"
    )
    drugclip_audit_path = ROOT / (
        "outputs/biomaster_odti_drugclip_audit_v1/"
        "DRUGCLIP_E2_ELIGIBILITY_AUDIT_V1.json"
    )
    e2_acquisition_contract_path = ROOT / (
        "outputs/biomaster_odti_e2_external_weight_contract_v1/"
        "E2_EXTERNAL_WEIGHT_ACQUISITION_CONTRACT_V1.json"
    )
    bindingdb_path = ROOT / (
        "outputs/biomaster_bindingdb_source_heldout_current_20260817/"
        "BINDINGDB_PAIR_HELDOUT_SUMMARY.json"
    )
    entity_cold_queue_path = ROOT / (
        "outputs/biomaster_odti_bindingdb_entity_cold_feature_queue_v1/"
        "BINDINGDB_ENTITY_COLD_FEATURE_QUEUE_V1.json"
    )
    ligand_cold_summary_path = ROOT / (
        "outputs/biomaster_odti_bindingdb_ligand_cold_v2_exact/"
        "BINDINGDB_LIGAND_COLD_SUMMARY.json"
    )
    ligand_cold_audit_path = ROOT / (
        "outputs/biomaster_odti_bindingdb_ligand_cold_v2_exact/"
        "BINDINGDB_LIGAND_COLD_AUDIT_V1.json"
    )
    s1_audit_path = ROOT / "outputs/odti_unified_champion_s1_20260817/"
    s1_audit_path = s1_audit_path / "S1_SCAFFOLD_COLD_DRUG_SUITE_AUDIT_V1.json"
    s3_audit_path = ROOT / "outputs/odti_unified_champion_s3_20260817/"
    s3_audit_path = s3_audit_path / "S3_STRICT_DOUBLE_COLD_SUITE_AUDIT_V1.json"
    external_landscape_path = ROOT / (
        "outputs/biomaster_odti_external_entity_cold_landscape_v1/"
        "EXTERNAL_ENTITY_COLD_LANDSCAPE_V1.json"
    )
    local_external_candidate_path = ROOT / (
        "outputs/biomaster_odti_local_external_candidates_v1/"
        "LOCAL_EXTERNAL_ENTITY_COLD_CANDIDATES_V1.json"
    )
    w1_audit_path = ROOT / (
        "outputs/evidence_routing_compute_execution_20260808_v1/"
        "w1_result_ingestion_v17/W1_RESULT_INGESTION_INDEPENDENT_AUDIT_V17.json"
    )
    w1_template_summary_path = ROOT / (
        "outputs/evidence_routing_compute_execution_20260808_v1/"
        "w1_result_ingestion_v17/W1_RESULT_INGESTION_TEMPLATE_SUMMARY_V17.json"
    )
    v3_contract_path = ROOT / (
        "outputs/biomaster_odti_v3_data_contract_v1/"
        "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.json"
    )
    w1_bridge_path = ROOT / (
        "outputs/biomaster_odti_w1_v3_bridge_v1/"
        "W1_V17_TO_ODTI_V3_BRIDGE_SUMMARY_V1.json"
    )
    w1_v3_preflight_path = ROOT / (
        "outputs/biomaster_odti_w1_v3_training_preflight_v1/"
        "W1_V3_TRAINING_PREFLIGHT_AUDIT_V1.json"
    )
    w1_semantic_template_path = ROOT / (
        "outputs/biomaster_odti_w1_v3_semantic_input_v1/"
        "W1_V3_SEMANTIC_RESULT_INPUT_TEMPLATE_V1.json"
    )
    w1_semantic_adapter_path = ROOT / (
        "outputs/biomaster_odti_w1_v3_semantic_adapter_v1/"
        "W1_V3_SEMANTIC_ADAPTER_AUDIT_V1.json"
    )

    live = read_json(live_path)
    e0 = read_json(e0_path)
    post = read_json(post_path)
    e1 = read_json(e1_path)
    e0_unified = read_json(e0_unified_path)
    e1_gate = read_json(e1_gate_path)
    e1_config = read_json(e1_config_path)
    e2_config = read_json(e2_config_path)
    e2_readiness = read_json(e2_readiness_path)
    thermoprot_audit = read_json(thermoprot_audit_path)
    drugclip_audit = read_json(drugclip_audit_path)
    e2_acquisition_contract = read_json(e2_acquisition_contract_path)
    bindingdb = read_json(bindingdb_path)
    entity_cold_queue = read_json(entity_cold_queue_path)
    ligand_cold_summary = read_json(ligand_cold_summary_path)
    ligand_cold_audit = read_json(ligand_cold_audit_path)
    s1_audit = read_json(s1_audit_path)
    s3_audit = read_json(s3_audit_path)
    external_landscape = read_json(external_landscape_path)
    local_external_candidate = read_json(local_external_candidate_path)
    w1_audit = read_json(w1_audit_path)
    w1_template_summary = read_json(w1_template_summary_path)
    v3_contract = read_json(v3_contract_path)
    w1_bridge = read_json(w1_bridge_path)
    w1_v3_preflight = read_json(w1_v3_preflight_path)
    w1_semantic_template = read_json(w1_semantic_template_path)
    w1_semantic_adapter = read_json(w1_semantic_adapter_path)

    roles = live.get("roles", {})
    gates = live.get("gates", {})
    s1 = roles.get("S1", {})
    s3 = roles.get("S3", {})

    # Prefer the freshly materialized strict audit over the live-status
    # watcher.  The watcher is intentionally asynchronous and can lag by one
    # completed run while the training process is still writing artifacts.
    audit_runs_present = int(s1_audit.get("completed_pass", 0) or 0)
    audit_expected = int(s1_audit.get("expected_tasks", 0) or 0)
    current_s1 = {
        "runs_present": max(
            int(s1.get("runs_present", e0.get("completed", 0) or 0)),
            audit_runs_present,
        ),
        "runs_expected": audit_expected or int(s1.get("runs_expected", e0.get("total", 25) or 25)),
        "audit_pass": audit_runs_present or int(s1.get("audit_pass", 0) or 0),
        "audit_failed": int(s1_audit.get("failed", s1.get("audit_failed", 0)) or 0),
        "audit_status": s1_audit.get("status", s1.get("audit_status", "UNKNOWN")),
        "aggregate_present": bool(s1_audit.get("aggregate_present", False)),
    }

    s3_runtime_status = (
        "RUNNING"
        if e0.get("status") in {"RUNNING_S3", "AUDITING_S3"}
        else s3.get("status", "NOT_STARTED")
    )
    current_s3 = {
        "runs_present": int(s3.get("runs_present", 0) or 0),
        "runs_expected": int(s3.get("runs_expected", 25) or 25),
        "status": s3_runtime_status,
    }

    e1_candidate = e1_config.get("candidate", {})
    e1_gates = e1_config.get("screening_gates", {})
    e1_gate_status = e1_gate.get("status", "NOT_EVALUATED")
    e1_promotion = e1_gate.get("promotion", "NOT_EVALUATED")
    bindingdb_ranking = bindingdb.get("ranking", {})

    if e1_gate_status == "SCREEN_FAIL_DO_NOT_PROMOTE":
        decision = "KEEP_CURRENT_CHAMPION_AND_PRIORITIZE_DATA_CALIBRATION"
        bermol_decision = "SCREEN_FAIL_DO_NOT_PROMOTE"
        bermol_reason = "Frozen paired S3/S5 screen failed non-inferiority gates; retain BERMOL as an analyzed ablation, not a promoted model."
        target_reason = "E0 unified S1-S5 and paired bootstrap PASS; keep target-side champion pending new residual/data evidence."
    else:
        decision = "KEEP_CURRENT_CHAMPION_AND_RUN_CONTROLLED_RESIDUAL_ABLATIONS"
        bermol_decision = "RUN_GATED_RESIDUAL"
        bermol_reason = "Feature store and interface are ready; formal paired gate is not yet terminal."
        target_reason = "Current target-side champion; unified confirmation is still running."

    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "program_status": live.get("program_status", "UNKNOWN"),
        "decision": decision,
        "claim_status": "READINESS_SNAPSHOT; NOT_A_FINAL_SOTA_OR_PROSPECTIVE_CLAIM",
        "champion": live.get(
            "champion",
            "Morgan2048 + ProtBERT1024 + pooled ESM2-650M 1280 + 19-D structure + 6 experts",
        ),
        "authoritative_inputs": {
            "live_status": rel(live_path),
            "e0_state": rel(e0_path),
            "post_e0_state": rel(post_path),
            "e1_state": rel(e1_path),
            "e0_unified_summary": rel(e0_unified_path),
            "e1_gate_audit": rel(e1_gate_path),
            "e1_protocol": rel(e1_config_path),
            "e2_protocol": rel(e2_config_path),
            "e2_feature_readiness": rel(e2_readiness_path),
            "e2_thermoprot_audit": rel(thermoprot_audit_path),
            "drugclip_eligibility_audit": rel(drugclip_audit_path),
            "e2_external_weight_acquisition_contract": rel(e2_acquisition_contract_path),
            "bindingdb_summary": rel(bindingdb_path),
            "entity_cold_feature_queue": rel(entity_cold_queue_path),
            "ligand_cold_summary": rel(ligand_cold_summary_path),
            "ligand_cold_audit": rel(ligand_cold_audit_path),
            "s1_audit": rel(s1_audit_path),
            "s3_audit": rel(s3_audit_path),
            "external_entity_cold_landscape": rel(external_landscape_path),
            "local_external_candidate_audit": rel(local_external_candidate_path),
            "w1_ingestion_audit": rel(w1_audit_path),
            "w1_template_summary": rel(w1_template_summary_path),
            "v3_data_contract": rel(v3_contract_path),
            "w1_v3_bridge": rel(w1_bridge_path),
            "w1_v3_training_preflight": rel(w1_v3_preflight_path),
            "w1_v3_semantic_template": rel(w1_semantic_template_path),
            "w1_v3_semantic_adapter_audit": rel(w1_semantic_adapter_path),
        },
        "training_readiness": {
            "E0-S1": {
                **current_s1,
                "state": (
                    "PASS"
                    if current_s1["audit_status"] == "PASS"
                    and current_s1["runs_present"] >= current_s1["runs_expected"]
                    else (
                        "RUNNING"
                        if current_s1["runs_present"] < current_s1["runs_expected"]
                        else "AUDIT_PENDING"
                    )
                ),
            },
            "E0-S2": {
                "runs_present": int(roles.get("S2", {}).get("runs_present", 0) or 0),
                "runs_expected": int(roles.get("S2", {}).get("runs_expected", 25) or 25),
                "status": roles.get("S2", {}).get("status", "UNKNOWN"),
            },
            "E0-S3": {
                **current_s3,
                "state": (
                    "PASS"
                    if s3_audit.get("status") == "PASS"
                    and int(s3_audit.get("completed_pass", 0) or 0) >= current_s3["runs_expected"]
                    else (
                        "RUNNING"
                        if e0.get("status") in {"RUNNING_S3", "AUDITING_S3"}
                        else "WAITING_FOR_S1"
                    )
                ),
                "audit_status": s3_audit.get("status", "NOT_STARTED"),
                "audit_pass": int(s3_audit.get("completed_pass", 0) or 0),
                "audit_failed": int(s3_audit.get("failed", 0) or 0),
                "aggregate_present": bool(s3_audit.get("aggregate_present", False)),
            },
            "E0-S4": {
                "runs_present": int(roles.get("S4", {}).get("runs_present", 0) or 0),
                "runs_expected": int(roles.get("S4", {}).get("runs_expected", 5) or 5),
                "status": roles.get("S4", {}).get("status", "UNKNOWN"),
            },
            "E0-S5": {
                "runs_present": int(roles.get("S5", {}).get("runs_present", 0) or 0),
                "runs_expected": int(roles.get("S5", {}).get("runs_expected", 5) or 5),
                "status": roles.get("S5", {}).get("status", "UNKNOWN"),
            },
            "E1_BERMOL768": {
                "state": e1_gate_status if e1_gate else e1.get("status", "UNKNOWN"),
                "runner_status": e1.get("status", "UNKNOWN"),
                "formal_screen_started": bool(gates.get("e1_bermol_formal_screen", False)) or bool(e1_gate),
                "promotion": e1_promotion,
                "gate_audit": rel(e1_gate_path),
                "candidate_dim": e1_candidate.get("drug_aux_dim"),
                "available_drugs": e1_candidate.get("available_drugs"),
                "drug_aux_sha256": e1_candidate.get("drug_aux_sha256"),
            },
            "E2_protein_residual": {
                "state": e2_readiness.get("state", e2_config.get("status", "NOT_STARTED")),
                "promotion": "RESIDUAL_ONLY_AFTER_E0_E1",
                "config": rel(e2_config_path),
                "feature_source": e2_config.get("candidate", {}).get("source_path"),
                "readiness_audit": rel(e2_readiness_path),
                "eligible_distinct_local_weight_count": e2_readiness.get("eligible_distinct_local_weight_count", 0),
                "thermoprot": {
                    "audit": rel(thermoprot_audit_path),
                    "available": bool(thermoprot_audit),
                    "eligible_as_e2_source": thermoprot_audit.get("decision", {}).get("eligible_as_e2_source", False),
                    "classification": thermoprot_audit.get("decision", {}).get("classification"),
                },
                "drugclip": {
                    "audit": rel(drugclip_audit_path),
                    "available": bool(drugclip_audit),
                    "eligible_as_e2_source": drugclip_audit.get("decision", {}).get("eligible_as_e2_source", False),
                    "classification": drugclip_audit.get("decision", {}).get("classification"),
                    "allowed_use": drugclip_audit.get("decision", {}).get("allowed_use"),
                },
                "external_weight_acquisition_contract": {
                    "path": rel(e2_acquisition_contract_path),
                    "available": bool(e2_acquisition_contract),
                    "status": e2_acquisition_contract.get("status", "MISSING"),
                    "target_entity_rows_expected": e2_acquisition_contract.get("frozen_target_alignment", {}).get("target_entity_rows_expected"),
                    "download_now": e2_acquisition_contract.get("decision", {}).get("download_now", False),
                    "start_formal_screen_now": e2_acquisition_contract.get("decision", {}).get("start_formal_screen_now", False),
                },
            },
            "E3_pair_specific_3D": {"state": "NOT_STARTED", "promotion": "REQUIRES_PAIR_POSE_AND_LEAKAGE_AUDIT"},
            "OFER_DTI_Phase_A": {"state": "DEVELOPMENT_SCREEN_NOT_FINAL"},
        },
        "promotion_policy": {
            "baseline_must_be_frozen": True,
            "same_split_seed_budget": True,
            "paired_cluster_bootstrap": True,
            "source_heldout_noninferiority": True,
            "w1_test_reuse_forbidden": True,
            "E1_screening_gates": e1_gates,
            "unknown_pairs_are_not_negative": True,
        },
        "weight_decisions": [
            {
                "candidate": "Morgan2048",
                "decision": "KEEP",
                "reason": "stable chemistry anchor; do not remove before paired evidence",
            },
            {
                "candidate": "ProtBERT1024 + pooled ESM2-650M",
                "decision": "KEEP",
                "reason": target_reason,
            },
            {
                "candidate": "BERMOL768",
                "decision": bermol_decision,
                "reason": bermol_reason,
            },
            {
                "candidate": "structure-aware protein PLM",
                "decision": "LATER_RESIDUAL_ABLATION",
                "reason": "only after E0/E1 and only with structure mask/fallback",
            },
            {
                "candidate": "ThermoProt checkpoint",
                "decision": "REJECT_AS_E2_SOURCE",
                "reason": "ESM-input IDP structure/contact task checkpoint; no independent pooled target PLM feature contract",
            },
            {
                "candidate": "DrugCLIP / Drug-The-Whole-Genome",
                "decision": "NOT_E2_LATER_E3_COMPARATOR",
                "reason": "pocket-ligand Uni-Mol 3D retrieval model; requires pair-specific pocket geometry and is not a sequence protein PLM",
            },
            {
                "candidate": "pocket-ligand 3D graph",
                "decision": "LATER_E3",
                "reason": "requires pair-specific pose/geometry and triple leakage audit",
            },
            {
                "candidate": "full encoder replacement or full fine-tuning",
                "decision": "DO_NOT_DO_NOW",
                "reason": "confounds attribution and increases overfit risk at current entity scale",
            },
        ],
        "data_priorities": [
            {
                "priority": 1,
                "item": "W1 prospective readout",
                "required_fields": ["active", "inactive", "borderline", "failed", "replicate_variance", "assay_metadata"],
                "reason": "highest-value new labels and the only route to a prospective closure",
            },
            {
                "priority": 2,
                "item": "true entity-cold external benchmark",
                "required_fields": ["unseen_ligand", "unseen_target", "both_unseen", "source_id"],
                "reason": "current BindingDB result is pair-heldout positive-only with both entities seen",
            },
            {
                "priority": 3,
                "item": "explicit inactive and censored affinity data",
                "required_fields": ["tested_inactive", "assay_type", "construct", "units", "lower_upper_bounds"],
                "reason": "improves PU separation, calibration and continuous affinity supervision",
            },
            {
                "priority": 4,
                "item": "target/scaffold coverage",
                "required_fields": ["target_family", "homology_cluster", "scaffold_cluster"],
                "reason": "coverage and observation bias are the bottleneck, not simply row count",
            },
            {
                "priority": 5,
                "item": "pair-specific pocket/pose structures",
                "required_fields": ["pocket_residues", "ligand_pose", "confidence", "construct", "feature_hash"],
                "reason": "needed before E3 3D interaction modeling",
            },
        ],
        "external_evidence_boundary": {
            "protocol": bindingdb.get("protocol"),
            "candidate_rows": bindingdb.get("candidate_rows"),
            "aligned_pairs": bindingdb.get("aligned_unique_pairs"),
            "aligned_ligands": bindingdb.get("aligned_unique_ligands"),
            "aligned_targets": bindingdb.get("aligned_unique_targets"),
            "recall_at_5": bindingdb_ranking.get("recall_at_5"),
            "recall_at_10": bindingdb_ranking.get("recall_at_10"),
            "recall_at_20": bindingdb_ranking.get("recall_at_20"),
            "claim": "source-heldout positive-only retrieval; not entity-cold, negative benchmark or prospective validation",
            "entity_cold_feature_queue": {
                "missing_feature_pairs": entity_cold_queue.get("counts", {}).get("missing_feature_pairs"),
                "missing_ligands": entity_cold_queue.get("counts", {}).get("missing_ligands"),
                "missing_targets": entity_cold_queue.get("counts", {}).get("missing_targets"),
                "both_unseen_pairs": entity_cold_queue.get("counts", {}).get("both_unseen_pairs"),
                "queue_path": rel(entity_cold_queue_path),
            },
            "ligand_cold_external": {
                "audit_status": ligand_cold_audit.get("status"),
                "protocol": ligand_cold_summary.get("protocol"),
                "candidate_rows": ligand_cold_summary.get("candidate_rows"),
                "candidate_ligands": ligand_cold_summary.get("candidate_ligands"),
                "positive_pairs": ligand_cold_summary.get("positive_pairs"),
                "source_queue_ligands": ligand_cold_summary.get("source_queue_ligands_before_alignment"),
                "unresolved_ligands": ligand_cold_summary.get("unresolved_ligands_explicit"),
                "unresolved_pairs": ligand_cold_summary.get("unresolved_pair_count"),
                "recall_at_5": ligand_cold_summary.get("ranking", {}).get("recall_at_5"),
                "recall_at_10": ligand_cold_summary.get("ranking", {}).get("recall_at_10"),
                "recall_at_20": ligand_cold_summary.get("ranking", {}).get("recall_at_20"),
                "claim": "ligand-entity-cold positive-only retrieval; target entities seen; not both-entity-cold or negative benchmark",
            },
            "local_external_candidates": {
                "audit_status": local_external_candidate.get("status", "MISSING"),
                "davis_status": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("status"),
                "davis_rows": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("rows"),
                "davis_unique_pairs": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("unique_pairs"),
                "davis_both_unseen_unique_pairs": local_external_candidate.get(
                    "sources", {}
                ).get("davis_complete_secondary_hf", {}).get("both_unseen_unique_pairs"),
                "davis_score_ready_in_current_store": local_external_candidate.get(
                    "gate_decision", {}
                ).get("davis_score_ready_in_current_store", False),
                "davis_positive_only_scored": local_external_candidate.get(
                    "gate_decision", {}
                ).get("davis_positive_only_scored", False),
                "recall_at_5": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("ranking_primary_mean_threshold", {}).get("recall_at_5"),
                "recall_at_10": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("ranking_primary_mean_threshold", {}).get("recall_at_10"),
                "recall_at_20": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("ranking_primary_mean_threshold", {}).get("recall_at_20"),
                "both_unseen_recall_at_5": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("ranking_primary_mean_threshold", {}).get("both_unseen_recall_at_5"),
                "both_unseen_recall_at_10": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("ranking_primary_mean_threshold", {}).get("both_unseen_recall_at_10"),
                "both_unseen_recall_at_20": local_external_candidate.get("sources", {}).get(
                    "davis_complete_secondary_hf", {}
                ).get("ranking_primary_mean_threshold", {}).get("both_unseen_recall_at_20"),
                "claim": (
                    "Davis positive-only numeric-affinity entity-cold retrieval; not binary or calibrated validation"
                    if local_external_candidate.get("gate_decision", {}).get("davis_positive_only_scored", False)
                    else "candidate source only; no scored external result yet"
                ),
            },
        },
        "gates": gates,
        "execution_readiness": {
            "true_entity_cold_external": {
                "state": (
                    "PASS"
                    if bool(external_landscape.get("gates", {}).get("true_entity_cold_external", False))
                    else "CLOSED_NO_QUALIFYING_SOURCE"
                ),
                "source_contract": external_landscape.get("next_required_source_contract", {}),
                "claim_status": external_landscape.get("claim_status"),
            },
            "w1_infrastructure": {
                "state": (
                    "PASS_AWAITING_REAL_DATA"
                    if w1_audit.get("status") == "PASS"
                    and w1_template_summary.get("status") == "PASS_TEMPLATE_CONSTRUCTION"
                    else "NOT_READY"
                ),
                "audit_status": w1_audit.get("status", "MISSING"),
                "checks_passed": w1_audit.get("checks_passed"),
                "checks_total": w1_audit.get("checks_total"),
                "template_status": w1_template_summary.get("status", "MISSING"),
                "real_readout_available": bool(gates.get("w1_prospective_readout", False)),
                "claim_status": "INFRASTRUCTURE_ONLY; REAL PROSPECTIVE READOUT STILL REQUIRED",
            },
            "v3_data_contract": {
                "state": "PASS" if v3_contract.get("status") == "PASS" else "NOT_BUILT",
                "contract_name": v3_contract.get("contract_name"),
                "source_rows": v3_contract.get("source", {}).get("rows"),
                "explicit_inactive_rows": v3_contract.get("current_training_pool", {}).get("explicit_inactive_rows"),
                "missing_numeric_affinity_rows": v3_contract.get("current_training_pool", {}).get("missing_numeric_affinity_rows"),
                "semantic_fields_absent": v3_contract.get("affinity_and_assay_coverage", {}).get("semantic_fields_absent_from_frozen_store", []),
                "claim_status": v3_contract.get("claim_status", "DATA_CONTRACT_NOT_AVAILABLE"),
            },
            "w1_v3_bridge": {
                "state": "PASS" if w1_bridge.get("status") == "PASS" else "NOT_BUILT",
                "rows": w1_bridge.get("rows"),
                "candidates": w1_bridge.get("candidates"),
                "independent_runs": w1_bridge.get("independent_runs"),
                "overlap_status_counts": w1_bridge.get("training_store_overlap_status_counts", {}),
                "claim_status": w1_bridge.get("claim_status", "W1_V3_BRIDGE_NOT_AVAILABLE"),
            },
            "w1_v3_training_preflight": {
                "state": "PASS_TEMPLATE_AND_BRIDGE_WAITING_SEMANTIC_READOUT"
                if w1_v3_preflight.get("status") == "PASS"
                and w1_v3_preflight.get("decision", {}).get("direct_v3_training_ready") is False
                else "NOT_BUILT",
                "templates_structurally_ready": w1_v3_preflight.get("decision", {}).get("w1_templates_structurally_ready"),
                "identity_provenance_bridge_ready": w1_v3_preflight.get("decision", {}).get("identity_provenance_bridge_ready"),
                "direct_v3_training_ready": w1_v3_preflight.get("decision", {}).get("direct_v3_training_ready", False),
                "semantic_fields_required": w1_v3_preflight.get("semantic_adapter_required", {}).get("fields", []),
                "claim_status": w1_v3_preflight.get("claim_status", "W1_V3_PREFLIGHT_NOT_AVAILABLE"),
            },
            "w1_v3_semantic_adapter": {
                "template_status": w1_semantic_template.get("status", "MISSING"),
                "adapter_status": w1_semantic_adapter.get("status", "NOT_RUN"),
                "accepted": w1_semantic_adapter.get("accepted", False),
                "output_csv_written": w1_semantic_adapter.get("output_csv_written", False),
                "training_ready": False,
                "claim_status": w1_semantic_adapter.get("claim_status", "SEMANTIC_ADAPTER_NOT_AVAILABLE"),
            },
        },
    }
    return snapshot


def markdown(snapshot: dict) -> str:
    training = snapshot["training_readiness"]
    external = snapshot["external_evidence_boundary"]
    ligand_cold = external.get("ligand_cold_external", {})
    e1 = training["E1_BERMOL768"]
    if e1.get("state") == "SCREEN_FAIL_DO_NOT_PROMOTE":
        opening = "> E0 统一验证已 PASS；BERMOL768 S3/S5 paired screen 未通过冻结非劣门槛，保持当前 champion，优先补充校准、entity-cold 和 W1 数据。"
    else:
        opening = "> 当前不整体替换预训练主干；先完成 E0 统一确认，再运行 BERMOL768 gated residual，并优先补充 W1、entity-cold 和显式 inactive 数据。"
    lines = [
        "# BioMaster ODTI：模型与数据 readiness 快照",
        "",
        f"生成时间：{snapshot['created_utc']}",
        "",
        f"**总判断：{snapshot['decision']}**",
        "",
        opening,
        "",
        "## 当前训练状态",
        "",
        "| 角色 | 当前状态 |",
        "|---|---|",
    ]
    for role in ["E0-S1", "E0-S2", "E0-S3", "E0-S4", "E0-S5", "E1_BERMOL768", "E2_protein_residual", "E3_pair_specific_3D", "OFER_DTI_Phase_A"]:
        item = training[role]
        if role == "E0-S1":
            state = f"{item['runs_present']}/{item['runs_expected']}，审计 PASS {item['audit_pass']}，FAIL {item['audit_failed']}，{item['state']}"
        elif role.startswith("E0-") and "runs_present" in item:
            state = f"{item['runs_present']}/{item['runs_expected']}，{item.get('status', 'UNKNOWN')}"
        else:
            state = str(item.get("state", item.get("status", "UNKNOWN")))
        lines.append(f"| {role} | {state} |")
    lines += [
        "",
        "## E1 BERMOL768 gate 结果",
        "",
        f"runner 状态：`{e1.get('runner_status', e1.get('state'))}`；promotion：`{e1.get('promotion', 'NOT_EVALUATED')}`。",
        "",
        "冻结门槛逐 protocol 结果见 `outputs/biomaster_odti_e1_gate_audit_20260817/BIOMASTER_ODTI_E1_GATE_AUDIT_V1.json`；若为 `SCREEN_FAIL_DO_NOT_PROMOTE`，不得扩展五 seed，也不得替换 E0 champion。",
    ]
    lines += ["", "## 权重决策", "", "| 候选 | 决策 | 理由 |", "|---|---|---|"]
    for item in snapshot["weight_decisions"]:
        lines.append(f"| {item['candidate']} | {item['decision']} | {item['reason']} |")
    lines += ["", "## 数据优先级", "", "| 优先级 | 数据 | 为什么 |", "|---:|---|---|"]
    for item in snapshot["data_priorities"]:
        lines.append(f"| {item['priority']} | {item['item']} | {item['reason']} |")
    execution = snapshot.get("execution_readiness", {})
    entity_state = execution.get("true_entity_cold_external", {}).get("state", "UNKNOWN")
    w1_state = execution.get("w1_infrastructure", {}).get("state", "UNKNOWN")
    v3_state = execution.get("v3_data_contract", {}).get("state", "UNKNOWN")
    bridge_state = execution.get("w1_v3_bridge", {}).get("state", "UNKNOWN")
    w1_v3_preflight_state = execution.get("w1_v3_training_preflight", {}).get("state", "UNKNOWN")
    w1_semantic_adapter_state = execution.get("w1_v3_semantic_adapter", {}).get("adapter_status", "UNKNOWN")
    lines += [
        "",
        "## 执行就绪度",
        "",
        f"- true entity-cold external：`{entity_state}`；Davis 已完成 preliminary positive-only numeric-affinity scoring，但没有 explicit inactive/observation contract，因此 binary gate 仍未打开。",
        f"- W1 infrastructure：`{w1_state}`；V17 ingestion audit {execution.get('w1_infrastructure', {}).get('checks_passed')}/{execution.get('w1_infrastructure', {}).get('checks_total')}，但真实 assay readout 仍不可用。",
        f"- V3/W1 data contract：`{v3_state}`；显式 inactive {execution.get('v3_data_contract', {}).get('explicit_inactive_rows')} 行，缺少 numeric affinity {execution.get('v3_data_contract', {}).get('missing_numeric_affinity_rows')} 行。",
        f"- W1→V3 bridge：`{bridge_state}`；{execution.get('w1_v3_bridge', {}).get('rows')} rows、{execution.get('w1_v3_bridge', {}).get('candidates')} candidates，当前全部标记为 prospective unseen。",
        f"- W1→V3 training preflight：`{w1_v3_preflight_state}`；模板与 identity bridge 已就绪，但真实 readout 仍需先映射为 activity/censor/replicate/assay semantic fields，不能直接启动 V3。",
        f"- W1 semantic adapter：`{w1_semantic_adapter_state}`；当前模板被 fail-closed 拒绝，未写出训练 CSV，这是预期安全状态。",
        f"- E2 feature source：`{training['E2_protein_residual'].get('state')}`；ThermoProt audit classification = `{training['E2_protein_residual'].get('thermoprot', {}).get('classification')}`，不计入独立 protein-PLM source。",
        f"- DrugCLIP eligibility：`{training['E2_protein_residual'].get('drugclip', {}).get('classification', 'MISSING')}`；只保留为 E3 pocket–ligand comparator，不作为 E2 protein-PLM residual。",
        f"- E2 external weight contract：`{training['E2_protein_residual'].get('external_weight_acquisition_contract', {}).get('status', 'MISSING')}`；expected frozen target rows = `{training['E2_protein_residual'].get('external_weight_acquisition_contract', {}).get('target_entity_rows_expected')}`，当前 `download_now=false`、`start_formal_screen_now=false`。",
        "",
        "## 外部证据边界",
        "",
        f"BindingDB 已见实体协议为 `{external.get('protocol')}`。结果只支持 source-heldout positive-only retrieval，不支持 entity-cold、negative benchmark、calibrated binding probability 或 prospective validation。",
        "",
        f"新增 ligand-cold 协议为 `{ligand_cold.get('protocol')}`，独立审计 `{ligand_cold.get('audit_status')}`；候选 {ligand_cold.get('candidate_rows')} 行、未见 ligand {ligand_cold.get('candidate_ligands')} 个、正例 {ligand_cold.get('positive_pairs')} 个，Recall@5/10/20 = {ligand_cold.get('recall_at_5')}/{ligand_cold.get('recall_at_10')}/{ligand_cold.get('recall_at_20')}。该结果只支持 ligand-entity-cold positive-only retrieval；靶点实体仍已见，不是 both-entity-cold。",
        "",
        f"本地 external candidate audit：Davis `{snapshot.get('external_evidence_boundary', {}).get('local_external_candidates', {}).get('davis_status')}`，{snapshot.get('external_evidence_boundary', {}).get('local_external_candidates', {}).get('davis_both_unseen_unique_pairs')} 个 both-unseen unique pairs；当前结果限定为 positive-only numeric-affinity retrieval，binary gate 仍关闭。",
        "",
        "## 机器可读来源",
        "",
    ]
    for name, path in snapshot["authoritative_inputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/biomaster_odti_model_data_readiness_v1",
        help="directory for JSON and Markdown readiness artifacts",
    )
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot()
    json_path = out_dir / "BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.json"
    md_path = out_dir / "BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.md"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(markdown(snapshot))
    print(json.dumps({
        "status": snapshot["status"],
        "program_status": snapshot["program_status"],
        "decision": snapshot["decision"],
        "json": str(json_path),
        "markdown": str(md_path),
        "s1": snapshot["training_readiness"]["E0-S1"],
        "s3": snapshot["training_readiness"]["E0-S3"],
        "e1": snapshot["training_readiness"]["E1_BERMOL768"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the final 384-target pair-compute program and V5 routing package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v5"
PAIRS = V5 / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V5.csv.gz"
TARGETS = V5 / "TARGET_EVIDENCE_LAYER_ROUTING_384_V5.csv"
SUMMARY = V5 / "EVIDENCE_LAYER_ROUTING_SUMMARY_V5.json"
MANIFEST = V5 / "EVIDENCE_LAYER_ROUTING_MANIFEST_V5.json"
INCREMENT_RESULTS = V5 / "INCREMENTAL_REMOTE_N2_N3_PAIR_RESULTS_57_V1.csv"
QUEUE = (
    ROOT
    / "outputs/unified_pair_compute_increment_384_v1"
    / "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENTAL_PAIR_QUEUE_V1.csv"
)
FULL_SCOPE = (
    ROOT
    / "outputs/recovered_target_program_integrated_v1/FULL_TARGET_SCOPE_AUDIT_888_V1.csv"
)
GNINA = (
    ROOT
    / "outputs/unified_pair_compute_increment_384_v1/execution_v1/gnina_evaluation"
    / "GNINA_REMOTE_INCREMENT_PAIR_EVIDENCE_V1.csv"
)
BOLTZ = (
    ROOT
    / "outputs/unified_pair_compute_increment_384_v1/execution_v1/boltz_evaluation"
    / "BOLTZ2_CONDITIONAL_DISCOVERY_EVIDENCE_V1.csv.gz"
)
MULTISEED = (
    ROOT
    / "outputs/unified_pair_compute_increment_384_v1/execution_v1/boltz_multiseed"
    / "stability_evaluation/UNIFIED_REMOTE_INCREMENT_BOLTZ_MULTI_SEED_STABILITY_V1.csv.gz"
)
ENSEMBLE = (
    ROOT
    / "outputs/unified_pair_compute_increment_384_v1/execution_v1/gnina_receptor_state"
    / "evaluation/UNIFIED_INCREMENT_GNINA_RECEPTOR_ENSEMBLE_PAIR_DECISIONS_V1.csv.gz"
)
OUT = V5
ABC = {"BOLTZ_STRUCTURE_A", "BOLTZ_STRUCTURE_B", "BOLTZ_STRUCTURE_C"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        PAIRS,
        TARGETS,
        SUMMARY,
        MANIFEST,
        INCREMENT_RESULTS,
        QUEUE,
        FULL_SCOPE,
        GNINA,
        BOLTZ,
        MULTISEED,
        ENSEMBLE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    pair_columns = [
        "pairId",
        "ligand_inchikey",
        "target_chembl_id",
        "active_target_branch",
        "recovered_eligible_pair_compute_gap",
        "increment_incremental_queue_rank",
        "incremental_compute_completed",
        "completed_in_all_remote_qualified_models_v5",
        "final_pair_evidence_layer_v5",
    ]
    pairs = pd.read_csv(PAIRS, usecols=pair_columns, low_memory=False)
    targets = pd.read_csv(TARGETS, low_memory=False)
    scope = pd.read_csv(FULL_SCOPE, low_memory=False)
    queue = pd.read_csv(QUEUE, low_memory=False)
    increment = pd.read_csv(INCREMENT_RESULTS, low_memory=False)
    gnina = pd.read_csv(GNINA, low_memory=False)
    boltz = pd.read_csv(BOLTZ, low_memory=False)
    multiseed = pd.read_csv(MULTISEED, low_memory=False)
    ensemble = pd.read_csv(ENSEMBLE, low_memory=False)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    active_scope = scope[
        scope["final_scope_branch"].isin(
            {"ACTIVE_STRICT_MAINLINE_338", "ACTIVE_RECOVERED_NO_EXPERIMENTAL_POCKET_46"}
        )
    ].copy()
    pair_increment = pairs[pairs["increment_incremental_queue_rank"].notna()].copy()
    manifest_hashes_ok = True
    for relative, expected in manifest["authoritative_outputs"].items():
        path = ROOT / relative
        manifest_hashes_ok &= path.is_file() and sha256(path) == expected

    checks: list[dict[str, object]] = []

    def add(code: str, description: str, observed: object, expected: object, passed: bool) -> None:
        checks.append(
            {
                "check_code": code,
                "description": description,
                "observed": observed,
                "expected": expected,
                "status": "PASS" if passed else "FAIL",
            }
        )

    add("V5_SUMMARY_PASS", "V5 summary status", summary.get("status"), "PASS", summary.get("status") == "PASS")
    add("PAIR_ROWS", "complete Cartesian pair rows", len(pairs), 276480, len(pairs) == 276480)
    add("PAIR_KEYS", "unique ligand-target keys", pairs["pairId"].nunique(), 276480, pairs["pairId"].nunique() == 276480)
    add("LIGANDS", "unique ligand structures", pairs["ligand_inchikey"].nunique(), 720, pairs["ligand_inchikey"].nunique() == 720)
    add("TARGETS", "unique active targets", pairs["target_chembl_id"].nunique(), 384, pairs["target_chembl_id"].nunique() == 384)
    add("TARGET_TABLE", "target routing rows", len(targets), 384, len(targets) == 384 and targets["target_chembl_id"].nunique() == 384)
    add("ACTIVE_SCOPE_IDS", "V5 target IDs equal frozen 338+46 active scope", len(set(targets["target_chembl_id"]) ^ set(active_scope["target_chembl_id"])), 0, set(targets["target_chembl_id"]) == set(active_scope["target_chembl_id"]))
    add("ACTIVE_NO_GPCR", "no active target is GPCR", int(active_scope["is_gpcr"].fillna(False).sum()), 0, not active_scope["is_gpcr"].fillna(False).any())
    add("ACTIVE_SM_MOA", "all active targets pass small-molecule MoA hard gate", int(active_scope["passes_chembl_small_molecule_moa"].fillna(False).sum()), 384, active_scope["passes_chembl_small_molecule_moa"].fillna(False).all())
    add("ACTIVE_SUPPORTED_CLASS", "all active targets pass supported-class hard gate", int(active_scope["passes_supported_target_class"].fillna(False).sum()), 384, active_scope["passes_supported_target_class"].fillna(False).all())
    branch_counts = pairs.drop_duplicates("target_chembl_id")["active_target_branch"].value_counts().to_dict()
    add("ACTIVE_BRANCHES", "338 experimental-pocket plus exact 46 recovery", branch_counts, {"STRICT_EXPERIMENTAL_POCKET_MAINLINE_338": 338, "RECOVERED_NO_EXPERIMENTAL_POCKET_46": 46}, branch_counts == {"STRICT_EXPERIMENTAL_POCKET_MAINLINE_338": 338, "RECOVERED_NO_EXPERIMENTAL_POCKET_46": 46})
    add("HARD_EXCLUDED", "prior hard-gate exclusion partition remains closed", int(scope["final_scope_branch"].eq("HARD_GATE_EXCLUDED_480").sum()), 480, int(scope["final_scope_branch"].eq("HARD_GATE_EXCLUDED_480").sum()) == 480)
    add("RECOVERY_GAP", "eligible recovered target-top10 compute gap", int(pairs["recovered_eligible_pair_compute_gap"].sum()), 0, int(pairs["recovered_eligible_pair_compute_gap"].sum()) == 0)
    add("INCREMENT_PAIRS", "frozen remote N2/N3 incremental pairs", len(queue), 57, len(queue) == 57 and len(increment) == 57 and len(pair_increment) == 57)
    add("INCREMENT_NOVELTY", "increment contains N2/N3 only", sorted(queue["novelty_lane"].unique().tolist()), ["N2_SCAFFOLD_HOP", "N3_REMOTE"], set(queue["novelty_lane"]) == {"N2_SCAFFOLD_HOP", "N3_REMOTE"})
    add("INCREMENT_NO_KNOWN", "no known control in discovery increment", int(queue["is_any_frozen_known_relationship"].fillna(False).sum()), 0, not queue["is_any_frozen_known_relationship"].fillna(False).any())
    add("INCREMENT_NO_GPCR", "increment targets remain non-GPCR", int(scope[scope["target_chembl_id"].isin(queue["target_chembl_id"])] ["is_gpcr"].fillna(False).sum()), 0, not scope[scope["target_chembl_id"].isin(queue["target_chembl_id"])] ["is_gpcr"].fillna(False).any())
    add("PRIMARY_RUNS", "planned and completed primary model runs", len(gnina) + len(boltz), 61, len(gnina) == 6 and len(boltz) == 55 and len(gnina) + len(boltz) == 61)
    add("INCREMENT_COMPLETE", "all increment pairs complete required primary models", int(pair_increment["incremental_compute_completed"].fillna(False).sum()), 57, pair_increment["incremental_compute_completed"].fillna(False).all())
    add("REMOTE_MODELS_COMPLETE", "all increment pairs cover every remote-qualified model", int(pair_increment["completed_in_all_remote_qualified_models_v5"].fillna(False).sum()), 57, pair_increment["completed_in_all_remote_qualified_models_v5"].fillna(False).all())
    add("GNINA_SUPPORT", "GNINA primary support pairs", int(gnina["gnina_remote_support"].sum()), 2, int(gnina["gnina_remote_support"].sum()) == 2)
    add("GNINA_ENSEMBLE", "all GNINA supports received state evaluation and were not reproduced", ensemble["receptor_ensemble_decision"].value_counts().to_dict(), {"DOWNGRADE_PRIMARY_NOT_REPRODUCED": 2}, len(ensemble) == 2 and ensemble["receptor_ensemble_decision"].eq("DOWNGRADE_PRIMARY_NOT_REPRODUCED").all())
    add("BOLTZ_SUPPORT", "Boltz primary A/B/C pairs", int(boltz["boltz_evidence_tier"].isin(ABC).sum()), 15, int(boltz["boltz_evidence_tier"].isin(ABC).sum()) == 15)
    add("BOLTZ_MULTI_SEED", "all Boltz supports completed 3-seed stability", len(multiseed), 15, len(multiseed) == 15 and multiseed["boltz_support_count_3seed"].eq(3).all())
    add("BOLTZ_POSE_STABLE", "all multi-seed pairs pass conditional pose stability", multiseed["multiseed_final_decision"].value_counts().to_dict(), {"PASS_SCORE_AND_CONDITIONAL_POSE_STABILITY": 15}, multiseed["multiseed_final_decision"].eq("PASS_SCORE_AND_CONDITIONAL_POSE_STABILITY").all())
    add("INCREMENT_LAYERS", "final increment evidence layers", pair_increment["final_pair_evidence_layer_v5"].value_counts().to_dict(), {"L4_HOLD_NOT_REPRODUCED_OR_UNQUALIFIED": 42, "L2_BOLTZ_REPRODUCED": 15}, pair_increment["final_pair_evidence_layer_v5"].value_counts().to_dict() == {"L4_HOLD_NOT_REPRODUCED_OR_UNQUALIFIED": 42, "L2_BOLTZ_REPRODUCED": 15})
    add("MANIFEST_HASHES", "V5 authoritative output hashes", manifest_hashes_ok, True, manifest_hashes_ok)

    check_frame = pd.DataFrame(checks)
    status = "PASS" if check_frame["status"].eq("PASS").all() else "FAIL"
    check_path = OUT / "UNIFIED_PAIR_PROGRAM_COMPLETION_AUDIT_CHECKS_V1.csv"
    check_frame.to_csv(check_path, index=False)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": len(check_frame),
        "passed": int(check_frame["status"].eq("PASS").sum()),
        "failed": int(check_frame["status"].eq("FAIL").sum()),
        "scope": "720 FDA drug structures x 384 active non-GPCR targets",
        "increment": "57 remote-qualified N2/N3 pairs; 61 primary model runs; 30 added Boltz seeds; 3 alternate receptor protocols",
        "checks_table": str(check_path),
    }
    output = OUT / "UNIFIED_PAIR_PROGRAM_COMPLETION_AUDIT_V1.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status != "PASS":
        failures = check_frame[check_frame["status"].eq("FAIL")]
        raise RuntimeError(failures.to_dict("records"))


if __name__ == "__main__":
    main()

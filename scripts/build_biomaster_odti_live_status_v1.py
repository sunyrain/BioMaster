#!/usr/bin/env python3
"""Build one machine-readable live status for the BioMaster ODTI program."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/biomaster_odti_live_status_v1"


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - status builder preserves errors
        return {"status": "UNREADABLE", "path": str(path), "error": str(exc)}
    data["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return data


def run_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*/RUN_SUMMARY_V2.json")) if path.is_dir() else 0


def main() -> None:
    e0 = read_json(ROOT / "outputs/odti_e0_continuation_20260817/CONTINUATION_STATE_V1.json")
    post_e0 = read_json(ROOT / "outputs/odti_post_e0_continuation_20260817/POST_E0_STATE_V1.json")
    e1 = read_json(ROOT / "outputs/odti_e1_continuation_20260817/E1_STATE_V1.json")
    unified = read_json(
        ROOT / "outputs/biomaster_odti_unified_e0_summary_v1/BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json"
    )
    external = read_json(
        ROOT / "outputs/biomaster_odti_external_evaluation_freeze_audit_20260817/EXTERNAL_EVALUATION_FREEZE_AUDIT_V1.json"
    )
    s1_audit = read_json(
        ROOT / "outputs/odti_unified_champion_s1_20260817/S1_SCAFFOLD_COLD_DRUG_SUITE_AUDIT_V1.json"
    )
    s3_audit = read_json(
        ROOT / "outputs/odti_unified_champion_s3_20260817/S3_STRICT_DOUBLE_COLD_SUITE_AUDIT_V1.json"
    )

    roles = {
        "S1": {
            "runs_present": run_count(ROOT / "outputs/odti_unified_champion_s1_20260817"),
            "runs_expected": 25,
            "audit_status": s1_audit.get("status"),
            "audit_pass": s1_audit.get("completed_pass", 0),
            "audit_failed": s1_audit.get("failed", 0),
            "aggregate_present": bool(s1_audit.get("aggregate_present", False)),
            "status": s1_audit.get("status", "UNKNOWN"),
        },
        "S2": {"runs_present": 25, "runs_expected": 25, "status": "PASS"},
        "S3": {
            "runs_present": run_count(ROOT / "outputs/odti_unified_champion_s3_20260817"),
            "runs_expected": 25,
            "audit_status": s3_audit.get("status"),
            "audit_pass": s3_audit.get("completed_pass", 0),
            "audit_failed": s3_audit.get("failed", 0),
            "aggregate_present": bool(s3_audit.get("aggregate_present", False)),
            "status": (
                s3_audit.get("status")
                if s3_audit.get("status") not in {None, "MISSING"}
                else ("RUNNING" if e0.get("status") in {"RUNNING_S3", "AUDITING_S3"} else "NOT_STARTED")
            ),
        },
        "S4": {"runs_present": 5, "runs_expected": 5, "status": "PASS"},
        "S5": {"runs_present": 5, "runs_expected": 5, "status": "PASS"},
    }
    internal_complete = bool(
        unified.get("status") == "PASS"
        and roles["S1"]["runs_present"] == 25
        and roles["S1"]["audit_status"] == "PASS"
        and roles["S3"]["runs_present"] == 25
        and roles["S3"]["audit_status"] == "PASS"
    )
    external_frozen = bool(external.get("status") == "PASS")
    remaining = []
    if roles["S1"]["runs_present"] < 25 or roles["S1"]["audit_status"] != "PASS":
        remaining.append("finish and strictly audit E0-S1")
    if roles["S3"]["runs_present"] < 25 or roles["S3"]["audit_status"] != "PASS":
        remaining.append("run and strictly audit E0-S3")
    if not internal_complete:
        remaining.append("build unified S1-S5 aggregate and paired bootstrap")
    if e1.get("status") != "PASS":
        remaining.append("run guarded E1 BERMOL768 S3/S5 screen")
    if not external_frozen:
        remaining.append("construct true entity-cold external feature stores and benchmark")
    remaining.append("ingest prospective W1 active/inactive/failed readout without test reuse")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if external_frozen else "ATTENTION",
        "program_status": "IN_PROGRESS" if not internal_complete else "INTERNAL_E0_COMPLETE",
        "champion": "Morgan2048 + ProtBERT1024 + pooled ESM2-650M 1280 + 19-D structure + 6 experts",
        "roles": roles,
        "continuations": {
            "e0": {"status": e0.get("status"), "path": e0.get("path")},
            "post_e0": {"status": post_e0.get("status"), "path": post_e0.get("path")},
            "e1": {"status": e1.get("status"), "path": e1.get("path")},
        },
        "gates": {
            "unified_e0_internal": internal_complete,
            "bindingdb_positive_only_freeze": external_frozen,
            "true_entity_cold_external": False,
            "w1_prospective_readout": False,
            "e1_bermol_formal_screen": e1.get("status") == "PASS",
        },
        "remaining": remaining,
        "claim_status": "LIVE_OPERATIONAL_STATUS; NOT_A_FINAL_SOTA_OR_PROSPECTIVE_CLAIM",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "BIOMASTER_ODTI_LIVE_STATUS_V1.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

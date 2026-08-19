#!/usr/bin/env python3
"""Build the final internal E0 S1--S5 summary from audited role artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLE_DIRS = {
    "S1": ROOT / "outputs/odti_unified_champion_s1_20260817",
    "S2": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s2_esm2_full",
    "S3": ROOT / "outputs/odti_unified_champion_s3_20260817",
    "S4": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s4_esm2_formal",
    "S5": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s5_esm2_formal",
}
PROTOCOLS = {
    "S1": "S1_SCAFFOLD_COLD_DRUG",
    "S2": "S2_HOMOLOGY_COLD_TARGET",
    "S3": "S3_STRICT_DOUBLE_COLD",
    "S4": "S4_FIRST_SEEN_TEMPORAL_2023_2025",
    "S5": "S5_OLD_DRUG_ENTITY_COLD",
}


def role_summary(role: str, path: Path) -> dict[str, object]:
    protocol = PROTOCOLS[role]
    aggregate_path = path / "V2_MULTI_SEED_SUMMARY.json"
    audit_path = path / f"{protocol}_SUITE_AUDIT_V1.json"
    if not aggregate_path.is_file() or not audit_path.is_file():
        return {
            "role": role,
            "protocol": protocol,
            "status": "MISSING",
            "aggregate_present": aggregate_path.is_file(),
            "audit_present": audit_path.is_file(),
        }
    aggregate = json.loads(aggregate_path.read_text())
    audit = json.loads(audit_path.read_text())
    entries = aggregate.get("aggregates", [])
    metric = entries[0].get("metric", {}) if entries else {}
    return {
        "role": role,
        "protocol": protocol,
        "status": "PASS" if aggregate.get("status") == "PASS" and audit.get("status") == "PASS" else "FAIL",
        "aggregate_present": True,
        "audit_present": True,
        "runs": metric.get("runs"),
        "rows": metric.get("rows"),
        "positives": metric.get("positives"),
        "prevalence": metric.get("prevalence"),
        "micro_auroc": metric.get("micro_auroc"),
        "micro_auprc": metric.get("micro_auprc"),
        "target_macro_auprc": metric.get("target_macro_auprc"),
        "drug_macro_auprc": metric.get("drug_macro_auprc"),
        "ece_15": metric.get("ece_15"),
        "target_recall_at_20": metric.get("target_macro_recall_at_20"),
        "drug_recall_at_20": metric.get("drug_macro_recall_at_20"),
        "suite_audit_completed": audit.get("completed_pass"),
        "suite_audit_expected": audit.get("expected_tasks"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/biomaster_odti_unified_e0_summary_v1")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = [role_summary(role, path) for role, path in ROLE_DIRS.items()]
    role_pass = all(item["status"] == "PASS" for item in rows)
    bootstrap = {}
    for role in ["S1", "S3"]:
        protocol = PROTOCOLS[role]
        path = ROLE_DIRS[role] / "paired_bootstrap_v1" / f"{protocol}_V2_PAIRED_CLUSTER_BOOTSTRAP_SUMMARY_V1.json"
        bootstrap[role] = json.loads(path.read_text()) if path.is_file() else {"status": "MISSING"}
    bootstrap_pass = all(item.get("status") == "PASS" for item in bootstrap.values())
    status = "PASS" if role_pass and bootstrap_pass else "INCOMPLETE"
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "BIOMASTER_ODTI_UNIFIED_E0_ROLE_METRICS_V1.csv", index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model": "ODTI_V2_MORGAN2048_PROTBERT1024_POOLED_ESM2_1280_STRUCTURE_CONTEXT_19D_6_EXPERTS",
        "roles": rows,
        "paired_cluster_bootstrap": bootstrap,
        "checks": {
            "all_s1_s5_role_aggregates_and_audits_pass": role_pass,
            "s1_and_s3_paired_bootstrap_pass": bootstrap_pass,
        },
        "remaining_external_gates": [
            "source/entity-heldout external benchmark",
            "prospective W1 wet-lab readout",
            "post-W1 model update without reusing W1 as test",
        ],
        "claim_status": "UNIFIED_INTERNAL_CONFIRMATORY_EVIDENCE_ONLY; EXTERNAL_AND_PROSPECTIVE_GATES_REMAIN",
    }
    (output_dir / "BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

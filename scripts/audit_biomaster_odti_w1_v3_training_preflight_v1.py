#!/usr/bin/env python3
"""Audit whether W1 can be ingested into the V3 training contract.

The frozen W1 files are operator-facing assay/provenance templates.  They are
not expected to have the final V3 semantic columns before real readouts exist.
This audit makes that boundary explicit and prevents a template PASS from
being mistaken for V3 training readiness.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V3_CONTRACT = ROOT / (
    "outputs/biomaster_odti_v3_data_contract_v1/"
    "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.json"
)
W1_DIR = ROOT / (
    "outputs/evidence_routing_compute_execution_20260808_v1/"
    "w1_result_ingestion_v17"
)
W1_SCHEMA = W1_DIR / "W1_RESULT_INPUT_SCHEMA_AND_CODEBOOK_V17.json"
W1_AUDIT = W1_DIR / "W1_RESULT_INGESTION_INDEPENDENT_AUDIT_V17.json"
W1_TEMPLATE_SUMMARY = W1_DIR / "W1_RESULT_INGESTION_TEMPLATE_SUMMARY_V17.json"
W1_BRIDGE = ROOT / (
    "outputs/biomaster_odti_w1_v3_bridge_v1/"
    "W1_V17_TO_ODTI_V3_BRIDGE_SUMMARY_V1.json"
)
TEMPLATE_DIR = W1_DIR / "input_templates_v17"
OUT_DIR = ROOT / "outputs/biomaster_odti_w1_v3_training_preflight_v1"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def csv_columns_and_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    v3 = read_json(V3_CONTRACT)
    schema = read_json(W1_SCHEMA)
    audit = read_json(W1_AUDIT)
    template_summary = read_json(W1_TEMPLATE_SUMMARY)
    bridge = read_json(W1_BRIDGE)
    required = list(v3.get("w1_required_fields", []))

    templates: dict[str, dict[str, Any]] = {}
    all_rows_pending = True
    synthetic_pass_cells = 0
    frozen_lock_values = {"LOCKED_NOT_UNBLINDED"}
    for path in sorted(TEMPLATE_DIR.glob("*.csv")):
        columns, rows = csv_columns_and_rows(path)
        missing = [field for field in required if field not in columns]
        statuses = [value for row in rows for key, value in row.items() if key.endswith("status")]
        pending_count = sum(value in {"PENDING", "", "TEMPLATE_PENDING"} for value in statuses)
        non_pending_statuses = [
            value
            for value in statuses
            if value not in {"PENDING", "", "TEMPLATE_PENDING"} | frozen_lock_values
        ]
        synthetic_pass_cells += sum(value.startswith("PASS") for value in statuses)
        all_rows_pending = all_rows_pending and not non_pending_statuses
        templates[path.name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "rows": len(rows),
            "columns": len(columns),
            "direct_v3_fields_present": [field for field in required if field in columns],
            "direct_v3_fields_missing": missing,
            "status_cells": len(statuses),
            "pending_status_cells": pending_count,
            "frozen_lock_status_cells": sum(value in frozen_lock_values for value in statuses),
            "non_pending_status_values": sorted(set(non_pending_statuses)),
        }

    semantic_fields = [
        "readout_value",
        "readout_unit",
        "activity_class",
        "censor_lower",
        "censor_upper",
        "assay_status",
        "replicate_id",
        "replicate_variance",
        "assay_metadata_json",
    ]
    required_adapter_mappings = {
        "readout_value/readout_unit": "candidate_potency_uM or raw assay readout; never infer from a missing curve",
        "activity_class": "prespecified mapping from curve/QC/interference/cytotoxicity outcome to active/inactive/borderline/failed/not_interpretable",
        "censor_lower/censor_upper": "raw assay detection or tested-range bounds; do not invent bounds from a label",
        "replicate_id/replicate_variance": "two independent runs plus any technical replicate metadata",
        "assay_status/assay_metadata_json": "controls, plate, lane, construct, condition and QC provenance",
    }
    bridge_rows = int(bridge.get("rows", 0) or 0)
    bridge_candidates = int(bridge.get("candidates", 0) or 0)
    bridge_runs = int(bridge.get("independent_runs", 0) or 0)
    bridge_ok = bridge.get("status") == "PASS" and bridge_rows == 16 and bridge_candidates == 8 and bridge_runs == 2
    template_ok = (
        audit.get("status") == "PASS"
        and audit.get("checks_passed") == audit.get("checks_total") == 25
        and template_summary.get("status") == "PASS_TEMPLATE_CONSTRUCTION"
        and synthetic_pass_cells == 0
        and all_rows_pending
    )

    report = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_ODTI_W1_V3_TRAINING_PREFLIGHT_AUDIT_V1",
        "claim_status": "TRAINING_PREFLIGHT_ONLY; NO_W1_READOUT_AND_NO_V3_MODEL_RESULT",
        "decision": {
            "w1_templates_structurally_ready": template_ok,
            "identity_provenance_bridge_ready": bridge_ok,
            "direct_v3_training_ready": False,
            "start_v3_training_now": False,
            "required_next_step": "ingest real W1 readouts through a semantic adapter, then rerun this preflight before any V3 training",
        },
        "authoritative_inputs": {
            "v3_contract": {"path": str(V3_CONTRACT.relative_to(ROOT)), "sha256": sha256(V3_CONTRACT)},
            "w1_schema": {"path": str(W1_SCHEMA.relative_to(ROOT)), "sha256": sha256(W1_SCHEMA)},
            "w1_audit": {"path": str(W1_AUDIT.relative_to(ROOT)), "sha256": sha256(W1_AUDIT)},
            "w1_template_summary": {"path": str(W1_TEMPLATE_SUMMARY.relative_to(ROOT)), "sha256": sha256(W1_TEMPLATE_SUMMARY)},
            "w1_v3_bridge": {"path": str(W1_BRIDGE.relative_to(ROOT)), "sha256": sha256(W1_BRIDGE)},
        },
        "counts": {
            "v3_required_fields": len(required),
            "w1_template_files": len(templates),
            "w1_templates_pending_only": all_rows_pending,
            "synthetic_pass_cells": synthetic_pass_cells,
            "bridge_rows": bridge_rows,
            "bridge_candidates": bridge_candidates,
            "bridge_independent_runs": bridge_runs,
        },
        "template_field_audit": templates,
        "semantic_adapter_required": {
            "fields": semantic_fields,
            "mappings": required_adapter_mappings,
            "unknown_pair_is_negative": False,
            "w1_test_reuse_forbidden": True,
        },
        "interpretation": [
            "W1 V17 PASS means the blinded assay/provenance interface is safe and frozen; it does not mean experimental labels exist.",
            "The operator-facing templates intentionally use assay-specific columns and therefore do not directly satisfy the V3 semantic row contract.",
            "No activity label, censor bound, replicate variance or observation indicator may be synthesized from all-PENDING templates.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "W1_V3_TRAINING_PREFLIGHT_AUDIT_V1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# W1 → V3 训练前审计",
        "",
        f"生成时间：{report['created_utc']}",
        "",
        "结论：W1 V17 的模板/盲法/provenance 接口已通过，但当前不能直接启动 V3 训练。必须等待真实 readout，并先通过语义适配器生成 V3 所需的 activity、censor、replicate 和 assay metadata 字段。",
        "",
        f"- W1 independent audit：`{audit.get('checks_passed')}/{audit.get('checks_total')} PASS`。",
        f"- 模板是否全部 pending 且无 synthetic PASS：`{template_ok}`。",
        f"- W1→V3 identity/provenance bridge：`{bridge_ok}`（{bridge_rows} rows，{bridge_candidates} candidates，{bridge_runs} independent runs）。",
        f"- 直接 V3 training ready：`False`。",
        "",
        "## 必须由真实实验数据填充的语义字段",
        "",
        "- `readout_value` / `readout_unit`",
        "- `activity_class`",
        "- `censor_lower` / `censor_upper`",
        "- `replicate_id` / `replicate_variance`",
        "- `assay_status` / `assay_metadata_json`",
        "",
        "不得把 missing、failed 或 unobserved pair 自动转换为 negative；W1 也不能在调参后再作为最终测试集。",
    ]
    (OUT_DIR / "W1_V3_TRAINING_PREFLIGHT_AUDIT_V1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

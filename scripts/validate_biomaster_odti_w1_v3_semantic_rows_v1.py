#!/usr/bin/env python3
"""Validate explicit W1 semantic rows before any V3 training append."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biomaster.odti_w1_v3 import W1V3AdapterError, adapt_w1_v3_rows  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "outputs/biomaster_odti_w1_v3_semantic_input_v1/"
    "W1_V3_SEMANTIC_RESULT_INPUT_16_V1.csv"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_w1_v3_semantic_adapter_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    input_csv = args.input_csv.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)

    rows = load_rows(input_csv)
    base = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_ODTI_W1_V3_SEMANTIC_ADAPTER_V1",
        "input_csv": str(input_csv),
        "input_sha256": sha256(input_csv),
        "input_rows": len(rows),
        "claim_status": "SEMANTIC_ADAPTER_AUDIT_ONLY; NO_MODEL_TRAINING",
    }
    try:
        normalized, summary = adapt_w1_v3_rows(rows)
    except W1V3AdapterError as exc:
        report = {
            **base,
            "status": "REJECTED_NOT_TRAINING_READY",
            "accepted": False,
            "output_csv_written": False,
            "errors": exc.errors,
            "decision": {
                "start_v3_training": False,
                "unknown_pair_is_negative": False,
                "required_next_step": "replace template/pending fields with controlled post-assay semantic observations and rerun validation",
            },
        }
        (out_dir / "W1_V3_SEMANTIC_ADAPTER_AUDIT_V1.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    output_csv = out_dir / "W1_V3_SEMANTIC_ROWS_VALIDATED_V1.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(normalized[0]) if normalized else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
    report = {
        **base,
        "status": "PASS_SEMANTIC_ROWS_VALIDATED",
        "accepted": True,
        "output_csv_written": True,
        "output_csv": str(output_csv),
        "output_sha256": sha256(output_csv),
        "summary": summary,
        "decision": {
            "start_v3_training": False,
            "safe_for_append_only_observation_store": True,
            "requires_split_freeze_and_external_w1_test_exclusion": True,
        },
    }
    (out_dir / "W1_V3_SEMANTIC_ADAPTER_AUDIT_V1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

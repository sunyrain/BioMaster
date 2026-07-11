#!/usr/bin/env python3
"""Prepare standardized second-pass adjudication batches for raw grade-D reviews."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "pair_id",
    "adjudicated_feasibility_grade",
    "adjudicated_verdict",
    "adjudicated_hard_exclusion",
    "adjudicated_exclusion_reason",
    "adjudicated_exposure_status",
    "adjudicated_assay_status",
    "adjudicated_confidence",
    "adjudicated_sources",
    "adjudicated_utc",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()
    reviewed = pd.read_csv(args.reviewed, low_memory=False).fillna("")
    raw_d = reviewed.loc[reviewed["agent_feasibility_grade"].astype(str).eq("D")].copy()
    if raw_d.empty or raw_d["pair_id"].duplicated().any():
        raise RuntimeError("Raw grade-D adjudication scope is empty or contains duplicate pair_id")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_d.to_csv(out_dir / "grade_d_recalibration_input_all.csv", index=False)
    batch_size = math.ceil(len(raw_d) / args.batches)
    manifest = []
    for index in range(args.batches):
        start = index * batch_size
        batch = raw_d.iloc[start : start + batch_size].copy()
        if batch.empty:
            continue
        name = f"recal_batch_{index + 1:02d}"
        input_path = out_dir / f"{name}_input.csv"
        output_path = out_dir / f"{name}_review.csv"
        batch.to_csv(input_path, index=False)
        manifest.append(
            {
                "batch": name,
                "rows": len(batch),
                "input": str(input_path),
                "output": str(output_path),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_grade_d_rows": len(raw_d),
        "unique_pairs": raw_d["pair_id"].nunique(),
        "batches": len(manifest),
        "batch_size_max": max(item["rows"] for item in manifest),
        "required_output_columns": OUTPUT_COLUMNS,
        "batch_manifest": manifest,
    }
    (out_dir / "grade_d_recalibration_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

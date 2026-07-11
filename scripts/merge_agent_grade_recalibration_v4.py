#!/usr/bin/env python3
"""Validate grade-D adjudications and merge calibrated grades into review512."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_agent_grade_recalibration_v4 import OUTPUT_COLUMNS


ENUMS = {
    "adjudicated_feasibility_grade": {"A", "B", "C", "D"},
    "adjudicated_exposure_status": {"plausible", "uncertain", "implausible"},
    "adjudicated_assay_status": {"executable", "requires_development", "not_executable"},
    "adjudicated_confidence": {"high", "medium", "low"},
}
SOURCE_RE = re.compile(r"(?:https?://|doi:|pmid:|10\.\d{4,9}/)", re.IGNORECASE)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--recal-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reviewed = pd.read_csv(args.reviewed, low_memory=False).fillna("")
    raw_d_pairs = set(
        reviewed.loc[reviewed["agent_feasibility_grade"].astype(str).eq("D"), "pair_id"].astype(str)
    )
    files = sorted(Path(args.recal_dir).glob("recal_batch_*_review.csv"))
    if not files:
        raise FileNotFoundError("No recalibration review files found")
    frames = [pd.read_csv(path, low_memory=False).fillna("") for path in files]
    for path, frame in zip(files, frames):
        if list(frame.columns) != OUTPUT_COLUMNS:
            raise RuntimeError(f"Unexpected adjudication columns: {path}")
    adjudicated = pd.concat(frames, ignore_index=True)
    if adjudicated["pair_id"].duplicated().any() or set(adjudicated["pair_id"].astype(str)) != raw_d_pairs:
        raise RuntimeError("Adjudication pair coverage does not exactly match raw grade-D scope")
    if adjudicated.astype(str).apply(lambda column: column.str.strip().eq("")).any().any():
        raise RuntimeError("Adjudication contains empty required fields")
    for column, allowed in ENUMS.items():
        invalid = set(adjudicated[column].astype(str)) - allowed
        if invalid:
            raise RuntimeError(f"Invalid {column}: {sorted(invalid)}")
    hard = adjudicated["adjudicated_hard_exclusion"].map(as_bool)
    if (~adjudicated["adjudicated_hard_exclusion"].astype(str).str.lower().isin({"true", "false", "1", "0", "1.0", "0.0", "yes", "no"})).any():
        raise RuntimeError("adjudicated_hard_exclusion is not boolean-like")
    if (hard & ~adjudicated["adjudicated_feasibility_grade"].astype(str).eq("D")).any():
        raise RuntimeError("Hard-excluded adjudications must remain grade D")
    if ((~hard) & adjudicated["adjudicated_exclusion_reason"].astype(str).str.lower().ne("none")).any():
        raise RuntimeError("Non-excluded adjudications must use exclusion reason 'none'")
    if (~adjudicated["adjudicated_sources"].astype(str).map(lambda value: bool(SOURCE_RE.search(value)))).any():
        raise RuntimeError("Every adjudication needs a URL, PMID, or DOI source")

    output = reviewed.copy()
    output["agent_feasibility_grade_initial"] = output["agent_feasibility_grade"]
    output["agent_verdict_initial"] = output["agent_verdict"]
    output["agent_grade_adjudication_applied"] = False
    adjudicated_index = adjudicated.set_index("pair_id")
    target = output["pair_id"].astype(str).isin(raw_d_pairs)
    for index in output.index[target]:
        pair_id = str(output.at[index, "pair_id"])
        row = adjudicated_index.loc[pair_id]
        output.at[index, "agent_feasibility_grade"] = row["adjudicated_feasibility_grade"]
        output.at[index, "agent_verdict"] = row["adjudicated_verdict"]
        output.at[index, "agent_grade_adjudication_applied"] = True
        for column in OUTPUT_COLUMNS[2:]:
            output.at[index, column] = row[column]
    for column in OUTPUT_COLUMNS[2:]:
        if column not in output.columns:
            output[column] = "not_applicable_initial_non_d"
        else:
            output[column] = output[column].fillna("").replace("", "not_applicable_initial_non_d")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": len(output),
        "raw_grade_d_rows": len(raw_d_pairs),
        "adjudication_files": len(files),
        "hard_exclusion_rows": int(hard.sum()),
        "calibrated_grade_counts": output["agent_feasibility_grade"].value_counts().to_dict(),
        "initial_grade_counts": output["agent_feasibility_grade_initial"].value_counts().to_dict(),
    }
    (output_path.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

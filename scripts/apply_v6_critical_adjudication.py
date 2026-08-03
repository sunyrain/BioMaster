#!/usr/bin/env python3
"""Merge validated critical adjudications into the 625-row V6 deep-review table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True)
    parser.add_argument("--stage2", required=True)
    parser.add_argument("--adjudications-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-adjudications", type=int, default=384)
    args = parser.parse_args()

    first = pd.concat(
        [
            pd.read_csv(args.stage1, low_memory=False).fillna(""),
            pd.read_csv(args.stage2, low_memory=False).fillna(""),
        ],
        ignore_index=True,
    )
    if len(first) != 625 or first["pair_id"].nunique() != 625:
        raise ValueError("Expected 625 unique first-pass reviews")
    paths = []
    for path in sorted(Path(args.adjudications_dir).glob("batch_*_adjudication.csv")):
        audit_path = path.with_suffix(".audit.json")
        if not audit_path.exists():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            raise ValueError(f"Adjudication audit did not pass: {audit_path}")
        paths.append(path)
    if not paths:
        raise ValueError("No audited adjudication files found")
    adjudication = pd.concat([pd.read_csv(path, low_memory=False).fillna("") for path in paths])
    if len(adjudication) != args.expected_adjudications:
        raise ValueError(
            f"Expected {args.expected_adjudications} audited adjudications, observed "
            f"{len(adjudication)} across {len(paths)} batches"
        )
    if adjudication["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id across adjudication batches")
    if adjudication["adjudication_decision"].astype(str).str.strip().eq("").any():
        raise ValueError("Audited adjudication set contains empty decisions")
    missing = set(adjudication["pair_id"]) - set(first["pair_id"])
    if missing:
        raise ValueError(f"Unexpected adjudication pair_id values: {sorted(missing)[:10]}")

    frame = first.merge(adjudication, on="pair_id", how="left", validate="one_to_one").fillna("")
    mask = frame["adjudication_decision"].ne("")
    replacements = {
        "agent_feasibility_grade": "adjudicated_feasibility_grade",
        "agent_literature_class": "adjudicated_literature_class",
        "agent_active_species_status": "adjudicated_active_species_status",
        "agent_database_query_resolution": "adjudicated_database_query_resolution",
        "agent_exposure_feasibility": "adjudicated_exposure_feasibility",
        "agent_verdict": "adjudicated_verdict",
        "agent_confidence": "adjudication_confidence",
        "agent_reviewed_utc": "adjudicated_utc",
    }
    for target, source in replacements.items():
        frame.loc[mask, target] = frame.loc[mask, source]
    frame.loc[mask, "agent_sources"] = (
        frame.loc[mask, "agent_sources"].astype(str)
        + "; independent adjudication: "
        + frame.loc[mask, "adjudication_sources"].astype(str)
    )
    frame.loc[mask, "agent_key_risks"] = (
        frame.loc[mask, "agent_key_risks"].astype(str)
        + "；独立裁决："
        + frame.loc[mask, "adjudication_rationale"].astype(str)
    )
    frame["agent_adjudication_decision"] = frame["adjudication_decision"]
    frame["agent_adjudication_rationale"] = frame["adjudication_rationale"]
    frame["agent_adjudication_sources"] = frame["adjudication_sources"]
    frame["agent_adjudicated_utc"] = frame["adjudicated_utc"]
    frame["agent_adjudication_applied"] = mask

    grade_change = mask & frame["adjudicated_feasibility_grade"].ne(
        first.set_index("pair_id").loc[frame["pair_id"], "agent_feasibility_grade"].to_numpy()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(frame)),
        "adjudicated_rows": int(mask.sum()),
        "grade_changes": int(grade_change.sum()),
        "final_grade_counts": frame["agent_feasibility_grade"].value_counts().to_dict(),
        "final_literature_counts": frame["agent_literature_class"].value_counts().to_dict(),
        "final_active_species_counts": frame["agent_active_species_status"].value_counts().to_dict(),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

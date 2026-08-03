#!/usr/bin/env python3
"""Fail-closed validation for one V6 critical-adjudication batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COLUMNS = [
    "pair_id",
    "adjudication_decision",
    "adjudicated_feasibility_grade",
    "adjudicated_literature_class",
    "adjudicated_active_species_status",
    "adjudicated_database_query_resolution",
    "adjudicated_exposure_feasibility",
    "adjudicated_verdict",
    "adjudication_rationale",
    "adjudication_confidence",
    "adjudication_sources",
    "adjudicated_utc",
]

ALLOWED = {
    "adjudication_decision": {"confirm", "upgrade", "downgrade", "revise_non_grade"},
    "adjudicated_feasibility_grade": {"A", "B", "C", "D"},
    "adjudicated_literature_class": {
        "exact_pair_validated",
        "functional_only",
        "indirect_or_family_only",
        "no_exact_report_found",
        "contradictory",
    },
    "adjudicated_active_species_status": {
        "parent_drug_relevant",
        "salt_normalization_adequate",
        "active_species_uncertain",
        "prodrug_active_metabolite_requires_rerun",
    },
    "adjudicated_database_query_resolution": {"not_needed", "resolved_manually", "unresolved"},
    "adjudication_confidence": {"high", "medium", "low"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--adjudication", required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input, low_memory=False).fillna("")
    result = pd.read_csv(args.adjudication, low_memory=False).fillna("")
    missing = set(COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"Missing adjudication columns: {sorted(missing)}")
    result = result[COLUMNS]
    if result["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id in adjudication")
    expected = set(source["pair_id"].astype(str))
    observed = set(result["pair_id"].astype(str))
    if len(source) != len(result) or expected != observed:
        raise ValueError(
            f"Coverage mismatch: source={len(source)}, adjudication={len(result)}, "
            f"missing={sorted(expected-observed)[:5]}, unexpected={sorted(observed-expected)[:5]}"
        )
    empties = {
        column: int(result[column].astype(str).str.strip().eq("").sum()) for column in COLUMNS
    }
    empties = {column: count for column, count in empties.items() if count}
    if empties:
        raise ValueError(f"Empty required fields: {empties}")
    for column, allowed in ALLOWED.items():
        invalid = sorted(set(result[column].astype(str)) - allowed)
        if invalid:
            raise ValueError(f"Invalid {column}: {invalid}")
    source_ok = result["adjudication_sources"].astype(str).str.contains(
        r"https?://|\bPMID\b|\bDOI\b|\b10\.\d{4,9}/", case=False, regex=True, na=False
    )
    if not source_ok.all():
        raise ValueError(
            f"Rows without resolvable source: {result.loc[~source_ok, 'pair_id'].tolist()[:10]}"
        )
    reviewed_time = pd.to_datetime(
        result["adjudicated_utc"], utc=True, errors="coerce", format="mixed"
    )
    if reviewed_time.isna().any():
        raise ValueError(
            f"Rows with invalid UTC: {result.loc[reviewed_time.isna(), 'pair_id'].tolist()[:10]}"
        )
    initial = source.set_index("pair_id")
    by_pair = result.set_index("pair_id")
    grade_changed = by_pair["adjudicated_feasibility_grade"].ne(
        initial.loc[by_pair.index, "agent_feasibility_grade"]
    )
    summary = {
        "status": "pass",
        "rows": int(len(result)),
        "decision_counts": result["adjudication_decision"].value_counts().to_dict(),
        "grade_counts": result["adjudicated_feasibility_grade"].value_counts().to_dict(),
        "grade_changes": int(grade_changed.sum()),
        "literature_counts": result["adjudicated_literature_class"].value_counts().to_dict(),
        "active_species_counts": result["adjudicated_active_species_status"].value_counts().to_dict(),
    }
    path = Path(args.adjudication).with_suffix(".audit.json")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed validation for one V6 agent deep-review batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = [
    "pair_id",
    "agent_feasibility_grade",
    "agent_verdict",
    "agent_literature_class",
    "agent_primary_disease",
    "agent_repurposing_status",
    "agent_disease_evidence",
    "agent_mechanism_rationale",
    "agent_exposure_feasibility",
    "agent_active_species_status",
    "agent_assay_plan",
    "agent_key_risks",
    "agent_database_query_resolution",
    "agent_confidence",
    "agent_sources",
    "agent_reviewed_utc",
]

ALLOWED = {
    "agent_feasibility_grade": {"A", "B", "C", "D"},
    "agent_literature_class": {
        "exact_pair_validated",
        "functional_only",
        "indirect_or_family_only",
        "no_exact_report_found",
        "contradictory",
    },
    "agent_repurposing_status": {
        "new_disease_area",
        "new_indication_same_area",
        "target_only_no_disease_claim",
        "original_indication_or_not_repurposing",
    },
    "agent_active_species_status": {
        "parent_drug_relevant",
        "salt_normalization_adequate",
        "active_species_uncertain",
        "prodrug_active_metabolite_requires_rerun",
    },
    "agent_database_query_resolution": {"not_needed", "resolved_manually", "unresolved"},
    "agent_confidence": {"high", "medium", "low"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    source = pd.read_csv(args.input, low_memory=False).fillna("")
    review = pd.read_csv(args.review, low_memory=False).fillna("")
    missing = set(REVIEW_COLUMNS) - set(review.columns)
    if missing:
        raise ValueError(f"Missing review columns: {sorted(missing)}")
    review = review[REVIEW_COLUMNS]
    if review["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id in review")
    expected = set(source["pair_id"].astype(str))
    observed = set(review["pair_id"].astype(str))
    if expected != observed or len(source) != len(review):
        raise ValueError(
            f"Pair coverage mismatch: source={len(source)}, review={len(review)}, "
            f"missing={sorted(expected-observed)[:5]}, unexpected={sorted(observed-expected)[:5]}"
        )
    empties = {
        column: int(review[column].astype(str).str.strip().eq("").sum())
        for column in REVIEW_COLUMNS
    }
    empties = {column: count for column, count in empties.items() if count}
    if empties:
        raise ValueError(f"Empty required fields: {empties}")
    for column, allowed in ALLOWED.items():
        invalid = sorted(set(review[column].astype(str)) - allowed)
        if invalid:
            raise ValueError(f"Invalid {column}: {invalid}")
    source_ok = review["agent_sources"].astype(str).str.contains(
        r"https?://|\bPMID\b|\bDOI\b|\b10\.\d{4,9}/", case=False, regex=True, na=False
    )
    if not source_ok.all():
        bad = review.loc[~source_ok, "pair_id"].tolist()
        raise ValueError(f"Rows without resolvable source: {bad[:10]}")
    reviewed_time = pd.to_datetime(
        review["agent_reviewed_utc"], utc=True, errors="coerce", format="mixed"
    )
    if reviewed_time.isna().any():
        bad = review.loc[reviewed_time.isna(), "pair_id"].tolist()
        raise ValueError(f"Rows with invalid review UTC: {bad[:10]}")
    summary = {
        "status": "pass",
        "rows": int(len(review)),
        "unique_pairs": int(review["pair_id"].nunique()),
        "grade_counts": review["agent_feasibility_grade"].value_counts().to_dict(),
        "literature_counts": review["agent_literature_class"].value_counts().to_dict(),
        "active_species_counts": review["agent_active_species_status"].value_counts().to_dict(),
        "database_resolution_counts": review["agent_database_query_resolution"].value_counts().to_dict(),
        "confidence_counts": review["agent_confidence"].value_counts().to_dict(),
    }
    summary_path = Path(args.summary) if args.summary else Path(args.review).with_suffix(".audit.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

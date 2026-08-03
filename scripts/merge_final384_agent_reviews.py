#!/usr/bin/env python3
"""Validate and merge all final384 subagent review files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-all", required=True)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input_all, low_memory=False).fillna("")
    review_paths = sorted(Path(args.reviews_dir).glob("batch_*_review.csv"))
    if not review_paths:
        raise FileNotFoundError(f"No batch review files in {args.reviews_dir}")
    reviews = []
    for path in review_paths:
        frame = pd.read_csv(path, low_memory=False).fillna("")
        missing = set(REVIEW_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing review columns: {sorted(missing)}")
        frame = frame[REVIEW_COLUMNS].copy()
        frame["agent_review_batch"] = path.stem
        reviews.append(frame)
    review = pd.concat(reviews, ignore_index=True)
    if review["pair_id"].duplicated().any():
        duplicate_ids = review.loc[review["pair_id"].duplicated(False), "pair_id"].unique().tolist()
        raise ValueError(f"Duplicate reviewed pair_id values: {duplicate_ids[:10]}")
    expected = set(source["pair_id"].astype(str))
    observed = set(review["pair_id"].astype(str))
    if expected != observed:
        raise ValueError(
            f"Review coverage mismatch: missing={sorted(expected - observed)[:10]}, "
            f"unexpected={sorted(observed - expected)[:10]}"
        )

    required_nonempty = list(REVIEW_COLUMNS)
    missing_values = {
        column: int(review[column].astype(str).str.strip().eq("").sum()) for column in required_nonempty
    }
    missing_values = {column: count for column, count in missing_values.items() if count}
    if missing_values:
        raise ValueError(f"Incomplete agent reviews: {missing_values}")
    invalid_grades = sorted(set(review["agent_feasibility_grade"]) - {"A", "B", "C", "D"})
    if invalid_grades:
        raise ValueError(f"Invalid feasibility grades: {invalid_grades}")
    invalid_confidence = sorted(set(review["agent_confidence"]) - {"high", "medium", "low"})
    if invalid_confidence:
        raise ValueError(f"Invalid confidence values: {invalid_confidence}")
    invalid_resolution = sorted(
        set(review["agent_database_query_resolution"])
        - {"not_needed", "resolved_manually", "unresolved"}
    )
    if invalid_resolution:
        raise ValueError(f"Invalid database query resolution values: {invalid_resolution}")
    invalid_repurposing = sorted(
        set(review["agent_repurposing_status"])
        - {
            "new_disease_area",
            "new_indication_same_area",
            "target_only_no_disease_claim",
            "original_indication_or_not_repurposing",
        }
    )
    if invalid_repurposing:
        raise ValueError(f"Invalid repurposing status values: {invalid_repurposing}")
    invalid_active_species = sorted(
        set(review["agent_active_species_status"])
        - {
            "parent_drug_relevant",
            "salt_normalization_adequate",
            "active_species_uncertain",
            "prodrug_active_metabolite_requires_rerun",
        }
    )
    if invalid_active_species:
        raise ValueError(f"Invalid active-species status values: {invalid_active_species}")
    allowed_literature = {
        "exact_pair_validated",
        "functional_only",
        "indirect_or_family_only",
        "no_exact_report_found",
        "contradictory",
    }
    invalid_literature = sorted(set(review["agent_literature_class"]) - allowed_literature)
    if invalid_literature:
        raise ValueError(f"Invalid literature classes: {invalid_literature}")
    source_ok = review["agent_sources"].astype(str).str.contains(
        r"https?://|\bPMID\b|\bDOI\b|\b10\.\d{4,9}/", case=False, regex=True, na=False
    )
    if not source_ok.all():
        raise ValueError(
            f"Agent reviews without a resolvable DOI/PMID/URL source: {int((~source_ok).sum())}"
        )
    manually_resolved = review["agent_database_query_resolution"].eq("resolved_manually")
    if (manually_resolved & ~source_ok).any():
        raise ValueError("A manually resolved database failure lacks a resolvable source")
    reviewed_time = pd.to_datetime(
        review["agent_reviewed_utc"], utc=True, errors="coerce", format="mixed"
    )
    if reviewed_time.isna().any():
        raise ValueError(f"Invalid agent_reviewed_utc values: {int(reviewed_time.isna().sum())}")

    output = source.merge(review, on="pair_id", how="left", validate="one_to_one")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": len(output),
        "unique_pairs": int(output["pair_id"].nunique()),
        "review_files": len(review_paths),
        "grade_counts": output["agent_feasibility_grade"].value_counts().to_dict(),
        "literature_class_counts": output["agent_literature_class"].value_counts().to_dict(),
        "confidence_counts": output["agent_confidence"].value_counts().to_dict(),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

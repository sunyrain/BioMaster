#!/usr/bin/env python3
"""Merge legacy 375 and newly reviewed 625 rows into one current V6 review table."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


AGENT_COLUMNS = [
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


def load_reviews(path: str, provenance: str, current_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).fillna("")
    required = {"pair_id", *AGENT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.loc[frame["pair_id"].astype(str).isin(current_ids)].copy()
    if frame["pair_id"].duplicated().any():
        raise ValueError(f"Duplicate pair_id in {path}")
    adjudication_columns = sorted(
        column for column in frame.columns if column.startswith("agent_adjudicat")
    )
    keep = ["pair_id", *AGENT_COLUMNS, *adjudication_columns]
    if "agent_review_batch" in frame.columns:
        keep.append("agent_review_batch")
    frame = frame[keep]
    if "agent_review_batch" not in frame.columns:
        frame["agent_review_batch"] = provenance
    frame["deep_review_provenance_v6"] = provenance
    return frame


def validate(frame: pd.DataFrame, expected_ids: set[str]) -> dict[str, object]:
    if frame["pair_id"].duplicated().any():
        duplicated = frame.loc[frame["pair_id"].duplicated(False), "pair_id"].tolist()
        raise ValueError(f"Duplicate combined pair_id values: {duplicated[:10]}")
    observed = set(frame["pair_id"].astype(str))
    if observed != expected_ids:
        raise ValueError(
            f"Combined coverage mismatch: missing={sorted(expected_ids-observed)[:10]}, "
            f"unexpected={sorted(observed-expected_ids)[:10]}"
        )
    empties = {
        column: int(frame[column].astype(str).str.strip().eq("").sum())
        for column in AGENT_COLUMNS
    }
    empties = {column: count for column, count in empties.items() if count}
    if empties:
        raise ValueError(f"Empty deep-review fields: {empties}")
    for column, allowed in ALLOWED.items():
        invalid = sorted(set(frame[column].astype(str)) - allowed)
        if invalid:
            raise ValueError(f"Invalid {column}: {invalid}")
    source_ok = frame["agent_sources"].astype(str).str.contains(
        r"https?://|\bPMID\b|\bDOI\b|\b10\.\d{4,9}/", case=False, regex=True, na=False
    )
    if not source_ok.all():
        raise ValueError(f"Rows without resolvable sources: {int((~source_ok).sum())}")
    reviewed_time = pd.to_datetime(
        frame["agent_reviewed_utc"], utc=True, errors="coerce", format="mixed"
    )
    if reviewed_time.isna().any():
        raise ValueError(f"Invalid reviewed UTC values: {int(reviewed_time.isna().sum())}")
    return {
        "rows": int(len(frame)),
        "unique_pairs": int(frame["pair_id"].nunique()),
        "grade_counts": frame["agent_feasibility_grade"].value_counts().to_dict(),
        "literature_counts": frame["agent_literature_class"].value_counts().to_dict(),
        "active_species_counts": frame["agent_active_species_status"].value_counts().to_dict(),
        "repurposing_counts": frame["agent_repurposing_status"].value_counts().to_dict(),
        "confidence_counts": frame["agent_confidence"].value_counts().to_dict(),
        "provenance_counts": frame["deep_review_provenance_v6"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current1000", required=True)
    parser.add_argument("--legacy-review", required=True)
    parser.add_argument("--stage1-review", required=True)
    parser.add_argument("--stage2-review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    current = pd.read_csv(args.current1000, usecols=["pair_id"], low_memory=False).fillna("")
    expected_ids = set(current["pair_id"].astype(str))
    frames = [
        load_reviews(args.legacy_review, "legacy_v4_deep_review", expected_ids),
        load_reviews(args.stage1_review, "v6_stage1_current_top500_deep_review", expected_ids),
        load_reviews(args.stage2_review, "v6_stage2_reserve_deep_review", expected_ids),
    ]
    combined = pd.concat(frames, ignore_index=True).fillna("")
    summary = validate(combined, expected_ids)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    summary["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

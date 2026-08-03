#!/usr/bin/env python3
"""Create the deterministic 625-pair V6 deep-review priority queue."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "1.0", "yes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False).fillna("")
    queue = frame.loc[frame["review_depth_v6"].eq("systematic_database_and_rule_review")].copy()
    if len(queue) != 625 or queue["pair_id"].nunique() != 625:
        raise ValueError(f"Expected 625 unique systematic-review pairs, found {len(queue)}")
    queue["in_final500_v6_bool"] = truthy(queue["in_final500_v6"])
    queue["deep_review_stage_v6"] = queue["in_final500_v6_bool"].map(
        {True: "stage1_current_top500", False: "stage2_reserve_pool"}
    )
    queue["top500_rank_sort"] = pd.to_numeric(queue["top500_rank_v6"], errors="coerce").fillna(10**9)
    queue["strength_sort"] = pd.to_numeric(queue["v5_strength_order_v6"], errors="coerce").fillna(9)
    queue["score_sort"] = pd.to_numeric(queue["top500_selection_score_v6"], errors="coerce").fillna(-1)
    queue = queue.sort_values(
        ["in_final500_v6_bool", "top500_rank_sort", "strength_sort", "score_sort", "pair_id"],
        ascending=[False, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    queue.insert(0, "deep_review_priority_v6", range(1, len(queue) + 1))
    queue = queue.drop(columns=["in_final500_v6_bool", "top500_rank_sort", "strength_sort", "score_sort"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False)
    stage1_path = output.parent / "STAGE1_CURRENT_TOP500_UNREVIEWED183.csv"
    stage2_path = output.parent / "STAGE2_RESERVE_UNREVIEWED442.csv"
    queue.loc[queue["deep_review_stage_v6"].eq("stage1_current_top500")].to_csv(stage1_path, index=False)
    queue.loc[queue["deep_review_stage_v6"].eq("stage2_reserve_pool")].to_csv(stage2_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": int(len(queue)),
        "unique_pairs": int(queue["pair_id"].nunique()),
        "stage_counts": queue["deep_review_stage_v6"].value_counts().to_dict(),
        "stage1_priority_range": [
            int(queue.loc[queue["deep_review_stage_v6"].eq("stage1_current_top500"), "deep_review_priority_v6"].min()),
            int(queue.loc[queue["deep_review_stage_v6"].eq("stage1_current_top500"), "deep_review_priority_v6"].max()),
        ],
        "stage2_priority_range": [
            int(queue.loc[queue["deep_review_stage_v6"].eq("stage2_reserve_pool"), "deep_review_priority_v6"].min()),
            int(queue.loc[queue["deep_review_stage_v6"].eq("stage2_reserve_pool"), "deep_review_priority_v6"].max()),
        ],
        "stage1_file": str(stage1_path),
        "stage2_file": str(stage2_path),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

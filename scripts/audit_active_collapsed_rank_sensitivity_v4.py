#!/usr/bin/env python3
"""Compare ID-weighted and structure-collapsed Top3000 prefilter selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "pre_boltz_top3000_v4.csv"
        ),
    )
    parser.add_argument(
        "--collapsed",
        default=(
            "outputs/current_production_package_v2/"
            "full_untruncated_universe_v4_active_collapsed_sensitivity/"
            "pre_boltz_top3000_v4_active_collapsed_sensitivity.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/current_production_package_v2/"
            "full_untruncated_universe_v4_active_collapsed_sensitivity"
        ),
    )
    args = parser.parse_args()
    baseline_path = Path(args.baseline)
    collapsed_path = Path(args.collapsed)
    baseline = pd.read_csv(baseline_path, low_memory=False).fillna("")
    collapsed = pd.read_csv(collapsed_path, low_memory=False).fillna("")
    for label, frame in [("baseline", baseline), ("collapsed", collapsed)]:
        if len(frame) != 3000 or frame["pair_id"].duplicated().any():
            raise ValueError(f"{label} Top3000 contract failed")
    baseline_ids = set(baseline["pair_id"].astype(str))
    collapsed_ids = set(collapsed["pair_id"].astype(str))
    differences = pd.concat(
        [
            baseline[baseline["pair_id"].isin(baseline_ids - collapsed_ids)].assign(
                sensitivity_disposition="baseline_only"
            ),
            collapsed[collapsed["pair_id"].isin(collapsed_ids - baseline_ids)].assign(
                sensitivity_disposition="active_collapsed_only"
            ),
        ],
        ignore_index=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    difference_path = out_dir / "active_collapsed_top3000_symmetric_difference_v4.csv"
    differences.to_csv(difference_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_rows": len(baseline),
        "collapsed_rows": len(collapsed),
        "overlap_rows": len(baseline_ids & collapsed_ids),
        "overlap_fraction": len(baseline_ids & collapsed_ids) / 3000,
        "baseline_only_rows": len(baseline_ids - collapsed_ids),
        "active_collapsed_only_rows": len(collapsed_ids - baseline_ids),
        "formal_policy": (
            "Keep the already signed 3000-run as the physical refinement pool; recompute "
            "active-collapsed target rank and shared percentiles in final scoring. The 99%+ "
            "prefilter overlap is reported as sensitivity, not hidden."
        ),
        "source_sha256": {
            "baseline": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "collapsed": hashlib.sha256(collapsed_path.read_bytes()).hexdigest(),
        },
        "difference_sha256": hashlib.sha256(difference_path.read_bytes()).hexdigest(),
    }
    output = out_dir / "active_collapsed_rank_sensitivity_v4.summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

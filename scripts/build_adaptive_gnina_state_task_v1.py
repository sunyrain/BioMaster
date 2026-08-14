#!/usr/bin/env python3
"""Freeze adaptive GNINA-supported pairs for alternate-receptor state testing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1"
    / "adaptive_no_stable_pair_pilot_v1/execution_v1"
)


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=BASE / "gnina_evaluation/GNINA_ADAPTIVE_PILOT_PAIR_EVIDENCE_V1.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "gnina_receptor_state"
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(args.evidence.resolve(), low_memory=False)
    support_column = (
        "gnina_local_support"
        if "gnina_local_support" in evidence.columns
        else "gnina_remote_support"
    )
    tier_column = (
        "gnina_local_evidence_tier"
        if "gnina_local_evidence_tier" in evidence.columns
        else "gnina_remote_evidence_tier"
    )
    selected = evidence[evidence[support_column].map(truth)].copy()
    if selected.empty:
        raise RuntimeError("No adaptive GNINA-supported pair is available")

    tier_order = {
        "GNINA_LOCAL_A": 1, "GNINA_LOCAL_B": 2, "GNINA_LOCAL_C": 3,
        "GNINA_REMOTE_A": 1, "GNINA_REMOTE_B": 2, "GNINA_REMOTE_C": 3,
    }
    selected["pair_lane_priority"] = selected[tier_column].map(
        tier_order
    ).fillna(9).astype(int)
    selected["gnina_support"] = True
    selected["pair_evidence_lane"] = "ADAPTIVE_GNINA_PRIMARY_SUPPORT"
    selected["pair_evidence_lane_zh"] = "自适应队列GNINA主结构局部支持"
    selected["unified_novelty_class"] = selected.get(
        "novelty_lane", pd.Series("", index=selected.index)
    )
    selected["target_route"] = selected.get(
        "current_target_route", pd.Series("", index=selected.index)
    )
    selected["task_status"] = "READY_FOR_ALTERNATE_SELECTION"
    selected = selected.sort_values(
        ["pair_lane_priority", "target_chembl_id", "pairId"], kind="mergesort"
    ).reset_index(drop=True)

    task_path = output_dir / "GNINA_ADAPTIVE_PRIMARY_SUPPORT_STATE_TASK_V1.csv"
    selected.to_csv(task_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "pairs": len(selected),
        "targets": int(selected["target_chembl_id"].nunique()),
        "tier_counts": selected[tier_column].value_counts().to_dict(),
        "purpose": (
            "Test whether primary-structure GNINA support persists in independently "
            "redocked receptor states; absence of an eligible alternate is inconclusive."
        ),
        "output": str(task_path),
    }
    (output_dir / "GNINA_ADAPTIVE_STATE_TASK_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

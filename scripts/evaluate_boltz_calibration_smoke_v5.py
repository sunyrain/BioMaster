#!/usr/bin/env python3
"""Evaluate paired ChEMBL positive/negative Boltz smoke results and gate expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "outputs/current_production_package_v2/boltz_target_calibration_smoke_v5/audit"
    / "boltz2_complex_validation_candidate_audit.csv"
)
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/boltz_target_calibration_smoke_v5/evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.audit.is_absolute():
        args.audit = (ROOT / args.audit).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    data = pd.read_csv(args.audit)
    if len(data) != 20 or not data["boltzCompleted"].astype(bool).all():
        raise ValueError("Expected 20 complete Boltz smoke rows")
    labels = data["knownDrugTargetPair"].astype(bool).astype(int)
    score_columns = [
        "boltzAffinityProbabilityBinary",
        "boltzConfidenceScore",
        "boltzLigandIptm",
        "boltzCompositeScore",
    ]
    rows = []
    for column in score_columns:
        rows.append(
            {
                "score": column,
                "roc_auc": float(roc_auc_score(labels, data[column])),
                "pr_auc": float(average_precision_score(labels, data[column])),
                "positive_median": float(data.loc[labels.eq(1), column].median()),
                "negative_median": float(data.loc[labels.eq(0), column].median()),
            }
        )
    metrics = pd.DataFrame(rows)
    paired = []
    for target, group in data.groupby("target", sort=True):
        if set(group["knownDrugTargetPair"].astype(bool)) != {False, True}:
            raise ValueError(f"Target is not paired: {target}")
        negative = group.loc[~group["knownDrugTargetPair"].astype(bool)].iloc[0]
        positive = group.loc[group["knownDrugTargetPair"].astype(bool)].iloc[0]
        row = {"target": target, "negative_pair_id": negative["pairId"], "positive_pair_id": positive["pairId"]}
        for column in score_columns:
            row[f"{column}_negative"] = float(negative[column])
            row[f"{column}_positive"] = float(positive[column])
            row[f"{column}_delta_positive_minus_negative"] = float(positive[column] - negative[column])
            row[f"{column}_positive_wins"] = bool(positive[column] > negative[column])
        paired.append(row)
    paired_table = pd.DataFrame(paired)
    affinity = metrics.set_index("score").loc["boltzAffinityProbabilityBinary"]
    composite = metrics.set_index("score").loc["boltzCompositeScore"]
    affinity_wins = int(paired_table["boltzAffinityProbabilityBinary_positive_wins"].sum())
    composite_wins = int(paired_table["boltzCompositeScore_positive_wins"].sum())
    expand = bool(
        affinity["roc_auc"] >= 0.70
        and affinity_wins >= 7
        and composite["roc_auc"] >= 0.65
        and composite_wins >= 6
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "BOLTZ_SMOKE_DISCRIMINATION_METRICS_V5.csv", index=False)
    paired_table.to_csv(args.output_dir / "BOLTZ_SMOKE_PAIRED_TARGET_COMPARISON_V5.csv", index=False)
    summary = {
        "status": "passed_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(data)),
        "paired_targets": int(len(paired_table)),
        "affinity_roc_auc": float(affinity["roc_auc"]),
        "affinity_pr_auc": float(affinity["pr_auc"]),
        "affinity_positive_wins": affinity_wins,
        "composite_roc_auc": float(composite["roc_auc"]),
        "composite_pr_auc": float(composite["pr_auc"]),
        "composite_positive_wins": composite_wins,
        "expansion_gate": "pass_expand_to_1000" if expand else "stop_do_not_expand",
        "decision": "Boltz may be used as conditional pose-generation evidence, not as calibrated binding discrimination.",
        "audit_sha256": sha256(args.audit),
    }
    (args.output_dir / "BOLTZ_SMOKE_CALIBRATION_DECISION_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

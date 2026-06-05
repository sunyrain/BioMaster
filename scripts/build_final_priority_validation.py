from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CUTOFFS = [10, 20, 50, 100, 200, 500, 1000, 2000, 3921]


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_auc(y_true: pd.Series, y_score: pd.Series) -> float | None:
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    scores = pd.to_numeric(y_score, errors="coerce").fillna(0)
    if labels.nunique() < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_ap(y_true: pd.Series, y_score: pd.Series) -> float | None:
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    scores = pd.to_numeric(y_score, errors="coerce").fillna(0)
    if labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def cutoff_rows(df: pd.DataFrame, score_col: str, label_col: str, group_type: str, group_value: str) -> list[dict[str, Any]]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    total = len(ranked)
    positives = int(ranked[label_col].sum())
    base_rate = positives / total if total else 0
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        if cutoff > total:
            continue
        top = ranked.head(cutoff)
        hits = int(top[label_col].sum())
        precision = hits / cutoff if cutoff else 0
        recall = hits / positives if positives else 0
        expected = cutoff * base_rate
        enrichment = hits / expected if expected else None
        rows.append(
            {
                "groupType": group_type,
                "groupValue": group_value,
                "cutoff": cutoff,
                "rows": total,
                "positives": positives,
                "hits": hits,
                "precisionPct": round(precision * 100, 4),
                "recallPct": round(recall * 100, 4),
                "randomExpectedHits": round(expected, 4),
                "enrichmentVsRandom": round(enrichment, 4) if enrichment is not None else "",
            }
        )
    return rows


def group_metric(df: pd.DataFrame, group_type: str, group_value: str) -> dict[str, Any]:
    labels = df["knownDrugTargetPair"].astype(int)
    scores = pd.to_numeric(df["finalPriorityScore"], errors="coerce").fillna(0)
    return {
        "groupType": group_type,
        "groupValue": group_value,
        "rows": int(len(df)),
        "positiveRows": int(labels.sum()),
        "positiveRatePct": round(pct(int(labels.sum()), len(df)), 4),
        "auroc": safe_auc(labels, scores),
        "averagePrecision": safe_ap(labels, scores),
        "medianScorePositive": float(scores[labels.eq(1)].median()) if labels.sum() else None,
        "medianScoreNegative": float(scores[labels.eq(0)].median()) if labels.eq(0).sum() else None,
    }


def build_group_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [group_metric(df, "all", "all")]
    for group_col, group_type in [
        ("direction", "direction"),
        ("finalPriorityTier", "finalPriorityTier"),
        ("reviewTrack", "reviewTrack"),
        ("noveltyClass", "noveltyClass"),
        ("poseAuditStatus", "poseAuditStatus"),
        ("admetTier", "admetTier"),
    ]:
        for value, group in df.groupby(group_col, dropna=False):
            rows.append(group_metric(group, group_type, str(value)))
    return rows


def shortlist_metrics(df: pd.DataFrame, root: Path, paths: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known_pairs = set(df[df["knownDrugTargetPair"].eq(1)]["pairId"].astype(str))
    for name, rel_path in paths.items():
        path = root / rel_path
        if not path.exists():
            rows.append({"shortlist": name, "exists": 0})
            continue
        short = pd.read_csv(path).fillna("")
        known = int(short["pairId"].astype(str).isin(known_pairs).sum()) if "pairId" in short else 0
        rows.append(
            {
                "shortlist": name,
                "exists": 1,
                "rows": int(len(short)),
                "knownDrugTargetRows": known,
                "knownDrugTargetPct": round(pct(known, len(short)), 4),
                "uniqueDrugs": int(short["drug"].nunique()) if "drug" in short else "",
                "uniqueTargets": int(short["target"].nunique()) if "target" in short else "",
                "tierCounts": dict(Counter(short["finalPriorityTier"])) if "finalPriorityTier" in short else {},
                "reviewTrackCounts": dict(Counter(short["reviewTrack"])) if "reviewTrack" in short else {},
            }
        )
    return rows


def top_known_hits(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    cols = [
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "directionLabelZhFinal",
        "pairId",
        "drug",
        "target",
        "protein",
        "knownTargetRank",
        "knownTargetRecord",
        "knownTargetName",
        "finalPriorityScore",
        "finalPriorityTier",
        "reviewTrack",
        "noveltyLabelZh",
        "admetTier",
        "kgEvidenceScore",
        "poseAuditStatus",
        "poseAuditReason",
    ]
    hits = df[df["knownDrugTargetPair"].eq(1)].sort_values("finalPriorityScore", ascending=False).head(limit)
    return hits[[col for col in cols if col in hits.columns]].to_dict("records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final candidate priority ranking against known drug-target positives.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / args.final_table).fillna("")
    df["knownDrugTargetPair"] = pd.to_numeric(df["knownDrugTargetPair"], errors="coerce").fillna(0).astype(int)
    df["finalPriorityScore"] = pd.to_numeric(df["finalPriorityScore"], errors="coerce").fillna(0)

    cutoff = cutoff_rows(df, "finalPriorityScore", "knownDrugTargetPair", "all", "all")
    for direction, group in df.groupby("direction"):
        cutoff.extend(cutoff_rows(group, "finalPriorityScore", "knownDrugTargetPair", "direction", direction))

    groups = build_group_metrics(df)
    shortlists = shortlist_metrics(
        df,
        root,
        {
            "top20_per_direction": "outputs/sota_validation/final_prioritization/final_candidate_priority_top20_per_direction.csv",
            "novel_top20_per_direction": "outputs/sota_validation/final_prioritization/final_candidate_priority_novel_top20_per_direction.csv",
            "known_controls": "outputs/sota_validation/final_prioritization/final_candidate_priority_known_controls.csv",
            "caution_cases": "outputs/sota_validation/final_prioritization/final_candidate_priority_caution_cases.csv",
            "diverse_expert_shortlist": "outputs/sota_validation/final_prioritization/final_candidate_diverse_expert_shortlist.csv",
        },
    )

    overall = next(row for row in groups if row["groupType"] == "all")
    cutoff_by_value = {row["cutoff"]: row for row in cutoff if row["groupType"] == "all"}
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "knownDrugTargetRows": int(df["knownDrugTargetPair"].sum()),
        "knownDrugTargetRatePct": round(pct(int(df["knownDrugTargetPair"].sum()), len(df)), 4),
        "auroc": overall["auroc"],
        "averagePrecision": overall["averagePrecision"],
        "recallAt100Pct": cutoff_by_value.get(100, {}).get("recallPct"),
        "precisionAt100Pct": cutoff_by_value.get(100, {}).get("precisionPct"),
        "enrichmentAt100": cutoff_by_value.get(100, {}).get("enrichmentVsRandom"),
        "recallAt500Pct": cutoff_by_value.get(500, {}).get("recallPct"),
        "precisionAt500Pct": cutoff_by_value.get(500, {}).get("precisionPct"),
        "enrichmentAt500": cutoff_by_value.get(500, {}).get("enrichmentVsRandom"),
        "shortlistKnownTargetRows": {row["shortlist"]: row.get("knownDrugTargetRows") for row in shortlists},
        "methodNote": "Final-priority validation treats benchmarked known drug-target pairs inside the final 3,921 candidate rows as positives. This tests triage ranking quality, not full-matrix discovery performance.",
        "inputs": {"finalTable": args.final_table},
        "outputs": {
            "cutoffMetrics": str((out_dir / "final_priority_known_target_enrichment_by_cutoff.csv").resolve()),
            "groupMetrics": str((out_dir / "final_priority_known_target_group_metrics.csv").resolve()),
            "shortlistMetrics": str((out_dir / "final_priority_shortlist_known_target_audit.csv").resolve()),
            "topKnownHits": str((out_dir / "final_priority_top_known_target_hits.csv").resolve()),
            "summary": str((out_dir / "final_priority_validation_summary.json").resolve()),
        },
    }

    write_csv(out_dir / "final_priority_known_target_enrichment_by_cutoff.csv", cutoff)
    write_csv(out_dir / "final_priority_known_target_group_metrics.csv", groups)
    write_csv(out_dir / "final_priority_shortlist_known_target_audit.csv", shortlists)
    write_csv(out_dir / "final_priority_top_known_target_hits.csv", top_known_hits(df, 100))
    write_json(out_dir / "final_priority_validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CUTOFFS = [10, 20, 50, 100, 200, 300, 500, 1000, 2000]


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def known_series(df: pd.DataFrame) -> pd.Series:
    if "knownDrugTargetPair" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["knownDrugTargetPair"].map(truthy)


def novel_series(df: pd.DataFrame) -> pd.Series:
    novelty = df.get("noveltyGroup", pd.Series("", index=df.index)).astype(str)
    strict = df.get("strictNovelPairFlag", pd.Series(False, index=df.index)).map(truthy)
    return strict | novelty.eq("novel_pair_or_new_target")


def interpretable_series(df: pd.DataFrame) -> pd.Series:
    return df.get("poseInterpretabilityTier", pd.Series("", index=df.index)).astype(str).str.startswith(("A_", "B_"))


def score_metrics(df: pd.DataFrame) -> dict[str, Any]:
    known = known_series(df).astype(int)
    score = pd.to_numeric(df["validationScore"], errors="coerce").fillna(0.0)
    if known.nunique() < 2:
        auroc = None
        ap = None
    else:
        auroc = float(roc_auc_score(known, score))
        ap = float(average_precision_score(known, score))
    return {
        "candidateRows": int(len(df)),
        "knownDrugTargetRows": int(known.sum()),
        "knownDrugTargetRatePct": round(pct(int(known.sum()), len(df)), 4),
        "validationAuroc": round(auroc, 6) if auroc is not None else None,
        "validationAveragePrecision": round(ap, 6) if ap is not None else None,
    }


def cutoff_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = df.sort_values(["validationRankGlobal", "pairId"], ascending=[True, True]).reset_index(drop=True)
    known = known_series(ordered)
    total = len(ordered)
    positives = int(known.sum())
    rows = []
    for cutoff in CUTOFFS + [total]:
        n = min(cutoff, total)
        top = ordered.head(n)
        hits = int(known.head(n).sum())
        expected = positives * n / total if total else 0.0
        rows.append(
            {
                "groupType": "all",
                "groupValue": "all",
                "cutoff": n,
                "rows": total,
                "positives": positives,
                "hits": hits,
                "precisionPct": round(pct(hits, n), 4),
                "recallPct": round(pct(hits, positives), 4),
                "randomExpectedHits": round(expected, 6),
                "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                "novelRows": int(novel_series(top).sum()),
                "interpretableRows": int(interpretable_series(top).sum()),
            }
        )
    for direction, group in ordered.groupby("direction", sort=True):
        group = group.sort_values(["validationRankWithinDirection", "pairId"], ascending=[True, True]).reset_index(drop=True)
        group_known = known_series(group)
        group_total = len(group)
        group_pos = int(group_known.sum())
        for cutoff in [10, 20, 50, 100, min(200, group_total), group_total]:
            n = min(cutoff, group_total)
            if n <= 0:
                continue
            hits = int(group_known.head(n).sum())
            expected = group_pos * n / group_total if group_total else 0.0
            rows.append(
                {
                    "groupType": "direction",
                    "groupValue": direction,
                    "cutoff": n,
                    "rows": group_total,
                    "positives": group_pos,
                    "hits": hits,
                    "precisionPct": round(pct(hits, n), 4),
                    "recallPct": round(pct(hits, group_pos), 4),
                    "randomExpectedHits": round(expected, 6),
                    "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                    "novelRows": int(novel_series(group.head(n)).sum()),
                    "interpretableRows": int(interpretable_series(group.head(n)).sum()),
                }
            )
    return rows


def group_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_type, column in [
        ("validationTier", "validationTier"),
        ("validationGate", "validationGate"),
        ("noveltyGroup", "noveltyGroup"),
        ("assayModality", "assayModality"),
        ("direction", "direction"),
        ("poseInterpretabilityTier", "poseInterpretabilityTier"),
        ("targetDruggabilityTier", "targetDruggabilityTier"),
    ]:
        if column not in df.columns:
            continue
        for value, group in df.groupby(column, dropna=False, sort=True):
            known = known_series(group)
            novel = novel_series(group)
            interpretable = interpretable_series(group)
            rows.append(
                {
                    "groupType": group_type,
                    "groupValue": str(value),
                    "rows": len(group),
                    "knownRows": int(known.sum()),
                    "knownRatePct": round(pct(int(known.sum()), len(group)), 4),
                    "novelRows": int(novel.sum()),
                    "novelRatePct": round(pct(int(novel.sum()), len(group)), 4),
                    "interpretableRows": int(interpretable.sum()),
                    "interpretablePct": round(pct(int(interpretable.sum()), len(group)), 4),
                    "medianValidationScore": round(float(pd.to_numeric(group["validationScore"], errors="coerce").median()), 4),
                }
            )
    return rows


def queue_metrics(root: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    queue_paths = {
        "full_panel": root / "outputs/sota_validation/experimental_validation/experimental_validation_panel.csv",
        "balanced_shortlist": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_balanced_shortlist.csv",
        "novel_shortlist": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_novel_shortlist.csv",
        "positive_controls": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_positive_controls.csv",
        "risk_review": root / "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_risk_review.csv",
    }
    rows: list[dict[str, Any]] = []
    for queue, path in queue_paths.items():
        if not path.exists():
            continue
        group = pd.read_csv(path).fillna("")
        known = known_series(group)
        novel = novel_series(group)
        interpretable = interpretable_series(group)
        rows.append(
            {
                "queue": queue,
                "rows": len(group),
                "knownRows": int(known.sum()),
                "knownRatePct": round(pct(int(known.sum()), len(group)), 4),
                "novelRows": int(novel.sum()),
                "novelRatePct": round(pct(int(novel.sum()), len(group)), 4),
                "interpretableRows": int(interpretable.sum()),
                "interpretablePct": round(pct(int(interpretable.sum()), len(group)), 4),
                "uniqueDrugs": group["drugId"].nunique() if "drugId" in group else group["drug"].nunique(),
                "uniqueTargets": group["protein"].nunique() if "protein" in group else group["target"].nunique(),
                "uniqueScaffolds": group["murckoScaffold"].nunique() if "murckoScaffold" in group else None,
                "medianValidationScore": round(float(pd.to_numeric(group["validationScore"], errors="coerce").median()), 4),
                "maxDrugPct": concentration_pct(group, "drugId" if "drugId" in group else "drug"),
                "maxTargetPct": concentration_pct(group, "protein" if "protein" in group else "target"),
                "maxScaffoldPct": concentration_pct(group, "murckoScaffold") if "murckoScaffold" in group else None,
            }
        )
    return rows


def concentration_pct(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    counts = df[column].astype(str).value_counts()
    return round(pct(int(counts.iloc[0]), len(df)), 4) if not counts.empty else 0.0


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    panel_path = root / args.panel
    df = pd.read_csv(panel_path).fillna("")
    metrics = score_metrics(df)
    cutoffs = cutoff_metrics(df)
    groups = group_metrics(df)
    queues = queue_metrics(root, df)

    cutoff100 = next((row for row in cutoffs if row["groupType"] == "all" and row["cutoff"] == 100), {})
    cutoff300 = next((row for row in cutoffs if row["groupType"] == "all" and row["cutoff"] == 300), {})
    balanced = next((row for row in queues if row["queue"] == "balanced_shortlist"), {})
    novel = next((row for row in queues if row["queue"] == "novel_shortlist"), {})
    positive = next((row for row in queues if row["queue"] == "positive_controls"), {})

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **metrics,
        "top100KnownRows": cutoff100.get("hits"),
        "top100KnownPrecisionPct": cutoff100.get("precisionPct"),
        "top100KnownRecallPct": cutoff100.get("recallPct"),
        "top100KnownEnrichment": cutoff100.get("enrichmentVsRandom"),
        "top300KnownRows": cutoff300.get("hits"),
        "top300KnownPrecisionPct": cutoff300.get("precisionPct"),
        "top300KnownRecallPct": cutoff300.get("recallPct"),
        "top300KnownEnrichment": cutoff300.get("enrichmentVsRandom"),
        "balancedKnownRows": balanced.get("knownRows"),
        "balancedNovelRows": balanced.get("novelRows"),
        "balancedInterpretableRows": balanced.get("interpretableRows"),
        "balancedUniqueDrugs": balanced.get("uniqueDrugs"),
        "balancedUniqueTargets": balanced.get("uniqueTargets"),
        "balancedUniqueScaffolds": balanced.get("uniqueScaffolds"),
        "balancedMaxDrugPct": balanced.get("maxDrugPct"),
        "balancedMaxTargetPct": balanced.get("maxTargetPct"),
        "balancedMaxScaffoldPct": balanced.get("maxScaffoldPct"),
        "novelShortlistRows": novel.get("rows"),
        "novelShortlistInterpretableRows": novel.get("interpretableRows"),
        "positiveControlRows": positive.get("rows"),
        "positiveControlKnownRows": positive.get("knownRows"),
        "methodNote": (
            "This benchmark audits the experimental validation panel against known drug-target positives "
            "available inside the final candidate table. It tests triage calibration and control recovery, "
            "not prospective biological truth for novel candidates."
        ),
    }

    out_dir = root / args.out_dir
    final_dir = root / "outputs/sota_validation/final_prioritization"
    write_csv(out_dir / "validation_panel_known_target_cutoff_metrics.csv", cutoffs)
    write_csv(out_dir / "validation_panel_group_calibration.csv", groups)
    write_csv(out_dir / "validation_panel_queue_metrics.csv", queues)
    write_json(out_dir / "validation_panel_benchmark_summary.json", summary)
    write_csv(final_dir / "final_priority_validation_panel_known_target_cutoff_metrics.csv", cutoffs)
    write_csv(final_dir / "final_priority_validation_panel_group_calibration.csv", groups)
    write_csv(final_dir / "final_priority_validation_panel_queue_metrics.csv", queues)
    write_json(final_dir / "final_priority_validation_panel_benchmark_summary.json", summary)
    (final_dir / "FINAL_PRIORITY_VALIDATION_PANEL_BENCHMARK.md").write_text(
        markdown(summary, queues, groups),
        encoding="utf-8",
    )
    return {"summary": summary}


def markdown(summary: dict[str, Any], queues: list[dict[str, Any]], groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Validation Panel Benchmark and Calibration",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["methodNote"],
        "",
        "## Headline Metrics",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Known drug-target positives: {summary['knownDrugTargetRows']} ({summary['knownDrugTargetRatePct']:.2f}%)",
        f"- Validation-score AUROC/AP: {summary['validationAuroc']} / {summary['validationAveragePrecision']}",
        f"- Top100 known rows: {summary['top100KnownRows']}; precision {summary['top100KnownPrecisionPct']}%; recall {summary['top100KnownRecallPct']}%; enrichment {summary['top100KnownEnrichment']}x",
        f"- Top300 known rows: {summary['top300KnownRows']}; precision {summary['top300KnownPrecisionPct']}%; recall {summary['top300KnownRecallPct']}%; enrichment {summary['top300KnownEnrichment']}x",
        f"- Balanced shortlist: known {summary['balancedKnownRows']}, novel {summary['balancedNovelRows']}, interpretable {summary['balancedInterpretableRows']}, "
        f"unique drugs/targets/scaffolds {summary['balancedUniqueDrugs']}/{summary['balancedUniqueTargets']}/{summary['balancedUniqueScaffolds']}",
        f"- Novel shortlist: rows {summary['novelShortlistRows']}; interpretable {summary['novelShortlistInterpretableRows']}",
        f"- Positive controls: rows {summary['positiveControlRows']}; known target rows {summary['positiveControlKnownRows']}",
        "",
        "## Queue Metrics",
        "",
    ]
    for row in queues:
        lines.append(
            f"- {row['queue']}: rows {row['rows']}; known {row['knownRows']} ({row['knownRatePct']}%); "
            f"novel {row['novelRows']} ({row['novelRatePct']}%); interpretable {row['interpretableRows']} ({row['interpretablePct']}%)."
        )
    lines.extend(["", "## Tier Calibration", ""])
    for row in groups:
        if row["groupType"] != "validationTier":
            continue
        lines.append(
            f"- {row['groupValue']}: rows {row['rows']}; known rate {row['knownRatePct']}%; "
            f"novel rate {row['novelRatePct']}%; interpretable {row['interpretablePct']}%."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark and calibrate experimental validation panel.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--panel", default="outputs/sota_validation/experimental_validation/experimental_validation_panel.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/validation_panel_benchmark")
    args = parser.parse_args()
    result = build(Path(args.root).resolve(), args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

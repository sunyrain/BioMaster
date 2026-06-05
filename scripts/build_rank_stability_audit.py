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
from scipy.stats import kendalltau, spearmanr


CUTOFFS = [20, 50, 100, 200, 500, 1000]


RANKINGS = {
    "final_priority": {
        "path": "outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv",
        "score": "finalPriorityScore",
        "rank": "finalRankGlobal",
    },
    "structure_adjusted": {
        "path": "outputs/sota_validation/final_prioritization/final_priority_structure_adjusted_table.csv",
        "score": "structureAdjustedPriorityScore",
        "rank": "structureAdjustedRankGlobal",
    },
    "target_adjusted": {
        "path": "outputs/sota_validation/final_prioritization/final_priority_target_druggability_augmented_table.csv",
        "score": "targetAdjustedPriorityScore",
        "rank": "targetAdjustedRankGlobal",
    },
    "sota_ready": {
        "path": "outputs/sota_validation/final_prioritization/final_priority_sota_ready_matrix.csv",
        "score": "sotaReadyScore",
        "rank": "sotaReadyRankGlobal",
    },
}


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_ranking(root: Path, name: str, spec: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(root / spec["path"]).fillna("")
    if spec["rank"] not in df:
        df = df.sort_values(spec["score"], ascending=False).reset_index(drop=True)
        df[spec["rank"]] = np.arange(1, len(df) + 1)
    keep = [
        "direction",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "reviewTrack",
        "noveltyClass",
        "finalPriorityTier",
        "sotaReadyTier",
        "sotaReadyAction",
        "structureConfidenceTier",
        "targetDruggabilityTier",
        "murckoScaffold",
        spec["score"],
        spec["rank"],
    ]
    existing = [col for col in keep if col in df.columns]
    out = df[existing].copy()
    out["candidateKey"] = out["direction"].astype(str) + "||" + out["pairId"].astype(str)
    out[f"{name}Score"] = pd.to_numeric(out[spec["score"]], errors="coerce").fillna(0.0)
    out[f"{name}Rank"] = pd.to_numeric(out[spec["rank"]], errors="coerce").fillna(len(out) + 1).astype(int)
    return out


def build_combined(root: Path) -> pd.DataFrame:
    base = read_ranking(root, "final_priority", RANKINGS["final_priority"])
    metadata_cols = [
        "candidateKey",
        "direction",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "reviewTrack",
        "noveltyClass",
        "finalPriorityTier",
    ]
    combined = base[[col for col in metadata_cols if col in base.columns] + ["final_priorityScore", "final_priorityRank"]].copy()
    for name, spec in RANKINGS.items():
        if name == "final_priority":
            continue
        item = read_ranking(root, name, spec)
        keep = ["candidateKey", f"{name}Score", f"{name}Rank"]
        for col in ["sotaReadyTier", "sotaReadyAction", "structureConfidenceTier", "targetDruggabilityTier", "murckoScaffold"]:
            if col in item.columns and col not in combined.columns:
                keep.append(col)
        combined = combined.merge(item[keep].drop_duplicates("candidateKey"), on="candidateKey", how="left")
    for name in RANKINGS:
        combined[f"{name}Rank"] = pd.to_numeric(combined[f"{name}Rank"], errors="coerce").fillna(len(combined) + 1).astype(int)
        combined[f"{name}Score"] = pd.to_numeric(combined[f"{name}Score"], errors="coerce").fillna(0.0)
    rank_cols = [f"{name}Rank" for name in RANKINGS]
    combined["meanRankAcrossMethods"] = combined[rank_cols].mean(axis=1).round(4)
    combined["medianRankAcrossMethods"] = combined[rank_cols].median(axis=1).round(4)
    combined["bestRankAcrossMethods"] = combined[rank_cols].min(axis=1)
    combined["worstRankAcrossMethods"] = combined[rank_cols].max(axis=1)
    combined["rankRangeAcrossMethods"] = combined["worstRankAcrossMethods"] - combined["bestRankAcrossMethods"]
    combined["top100MethodCount"] = sum((combined[f"{name}Rank"] <= 100).astype(int) for name in RANKINGS)
    combined["top500MethodCount"] = sum((combined[f"{name}Rank"] <= 500).astype(int) for name in RANKINGS)
    combined["consensusTop100Flag"] = combined["top100MethodCount"].ge(3).astype(int)
    combined["consensusTop500Flag"] = combined["top500MethodCount"].ge(3).astype(int)
    combined["sotaVsFinalRankDelta"] = combined["sota_readyRank"] - combined["final_priorityRank"]
    combined = combined.sort_values(["meanRankAcrossMethods", "bestRankAcrossMethods"]).reset_index(drop=True)
    combined["consensusRank"] = np.arange(1, len(combined) + 1)
    return combined


def top_set(df: pd.DataFrame, method: str, cutoff: int) -> set[str]:
    return set(df[df[f"{method}Rank"].le(cutoff)]["candidateKey"].astype(str))


def overlap_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = list(RANKINGS)
    for cutoff in CUTOFFS:
        for i, left in enumerate(methods):
            left_set = top_set(df, left, cutoff)
            for right in methods[i + 1 :]:
                right_set = top_set(df, right, cutoff)
                inter = left_set & right_set
                union = left_set | right_set
                rows.append(
                    {
                        "cutoff": cutoff,
                        "leftRanking": left,
                        "rightRanking": right,
                        "leftRows": len(left_set),
                        "rightRows": len(right_set),
                        "overlapRows": len(inter),
                        "overlapPctOfCutoff": round(pct(len(inter), cutoff), 4),
                        "jaccard": round(len(inter) / len(union), 6) if union else None,
                    }
                )
        all_sets = [top_set(df, method, cutoff) for method in methods]
        intersection = set.intersection(*all_sets)
        union = set.union(*all_sets)
        rows.append(
            {
                "cutoff": cutoff,
                "leftRanking": "all_methods",
                "rightRanking": "all_methods",
                "leftRows": cutoff,
                "rightRows": cutoff,
                "overlapRows": len(intersection),
                "overlapPctOfCutoff": round(pct(len(intersection), cutoff), 4),
                "jaccard": round(len(intersection) / len(union), 6) if union else None,
            }
        )
    return rows


def correlation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = list(RANKINGS)
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            left_rank = df[f"{left}Rank"].astype(float)
            right_rank = df[f"{right}Rank"].astype(float)
            spearman = spearmanr(left_rank, right_rank).statistic
            kendall = kendalltau(left_rank, right_rank).statistic
            rows.append(
                {
                    "leftRanking": left,
                    "rightRanking": right,
                    "spearmanRankCorrelation": round(float(spearman), 6) if not math.isnan(float(spearman)) else None,
                    "kendallTau": round(float(kendall), 6) if not math.isnan(float(kendall)) else None,
                    "medianAbsRankDelta": round(float((left_rank - right_rank).abs().median()), 4),
                    "meanAbsRankDelta": round(float((left_rank - right_rank).abs().mean()), 4),
                }
            )
    return rows


def topk_quality(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = pd.to_numeric(df.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).astype(int)
    total_positive = int(labels.sum())
    for method in list(RANKINGS) + ["consensus"]:
        rank_col = "consensusRank" if method == "consensus" else f"{method}Rank"
        ranked = df.sort_values(rank_col)
        for cutoff in CUTOFFS:
            if cutoff > len(ranked):
                continue
            top = ranked.head(cutoff)
            hits = int(pd.to_numeric(top.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).astype(int).sum())
            rows.append(
                {
                    "ranking": method,
                    "cutoff": cutoff,
                    "knownDrugTargetRows": hits,
                    "knownDrugTargetPct": round(pct(hits, cutoff), 4),
                    "knownDrugTargetRecallPct": round(pct(hits, total_positive), 4),
                    "novelRows": int(top["noveltyClass"].astype(str).isin({"disease_context_supported_new_pair", "model_priority_without_txgnn_kg_path"}).sum())
                    if "noveltyClass" in top
                    else "",
                    "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else "",
                    "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else "",
                    "consensusTop100Rows": int(top.get("consensusTop100Flag", pd.Series(dtype=int)).sum()),
                    "consensusTop500Rows": int(top.get("consensusTop500Flag", pd.Series(dtype=int)).sum()),
                }
            )
    return rows


def summary_payload(df: pd.DataFrame, overlaps: list[dict[str, Any]], correlations: list[dict[str, Any]], topk: list[dict[str, Any]]) -> dict[str, Any]:
    top100_all = next(row for row in overlaps if row["cutoff"] == 100 and row["leftRanking"] == "all_methods")
    top500_all = next(row for row in overlaps if row["cutoff"] == 500 and row["leftRanking"] == "all_methods")
    quality = {(row["ranking"], row["cutoff"]): row for row in topk}
    corr_lookup = {(row["leftRanking"], row["rightRanking"]): row for row in correlations}
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "rankingMethods": list(RANKINGS),
        "consensusTop100Rows3PlusMethods": int(df["consensusTop100Flag"].sum()),
        "consensusTop500Rows3PlusMethods": int(df["consensusTop500Flag"].sum()),
        "allMethodTop100IntersectionRows": top100_all["overlapRows"],
        "allMethodTop100IntersectionPct": top100_all["overlapPctOfCutoff"],
        "allMethodTop500IntersectionRows": top500_all["overlapRows"],
        "allMethodTop500IntersectionPct": top500_all["overlapPctOfCutoff"],
        "finalVsSotaReadyTop100OverlapRows": next(
            row
            for row in overlaps
            if row["cutoff"] == 100 and row["leftRanking"] == "final_priority" and row["rightRanking"] == "sota_ready"
        )["overlapRows"],
        "finalVsSotaReadyTop100Jaccard": next(
            row
            for row in overlaps
            if row["cutoff"] == 100 and row["leftRanking"] == "final_priority" and row["rightRanking"] == "sota_ready"
        )["jaccard"],
        "finalVsSotaReadySpearman": corr_lookup.get(("final_priority", "sota_ready"), {}).get("spearmanRankCorrelation"),
        "finalTop100KnownRows": quality.get(("final_priority", 100), {}).get("knownDrugTargetRows"),
        "sotaReadyTop100KnownRows": quality.get(("sota_ready", 100), {}).get("knownDrugTargetRows"),
        "consensusTop100KnownRows": quality.get(("consensus", 100), {}).get("knownDrugTargetRows"),
        "consensusTop100NovelRows": quality.get(("consensus", 100), {}).get("novelRows"),
        "consensusTop300Rows": int(min(300, len(df))),
        "largeRankDeltaRowsAbs1000": int(df["sotaVsFinalRankDelta"].abs().ge(1000).sum()),
        "methodNote": "Rank stability compares final-priority, structure-adjusted, target-adjusted, SOTA-ready, and consensus rankings on the same direction+pairId candidate keys. SOTA-ready intentionally trades some known-target recall for stricter structural, target, chemotype, and risk gates.",
    }


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final Priority Rank Stability Audit",
            "",
            f"Generated: {summary.get('created_utc')}",
            "",
            "## Summary",
            "",
            f"- Candidate rows: {summary.get('candidateRows')}",
            f"- Ranking methods: {summary.get('rankingMethods')}",
            f"- Consensus Top100 rows in 3+ methods: {summary.get('consensusTop100Rows3PlusMethods')}",
            f"- All-method Top100 intersection: {summary.get('allMethodTop100IntersectionRows')} ({summary.get('allMethodTop100IntersectionPct')}%)",
            f"- Final vs SOTA-ready Top100 overlap / Jaccard: {summary.get('finalVsSotaReadyTop100OverlapRows')} / {summary.get('finalVsSotaReadyTop100Jaccard')}",
            f"- Final vs SOTA-ready Spearman rank correlation: {summary.get('finalVsSotaReadySpearman')}",
            f"- Known-target Top100 rows final / SOTA-ready / consensus: {summary.get('finalTop100KnownRows')} / {summary.get('sotaReadyTop100KnownRows')} / {summary.get('consensusTop100KnownRows')}",
            f"- Consensus Top100 novel rows: {summary.get('consensusTop100NovelRows')}",
            f"- Large final-vs-SOTA-ready rank delta rows (absolute >= 1000): {summary.get('largeRankDeltaRowsAbs1000')}",
            "",
            "## Outputs",
            "",
            "- Stability audit: `outputs/sota_validation/final_prioritization/final_priority_rank_stability_audit.csv`",
            "- TopK overlap: `outputs/sota_validation/final_prioritization/final_priority_rank_stability_topk_overlap.csv`",
            "- Rank correlations: `outputs/sota_validation/final_prioritization/final_priority_rank_stability_correlations.csv`",
            "- Consensus shortlist: `outputs/sota_validation/final_prioritization/final_priority_rank_stability_consensus_shortlist.csv`",
            "- Large rank-delta review: `outputs/sota_validation/final_prioritization/final_priority_rank_stability_large_delta_review.csv`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ranking stability across final, adjusted, and SOTA-ready rankings.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = build_combined(root)
    overlaps = overlap_rows(combined)
    correlations = correlation_rows(combined)
    quality = topk_quality(combined)
    summary = summary_payload(combined, overlaps, correlations, quality)

    combined.to_csv(out_dir / "final_priority_rank_stability_audit.csv", index=False)
    write_csv(out_dir / "final_priority_rank_stability_topk_overlap.csv", overlaps)
    write_csv(out_dir / "final_priority_rank_stability_correlations.csv", correlations)
    write_csv(out_dir / "final_priority_rank_stability_topk_quality.csv", quality)
    consensus = combined.sort_values("consensusRank").head(300)
    consensus.to_csv(out_dir / "final_priority_rank_stability_consensus_shortlist.csv", index=False)
    delta_review = combined.sort_values("sotaVsFinalRankDelta", key=lambda item: item.abs(), ascending=False).head(500)
    delta_review.to_csv(out_dir / "final_priority_rank_stability_large_delta_review.csv", index=False)
    write_json(out_dir / "final_priority_rank_stability_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_RANK_STABILITY_AUDIT.md").write_text(markdown(summary), encoding="utf-8")

    print(json.dumps({"summary": summary, "out_dir": args.out_dir}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

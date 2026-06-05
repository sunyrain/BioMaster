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
from scipy.stats import hypergeom


CUTOFFS = [10, 20, 50, 100, 200, 500, 1000, 2000, 3921]
DIRECTION_CUTOFFS = [10, 20, 50, 100, 200, 500]


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def as_number(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


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


def empirical_p_ge(samples: np.ndarray, observed: int) -> float:
    return float((np.sum(samples >= observed) + 1) / (len(samples) + 1)) if len(samples) else 1.0


def z_score(observed: int | float, mean: float, sd: float) -> float | None:
    if sd <= 0 or not math.isfinite(sd):
        return None
    return float((observed - mean) / sd)


def sample_global(labels: np.ndarray, cutoff: int, iterations: int, rng: np.random.Generator) -> np.ndarray:
    n = len(labels)
    samples = np.empty(iterations, dtype=int)
    for i in range(iterations):
        idx = rng.choice(n, size=cutoff, replace=False)
        samples[i] = int(labels[idx].sum())
    return samples


def sample_stratified(
    df: pd.DataFrame,
    top_counts: dict[str, int],
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_direction = {
        direction: group["knownDrugTargetPair"].to_numpy(dtype=int)
        for direction, group in df.groupby("direction", dropna=False)
    }
    samples = np.zeros(iterations, dtype=int)
    for direction, k in top_counts.items():
        labels = by_direction.get(direction)
        if labels is None or k <= 0:
            continue
        n = len(labels)
        if k >= n:
            samples += int(labels.sum())
            continue
        for i in range(iterations):
            idx = rng.choice(n, size=k, replace=False)
            samples[i] += int(labels[idx].sum())
    return samples


def distribution_stats(samples: np.ndarray, observed: int) -> dict[str, Any]:
    if len(samples) == 0:
        return {
            "mean": None,
            "sd": None,
            "pValueGe": None,
            "zScore": None,
            "q025": None,
            "q500": None,
            "q975": None,
        }
    mean = float(samples.mean())
    sd = float(samples.std(ddof=0))
    return {
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "pValueGe": round(empirical_p_ge(samples, observed), 8),
        "zScore": round(z_score(observed, mean, sd), 6) if z_score(observed, mean, sd) is not None else None,
        "q025": round(float(np.quantile(samples, 0.025)), 6),
        "q500": round(float(np.quantile(samples, 0.5)), 6),
        "q975": round(float(np.quantile(samples, 0.975)), 6),
    }


def topk_rows(df: pd.DataFrame, iterations: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ranked = df.sort_values("finalPriorityScore", ascending=False).reset_index(drop=True)
    labels = ranked["knownDrugTargetPair"].to_numpy(dtype=int)
    total = len(ranked)
    positives = int(labels.sum())
    base_rate = positives / total if total else 0.0
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        if cutoff > total:
            continue
        top = ranked.head(cutoff)
        observed = int(top["knownDrugTargetPair"].sum())
        expected = cutoff * base_rate
        direction_counts = Counter(top["direction"].astype(str))

        global_samples = sample_global(labels, cutoff, iterations, rng)
        stratified_samples = sample_stratified(ranked, dict(direction_counts), iterations, rng)
        global_stats = distribution_stats(global_samples, observed)
        stratified_stats = distribution_stats(stratified_samples, observed)
        exact_p = float(hypergeom.sf(observed - 1, total, positives, cutoff)) if observed > 0 else 1.0

        rows.append(
            {
                "cutoff": cutoff,
                "rows": total,
                "positives": positives,
                "observedHits": observed,
                "observedPrecisionPct": round(pct(observed, cutoff), 4),
                "observedRecallPct": round(pct(observed, positives), 4),
                "globalExpectedHits": round(expected, 6),
                "globalEnrichmentVsRandom": round(observed / expected, 6) if expected else "",
                "globalHypergeomPGe": f"{exact_p:.8g}",
                "globalPermutationMean": global_stats["mean"],
                "globalPermutationSd": global_stats["sd"],
                "globalPermutationPGe": global_stats["pValueGe"],
                "globalPermutationZ": global_stats["zScore"],
                "globalQ025": global_stats["q025"],
                "globalQ500": global_stats["q500"],
                "globalQ975": global_stats["q975"],
                "stratifiedExpectedHits": stratified_stats["mean"],
                "stratifiedEnrichmentVsRandom": round(observed / stratified_stats["mean"], 6) if stratified_stats["mean"] else "",
                "stratifiedPermutationSd": stratified_stats["sd"],
                "stratifiedPermutationPGe": stratified_stats["pValueGe"],
                "stratifiedPermutationZ": stratified_stats["zScore"],
                "stratifiedQ025": stratified_stats["q025"],
                "stratifiedQ500": stratified_stats["q500"],
                "stratifiedQ975": stratified_stats["q975"],
                "directionComposition": dict(direction_counts),
            }
        )
    return rows


def direction_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, group in df.groupby("direction", dropna=False):
        ranked = group.sort_values("finalPriorityScore", ascending=False)
        total = len(ranked)
        positives = int(as_number(ranked["knownDrugTargetPair"]).sum())
        base_rate = positives / total if total else 0
        for cutoff in DIRECTION_CUTOFFS:
            if cutoff > total:
                continue
            top = ranked.head(cutoff)
            hits = int(as_number(top["knownDrugTargetPair"]).sum())
            expected = cutoff * base_rate
            exact_p = float(hypergeom.sf(hits - 1, total, positives, cutoff)) if hits > 0 else 1.0
            rows.append(
                {
                    "direction": direction,
                    "cutoff": cutoff,
                    "rows": total,
                    "positives": positives,
                    "observedHits": hits,
                    "observedPrecisionPct": round(pct(hits, cutoff), 4),
                    "observedRecallPct": round(pct(hits, positives), 4),
                    "expectedHits": round(expected, 6),
                    "enrichmentVsRandom": round(hits / expected, 6) if expected else "",
                    "hypergeomPGe": f"{exact_p:.8g}",
                }
            )
    return rows


def build_markdown(summary: dict[str, Any]) -> str:
    top100 = summary["topK"]["100"]
    top500 = summary["topK"]["500"]
    lines = [
        "# TopK Significance Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit tests whether known drug-target recovery in the final ranking exceeds random and disease-direction-stratified random baselines.",
        "",
        "## Headline",
        "",
        f"- Top100: {top100['observedHits']} known hits; global enrichment {top100['globalEnrichmentVsRandom']:.2f}x; "
        f"global hypergeometric p={top100['globalHypergeomPGe']}; stratified p={top100['stratifiedPermutationPGe']}.",
        f"- Top500: {top500['observedHits']} known hits; global enrichment {top500['globalEnrichmentVsRandom']:.2f}x; "
        f"global hypergeometric p={top500['globalHypergeomPGe']}; stratified p={top500['stratifiedPermutationPGe']}.",
        "",
        "## Interpretation",
        "",
        "The stratified baseline keeps the final TopK disease-direction composition fixed. This separates ranking quality from artifacts caused by one disease direction containing more known positives.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TopK known-target significance audit for final priority ranking.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    df = pd.read_csv(root / args.final_table).fillna("")
    df["knownDrugTargetPair"] = as_number(df["knownDrugTargetPair"]).astype(int)
    df["finalPriorityScore"] = as_number(df["finalPriorityScore"])

    topk = topk_rows(df, args.iterations, args.seed)
    direction = direction_rows(df)
    lookup = {str(row["cutoff"]): row for row in topk}
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": args.iterations,
        "candidateRows": int(len(df)),
        "knownDrugTargetRows": int(df["knownDrugTargetPair"].sum()),
        "knownDrugTargetRatePct": round(pct(int(df["knownDrugTargetPair"].sum()), len(df)), 4),
        "topK": lookup,
        "methodNote": (
            "Global baseline samples TopK rows uniformly from all final candidates. Stratified baseline preserves the disease-direction composition "
            "of each observed TopK and samples within each direction. P-values are right-tail probabilities for observing at least the current number of known hits."
        ),
        "outputs": {
            "topkSignificance": str((out_dir / "final_priority_topk_significance.csv").resolve()),
            "directionSignificance": str((out_dir / "final_priority_direction_topk_significance.csv").resolve()),
            "summary": str((out_dir / "final_priority_topk_significance_summary.json").resolve()),
            "markdown": str((out_dir / "FINAL_PRIORITY_TOPK_SIGNIFICANCE_AUDIT.md").resolve()),
        },
    }

    write_csv(out_dir / "final_priority_topk_significance.csv", topk)
    write_csv(out_dir / "final_priority_direction_topk_significance.csv", direction)
    write_json(out_dir / "final_priority_topk_significance_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_TOPK_SIGNIFICANCE_AUDIT.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

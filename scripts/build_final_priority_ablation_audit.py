from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


CUTOFFS = [10, 20, 50, 100, 200, 500, 1000, 2000, 3921]
DIRECTION_CUTOFFS = [10, 20, 50, 100, 200, 500]
TOPK_OVERLAP = [20, 50, 100, 500, 1000]
BOOTSTRAP_VARIANTS = [
    "final_priority",
    "raw_without_risk",
    "model_only",
    "disease_evidence_only",
    "kg_only",
    "admet_only",
    "pose_only",
    "without_model_component",
    "without_disease_component",
    "without_kg_component",
    "without_admet_component",
    "without_pose_component",
    "without_risk_penalty",
]

WEIGHTS = {
    "model": 0.24,
    "disease": 0.18,
    "credibility": 0.14,
    "kg": 0.16,
    "admet": 0.14,
    "pose": 0.10,
    "label_fit": 0.04,
}


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def as_number(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def bounded(series: pd.Series, lower: float = 0.0, upper: float = 100.0) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def minmax_score(series: pd.Series, default: float = 50.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(default, index=series.index, dtype=float)
    lo = numeric.min()
    hi = numeric.max()
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return pd.Series(default, index=series.index, dtype=float)
    return ((numeric - lo) / (hi - lo) * 100).fillna(default)


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


def safe_auc(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y = pd.Series(labels).astype(int)
    s = pd.Series(scores).astype(float)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, s))


def safe_ap(labels: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float | None:
    y = pd.Series(labels).astype(int)
    s = pd.Series(scores).astype(float)
    if int(y.sum()) == 0:
        return None
    return float(average_precision_score(y, s))


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    x = as_number(left)
    y = as_number(right)
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    corr = spearmanr(x, y, nan_policy="omit").correlation
    return float(corr) if corr is not None and math.isfinite(float(corr)) else None


def cutoff_metrics(
    df: pd.DataFrame,
    variant: str,
    score_col: str,
    group_type: str,
    group_value: str,
    cutoffs: list[int],
) -> list[dict[str, Any]]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    total = len(ranked)
    positives = int(ranked["knownDrugTargetPair"].sum())
    base_rate = positives / total if total else 0.0
    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        if cutoff > total:
            continue
        top = ranked.head(cutoff)
        hits = int(top["knownDrugTargetPair"].sum())
        expected = cutoff * base_rate
        rows.append(
            {
                "variant": variant,
                "groupType": group_type,
                "groupValue": group_value,
                "cutoff": cutoff,
                "rows": total,
                "positives": positives,
                "hits": hits,
                "precisionPct": round(pct(hits, cutoff), 4),
                "recallPct": round(pct(hits, positives), 4),
                "randomExpectedHits": round(expected, 4),
                "enrichmentVsRandom": round(hits / expected, 4) if expected else "",
            }
        )
    return rows


def rank_metrics(df: pd.DataFrame, variant: str, score_col: str, group_type: str = "all", group_value: str = "all") -> dict[str, Any]:
    labels = df["knownDrugTargetPair"].astype(int)
    scores = as_number(df[score_col])
    pos = scores[labels.eq(1)]
    neg = scores[labels.eq(0)]
    corr = safe_spearman(df["finalPriorityScore"], scores)
    return {
        "variant": variant,
        "groupType": group_type,
        "groupValue": group_value,
        "rows": int(len(df)),
        "positives": int(labels.sum()),
        "positiveRatePct": round(pct(int(labels.sum()), len(df)), 4),
        "auroc": safe_auc(labels, scores),
        "averagePrecision": safe_ap(labels, scores),
        "medianScorePositive": float(pos.median()) if len(pos) else None,
        "medianScoreNegative": float(neg.median()) if len(neg) else None,
        "spearmanVsFinalPriority": corr,
    }


def weighted_component_score(components: dict[str, pd.Series], include: set[str]) -> pd.Series:
    denom = sum(WEIGHTS[name] for name in include)
    if denom <= 0:
        return pd.Series(50.0, index=next(iter(components.values())).index)
    score = sum(WEIGHTS[name] * components[name] for name in include) / denom
    return score.astype(float)


def add_variant_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = df.copy()
    out["knownDrugTargetPair"] = as_number(out["knownDrugTargetPair"]).astype(int)
    out["directionLabelFitScore"] = np.where(as_number(out["directionLabelFit"]).eq(1), 100.0, 55.0)
    out["diffdockConfidenceMinmax"] = minmax_score(out["diffdock"], default=50.0)
    out["rankOnlyScore"] = 100 - minmax_score(out["rank"], default=50.0)

    components = {
        "model": as_number(out["modelComponent"]),
        "disease": as_number(out["diseaseEvidenceComponent"]),
        "credibility": as_number(out["credibilityScore"]),
        "kg": as_number(out["kgEvidenceScore"]),
        "admet": as_number(out["admetScore"]),
        "pose": as_number(out["structureGeometryScore"]),
        "label_fit": as_number(out["directionLabelFitScore"]),
    }
    risk = as_number(out["riskPenalty"])
    all_components = set(WEIGHTS)

    variant_notes: dict[str, str] = {
        "final_priority": "Published final integrated score after risk penalties.",
        "raw_without_risk": "Integrated weighted evidence score before risk penalties.",
        "model_only": "ConPLex direction and affinity components only.",
        "direction_score_only": "Disease-direction rank score only.",
        "affinity_score_only": "ConPLex affinity score only.",
        "disease_evidence_only": "Disease evidence component only: direction, Open Targets, and TxGNN.",
        "open_targets_only": "Open Targets target-disease evidence only.",
        "txgnn_only": "TxGNN drug-disease evidence only.",
        "kg_only": "TxGNN/DrugBank shallow KG path score only.",
        "admet_only": "Rule-based ADMET/repurposability score only.",
        "pose_only": "Pose geometry sanity score only.",
        "credibility_only": "Existing credibility score only.",
        "diffdock_confidence_only": "Raw DiffDock confidence after min-max normalization; higher confidence ranked first.",
        "rank_only": "Original within-direction candidate rank converted to a 0-100 descending score.",
        "without_model_component": "Integrated score with the model component removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_disease_component": "Integrated score with disease evidence removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_kg_component": "Integrated score with KG evidence removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_admet_component": "Integrated score with ADMET score removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_pose_component": "Integrated score with pose geometry removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_credibility_component": "Integrated score with credibility removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_label_fit_component": "Integrated score with direction-label fit removed and remaining evidence weights renormalized; original risk penalty is retained.",
        "without_risk_penalty": "Final integrated evidence score before subtracting safety/structure/label penalties.",
    }

    out["score_final_priority"] = as_number(out["finalPriorityScore"])
    out["score_raw_without_risk"] = as_number(out["rawIntegratedScore"])
    out["score_model_only"] = components["model"]
    out["score_direction_score_only"] = as_number(out["directionScore"]) * 100
    out["score_affinity_score_only"] = as_number(out["affinityScore"]) * 100
    out["score_disease_evidence_only"] = components["disease"]
    out["score_open_targets_only"] = as_number(out["openTargetsScore"]) * 100
    out["score_txgnn_only"] = as_number(out["integratedTxgnnScore"]) * 100
    out["score_kg_only"] = components["kg"]
    out["score_admet_only"] = components["admet"]
    out["score_pose_only"] = components["pose"]
    out["score_credibility_only"] = components["credibility"]
    out["score_diffdock_confidence_only"] = out["diffdockConfidenceMinmax"]
    out["score_rank_only"] = out["rankOnlyScore"]
    out["score_without_risk_penalty"] = as_number(out["rawIntegratedScore"])

    for removed in ["model", "disease", "kg", "admet", "pose", "credibility", "label_fit"]:
        include = all_components - {removed}
        out[f"score_without_{removed}_component"] = bounded(weighted_component_score(components, include) - risk)

    return out, variant_notes


def build_topk_overlap(df: pd.DataFrame, variants: list[str]) -> list[dict[str, Any]]:
    full_sorted = df.sort_values("score_final_priority", ascending=False)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        ranked = df.sort_values(f"score_{variant}", ascending=False)
        for cutoff in TOPK_OVERLAP:
            if cutoff > len(df):
                continue
            full_top = set(full_sorted.head(cutoff)["pairId"].astype(str))
            variant_top = set(ranked.head(cutoff)["pairId"].astype(str))
            intersection = full_top & variant_top
            union = full_top | variant_top
            rows.append(
                {
                    "variant": variant,
                    "cutoff": cutoff,
                    "overlapWithFinal": len(intersection),
                    "jaccardWithFinal": round(len(intersection) / len(union), 4) if union else "",
                    "knownHitsInVariantTopK": int(ranked.head(cutoff)["knownDrugTargetPair"].sum()),
                    "knownHitsInFinalTopK": int(full_sorted.head(cutoff)["knownDrugTargetPair"].sum()),
                }
            )
    return rows


def bootstrap_ci(df: pd.DataFrame, variants: list[str], iterations: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(df)
    rows: list[dict[str, Any]] = []
    if n == 0 or iterations <= 0:
        return rows
    indices = np.arange(n)
    for variant in variants:
        score_col = f"score_{variant}"
        auc_values: list[float] = []
        ap_values: list[float] = []
        recall100_values: list[float] = []
        enrichment100_values: list[float] = []
        for _ in range(iterations):
            sample_idx = rng.choice(indices, size=n, replace=True)
            sample = df.iloc[sample_idx].copy()
            labels = sample["knownDrugTargetPair"].astype(int)
            scores = as_number(sample[score_col])
            auc = safe_auc(labels, scores)
            ap = safe_ap(labels, scores)
            if auc is not None:
                auc_values.append(auc)
            if ap is not None:
                ap_values.append(ap)
            ranked = sample.sort_values(score_col, ascending=False).head(min(100, len(sample)))
            positives = int(labels.sum())
            hits = int(ranked["knownDrugTargetPair"].sum())
            if positives:
                recall100_values.append(hits / positives * 100)
            expected = min(100, len(sample)) * (positives / len(sample)) if len(sample) else 0
            if expected:
                enrichment100_values.append(hits / expected)

        for metric_name, values in [
            ("auroc", auc_values),
            ("averagePrecision", ap_values),
            ("recallAt100Pct", recall100_values),
            ("enrichmentAt100", enrichment100_values),
        ]:
            if not values:
                continue
            arr = np.array(values, dtype=float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric_name,
                    "iterations": len(arr),
                    "mean": round(float(arr.mean()), 6),
                    "ciLowerP025": round(float(np.quantile(arr, 0.025)), 6),
                    "ciUpperP975": round(float(np.quantile(arr, 0.975)), 6),
                }
            )
    return rows


def build_markdown(summary: dict[str, Any]) -> str:
    full = summary["fullVariant"]
    model = summary["selectedVariants"].get("model_only", {})
    disease = summary["selectedVariants"].get("disease_evidence_only", {})
    kg = summary["selectedVariants"].get("kg_only", {})
    no_risk = summary["selectedVariants"].get("without_risk_penalty", {})
    lines = [
        "# Final Priority Ablation and Robustness Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit evaluates whether the final multi-evidence priority score enriches known drug-target positives beyond single evidence layers.",
        "",
        "## Headline",
        "",
        f"- Final priority: AUROC {full.get('auroc'):.4f}, AP {full.get('averagePrecision'):.4f}, "
        f"Recall@100 {full.get('recallAt100Pct'):.2f}%, enrichment@100 {full.get('enrichmentAt100'):.2f}x.",
        f"- Model-only: AUROC {model.get('auroc'):.4f}, AP {model.get('averagePrecision'):.4f}, "
        f"Recall@100 {model.get('recallAt100Pct'):.2f}%, enrichment@100 {model.get('enrichmentAt100'):.2f}x.",
        f"- Disease-evidence-only: AUROC {disease.get('auroc'):.4f}, AP {disease.get('averagePrecision'):.4f}, "
        f"Recall@100 {disease.get('recallAt100Pct'):.2f}%, enrichment@100 {disease.get('enrichmentAt100'):.2f}x.",
        f"- KG-only: AUROC {kg.get('auroc'):.4f}, AP {kg.get('averagePrecision'):.4f}, "
        f"Recall@100 {kg.get('recallAt100Pct'):.2f}%, enrichment@100 {kg.get('enrichmentAt100'):.2f}x.",
        f"- Without risk penalty: AUROC {no_risk.get('auroc'):.4f}, AP {no_risk.get('averagePrecision'):.4f}, "
        f"Recall@100 {no_risk.get('recallAt100Pct'):.2f}%, enrichment@100 {no_risk.get('enrichmentAt100'):.2f}x.",
        "",
        "## Largest Ablation Effects",
        "",
    ]
    for row in summary["largestApDrops"]:
        lines.append(
            f"- {row['variant']}: AP delta {row['deltaAveragePrecisionVsFinal']:.4f}, "
            f"Recall@100 delta {row['deltaRecallAt100PctVsFinal']:.2f}%."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The final score is a triage score, not a claim of clinical efficacy. Known-target recall is used as a positive-control audit: a useful prioritization layer should recover known biology while still leaving room for novel candidate review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ablation and robustness audit for final BioMaster priority ranking.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    df = pd.read_csv(root / args.final_table).fillna("")
    df, variant_notes = add_variant_scores(df)
    variants = list(variant_notes)

    model_rows = [rank_metrics(df, variant, f"score_{variant}") for variant in variants]
    cutoff_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for variant in variants:
        score_col = f"score_{variant}"
        cutoff_rows.extend(cutoff_metrics(df, variant, score_col, "all", "all", CUTOFFS))
        for direction, group in df.groupby("direction", dropna=False):
            direction_rows.append(rank_metrics(group, variant, score_col, "direction", str(direction)))
            cutoff_rows.extend(cutoff_metrics(group, variant, score_col, "direction", str(direction), DIRECTION_CUTOFFS))

    by_variant = {row["variant"]: row for row in model_rows}
    cutoff_lookup = {
        (row["variant"], row["groupType"], row["groupValue"], row["cutoff"]): row
        for row in cutoff_rows
    }
    for row in model_rows:
        at100 = cutoff_lookup.get((row["variant"], "all", "all", 100), {})
        at500 = cutoff_lookup.get((row["variant"], "all", "all", 500), {})
        row["recallAt100Pct"] = at100.get("recallPct")
        row["precisionAt100Pct"] = at100.get("precisionPct")
        row["enrichmentAt100"] = at100.get("enrichmentVsRandom")
        row["recallAt500Pct"] = at500.get("recallPct")
        row["precisionAt500Pct"] = at500.get("precisionPct")
        row["enrichmentAt500"] = at500.get("enrichmentVsRandom")
        row["methodNote"] = variant_notes[row["variant"]]

    full = by_variant["final_priority"]
    ablation_delta_rows: list[dict[str, Any]] = []
    for row in model_rows:
        ap = row.get("averagePrecision")
        recall = row.get("recallAt100Pct")
        enrich = row.get("enrichmentAt100")
        ablation_delta_rows.append(
            {
                "variant": row["variant"],
                "methodNote": row["methodNote"],
                "averagePrecision": ap,
                "deltaAveragePrecisionVsFinal": None if ap is None or full.get("averagePrecision") is None else ap - full["averagePrecision"],
                "recallAt100Pct": recall,
                "deltaRecallAt100PctVsFinal": None if recall in (None, "") else recall - full["recallAt100Pct"],
                "enrichmentAt100": enrich,
                "deltaEnrichmentAt100VsFinal": None if enrich in (None, "") else enrich - full["enrichmentAt100"],
            }
        )

    topk_overlap_rows = build_topk_overlap(df, variants)
    bootstrap_rows = bootstrap_ci(df, [v for v in BOOTSTRAP_VARIANTS if v in variants], args.bootstrap_iterations, args.bootstrap_seed)

    selected = {
        name: by_variant[name]
        for name in [
            "final_priority",
            "raw_without_risk",
            "model_only",
            "disease_evidence_only",
            "kg_only",
            "admet_only",
            "pose_only",
            "without_model_component",
            "without_disease_component",
            "without_kg_component",
            "without_admet_component",
            "without_pose_component",
            "without_risk_penalty",
        ]
        if name in by_variant
    }
    largest_ap_drops = sorted(
        [
            row
            for row in ablation_delta_rows
            if row["variant"].startswith("without_") and row["variant"] != "without_risk_penalty" and row["deltaAveragePrecisionVsFinal"] is not None
        ],
        key=lambda row: row["deltaAveragePrecisionVsFinal"],
    )[:5]
    best_by_ap = sorted(model_rows, key=lambda row: row["averagePrecision"] if row["averagePrecision"] is not None else -1, reverse=True)[:8]
    best_by_recall100 = sorted(model_rows, key=lambda row: row["recallAt100Pct"] if row["recallAt100Pct"] not in (None, "") else -1, reverse=True)[:8]

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "knownDrugTargetRows": int(df["knownDrugTargetPair"].sum()),
        "knownDrugTargetRatePct": round(pct(int(df["knownDrugTargetPair"].sum()), len(df)), 4),
        "fullVariant": {
            "auroc": full.get("auroc"),
            "averagePrecision": full.get("averagePrecision"),
            "recallAt100Pct": full.get("recallAt100Pct"),
            "precisionAt100Pct": full.get("precisionAt100Pct"),
            "enrichmentAt100": full.get("enrichmentAt100"),
            "recallAt500Pct": full.get("recallAt500Pct"),
            "precisionAt500Pct": full.get("precisionAt500Pct"),
            "enrichmentAt500": full.get("enrichmentAt500"),
        },
        "selectedVariants": {
            key: {
                "auroc": value.get("auroc"),
                "averagePrecision": value.get("averagePrecision"),
                "recallAt100Pct": value.get("recallAt100Pct"),
                "precisionAt100Pct": value.get("precisionAt100Pct"),
                "enrichmentAt100": value.get("enrichmentAt100"),
            }
            for key, value in selected.items()
        },
        "largestApDrops": largest_ap_drops,
        "bestByAveragePrecision": best_by_ap,
        "bestByRecallAt100": best_by_recall100,
        "methodNote": (
            "Ablations remove one score component and renormalize remaining evidence weights while retaining the original risk penalty. "
            "The without_risk_penalty variant isolates the effect of safety, pose, route, and hard-flag penalties. "
            "Known drug-target positives are positive-control labels, not a complete ground truth for novel repurposing."
        ),
        "outputs": {
            "modelMetrics": str((out_dir / "final_priority_ablation_model_metrics.csv").resolve()),
            "cutoffMetrics": str((out_dir / "final_priority_ablation_cutoff_metrics.csv").resolve()),
            "directionMetrics": str((out_dir / "final_priority_ablation_direction_metrics.csv").resolve()),
            "deltaMetrics": str((out_dir / "final_priority_ablation_delta_metrics.csv").resolve()),
            "topkOverlap": str((out_dir / "final_priority_ablation_topk_overlap.csv").resolve()),
            "bootstrapCi": str((out_dir / "final_priority_ablation_bootstrap_ci.csv").resolve()),
            "summary": str((out_dir / "final_priority_ablation_summary.json").resolve()),
            "markdown": str((out_dir / "FINAL_PRIORITY_ABLATION_AUDIT.md").resolve()),
        },
    }

    write_csv(out_dir / "final_priority_ablation_model_metrics.csv", model_rows)
    write_csv(out_dir / "final_priority_ablation_cutoff_metrics.csv", cutoff_rows)
    write_csv(out_dir / "final_priority_ablation_direction_metrics.csv", direction_rows)
    write_csv(out_dir / "final_priority_ablation_delta_metrics.csv", ablation_delta_rows)
    write_csv(out_dir / "final_priority_ablation_topk_overlap.csv", topk_overlap_rows)
    write_csv(out_dir / "final_priority_ablation_bootstrap_ci.csv", bootstrap_rows)
    write_json(out_dir / "final_priority_ablation_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_ABLATION_AUDIT.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

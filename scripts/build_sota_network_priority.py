from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


POSITIVE_NETWORK_TIERS = {
    "A_direct_disease_module",
    "B_direct_low_score",
    "B_network_close",
    "C_network_reachable",
}

NOVEL_CLASSES = {
    "disease_context_supported_new_pair",
    "model_priority_without_txgnn_kg_path",
}

NETWORK_TIER_SCORE = {
    "A_direct_disease_module": 100.0,
    "B_network_close": 90.0,
    "B_direct_low_score": 82.0,
    "C_network_reachable": 68.0,
    "D_network_distant": 35.0,
    "U_unreachable_in_string": 45.0,
    "U_uncovered_by_string": 50.0,
}

NETWORK_TIER_ADJUSTMENT = {
    "A_direct_disease_module": 4.0,
    "B_network_close": 3.0,
    "B_direct_low_score": 2.0,
    "C_network_reachable": 1.0,
    "D_network_distant": -1.5,
    "U_unreachable_in_string": -0.5,
    "U_uncovered_by_string": 0.0,
}

CUTOFFS = [20, 50, 100, 200, 500, 1000, 2000]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def bool_value(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def network_score(row: pd.Series) -> float:
    tier = str(row.get("networkEvidenceTier") or "U_uncovered_by_string")
    tier_score = NETWORK_TIER_SCORE.get(tier, 50.0)
    proximity = number(row.get("networkProximityScore"))
    if proximity is None:
        return tier_score
    return max(0.0, min(100.0, tier_score * 0.75 + proximity * 100.0 * 0.25))


def network_adjustment(row: pd.Series) -> float:
    tier = str(row.get("networkEvidenceTier") or "U_uncovered_by_string")
    adjustment = NETWORK_TIER_ADJUSTMENT.get(tier, 0.0)
    proximity = number(row.get("networkProximityScore"))
    if tier in POSITIVE_NETWORK_TIERS and proximity is not None:
        adjustment += max(-0.5, min(1.0, (proximity - 0.5) * 1.2))
    return round(adjustment, 4)


def classify_action(row: pd.Series) -> tuple[str, str]:
    base_action = str(row.get("sotaReadyAction", ""))
    tier = str(row.get("networkEvidenceTier", ""))
    novelty = str(row.get("noveltyClass", ""))
    strict_novel = bool_value(row.get("strictNovelPairFlag")) or novelty in NOVEL_CLASSES
    known_pair = bool_value(row.get("knownDrugTargetPair")) or bool_value(row.get("knownDrugTargetPairFlag"))
    direct_mechanism = bool_value(row.get("directKgDrugTargetFlag")) or bool_value(row.get("hasDirectDrugTargetEdge"))

    if base_action in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return base_action, "Network support is reported but does not override safety, structure, or hard-flag review gates."
    if tier in {"A_direct_disease_module", "B_direct_low_score"} and (known_pair or direct_mechanism):
        return "positive_control_network_supported", "Known or mechanism-supported candidate with direct disease-module target evidence."
    if tier in POSITIVE_NETWORK_TIERS and strict_novel:
        return "novel_network_medicine_review", "Novel or new-context candidate with orthogonal disease-module or PPI-proximity support."
    if tier in POSITIVE_NETWORK_TIERS:
        return "network_supported_expert_review", "Candidate has additional network-medicine support for expert triage."
    if tier == "D_network_distant":
        return "network_distant_context_review", "Candidate target is reachable but distant from the current disease module; review context before prioritization."
    if tier.startswith("U_"):
        return "network_coverage_gap_review", "The local STRING subnet does not provide decisive evidence for this target; treat as a data coverage gap."
    return base_action or "secondary_expert_review", "No network-specific action was assigned."


def classify_tier(row: pd.Series) -> str:
    score = float(row.get("sotaNetworkScore", 0.0))
    action = str(row.get("sotaNetworkAction", ""))
    base_tier = str(row.get("sotaReadyTier", ""))
    network_tier = str(row.get("networkEvidenceTier", ""))
    structure_good = str(row.get("structureConfidenceTier", "")).startswith(("A_", "B_"))
    target_good = str(row.get("targetDruggabilityTier", "")).startswith(("A_", "B_"))

    if action in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return "D_blocked_or_requires_resolution"
    if score >= 90.0 and base_tier.startswith(("A_", "B_")) and network_tier in POSITIVE_NETWORK_TIERS and structure_good and target_good:
        return "A_network_supported_expert_priority"
    if score >= 80.0 and base_tier.startswith(("A_", "B_", "C_")) and structure_good and target_good:
        return "B_network_supported_review_priority"
    if score >= 65.0:
        return "C_context_or_secondary_review"
    return "D_low_priority_or_sparse_support"


def safe_auc_ap(df: pd.DataFrame, score_col: str) -> tuple[float | None, float | None]:
    labels = num_col(df, "knownDrugTargetPair").astype(int)
    scores = num_col(df, score_col)
    auc = float(roc_auc_score(labels, scores)) if labels.nunique() >= 2 else None
    ap = float(average_precision_score(labels, scores)) if int(labels.sum()) else None
    return auc, ap


def topk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positives_total = int(num_col(df, "knownDrugTargetPair").sum())
    total = len(df)
    base_rate = positives_total / total if total else 0.0
    ranked = df.sort_values("sotaNetworkScore", ascending=False).reset_index(drop=True)
    for cutoff in CUTOFFS:
        if cutoff > len(ranked):
            continue
        top = ranked.head(cutoff)
        hits = int(num_col(top, "knownDrugTargetPair").sum())
        expected = cutoff * base_rate
        rows.append(
            {
                "groupType": "all",
                "groupValue": "all",
                "cutoff": cutoff,
                "knownDrugTargetRows": hits,
                "knownDrugTargetPct": round(pct(hits, cutoff), 4),
                "recallKnownDrugTargetPct": round(pct(hits, positives_total), 4),
                "randomExpectedHits": round(expected, 4),
                "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
                "networkPositiveRows": int(top["networkPositiveFlag"].sum()),
                "networkDirectRows": int(top["networkDirectFlag"].sum()),
                "networkUncoveredRows": int(top["networkCoverageGapFlag"].sum()),
                "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()),
                "tierCounts": dict(Counter(top["sotaNetworkTier"].astype(str))),
                "networkTierCounts": dict(Counter(top["networkEvidenceTier"].astype(str))),
                "uniqueDrugs": int(top["drugId"].nunique()),
                "uniqueTargets": int(top["protein"].nunique()),
            }
        )
    for direction, group in df.groupby("direction"):
        ranked_dir = group.sort_values("sotaNetworkScore", ascending=False).reset_index(drop=True)
        positives_dir = int(num_col(group, "knownDrugTargetPair").sum())
        base_rate_dir = positives_dir / len(group) if len(group) else 0.0
        for cutoff in [20, 50, 100, 200]:
            if cutoff > len(ranked_dir):
                continue
            top = ranked_dir.head(cutoff)
            hits = int(num_col(top, "knownDrugTargetPair").sum())
            expected = cutoff * base_rate_dir
            rows.append(
                {
                    "groupType": "direction",
                    "groupValue": direction,
                    "cutoff": cutoff,
                    "knownDrugTargetRows": hits,
                    "knownDrugTargetPct": round(pct(hits, cutoff), 4),
                    "recallKnownDrugTargetPct": round(pct(hits, positives_dir), 4),
                    "randomExpectedHits": round(expected, 4),
                    "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
                    "networkPositiveRows": int(top["networkPositiveFlag"].sum()),
                    "networkDirectRows": int(top["networkDirectFlag"].sum()),
                    "networkUncoveredRows": int(top["networkCoverageGapFlag"].sum()),
                    "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()),
                    "tierCounts": dict(Counter(top["sotaNetworkTier"].astype(str))),
                    "networkTierCounts": dict(Counter(top["networkEvidenceTier"].astype(str))),
                    "uniqueDrugs": int(top["drugId"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def direction_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, group in df.groupby("direction"):
        top100 = group.sort_values("sotaNetworkScore", ascending=False).head(min(100, len(group)))
        rows.append(
            {
                "direction": direction,
                "directionLabelZhFinal": group["directionLabelZhFinal"].iloc[0] if "directionLabelZhFinal" in group else "",
                "candidateRows": int(len(group)),
                "networkPositiveRows": int(group["networkPositiveFlag"].sum()),
                "networkPositivePct": round(pct(int(group["networkPositiveFlag"].sum()), len(group)), 4),
                "networkDirectRows": int(group["networkDirectFlag"].sum()),
                "networkCoverageGapRows": int(group["networkCoverageGapFlag"].sum()),
                "top100NetworkPositiveRows": int(top100["networkPositiveFlag"].sum()),
                "top100KnownDrugTargetRows": int(num_col(top100, "knownDrugTargetPair").sum()),
                "top100NovelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()),
                "medianNetworkMedicineScore": round(float(group["networkMedicineScore"].median()), 4),
                "medianSotaNetworkScore": round(float(group["sotaNetworkScore"].median()), 4),
                "networkTierCounts": dict(Counter(group["networkEvidenceTier"].astype(str))),
                "sotaNetworkTierCounts": dict(Counter(group["sotaNetworkTier"].astype(str))),
            }
        )
    return pd.DataFrame(rows)


def output_columns(df: pd.DataFrame) -> list[str]:
    leading = [
        "sotaNetworkRankGlobal",
        "sotaNetworkRankWithinDirection",
        "sotaReadyRankGlobal",
        "sotaReadyRankWithinDirection",
        "globalRankDeltaPositivePromoted",
        "directionRankDeltaPositivePromoted",
        "direction",
        "directionLabelZhFinal",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "sotaNetworkScore",
        "sotaNetworkTier",
        "sotaNetworkAction",
        "sotaNetworkActionNote",
        "networkMedicineScore",
        "networkMedicineAdjustment",
        "networkPositiveFlag",
        "networkDirectFlag",
        "networkCoverageGapFlag",
        "networkEvidenceTier",
        "networkEvidenceReason",
        "directOpenTargetsScore",
        "directDiseaseName",
        "shortestHopToDiseaseModule",
        "weightedDistanceToDiseaseModule",
        "networkProximityPercentile",
        "networkProximityZ",
        "networkProximityScore",
        "sotaReadyScore",
        "sotaReadyTier",
        "sotaReadyAction",
        "finalPriorityScore",
        "knownDrugTargetPair",
        "noveltyClass",
        "strictNovelPairFlag",
        "mechanismExtensionFlag",
        "structureConfidenceTier",
        "targetDruggabilityTier",
        "evidenceConcordanceTier",
        "admetTier",
        "kgEvidenceScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "diffdock",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    seen: set[str] = set()
    columns: list[str] = []
    for col in leading + list(df.columns):
        if col in df.columns and col not in seen:
            columns.append(col)
            seen.add(col)
    return columns


def build_network_priority(root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    sota = read_csv(root / args.sota_ready)
    network = read_csv(root / args.network_audit)
    required = {"direction", "pairId"}
    if not required.issubset(sota.columns) or not required.issubset(network.columns):
        raise ValueError("Both inputs must contain direction and pairId columns.")

    network_cols = [
        "direction",
        "pairId",
        "stringCovered",
        "directOpenTargetsScore",
        "directDiseaseName",
        "shortestHopToDiseaseModule",
        "weightedDistanceToDiseaseModule",
        "networkProximityPercentile",
        "networkProximityZ",
        "networkProximityScore",
        "networkEvidenceTier",
        "networkEvidenceReason",
    ]
    network_small = network[[col for col in network_cols if col in network.columns]].drop_duplicates(["direction", "pairId"])
    df = sota.merge(network_small, on=["direction", "pairId"], how="left")
    missing_network_rows = int(df["networkEvidenceTier"].eq("").sum() if "networkEvidenceTier" in df else len(df))
    df["networkEvidenceTier"] = df["networkEvidenceTier"].replace("", "U_uncovered_by_string").fillna("U_uncovered_by_string")
    df["networkMedicineScore"] = df.apply(network_score, axis=1).round(4)
    df["networkMedicineAdjustment"] = df.apply(network_adjustment, axis=1)
    df["networkPositiveFlag"] = df["networkEvidenceTier"].isin(POSITIVE_NETWORK_TIERS)
    df["networkDirectFlag"] = df["networkEvidenceTier"].isin({"A_direct_disease_module", "B_direct_low_score"})
    df["networkCoverageGapFlag"] = df["networkEvidenceTier"].astype(str).str.startswith("U_")
    df["sotaNetworkScore"] = (num_col(df, "sotaReadyScore") + df["networkMedicineAdjustment"]).clip(0, 100).round(4)

    actions = df.apply(classify_action, axis=1)
    df["sotaNetworkAction"] = [item[0] for item in actions]
    df["sotaNetworkActionNote"] = [item[1] for item in actions]
    df["sotaNetworkTier"] = df.apply(classify_tier, axis=1)

    df = df.sort_values("sotaNetworkScore", ascending=False).reset_index(drop=True)
    df["sotaNetworkRankGlobal"] = np.arange(1, len(df) + 1)
    df["sotaNetworkRankWithinDirection"] = (
        df.groupby("direction")["sotaNetworkScore"].rank(method="first", ascending=False).astype(int)
    )
    df["globalRankDeltaPositivePromoted"] = num_col(df, "sotaReadyRankGlobal") - df["sotaNetworkRankGlobal"]
    df["directionRankDeltaPositivePromoted"] = num_col(df, "sotaReadyRankWithinDirection") - df["sotaNetworkRankWithinDirection"]

    old_auc, old_ap = safe_auc_ap(df, "sotaReadyScore")
    new_auc, new_ap = safe_auc_ap(df, "sotaNetworkScore")
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "missingNetworkRowsAfterJoin": missing_network_rows,
        "networkTierCounts": dict(Counter(df["networkEvidenceTier"].astype(str))),
        "sotaNetworkTierCounts": dict(Counter(df["sotaNetworkTier"].astype(str))),
        "sotaNetworkActionCounts": dict(Counter(df["sotaNetworkAction"].astype(str))),
        "networkPositiveRows": int(df["networkPositiveFlag"].sum()),
        "networkDirectRows": int(df["networkDirectFlag"].sum()),
        "networkCoverageGapRows": int(df["networkCoverageGapFlag"].sum()),
        "networkPositivePct": pct(int(df["networkPositiveFlag"].sum()), len(df)),
        "oldSotaReadyAuroc": old_auc,
        "oldSotaReadyAveragePrecision": old_ap,
        "sotaNetworkAuroc": new_auc,
        "sotaNetworkAveragePrecision": new_ap,
        "top100": {
            "knownDrugTargetRows": int(num_col(df.head(100), "knownDrugTargetPair").sum()),
            "networkPositiveRows": int(df.head(100)["networkPositiveFlag"].sum()),
            "networkDirectRows": int(df.head(100)["networkDirectFlag"].sum()),
            "novelRows": int(df.head(100)["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()),
            "tierCounts": dict(Counter(df.head(100)["sotaNetworkTier"].astype(str))),
            "networkTierCounts": dict(Counter(df.head(100)["networkEvidenceTier"].astype(str))),
        },
        "methodNote": (
            "SOTA-network score is an additive, coverage-aware adjustment to the existing SOTA-ready score. "
            "Direct disease-module targets and close/reachable PPI proximity add small positive evidence; "
            "STRING-uncovered targets are kept neutral and treated as data gaps, not negative biology."
        ),
        "inputs": {
            "sotaReady": args.sota_ready,
            "networkAudit": args.network_audit,
        },
    }
    return df, summary


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    top100 = summary["top100"]
    lines = [
        "# SOTA + Network Medicine Priority Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit integrates the PPI/network-medicine proximity layer into the existing SOTA-ready candidate triage.",
        "",
        "## Summary",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Network-positive rows: {summary['networkPositiveRows']} ({summary['networkPositivePct']:.2f}%)",
        f"- Direct disease-module rows: {summary['networkDirectRows']}",
        f"- Network coverage-gap rows: {summary['networkCoverageGapRows']}",
        f"- Missing network rows after join: {summary['missingNetworkRowsAfterJoin']}",
        f"- SOTA-ready AP -> SOTA-network AP: {summary['oldSotaReadyAveragePrecision']:.6f} -> {summary['sotaNetworkAveragePrecision']:.6f}",
        f"- Top100 known target rows: {top100['knownDrugTargetRows']}",
        f"- Top100 network-positive rows: {top100['networkPositiveRows']}",
        f"- Top100 novel rows: {top100['novelRows']}",
        "",
        "## Direction Summary",
        "",
    ]
    for _, row in direction_df.iterrows():
        lines.append(
            f"- {row['direction']}: rows={row['candidateRows']}, "
            f"networkPositive={row['networkPositiveRows']} ({row['networkPositivePct']:.2f}%), "
            f"top100NetworkPositive={row['top100NetworkPositiveRows']}, "
            f"top100Known={row['top100KnownDrugTargetRows']}, top100Novel={row['top100NovelRows']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an orthogonal validation layer; it does not replace ConPLex, Open Targets, TxGNN, DiffDock, ADMET, target druggability, or chemotype checks.",
            "- Direct disease-module targets and close PPI proximity strengthen candidate plausibility for expert review.",
            "- STRING-uncovered targets remain neutral because the current local STRING subnet is intentionally narrow.",
            "- Safety, structure, and hard-flag review gates still override network support.",
            "",
            "## Outputs",
            "",
            "- Full matrix: `final_priority_sota_network_matrix.csv`",
            "- Expert shortlist: `final_priority_sota_network_top300_expert_shortlist.csv`",
            "- Novel network-supported shortlist: `final_priority_sota_network_novel_shortlist.csv`",
            "- Rank-shift review: `final_priority_sota_network_rank_shift_review.csv`",
            "- TopK metrics: `final_priority_sota_network_topk_metrics.csv`",
            "- Direction summary: `final_priority_sota_network_direction_summary.csv`",
        ]
    )
    all_top100 = metrics_df[(metrics_df["groupType"].eq("all")) & (metrics_df["cutoff"].eq(100))]
    if not all_top100.empty:
        item = all_top100.iloc[0]
        lines.extend(
            [
                "",
                "## Top100 Metrics",
                "",
                f"- Known target precision: {item['knownDrugTargetPct']:.2f}%",
                f"- Known target recall: {item['recallKnownDrugTargetPct']:.2f}%",
                f"- Enrichment versus random: {item['enrichmentVsRandom']}",
                f"- Network-positive rows: {item['networkPositiveRows']}",
                f"- Network coverage-gap rows: {item['networkUncoveredRows']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate network-medicine evidence into SOTA-ready candidate priority.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--sota-ready",
        default="outputs/sota_validation/final_prioritization/final_priority_sota_ready_matrix.csv",
    )
    parser.add_argument(
        "--network-audit",
        default="outputs/sota_validation/network_proximity/final_priority_network_proximity_audit.csv",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, summary = build_network_priority(root, args)
    columns = output_columns(df)
    matrix_path = out_dir / "final_priority_sota_network_matrix.csv"
    df.to_csv(matrix_path, index=False, columns=columns)

    expert = df[~df["sotaNetworkTier"].astype(str).str.startswith("D_")].head(300).copy()
    expert.to_csv(out_dir / "final_priority_sota_network_top300_expert_shortlist.csv", index=False, columns=columns)

    novel = df[
        (df["networkPositiveFlag"])
        & (df["noveltyClass"].astype(str).isin(NOVEL_CLASSES) | df["strictNovelPairFlag"].map(bool_value))
        & ~df["sotaNetworkTier"].astype(str).str.startswith("D_")
    ].head(300)
    novel.to_csv(out_dir / "final_priority_sota_network_novel_shortlist.csv", index=False, columns=columns)

    rank_shift = df.copy()
    rank_shift["absoluteGlobalRankDelta"] = rank_shift["globalRankDeltaPositivePromoted"].abs()
    rank_shift = rank_shift.sort_values("absoluteGlobalRankDelta", ascending=False).head(300)
    rank_shift_cols = ["absoluteGlobalRankDelta"] + [col for col in columns if col in rank_shift.columns]
    rank_shift.to_csv(out_dir / "final_priority_sota_network_rank_shift_review.csv", index=False, columns=rank_shift_cols)

    metrics = topk_metrics(df)
    direction = direction_summary(df)
    metrics.to_csv(out_dir / "final_priority_sota_network_topk_metrics.csv", index=False)
    direction.to_csv(out_dir / "final_priority_sota_network_direction_summary.csv", index=False)

    summary["outputs"] = {
        "matrix": str(matrix_path),
        "expertShortlist": str(out_dir / "final_priority_sota_network_top300_expert_shortlist.csv"),
        "novelShortlist": str(out_dir / "final_priority_sota_network_novel_shortlist.csv"),
        "rankShiftReview": str(out_dir / "final_priority_sota_network_rank_shift_review.csv"),
        "topkMetrics": str(out_dir / "final_priority_sota_network_topk_metrics.csv"),
        "directionSummary": str(out_dir / "final_priority_sota_network_direction_summary.csv"),
        "summary": str(out_dir / "final_priority_sota_network_summary.json"),
        "markdown": str(out_dir / "FINAL_PRIORITY_SOTA_NETWORK_MEDICINE_AUDIT.md"),
    }
    write_json(out_dir / "final_priority_sota_network_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_SOTA_NETWORK_MEDICINE_AUDIT.md").write_text(
        markdown(summary, direction, metrics),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

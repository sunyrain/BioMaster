from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


TOPK_CUTOFFS = [20, 50, 100, 200, 500, 1000, 2000, 3921]
DIRECTION_TOPK_CUTOFFS = [20, 50, 100, 200, 500]


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


def margin_class(n_directions: int, margin: float | None) -> str:
    if n_directions <= 1:
        return "single_direction_observed"
    if margin is None or not math.isfinite(float(margin)):
        return "multi_direction_unranked_margin"
    if margin >= 15:
        return "highly_direction_specific"
    if margin >= 7.5:
        return "moderately_direction_specific"
    if margin >= 3:
        return "weakly_direction_specific"
    return "broad_multi_direction_generalist"


def pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, group in df.groupby("pairId", dropna=False):
        ranked = group.sort_values(["finalPriorityScore", "finalRankGlobal"], ascending=[False, True])
        top = ranked.iloc[0]
        second_score = float(ranked.iloc[1]["finalPriorityScore"]) if len(ranked) > 1 else None
        top_score = float(top["finalPriorityScore"])
        margin = top_score - second_score if second_score is not None else None
        directions = ranked["direction"].astype(str).tolist()
        rows.append(
            {
                "pairId": pair_id,
                "drugId": top.get("drugId", ""),
                "drug": top.get("drug", ""),
                "target": top.get("target", ""),
                "protein": top.get("protein", ""),
                "proteinName": top.get("proteinName", ""),
                "nDirections": int(group["direction"].nunique()),
                "directions": "; ".join(sorted(set(group["direction"].astype(str)))),
                "topDirection": top["direction"],
                "topDirectionLabelZh": top.get("directionLabelZhFinal", ""),
                "topScore": round(top_score, 4),
                "secondScore": round(second_score, 4) if second_score is not None else "",
                "topSecondMargin": round(margin, 4) if margin is not None else "",
                "scoreMean": round(float(group["finalPriorityScore"].mean()), 4),
                "scoreStd": round(float(group["finalPriorityScore"].std(ddof=0)), 4),
                "scoreMin": round(float(group["finalPriorityScore"].min()), 4),
                "scoreMax": round(float(group["finalPriorityScore"].max()), 4),
                "rankBest": int(group["finalRankGlobal"].min()),
                "rankWorst": int(group["finalRankGlobal"].max()),
                "top100Any": int(group["finalRankGlobal"].le(100).any()),
                "top500Any": int(group["finalRankGlobal"].le(500).any()),
                "knownDrugTargetPairAny": int(as_number(group["knownDrugTargetPair"]).gt(0).any()),
                "directKgDrugTargetAny": int(as_number(group["hasDirectDrugTargetEdge"]).gt(0).any()),
                "positiveDrugDiseaseAny": int(as_number(group["hasPositiveDrugDiseaseEdge"]).gt(0).any()),
                "contraindicationAny": int(as_number(group["hasContraindicationDiseaseEdge"]).gt(0).any()),
                "strictNovelLikeAny": int(
                    (
                        as_number(group["knownDrugTargetPair"]).eq(0)
                        & as_number(group["hasDirectDrugTargetEdge"]).eq(0)
                        & as_number(group["hasPositiveDrugDiseaseEdge"]).eq(0)
                        & as_number(group["hasContraindicationDiseaseEdge"]).eq(0)
                        & (
                            as_number(group["hasDirectTargetDiseaseEdge"]).gt(0)
                            | as_number(group["ppiDiseaseBridgeCount"]).gt(0)
                            | as_number(group["drugTargetDiseaseBridgeCount"]).gt(0)
                        )
                    ).any()
                ),
                "directionSpecificityClass": margin_class(int(group["direction"].nunique()), margin),
                "rankedDirectionsByScore": "; ".join(
                    f"{row.direction}:{row.finalPriorityScore:.3f}" for row in ranked.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def add_row_specificity(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        pairs[
            [
                "pairId",
                "nDirections",
                "directions",
                "topDirection",
                "topScore",
                "secondScore",
                "topSecondMargin",
                "directionSpecificityClass",
                "rankedDirectionsByScore",
            ]
        ],
        on="pairId",
        how="left",
        suffixes=("", "Pair"),
    )
    merged["isPairTopDirection"] = merged["direction"].astype(str).eq(merged["topDirection"].astype(str))
    merged["isBroadMultiDirectionPair"] = merged["directionSpecificityClass"].eq("broad_multi_direction_generalist")
    merged["isDirectionSpecificPair"] = merged["directionSpecificityClass"].isin(
        ["single_direction_observed", "highly_direction_specific", "moderately_direction_specific"]
    )
    merged["scoreDeltaFromPairTop"] = as_number(merged["topScore"]) - as_number(merged["finalPriorityScore"])
    return merged


def composition_row(df: pd.DataFrame, group_type: str, group_value: str, cutoff: int | str) -> dict[str, Any]:
    total = len(df)
    multi_rows = int(as_number(df["nDirections"]).gt(1).sum())
    broad_rows = int(df["isBroadMultiDirectionPair"].sum())
    specific_rows = int(df["isDirectionSpecificPair"].sum())
    pair_top_rows = int(df["isPairTopDirection"].sum())
    repeated_pair_ids = int(df.loc[as_number(df["nDirections"]).gt(1), "pairId"].nunique())
    return {
        "groupType": group_type,
        "groupValue": group_value,
        "cutoff": cutoff,
        "rows": total,
        "uniquePairs": int(df["pairId"].nunique()) if total else 0,
        "multiDirectionPairRows": multi_rows,
        "multiDirectionPairPct": round(pct(multi_rows, total), 4),
        "broadGeneralistRows": broad_rows,
        "broadGeneralistPct": round(pct(broad_rows, total), 4),
        "directionSpecificRows": specific_rows,
        "directionSpecificPct": round(pct(specific_rows, total), 4),
        "pairTopDirectionRows": pair_top_rows,
        "pairTopDirectionPct": round(pct(pair_top_rows, total), 4),
        "repeatedPairIds": repeated_pair_ids,
        "uniqueDrugs": int(df["drug"].nunique()) if total else 0,
        "uniqueTargets": int(df["target"].nunique()) if total else 0,
        "specificityClassCounts": dict(Counter(df["directionSpecificityClass"])),
        "directionCounts": dict(Counter(df["direction"])),
        "reviewTrackCounts": dict(Counter(df["reviewTrack"])),
    }


def topk_composition(row_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranked = row_df.sort_values("finalPriorityScore", ascending=False)
    for cutoff in TOPK_CUTOFFS:
        if cutoff <= len(ranked):
            rows.append(composition_row(ranked.head(cutoff), "all", "all", cutoff))
    for direction, group in row_df.groupby("direction", dropna=False):
        direction_ranked = group.sort_values("finalPriorityScore", ascending=False)
        for cutoff in DIRECTION_TOPK_CUTOFFS:
            if cutoff <= len(direction_ranked):
                rows.append(composition_row(direction_ranked.head(cutoff), "direction", str(direction), cutoff))
    return rows


def group_composition(row_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [composition_row(row_df, "all", "all", "all")]
    for col, group_type in [
        ("direction", "direction"),
        ("directionSpecificityClass", "directionSpecificityClass"),
        ("reviewTrack", "reviewTrack"),
        ("finalPriorityTier", "finalPriorityTier"),
        ("noveltyClass", "noveltyClass"),
    ]:
        for value, group in row_df.groupby(col, dropna=False):
            rows.append(composition_row(group, group_type, str(value), "all"))
    return rows


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "directionLabelZhFinal",
        "pairId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "finalPriorityScore",
        "finalPriorityTier",
        "reviewTrack",
        "noveltyClass",
        "knownDrugTargetPair",
        "nDirections",
        "topDirection",
        "topScore",
        "topSecondMargin",
        "directionSpecificityClass",
        "isPairTopDirection",
        "scoreDeltaFromPairTop",
        "rankedDirectionsByScore",
        "hasDirectDrugTargetEdge",
        "hasPositiveDrugDiseaseEdge",
        "hasContraindicationDiseaseEdge",
        "hasDirectTargetDiseaseEdge",
        "ppiDiseaseBridgeCount",
        "drugTargetDiseaseBridgeCount",
        "status",
        "poseAuditStatus",
        "admetTier",
        "directionScore",
        "affinityScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "kgEvidenceScore",
        "diffdock",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "validationGatesZh",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    return df[[col for col in cols if col in df.columns]]


def build_markdown(summary: dict[str, Any]) -> str:
    top100 = summary["topK"]["100"]
    lines = [
        "# Direction Specificity Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit separates disease-direction-specific candidates from broad multi-direction candidates.",
        "",
        "## Headline",
        "",
        f"- Final candidate rows: {summary['candidateRows']}; unique drug-target pairs: {summary['uniquePairs']}.",
        f"- Repeated drug-target pairs across multiple directions: {summary['multiDirectionPairIds']} ({summary['multiDirectionPairIdPct']:.2f}% of unique pairs).",
        f"- Top100 rows from multi-direction pairs: {top100['multiDirectionPairRows']} ({top100['multiDirectionPairPct']:.2f}%).",
        f"- Top100 broad-generalist rows: {top100['broadGeneralistRows']} ({top100['broadGeneralistPct']:.2f}%).",
        f"- Top100 rows that are the pair's best-scoring disease direction: {top100['pairTopDirectionRows']} ({top100['pairTopDirectionPct']:.2f}%).",
        f"- Direction-specific pair rows in Top100: {top100['directionSpecificRows']} ({top100['directionSpecificPct']:.2f}%).",
        "",
        "## Interpretation",
        "",
        "Repeated pairs are not necessarily false positives. They identify general mechanisms or broad repurposing hypotheses. For disease-focused expert review, prioritize rows where the disease direction is the pair's top-scoring direction and the score margin is moderate or high.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit cross-disease direction specificity for final prioritized candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--top-review-limit", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    df = pd.read_csv(root / args.final_table).fillna("")
    for col in ["finalPriorityScore", "finalRankGlobal", "finalRankWithinDirection", "rank"]:
        df[col] = as_number(df[col])

    pair_df = pair_summary(df).sort_values(["topScore", "pairId"], ascending=[False, True])
    row_df = add_row_specificity(df, pair_df).sort_values("finalPriorityScore", ascending=False)
    topk_rows = topk_composition(row_df)
    group_rows = group_composition(row_df)
    topk_lookup = {str(row["cutoff"]): row for row in topk_rows if row["groupType"] == "all"}

    focused_mask = (
        row_df["isPairTopDirection"]
        & row_df["directionSpecificityClass"].isin(
            ["single_direction_observed", "highly_direction_specific", "moderately_direction_specific"]
        )
        & row_df["reviewTrack"].isin(
            ["A_repurposing_mechanism_review", "B_novel_pair_disease_context_review", "B_mechanism_review"]
        )
        & row_df["finalPriorityTier"].isin(["A", "B"])
    )
    broad_mask = row_df["isBroadMultiDirectionPair"] & row_df["finalRankGlobal"].le(1000)

    focused_shortlist = select_columns(row_df[focused_mask].head(args.top_review_limit))
    broad_review = select_columns(row_df[broad_mask].head(args.top_review_limit))
    row_audit = select_columns(row_df)

    unique_pairs = int(pair_df["pairId"].nunique())
    multi_pair_ids = int(pair_df["nDirections"].gt(1).sum())
    broad_pair_ids = int(pair_df["directionSpecificityClass"].eq("broad_multi_direction_generalist").sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(row_df)),
        "uniquePairs": unique_pairs,
        "multiDirectionPairIds": multi_pair_ids,
        "multiDirectionPairIdPct": round(pct(multi_pair_ids, unique_pairs), 4),
        "broadGeneralistPairIds": broad_pair_ids,
        "broadGeneralistPairIdPct": round(pct(broad_pair_ids, unique_pairs), 4),
        "directionSpecificPairClassCounts": dict(Counter(pair_df["directionSpecificityClass"])),
        "topK": topk_lookup,
        "focusedDirectionSpecificShortlistRows": int(len(focused_shortlist)),
        "broadGeneralistReviewRows": int(len(broad_review)),
        "methodNote": (
            "Pair-level specificity compares the same drug-target pair across disease directions using finalPriorityScore. "
            "High or moderate margin means the pair is more disease-direction-specific; low margin across multiple directions means broad/generalist behavior."
        ),
        "outputs": {
            "pairAudit": str((out_dir / "final_priority_direction_specificity_pair_audit.csv").resolve()),
            "rowAudit": str((out_dir / "final_priority_direction_specificity_row_audit.csv").resolve()),
            "topkComposition": str((out_dir / "final_priority_direction_specificity_topk_composition.csv").resolve()),
            "groupComposition": str((out_dir / "final_priority_direction_specificity_group_composition.csv").resolve()),
            "focusedShortlist": str((out_dir / "final_priority_direction_specific_shortlist.csv").resolve()),
            "broadGeneralistReview": str((out_dir / "final_priority_broad_generalist_review.csv").resolve()),
            "summary": str((out_dir / "final_priority_direction_specificity_summary.json").resolve()),
            "markdown": str((out_dir / "FINAL_PRIORITY_DIRECTION_SPECIFICITY_AUDIT.md").resolve()),
        },
    }

    pair_df.to_csv(out_dir / "final_priority_direction_specificity_pair_audit.csv", index=False)
    row_audit.to_csv(out_dir / "final_priority_direction_specificity_row_audit.csv", index=False)
    write_csv(out_dir / "final_priority_direction_specificity_topk_composition.csv", topk_rows)
    write_csv(out_dir / "final_priority_direction_specificity_group_composition.csv", group_rows)
    focused_shortlist.to_csv(out_dir / "final_priority_direction_specific_shortlist.csv", index=False)
    broad_review.to_csv(out_dir / "final_priority_broad_generalist_review.csv", index=False)
    write_json(out_dir / "final_priority_direction_specificity_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_DIRECTION_SPECIFICITY_AUDIT.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

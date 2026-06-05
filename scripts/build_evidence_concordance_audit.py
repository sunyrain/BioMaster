from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


TOPK_CUTOFFS = [20, 50, 100, 200, 500, 1000, 2000, 3921]


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


def add_concordance_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["modelSupportFlag"] = as_number(out["modelComponent"]).ge(80)
    out["modelStrongFlag"] = as_number(out["modelComponent"]).ge(85)
    out["diseaseSupportFlag"] = (
        as_number(out["diseaseEvidenceComponent"]).ge(52)
        | as_number(out["openTargetsScore"]).gt(0)
        | as_number(out["integratedTxgnnScore"]).gt(0)
        | as_number(out["directionLabelFit"]).eq(1)
    )
    out["diseaseStrongFlag"] = (
        as_number(out["diseaseEvidenceComponent"]).ge(60)
        | as_number(out["openTargetsScore"]).ge(0.5)
        | as_number(out["integratedTxgnnScore"]).ge(0.5)
    )
    out["kgSupportFlag"] = (
        as_number(out["kgEvidenceScore"]).ge(35)
        | as_number(out["pathCount"]).gt(0)
        | as_number(out["hasDirectTargetDiseaseEdge"]).gt(0)
        | as_number(out["ppiDiseaseBridgeCount"]).gt(0)
        | as_number(out["drugTargetDiseaseBridgeCount"]).gt(0)
    )
    out["kgStrongFlag"] = (
        as_number(out["kgEvidenceScore"]).ge(60)
        | as_number(out["hasDirectDrugTargetEdge"]).gt(0)
        | as_number(out["hasPositiveDrugDiseaseEdge"]).gt(0)
        | as_number(out["hasDirectTargetDiseaseEdge"]).gt(0)
    )
    out["admetSupportFlag"] = out["admetTier"].astype(str).isin(["A", "B"]) & as_number(out["admetScore"]).ge(75)
    out["admetStrongFlag"] = out["admetTier"].astype(str).eq("A") & as_number(out["admetScore"]).ge(90)
    out["structureSupportFlag"] = out["status"].astype(str).eq("completed") & as_number(out["structureGeometryScore"]).ge(72)
    out["structureStrongFlag"] = out["poseAuditStatus"].astype(str).eq("pass") & as_number(out["structureGeometryScore"]).ge(90)
    out["labelFitSupportFlag"] = as_number(out["directionLabelFit"]).eq(1)

    hard_flags = out["hardFlags"].astype(str).str.lower()
    out["contraindicationFlag"] = as_number(out["hasContraindicationDiseaseEdge"]).gt(0)
    out["hardFlagCleanFlag"] = hard_flags.eq("none") | hard_flags.eq("")
    out["riskCleanFlag"] = (
        out["hardFlagCleanFlag"]
        & ~out["contraindicationFlag"]
        & out["admetTier"].astype(str).ne("D")
        & out["status"].astype(str).eq("completed")
        & out["poseAuditStatus"].astype(str).ne("fail")
        & as_number(out["riskPenalty"]).le(8)
    )

    support_cols = [
        "modelSupportFlag",
        "diseaseSupportFlag",
        "kgSupportFlag",
        "admetSupportFlag",
        "structureSupportFlag",
        "labelFitSupportFlag",
    ]
    strong_cols = [
        "modelStrongFlag",
        "diseaseStrongFlag",
        "kgStrongFlag",
        "admetStrongFlag",
        "structureStrongFlag",
    ]
    out["evidenceSupportCount"] = out[support_cols].sum(axis=1).astype(int)
    out["strongEvidenceCount"] = out[strong_cols].sum(axis=1).astype(int)
    out["singleEvidenceDominatedFlag"] = out["evidenceSupportCount"].le(2)
    out["multiEvidenceFlag"] = out["evidenceSupportCount"].ge(4)
    out["highConcordanceFlag"] = out["evidenceSupportCount"].ge(5) & out["riskCleanFlag"]

    tiers: list[str] = []
    labels: list[str] = []
    for _, row in out.iterrows():
        if not row["hardFlagCleanFlag"] or row["admetTier"] == "D" or row["status"] != "completed":
            tiers.append("D")
            labels.append("blocked_by_safety_structure_or_hard_flag")
        elif row["highConcordanceFlag"] and row["strongEvidenceCount"] >= 3:
            tiers.append("A")
            labels.append("high_multi_evidence_concordance")
        elif row["multiEvidenceFlag"] and row["riskCleanFlag"]:
            tiers.append("B")
            labels.append("moderate_multi_evidence_concordance")
        elif row["evidenceSupportCount"] >= 3:
            tiers.append("C")
            labels.append("partial_evidence_or_risk_review")
        else:
            tiers.append("D")
            labels.append("single_or_sparse_evidence_priority")
    out["evidenceConcordanceTier"] = tiers
    out["evidenceConcordanceClass"] = labels
    out["supportFlagsText"] = [
        "; ".join(col.replace("Flag", "") for col in support_cols if bool(row[col])) or "none"
        for _, row in out.iterrows()
    ]
    return out


def composition_row(df: pd.DataFrame, group_type: str, group_value: str, cutoff: int | str) -> dict[str, Any]:
    total = len(df)
    multi = int(df["multiEvidenceFlag"].sum())
    high = int(df["highConcordanceFlag"].sum())
    single = int(df["singleEvidenceDominatedFlag"].sum())
    risk_clean = int(df["riskCleanFlag"].sum())
    known = int(as_number(df["knownDrugTargetPair"]).sum())
    return {
        "groupType": group_type,
        "groupValue": group_value,
        "cutoff": cutoff,
        "rows": total,
        "knownDrugTargetRows": known,
        "knownDrugTargetPct": round(pct(known, total), 4),
        "multiEvidenceRows": multi,
        "multiEvidencePct": round(pct(multi, total), 4),
        "highConcordanceRows": high,
        "highConcordancePct": round(pct(high, total), 4),
        "singleEvidenceDominatedRows": single,
        "singleEvidenceDominatedPct": round(pct(single, total), 4),
        "riskCleanRows": risk_clean,
        "riskCleanPct": round(pct(risk_clean, total), 4),
        "meanEvidenceSupportCount": round(float(df["evidenceSupportCount"].mean()), 4) if total else 0,
        "meanStrongEvidenceCount": round(float(df["strongEvidenceCount"].mean()), 4) if total else 0,
        "tierCounts": dict(Counter(df["evidenceConcordanceTier"])),
        "classCounts": dict(Counter(df["evidenceConcordanceClass"])),
        "reviewTrackCounts": dict(Counter(df["reviewTrack"])),
        "finalPriorityTierCounts": dict(Counter(df["finalPriorityTier"])),
    }


def topk_composition(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranked = df.sort_values("finalPriorityScore", ascending=False)
    for cutoff in TOPK_CUTOFFS:
        if cutoff <= len(ranked):
            rows.append(composition_row(ranked.head(cutoff), "all", "all", cutoff))
    for direction, group in df.groupby("direction", dropna=False):
        ranked_group = group.sort_values("finalPriorityScore", ascending=False)
        for cutoff in [20, 50, 100, 200, 500]:
            if cutoff <= len(ranked_group):
                rows.append(composition_row(ranked_group.head(cutoff), "direction", str(direction), cutoff))
    return rows


def group_composition(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [composition_row(df, "all", "all", "all")]
    for col, group_type in [
        ("direction", "direction"),
        ("evidenceConcordanceTier", "evidenceConcordanceTier"),
        ("evidenceConcordanceClass", "evidenceConcordanceClass"),
        ("reviewTrack", "reviewTrack"),
        ("finalPriorityTier", "finalPriorityTier"),
        ("noveltyClass", "noveltyClass"),
    ]:
        for value, group in df.groupby(col, dropna=False):
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
        "evidenceConcordanceTier",
        "evidenceConcordanceClass",
        "evidenceSupportCount",
        "strongEvidenceCount",
        "supportFlagsText",
        "riskCleanFlag",
        "singleEvidenceDominatedFlag",
        "knownDrugTargetPair",
        "modelComponent",
        "diseaseEvidenceComponent",
        "kgEvidenceScore",
        "admetScore",
        "structureGeometryScore",
        "riskPenalty",
        "status",
        "poseAuditStatus",
        "admetTier",
        "openTargetsScore",
        "integratedTxgnnScore",
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
    top500 = summary["topK"]["500"]
    lines = [
        "# Evidence Concordance Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit labels candidates by how many independent evidence layers support the final priority.",
        "",
        "## Headline",
        "",
        f"- Final candidate rows: {summary['candidateRows']}; high-concordance rows: {summary['highConcordanceRows']} ({summary['highConcordancePct']:.2f}%).",
        f"- Multi-evidence rows: {summary['multiEvidenceRows']} ({summary['multiEvidencePct']:.2f}%).",
        f"- Top100: multi-evidence {top100['multiEvidenceRows']} ({top100['multiEvidencePct']:.2f}%), high-concordance {top100['highConcordanceRows']} ({top100['highConcordancePct']:.2f}%), single/sparse evidence {top100['singleEvidenceDominatedRows']} ({top100['singleEvidenceDominatedPct']:.2f}%).",
        f"- Top500: multi-evidence {top500['multiEvidenceRows']} ({top500['multiEvidencePct']:.2f}%), high-concordance {top500['highConcordanceRows']} ({top500['highConcordancePct']:.2f}%).",
        "",
        "## Interpretation",
        "",
        "High-concordance candidates are better suited for near-term expert review. Single/sparse-evidence candidates can remain as exploration leads but should not be presented as mature repurposing hypotheses.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit multi-evidence concordance for final priority candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--review-limit", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    df = pd.read_csv(root / args.final_table).fillna("")
    for col in ["finalPriorityScore", "finalRankGlobal", "finalRankWithinDirection"]:
        df[col] = as_number(df[col])
    audited = add_concordance_flags(df).sort_values("finalPriorityScore", ascending=False)

    topk = topk_composition(audited)
    groups = group_composition(audited)
    topk_lookup = {str(row["cutoff"]): row for row in topk if row["groupType"] == "all"}

    high_shortlist = select_columns(
        audited[audited["highConcordanceFlag"]].sort_values("finalPriorityScore", ascending=False).head(args.review_limit)
    )
    single_review = select_columns(
        audited[audited["singleEvidenceDominatedFlag"]].sort_values("finalPriorityScore", ascending=False).head(args.review_limit)
    )
    audit_table = select_columns(audited)

    total = len(audited)
    multi = int(audited["multiEvidenceFlag"].sum())
    high = int(audited["highConcordanceFlag"].sum())
    single = int(audited["singleEvidenceDominatedFlag"].sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(total),
        "multiEvidenceRows": multi,
        "multiEvidencePct": round(pct(multi, total), 4),
        "highConcordanceRows": high,
        "highConcordancePct": round(pct(high, total), 4),
        "singleEvidenceDominatedRows": single,
        "singleEvidenceDominatedPct": round(pct(single, total), 4),
        "meanEvidenceSupportCount": round(float(audited["evidenceSupportCount"].mean()), 4),
        "meanStrongEvidenceCount": round(float(audited["strongEvidenceCount"].mean()), 4),
        "concordanceTierCounts": dict(Counter(audited["evidenceConcordanceTier"])),
        "concordanceClassCounts": dict(Counter(audited["evidenceConcordanceClass"])),
        "topK": topk_lookup,
        "highConcordanceShortlistRows": int(len(high_shortlist)),
        "singleEvidenceReviewRows": int(len(single_review)),
        "methodNote": (
            "Evidence support is counted across model, disease, KG, ADMET, structure, and direction-label layers. "
            "Risk-clean status is handled separately so high evidence does not hide safety, structure, or hard-flag concerns."
        ),
        "outputs": {
            "auditTable": str((out_dir / "final_priority_evidence_concordance_audit.csv").resolve()),
            "topkComposition": str((out_dir / "final_priority_evidence_concordance_topk.csv").resolve()),
            "groupComposition": str((out_dir / "final_priority_evidence_concordance_groups.csv").resolve()),
            "highConcordanceShortlist": str((out_dir / "final_priority_high_concordance_shortlist.csv").resolve()),
            "singleEvidenceReview": str((out_dir / "final_priority_single_evidence_review.csv").resolve()),
            "summary": str((out_dir / "final_priority_evidence_concordance_summary.json").resolve()),
            "markdown": str((out_dir / "FINAL_PRIORITY_EVIDENCE_CONCORDANCE_AUDIT.md").resolve()),
        },
    }

    audit_table.to_csv(out_dir / "final_priority_evidence_concordance_audit.csv", index=False)
    write_csv(out_dir / "final_priority_evidence_concordance_topk.csv", topk)
    write_csv(out_dir / "final_priority_evidence_concordance_groups.csv", groups)
    high_shortlist.to_csv(out_dir / "final_priority_high_concordance_shortlist.csv", index=False)
    single_review.to_csv(out_dir / "final_priority_single_evidence_review.csv", index=False)
    write_json(out_dir / "final_priority_evidence_concordance_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_EVIDENCE_CONCORDANCE_AUDIT.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

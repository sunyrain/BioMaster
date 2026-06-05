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


def as_bool(series: pd.Series) -> pd.Series:
    return as_number(series).astype(int).ne(0)


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


def add_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["knownDrugTargetPairFlag"] = as_bool(out["knownDrugTargetPair"])
    out["directKgDrugTargetFlag"] = as_bool(out["hasDirectDrugTargetEdge"])
    out["positiveDrugDiseaseFlag"] = as_bool(out["hasPositiveDrugDiseaseEdge"])
    out["contraindicationFlag"] = as_bool(out["hasContraindicationDiseaseEdge"])
    out["directTargetDiseaseFlag"] = as_bool(out["hasDirectTargetDiseaseEdge"])
    out["ppiBridgeFlag"] = as_number(out["ppiDiseaseBridgeCount"]).gt(0)
    out["drugTargetDiseaseBridgeFlag"] = as_number(out["drugTargetDiseaseBridgeCount"]).gt(0)
    out["completedStructureFlag"] = out["status"].astype(str).eq("completed")
    out["posePassFlag"] = out["poseAuditStatus"].astype(str).eq("pass")
    out["abTierFlag"] = out["finalPriorityTier"].astype(str).isin(["A", "B"])
    out["directKnownMechanismFlag"] = out["knownDrugTargetPairFlag"] | out["directKgDrugTargetFlag"]
    out["anyDiseaseContextFlag"] = (
        out["positiveDrugDiseaseFlag"]
        | out["directTargetDiseaseFlag"]
        | out["ppiBridgeFlag"]
        | out["drugTargetDiseaseBridgeFlag"]
    )
    out["networkDiseaseSupportFlag"] = (
        out["directTargetDiseaseFlag"] | out["ppiBridgeFlag"] | out["drugTargetDiseaseBridgeFlag"]
    )
    out["mechanismExtensionFlag"] = (
        ~out["directKnownMechanismFlag"]
        & out["positiveDrugDiseaseFlag"]
        & ~out["contraindicationFlag"]
    )
    out["strictNovelPairFlag"] = (
        ~out["directKnownMechanismFlag"]
        & ~out["positiveDrugDiseaseFlag"]
        & ~out["contraindicationFlag"]
        & out["networkDiseaseSupportFlag"]
    )
    out["strictNovelCompletedFlag"] = out["strictNovelPairFlag"] & out["completedStructureFlag"]
    out["strictNovelAbFlag"] = out["strictNovelCompletedFlag"] & out["abTierFlag"]
    out["evidenceSparseFlag"] = (
        ~out["directKnownMechanismFlag"]
        & ~out["positiveDrugDiseaseFlag"]
        & ~out["contraindicationFlag"]
        & ~out["networkDiseaseSupportFlag"]
    )

    classes: list[str] = []
    for _, row in out.iterrows():
        if row["knownDrugTargetPairFlag"]:
            classes.append("known_benchmark_drug_target_pair")
        elif row["directKgDrugTargetFlag"]:
            classes.append("known_kg_drug_target_mechanism")
        elif row["contraindicationFlag"]:
            classes.append("safety_or_contraindication_context")
        elif row["positiveDrugDiseaseFlag"]:
            classes.append("known_disease_use_new_target_hypothesis")
        elif row["directTargetDiseaseFlag"] and (row["ppiBridgeFlag"] or row["drugTargetDiseaseBridgeFlag"]):
            classes.append("target_disease_network_supported_new_pair")
        elif row["directTargetDiseaseFlag"]:
            classes.append("target_disease_supported_new_pair")
        elif row["ppiBridgeFlag"] or row["drugTargetDiseaseBridgeFlag"]:
            classes.append("network_supported_new_pair")
        else:
            classes.append("model_priority_sparse_external_context")
    out["auditNoveltyClass"] = classes
    return out


def composition_row(df: pd.DataFrame, group_type: str, group_value: str, cutoff: int | str = "all") -> dict[str, Any]:
    total = len(df)
    known_pair = int(df["knownDrugTargetPairFlag"].sum())
    direct_kg = int(df["directKgDrugTargetFlag"].sum())
    known_mechanism = int(df["directKnownMechanismFlag"].sum())
    mechanism_extension = int(df["mechanismExtensionFlag"].sum())
    strict_novel = int(df["strictNovelPairFlag"].sum())
    strict_novel_ab = int(df["strictNovelAbFlag"].sum())
    safety = int(df["contraindicationFlag"].sum())
    sparse = int(df["evidenceSparseFlag"].sum())
    completed = int(df["completedStructureFlag"].sum())
    pose_pass = int(df["posePassFlag"].sum())
    return {
        "groupType": group_type,
        "groupValue": group_value,
        "cutoff": cutoff,
        "rows": total,
        "knownBenchmarkPairRows": known_pair,
        "knownBenchmarkPairPct": round(pct(known_pair, total), 4),
        "directKgDrugTargetRows": direct_kg,
        "directKnownMechanismRows": known_mechanism,
        "directKnownMechanismPct": round(pct(known_mechanism, total), 4),
        "knownDiseaseUseNewTargetRows": mechanism_extension,
        "knownDiseaseUseNewTargetPct": round(pct(mechanism_extension, total), 4),
        "strictNovelPairRows": strict_novel,
        "strictNovelPairPct": round(pct(strict_novel, total), 4),
        "strictNovelAbRows": strict_novel_ab,
        "strictNovelAbPct": round(pct(strict_novel_ab, total), 4),
        "safetyOrContraindicationRows": safety,
        "safetyOrContraindicationPct": round(pct(safety, total), 4),
        "evidenceSparseRows": sparse,
        "evidenceSparsePct": round(pct(sparse, total), 4),
        "completedStructureRows": completed,
        "completedStructurePct": round(pct(completed, total), 4),
        "posePassRows": pose_pass,
        "posePassPct": round(pct(pose_pass, total), 4),
        "uniqueDrugs": int(df["drug"].nunique()) if total else 0,
        "uniqueTargets": int(df["target"].nunique()) if total else 0,
        "auditNoveltyClassCounts": dict(Counter(df["auditNoveltyClass"])),
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
        ("auditNoveltyClass", "auditNoveltyClass"),
        ("noveltyClass", "originalNoveltyClass"),
        ("reviewTrack", "reviewTrack"),
        ("finalPriorityTier", "finalPriorityTier"),
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
        "auditNoveltyClass",
        "noveltyClass",
        "knownDrugTargetPairFlag",
        "directKgDrugTargetFlag",
        "positiveDrugDiseaseFlag",
        "contraindicationFlag",
        "directTargetDiseaseFlag",
        "ppiDiseaseBridgeCount",
        "drugTargetDiseaseBridgeCount",
        "strictNovelPairFlag",
        "mechanismExtensionFlag",
        "evidenceSparseFlag",
        "status",
        "poseAuditStatus",
        "admetTier",
        "directionScore",
        "affinityScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "kgEvidenceScore",
        "diffdock",
        "therapeuticArea",
        "indication",
        "targetDiseaseExamples",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "validationGatesZh",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    return df[[col for col in cols if col in df.columns]]


def per_direction_shortlist(df: pd.DataFrame, mask: pd.Series, per_direction: int, global_limit: int) -> pd.DataFrame:
    selected = (
        df[mask]
        .sort_values(["direction", "finalPriorityScore", "finalRankGlobal"], ascending=[True, False, True])
        .groupby("direction", dropna=False)
        .head(per_direction)
        .sort_values("finalPriorityScore", ascending=False)
        .head(global_limit)
    )
    return select_columns(selected)


def build_markdown(summary: dict[str, Any]) -> str:
    top100 = summary["topK"]["100"]
    strict = summary["strictNovel"]
    lines = [
        "# Novelty and Leakage Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit separates known mechanism recovery from novel pair prioritization.",
        "",
        "## Headline",
        "",
        f"- Final table rows: {summary['candidateRows']}; known benchmark drug-target pairs: {summary['knownBenchmarkPairRows']}.",
        f"- Direct known mechanism rows, including benchmark positives and KG drug-target edges: {summary['directKnownMechanismRows']} ({summary['directKnownMechanismPct']:.2f}%).",
        f"- Strict novel pair rows: {strict['rows']} ({strict['pct']:.2f}%); strict novel A/B rows: {strict['abRows']}.",
        f"- Top100 composition: known mechanism {top100['directKnownMechanismRows']} ({top100['directKnownMechanismPct']:.2f}%), strict novel {top100['strictNovelPairRows']} ({top100['strictNovelPairPct']:.2f}%), known disease-use/new-target {top100['knownDiseaseUseNewTargetRows']} ({top100['knownDiseaseUseNewTargetPct']:.2f}%).",
        f"- Top100 safety/contraindication context rows: {top100['safetyOrContraindicationRows']} ({top100['safetyOrContraindicationPct']:.2f}%).",
        "",
        "## Interpretation",
        "",
        "Known mechanism rows are useful positive controls, but they should not be presented as novel discovery. The strict novel shortlist removes direct drug-target recovery, known drug-disease use, and contraindication context, while retaining disease/network support.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit novelty and known-mechanism leakage in the final priority table.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--per-direction", type=int, default=25)
    parser.add_argument("--global-limit", type=int, default=150)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    df = pd.read_csv(root / args.final_table).fillna("")
    df["finalPriorityScore"] = as_number(df["finalPriorityScore"])
    df["finalRankGlobal"] = as_number(df["finalRankGlobal"]).astype(int)
    audited = add_audit_flags(df)

    topk_rows = topk_composition(audited)
    group_rows = group_composition(audited)
    topk_by_cutoff = {str(row["cutoff"]): row for row in topk_rows if row["groupType"] == "all"}

    strict_mask = audited["strictNovelPairFlag"] & audited["completedStructureFlag"] & audited["abTierFlag"]
    extension_mask = audited["mechanismExtensionFlag"] & audited["completedStructureFlag"] & audited["abTierFlag"]
    positive_control_mask = audited["directKnownMechanismFlag"]
    safety_mask = audited["contraindicationFlag"] | audited["reviewTrack"].astype(str).str.contains("safety", case=False, na=False)

    audit_table = select_columns(audited.sort_values("finalPriorityScore", ascending=False))
    strict_shortlist = per_direction_shortlist(audited, strict_mask, args.per_direction, args.global_limit)
    extension_shortlist = per_direction_shortlist(audited, extension_mask, args.per_direction, args.global_limit)
    positive_controls = select_columns(
        audited[positive_control_mask].sort_values("finalPriorityScore", ascending=False).head(args.global_limit)
    )
    safety_review = select_columns(
        audited[safety_mask].sort_values("finalPriorityScore", ascending=False).head(args.global_limit)
    )

    known = int(audited["knownDrugTargetPairFlag"].sum())
    direct_known = int(audited["directKnownMechanismFlag"].sum())
    strict_rows = int(audited["strictNovelPairFlag"].sum())
    strict_ab = int(audited["strictNovelAbFlag"].sum())
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(audited)),
        "knownBenchmarkPairRows": known,
        "directKgDrugTargetRows": int(audited["directKgDrugTargetFlag"].sum()),
        "directKnownMechanismRows": direct_known,
        "directKnownMechanismPct": round(pct(direct_known, len(audited)), 4),
        "knownDiseaseUseNewTargetRows": int(audited["mechanismExtensionFlag"].sum()),
        "knownDiseaseUseNewTargetPct": round(pct(int(audited["mechanismExtensionFlag"].sum()), len(audited)), 4),
        "strictNovel": {
            "rows": strict_rows,
            "pct": round(pct(strict_rows, len(audited)), 4),
            "completedRows": int(audited["strictNovelCompletedFlag"].sum()),
            "abRows": strict_ab,
            "shortlistRows": int(len(strict_shortlist)),
        },
        "safetyOrContraindicationRows": int(audited["contraindicationFlag"].sum()),
        "evidenceSparseRows": int(audited["evidenceSparseFlag"].sum()),
        "topK": topk_by_cutoff,
        "auditNoveltyClassCounts": dict(Counter(audited["auditNoveltyClass"])),
        "methodNote": (
            "Strict novel pairs exclude known benchmark drug-target pairs, direct KG drug-target edges, positive drug-disease use, "
            "and contraindication context, while requiring direct target-disease or network disease support. This is a leakage audit, "
            "not proof that remaining pairs are biologically novel in the literature."
        ),
        "outputs": {
            "auditTable": str((out_dir / "final_priority_novelty_leakage_audit.csv").resolve()),
            "topkComposition": str((out_dir / "final_priority_novelty_topk_composition.csv").resolve()),
            "groupComposition": str((out_dir / "final_priority_novelty_group_composition.csv").resolve()),
            "strictNovelShortlist": str((out_dir / "final_priority_strict_novel_shortlist.csv").resolve()),
            "mechanismExtensionShortlist": str((out_dir / "final_priority_mechanism_extension_shortlist.csv").resolve()),
            "positiveControls": str((out_dir / "final_priority_known_mechanism_positive_controls.csv").resolve()),
            "safetyReview": str((out_dir / "final_priority_safety_context_review.csv").resolve()),
            "summary": str((out_dir / "final_priority_novelty_leakage_summary.json").resolve()),
            "markdown": str((out_dir / "FINAL_PRIORITY_NOVELTY_LEAKAGE_AUDIT.md").resolve()),
        },
    }

    audit_table.to_csv(out_dir / "final_priority_novelty_leakage_audit.csv", index=False)
    write_csv(out_dir / "final_priority_novelty_topk_composition.csv", topk_rows)
    write_csv(out_dir / "final_priority_novelty_group_composition.csv", group_rows)
    strict_shortlist.to_csv(out_dir / "final_priority_strict_novel_shortlist.csv", index=False)
    extension_shortlist.to_csv(out_dir / "final_priority_mechanism_extension_shortlist.csv", index=False)
    positive_controls.to_csv(out_dir / "final_priority_known_mechanism_positive_controls.csv", index=False)
    safety_review.to_csv(out_dir / "final_priority_safety_context_review.csv", index=False)
    write_json(out_dir / "final_priority_novelty_leakage_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_NOVELTY_LEAKAGE_AUDIT.md").write_text(build_markdown(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

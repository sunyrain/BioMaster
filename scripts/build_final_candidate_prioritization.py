from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}

NOVELTY_LABEL_ZH = {
    "known_mechanism_or_known_disease_use": "已知机制/已知疾病用途召回",
    "known_drug_target_mechanism": "已知药物-靶点机制",
    "known_disease_use_with_predicted_target": "已知疾病用途下的新靶点机制",
    "disease_context_supported_new_pair": "疾病图谱支撑的新组合",
    "known_negative_or_safety_context": "禁忌/安全性上下文",
    "model_priority_without_txgnn_kg_path": "模型优先但 KG 浅层路径不足",
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


def bounded(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def bool_int(value: Any) -> int:
    if value in (True, 1, "1", "true", "True", "yes", "YES"):
        return 1
    return 0


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


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def minmax_score(series: pd.Series, default: float = 50.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([default] * len(series), index=series.index)
    lo = numeric.min()
    hi = numeric.max()
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return pd.Series([default] * len(series), index=series.index)
    return ((numeric - lo) / (hi - lo) * 100).fillna(default)


def pose_geometry_score(status: str, reason: str) -> float:
    status = (status or "").strip()
    reason = (reason or "").strip()
    if status == "pass":
        return 100.0
    if status == "warning":
        if reason == "possible_ligand_receptor_clash":
            return 72.0
        if reason == "severe_ligand_receptor_clash":
            return 45.0
        if reason == "no_receptor_contact_within_4A":
            return 55.0
        if reason == "ligand_coordinate_span_unusually_large":
            return 55.0
        return 60.0
    if status == "fail":
        return 0.0
    if status == "not_applicable":
        return 25.0
    return 40.0


def structural_score(row: pd.Series) -> float:
    if row.get("status", "") != "completed":
        return 20.0
    return pose_geometry_score(row.get("poseAuditStatus", ""), row.get("poseAuditReason", ""))


def flag_text(row: pd.Series) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    if row.get("status", "") != "completed":
        hard.append("structure_missing")
    if row.get("admetTier", "") == "D":
        hard.append("admet_tier_D")
    if bool_int(row.get("painsAlerts")):
        hard.append("pains_alert")
    if bool_int(row.get("diagnosticProcedureTextFlag")):
        hard.append("diagnostic_or_procedure_like")
    if bool_int(row.get("hasContraindicationDiseaseEdge")):
        soft.append("txgnn_contraindication_context")
    if row.get("admetTier", "") == "C":
        soft.append("admet_tier_C")
    if bool_int(row.get("cypDdiTextFlag")):
        soft.append("cyp_or_transporter_ddi_text")
    if str(row.get("poseAuditStatus", "")) == "fail":
        hard.append("pose_sanity_fail")
    elif str(row.get("poseAuditStatus", "")) == "warning":
        soft.append("pose_sanity_warning:" + str(row.get("poseAuditReason", "")))
    if row.get("routeClass", "") == "local":
        soft.append("local_route")
    if row.get("noveltyClass", "") == "model_priority_without_txgnn_kg_path":
        soft.append("kg_path_sparse")
    return hard, soft


def review_track(row: pd.Series, hard_flags: list[str], soft_flags: list[str]) -> str:
    if hard_flags:
        return "D_deprioritize_until_issue_resolved"
    if "txgnn_contraindication_context" in soft_flags or row.get("noveltyClass") == "known_negative_or_safety_context":
        return "C_safety_or_contraindication_review"
    if bool_int(row.get("knownDrugTargetPair")) or row.get("noveltyClass") == "known_mechanism_or_known_disease_use":
        return "A_positive_control_or_known_mechanism"
    if row.get("noveltyClass") == "known_disease_use_with_predicted_target":
        return "A_repurposing_mechanism_review"
    if row.get("noveltyClass") == "disease_context_supported_new_pair":
        return "B_novel_pair_disease_context_review"
    if row.get("kgEvidenceTier", "") in {"A", "B"}:
        return "B_mechanism_review"
    return "C_secondary_model_priority"


def priority_tier(score: float, hard_flags: list[str], soft_flags: list[str]) -> str:
    if hard_flags:
        return "D"
    if score >= 82 and "txgnn_contraindication_context" not in soft_flags:
        return "A"
    if score >= 68:
        return "B"
    if score >= 52:
        return "C"
    return "D"


def score_row(row: pd.Series) -> dict[str, Any]:
    direction = number(row.get("directionScore")) or 0.0
    affinity = number(row.get("affinityScore")) or 0.0
    credibility = number(row.get("credibilityScore")) or 0.0
    kg = number(row.get("kgEvidenceScore")) or 0.0
    admet = number(row.get("admetScore")) or 0.0
    open_targets = number(row.get("openTargetsScore")) or 0.0
    txgnn = number(row.get("integratedTxgnnScore")) or number(row.get("txgnnScore")) or 0.0
    pose = structural_score(row)

    model_component = (direction * 100 + affinity * 100) / 2
    disease_component = 0.60 * direction * 100 + 0.25 * open_targets * 100 + 0.15 * txgnn * 100
    translation_component = (
        0.24 * model_component
        + 0.18 * disease_component
        + 0.14 * credibility
        + 0.16 * kg
        + 0.14 * admet
        + 0.10 * pose
        + 0.04 * (100 if bool_int(row.get("directionLabelFit")) else 55)
    )

    hard_flags, soft_flags = flag_text(row)
    penalty = 0.0
    if "txgnn_contraindication_context" in soft_flags:
        penalty += 18
    if "admet_tier_C" in soft_flags:
        penalty += 8
    if row.get("admetTier", "") == "B":
        penalty += 2
    if any(flag.startswith("pose_sanity_warning:severe") for flag in soft_flags):
        penalty += 10
    elif any(flag.startswith("pose_sanity_warning") for flag in soft_flags):
        penalty += 4
    if "local_route" in soft_flags:
        penalty += 3
    if hard_flags:
        penalty += 35

    final_score = bounded(translation_component - penalty)
    track = review_track(row, hard_flags, soft_flags)
    tier = priority_tier(final_score, hard_flags, soft_flags)
    return {
        "modelComponent": round(model_component, 3),
        "diseaseEvidenceComponent": round(disease_component, 3),
        "structureGeometryScore": round(pose, 3),
        "rawIntegratedScore": round(translation_component, 3),
        "riskPenalty": round(penalty, 3),
        "finalPriorityScore": round(final_score, 3),
        "finalPriorityTier": tier,
        "reviewTrack": track,
        "hardFlags": "; ".join(hard_flags) if hard_flags else "none",
        "softFlags": "; ".join(soft_flags) if soft_flags else "none",
    }


def load_known_pairs(path: Path) -> pd.DataFrame:
    known = read_frame(path)
    if known.empty:
        return pd.DataFrame(columns=["pairId", "knownDrugTargetPair", "knownTargetRank", "knownTargetRecord", "knownTargetName"])
    known["rankNumeric"] = pd.to_numeric(known["rank"], errors="coerce")
    grouped = []
    for pair_id, group in known.groupby("pair_id"):
        best = group.sort_values("rankNumeric").iloc[0]
        grouped.append(
            {
                "pairId": pair_id,
                "knownDrugTargetPair": 1,
                "knownTargetRank": int(best["rankNumeric"]) if pd.notna(best["rankNumeric"]) else "",
                "knownTargetRecord": best.get("known_target_record", ""),
                "knownTargetName": best.get("target_name", ""),
                "knownTargetChemblId": best.get("target_chembl_id", ""),
            }
        )
    return pd.DataFrame(grouped)


def build_table(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    base = read_frame(root / args.admet_candidates)
    integrated_cols = [
        "direction",
        "pairId",
        "openTargetsScore",
        "txgnnScore",
        "scoreSource",
        "categoryZh",
        "credibilityTierZh",
        "evidenceSummaryZh",
        "rationaleZh",
        "nextStepZh",
        "validationGatesZh",
        "therapeuticArea",
        "indication",
        "proteinName",
        "representedPairCount",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    integrated = read_frame(root / args.integrated_candidates)
    integrated = integrated[[col for col in integrated_cols if col in integrated.columns]].drop_duplicates(["direction", "pairId"])

    kg_cols = [
        "direction",
        "pairId",
        "txgnnDrugbankId",
        "txgnnDrugName",
        "txgnnDiseaseName",
        "openTargetsScore",
        "integratedTxgnnScore",
        "txgnnIndicationScore",
        "kgEvidenceScore",
        "kgEvidenceTier",
        "noveltyClass",
        "hasTxgnnDrugMapping",
        "hasDirectDrugTargetEdge",
        "hasPositiveDrugDiseaseEdge",
        "hasContraindicationDiseaseEdge",
        "hasDirectTargetDiseaseEdge",
        "ppiDiseaseBridgeCount",
        "drugTargetDiseaseBridgeCount",
        "pathCount",
        "directDrugTargetRelations",
        "drugDiseaseRelations",
        "targetDiseaseExamples",
        "ppiDiseaseBridgeGenes",
        "drugTargetDiseaseBridgeGenes",
        "kgExplanationZh",
    ]
    kg = read_frame(root / args.kg_summary)
    kg = kg[[col for col in kg_cols if col in kg.columns]].drop_duplicates(["direction", "pairId"])

    pose_cols = [
        "direction",
        "pairId",
        "poseAuditStatus",
        "poseAuditReason",
        "ligandReadStatus",
        "receptorReadStatus",
        "ligandHeavyAtomCount",
        "receptorResidueCount",
        "minHeavyAtomDistance",
        "medianNearestDistance",
        "ligandAtomsWithContact4A",
        "receptorAtomsWithin4A",
        "centroidDistanceToReceptor",
        "ligandSdfPath",
        "receptorPdbPath",
    ]
    pose = read_frame(root / args.pose_audit)
    pose = pose[[col for col in pose_cols if col in pose.columns]].drop_duplicates(["direction", "pairId"])

    known = load_known_pairs(root / args.known_pairs)

    df = base.merge(integrated, on=["direction", "pairId"], how="left", suffixes=("", "_integrated"))
    df = df.merge(kg, on=["direction", "pairId"], how="left", suffixes=("", "_kg"))
    df = df.merge(pose, on=["direction", "pairId"], how="left", suffixes=("", "_pose"))
    df = df.merge(known, on="pairId", how="left")
    df["knownDrugTargetPair"] = df["knownDrugTargetPair"].fillna(0).astype(int)
    df["knownTargetRank"] = df["knownTargetRank"].fillna("")

    for col in [
        "directionScore",
        "affinityScore",
        "diffdock",
        "credibilityScore",
        "admetScore",
        "translationalScore",
        "kgEvidenceScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "txgnnScore",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    score_rows = [score_row(row) for _, row in df.iterrows()]
    score_df = pd.DataFrame(score_rows)
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    df["noveltyLabelZh"] = df["noveltyClass"].map(NOVELTY_LABEL_ZH).fillna("未分类")
    df["directionLabelZhFinal"] = df["labelZh"].where(df["labelZh"].astype(str).str.len() > 0, df["direction"].map(DIRECTION_LABELS))

    df = df.sort_values(["direction", "finalPriorityScore", "rank"], ascending=[True, False, True]).copy()
    df["finalRankWithinDirection"] = df.groupby("direction").cumcount() + 1
    df = df.sort_values(["finalPriorityScore", "direction", "rank"], ascending=[False, True, True]).copy()
    df["finalRankGlobal"] = range(1, len(df) + 1)
    return df


def select_shortlists(df: pd.DataFrame, per_direction: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = df.sort_values(["direction", "finalPriorityScore", "rank"], ascending=[True, False, True]).groupby("direction").head(per_direction)
    novel_mask = (
        df["knownDrugTargetPair"].eq(0)
        & df["noveltyClass"].isin(["disease_context_supported_new_pair", "known_disease_use_with_predicted_target"])
        & df["hardFlags"].eq("none")
        & ~df["softFlags"].str.contains("txgnn_contraindication_context", na=False)
    )
    novel = df[novel_mask].sort_values(["direction", "finalPriorityScore", "rank"], ascending=[True, False, True]).groupby("direction").head(per_direction)
    controls = df[
        df["knownDrugTargetPair"].eq(1)
        | df["noveltyClass"].eq("known_mechanism_or_known_disease_use")
    ].sort_values(["direction", "finalPriorityScore", "rank"], ascending=[True, False, True]).groupby("direction").head(per_direction)
    caution = df[
        df["reviewTrack"].isin(["C_safety_or_contraindication_review", "D_deprioritize_until_issue_resolved"])
    ].sort_values(["direction", "finalPriorityScore", "rank"], ascending=[True, False, True]).groupby("direction").head(per_direction)
    return overall, novel, controls, caution


def output_columns() -> list[str]:
    return [
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "directionLabelZhFinal",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "knownTargetRank",
        "knownTargetRecord",
        "knownTargetName",
        "directionScore",
        "affinityScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "diffdock",
        "status",
        "credibilityScore",
        "admetScore",
        "admetTier",
        "routeClass",
        "directionLabelFit",
        "kgEvidenceScore",
        "kgEvidenceTier",
        "noveltyClass",
        "noveltyLabelZh",
        "pathCount",
        "hasDirectDrugTargetEdge",
        "hasPositiveDrugDiseaseEdge",
        "hasContraindicationDiseaseEdge",
        "hasDirectTargetDiseaseEdge",
        "ppiDiseaseBridgeCount",
        "drugTargetDiseaseBridgeCount",
        "poseAuditStatus",
        "poseAuditReason",
        "minHeavyAtomDistance",
        "ligandAtomsWithContact4A",
        "modelComponent",
        "diseaseEvidenceComponent",
        "structureGeometryScore",
        "rawIntegratedScore",
        "riskPenalty",
        "finalPriorityScore",
        "finalPriorityTier",
        "reviewTrack",
        "hardFlags",
        "softFlags",
        "therapeuticArea",
        "indication",
        "targetDiseaseExamples",
        "ppiDiseaseBridgeGenes",
        "drugTargetDiseaseBridgeGenes",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "validationGatesZh",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]


def summarize(df: pd.DataFrame, shortlists: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    overall, novel, controls, caution = shortlists
    by_direction: dict[str, dict[str, Any]] = {}
    for direction, group in df.groupby("direction"):
        by_direction[direction] = {
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "rows": int(len(group)),
            "tierCounts": dict(Counter(group["finalPriorityTier"])),
            "reviewTrackCounts": dict(Counter(group["reviewTrack"])),
            "medianFinalPriorityScore": float(group["finalPriorityScore"].median()),
            "knownDrugTargetRows": int(group["knownDrugTargetPair"].sum()),
            "novelReviewRows": int(
                (
                    group["knownDrugTargetPair"].eq(0)
                    & group["noveltyClass"].isin(["disease_context_supported_new_pair", "known_disease_use_with_predicted_target"])
                    & group["hardFlags"].eq("none")
                    & ~group["softFlags"].str.contains("txgnn_contraindication_context", na=False)
                ).sum()
            ),
            "safetyReviewRows": int(group["reviewTrack"].eq("C_safety_or_contraindication_review").sum()),
            "deprioritizedRows": int(group["reviewTrack"].eq("D_deprioritize_until_issue_resolved").sum()),
        }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "tierCounts": dict(Counter(df["finalPriorityTier"])),
        "reviewTrackCounts": dict(Counter(df["reviewTrack"])),
        "knownDrugTargetRows": int(df["knownDrugTargetPair"].sum()),
        "kgPathRows": int((pd.to_numeric(df["pathCount"], errors="coerce").fillna(0) > 0).sum()),
        "posePassRows": int(df["poseAuditStatus"].eq("pass").sum()),
        "poseWarningRows": int(df["poseAuditStatus"].eq("warning").sum()),
        "hardFlagRows": int(~df["hardFlags"].eq("none").sum()) if False else int(df["hardFlags"].ne("none").sum()),
        "softFlagRows": int(df["softFlags"].ne("none").sum()),
        "shortlistRows": {
            "overall": int(len(overall)),
            "novel": int(len(novel)),
            "knownControls": int(len(controls)),
            "caution": int(len(caution)),
        },
        "byDirection": by_direction,
        "scoreFormula": {
            "modelComponent": "mean(directionScore, affinityScore) * 100",
            "diseaseEvidenceComponent": "0.60*directionScore + 0.25*OpenTargets + 0.15*TxGNN, scaled to 0-100",
            "rawIntegratedScore": "0.24*model + 0.18*diseaseEvidence + 0.14*credibility + 0.16*KG + 0.14*ADMET + 0.10*poseGeometry + 0.04*direction-label-fit",
            "penalties": "contraindication, ADMET C/D, PAINS/diagnostic flags, pose warnings/fails, missing structure, local route",
            "interpretation": "Ranking score for expert triage, not an efficacy probability.",
        },
        "methodNote": "Final prioritization integrates existing evidence layers with transparent penalties. It separates positive controls, repurposing mechanisms, novel disease-context pairs, and safety/deprioritization cases.",
    }


def discussion_markdown(df: pd.DataFrame, shortlists: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], summary: dict[str, Any]) -> str:
    overall, novel, controls, caution = shortlists

    def line(row: pd.Series) -> str:
        known = "known target" if bool_int(row.get("knownDrugTargetPair")) else "not known target in benchmark"
        return (
            f"- {row['directionLabelZhFinal']} | {row['drug']} - {row['target']} ({row['protein']}): "
            f"score {row['finalPriorityScore']:.1f}, tier {row['finalPriorityTier']}, "
            f"{row['noveltyLabelZh']}, {known}. "
            f"KG={row.get('kgEvidenceScore', '')}, ADMET={row.get('admetTier', '')}/{row.get('admetScore', '')}, "
            f"pose={row.get('poseAuditStatus', '')}:{row.get('poseAuditReason', '')}. "
            f"{row.get('kgExplanationZh', '')}"
        )

    lines = [
        "# Final Candidate Prioritization for Expert Review",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This file is a triage aid. The final score is not an efficacy probability; it combines model ranking, disease evidence, KG explanation, ADMET feasibility, and structure sanity checks.",
        "",
        "## Overall",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Final tier counts: {summary['tierCounts']}",
        f"- Review tracks: {summary['reviewTrackCounts']}",
        "",
        "## Top Overall Candidates",
        "",
    ]
    for _, row in overall.sort_values(["direction", "finalPriorityScore"], ascending=[True, False]).groupby("direction").head(5).iterrows():
        lines.append(line(row))

    lines.extend(["", "## Mechanistic / Novel Review Candidates", ""])
    for _, row in novel.sort_values(["direction", "finalPriorityScore"], ascending=[True, False]).groupby("direction").head(5).iterrows():
        lines.append(line(row))

    lines.extend(["", "## Positive Controls / Known Mechanisms", ""])
    for _, row in controls.sort_values(["direction", "finalPriorityScore"], ascending=[True, False]).groupby("direction").head(4).iterrows():
        lines.append(line(row))

    lines.extend(["", "## Safety Or Deprioritization Cases", ""])
    for _, row in caution.sort_values(["direction", "finalPriorityScore"], ascending=[True, False]).groupby("direction").head(4).iterrows():
        lines.append(line(row))

    lines.extend(["", "## Direction-Level Counts", ""])
    for direction, item in summary["byDirection"].items():
        lines.append(
            f"- {item['labelZh']}: rows={item['rows']}, tiers={item['tierCounts']}, "
            f"novelReviewRows={item['novelReviewRows']}, safetyReviewRows={item['safetyReviewRows']}, "
            f"deprioritizedRows={item['deprioritizedRows']}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final multi-evidence candidate prioritization tables.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--admet-candidates", default="outputs/sota_validation/admet_repurposing/candidate_admet_repurposing_audit_topn.csv")
    parser.add_argument("--integrated-candidates", default="outputs/disease_directions/disease_direction_integrated_candidates.csv")
    parser.add_argument("--kg-summary", default="outputs/sota_validation/kg_explainability_top1000/candidate_kg_explanation_summary.csv")
    parser.add_argument("--pose-audit", default="outputs/sota_validation/pose_sanity_top10000/candidate_pose_sanity_audit.csv")
    parser.add_argument("--known-pairs", default="outputs/sota_validation/known_target_positive_pair_ranks.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--per-direction", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_table(root, args)
    shortlists = select_shortlists(df, args.per_direction)
    summary = summarize(df, shortlists)
    summary["inputs"] = {
        "admetCandidates": args.admet_candidates,
        "integratedCandidates": args.integrated_candidates,
        "kgSummary": args.kg_summary,
        "poseAudit": args.pose_audit,
        "knownPairs": args.known_pairs,
    }
    summary["outputs"] = {
        "finalTable": str((out_dir / "final_candidate_priority_table.csv").resolve()),
        "topOverall": str((out_dir / "final_candidate_priority_top20_per_direction.csv").resolve()),
        "topNovel": str((out_dir / "final_candidate_priority_novel_top20_per_direction.csv").resolve()),
        "knownControls": str((out_dir / "final_candidate_priority_known_controls.csv").resolve()),
        "cautionCases": str((out_dir / "final_candidate_priority_caution_cases.csv").resolve()),
        "discussionMarkdown": str((out_dir / "FINAL_CANDIDATE_EXPERT_REVIEW.md").resolve()),
        "summary": str((out_dir / "final_candidate_priority_summary.json").resolve()),
    }

    cols = [col for col in output_columns() if col in df.columns]
    df[cols].to_csv(out_dir / "final_candidate_priority_table.csv", index=False)
    names = [
        "final_candidate_priority_top20_per_direction.csv",
        "final_candidate_priority_novel_top20_per_direction.csv",
        "final_candidate_priority_known_controls.csv",
        "final_candidate_priority_caution_cases.csv",
    ]
    for frame, name in zip(shortlists, names):
        frame[cols].to_csv(out_dir / name, index=False)
    write_json(out_dir / "final_candidate_priority_summary.json", summary)
    (out_dir / "FINAL_CANDIDATE_EXPERT_REVIEW.md").write_text(discussion_markdown(df, shortlists, summary), encoding="utf-8")
    print(json.dumps({"summary": summary, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

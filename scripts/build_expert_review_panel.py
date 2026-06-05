from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DIRECTION_LABELS_ZH = {
    "oncology": "肿瘤",
    "cardiovascular": "心血管",
    "infectious_disease": "感染性疾病",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫炎症",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def bounded(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def add_prefixed_columns(base: pd.DataFrame, extra: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    if extra.empty or key not in extra.columns:
        return base
    use = [key] + [col for col in columns if col in extra.columns and col != key]
    if len(use) <= 1:
        return base
    return base.merge(extra[use].drop_duplicates(key), on=key, how="left")


def tier_score(value: Any, mapping: dict[str, float], default: float = 45.0) -> float:
    text = str(value or "")
    for prefix, score in mapping.items():
        if text.startswith(prefix):
            return score
    return default


def cmap_score(row: pd.Series) -> float:
    if not truthy(row.get("cmapMapped")):
        return 35.0
    tier = str(row.get("cmapReversalTier", ""))
    base = tier_score(
        tier,
        {
            "A_": 98.0,
            "B_": 84.0,
            "C_": 62.0,
            "D_": 30.0,
            "U_": 35.0,
        },
        45.0,
    )
    raw = num(row.get("cmapBestRawReversal"), 0.0)
    return bounded(base + min(8.0, max(-8.0, raw * 3.0)))


def pathway_score(row: pd.Series) -> float:
    tier = str(row.get("pathwayDiseaseContextTier", ""))
    base = tier_score(
        tier,
        {
            "A_": 95.0,
            "B_": 82.0,
            "C_": 60.0,
            "D_": 25.0,
        },
        45.0,
    )
    return max(base, num(row.get("pathwayDiseaseContextScore"), 0.0))


def tissue_score(row: pd.Series) -> float:
    score = max(num(row.get("sotaGtexContextScore"), 0.0), num(row.get("sotaContextScore"), 0.0), num(row.get("tissueContextScore"), 0.0))
    if score:
        return bounded(score)
    if truthy(row.get("gtexContextPositiveFlag")) or truthy(row.get("tissueContextPositiveFlag")):
        return 80.0
    return 45.0


def structure_score(row: pd.Series) -> float:
    scores = [
        num(row.get("standardPoseValidationScore"), 0.0),
        num(row.get("structureConfidenceScore"), 0.0),
        tier_score(str(row.get("structureConfidenceTier", "")), {"A_": 96.0, "B_": 82.0, "C_": 55.0, "D_": 20.0}, 0.0),
    ]
    score = max(scores)
    if str(row.get("poseAuditStatus", "")).lower() == "pass":
        score = max(score, 78.0)
    if str(row.get("standardPoseValidationTier", "")).startswith("D_"):
        score = min(score, 45.0)
    return bounded(score if score else 45.0)


def admet_score(row: pd.Series) -> float:
    score = max(
        num(row.get("sotaMlAdmetScore"), 0.0),
        num(row.get("mlAdmetCandidateSafetyScore"), 0.0),
        num(row.get("admetScore"), 0.0),
    )
    tier = str(row.get("admetTier", ""))
    tier_floor = {"A": 88.0, "B": 72.0, "C": 45.0, "D": 18.0}.get(tier, 45.0)
    return bounded(max(score, tier_floor))


def depmap_score(row: pd.Series) -> float:
    if str(row.get("direction")) != "oncology":
        return 65.0
    if truthy(row.get("depmapDependencyPositiveFlag")):
        return max(80.0, num(row.get("depmapDependencyScore"), 0.0))
    tier = str(row.get("depmapDependencyTier", ""))
    return tier_score(tier, {"A_": 95.0, "B_": 82.0, "C_": 55.0, "D_": 30.0}, 45.0)


def risk_penalty(row: pd.Series) -> float:
    penalty = 0.0
    if str(row.get("admetTier", "")) == "D":
        penalty += 14.0
    elif str(row.get("admetTier", "")) == "C":
        penalty += 6.0
    if truthy(row.get("contraindicationFlag")) or truthy(row.get("hasContraindicationDiseaseEdge")):
        penalty += 14.0
    if str(row.get("standardPoseValidationTier", "")).startswith("D_"):
        penalty += 8.0
    elif str(row.get("structureConfidenceTier", "")).startswith("D_"):
        penalty += 8.0
    if str(row.get("cmapReversalTier", "")).startswith("D_"):
        penalty += 5.0
    if "safety" in str(row.get("reviewTrack", "")).lower() or "contraindication" in str(row.get("sotaReadyAction", "")).lower():
        penalty += 8.0
    hard = str(row.get("hardFlags", "")).strip().lower()
    if hard and hard not in {"none", "nan"}:
        penalty += 10.0
    return penalty


def evidence_count(row: pd.Series) -> int:
    flags = [
        num(row.get("affinityScore"), 0.0) > 0.85 or num(row.get("modelReadinessScore"), 0.0) >= 75,
        num(row.get("openTargetsScore"), 0.0) > 0.5 or num(row.get("integratedTxgnnScore"), 0.0) > 0.5 or num(row.get("kgEvidenceScore"), 0.0) >= 70,
        pathway_score(row) >= 75,
        cmap_score(row) >= 70,
        tissue_score(row) >= 75,
        structure_score(row) >= 70,
        admet_score(row) >= 70,
        depmap_score(row) >= 80 if str(row.get("direction")) == "oncology" else False,
    ]
    return int(sum(bool(x) for x in flags))


def novelty_group(row: pd.Series) -> str:
    if truthy(row.get("knownDrugTargetPair")) or "known" in str(row.get("reviewTrack", "")).lower():
        return "positive_control_or_known_mechanism"
    if truthy(row.get("mechanismExtensionFlag")) or "mechanism_extension" in str(row.get("noveltyClass", "")):
        return "mechanism_extension"
    if truthy(row.get("strictNovelPairFlag")) or "new_pair" in str(row.get("noveltyClass", "")):
        return "novel_repurposing_candidate"
    if truthy(row.get("contraindicationFlag")) or "safety" in str(row.get("reviewTrack", "")).lower():
        return "risk_review"
    return "secondary_review"


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expertModelScore"] = out.apply(lambda row: max(num(row.get("sotaReadyScore"), 0.0), num(row.get("finalPriorityScore"), 0.0)), axis=1)
    out["expertDiseaseEvidenceScore"] = out.apply(
        lambda row: bounded(
            max(
                num(row.get("evidenceReadinessScore"), 0.0),
                100.0 * num(row.get("openTargetsScore"), 0.0),
                100.0 * num(row.get("integratedTxgnnScore"), 0.0),
                num(row.get("kgEvidenceScore"), 0.0),
            )
        ),
        axis=1,
    )
    out["expertPathwayScore"] = out.apply(pathway_score, axis=1)
    out["expertCmapScore"] = out.apply(cmap_score, axis=1)
    out["expertTissueScore"] = out.apply(tissue_score, axis=1)
    out["expertStructureScore"] = out.apply(structure_score, axis=1)
    out["expertAdmetScore"] = out.apply(admet_score, axis=1)
    out["expertDepmapScore"] = out.apply(depmap_score, axis=1)
    out["expertEvidenceSupportCount"] = out.apply(evidence_count, axis=1)
    out["expertRiskPenalty"] = out.apply(risk_penalty, axis=1)
    out["expertNoveltyGroup"] = out.apply(novelty_group, axis=1)
    out["expertReviewScore"] = (
        0.16 * out["expertModelScore"]
        + 0.16 * out["expertDiseaseEvidenceScore"]
        + 0.11 * out["expertPathwayScore"]
        + 0.12 * out["expertCmapScore"]
        + 0.11 * out["expertTissueScore"]
        + 0.14 * out["expertStructureScore"]
        + 0.10 * out["expertAdmetScore"]
        + 0.04 * out["expertDepmapScore"]
        + 0.06 * (out["expertEvidenceSupportCount"] / 8.0 * 100.0)
        - out["expertRiskPenalty"]
    ).map(lambda x: round(bounded(float(x)), 4))
    out["expertReviewTier"] = out["expertReviewScore"].map(
        lambda s: "A_expert_review_priority"
        if s >= 82
        else "B_review_ready"
        if s >= 70
        else "C_secondary_review"
        if s >= 55
        else "D_hold"
    )
    return out


def short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def build_reason(row: pd.Series) -> str:
    parts = []
    if num(row.get("affinityScore"), 0.0) > 0:
        parts.append(f"亲和/模型支持 {num(row.get('affinityScore')):.3f}")
    if num(row.get("openTargetsScore"), 0.0) > 0 or num(row.get("integratedTxgnnScore"), 0.0) > 0:
        parts.append(f"疾病证据 OT {num(row.get('openTargetsScore')):.2f}, TxGNN {num(row.get('integratedTxgnnScore')):.2f}")
    if str(row.get("pathwayDiseaseContextTier", "")).startswith(("A_", "B_")):
        parts.append("Reactome/GO/CREEDS 支持靶点疾病上下文")
    if str(row.get("cmapReversalTier", "")).startswith(("A_", "B_", "C_")) and truthy(row.get("cmapMapped")):
        parts.append(f"CMap {row.get('cmapReversalTier')}，最佳细胞 {row.get('cmapBestCell', '')}")
    if truthy(row.get("gtexContextPositiveFlag")) or truthy(row.get("tissueContextPositiveFlag")):
        parts.append("GTEx/HPA 相关组织表达支持")
    if str(row.get("direction")) == "oncology" and truthy(row.get("depmapDependencyPositiveFlag")):
        parts.append("DepMap 癌症依赖性支持")
    if structure_score(row) >= 70:
        parts.append("结构/口袋审计可解释")
    if admet_score(row) >= 70:
        parts.append(f"ADMET {row.get('admetTier', '')}，安全性可优先审阅")
    return "；".join(parts[:7])


def build_gap(row: pd.Series) -> str:
    gaps = []
    if not truthy(row.get("cmapMapped")):
        gaps.append("CMap 未覆盖该药物")
    elif str(row.get("cmapReversalTier", "")).startswith("D_"):
        gaps.append("CMap 未见正向反转")
    if str(row.get("standardPoseValidationTier", "")).startswith("D_") or str(row.get("structureConfidenceTier", "")).startswith("D_"):
        gaps.append("结构姿势需复核")
    if str(row.get("admetTier", "")) in {"C", "D"}:
        gaps.append("ADMET/毒性需人工审阅")
    if truthy(row.get("contraindicationFlag")) or truthy(row.get("hasContraindicationDiseaseEdge")):
        gaps.append("禁忌证/疾病安全上下文需核查")
    if str(row.get("direction")) == "oncology" and not truthy(row.get("depmapDependencyPositiveFlag")):
        gaps.append("DepMap 依赖性信号弱或缺失")
    if not gaps:
        gaps.append("补充 PubMed/ClinicalTrials 文献核查")
    return "；".join(gaps)


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expertRationaleZh"] = out.apply(build_reason, axis=1)
    out["expertReviewGapsZh"] = out.apply(build_gap, axis=1)
    out["expertDiscussionNoteZh"] = out.apply(
        lambda row: f"{row.get('drug')} - {row.get('target')} ({DIRECTION_LABELS_ZH.get(str(row.get('direction')), row.get('direction'))})：{row.get('expertRationaleZh')}。需关注：{row.get('expertReviewGapsZh')}。",
        axis=1,
    )
    return out


def diverse_select(
    df: pd.DataFrame,
    limit: int,
    per_direction_min: int,
    max_per_drug: int,
    max_per_target: int,
    max_per_scaffold: int,
) -> pd.DataFrame:
    ranked = df.sort_values(["expertReviewScore", "finalPriorityScore"], ascending=[False, False]).copy()
    selected: list[int] = []
    drug_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    scaffold_counts: Counter[str] = Counter()

    def can_take(row: pd.Series) -> bool:
        if drug_counts[str(row.get("drugId"))] >= max_per_drug:
            return False
        if target_counts[str(row.get("protein"))] >= max_per_target:
            return False
        scaffold = str(row.get("murckoScaffold", "") or "")
        if scaffold and scaffold_counts[scaffold] >= max_per_scaffold:
            return False
        return True

    def take(idx: int, row: pd.Series) -> None:
        selected.append(idx)
        drug_counts[str(row.get("drugId"))] += 1
        target_counts[str(row.get("protein"))] += 1
        scaffold = str(row.get("murckoScaffold", "") or "")
        if scaffold:
            scaffold_counts[scaffold] += 1

    for direction, group in ranked.groupby("direction", sort=False):
        count = 0
        for idx, row in group.iterrows():
            if idx in selected or not can_take(row):
                continue
            take(idx, row)
            count += 1
            if count >= per_direction_min:
                break

    for idx, row in ranked.iterrows():
        if len(selected) >= limit:
            break
        if idx in selected or not can_take(row):
            continue
        take(idx, row)

    return df.loc[selected].sort_values(["expertReviewScore", "finalPriorityScore"], ascending=[False, False]).reset_index(drop=True)


def load_and_merge(args: argparse.Namespace) -> pd.DataFrame:
    base = read_table(args.base)
    if base.empty:
        raise ValueError(f"Base table is empty or missing: {args.base}")
    cmap = read_table(args.cmap)
    pathway = read_table(args.pathway)
    ml_admet = read_table(args.ml_admet)
    std_pose = read_table(args.standard_pose)
    depmap = read_table(args.depmap)
    validation = read_table(args.validation)

    out = base.copy()
    out = add_prefixed_columns(
        out,
        cmap,
        "pairId",
        [
            "pairId",
            "cmapMapped",
            "cmapPerturbagenCount",
            "cmapSignatureCount",
            "cmapBestRawReversal",
            "cmapMedianRawReversal",
            "cmapTop5MeanRawReversal",
            "cmapPositiveSignaturePct",
            "cmapReversalPercentileWithinDirection",
            "cmapReversalScore",
            "cmapReversalTier",
            "cmapBestSigId",
            "cmapBestPertId",
            "cmapBestCmapName",
            "cmapBestCell",
            "cmapBestDose",
            "cmapBestDoseUnit",
            "cmapBestTime",
            "cmapBestTas",
            "cmapMatchTypes",
            "cmapReversalInterpretation",
        ],
    )
    out = add_prefixed_columns(
        out,
        pathway,
        "pairId",
        [
            "pairId",
            "reactomePathwayCount",
            "reactomeTopPathways",
            "goBpCount",
            "goMfCount",
            "goCcCount",
            "goTopBpIds",
            "goTopMfIds",
            "goTopCcIds",
            "creedsDirectionSignatureCount",
            "creedsTargetDirectionHit",
            "creedsUpSignatureCount",
            "creedsDownSignatureCount",
            "creedsTotalSignatureCount",
            "creedsMatchedDiseases",
            "pathwayAnnotationSupportScore",
            "diseaseSignatureSupportScore",
            "pathwayDiseaseContextScore",
            "pathwayDiseaseContextTier",
        ],
    )
    out = add_prefixed_columns(
        out,
        ml_admet,
        "pairId",
        [
            "pairId",
            "sotaMlAdmetScore",
            "sotaMlAdmetTier",
            "sotaMlAdmetAction",
            "mlAdmetCandidateSafetyScore",
            "mlAdmetSafetyTier",
            "mlAdmetRiskMean",
            "mlAdmetRiskMax",
            "mlAdmetHighRiskEndpointCount",
            "mlAdmetModerateRiskEndpointCount",
            "mlAdmetRiskFlags",
            "mlAdmetSafetyScore",
        ],
    )
    out = add_prefixed_columns(
        out,
        std_pose,
        "pairId",
        [
            "pairId",
            "standardPoseValidationScore",
            "standardPoseValidationTier",
            "standardPoseValidationAction",
            "standardPoseValidationReason",
            "posebustersPass",
            "posebustersFailedCheckCount",
            "posebustersCriticalFailedChecks",
            "prolifStatus",
            "prolifInteractionCount",
            "prolifUniqueResidueCount",
            "prolifInteractionTypes",
            "prolifTopInteractions",
            "minLigandReceptorDistance",
            "ligandContactCoverage4APct",
            "severeLigandReceptorClashPairs075A",
            "warningLigandReceptorClashPairs1A",
        ],
    )
    out = add_prefixed_columns(
        out,
        depmap,
        "pairId",
        [
            "pairId",
            "depmapMatchedGeneFlag",
            "depmapDependencyTier",
            "depmapDependencyPositiveFlag",
            "depmapDependencyScore",
            "depmapDependencyAdjustment",
            "depmapDependencyReason",
            "sotaDepmapOncologyScore",
            "sotaDepmapOncologyTier",
        ],
    )
    out = add_prefixed_columns(
        out,
        validation,
        "pairId",
        [
            "pairId",
            "validationScore",
            "validationTier",
            "validationGate",
            "assayModality",
            "assayRationale",
            "noveltyGroup",
            "validationSummary",
        ],
    )
    return out


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "expertReviewRank",
        "expertReviewScore",
        "expertReviewTier",
        "expertNoveltyGroup",
        "expertEvidenceSupportCount",
        "expertRiskPenalty",
        "direction",
        "directionLabelZhFinal",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "noveltyClass",
        "reviewTrack",
        "finalPriorityScore",
        "sotaContextScore",
        "affinityScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "kgEvidenceScore",
        "expertModelScore",
        "expertDiseaseEvidenceScore",
        "expertPathwayScore",
        "expertCmapScore",
        "expertTissueScore",
        "expertStructureScore",
        "expertAdmetScore",
        "expertDepmapScore",
        "cmapReversalTier",
        "cmapBestRawReversal",
        "cmapReversalScore",
        "cmapBestCell",
        "cmapSignatureCount",
        "pathwayDiseaseContextTier",
        "reactomeTopPathways",
        "creedsTargetDirectionHit",
        "creedsMatchedDiseases",
        "gtexContextTier",
        "gtexTopRelevantTissuesByTpm",
        "tissueContextTier",
        "topRelevantTissuesByNtpm",
        "depmapDependencyTier",
        "depmapDependencyPositiveFlag",
        "admetTier",
        "sotaMlAdmetTier",
        "mlAdmetRiskFlags",
        "structureConfidenceTier",
        "standardPoseValidationTier",
        "standardPoseValidationReason",
        "prolifInteractionTypes",
        "prolifTopInteractions",
        "poseAuditStatus",
        "diffdock",
        "contraindicationFlag",
        "hasContraindicationDiseaseEdge",
        "assayModality",
        "assayRationale",
        "expertRationaleZh",
        "expertReviewGapsZh",
        "expertDiscussionNoteZh",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    available = [col for col in cols if col in df.columns]
    return df[available]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any], top_rows: pd.DataFrame) -> None:
    lines = [
        "# Expert Review Panel",
        "",
        f"Created UTC: {summary['createdUtc']}",
        "",
        "## Scope",
        "",
        "The panel integrates affinity, disease evidence, pathways, CREEDS, LINCS/CMap reversal, tissue expression, DepMap, ADMET, contraindication flags, structural audits, and known-target/novelty labels.",
        "",
        "## Summary",
        "",
        f"- Candidate rows considered: {summary['candidateRows']}",
        f"- Expert panel rows: {summary['panelRows']}",
        f"- Wave-1 rows: {summary['wave1Rows']}",
        f"- Unique drugs in panel: {summary['panelUniqueDrugs']}",
        f"- Unique targets in panel: {summary['panelUniqueTargets']}",
        f"- Unique scaffolds in panel: {summary['panelUniqueScaffolds']}",
        "",
        "## Panel Composition",
        "",
    ]
    for key, value in summary["panelNoveltyGroupCounts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Disease Direction Counts", ""])
    for key, value in summary["panelDirectionCounts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Candidates", ""])
    for row in top_rows.head(12).itertuples(index=False):
        lines.append(
            f"- {row.drug} - {row.target} ({row.direction}), score {row.expertReviewScore}: {short_text(row.expertRationaleZh, 180)}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is an expert-review shortlist, not a final efficacy claim. Candidates with CMap or structural gaps are retained when other evidence layers are strong, but their gap fields should guide manual review and assay design.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an integrated 20-50 candidate expert review panel.")
    parser.add_argument("--base", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_gtex_context_matrix.csv"))
    parser.add_argument("--cmap", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_lincs_cmap_augmented_table.csv"))
    parser.add_argument("--pathway", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_pathway_disease_context_augmented_table.csv"))
    parser.add_argument("--ml-admet", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_ml_admet_matrix.csv"))
    parser.add_argument("--standard-pose", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_standard_pose_validation_matrix.csv"))
    parser.add_argument("--depmap", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_depmap_oncology_matrix.csv"))
    parser.add_argument("--validation", type=Path, default=Path("outputs/sota_validation/final_prioritization/final_priority_experimental_validation_panel.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/sota_validation/expert_review_panel"))
    parser.add_argument("--panel-size", type=int, default=50)
    parser.add_argument("--wave1-size", type=int, default=24)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    final_out = Path("outputs/sota_validation/final_prioritization")
    final_out.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge(args)
    scored = add_explanations(compute_scores(merged))
    scored = scored.sort_values(["expertReviewScore", "finalPriorityScore"], ascending=[False, False]).reset_index(drop=True)
    scored["expertReviewRankAllCandidates"] = range(1, len(scored) + 1)

    panel = diverse_select(
        scored,
        limit=args.panel_size,
        per_direction_min=5,
        max_per_drug=3,
        max_per_target=3,
        max_per_scaffold=3,
    )
    panel["expertReviewRank"] = range(1, len(panel) + 1)
    panel_selected = select_columns(panel)

    wave1_source = panel[
        panel["expertReviewTier"].isin(["A_expert_review_priority", "B_review_ready"])
        & ~panel["expertNoveltyGroup"].eq("risk_review")
    ].copy()
    wave1 = diverse_select(
        wave1_source if not wave1_source.empty else panel,
        limit=args.wave1_size,
        per_direction_min=3,
        max_per_drug=2,
        max_per_target=2,
        max_per_scaffold=2,
    )
    wave1["expertReviewRank"] = range(1, len(wave1) + 1)
    wave1_selected = select_columns(wave1)

    full_selected = select_columns(scored.assign(expertReviewRank=scored["expertReviewRankAllCandidates"]))

    full_selected.to_csv(args.outdir / "integrated_expert_review_scored_candidates.csv", index=False)
    panel_selected.to_csv(args.outdir / "integrated_expert_review_panel_top50.csv", index=False)
    wave1_selected.to_csv(args.outdir / "integrated_expert_review_wave1_24.csv", index=False)
    panel_selected.to_csv(final_out / "final_priority_integrated_expert_review_panel_top50.csv", index=False)
    wave1_selected.to_csv(final_out / "final_priority_integrated_expert_review_wave1_24.csv", index=False)

    summary = {
        "createdUtc": utc_now(),
        "candidateRows": int(len(scored)),
        "panelRows": int(len(panel)),
        "wave1Rows": int(len(wave1)),
        "panelUniqueDrugs": int(panel["drugId"].nunique()),
        "panelUniqueTargets": int(panel["protein"].nunique()),
        "panelUniqueScaffolds": int(panel.get("murckoScaffold", pd.Series(dtype=str)).fillna("").astype(str).replace("", pd.NA).nunique()),
        "panelDirectionCounts": panel["direction"].value_counts().to_dict(),
        "panelNoveltyGroupCounts": panel["expertNoveltyGroup"].value_counts().to_dict(),
        "panelTierCounts": panel["expertReviewTier"].value_counts().to_dict(),
        "panelCmapTierCounts": panel["cmapReversalTier"].fillna("NA").value_counts().to_dict(),
        "panelStructureTierCounts": panel["structureConfidenceTier"].fillna("NA").value_counts().to_dict(),
        "panelAdmetTierCounts": panel["admetTier"].fillna("NA").value_counts().to_dict(),
        "wave1DirectionCounts": wave1["direction"].value_counts().to_dict(),
        "wave1NoveltyGroupCounts": wave1["expertNoveltyGroup"].value_counts().to_dict(),
        "outputs": {
            "scoredCandidates": str(args.outdir / "integrated_expert_review_scored_candidates.csv"),
            "panelTop50": str(args.outdir / "integrated_expert_review_panel_top50.csv"),
            "wave1": str(args.outdir / "integrated_expert_review_wave1_24.csv"),
            "finalPanelTop50": str(final_out / "final_priority_integrated_expert_review_panel_top50.csv"),
            "finalWave1": str(final_out / "final_priority_integrated_expert_review_wave1_24.csv"),
        },
        "methodNote": "Expert review score combines model, disease evidence, pathway/CREEDS, LINCS/CMap reversal, tissue, structure, ADMET, oncology DepMap context, evidence count, and explicit risk penalties with diversity-constrained selection.",
    }
    write_summary(args.outdir / "integrated_expert_review_panel_summary.json", summary)
    write_summary(final_out / "final_priority_integrated_expert_review_panel_summary.json", summary)
    write_markdown(args.outdir / "INTEGRATED_EXPERT_REVIEW_PANEL.md", summary, panel_selected)
    write_markdown(final_out / "FINAL_PRIORITY_INTEGRATED_EXPERT_REVIEW_PANEL.md", summary, panel_selected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

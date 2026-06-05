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
from sklearn.metrics import average_precision_score, roc_auc_score


CUTOFFS = [20, 50, 100, 200, 500, 1000, 2000]
NOVEL_CLASSES = {"disease_context_supported_new_pair", "model_priority_without_txgnn_kg_path"}


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


def read_table(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).fillna("")


def merge_small(base: pd.DataFrame, other: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = ["direction", "pairId"] + [col for col in cols if col in other.columns]
    small = other[keep].drop_duplicates(["direction", "pairId"])
    return base.merge(small, how="left", on=["direction", "pairId"])


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    values = df[col]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def tier_to_score(value: Any, mapping: dict[str, float], default: float) -> float:
    text = str(value or "")
    for prefix, score in mapping.items():
        if text.startswith(prefix) or text == prefix:
            return score
    return default


def structure_score(row: pd.Series) -> float:
    numeric = number(row.get("structureConfidenceScore"))
    if numeric is not None:
        return max(0.0, min(100.0, numeric))
    return tier_to_score(
        row.get("structureConfidenceTier"),
        {
            "A_": 100.0,
            "B_": 85.0,
            "C_": 55.0,
            "D_": 25.0,
            "not_applicable": 35.0,
        },
        45.0,
    )


def evidence_score(row: pd.Series) -> float:
    tier = tier_to_score(
        row.get("evidenceConcordanceTier"),
        {"A": 100.0, "B": 85.0, "C": 60.0, "D": 25.0},
        50.0,
    )
    support = number(row.get("evidenceSupportCount")) or 0.0
    strong = number(row.get("strongEvidenceCount")) or 0.0
    return max(0.0, min(100.0, tier * 0.70 + min(support, 6.0) / 6.0 * 15.0 + min(strong, 5.0) / 5.0 * 15.0))


def direction_specificity_score(row: pd.Series) -> float:
    is_top = str(row.get("isPairTopDirection", "")).lower() in {"1", "true", "yes"}
    text = str(row.get("directionSpecificityClass", "")).lower()
    if "broad" in text:
        base = 70.0
    elif "single" in text or "direction" in text:
        base = 95.0
    elif text:
        base = 85.0
    else:
        base = 75.0
    if is_top:
        base += 5.0
    return max(0.0, min(100.0, base))


def chemotype_score(row: pd.Series) -> float:
    valid = str(row.get("smilesValid", "")).lower() in {"1", "true", "yes"}
    match_status = str(row.get("structureMatchStatus", "")).lower()
    cluster_size = number(row.get("chemotypeClusterSizeUniqueDrugs")) or 99.0
    nn = number(row.get("nearestNeighborSimilarity")) or 0.0
    if not valid:
        return 20.0
    match_score = 100.0 if "exact" in match_status else 90.0 if "base" in match_status else 50.0
    if cluster_size <= 3:
        cluster_score = 100.0
    elif cluster_size <= 6:
        cluster_score = 90.0
    elif cluster_size <= 12:
        cluster_score = 78.0
    else:
        cluster_score = 65.0
    similarity_score = 100.0 - max(0.0, min(1.0, nn)) * 20.0
    return max(0.0, min(100.0, 0.45 * match_score + 0.35 * cluster_score + 0.20 * similarity_score))


def risk_penalty(row: pd.Series) -> float:
    penalty = 0.0
    hard_flags = str(row.get("hardFlags", "")).strip().lower()
    if hard_flags and hard_flags != "none":
        penalty += 20.0
    review = str(row.get("reviewTrack", ""))
    novelty = str(row.get("auditNoveltyClass", row.get("noveltyClass", "")))
    structure_tier = str(row.get("structureConfidenceTier", ""))
    target_tier = str(row.get("targetDruggabilityTier", ""))
    if review.startswith("D_"):
        penalty += 20.0
    if "safety" in review or "negative_or_safety" in novelty:
        penalty += 12.0
    if bool(row.get("hasContraindicationDiseaseEdge")) or bool(row.get("contraindicationFlag")):
        penalty += 8.0
    if structure_tier.startswith("D_") or structure_tier == "fail":
        penalty += 12.0
    elif structure_tier.startswith("C_") or structure_tier == "not_applicable":
        penalty += 5.0
    if target_tier.startswith("D_"):
        penalty += 10.0
    elif target_tier.startswith("C_"):
        penalty += 4.0
    if str(row.get("singleEvidenceDominatedFlag", "")).lower() in {"1", "true", "yes"}:
        penalty += 10.0
    if str(row.get("isPairTopDirection", "")).lower() in {"0", "false", "no"}:
        penalty += 4.0
    return penalty


def bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def classify_action(row: pd.Series) -> tuple[str, str]:
    novelty = str(row.get("auditNoveltyClass", row.get("noveltyClass", "")))
    review = str(row.get("reviewTrack", ""))
    structure_tier = str(row.get("structureConfidenceTier", ""))
    target_tier = str(row.get("targetDruggabilityTier", ""))
    known_pair = bool(row.get("knownDrugTargetPair")) or bool(row.get("knownDrugTargetPairFlag"))
    direct_mechanism = bool(row.get("directKgDrugTargetFlag")) or bool(row.get("hasDirectDrugTargetEdge"))
    strict_novel = bool(row.get("strictNovelPairFlag")) or str(row.get("noveltyClass", "")) in NOVEL_CLASSES
    mechanism_extension = bool(row.get("mechanismExtensionFlag")) or "known_disease_use" in str(row.get("noveltyClass", ""))
    safety = "safety" in review or "negative_or_safety" in novelty or bool(row.get("contraindicationFlag"))
    if safety:
        return "safety_or_contraindication_review", "Check label safety, contraindication, and disease-context direction before any prioritization."
    if structure_tier.startswith("D_") or structure_tier == "fail":
        return "structure_low_confidence_review", "Docked pose falls in low-confidence or failed structural context; prioritize pocket/pose review."
    if target_tier.startswith("C_") or target_tier.startswith("D_"):
        return "target_context_review", "Target is druggable but needs clinical-stage, modality, or disease-context review."
    if known_pair or direct_mechanism:
        return "positive_control_or_known_mechanism", "Use as benchmark recall, known-mechanism support, or near-term repurposing control."
    if mechanism_extension:
        return "mechanism_extension_repurposing", "Known disease-use or mechanism-adjacent signal; review literature and clinical plausibility."
    if strict_novel:
        return "novel_pair_expert_review", "Novel drug-target/disease-context candidate; review biology, literature, and assay feasibility."
    if review.startswith("D_"):
        return "deprioritize_until_issue_resolved", "Do not advance until hard flags, missing evidence, or structural issues are resolved."
    return "secondary_expert_review", "Candidate has partial support and can be retained as a secondary review item."


def classify_tier(row: pd.Series) -> str:
    score = float(row.get("sotaReadyScore", 0.0))
    action = str(row.get("sotaReadyAction", ""))
    risk_clean = str(row.get("riskCleanFlag", "")).lower() in {"1", "true", "yes"}
    structure_good = str(row.get("structureConfidenceTier", "")).startswith(("A_", "B_"))
    target_good = str(row.get("targetDruggabilityTier", "")).startswith(("A_", "B_"))
    high_evidence = str(row.get("evidenceConcordanceTier", "")).startswith(("A", "B"))
    if action in {"safety_or_contraindication_review", "structure_low_confidence_review", "deprioritize_until_issue_resolved"}:
        return "D_blocked_or_requires_resolution"
    if score >= 88.0 and risk_clean and structure_good and target_good and high_evidence:
        return "A_sota_ready_expert_priority"
    if score >= 78.0 and structure_good and target_good:
        return "B_review_ready_priority"
    if score >= 65.0:
        return "C_context_or_secondary_review"
    return "D_low_priority_or_sparse_support"


def build_matrix(root: Path) -> pd.DataFrame:
    base = read_table(root, "outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    evidence = read_table(root, "outputs/sota_validation/final_prioritization/final_priority_evidence_concordance_audit.csv")
    novelty = read_table(root, "outputs/sota_validation/final_prioritization/final_priority_novelty_leakage_audit.csv")
    specificity = read_table(root, "outputs/sota_validation/final_prioritization/final_priority_direction_specificity_row_audit.csv")
    structure = read_table(root, "outputs/sota_validation/final_prioritization/final_priority_structure_confidence_audit.csv")
    target = read_table(root, "outputs/sota_validation/final_prioritization/final_priority_target_druggability_audit.csv")
    chemotype = read_table(root, "outputs/sota_validation/final_prioritization/final_priority_chemotype_diversity_audit.csv")

    df = base.copy()
    df = merge_small(
        df,
        evidence,
        [
            "evidenceConcordanceTier",
            "evidenceConcordanceClass",
            "evidenceSupportCount",
            "strongEvidenceCount",
            "supportFlagsText",
            "riskCleanFlag",
            "singleEvidenceDominatedFlag",
        ],
    )
    df = merge_small(
        df,
        novelty,
        [
            "auditNoveltyClass",
            "knownDrugTargetPairFlag",
            "directKgDrugTargetFlag",
            "positiveDrugDiseaseFlag",
            "contraindicationFlag",
            "strictNovelPairFlag",
            "mechanismExtensionFlag",
            "evidenceSparseFlag",
        ],
    )
    df = merge_small(
        df,
        specificity,
        [
            "nDirections",
            "topDirection",
            "topSecondMargin",
            "directionSpecificityClass",
            "isPairTopDirection",
            "scoreDeltaFromPairTop",
        ],
    )
    df = merge_small(
        df,
        structure,
        [
            "structureAdjustedRankGlobal",
            "structureAdjustedPriorityScore",
            "structureConfidenceTier",
            "structureConfidenceScore",
            "structureConfidencePenalty",
            "pocketResiduesWithin5A",
            "pocketMeanPlddt5A",
            "pocketLowPlddtResiduePct5A",
        ],
    )
    df = merge_small(
        df,
        target,
        [
            "targetAdjustedRankGlobal",
            "targetAdjustedPriorityScore",
            "targetDruggabilityScore",
            "targetDruggabilityTier",
            "targetDruggabilityReason",
            "catalogMax_Clinical_Phase",
            "catalogTarget_Class",
            "catalogDruggable_Modalities",
            "smallMoleculeModality",
            "targetDiseaseDirectionFit",
            "targetDiseaseDirectionMatchedIcdCodes",
        ],
    )
    df = merge_small(
        df,
        chemotype,
        [
            "structureMatchStatus",
            "smilesValid",
            "murckoScaffold",
            "chemotypeClusterId",
            "chemotypeClusterSizeUniqueDrugs",
            "nearestNeighborSimilarity",
            "nearestNeighborChemblId",
            "molecularWeight",
            "logP",
            "tpsa",
            "qed",
        ],
    )

    df["finalPriorityScore"] = num_col(df, "finalPriorityScore")
    df["modelReadinessScore"] = df["finalPriorityScore"].clip(0, 100)
    df["evidenceReadinessScore"] = df.apply(evidence_score, axis=1)
    df["structureReadinessScore"] = df.apply(structure_score, axis=1)
    df["targetReadinessScore"] = num_col(df, "targetDruggabilityScore", 50.0).clip(0, 100)
    df["chemotypeReadinessScore"] = df.apply(chemotype_score, axis=1)
    df["directionReadinessScore"] = df.apply(direction_specificity_score, axis=1)
    df["sotaRiskPenalty"] = df.apply(risk_penalty, axis=1)
    df["sotaReadyScore"] = (
        0.35 * df["modelReadinessScore"]
        + 0.20 * df["evidenceReadinessScore"]
        + 0.15 * df["structureReadinessScore"]
        + 0.15 * df["targetReadinessScore"]
        + 0.07 * df["chemotypeReadinessScore"]
        + 0.08 * df["directionReadinessScore"]
        - df["sotaRiskPenalty"]
    ).clip(0, 100).round(4)
    actions = df.apply(classify_action, axis=1)
    df["sotaReadyAction"] = [item[0] for item in actions]
    df["sotaReadyActionNote"] = [item[1] for item in actions]
    df["sotaReadyTier"] = df.apply(classify_tier, axis=1)
    df = df.sort_values("sotaReadyScore", ascending=False).reset_index(drop=True)
    df["sotaReadyRankGlobal"] = np.arange(1, len(df) + 1)
    df["sotaReadyRankWithinDirection"] = (
        df.groupby("direction")["sotaReadyScore"].rank(method="first", ascending=False).astype(int)
    )
    return df


def topk_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positives_total = int(num_col(df, "knownDrugTargetPair").sum())
    total = len(df)
    base_rate = positives_total / total if total else 0.0
    ranked = df.sort_values("sotaReadyScore", ascending=False).reset_index(drop=True)
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
                "rows": cutoff,
                "knownDrugTargetRows": hits,
                "knownDrugTargetPct": round(pct(hits, cutoff), 4),
                "recallKnownDrugTargetPct": round(pct(hits, positives_total), 4),
                "randomExpectedHits": round(expected, 4),
                "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
                "tierCounts": dict(Counter(top["sotaReadyTier"].astype(str))),
                "actionCounts": dict(Counter(top["sotaReadyAction"].astype(str))),
                "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()),
                "structureABRows": int(top["structureConfidenceTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                "targetABRows": int(top["targetDruggabilityTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                "uniqueDrugs": int(top["drugId"].nunique()),
                "uniqueTargets": int(top["protein"].nunique()),
                "uniqueScaffolds": int(top["murckoScaffold"].astype(str).replace("", np.nan).nunique(dropna=True)),
            }
        )
    for direction, group in df.groupby("direction"):
        ranked_dir = group.sort_values("sotaReadyScore", ascending=False).reset_index(drop=True)
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
                    "rows": cutoff,
                    "knownDrugTargetRows": hits,
                    "knownDrugTargetPct": round(pct(hits, cutoff), 4),
                    "recallKnownDrugTargetPct": round(pct(hits, positives_dir), 4),
                    "randomExpectedHits": round(expected, 4),
                    "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
                    "tierCounts": dict(Counter(top["sotaReadyTier"].astype(str))),
                    "actionCounts": dict(Counter(top["sotaReadyAction"].astype(str))),
                    "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()),
                    "structureABRows": int(top["structureConfidenceTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                    "targetABRows": int(top["targetDruggabilityTier"].astype(str).str.startswith(("A_", "B_")).sum()),
                    "uniqueDrugs": int(top["drugId"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()),
                    "uniqueScaffolds": int(top["murckoScaffold"].astype(str).replace("", np.nan).nunique(dropna=True)),
                }
            )
    return rows


def safe_auc_ap(df: pd.DataFrame, score_col: str) -> tuple[float | None, float | None]:
    labels = num_col(df, "knownDrugTargetPair").astype(int)
    scores = num_col(df, score_col)
    auc = float(roc_auc_score(labels, scores)) if labels.nunique() >= 2 else None
    ap = float(average_precision_score(labels, scores)) if int(labels.sum()) else None
    return auc, ap


def select_diverse(df: pd.DataFrame, per_direction: int = 50) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for direction, group in df.groupby("direction"):
        ranked = group.sort_values("sotaReadyScore", ascending=False)
        drug_counts: Counter[str] = Counter()
        scaffold_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        direction_rows: list[pd.Series] = []
        for _, row in ranked.iterrows():
            if len(direction_rows) >= per_direction:
                break
            if str(row.get("sotaReadyTier", "")).startswith("D_"):
                continue
            drug = str(row.get("drugId", ""))
            scaffold = str(row.get("murckoScaffold", "")) or "unmapped"
            target = str(row.get("protein", ""))
            if drug_counts[drug] >= 2 or scaffold_counts[scaffold] >= 4 or target_counts[target] >= 2:
                continue
            direction_rows.append(row)
            drug_counts[drug] += 1
            scaffold_counts[scaffold] += 1
            target_counts[target] += 1
        for _, row in ranked.iterrows():
            if len(direction_rows) >= per_direction:
                break
            if any(existing["pairId"] == row["pairId"] and existing["direction"] == row["direction"] for existing in direction_rows):
                continue
            if str(row.get("sotaReadyTier", "")).startswith("D_"):
                continue
            drug = str(row.get("drugId", ""))
            if drug_counts[drug] >= 3:
                continue
            direction_rows.append(row)
            drug_counts[drug] += 1
            scaffold_counts[str(row.get("murckoScaffold", "")) or "unmapped"] += 1
            target_counts[str(row.get("protein", ""))] += 1
        selected.extend(direction_rows)
    if not selected:
        return pd.DataFrame(columns=df.columns)
    result = pd.DataFrame(selected).sort_values(["direction", "sotaReadyScore"], ascending=[True, False]).copy()
    result["sotaReadyDiverseRankWithinDirection"] = result.groupby("direction").cumcount() + 1
    return result


def output_columns() -> list[str]:
    return [
        "sotaReadyRankGlobal",
        "sotaReadyRankWithinDirection",
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "directionLabelZhFinal",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "sotaReadyScore",
        "sotaReadyTier",
        "sotaReadyAction",
        "sotaReadyActionNote",
        "finalPriorityScore",
        "modelReadinessScore",
        "evidenceReadinessScore",
        "structureReadinessScore",
        "targetReadinessScore",
        "chemotypeReadinessScore",
        "directionReadinessScore",
        "sotaRiskPenalty",
        "finalPriorityTier",
        "reviewTrack",
        "noveltyClass",
        "auditNoveltyClass",
        "knownDrugTargetPair",
        "strictNovelPairFlag",
        "mechanismExtensionFlag",
        "evidenceConcordanceTier",
        "evidenceConcordanceClass",
        "evidenceSupportCount",
        "strongEvidenceCount",
        "riskCleanFlag",
        "supportFlagsText",
        "structureConfidenceTier",
        "structureConfidenceScore",
        "pocketMeanPlddt5A",
        "targetDruggabilityTier",
        "targetDruggabilityScore",
        "catalogMax_Clinical_Phase",
        "catalogTarget_Class",
        "catalogDruggable_Modalities",
        "smallMoleculeModality",
        "targetDiseaseDirectionFit",
        "murckoScaffold",
        "chemotypeClusterId",
        "chemotypeClusterSizeUniqueDrugs",
        "nearestNeighborSimilarity",
        "directionSpecificityClass",
        "isPairTopDirection",
        "nDirections",
        "admetTier",
        "kgEvidenceScore",
        "poseAuditStatus",
        "poseAuditReason",
        "openTargetsScore",
        "integratedTxgnnScore",
        "diffdock",
        "hardFlags",
        "softFlags",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "validationGatesZh",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]


def build_summary(df: pd.DataFrame, topk: list[dict[str, Any]], diverse: pd.DataFrame) -> dict[str, Any]:
    original_auc, original_ap = safe_auc_ap(df, "finalPriorityScore")
    sota_auc, sota_ap = safe_auc_ap(df, "sotaReadyScore")
    top_by_key = {(row["groupType"], row["groupValue"], row["cutoff"]): row for row in topk}
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(df)),
        "uniqueDrugs": int(df["drugId"].nunique()),
        "uniqueTargets": int(df["protein"].nunique()),
        "tierCounts": dict(Counter(df["sotaReadyTier"].astype(str))),
        "actionCounts": dict(Counter(df["sotaReadyAction"].astype(str))),
        "originalAuroc": original_auc,
        "sotaReadyAuroc": sota_auc,
        "originalAveragePrecision": original_ap,
        "sotaReadyAveragePrecision": sota_ap,
        "originalTop100KnownRows": int(num_col(df.sort_values("finalPriorityScore", ascending=False).head(100), "knownDrugTargetPair").sum()),
        "sotaReadyTop100KnownRows": top_by_key.get(("all", "all", 100), {}).get("knownDrugTargetRows"),
        "sotaReadyTop100KnownPct": top_by_key.get(("all", "all", 100), {}).get("knownDrugTargetPct"),
        "sotaReadyTop100NovelRows": top_by_key.get(("all", "all", 100), {}).get("novelRows"),
        "sotaReadyTop100StructureABRows": top_by_key.get(("all", "all", 100), {}).get("structureABRows"),
        "sotaReadyTop100TargetABRows": top_by_key.get(("all", "all", 100), {}).get("targetABRows"),
        "sotaReadyTop100UniqueDrugs": top_by_key.get(("all", "all", 100), {}).get("uniqueDrugs"),
        "sotaReadyTop100UniqueTargets": top_by_key.get(("all", "all", 100), {}).get("uniqueTargets"),
        "expertShortlistRows": int(min(300, len(df[df["sotaReadyTier"].astype(str).str.startswith(("A_", "B_"))]))),
        "novelShortlistRows": int(len(df[df["sotaReadyAction"].eq("novel_pair_expert_review")].head(300))),
        "diverseShortlistRows": int(len(diverse)),
        "diverseShortlistUniqueDrugs": int(diverse["drugId"].nunique()) if not diverse.empty else 0,
        "diverseShortlistUniqueTargets": int(diverse["protein"].nunique()) if not diverse.empty else 0,
        "diverseShortlistUniqueScaffolds": int(diverse["murckoScaffold"].nunique()) if not diverse.empty else 0,
        "methodNote": "SOTA-ready score integrates the existing final priority score with multi-evidence concordance, AlphaFold pocket pLDDT, target druggability, chemotype diversity, and direction specificity. It preserves known controls, mechanism-extension candidates, strict novel candidates, and review queues as separate action classes.",
    }


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final SOTA-Ready Decision Matrix",
            "",
            f"Generated: {summary.get('created_utc')}",
            "",
            "## Summary",
            "",
            f"- Candidate rows: {summary.get('candidateRows')}",
            f"- Unique drugs / targets: {summary.get('uniqueDrugs')} / {summary.get('uniqueTargets')}",
            f"- SOTA-ready tiers: {summary.get('tierCounts')}",
            f"- SOTA-ready actions: {summary.get('actionCounts')}",
            f"- Original AP vs SOTA-ready AP: {summary.get('originalAveragePrecision')} / {summary.get('sotaReadyAveragePrecision')}",
            f"- Original Top100 known rows vs SOTA-ready Top100 known rows: {summary.get('originalTop100KnownRows')} / {summary.get('sotaReadyTop100KnownRows')}",
            f"- SOTA-ready Top100 known pct / novel rows: {summary.get('sotaReadyTop100KnownPct')}% / {summary.get('sotaReadyTop100NovelRows')}",
            f"- SOTA-ready Top100 structure A/B rows: {summary.get('sotaReadyTop100StructureABRows')}",
            f"- SOTA-ready Top100 target A/B rows: {summary.get('sotaReadyTop100TargetABRows')}",
            f"- Diverse shortlist rows: {summary.get('diverseShortlistRows')}",
            "",
            "## Outputs",
            "",
            "- Full decision matrix: `outputs/sota_validation/final_prioritization/final_priority_sota_ready_matrix.csv`",
            "- Expert shortlist: `outputs/sota_validation/final_prioritization/final_priority_sota_ready_top300_expert_shortlist.csv`",
            "- Novel shortlist: `outputs/sota_validation/final_prioritization/final_priority_sota_ready_novel_shortlist.csv`",
            "- Positive controls: `outputs/sota_validation/final_prioritization/final_priority_sota_ready_positive_controls.csv`",
            "- Review queue: `outputs/sota_validation/final_prioritization/final_priority_sota_ready_review_queue.csv`",
            "- Diverse shortlist: `outputs/sota_validation/final_prioritization/final_priority_sota_ready_diverse_shortlist.csv`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a final SOTA-ready multi-evidence decision matrix.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_matrix(root)
    ordered_cols = [col for col in output_columns() if col in df.columns] + [col for col in df.columns if col not in output_columns()]
    df[ordered_cols].to_csv(out_dir / "final_priority_sota_ready_matrix.csv", index=False)

    shortlist = df[df["sotaReadyTier"].astype(str).str.startswith(("A_", "B_"))].head(300)
    shortlist[ordered_cols].to_csv(out_dir / "final_priority_sota_ready_top300_expert_shortlist.csv", index=False)
    df[df["sotaReadyAction"].eq("novel_pair_expert_review")].head(300)[ordered_cols].to_csv(
        out_dir / "final_priority_sota_ready_novel_shortlist.csv", index=False
    )
    df[df["sotaReadyAction"].eq("positive_control_or_known_mechanism")].head(300)[ordered_cols].to_csv(
        out_dir / "final_priority_sota_ready_positive_controls.csv", index=False
    )
    df[df["sotaReadyAction"].isin(["safety_or_contraindication_review", "structure_low_confidence_review", "target_context_review"])].head(500)[ordered_cols].to_csv(
        out_dir / "final_priority_sota_ready_review_queue.csv", index=False
    )
    diverse = select_diverse(df, per_direction=50)
    diverse_cols = [col for col in ["sotaReadyDiverseRankWithinDirection"] + ordered_cols if col in diverse.columns]
    diverse[diverse_cols].to_csv(out_dir / "final_priority_sota_ready_diverse_shortlist.csv", index=False)

    topk = topk_metrics(df)
    write_csv(out_dir / "final_priority_sota_ready_topk_composition.csv", topk)
    summary = build_summary(df, topk, diverse)
    write_json(out_dir / "final_priority_sota_ready_summary.json", summary)
    (out_dir / "FINAL_PRIORITY_SOTA_READY_DECISION_MATRIX.md").write_text(markdown(summary), encoding="utf-8")

    print(json.dumps({"summary": summary, "out_dir": args.out_dir}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

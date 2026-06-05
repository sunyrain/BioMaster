from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


NOVEL_CLASSES = {
    "disease_context_supported_new_pair",
    "model_priority_without_txgnn_kg_path",
}

POSITIVE_NETWORK_TIERS = {
    "A_direct_disease_module",
    "B_direct_low_score",
    "B_network_close",
    "C_network_reachable",
}

CUTOFFS = [20, 50, 100, 200, 500, 1000, 2000]

RELEVANT_TISSUES_BY_DIRECTION: dict[str, list[str]] = {
    "oncology": [
        "bone marrow",
        "breast",
        "cervix",
        "colon",
        "endometrium",
        "fallopian tube",
        "kidney",
        "liver",
        "lung",
        "lymph node",
        "ovary",
        "pancreas",
        "prostate",
        "rectum",
        "skin",
        "small intestine",
        "spleen",
        "stomach",
        "testis",
        "thymus",
        "thyroid gland",
        "tonsil",
        "urinary bladder",
        "vagina",
    ],
    "cardiovascular": [
        "blood vessel",
        "heart muscle",
        "smooth muscle",
    ],
    "infectious_disease": [
        "bone marrow",
        "colon",
        "duodenum",
        "kidney",
        "liver",
        "lung",
        "lymph node",
        "rectum",
        "skin",
        "small intestine",
        "spleen",
        "stomach",
        "urinary bladder",
    ],
    "neurology_psychiatry": [
        "amygdala",
        "basal ganglia",
        "cerebellum",
        "cerebral cortex",
        "choroid plexus",
        "hippocampal formation",
        "hypothalamus",
        "midbrain",
        "retina",
        "spinal cord",
    ],
    "immunology_inflammation": [
        "appendix",
        "bone marrow",
        "colon",
        "lung",
        "lymph node",
        "skin",
        "small intestine",
        "spleen",
        "thymus",
        "tonsil",
    ],
}

TIER_SCORE = {
    "A_high_relevant_tissue_expression": 94.0,
    "B_moderate_relevant_tissue_expression": 78.0,
    "C_low_relevant_tissue_expression": 56.0,
    "D_no_relevant_tissue_expression": 32.0,
    "E_no_hpa_detected_expression": 20.0,
    "U_no_hpa_gene_match": 50.0,
}

TIER_ADJUSTMENT = {
    "A_high_relevant_tissue_expression": 2.5,
    "B_moderate_relevant_tissue_expression": 1.2,
    "C_low_relevant_tissue_expression": 0.0,
    "D_no_relevant_tissue_expression": -1.0,
    "E_no_hpa_detected_expression": -1.5,
    "U_no_hpa_gene_match": 0.0,
}


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


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


def pct_str(value: float | int | None) -> str:
    return "NA" if value is None else f"{float(value):.2f}%"


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def top_tissue_string(rows: pd.DataFrame, limit: int = 6) -> str:
    if rows.empty:
        return ""
    top = rows.sort_values("nTPM", ascending=False).head(limit)
    return "; ".join(f"{row.Tissue}:{float(row.nTPM):.2f}" for row in top.itertuples(index=False))


def classify_tissue_tier(matched: bool, max_relevant: float, max_all: float) -> str:
    if not matched:
        return "U_no_hpa_gene_match"
    if max_relevant >= 10.0:
        return "A_high_relevant_tissue_expression"
    if max_relevant >= 1.0:
        return "B_moderate_relevant_tissue_expression"
    if max_relevant > 0.0:
        return "C_low_relevant_tissue_expression"
    if max_all > 0.0:
        return "D_no_relevant_tissue_expression"
    return "E_no_hpa_detected_expression"


def tissue_score(tier: str, max_relevant: float) -> float:
    base = TIER_SCORE.get(tier, 50.0)
    if max_relevant <= 0:
        return base
    bonus = min(4.0, math.log10(max_relevant + 1.0) * 1.5)
    return round(max(0.0, min(100.0, base + bonus)), 4)


def tissue_adjustment(tier: str, max_relevant: float) -> float:
    adjustment = TIER_ADJUSTMENT.get(tier, 0.0)
    if tier.startswith(("A_", "B_")) and max_relevant > 0:
        adjustment += min(0.8, math.log10(max_relevant + 1.0) * 0.25)
    return round(adjustment, 4)


def reason_for_tier(tier: str, max_relevant: float, top_relevant: str, max_all: float, top_all: str) -> str:
    if tier == "U_no_hpa_gene_match":
        return "No exact HPA gene-symbol match was found; treat as data coverage gap, not negative biology."
    if tier == "A_high_relevant_tissue_expression":
        return f"Relevant tissue expression is high (max nTPM {max_relevant:.2f}; {top_relevant})."
    if tier == "B_moderate_relevant_tissue_expression":
        return f"Relevant tissue expression is detectable (max nTPM {max_relevant:.2f}; {top_relevant})."
    if tier == "C_low_relevant_tissue_expression":
        return f"Relevant tissue expression is weak but nonzero (max nTPM {max_relevant:.2f}; {top_relevant})."
    if tier == "D_no_relevant_tissue_expression":
        return f"HPA detects the target in other tissues (max nTPM {max_all:.2f}; {top_all}), but not in the mapped disease-relevant tissue set."
    return "HPA has a gene match but no detected consensus tissue expression."


def read_hpa(hpa_path: Path, tissue_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hpa = pd.read_csv(hpa_path, sep="\t", compression="zip")
    tissue_meta = pd.read_csv(tissue_path, sep="\t", compression="zip")
    required = {"Gene", "Gene name", "Tissue", "nTPM"}
    missing = required - set(hpa.columns)
    if missing:
        raise ValueError(f"HPA table is missing required columns: {sorted(missing)}")
    hpa = hpa.rename(columns={"Gene name": "geneName"})
    hpa["symbolNorm"] = hpa["geneName"].map(normalize_symbol)
    hpa["nTPM"] = pd.to_numeric(hpa["nTPM"], errors="coerce").fillna(0.0).astype(float)
    hpa["Tissue"] = hpa["Tissue"].astype(str).str.strip()
    hpa = hpa[hpa["symbolNorm"] != ""].copy()
    hpa = (
        hpa.groupby(["symbolNorm", "geneName", "Gene", "Tissue"], as_index=False)["nTPM"]
        .max()
        .sort_values(["symbolNorm", "nTPM"], ascending=[True, False])
    )
    tissue_meta["Tissue"] = tissue_meta["Tissue"].astype(str).str.strip()
    return hpa, tissue_meta


def build_expression_lookup(hpa: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {symbol: group.copy() for symbol, group in hpa.groupby("symbolNorm", sort=False)}


def target_direction_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    cols = ["target", "protein", "proteinName", "direction"]
    if "directionLabelZhFinal" in candidates:
        cols.append("directionLabelZhFinal")
    target_df = candidates[cols].drop_duplicates().copy()
    target_df["targetSymbolNorm"] = target_df["target"].map(normalize_symbol)
    return target_df.reset_index(drop=True)


def audit_target_direction(target_df: pd.DataFrame, hpa_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in target_df.itertuples(index=False):
        data = row._asdict()
        direction = str(data.get("direction") or "")
        relevant_tissues = RELEVANT_TISSUES_BY_DIRECTION.get(direction, [])
        relevant_set = set(relevant_tissues)
        symbol = str(data.get("targetSymbolNorm") or "")
        expr = hpa_by_symbol.get(symbol)
        matched = expr is not None and not expr.empty
        if matched:
            relevant_expr = expr[expr["Tissue"].isin(relevant_set)].copy()
            max_all = float(expr["nTPM"].max())
            median_all = float(expr["nTPM"].median())
            max_relevant = float(relevant_expr["nTPM"].max()) if not relevant_expr.empty else 0.0
            median_relevant = float(relevant_expr["nTPM"].median()) if not relevant_expr.empty else 0.0
            expressed_relevant = int((relevant_expr["nTPM"] >= 1.0).sum()) if not relevant_expr.empty else 0
            strong_relevant = int((relevant_expr["nTPM"] >= 10.0).sum()) if not relevant_expr.empty else 0
            top_relevant = top_tissue_string(relevant_expr)
            top_all = top_tissue_string(expr)
            gene_names = ";".join(sorted(set(expr["geneName"].astype(str))))
            gene_ids = ";".join(sorted(set(expr["Gene"].astype(str))))
        else:
            relevant_expr = pd.DataFrame()
            max_all = 0.0
            median_all = 0.0
            max_relevant = 0.0
            median_relevant = 0.0
            expressed_relevant = 0
            strong_relevant = 0
            top_relevant = ""
            top_all = ""
            gene_names = ""
            gene_ids = ""
        tier = classify_tissue_tier(matched, max_relevant, max_all)
        rows.append(
            {
                "target": data.get("target", ""),
                "protein": data.get("protein", ""),
                "proteinName": data.get("proteinName", ""),
                "direction": direction,
                "directionLabelZhFinal": data.get("directionLabelZhFinal", ""),
                "hpaMatchedGeneFlag": bool(matched),
                "hpaGeneName": gene_names,
                "hpaGeneIds": gene_ids,
                "directionRelevantTissues": ";".join(relevant_tissues),
                "relevantTissueCount": len(relevant_tissues),
                "relevantTissueExpressedCountNTPM1": expressed_relevant,
                "relevantTissueStrongCountNTPM10": strong_relevant,
                "maxNtpmRelevantTissues": round(max_relevant, 6),
                "medianNtpmRelevantTissues": round(median_relevant, 6),
                "maxNtpmAllTissues": round(max_all, 6),
                "medianNtpmAllTissues": round(median_all, 6),
                "topRelevantTissuesByNtpm": top_relevant,
                "topAllTissuesByNtpm": top_all,
                "tissueContextTier": tier,
                "tissueContextScore": tissue_score(tier, max_relevant),
                "tissueContextAdjustment": tissue_adjustment(tier, max_relevant),
                "tissueContextPositiveFlag": tier.startswith(("A_", "B_")),
                "tissueContextCoverageGapFlag": tier.startswith("U_"),
                "tissueContextReason": reason_for_tier(tier, max_relevant, top_relevant, max_all, top_all),
            }
        )
    return pd.DataFrame(rows)


def build_protein_summary(target_df: pd.DataFrame, hpa_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = target_df[["target", "protein", "proteinName", "targetSymbolNorm"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        expr = hpa_by_symbol.get(str(row.targetSymbolNorm))
        matched = expr is not None and not expr.empty
        rows.append(
            {
                "target": row.target,
                "protein": row.protein,
                "proteinName": row.proteinName,
                "hpaMatchedGeneFlag": bool(matched),
                "hpaGeneName": ";".join(sorted(set(expr["geneName"].astype(str)))) if matched else "",
                "hpaGeneIds": ";".join(sorted(set(expr["Gene"].astype(str)))) if matched else "",
                "maxNtpmAllTissues": round(float(expr["nTPM"].max()), 6) if matched else 0.0,
                "medianNtpmAllTissues": round(float(expr["nTPM"].median()), 6) if matched else 0.0,
                "expressedTissueCountNTPM1": int((expr["nTPM"] >= 1.0).sum()) if matched else 0,
                "strongTissueCountNTPM10": int((expr["nTPM"] >= 10.0).sum()) if matched else 0,
                "topAllTissuesByNtpm": top_tissue_string(expr) if matched else "",
            }
        )
    return pd.DataFrame(rows)


def classify_action(row: pd.Series) -> tuple[str, str]:
    base_action = str(row.get("sotaNetworkAction") or row.get("sotaReadyAction") or "")
    if base_action in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return base_action, "Tissue context is reported but does not override safety, structure, or hard-flag review gates."

    tier = str(row.get("tissueContextTier") or "")
    network_tier = str(row.get("networkEvidenceTier") or "")
    strict_novel = bool_value(row.get("strictNovelPairFlag")) or str(row.get("noveltyClass") or "") in NOVEL_CLASSES
    network_positive = bool_value(row.get("networkPositiveFlag")) or network_tier in POSITIVE_NETWORK_TIERS

    if tier.startswith(("A_", "B_")) and network_positive and strict_novel:
        return "novel_tissue_network_review", "Novel or new-context candidate has both relevant tissue expression and network-medicine support."
    if tier.startswith(("A_", "B_")) and network_positive:
        return "tissue_network_supported_review", "Candidate has relevant tissue expression plus orthogonal network support."
    if tier.startswith(("A_", "B_")):
        return "tissue_context_supported_review", "Candidate target is expressed in mapped disease-relevant tissue context."
    if tier.startswith(("D_", "E_")):
        return "tissue_context_mismatch_review", "Candidate lacks expression in the mapped disease-relevant tissue set and should be reviewed before prioritization."
    if tier.startswith("U_"):
        return "hpa_gene_coverage_gap_review", "HPA exact gene-symbol matching did not resolve this target; this is a data gap."
    return "secondary_context_review", "No strong tissue-context action was assigned."


def classify_context_tier(row: pd.Series) -> str:
    score = float(row.get("sotaContextScore", 0.0))
    action = str(row.get("sotaContextAction") or "")
    tissue_tier = str(row.get("tissueContextTier") or "")
    network_tier = str(row.get("networkEvidenceTier") or "")
    structure_good = str(row.get("structureConfidenceTier") or "").startswith(("A_", "B_"))
    target_good = str(row.get("targetDruggabilityTier") or "").startswith(("A_", "B_"))
    base_tier = str(row.get("sotaNetworkTier") or row.get("sotaReadyTier") or "")

    if action in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return "D_blocked_or_requires_resolution"
    if (
        score >= 90.0
        and tissue_tier.startswith(("A_", "B_"))
        and network_tier in POSITIVE_NETWORK_TIERS
        and structure_good
        and target_good
        and base_tier.startswith(("A_", "B_"))
    ):
        return "A_tissue_network_expert_priority"
    if score >= 80.0 and tissue_tier.startswith(("A_", "B_")) and structure_good and target_good:
        return "B_tissue_supported_review_priority"
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
    ranked = df.sort_values("sotaContextScore", ascending=False).reset_index(drop=True)
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
                "tissuePositiveRows": int(top["tissueContextPositiveFlag"].sum()),
                "hpaMatchedRows": int(top["hpaMatchedGeneFlag"].sum()),
                "networkPositiveRows": int(top["networkPositiveFlag"].sum()) if "networkPositiveFlag" in top else 0,
                "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top else 0,
                "tierCounts": dict(Counter(top["sotaContextTier"].astype(str))),
                "tissueTierCounts": dict(Counter(top["tissueContextTier"].astype(str))),
                "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else int(top["drug"].nunique()),
                "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else int(top["target"].nunique()),
            }
        )
    for direction, group in df.groupby("direction"):
        ranked_dir = group.sort_values("sotaContextScore", ascending=False).reset_index(drop=True)
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
                    "tissuePositiveRows": int(top["tissueContextPositiveFlag"].sum()),
                    "hpaMatchedRows": int(top["hpaMatchedGeneFlag"].sum()),
                    "networkPositiveRows": int(top["networkPositiveFlag"].sum()) if "networkPositiveFlag" in top else 0,
                    "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top else 0,
                    "tierCounts": dict(Counter(top["sotaContextTier"].astype(str))),
                    "tissueTierCounts": dict(Counter(top["tissueContextTier"].astype(str))),
                    "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else int(top["drug"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else int(top["target"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def direction_summary(df: pd.DataFrame, target_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_counts = target_audit.groupby("direction").size().to_dict()
    target_positive = target_audit.groupby("direction")["tissueContextPositiveFlag"].sum().to_dict()
    for direction, group in df.groupby("direction"):
        top100 = group.sort_values("sotaContextScore", ascending=False).head(min(100, len(group)))
        tissue_positive = int(group["tissueContextPositiveFlag"].sum())
        hpa_matched = int(group["hpaMatchedGeneFlag"].sum())
        rows.append(
            {
                "direction": direction,
                "directionLabelZhFinal": group["directionLabelZhFinal"].iloc[0] if "directionLabelZhFinal" in group else "",
                "candidateRows": int(len(group)),
                "targetDirectionRows": int(target_counts.get(direction, 0)),
                "targetDirectionTissuePositiveRows": int(target_positive.get(direction, 0)),
                "candidateHpaMatchedRows": hpa_matched,
                "candidateHpaMatchedPct": round(pct(hpa_matched, len(group)), 4),
                "candidateTissuePositiveRows": tissue_positive,
                "candidateTissuePositivePct": round(pct(tissue_positive, len(group)), 4),
                "medianRelevantNtpm": round(float(group["maxNtpmRelevantTissues"].median()), 4),
                "maxRelevantNtpm": round(float(group["maxNtpmRelevantTissues"].max()), 4),
                "tissueTierCounts": dict(Counter(group["tissueContextTier"].astype(str))),
                "contextTierCounts": dict(Counter(group["sotaContextTier"].astype(str))),
                "top100TissuePositiveRows": int(top100["tissueContextPositiveFlag"].sum()),
                "top100HpaMatchedRows": int(top100["hpaMatchedGeneFlag"].sum()),
                "top100KnownRows": int(num_col(top100, "knownDrugTargetPair").sum()),
                "top100NovelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top100 else 0,
                "relevantTissuePanel": ";".join(RELEVANT_TISSUES_BY_DIRECTION.get(direction, [])),
            }
        )
    return pd.DataFrame(rows)


def build_context_matrix(candidates: pd.DataFrame, target_audit: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["target", "protein", "direction"]
    add_cols = [
        "hpaMatchedGeneFlag",
        "hpaGeneName",
        "hpaGeneIds",
        "directionRelevantTissues",
        "relevantTissueCount",
        "relevantTissueExpressedCountNTPM1",
        "relevantTissueStrongCountNTPM10",
        "maxNtpmRelevantTissues",
        "medianNtpmRelevantTissues",
        "maxNtpmAllTissues",
        "medianNtpmAllTissues",
        "topRelevantTissuesByNtpm",
        "topAllTissuesByNtpm",
        "tissueContextTier",
        "tissueContextScore",
        "tissueContextAdjustment",
        "tissueContextPositiveFlag",
        "tissueContextCoverageGapFlag",
        "tissueContextReason",
    ]
    merged = candidates.merge(target_audit[join_cols + add_cols], on=join_cols, how="left")
    merged["hpaMatchedGeneFlag"] = merged["hpaMatchedGeneFlag"].fillna(False).astype(bool)
    merged["tissueContextPositiveFlag"] = merged["tissueContextPositiveFlag"].fillna(False).astype(bool)
    merged["tissueContextCoverageGapFlag"] = merged["tissueContextCoverageGapFlag"].fillna(True).astype(bool)
    merged["tissueContextTier"] = merged["tissueContextTier"].fillna("U_no_hpa_gene_match")
    merged["tissueContextScore"] = num_col(merged, "tissueContextScore", 50.0)
    merged["tissueContextAdjustment"] = num_col(merged, "tissueContextAdjustment", 0.0)
    base_score_col = "sotaNetworkScore" if "sotaNetworkScore" in merged else "sotaReadyScore"
    merged["sotaContextScore"] = (num_col(merged, base_score_col) + merged["tissueContextAdjustment"]).clip(0, 100).round(4)
    actions = merged.apply(classify_action, axis=1)
    merged["sotaContextAction"] = [item[0] for item in actions]
    merged["sotaContextActionNote"] = [item[1] for item in actions]
    merged["sotaContextTier"] = merged.apply(classify_context_tier, axis=1)
    merged = merged.sort_values(["sotaContextScore", base_score_col], ascending=[False, False]).reset_index(drop=True).copy()
    merged["sotaContextRankGlobal"] = range(1, len(merged) + 1)
    merged["sotaContextRankWithinDirection"] = (
        merged.groupby("direction")["sotaContextScore"].rank(method="first", ascending=False).astype(int)
    )
    front = [
        "sotaContextRankGlobal",
        "sotaContextRankWithinDirection",
        "sotaContextScore",
        "sotaContextTier",
        "sotaContextAction",
        "sotaContextActionNote",
        "tissueContextScore",
        "tissueContextAdjustment",
        "tissueContextTier",
        "tissueContextPositiveFlag",
        "hpaMatchedGeneFlag",
        "maxNtpmRelevantTissues",
        "topRelevantTissuesByNtpm",
    ]
    ordered = front + [col for col in merged.columns if col not in front]
    return merged[ordered]


def build_summary(
    candidates: pd.DataFrame,
    context: pd.DataFrame,
    target_audit: pd.DataFrame,
    protein_summary: pd.DataFrame,
    hpa: pd.DataFrame,
    tissue_meta: pd.DataFrame,
) -> dict[str, Any]:
    old_auc, old_ap = safe_auc_ap(context, "sotaNetworkScore" if "sotaNetworkScore" in context else "sotaReadyScore")
    new_auc, new_ap = safe_auc_ap(context, "sotaContextScore")
    top100 = context.sort_values("sotaContextScore", ascending=False).head(min(100, len(context)))
    target_positive = int(target_audit["tissueContextPositiveFlag"].sum())
    candidate_positive = int(context["tissueContextPositiveFlag"].sum())
    hpa_candidate_matched = int(context["hpaMatchedGeneFlag"].sum())
    hpa_target_matched = int(protein_summary["hpaMatchedGeneFlag"].sum())
    all_tissues = sorted(set(hpa["Tissue"].astype(str)))
    map_missing = {
        direction: sorted(set(tissues) - set(all_tissues))
        for direction, tissues in RELEVANT_TISSUES_BY_DIRECTION.items()
    }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "HPA consensus RNA tissue-expression context for final SOTA-network candidates.",
        "interpretationNote": "This layer checks target tissue plausibility. It is not disease-causality evidence and does not replace disease genetics, perturbational transcriptomics, or dependency screens.",
        "candidateRows": int(len(context)),
        "inputCandidateRows": int(len(candidates)),
        "uniqueTargets": int(context["protein"].nunique()) if "protein" in context else int(context["target"].nunique()),
        "targetDirectionRows": int(len(target_audit)),
        "hpaExpressionRows": int(len(hpa)),
        "hpaUniqueGeneSymbols": int(hpa["symbolNorm"].nunique()),
        "hpaTissues": int(hpa["Tissue"].nunique()),
        "hpaTissueMetaRows": int(len(tissue_meta)),
        "candidateHpaMatchedRows": hpa_candidate_matched,
        "candidateHpaMatchedPct": round(pct(hpa_candidate_matched, len(context)), 4),
        "uniqueTargetHpaMatchedRows": hpa_target_matched,
        "uniqueTargetHpaMatchedPct": round(pct(hpa_target_matched, len(protein_summary)), 4),
        "candidateTissuePositiveRows": candidate_positive,
        "candidateTissuePositivePct": round(pct(candidate_positive, len(context)), 4),
        "targetDirectionTissuePositiveRows": target_positive,
        "targetDirectionTissuePositivePct": round(pct(target_positive, len(target_audit)), 4),
        "candidateTissueTierCounts": dict(Counter(context["tissueContextTier"].astype(str))),
        "targetDirectionTissueTierCounts": dict(Counter(target_audit["tissueContextTier"].astype(str))),
        "sotaContextTierCounts": dict(Counter(context["sotaContextTier"].astype(str))),
        "sotaContextActionCounts": dict(Counter(context["sotaContextAction"].astype(str))),
        "oldSotaNetworkAuroc": old_auc,
        "oldSotaNetworkAveragePrecision": old_ap,
        "sotaContextAuroc": new_auc,
        "sotaContextAveragePrecision": new_ap,
        "top100": {
            "knownDrugTargetRows": int(num_col(top100, "knownDrugTargetPair").sum()),
            "novelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top100 else 0,
            "tissuePositiveRows": int(top100["tissueContextPositiveFlag"].sum()),
            "hpaMatchedRows": int(top100["hpaMatchedGeneFlag"].sum()),
            "networkPositiveRows": int(top100["networkPositiveFlag"].sum()) if "networkPositiveFlag" in top100 else 0,
            "tierCounts": dict(Counter(top100["sotaContextTier"].astype(str))),
            "tissueTierCounts": dict(Counter(top100["tissueContextTier"].astype(str))),
            "uniqueDrugs": int(top100["drugId"].nunique()) if "drugId" in top100 else int(top100["drug"].nunique()),
            "uniqueTargets": int(top100["protein"].nunique()) if "protein" in top100 else int(top100["target"].nunique()),
        },
        "relevantTissuesByDirection": RELEVANT_TISSUES_BY_DIRECTION,
        "missingMappedTissuesByDirection": map_missing,
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame) -> str:
    lines = [
        "# HPA Tissue Context Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit adds a tissue-expression plausibility layer to the final SOTA-network candidate matrix using Human Protein Atlas consensus RNA nTPM values.",
        "",
        "## Summary",
        "",
        f"- Candidate rows audited: {summary['candidateRows']}",
        f"- Unique targets audited: {summary['uniqueTargets']}",
        f"- Target-direction rows audited: {summary['targetDirectionRows']}",
        f"- HPA exact gene-symbol match in candidates: {summary['candidateHpaMatchedRows']} ({pct_str(summary['candidateHpaMatchedPct'])})",
        f"- HPA exact gene-symbol match in unique targets: {summary['uniqueTargetHpaMatchedRows']} ({pct_str(summary['uniqueTargetHpaMatchedPct'])})",
        f"- Candidate rows with A/B tissue-context support: {summary['candidateTissuePositiveRows']} ({pct_str(summary['candidateTissuePositivePct'])})",
        f"- Target-direction rows with A/B tissue-context support: {summary['targetDirectionTissuePositiveRows']} ({pct_str(summary['targetDirectionTissuePositivePct'])})",
        f"- SOTA-network AP to SOTA-context AP: {fmt(summary['oldSotaNetworkAveragePrecision'], 4)} -> {fmt(summary['sotaContextAveragePrecision'], 4)}",
        f"- Top100 tissue-positive rows: {summary['top100']['tissuePositiveRows']}; HPA-matched rows: {summary['top100']['hpaMatchedRows']}; known rows: {summary['top100']['knownDrugTargetRows']}; novel rows: {summary['top100']['novelRows']}",
        "",
        "## Direction Summary",
        "",
        "| Direction | Candidates | Target-direction rows | Candidate A/B tissue support | Top100 A/B support | Median relevant nTPM | Relevant tissue panel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in direction_df.sort_values("direction").itertuples(index=False):
        lines.append(
            f"| {row.direction} | {row.candidateRows} | {row.targetDirectionRows} | "
            f"{row.candidateTissuePositiveRows} ({pct_str(row.candidateTissuePositivePct)}) | "
            f"{row.top100TissuePositiveRows} | {fmt(row.medianRelevantNtpm, 2)} | {row.relevantTissuePanel} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/B tissue-context support means the target has detectable HPA consensus RNA expression in the tissue panel mapped to that disease direction.",
            "- No HPA match or no relevant-tissue expression should be interpreted as a context or data-coverage warning, not as proof that the biology is impossible.",
            "- This layer is orthogonal to ConPLex, DiffDock, Open Targets, TxGNN, and network medicine. It strengthens expert triage by asking whether the target appears in the right biological tissue context.",
            "",
            "## Machine-Readable Outputs",
            "",
            "- Candidate audit: `outputs/sota_validation/tissue_context/candidate_tissue_context_audit.csv`",
            "- Target-direction audit: `outputs/sota_validation/tissue_context/target_tissue_context_audit.csv`",
            "- Protein expression summary: `outputs/sota_validation/tissue_context/protein_hpa_expression_summary.csv`",
            "- Integrated SOTA-context matrix: `outputs/sota_validation/final_prioritization/final_priority_sota_context_matrix.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HPA tissue-expression context audit for final candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--candidates",
        default="outputs/sota_validation/final_prioritization/final_priority_sota_network_matrix.csv",
    )
    parser.add_argument("--hpa-rna", default="data/external/hpa/rna_tissue_consensus.tsv.zip")
    parser.add_argument("--hpa-tissues", default="data/external/hpa/rna_tissue_consensus_tissues.tsv.zip")
    parser.add_argument("--out-dir", default="outputs/sota_validation/tissue_context")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates = pd.read_csv(root / args.candidates).fillna("")
    hpa, tissue_meta = read_hpa(root / args.hpa_rna, root / args.hpa_tissues)
    hpa_by_symbol = build_expression_lookup(hpa)
    target_df = target_direction_rows(candidates)
    target_audit = audit_target_direction(target_df, hpa_by_symbol)
    protein_summary = build_protein_summary(target_df, hpa_by_symbol)
    context = build_context_matrix(candidates, target_audit)
    topk = topk_metrics(context)
    direction_df = direction_summary(context, target_audit)
    summary = build_summary(candidates, context, target_audit, protein_summary, hpa, tissue_meta)

    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    target_audit.to_csv(out_dir / "target_tissue_context_audit.csv", index=False)
    protein_summary.to_csv(out_dir / "protein_hpa_expression_summary.csv", index=False)
    context.to_csv(out_dir / "candidate_tissue_context_audit.csv", index=False)
    topk.to_csv(out_dir / "tissue_context_topk_metrics.csv", index=False)
    direction_df.to_csv(out_dir / "tissue_context_direction_summary.csv", index=False)
    write_json(out_dir / "tissue_context_summary.json", summary)
    (out_dir / "TISSUE_CONTEXT_AUDIT.md").write_text(markdown(summary, direction_df), encoding="utf-8")

    context.to_csv(final_dir / "final_priority_sota_context_matrix.csv", index=False)
    context.head(300).to_csv(final_dir / "final_priority_sota_context_top300_expert_shortlist.csv", index=False)
    novel_mask = context["noveltyClass"].astype(str).isin(NOVEL_CLASSES) if "noveltyClass" in context else pd.Series(False, index=context.index)
    context[novel_mask].head(300).to_csv(final_dir / "final_priority_sota_context_novel_shortlist.csv", index=False)
    context[context["sotaContextAction"].isin(["tissue_context_mismatch_review", "hpa_gene_coverage_gap_review"])].head(300).to_csv(
        final_dir / "final_priority_sota_context_review_queue.csv", index=False
    )
    direction_df.to_csv(final_dir / "final_priority_sota_context_direction_summary.csv", index=False)
    (final_dir / "FINAL_PRIORITY_SOTA_TISSUE_CONTEXT_AUDIT.md").write_text(markdown(summary, direction_df), encoding="utf-8")

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

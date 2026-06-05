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

POSITIVE_CONTEXT_TIERS = {
    "A_tissue_network_expert_priority",
    "B_tissue_supported_review_priority",
    "C_context_or_secondary_review",
}

CUTOFFS = [20, 50, 100, 200, 500, 1000, 2000]

GTEX_RELEVANT_TISSUES_BY_DIRECTION: dict[str, list[str]] = {
    "oncology": [
        "Bladder",
        "Breast_Mammary_Tissue",
        "Cervix_Ectocervix",
        "Cervix_Endocervix",
        "Colon_Sigmoid",
        "Colon_Transverse",
        "Colon_Transverse_Mucosa",
        "Colon_Transverse_Muscularis",
        "Fallopian_Tube",
        "Kidney_Cortex",
        "Kidney_Medulla",
        "Liver",
        "Liver_Hepatocyte",
        "Liver_Portal_Tract",
        "Lung",
        "Ovary",
        "Pancreas",
        "Pancreas_Acini",
        "Pancreas_Islets",
        "Prostate",
        "Skin_Not_Sun_Exposed_Suprapubic",
        "Skin_Sun_Exposed_Lower_leg",
        "Small_Intestine_Terminal_Ileum",
        "Small_Intestine_Terminal_Ileum_Lymphode_Aggregate",
        "Small_Intestine_Terminal_Ileum_Mixed_Cell",
        "Spleen",
        "Stomach",
        "Stomach_Mucosa",
        "Stomach_Muscularis",
        "Testis",
        "Thyroid",
        "Uterus",
        "Vagina",
        "Whole_Blood",
    ],
    "cardiovascular": [
        "Artery_Aorta",
        "Artery_Coronary",
        "Artery_Tibial",
        "Heart_Atrial_Appendage",
        "Heart_Left_Ventricle",
        "Muscle_Skeletal",
    ],
    "infectious_disease": [
        "Bladder",
        "Colon_Sigmoid",
        "Colon_Transverse",
        "Colon_Transverse_Mucosa",
        "Kidney_Cortex",
        "Kidney_Medulla",
        "Liver",
        "Lung",
        "Skin_Not_Sun_Exposed_Suprapubic",
        "Skin_Sun_Exposed_Lower_leg",
        "Small_Intestine_Terminal_Ileum",
        "Small_Intestine_Terminal_Ileum_Lymphode_Aggregate",
        "Spleen",
        "Stomach",
        "Stomach_Mucosa",
        "Whole_Blood",
    ],
    "neurology_psychiatry": [
        "Brain_Amygdala",
        "Brain_Anterior_cingulate_cortex_BA24",
        "Brain_Caudate_basal_ganglia",
        "Brain_Cerebellar_Hemisphere",
        "Brain_Cerebellum",
        "Brain_Cortex",
        "Brain_Frontal_Cortex_BA9",
        "Brain_Hippocampus",
        "Brain_Hypothalamus",
        "Brain_Nucleus_accumbens_basal_ganglia",
        "Brain_Putamen_basal_ganglia",
        "Brain_Spinal_cord_cervical_c-1",
        "Brain_Substantia_nigra",
        "Nerve_Tibial",
        "Pituitary",
    ],
    "immunology_inflammation": [
        "Cells_EBV-transformed_lymphocytes",
        "Colon_Sigmoid",
        "Colon_Transverse",
        "Colon_Transverse_Mucosa",
        "Lung",
        "Skin_Not_Sun_Exposed_Suprapubic",
        "Skin_Sun_Exposed_Lower_leg",
        "Small_Intestine_Terminal_Ileum",
        "Small_Intestine_Terminal_Ileum_Lymphode_Aggregate",
        "Spleen",
        "Whole_Blood",
    ],
}

TIER_SCORE = {
    "A_high_relevant_gtex_expression": 93.0,
    "B_moderate_relevant_gtex_expression": 77.0,
    "C_low_relevant_gtex_expression": 55.0,
    "D_no_relevant_gtex_expression": 33.0,
    "E_no_gtex_detected_expression": 20.0,
    "U_no_gtex_gene_match": 50.0,
}

TIER_ADJUSTMENT = {
    "A_high_relevant_gtex_expression": 1.2,
    "B_moderate_relevant_gtex_expression": 0.6,
    "C_low_relevant_gtex_expression": 0.0,
    "D_no_relevant_gtex_expression": -0.5,
    "E_no_gtex_detected_expression": -0.8,
    "U_no_gtex_gene_match": 0.0,
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


def top_tissue_string(values: pd.Series, limit: int = 6) -> str:
    if values.empty:
        return ""
    top = values.sort_values(ascending=False).head(limit)
    return "; ".join(f"{idx}:{float(val):.2f}" for idx, val in top.items())


def classify_gtex_tier(matched: bool, max_relevant: float, max_all: float) -> str:
    if not matched:
        return "U_no_gtex_gene_match"
    if max_relevant >= 10.0:
        return "A_high_relevant_gtex_expression"
    if max_relevant >= 1.0:
        return "B_moderate_relevant_gtex_expression"
    if max_relevant > 0.0:
        return "C_low_relevant_gtex_expression"
    if max_all > 0.0:
        return "D_no_relevant_gtex_expression"
    return "E_no_gtex_detected_expression"


def gtex_score(tier: str, max_relevant: float) -> float:
    base = TIER_SCORE.get(tier, 50.0)
    if max_relevant <= 0:
        return base
    bonus = min(4.0, math.log10(max_relevant + 1.0) * 1.4)
    return round(max(0.0, min(100.0, base + bonus)), 4)


def gtex_adjustment(tier: str, max_relevant: float) -> float:
    adjustment = TIER_ADJUSTMENT.get(tier, 0.0)
    if tier.startswith(("A_", "B_")) and max_relevant > 0:
        adjustment += min(0.5, math.log10(max_relevant + 1.0) * 0.15)
    return round(adjustment, 4)


def reason_for_tier(tier: str, max_relevant: float, top_relevant: str, max_all: float, top_all: str) -> str:
    if tier == "U_no_gtex_gene_match":
        return "No exact GTEx gene-symbol match was found; treat as data coverage gap, not negative biology."
    if tier == "A_high_relevant_gtex_expression":
        return f"GTEx relevant-tissue median TPM is high (max {max_relevant:.2f}; {top_relevant})."
    if tier == "B_moderate_relevant_gtex_expression":
        return f"GTEx relevant-tissue median TPM is detectable (max {max_relevant:.2f}; {top_relevant})."
    if tier == "C_low_relevant_gtex_expression":
        return f"GTEx relevant-tissue median TPM is weak but nonzero (max {max_relevant:.2f}; {top_relevant})."
    if tier == "D_no_relevant_gtex_expression":
        return f"GTEx detects the target in other tissues (max {max_all:.2f}; {top_all}), but not in the mapped disease-relevant tissue set."
    return "GTEx has a gene match but no detected median TPM expression."


def read_gtex(gtex_path: Path) -> tuple[pd.DataFrame, list[str]]:
    gtex = pd.read_csv(gtex_path, sep="\t", compression="gzip", skiprows=2)
    if "Name" not in gtex or "Description" not in gtex:
        raise ValueError("GTEx GCT file must contain Name and Description columns.")
    tissue_cols = [col for col in gtex.columns if col not in {"Name", "Description"}]
    gtex["symbolNorm"] = gtex["Description"].map(normalize_symbol)
    for col in tissue_cols:
        gtex[col] = pd.to_numeric(gtex[col], errors="coerce").fillna(0.0).astype(float)
    gtex = gtex[gtex["symbolNorm"] != ""].copy()
    grouped = gtex.groupby("symbolNorm", as_index=False)[tissue_cols].max()
    names = (
        gtex.groupby("symbolNorm", as_index=False)
        .agg(gtexGeneSymbols=("Description", lambda vals: ";".join(sorted(set(map(str, vals))))),
             gtexGeneIds=("Name", lambda vals: ";".join(sorted(set(map(str, vals))))))
    )
    grouped = grouped.merge(names, on="symbolNorm", how="left")
    return grouped, tissue_cols


def build_expression_lookup(gtex: pd.DataFrame, tissue_cols: list[str]) -> dict[str, pd.Series]:
    lookup: dict[str, pd.Series] = {}
    for _, row in gtex.iterrows():
        symbol = str(row["symbolNorm"])
        series = row[tissue_cols].astype(float).copy()
        series.attrs["gtexGeneSymbols"] = row.get("gtexGeneSymbols", "")
        series.attrs["gtexGeneIds"] = row.get("gtexGeneIds", "")
        lookup[symbol] = series
    return lookup


def target_direction_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    cols = ["target", "protein", "proteinName", "direction"]
    if "directionLabelZhFinal" in candidates:
        cols.append("directionLabelZhFinal")
    target_df = candidates[cols].drop_duplicates().copy()
    target_df["targetSymbolNorm"] = target_df["target"].map(normalize_symbol)
    return target_df.reset_index(drop=True)


def audit_target_direction(
    target_df: pd.DataFrame,
    gtex_by_symbol: dict[str, pd.Series],
    tissue_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    available_tissues = set(tissue_cols)
    for row in target_df.itertuples(index=False):
        data = row._asdict()
        direction = str(data.get("direction") or "")
        relevant_tissues = [t for t in GTEX_RELEVANT_TISSUES_BY_DIRECTION.get(direction, []) if t in available_tissues]
        symbol = str(data.get("targetSymbolNorm") or "")
        expr = gtex_by_symbol.get(symbol)
        matched = expr is not None and not expr.empty
        if matched:
            relevant_expr = expr[relevant_tissues] if relevant_tissues else pd.Series(dtype=float)
            max_all = float(expr[tissue_cols].max())
            median_all = float(expr[tissue_cols].median())
            max_relevant = float(relevant_expr.max()) if not relevant_expr.empty else 0.0
            median_relevant = float(relevant_expr.median()) if not relevant_expr.empty else 0.0
            expressed_relevant = int((relevant_expr >= 1.0).sum()) if not relevant_expr.empty else 0
            strong_relevant = int((relevant_expr >= 10.0).sum()) if not relevant_expr.empty else 0
            top_relevant = top_tissue_string(relevant_expr)
            top_all = top_tissue_string(expr[tissue_cols])
            gene_names = str(expr.attrs.get("gtexGeneSymbols", ""))
            gene_ids = str(expr.attrs.get("gtexGeneIds", ""))
        else:
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
        tier = classify_gtex_tier(matched, max_relevant, max_all)
        rows.append(
            {
                "target": data.get("target", ""),
                "protein": data.get("protein", ""),
                "proteinName": data.get("proteinName", ""),
                "direction": direction,
                "directionLabelZhFinal": data.get("directionLabelZhFinal", ""),
                "gtexMatchedGeneFlag": bool(matched),
                "gtexGeneSymbols": gene_names,
                "gtexGeneIds": gene_ids,
                "gtexRelevantTissues": ";".join(relevant_tissues),
                "gtexRelevantTissueCount": len(relevant_tissues),
                "gtexRelevantTissueExpressedCountTPM1": expressed_relevant,
                "gtexRelevantTissueStrongCountTPM10": strong_relevant,
                "gtexMaxTpmRelevantTissues": round(max_relevant, 6),
                "gtexMedianTpmRelevantTissues": round(median_relevant, 6),
                "gtexMaxTpmAllTissues": round(max_all, 6),
                "gtexMedianTpmAllTissues": round(median_all, 6),
                "gtexTopRelevantTissuesByTpm": top_relevant,
                "gtexTopAllTissuesByTpm": top_all,
                "gtexContextTier": tier,
                "gtexContextScore": gtex_score(tier, max_relevant),
                "gtexContextAdjustment": gtex_adjustment(tier, max_relevant),
                "gtexContextPositiveFlag": tier.startswith(("A_", "B_")),
                "gtexContextCoverageGapFlag": tier.startswith("U_"),
                "gtexContextReason": reason_for_tier(tier, max_relevant, top_relevant, max_all, top_all),
            }
        )
    return pd.DataFrame(rows)


def build_protein_summary(target_df: pd.DataFrame, gtex_by_symbol: dict[str, pd.Series], tissue_cols: list[str]) -> pd.DataFrame:
    base = target_df[["target", "protein", "proteinName", "targetSymbolNorm"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        expr = gtex_by_symbol.get(str(row.targetSymbolNorm))
        matched = expr is not None and not expr.empty
        values = expr[tissue_cols] if matched else pd.Series(dtype=float)
        rows.append(
            {
                "target": row.target,
                "protein": row.protein,
                "proteinName": row.proteinName,
                "gtexMatchedGeneFlag": bool(matched),
                "gtexGeneSymbols": str(expr.attrs.get("gtexGeneSymbols", "")) if matched else "",
                "gtexGeneIds": str(expr.attrs.get("gtexGeneIds", "")) if matched else "",
                "gtexMaxTpmAllTissues": round(float(values.max()), 6) if matched else 0.0,
                "gtexMedianTpmAllTissues": round(float(values.median()), 6) if matched else 0.0,
                "gtexExpressedTissueCountTPM1": int((values >= 1.0).sum()) if matched else 0,
                "gtexStrongTissueCountTPM10": int((values >= 10.0).sum()) if matched else 0,
                "gtexTopAllTissuesByTpm": top_tissue_string(values) if matched else "",
            }
        )
    return pd.DataFrame(rows)


def classify_action(row: pd.Series) -> tuple[str, str]:
    base_action = str(row.get("sotaContextAction") or row.get("sotaNetworkAction") or row.get("sotaReadyAction") or "")
    if base_action in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return base_action, "GTEx context is reported but does not override safety, structure, or hard-flag review gates."

    tier = str(row.get("gtexContextTier") or "")
    hpa_positive = bool_value(row.get("tissueContextPositiveFlag"))
    strict_novel = bool_value(row.get("strictNovelPairFlag")) or str(row.get("noveltyClass") or "") in NOVEL_CLASSES
    context_tier = str(row.get("sotaContextTier") or "")
    context_positive = hpa_positive or context_tier in POSITIVE_CONTEXT_TIERS

    if tier.startswith(("A_", "B_")) and context_positive and strict_novel:
        return "novel_gtex_context_review", "Novel or new-context candidate has GTEx relevant-tissue expression plus existing context support."
    if tier.startswith(("A_", "B_")) and context_positive:
        return "gtex_context_supported_review", "Candidate has GTEx relevant-tissue expression plus existing tissue/network context support."
    if tier.startswith(("A_", "B_")):
        return "gtex_tissue_supported_review", "Candidate target is expressed in GTEx mapped disease-relevant tissue context."
    if tier.startswith(("D_", "E_")):
        return "gtex_context_mismatch_review", "Candidate lacks GTEx expression in the mapped disease-relevant tissue set and should be reviewed before prioritization."
    if tier.startswith("U_"):
        return "gtex_gene_coverage_gap_review", "GTEx exact gene-symbol matching did not resolve this target; this is a data gap."
    return "secondary_context_review", "No strong GTEx context action was assigned."


def classify_gtex_integrated_tier(row: pd.Series) -> str:
    score = float(row.get("sotaGtexContextScore", 0.0))
    action = str(row.get("sotaGtexContextAction") or "")
    gtex_tier = str(row.get("gtexContextTier") or "")
    hpa_tier = str(row.get("tissueContextTier") or "")
    network_tier = str(row.get("networkEvidenceTier") or "")
    structure_good = str(row.get("structureConfidenceTier") or "").startswith(("A_", "B_"))
    target_good = str(row.get("targetDruggabilityTier") or "").startswith(("A_", "B_"))
    base_tier = str(row.get("sotaContextTier") or row.get("sotaNetworkTier") or row.get("sotaReadyTier") or "")

    if action in {
        "safety_or_contraindication_review",
        "structure_low_confidence_review",
        "deprioritize_until_issue_resolved",
    }:
        return "D_blocked_or_requires_resolution"
    if (
        score >= 90.0
        and gtex_tier.startswith(("A_", "B_"))
        and hpa_tier.startswith(("A_", "B_"))
        and network_tier in {"A_direct_disease_module", "B_direct_low_score", "B_network_close", "C_network_reachable"}
        and structure_good
        and target_good
        and base_tier.startswith(("A_", "B_", "C_"))
    ):
        return "A_multi_tissue_context_expert_priority"
    if score >= 80.0 and gtex_tier.startswith(("A_", "B_")) and structure_good and target_good:
        return "B_gtex_supported_review_priority"
    if score >= 65.0:
        return "C_context_or_secondary_review"
    return "D_low_priority_or_sparse_support"


def build_context_matrix(candidates: pd.DataFrame, target_audit: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["target", "protein", "direction"]
    add_cols = [
        "gtexMatchedGeneFlag",
        "gtexGeneSymbols",
        "gtexGeneIds",
        "gtexRelevantTissues",
        "gtexRelevantTissueCount",
        "gtexRelevantTissueExpressedCountTPM1",
        "gtexRelevantTissueStrongCountTPM10",
        "gtexMaxTpmRelevantTissues",
        "gtexMedianTpmRelevantTissues",
        "gtexMaxTpmAllTissues",
        "gtexMedianTpmAllTissues",
        "gtexTopRelevantTissuesByTpm",
        "gtexTopAllTissuesByTpm",
        "gtexContextTier",
        "gtexContextScore",
        "gtexContextAdjustment",
        "gtexContextPositiveFlag",
        "gtexContextCoverageGapFlag",
        "gtexContextReason",
    ]
    merged = candidates.merge(target_audit[join_cols + add_cols], on=join_cols, how="left")
    merged["gtexMatchedGeneFlag"] = merged["gtexMatchedGeneFlag"].fillna(False).astype(bool)
    merged["gtexContextPositiveFlag"] = merged["gtexContextPositiveFlag"].fillna(False).astype(bool)
    merged["gtexContextCoverageGapFlag"] = merged["gtexContextCoverageGapFlag"].fillna(True).astype(bool)
    merged["gtexContextTier"] = merged["gtexContextTier"].fillna("U_no_gtex_gene_match")
    merged["gtexContextScore"] = num_col(merged, "gtexContextScore", 50.0)
    merged["gtexContextAdjustment"] = num_col(merged, "gtexContextAdjustment", 0.0)
    base_score_col = "sotaContextScore" if "sotaContextScore" in merged else "sotaNetworkScore"
    merged["sotaGtexContextScore"] = (num_col(merged, base_score_col) + merged["gtexContextAdjustment"]).clip(0, 100).round(4)
    actions = merged.apply(classify_action, axis=1)
    merged["sotaGtexContextAction"] = [item[0] for item in actions]
    merged["sotaGtexContextActionNote"] = [item[1] for item in actions]
    merged["sotaGtexContextTier"] = merged.apply(classify_gtex_integrated_tier, axis=1)
    merged = merged.sort_values(["sotaGtexContextScore", base_score_col], ascending=[False, False]).reset_index(drop=True).copy()
    merged["sotaGtexContextRankGlobal"] = range(1, len(merged) + 1)
    merged["sotaGtexContextRankWithinDirection"] = (
        merged.groupby("direction")["sotaGtexContextScore"].rank(method="first", ascending=False).astype(int)
    )
    front = [
        "sotaGtexContextRankGlobal",
        "sotaGtexContextRankWithinDirection",
        "sotaGtexContextScore",
        "sotaGtexContextTier",
        "sotaGtexContextAction",
        "sotaGtexContextActionNote",
        "gtexContextScore",
        "gtexContextAdjustment",
        "gtexContextTier",
        "gtexContextPositiveFlag",
        "gtexMatchedGeneFlag",
        "gtexMaxTpmRelevantTissues",
        "gtexTopRelevantTissuesByTpm",
    ]
    ordered = front + [col for col in merged.columns if col not in front]
    return merged[ordered]


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
    ranked = df.sort_values("sotaGtexContextScore", ascending=False).reset_index(drop=True)
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
                "gtexPositiveRows": int(top["gtexContextPositiveFlag"].sum()),
                "gtexMatchedRows": int(top["gtexMatchedGeneFlag"].sum()),
                "hpaPositiveRows": int(top["tissueContextPositiveFlag"].sum()) if "tissueContextPositiveFlag" in top else 0,
                "networkPositiveRows": int(top["networkPositiveFlag"].sum()) if "networkPositiveFlag" in top else 0,
                "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top else 0,
                "tierCounts": dict(Counter(top["sotaGtexContextTier"].astype(str))),
                "gtexTierCounts": dict(Counter(top["gtexContextTier"].astype(str))),
                "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else int(top["drug"].nunique()),
                "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else int(top["target"].nunique()),
            }
        )
    for direction, group in df.groupby("direction"):
        ranked_dir = group.sort_values("sotaGtexContextScore", ascending=False).reset_index(drop=True)
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
                    "gtexPositiveRows": int(top["gtexContextPositiveFlag"].sum()),
                    "gtexMatchedRows": int(top["gtexMatchedGeneFlag"].sum()),
                    "hpaPositiveRows": int(top["tissueContextPositiveFlag"].sum()) if "tissueContextPositiveFlag" in top else 0,
                    "networkPositiveRows": int(top["networkPositiveFlag"].sum()) if "networkPositiveFlag" in top else 0,
                    "novelRows": int(top["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top else 0,
                    "tierCounts": dict(Counter(top["sotaGtexContextTier"].astype(str))),
                    "gtexTierCounts": dict(Counter(top["gtexContextTier"].astype(str))),
                    "uniqueDrugs": int(top["drugId"].nunique()) if "drugId" in top else int(top["drug"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()) if "protein" in top else int(top["target"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def direction_summary(df: pd.DataFrame, target_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_counts = target_audit.groupby("direction").size().to_dict()
    target_positive = target_audit.groupby("direction")["gtexContextPositiveFlag"].sum().to_dict()
    for direction, group in df.groupby("direction"):
        top100 = group.sort_values("sotaGtexContextScore", ascending=False).head(min(100, len(group)))
        gtex_positive = int(group["gtexContextPositiveFlag"].sum())
        gtex_matched = int(group["gtexMatchedGeneFlag"].sum())
        rows.append(
            {
                "direction": direction,
                "directionLabelZhFinal": group["directionLabelZhFinal"].iloc[0] if "directionLabelZhFinal" in group else "",
                "candidateRows": int(len(group)),
                "targetDirectionRows": int(target_counts.get(direction, 0)),
                "targetDirectionGtexPositiveRows": int(target_positive.get(direction, 0)),
                "candidateGtexMatchedRows": gtex_matched,
                "candidateGtexMatchedPct": round(pct(gtex_matched, len(group)), 4),
                "candidateGtexPositiveRows": gtex_positive,
                "candidateGtexPositivePct": round(pct(gtex_positive, len(group)), 4),
                "medianRelevantTpm": round(float(group["gtexMaxTpmRelevantTissues"].median()), 4),
                "maxRelevantTpm": round(float(group["gtexMaxTpmRelevantTissues"].max()), 4),
                "gtexTierCounts": dict(Counter(group["gtexContextTier"].astype(str))),
                "integratedTierCounts": dict(Counter(group["sotaGtexContextTier"].astype(str))),
                "top100GtexPositiveRows": int(top100["gtexContextPositiveFlag"].sum()),
                "top100GtexMatchedRows": int(top100["gtexMatchedGeneFlag"].sum()),
                "top100HpaPositiveRows": int(top100["tissueContextPositiveFlag"].sum()) if "tissueContextPositiveFlag" in top100 else 0,
                "top100KnownRows": int(num_col(top100, "knownDrugTargetPair").sum()),
                "top100NovelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top100 else 0,
                "relevantGtexTissuePanel": ";".join(GTEX_RELEVANT_TISSUES_BY_DIRECTION.get(direction, [])),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    candidates: pd.DataFrame,
    context: pd.DataFrame,
    target_audit: pd.DataFrame,
    protein_summary: pd.DataFrame,
    gtex: pd.DataFrame,
    tissue_cols: list[str],
    source_path: Path,
) -> dict[str, Any]:
    old_auc, old_ap = safe_auc_ap(context, "sotaContextScore" if "sotaContextScore" in context else "sotaNetworkScore")
    new_auc, new_ap = safe_auc_ap(context, "sotaGtexContextScore")
    top100 = context.sort_values("sotaGtexContextScore", ascending=False).head(min(100, len(context)))
    target_positive = int(target_audit["gtexContextPositiveFlag"].sum())
    candidate_positive = int(context["gtexContextPositiveFlag"].sum())
    gtex_candidate_matched = int(context["gtexMatchedGeneFlag"].sum())
    gtex_target_matched = int(protein_summary["gtexMatchedGeneFlag"].sum())
    available_tissues = set(tissue_cols)
    map_missing = {
        direction: sorted(set(tissues) - available_tissues)
        for direction, tissues in GTEX_RELEVANT_TISSUES_BY_DIRECTION.items()
    }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "GTEx V10 median-TPM tissue-expression context over final SOTA-context candidates.",
        "source": str(source_path),
        "gtexRelease": "GTEx Analysis V10",
        "gtexFile": source_path.name,
        "interpretationNote": "This layer checks independent bulk-tissue expression plausibility. It is not disease-causality evidence and does not prove drug binding, clinical efficacy, or safety.",
        "candidateRows": int(len(context)),
        "inputCandidateRows": int(len(candidates)),
        "uniqueTargets": int(context["protein"].nunique()) if "protein" in context else int(context["target"].nunique()),
        "targetDirectionRows": int(len(target_audit)),
        "gtexExpressionRows": int(len(gtex)),
        "gtexUniqueGeneSymbols": int(gtex["symbolNorm"].nunique()),
        "gtexTissues": int(len(tissue_cols)),
        "candidateGtexMatchedRows": gtex_candidate_matched,
        "candidateGtexMatchedPct": round(pct(gtex_candidate_matched, len(context)), 4),
        "uniqueTargetGtexMatchedRows": gtex_target_matched,
        "uniqueTargetGtexMatchedPct": round(pct(gtex_target_matched, len(protein_summary)), 4),
        "candidateGtexPositiveRows": candidate_positive,
        "candidateGtexPositivePct": round(pct(candidate_positive, len(context)), 4),
        "targetDirectionGtexPositiveRows": target_positive,
        "targetDirectionGtexPositivePct": round(pct(target_positive, len(target_audit)), 4),
        "candidateGtexTierCounts": dict(Counter(context["gtexContextTier"].astype(str))),
        "targetDirectionGtexTierCounts": dict(Counter(target_audit["gtexContextTier"].astype(str))),
        "sotaGtexContextTierCounts": dict(Counter(context["sotaGtexContextTier"].astype(str))),
        "sotaGtexContextActionCounts": dict(Counter(context["sotaGtexContextAction"].astype(str))),
        "oldSotaContextAuroc": old_auc,
        "oldSotaContextAveragePrecision": old_ap,
        "sotaGtexContextAuroc": new_auc,
        "sotaGtexContextAveragePrecision": new_ap,
        "top100": {
            "knownDrugTargetRows": int(num_col(top100, "knownDrugTargetPair").sum()),
            "novelRows": int(top100["noveltyClass"].astype(str).isin(NOVEL_CLASSES).sum()) if "noveltyClass" in top100 else 0,
            "gtexPositiveRows": int(top100["gtexContextPositiveFlag"].sum()),
            "gtexMatchedRows": int(top100["gtexMatchedGeneFlag"].sum()),
            "hpaPositiveRows": int(top100["tissueContextPositiveFlag"].sum()) if "tissueContextPositiveFlag" in top100 else 0,
            "networkPositiveRows": int(top100["networkPositiveFlag"].sum()) if "networkPositiveFlag" in top100 else 0,
            "tierCounts": dict(Counter(top100["sotaGtexContextTier"].astype(str))),
            "gtexTierCounts": dict(Counter(top100["gtexContextTier"].astype(str))),
            "uniqueDrugs": int(top100["drugId"].nunique()) if "drugId" in top100 else int(top100["drug"].nunique()),
            "uniqueTargets": int(top100["protein"].nunique()) if "protein" in top100 else int(top100["target"].nunique()),
        },
        "relevantGtexTissuesByDirection": GTEX_RELEVANT_TISSUES_BY_DIRECTION,
        "missingMappedGtexTissuesByDirection": map_missing,
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame) -> str:
    lines = [
        "# GTEx Tissue Context Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit adds an independent GTEx V10 median-TPM tissue-expression plausibility layer to the final SOTA-context candidate matrix.",
        "",
        "## Summary",
        "",
        f"- Candidate rows audited: {summary['candidateRows']}",
        f"- Unique targets audited: {summary['uniqueTargets']}",
        f"- Target-direction rows audited: {summary['targetDirectionRows']}",
        f"- GTEx exact gene-symbol match in candidates: {summary['candidateGtexMatchedRows']} ({pct_str(summary['candidateGtexMatchedPct'])})",
        f"- GTEx exact gene-symbol match in unique targets: {summary['uniqueTargetGtexMatchedRows']} ({pct_str(summary['uniqueTargetGtexMatchedPct'])})",
        f"- Candidate rows with A/B GTEx context support: {summary['candidateGtexPositiveRows']} ({pct_str(summary['candidateGtexPositivePct'])})",
        f"- Target-direction rows with A/B GTEx context support: {summary['targetDirectionGtexPositiveRows']} ({pct_str(summary['targetDirectionGtexPositivePct'])})",
        f"- SOTA-context AP to SOTA-GTEx-context AP: {fmt(summary['oldSotaContextAveragePrecision'], 4)} -> {fmt(summary['sotaGtexContextAveragePrecision'], 4)}",
        f"- Top100 GTEx-positive rows: {summary['top100']['gtexPositiveRows']}; GTEx-matched rows: {summary['top100']['gtexMatchedRows']}; HPA-positive rows: {summary['top100']['hpaPositiveRows']}; known rows: {summary['top100']['knownDrugTargetRows']}; novel rows: {summary['top100']['novelRows']}",
        "",
        "## Direction Summary",
        "",
        "| Direction | Candidates | Target-direction rows | Candidate A/B GTEx support | Top100 A/B support | Median relevant TPM | Relevant GTEx tissue panel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in direction_df.sort_values("direction").itertuples(index=False):
        lines.append(
            f"| {row.direction} | {row.candidateRows} | {row.targetDirectionRows} | "
            f"{row.candidateGtexPositiveRows} ({pct_str(row.candidateGtexPositivePct)}) | "
            f"{row.top100GtexPositiveRows} | {fmt(row.medianRelevantTpm, 2)} | {row.relevantGtexTissuePanel} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/B GTEx support means the target has detectable GTEx median TPM expression in the tissue panel mapped to that disease direction.",
            "- GTEx is an independent tissue-context corroboration layer for HPA; disagreement should be treated as review context rather than a hard rejection rule.",
            "- This layer is orthogonal to ConPLex, DiffDock, Open Targets, TxGNN, network medicine, HPA, and DepMap. It strengthens expert triage by checking whether the target appears in the expected human tissue context.",
            "",
            "## Machine-Readable Outputs",
            "",
            "- Candidate audit: `outputs/sota_validation/gtex_context/candidate_gtex_context_audit.csv`",
            "- Target-direction audit: `outputs/sota_validation/gtex_context/target_gtex_context_audit.csv`",
            "- Protein expression summary: `outputs/sota_validation/gtex_context/protein_gtex_expression_summary.csv`",
            "- Integrated matrix: `outputs/sota_validation/final_prioritization/final_priority_gtex_context_matrix.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GTEx tissue-expression context audit for final candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--candidates",
        default="outputs/sota_validation/final_prioritization/final_priority_sota_context_matrix.csv",
    )
    parser.add_argument(
        "--gtex",
        default="data/external/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/gtex_context")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates = pd.read_csv(root / args.candidates).fillna("")
    gtex, tissue_cols = read_gtex(root / args.gtex)
    gtex_by_symbol = build_expression_lookup(gtex, tissue_cols)
    target_df = target_direction_rows(candidates)
    target_audit = audit_target_direction(target_df, gtex_by_symbol, tissue_cols)
    protein_summary = build_protein_summary(target_df, gtex_by_symbol, tissue_cols)
    context = build_context_matrix(candidates, target_audit)
    topk = topk_metrics(context)
    direction_df = direction_summary(context, target_audit)
    summary = build_summary(candidates, context, target_audit, protein_summary, gtex, tissue_cols, root / args.gtex)

    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    target_audit.to_csv(out_dir / "target_gtex_context_audit.csv", index=False)
    protein_summary.to_csv(out_dir / "protein_gtex_expression_summary.csv", index=False)
    context.to_csv(out_dir / "candidate_gtex_context_audit.csv", index=False)
    topk.to_csv(out_dir / "gtex_context_topk_metrics.csv", index=False)
    direction_df.to_csv(out_dir / "gtex_context_direction_summary.csv", index=False)
    write_json(out_dir / "gtex_context_summary.json", summary)
    (out_dir / "GTEX_CONTEXT_AUDIT.md").write_text(markdown(summary, direction_df), encoding="utf-8")

    context.to_csv(final_dir / "final_priority_gtex_context_matrix.csv", index=False)
    context.head(300).to_csv(final_dir / "final_priority_gtex_context_top300_expert_shortlist.csv", index=False)
    novel_mask = context["noveltyClass"].astype(str).isin(NOVEL_CLASSES) if "noveltyClass" in context else pd.Series(False, index=context.index)
    context[novel_mask].head(300).to_csv(final_dir / "final_priority_gtex_context_novel_shortlist.csv", index=False)
    context[context["sotaGtexContextAction"].isin(["gtex_context_mismatch_review", "gtex_gene_coverage_gap_review"])].head(300).to_csv(
        final_dir / "final_priority_gtex_context_review_queue.csv", index=False
    )
    direction_df.to_csv(final_dir / "final_priority_gtex_context_direction_summary.csv", index=False)
    (final_dir / "FINAL_PRIORITY_GTEX_CONTEXT_AUDIT.md").write_text(markdown(summary, direction_df), encoding="utf-8")

    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

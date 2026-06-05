from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DIRECTION_ICD_CODES = {
    "oncology": {"02"},
    "infectious_disease": {"01"},
    "cardiovascular": {"11"},
    "neurology_psychiatry": {"08", "06"},
    "immunology_inflammation": {"04", "15"},
}

CUTOFFS = [20, 50, 100, 200, 500, 1000]


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


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def split_terms(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[;|,/]+", norm(value)) if part.strip()]


def icd_codes(value: Any) -> set[str]:
    codes = set()
    for part in split_terms(value):
        match = re.match(r"^(\d{2})\b", part)
        if match:
            codes.add(match.group(1))
    return codes


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


def phase_score(phase: Any) -> float:
    parsed = number(phase)
    if parsed is None:
        return 20.0
    return max(0.0, min(100.0, parsed / 4.0 * 100.0))


def modality_flags(modalities: Any) -> dict[str, int]:
    lower = norm(modalities).lower()
    return {
        "smallMoleculeModality": int("small molecule" in lower),
        "antibodyModality": int("antibody" in lower),
        "protacDegraderModality": int("protac" in lower or "degrader" in lower),
        "otherClinicalModality": int("other clinical" in lower),
        "hasAnyModality": int(bool(lower)),
    }


def modality_score(flags: dict[str, int]) -> float:
    if flags["smallMoleculeModality"]:
        return 100.0
    if flags["protacDegraderModality"] or flags["antibodyModality"] or flags["otherClinicalModality"]:
        return 70.0
    return 30.0


def target_class_score(target_class: Any) -> float:
    lower = norm(target_class).lower()
    if not lower:
        return 40.0
    if any(term in lower for term in ["enzyme", "membrane receptor", "ion channel", "transporter"]):
        return 100.0
    if "epigenetic regulator" in lower:
        return 85.0
    if "transcription factor" in lower:
        return 70.0
    if any(term in lower for term in ["secreted protein", "surface antigen", "adhesion"]):
        return 65.0
    if "unclassified" in lower:
        return 55.0
    return 60.0


def disease_context_score(direction: str, disease_classes: Any) -> tuple[float, int, str]:
    codes = icd_codes(disease_classes)
    wanted = DIRECTION_ICD_CODES.get(direction, set())
    if codes & wanted:
        return 100.0, 1, ";".join(sorted(codes & wanted))
    if codes:
        return 70.0, 0, ""
    return 45.0, 0, ""


def druggability_tier(score: float, phase: Any, flags: dict[str, int], direction_fit: int) -> tuple[str, str]:
    parsed_phase = number(phase) or 0.0
    if score >= 85.0 and parsed_phase >= 4.0 and flags["smallMoleculeModality"] and direction_fit:
        return "A_established_small_molecule_direction_target", "phase4_small_molecule_target_with_direction_context"
    if score >= 80.0 and parsed_phase >= 4.0 and flags["smallMoleculeModality"]:
        return "A_established_small_molecule_target", "phase4_small_molecule_target"
    if score >= 70.0 and parsed_phase >= 3.0:
        return "B_clinical_or_multimodal_target", "clinical_stage_or_multimodal_target"
    if score >= 55.0:
        return "C_druggable_but_context_review", "druggable_target_with_context_gap"
    return "D_weak_target_druggability_context", "weak_or_missing_target_druggability_context"


def safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    y_true = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_ap(labels: pd.Series, scores: pd.Series) -> float | None:
    y_true = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def topk_metrics(df: pd.DataFrame, score_col: str, label_col: str) -> list[dict[str, Any]]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    total = len(ranked)
    labels = pd.to_numeric(ranked[label_col], errors="coerce").fillna(0).astype(int)
    positives = int(labels.sum())
    base_rate = positives / total if total else 0.0
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        if cutoff > total:
            continue
        top = ranked.head(cutoff)
        hits = int(pd.to_numeric(top[label_col], errors="coerce").fillna(0).astype(int).sum())
        expected = cutoff * base_rate
        rows.append(
            {
                "scoreColumn": score_col,
                "cutoff": cutoff,
                "hits": hits,
                "positives": positives,
                "precisionPct": round(pct(hits, cutoff), 4),
                "recallPct": round(pct(hits, positives), 4),
                "randomExpectedHits": round(expected, 4),
                "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
            }
        )
    return rows


def read_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_excel(path, sheet_name="ChEMBL_Targets").fillna("")
    catalog["UniProt_ID"] = catalog["UniProt_ID"].astype(str).str.strip()
    catalog["Gene_Symbol"] = catalog["Gene_Symbol"].astype(str).str.strip()
    catalog["catalogIcdCodes"] = catalog["Disease_ICD11_Classes"].apply(lambda value: ";".join(sorted(icd_codes(value))))
    return catalog


def build_audit(final: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    cat = catalog.add_prefix("catalog")
    merged = final.merge(cat, how="left", left_on="protein", right_on="catalogUniProt_ID")
    merged["targetExactUniProtMatch"] = merged["catalogUniProt_ID"].astype(str).ne("").astype(int)
    merged["targetGeneSymbolMatch"] = (
        merged["target"].astype(str).str.upper().eq(merged["catalogGene_Symbol"].astype(str).str.upper())
    ).astype(int)
    merged["targetGeneSymbolHasMultipleAccessions"] = (
        merged["target"].astype(str).str.upper().map(
            catalog.groupby(catalog["Gene_Symbol"].astype(str).str.upper())["UniProt_ID"].nunique().to_dict()
        ).fillna(0).astype(int)
        > 1
    ).astype(int)

    flags = merged["catalogDruggable_Modalities"].apply(modality_flags)
    for key in ["smallMoleculeModality", "antibodyModality", "protacDegraderModality", "otherClinicalModality", "hasAnyModality"]:
        merged[key] = flags.apply(lambda item: item[key])

    phase_scores = merged["catalogMax_Clinical_Phase"].apply(phase_score)
    modality_scores = flags.apply(modality_score)
    class_scores = merged["catalogTarget_Class"].apply(target_class_score)
    disease_scores = []
    direction_fits = []
    matched_codes = []
    for _, row in merged.iterrows():
        score, fit, codes = disease_context_score(str(row.get("direction", "")), row.get("catalogDisease_ICD11_Classes", ""))
        disease_scores.append(score)
        direction_fits.append(fit)
        matched_codes.append(codes)
    merged["targetClinicalPhaseScore"] = phase_scores
    merged["targetModalityScore"] = modality_scores
    merged["targetClassScore"] = class_scores
    merged["targetDiseaseDirectionScore"] = disease_scores
    merged["targetDiseaseDirectionFit"] = direction_fits
    merged["targetDiseaseDirectionMatchedIcdCodes"] = matched_codes
    merged["targetDruggabilityScore"] = (
        0.35 * merged["targetClinicalPhaseScore"]
        + 0.30 * merged["targetModalityScore"]
        + 0.20 * merged["targetClassScore"]
        + 0.15 * merged["targetDiseaseDirectionScore"]
    ).round(4)

    tiers = [
        druggability_tier(float(score), phase, flag, int(fit))
        for score, phase, flag, fit in zip(
            merged["targetDruggabilityScore"],
            merged["catalogMax_Clinical_Phase"],
            flags,
            merged["targetDiseaseDirectionFit"],
            strict=False,
        )
    ]
    merged["targetDruggabilityTier"] = [tier for tier, _ in tiers]
    merged["targetDruggabilityReason"] = [reason for _, reason in tiers]
    merged["finalPriorityScore"] = pd.to_numeric(merged["finalPriorityScore"], errors="coerce").fillna(0.0)
    merged["targetDruggabilityPenalty"] = 0.0
    merged.loc[merged["targetExactUniProtMatch"].eq(0), "targetDruggabilityPenalty"] += 20.0
    merged.loc[merged["smallMoleculeModality"].eq(0), "targetDruggabilityPenalty"] += 6.0
    merged.loc[merged["targetDiseaseDirectionFit"].eq(0), "targetDruggabilityPenalty"] += 4.0
    merged.loc[pd.to_numeric(merged["catalogMax_Clinical_Phase"], errors="coerce").fillna(0).lt(3), "targetDruggabilityPenalty"] += 5.0
    merged["targetAdjustedPriorityScore"] = (
        0.88 * merged["finalPriorityScore"] + 0.12 * merged["targetDruggabilityScore"] - merged["targetDruggabilityPenalty"]
    ).round(4)
    merged = merged.sort_values("targetAdjustedPriorityScore", ascending=False).reset_index(drop=True)
    merged["targetAdjustedRankGlobal"] = np.arange(1, len(merged) + 1)
    merged["targetAdjustedRankWithinDirection"] = (
        merged.groupby("direction")["targetAdjustedPriorityScore"].rank(method="first", ascending=False).astype(int)
    )
    return merged


def build_target_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protein, group in audit.groupby("protein"):
        first = group.sort_values("finalPriorityScore", ascending=False).iloc[0]
        rows.append(
            {
                "protein": protein,
                "target": first.get("target", ""),
                "proteinName": first.get("proteinName", ""),
                "candidateRows": int(len(group)),
                "directions": ";".join(sorted(group["direction"].astype(str).unique())),
                "directionCount": int(group["direction"].nunique()),
                "bestFinalRankGlobal": int(pd.to_numeric(group["finalRankGlobal"], errors="coerce").min()),
                "bestFinalPriorityScore": round(float(pd.to_numeric(group["finalPriorityScore"], errors="coerce").max()), 4),
                "targetDruggabilityScore": first.get("targetDruggabilityScore", ""),
                "targetDruggabilityTier": first.get("targetDruggabilityTier", ""),
                "maxClinicalPhase": first.get("catalogMax_Clinical_Phase", ""),
                "targetClass": first.get("catalogTarget_Class", ""),
                "druggableModalities": first.get("catalogDruggable_Modalities", ""),
                "smallMoleculeModality": int(first.get("smallMoleculeModality", 0)),
                "diseaseIcdClasses": first.get("catalogDisease_ICD11_Classes", ""),
                "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["bestFinalPriorityScore", "candidateRows"], ascending=[False, False])


def build_direction_summary(audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, group in audit.groupby("direction"):
        phase = pd.to_numeric(group["catalogMax_Clinical_Phase"], errors="coerce").fillna(0)
        rows.append(
            {
                "direction": direction,
                "candidateRows": int(len(group)),
                "uniqueTargets": int(group["protein"].nunique()),
                "exactUniProtMatchRows": int(group["targetExactUniProtMatch"].sum()),
                "exactUniProtMatchPct": round(pct(int(group["targetExactUniProtMatch"].sum()), len(group)), 4),
                "phase4Rows": int(phase.eq(4).sum()),
                "phase3Or4Rows": int(phase.ge(3).sum()),
                "smallMoleculeModalityRows": int(group["smallMoleculeModality"].sum()),
                "smallMoleculeModalityPct": round(pct(int(group["smallMoleculeModality"].sum()), len(group)), 4),
                "directionDiseaseFitRows": int(group["targetDiseaseDirectionFit"].sum()),
                "directionDiseaseFitPct": round(pct(int(group["targetDiseaseDirectionFit"].sum()), len(group)), 4),
                "medianTargetDruggabilityScore": round(float(group["targetDruggabilityScore"].median()), 4),
                "tierCounts": dict(Counter(group["targetDruggabilityTier"].astype(str))),
            }
        )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final Priority Target Druggability Audit",
            "",
            f"Generated: {summary.get('created_utc')}",
            "",
            "## Method",
            "",
            "Final candidates were matched to the ChEMBL druggable-proteome table by exact UniProt accession. Gene-symbol multiplicity is reported only as a caution field and is not used as the primary match key.",
            "",
            "## Summary",
            "",
            f"- Candidate rows: {summary.get('candidateRows')}",
            f"- Unique candidate targets: {summary.get('uniqueTargets')}",
            f"- Exact UniProt match rows: {summary.get('exactUniProtMatchRows')} ({summary.get('exactUniProtMatchPct')}%)",
            f"- Phase 4 rows: {summary.get('phase4Rows')} ({summary.get('phase4Pct')}%)",
            f"- Phase 3/4 rows: {summary.get('phase3Or4Rows')} ({summary.get('phase3Or4Pct')}%)",
            f"- Small-molecule modality rows: {summary.get('smallMoleculeModalityRows')} ({summary.get('smallMoleculeModalityPct')}%)",
            f"- Direction disease-context fit rows: {summary.get('directionDiseaseFitRows')} ({summary.get('directionDiseaseFitPct')}%)",
            f"- Target druggability tier counts: {summary.get('targetDruggabilityTierCounts')}",
            f"- Final AP vs target-adjusted AP: {summary.get('originalAveragePrecision')} / {summary.get('targetAdjustedAveragePrecision')}",
            f"- Final Recall@100 vs target-adjusted Recall@100: {summary.get('originalRecallAt100Pct')}% / {summary.get('targetAdjustedRecallAt100Pct')}%",
            f"- High-translatability shortlist rows: {summary.get('highTranslatabilityShortlistRows')}",
            f"- Low-phase/context review rows: {summary.get('lowPhaseOrContextReviewRows')}",
            "",
            "## Outputs",
            "",
            "- Candidate audit: `outputs/sota_validation/final_prioritization/final_priority_target_druggability_audit.csv`",
            "- Target summary: `outputs/sota_validation/target_druggability/target_druggability_target_summary.csv`",
            "- Direction summary: `outputs/sota_validation/target_druggability/target_druggability_direction_summary.csv`",
            "- Target-adjusted table: `outputs/sota_validation/final_prioritization/final_priority_target_druggability_augmented_table.csv`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit target-level druggability and clinical-stage context.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--catalog", default="druggable_proteome_chembl(1).xlsx")
    parser.add_argument("--final-table", default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/target_druggability")
    parser.add_argument("--final-out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    final_out_dir = root / args.final_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_out_dir.mkdir(parents=True, exist_ok=True)

    catalog = read_catalog(root / args.catalog)
    final = pd.read_csv(root / args.final_table).fillna("")
    audit = build_audit(final, catalog)

    audit_cols = [
        "targetAdjustedRankGlobal",
        "targetAdjustedRankWithinDirection",
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "pairId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "finalPriorityScore",
        "targetAdjustedPriorityScore",
        "targetDruggabilityScore",
        "targetDruggabilityTier",
        "targetDruggabilityReason",
        "targetExactUniProtMatch",
        "targetGeneSymbolMatch",
        "targetGeneSymbolHasMultipleAccessions",
        "catalogMax_Clinical_Phase",
        "catalogTarget_Class",
        "catalogDruggable_Modalities",
        "smallMoleculeModality",
        "antibodyModality",
        "protacDegraderModality",
        "otherClinicalModality",
        "targetDiseaseDirectionFit",
        "targetDiseaseDirectionMatchedIcdCodes",
        "catalogDisease_ICD11_Classes",
        "finalPriorityTier",
        "reviewTrack",
        "noveltyClass",
        "knownDrugTargetPair",
    ]
    audit[[col for col in audit_cols if col in audit.columns]].to_csv(
        final_out_dir / "final_priority_target_druggability_audit.csv", index=False
    )
    audit.to_csv(final_out_dir / "final_priority_target_druggability_augmented_table.csv", index=False)

    target_summary = build_target_summary(audit)
    target_summary.to_csv(out_dir / "target_druggability_target_summary.csv", index=False)
    direction_summary = build_direction_summary(audit)
    write_csv(out_dir / "target_druggability_direction_summary.csv", direction_summary)

    high_shortlist = audit[
        audit["targetDruggabilityTier"].astype(str).str.startswith(("A_", "B_"))
        & audit["smallMoleculeModality"].eq(1)
        & audit["targetExactUniProtMatch"].eq(1)
    ].head(200)
    high_shortlist.to_csv(final_out_dir / "final_priority_target_druggability_high_translation_shortlist.csv", index=False)

    low_review = audit[
        (pd.to_numeric(audit["finalRankGlobal"], errors="coerce").fillna(999999) <= 500)
        & (
            pd.to_numeric(audit["catalogMax_Clinical_Phase"], errors="coerce").fillna(0).lt(3)
            | audit["smallMoleculeModality"].eq(0)
            | audit["targetDiseaseDirectionFit"].eq(0)
        )
    ].head(200)
    low_review.to_csv(final_out_dir / "final_priority_target_druggability_context_review.csv", index=False)

    validation_rows = topk_metrics(audit, "finalPriorityScore", "knownDrugTargetPair")
    validation_rows.extend(topk_metrics(audit, "targetAdjustedPriorityScore", "knownDrugTargetPair"))
    write_csv(final_out_dir / "final_priority_target_druggability_topk_validation.csv", validation_rows)

    phase = pd.to_numeric(audit["catalogMax_Clinical_Phase"], errors="coerce").fillna(0)
    labels = pd.to_numeric(audit["knownDrugTargetPair"], errors="coerce").fillna(0).astype(int)
    original_metrics = {row["cutoff"]: row for row in topk_metrics(audit, "finalPriorityScore", "knownDrugTargetPair")}
    adjusted_metrics = {
        row["cutoff"]: row for row in topk_metrics(audit, "targetAdjustedPriorityScore", "knownDrugTargetPair")
    }
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(audit)),
        "uniqueTargets": int(audit["protein"].nunique()),
        "catalogRows": int(len(catalog)),
        "catalogUniqueGenes": int(catalog["Gene_Symbol"].nunique()),
        "exactUniProtMatchRows": int(audit["targetExactUniProtMatch"].sum()),
        "exactUniProtMatchPct": round(pct(int(audit["targetExactUniProtMatch"].sum()), len(audit)), 4),
        "exactUniProtMatchedUniqueTargets": int(audit.loc[audit["targetExactUniProtMatch"].eq(1), "protein"].nunique()),
        "geneSymbolMultiAccessionRows": int(audit["targetGeneSymbolHasMultipleAccessions"].sum()),
        "phase4Rows": int(phase.eq(4).sum()),
        "phase4Pct": round(pct(int(phase.eq(4).sum()), len(audit)), 4),
        "phase3Or4Rows": int(phase.ge(3).sum()),
        "phase3Or4Pct": round(pct(int(phase.ge(3).sum()), len(audit)), 4),
        "smallMoleculeModalityRows": int(audit["smallMoleculeModality"].sum()),
        "smallMoleculeModalityPct": round(pct(int(audit["smallMoleculeModality"].sum()), len(audit)), 4),
        "directionDiseaseFitRows": int(audit["targetDiseaseDirectionFit"].sum()),
        "directionDiseaseFitPct": round(pct(int(audit["targetDiseaseDirectionFit"].sum()), len(audit)), 4),
        "medianTargetDruggabilityScore": round(float(audit["targetDruggabilityScore"].median()), 4),
        "targetDruggabilityTierCounts": dict(Counter(audit["targetDruggabilityTier"].astype(str))),
        "targetClassCountsTop": dict(Counter(audit["catalogTarget_Class"].astype(str)).most_common(12)),
        "modalityCountsTop": dict(Counter(audit["catalogDruggable_Modalities"].astype(str)).most_common(12)),
        "originalAuroc": safe_auc(labels, audit["finalPriorityScore"]),
        "targetAdjustedAuroc": safe_auc(labels, audit["targetAdjustedPriorityScore"]),
        "originalAveragePrecision": safe_ap(labels, audit["finalPriorityScore"]),
        "targetAdjustedAveragePrecision": safe_ap(labels, audit["targetAdjustedPriorityScore"]),
        "originalRecallAt100Pct": original_metrics.get(100, {}).get("recallPct"),
        "targetAdjustedRecallAt100Pct": adjusted_metrics.get(100, {}).get("recallPct"),
        "originalPrecisionAt100Pct": original_metrics.get(100, {}).get("precisionPct"),
        "targetAdjustedPrecisionAt100Pct": adjusted_metrics.get(100, {}).get("precisionPct"),
        "highTranslatabilityShortlistRows": int(len(high_shortlist)),
        "lowPhaseOrContextReviewRows": int(len(low_review)),
        "methodNote": "Target druggability is matched by exact UniProt accession against the ChEMBL druggable-proteome table. Gene symbol multiplicity is reported as a caution flag only.",
    }
    write_json(out_dir / "target_druggability_summary.json", summary)
    write_json(final_out_dir / "final_priority_target_druggability_summary.json", summary)
    (final_out_dir / "FINAL_PRIORITY_TARGET_DRUGGABILITY_AUDIT.md").write_text(markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": summary,
                "out_dir": args.out_dir,
                "final_out_dir": args.final_out_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

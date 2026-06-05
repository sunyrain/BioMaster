from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFFS = [10, 20, 50, 100, 200, 300, 500, 1000, 2000]
QUEUE_PATHS = {
    "full_validation_panel": "outputs/sota_validation/experimental_validation/experimental_validation_panel.csv",
    "balanced_validation_shortlist": "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_balanced_shortlist.csv",
    "novel_validation_shortlist": "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_novel_shortlist.csv",
    "positive_control_queue": "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_positive_controls.csv",
    "wave1_diverse_validation_panel": "outputs/sota_validation/final_prioritization/final_priority_validation_panel_wave1_diverse_panel.csv",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "1.0", "true", "yes", "y"}


def accession_base(accession: Any) -> str:
    text = clean(accession)
    return text.split("-", 1)[0] if "-" in text else text


def drug_base(drug_id: Any) -> str:
    text = clean(drug_id)
    return text.split("__", 1)[0] if "__" in text else text


def join_unique(values: list[Any], limit: int = 12) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean(value)
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
        if len(out) >= limit:
            break
    return "; ".join(out)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    for column in ["validationRankGlobal", "validationRankWithinDirection", "validationScore", "finalRankGlobal"]:
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    return ranked.sort_values(["validationRankGlobal", "pairId"], ascending=[True, True]).reset_index(drop=True)


def load_target_cache(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_fda_table(path: Path, sheet: str) -> pd.DataFrame:
    fda = pd.read_excel(path, sheet_name=sheet).fillna("")
    for column in fda.columns:
        if fda[column].dtype == object:
            fda[column] = fda[column].astype(str).str.strip()
    return fda


def build_fda_drug_meta(fda: pd.DataFrame) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for drug_id, group in fda.groupby("ChEMBL ID", dropna=False):
        base = clean(drug_id)
        if not base:
            continue
        meta[base] = {
            "fdaDrugRows": int(len(group)),
            "fdaDrugName": join_unique(group.get("Generic Name (INN)", pd.Series(dtype=str)).tolist(), 6),
            "brandNames": join_unique(group.get("Brand Name", pd.Series(dtype=str)).tolist(), 6),
            "approvalYears": join_unique(group.get("Approval Year", pd.Series(dtype=str)).tolist(), 8),
            "routes": join_unique(group.get("Route", pd.Series(dtype=str)).tolist(), 8),
            "therapeuticAreas": join_unique(group.get("Therapeutic Area", pd.Series(dtype=str)).tolist(), 8),
            "indications": join_unique(group.get("Indication", pd.Series(dtype=str)).tolist(), 5),
            "fdaRowsWithTarget": int(group.get("Target ChEMBL ID", pd.Series(dtype=str)).astype(str).str.strip().ne("").sum()),
        }
    return meta


def expand_fda_label_targets(fda: pd.DataFrame, target_cache: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in fda.iterrows():
        drug_id = clean(row.get("ChEMBL ID", ""))
        target_chembl_id = clean(row.get("Target ChEMBL ID", ""))
        if not drug_id or not target_chembl_id:
            continue
        cached = target_cache.get(target_chembl_id, {})
        components = cached.get("components") or []
        if not components:
            rows.append(
                {
                    "fdaRowIndex": int(idx),
                    "drugIdBase": drug_id,
                    "fdaDrugName": clean(row.get("Generic Name (INN)", "")),
                    "targetChemblId": target_chembl_id,
                    "targetName": clean(row.get("Target Name", "")),
                    "targetPrefName": clean(cached.get("pref_name", "")),
                    "targetOrganism": clean(cached.get("organism", "")),
                    "targetAccession": "",
                    "targetAccessionBase": "",
                    "targetGenes": "",
                    "componentType": "",
                    "componentDescription": "",
                    "relationship": "",
                    "actionType": clean(row.get("Action Type", "")),
                    "mechanismOfAction": clean(row.get("Mechanism of Action", "")),
                    "therapeuticArea": clean(row.get("Therapeutic Area", "")),
                    "indication": clean(row.get("Indication", "")),
                    "route": clean(row.get("Route", "")),
                    "approvalYear": clean(row.get("Approval Year", "")),
                }
            )
            continue
        for component in components:
            accession = clean(component.get("accession", ""))
            rows.append(
                {
                    "fdaRowIndex": int(idx),
                    "drugIdBase": drug_id,
                    "fdaDrugName": clean(row.get("Generic Name (INN)", "")),
                    "targetChemblId": target_chembl_id,
                    "targetName": clean(row.get("Target Name", "")),
                    "targetPrefName": clean(cached.get("pref_name", "")),
                    "targetOrganism": clean(cached.get("organism", "")),
                    "targetAccession": accession,
                    "targetAccessionBase": accession_base(accession),
                    "targetGenes": ";".join(clean(gene) for gene in (component.get("genes") or []) if clean(gene)),
                    "componentType": clean(component.get("component_type", "")),
                    "componentDescription": clean(component.get("component_description", "")),
                    "relationship": clean(component.get("relationship", "")),
                    "actionType": clean(row.get("Action Type", "")),
                    "mechanismOfAction": clean(row.get("Mechanism of Action", "")),
                    "therapeuticArea": clean(row.get("Therapeutic Area", "")),
                    "indication": clean(row.get("Indication", "")),
                    "route": clean(row.get("Route", "")),
                    "approvalYear": clean(row.get("Approval Year", "")),
                }
            )
    return rows


def build_label_indexes(label_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_drug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_drug_accession: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_accession: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        drug_id = clean(row.get("drugIdBase"))
        accession = accession_base(row.get("targetAccessionBase") or row.get("targetAccession"))
        by_drug[drug_id].append(row)
        if accession:
            by_drug_accession[(drug_id, accession)].append(row)
            by_accession[accession].append(row)
    return {
        "byDrug": by_drug,
        "byDrugAccession": by_drug_accession,
        "byAccession": by_accession,
    }


def aggregate_label_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "matchedFdaTargetChemblIds": join_unique([row.get("targetChemblId") for row in rows], 20),
        "matchedFdaTargetNames": join_unique([row.get("targetName") or row.get("targetPrefName") for row in rows], 12),
        "matchedFdaTargetGenes": join_unique([row.get("targetGenes") for row in rows], 20),
        "matchedFdaActionTypes": join_unique([row.get("actionType") for row in rows], 12),
        "matchedFdaMechanisms": join_unique([row.get("mechanismOfAction") for row in rows], 8),
        "matchedFdaIndications": join_unique([row.get("indication") for row in rows], 4),
        "matchedFdaTherapeuticAreas": join_unique([row.get("therapeuticArea") for row in rows], 8),
        "matchedFdaTargetRecordCount": len(rows),
    }


def candidate_mechanism_class(row: pd.Series, exact_rows: list[dict[str, Any]], same_drug_rows: list[dict[str, Any]], same_target_rows: list[dict[str, Any]]) -> str:
    if exact_rows:
        return "fda_label_target_recall"
    if truthy(row.get("knownDrugTargetPair")) or truthy(row.get("hasDirectDrugTargetEdge")):
        return "direct_known_or_kg_target_not_fda_label"
    if same_target_rows:
        return "clinically_labeled_target_new_drug_pair"
    if same_drug_rows:
        return "same_fda_drug_new_target_extension"
    return "missing_or_unmapped_fda_drug_label"


def audit_candidates(panel: pd.DataFrame, fda_meta: dict[str, dict[str, Any]], indexes: dict[str, Any]) -> pd.DataFrame:
    by_drug = indexes["byDrug"]
    by_drug_accession = indexes["byDrugAccession"]
    by_accession = indexes["byAccession"]
    rows: list[dict[str, Any]] = []
    for _, item in panel.iterrows():
        drug_id = drug_base(item.get("drugId"))
        accession = accession_base(item.get("protein"))
        exact_rows = by_drug_accession.get((drug_id, accession), [])
        same_drug_rows = by_drug.get(drug_id, [])
        same_target_rows = by_accession.get(accession, [])
        exact = bool(exact_rows)
        same_drug_accessions = sorted({clean(row.get("targetAccessionBase") or row.get("targetAccession")) for row in same_drug_rows if clean(row.get("targetAccessionBase") or row.get("targetAccession"))})
        same_target_label_drugs = sorted({clean(row.get("drugIdBase")) for row in same_target_rows if clean(row.get("drugIdBase"))})
        fda_drug = fda_meta.get(drug_id, {})
        audited = item.to_dict()
        audited.update(
            {
                "drugIdBase": drug_id,
                "fdaDrugMapped": int(drug_id in fda_meta),
                "fdaDrugRows": fda_drug.get("fdaDrugRows", 0),
                "fdaRowsWithTarget": fda_drug.get("fdaRowsWithTarget", 0),
                "fdaDrugName": fda_drug.get("fdaDrugName", ""),
                "fdaBrandNames": fda_drug.get("brandNames", ""),
                "fdaApprovalYears": fda_drug.get("approvalYears", ""),
                "fdaRoutes": fda_drug.get("routes", ""),
                "fdaTherapeuticAreas": fda_drug.get("therapeuticAreas", ""),
                "fdaIndications": fda_drug.get("indications", ""),
                "fdaMappedHumanOrProteinTargetAccessions": ";".join(same_drug_accessions[:60]),
                "fdaMappedHumanOrProteinTargetCount": len(same_drug_accessions),
                "candidateTargetFdaLabeledByAnyDrug": int(bool(same_target_rows)),
                "candidateTargetFdaLabelDrugCount": len(same_target_label_drugs),
                "candidateTargetFdaLabelDrugs": ";".join(same_target_label_drugs[:40]),
                "fdaLabelTargetMatch": int(exact),
                "fdaLabelMechanismClass": candidate_mechanism_class(item, exact_rows, same_drug_rows, same_target_rows),
            }
        )
        audited.update(aggregate_label_rows(exact_rows))
        if not exact:
            audited.update(
                {
                    "matchedFdaTargetChemblIds": "",
                    "matchedFdaTargetNames": "",
                    "matchedFdaTargetGenes": "",
                    "matchedFdaActionTypes": "",
                    "matchedFdaMechanisms": "",
                    "matchedFdaIndications": "",
                    "matchedFdaTherapeuticAreas": "",
                    "matchedFdaTargetRecordCount": 0,
                }
            )
        rows.append(audited)
    return sort_panel(pd.DataFrame(rows))


def topk_metrics(audit: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = sort_panel(audit)
    total = len(ordered)
    positives = int(ordered["fdaLabelTargetMatch"].sum())
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS + [total]:
        n = min(cutoff, total)
        top = ordered.head(n)
        hits = int(top["fdaLabelTargetMatch"].sum())
        expected = positives * n / total if total else 0
        rows.append(
            {
                "groupType": "global_topk",
                "groupValue": "all",
                "cutoff": n,
                "rows": total,
                "labelTargetPositives": positives,
                "labelTargetHits": hits,
                "labelTargetPrecisionPct": round(pct(hits, n), 4),
                "labelTargetRecallPct": round(pct(hits, positives), 4),
                "randomExpectedHits": round(expected, 6),
                "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                "knownDrugTargetRows": int(pd.to_numeric(top.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                "clinicallyLabeledTargetNewDrugRows": int(top["fdaLabelMechanismClass"].eq("clinically_labeled_target_new_drug_pair").sum()),
                "sameFdaDrugNewTargetExtensionRows": int(top["fdaLabelMechanismClass"].eq("same_fda_drug_new_target_extension").sum()),
                "uniqueDrugs": int(top["drugIdBase"].nunique()),
                "uniqueTargets": int(top["protein"].nunique()),
            }
        )
    for direction, group in ordered.groupby("direction", sort=True):
        group = group.sort_values(["validationRankWithinDirection", "pairId"], ascending=[True, True]).reset_index(drop=True)
        group_total = len(group)
        group_pos = int(group["fdaLabelTargetMatch"].sum())
        for cutoff in [20, 50, 100, 200, group_total]:
            n = min(cutoff, group_total)
            top = group.head(n)
            hits = int(top["fdaLabelTargetMatch"].sum())
            expected = group_pos * n / group_total if group_total else 0
            rows.append(
                {
                    "groupType": "direction_topk",
                    "groupValue": direction,
                    "cutoff": n,
                    "rows": group_total,
                    "labelTargetPositives": group_pos,
                    "labelTargetHits": hits,
                    "labelTargetPrecisionPct": round(pct(hits, n), 4),
                    "labelTargetRecallPct": round(pct(hits, group_pos), 4),
                    "randomExpectedHits": round(expected, 6),
                    "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                    "knownDrugTargetRows": int(pd.to_numeric(top.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                    "clinicallyLabeledTargetNewDrugRows": int(top["fdaLabelMechanismClass"].eq("clinically_labeled_target_new_drug_pair").sum()),
                    "sameFdaDrugNewTargetExtensionRows": int(top["fdaLabelMechanismClass"].eq("same_fda_drug_new_target_extension").sum()),
                    "uniqueDrugs": int(top["drugIdBase"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()),
                }
            )
    return rows


def group_metrics(audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ["fdaLabelMechanismClass", "direction", "validationGate", "validationTier", "assayModality", "targetDruggabilityTier", "poseInterpretabilityTier"]:
        if column not in audit.columns:
            continue
        for value, group in audit.groupby(column, dropna=False, sort=True):
            label_hits = int(group["fdaLabelTargetMatch"].sum())
            rows.append(
                {
                    "groupType": column,
                    "groupValue": clean(value),
                    "rows": int(len(group)),
                    "labelTargetRows": label_hits,
                    "labelTargetPct": round(pct(label_hits, len(group)), 4),
                    "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                    "clinicallyLabeledTargetNewDrugRows": int(group["fdaLabelMechanismClass"].eq("clinically_labeled_target_new_drug_pair").sum()),
                    "sameFdaDrugNewTargetExtensionRows": int(group["fdaLabelMechanismClass"].eq("same_fda_drug_new_target_extension").sum()),
                    "uniqueDrugs": int(group["drugIdBase"].nunique()),
                    "uniqueTargets": int(group["protein"].nunique()),
                    "medianValidationScore": round(float(pd.to_numeric(group.get("validationScore", 0), errors="coerce").fillna(0).median()), 4),
                }
            )
    return rows


def drug_summary(audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for drug_id, group in audit.groupby("drugIdBase", sort=True):
        exact = group[group["fdaLabelTargetMatch"].eq(1)]
        rows.append(
            {
                "drugIdBase": drug_id,
                "drug": join_unique(group["drug"].tolist(), 3),
                "candidateRows": int(len(group)),
                "candidateTargets": int(group["protein"].nunique()),
                "fdaDrugRows": int(pd.to_numeric(group["fdaDrugRows"], errors="coerce").fillna(0).max()),
                "fdaRowsWithTarget": int(pd.to_numeric(group["fdaRowsWithTarget"], errors="coerce").fillna(0).max()),
                "fdaMappedTargetCount": int(pd.to_numeric(group["fdaMappedHumanOrProteinTargetCount"], errors="coerce").fillna(0).max()),
                "labelTargetRows": int(exact.shape[0]),
                "labelTargetProteins": int(exact["protein"].nunique()) if not exact.empty else 0,
                "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                "clinicallyLabeledTargetNewDrugRows": int(group["fdaLabelMechanismClass"].eq("clinically_labeled_target_new_drug_pair").sum()),
                "sameFdaDrugNewTargetExtensionRows": int(group["fdaLabelMechanismClass"].eq("same_fda_drug_new_target_extension").sum()),
                "bestValidationRank": int(pd.to_numeric(group["validationRankGlobal"], errors="coerce").min()),
                "bestValidationScore": round(float(pd.to_numeric(group["validationScore"], errors="coerce").max()), 4),
                "matchedFdaActionTypes": join_unique(exact["matchedFdaActionTypes"].tolist(), 8) if not exact.empty else "",
                "matchedFdaMechanismExamples": join_unique(exact["matchedFdaMechanisms"].tolist(), 4) if not exact.empty else "",
                "validationGateCounts": dict(Counter(group["validationGate"].astype(str))),
                "directions": join_unique(group["direction"].tolist(), 8),
            }
        )
    rows.sort(key=lambda row: (-row["labelTargetRows"], row["bestValidationRank"]))
    return rows


def target_summary(audit: pd.DataFrame, indexes: dict[str, Any]) -> list[dict[str, Any]]:
    by_accession = indexes["byAccession"]
    rows: list[dict[str, Any]] = []
    for protein, group in audit.groupby("protein", sort=True):
        accession = accession_base(protein)
        label_rows = by_accession.get(accession, [])
        exact = group[group["fdaLabelTargetMatch"].eq(1)]
        rows.append(
            {
                "protein": protein,
                "target": join_unique(group["target"].tolist(), 3),
                "proteinName": join_unique(group.get("proteinName", pd.Series(dtype=str)).tolist(), 2),
                "candidateRows": int(len(group)),
                "candidateDrugs": int(group["drugIdBase"].nunique()),
                "fdaLabelDrugCountAnyDrug": len({row.get("drugIdBase") for row in label_rows}),
                "fdaLabelRecordCountAnyDrug": len(label_rows),
                "labelTargetRows": int(exact.shape[0]),
                "labelTargetDrugs": int(exact["drugIdBase"].nunique()) if not exact.empty else 0,
                "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                "clinicallyLabeledTargetNewDrugRows": int(group["fdaLabelMechanismClass"].eq("clinically_labeled_target_new_drug_pair").sum()),
                "bestValidationRank": int(pd.to_numeric(group["validationRankGlobal"], errors="coerce").min()),
                "bestValidationScore": round(float(pd.to_numeric(group["validationScore"], errors="coerce").max()), 4),
                "fdaActionTypesAnyDrug": join_unique([row.get("actionType") for row in label_rows], 8),
                "fdaMechanismExamplesAnyDrug": join_unique([row.get("mechanismOfAction") for row in label_rows], 4),
                "directions": join_unique(group["direction"].tolist(), 8),
            }
        )
    rows.sort(key=lambda row: (-row["labelTargetRows"], row["bestValidationRank"]))
    return rows


def queue_metrics(root: Path, audit: pd.DataFrame) -> list[dict[str, Any]]:
    audit_rows = audit.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for queue_name, rel_path in QUEUE_PATHS.items():
        path = root / rel_path
        if not path.exists():
            rows.append({"queue": queue_name, "exists": 0})
            continue
        queue = pd.read_csv(path).fillna("")
        if "pairId" not in queue.columns:
            rows.append({"queue": queue_name, "exists": 1, "rows": len(queue), "mappedRows": 0})
            continue
        merge_keys = ["pairId", "direction"] if "direction" in queue.columns and "direction" in audit_rows.columns else ["pairId"]
        lookup_columns = merge_keys + [
            "fdaLabelTargetMatch",
            "fdaLabelMechanismClass",
            "knownDrugTargetPair",
            "drugIdBase",
            "protein",
            "validationScore",
        ]
        lookup = (
            audit_rows[lookup_columns]
            .sort_values(["validationScore", "pairId"], ascending=[False, True])
            .drop_duplicates(subset=merge_keys, keep="first")
        )
        joined = queue[merge_keys].merge(
            lookup,
            on=merge_keys,
            how="left",
        )
        rows.append(
            {
                "queue": queue_name,
                "exists": 1,
                "rows": int(len(queue)),
                "mappedRows": int(joined["fdaLabelMechanismClass"].notna().sum()),
                "labelTargetRows": int(pd.to_numeric(joined["fdaLabelTargetMatch"], errors="coerce").fillna(0).sum()),
                "labelTargetPct": round(pct(int(pd.to_numeric(joined["fdaLabelTargetMatch"], errors="coerce").fillna(0).sum()), len(joined)), 4),
                "knownDrugTargetRows": int(pd.to_numeric(joined["knownDrugTargetPair"], errors="coerce").fillna(0).sum()),
                "clinicallyLabeledTargetNewDrugRows": int(joined["fdaLabelMechanismClass"].eq("clinically_labeled_target_new_drug_pair").sum()),
                "sameFdaDrugNewTargetExtensionRows": int(joined["fdaLabelMechanismClass"].eq("same_fda_drug_new_target_extension").sum()),
                "uniqueDrugs": int(joined["drugIdBase"].nunique()),
                "uniqueTargets": int(joined["protein"].nunique()),
                "medianValidationScore": round(float(pd.to_numeric(joined["validationScore"], errors="coerce").fillna(0).median()), 4),
                "mechanismClassCounts": dict(Counter(joined["fdaLabelMechanismClass"].dropna().astype(str))),
            }
        )
    return rows


def top_label_hits(audit: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    cols = [
        "validationRankGlobal",
        "validationRankWithinDirection",
        "direction",
        "validationScore",
        "validationTier",
        "validationGate",
        "pairId",
        "drug",
        "drugIdBase",
        "target",
        "protein",
        "proteinName",
        "knownDrugTargetPair",
        "fdaLabelMechanismClass",
        "matchedFdaTargetChemblIds",
        "matchedFdaTargetNames",
        "matchedFdaTargetGenes",
        "matchedFdaActionTypes",
        "matchedFdaMechanisms",
        "matchedFdaIndications",
        "poseInterpretabilityTier",
        "targetDruggabilityTier",
        "assayModality",
    ]
    hits = audit[audit["fdaLabelTargetMatch"].eq(1)].sort_values(["validationRankGlobal", "pairId"]).head(limit)
    return hits[[col for col in cols if col in hits.columns]].to_dict("records")


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    fda = load_fda_table(root / args.fda_xlsx, args.fda_sheet)
    target_cache = load_target_cache(root / args.target_cache)
    panel = sort_panel(pd.read_csv(root / args.panel).fillna(""))
    fda_meta = build_fda_drug_meta(fda)
    label_rows = expand_fda_label_targets(fda, target_cache)
    indexes = build_label_indexes(label_rows)
    audit = audit_candidates(panel, fda_meta, indexes)
    topk = topk_metrics(audit)
    groups = group_metrics(audit)
    drugs = drug_summary(audit)
    targets = target_summary(audit, indexes)
    queues = queue_metrics(root, audit)
    mechanism_counts = dict(Counter(audit["fdaLabelMechanismClass"].astype(str)))

    top100 = next((row for row in topk if row["groupType"] == "global_topk" and row["cutoff"] == 100), {})
    top300 = next((row for row in topk if row["groupType"] == "global_topk" and row["cutoff"] == 300), {})
    balanced = next((row for row in queues if row.get("queue") == "balanced_validation_shortlist"), {})
    wave1 = next((row for row in queues if row.get("queue") == "wave1_diverse_validation_panel"), {})
    positive = next((row for row in queues if row.get("queue") == "positive_control_queue"), {})
    label_hit_rows = audit[audit["fdaLabelTargetMatch"].eq(1)]
    human_or_protein_label_pairs = {
        (row["drugIdBase"], row["targetAccessionBase"])
        for row in label_rows
        if clean(row.get("drugIdBase")) and clean(row.get("targetAccessionBase"))
    }
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fdaRows": int(len(fda)),
        "fdaRowsWithTarget": int(fda["Target ChEMBL ID"].astype(str).str.strip().ne("").sum()),
        "uniqueFdaDrugs": int(fda["ChEMBL ID"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
        "uniqueFdaTargetChemblIds": int(fda["Target ChEMBL ID"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
        "expandedFdaDrugUniprotPairs": len(human_or_protein_label_pairs),
        "expandedFdaLabelRows": len(label_rows),
        "candidateRows": int(len(audit)),
        "uniqueCandidateDrugs": int(audit["drugIdBase"].nunique()),
        "uniqueCandidateTargets": int(audit["protein"].nunique()),
        "fdaDrugMappedRows": int(audit["fdaDrugMapped"].sum()),
        "fdaDrugMappedPct": round(pct(int(audit["fdaDrugMapped"].sum()), len(audit)), 4),
        "fdaLabelTargetMatchRows": int(audit["fdaLabelTargetMatch"].sum()),
        "fdaLabelTargetMatchPct": round(pct(int(audit["fdaLabelTargetMatch"].sum()), len(audit)), 4),
        "fdaLabelMatchedDrugs": int(label_hit_rows["drugIdBase"].nunique()),
        "fdaLabelMatchedTargets": int(label_hit_rows["protein"].nunique()),
        "candidateTargetFdaLabeledByAnyDrugRows": int(audit["candidateTargetFdaLabeledByAnyDrug"].sum()),
        "candidateTargetFdaLabeledByAnyDrugPct": round(pct(int(audit["candidateTargetFdaLabeledByAnyDrug"].sum()), len(audit)), 4),
        "mechanismClassCounts": mechanism_counts,
        "top100LabelTargetRows": top100.get("labelTargetHits"),
        "top100LabelTargetPrecisionPct": top100.get("labelTargetPrecisionPct"),
        "top100LabelTargetRecallPct": top100.get("labelTargetRecallPct"),
        "top100LabelTargetEnrichment": top100.get("enrichmentVsRandom"),
        "top300LabelTargetRows": top300.get("labelTargetHits"),
        "top300LabelTargetPrecisionPct": top300.get("labelTargetPrecisionPct"),
        "top300LabelTargetRecallPct": top300.get("labelTargetRecallPct"),
        "top300LabelTargetEnrichment": top300.get("enrichmentVsRandom"),
        "balancedLabelTargetRows": balanced.get("labelTargetRows"),
        "balancedClinicallyLabeledTargetNewDrugRows": balanced.get("clinicallyLabeledTargetNewDrugRows"),
        "wave1LabelTargetRows": wave1.get("labelTargetRows"),
        "wave1ClinicallyLabeledTargetNewDrugRows": wave1.get("clinicallyLabeledTargetNewDrugRows"),
        "positiveControlLabelTargetRows": positive.get("labelTargetRows"),
        "methodNote": (
            "FDA label mechanism audit maps FDA Target ChEMBL IDs to UniProt accessions through the local ChEMBL target "
            "component cache, then matches candidate drug-protein pairs by FDA drug ChEMBL ID and UniProt accession. "
            "It distinguishes label-target recalls from same-drug new-target extensions and clinically labeled targets "
            "paired with a different approved drug."
        ),
    }

    out_dir = root / args.out_dir
    final_dir = root / "outputs/sota_validation/final_prioritization"
    write_csv(out_dir / "fda_label_mechanism_expanded_targets.csv", label_rows)
    write_csv(out_dir / "fda_label_mechanism_candidate_audit.csv", audit.to_dict("records"))
    write_csv(out_dir / "fda_label_mechanism_topk.csv", topk)
    write_csv(out_dir / "fda_label_mechanism_group_summary.csv", groups)
    write_csv(out_dir / "fda_label_mechanism_drug_summary.csv", drugs)
    write_csv(out_dir / "fda_label_mechanism_target_summary.csv", targets)
    write_csv(out_dir / "fda_label_mechanism_queue_summary.csv", queues)
    write_csv(out_dir / "fda_label_mechanism_top_label_hits.csv", top_label_hits(audit))
    write_json(out_dir / "fda_label_mechanism_summary.json", summary)

    write_csv(final_dir / "final_priority_fda_label_mechanism_candidate_audit.csv", audit.to_dict("records"))
    write_csv(final_dir / "final_priority_fda_label_mechanism_topk.csv", topk)
    write_csv(final_dir / "final_priority_fda_label_mechanism_group_summary.csv", groups)
    write_csv(final_dir / "final_priority_fda_label_mechanism_drug_summary.csv", drugs)
    write_csv(final_dir / "final_priority_fda_label_mechanism_target_summary.csv", targets)
    write_csv(final_dir / "final_priority_fda_label_mechanism_queue_summary.csv", queues)
    write_csv(final_dir / "final_priority_fda_label_mechanism_top_label_hits.csv", top_label_hits(audit))
    write_json(final_dir / "final_priority_fda_label_mechanism_summary.json", summary)
    (final_dir / "FINAL_PRIORITY_FDA_LABEL_MECHANISM_AUDIT.md").write_text(markdown(summary, queues), encoding="utf-8")
    return {"summary": summary}


def markdown(summary: dict[str, Any], queues: list[dict[str, Any]]) -> str:
    lines = [
        "# FDA Label Mechanism Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["methodNote"],
        "",
        "## Headline Metrics",
        "",
        f"- FDA rows: {summary['fdaRows']}; rows with target annotation: {summary['fdaRowsWithTarget']}; "
        f"unique FDA drugs: {summary['uniqueFdaDrugs']}; unique FDA Target ChEMBL IDs: {summary['uniqueFdaTargetChemblIds']}.",
        f"- Expanded FDA drug-UniProt pairs: {summary['expandedFdaDrugUniprotPairs']}; expanded label rows: {summary['expandedFdaLabelRows']}.",
        f"- Candidate rows: {summary['candidateRows']}; FDA-drug mapped rows: {summary['fdaDrugMappedRows']} "
        f"({summary['fdaDrugMappedPct']}%).",
        f"- FDA label target matches: {summary['fdaLabelTargetMatchRows']} rows "
        f"({summary['fdaLabelTargetMatchPct']}%), covering {summary['fdaLabelMatchedDrugs']} drugs and "
        f"{summary['fdaLabelMatchedTargets']} targets.",
        f"- Candidate targets that are FDA-labeled for any approved drug: {summary['candidateTargetFdaLabeledByAnyDrugRows']} "
        f"({summary['candidateTargetFdaLabeledByAnyDrugPct']}%).",
        f"- Mechanism classes: {summary['mechanismClassCounts']}.",
        f"- Top100 label-target recall: {summary['top100LabelTargetRows']} rows; precision "
        f"{summary['top100LabelTargetPrecisionPct']}%; recall {summary['top100LabelTargetRecallPct']}%; "
        f"enrichment {summary['top100LabelTargetEnrichment']}x.",
        f"- Top300 label-target recall: {summary['top300LabelTargetRows']} rows; precision "
        f"{summary['top300LabelTargetPrecisionPct']}%; recall {summary['top300LabelTargetRecallPct']}%; "
        f"enrichment {summary['top300LabelTargetEnrichment']}x.",
        f"- Balanced shortlist: label-target rows {summary['balancedLabelTargetRows']}; clinically labeled target/new-drug rows "
        f"{summary['balancedClinicallyLabeledTargetNewDrugRows']}.",
        f"- Wave-1 panel: label-target rows {summary['wave1LabelTargetRows']}; clinically labeled target/new-drug rows "
        f"{summary['wave1ClinicallyLabeledTargetNewDrugRows']}.",
        "",
        "## Queue Summary",
        "",
    ]
    for row in queues:
        if not row.get("exists"):
            lines.append(f"- {row['queue']}: missing.")
            continue
        lines.append(
            f"- {row['queue']}: rows {row['rows']}; label-target rows {row['labelTargetRows']} "
            f"({row['labelTargetPct']}%); clinically labeled target/new-drug rows "
            f"{row['clinicallyLabeledTargetNewDrugRows']}; same-FDA-drug new-target rows "
            f"{row['sameFdaDrugNewTargetExtensionRows']}."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FDA label target/mechanism support for final validation candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--panel", default="outputs/sota_validation/experimental_validation/experimental_validation_panel.csv")
    parser.add_argument("--fda-xlsx", default="FDA_approved_small_molecules_2005_2026_with_structures.xlsx")
    parser.add_argument("--fda-sheet", default="FDA Small Molecules 2005-2026")
    parser.add_argument("--target-cache", default="outputs/druggable_proteome/fda_chembl_target_component_cache.json")
    parser.add_argument("--out-dir", default="outputs/sota_validation/fda_label_mechanism")
    args = parser.parse_args()
    result = build(Path(args.root).resolve(), args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

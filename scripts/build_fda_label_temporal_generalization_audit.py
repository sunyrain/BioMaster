from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFFS = [10, 20, 50, 100, 200, 300, 500, 1000, 2000]
TEMPORAL_SPLITS = [2010, 2015, 2020]
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


def parse_years(value: Any) -> list[int]:
    years: list[int] = []
    for match in re.findall(r"(?:19|20)\d{2}", clean(value)):
        year = int(match)
        if 1900 <= year <= 2100 and year not in years:
            years.append(year)
    return sorted(years)


def year_bin(year: Any) -> str:
    try:
        parsed = int(float(year))
    except (TypeError, ValueError):
        return "unknown"
    if 2005 <= parsed <= 2010:
        return "2005-2010"
    if 2011 <= parsed <= 2015:
        return "2011-2015"
    if 2016 <= parsed <= 2020:
        return "2016-2020"
    if 2021 <= parsed <= 2026:
        return "2021-2026"
    return "outside_2005_2026"


def year_era(year: Any) -> str:
    try:
        parsed = int(float(year))
    except (TypeError, ValueError):
        return "unknown"
    if parsed >= 2021:
        return "recent_2021_2026"
    if parsed >= 2016:
        return "modern_2016_2020"
    if parsed >= 2011:
        return "mid_2011_2015"
    if parsed >= 2005:
        return "early_2005_2010"
    return "outside_2005_2026"


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
    out = df.copy()
    for column in ["validationRankGlobal", "validationRankWithinDirection", "validationScore", "finalRankGlobal"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["validationRankGlobal", "pairId"], ascending=[True, True]).reset_index(drop=True)


def build_label_indexes(expanded: pd.DataFrame) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_drug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in expanded.iterrows():
        drug = clean(row.get("drugIdBase"))
        accession = accession_base(row.get("targetAccessionBase") or row.get("targetAccession"))
        if not drug:
            continue
        item = row.to_dict()
        years = parse_years(item.get("approvalYear"))
        item["approvalYearParsed"] = years[0] if years else ""
        by_drug[drug].append(item)
        if accession:
            by_pair[(drug, accession)].append(item)
            by_target[accession].append(item)
    return {"byPair": by_pair, "byTarget": by_target, "byDrug": by_drug}


def row_year_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    years: list[int] = []
    actions: list[Any] = []
    mechanisms: list[Any] = []
    indications: list[Any] = []
    areas: list[Any] = []
    targets: list[Any] = []
    drugs: list[Any] = []
    for row in rows:
        years.extend(parse_years(row.get("approvalYearParsed") or row.get("approvalYear")))
        actions.append(row.get("actionType", ""))
        mechanisms.append(row.get("mechanismOfAction", ""))
        indications.append(row.get("indication", ""))
        areas.append(row.get("therapeuticArea", ""))
        targets.append(row.get("targetName") or row.get("targetPrefName", ""))
        drugs.append(row.get("drugIdBase", ""))
    years = sorted(set(years))
    return {
        "years": years,
        "earliestYear": years[0] if years else "",
        "latestYear": years[-1] if years else "",
        "yearSpan": ";".join(str(year) for year in years),
        "actions": join_unique(actions, 10),
        "mechanisms": join_unique(mechanisms, 8),
        "indications": join_unique(indications, 4),
        "areas": join_unique(areas, 8),
        "targets": join_unique(targets, 10),
        "drugs": join_unique(drugs, 20),
        "recordCount": len(rows),
    }


def temporal_class(row: pd.Series) -> str:
    exact_latest = row.get("exactFdaLabelLatestYear", "")
    target_latest = row.get("targetAnyFdaLabelLatestYear", "")
    drug_latest = row.get("drugFdaLatestApprovalYear", "")
    if truthy(row.get("fdaLabelTargetMatch")):
        era = year_era(exact_latest)
        return f"exact_label_target_{era}"
    if truthy(row.get("candidateTargetFdaLabeledByAnyDrug")):
        era = year_era(target_latest)
        return f"clinically_labeled_target_context_{era}"
    if drug_latest != "":
        era = year_era(drug_latest)
        return f"same_fda_drug_new_target_{era}"
    return "unmapped_or_no_temporal_label"


def audit_candidates(candidates: pd.DataFrame, indexes: dict[str, Any]) -> pd.DataFrame:
    by_pair = indexes["byPair"]
    by_target = indexes["byTarget"]
    by_drug = indexes["byDrug"]
    rows: list[dict[str, Any]] = []
    for _, item in candidates.iterrows():
        drug = clean(item.get("drugIdBase") or clean(item.get("drugId")).split("__", 1)[0])
        accession = accession_base(item.get("protein"))
        exact = row_year_summary(by_pair.get((drug, accession), []))
        target = row_year_summary(by_target.get(accession, []))
        drug_rows = row_year_summary(by_drug.get(drug, []))
        drug_meta_years = parse_years(item.get("fdaApprovalYears"))
        drug_years = sorted(set(drug_rows["years"]) | set(drug_meta_years))
        out = item.to_dict()
        out.update(
            {
                "exactFdaLabelEarliestYear": exact["earliestYear"],
                "exactFdaLabelLatestYear": exact["latestYear"],
                "exactFdaLabelYearSpan": exact["yearSpan"],
                "exactFdaLabelYearBin": year_bin(exact["latestYear"]),
                "exactFdaLabelEra": year_era(exact["latestYear"]),
                "exactFdaLabelActionTypesTemporal": exact["actions"],
                "exactFdaLabelMechanismsTemporal": exact["mechanisms"],
                "exactFdaLabelIndicationsTemporal": exact["indications"],
                "exactFdaLabelRecordCountTemporal": exact["recordCount"],
                "targetAnyFdaLabelEarliestYear": target["earliestYear"],
                "targetAnyFdaLabelLatestYear": target["latestYear"],
                "targetAnyFdaLabelYearSpan": target["yearSpan"],
                "targetAnyFdaLabelEra": year_era(target["latestYear"]),
                "targetAnyFdaLabelDrugCount": len([x for x in target["drugs"].split(";") if x.strip()]),
                "targetAnyFdaLabelDrugsTemporal": target["drugs"],
                "targetAnyFdaLabelActionTypesTemporal": target["actions"],
                "targetAnyFdaLabelMechanismsTemporal": target["mechanisms"],
                "drugFdaEarliestApprovalYear": drug_years[0] if drug_years else "",
                "drugFdaLatestApprovalYear": drug_years[-1] if drug_years else "",
                "drugFdaApprovalYearSpan": ";".join(str(year) for year in drug_years),
                "drugFdaApprovalEra": year_era(drug_years[-1] if drug_years else ""),
            }
        )
        out["exactFdaLabel2016PlusFlag"] = int(bool(exact["latestYear"] != "" and int(exact["latestYear"]) >= 2016))
        out["exactFdaLabel2021PlusFlag"] = int(bool(exact["latestYear"] != "" and int(exact["latestYear"]) >= 2021))
        out["targetAnyFdaLabel2016PlusFlag"] = int(bool(target["latestYear"] != "" and int(target["latestYear"]) >= 2016))
        out["targetAnyFdaLabel2021PlusFlag"] = int(bool(target["latestYear"] != "" and int(target["latestYear"]) >= 2021))
        out["fdaTemporalMechanismClass"] = temporal_class(pd.Series(out))
        rows.append(out)
    return sort_panel(pd.DataFrame(rows))


def topk_metrics(audit: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = sort_panel(audit)
    definitions = [
        ("exact_label_all", ordered["fdaLabelTargetMatch"].map(truthy)),
        ("exact_label_2016plus", pd.to_numeric(ordered["exactFdaLabel2016PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)),
        ("exact_label_2021plus", pd.to_numeric(ordered["exactFdaLabel2021PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)),
        ("target_context_2016plus", pd.to_numeric(ordered["targetAnyFdaLabel2016PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)),
        ("target_context_2021plus", pd.to_numeric(ordered["targetAnyFdaLabel2021PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)),
    ]
    rows: list[dict[str, Any]] = []
    total = len(ordered)
    for label_set, mask in definitions:
        positives = int(mask.sum())
        for cutoff in CUTOFFS + [total]:
            n = min(cutoff, total)
            top = ordered.head(n)
            hits = int(mask.head(n).sum())
            expected = positives * n / total if total else 0
            rows.append(
                {
                    "labelSet": label_set,
                    "groupType": "global_topk",
                    "groupValue": "all",
                    "cutoff": n,
                    "rows": total,
                    "positives": positives,
                    "hits": hits,
                    "precisionPct": round(pct(hits, n), 4),
                    "recallPct": round(pct(hits, positives), 4),
                    "randomExpectedHits": round(expected, 6),
                    "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                    "uniqueDrugs": int(top["drugIdBase"].nunique()) if "drugIdBase" in top else int(top["drugId"].nunique()),
                    "uniqueTargets": int(top["protein"].nunique()),
                }
            )
    for direction, group in ordered.groupby("direction", sort=True):
        group = group.sort_values(["validationRankWithinDirection", "pairId"], ascending=[True, True]).reset_index(drop=True)
        definitions = [
            ("exact_label_all", group["fdaLabelTargetMatch"].map(truthy)),
            ("exact_label_2016plus", pd.to_numeric(group["exactFdaLabel2016PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)),
            ("exact_label_2021plus", pd.to_numeric(group["exactFdaLabel2021PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)),
        ]
        for label_set, mask in definitions:
            positives = int(mask.sum())
            for cutoff in [20, 50, 100, 200, len(group)]:
                n = min(cutoff, len(group))
                top = group.head(n)
                hits = int(mask.head(n).sum())
                expected = positives * n / len(group) if len(group) else 0
                rows.append(
                    {
                        "labelSet": label_set,
                        "groupType": "direction_topk",
                        "groupValue": direction,
                        "cutoff": n,
                        "rows": len(group),
                        "positives": positives,
                        "hits": hits,
                        "precisionPct": round(pct(hits, n), 4),
                        "recallPct": round(pct(hits, positives), 4),
                        "randomExpectedHits": round(expected, 6),
                        "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                        "uniqueDrugs": int(top["drugIdBase"].nunique()) if "drugIdBase" in top else int(top["drugId"].nunique()),
                        "uniqueTargets": int(top["protein"].nunique()),
                    }
                )
    return rows


def temporal_split_metrics(audit: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = sort_panel(audit)
    exact_year = pd.to_numeric(ordered["exactFdaLabelLatestYear"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for split_year in TEMPORAL_SPLITS:
        masks = {
            "historical_label_at_or_before_split": ordered["fdaLabelTargetMatch"].map(truthy) & exact_year.le(split_year),
            "future_label_after_split": ordered["fdaLabelTargetMatch"].map(truthy) & exact_year.gt(split_year),
        }
        for label_set, mask in masks.items():
            positives = int(mask.sum())
            for cutoff in [100, 300, 500, 1000, 2000, len(ordered)]:
                n = min(cutoff, len(ordered))
                hits = int(mask.head(n).sum())
                expected = positives * n / len(ordered) if len(ordered) else 0
                rows.append(
                    {
                        "splitYear": split_year,
                        "labelSet": label_set,
                        "cutoff": n,
                        "rows": len(ordered),
                        "positives": positives,
                        "hits": hits,
                        "precisionPct": round(pct(hits, n), 4),
                        "recallPct": round(pct(hits, positives), 4),
                        "randomExpectedHits": round(expected, 6),
                        "enrichmentVsRandom": round(hits / expected, 6) if expected else None,
                    }
                )
    return rows


def group_summary(audit: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in [
        "exactFdaLabelYearBin",
        "exactFdaLabelEra",
        "targetAnyFdaLabelEra",
        "drugFdaApprovalEra",
        "fdaTemporalMechanismClass",
        "direction",
        "validationGate",
        "validationTier",
    ]:
        if column not in audit.columns:
            continue
        for value, group in audit.groupby(column, dropna=False, sort=True):
            exact = int(group["fdaLabelTargetMatch"].map(truthy).sum())
            exact_2016 = int(pd.to_numeric(group["exactFdaLabel2016PlusFlag"], errors="coerce").fillna(0).sum())
            exact_2021 = int(pd.to_numeric(group["exactFdaLabel2021PlusFlag"], errors="coerce").fillna(0).sum())
            target_2021 = int(pd.to_numeric(group["targetAnyFdaLabel2021PlusFlag"], errors="coerce").fillna(0).sum())
            rows.append(
                {
                    "groupType": column,
                    "groupValue": clean(value),
                    "rows": int(len(group)),
                    "exactLabelRows": exact,
                    "exactLabelPct": round(pct(exact, len(group)), 4),
                    "exactLabel2016PlusRows": exact_2016,
                    "exactLabel2021PlusRows": exact_2021,
                    "targetContext2021PlusRows": target_2021,
                    "knownDrugTargetRows": int(pd.to_numeric(group.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()),
                    "uniqueDrugs": int(group["drugIdBase"].nunique()) if "drugIdBase" in group else int(group["drugId"].nunique()),
                    "uniqueTargets": int(group["protein"].nunique()),
                    "medianValidationScore": round(float(pd.to_numeric(group.get("validationScore", 0), errors="coerce").fillna(0).median()), 4),
                }
            )
    return rows


def queue_metrics(root: Path, audit: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = audit.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for queue_name, rel_path in QUEUE_PATHS.items():
        path = root / rel_path
        if not path.exists():
            rows.append({"queue": queue_name, "exists": 0})
            continue
        queue = pd.read_csv(path).fillna("")
        merge_keys = ["pairId", "direction"] if "direction" in queue.columns else ["pairId"]
        keep = merge_keys + [
            "fdaLabelTargetMatch",
            "exactFdaLabel2016PlusFlag",
            "exactFdaLabel2021PlusFlag",
            "targetAnyFdaLabel2016PlusFlag",
            "targetAnyFdaLabel2021PlusFlag",
            "fdaTemporalMechanismClass",
            "knownDrugTargetPair",
            "drugIdBase",
            "protein",
            "validationScore",
        ]
        joined = queue[merge_keys].merge(
            lookup[keep].drop_duplicates(subset=merge_keys, keep="first"),
            on=merge_keys,
            how="left",
        )
        rows.append(
            {
                "queue": queue_name,
                "exists": 1,
                "rows": int(len(queue)),
                "mappedRows": int(joined["fdaTemporalMechanismClass"].notna().sum()),
                "exactLabelRows": int(joined["fdaLabelTargetMatch"].map(truthy).sum()),
                "exactLabel2016PlusRows": int(pd.to_numeric(joined["exactFdaLabel2016PlusFlag"], errors="coerce").fillna(0).sum()),
                "exactLabel2021PlusRows": int(pd.to_numeric(joined["exactFdaLabel2021PlusFlag"], errors="coerce").fillna(0).sum()),
                "targetContext2016PlusRows": int(pd.to_numeric(joined["targetAnyFdaLabel2016PlusFlag"], errors="coerce").fillna(0).sum()),
                "targetContext2021PlusRows": int(pd.to_numeric(joined["targetAnyFdaLabel2021PlusFlag"], errors="coerce").fillna(0).sum()),
                "knownDrugTargetRows": int(pd.to_numeric(joined["knownDrugTargetPair"], errors="coerce").fillna(0).sum()),
                "uniqueDrugs": int(joined["drugIdBase"].nunique()),
                "uniqueTargets": int(joined["protein"].nunique()),
                "medianValidationScore": round(float(pd.to_numeric(joined["validationScore"], errors="coerce").fillna(0).median()), 4),
                "temporalClassCounts": dict(Counter(joined["fdaTemporalMechanismClass"].dropna().astype(str))),
            }
        )
    return rows


def select_recent_label_hits(audit: pd.DataFrame, min_year: int, limit: int = 200) -> list[dict[str, Any]]:
    exact_year = pd.to_numeric(audit["exactFdaLabelLatestYear"], errors="coerce")
    hits = audit[audit["fdaLabelTargetMatch"].map(truthy) & exact_year.ge(min_year)].sort_values(["validationRankGlobal", "pairId"]).head(limit)
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
        "knownDrugTargetPair",
        "exactFdaLabelLatestYear",
        "exactFdaLabelYearBin",
        "exactFdaLabelActionTypesTemporal",
        "exactFdaLabelMechanismsTemporal",
        "matchedFdaTargetNames",
        "poseInterpretabilityTier",
        "targetDruggabilityTier",
        "assayModality",
    ]
    return hits[[col for col in cols if col in hits.columns]].to_dict("records")


def select_recent_context_candidates(audit: pd.DataFrame, min_year: int, limit: int = 300) -> list[dict[str, Any]]:
    target_year = pd.to_numeric(audit["targetAnyFdaLabelLatestYear"], errors="coerce")
    exact = audit["fdaLabelTargetMatch"].map(truthy)
    candidates = audit[~exact & target_year.ge(min_year)].sort_values(["validationRankGlobal", "pairId"]).head(limit)
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
        "fdaLabelMechanismClass",
        "targetAnyFdaLabelLatestYear",
        "targetAnyFdaLabelActionTypesTemporal",
        "targetAnyFdaLabelMechanismsTemporal",
        "targetAnyFdaLabelDrugsTemporal",
        "poseInterpretabilityTier",
        "targetDruggabilityTier",
        "assayModality",
    ]
    return candidates[[col for col in cols if col in candidates.columns]].to_dict("records")


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    candidates = sort_panel(pd.read_csv(root / args.candidate_audit).fillna(""))
    expanded = pd.read_csv(root / args.expanded_targets).fillna("")
    indexes = build_label_indexes(expanded)
    audit = audit_candidates(candidates, indexes)
    topk = topk_metrics(audit)
    splits = temporal_split_metrics(audit)
    groups = group_summary(audit)
    queues = queue_metrics(root, audit)
    recent_hits = select_recent_label_hits(audit, min_year=2021)
    modern_hits = select_recent_label_hits(audit, min_year=2016)
    recent_context = select_recent_context_candidates(audit, min_year=2021)

    def find_topk(label_set: str, cutoff: int) -> dict[str, Any]:
        return next((row for row in topk if row["labelSet"] == label_set and row["groupType"] == "global_topk" and row["cutoff"] == cutoff), {})

    def find_split(split_year: int, label_set: str, cutoff: int) -> dict[str, Any]:
        return next((row for row in splits if row["splitYear"] == split_year and row["labelSet"] == label_set and row["cutoff"] == cutoff), {})

    balanced = next((row for row in queues if row.get("queue") == "balanced_validation_shortlist"), {})
    wave1 = next((row for row in queues if row.get("queue") == "wave1_diverse_validation_panel"), {})
    positive = next((row for row in queues if row.get("queue") == "positive_control_queue"), {})
    exact = audit[audit["fdaLabelTargetMatch"].map(truthy)]
    exact_2016 = audit[pd.to_numeric(audit["exactFdaLabel2016PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)]
    exact_2021 = audit[pd.to_numeric(audit["exactFdaLabel2021PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)]
    target_2021 = audit[pd.to_numeric(audit["targetAnyFdaLabel2021PlusFlag"], errors="coerce").fillna(0).astype(int).eq(1)]

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(audit)),
        "exactLabelRows": int(len(exact)),
        "exactLabel2016PlusRows": int(len(exact_2016)),
        "exactLabel2021PlusRows": int(len(exact_2021)),
        "exactLabel2016PlusPctOfExact": round(pct(len(exact_2016), len(exact)), 4),
        "exactLabel2021PlusPctOfExact": round(pct(len(exact_2021), len(exact)), 4),
        "targetContext2021PlusRows": int(len(target_2021)),
        "targetContext2021PlusPct": round(pct(len(target_2021), len(audit)), 4),
        "exactLabelEraCounts": dict(Counter(exact["exactFdaLabelEra"].astype(str))),
        "temporalMechanismClassCounts": dict(Counter(audit["fdaTemporalMechanismClass"].astype(str))),
        "top100Exact2016PlusRows": find_topk("exact_label_2016plus", 100).get("hits"),
        "top100Exact2016PlusRecallPct": find_topk("exact_label_2016plus", 100).get("recallPct"),
        "top100Exact2016PlusEnrichment": find_topk("exact_label_2016plus", 100).get("enrichmentVsRandom"),
        "top100Exact2021PlusRows": find_topk("exact_label_2021plus", 100).get("hits"),
        "top100Exact2021PlusRecallPct": find_topk("exact_label_2021plus", 100).get("recallPct"),
        "top100Exact2021PlusEnrichment": find_topk("exact_label_2021plus", 100).get("enrichmentVsRandom"),
        "top300Exact2016PlusRows": find_topk("exact_label_2016plus", 300).get("hits"),
        "top300Exact2016PlusRecallPct": find_topk("exact_label_2016plus", 300).get("recallPct"),
        "top300Exact2021PlusRows": find_topk("exact_label_2021plus", 300).get("hits"),
        "top300Exact2021PlusRecallPct": find_topk("exact_label_2021plus", 300).get("recallPct"),
        "split2015FutureLabelTop300Rows": find_split(2015, "future_label_after_split", 300).get("hits"),
        "split2015FutureLabelTop300RecallPct": find_split(2015, "future_label_after_split", 300).get("recallPct"),
        "split2015FutureLabelTop300Enrichment": find_split(2015, "future_label_after_split", 300).get("enrichmentVsRandom"),
        "split2020FutureLabelRows": find_split(2020, "future_label_after_split", len(audit)).get("positives"),
        "split2020FutureLabelTop100Rows": find_split(2020, "future_label_after_split", 100).get("hits"),
        "split2020FutureLabelTop300Rows": find_split(2020, "future_label_after_split", 300).get("hits"),
        "balancedExact2016PlusRows": balanced.get("exactLabel2016PlusRows"),
        "balancedExact2021PlusRows": balanced.get("exactLabel2021PlusRows"),
        "balancedTargetContext2021PlusRows": balanced.get("targetContext2021PlusRows"),
        "wave1Exact2016PlusRows": wave1.get("exactLabel2016PlusRows"),
        "wave1Exact2021PlusRows": wave1.get("exactLabel2021PlusRows"),
        "wave1TargetContext2021PlusRows": wave1.get("targetContext2021PlusRows"),
        "positiveControlExact2016PlusRows": positive.get("exactLabel2016PlusRows"),
        "positiveControlExact2021PlusRows": positive.get("exactLabel2021PlusRows"),
        "methodNote": (
            "This is a retrospective FDA-label time-sliced audit, not a true prospective deployment test. "
            "It uses FDA approval years attached to label target annotations to stress-test whether current "
            "validation ranking and assay queues recover newer label-target mechanisms and recent clinically labeled target contexts."
        ),
    }

    out_dir = root / args.out_dir
    final_dir = root / "outputs/sota_validation/final_prioritization"
    write_csv(out_dir / "fda_label_temporal_candidate_audit.csv", audit.to_dict("records"))
    write_csv(out_dir / "fda_label_temporal_topk.csv", topk)
    write_csv(out_dir / "fda_label_temporal_split_summary.csv", splits)
    write_csv(out_dir / "fda_label_temporal_group_summary.csv", groups)
    write_csv(out_dir / "fda_label_temporal_queue_summary.csv", queues)
    write_csv(out_dir / "fda_label_temporal_recent_2021plus_label_hits.csv", recent_hits)
    write_csv(out_dir / "fda_label_temporal_modern_2016plus_label_hits.csv", modern_hits)
    write_csv(out_dir / "fda_label_temporal_recent_context_candidates.csv", recent_context)
    write_json(out_dir / "fda_label_temporal_summary.json", summary)

    write_csv(final_dir / "final_priority_fda_label_temporal_candidate_audit.csv", audit.to_dict("records"))
    write_csv(final_dir / "final_priority_fda_label_temporal_topk.csv", topk)
    write_csv(final_dir / "final_priority_fda_label_temporal_split_summary.csv", splits)
    write_csv(final_dir / "final_priority_fda_label_temporal_group_summary.csv", groups)
    write_csv(final_dir / "final_priority_fda_label_temporal_queue_summary.csv", queues)
    write_csv(final_dir / "final_priority_fda_label_temporal_recent_2021plus_label_hits.csv", recent_hits)
    write_csv(final_dir / "final_priority_fda_label_temporal_modern_2016plus_label_hits.csv", modern_hits)
    write_csv(final_dir / "final_priority_fda_label_temporal_recent_context_candidates.csv", recent_context)
    write_json(final_dir / "final_priority_fda_label_temporal_summary.json", summary)
    (final_dir / "FINAL_PRIORITY_FDA_LABEL_TEMPORAL_GENERALIZATION_AUDIT.md").write_text(markdown(summary, queues), encoding="utf-8")
    return {"summary": summary}


def markdown(summary: dict[str, Any], queues: list[dict[str, Any]]) -> str:
    lines = [
        "# FDA Label Temporal Generalization Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["methodNote"],
        "",
        "## Headline Metrics",
        "",
        f"- Candidate rows: {summary['candidateRows']}; FDA exact label-target rows: {summary['exactLabelRows']}.",
        f"- Exact label-target rows from 2016 onward: {summary['exactLabel2016PlusRows']} "
        f"({summary['exactLabel2016PlusPctOfExact']}% of exact label rows).",
        f"- Exact label-target rows from 2021 onward: {summary['exactLabel2021PlusRows']} "
        f"({summary['exactLabel2021PlusPctOfExact']}% of exact label rows).",
        f"- Candidate rows whose target has any 2021+ FDA label context: {summary['targetContext2021PlusRows']} "
        f"({summary['targetContext2021PlusPct']}%).",
        f"- Exact label era counts: {summary['exactLabelEraCounts']}.",
        f"- Temporal mechanism class counts: {summary['temporalMechanismClassCounts']}.",
        f"- Top100 2016+ exact label hits: {summary['top100Exact2016PlusRows']}; recall "
        f"{summary['top100Exact2016PlusRecallPct']}%; enrichment {summary['top100Exact2016PlusEnrichment']}x.",
        f"- Top100 2021+ exact label hits: {summary['top100Exact2021PlusRows']}; recall "
        f"{summary['top100Exact2021PlusRecallPct']}%; enrichment {summary['top100Exact2021PlusEnrichment']}x.",
        f"- Top300 2016+/2021+ exact label hits: {summary['top300Exact2016PlusRows']}/{summary['top300Exact2021PlusRows']}.",
        f"- Split-2015 future-label Top300 hits: {summary['split2015FutureLabelTop300Rows']}; recall "
        f"{summary['split2015FutureLabelTop300RecallPct']}%; enrichment {summary['split2015FutureLabelTop300Enrichment']}x.",
        f"- Split-2020 future-label positives: {summary['split2020FutureLabelRows']}; Top100/Top300 hits "
        f"{summary['split2020FutureLabelTop100Rows']}/{summary['split2020FutureLabelTop300Rows']}.",
        f"- Balanced shortlist 2016+/2021+ exact labels: {summary['balancedExact2016PlusRows']}/"
        f"{summary['balancedExact2021PlusRows']}; 2021+ target-context rows: {summary['balancedTargetContext2021PlusRows']}.",
        f"- Wave-1 panel 2016+/2021+ exact labels: {summary['wave1Exact2016PlusRows']}/"
        f"{summary['wave1Exact2021PlusRows']}; 2021+ target-context rows: {summary['wave1TargetContext2021PlusRows']}.",
        "",
        "## Queue Summary",
        "",
    ]
    for row in queues:
        if not row.get("exists"):
            lines.append(f"- {row['queue']}: missing.")
            continue
        lines.append(
            f"- {row['queue']}: rows {row['rows']}; exact labels {row['exactLabelRows']}; "
            f"2016+ exact {row['exactLabel2016PlusRows']}; 2021+ exact {row['exactLabel2021PlusRows']}; "
            f"2021+ target-context {row['targetContext2021PlusRows']}."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrospective FDA-label temporal generalization audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidate-audit", default="outputs/sota_validation/fda_label_mechanism/fda_label_mechanism_candidate_audit.csv")
    parser.add_argument("--expanded-targets", default="outputs/sota_validation/fda_label_mechanism/fda_label_mechanism_expanded_targets.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/fda_label_temporal")
    args = parser.parse_args()
    result = build(Path(args.root).resolve(), args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

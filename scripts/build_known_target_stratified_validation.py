from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFFS = [100, 1000, 10000, 100000, 1000000]
YEAR_BINS = [
    ("2005-2010", 2005, 2010),
    ("2011-2015", 2011, 2015),
    ("2016-2020", 2016, 2020),
    ("2021-2026", 2021, 2026),
]


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def year_bin(year: Any) -> str:
    parsed = number(year)
    if parsed is None:
        return "unknown"
    value = int(parsed)
    for label, lower, upper in YEAR_BINS:
        if lower <= value <= upper:
            return label
    return "outside_2005_2026"


def split_accessions(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def load_drug_metadata(path: Path) -> pd.DataFrame:
    cols = [
        "drug_id",
        "drug_name",
        "approval_year",
        "route",
        "therapeutic_area",
        "indication",
        "target_name",
        "target_chembl_id",
        "chembl_first_approval",
    ]
    return pd.read_csv(path, dtype=str, usecols=lambda col: col in cols).fillna("")


def load_record_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    numeric_cols = [
        "in_scope_accession_count",
        "top10000_exact_hit_count",
        "top10000_exact_min_rank",
        "top10000_sequence_hit_count",
        "top10000_sequence_min_rank",
        "completed_hit_count",
        "missing_output_hit_count",
        "all_exact_min_rank",
        "all_exact_best_affinity",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def attach_metadata(records: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
    # The seed file can contain multiple records for the same drug. Keep the first
    # row because approval year, route, and therapeutic area are drug-level fields.
    drug_meta = drugs.sort_values(["drug_id"]).drop_duplicates("drug_id")
    merged = records.merge(
        drug_meta,
        on="drug_id",
        how="left",
        suffixes=("", "_library"),
    )
    merged["approvalYear"] = pd.to_numeric(merged.get("approval_year", ""), errors="coerce")
    merged["approvalYearBin"] = merged["approvalYear"].map(year_bin)
    merged["therapeuticArea"] = merged.get("therapeutic_area", "").replace("", "unknown")
    merged["routeClass"] = merged.get("route", "").replace("", "unknown")
    merged["knownTargetAccessions"] = merged["in_scope_accessions"].map(split_accessions)
    merged["knownTargetCount"] = merged["knownTargetAccessions"].map(len)
    return merged


def expand_pair_hits(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in records.iterrows():
        pair_ids = split_accessions(row.get("all_exact_hit_pairs", ""))
        for pair_id in pair_ids:
            protein_id = pair_id.rsplit("__", 1)[-1] if "__" in pair_id else ""
            rows.append(
                {
                    "drug_id": row.get("drug_id", ""),
                    "drug_name": row.get("drug_name", ""),
                    "approvalYear": row.get("approvalYear"),
                    "approvalYearBin": row.get("approvalYearBin", "unknown"),
                    "therapeuticArea": row.get("therapeuticArea", "unknown"),
                    "routeClass": row.get("routeClass", "unknown"),
                    "target_chembl_id": row.get("target_chembl_id", ""),
                    "target_name": row.get("target_name", ""),
                    "protein_id": protein_id,
                    "pair_id": pair_id,
                    "allExactMinRankForTargetRecord": row.get("all_exact_min_rank"),
                    "allExactBestAffinityForTargetRecord": row.get("all_exact_best_affinity"),
                }
            )
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame, group_type: str, group_value: str) -> dict[str, Any]:
    denom = len(df)
    in_scope_accessions = int(pd.to_numeric(df["in_scope_accession_count"], errors="coerce").fillna(0).sum())
    ranks = pd.to_numeric(df["all_exact_min_rank"], errors="coerce").dropna()
    row: dict[str, Any] = {
        "groupType": group_type,
        "groupValue": group_value,
        "targetRecords": denom,
        "inScopeAccessions": in_scope_accessions,
        "drugCount": int(df["drug_id"].nunique()),
        "targetChemblCount": int(df["target_chembl_id"].nunique()),
        "medianBestRank": float(ranks.median()) if len(ranks) else None,
        "meanBestRank": float(ranks.mean()) if len(ranks) else None,
        "missingBestRankRecords": int(df["all_exact_min_rank"].isna().sum()),
    }
    for cutoff in CUTOFFS:
        hits = int((df["all_exact_min_rank"] <= cutoff).sum())
        row[f"recordHitsAt{cutoff}"] = hits
        row[f"recordRecallAt{cutoff}Pct"] = pct(hits, denom)
    completed = int(pd.to_numeric(df["completed_hit_count"], errors="coerce").fillna(0).gt(0).sum())
    missing = int(pd.to_numeric(df["missing_output_hit_count"], errors="coerce").fillna(0).gt(0).sum())
    row["completedDockingHitRecords"] = completed
    row["missingOutputHitRecords"] = missing
    row["completedDockingHitPct"] = pct(completed, denom)
    row["missingOutputHitPct"] = pct(missing, denom)
    return row


def build_group_rows(records: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(summarize_group(records, "all", "all"))
    for group_col, group_type in [
        ("approvalYearBin", "approvalYearBin"),
        ("therapeuticArea", "therapeuticArea"),
        ("routeClass", "route"),
    ]:
        grouped = records.groupby(group_col, dropna=False)
        for value, group in grouped:
            rows.append(summarize_group(group, group_type, str(value or "unknown")))
    return sorted(rows, key=lambda row: (row["groupType"], row["groupValue"]))


def build_gap_rows(records: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    missing = records[
        pd.to_numeric(records["all_exact_min_rank"], errors="coerce").isna()
        | (pd.to_numeric(records["all_exact_min_rank"], errors="coerce") > 100000)
    ].copy()
    missing["inScopeCountSort"] = pd.to_numeric(missing["in_scope_accession_count"], errors="coerce").fillna(0)
    missing["bestRankSort"] = pd.to_numeric(missing["all_exact_min_rank"], errors="coerce").fillna(999999999)
    missing = missing.sort_values(["inScopeCountSort", "bestRankSort", "drug_name"], ascending=[False, True, True])
    rows: list[dict[str, Any]] = []
    for _, row in missing.head(top_n).iterrows():
        rows.append(
            {
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "approvalYear": row.get("approvalYear", ""),
                "approvalYearBin": row.get("approvalYearBin", ""),
                "therapeuticArea": row.get("therapeuticArea", ""),
                "route": row.get("routeClass", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
                "target_name": row.get("target_name", ""),
                "inScopeAccessions": row.get("in_scope_accessions", ""),
                "inScopeGenes": row.get("in_scope_genes", ""),
                "allExactMinRank": row.get("all_exact_min_rank", ""),
                "allExactBestAffinity": row.get("all_exact_best_affinity", ""),
                "top10000ExactHitCount": row.get("top10000_exact_hit_count", ""),
                "completedHitCount": row.get("completed_hit_count", ""),
                "missingOutputHitCount": row.get("missing_output_hit_count", ""),
                "gapReason": "not recovered within Top100000" if not pd.isna(row.get("all_exact_min_rank")) else "no exact hit found in scored matrix",
            }
        )
    return rows


def build_summary(records: pd.DataFrame, group_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_row = next(row for row in group_rows if row["groupType"] == "all")
    year_rows = [row for row in group_rows if row["groupType"] == "approvalYearBin"]
    area_counts = Counter(records["therapeuticArea"])
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targetRecords": int(len(records)),
        "drugCount": int(records["drug_id"].nunique()),
        "targetChemblCount": int(records["target_chembl_id"].nunique()),
        "inScopeAccessions": int(pd.to_numeric(records["in_scope_accession_count"], errors="coerce").fillna(0).sum()),
        "overall": all_row,
        "approvalYearBins": year_rows,
        "therapeuticAreaCounts": dict(area_counts.most_common()),
        "gapRowsTopN": len(gap_rows),
        "methodNote": (
            "Stratified known-target validation over FDA small-molecule target records with in-scope UniProt accessions. "
            "Recall is record-level: a target record is counted as recovered if at least one exact accession for that record is ranked within K."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stratified known-target validation tables.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--drug-library", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--known-target-audit", default="outputs/druggable_proteome/fda_known_target_recall_top100000_audit.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/known_target_stratified")
    parser.add_argument("--gap-top-n", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    drugs = load_drug_metadata(root / args.drug_library)
    records = attach_metadata(load_record_audit(root / args.known_target_audit), drugs)
    group_rows = build_group_rows(records)
    gap_rows = build_gap_rows(records, args.gap_top_n)
    pair_hits = expand_pair_hits(records)
    summary = build_summary(records, group_rows, gap_rows)
    summary["inputs"] = {
        "drugLibrary": args.drug_library,
        "knownTargetAudit": args.known_target_audit,
    }
    summary["outputs"] = {
        "groupSummary": str((out_dir / "known_target_stratified_summary.csv").resolve()),
        "gapTable": str((out_dir / "known_target_top100k_gap_records.csv").resolve()),
        "pairHits": str((out_dir / "known_target_pair_hits_expanded.csv").resolve()),
        "summary": str((out_dir / "known_target_stratified_summary.json").resolve()),
    }

    write_csv(out_dir / "known_target_stratified_summary.csv", group_rows)
    write_csv(out_dir / "known_target_top100k_gap_records.csv", gap_rows)
    if len(pair_hits):
        pair_hits.to_csv(out_dir / "known_target_pair_hits_expanded.csv", index=False)
    else:
        write_csv(out_dir / "known_target_pair_hits_expanded.csv", [])
    write_json(out_dir / "known_target_stratified_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

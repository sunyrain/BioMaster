#!/usr/bin/env python3
"""Run PubMed drug-target literature audit for the strict895 set.

This supplements the older 12,696-pair comprehensive audit. The strict895
candidate set was generated later from the broad mechanism v2 top-ready layer,
so it is not guaranteed to be a subset of the earlier literature-audited table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_comprehensive_repurposing_literature_report import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    clean,
    run_pair_literature_audit,
)


BASE = ROOT / "outputs/broad_mechanism_layer_v2"
STRICT_SOURCE = BASE / "strict895_agent_review/strict_top_ready_895_source_rows.csv"
COMPREHENSIVE_AUDIT = ROOT / "outputs/comprehensive_repurposing_literature/unique_drug_target_pairs_literature_audit.csv"
COMPREHENSIVE_CACHE = ROOT / "outputs/comprehensive_repurposing_literature/pubmed_pair_audit_cache.json"
OUTDIR = BASE / "strict895_pubmed_literature_audit"


def norm_gene(value: Any) -> str:
    text = clean(value).upper()
    for sep in [";", ",", "|"]:
        if sep in text:
            text = next((part.strip() for part in text.split(sep) if part.strip()), "")
    return text


def normalize_drug_id(value: Any) -> str:
    text = clean(value)
    # Some restarted-pool rows used a synthetic suffix to keep duplicate FDA
    # representations apart. PubMed evidence is molecule-name based, so the
    # base ChEMBL identifier is the stable pair key.
    return re.sub(r"__.*$", "", text)


def truthy(value: Any) -> bool:
    return clean(value).upper() in {"TRUE", "1", "YES", "Y"}


def pick_first(values: pd.Series) -> str:
    for value in values:
        text = clean(value)
        if text:
            return text
    return ""


def build_strict_pairs(source: pd.DataFrame) -> pd.DataFrame:
    work = source.copy()
    work["drug_chembl_id_original"] = work["drug_chembl_id"].map(clean)
    work["drug_chembl_id"] = work["drug_chembl_id"].map(normalize_drug_id)
    work["generic_name"] = work.get("generic_name", work.get("drug_names", "")).map(clean)
    if "drug_names" in work.columns:
        work["generic_name"] = work["generic_name"].where(work["generic_name"].ne(""), work["drug_names"].map(clean))
    work["candidate_anchor_gene"] = work.get("target_gene_norm", work.get("target_gene", "")).map(norm_gene)
    work["candidate_anchor_name"] = work.get("protein_names", work.get("target_name", "")).map(clean)
    work["first_approval_year"] = pd.to_numeric(work.get("approval_year_min"), errors="coerce")
    work["max_candidate_total_score"] = pd.to_numeric(work.get("review_priority_score"), errors="coerce").fillna(0.0)
    work["max_non_table_evidence_count"] = pd.to_numeric(work.get("non_conplex_evidence_count_audit"), errors="coerce").fillna(0.0)
    work["any_direct_known_label"] = work.get("exact_known_target_by_profile_or_control", False).map(truthy)
    work["any_known_drug_target_pair"] = work.get("is_known_fda_target_pair", False).map(truthy)
    work["fda_text_known_target_match"] = work["any_direct_known_label"] | work["any_known_drug_target_pair"]
    work["automated_novel_target_candidate"] = ~work[
        ["any_direct_known_label", "any_known_drug_target_pair", "fda_text_known_target_match"]
    ].any(axis=1)
    work["pair_priority_class"] = "strict895"

    grouped = (
        work.groupby(["drug_chembl_id", "candidate_anchor_gene"], dropna=False)
        .agg(
            drug_chembl_id_originals=("drug_chembl_id_original", lambda s: ";".join(sorted(set(clean(v) for v in s if clean(v))))),
            generic_name=("generic_name", pick_first),
            candidate_anchor_name=("candidate_anchor_name", pick_first),
            first_approval_year=("first_approval_year", "min"),
            directions=("direction", lambda s: ";".join(sorted(set(clean(v) for v in s if clean(v))))),
            review_ids=("review_id", lambda s: ";".join(clean(v) for v in s if clean(v))),
            strict895_row_count=("review_id", "size"),
            pair_priority_class=("pair_priority_class", pick_first),
            max_candidate_total_score=("max_candidate_total_score", "max"),
            max_non_table_evidence_count=("max_non_table_evidence_count", "max"),
            any_direct_known_label=("any_direct_known_label", "max"),
            any_known_drug_target_pair=("any_known_drug_target_pair", "max"),
            fda_text_known_target_match=("fda_text_known_target_match", "max"),
            automated_novel_target_candidate=("automated_novel_target_candidate", "max"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["drug_chembl_id"].map(clean).ne("") & grouped["candidate_anchor_gene"].map(clean).ne("")]
    return grouped


def seed_cache_from_comprehensive(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_cache = out_dir / "pubmed_pair_audit_cache.json"
    if target_cache.exists() or not COMPREHENSIVE_CACHE.exists():
        return 0
    source_cache = json.loads(COMPREHENSIVE_CACHE.read_text(encoding="utf-8"))
    target_cache.write_text(json.dumps(source_cache, ensure_ascii=False), encoding="utf-8")
    return len(source_cache)


def add_overlap_flag(audited: pd.DataFrame) -> pd.DataFrame:
    out = audited.copy()
    if not COMPREHENSIVE_AUDIT.exists():
        out["in_previous_12696_literature_audit"] = False
        return out
    comp = pd.read_csv(COMPREHENSIVE_AUDIT, low_memory=False)
    comp_key = set(
        comp["drug_chembl_id"].map(normalize_drug_id).map(clean)
        + "__"
        + comp["candidate_anchor_gene"].map(norm_gene)
    )
    key = out["drug_chembl_id"].map(normalize_drug_id).map(clean) + "__" + out["candidate_anchor_gene"].map(norm_gene)
    out["in_previous_12696_literature_audit"] = key.isin(comp_key)
    return out


def status_for_final(row: pd.Series) -> str:
    def as_int(value: Any) -> int:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return 0
        return int(numeric)

    if not bool(row.get("lit_ok", False)):
        return "pubmed_query_failed_needs_retry"
    if bool(row.get("any_direct_known_label")) or bool(row.get("any_known_drug_target_pair")) or bool(row.get("fda_text_known_target_match")):
        return "known_pharmacology_or_control_not_new"
    post = as_int(row.get("post_approval_pair_pubmed_count"))
    full = as_int(row.get("pair_pubmed_count_2000_2026"))
    if post > 0:
        return "reported_post_approval_old_drug_new_target"
    if full > 0:
        return "reported_only_before_or_without_approval_window"
    return "unreported_in_pubmed_pair_audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=STRICT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--delay-s", type=float, default=0.34)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeded = seed_cache_from_comprehensive(args.out_dir)
    source = pd.read_csv(args.source, low_memory=False)
    pairs = build_strict_pairs(source)
    pairs.to_csv(args.out_dir / "strict895_pairs_for_pubmed_audit.csv", index=False)

    audited = run_pair_literature_audit(
        pairs,
        args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        delay_s=args.delay_s,
        refresh=args.refresh,
        max_pairs=args.max_pairs,
        workers=args.workers,
    )
    audited = add_overlap_flag(audited)
    audited["strict895_literature_status"] = audited.apply(status_for_final, axis=1)
    audited.to_csv(args.out_dir / "strict895_pair_pubmed_literature_audit.csv", index=False)

    summary = {
        "source": str(args.source),
        "strict895_source_rows": int(len(source)),
        "unique_drug_target_pairs": int(len(pairs)),
        "seeded_cache_entries_from_previous_comprehensive_run": int(seeded),
        "audited_rows": int(len(audited)),
        "in_previous_12696_literature_audit": int(audited["in_previous_12696_literature_audit"].sum()),
        "new_pairs_not_in_previous_12696": int((~audited["in_previous_12696_literature_audit"]).sum()),
        "status_counts": audited["strict895_literature_status"].value_counts(dropna=False).to_dict(),
        "pubmed_pair_count_gt0": int((pd.to_numeric(audited["pair_pubmed_count_2000_2026"], errors="coerce").fillna(0) > 0).sum()),
        "post_approval_pair_count_gt0": int((pd.to_numeric(audited["post_approval_pair_pubmed_count"], errors="coerce").fillna(0) > 0).sum()),
    }
    (args.out_dir / "strict895_pubmed_literature_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

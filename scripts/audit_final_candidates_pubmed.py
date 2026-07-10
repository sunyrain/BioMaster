#!/usr/bin/env python3
"""Run an exact-name PubMed screening audit for a formal candidate table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_comprehensive_repurposing_literature_report import (  # noqa: E402
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    run_pair_literature_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay-s", type=float, default=0.34)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, low_memory=False).fillna("")
    pairs = pd.DataFrame(
        {
            "drug_chembl_id": candidates["drug_chembl_id"],
            "generic_name": candidates["drug_names"],
            "candidate_anchor_gene": candidates["primary_gene"],
            "candidate_anchor_name": candidates["protein_names"],
            "first_approval_year": pd.to_numeric(candidates.get("fda_approval_year"), errors="coerce"),
            "pair_priority_class": "formal_v4",
            "max_candidate_total_score": pd.to_numeric(candidates["priority_score_v2"], errors="coerce"),
            "max_non_table_evidence_count": 2,
            "automated_novel_target_candidate": True,
            "fda_text_known_target_match": False,
        }
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "formal_pairs_for_pubmed_audit.csv", index=False)
    audited = run_pair_literature_audit(
        pairs,
        out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        delay_s=args.delay_s,
        refresh=args.refresh,
        max_pairs=None,
        workers=args.workers,
    )
    audited.to_csv(out_dir / "formal_pair_pubmed_audit.csv", index=False)
    pair_count = pd.to_numeric(audited["pair_pubmed_count_2000_2026"], errors="coerce").fillna(0)
    post_count = pd.to_numeric(audited["post_approval_pair_pubmed_count"], errors="coerce").fillna(0)
    summary = {
        "rows": int(len(audited)),
        "query_ok": int(audited["lit_ok"].astype(bool).sum()),
        "pubmed_count_gt0": int((pair_count > 0).sum()),
        "post_approval_count_gt0": int((post_count > 0).sum()),
        "warning": "Automated name/gene co-occurrence is a screening signal; PMIDs require manual exact-pair validation.",
    }
    (out_dir / "formal_pair_pubmed_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

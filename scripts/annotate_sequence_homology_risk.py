#!/usr/bin/env python3
"""Annotate candidate pairs with sequence homology to each drug's known targets."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.Align import PairwiseAligner


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    return aligner


def annotate(candidates: pd.DataFrame, known: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    sequence_by_key = sequences.drop_duplicates("sequence_key").set_index("sequence_key")["sequence"].map(clean).to_dict()
    gene_by_key = sequences.drop_duplicates("sequence_key").set_index("sequence_key")["gene_names"].map(clean).to_dict()
    known_by_drug = {
        clean(drug): sorted(set(group["sequence_key"].map(clean)) - {""})
        for drug, group in known.groupby("drug_chembl_id")
    }
    aligner = make_aligner()

    @lru_cache(maxsize=None)
    def compare(first_key: str, second_key: str) -> tuple[float, float, int]:
        first = sequence_by_key.get(first_key, "")
        second = sequence_by_key.get(second_key, "")
        if not first or not second:
            return 0.0, 0.0, 0
        if first == second:
            return 1.0, 1.0, min(len(first), len(second))
        alignment = aligner.align(first, second)[0]
        counts = alignment.counts()
        aligned = int(counts.identities + counts.mismatches)
        identity = float(counts.identities / aligned) if aligned else 0.0
        coverage = float(aligned / min(len(first), len(second))) if first and second else 0.0
        return identity, coverage, aligned

    max_identity: list[float] = []
    max_coverage: list[float] = []
    max_aligned: list[int] = []
    matched_sequence: list[str] = []
    matched_gene: list[str] = []
    risk: list[bool] = []
    for _, row in candidates.iterrows():
        drug = clean(row.get("drug_chembl_id"))
        candidate_key = clean(row.get("sequence_key"))
        best = (0.0, 0.0, 0, "")
        for known_key in known_by_drug.get(drug, []):
            identity, coverage, aligned = compare(candidate_key, known_key)
            # Prefer broad, high-identity alignments over short local matches.
            merit = identity * min(1.0, coverage / 0.60)
            best_merit = best[0] * min(1.0, best[1] / 0.60)
            if merit > best_merit:
                best = (identity, coverage, aligned, known_key)
        identity, coverage, aligned, known_key = best
        extension = candidate_key != known_key and (
            (identity >= 0.40 and coverage >= 0.60)
            or (identity >= 0.55 and coverage >= 0.30)
        )
        max_identity.append(identity)
        max_coverage.append(coverage)
        max_aligned.append(aligned)
        matched_sequence.append(known_key)
        matched_gene.append(gene_by_key.get(known_key, ""))
        risk.append(bool(extension))

    out = candidates.copy()
    out["max_known_target_local_identity"] = max_identity
    out["max_known_target_local_coverage"] = max_coverage
    out["max_known_target_aligned_residues"] = max_aligned
    out["nearest_known_target_sequence_key"] = matched_sequence
    out["nearest_known_target_gene"] = matched_gene
    out["sequence_homology_extension_risk"] = risk
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--known-controls", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates, low_memory=False).fillna("")
    known = pd.read_csv(args.known_controls, low_memory=False).fillna("")
    sequences = pd.read_csv(args.sequences, low_memory=False).fillna("")
    result = annotate(candidates, known, sequences)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "rows": int(len(result)),
        "rows_with_known_target_context": int(result["nearest_known_target_sequence_key"].astype(str).ne("").sum()),
        "sequence_homology_extension_risk_rows": int(result["sequence_homology_extension_risk"].sum()),
        "risk_unique_drugs": int(result.loc[result["sequence_homology_extension_risk"], "drug_chembl_id"].nunique()),
        "risk_unique_targets": int(result.loc[result["sequence_homology_extension_risk"], "sequence_key"].nunique()),
        "method": "Biopython local alignment; risk if identity>=0.40 and shorter-sequence coverage>=0.60, or identity>=0.55 and coverage>=0.30.",
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

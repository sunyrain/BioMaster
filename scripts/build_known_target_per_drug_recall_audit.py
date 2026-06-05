from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFFS = [1, 3, 5, 10, 20, 50, 53, 100, 200, 266, 500, 531, 1000, 2000, 5306]


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


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


def split_semicolon(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def random_hit_probability(candidate_count: int, positive_count: int, cutoff: int) -> float:
    if candidate_count <= 0 or positive_count <= 0 or cutoff <= 0:
        return 0.0
    k = min(cutoff, candidate_count)
    if k >= candidate_count or positive_count >= candidate_count:
        return 1.0
    if candidate_count - positive_count < k:
        return 1.0
    log_miss = log_comb(candidate_count - positive_count, k) - log_comb(candidate_count, k)
    return 1.0 - math.exp(log_miss)


def load_protein_sequence_equivalence(path: Path) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    df = pd.read_csv(path, dtype=str, usecols=lambda col: col in {"protein_id", "sequence_key", "gene_name"}).fillna("")
    accession_to_sequence: dict[str, str] = {}
    sequence_to_accessions: dict[str, set[str]] = defaultdict(set)
    accession_to_gene: dict[str, str] = {}
    for row in df.itertuples(index=False):
        protein_id = getattr(row, "protein_id")
        sequence_key = getattr(row, "sequence_key", "")
        gene_name = getattr(row, "gene_name", "")
        accession_to_gene[protein_id] = gene_name
        if sequence_key:
            accession_to_sequence[protein_id] = sequence_key
            sequence_to_accessions[sequence_key].add(protein_id)
    return accession_to_sequence, sequence_to_accessions, accession_to_gene


def load_known_records(path: Path, accession_to_sequence: dict[str, str], sequence_to_accessions: dict[str, set[str]]) -> list[dict[str, Any]]:
    df = pd.read_csv(path, dtype=str).fillna("")
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        exact_accessions = split_semicolon(row.get("in_scope_accessions"))
        sequence_keys = sorted({accession_to_sequence.get(acc, "") for acc in exact_accessions} - {""})
        sequence_accessions: set[str] = set(exact_accessions)
        for key in sequence_keys:
            sequence_accessions.update(sequence_to_accessions.get(key, set()))
        records.append(
            {
                "record_id": f"{row.get('drug_id', '')}__{row.get('target_chembl_id', '')}",
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
                "target_name": row.get("target_name", ""),
                "target_pref_name": row.get("target_pref_name", ""),
                "target_organism": row.get("target_organism", ""),
                "exact_accessions": sorted(set(exact_accessions)),
                "sequence_equivalent_accessions": sorted(sequence_accessions),
                "sequence_keys": sequence_keys,
            }
        )
    return records


def collect_affinity_scores(path: Path, target_drugs: set[str], chunksize: int) -> tuple[dict[str, list[tuple[str, float]]], int, dict[str, int]]:
    scores_by_drug: dict[str, list[tuple[str, float]]] = defaultdict(list)
    pair_total = 0
    pair_count_by_drug: dict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(path, usecols=["pair_id", "affinity_score"], dtype={"pair_id": str}, chunksize=chunksize):
        parts = chunk["pair_id"].astype(str).str.rsplit("__", n=1, expand=True)
        chunk["drug_id"] = parts[0]
        chunk["protein_id"] = parts[1]
        chunk["affinity_score"] = pd.to_numeric(chunk["affinity_score"], errors="coerce").fillna(0.0)
        pair_total += len(chunk)
        for drug_id, count in chunk["drug_id"].value_counts().items():
            pair_count_by_drug[str(drug_id)] += int(count)
        subset = chunk[chunk["drug_id"].isin(target_drugs)]
        for row in subset.itertuples(index=False):
            scores_by_drug[getattr(row, "drug_id")].append((getattr(row, "protein_id"), float(getattr(row, "affinity_score"))))
    return scores_by_drug, pair_total, dict(pair_count_by_drug)


def rank_drug_scores(items: list[tuple[str, float]]) -> dict[str, dict[str, Any]]:
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    rank_map: dict[str, dict[str, Any]] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        score = ordered[start][1]
        while end < len(ordered) and ordered[end][1] == score:
            end += 1
        min_rank = start + 1
        max_rank = end
        avg_rank = (min_rank + max_rank) / 2
        tie_size = end - start
        for idx in range(start, end):
            protein_id = ordered[idx][0]
            rank_map[protein_id] = {
                "score": score,
                "minRank": min_rank,
                "averageRank": avg_rank,
                "maxRank": max_rank,
                "tieSize": tie_size,
            }
        start = end
    return rank_map


def best_hit(accessions: set[str], rank_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hits = []
    for accession in accessions:
        item = rank_map.get(accession)
        if item:
            hits.append((accession, item))
    if not hits:
        return {"accession": "", "minRank": None, "averageRank": None, "score": None, "tieSize": None}
    accession, item = min(hits, key=lambda pair: (pair[1]["minRank"], pair[0]))
    return {
        "accession": accession,
        "minRank": int(item["minRank"]),
        "averageRank": float(item["averageRank"]),
        "score": float(item["score"]),
        "tieSize": int(item["tieSize"]),
    }


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def summarize_cutoffs(record_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], candidate_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    record_denominator = len(record_rows)
    pair_denominator = len(pair_rows)
    for cutoff in CUTOFFS:
        k = min(cutoff, candidate_count)
        exact_hits = sum(1 for row in record_rows if number(row.get("bestExactMinRank")) is not None and float(row["bestExactMinRank"]) <= k)
        seq_hits = sum(1 for row in record_rows if number(row.get("bestSequenceMinRank")) is not None and float(row["bestSequenceMinRank"]) <= k)
        exact_pair_hits = sum(1 for row in pair_rows if number(row.get("exactMinRank")) is not None and float(row["exactMinRank"]) <= k)
        expected_exact = sum(random_hit_probability(candidate_count, int(row.get("exactCandidateCount", 0)), k) for row in record_rows)
        expected_seq = sum(random_hit_probability(candidate_count, int(row.get("sequenceCandidateCount", 0)), k) for row in record_rows)
        expected_pairs = pair_denominator * k / candidate_count if candidate_count else 0.0
        rows.append(
            {
                "cutoff": k,
                "cutoffLabel": f"Top{k}",
                "exactRecordHits": exact_hits,
                "exactRecordRecallPct": pct(exact_hits, record_denominator),
                "sequenceRecordHits": seq_hits,
                "sequenceRecordRecallPct": pct(seq_hits, record_denominator),
                "exactPairHits": exact_pair_hits,
                "exactPairRecallPct": pct(exact_pair_hits, pair_denominator),
                "randomExpectedExactRecordHits": expected_exact,
                "randomExpectedExactRecordRecallPct": pct(expected_exact, record_denominator),
                "exactRecordEnrichmentVsRandom": exact_hits / expected_exact if expected_exact else None,
                "randomExpectedSequenceRecordHits": expected_seq,
                "randomExpectedSequenceRecordRecallPct": pct(expected_seq, record_denominator),
                "sequenceRecordEnrichmentVsRandom": seq_hits / expected_seq if expected_seq else None,
                "randomExpectedExactPairHits": expected_pairs,
                "randomExpectedExactPairRecallPct": pct(expected_pairs, pair_denominator),
                "exactPairEnrichmentVsRandom": exact_pair_hits / expected_pairs if expected_pairs else None,
            }
        )
    return rows


def load_coverage(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    fda_path = root / args.fda_xlsx
    drug_library_path = root / args.drug_library
    all_audit_path = root / args.all_known_audit
    coverage: dict[str, Any] = {}
    if fda_path.exists():
        fda = pd.read_excel(fda_path)
        coverage["fdaRows"] = int(len(fda))
        coverage["fdaRowsWithTargetChembl"] = int(fda["Target ChEMBL ID"].notna().sum()) if "Target ChEMBL ID" in fda else None
        coverage["fdaUniqueChemblIds"] = int(fda["ChEMBL ID"].dropna().nunique()) if "ChEMBL ID" in fda else None
    if drug_library_path.exists():
        library = pd.read_csv(drug_library_path, dtype=str).fillna("")
        coverage["drugLibraryRows"] = int(len(library))
        coverage["drugLibraryRowsWithTargetChembl"] = int(library["target_chembl_id"].astype(str).ne("").sum()) if "target_chembl_id" in library else None
        coverage["drugLibraryUniqueDrugIds"] = int(library["drug_id"].nunique()) if "drug_id" in library else None
    if all_audit_path.exists():
        audit = pd.read_csv(all_audit_path, dtype=str).fillna("")
        in_scope = pd.to_numeric(audit.get("in_scope_accession_count", 0), errors="coerce").fillna(0).gt(0)
        mapped = pd.to_numeric(audit.get("mapped_accession_count", 0), errors="coerce").fillna(0).gt(0)
        coverage["uniqueDrugTargetChemblRecords"] = int(len(audit))
        coverage["recordsWithMappedAccessions"] = int(mapped.sum())
        coverage["recordsInCurrentProteinScope"] = int(in_scope.sum())
        coverage["recordsOutsideCurrentProteinScopeOrUnmapped"] = int((~in_scope).sum())
    return coverage


def build_markdown(summary: dict[str, Any], cutoffs: list[dict[str, Any]]) -> str:
    selected = [1, 5, 10, 20, 50, 100, 531, 1000, 5306]
    lines = [
        "# Known-Target Recall Audit for Drug Repurposing",
        "",
        f"Created UTC: {summary['createdUtc']}",
        "",
        "## Evaluation Scope",
        "",
        f"- FDA rows: {summary['coverage'].get('fdaRows')}",
        f"- FDA rows with target ChEMBL ID: {summary['coverage'].get('fdaRowsWithTargetChembl')}",
        f"- Unique drug-target ChEMBL records in audit: {summary['coverage'].get('uniqueDrugTargetChemblRecords')}",
        f"- Evaluable records in current protein scope: {summary['evaluableTargetRecords']}",
        f"- Candidate proteins ranked per drug: {summary['candidateProteinsPerDrug']}",
        "",
        "## Main Result",
        "",
        "| Cutoff | Exact record recall | Sequence-equivalent record recall | Exact pair recall | Exact enrichment vs random |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_cutoff = {row["cutoff"]: row for row in cutoffs}
    for cutoff in selected:
        row = by_cutoff.get(min(cutoff, summary["candidateProteinsPerDrug"]))
        if not row:
            continue
        lines.append(
            "| {cutoff} | {exact:.2f}% ({eh}/{den}) | {seq:.2f}% ({sh}/{den}) | {pair:.2f}% ({ph}/{pden}) | {enrich:.1f}x |".format(
                cutoff=row["cutoff"],
                exact=row["exactRecordRecallPct"],
                eh=row["exactRecordHits"],
                den=summary["evaluableTargetRecords"],
                seq=row["sequenceRecordRecallPct"],
                sh=row["sequenceRecordHits"],
                pair=row["exactPairRecallPct"],
                ph=row["exactPairHits"],
                pden=summary["expandedKnownDrugUniprotPairs"],
                enrich=row["exactRecordEnrichmentVsRandom"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Exact record recall asks whether at least one known UniProt accession for a drug's known target appears within the drug-specific Top-K list.",
            "Sequence-equivalent recall also credits accessions with the same protein sequence, which is important when UniProt contains duplicate or isoform-equivalent entries in the druggable proteome table.",
            "The denominator is the subset that can be evaluated with the current protein library; FDA records without a mapped in-scope protein are counted in coverage, not as model failures.",
            "",
        ]
    )
    return "\n".join(lines)


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / args.out_dir
    accession_to_sequence, sequence_to_accessions, accession_to_gene = load_protein_sequence_equivalence(root / args.protein_library)
    records = load_known_records(root / args.known_audit, accession_to_sequence, sequence_to_accessions)
    target_drugs = {row["drug_id"] for row in records}
    scores_by_drug, pair_total, pair_count_by_drug = collect_affinity_scores(root / args.affinity, target_drugs, args.chunksize)
    candidate_counts = sorted(set(pair_count_by_drug.values()))
    candidate_count = candidate_counts[0] if len(candidate_counts) == 1 else max(candidate_counts or [0])

    record_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for record in records:
        drug_id = record["drug_id"]
        rank_map = rank_drug_scores(scores_by_drug.get(drug_id, []))
        exact_accessions = set(record["exact_accessions"])
        sequence_accessions = set(record["sequence_equivalent_accessions"])
        exact_best = best_hit(exact_accessions, rank_map)
        sequence_best = best_hit(sequence_accessions, rank_map)
        record_rows.append(
            {
                "recordId": record["record_id"],
                "drugId": drug_id,
                "drugName": record["drug_name"],
                "targetChemblId": record["target_chembl_id"],
                "targetName": record["target_name"],
                "targetPrefName": record["target_pref_name"],
                "exactAccessions": ";".join(record["exact_accessions"]),
                "sequenceKeys": ";".join(record["sequence_keys"]),
                "exactCandidateCount": len(exact_accessions),
                "sequenceCandidateCount": len(sequence_accessions),
                "candidateProteinsForDrug": len(rank_map),
                "bestExactAccession": exact_best["accession"],
                "bestExactGene": accession_to_gene.get(exact_best["accession"], ""),
                "bestExactMinRank": exact_best["minRank"],
                "bestExactAverageRank": exact_best["averageRank"],
                "bestExactScore": exact_best["score"],
                "bestExactTieSize": exact_best["tieSize"],
                "bestSequenceAccession": sequence_best["accession"],
                "bestSequenceGene": accession_to_gene.get(sequence_best["accession"], ""),
                "bestSequenceMinRank": sequence_best["minRank"],
                "bestSequenceAverageRank": sequence_best["averageRank"],
                "bestSequenceScore": sequence_best["score"],
                "bestSequenceTieSize": sequence_best["tieSize"],
            }
        )
        for accession in sorted(exact_accessions):
            item = rank_map.get(accession)
            seq_key = accession_to_sequence.get(accession, "")
            seq_accessions = sequence_to_accessions.get(seq_key, set()) if seq_key else set()
            seq_best = best_hit(seq_accessions, rank_map)
            pair_rows.append(
                {
                    "pairId": f"{drug_id}__{accession}",
                    "drugId": drug_id,
                    "drugName": record["drug_name"],
                    "targetChemblId": record["target_chembl_id"],
                    "targetName": record["target_name"],
                    "proteinId": accession,
                    "geneName": accession_to_gene.get(accession, ""),
                    "sequenceKey": seq_key,
                    "exactMinRank": int(item["minRank"]) if item else None,
                    "exactAverageRank": float(item["averageRank"]) if item else None,
                    "exactScore": float(item["score"]) if item else None,
                    "exactTieSize": int(item["tieSize"]) if item else None,
                    "sequenceBestAccession": seq_best["accession"],
                    "sequenceBestMinRank": seq_best["minRank"],
                    "sequenceBestScore": seq_best["score"],
                }
            )

    cutoffs = summarize_cutoffs(record_rows, pair_rows, candidate_count)
    exact_ranks = [float(row["bestExactMinRank"]) for row in record_rows if number(row.get("bestExactMinRank")) is not None]
    sequence_ranks = [float(row["bestSequenceMinRank"]) for row in record_rows if number(row.get("bestSequenceMinRank")) is not None]
    pair_ranks = [float(row["exactMinRank"]) for row in pair_rows if number(row.get("exactMinRank")) is not None]
    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "affinity": args.affinity,
            "knownAudit": args.known_audit,
            "allKnownAudit": args.all_known_audit,
            "proteinLibrary": args.protein_library,
            "drugLibrary": args.drug_library,
            "fdaXlsx": args.fda_xlsx,
        },
        "coverage": load_coverage(root, args),
        "affinityPairs": pair_total,
        "affinityDrugCount": len(pair_count_by_drug),
        "candidateProteinsPerDrug": candidate_count,
        "candidateProteinsPerDrugDistinctCounts": candidate_counts,
        "evaluableTargetRecords": len(record_rows),
        "evaluableDrugs": len({row["drugId"] for row in record_rows}),
        "expandedKnownDrugUniprotPairs": len(pair_rows),
        "exactMedianBestRank": median(exact_ranks),
        "sequenceMedianBestRank": median(sequence_ranks),
        "exactPairMedianRank": median(pair_ranks),
        "exactMeanReciprocalRank": sum(1.0 / rank for rank in exact_ranks) / len(exact_ranks) if exact_ranks else None,
        "sequenceMeanReciprocalRank": sum(1.0 / rank for rank in sequence_ranks) / len(sequence_ranks) if sequence_ranks else None,
        "cutoffs": cutoffs,
        "methodNote": (
            "Drug-specific known-target recall. For each FDA drug with an in-scope known target, all candidate proteins are ranked by ConPLex affinity_score within that drug. "
            "Exact recall credits only the mapped UniProt accessions; sequence recall also credits protein-library accessions with the same sequence_key."
        ),
    }

    gap_rows = sorted(
        record_rows,
        key=lambda row: (
            float(row["bestExactMinRank"]) if number(row.get("bestExactMinRank")) is not None else 999999999.0,
            row["drugName"],
        ),
        reverse=True,
    )[:200]
    write_csv(out_dir / "known_target_per_drug_record_ranks.csv", record_rows)
    write_csv(out_dir / "known_target_per_drug_pair_ranks.csv", pair_rows)
    write_csv(out_dir / "known_target_per_drug_cutoffs.csv", cutoffs)
    write_csv(out_dir / "known_target_per_drug_worst_records.csv", gap_rows)
    write_json(out_dir / "known_target_per_drug_recall_summary.json", summary)
    (out_dir / "KNOWN_TARGET_PER_DRUG_RECALL.md").write_text(build_markdown(summary, cutoffs), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-drug known-target recall audit for FDA drug repurposing validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--affinity", default="outputs/druggable_proteome/conplex_affinity_scores_druggable.csv")
    parser.add_argument("--known-audit", default="outputs/druggable_proteome/fda_known_target_recall_top100000_audit.csv")
    parser.add_argument("--all-known-audit", default="outputs/druggable_proteome/fda_known_target_recall_audit.csv")
    parser.add_argument("--protein-library", default="outputs/druggable_proteome/protein_library_druggable_chembl.csv")
    parser.add_argument("--drug-library", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--fda-xlsx", default="FDA_approved_small_molecules_2005_2026_with_structures.xlsx")
    parser.add_argument("--out-dir", default="outputs/sota_validation/known_target_per_drug_recall")
    parser.add_argument("--chunksize", type=int, default=500000)
    args = parser.parse_args()

    summary = build(Path(args.root).resolve(), args)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

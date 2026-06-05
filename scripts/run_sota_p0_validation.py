from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def number(value: str | float | int | None) -> float | None:
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


def positive_pairs_from_known_targets(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    positives: set[str] = set()
    expanded_rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        drug_id = row.get("drug_id", "")
        accessions = [item.strip() for item in (row.get("in_scope_accessions") or "").split(";") if item.strip()]
        for accession in accessions:
            pair_id = f"{drug_id}__{accession}"
            positives.add(pair_id)
            expanded_rows.append(
                {
                    "pair_id": pair_id,
                    "drug_id": drug_id,
                    "drug_name": row.get("drug_name", ""),
                    "protein_id": accession,
                    "target_chembl_id": row.get("target_chembl_id", ""),
                    "target_name": row.get("target_name", ""),
                    "known_target_record": row.get("target_pref_name", ""),
                }
            )
    return positives, expanded_rows


def average_ranks_for_ties(scores: array) -> list[float]:
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        score = scores[order[start]]
        while end < len(order) and scores[order[end]] == score:
            end += 1
        avg_rank = (start + 1 + end) / 2
        for i in range(start, end):
            ranks[order[i]] = avg_rank
        start = end
    return ranks


def auc_from_ranks(labels: array, ranks_desc: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    # Ranks are descending. Convert to ascending ranks for the Mann-Whitney U formula.
    n = len(labels)
    sum_pos_ascending_ranks = sum((n + 1 - ranks_desc[i]) for i, label in enumerate(labels) if label)
    auc = (sum_pos_ascending_ranks - positives * (positives + 1) / 2) / (positives * negatives)
    return auc


def average_precision_from_sorted(labels_sorted: list[int]) -> float | None:
    positives = sum(labels_sorted)
    if positives == 0:
        return None
    hits = 0
    precision_sum = 0.0
    for idx, label in enumerate(labels_sorted, start=1):
        if label:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / positives


def enrichment_rows(
    labels_sorted: list[int],
    cutoffs: list[int],
    positive_total: int,
    total: int,
    rng_seed: int,
    permutations: int,
) -> list[dict[str, Any]]:
    rng = random.Random(rng_seed)
    positive_indices = list(range(positive_total))
    rows: list[dict[str, Any]] = []
    random_expected_rate = positive_total / total if total else 0.0
    random_hits_by_cutoff: dict[int, list[int]] = {cutoff: [] for cutoff in cutoffs}
    population = range(total)
    for _ in range(permutations):
        random_pos = set(rng.sample(population, positive_total))
        for cutoff in cutoffs:
            random_hits_by_cutoff[cutoff].append(sum(1 for idx in range(min(cutoff, total)) if idx in random_pos))

    cumulative_hits = 0
    cutoff_set = set(cutoffs)
    hits_at: dict[int, int] = {}
    for idx, label in enumerate(labels_sorted, start=1):
        cumulative_hits += int(label)
        if idx in cutoff_set:
            hits_at[idx] = cumulative_hits
    for cutoff in cutoffs:
        cutoff = min(cutoff, total)
        hits = hits_at.get(cutoff, sum(labels_sorted[:cutoff]))
        expected = cutoff * random_expected_rate
        samples = sorted(random_hits_by_cutoff.get(cutoff, []))
        p95 = samples[int(0.95 * (len(samples) - 1))] if samples else None
        rows.append(
            {
                "cutoff": cutoff,
                "hits": hits,
                "recallPct": pct(hits, positive_total),
                "precisionPct": pct(hits, cutoff),
                "randomExpectedHits": expected,
                "randomExpectedRecallPct": pct(expected, positive_total),
                "randomPermutationP95Hits": p95,
                "enrichmentVsRandom": hits / expected if expected else None,
            }
        )
    return rows


def compute_known_target_benchmark(
    affinity_path: Path,
    known_audit_path: Path,
    out_dir: Path,
    permutations: int,
    rng_seed: int,
) -> dict[str, Any]:
    positive_pairs, expanded_positive_rows = positive_pairs_from_known_targets(known_audit_path)
    scores = array("f")
    labels = array("b")
    pair_ids: list[str] = []
    positives_seen: set[str] = set()

    with affinity_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pair_id = row["pair_id"]
            score = number(row.get("affinity_score")) or 0.0
            label = 1 if pair_id in positive_pairs else 0
            if label:
                positives_seen.add(pair_id)
            pair_ids.append(pair_id)
            scores.append(score)
            labels.append(label)

    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    labels_sorted = [int(labels[idx]) for idx in order]
    ranks_desc = average_ranks_for_ties(scores)
    auc = auc_from_ranks(labels, ranks_desc)
    ap = average_precision_from_sorted(labels_sorted)
    cutoffs = [10, 20, 50, 100, 200, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    enrichment = enrichment_rows(labels_sorted, cutoffs, sum(labels), len(labels), rng_seed, permutations)

    positive_detail: list[dict[str, Any]] = []
    positive_lookup = {row["pair_id"]: row for row in expanded_positive_rows}
    for pair_id in positive_pairs:
        if pair_id not in positive_lookup:
            continue
        try:
            idx = pair_ids.index(pair_id)
        except ValueError:
            continue
        item = dict(positive_lookup[pair_id])
        item.update({"affinityScore": float(scores[idx]), "rank": int(ranks_desc[idx])})
        positive_detail.append(item)
    positive_detail.sort(key=lambda row: row["rank"])

    write_csv(
        out_dir / "known_target_positive_pair_ranks.csv",
        [
            "pair_id",
            "drug_id",
            "drug_name",
            "protein_id",
            "target_chembl_id",
            "target_name",
            "known_target_record",
            "affinityScore",
            "rank",
        ],
        positive_detail,
    )
    write_csv(
        out_dir / "known_target_enrichment_by_cutoff.csv",
        [
            "cutoff",
            "hits",
            "recallPct",
            "precisionPct",
            "randomExpectedHits",
            "randomExpectedRecallPct",
            "randomPermutationP95Hits",
            "enrichmentVsRandom",
        ],
        enrichment,
    )
    return {
        "totalPairs": len(labels),
        "positivePairsFromKnownTargets": len(positive_pairs),
        "positivePairsInAffinityMatrix": int(sum(labels)),
        "positivePairsMissingFromAffinityMatrix": len(positive_pairs - positives_seen),
        "negativePairs": len(labels) - int(sum(labels)),
        "auroc": auc,
        "averagePrecision": ap,
        "randomPositiveRatePct": pct(sum(labels), len(labels)),
        "enrichment": enrichment,
        "positiveRankRows": len(positive_detail),
        "outputs": {
            "positivePairRanks": str(out_dir / "known_target_positive_pair_ranks.csv"),
            "enrichmentByCutoff": str(out_dir / "known_target_enrichment_by_cutoff.csv"),
        },
    }


def score_token(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return int(any(term in lower for term in terms))


def disease_terms() -> dict[str, dict[str, list[str]]]:
    return {
        "oncology": {
            "opentargets": ["cancer", "neoplasm", "tumor", "tumour", "carcinoma", "sarcoma", "leukemia", "lymphoma", "melanoma", "myeloma"],
            "txgnn": ["cancer", "neoplasm", "tumor", "carcinoma"],
        },
        "infectious_disease": {
            "opentargets": ["infection", "infectious", "bacterial", "viral", "virus", "hiv", "hepatitis", "influenza", "tuberculosis", "pneumonia", "fungal", "malaria"],
            "txgnn": ["infection", "infectious", "hiv", "hepatitis", "influenza", "tuberculosis", "pneumonia", "malaria"],
        },
        "cardiovascular": {
            "opentargets": ["cardiovascular", "heart", "hypertension", "coronary", "myocardial", "stroke", "thrombosis", "atherosclerosis", "arrhythmia"],
            "txgnn": ["cardiovascular", "heart", "hypertension", "coronary", "stroke", "thrombosis"],
        },
        "neurology_psychiatry": {
            "opentargets": ["neurolog", "psychiatr", "brain", "parkinson", "alzheimer", "epilepsy", "seizure", "depression", "schizophrenia", "migraine", "pain"],
            "txgnn": ["neurolog", "brain", "parkinson", "alzheimer", "epilepsy", "depression", "schizophrenia", "migraine", "pain"],
        },
        "immunology_inflammation": {
            "opentargets": ["immune", "inflamm", "arthritis", "psoriasis", "dermatitis", "asthma", "lupus", "crohn", "colitis", "allergy"],
            "txgnn": ["immune", "inflamm", "arthritis", "psoriasis", "asthma", "lupus", "crohn", "colitis"],
        },
    }


def build_external_evidence_maps(opentargets_path: Path, txgnn_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    terms = disease_terms()
    ot: dict[str, dict[str, float]] = {direction: {} for direction in terms}
    for row in read_csv(opentargets_path):
        protein_id = row.get("protein_id", "")
        disease_text = " ".join([row.get("disease_name", ""), row.get("disease_id", "")])
        score = number(row.get("overall_score")) or 0.0
        for direction, term_sets in terms.items():
            if score_token(disease_text, term_sets["opentargets"]):
                ot[direction][protein_id] = max(ot[direction].get(protein_id, 0.0), score)

    tx: dict[str, dict[str, float]] = {direction: {} for direction in terms}
    for row in read_csv(txgnn_path):
        drug_id = row.get("drug_id", "")
        disease_text = " ".join([row.get("disease_name", ""), row.get("disease_id", "")])
        score = number(row.get("txgnn_indication_score")) or 0.0
        for direction, term_sets in terms.items():
            if score_token(disease_text, term_sets["txgnn"]):
                tx[direction][drug_id] = max(tx[direction].get(drug_id, 0.0), score)
    return ot, tx


def compute_disease_evidence_gap(
    candidates_path: Path,
    opentargets_path: Path,
    txgnn_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    candidates = read_csv(candidates_path)
    ot_map, tx_map = build_external_evidence_maps(opentargets_path, txgnn_path)
    summary_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    rescue_rows: list[dict[str, Any]] = []

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[row.get("direction", "")].append(row)

    for direction, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(float(row.get("rank") or 999999)))
        direct_ot = sum(1 for row in rows if (number(row.get("openTargetsScore")) or 0) > 0)
        direct_tx = sum(1 for row in rows if (number(row.get("txgnnScore")) or 0) > 0)
        expanded_ot = 0
        expanded_tx = 0
        both_expanded = 0
        high_rank_gap = 0
        for row in rows:
            protein = row.get("protein", "")
            drug = row.get("drugId", "")
            rank = int(float(row.get("rank") or 999999))
            ot_score = max(number(row.get("openTargetsScore")) or 0.0, ot_map.get(direction, {}).get(protein, 0.0))
            tx_score = max(number(row.get("txgnnScore")) or 0.0, tx_map.get(direction, {}).get(drug, 0.0))
            if ot_score > 0:
                expanded_ot += 1
            if tx_score > 0:
                expanded_tx += 1
            if ot_score > 0 and tx_score > 0:
                both_expanded += 1
            if rank <= 1000 and (ot_score == 0 or tx_score == 0):
                high_rank_gap += 1
                gap_rows.append(
                    {
                        "direction": direction,
                        "labelZh": DIRECTION_LABELS.get(direction, direction),
                        "rank": rank,
                        "drugId": drug,
                        "drug": row.get("drug", ""),
                        "protein": protein,
                        "target": row.get("target", ""),
                        "directionScore": row.get("directionScore", ""),
                        "affinityScore": row.get("affinityScore", ""),
                        "missingOpenTargets": int(ot_score == 0),
                        "missingTxGNN": int(tx_score == 0),
                        "status": row.get("status", ""),
                    }
                )
            if rank <= 1000 and (number(row.get("openTargetsScore")) or 0) == 0 and ot_score > 0:
                rescue_rows.append(
                    {
                        "direction": direction,
                        "rank": rank,
                        "drug": row.get("drug", ""),
                        "target": row.get("target", ""),
                        "protein": protein,
                        "externalOpenTargetsScore": ot_score,
                        "note": "candidate gains disease-direction Open Targets evidence under text-term expansion",
                    }
                )
        summary_rows.append(
            {
                "direction": direction,
                "labelZh": DIRECTION_LABELS.get(direction, direction),
                "candidateRows": len(rows),
                "directOpenTargetsRows": direct_ot,
                "directOpenTargetsPct": pct(direct_ot, len(rows)),
                "expandedOpenTargetsRows": expanded_ot,
                "expandedOpenTargetsPct": pct(expanded_ot, len(rows)),
                "directTxGNNRows": direct_tx,
                "directTxGNNPct": pct(direct_tx, len(rows)),
                "expandedTxGNNRows": expanded_tx,
                "expandedTxGNNPct": pct(expanded_tx, len(rows)),
                "expandedBothRows": both_expanded,
                "expandedBothPct": pct(both_expanded, len(rows)),
                "top1000RowsMissingAnyExpandedEvidence": high_rank_gap,
            }
        )

    write_csv(
        out_dir / "disease_direction_external_evidence_coverage.csv",
        [
            "direction",
            "labelZh",
            "candidateRows",
            "directOpenTargetsRows",
            "directOpenTargetsPct",
            "expandedOpenTargetsRows",
            "expandedOpenTargetsPct",
            "directTxGNNRows",
            "directTxGNNPct",
            "expandedTxGNNRows",
            "expandedTxGNNPct",
            "expandedBothRows",
            "expandedBothPct",
            "top1000RowsMissingAnyExpandedEvidence",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / "top1000_external_evidence_gap_candidates.csv",
        [
            "direction",
            "labelZh",
            "rank",
            "drugId",
            "drug",
            "protein",
            "target",
            "directionScore",
            "affinityScore",
            "missingOpenTargets",
            "missingTxGNN",
            "status",
        ],
        gap_rows,
    )
    write_csv(
        out_dir / "top1000_opentargets_expansion_rescue_candidates.csv",
        ["direction", "rank", "drug", "target", "protein", "externalOpenTargetsScore", "note"],
        rescue_rows,
    )
    return {
        "summary": summary_rows,
        "top1000GapRows": len(gap_rows),
        "opentargetsExpansionRescueRows": len(rescue_rows),
        "outputs": {
            "coverage": str(out_dir / "disease_direction_external_evidence_coverage.csv"),
            "top1000EvidenceGaps": str(out_dir / "top1000_external_evidence_gap_candidates.csv"),
            "opentargetsExpansionRescues": str(out_dir / "top1000_opentargets_expansion_rescue_candidates.csv"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P0 SOTA validation calculations for BioMaster.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/sota_validation")
    parser.add_argument("--affinity", default="outputs/druggable_proteome/conplex_affinity_scores_druggable.csv")
    parser.add_argument("--known-audit", default="outputs/druggable_proteome/fda_known_target_recall_top100000_audit.csv")
    parser.add_argument("--candidates", default="outputs/disease_directions/disease_direction_integrated_candidates.csv")
    parser.add_argument("--opentargets", default="data/processed/opentargets_target_disease_scores.csv")
    parser.add_argument("--txgnn", default="data/processed/txgnn_drug_disease_scores.csv")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    known = compute_known_target_benchmark(
        root / args.affinity,
        root / args.known_audit,
        out_dir,
        args.permutations,
        args.seed,
    )
    evidence = compute_disease_evidence_gap(
        root / args.candidates,
        root / args.opentargets,
        root / args.txgnn,
        out_dir,
    )
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "affinity": args.affinity,
            "known_audit": args.known_audit,
            "candidates": args.candidates,
            "opentargets": args.opentargets,
            "txgnn": args.txgnn,
        },
        "known_target_benchmark": known,
        "disease_evidence_gap": evidence,
    }
    write_json(out_dir / "sota_p0_validation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

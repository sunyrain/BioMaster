from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "stage5_rank",
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "combined_ai_score",
    "disease_id",
    "disease_name",
    "direct_disease_score",
    "network_disease_score",
    "disease_evidence_status",
    "disease_source",
    "disease_source_detail",
    "network_source",
    "network_filter_rule",
    "final_priority_score",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 if maximum > 0 else 0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def best_open_targets_scores(
    rows: list[dict[str, str]],
    disease_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    by_protein: dict[str, dict[str, Any]] = {}
    by_gene: dict[str, dict[str, Any]] = {}
    disease_name = ""
    for row in rows:
        if row.get("disease_id") != disease_id:
            continue
        score = as_float(row.get("overall_score"), 0.0)
        disease_name = disease_name or row.get("disease_name", "")
        info = {
            "score": score,
            "disease_id": row.get("disease_id", ""),
            "disease_name": row.get("disease_name", ""),
            "source_detail": "OpenTargets associationByOverallDirect via GraphQL filtered to screening targets",
        }
        protein_id = (row.get("protein_id") or "").strip()
        if protein_id and score > as_float((by_protein.get(protein_id) or {}).get("score"), -1.0):
            by_protein[protein_id] = info
        for key in ("approved_symbol", "gene_name"):
            gene = (row.get(key) or "").strip().upper()
            if gene and score > as_float((by_gene.get(gene) or {}).get("score"), -1.0):
                by_gene[gene] = info
    return by_protein, by_gene, disease_name


def compute_string_network_scores(
    rows: list[dict[str, str]],
    disease_by_gene: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], str]:
    network_by_gene: dict[str, float] = {}
    disease_genes = set(disease_by_gene)
    filter_rule = ""
    for row in rows:
        gene_a = (row.get("preferredName_A") or row.get("protein1") or row.get("gene_a") or "").strip().upper()
        gene_b = (row.get("preferredName_B") or row.get("protein2") or row.get("gene_b") or "").strip().upper()
        if not gene_a or not gene_b:
            continue
        raw_score = as_float(row.get("score") or row.get("combined_score") or row.get("string_score"), 0.0)
        string_score = raw_score / 1000.0 if raw_score > 1.0 else raw_score
        filter_rule = filter_rule or row.get("filter_rule", "")
        if gene_a in disease_genes:
            propagated = string_score * as_float(disease_by_gene[gene_a].get("score"), 0.0)
            network_by_gene[gene_b] = max(network_by_gene.get(gene_b, 0.0), propagated)
        if gene_b in disease_genes:
            propagated = string_score * as_float(disease_by_gene[gene_b].get("score"), 0.0)
            network_by_gene[gene_a] = max(network_by_gene.get(gene_a, 0.0), propagated)
    return network_by_gene, filter_rule


def evidence_status(direct: float, network: float) -> str:
    if direct > 0 and network > 0:
        return "direct_and_network"
    if direct > 0:
        return "direct"
    if network > 0:
        return "network"
    return "none"


def rerank(
    stage4_csv: Path,
    opentargets_csv: Path,
    string_csv: Path,
    out_csv: Path,
    metadata_out: Path,
    disease_id: str,
    top_csv: Path | None = None,
    top_n: int = 100,
) -> list[dict[str, Any]]:
    stage4_rows = read_rows(stage4_csv)
    ot_rows = read_rows(opentargets_csv)
    string_rows = read_rows(string_csv)

    disease_by_protein, disease_by_gene, disease_name = best_open_targets_scores(ot_rows, disease_id)
    network_by_gene, string_filter_rule = compute_string_network_scores(string_rows, disease_by_gene)
    ai_norm = normalize([as_float(row.get("combined_ai_score"), 0.0) for row in stage4_rows])

    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(stage4_rows):
        protein_id = (row.get("protein_id") or "").strip()
        gene = (row.get("gene_name") or "").strip().upper()
        direct_info = disease_by_protein.get(protein_id) or disease_by_gene.get(gene) or {}
        direct = as_float(direct_info.get("score"), 0.0)
        network = network_by_gene.get(gene, 0.0)
        status = evidence_status(direct, network)
        has_evidence = status != "none"
        final_score = (0.55 * ai_norm[index]) + (0.30 * direct) + (0.15 * network)
        ranked.append(
            {
                "pair_id": row.get("pair_id", ""),
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "protein_id": row.get("protein_id", ""),
                "gene_name": row.get("gene_name", ""),
                "protein_name": row.get("protein_name", ""),
                "combined_ai_score": row.get("combined_ai_score", ""),
                "disease_id": disease_id if has_evidence else "",
                "disease_name": (direct_info.get("disease_name") or disease_name) if has_evidence else "",
                "direct_disease_score": round(direct, 9),
                "network_disease_score": round(network, 9),
                "disease_evidence_status": status,
                "disease_source": "Open Targets Platform" if direct > 0 else "",
                "disease_source_detail": direct_info.get("source_detail", "") if direct > 0 else "",
                "network_source": "STRING v12.0 API filtered human subnet" if network > 0 else "",
                "network_filter_rule": string_filter_rule if network > 0 else "",
                "final_priority_score": round(final_score, 9),
            }
        )

    ranked.sort(key=lambda item: as_float(item["final_priority_score"], 0.0), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["stage5_rank"] = rank

    write_rows(out_csv, OUTPUT_FIELDS, ranked)
    if top_csv:
        write_rows(top_csv, OUTPUT_FIELDS, ranked[:top_n])

    status_counts: dict[str, int] = {}
    for row in ranked:
        status_counts[row["disease_evidence_status"]] = status_counts.get(row["disease_evidence_status"], 0) + 1

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage4_csv": str(stage4_csv),
        "opentargets_csv": str(opentargets_csv),
        "string_csv": str(string_csv),
        "output_csv": str(out_csv),
        "top_csv": str(top_csv) if top_csv else "",
        "rows": len(ranked),
        "disease_id": disease_id,
        "disease_name": disease_name,
        "open_targets_direct_genes": len(disease_by_gene),
        "open_targets_direct_proteins": len(disease_by_protein),
        "string_network_genes": len(network_by_gene),
        "evidence_status_counts": status_counts,
        "score_formula": "0.55 * normalized_combined_ai_score + 0.30 * OpenTargets_direct_score + 0.15 * STRING_network_score",
        "notes": "Rows without Open Targets direct evidence or STRING propagated evidence leave disease_id/disease_name blank and are marked disease_evidence_status=none.",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerank Stage 5 with real Open Targets and STRING evidence.")
    parser.add_argument("--stage4", default="outputs/report_scale/stage4_affinity_candidates_915k.csv")
    parser.add_argument("--opentargets", default="data/processed/opentargets_target_disease_scores.csv")
    parser.add_argument("--string", default="data/processed/string_human_filtered_edges.csv")
    parser.add_argument("--out", default="outputs/report_scale/stage5_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--metadata-out", default="outputs/report_scale/stage5_open_targets_string_ranked_candidates_915k.metadata.json")
    parser.add_argument("--top-csv", default="outputs/report_scale/top100_stage5_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--disease-id", default="MONDO_0004992")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    rerank(
        stage4_csv=Path(args.stage4),
        opentargets_csv=Path(args.opentargets),
        string_csv=Path(args.string),
        out_csv=Path(args.out),
        metadata_out=Path(args.metadata_out),
        disease_id=args.disease_id,
        top_csv=Path(args.top_csv) if args.top_csv else None,
        top_n=args.top_n,
    )
    print(Path(args.out))
    print(Path(args.metadata_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

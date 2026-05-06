from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

import requests


STRING_BASE_URL = "https://version-12-0.string-db.org/api/tsv"
OUTPUT_FIELDS = [
    "stringId_A",
    "stringId_B",
    "preferredName_A",
    "preferredName_B",
    "score",
    "nscore",
    "fscore",
    "pscore",
    "ascore",
    "escore",
    "dscore",
    "tscore",
    "source_query",
    "filter_rule",
]


def read_gene_names(path: Path) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gene = (row.get("gene_name") or "").strip()
            if not gene or gene in seen:
                continue
            genes.append(gene)
            seen.add(gene)
    return genes


def fetch_string_tsv(
    endpoint: str,
    identifiers: Iterable[str],
    *,
    species: int,
    required_score: int,
    caller_identity: str,
    limit: int | None = None,
    timeout: int = 180,
) -> list[dict[str, str]]:
    params: dict[str, str | int] = {
        "identifiers": "\r".join(identifiers),
        "species": species,
        "required_score": required_score,
        "caller_identity": caller_identity,
    }
    if limit is not None:
        params["limit"] = limit

    response = requests.post(f"{STRING_BASE_URL}/{endpoint}", data=params, timeout=timeout)
    response.raise_for_status()
    text = response.text.strip()
    if not text:
        return []
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    return list(reader)


def normalize_row(row: dict[str, str], source_query: str, filter_rule: str) -> dict[str, str]:
    normalized = {field: row.get(field, "") for field in OUTPUT_FIELDS}
    normalized["source_query"] = source_query
    normalized["filter_rule"] = filter_rule
    return normalized


def score_value(row: dict[str, str]) -> float:
    try:
        return float(row.get("score") or 0.0)
    except ValueError:
        return 0.0


def merge_edges(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        a = row.get("stringId_A", "")
        b = row.get("stringId_B", "")
        if not a or not b:
            continue
        key = tuple(sorted((a, b)))
        current = merged.get(key)
        if current is None or score_value(row) > score_value(current):
            if current is not None:
                row["source_query"] = merge_labels(current.get("source_query", ""), row.get("source_query", ""))
                row["filter_rule"] = merge_labels(current.get("filter_rule", ""), row.get("filter_rule", ""))
            merged[key] = row
        else:
            current["source_query"] = merge_labels(current.get("source_query", ""), row.get("source_query", ""))
            current["filter_rule"] = merge_labels(current.get("filter_rule", ""), row.get("filter_rule", ""))
    return sorted(
        merged.values(),
        key=lambda item: (
            -score_value(item),
            item.get("preferredName_A", ""),
            item.get("preferredName_B", ""),
        ),
    )


def merge_labels(left: str, right: str) -> str:
    labels: list[str] = []
    for value in (left, right):
        for label in value.split(";"):
            label = label.strip()
            if label and label not in labels:
                labels.append(label)
    return ";".join(labels)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a filtered human STRING v12.0 subnet for BioMaster Stage 5.")
    parser.add_argument(
        "--proteins",
        default="outputs/report_scale/protein_library_1000.csv",
        help="Protein library CSV with gene_name column.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/string_human_filtered_edges.csv",
        help="Output filtered STRING edge CSV.",
    )
    parser.add_argument(
        "--metadata-out",
        default="data/processed/string_human_filtered_edges.metadata.json",
        help="Output metadata JSON.",
    )
    parser.add_argument("--required-score", type=int, default=700)
    parser.add_argument("--partner-limit", type=int, default=100)
    parser.add_argument("--species", type=int, default=9606)
    parser.add_argument("--caller-identity", default="BioMaster")
    parser.add_argument("--disease-seeds", default="TP53,EGFR,BCL2,JAK1")
    args = parser.parse_args()

    protein_path = Path(args.proteins)
    out_path = Path(args.out)
    metadata_path = Path(args.metadata_out)
    screening_genes = read_gene_names(protein_path)
    disease_seed_genes = [gene.strip() for gene in args.disease_seeds.split(",") if gene.strip()]

    if not screening_genes:
        raise ValueError(f"No gene_name values found in {protein_path}")
    if not disease_seed_genes:
        raise ValueError("No disease seed genes supplied")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    all_rows: list[dict[str, str]] = []

    internal_rows = fetch_string_tsv(
        "network",
        screening_genes,
        species=args.species,
        required_score=args.required_score,
        caller_identity=args.caller_identity,
    )
    all_rows.extend(
        normalize_row(
            row,
            source_query="screening_protein_library_1000",
            filter_rule=f"network_required_score_{args.required_score}",
        )
        for row in internal_rows
    )

    partner_rows = fetch_string_tsv(
        "interaction_partners",
        disease_seed_genes,
        species=args.species,
        required_score=args.required_score,
        caller_identity=args.caller_identity,
        limit=args.partner_limit,
    )
    seed_set = set(disease_seed_genes)
    for row in partner_rows:
        row_genes = {row.get("preferredName_A", ""), row.get("preferredName_B", "")}
        source_query = ";".join(gene for gene in disease_seed_genes if gene in row_genes) or "disease_seed_genes"
        all_rows.append(
            normalize_row(
                row,
                source_query=source_query,
                filter_rule=f"interaction_partners_limit_{args.partner_limit}_required_score_{args.required_score}",
            )
        )

    merged_rows = merge_edges(all_rows)
    write_csv(out_path, merged_rows)

    metadata = {
        "created_utc": started,
        "string_version": "12.0",
        "species": args.species,
        "required_score": args.required_score,
        "partner_limit": args.partner_limit,
        "screening_gene_count": len(screening_genes),
        "disease_seed_genes": disease_seed_genes,
        "raw_internal_edge_rows": len(internal_rows),
        "raw_partner_edge_rows": len(partner_rows),
        "merged_edge_rows": len(merged_rows),
        "output_csv": str(out_path),
        "api_endpoints": {
            "network": f"{STRING_BASE_URL}/network",
            "interaction_partners": f"{STRING_BASE_URL}/interaction_partners",
        },
    }
    write_metadata(metadata_path, metadata)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

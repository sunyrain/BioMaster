from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import requests


GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
ASSOCIATED_TARGETS_QUERY = """
query diseaseTargets($efoId: String!, $pageIndex: Int!, $pageSize: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: $pageIndex, size: $pageSize}) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
          biotype
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""

OUTPUT_FIELDS = [
    "protein_id",
    "gene_name",
    "ensembl_gene_id",
    "approved_symbol",
    "approved_name",
    "disease_id",
    "disease_name",
    "overall_score",
    "genetic_association_score",
    "somatic_mutation_score",
    "known_drug_score",
    "clinical_score",
    "affected_pathway_score",
    "literature_score",
    "animal_model_score",
    "rna_expression_score",
    "datatype_scores_json",
    "source",
]


def read_screening_targets(path: Path) -> tuple[set[str], dict[str, str]]:
    genes: set[str] = set()
    gene_to_protein: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gene = (row.get("gene_name") or "").strip()
            protein = (row.get("protein_id") or "").strip()
            if not gene:
                continue
            genes.add(gene)
            gene_to_protein.setdefault(gene, protein)
    return genes, gene_to_protein


def post_graphql(query: str, variables: dict[str, object], timeout: int = 90) -> dict[str, object]:
    response = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
    return payload


def fetch_disease_targets(disease_id: str, page_size: int, sleep_seconds: float) -> tuple[dict[str, str], list[dict[str, object]]]:
    first = post_graphql(
        ASSOCIATED_TARGETS_QUERY,
        {"efoId": disease_id, "pageIndex": 0, "pageSize": page_size},
    )
    disease = ((first.get("data") or {}).get("disease") or {})
    if not disease:
        raise ValueError(f"Open Targets disease not found: {disease_id}")
    associated = disease["associatedTargets"]
    count = int(associated["count"])
    rows = list(associated["rows"])
    pages = math.ceil(count / page_size)

    for page in range(1, pages):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        payload = post_graphql(
            ASSOCIATED_TARGETS_QUERY,
            {"efoId": disease_id, "pageIndex": page, "pageSize": page_size},
        )
        disease_payload = ((payload.get("data") or {}).get("disease") or {})
        rows.extend(disease_payload["associatedTargets"]["rows"])

    disease_meta = {"id": disease["id"], "name": disease["name"], "count": count}
    return disease_meta, rows


def datatype_score_map(row: dict[str, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in row.get("datatypeScores") or []:
        if not isinstance(item, dict):
            continue
        score_id = str(item.get("id") or "")
        if not score_id:
            continue
        try:
            scores[score_id] = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            scores[score_id] = 0.0
    return scores


def output_row(
    row: dict[str, object],
    disease_meta: dict[str, str],
    gene_to_protein: dict[str, str],
) -> dict[str, object]:
    target = row["target"]
    symbol = str(target.get("approvedSymbol") or "")
    scores = datatype_score_map(row)
    return {
        "protein_id": gene_to_protein.get(symbol, ""),
        "gene_name": symbol,
        "ensembl_gene_id": target.get("id", ""),
        "approved_symbol": symbol,
        "approved_name": target.get("approvedName", ""),
        "disease_id": disease_meta["id"],
        "disease_name": disease_meta["name"],
        "overall_score": round(float(row.get("score") or 0.0), 9),
        "genetic_association_score": round(scores.get("genetic_association", 0.0), 9),
        "somatic_mutation_score": round(scores.get("somatic_mutation", 0.0), 9),
        "known_drug_score": round(scores.get("known_drug", 0.0), 9),
        "clinical_score": round(scores.get("clinical", 0.0), 9),
        "affected_pathway_score": round(scores.get("affected_pathway", 0.0), 9),
        "literature_score": round(scores.get("literature", 0.0), 9),
        "animal_model_score": round(scores.get("animal_model", 0.0), 9),
        "rna_expression_score": round(scores.get("rna_expression", 0.0), 9),
        "datatype_scores_json": json.dumps(scores, ensure_ascii=False, sort_keys=True),
        "source": "OpenTargets Platform GraphQL API",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download filtered Open Targets disease-target associations.")
    parser.add_argument("--proteins", default="outputs/report_scale/protein_library_1000.csv")
    parser.add_argument("--out", default="data/processed/opentargets_target_disease_scores.csv")
    parser.add_argument("--metadata-out", default="data/processed/opentargets_target_disease_scores.metadata.json")
    parser.add_argument("--disease-ids", default="MONDO_0004992,EFO_0000616")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    genes, gene_to_protein = read_screening_targets(Path(args.proteins))
    disease_ids = [item.strip() for item in args.disease_ids.split(",") if item.strip()]
    if not genes:
        raise ValueError("No gene names found in protein library")
    if not disease_ids:
        raise ValueError("No disease IDs supplied")

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output: list[dict[str, object]] = []
    disease_stats: list[dict[str, object]] = []
    for disease_id in disease_ids:
        disease_meta, rows = fetch_disease_targets(disease_id, args.page_size, args.sleep)
        filtered_rows = []
        for row in rows:
            target = row.get("target") or {}
            symbol = str(target.get("approvedSymbol") or "")
            if symbol in genes:
                filtered_rows.append(output_row(row, disease_meta, gene_to_protein))
        output.extend(filtered_rows)
        disease_stats.append(
            {
                "disease_id": disease_meta["id"],
                "disease_name": disease_meta["name"],
                "open_targets_associated_target_count": disease_meta["count"],
                "matched_screening_targets": len(filtered_rows),
            }
        )

    output.sort(
        key=lambda row: (
            str(row["disease_id"]),
            -float(row["overall_score"]),
            str(row["gene_name"]),
        )
    )
    write_csv(Path(args.out), output)
    metadata = {
        "created_utc": started,
        "source": "Open Targets Platform GraphQL API",
        "graphql_url": GRAPHQL_URL,
        "protein_library": args.proteins,
        "screening_gene_count": len(genes),
        "disease_stats": disease_stats,
        "output_rows": len(output),
        "output_csv": args.out,
        "notes": "Filtered to genes in the 1000-protein screening library. MONDO_0004992/cancer is the primary replacement for the previous C0006826 malignant neoplasm placeholder; EFO_0000616/neoplasm is retained as a broader comparison disease.",
    }
    write_metadata(Path(args.metadata_out), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

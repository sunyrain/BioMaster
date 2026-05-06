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
    "open_targets_string_priority_score",
    "txgnn_indication_score",
    "txgnn_indication_logit",
    "txgnn_drugbank_id",
    "txgnn_drug_name",
    "txgnn_mapping_status",
    "txgnn_component_status",
    "final_priority_score",
]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


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


def merge(
    stage5_csv: Path,
    txgnn_csv: Path,
    out_csv: Path,
    top_csv: Path,
    metadata_out: Path,
    top_n: int,
) -> dict[str, Any]:
    _, stage5_rows = read_rows(stage5_csv)
    _, txgnn_rows = read_rows(txgnn_csv)
    txgnn_by_drug = {row["drug_id"]: row for row in txgnn_rows}

    merged: list[dict[str, Any]] = []
    for row in stage5_rows:
        txgnn = txgnn_by_drug.get(row.get("drug_id", ""))
        open_targets_string_score = as_float(row.get("final_priority_score"), 0.0)
        if txgnn:
            txgnn_score = as_float(txgnn.get("txgnn_indication_score"), 0.0)
            final_score = (0.80 * open_targets_string_score) + (0.20 * txgnn_score)
            txgnn_component_status = "mapped"
        else:
            txgnn_score = None
            final_score = open_targets_string_score
            txgnn_component_status = "unmapped"

        merged.append(
            {
                "pair_id": row.get("pair_id", ""),
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "protein_id": row.get("protein_id", ""),
                "gene_name": row.get("gene_name", ""),
                "protein_name": row.get("protein_name", ""),
                "combined_ai_score": row.get("combined_ai_score", ""),
                "disease_id": row.get("disease_id", ""),
                "disease_name": row.get("disease_name", ""),
                "direct_disease_score": row.get("direct_disease_score", ""),
                "network_disease_score": row.get("network_disease_score", ""),
                "disease_evidence_status": row.get("disease_evidence_status", ""),
                "disease_source": row.get("disease_source", ""),
                "disease_source_detail": row.get("disease_source_detail", ""),
                "network_source": row.get("network_source", ""),
                "network_filter_rule": row.get("network_filter_rule", ""),
                "open_targets_string_priority_score": row.get("final_priority_score", ""),
                "txgnn_indication_score": "" if txgnn_score is None else txgnn.get("txgnn_indication_score", ""),
                "txgnn_indication_logit": "" if txgnn is None else txgnn.get("txgnn_indication_logit", ""),
                "txgnn_drugbank_id": "" if txgnn is None else txgnn.get("txgnn_drugbank_id", ""),
                "txgnn_drug_name": "" if txgnn is None else txgnn.get("txgnn_drug_name", ""),
                "txgnn_mapping_status": "" if txgnn is None else txgnn.get("mapping_status", ""),
                "txgnn_component_status": txgnn_component_status,
                "final_priority_score": round(final_score, 9),
            }
        )

    merged.sort(key=lambda item: as_float(item["final_priority_score"], 0.0), reverse=True)
    for rank, row in enumerate(merged, start=1):
        row["stage5_rank"] = rank

    write_rows(out_csv, OUTPUT_FIELDS, merged)
    write_rows(top_csv, OUTPUT_FIELDS, merged[:top_n])

    mapped_rows = sum(1 for row in merged if row["txgnn_component_status"] == "mapped")
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage5_input_csv": str(stage5_csv),
        "txgnn_scores_csv": str(txgnn_csv),
        "output_csv": str(out_csv),
        "top_csv": str(top_csv),
        "rows": len(merged),
        "txgnn_drug_score_rows": len(txgnn_rows),
        "rows_with_txgnn_score": mapped_rows,
        "rows_without_txgnn_score": len(merged) - mapped_rows,
        "score_formula": "Rows with TxGNN mapping: 0.80 * OpenTargets_STRING_priority + 0.20 * TxGNN_indication_score. Rows without TxGNN mapping retain OpenTargets_STRING_priority; missing TxGNN is not treated as negative evidence.",
        "notes": "TxGNN score is drug-disease level and is applied to all protein pairs for the mapped drug.",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge TxGNN drug-disease indication scores into Stage 5 ranking.")
    parser.add_argument("--stage5", default="outputs/report_scale/stage5_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--txgnn", default="data/processed/txgnn_drug_disease_scores.csv")
    parser.add_argument("--out", default="outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--top-csv", default="outputs/report_scale/top100_stage5_txgnn_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--metadata-out", default="outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.metadata.json")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    metadata = merge(
        stage5_csv=Path(args.stage5),
        txgnn_csv=Path(args.txgnn),
        out_csv=Path(args.out),
        top_csv=Path(args.top_csv),
        metadata_out=Path(args.metadata_out),
        top_n=args.top_n,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

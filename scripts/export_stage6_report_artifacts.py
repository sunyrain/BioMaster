from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def counter_dict(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(Counter(row.get(field, "") or "NA" for row in rows))


def compact_candidate(row: dict[str, str]) -> dict[str, str]:
    fields = [
        "stage6_rank",
        "drug_name",
        "gene_name",
        "protein_id",
        "stage5_final_priority_score",
        "diffdock_confidence",
        "stage6_consensus_score",
        "structural_status",
        "disease_evidence_status",
        "txgnn_component_status",
    ]
    return {field: row.get(field, "") for field in fields}


def export(args: argparse.Namespace) -> dict[str, Any]:
    input_csv = Path(args.input)
    rows = read_csv(input_csv)
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    fieldnames = list(rows[0].keys())
    top_rows = rows[: args.top_n]
    failed_rows = [row for row in rows if row.get("structural_status") != "completed"]

    write_csv(Path(args.top_out), fieldnames, top_rows)
    write_csv(Path(args.failure_out), fieldnames, failed_rows)

    completed = [row for row in rows if row.get("structural_status") == "completed"]
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(input_csv),
        "rows": len(rows),
        "unique_drugs": len({row.get("drug_id", "") for row in rows}),
        "unique_proteins": len({row.get("protein_id", "") for row in rows}),
        "diffdock_completed": len(completed),
        "diffdock_missing_or_failed": len(failed_rows),
        "top_n": args.top_n,
        "top_unique_drugs": len({row.get("drug_id", "") for row in top_rows}),
        "top_unique_proteins": len({row.get("protein_id", "") for row in top_rows}),
        "structural_status": counter_dict(rows, "structural_status"),
        "disease_evidence_status": counter_dict(rows, "disease_evidence_status"),
        "txgnn_component_status": counter_dict(rows, "txgnn_component_status"),
        "diffdock_receptor_status": counter_dict(rows, "diffdock_receptor_status"),
        "top_candidates": [compact_candidate(row) for row in rows[: args.preview_n]],
        "top_out": args.top_out,
        "failure_out": args.failure_out,
    }

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Stage 6 report tables and audit artifacts.")
    parser.add_argument("--input", default="outputs/report_scale/stage6_top1000_consensus_candidates.csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--preview-n", type=int, default=20)
    parser.add_argument("--top-out", default="outputs/report_scale/stage6_top100_consensus_candidates.csv")
    parser.add_argument("--failure-out", default="outputs/report_scale/diffdock_top1000_smiles_failure_audit.csv")
    parser.add_argument("--summary-out", default="outputs/report_scale/stage6_top1000_report_summary.json")
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

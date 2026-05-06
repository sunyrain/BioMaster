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


def counter_rows(counter: Counter[str], key_name: str, value_name: str = "count", limit: int = 25) -> list[dict[str, Any]]:
    return [{key_name: key, value_name: count} for key, count in counter.most_common(limit)]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(Path(args.input))
    top_rows = rows[: args.top_n]
    gene_counter = Counter(row.get("gene_name") or row.get("stage5_gene_name", "") for row in top_rows)
    drug_counter = Counter(row.get("drug_name") or row.get("stage5_drug_name", "") for row in top_rows)
    evidence_counter = Counter(row.get("stage5_disease_evidence_status") or row.get("disease_evidence_status", "") for row in top_rows)
    txgnn_counter = Counter(row.get("stage5_txgnn_component_status") or row.get("txgnn_component_status", "") for row in top_rows)
    receptor_counter = Counter(row.get("diffdock_receptor_status", "") for row in top_rows)

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": args.input,
        "top_n": args.top_n,
        "rows": len(top_rows),
        "unique_drugs": len({row.get("drug_id") or row.get("stage5_drug_id", "") for row in top_rows}),
        "unique_proteins": len({row.get("protein_id") or row.get("stage5_protein_id", "") for row in top_rows}),
        "unique_genes": len({row.get("gene_name") or row.get("stage5_gene_name", "") for row in top_rows}),
        "top_genes": counter_rows(gene_counter, "gene_name", limit=20),
        "top_drugs": counter_rows(drug_counter, "drug_name", limit=20),
        "disease_evidence_status": dict(evidence_counter),
        "txgnn_component_status": dict(txgnn_counter),
        "diffdock_receptor_status": dict(receptor_counter),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(out_json.with_suffix(".top_genes.csv"), ["gene_name", "count"], summary["top_genes"])
    write_csv(out_json.with_suffix(".top_drugs.csv"), ["drug_name", "count"], summary["top_drugs"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 5 or Stage 5 Top-N manifest.")
    parser.add_argument("--input", default="outputs/report_scale/stage5_top1000_diffdock_ready_manifest.csv")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--out-json", default="outputs/report_scale/stage5_top1000_summary.json")
    args = parser.parse_args()
    summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

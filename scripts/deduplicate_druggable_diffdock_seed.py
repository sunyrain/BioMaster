from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


EXTRA_FIELDS = ["represented_pair_count", "represented_protein_ids", "represented_gene_names"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate druggable proteome DiffDock seed rows by drug and protein sequence.")
    parser.add_argument("--input", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest.csv")
    parser.add_argument("--output", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest_unique_sequences.csv")
    parser.add_argument("--metadata", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest_unique_sequences.metadata.json")
    args = parser.parse_args()

    rows = read_csv(Path(args.input))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["drug_id"], row["sequence_key"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: min(int(row.get("stage4_rank") or 10**12) for row in groups[item])):
        members = groups[key]
        representative = dict(members[0])
        representative["represented_pair_count"] = len(members)
        representative["represented_protein_ids"] = ";".join(sorted({row["protein_id"] for row in members if row.get("protein_id")}))
        representative["represented_gene_names"] = ";".join(sorted({row["gene_name"] for row in members if row.get("gene_name")}))
        output_rows.append(representative)

    fieldnames = list(rows[0].keys()) if rows else []
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(Path(args.output), fieldnames, output_rows)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": args.input,
        "output": args.output,
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "deduplication_key": ["drug_id", "sequence_key"],
        "represented_pair_count_total": sum(len(members) for members in groups.values()),
    }
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

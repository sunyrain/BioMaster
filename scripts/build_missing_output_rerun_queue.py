from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


DIRECTIONS = [
    "oncology",
    "infectious_disease",
    "cardiovascular",
    "neurology_psychiatry",
    "immunology_inflammation",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any, default: float = 0.0) -> float:
    if value in ("", None, "NA"):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_ready_rows(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for direction in DIRECTIONS:
        ready_path = root / "outputs/disease_directions" / direction / "top10000_diffdock_ready.csv"
        if not ready_path.exists():
            continue
        for row in read_csv(ready_path):
            rows[(direction, row["pair_id"])] = row
    return rows


def sort_key(row: dict[str, str]) -> tuple[float, float, float, int]:
    return (
        -number(row.get("credibilityScore")),
        -number(row.get("directionScore")),
        -number(row.get("affinityScore")),
        int(number(row.get("rank"), 999999)),
    )


def select_candidates(rows: list[dict[str, str]], limit: int, per_direction_min: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("status") != "completed"
        and row.get("credibilityTierZh", "").startswith("D｜结构补跑")
    ]
    by_direction: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(candidates, key=sort_key):
        by_direction[row["direction"]].append(row)

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for direction in DIRECTIONS:
        for row in by_direction.get(direction, [])[:per_direction_min]:
            key = (row["direction"], row["pairId"])
            if key not in seen:
                selected.append(row)
                seen.add(key)

    for row in sorted(candidates, key=sort_key):
        if len(selected) >= limit:
            break
        key = (row["direction"], row["pairId"])
        if key in seen:
            continue
        selected.append(row)
        seen.add(key)

    return selected[:limit]


def build_queue(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    integrated = read_csv(root / "outputs/disease_directions/disease_direction_integrated_candidates.csv")
    ready = load_ready_rows(root)
    selected = select_candidates(integrated, args.limit, args.per_direction_min)

    out_dir = root / args.out_dir
    input_dir = out_dir / "inputs"
    output_dir = out_dir / "outputs"
    score_dir = out_dir / "scores"
    log_dir = out_dir / "logs"

    manifest_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in selected:
        ready_row = ready.get((row["direction"], row["pairId"]))
        if not ready_row:
            skipped.append({**row, "skip_reason": "ready_row_missing"})
            continue
        protein_path = ready_row.get("diffdock_receptor_pdb_path") or ready_row.get("receptor_pdb_path")
        ligand_description = ready_row.get("ligand_smiles") or ready_row.get("ligand_sdf_path")
        if not protein_path or not Path(protein_path).exists():
            skipped.append({**row, "skip_reason": "protein_path_missing"})
            continue
        if not ligand_description:
            skipped.append({**row, "skip_reason": "ligand_description_missing"})
            continue
        input_rows.append(
            {
                "complex_name": row["pairId"],
                "protein_path": protein_path,
                "protein_sequence": "",
                "ligand_description": ligand_description,
            }
        )
        manifest_rows.append(
            {
                "direction": row["direction"],
                "directionLabelZh": row.get("directionLabelZh", ""),
                "rank": row.get("rank", ""),
                "pairId": row["pairId"],
                "drug": row.get("drug", ""),
                "target": row.get("target", ""),
                "protein": row.get("protein", ""),
                "directionScore": row.get("directionScore", ""),
                "affinityScore": row.get("affinityScore", ""),
                "credibilityScore": row.get("credibilityScore", ""),
                "credibilityTierZh": row.get("credibilityTierZh", ""),
                "nextStepZh": row.get("nextStepZh", ""),
                "protein_path": protein_path,
                "ligand_description": ligand_description,
            }
        )

    job_rows: list[dict[str, Any]] = []
    for job_id, start in enumerate(range(0, len(input_rows), args.chunk_size)):
        chunk_rows = input_rows[start : start + args.chunk_size]
        chunk_name = f"diffdock_missing_priority_chunk_{job_id:05d}.csv"
        chunk_csv = input_dir / chunk_name
        write_csv(chunk_csv, ["complex_name", "protein_path", "protein_sequence", "ligand_description"], chunk_rows)
        job_rows.append(
            {
                "job_id": job_id,
                "chunk_csv": rel(root, chunk_csv),
                "out_dir": rel(root, output_dir / f"chunk_{job_id:05d}"),
                "score_csv": rel(root, score_dir / f"diffdock_missing_priority_chunk_{job_id:05d}.scores.csv"),
                "log_file": rel(root, log_dir / f"diffdock_missing_priority_chunk_{job_id:05d}.log"),
                "row_count": len(chunk_rows),
                "gpu_slot": job_id % 4,
                "status": "pending",
            }
        )

    job_index = out_dir / "diffdock_missing_priority_job_index.csv"
    write_csv(
        job_index,
        ["job_id", "chunk_csv", "out_dir", "score_csv", "log_file", "row_count", "gpu_slot", "status"],
        job_rows,
    )
    write_csv(
        out_dir / "missing_priority_manifest.csv",
        [
            "direction",
            "directionLabelZh",
            "rank",
            "pairId",
            "drug",
            "target",
            "protein",
            "directionScore",
            "affinityScore",
            "credibilityScore",
            "credibilityTierZh",
            "nextStepZh",
            "protein_path",
            "ligand_description",
        ],
        manifest_rows,
    )
    if skipped:
        write_csv(out_dir / "missing_priority_skipped.csv", sorted(skipped[0]), skipped)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected": len(selected),
        "queued": len(input_rows),
        "skipped": len(skipped),
        "jobs": len(job_rows),
        "chunk_size": args.chunk_size,
        "limit": args.limit,
        "per_direction_min": args.per_direction_min,
        "job_index": rel(root, job_index),
    }
    (out_dir / "missing_priority_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a small rerun queue for high-priority missing DiffDock outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/disease_directions/missing_output_priority_rerun")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--per-direction-min", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    metadata = build_queue(root, args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

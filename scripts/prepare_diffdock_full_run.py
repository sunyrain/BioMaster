from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_complex_name(pair_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_id)


def abs_path(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def prepare(
    manifest: Path,
    out_dir: Path,
    root: Path,
    chunk_size: int,
    gpu_count: int,
    samples_per_complex: int,
    inference_steps: int,
    actual_steps: int,
    batch_size: int,
    limit: int | None,
    ligand_source: str,
) -> dict[str, Any]:
    inputs_dir = out_dir / "inputs"
    outputs_dir = out_dir / "outputs"
    logs_dir = out_dir / "logs"
    scores_dir = out_dir / "scores"
    for path in [inputs_dir, outputs_dir, logs_dir, scores_dir]:
        path.mkdir(parents=True, exist_ok=True)

    input_fields = ["complex_name", "protein_path", "protein_sequence", "ligand_description"]
    job_fields = [
        "job_id",
        "chunk_csv",
        "out_dir",
        "score_csv",
        "log_file",
        "row_count",
        "gpu_slot",
        "status",
    ]

    jobs: list[dict[str, Any]] = []
    buffer: list[dict[str, str]] = []
    total_rows = 0
    skipped = 0

    def flush() -> None:
        if not buffer:
            return
        job_id = len(jobs)
        chunk_csv = inputs_dir / f"diffdock_full_chunk_{job_id:05d}.csv"
        chunk_out = outputs_dir / f"chunk_{job_id:05d}"
        score_csv = scores_dir / f"diffdock_full_chunk_{job_id:05d}.scores.csv"
        log_file = logs_dir / f"diffdock_full_chunk_{job_id:05d}.log"
        write_csv(chunk_csv, input_fields, buffer)
        jobs.append(
            {
                "job_id": job_id,
                "chunk_csv": str(chunk_csv),
                "out_dir": str(chunk_out),
                "score_csv": str(score_csv),
                "log_file": str(log_file),
                "row_count": len(buffer),
                "gpu_slot": job_id % gpu_count,
                "status": "pending",
            }
        )
        buffer.clear()

    for row in read_rows(manifest):
        ready_value = row.get("diffdock_ready", row.get("structure_ready", ""))
        if ready_value not in {"true", "True", "1", "yes"}:
            skipped += 1
            continue
        ligand_sdf = row.get("ligand_sdf_path", "")
        ligand_smiles = row.get("ligand_smiles", "")
        ligand = ligand_smiles if ligand_source == "smiles" and ligand_smiles else ligand_sdf
        receptor = row.get("diffdock_receptor_pdb_path") or row.get("receptor_pdb_path") or row.get("receptor_pdb_url") or ""
        if not ligand or not receptor:
            skipped += 1
            continue
        buffer.append(
            {
                "complex_name": safe_complex_name(row["pair_id"]),
                "protein_path": abs_path(root, receptor),
                "protein_sequence": "",
                "ligand_description": ligand if ligand_source == "smiles" and ligand == ligand_smiles else abs_path(root, ligand),
            }
        )
        total_rows += 1
        if len(buffer) >= chunk_size:
            flush()
        if limit is not None and total_rows >= limit:
            break
    flush()

    job_index = out_dir / "diffdock_full_job_index.csv"
    write_csv(job_index, job_fields, jobs)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest),
        "run_dir": str(out_dir),
        "job_index": str(job_index),
        "input_chunks": len(jobs),
        "structure_ready_rows": total_rows,
        "skipped_rows": skipped,
        "chunk_size": chunk_size,
        "gpu_count": gpu_count,
        "samples_per_complex": samples_per_complex,
        "inference_steps": inference_steps,
        "actual_steps": actual_steps,
        "batch_size": batch_size,
        "limit": limit,
        "ligand_source": ligand_source,
        "estimated_rank1_output_gb": round((total_rows * 9_000) / (1024**3), 2),
        "notes": "Prepared for full DiffDock execution. This script does not run inference.",
    }
    metadata_path = out_dir / "diffdock_full_run.metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare full DiffDock chunk CSVs and job index.")
    parser.add_argument("--manifest", default="outputs/report_scale/manifest_915k_structure_ready.csv")
    parser.add_argument("--out-dir", default="outputs/report_scale/diffdock_full_run")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--samples-per-complex", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--actual-steps", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ligand-source", choices=["sdf", "smiles"], default="sdf")
    args = parser.parse_args()

    prepare(
        manifest=Path(args.manifest),
        out_dir=Path(args.out_dir),
        root=Path.cwd(),
        chunk_size=args.chunk_size,
        gpu_count=args.gpu_count,
        samples_per_complex=args.samples_per_complex,
        inference_steps=args.inference_steps,
        actual_steps=args.actual_steps,
        batch_size=args.batch_size,
        limit=args.limit,
        ligand_source=args.ligand_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

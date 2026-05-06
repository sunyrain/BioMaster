from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any


CONFIDENCE_RE = re.compile(r"rank1_confidence([-+]?\d+(?:\.\d+)?)\.sdf$")
FIELDNAMES = [
    "pair_id",
    "complex_name",
    "diffdock_confidence",
    "docking_score",
    "rank1_sdf_path",
    "confidence_sdf_path",
    "source_chunk",
    "status",
    "error",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def parse_confidence_file(complex_dir: Path) -> tuple[float | None, Path | None]:
    matches: list[tuple[float, Path]] = []
    for path in complex_dir.glob("rank1_confidence*.sdf"):
        match = CONFIDENCE_RE.match(path.name)
        if match:
            matches.append((float(match.group(1)), path))
    if not matches:
        return None, None
    matches.sort(key=lambda item: item[1].name)
    return matches[0]


def parse_chunk_outputs(chunk_csv: Path, out_dir: Path, source_chunk: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_row in read_csv(chunk_csv):
        complex_name = input_row["complex_name"]
        complex_dir = out_dir / complex_name
        confidence, confidence_sdf = parse_confidence_file(complex_dir)
        rank1_sdf = complex_dir / "rank1.sdf"
        rows.append(
            {
                "pair_id": complex_name,
                "complex_name": complex_name,
                "diffdock_confidence": "" if confidence is None else f"{confidence:.2f}",
                "docking_score": "",
                "rank1_sdf_path": str(rank1_sdf) if rank1_sdf.exists() else "",
                "confidence_sdf_path": str(confidence_sdf) if confidence_sdf and confidence_sdf.exists() else "",
                "source_chunk": source_chunk,
                "status": "completed" if confidence is not None else "missing_output",
                "error": "" if confidence is not None else "rank1_confidence_sdf_missing",
            }
        )
    return rows


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    job_index = resolve(root, args.job_index)
    rows: list[dict[str, Any]] = []
    jobs = read_csv(job_index)
    parsed_jobs = 0
    score_csv_jobs = 0

    for job in jobs:
        job_id = int(job["job_id"])
        if args.start_job_id is not None and job_id < args.start_job_id:
            continue
        if args.end_job_id is not None and job_id > args.end_job_id:
            continue

        score_csv = resolve(root, job["score_csv"])
        if score_csv.exists() and not args.reparse:
            rows.extend(read_csv(score_csv))
            score_csv_jobs += 1
            continue

        if args.include_missing_jobs or resolve(root, job["out_dir"]).exists():
            rows.extend(
                parse_chunk_outputs(
                    chunk_csv=resolve(root, job["chunk_csv"]),
                    out_dir=resolve(root, job["out_dir"]),
                    source_chunk=str(resolve(root, job["chunk_csv"])),
                )
            )
            parsed_jobs += 1

    out_csv = resolve(root, args.out_csv)
    write_csv(out_csv, FIELDNAMES, rows)
    completed = sum(1 for row in rows if row["status"] == "completed")
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_index": str(job_index),
        "out_csv": str(out_csv),
        "jobs_total": len(jobs),
        "jobs_from_score_csv": score_csv_jobs,
        "jobs_parsed_from_output_dirs": parsed_jobs,
        "rows": len(rows),
        "completed": completed,
        "missing_or_failed": len(rows) - completed,
    }
    out_csv.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect full DiffDock rank1 confidence scores.")
    parser.add_argument("--job-index", default="outputs/report_scale/diffdock_full_run/diffdock_full_job_index.csv")
    parser.add_argument("--out-csv", default="outputs/report_scale/diffdock_scores_full_913170.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--start-job-id", type=int, default=None)
    parser.add_argument("--end-job-id", type=int, default=None)
    parser.add_argument("--include-missing-jobs", action="store_true")
    parser.add_argument("--reparse", action="store_true")
    args = parser.parse_args()
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

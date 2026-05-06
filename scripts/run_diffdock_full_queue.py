from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def output_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def is_done(root: Path, job: dict[str, str]) -> bool:
    score_csv = resolve(root, job["score_csv"])
    return output_count(score_csv) >= int(job["row_count"])


def selected_jobs(args: argparse.Namespace, root: Path) -> list[dict[str, str]]:
    rows = read_csv(resolve(root, args.job_index))
    jobs: list[dict[str, str]] = []
    for row in rows:
        job_id = int(row["job_id"])
        if args.start_job_id is not None and job_id < args.start_job_id:
            continue
        if args.end_job_id is not None and job_id > args.end_job_id:
            continue
        if not args.force and is_done(root, row):
            continue
        jobs.append(row)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    return jobs


def launch_command(args: argparse.Namespace, root: Path, job_id: int, device: str) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "run_diffdock_full_job.py"),
        "--job-index",
        str(resolve(root, args.job_index)),
        "--job-id",
        str(job_id),
        "--root",
        str(root),
        "--diffdock-dir",
        str(resolve(root, args.diffdock_dir)),
        "--cuda-device",
        device,
        "--samples-per-complex",
        str(args.samples_per_complex),
        "--inference-steps",
        str(args.inference_steps),
        "--actual-steps",
        str(args.actual_steps),
        "--batch-size",
        str(args.batch_size),
        "--sdf-retention",
        args.sdf_retention,
        "--min-free-gb",
        str(args.min_free_gb),
        "--omp-threads",
        str(args.omp_threads),
        "--loglevel",
        args.loglevel,
    ]
    if args.force:
        command.append("--force")
    return command


def run_queue(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must name at least one CUDA device")

    jobs = selected_jobs(args, root)
    print(
        json.dumps(
            {
                "selected_jobs": len(jobs),
                "devices": devices,
                "dry_run": args.dry_run,
                "sdf_retention": args.sdf_retention,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for row in jobs[:20]:
            print(json.dumps({"job_id": row["job_id"], "row_count": row["row_count"]}, ensure_ascii=False), flush=True)
        return 0

    pending = list(jobs)
    running: dict[subprocess.Popen[bytes], dict[str, Any]] = {}
    failures = 0
    completed = 0

    while pending or running:
        for device in devices:
            if not pending:
                break
            if any(meta["device"] == device for meta in running.values()):
                continue
            job = pending.pop(0)
            job_id = int(job["job_id"])
            command = launch_command(args, root, job_id, device)
            process = subprocess.Popen(command, cwd=root)
            running[process] = {"job_id": job_id, "device": device, "started": time.time()}
            print(json.dumps({"event": "started", "job_id": job_id, "device": device}, ensure_ascii=False), flush=True)

        if not running:
            continue

        time.sleep(args.poll_seconds)
        for process, meta in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            elapsed = round(time.time() - meta["started"], 2)
            completed += 1
            if returncode != 0:
                failures += 1
            print(
                json.dumps(
                    {
                        "event": "finished",
                        "job_id": meta["job_id"],
                        "device": meta["device"],
                        "returncode": returncode,
                        "elapsed_seconds": elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del running[process]

    print(json.dumps({"completed_jobs": completed, "failed_jobs": failures}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pending full DiffDock chunks across CUDA devices.")
    parser.add_argument("--job-index", default="outputs/report_scale/diffdock_full_run/diffdock_full_job_index.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--diffdock-dir", default="third_party/DiffDock")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means all selected jobs.")
    parser.add_argument("--start-job-id", type=int, default=None)
    parser.add_argument("--end-job-id", type=int, default=None)
    parser.add_argument("--samples-per-complex", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--actual-steps", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sdf-retention", choices=["all", "rank1_confidence", "none"], default="rank1_confidence")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())

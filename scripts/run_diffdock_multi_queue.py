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


def lock_path_for_job(root: Path, job: dict[str, Any]) -> Path:
    score_csv = resolve(root, str(job["score_csv"]))
    return score_csv.parent / f"{score_csv.name}.lock"


def is_done(root: Path, job: dict[str, str]) -> bool:
    score_csv = resolve(root, job["score_csv"])
    return output_count(score_csv) >= int(job["row_count"])


def is_locked(root: Path, job: dict[str, Any]) -> bool:
    return lock_path_for_job(root, job).exists()


def gpu_memory_mb() -> dict[str, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return {}
    memory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            memory[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return memory


def parse_skip_jobs(root: Path, values: list[str]) -> set[tuple[str, int]]:
    skipped: set[tuple[str, int]] = set()
    for value in values:
        if "::" not in value:
            raise ValueError("--skip-job must use '<job_index>::<job_id>' format")
        job_index, job_id = value.rsplit("::", 1)
        skipped.add((str(resolve(root, job_index)), int(job_id)))
    return skipped


def load_pending_jobs(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    skipped = parse_skip_jobs(root, args.skip_job or [])
    pending: list[dict[str, Any]] = []
    for queue_order, job_index_value in enumerate(args.job_index):
        job_index = resolve(root, job_index_value)
        direction = job_index.parent.parent.name
        for row in read_csv(job_index):
            job_id = int(row["job_id"])
            if (str(job_index), job_id) in skipped:
                continue
            if not args.force and is_done(root, row):
                continue
            pending.append(
                {
                    "queue_order": queue_order,
                    "direction": direction,
                    "job_index": str(job_index),
                    "job_id": job_id,
                    "row_count": int(row["row_count"]),
                    "score_csv": row["score_csv"],
                }
            )
    pending.sort(key=lambda row: (row["queue_order"], row["job_id"]))
    return pending


def launch_command(args: argparse.Namespace, root: Path, job: dict[str, Any], device: str) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "run_diffdock_full_job.py"),
        "--job-index",
        job["job_index"],
        "--job-id",
        str(job["job_id"]),
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


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("--devices must name at least one CUDA device")
    pending = load_pending_jobs(args, root)
    if args.max_jobs > 0:
        pending = pending[: args.max_jobs]
    print(
        json.dumps(
            {
                "selected_jobs": len(pending),
                "job_indices": args.job_index,
                "devices": devices,
                "external_busy_memory_mb": args.external_busy_memory_mb,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for job in pending[:50]:
            print(json.dumps(job, ensure_ascii=False), flush=True)
        return 0

    running: dict[subprocess.Popen[bytes], dict[str, Any]] = {}
    failures = 0
    completed = 0
    skipped_locked = 0
    while pending or running:
        memory = gpu_memory_mb()
        for device in devices:
            if not pending:
                break
            if any(meta["device"] == device for meta in running.values()):
                continue
            if int(memory.get(device, 0)) > args.external_busy_memory_mb:
                continue
            job = None
            deferred_locked: list[dict[str, Any]] = []
            while pending:
                candidate = pending.pop(0)
                if is_done(root, candidate):
                    continue
                if is_locked(root, candidate):
                    deferred_locked.append(candidate)
                    continue
                job = candidate
                break
            pending.extend(deferred_locked)
            if job is None:
                continue
            command = launch_command(args, root, job, device)
            process = subprocess.Popen(command, cwd=root)
            running[process] = {**job, "device": device, "started": time.time()}
            print(
                json.dumps(
                    {
                        "event": "started",
                        "direction": job["direction"],
                        "job_id": job["job_id"],
                        "device": device,
                        "row_count": job["row_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if not running:
            time.sleep(args.poll_seconds)
            continue

        time.sleep(args.poll_seconds)
        for process, meta in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            elapsed = round(time.time() - meta["started"], 2)
            completed += 1
            if returncode == 76:
                skipped_locked += 1
            elif returncode != 0:
                failures += 1
            print(
                json.dumps(
                    {
                        "event": "finished",
                        "direction": meta["direction"],
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

    print(
        json.dumps(
            {"completed_jobs": completed, "failed_jobs": failures, "skipped_locked_jobs": skipped_locked},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple DiffDock job indices across available CUDA devices.")
    parser.add_argument("--job-index", action="append", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--diffdock-dir", default="third_party/DiffDock")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--skip-job", action="append", default=[])
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means all selected jobs.")
    parser.add_argument("--samples-per-complex", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--actual-steps", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sdf-retention", choices=["all", "rank1_confidence", "none"], default="rank1_confidence")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--external-busy-memory-mb", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

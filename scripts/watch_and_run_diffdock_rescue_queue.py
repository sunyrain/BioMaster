from __future__ import annotations

import argparse
import csv
import json
import os
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
    return path if path.is_absolute() else (root / path).resolve()


def output_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_locks(score_dir: Path) -> list[dict[str, Any]]:
    locks: list[dict[str, Any]] = []
    for path in sorted(score_dir.glob("*.lock")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except Exception:
            pid = 0
            payload = {}
        if pid and process_alive(pid):
            locks.append({"path": str(path), "pid": pid, "jobId": payload.get("job_id")})
    return locks


def queue_status(root: Path, job_index: Path) -> dict[str, Any]:
    jobs = read_csv(job_index)
    total_rows = 0
    scored_rows = 0
    completed_jobs = 0
    for job in jobs:
        row_count = int(job["row_count"])
        total_rows += row_count
        rows = output_count(resolve(root, job["score_csv"]))
        scored_rows += rows
        if rows >= row_count:
            completed_jobs += 1
    return {
        "completedJobs": completed_jobs,
        "totalJobs": len(jobs),
        "scoredRows": scored_rows,
        "totalRows": total_rows,
        "allComplete": completed_jobs >= len(jobs),
    }


def gpu_memory_used() -> dict[str, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError:
        return {}
    memory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                memory[parts[0]] = int(float(parts[1]))
            except ValueError:
                pass
    return memory


def emit(event: str, **payload: Any) -> None:
    payload.update({"event": event, "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    main_job_index = resolve(root, args.main_job_index)
    rescue_job_index = resolve(root, args.rescue_job_index)
    main_score_dir = resolve(root, args.main_score_dir)

    emit(
        "watcher_started",
        mainJobIndex=str(main_job_index),
        rescueJobIndex=str(rescue_job_index),
        devices=args.devices,
        pollSeconds=args.poll_seconds,
        idleGpuMemoryMb=args.idle_gpu_memory_mb,
    )

    last_status = 0.0
    while True:
        status = queue_status(root, main_job_index)
        locks = active_locks(main_score_dir)
        memory = gpu_memory_used()
        gpu_idle = bool(memory) and all(value <= args.idle_gpu_memory_mb for value in memory.values())
        ready = status["allComplete"] and not locks and gpu_idle

        now = time.time()
        if now - last_status >= args.status_every_sec:
            emit("watcher_status", mainQueue=status, activeMainLocks=locks, gpuMemoryUsedMb=memory, ready=ready)
            last_status = now

        if ready:
            break
        time.sleep(args.poll_seconds)

    command = [
        sys.executable,
        str(root / "scripts" / "run_diffdock_dynamic_queue.py"),
        "--job-index",
        str(rescue_job_index),
        "--root",
        str(root),
        "--diffdock-dir",
        str(resolve(root, args.diffdock_dir)),
        "--devices",
        args.devices,
        "--order",
        "asc",
        "--respect-existing-gpu-use",
        "--idle-gpu-memory-mb",
        str(args.idle_gpu_memory_mb),
        "--remove-stale-locks",
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
        "--poll-seconds",
        str(args.rescue_poll_seconds),
        "--status-every-sec",
        str(args.rescue_status_every_sec),
    ]
    emit("rescue_queue_starting", command=command)
    returncode = subprocess.call(command, cwd=root)
    emit("rescue_queue_finished", returncode=returncode)

    for refresh_command in [
        [sys.executable, str(root / "scripts" / "build_sota_compute_closure_summary.py"), "--root", str(root)],
        [sys.executable, str(root / "scripts" / "build_sota_artifact_manifest.py"), "--root", str(root)],
    ]:
        emit("refresh_starting", command=refresh_command)
        refresh_code = subprocess.call(refresh_command, cwd=root)
        emit("refresh_finished", command=refresh_command, returncode=refresh_code)

    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Start a DiffDock rescue queue only after the main full queue is complete and GPUs are idle.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--main-job-index", default="outputs/report_scale/diffdock_full_run/diffdock_full_job_index.csv")
    parser.add_argument("--main-score-dir", default="outputs/report_scale/diffdock_full_run/scores")
    parser.add_argument(
        "--rescue-job-index",
        default="outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent/diffdock_ligand_rescue_job_index.csv",
    )
    parser.add_argument("--diffdock-dir", default="third_party/DiffDock")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--status-every-sec", type=float, default=900.0)
    parser.add_argument("--idle-gpu-memory-mb", type=int, default=500)
    parser.add_argument("--samples-per-complex", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--actual-steps", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sdf-retention", choices=["all", "rank1_confidence", "none"], default="rank1_confidence")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--rescue-poll-seconds", type=float, default=20.0)
    parser.add_argument("--rescue-status-every-sec", type=float, default=120.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

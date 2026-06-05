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


def parse_device_slots(raw_devices: str) -> list[dict[str, str]]:
    slots: list[dict[str, str]] = []
    for raw in raw_devices.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            cuda_device, slot_name = token.split(":", 1)
            cuda_device = cuda_device.strip()
            slot_name = slot_name.strip() or "default"
        else:
            cuda_device = token
            slot_name = "default"
        label = cuda_device if slot_name == "default" else f"{cuda_device}:{slot_name}"
        slots.append({"label": label, "cuda_device": cuda_device, "lock_token": label})
    return slots


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


def active_lock(lock_path: Path, remove_stale: bool) -> bool:
    if not lock_path.exists():
        return False
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
    except Exception:
        pid = 0
    if pid and process_alive(pid):
        return True
    if remove_stale:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return False


def is_done(root: Path, job: dict[str, str]) -> bool:
    return output_count(resolve(root, job["score_csv"])) >= int(job["row_count"])


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


def pick_next_job(
    root: Path,
    jobs: list[dict[str, str]],
    remove_stale_locks: bool,
    reserved_job_ids: set[int],
) -> dict[str, str] | None:
    for job in jobs:
        job_id = int(job["job_id"])
        if job_id in reserved_job_ids:
            continue
        if is_done(root, job):
            continue
        score_csv = resolve(root, job["score_csv"])
        lock_path = score_csv.parent / f"{score_csv.name}.lock"
        if active_lock(lock_path, remove_stale_locks):
            continue
        return job
    return None


def launch_command(args: argparse.Namespace, root: Path, job_id: int, slot: dict[str, str]) -> list[str]:
    return [
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
        slot["cuda_device"],
        "--device-lock-token",
        slot["lock_token"],
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


def status(root: Path, jobs: list[dict[str, str]]) -> dict[str, Any]:
    completed_jobs = 0
    scored_rows = 0
    total_rows = 0
    for job in jobs:
        total_rows += int(job["row_count"])
        score_csv = resolve(root, job["score_csv"])
        rows = output_count(score_csv)
        scored_rows += rows
        if rows >= int(job["row_count"]):
            completed_jobs += 1
    return {"completedJobs": completed_jobs, "scoredRows": scored_rows, "totalRows": total_rows}


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    device_slots = parse_device_slots(args.devices)
    jobs = read_csv(resolve(root, args.job_index))
    if args.order == "desc":
        jobs = list(reversed(jobs))

    running: dict[subprocess.Popen[bytes], dict[str, Any]] = {}
    failures = 0
    last_status = 0.0
    idle_rounds = 0
    print(
        json.dumps(
            {
                "event": "dynamic_queue_started",
                "devices": [slot["label"] for slot in device_slots],
                "physicalCudaDevices": [slot["cuda_device"] for slot in device_slots],
                "jobs": len(jobs),
                "order": args.order,
                "idleGpuMemoryMb": args.idle_gpu_memory_mb,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    while True:
        for process, meta in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            if returncode != 0:
                failures += 1
            print(
                json.dumps(
                    {
                        "event": "finished",
                        "job_id": meta["job_id"],
                        "device": meta["device"],
                        "returncode": returncode,
                        "elapsed_seconds": round(time.time() - meta["started"], 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del running[process]

        memory = gpu_memory_used()
        launched = 0
        reserved_job_ids = {int(meta["job_id"]) for meta in running.values()}
        for slot in device_slots:
            if any(meta["device"] == slot["label"] for meta in running.values()):
                continue
            if args.respect_existing_gpu_use and memory.get(slot["cuda_device"], 0) > args.idle_gpu_memory_mb:
                continue
            job = pick_next_job(root, jobs, args.remove_stale_locks, reserved_job_ids)
            if job is None:
                continue
            job_id = int(job["job_id"])
            reserved_job_ids.add(job_id)
            process = subprocess.Popen(launch_command(args, root, job_id, slot), cwd=root)
            running[process] = {
                "job_id": job_id,
                "device": slot["label"],
                "cuda_device": slot["cuda_device"],
                "started": time.time(),
            }
            launched += 1
            print(
                json.dumps(
                    {
                        "event": "started",
                        "job_id": job_id,
                        "device": slot["label"],
                        "cuda_device": slot["cuda_device"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        now = time.time()
        if args.status_every_sec > 0 and now - last_status >= args.status_every_sec:
            payload = status(root, jobs)
            payload.update(
                {
                    "event": "status",
                    "runningJobs": [meta["job_id"] for meta in running.values()],
                    "gpuMemoryUsedMb": memory,
                    "failures": failures,
                }
            )
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            last_status = now

        current = status(root, jobs)
        if current["completedJobs"] >= len(jobs):
            print(json.dumps({"event": "all_jobs_completed", **current, "failures": failures}, ensure_ascii=False), flush=True)
            return 1 if failures else 0

        if not running and launched == 0:
            idle_rounds += 1
            if args.exit_after_idle_rounds > 0 and idle_rounds >= args.exit_after_idle_rounds:
                print(json.dumps({"event": "idle_exit", **current, "failures": failures}, ensure_ascii=False), flush=True)
                return 1 if failures else 0
        else:
            idle_rounds = 0

        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamically run pending DiffDock jobs while respecting active locks and GPU use.")
    parser.add_argument("--job-index", default="outputs/report_scale/diffdock_top10000_run/diffdock_full_job_index.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--diffdock-dir", default="third_party/DiffDock")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--samples-per-complex", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--actual-steps", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sdf-retention", choices=["all", "rank1_confidence", "none"], default="rank1_confidence")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--status-every-sec", type=float, default=120.0)
    parser.add_argument("--respect-existing-gpu-use", action="store_true")
    parser.add_argument("--idle-gpu-memory-mb", type=int, default=500)
    parser.add_argument("--remove-stale-locks", action="store_true")
    parser.add_argument("--exit-after-idle-rounds", type=int, default=0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_from_timestamp(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_rank1_sdfs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("rank1_confidence*.sdf"))


def parse_elapsed_seconds(log_dir: Path) -> list[float]:
    elapsed: list[float] = []
    if not log_dir.exists():
        return elapsed
    pattern = re.compile(r'"elapsed_seconds":\s*([0-9.]+)')
    for log_path in sorted(log_dir.glob("full_queue*.log")):
        text = log_path.read_text(errors="ignore")
        for match in pattern.finditer(text):
            value = float(match.group(1))
            if 100 <= value <= 5000:
                elapsed.append(value)
    return elapsed


def run_query(command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def gpu_snapshot() -> dict[str, Any]:
    gpu_lines = run_query(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    process_lines = run_query(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in gpu_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "busId": parts[1],
                "memoryUsedMb": float(parts[2]),
                "memoryTotalMb": float(parts[3]),
                "utilizationPct": float(parts[4]),
                "temperatureC": float(parts[5]),
                "powerW": float(parts[6]),
            }
        )
    processes = []
    for line in process_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        processes.append(
            {
                "busId": parts[0],
                "pid": int(parts[1]),
                "processName": parts[2],
                "memoryUsedMb": float(parts[3]),
            }
        )
    busy_bus_ids = {row["busId"] for row in processes}
    return {
        "gpuCount": len(gpus),
        "busyGpuCount": sum(1 for gpu in gpus if gpu["busId"] in busy_bus_ids),
        "gpus": gpus,
        "computeProcesses": processes,
    }


def build_summary(run_dir: Path) -> dict[str, Any]:
    index_path = run_dir / "diffdock_full_job_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing job index: {index_path}")

    rows = read_csv_rows(index_path)
    total_jobs = len(rows)
    total_rows = sum(int(row["row_count"]) for row in rows)
    completed_jobs = 0
    scored_rows = 0
    completed_outputs = 0
    missing_outputs = 0
    active_locks: list[int] = []
    active_details: list[dict[str, Any]] = []
    newest_scores: list[tuple[float, dict[str, Any]]] = []
    recent_completed: list[dict[str, Any]] = []

    for row in rows:
        job_id = int(row["job_id"])
        expected = int(row["row_count"])
        score_path = Path(row["score_csv"])
        out_dir = Path(row["out_dir"])
        lock_path = score_path.parent / f"{score_path.name}.lock"
        score_count = 0
        completed_count = 0
        status_counts: dict[str, int] = {}

        if score_path.exists():
            try:
                score_rows = read_csv_rows(score_path)
                score_count = len(score_rows)
                for score_row in score_rows:
                    status = score_row.get("status", "")
                    status_counts[status] = status_counts.get(status, 0) + 1
                completed_count = status_counts.get("completed", 0)
            except (csv.Error, UnicodeDecodeError):
                status_counts["read_error"] = 1

            scored_rows += score_count
            completed_outputs += completed_count
            missing_outputs += max(score_count - completed_count, 0)
            if score_count >= expected:
                completed_jobs += 1
                recent_completed.append(
                    {
                        "jobId": job_id,
                        "rows": score_count,
                        "expectedRows": expected,
                        "completedOutputs": completed_count,
                        "missingOutputs": max(score_count - completed_count, 0),
                        "scoreMtimeUtc": utc_from_timestamp(score_path.stat().st_mtime),
                    }
                )
            newest_scores.append(
                (
                    score_path.stat().st_mtime,
                    {
                        "jobId": job_id,
                        "rows": score_count,
                        "expectedRows": expected,
                        "completedOutputs": completed_count,
                        "missingOutputs": max(score_count - completed_count, 0),
                        "scoreMtimeUtc": utc_from_timestamp(score_path.stat().st_mtime),
                    },
                )
            )

        if lock_path.exists():
            active_locks.append(job_id)
            active_details.append(
                {
                    "jobId": job_id,
                    "expectedRows": expected,
                    "scoreRows": score_count,
                    "completedOutputs": completed_count,
                    "inFlightRank1SdfCount": count_rank1_sdfs(out_dir),
                    "scoreExists": score_path.exists(),
                    "lockPath": str(lock_path),
                }
            )

    elapsed = parse_elapsed_seconds(run_dir / "screen_logs")
    recent_elapsed = elapsed[-32:]
    median_recent = statistics.median(recent_elapsed) if recent_elapsed else None
    median_all = statistics.median(elapsed) if elapsed else None
    active_gpu_count = len(active_locks) or 4
    remaining_jobs = total_jobs - completed_jobs
    eta_hours = None
    estimated_finish_utc = None
    if median_recent:
        eta_hours = remaining_jobs * median_recent / max(active_gpu_count, 1) / 3600
        estimated_finish_utc = utc_from_timestamp(time.time() + eta_hours * 3600)

    gpu = gpu_snapshot()
    newest_scores_sorted = [item for _, item in sorted(newest_scores, reverse=True)[:10]]
    recent_completed_sorted = sorted(
        recent_completed, key=lambda item: item["scoreMtimeUtc"], reverse=True
    )[:10]

    return {
        "createdUtc": utc_now(),
        "runDir": str(run_dir),
        "completedJobs": completed_jobs,
        "totalJobs": total_jobs,
        "remainingJobs": remaining_jobs,
        "completedJobPct": round(completed_jobs / total_jobs * 100, 4) if total_jobs else None,
        "scoredRows": scored_rows,
        "totalRows": total_rows,
        "remainingRowsByCompletedScores": total_rows - scored_rows,
        "scoredRowPct": round(scored_rows / total_rows * 100, 4) if total_rows else None,
        "completedOutputs": completed_outputs,
        "missingOutputsInScoredJobs": missing_outputs,
        "completedOutputPctInScoredRows": round(completed_outputs / scored_rows * 100, 4)
        if scored_rows
        else None,
        "activeLocks": active_locks,
        "activeDetails": active_details,
        "elapsedSamples": len(elapsed),
        "medianChunkElapsedSecRecent32": median_recent,
        "medianChunkElapsedSecAll": median_all,
        "etaHours": round(eta_hours, 4) if eta_hours is not None else None,
        "etaDays": round(eta_hours / 24, 4) if eta_hours is not None else None,
        "estimatedFinishUtc": estimated_finish_utc,
        "gpu": gpu,
        "newestScoreFiles": newest_scores_sorted,
        "recentCompletedJobs": recent_completed_sorted,
        "interpretation": {
            "missingOutputsInScoredJobs": "DiffDock rows with no completed rank1 pose in already finalized chunks; these are not biological negatives.",
            "eta": "ETA is based on recent screen-log chunk elapsed_seconds medians and current active lock count.",
            "gpuExpansion": "Do not start extra GPU queues unless busyGpuCount drops below GPU count and active locks/processes confirm a real idle device.",
        },
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    gpu = summary["gpu"]
    lines = [
        "# DiffDock Full Run Progress",
        "",
        f"Generated: {summary['createdUtc']}",
        "",
        "## Queue Status",
        "",
        f"- Jobs completed: {summary['completedJobs']} / {summary['totalJobs']} ({summary['completedJobPct']}%).",
        f"- Rows scored: {summary['scoredRows']} / {summary['totalRows']} ({summary['scoredRowPct']}%).",
        f"- Completed outputs in finalized score files: {summary['completedOutputs']}.",
        f"- Missing outputs in finalized score files: {summary['missingOutputsInScoredJobs']} "
        f"({100 - summary['completedOutputPctInScoredRows']:.2f}% of scored rows).",
        f"- Active locked jobs: {summary['activeLocks']}.",
        "",
        "## ETA",
        "",
        f"- Recent median chunk runtime: {summary['medianChunkElapsedSecRecent32']} seconds.",
        f"- Estimated remaining time: {summary['etaHours']} hours ({summary['etaDays']} days).",
        f"- Estimated finish UTC: {summary['estimatedFinishUtc']}.",
        "",
        "## GPU Status",
        "",
        f"- Busy GPUs: {gpu['busyGpuCount']} / {gpu['gpuCount']}.",
    ]
    for item in gpu["gpus"]:
        lines.append(
            f"- GPU {item['index']}: {item['memoryUsedMb']:.0f}/{item['memoryTotalMb']:.0f} MB, "
            f"{item['utilizationPct']:.0f}% util, {item['powerW']:.1f} W."
        )
    lines.extend(
        [
            "",
            "## Active Jobs",
            "",
        ]
    )
    for item in summary["activeDetails"]:
        lines.append(
            f"- Job {item['jobId']}: expected {item['expectedRows']} rows, "
            f"in-flight rank1 SDFs {item['inFlightRank1SdfCount']}, score exists {item['scoreExists']}."
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Missing outputs are retained as technical missing outputs, not treated as negative biology.",
            "- The current run already occupies all visible GPUs; expansion should wait for a verified idle GPU.",
            "- Keep TMPDIR on `/root/autodl-tmp/tmp` because the root filesystem is full.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/report_scale/diffdock_full_run"),
        help="DiffDock full run directory containing diffdock_full_job_index.csv.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/report_scale/diffdock_full_run/DIFFDOCK_FULL_PROGRESS_SUMMARY.md"),
    )
    args = parser.parse_args()

    summary = build_summary(args.run_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, args.out_md)
    print(json.dumps({"json": str(args.out_json), "markdown": str(args.out_md), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

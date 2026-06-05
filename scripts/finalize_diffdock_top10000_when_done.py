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
    return path if path.is_absolute() else (root / path).resolve()


def output_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def status(root: Path, job_index: Path) -> dict[str, Any]:
    jobs = read_csv(job_index)
    completed_jobs = 0
    scored_rows = 0
    total_rows = 0
    for job in jobs:
        expected = int(job["row_count"])
        total_rows += expected
        rows = output_count(resolve(root, job["score_csv"]))
        scored_rows += rows
        if rows >= expected:
            completed_jobs += 1
    return {
        "completedJobs": completed_jobs,
        "totalJobs": len(jobs),
        "scoredRows": scored_rows,
        "totalRows": total_rows,
        "done": completed_jobs >= len(jobs),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(root: Path, command: list[str], log_path: Path, required: bool) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n$ " + " ".join(command) + "\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result = {
        "command": command,
        "returncode": process.returncode,
        "required": required,
        "elapsedSeconds": round(time.time() - started, 2),
        "log": str(log_path),
    }
    if required and process.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    job_index = resolve(root, args.job_index)
    status_path = resolve(root, args.status_json)
    log_path = resolve(root, args.log_file)
    start = time.time()
    last_status = 0.0

    while True:
        current = status(root, job_index)
        current.update({"phase": "waiting_for_diffdock", "updatedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if time.time() - last_status >= args.status_every_sec:
            print(json.dumps(current, ensure_ascii=False), flush=True)
            write_json(status_path, current)
            last_status = time.time()
        if current["done"]:
            break
        if args.timeout_sec > 0 and time.time() - start > args.timeout_sec:
            current["phase"] = "timeout"
            write_json(status_path, current)
            return current
        time.sleep(args.poll_seconds)

    commands: list[tuple[list[str], bool]] = [
        (
            [
                sys.executable,
                "scripts/collect_diffdock_full_scores.py",
                "--root",
                str(root),
                "--job-index",
                str(job_index),
                "--out-csv",
                args.out_csv,
                "--include-missing-jobs",
            ],
            True,
        )
    ]

    top_manifest = resolve(root, args.top_manifest)
    if top_manifest.exists():
        commands.append(
            (
                [
                    sys.executable,
                    "scripts/merge_stage6_top_diffdock.py",
                    "--root",
                    str(root),
                    "--top-manifest",
                    str(top_manifest),
                    "--score-dir",
                    str(resolve(root, args.score_dir)),
                    "--out-csv",
                    args.stage6_out_csv,
                ],
                True,
            )
        )

    for script in [
        "scripts/audit_sota_external_dependencies.py",
        "scripts/build_sota_model_feasibility_audit.py",
        "scripts/build_sota_artifact_manifest.py",
        "scripts/build_sota_compute_closure_summary.py",
    ]:
        if (root / script).exists():
            commands.append(([sys.executable, script, "--root", str(root)], False))

    results: list[dict[str, Any]] = []
    final_payload: dict[str, Any] = {
        **status(root, job_index),
        "phase": "finalizing",
        "updatedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    write_json(status_path, final_payload)

    for command, required in commands:
        try:
            result = run_command(root, command, log_path, required)
        except RuntimeError as exc:
            final_payload.update(
                {
                    "phase": "failed",
                    "error": str(exc),
                    "updatedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            write_json(status_path, final_payload)
            raise
        results.append(result)
        final_payload.update({"results": results, "updatedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_json(status_path, final_payload)

    final_payload.update({"phase": "completed", "updatedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(status_path, final_payload)
    print(json.dumps(final_payload, ensure_ascii=False), flush=True)
    return final_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for Top10000 DiffDock scores, then collect and rebuild summaries.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--job-index", default="outputs/report_scale/diffdock_top10000_run/diffdock_full_job_index.csv")
    parser.add_argument("--score-dir", default="outputs/report_scale/diffdock_top10000_run/scores")
    parser.add_argument("--out-csv", default="outputs/report_scale/diffdock_top10000_run/diffdock_scores_top10000.csv")
    parser.add_argument("--top-manifest", default="outputs/report_scale/stage5_top10000_diffdock_ready_manifest.csv")
    parser.add_argument("--stage6-out-csv", default="outputs/report_scale/stage6_top10000_consensus_candidates.csv")
    parser.add_argument("--status-json", default="outputs/report_scale/diffdock_top10000_run/finalize_status.json")
    parser.add_argument("--log-file", default="outputs/report_scale/diffdock_top10000_run/logs/finalize_top10000.log")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--status-every-sec", type=float, default=300.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    args = parser.parse_args()
    finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

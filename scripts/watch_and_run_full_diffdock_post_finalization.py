from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit status should preserve parse failures.
        return {"_parse_error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def command_result(root: Path, command: list[str], log_path: Path, required: bool = True) -> dict[str, Any]:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n$ " + " ".join(command) + "\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    result = {
        "command": command,
        "returncode": process.returncode,
        "required": required,
        "elapsedSeconds": round(time.time() - started, 3),
        "log": str(log_path),
    }
    if required and process.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def build_status(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    finalizer_status = read_json(resolve(root, args.finalizer_status_json))
    merged_scores = resolve(root, args.merged_scores)
    post_summary = resolve(root, args.post_summary_json)
    ready = (
        finalizer_status.get("phase") == "completed"
        and merged_scores.exists()
        and file_row_count(merged_scores) >= args.min_merged_rows
    )
    already_completed = post_summary.exists() and read_json(post_summary).get("phase") == "completed"
    return {
        "createdUtc": utc_now(),
        "phase": "completed" if already_completed else "waiting_for_finalizer",
        "readyToRun": bool(ready and not already_completed),
        "alreadyCompleted": bool(already_completed),
        "finalizerPhase": finalizer_status.get("phase", ""),
        "finalizerCompletedUtc": finalizer_status.get("completedUtc", ""),
        "mergedScores": str(merged_scores),
        "mergedScoresExists": merged_scores.exists(),
        "mergedScoreRows": file_row_count(merged_scores),
        "minMergedRows": args.min_merged_rows,
        "postSummary": str(post_summary),
    }


def run_postprocess(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    status_path = resolve(root, args.status_json)
    log_path = resolve(root, args.log_file)
    results: list[dict[str, Any]] = []
    status = build_status(root, args)
    status["phase"] = "running"
    status["startedUtc"] = utc_now()
    status["results"] = results
    write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)

    out_dir = args.out_dir
    audit_input = f"{out_dir}/full_diffdock_structure_audit_input_top{args.top_n}.csv"
    pose_out = f"outputs/sota_validation/pose_sanity_full_diffdock_top{args.top_n}"
    structure_out = f"outputs/sota_validation/structure_confidence_full_diffdock_top{args.top_n}"
    structure_final_out = f"outputs/sota_validation/final_prioritization_full_diffdock_top{args.top_n}"

    commands = [
        [
            sys.executable,
            "scripts/build_full_diffdock_merged_candidate_table.py",
            "--root",
            str(root),
            "--scores",
            args.merged_scores,
            "--out-dir",
            out_dir,
            "--top-n",
            str(args.top_n),
            "--large-top-n",
            str(args.large_top_n),
        ],
        [
            sys.executable,
            "scripts/build_pose_sanity_audit.py",
            "--root",
            str(root),
            "--candidates",
            audit_input,
            "--out-dir",
            pose_out,
            "--top-n-per-direction",
            str(args.top_n),
        ],
        [
            sys.executable,
            "scripts/build_structure_confidence_audit.py",
            "--root",
            str(root),
            "--pose-audit",
            f"{pose_out}/candidate_pose_sanity_audit.csv",
            "--final-table",
            audit_input,
            "--out-dir",
            structure_out,
            "--final-out-dir",
            structure_final_out,
        ],
        [sys.executable, "scripts/build_wetlab_validation_package.py", "--root", str(root)],
        [sys.executable, "scripts/build_final_pre_experiment_gate.py", "--root", str(root)],
        [sys.executable, "scripts/build_sota_compute_closure_summary.py", "--root", str(root)],
        [sys.executable, "scripts/build_sota_artifact_manifest.py", "--root", str(root)],
        [sys.executable, "scripts/build_compute_status_dashboard_asset.py", "--root", str(root)],
    ]

    try:
        for command in commands:
            result = command_result(root, command, log_path, required=True)
            results.append({"label": Path(command[1]).stem if len(command) > 1 else command[0], "result": result})
            status["results"] = results
            status["updatedUtc"] = utc_now()
            write_json(status_path, status)
    except Exception as exc:  # noqa: BLE001 - preserve failure for restartable watcher.
        status["phase"] = "failed"
        status["failedUtc"] = utc_now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["results"] = results
        write_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        raise

    summary = read_json(resolve(root, f"{out_dir}/full_diffdock_merged_stage5_summary.json"))
    status.update(
        {
            "phase": "completed",
            "completedUtc": utc_now(),
            "readyToRun": False,
            "postProcessSummary": summary,
            "outputs": {
                "fullMergedSummary": f"{out_dir}/full_diffdock_merged_stage5_summary.json",
                "fullMergedTop": f"{out_dir}/full_diffdock_merged_stage5_top{args.top_n}.csv",
                "fullMergedLargeTop": f"{out_dir}/full_diffdock_merged_stage5_top{args.large_top_n}.csv",
                "poseSanity": f"{pose_out}/candidate_pose_sanity_audit.csv",
                "structureConfidence": f"{structure_out}/candidate_structure_confidence_audit.csv",
                "structureAdjusted": f"{structure_final_out}/final_priority_structure_adjusted_table.csv",
            },
            "results": results,
        }
    )
    write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    status_path = resolve(root, args.status_json)
    last_status = 0.0
    started = time.time()
    while True:
        status = build_status(root, args)
        now = time.time()
        if args.once or now - last_status >= args.status_every_sec:
            write_json(status_path, status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            last_status = now
        if status.get("alreadyCompleted"):
            return 0
        if status.get("readyToRun"):
            run_postprocess(root, args)
            return 0
        if args.once:
            return 0
        if args.timeout_sec > 0 and now - started > args.timeout_sec:
            status["phase"] = "timeout"
            status["timeoutSec"] = args.timeout_sec
            write_json(status_path, status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return 124
        time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPU post-processing after full DiffDock final scores are merged.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--finalizer-status-json",
        default="outputs/report_scale/diffdock_full_final_scores/finalize_after_rescues_status.json",
    )
    parser.add_argument("--merged-scores", default="outputs/report_scale/diffdock_scores_full_913170_with_rescues.csv")
    parser.add_argument("--out-dir", default="outputs/report_scale/full_diffdock_final")
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--large-top-n", type=int, default=100000)
    parser.add_argument("--min-merged-rows", type=int, default=900000)
    parser.add_argument(
        "--status-json",
        default="outputs/report_scale/full_diffdock_final/post_finalization_status.json",
    )
    parser.add_argument(
        "--post-summary-json",
        default="outputs/report_scale/full_diffdock_final/post_finalization_status.json",
    )
    parser.add_argument(
        "--log-file",
        default="outputs/report_scale/full_diffdock_final/logs/post_finalization.log",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--status-every-sec", type=float, default=900.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

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


SCORE_FIELDNAMES = [
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

MERGED_FIELDNAMES = SCORE_FIELDNAMES + [
    "final_source",
    "rescue_attempted",
    "rescue_source",
    "rescue_status",
    "rescue_error",
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve(root: Path, value: str | Path) -> Path:
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


def active_locks(score_dirs: list[Path]) -> list[dict[str, Any]]:
    locks: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for score_dir in score_dirs:
        if score_dir in seen or not score_dir.exists():
            continue
        seen.add(score_dir)
        for path in sorted(score_dir.glob("*.lock")):
            payload: dict[str, Any] = {}
            pid = 0
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", 0))
            except Exception:
                pass
            if pid and process_alive(pid):
                locks.append({"path": str(path), "pid": pid, "jobId": payload.get("job_id")})
    return locks


def score_dirs_from_job_index(root: Path, job_index: Path) -> list[Path]:
    if not job_index.exists():
        return []
    dirs: list[Path] = []
    for job in read_csv(job_index):
        dirs.append(resolve(root, job["score_csv"]).parent)
    return dirs


def queue_status(root: Path, job_index: Path, allow_missing: bool = False, allow_empty: bool = False) -> dict[str, Any]:
    if not job_index.exists():
        return {
            "jobIndex": str(job_index),
            "exists": False,
            "completedJobs": 0,
            "totalJobs": 0,
            "scoredRows": 0,
            "totalRows": 0,
            "allComplete": bool(allow_missing),
        }
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
        "jobIndex": str(job_index),
        "exists": True,
        "completedJobs": completed_jobs,
        "totalJobs": len(jobs),
        "scoredRows": scored_rows,
        "totalRows": total_rows,
        "allComplete": (not jobs and allow_empty) or (bool(jobs) and completed_jobs >= len(jobs)),
    }


def command_result(root: Path, command: list[str], log_path: Path, required: bool) -> dict[str, Any]:
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


def collect_queue_scores(
    root: Path,
    job_index: Path,
    out_csv: Path,
    log_path: Path,
    include_missing_jobs: bool = True,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/collect_diffdock_full_scores.py",
        "--root",
        str(root),
        "--job-index",
        str(job_index),
        "--out-csv",
        str(out_csv),
    ]
    if include_missing_jobs:
        command.append("--include-missing-jobs")
    result = command_result(root, command, log_path, required=True)
    metadata_path = out_csv.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {"result": result, "metadata": metadata}


def better_rescue_row(row: dict[str, str]) -> bool:
    return row.get("status") == "completed" and bool(row.get("diffdock_confidence"))


def merge_scores(
    main_csv: Path,
    rescue_csvs: list[tuple[str, Path]],
    out_csv: Path,
    out_score_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    main_rows = read_csv(main_csv)
    by_pair: dict[str, dict[str, Any]] = {}
    for row in main_rows:
        pair_id = row["pair_id"]
        by_pair[pair_id] = {
            **{field: row.get(field, "") for field in SCORE_FIELDNAMES},
            "final_source": "main",
            "rescue_attempted": "0",
            "rescue_source": "",
            "rescue_status": "",
            "rescue_error": "",
        }

    rescue_rows_total = 0
    rescue_completed = 0
    rescue_overrides = 0
    rescue_attempted_pairs: set[str] = set()
    rescue_sources: dict[str, dict[str, int]] = {}
    for rescue_name, rescue_csv in rescue_csvs:
        if not rescue_csv.exists():
            continue
        source_counts = {"rows": 0, "completed": 0, "overrides": 0}
        for row in read_csv(rescue_csv):
            pair_id = row["pair_id"]
            rescue_rows_total += 1
            source_counts["rows"] += 1
            rescue_attempted_pairs.add(pair_id)
            if better_rescue_row(row):
                rescue_completed += 1
                source_counts["completed"] += 1
            existing = by_pair.get(pair_id)
            if existing is None:
                by_pair[pair_id] = {
                    **{field: row.get(field, "") for field in SCORE_FIELDNAMES},
                    "final_source": rescue_name if better_rescue_row(row) else "rescue_only_missing",
                    "rescue_attempted": "1",
                    "rescue_source": rescue_name,
                    "rescue_status": row.get("status", ""),
                    "rescue_error": row.get("error", ""),
                }
                if better_rescue_row(row):
                    rescue_overrides += 1
                    source_counts["overrides"] += 1
                continue

            existing["rescue_attempted"] = "1"
            existing["rescue_source"] = rescue_name
            existing["rescue_status"] = row.get("status", "")
            existing["rescue_error"] = row.get("error", "")
            if existing.get("status") != "completed" and better_rescue_row(row):
                for field in SCORE_FIELDNAMES:
                    existing[field] = row.get(field, "")
                existing["final_source"] = rescue_name
                rescue_overrides += 1
                source_counts["overrides"] += 1
        rescue_sources[rescue_name] = source_counts

    merged_rows = [by_pair[pair_id] for pair_id in sorted(by_pair)]
    completed = sum(1 for row in merged_rows if row.get("status") == "completed")
    missing = len(merged_rows) - completed
    write_csv(out_csv, MERGED_FIELDNAMES, merged_rows)
    write_csv(out_score_csv, SCORE_FIELDNAMES, [{field: row.get(field, "") for field in SCORE_FIELDNAMES} for row in merged_rows])

    summary = {
        "createdUtc": utc_now(),
        "mainCsv": str(main_csv),
        "rescueCsvs": [{"name": name, "path": str(path), "exists": path.exists()} for name, path in rescue_csvs],
        "outCsv": str(out_csv),
        "outScoreCsv": str(out_score_csv),
        "mainRows": len(main_rows),
        "mergedRows": len(merged_rows),
        "completedRows": completed,
        "missingRows": missing,
        "completedPct": round(completed / len(merged_rows) * 100.0, 4) if merged_rows else 0.0,
        "rescueRowsTotal": rescue_rows_total,
        "rescueCompletedRows": rescue_completed,
        "rescueAttemptedPairs": len(rescue_attempted_pairs),
        "rescueOverrides": rescue_overrides,
        "rescueSources": rescue_sources,
        "statusCounts": count_field(merged_rows, "status"),
        "finalSourceCounts": count_field(merged_rows, "final_source"),
    }
    write_json(summary_json, summary)
    out_csv.with_suffix(".metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out_score_csv.with_suffix(".metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Full DiffDock Final Merged Scores",
        "",
        f"- Generated: {summary['createdUtc']}",
        f"- Main rows: {summary['mainRows']}",
        f"- Merged rows: {summary['mergedRows']}",
        f"- Completed rows: {summary['completedRows']} ({summary['completedPct']}%)",
        f"- Missing rows: {summary['missingRows']}",
        f"- Rescue attempted pairs: {summary['rescueAttemptedPairs']}",
        f"- Rescue overrides: {summary['rescueOverrides']}",
        f"- Output CSV: `{out_csv}`",
        f"- Score-compatible CSV: `{out_score_csv}`",
        "",
        "Rows still marked `missing_output` are technical DiffDock output misses, not biological negatives.",
        "",
    ]
    summary_json.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return summary


def count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "") or "NA")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def emit_status(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_status(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    main_job_index = resolve(root, args.main_job_index)
    single_job_index = resolve(root, args.single_rescue_job_index)
    multi_job_index = resolve(root, args.multi_rescue_job_index)
    statuses = {
        "main": queue_status(root, main_job_index),
        "singleRescue": queue_status(
            root,
            single_job_index,
            allow_missing=args.allow_missing_single_rescue,
            allow_empty=True,
        ),
        "multiRescue": queue_status(
            root,
            multi_job_index,
            allow_missing=args.allow_missing_multi_rescue,
            allow_empty=True,
        ),
    }
    score_dirs = []
    for job_index in [main_job_index, single_job_index, multi_job_index]:
        score_dirs.extend(score_dirs_from_job_index(root, job_index))
    locks = active_locks(score_dirs)
    ready = all(item["allComplete"] for item in statuses.values()) and not locks
    return {
        "createdUtc": utc_now(),
        "phase": "waiting_for_queues",
        "readyToFinalize": ready,
        "queues": statuses,
        "activeLocks": locks,
    }


def finalize(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    status_path = resolve(root, args.status_json)
    log_path = resolve(root, args.log_file)

    main_job_index = resolve(root, args.main_job_index)
    single_job_index = resolve(root, args.single_rescue_job_index)
    multi_job_index = resolve(root, args.multi_rescue_job_index)

    main_scores = resolve(root, args.main_scores_out)
    single_scores = resolve(root, args.single_rescue_scores_out)
    multi_scores = resolve(root, args.multi_rescue_scores_out)
    merged_scores = resolve(root, args.merged_scores_out)
    score_compatible = resolve(root, args.score_compatible_out)
    merge_summary = resolve(root, args.merge_summary_json)

    results: list[dict[str, Any]] = []
    status = build_status(root, args)
    status["phase"] = "finalizing"
    status["results"] = results
    emit_status(status_path, status)

    for label, job_index, out_csv in [
        ("main", main_job_index, main_scores),
        ("singleRescue", single_job_index, single_scores),
        ("multiRescue", multi_job_index, multi_scores),
    ]:
        if not job_index.exists():
            results.append({"label": label, "skipped": True, "reason": "job_index_missing", "jobIndex": str(job_index)})
            continue
        collected = collect_queue_scores(root, job_index, out_csv, log_path)
        results.append({"label": label, "skipped": False, **collected})
        status["results"] = results
        status["updatedUtc"] = utc_now()
        emit_status(status_path, status)

    merged = merge_scores(
        main_csv=main_scores,
        rescue_csvs=[("single_rescue", single_scores), ("multi_ligand_rescue", multi_scores)],
        out_csv=merged_scores,
        out_score_csv=score_compatible,
        summary_json=merge_summary,
    )
    results.append({"label": "mergeScores", "summary": merged})
    status["results"] = results
    status["updatedUtc"] = utc_now()
    emit_status(status_path, status)

    refresh_commands = [
        [sys.executable, "scripts/summarize_diffdock_full_progress.py", "--run-dir", args.main_run_dir],
        [sys.executable, "scripts/build_diffdock_full_ligand_failure_audit.py", "--root", str(root)],
        [sys.executable, "scripts/audit_sota_external_dependencies.py", "--root", str(root)],
        [sys.executable, "scripts/build_sota_model_feasibility_audit.py", "--root", str(root)],
        [sys.executable, "scripts/build_sota_compute_closure_summary.py", "--root", str(root)],
        [sys.executable, "scripts/build_sota_artifact_manifest.py", "--root", str(root)],
        [sys.executable, "scripts/build_compute_status_dashboard_asset.py", "--root", str(root)],
    ]
    for command in refresh_commands:
        if not (root / command[1]).exists():
            results.append({"label": "refresh", "command": command, "skipped": True, "reason": "script_missing"})
            continue
        result = command_result(root, command, log_path, required=False)
        results.append({"label": "refresh", "result": result})
        status["results"] = results
        status["updatedUtc"] = utc_now()
        emit_status(status_path, status)

    status.update(
        {
            "phase": "completed",
            "completedUtc": utc_now(),
            "readyToFinalize": False,
            "mergeSummary": merged,
            "results": results,
        }
    )
    emit_status(status_path, status)
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
            emit_status(status_path, status)
            last_status = now
        if args.once:
            return 0
        if status["readyToFinalize"]:
            break
        if args.timeout_sec > 0 and now - started > args.timeout_sec:
            status["phase"] = "timeout"
            status["timeoutSec"] = args.timeout_sec
            emit_status(status_path, status)
            return 124
        time.sleep(args.poll_seconds)

    finalize(root, args)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize full DiffDock scores after main and rescue queues complete.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--main-run-dir", default="outputs/report_scale/diffdock_full_run")
    parser.add_argument("--main-job-index", default="outputs/report_scale/diffdock_full_run/diffdock_full_job_index.csv")
    parser.add_argument(
        "--single-rescue-job-index",
        default="outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent/diffdock_ligand_rescue_job_index.csv",
    )
    parser.add_argument(
        "--multi-rescue-job-index",
        default="outputs/report_scale/diffdock_full_multi_ligand_rescue/aggregate/diffdock_multi_ligand_rescue_job_index.csv",
    )
    parser.add_argument("--main-scores-out", default="outputs/report_scale/diffdock_scores_full_913170.csv")
    parser.add_argument(
        "--single-rescue-scores-out",
        default="outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent/diffdock_ligand_rescue_scores.csv",
    )
    parser.add_argument(
        "--multi-rescue-scores-out",
        default="outputs/report_scale/diffdock_full_multi_ligand_rescue/aggregate/diffdock_multi_ligand_rescue_scores.csv",
    )
    parser.add_argument(
        "--merged-scores-out",
        default="outputs/report_scale/diffdock_scores_full_913170_with_rescues.csv",
    )
    parser.add_argument(
        "--score-compatible-out",
        default="outputs/report_scale/diffdock_full_final_scores/scores/diffdock_full_final_merged.scores.csv",
    )
    parser.add_argument(
        "--merge-summary-json",
        default="outputs/report_scale/diffdock_full_final_scores/diffdock_full_final_merged_summary.json",
    )
    parser.add_argument(
        "--status-json",
        default="outputs/report_scale/diffdock_full_final_scores/finalize_after_rescues_status.json",
    )
    parser.add_argument(
        "--log-file",
        default="outputs/report_scale/diffdock_full_final_scores/logs/finalize_after_rescues.log",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--status-every-sec", type=float, default=900.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--once", action="store_true", help="Write one status snapshot and exit without waiting/finalizing.")
    parser.add_argument("--allow-missing-single-rescue", action="store_true")
    parser.add_argument("--allow-missing-multi-rescue", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

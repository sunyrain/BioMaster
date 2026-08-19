from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_OUT = "outputs/report_scale/biomaster_status_audit_latest.json"
REPORT_DIR = Path("outputs/report_scale")
FULL_DIFFDOCK_DIR = REPORT_DIR / "diffdock_full_run"
FULL_DIFFDOCK_SCORES_DIR = FULL_DIFFDOCK_DIR / "scores"
FULL_DIFFDOCK_JOB_INDEX = FULL_DIFFDOCK_DIR / "diffdock_full_job_index.csv"


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def run_command(args: list[str], root: Path, timeout: int = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_git(root: Path) -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "--short", "HEAD"], root)
    branch = run_command(["git", "branch", "--show-current"], root)
    status = run_command(["git", "status", "--short"], root)
    return {
        "commit": commit["stdout"] if commit["ok"] else None,
        "branch": branch["stdout"] if branch["ok"] else None,
        "status_short": status["stdout"] if status["ok"] else None,
        "is_clean": status["ok"] and status["stdout"] == "",
    }


def audit_disk(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    total_gb = usage.total / 1024**3
    used_gb = usage.used / 1024**3
    free_gb = usage.free / 1024**3
    return {
        "path": str(root),
        "total_gb": round(total_gb, 2),
        "used_gb": round(used_gb, 2),
        "free_gb": round(free_gb, 2),
        "used_pct": round(used_gb / total_gb * 100, 2) if total_gb else None,
    }


def parse_processes(ps_text: str) -> dict[str, Any]:
    queue_processes: list[dict[str, str]] = []
    job_processes: list[dict[str, str]] = []
    job_id_re = re.compile(r"--job-id\s+(\d+)")
    device_re = re.compile(r"--cuda-device\s+(\d+)")
    for line in ps_text.splitlines():
        if "run_diffdock_full_queue.py" in line and "grep" not in line:
            parts = line.split(maxsplit=7)
            queue_processes.append({"pid": parts[1] if len(parts) > 1 else "", "command": line})
        if "run_diffdock_full_job.py" in line and "grep" not in line:
            parts = line.split(maxsplit=7)
            job_processes.append(
                {
                    "pid": parts[1] if len(parts) > 1 else "",
                    "job_id": job_id_re.search(line).group(1) if job_id_re.search(line) else "",
                    "cuda_device": device_re.search(line).group(1) if device_re.search(line) else "",
                    "command": line,
                }
            )
    return {
        "queue_running": bool(queue_processes),
        "queue_processes": queue_processes,
        "active_job_count": len(job_processes),
        "active_jobs": job_processes,
    }


def audit_processes(root: Path) -> dict[str, Any]:
    ps = run_command(["ps", "-ef"], root)
    if not ps["ok"]:
        return {"queue_running": False, "active_job_count": 0, "active_jobs": [], "error": ps["stderr"]}
    audit = parse_processes(ps["stdout"])
    pid_file = resolve(root, FULL_DIFFDOCK_DIR / "full_queue.pid")
    audit["pid_file"] = str(pid_file)
    audit["pid_file_value"] = pid_file.read_text(encoding="utf-8").strip() if pid_file.exists() else None
    return audit


def audit_gpus(root: Path) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(cmd, root, timeout=10)
    if not result["ok"]:
        return {"available": False, "error": result["stderr"] or result["error"]}
    gpus: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            }
        )
    return {"available": True, "gpus": gpus}


def audit_stage_outputs(root: Path) -> dict[str, Any]:
    paths = {
        "drug_library": "data/processed/drug_library_pubchem_chembl_mapped.csv",
        "protein_library": "data/processed/protein_library_1000_alphafold_paths.csv",
        "opentargets_target_disease_scores": "data/processed/opentargets_target_disease_scores.csv",
        "string_human_filtered_edges": "data/processed/string_human_filtered_edges.csv",
        "txgnn_drug_disease_scores": "data/processed/txgnn_drug_disease_scores.csv",
        "conplex_affinity_scores": "outputs/report_scale/conplex_affinity_scores_915k.csv",
        "stage5_ranked_candidates": "outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv",
        "stage6_top1000_consensus": "outputs/report_scale/stage6_top1000_consensus_candidates.csv",
    }
    row_counts = {name: count_csv_rows(resolve(root, path)) for name, path in paths.items()}
    return {
        "row_counts": row_counts,
        "opentargets_metadata": load_json(resolve(root, "data/processed/opentargets_target_disease_scores.metadata.json")),
        "stage5_open_targets_string_metadata": load_json(resolve(root, "outputs/report_scale/stage5_open_targets_string_ranked_candidates_915k.metadata.json")),
        "stage5_metadata": load_json(resolve(root, "outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.metadata.json")),
        "stage6_metadata": load_json(resolve(root, "outputs/report_scale/stage6_top1000_consensus_candidates.metadata.json")),
        "stage6_summary": load_json(resolve(root, "outputs/report_scale/stage6_top1000_report_summary.json")),
        "diffdock_ready_metadata": load_json(resolve(root, "outputs/report_scale/manifest_915k_diffdock_ready.metadata.json")),
    }


def audit_full_diffdock(root: Path) -> dict[str, Any]:
    job_index = resolve(root, FULL_DIFFDOCK_JOB_INDEX)
    report_scale_dir = resolve(root, REPORT_DIR)
    jobs = read_csv(job_index)
    jobs_total = len(jobs)
    ready_meta = load_json(resolve(root, "outputs/report_scale/manifest_915k_diffdock_ready.metadata.json")) or {}
    rows_total = ready_meta.get("pairs_diffdock_ready")
    if rows_total is None:
        rows_total = count_csv_rows(resolve(root, "outputs/report_scale/manifest_915k_diffdock_ready.csv"))

    score_files = sorted(resolve(root, FULL_DIFFDOCK_SCORES_DIR).glob("*.scores.csv"))
    rows_scored = 0
    completed = 0
    missing = 0
    completed_per_chunk: list[int] = []
    zero_chunks: list[dict[str, Any]] = []

    for score_file in score_files:
        rows = read_csv(score_file)
        chunk_completed = sum(1 for row in rows if row.get("status") == "completed")
        rows_scored += len(rows)
        completed += chunk_completed
        missing += len(rows) - chunk_completed
        completed_per_chunk.append(chunk_completed)
        if rows and chunk_completed == 0:
            drugs = sorted({row.get("pair_id", "").split("__")[0] for row in rows if row.get("pair_id")})
            zero_chunks.append({"file": score_file.name, "drugs": drugs})

    return {
        "scope_status": (
            "NOT_INITIALIZED_IN_CURRENT_WORKSPACE"
            if not job_index.exists() and not report_scale_dir.exists()
            else ("PREPARED_OR_PARTIAL" if job_index.exists() else "MISSING_JOB_INDEX")
        ),
        "job_index": str(job_index),
        "score_dir": str(resolve(root, FULL_DIFFDOCK_SCORES_DIR)),
        "jobs_total": jobs_total,
        "score_files": len(score_files),
        "job_progress_pct": round(len(score_files) / jobs_total * 100, 2) if jobs_total else None,
        "rows_total_diffdock_ready": rows_total,
        "rows_scored": rows_scored,
        "row_progress_pct": round(rows_scored / rows_total * 100, 2) if rows_total else None,
        "completed_outputs": completed,
        "missing_outputs": missing,
        "success_rate_among_scored_pct": round(completed / rows_scored * 100, 2) if rows_scored else None,
        "zero_completed_chunks": len(zero_chunks),
        "zero_chunk_drugs": sorted({drug for chunk in zero_chunks for drug in chunk["drugs"]}),
        "completed_per_chunk_median": statistics.median(completed_per_chunk) if completed_per_chunk else None,
        "completed_per_chunk_min": min(completed_per_chunk) if completed_per_chunk else None,
        "completed_per_chunk_max": max(completed_per_chunk) if completed_per_chunk else None,
        "zero_chunks": zero_chunks[:30],
    }


def build_warnings(audit: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    full = audit["full_diffdock"]
    disk = audit["disk"]
    processes = audit["processes"]
    stage = audit["stage_outputs"]

    if full["scope_status"] == "NOT_INITIALIZED_IN_CURRENT_WORKSPACE":
        warnings.append(
            "当前工作区没有旧 report_scale/full DiffDock 运行产物；六月 tracker 的后台运行状态已失效，"
            "不能将该分支报告为运行中或已完成。若重新纳入主线，必须先重建 manifest、job index 和冻结执行协议。"
        )
        return warnings

    if full["row_progress_pct"] is not None and full["row_progress_pct"] < 100:
        warnings.append("全量 DiffDock 尚未完成；当前只能声明 Top1000 结构增强完成 940/1000，全量结构扩展仍在后台运行。")
    if full["zero_completed_chunks"]:
        warnings.append("部分 DiffDock chunk 为 0 completed，需后续建立失败 pair 的 SMILES/参数补跑队列。")
    if disk["used_pct"] is not None and disk["used_pct"] >= 85:
        warnings.append("磁盘使用率已达到或超过 85%，继续运行全量 DiffDock 时应监控可用空间。")
    if not processes["queue_running"] and full["jobs_total"] > 0:
        warnings.append("未检测到 full DiffDock queue 进程；如预期应继续运行，需要检查后台任务。")
    stage6_meta = stage.get("stage6_metadata") or {}
    if stage6_meta and stage6_meta.get("diffdock_completed") != 940:
        warnings.append("Stage6 Top1000 DiffDock 完成数不是既定 940/1000，请复核结构增强结果。")
    elif not stage6_meta:
        warnings.append("当前工作区缺少 Stage6 DiffDock metadata；不能声明 Top1000 结构增强完成。")
    return warnings


def audit(root: Path) -> dict[str, Any]:
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "git": audit_git(root),
        "disk": audit_disk(root),
        "processes": audit_processes(root),
        "gpus": audit_gpus(root),
        "stage_outputs": audit_stage_outputs(root),
        "full_diffdock": audit_full_diffdock(root),
    }
    result["warnings"] = build_warnings(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit BioMaster current result files and full DiffDock progress.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-json", default=DEFAULT_OUT)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = audit(root)
    out_json = resolve(root, args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def file_info(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": rel(path, root)}
    stat = path.stat()
    return {
        "exists": True,
        "path": rel(path, root),
        "sizeBytes": stat.st_size,
        "mtimeUtc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


def csv_data_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def job_index_summary(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "totalJobs": 0, "scoreFiles": 0, "scoreRows": 0, "totalRows": 0}
    total_jobs = 0
    score_files = 0
    score_rows = 0
    total_rows = 0
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_jobs += 1
            try:
                total_rows += int(float(row.get("row_count") or 0))
            except ValueError:
                pass
            score_path = path.parent
            raw_score = row.get("score_csv") or ""
            score_file = Path(raw_score)
            if not score_file.is_absolute():
                score_file = root / score_file if raw_score.startswith("outputs/") else score_path / score_file
            if score_file.exists():
                score_files += 1
                rows = csv_data_rows(score_file)
                score_rows += rows or 0
    return {
        "exists": True,
        "totalJobs": total_jobs,
        "scoreFiles": score_files,
        "scoreRows": score_rows,
        "totalRows": total_rows,
        "allScoreFilesExist": score_files == total_jobs and total_jobs > 0,
        "allRowsScored": score_rows >= total_rows and total_rows > 0,
    }


def last_json_line(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def check(
    key: str,
    label: str,
    passed: bool,
    evidence: dict[str, Any],
    required: bool = True,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "required": required,
        "status": "pass" if passed else "fail",
        "passed": bool(passed),
        "reason": reason,
        "evidence": evidence,
    }


def build_audit(root: Path) -> dict[str, Any]:
    progress_path = root / "outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json"
    progress = read_json(progress_path)
    closure = read_json(root / "outputs/sota_validation/sota_compute_closure_summary.json")
    headline = closure.get("headline", {})
    wetlab = read_json(root / "outputs/sota_validation/wetlab_validation_package/wetlab_validation_summary.json")
    finalizer = read_json(root / "outputs/report_scale/diffdock_full_final_scores/finalize_after_rescues_status.json")
    post_final = read_json(root / "outputs/report_scale/full_diffdock_final/post_finalization_status.json")

    main_total_jobs = progress.get("totalJobs") or headline.get("fullDiffdockTotalJobs")
    main_completed_jobs = progress.get("completedJobs") or headline.get("fullDiffdockCompletedJobs")
    main_total_rows = progress.get("totalRows") or headline.get("fullDiffdockTotalRows")
    main_scored_rows = progress.get("scoredRows") or headline.get("fullDiffdockScoredRows")
    main_complete = (
        main_total_jobs is not None
        and main_completed_jobs == main_total_jobs
        and main_total_rows is not None
        and main_scored_rows == main_total_rows
    )

    single_job_index = (
        root
        / "outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent/diffdock_ligand_rescue_job_index.csv"
    )
    multi_job_index = (
        root
        / "outputs/report_scale/diffdock_full_multi_ligand_rescue/aggregate/diffdock_multi_ligand_rescue_job_index.csv"
    )
    single_queue = job_index_summary(single_job_index, root)
    multi_queue = job_index_summary(multi_job_index, root)

    full_raw_scores = root / "outputs/report_scale/diffdock_scores_full_913170.csv"
    full_rescued_scores = root / "outputs/report_scale/diffdock_scores_full_913170_with_rescues.csv"
    final_compatible_scores = (
        root / "outputs/report_scale/diffdock_full_final_scores/scores/diffdock_full_final_merged.scores.csv"
    )
    final_summary = root / "outputs/report_scale/diffdock_full_final_scores/diffdock_full_final_merged_summary.json"

    rescued_rows = csv_data_rows(full_rescued_scores)
    compatible_rows = csv_data_rows(final_compatible_scores)
    min_full_rows = 900000

    required_files = [
        full_raw_scores,
        full_rescued_scores,
        final_compatible_scores,
        final_summary,
        root / "outputs/sota_validation/sota_compute_closure_summary.json",
        root / "outputs/sota_validation/sota_artifact_manifest.json",
        root / "docs/assets/compute-status.js",
        root / "outputs/sota_validation/wetlab_validation_package/wetlab_validation_summary.json",
        root / "outputs/sota_validation/wetlab_validation_package/wetlab_platform_procurement_checklist_12.csv",
    ]

    required_file_infos = [file_info(path, root) for path in required_files]
    all_required_files_exist = all(info["exists"] for info in required_file_infos)

    checks = [
        check(
            "main_diffdock_complete",
            "Full DiffDock main queue is complete",
            bool(main_complete),
            {
                "progressFile": rel(progress_path, root),
                "completedJobs": main_completed_jobs,
                "totalJobs": main_total_jobs,
                "scoredRows": main_scored_rows,
                "totalRows": main_total_rows,
                "activeLocks": progress.get("activeLocks") or [],
            },
            reason="Main queue must reach all jobs and all rows before rescue/finalization can close.",
        ),
        check(
            "single_ligand_rescue_complete_or_documented",
            "CHEMBL3039504 ligand rescue completed or formally documented as stable technical failure",
            bool(single_queue.get("allRowsScored")),
            {
                "jobIndex": rel(single_job_index, root),
                **single_queue,
                "watcher": last_json_line(root / "logs/diffdock_ligand_rescue_after_full.log"),
            },
            reason="Single-ligand rescue is currently required after main queue completion.",
        ),
        check(
            "multi_ligand_rescue_complete",
            "Multi-ligand technical rescue queue completed after latest failure audit",
            bool(multi_queue.get("allRowsScored")),
            {
                "jobIndex": rel(multi_job_index, root),
                **multi_queue,
                "watcher": last_json_line(root / "logs/diffdock_multi_ligand_rescue_after_single.log"),
            },
            reason="Aggregate rescue queue must complete or be rebuilt and completed from the latest failure audit.",
        ),
        check(
            "final_merge_outputs_exist",
            "Final full DiffDock merged score files exist",
            all_required_files_exist,
            {"files": required_file_infos},
            reason="Final reporting requires raw full scores, rescued scores, compatible scores, summaries, and refreshed web/status assets.",
        ),
        check(
            "final_merged_rows_sufficient",
            "Final rescued merged score table has at least 900000 rows",
            bool(rescued_rows is not None and rescued_rows >= min_full_rows),
            {
                "file": rel(full_rescued_scores, root),
                "rows": rescued_rows,
                "minRows": min_full_rows,
                "compatibleRows": compatible_rows,
            },
            reason="Post-finalization expects near-complete full docking coverage.",
        ),
        check(
            "finalizer_completed",
            "DiffDock finalizer completed",
            finalizer.get("phase") == "completed" or finalizer.get("completed") is True,
            {"statusFile": rel(root / "outputs/report_scale/diffdock_full_final_scores/finalize_after_rescues_status.json", root), **finalizer},
            reason="Finalizer must run after main and rescue queues.",
        ),
        check(
            "post_finalization_completed",
            "Post-finalization audits and table rebuilds completed",
            post_final.get("phase") == "completed" or post_final.get("alreadyCompleted") is True,
            {"statusFile": rel(root / "outputs/report_scale/full_diffdock_final/post_finalization_status.json", root), **post_final},
            reason="Structure, pose, shortlist, manifest, and dashboard assets must be refreshed after final merge.",
        ),
        check(
            "wetlab_package_ready",
            "Wet-lab validation package is present and narrowed to a first panel",
            bool(
                wetlab.get("firstExperimentPanelRows", 0) >= 12
                and wetlab.get("experimentExecutionProtocolCoreRows", 0) >= 6
                and wetlab.get("procurementPlatformChecklistRows", 0) >= 12
            ),
            {
                "summaryFile": rel(root / "outputs/sota_validation/wetlab_validation_package/wetlab_validation_summary.json", root),
                "candidateRows": wetlab.get("candidateRows"),
                "expertTop50Rows": wetlab.get("expertTop50Rows"),
                "wave1Rows": wetlab.get("wave1Rows"),
                "firstExperimentPanelRows": wetlab.get("firstExperimentPanelRows"),
                "experimentExecutionProtocolCoreRows": wetlab.get("experimentExecutionProtocolCoreRows"),
                "procurementPlatformChecklistRows": wetlab.get("procurementPlatformChecklistRows"),
            },
            reason="Experiment planning can be ready before final full DiffDock closure, but completion still requires final computation closure.",
        ),
    ]

    required_checks = [row for row in checks if row["required"]]
    passed_required = sum(1 for row in required_checks if row["passed"])
    overall_status = "complete" if passed_required == len(required_checks) else "not_complete"

    return {
        "createdUtc": utc_now(),
        "overallStatus": overall_status,
        "passedRequiredChecks": passed_required,
        "totalRequiredChecks": len(required_checks),
        "failedRequiredChecks": [row["key"] for row in required_checks if not row["passed"]],
        "checks": checks,
        "interpretation": (
            "The experiment is computationally closed only when every required check passes. "
            "Current failures are expected while the full DiffDock main/rescue/finalization chain is still running."
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Experiment Closure Audit",
        "",
        f"Generated UTC: {payload['createdUtc']}",
        "",
        f"Overall status: `{payload['overallStatus']}`",
        "",
        f"Required checks: {payload['passedRequiredChecks']} / {payload['totalRequiredChecks']} passed",
        "",
        "## Check Results",
        "",
        "|Check|Status|Reason|",
        "|---|---:|---|",
    ]
    for row in payload["checks"]:
        status = "pass" if row["passed"] else "fail"
        lines.append(f"|{row['label']}|`{status}`|{row.get('reason', '')}|")
    lines.extend(
        [
            "",
            "## Failed Required Checks",
            "",
        ]
    )
    failed = payload.get("failedRequiredChecks") or []
    if failed:
        lines.extend([f"- `{item}`" for item in failed])
    else:
        lines.append("- none")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_audit(root)
    out_dir = root / "outputs/sota_validation/experiment_closure_audit"
    out_json = out_dir / "experiment_closure_audit.json"
    out_md = out_dir / "EXPERIMENT_CLOSURE_AUDIT.md"
    write_json(out_json, payload)
    write_markdown(out_md, payload)
    print(
        json.dumps(
            {
                "out_json": rel(out_json, root),
                "out_md": rel(out_md, root),
                "overallStatus": payload["overallStatus"],
                "passedRequiredChecks": payload["passedRequiredChecks"],
                "totalRequiredChecks": payload["totalRequiredChecks"],
                "failedRequiredChecks": payload["failedRequiredChecks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

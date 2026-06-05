from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def job_id_from_path(path: Path) -> int | None:
    match = re.search(r"chunk_(\d+)", path.name)
    return int(match.group(1)) if match else None


def ligand_id_from_pair(pair_id: str) -> str:
    return pair_id.split("__", 1)[0] if "__" in pair_id else pair_id


def pct(part: int | float, total: int | float) -> float:
    return round((float(part) / float(total) * 100.0), 4) if total else 0.0


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def first_existing(values: list[str]) -> str:
    for value in values:
        if value:
            return value
    return ""


def file_exists(value: str) -> bool:
    return bool(value) and Path(value).exists()


def load_ligand_inputs(run_dir: Path) -> dict[str, dict[str, Any]]:
    ligand_inputs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "inputRows": 0,
            "ligandDescriptions": Counter(),
            "sourceInputCsvs": Counter(),
            "proteinPathsSeen": 0,
            "proteinPathsMissing": 0,
        }
    )
    for input_csv in sorted((run_dir / "inputs").glob("diffdock_full_chunk_*.csv")):
        for row in read_csv(input_csv):
            pair_id = row.get("complex_name", "")
            ligand_id = ligand_id_from_pair(pair_id)
            item = ligand_inputs[ligand_id]
            item["inputRows"] += 1
            item["ligandDescriptions"].update([row.get("ligand_description", "")])
            item["sourceInputCsvs"].update([str(input_csv)])
            protein_path = row.get("protein_path", "")
            if protein_path and Path(protein_path).exists():
                item["proteinPathsSeen"] += 1
            else:
                item["proteinPathsMissing"] += 1
    return ligand_inputs


def log_failure_signal(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"logExists": False, "maskRotateExceptionCount": 0, "failedAllComplexes": False, "tailSignal": ""}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"logExists": True, "maskRotateExceptionCount": 0, "failedAllComplexes": False, "tailSignal": "read_error"}
    tail = "\n".join(text.splitlines()[-12:])
    return {
        "logExists": True,
        "maskRotateExceptionCount": text.count("mask rotate exception"),
        "failedAllComplexes": bool(re.search(r"Failed for\s+\d+\s*/\s+\d+\s+complexes", text)),
        "tailSignal": "mask_rotate_exception" if "mask rotate exception" in text else tail[:300],
    }


def build_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = (root / args.run_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    ligand_inputs = load_ligand_inputs(run_dir)

    ligand_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scoredRows": 0,
            "completedRows": 0,
            "missingRows": 0,
            "errorCounts": Counter(),
            "sourceScoreCsvs": Counter(),
            "sourceJobIds": Counter(),
            "completedJobIds": set(),
            "missingJobIds": set(),
            "zeroCompletedJobIds": set(),
            "recentJobIds": set(),
            "exampleMissingPairs": [],
            "exampleCompletedPairs": [],
            "logMaskRotateExceptions": 0,
            "logFailedAllChunks": 0,
        }
    )
    chunk_stats: list[dict[str, Any]] = []

    for score_csv in sorted((run_dir / "scores").glob("diffdock_full_chunk_*.scores.csv")):
        job_id = job_id_from_path(score_csv)
        rows = read_csv(score_csv)
        completed = 0
        missing = 0
        ligand_counts: Counter[str] = Counter()
        for row in rows:
            pair_id = row.get("pair_id") or row.get("complex_name", "")
            ligand_id = ligand_id_from_pair(pair_id)
            ligand_counts.update([ligand_id])
            item = ligand_stats[ligand_id]
            item["scoredRows"] += 1
            item["sourceScoreCsvs"].update([str(score_csv)])
            if job_id is not None:
                item["sourceJobIds"].update([job_id])
                item["recentJobIds"].add(job_id)
            status = row.get("status", "")
            if status == "completed":
                completed += 1
                item["completedRows"] += 1
                if job_id is not None:
                    item["completedJobIds"].add(job_id)
                if len(item["exampleCompletedPairs"]) < 5:
                    item["exampleCompletedPairs"].append(pair_id)
            else:
                missing += 1
                item["missingRows"] += 1
                if job_id is not None:
                    item["missingJobIds"].add(job_id)
                item["errorCounts"].update([row.get("error", "") or "missing_status"])
                if len(item["exampleMissingPairs"]) < 5:
                    item["exampleMissingPairs"].append(pair_id)

        log_path = run_dir / "logs" / f"diffdock_full_chunk_{job_id:05d}.log" if job_id is not None else Path()
        log_signal = log_failure_signal(log_path)
        if completed == 0 and rows:
            for ligand_id in ligand_counts:
                item = ligand_stats[ligand_id]
                if job_id is not None:
                    item["zeroCompletedJobIds"].add(job_id)
                item["logMaskRotateExceptions"] += int(log_signal["maskRotateExceptionCount"])
                item["logFailedAllChunks"] += int(bool(log_signal["failedAllComplexes"]))

        chunk_stats.append(
            {
                "jobId": "" if job_id is None else job_id,
                "scoreCsv": rel(root, score_csv),
                "rows": len(rows),
                "completedRows": completed,
                "missingRows": missing,
                "missingPct": pct(missing, len(rows)),
                "uniqueLigands": len(ligand_counts),
                "topLigands": ";".join(f"{ligand}:{count}" for ligand, count in ligand_counts.most_common(5)),
                "maskRotateExceptionCount": log_signal["maskRotateExceptionCount"],
                "failedAllComplexes": log_signal["failedAllComplexes"],
                "technicalSignal": (
                    "all_missing_mask_rotate"
                    if completed == 0 and log_signal["maskRotateExceptionCount"]
                    else ("all_missing" if completed == 0 and rows else "")
                ),
            }
        )

    ligand_rows: list[dict[str, Any]] = []
    for ligand_id, item in ligand_stats.items():
        input_item = ligand_inputs.get(ligand_id, {})
        total = int(item["scoredRows"])
        completed = int(item["completedRows"])
        missing = int(item["missingRows"])
        ligand_descriptions = input_item.get("ligandDescriptions", Counter())
        source_inputs = input_item.get("sourceInputCsvs", Counter())
        zero_completed_jobs = sorted(item["zeroCompletedJobIds"])
        missing_jobs = sorted(item["missingJobIds"])
        completed_jobs = sorted(item["completedJobIds"])
        error_counts = item["errorCounts"].most_common(5)
        ligand_description = first_existing([value for value, _ in ligand_descriptions.most_common(3)])
        likely_parent_rescue = (
            missing >= args.min_missing_rows
            and pct(missing, total) >= args.min_missing_pct
            and (item["logMaskRotateExceptions"] > 0 or len(zero_completed_jobs) >= args.min_zero_completed_jobs)
        )
        ligand_rows.append(
            {
                "ligandId": ligand_id,
                "scoredRows": total,
                "completedRows": completed,
                "missingRows": missing,
                "missingPct": pct(missing, total),
                "inputRowsSeen": input_item.get("inputRows", 0),
                "ligandDescription": ligand_description,
                "ligandDescriptionExists": file_exists(ligand_description),
                "proteinPathsMissing": input_item.get("proteinPathsMissing", 0),
                "zeroCompletedJobs": len(zero_completed_jobs),
                "missingJobs": len(missing_jobs),
                "completedJobs": len(completed_jobs),
                "zeroCompletedJobIds": ";".join(str(value) for value in zero_completed_jobs[:40]),
                "missingJobIds": ";".join(str(value) for value in missing_jobs[:40]),
                "completedJobIds": ";".join(str(value) for value in completed_jobs[:40]),
                "topErrors": ";".join(f"{key}:{value}" for key, value in error_counts),
                "exampleMissingPairs": ";".join(item["exampleMissingPairs"]),
                "exampleCompletedPairs": ";".join(item["exampleCompletedPairs"]),
                "sourceInputCsvSample": ";".join(rel(root, Path(value)) for value, _ in source_inputs.most_common(5)),
                "sourceScoreCsvSample": ";".join(rel(root, Path(value)) for value, _ in item["sourceScoreCsvs"].most_common(5)),
                "logMaskRotateExceptions": item["logMaskRotateExceptions"],
                "logFailedAllChunks": item["logFailedAllChunks"],
                "rescueRecommendation": (
                    "prepare_parent_ligand_rescue"
                    if likely_parent_rescue
                    else ("monitor_until_full_run_complete" if missing else "none")
                ),
            }
        )

    ligand_rows.sort(key=lambda row: (float(row["missingPct"]), int(row["missingRows"]), int(row["scoredRows"])), reverse=True)
    rescue_rows = [row for row in ligand_rows if row["rescueRecommendation"] == "prepare_parent_ligand_rescue"]
    zero_chunk_rows = [row for row in chunk_stats if row["completedRows"] == 0 and row["rows"]]
    chunk_stats.sort(key=lambda row: (int(row["jobId"] or 0)))

    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Ligand-level technical failure audit for the full DiffDock run.",
        "runDir": str(run_dir),
        "scoredLigands": len(ligand_rows),
        "scoredRows": sum(int(row["scoredRows"]) for row in ligand_rows),
        "completedRows": sum(int(row["completedRows"]) for row in ligand_rows),
        "missingRows": sum(int(row["missingRows"]) for row in ligand_rows),
        "missingPct": pct(
            sum(int(row["missingRows"]) for row in ligand_rows),
            sum(int(row["scoredRows"]) for row in ligand_rows),
        ),
        "rescueRecommendedLigands": len(rescue_rows),
        "rescueRecommendedRows": sum(int(row["missingRows"]) for row in rescue_rows),
        "zeroCompletedChunks": len(zero_chunk_rows),
        "maskRotateZeroCompletedChunks": sum(1 for row in zero_chunk_rows if int(row["maskRotateExceptionCount"]) > 0),
        "topRescueLigands": rescue_rows[:20],
        "outputs": {
            "ligandAuditCsv": str(out_dir / "diffdock_full_ligand_failure_audit.csv"),
            "chunkAuditCsv": str(out_dir / "diffdock_full_chunk_failure_audit.csv"),
            "summaryJson": str(out_dir / "diffdock_full_ligand_failure_summary.json"),
            "summaryMd": str(out_dir / "DIFFDOCK_FULL_LIGAND_FAILURE_AUDIT.md"),
        },
        "interpretation": {
            "technicalFailure": "Rows marked missing are DiffDock technical output failures, not biological negatives.",
            "rescueRecommendation": "Ligands with high missing rate and mask-rotate/all-missing chunk signal should be considered for parent-ligand or repaired-SDF rescue after the main GPU queue is complete.",
            "gpuPolicy": "This audit is CPU-only and does not start any additional GPU queue.",
        },
    }

    write_csv(
        out_dir / "diffdock_full_ligand_failure_audit.csv",
        [
            "ligandId",
            "scoredRows",
            "completedRows",
            "missingRows",
            "missingPct",
            "inputRowsSeen",
            "ligandDescription",
            "ligandDescriptionExists",
            "proteinPathsMissing",
            "zeroCompletedJobs",
            "missingJobs",
            "completedJobs",
            "zeroCompletedJobIds",
            "missingJobIds",
            "completedJobIds",
            "topErrors",
            "exampleMissingPairs",
            "exampleCompletedPairs",
            "sourceInputCsvSample",
            "sourceScoreCsvSample",
            "logMaskRotateExceptions",
            "logFailedAllChunks",
            "rescueRecommendation",
        ],
        ligand_rows,
    )
    write_csv(
        out_dir / "diffdock_full_chunk_failure_audit.csv",
        [
            "jobId",
            "scoreCsv",
            "rows",
            "completedRows",
            "missingRows",
            "missingPct",
            "uniqueLigands",
            "topLigands",
            "maskRotateExceptionCount",
            "failedAllComplexes",
            "technicalSignal",
        ],
        chunk_stats,
    )
    write_json(out_dir / "diffdock_full_ligand_failure_summary.json", summary)
    (out_dir / "DIFFDOCK_FULL_LIGAND_FAILURE_AUDIT.md").write_text(
        "\n".join(
            [
                "# Full DiffDock Ligand-Level Technical Failure Audit",
                "",
                f"- Generated: {summary['createdUtc']}",
                f"- Scored rows audited: {summary['scoredRows']}",
                f"- Missing rows: {summary['missingRows']} ({summary['missingPct']:.2f}%)",
                f"- Ligands with rescue recommendation: {summary['rescueRecommendedLigands']}",
                f"- Missing rows covered by those ligands: {summary['rescueRecommendedRows']}",
                f"- Zero-completed chunks: {summary['zeroCompletedChunks']}",
                f"- Zero-completed chunks with mask-rotate signal: {summary['maskRotateZeroCompletedChunks']}",
                "",
                "## Top Rescue Candidates",
                "",
                "| Ligand | Scored rows | Missing rows | Missing % | Zero-completed jobs | Recommendation |",
                "|---|---:|---:|---:|---:|---|",
                *[
                    f"| {row['ligandId']} | {row['scoredRows']} | {row['missingRows']} | {float(row['missingPct']):.2f}% | {row['zeroCompletedJobs']} | {row['rescueRecommendation']} |"
                    for row in rescue_rows[:30]
                ],
                "",
                "These are technical DiffDock failures and should not be interpreted as biological negatives.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ligand-level technical missing-output patterns in the full DiffDock run.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default="outputs/report_scale/diffdock_full_run")
    parser.add_argument("--out-dir", default="outputs/report_scale/diffdock_full_run/ligand_failure_audit")
    parser.add_argument("--min-missing-rows", type=int, default=100)
    parser.add_argument("--min-missing-pct", type=float, default=95.0)
    parser.add_argument("--min-zero-completed-jobs", type=int, default=1)
    args = parser.parse_args()
    summary = build_audit(Path(args.root).resolve(), args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

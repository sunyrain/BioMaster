from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_full_diffdock_ligand_rescue_queue import INPUT_FIELDNAMES, write_csv, write_json, write_parent_ligand


JOB_FIELDNAMES = ["job_id", "chunk_csv", "out_dir", "score_csv", "log_file", "row_count", "gpu_slot", "status"]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def job_id_from_chunk_path(path: Path) -> int | None:
    match = re.search(r"chunk_(\d+)", path.name)
    return int(match.group(1)) if match else None


def load_rescue_ligands(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    audit_summary_path = resolve(root, args.audit_summary)
    audit = read_json(audit_summary_path)
    audit_csv = resolve(root, audit.get("outputs", {}).get("ligandAuditCsv", ""))
    rows = read_optional_csv(audit_csv)
    if not rows:
        rows = audit.get("topRescueLigands", [])

    excluded = set(args.exclude_ligand or [])
    candidates = [
        row
        for row in rows
        if row.get("ligandId")
        and row.get("ligandId") not in excluded
        and row.get("rescueRecommendation") == "prepare_parent_ligand_rescue"
        and str(row.get("ligandDescriptionExists", "True")).lower() in {"true", "1", "yes"}
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("missingPct") or 0),
            int(float(row.get("missingRows") or 0)),
            int(float(row.get("scoredRows") or 0)),
        ),
        reverse=True,
    )
    if args.max_ligands > 0:
        candidates = candidates[: args.max_ligands]
    return candidates


def load_selected_input_rows(run_dir: Path, selected_ligands: set[str]) -> dict[str, dict[str, dict[str, str]]]:
    rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for input_csv in sorted((run_dir / "inputs").glob("diffdock_full_chunk_*.csv")):
        job_id = job_id_from_chunk_path(input_csv)
        for row in read_optional_csv(input_csv):
            pair_id = row.get("complex_name", "")
            ligand_id = pair_id.split("__", 1)[0] if "__" in pair_id else pair_id
            if ligand_id not in selected_ligands:
                continue
            rows[ligand_id][pair_id] = {
                **row,
                "source_input_csv": str(input_csv),
                "source_job_id": "" if job_id is None else str(job_id),
            }
    return rows


def collect_selected_missing_rows(
    run_dir: Path,
    selected_ligands: set[str],
    max_source_job_id: int | None,
) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for score_csv in sorted((run_dir / "scores").glob("diffdock_full_chunk_*.scores.csv")):
        job_id = job_id_from_chunk_path(score_csv)
        if max_source_job_id is not None and job_id is not None and job_id > max_source_job_id:
            continue
        for row in read_optional_csv(score_csv):
            pair_id = row.get("pair_id") or row.get("complex_name", "")
            ligand_id = pair_id.split("__", 1)[0] if "__" in pair_id else pair_id
            if ligand_id not in selected_ligands or pair_id in seen:
                continue
            if row.get("status") == "completed":
                continue
            rows[ligand_id].append(
                {
                    "pair_id": pair_id,
                    "source_score_csv": str(score_csv),
                    "source_job_id": "" if job_id is None else str(job_id),
                    "source_status": row.get("status", ""),
                    "source_error": row.get("error", ""),
                    "source_chunk": row.get("source_chunk", ""),
                }
            )
            seen.add(pair_id)
    for ligand_rows in rows.values():
        ligand_rows.sort(key=lambda item: (int(item["source_job_id"] or 0), item["pair_id"]))
    return rows


def build_multi_queue(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve(root, args.out_dir)
    per_ligand_dir = out_dir / "per_ligand"
    aggregate_dir = out_dir / "aggregate"
    candidates = load_rescue_ligands(root, args)
    selected_ligands = {row["ligandId"] for row in candidates}
    run_dir = resolve(root, args.run_dir)
    input_rows_by_ligand = load_selected_input_rows(run_dir, selected_ligands)
    missing_rows_by_ligand = collect_selected_missing_rows(run_dir, selected_ligands, args.max_source_job_id)

    aggregate_jobs: list[dict[str, Any]] = []
    ligand_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    global_job_id = 0

    for row in candidates:
        ligand_id = row["ligandId"]
        original_sdf = resolve(root, row.get("ligandDescription") or f"data/processed/ligands_sdf_chembl/{ligand_id}.sdf")
        if not original_sdf.exists():
            skipped.append({**row, "skipReason": "original_sdf_missing", "originalSdf": str(original_sdf)})
            continue
        parent_sdf = resolve(root, f"data/processed/ligands_sdf_chembl_parent/{ligand_id}_parent.sdf")
        ligand_out_dir = per_ligand_dir / f"{ligand_id}_parent"
        ligand_info = write_parent_ligand(original_sdf, parent_sdf, ligand_id)
        input_by_pair = input_rows_by_ligand.get(ligand_id, {})
        missing_rows = missing_rows_by_ligand.get(ligand_id, [])
        rescue_inputs: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, Any]] = []

        for missing in missing_rows:
            source = input_by_pair.get(missing["pair_id"])
            if not source:
                skipped_rows.append({**missing, "skip_reason": "source_input_missing"})
                continue
            protein_path = source.get("protein_path", "")
            if not protein_path or not Path(protein_path).exists():
                skipped_rows.append({**missing, "skip_reason": "protein_path_missing", **source})
                continue
            rescue_inputs.append(
                {
                    "complex_name": missing["pair_id"],
                    "protein_path": protein_path,
                    "protein_sequence": source.get("protein_sequence", ""),
                    "ligand_description": str(parent_sdf),
                }
            )
            manifest_rows.append(
                {
                    **missing,
                    "source_input_csv": source.get("source_input_csv", ""),
                    "protein_path": protein_path,
                    "original_ligand_description": source.get("ligand_description", ""),
                    "rescue_ligand_description": str(parent_sdf),
                    "rescue_reason": "largest_organic_parent_for_multifragment_ligand",
                }
            )

        input_dir = ligand_out_dir / "inputs"
        output_dir = ligand_out_dir / "outputs"
        score_dir = ligand_out_dir / "scores"
        log_dir = ligand_out_dir / "logs"
        ligand_job_rows: list[dict[str, Any]] = []
        for ligand_job_id, start in enumerate(range(0, len(rescue_inputs), args.chunk_size)):
            chunk_rows = rescue_inputs[start : start + args.chunk_size]
            chunk_csv = input_dir / f"diffdock_ligand_rescue_chunk_{ligand_job_id:05d}.csv"
            write_csv(chunk_csv, INPUT_FIELDNAMES, chunk_rows)
            ligand_job_rows.append(
                {
                    "job_id": ligand_job_id,
                    "chunk_csv": rel(root, chunk_csv),
                    "out_dir": rel(root, output_dir / f"chunk_{ligand_job_id:05d}"),
                    "score_csv": rel(root, score_dir / f"diffdock_ligand_rescue_chunk_{ligand_job_id:05d}.scores.csv"),
                    "log_file": rel(root, log_dir / f"diffdock_ligand_rescue_chunk_{ligand_job_id:05d}.log"),
                    "row_count": len(chunk_rows),
                    "gpu_slot": ligand_job_id % max(args.gpu_slots, 1),
                    "status": "pending",
                }
            )

        ligand_job_index = ligand_out_dir / "diffdock_ligand_rescue_job_index.csv"
        write_csv(ligand_job_index, JOB_FIELDNAMES, ligand_job_rows)
        if manifest_rows:
            write_csv(ligand_out_dir / "diffdock_ligand_rescue_manifest.csv", sorted(manifest_rows[0].keys()), manifest_rows)
        if skipped_rows:
            write_csv(ligand_out_dir / "diffdock_ligand_rescue_skipped.csv", sorted(skipped_rows[0].keys()), skipped_rows)

        summary = {
            "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "Full DiffDock ligand-specific technical rescue queue.",
            "ligandId": ligand_id,
            "runDir": str(run_dir),
            "outDir": str(ligand_out_dir),
            "originalSdf": str(original_sdf),
            "parentSdf": str(parent_sdf),
            "inputRowsForLigand": len(input_by_pair),
            "missingRowsSelected": len(missing_rows),
            "queuedRows": len(rescue_inputs),
            "skippedRows": len(skipped_rows),
            "jobs": len(ligand_job_rows),
            "chunkSize": args.chunk_size,
            "maxSourceJobId": args.max_source_job_id,
            "jobIndex": str(ligand_job_index),
            "manifest": str(ligand_out_dir / "diffdock_ligand_rescue_manifest.csv"),
            "ligandPreparation": ligand_info,
        }
        write_json(ligand_out_dir / "diffdock_ligand_rescue_summary.json", summary)
        (ligand_out_dir / "DIFFDOCK_LIGAND_RESCUE_QUEUE.md").write_text(
            "\n".join(
                [
                    "# Full DiffDock Ligand Rescue Queue",
                    "",
                    f"- Generated: {summary['createdUtc']}",
                    f"- Ligand: `{ligand_id}`",
                    f"- Original SDF: `{original_sdf}`",
                    f"- Parent SDF: `{parent_sdf}`",
                    f"- Missing rows selected: {len(missing_rows)}",
                    f"- Queued rows: {len(rescue_inputs)}",
                    f"- Jobs: {len(ligand_job_rows)}",
                    f"- Job index: `{rel(root, ligand_job_index)}`",
                    "",
                    "This queue is for technical rescue of DiffDock rows that failed with the original ligand representation.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ligand_summaries.append(summary)
        for job in ligand_job_rows:
            aggregate_jobs.append(
                {
                    "job_id": global_job_id,
                    "chunk_csv": job["chunk_csv"],
                    "out_dir": job["out_dir"],
                    "score_csv": job["score_csv"],
                    "log_file": job["log_file"],
                    "row_count": job["row_count"],
                    "gpu_slot": global_job_id % max(args.gpu_slots, 1),
                    "status": "pending",
                }
            )
            global_job_id += 1

    aggregate_job_index = aggregate_dir / "diffdock_multi_ligand_rescue_job_index.csv"
    write_csv(aggregate_job_index, JOB_FIELDNAMES, aggregate_jobs)
    write_csv(
        aggregate_dir / "diffdock_multi_ligand_rescue_ligands.csv",
        sorted(candidates[0].keys()) if candidates else ["ligandId"],
        candidates,
    )
    if skipped:
        write_csv(aggregate_dir / "diffdock_multi_ligand_rescue_skipped.csv", sorted(skipped[0].keys()), skipped)

    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Aggregate post-full-run parent-ligand rescue queue for ligand-level DiffDock technical failures.",
        "auditSummary": str(resolve(root, args.audit_summary)),
        "outDir": str(out_dir),
        "candidateLigands": len(candidates),
        "queuedLigands": len([item for item in ligand_summaries if int(item.get("queuedRows", 0)) > 0]),
        "skippedLigands": len(skipped),
        "queuedRows": sum(int(item.get("queuedRows", 0)) for item in ligand_summaries),
        "jobs": len(aggregate_jobs),
        "chunkSize": args.chunk_size,
        "excludeLigands": args.exclude_ligand,
        "aggregateJobIndex": str(aggregate_job_index),
        "ligandSummaries": ligand_summaries,
        "executionNote": (
            "Prepared only. Run after the main full DiffDock queue and the current single-ligand rescue queue are complete."
        ),
    }
    write_json(aggregate_dir / "diffdock_multi_ligand_rescue_summary.json", summary)
    (aggregate_dir / "DIFFDOCK_MULTI_LIGAND_RESCUE_QUEUE.md").write_text(
        "\n".join(
            [
                "# DiffDock Multi-Ligand Rescue Queue",
                "",
                f"- Generated: {summary['createdUtc']}",
                f"- Candidate ligands: {summary['candidateLigands']}",
                f"- Queued ligands: {summary['queuedLigands']}",
                f"- Queued rows: {summary['queuedRows']}",
                f"- Jobs: {summary['jobs']}",
                f"- Aggregate job index: `{rel(root, aggregate_job_index)}`",
                "",
                "This queue targets ligand-level technical DiffDock output failures. It is not a biological negative/positive call.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an aggregate multi-ligand rescue queue for full DiffDock technical failures.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default="outputs/report_scale/diffdock_full_run")
    parser.add_argument(
        "--audit-summary",
        default="outputs/report_scale/diffdock_full_run/ligand_failure_audit/diffdock_full_ligand_failure_summary.json",
    )
    parser.add_argument("--out-dir", default="outputs/report_scale/diffdock_full_multi_ligand_rescue")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-source-job-id", type=int, default=None)
    parser.add_argument("--max-ligands", type=int, default=0, help="0 means all recommended ligands.")
    parser.add_argument("--exclude-ligand", action="append", default=["CHEMBL3039504"])
    parser.add_argument("--gpu-slots", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summary = build_multi_queue(root, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

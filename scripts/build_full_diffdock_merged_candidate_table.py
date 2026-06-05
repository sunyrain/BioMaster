from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any


BASE_FIELDS = [
    "full_diffdock_rank",
    "finalRankGlobal",
    "finalRankWithinDirection",
    "rankNumeric",
    "stage5_rank",
    "pair_id",
    "pairId",
    "drug_id",
    "drugId",
    "drug_name",
    "drug",
    "protein_id",
    "protein",
    "gene_name",
    "target",
    "protein_name",
    "proteinName",
    "direction",
    "disease_id",
    "disease_name",
    "combined_ai_score",
    "affinityScore",
    "open_targets_string_priority_score",
    "txgnn_indication_score",
    "direct_disease_score",
    "network_disease_score",
    "disease_evidence_status",
    "txgnn_component_status",
    "final_priority_score",
    "originalStage5PriorityScore",
    "finalPriorityScore",
    "knownDrugTargetPair",
    "finalPriorityTier",
    "reviewTrack",
    "noveltyClass",
    "directionScore",
    "diffdock_confidence",
    "diffdock",
    "diffdock_confidence_norm",
    "full_diffdock_consensus_score",
    "structural_status",
    "status",
    "docking_error",
    "confidence_sdf_path",
    "confidenceSdfPath",
    "rank1_sdf_path",
    "rank1SdfPath",
    "diffdock_receptor_pdb_path",
    "receptorPdbPath",
    "diffdock_receptor_status",
    "ligand_sdf_path",
    "ligand_smiles",
    "final_source",
    "rescue_attempted",
    "rescue_source",
    "rescue_status",
    "rescue_error",
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "NA"):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def fmt(value: float | None, digits: int = 9) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return lo, lo + 1.0
    return lo, hi


def load_scores(path: Path) -> tuple[dict[str, dict[str, str]], list[float], Counter[str]]:
    scores: dict[str, dict[str, str]] = {}
    confidences: list[float] = []
    status_counts: Counter[str] = Counter()
    for row in read_csv(path):
        pair_id = row.get("pair_id") or row.get("complex_name")
        if not pair_id:
            continue
        scores[pair_id] = {
            "diffdock_confidence": row.get("diffdock_confidence", ""),
            "rank1_sdf_path": row.get("rank1_sdf_path", ""),
            "confidence_sdf_path": row.get("confidence_sdf_path", ""),
            "status": row.get("status", ""),
            "error": row.get("error", ""),
            "final_source": row.get("final_source", ""),
            "rescue_attempted": row.get("rescue_attempted", ""),
            "rescue_source": row.get("rescue_source", ""),
            "rescue_status": row.get("rescue_status", ""),
            "rescue_error": row.get("rescue_error", ""),
        }
        status = row.get("status") or "NA"
        status_counts[status] += 1
        confidence = to_float(row.get("diffdock_confidence"))
        if status == "completed" and confidence is not None:
            confidences.append(confidence)
    return scores, confidences, status_counts


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        pair_id = row.get("pair_id", "")
        if not pair_id:
            continue
        manifest[pair_id] = {
            "ligand_sdf_path": row.get("ligand_sdf_path", ""),
            "ligand_smiles": row.get("ligand_smiles", ""),
            "diffdock_receptor_pdb_path": row.get("diffdock_receptor_pdb_path", ""),
            "diffdock_receptor_status": row.get("diffdock_receptor_status", ""),
            "diffdock_ready": row.get("diffdock_ready", ""),
        }
    return manifest


def consensus_score(stage5_score: float, confidence_norm: float | None, stage5_weight: float, diffdock_weight: float) -> float:
    if confidence_norm is None:
        return stage5_score
    return stage5_weight * stage5_score + diffdock_weight * confidence_norm


def build_row(
    stage5: dict[str, str],
    manifest: dict[str, str],
    score: dict[str, str],
    confidence_norm: float | None,
    consensus: float,
    structural_status: str,
) -> dict[str, Any]:
    pair_id = stage5.get("pair_id", "")
    drug_id = stage5.get("drug_id", "")
    drug_name = stage5.get("drug_name", "")
    protein_id = stage5.get("protein_id", "")
    gene_name = stage5.get("gene_name", "")
    final_priority_score = stage5.get("final_priority_score", "")
    combined_ai_score = stage5.get("combined_ai_score", "")
    confidence = score.get("diffdock_confidence", "")
    rank1 = score.get("rank1_sdf_path", "")
    confidence_sdf = score.get("confidence_sdf_path", "")
    receptor = manifest.get("diffdock_receptor_pdb_path", "")
    return {
        "full_diffdock_rank": "",
        "finalRankGlobal": "",
        "finalRankWithinDirection": "",
        "rankNumeric": "",
        "stage5_rank": stage5.get("stage5_rank", ""),
        "pair_id": pair_id,
        "pairId": pair_id,
        "drug_id": drug_id,
        "drugId": drug_id,
        "drug_name": drug_name,
        "drug": drug_name,
        "protein_id": protein_id,
        "protein": protein_id,
        "gene_name": gene_name,
        "target": gene_name,
        "protein_name": stage5.get("protein_name", ""),
        "proteinName": stage5.get("protein_name", ""),
        "direction": "oncology",
        "disease_id": stage5.get("disease_id", ""),
        "disease_name": stage5.get("disease_name", ""),
        "combined_ai_score": combined_ai_score,
        "affinityScore": combined_ai_score,
        "open_targets_string_priority_score": stage5.get("open_targets_string_priority_score", ""),
        "txgnn_indication_score": stage5.get("txgnn_indication_score", ""),
        "direct_disease_score": stage5.get("direct_disease_score", ""),
        "network_disease_score": stage5.get("network_disease_score", ""),
        "disease_evidence_status": stage5.get("disease_evidence_status", ""),
        "txgnn_component_status": stage5.get("txgnn_component_status", ""),
        "final_priority_score": final_priority_score,
        "originalStage5PriorityScore": final_priority_score,
        "finalPriorityScore": fmt(consensus),
        "knownDrugTargetPair": "0",
        "finalPriorityTier": "",
        "reviewTrack": "full_diffdock_stage5_structural_review",
        "noveltyClass": "",
        "directionScore": final_priority_score,
        "diffdock_confidence": confidence,
        "diffdock": confidence,
        "diffdock_confidence_norm": fmt(confidence_norm, 6),
        "full_diffdock_consensus_score": fmt(consensus),
        "structural_status": structural_status,
        "status": structural_status,
        "docking_error": score.get("error", ""),
        "confidence_sdf_path": confidence_sdf,
        "confidenceSdfPath": confidence_sdf,
        "rank1_sdf_path": rank1,
        "rank1SdfPath": rank1,
        "diffdock_receptor_pdb_path": receptor,
        "receptorPdbPath": receptor,
        "diffdock_receptor_status": manifest.get("diffdock_receptor_status", ""),
        "ligand_sdf_path": manifest.get("ligand_sdf_path", ""),
        "ligand_smiles": manifest.get("ligand_smiles", ""),
        "final_source": score.get("final_source", ""),
        "rescue_attempted": score.get("rescue_attempted", ""),
        "rescue_source": score.get("rescue_source", ""),
        "rescue_status": score.get("rescue_status", ""),
        "rescue_error": score.get("rescue_error", ""),
    }


def push_top(heap: list[tuple[float, int, dict[str, Any]]], limit: int, score: float, counter: int, row: dict[str, Any]) -> None:
    if limit <= 0:
        return
    item = (score, counter, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] = BASE_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_all_stream(path: Path, fieldnames: list[str] = BASE_FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def ranked_rows(heap: list[tuple[float, int, dict[str, Any]]]) -> list[dict[str, Any]]:
    ordered = [row for _, _, row in sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)]
    for rank, row in enumerate(ordered, start=1):
        row["full_diffdock_rank"] = rank
        row["finalRankGlobal"] = rank
        row["finalRankWithinDirection"] = rank
        row["rankNumeric"] = rank
    return ordered


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    stage5_path = resolve(root, args.stage5)
    manifest_path = resolve(root, args.manifest)
    scores_path = resolve(root, args.scores)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, path in [("stage5", stage5_path), ("manifest", manifest_path), ("scores", scores_path)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} input missing: {path}")

    scores, confidences, score_status_counts = load_scores(scores_path)
    confidence_min, confidence_max = minmax(confidences)
    manifest = load_manifest(manifest_path)

    out_all = out_dir / "full_diffdock_merged_stage5_all.csv"
    all_handle = None
    all_writer = None
    if args.write_all:
        all_handle, all_writer = write_all_stream(out_all)

    top_limit = max(args.top_n, args.large_top_n)
    heap: list[tuple[float, int, dict[str, Any]]] = []
    status_counts: Counter[str] = Counter()
    rescue_counts: Counter[str] = Counter()
    rows_total = 0
    manifest_hits = 0
    score_hits = 0
    completed_rows = 0
    not_structure_ready_rows = 0

    try:
        for counter, stage5 in enumerate(read_csv(stage5_path), start=1):
            rows_total += 1
            pair_id = stage5.get("pair_id", "")
            manifest_row = manifest.get(pair_id, {})
            score_row = scores.get(pair_id, {})
            if manifest_row:
                manifest_hits += 1
            if score_row:
                score_hits += 1

            confidence = to_float(score_row.get("diffdock_confidence"))
            if score_row.get("status") == "completed" and confidence is not None:
                confidence_norm = (confidence - confidence_min) / (confidence_max - confidence_min)
                structural_status = "completed"
                completed_rows += 1
            elif score_row:
                confidence_norm = None
                structural_status = score_row.get("status") or "missing_output"
            elif manifest_row:
                confidence_norm = None
                structural_status = "not_scored"
            else:
                confidence_norm = None
                structural_status = "not_structure_ready"
                not_structure_ready_rows += 1

            stage5_score = to_float(stage5.get("final_priority_score"), 0.0) or 0.0
            full_score = consensus_score(stage5_score, confidence_norm, args.stage5_weight, args.diffdock_weight)
            row = build_row(stage5, manifest_row, score_row, confidence_norm, full_score, structural_status)
            status_counts[structural_status] += 1
            rescue_counts[row.get("final_source", "") or "NA"] += 1
            if all_writer is not None:
                all_writer.writerow(row)
            push_top(heap, top_limit, full_score, counter, row)
    finally:
        if all_handle is not None:
            all_handle.close()

    top_rows = ranked_rows(heap)
    top_large = top_rows[: args.large_top_n] if args.large_top_n > 0 else []
    top_small = top_rows[: args.top_n] if args.top_n > 0 else []

    out_top_large = out_dir / f"full_diffdock_merged_stage5_top{args.large_top_n}.csv"
    out_top_small = out_dir / f"full_diffdock_merged_stage5_top{args.top_n}.csv"
    audit_input = out_dir / f"full_diffdock_structure_audit_input_top{args.top_n}.csv"
    if top_large:
        write_rows(out_top_large, top_large)
    if top_small:
        write_rows(out_top_small, top_small)
        write_rows(audit_input, top_small)

    summary = {
        "createdUtc": utc_now(),
        "stage5": str(stage5_path),
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "outDir": str(out_dir),
        "allRowsCsv": str(out_all) if args.write_all else "",
        "topCsv": str(out_top_small) if top_small else "",
        "largeTopCsv": str(out_top_large) if top_large else "",
        "structureAuditInputCsv": str(audit_input) if top_small else "",
        "stage5Rows": rows_total,
        "manifestRows": len(manifest),
        "scoreRows": len(scores),
        "stage5RowsWithManifest": manifest_hits,
        "stage5RowsWithScores": score_hits,
        "completedRows": completed_rows,
        "completedPctOfStage5": round(completed_rows / rows_total * 100.0, 4) if rows_total else 0.0,
        "notStructureReadyRows": not_structure_ready_rows,
        "confidenceMin": confidence_min,
        "confidenceMax": confidence_max,
        "stage5Weight": args.stage5_weight,
        "diffdockWeight": args.diffdock_weight,
        "scoreStatusCounts": dict(score_status_counts),
        "mergedStructuralStatusCounts": dict(status_counts),
        "finalSourceCounts": dict(rescue_counts),
        "topN": args.top_n,
        "topNCompletedRows": sum(1 for row in top_small if row.get("structural_status") == "completed"),
        "largeTopN": args.large_top_n,
        "largeTopNCompletedRows": sum(1 for row in top_large if row.get("structural_status") == "completed"),
        "interpretation": {
            "fullDiffDockConsensusScore": "stage5_weight * Stage5 final_priority_score + diffdock_weight * normalized DiffDock confidence when a completed pose exists; otherwise the Stage5 score is retained.",
            "missingOutput": "Rows marked missing_output are technical DiffDock output misses, not biological negatives.",
            "notStructureReady": "Rows absent from the DiffDock-ready manifest were not included in the 913170-row full DiffDock queue.",
        },
    }
    summary_path = out_dir / "full_diffdock_merged_stage5_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "FULL_DIFFDOCK_MERGED_STAGE5_SUMMARY.md"
    md_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "topCsv": summary.get("topCsv"), "completedRows": completed_rows}, ensure_ascii=False, indent=2))
    return summary


def markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full DiffDock Merged Stage5 Summary",
            "",
            f"Generated: {summary.get('createdUtc')}",
            "",
            "## Inputs",
            "",
            f"- Stage5 table: `{summary.get('stage5')}`",
            f"- DiffDock-ready manifest: `{summary.get('manifest')}`",
            f"- Final merged DiffDock scores: `{summary.get('scores')}`",
            "",
            "## Coverage",
            "",
            f"- Stage5 rows: {summary.get('stage5Rows')}",
            f"- Manifest rows: {summary.get('manifestRows')}",
            f"- Score rows: {summary.get('scoreRows')}",
            f"- Completed DiffDock rows: {summary.get('completedRows')} ({summary.get('completedPctOfStage5')}% of Stage5 rows)",
            f"- Structural status counts: {summary.get('mergedStructuralStatusCounts')}",
            f"- Final source counts: {summary.get('finalSourceCounts')}",
            "",
            "## Outputs",
            "",
            f"- Full merged table: `{summary.get('allRowsCsv')}`",
            f"- Top{summary.get('topN')} table: `{summary.get('topCsv')}`",
            f"- Top{summary.get('largeTopN')} table: `{summary.get('largeTopCsv')}`",
            f"- Structure-audit input: `{summary.get('structureAuditInputCsv')}`",
            "",
            "Rows marked `missing_output` remain technical output misses rather than biological negatives.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge final full DiffDock scores back onto the 915k Stage5 candidate table.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--stage5", default="outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv")
    parser.add_argument("--manifest", default="outputs/report_scale/manifest_915k_diffdock_ready.csv")
    parser.add_argument("--scores", default="outputs/report_scale/diffdock_scores_full_913170_with_rescues.csv")
    parser.add_argument("--out-dir", default="outputs/report_scale/full_diffdock_final")
    parser.add_argument("--stage5-weight", type=float, default=0.85)
    parser.add_argument("--diffdock-weight", type=float, default=0.15)
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--large-top-n", type=int, default=100000)
    parser.add_argument("--write-all", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

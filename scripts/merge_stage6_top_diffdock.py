from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def to_float(value: str, default: float | None = None) -> float | None:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return default
    try:
        return float(value)
    except ValueError:
        return default


def minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return lo, lo + 1.0
    return lo, hi


def collect_scores(score_dir: Path) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    for path in sorted(score_dir.glob("*.scores.csv")):
        for row in read_csv(path):
            scores[row["pair_id"]] = row
    return scores


def merge(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    top_manifest = resolve(root, args.top_manifest)
    score_dir = resolve(root, args.score_dir)
    out_csv = resolve(root, args.out_csv)
    rows = read_csv(top_manifest)
    scores = collect_scores(score_dir)

    confidences = [
        value
        for value in (to_float(score.get("diffdock_confidence", "")) for score in scores.values())
        if value is not None
    ]
    lo, hi = minmax(confidences)

    merged: list[dict[str, Any]] = []
    for row in rows:
        pair_id = row["pair_id"]
        score = scores.get(pair_id, {})
        confidence = to_float(score.get("diffdock_confidence", ""))
        if confidence is None:
            confidence_norm = ""
            structural_status = score.get("status") or "not_yet_run"
        else:
            confidence_norm = (confidence - lo) / (hi - lo)
            structural_status = "completed"

        stage5_score = to_float(row.get("stage5_final_priority_score", ""), 0.0) or 0.0
        if confidence is None:
            stage6_score = stage5_score
        else:
            stage6_score = args.stage5_weight * stage5_score + args.diffdock_weight * float(confidence_norm)

        merged.append(
            {
                "pair_id": pair_id,
                "stage5_rank": row.get("stage5_stage5_rank", ""),
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "protein_id": row.get("protein_id", ""),
                "gene_name": row.get("gene_name", ""),
                "protein_name": row.get("protein_name", ""),
                "stage5_final_priority_score": row.get("stage5_final_priority_score", ""),
                "combined_ai_score": row.get("stage5_combined_ai_score", ""),
                "direct_disease_score": row.get("stage5_direct_disease_score", ""),
                "network_disease_score": row.get("stage5_network_disease_score", ""),
                "txgnn_indication_score": row.get("stage5_txgnn_indication_score", ""),
                "diffdock_confidence": "" if confidence is None else f"{confidence:.2f}",
                "diffdock_confidence_norm": "" if confidence is None else f"{confidence_norm:.6f}",
                "stage6_consensus_score": f"{stage6_score:.9f}",
                "structural_status": structural_status,
                "rank1_sdf_path": score.get("confidence_sdf_path", ""),
                "diffdock_receptor_status": row.get("diffdock_receptor_status", ""),
                "disease_evidence_status": row.get("stage5_disease_evidence_status", ""),
                "txgnn_component_status": row.get("stage5_txgnn_component_status", ""),
            }
        )

    merged.sort(key=lambda item: float(item["stage6_consensus_score"]), reverse=True)
    for idx, row in enumerate(merged, start=1):
        row["stage6_rank"] = idx

    fieldnames = [
        "stage6_rank",
        "pair_id",
        "stage5_rank",
        "drug_id",
        "drug_name",
        "protein_id",
        "gene_name",
        "protein_name",
        "stage5_final_priority_score",
        "combined_ai_score",
        "direct_disease_score",
        "network_disease_score",
        "txgnn_indication_score",
        "diffdock_confidence",
        "diffdock_confidence_norm",
        "stage6_consensus_score",
        "structural_status",
        "rank1_sdf_path",
        "diffdock_receptor_status",
        "disease_evidence_status",
        "txgnn_component_status",
    ]
    write_csv(out_csv, fieldnames, merged)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "top_manifest": str(top_manifest),
        "score_dir": str(score_dir),
        "out_csv": str(out_csv),
        "rows": len(merged),
        "score_files": len(list(score_dir.glob("*.scores.csv"))),
        "diffdock_completed": sum(1 for row in merged if row["structural_status"] == "completed"),
        "stage5_weight": args.stage5_weight,
        "diffdock_weight": args.diffdock_weight,
        "confidence_min": lo,
        "confidence_max": hi,
    }
    out_csv.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Stage 5 Top-N with completed DiffDock scores into a Stage 6 consensus table.")
    parser.add_argument("--top-manifest", default="outputs/report_scale/stage5_top1000_diffdock_ready_manifest.csv")
    parser.add_argument("--score-dir", default="outputs/report_scale/diffdock_top1000_run/scores")
    parser.add_argument("--out-csv", default="outputs/report_scale/stage6_top1000_consensus_candidates.csv")
    parser.add_argument("--stage5-weight", type=float, default=0.85)
    parser.add_argument("--diffdock-weight", type=float, default=0.15)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

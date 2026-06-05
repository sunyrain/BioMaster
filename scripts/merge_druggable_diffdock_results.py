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


def to_float(value: str | None, default: float | None = None) -> float | None:
    if value in (None, "", "NA"):
        return default
    try:
        number = float(value)
    except ValueError:
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def minmax(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return lo, lo + 1.0
    return lo, hi


def load_scores(score_dir: Path) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    for path in sorted(score_dir.glob("*.scores.csv")):
        for row in read_csv(path):
            scores[row["pair_id"]] = row
    return scores


def norm(value: float | None, lo: float, hi: float) -> float | None:
    if value is None:
        return None
    return (value - lo) / (hi - lo)


def fmt(value: float | None, digits: int = 9) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def build_unique_rows(
    representative_rows: list[dict[str, str]],
    scores: dict[str, dict[str, str]],
    affinity_weight: float,
    diffdock_weight: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    confidences = [
        value
        for value in (to_float(score.get("diffdock_confidence")) for score in scores.values())
        if value is not None
    ]
    lo, hi = minmax(confidences)
    merged: list[dict[str, Any]] = []

    for row in representative_rows:
        pair_id = row["pair_id"]
        score = scores.get(pair_id, {})
        confidence = to_float(score.get("diffdock_confidence"))
        confidence_norm = norm(confidence, lo, hi)
        base_score = (
            to_float(row.get("combined_ai_score"))
            or to_float(row.get("affinity_component"))
            or to_float(row.get("affinity_score"))
            or 0.0
        )
        if confidence_norm is None:
            consensus = base_score
            structural_status = score.get("status") or "not_yet_run"
        else:
            consensus = affinity_weight * base_score + diffdock_weight * confidence_norm
            structural_status = "completed"

        merged.append(
            {
                "pair_id": pair_id,
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "protein_id": row.get("protein_id", ""),
                "gene_name": row.get("gene_name", ""),
                "protein_name": row.get("protein_name", ""),
                "sequence_key": row.get("sequence_key", ""),
                "affinity_score": row.get("affinity_score", ""),
                "affinity_component": row.get("affinity_component", ""),
                "combined_ai_score": row.get("combined_ai_score", ""),
                "diffdock_confidence": "" if confidence is None else f"{confidence:.2f}",
                "diffdock_confidence_norm": fmt(confidence_norm, 6),
                "stage6_consensus_score": fmt(consensus),
                "structural_status": structural_status,
                "docking_error": score.get("error", ""),
                "confidence_sdf_path": score.get("confidence_sdf_path", ""),
                "rank1_sdf_path": score.get("rank1_sdf_path", ""),
                "diffdock_receptor_pdb_path": row.get("diffdock_receptor_pdb_path", ""),
                "diffdock_receptor_status": row.get("diffdock_receptor_status", ""),
                "diffdock_residue_count": row.get("diffdock_residue_count", ""),
                "represented_pair_count": row.get("represented_pair_count", ""),
                "represented_protein_ids": row.get("represented_protein_ids", ""),
                "represented_gene_names": row.get("represented_gene_names", ""),
                "source_pair_id": row.get("source_pair_id", ""),
                "representative_selection_reason": row.get("representative_selection_reason", ""),
            }
        )

    merged.sort(key=lambda item: float(item["stage6_consensus_score"] or 0.0), reverse=True)
    for rank, row in enumerate(merged, start=1):
        row["stage6_unique_rank"] = rank

    by_drug_sequence = {
        (row["drug_id"], row["sequence_key"]): row
        for row in merged
        if row.get("drug_id") and row.get("sequence_key")
    }
    metadata = {
        "diffdock_confidence_min": lo,
        "diffdock_confidence_max": hi,
        "diffdock_score_rows": len(scores),
        "diffdock_completed": sum(1 for row in merged if row["structural_status"] == "completed"),
        "diffdock_missing_or_failed": sum(1 for row in merged if row["structural_status"] != "completed"),
    }
    return merged, by_drug_sequence, metadata


def build_top10000_rows(
    top_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    unique_by_drug_sequence: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_pair = {row["pair_id"]: row for row in manifest_rows}
    expanded: list[dict[str, Any]] = []

    for row in top_rows:
        pair_id = row["pair_id"]
        manifest = manifest_by_pair.get(pair_id, {})
        sequence_key = manifest.get("sequence_key", "")
        representative = unique_by_drug_sequence.get((row.get("drug_id", ""), sequence_key), {})
        base_score = (
            to_float(row.get("combined_ai_score"))
            or to_float(row.get("affinity_component"))
            or to_float(row.get("affinity_score"))
            or 0.0
        )
        consensus = to_float(str(representative.get("stage6_consensus_score", "")), base_score) or base_score

        expanded.append(
            {
                "pair_id": pair_id,
                "stage4_rank": row.get("stage4_rank", ""),
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "protein_id": row.get("protein_id", ""),
                "gene_name": row.get("gene_name", ""),
                "protein_name": row.get("protein_name", ""),
                "sequence_key": sequence_key,
                "affinity_score": row.get("affinity_score", ""),
                "affinity_component": row.get("affinity_component", ""),
                "combined_ai_score": row.get("combined_ai_score", ""),
                "stage6_consensus_score": fmt(consensus),
                "representative_pair_id": representative.get("pair_id", ""),
                "representative_protein_id": representative.get("protein_id", ""),
                "representative_gene_name": representative.get("gene_name", ""),
                "stage6_unique_rank": representative.get("stage6_unique_rank", ""),
                "diffdock_confidence": representative.get("diffdock_confidence", ""),
                "diffdock_confidence_norm": representative.get("diffdock_confidence_norm", ""),
                "structural_status": representative.get("structural_status", "not_yet_run"),
                "docking_error": representative.get("docking_error", ""),
                "confidence_sdf_path": representative.get("confidence_sdf_path", ""),
                "rank1_sdf_path": representative.get("rank1_sdf_path", ""),
                "diffdock_receptor_pdb_path": representative.get("diffdock_receptor_pdb_path", ""),
                "diffdock_receptor_status": representative.get("diffdock_receptor_status", ""),
                "represented_pair_count": representative.get("represented_pair_count", ""),
                "represented_protein_ids": representative.get("represented_protein_ids", ""),
                "represented_gene_names": representative.get("represented_gene_names", ""),
            }
        )

    expanded.sort(key=lambda item: float(item["stage6_consensus_score"] or 0.0), reverse=True)
    for rank, row in enumerate(expanded, start=1):
        row["stage6_rank"] = rank
    return expanded


def merge(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    representative_manifest = resolve(root, args.representative_manifest)
    top_manifest = resolve(root, args.top_manifest)
    top_candidates = resolve(root, args.top_candidates)
    score_dir = resolve(root, args.score_dir)
    out_unique = resolve(root, args.out_unique)
    out_top10000 = resolve(root, args.out_top10000)

    representative_rows = read_csv(representative_manifest)
    top_rows = read_csv(top_candidates)
    manifest_rows = read_csv(top_manifest)
    scores = load_scores(score_dir)

    unique_rows, unique_by_drug_sequence, score_metadata = build_unique_rows(
        representative_rows=representative_rows,
        scores=scores,
        affinity_weight=args.affinity_weight,
        diffdock_weight=args.diffdock_weight,
    )
    top10000_rows = build_top10000_rows(
        top_rows=top_rows,
        manifest_rows=manifest_rows,
        unique_by_drug_sequence=unique_by_drug_sequence,
    )

    unique_fields = [
        "stage6_unique_rank",
        "pair_id",
        "drug_id",
        "drug_name",
        "protein_id",
        "gene_name",
        "protein_name",
        "sequence_key",
        "affinity_score",
        "affinity_component",
        "combined_ai_score",
        "diffdock_confidence",
        "diffdock_confidence_norm",
        "stage6_consensus_score",
        "structural_status",
        "docking_error",
        "confidence_sdf_path",
        "rank1_sdf_path",
        "diffdock_receptor_pdb_path",
        "diffdock_receptor_status",
        "diffdock_residue_count",
        "represented_pair_count",
        "represented_protein_ids",
        "represented_gene_names",
        "source_pair_id",
        "representative_selection_reason",
    ]
    top_fields = [
        "stage6_rank",
        "pair_id",
        "stage4_rank",
        "drug_id",
        "drug_name",
        "protein_id",
        "gene_name",
        "protein_name",
        "sequence_key",
        "affinity_score",
        "affinity_component",
        "combined_ai_score",
        "stage6_consensus_score",
        "representative_pair_id",
        "representative_protein_id",
        "representative_gene_name",
        "stage6_unique_rank",
        "diffdock_confidence",
        "diffdock_confidence_norm",
        "structural_status",
        "docking_error",
        "confidence_sdf_path",
        "rank1_sdf_path",
        "diffdock_receptor_pdb_path",
        "diffdock_receptor_status",
        "represented_pair_count",
        "represented_protein_ids",
        "represented_gene_names",
    ]
    write_csv(out_unique, unique_fields, unique_rows)
    write_csv(out_top10000, top_fields, top10000_rows)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "representative_manifest": str(representative_manifest),
        "top_manifest": str(top_manifest),
        "top_candidates": str(top_candidates),
        "score_dir": str(score_dir),
        "out_unique": str(out_unique),
        "out_top10000": str(out_top10000),
        "unique_rows": len(unique_rows),
        "top10000_rows": len(top10000_rows),
        "affinity_weight": args.affinity_weight,
        "diffdock_weight": args.diffdock_weight,
        **score_metadata,
    }
    metadata_path = out_top10000.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_unique.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge druggable-proteome DiffDock scores into final candidate tables.")
    parser.add_argument(
        "--representative-manifest",
        default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest_diffdock_ready.csv",
    )
    parser.add_argument(
        "--top-manifest",
        default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest.csv",
    )
    parser.add_argument(
        "--top-candidates",
        default="outputs/druggable_proteome/stage4_affinity_candidates_druggable_top10000.csv",
    )
    parser.add_argument("--score-dir", default="outputs/druggable_proteome/diffdock_top_unique_run/scores")
    parser.add_argument(
        "--out-unique",
        default="outputs/druggable_proteome/stage6_druggable_top_unique_diffdock_consensus.csv",
    )
    parser.add_argument(
        "--out-top10000",
        default="outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.csv",
    )
    parser.add_argument("--affinity-weight", type=float, default=0.85)
    parser.add_argument("--diffdock-weight", type=float, default=0.15)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

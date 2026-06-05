from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DOCS_ASSET_DIR = Path("docs/assets")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: str | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def rel_to_docs(path: Path) -> str:
    parts = path.resolve().parts
    docs_index = parts.index("docs")
    return Path(*parts[docs_index + 1 :]).as_posix()


def copy_asset(source_value: str, destination: Path) -> str:
    source = Path(source_value)
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return rel_to_docs(destination)


def build_candidate(row: dict[str, str]) -> dict[str, Any]:
    return {
        "rank": int(row["stage6_unique_rank"]),
        "pairId": row["pair_id"],
        "drugId": row["drug_id"],
        "drug": row["drug_name"],
        "target": row["gene_name"],
        "protein": row["protein_id"],
        "proteinName": row["protein_name"],
        "aiScore": number(row.get("combined_ai_score")),
        "diseasePriority": number(row.get("affinity_score")),
        "affinityScore": number(row.get("affinity_score")),
        "affinityComponent": number(row.get("affinity_component")),
        "representedPairCount": int(float(row.get("represented_pair_count") or 1)),
        "representedProteins": row.get("represented_protein_ids", ""),
        "diffdock": number(row.get("diffdock_confidence")),
        "consensus": number(row.get("stage6_consensus_score")),
        "status": row.get("structural_status", ""),
        "receptorStatus": row.get("diffdock_receptor_status", ""),
        "selectionReason": row.get("representative_selection_reason", ""),
    }


def build_structure_samples(unique_rows: list[dict[str, str]], structures_dir: Path, limit: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in unique_rows:
        if row.get("structural_status") != "completed":
            continue
        ligand_source = row.get("confidence_sdf_path") or row.get("rank1_sdf_path")
        receptor_source = row.get("diffdock_receptor_pdb_path")
        if not ligand_source or not receptor_source:
            continue
        safe_pair = row["pair_id"].replace("__", "_")
        try:
            ligand_url = copy_asset(ligand_source, structures_dir / f"{safe_pair}_ligand.sdf")
            receptor_url = copy_asset(receptor_source, structures_dir / f"{safe_pair}_receptor.pdb")
        except FileNotFoundError:
            continue
        samples.append(
            {
                "pairId": row["pair_id"],
                "rank": int(row["stage6_unique_rank"]),
                "drug": row["drug_name"],
                "target": row["gene_name"],
                "protein": row["protein_id"],
                "confidence": number(row.get("diffdock_confidence")),
                "consensus": number(row.get("stage6_consensus_score")),
                "receptorStatus": row.get("diffdock_receptor_status", ""),
                "ligandUrl": ligand_url,
                "receptorUrl": receptor_url,
            }
        )
        if len(samples) >= limit:
            break
    return samples


def counter_rows(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    items = counter.most_common(limit)
    return [{"label": label or "NA", "value": value} for label, value in items]


def build_payload(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    unique_rows = read_csv(root / args.unique_csv)
    top_rows = read_csv(root / args.top_csv)
    prep = load_json(root / args.prep_metadata)
    expansion = load_json(root / args.expansion_metadata)
    merge_metadata_path = root / args.merge_metadata
    merge_metadata = load_json(merge_metadata_path) if merge_metadata_path.exists() else {}

    structural_status = Counter(row.get("structural_status", "not_yet_run") for row in unique_rows)
    receptor_status = Counter(row.get("diffdock_receptor_status", "NA") for row in unique_rows)
    top_targets = Counter(row.get("gene_name", "NA") for row in top_rows)
    top_drugs = Counter(row.get("drug_name", "NA") for row in top_rows)
    selection_reasons = Counter(row.get("representative_selection_reason", "NA") for row in unique_rows)

    completed = structural_status.get("completed", 0)
    docking_total = len(unique_rows)
    processed = sum(1 for row in unique_rows if row.get("structural_status") != "not_yet_run")
    progress_rate = (processed / docking_total * 100.0) if docking_total else 0.0
    output_rate = (completed / docking_total * 100.0) if docking_total else 0.0

    return {
        "updated": merge_metadata.get("created_utc") or prep.get("created_utc"),
        "mode": "druggable_proteome",
        "labels": {
            "primaryScore": "Affinity",
            "primaryScoreLong": "ConPLex affinity score",
            "candidateScope": "Top druggable-proteome representatives",
        },
        "metrics": {
            "drugs": prep["drug_rows_usable"],
            "targets": prep["protein_rows_input_valid"],
            "uniqueSequences": prep["unique_sequences"],
            "pairs": expansion["expanded_affinity_rows_written"],
            "topCandidates": len(top_rows),
            "structureCandidates": docking_total,
            "structureCompleted": completed,
            "structureMissing": docking_total - completed,
            "top1000Completed": completed,
            "top1000Missing": docking_total - completed,
            "fullScoreFiles": len(list((root / args.score_dir).glob("*.scores.csv"))),
            "fullJobsTotal": 8,
            "fullRowsScored": processed,
            "fullRowsTotal": docking_total,
            "fullRowProgressPct": progress_rate,
            "fullCompletedOutputs": completed,
            "fullMissingOutputs": docking_total - completed,
            "fullOutputRatePct": output_rate,
            "zeroCompletedChunks": 0,
        },
        "charts": {
            "evidenceCoverage": [
                {"label": "ConPLex screened", "value": expansion["expanded_affinity_rows_written"]},
                {"label": "Top affinity set", "value": len(top_rows)},
                {"label": "Docking representatives", "value": docking_total},
                {"label": "Completed structures", "value": completed},
            ],
            "topTargets": counter_rows(top_targets, 10),
            "topDrugs": counter_rows(top_drugs, 10),
            "structuralStatus": counter_rows(structural_status),
            "receptorStatus": counter_rows(receptor_status),
            "txgnnStatus": counter_rows(selection_reasons),
        },
        "candidates": [build_candidate(row) for row in unique_rows[:80]],
        "structureSamples": build_structure_samples(
            unique_rows=unique_rows,
            structures_dir=root / DOCS_ASSET_DIR / "structures",
            limit=args.structure_samples,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dashboard assets for the druggable-proteome workflow.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--unique-csv",
        default="outputs/druggable_proteome/stage6_druggable_top_unique_diffdock_consensus.csv",
    )
    parser.add_argument(
        "--top-csv",
        default="outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.csv",
    )
    parser.add_argument(
        "--prep-metadata",
        default="outputs/druggable_proteome/druggable_proteome_conplex_prep.metadata.json",
    )
    parser.add_argument(
        "--expansion-metadata",
        default="outputs/druggable_proteome/druggable_conplex_expansion.metadata.json",
    )
    parser.add_argument(
        "--merge-metadata",
        default="outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.metadata.json",
    )
    parser.add_argument("--score-dir", default="outputs/druggable_proteome/diffdock_top_unique_run/scores")
    parser.add_argument("--out", default="docs/assets/dashboard-data.js")
    parser.add_argument("--structure-samples", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_payload(root, args)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "window.BIOMASTER_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(out_path),
                "candidates": len(payload["candidates"]),
                "structure_samples": len(payload["structureSamples"]),
                "completed": payload["metrics"]["top1000Completed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

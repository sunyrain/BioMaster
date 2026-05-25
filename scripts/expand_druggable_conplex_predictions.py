from __future__ import annotations

import argparse
import csv
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


AFFINITY_FIELDS = ["pair_id", "model", "affinity_score", "kd_nm"]

STAGE4_FIELDS = [
    "stage4_rank",
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "diffdock_confidence",
    "docking_score",
    "docking_component",
    "affinity_model",
    "affinity_score",
    "kd_nm",
    "affinity_component",
    "combined_ai_score",
]

DIFFDOCK_SEED_FIELDS = [
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "ligand_sdf_path",
    "ligand_smiles",
    "receptor_pdb_url",
    "receptor_cif_url",
    "protein_sequence",
    "diffdock_output_dir",
    "status",
    "sequence_key",
    "affinity_score",
    "affinity_component",
    "combined_ai_score",
]


@dataclass(frozen=True)
class Prediction:
    drug_id: str
    sequence_key: str
    score_text: str
    score: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


def as_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_predictions(path: Path) -> tuple[list[Prediction], int]:
    predictions: list[Prediction] = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for values in reader:
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) < 3:
                skipped += 1
                continue
            drug_id, sequence_key, score_text = [value.strip() for value in values[:3]]
            if drug_id.lower() in {"moleculeid", "molecule_id", "drug_id"}:
                continue
            score = as_float(score_text)
            if not drug_id or not sequence_key or score is None:
                skipped += 1
                continue
            predictions.append(Prediction(drug_id=drug_id, sequence_key=sequence_key, score_text=score_text, score=score))
    return predictions, skipped


def normalize(value: float, minimum: float, maximum: float) -> float:
    if minimum == maximum:
        return 1.0
    return round((value - minimum) / (maximum - minimum), 6)


def alphafold_pdb_url(protein_id: str, version: int) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v{version}.pdb"


def alphafold_cif_url(protein_id: str, version: int) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v{version}.cif"


def expanded_rows(
    predictions: Iterable[Prediction],
    proteins_by_sequence: dict[str, list[dict[str, str]]],
    drugs_by_id: dict[str, dict[str, str]],
    score_min: float,
    score_max: float,
    alphafold_version: int,
    diffdock_dir: str,
) -> Iterable[dict[str, Any]]:
    for prediction in predictions:
        drug = drugs_by_id.get(prediction.drug_id, {})
        affinity_component = normalize(prediction.score, score_min, score_max)
        for protein in proteins_by_sequence.get(prediction.sequence_key, []):
            protein_id = protein["protein_id"]
            pair_id = f"{prediction.drug_id}__{protein_id}"
            yield {
                "stage4_rank": "",
                "pair_id": pair_id,
                "drug_id": prediction.drug_id,
                "drug_name": drug.get("drug_name", ""),
                "protein_id": protein_id,
                "gene_name": protein.get("gene_name", ""),
                "protein_name": protein.get("protein_name", ""),
                "diffdock_confidence": "",
                "docking_score": "",
                "docking_component": 0.0,
                "affinity_model": "ConPLex",
                "affinity_score": prediction.score_text,
                "kd_nm": "",
                "affinity_component": affinity_component,
                "combined_ai_score": affinity_component,
                "ligand_sdf_path": drug.get("sdf_path", ""),
                "ligand_smiles": drug.get("canonical_smiles") or drug.get("isomeric_smiles") or drug.get("smiles", ""),
                "receptor_pdb_url": protein.get("alphafold_pdb_url") or alphafold_pdb_url(protein_id, alphafold_version),
                "receptor_cif_url": protein.get("alphafold_cif_url") or alphafold_cif_url(protein_id, alphafold_version),
                "protein_sequence": protein.get("sequence", ""),
                "diffdock_output_dir": f"{diffdock_dir}/{pair_id}",
                "status": "pending",
                "sequence_key": prediction.sequence_key,
            }


def write_full_affinity_scores(
    path: Path,
    predictions: list[Prediction],
    proteins_by_sequence: dict[str, list[dict[str, str]]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AFFINITY_FIELDS)
        writer.writeheader()
        for prediction in predictions:
            for protein in proteins_by_sequence.get(prediction.sequence_key, []):
                writer.writerow(
                    {
                        "pair_id": f"{prediction.drug_id}__{protein['protein_id']}",
                        "model": "ConPLex",
                        "affinity_score": prediction.score_text,
                        "kd_nm": "",
                    }
                )
                count += 1
    return count


def top_expanded_rows(
    predictions: list[Prediction],
    proteins_by_sequence: dict[str, list[dict[str, str]]],
    drugs_by_id: dict[str, dict[str, str]],
    score_min: float,
    score_max: float,
    top_n: int,
    alphafold_version: int,
    diffdock_dir: str,
) -> list[dict[str, Any]]:
    heap: list[tuple[float, str, dict[str, Any]]] = []
    for row in expanded_rows(
        predictions,
        proteins_by_sequence,
        drugs_by_id,
        score_min,
        score_max,
        alphafold_version,
        diffdock_dir,
    ):
        key = row["pair_id"]
        item = (float(row["combined_ai_score"]), key, row)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)

    rows = [item[2] for item in heap]
    rows.sort(key=lambda row: (float(row["combined_ai_score"]), row["pair_id"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["stage4_rank"] = rank
    return rows


def group_proteins_by_sequence(proteins: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for protein in proteins:
        grouped.setdefault(protein["sequence_key"], []).append(protein)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand unique-sequence ConPLex predictions back to all druggable proteome UniProt IDs.")
    parser.add_argument("--predictions", default="outputs/druggable_proteome/conplex_predictions_druggable_unique_sequences.tsv")
    parser.add_argument("--proteins", default="outputs/druggable_proteome/protein_library_druggable_chembl.csv")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--out-affinity", default="outputs/druggable_proteome/conplex_affinity_scores_druggable.csv")
    parser.add_argument("--out-top", default="outputs/druggable_proteome/stage4_affinity_candidates_druggable_top10000.csv")
    parser.add_argument("--out-diffdock-seed", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest.csv")
    parser.add_argument("--metadata", default="outputs/druggable_proteome/druggable_conplex_expansion.metadata.json")
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--alphafold-version", type=int, default=6)
    parser.add_argument("--diffdock-dir", default="outputs/druggable_proteome/diffdock_runs")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    proteins = read_csv(Path(args.proteins))
    drugs = read_csv(Path(args.drugs))
    predictions, skipped_prediction_rows = read_predictions(predictions_path)
    if not predictions:
        raise ValueError(f"No usable ConPLex predictions found in {predictions_path}")

    proteins_by_sequence = group_proteins_by_sequence(proteins)
    drugs_by_id = {row["drug_id"]: row for row in drugs}
    missing_sequence_predictions = sum(1 for prediction in predictions if prediction.sequence_key not in proteins_by_sequence)
    missing_drug_predictions = sum(1 for prediction in predictions if prediction.drug_id not in drugs_by_id)

    score_min = min(prediction.score for prediction in predictions)
    score_max = max(prediction.score for prediction in predictions)

    affinity_rows_written = write_full_affinity_scores(Path(args.out_affinity), predictions, proteins_by_sequence)
    top_rows = top_expanded_rows(
        predictions,
        proteins_by_sequence,
        drugs_by_id,
        score_min,
        score_max,
        args.top_n,
        args.alphafold_version,
        args.diffdock_dir,
    )
    top_rows_written = write_csv(Path(args.out_top), STAGE4_FIELDS, top_rows)
    diffdock_seed_rows_written = write_csv(Path(args.out_diffdock_seed), DIFFDOCK_SEED_FIELDS, top_rows)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions": str(predictions_path),
        "proteins": args.proteins,
        "drugs": args.drugs,
        "unique_sequence_predictions": len(predictions),
        "skipped_prediction_rows": skipped_prediction_rows,
        "protein_rows": len(proteins),
        "drug_rows": len(drugs),
        "expanded_affinity_rows_written": affinity_rows_written,
        "top_rows_written": top_rows_written,
        "diffdock_seed_rows_written": diffdock_seed_rows_written,
        "missing_sequence_predictions": missing_sequence_predictions,
        "missing_drug_predictions": missing_drug_predictions,
        "score_min": score_min,
        "score_max": score_max,
        "outputs": {
            "affinity_scores": args.out_affinity,
            "top_candidates": args.out_top,
            "diffdock_seed_manifest": args.out_diffdock_seed,
        },
        "notes": [
            "Top candidates are ranked by normalized ConPLex affinity score only.",
            "DiffDock seed manifest contains ligand and AlphaFold receptor references for downstream structure preparation.",
        ],
    }
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

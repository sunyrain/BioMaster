from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TOP_FIELDS = [
    "direction_rank",
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "direction",
    "direction_label",
    "direction_score",
    "affinity_model",
    "affinity_score",
    "affinity_component",
    "drug_direction_score",
    "protein_direction_score",
    "opentargets_direction_score",
    "txgnn_direction_score",
    "direction_evidence_summary",
    "therapeutic_area",
    "indication",
    "target_name",
    "target_chembl_id",
    "disease_icd11_classes",
    "sequence_key",
]

SEED_FIELDS = [
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
    "direction",
    "direction_label",
    "direction_rank",
    "direction_score",
    "drug_direction_score",
    "protein_direction_score",
    "opentargets_direction_score",
    "txgnn_direction_score",
    "direction_evidence_summary",
    "therapeutic_area",
    "indication",
    "target_name",
    "target_chembl_id",
    "disease_icd11_classes",
]


@dataclass(frozen=True)
class Direction:
    name: str
    label: str
    therapeutic_areas: tuple[str, ...]
    drug_terms: tuple[str, ...]
    protein_icd11_terms: tuple[str, ...]
    use_oncology_graph_evidence: bool = False


DIRECTIONS = [
    Direction(
        name="oncology",
        label="Oncology",
        therapeutic_areas=("Oncology", "Hematology"),
        drug_terms=(
            "cancer",
            "neoplasm",
            "tumor",
            "tumour",
            "carcinoma",
            "sarcoma",
            "leukemia",
            "leukaemia",
            "lymphoma",
            "melanoma",
            "myeloma",
            "glioblastoma",
            "metastatic",
        ),
        protein_icd11_terms=("02 肿瘤", "neoplasm", "cancer"),
        use_oncology_graph_evidence=True,
    ),
    Direction(
        name="infectious_disease",
        label="Infectious Disease",
        therapeutic_areas=("Infectious Disease",),
        drug_terms=(
            "infection",
            "infectious",
            "bacterial",
            "antibacterial",
            "viral",
            "antiviral",
            "virus",
            "hiv",
            "hepatitis",
            "influenza",
            "covid",
            "sars-cov",
            "pneumonia",
            "tuberculosis",
            "candidiasis",
            "fungal",
            "antifungal",
            "parasite",
            "malaria",
        ),
        protein_icd11_terms=("01 感染性或寄生虫病", "infectious", "infection"),
    ),
    Direction(
        name="cardiovascular",
        label="Cardiovascular",
        therapeutic_areas=("Cardiovascular",),
        drug_terms=(
            "hypertension",
            "heart failure",
            "atrial fibrillation",
            "angina",
            "cardiovascular",
            "myocardial",
            "coronary",
            "thrombosis",
            "embolism",
            "stroke",
            "hyperlipidemia",
            "dyslipidemia",
        ),
        protein_icd11_terms=("11 循环系统疾病", "cardiovascular", "circulatory"),
    ),
    Direction(
        name="neurology_psychiatry",
        label="Neurology/Psychiatry",
        therapeutic_areas=("Neurology/Psychiatry", "Anesthesia/Pain"),
        drug_terms=(
            "epilepsy",
            "seizure",
            "parkinson",
            "alzheimer",
            "dementia",
            "migraine",
            "neuropathic",
            "multiple sclerosis",
            "depression",
            "depressive",
            "psychosis",
            "psychotic",
            "schizophrenia",
            "bipolar",
            "anxiety",
            "adhd",
            "attention deficit",
            "pain",
        ),
        protein_icd11_terms=("08 神经系统疾病", "neurology", "nervous"),
    ),
    Direction(
        name="immunology_inflammation",
        label="Immunology/Inflammation",
        therapeutic_areas=("Immunology/Inflammation", "Dermatology", "Musculoskeletal"),
        drug_terms=(
            "arthritis",
            "rheumatoid",
            "psoriasis",
            "psoriatic",
            "dermatitis",
            "eczema",
            "inflammation",
            "inflammatory",
            "autoimmune",
            "lupus",
            "crohn",
            "ulcerative colitis",
            "asthma",
            "atopic",
        ),
        protein_icd11_terms=(
            "15 肌肉骨骼系统或结缔组织疾病",
            "immune",
            "immunology",
            "inflammation",
            "inflammatory",
        ),
    ),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def score_drugs(drugs: pd.DataFrame, direction: Direction) -> dict[str, float]:
    scores: dict[str, float] = {}
    area_set = {area.lower() for area in direction.therapeutic_areas}
    for row in drugs.itertuples(index=False):
        drug_id = normalise_text(getattr(row, "drug_id"))
        area = normalise_text(getattr(row, "therapeutic_area", "")).lower()
        indication = normalise_text(getattr(row, "indication", ""))
        target_name = normalise_text(getattr(row, "target_name", ""))
        mechanism = normalise_text(getattr(row, "mechanism_of_action", ""))
        score = 0.0
        if area and area in area_set:
            score = max(score, 1.0)
        haystack = " ".join([indication, target_name, mechanism])
        if contains_any(haystack, direction.drug_terms):
            score = max(score, 0.85)
        scores[drug_id] = score
    return scores


def score_proteins(proteins: pd.DataFrame, direction: Direction) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in proteins.itertuples(index=False):
        protein_id = normalise_text(getattr(row, "protein_id"))
        disease_classes = normalise_text(getattr(row, "disease_icd11_classes", ""))
        function = normalise_text(getattr(row, "function_description", ""))
        target_class = normalise_text(getattr(row, "target_class", ""))
        score = 0.0
        if contains_any(disease_classes, direction.protein_icd11_terms):
            score = max(score, 1.0)
        if contains_any(" ".join([function, target_class]), direction.drug_terms):
            score = max(score, 0.5)
        scores[protein_id] = score
    return scores


def load_oncology_opentargets(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=lambda col: col in {"protein_id", "disease_name", "overall_score"})
    if df.empty:
        return {}
    df["disease_name"] = df["disease_name"].fillna("").astype(str).str.lower()
    df = df[df["disease_name"].isin({"cancer", "neoplasm"})]
    if df.empty:
        return {}
    scores = df.groupby("protein_id")["overall_score"].max()
    max_score = float(scores.max()) if len(scores) else 0.0
    if max_score > 0:
        scores = scores / max_score
    return {str(key): float(value) for key, value in scores.items()}


def load_oncology_txgnn(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=lambda col: col in {"drug_id", "disease_name", "txgnn_indication_score"})
    if df.empty:
        return {}
    df["disease_name"] = df["disease_name"].fillna("").astype(str).str.lower()
    df = df[df["disease_name"].eq("cancer")]
    if df.empty:
        return {}
    scores = df.groupby("drug_id")["txgnn_indication_score"].max()
    return {str(key): float(value) for key, value in scores.items()}


def local_receptor_paths(protein_id: str, receptor_dir: Path) -> tuple[str, str]:
    pdb_path = receptor_dir / f"AF-{protein_id}-F1-model_v6.pdb"
    cif_path = receptor_dir / f"AF-{protein_id}-F1-model_v6.cif"
    return (
        str(pdb_path) if pdb_path.exists() and pdb_path.stat().st_size > 0 else "",
        str(cif_path) if cif_path.exists() and cif_path.stat().st_size > 0 else "",
    )


def alphafold_pdb_url(protein_id: str, version: int) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v{version}.pdb"


def alphafold_cif_url(protein_id: str, version: int) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v{version}.cif"


def evidence_summary(
    drug_score: float,
    protein_score: float,
    opentargets_score: float,
    txgnn_score: float,
    direction: Direction,
) -> str:
    evidence: list[str] = []
    if drug_score >= 1.0:
        evidence.append("FDA therapeutic area match")
    elif drug_score > 0:
        evidence.append("FDA indication/target text match")
    if protein_score >= 1.0:
        evidence.append("protein ICD-11 disease-class match")
    elif protein_score > 0:
        evidence.append("protein functional text match")
    if direction.use_oncology_graph_evidence and opentargets_score > 0:
        evidence.append("Open Targets cancer/neoplasm association")
    if direction.use_oncology_graph_evidence and txgnn_score > 0:
        evidence.append("TxGNN cancer drug-disease signal")
    return "; ".join(evidence) if evidence else "affinity-driven candidate"


def build_output_rows(
    selected: pd.DataFrame,
    direction: Direction,
    drugs_by_id: dict[str, dict[str, str]],
    proteins_by_id: dict[str, dict[str, str]],
    receptor_dir: Path,
    alphafold_version: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        drug_id = str(row.drug_id)
        protein_id = str(row.protein_id)
        pair_id = f"{drug_id}__{protein_id}"
        drug = drugs_by_id.get(drug_id, {})
        protein = proteins_by_id.get(protein_id, {})
        ligand_smiles = drug.get("canonical_smiles") or drug.get("isomeric_smiles") or drug.get("smiles", "")
        receptor_pdb_path, receptor_cif_path = local_receptor_paths(protein_id, receptor_dir)
        receptor_pdb = receptor_pdb_path or protein.get("alphafold_pdb_url") or alphafold_pdb_url(protein_id, alphafold_version)
        receptor_cif = receptor_cif_path or protein.get("alphafold_cif_url") or alphafold_cif_url(protein_id, alphafold_version)
        evidence = evidence_summary(
            float(row.drug_direction_score),
            float(row.protein_direction_score),
            float(row.opentargets_direction_score),
            float(row.txgnn_direction_score),
            direction,
        )
        common = {
            "direction_rank": rank,
            "pair_id": pair_id,
            "drug_id": drug_id,
            "drug_name": drug.get("drug_name", ""),
            "protein_id": protein_id,
            "gene_name": protein.get("gene_name", ""),
            "protein_name": protein.get("protein_name", ""),
            "direction": direction.name,
            "direction_label": direction.label,
            "direction_score": round(float(row.direction_score), 6),
            "affinity_model": "ConPLex",
            "affinity_score": float(row.affinity_score),
            "affinity_component": round(float(row.affinity_component), 6),
            "drug_direction_score": round(float(row.drug_direction_score), 6),
            "protein_direction_score": round(float(row.protein_direction_score), 6),
            "opentargets_direction_score": round(float(row.opentargets_direction_score), 6),
            "txgnn_direction_score": round(float(row.txgnn_direction_score), 6),
            "direction_evidence_summary": evidence,
            "therapeutic_area": drug.get("therapeutic_area", ""),
            "indication": drug.get("indication", ""),
            "target_name": drug.get("target_name", ""),
            "target_chembl_id": drug.get("target_chembl_id", ""),
            "disease_icd11_classes": protein.get("disease_icd11_classes", ""),
            "sequence_key": protein.get("sequence_key", ""),
        }
        top_rows.append(common)
        seed_rows.append(
            {
                **common,
                "ligand_sdf_path": drug.get("sdf_path", ""),
                "ligand_smiles": ligand_smiles,
                "receptor_pdb_url": receptor_pdb,
                "receptor_cif_url": receptor_cif,
                "protein_sequence": protein.get("sequence", ""),
                "diffdock_output_dir": str(out_dir / direction.name / "diffdock_outputs" / pair_id),
                "status": "pending",
                "combined_ai_score": round(float(row.direction_score), 6),
            }
        )
    return top_rows, seed_rows


def write_summary(
    path: Path,
    direction: Direction,
    top_rows: list[dict[str, Any]],
    selected: pd.DataFrame,
    score_formula: str,
) -> dict[str, Any]:
    drug_evidence = sum(1 for row in top_rows if float(row["drug_direction_score"]) > 0)
    protein_evidence = sum(1 for row in top_rows if float(row["protein_direction_score"]) > 0)
    ot_evidence = sum(1 for row in top_rows if float(row["opentargets_direction_score"]) > 0)
    txgnn_evidence = sum(1 for row in top_rows if float(row["txgnn_direction_score"]) > 0)
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "direction": direction.name,
        "direction_label": direction.label,
        "top_rows": len(top_rows),
        "unique_drugs": len({row["drug_id"] for row in top_rows}),
        "unique_proteins": len({row["protein_id"] for row in top_rows}),
        "unique_sequences": len({row["sequence_key"] for row in top_rows if row.get("sequence_key")}),
        "rows_with_drug_direction_evidence": drug_evidence,
        "rows_with_protein_direction_evidence": protein_evidence,
        "rows_with_opentargets_evidence": ot_evidence,
        "rows_with_txgnn_evidence": txgnn_evidence,
        "score_formula": score_formula,
        "score_range": {
            "min": round(float(selected["direction_score"].min()), 6) if len(selected) else None,
            "max": round(float(selected["direction_score"].max()), 6) if len(selected) else None,
        },
        "affinity_score_range": {
            "min": round(float(selected["affinity_score"].min()), 6) if len(selected) else None,
            "max": round(float(selected["affinity_score"].max()), 6) if len(selected) else None,
        },
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_dir = Path(args.receptor_dir)

    drugs = pd.read_csv(
        args.drugs,
        dtype=str,
        keep_default_na=False,
        usecols=lambda col: col
        in {
            "drug_id",
            "drug_name",
            "canonical_smiles",
            "isomeric_smiles",
            "sdf_path",
            "therapeutic_area",
            "indication",
            "target_name",
            "target_chembl_id",
            "mechanism_of_action",
        },
    )
    proteins = pd.read_csv(
        args.proteins,
        dtype=str,
        keep_default_na=False,
        usecols=lambda col: col
        in {
            "protein_id",
            "gene_name",
            "protein_name",
            "sequence",
            "sequence_key",
            "alphafold_pdb_url",
            "alphafold_cif_url",
            "function_description",
            "target_class",
            "disease_icd11_classes",
        },
    )
    affinity = pd.read_csv(args.affinity, usecols=["pair_id", "affinity_score"])
    pair_parts = affinity["pair_id"].str.rsplit("__", n=1, expand=True)
    affinity["drug_id"] = pair_parts[0]
    affinity["protein_id"] = pair_parts[1]
    affinity["affinity_score"] = pd.to_numeric(affinity["affinity_score"], errors="coerce").fillna(0.0)
    score_min = float(affinity["affinity_score"].min())
    score_max = float(affinity["affinity_score"].max())
    if score_max == score_min:
        affinity["affinity_component"] = 1.0
    else:
        affinity["affinity_component"] = (affinity["affinity_score"] - score_min) / (score_max - score_min)

    drugs_by_id = {str(row["drug_id"]): row for row in drugs.to_dict("records")}
    proteins_by_id = {str(row["protein_id"]): row for row in proteins.to_dict("records")}
    oncology_opentargets = load_oncology_opentargets(Path(args.opentargets))
    oncology_txgnn = load_oncology_txgnn(Path(args.txgnn))

    score_formula = (
        "non-oncology: 0.70*affinity_component + 0.15*drug_direction_score + "
        "0.15*protein_direction_score; oncology: 0.60*affinity_component + "
        "0.15*drug_direction_score + 0.15*protein_direction_score + "
        "0.07*OpenTargets + 0.03*TxGNN"
    )

    summaries: dict[str, Any] = {}
    for direction in DIRECTIONS:
        direction_dir = out_dir / direction.name
        direction_dir.mkdir(parents=True, exist_ok=True)
        drug_scores = score_drugs(drugs, direction)
        protein_scores = score_proteins(proteins, direction)
        work = affinity[
            ["pair_id", "drug_id", "protein_id", "affinity_score", "affinity_component"]
        ].copy()
        work["drug_direction_score"] = work["drug_id"].map(drug_scores).fillna(0.0).astype(float)
        work["protein_direction_score"] = work["protein_id"].map(protein_scores).fillna(0.0).astype(float)
        if direction.use_oncology_graph_evidence:
            work["opentargets_direction_score"] = work["protein_id"].map(oncology_opentargets).fillna(0.0).astype(float)
            work["txgnn_direction_score"] = work["drug_id"].map(oncology_txgnn).fillna(0.0).astype(float)
            work["direction_score"] = (
                (0.60 * work["affinity_component"])
                + (0.15 * work["drug_direction_score"])
                + (0.15 * work["protein_direction_score"])
                + (0.07 * work["opentargets_direction_score"])
                + (0.03 * work["txgnn_direction_score"])
            )
        else:
            work["opentargets_direction_score"] = 0.0
            work["txgnn_direction_score"] = 0.0
            work["direction_score"] = (
                (0.70 * work["affinity_component"])
                + (0.15 * work["drug_direction_score"])
                + (0.15 * work["protein_direction_score"])
            )

        selected = (
            work.nlargest(args.top_n, ["direction_score", "affinity_score"])
            .sort_values(["direction_score", "affinity_score", "pair_id"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        top_rows, seed_rows = build_output_rows(
            selected=selected,
            direction=direction,
            drugs_by_id=drugs_by_id,
            proteins_by_id=proteins_by_id,
            receptor_dir=receptor_dir,
            alphafold_version=args.alphafold_version,
            out_dir=out_dir,
        )
        write_csv(direction_dir / f"top{args.top_n}_candidates.csv", TOP_FIELDS, top_rows)
        write_csv(direction_dir / f"top{args.top_n}_diffdock_seed_manifest.csv", SEED_FIELDS, seed_rows)
        summary = write_summary(
            path=direction_dir / "summary.json",
            direction=direction,
            top_rows=top_rows,
            selected=selected,
            score_formula=score_formula,
        )
        summaries[direction.name] = summary

    run_summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "affinity": args.affinity,
        "drugs": args.drugs,
        "proteins": args.proteins,
        "top_n": args.top_n,
        "directions": list(summaries),
        "affinity_score_min": score_min,
        "affinity_score_max": score_max,
        "score_formula": score_formula,
        "direction_summaries": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build disease-direction Top-N drug-target candidates and DiffDock seed manifests.")
    parser.add_argument("--affinity", default="outputs/druggable_proteome/conplex_affinity_scores_druggable.csv")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--proteins", default="outputs/druggable_proteome/protein_library_druggable_chembl.csv")
    parser.add_argument("--opentargets", default="data/processed/opentargets_target_disease_scores.csv")
    parser.add_argument("--txgnn", default="data/processed/txgnn_drug_disease_scores.csv")
    parser.add_argument("--out-dir", default="outputs/disease_directions")
    parser.add_argument("--receptor-dir", default="data/processed/alphafold_receptors_v6")
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--alphafold-version", type=int, default=6)
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

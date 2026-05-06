from __future__ import annotations

import csv
from pathlib import Path

from biomaster.pipeline import (
    build_drug_library,
    convert_conplex_predictions,
    make_conplex_input,
    make_screening_manifest,
    merge_affinity_scores,
    rank_disease_relevance,
)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_drug_library_filters_approval_year_without_network(tmp_path: Path) -> None:
    seed = tmp_path / "drug_seed.csv"
    out = tmp_path / "drug_library.csv"
    write_rows(
        seed,
        ["drug_name", "approval_year", "drugbank_id", "pubchem_cid"],
        [
            {"drug_name": "NewDrug", "approval_year": 2021, "drugbank_id": "DBNEW", "pubchem_cid": "1"},
            {"drug_name": "OldDrug", "approval_year": 2010, "drugbank_id": "DBOLD", "pubchem_cid": "2"},
        ],
    )

    rows = build_drug_library(seed, out, year_min=2016, fetch_pubchem=False)

    assert [row["drug_id"] for row in rows] == ["DBNEW"]
    assert read_rows(out)[0]["drug_name"] == "NewDrug"


def test_build_drug_library_makes_duplicate_source_ids_unique(tmp_path: Path) -> None:
    seed = tmp_path / "drug_seed.csv"
    out = tmp_path / "drug_library.csv"
    write_rows(
        seed,
        ["drug_name", "approval_year", "chembl_id"],
        [
            {"drug_name": "Drug A", "approval_year": 2024, "chembl_id": "CHEMBL1"},
            {"drug_name": "Drug B", "approval_year": 2025, "chembl_id": "CHEMBL1"},
        ],
    )

    rows = build_drug_library(seed, out, fetch_pubchem=False)

    assert [row["drug_id"] for row in rows] == ["CHEMBL1", "CHEMBL1__DRUG_B"]
    assert [row["chembl_id"] for row in rows] == ["CHEMBL1", "CHEMBL1"]


def test_build_drug_library_accepts_fda_xlsx_without_network(tmp_path: Path) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        return

    seed = tmp_path / "fda_small_molecules.xlsx"
    out = tmp_path / "drug_library.csv"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "Brand Name",
            "Generic Name (INN)",
            "Approval Year",
            "Route",
            "Therapeutic Area",
            "Molecular Weight (Da)",
            "SMILES",
            "ChEMBL ID",
            "Target Name",
        ]
    )
    sheet.append(["BrandX", "GenericX", 2024, "Oral", "Oncology", 321.1, "CCO", "CHEMBL123", "TargetX"])
    workbook.save(seed)

    rows = build_drug_library(seed, out, fetch_pubchem=False)

    assert rows[0]["drug_id"] == "CHEMBL123"
    assert rows[0]["drug_name"] == "GenericX"
    assert rows[0]["brand_name"] == "BrandX"
    assert rows[0]["canonical_smiles"] == "CCO"
    assert read_rows(out)[0]["therapeutic_area"] == "Oncology"


def test_make_screening_manifest_cross_joins_drugs_and_proteins(tmp_path: Path) -> None:
    drugs = tmp_path / "drugs.csv"
    proteins = tmp_path / "proteins.csv"
    out = tmp_path / "manifest.csv"
    write_rows(
        drugs,
        ["drug_id", "drug_name", "sdf_path", "canonical_smiles"],
        [{"drug_id": "D1", "drug_name": "Drug 1", "sdf_path": "D1.sdf", "canonical_smiles": "CCO"}],
    )
    write_rows(
        proteins,
        ["protein_id", "gene_name", "protein_name", "sequence", "alphafold_pdb_url", "alphafold_cif_url"],
        [
            {"protein_id": "P1", "gene_name": "G1", "protein_name": "Protein 1", "sequence": "MA", "alphafold_pdb_url": "p1.pdb", "alphafold_cif_url": "p1.cif"},
            {"protein_id": "P2", "gene_name": "G2", "protein_name": "Protein 2", "sequence": "MT", "alphafold_pdb_url": "p2.pdb", "alphafold_cif_url": "p2.cif"},
        ],
    )

    rows = make_screening_manifest(drugs, proteins, out, output_prefix="runs")

    assert [row["pair_id"] for row in rows] == ["D1__P1", "D1__P2"]
    assert rows[1]["diffdock_output_dir"] == "runs/D1__P2"
    assert rows[1]["ligand_smiles"] == "CCO"
    assert rows[1]["protein_sequence"] == "MT"


def test_make_conplex_input_skips_pairs_missing_smiles_or_sequence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    out = tmp_path / "conplex_input.tsv"
    write_rows(
        manifest,
        ["drug_id", "protein_id", "ligand_smiles", "protein_sequence"],
        [
            {"drug_id": "D1", "protein_id": "P1", "ligand_smiles": "CCO", "protein_sequence": "MA"},
            {"drug_id": "D2", "protein_id": "P2", "ligand_smiles": "", "protein_sequence": "MT"},
        ],
    )

    rows = make_conplex_input(manifest, out)

    assert rows == [{"protein_id": "P1", "drug_id": "D1", "protein_sequence": "MA", "ligand_smiles": "CCO"}]
    assert out.read_text(encoding="utf-8") == "P1\tD1\tMA\tCCO\n"


def test_convert_conplex_predictions_writes_affinity_scores(tmp_path: Path) -> None:
    predictions = tmp_path / "conplex_results.tsv"
    out = tmp_path / "affinity_scores.csv"
    predictions.write_text("D1\tP1\t0.91\nD2\tP2\t0.42\n", encoding="utf-8")

    rows = convert_conplex_predictions(predictions, out)

    assert rows[0]["pair_id"] == "D1__P1"
    assert rows[0]["model"] == "ConPLex"
    assert read_rows(out)[1]["affinity_score"] == "0.42"


def test_merge_affinity_scores_ranks_combined_components(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    docking = tmp_path / "docking.csv"
    affinity = tmp_path / "affinity.csv"
    out = tmp_path / "stage4.csv"
    write_rows(
        manifest,
        ["pair_id", "drug_id", "drug_name", "protein_id", "gene_name", "protein_name"],
        [
            {"pair_id": "D1__P1", "drug_id": "D1", "drug_name": "Drug 1", "protein_id": "P1", "gene_name": "G1", "protein_name": "Protein 1"},
            {"pair_id": "D2__P2", "drug_id": "D2", "drug_name": "Drug 2", "protein_id": "P2", "gene_name": "G2", "protein_name": "Protein 2"},
        ],
    )
    write_rows(
        docking,
        ["pair_id", "diffdock_confidence", "docking_score"],
        [
            {"pair_id": "D1__P1", "diffdock_confidence": 0.2, "docking_score": -6.0},
            {"pair_id": "D2__P2", "diffdock_confidence": 0.9, "docking_score": -9.0},
        ],
    )
    write_rows(
        affinity,
        ["pair_id", "model", "affinity_score", "kd_nm"],
        [
            {"pair_id": "D1__P1", "model": "DeepDTA", "affinity_score": 0.1, "kd_nm": ""},
            {"pair_id": "D2__P2", "model": "DeepDTA", "affinity_score": 0.8, "kd_nm": ""},
        ],
    )

    rows = merge_affinity_scores(manifest, docking, affinity, out)

    assert rows[0]["pair_id"] == "D2__P2"
    assert rows[0]["stage4_rank"] == 1
    assert float(read_rows(out)[0]["combined_ai_score"]) > float(read_rows(out)[1]["combined_ai_score"])


def test_rank_disease_relevance_uses_direct_and_string_network_scores(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    string = tmp_path / "string.csv"
    disgenet = tmp_path / "disgenet.csv"
    out = tmp_path / "ranked.csv"
    write_rows(
        candidates,
        ["pair_id", "drug_id", "drug_name", "protein_id", "gene_name", "protein_name", "combined_ai_score"],
        [
            {"pair_id": "D1__P1", "drug_id": "D1", "drug_name": "Drug 1", "protein_id": "P1", "gene_name": "TARGET", "protein_name": "Target", "combined_ai_score": 0.5},
            {"pair_id": "D2__P2", "drug_id": "D2", "drug_name": "Drug 2", "protein_id": "P2", "gene_name": "DISEASE", "protein_name": "Disease", "combined_ai_score": 0.4},
        ],
    )
    write_rows(
        string,
        ["protein1", "protein2", "string_score"],
        [{"protein1": "TARGET", "protein2": "DISEASE", "string_score": 900}],
    )
    write_rows(
        disgenet,
        ["gene_name", "disease_id", "disease_name", "disgenet_score"],
        [{"gene_name": "DISEASE", "disease_id": "DX", "disease_name": "Disease X", "disgenet_score": 0.8}],
    )

    rows = rank_disease_relevance(candidates, string, disgenet, out, disease_id="DX")

    target = next(row for row in rows if row["gene_name"] == "TARGET")
    assert target["network_disease_score"] == 0.72
    assert read_rows(out)[0]["stage5_rank"] == "1"

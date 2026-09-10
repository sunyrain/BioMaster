"""Scientific identity and evidence boundary tests for explorer annotations."""
import csv
import gzip
import json
import sqlite3
import zipfile

from biomaster.explorer_annotations import (
    CHEMBL_PATH, ENTITY_PATH, GTEX_PATH, HPA_PATH, OT_DISEASE_PATHS, SPECIES_PATH, TX_PATH, load_annotations,
)


def write_csv(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_partial_data_preserves_identity_phase_and_snapshot_precedence(tmp_path):
    drugs = {"PARENT": {"name": "Reviewed Drug"}, "ACTIVE": {"name": "Reviewed Drug"},
             "UNMATCHED": {"name": "Reviewed Drug Hydrochloride"}}
    targets = {"CHEMBL_TARGET": {"name": "GENE", "identifiers": {"uniprot_id": "P12345"}}}
    write_csv(tmp_path, SPECIES_PATH, [
        {"ligand_inchikey": "PARENT", "project_entity_id": "UNII:1", "drug_name": "Reviewed Drug",
         "ligand_species_role": "ADMINISTERED_OR_APPROVED_ACTIVE_MOIETY", "ligand_species_name": "", "ligand_species_chembl_id": "CHEMBL_PARENT"},
        {"ligand_inchikey": "ACTIVE", "project_entity_id": "UNII:1", "drug_name": "Reviewed Drug",
         "ligand_species_role": "CH_EMBL_ANNOTATED_ACTIVE_SPECIES", "ligand_species_name": "Active metabolite", "ligand_species_chembl_id": "CHEMBL_ACTIVE"},
    ])
    write_csv(tmp_path, ENTITY_PATH, [{"project_entity_id": "UNII:1", "model_inchikey": "PARENT",
        "base_chembl_ids": "CHEMBL_PARENT", "gsrs_molecular_formula": "C2H6O", "molecular_weight": "46.07"}])
    write_csv(tmp_path, TX_PATH, [{"txgnn_drug_name": "Reviewed Drug", "txgnn_drugbank_id": "DB1",
        "disease_id": "MONDO_0004992", "disease_name": "cancer", "txgnn_indication_score": "0.02",
        "txgnn_indication_logit": "-3.89", "relation": "indication"}])
    association = {"approved_symbol": "GENE", "disease_id": "EFO_1", "disease_name": "Example disease",
                   "overall_score": "0.4", "snapshot": "new", "datatype_scores_json": '{"genetic_association": 0.5}'}
    write_csv(tmp_path, OT_DISEASE_PATHS[0], [association])
    write_csv(tmp_path, OT_DISEASE_PATHS[1], [{**association, "overall_score": "0.9", "snapshot": "old"}])

    database = tmp_path / CHEMBL_PATH
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:
        conn.executescript("""
            CREATE TABLE molecule_dictionary (molregno INTEGER, chembl_id TEXT);
            CREATE TABLE drug_indication (molregno INTEGER, max_phase_for_ind REAL, mesh_id TEXT,
                mesh_heading TEXT, efo_id TEXT, efo_term TEXT);
            CREATE TABLE drug_mechanism (molregno INTEGER, tid INTEGER, action_type TEXT,
                mechanism_of_action TEXT, direct_interaction INTEGER);
            CREATE TABLE target_dictionary (tid INTEGER, chembl_id TEXT, pref_name TEXT, organism TEXT);
            INSERT INTO molecule_dictionary VALUES (1, 'CHEMBL_PARENT');
            INSERT INTO drug_indication VALUES (1, 4, 'M1', 'Known disease', 'EFO:KNOWN', 'Known disease');
            INSERT INTO drug_indication VALUES (1, 2, 'M2', 'Investigational', 'EFO:TRIAL', 'Investigational');
            INSERT INTO drug_indication VALUES (1, 2, 'M1', 'Known disease', 'EFO:KNOWN', 'Known disease');
            INSERT INTO target_dictionary VALUES (2, 'CHEMBL_TARGET', 'Target name', 'Homo sapiens');
            INSERT INTO drug_mechanism VALUES (1, 2, 'INHIBITOR', 'Known inhibitor', 1);
        """)
    result = load_annotations(tmp_path, drugs, targets)
    parent, active = result["drugs"]["PARENT"], result["drugs"]["ACTIVE"]
    assert [(item["id"], item["phase"]) for item in parent["known_diseases"]] == [("EFO_KNOWN", 4), ("EFO_TRIAL", 2)]
    assert active["properties"]["species_role"] == "CH_EMBL_ANNOTATED_ACTIVE_SPECIES"
    assert active["identifiers"]["chembl_id"] == "CHEMBL_ACTIVE"
    assert "molecular_weight" not in active["properties"]  # Parent chemistry is not assigned to the metabolite.
    assert parent["properties"]["molecular_weight"] == 46.07
    assert active["txgnn_diseases"][0]["score"] == 0.02
    assert not result["drugs"]["UNMATCHED"]["txgnn_diseases"]  # No fuzzy/salt stripping.
    assert parent["known_targets"][0]["direct_interaction"] is True
    diseases = result["targets"]["CHEMBL_TARGET"]["target_diseases"]
    assert len(diseases) == 1 and diseases[0]["score"] == 0.4
    assert result["counts"]["target_disease_associations"] == 1
    json.dumps(result, allow_nan=False)


def test_missing_sources_remain_explicitly_empty(tmp_path):
    result = load_annotations(tmp_path, {"D": {"name": "D"}}, {"T": {"name": "T"}})
    assert not result["drugs"]["D"]["known_diseases"]
    assert not result["targets"]["T"]["pathways"]
    assert all(not item["available"] for item in result["sources"])
    assert all(value == 0 for value in result["counts"].values())


def test_tissue_expression_keeps_units_zeros_and_exact_ensembl_mapping(tmp_path):
    gtex = tmp_path / GTEX_PATH
    gtex.parent.mkdir(parents=True)
    with gzip.open(gtex, "wt") as handle:
        handle.write("#1.2\n2\t2\nName\tDescription\tLiver\tBrain\nENSG1.12\tGENE\t2.5\t0\nENSG2.1\tGENE\t999\t999\n")
    hpa = tmp_path / HPA_PATH
    hpa.parent.mkdir(parents=True)
    with zipfile.ZipFile(hpa, "w") as archive:
        archive.writestr("rna_tissue_consensus.tsv", "Gene\tGene name\tTissue\tnTPM\nENSG1\tGENE\tliver\t7.5\n")
    result = load_annotations(tmp_path, {}, {"T": {"name": "GENE", "identifiers": {"ensembl_id": "ENSG1"}}})
    expression = result["targets"]["T"]["properties"]["tissue_expression"]
    assert [(item["name"], item["value"], item["unit"]) for item in expression] == [
        ("Liver", 2.5, "TPM"), ("Brain", 0, "TPM"), ("liver", 7.5, "nTPM")]
    assert result["counts"]["tissue_expression_records"] == 3

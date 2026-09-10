import csv
import json

import pytest

from biomaster.explorer_experiments import CONTROL_PATH, MASTER_PATH, ROSTER_PATH, load_experiments


def write_csv(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_compound_mapping_keeps_off_catalog_controls_and_release_status(tmp_path):
    drugs = {"EXACT-INCHIKEY": {"name": "Same name"}, "OTHER-STEREO": {"name": "Control name"}}
    targets = {"CHEMBL1": {"name": "GENE"}}
    write_csv(tmp_path, ROSTER_PATH, [{"target_chembl_id": "CHEMBL1", "construct_recommendation": "Validated domain required",
        "construct_status": "CONFIRM_ACTIVITY_REQUIRED", "reference_control": "Control name", "release_status": "NOT_RELEASED_FOR_EXPERIMENT"}])
    common = {"target_chembl_id": "CHEMBL1", "gene_symbol": "GENE", "release_status": "DESIGN_REVIEW_NOT_RELEASED_FOR_EXPERIMENT"}
    write_csv(tmp_path, MASTER_PATH, [
        {**common, "experiment_id": "SPR64D-001", "pair_id": "EXACT-INCHIKEY__CHEMBL1", "ligand_inchikey": "EXACT-INCHIKEY",
         "drug_names": "Same name", "selection_role": "OUR_FROZEN_MODEL_HIGH", "retargetmap_rank_384": "4.0"},
        {**common, "experiment_id": "SPR64D-002", "pair_id": "CONTROL::OFF-CATALOG__CHEMBL1", "ligand_inchikey": "OFF-CATALOG",
         "drug_names": "Control name", "selection_role": "POSITIVE_CONTROL", "row_type": "REFERENCE_CONTROL"},
    ])
    write_csv(tmp_path, CONTROL_PATH, [{"pair_id": "CONTROL::OFF-CATALOG__CHEMBL1", "control_chembl_id": "CHEMBL99",
        "control_standard_types": "Ki", "control_mean_pchembl_mixed_endpoints": "8.0", "control_assay_ids": "123",
        "control_doc_ids": "456", "control_status": "LOCAL_CONFLICT_EXTERNAL_REFERENCE_REQUIRES_SCOUT"}])
    result = load_experiments(tmp_path, drugs, targets)
    assert len(result["targets"]["CHEMBL1"]["experiments"]) == 2
    assert len(result["drugs"]["EXACT-INCHIKEY"]["experiments"]) == 1
    assert result["drugs"]["OTHER-STEREO"]["experiments"] == []  # Equal names do not establish identity.
    control = result["targets"]["CHEMBL1"]["experiments"][1]
    assert control["drug_in_catalog"] is False
    assert control["control_evidence"]["standard_types"] == "Ki"
    assert control["control_evidence"]["assay_ids"] == "123"
    assert control["control_status"] == "LOCAL_CONFLICT_EXTERNAL_REFERENCE_REQUIRES_SCOUT"
    assert control["result_status"] == "NO_EXPERIMENTAL_RESULT_IN_DESIGN_ARTIFACT"
    assert result["counts"]["spr_not_released_pairs"] == 2
    assert result["counts"]["spr_off_catalog_compounds"] == 1
    json.dumps(result, allow_nan=False)


def test_missing_source_does_not_manufacture_spr_records(tmp_path):
    result = load_experiments(tmp_path, {"D": {}}, {"T": {}})
    assert result["drugs"]["D"]["experiments"] == []
    assert not any(result["counts"].values())
    assert all(not source["available"] for source in result["sources"])


def test_duplicate_experiment_ids_are_rejected(tmp_path):
    row = {"experiment_id": "SPR-1", "ligand_inchikey": "D", "target_chembl_id": "T"}
    write_csv(tmp_path, MASTER_PATH, [row, row])
    with pytest.raises(ValueError, match="unique"):
        load_experiments(tmp_path, {"D": {}}, {"T": {}})

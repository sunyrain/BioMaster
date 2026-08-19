from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_contract_module():
    path = Path(__file__).resolve().parents[1] / "scripts/build_biomaster_odti_v3_data_contract_v1.py"
    spec = importlib.util.spec_from_file_location("biomaster_v3_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_contract_preserves_inactive_and_missing_affinity(tmp_path: Path) -> None:
    module = load_contract_module()
    rows = []
    for index, label in enumerate(["positive", "negative_or_inactive"]):
        rows.append(
            {
                "calibration_pair_id": f"PAIR_{index}",
                "parent_molecule_chembl_id": f"CHEMBL_MOL_{index}",
                "target_chembl_id": "CHEMBL_TARGET_1",
                "calibration_label": label,
                "binary_label": int(label == "positive"),
                "any_explicit_inactive": int(label != "positive"),
                "explicit_inactive_positive_conflict": False,
                "numeric_positive_negative_conflict": False,
                "min_pchembl": 7.0 if index == 0 else float("nan"),
                "max_pchembl": 7.0 if index == 0 else float("nan"),
                "mean_pchembl": 7.0 if index == 0 else float("nan"),
                "numeric_rows": 1 if index == 0 else 0,
                "activity_rows": 1,
                "assay_count": 1,
                "document_count": 1,
                "standard_types": "IC50",
                "relationship_types": "D",
                "assay_ids": "ASSAY_1",
                "doc_ids": "DOC_1",
                "min_document_year": 2020,
                "max_document_year": 2020,
                "target_assay_family": "kinase",
                "scaffold_group": f"SCAFFOLD_{index}",
                "target_homology_cluster": "H1",
                "model_ligand_smiles": "CCO",
                "parent_standard_inchi_key": f"INCHI_{index}",
                "drug_feature_available": True,
                "murcko_scaffold": "c1ccccc1",
                "temporal_role": "TRAIN_POOL_THROUGH_2022",
            }
        )
    pairs = tmp_path / "pairs.csv.gz"
    pd.DataFrame(rows).to_csv(pairs, index=False, compression="gzip")

    contract = module.build_contract(pairs, tmp_path / "out")

    assert contract["status"] == "PASS"
    pool = contract["current_training_pool"]
    assert pool["positive_rows"] == 1
    assert pool["explicit_inactive_rows"] == 1
    assert pool["missing_numeric_affinity_rows"] == 1
    assert contract["label_contract"]["unknown_pair_is_negative"] is False
    assert contract["affinity_and_assay_coverage"]["censoring_contract_available"] is False

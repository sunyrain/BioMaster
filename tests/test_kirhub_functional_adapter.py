from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.build_kirhub_functional_training_v1 import scaffold_smiles
from scripts.train_kirhub_functional_adapter_v1 import (
    FunctionalInhibitionAdapter,
    rank_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
EXPANDED = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_pretrained_features_v1"
TRAINED = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_functional_adapter_v1"
FULL_FIT = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_full_fit_kinase_candidate_v1"


def test_functional_adapter_supports_frozen_prior_ablation() -> None:
    torch.manual_seed(31)
    drug = torch.randn(7, 768)
    target = torch.randn(7, 1280)
    priors = torch.randn(7, 4)
    embedding_only = FunctionalInhibitionAdapter(
        use_priors=False, projection_dim=16, hidden_dim=24, dropout=0.0
    ).eval()
    with_priors = FunctionalInhibitionAdapter(
        use_priors=True, projection_dim=16, hidden_dim=24, dropout=0.0
    ).eval()
    assert embedding_only(drug, target, priors).shape == (7,)
    assert with_priors(drug, target, priors).shape == (7,)
    assert torch.isfinite(embedding_only(drug, target, priors)).all()
    assert torch.isfinite(with_priors(drug, target, priors)).all()


def test_rank_pairs_are_within_drug_and_have_clear_continuous_order() -> None:
    frame = pd.DataFrame({
        "ligand_inchikey": ["A"] * 4 + ["B"] * 4,
        "inhibition_fraction_1uM": [0.95, 0.75, 0.20, 0.05, 0.80, 0.60, 0.30, 0.10],
    })
    positions = np.arange(len(frame))
    high, low, contrast = rank_pairs(frame, positions, 100, 17)
    assert len(high) == len(low) == len(contrast) > 0
    assert np.all(
        frame.iloc[high]["ligand_inchikey"].to_numpy()
        == frame.iloc[low]["ligand_inchikey"].to_numpy()
    )
    assert np.all(
        frame.iloc[high]["inhibition_fraction_1uM"].to_numpy()
        - frame.iloc[low]["inhibition_fraction_1uM"].to_numpy()
        >= 0.30 - 1e-7
    )
    assert np.all(contrast >= 0.30 - 1e-7)


def test_scaffold_contract_groups_ring_analogs_but_not_all_acyclics() -> None:
    assert scaffold_smiles("Cc1ccccc1") == scaffold_smiles("Oc1ccccc1")
    assert scaffold_smiles("CCO").startswith("ACYCLIC:")
    assert scaffold_smiles("CCO") != scaffold_smiles("CCN")


def test_expanded_feature_artifacts_obey_entity_and_label_contract() -> None:
    summary_path = EXPANDED / "KIRHUB_EXPANDED_PRETRAINED_FEATURE_SUMMARY_V1.json"
    pair_path = EXPANDED / "KIRHUB_EXPANDED_FUNCTIONAL_PAIRS_V1.csv.gz"
    if not summary_path.is_file() or not pair_path.is_file():
        pytest.skip("Expanded KiRHub artifacts are optional local outputs")
    summary = json.loads(summary_path.read_text())
    data = pd.read_csv(pair_path, low_memory=False)
    assert summary["status"] == "PASS"
    assert len(data) == 30973
    assert data["ligand_inchikey"].nunique() == 92
    assert data["uniprot_accession"].nunique() == 345
    assert int(data["strong_inhibition_label"].sum()) == 2306
    assert not data.duplicated(["ligand_inchikey", "uniprot_accession"]).any()
    assert data["inhibition_fraction_1uM"].between(0, 1).all()
    assert np.array_equal(
        data["strong_inhibition_label"].to_numpy(),
        data["inhibition_fraction_1uM"].ge(0.70).astype(np.int8).to_numpy(),
    )
    assert data.groupby("drug_scaffold_group")["drug_scaffold_cold_fold"].nunique().eq(1).all()
    assert data.groupby("target_homology_cluster")["target_homology_cold_fold"].nunique().eq(1).all()


def test_expanded_training_is_early_stopped_and_fusion_is_validation_only() -> None:
    summary_path = TRAINED / "KIRHUB_EXPANDED_FUNCTIONAL_ADAPTER_SUMMARY_V1.json"
    fit_path = TRAINED / "KIRHUB_EXPANDED_FIT_HISTORY_V1.json"
    selection_path = TRAINED / "KIRHUB_EXPANDED_CROSSFIT_SELECTION_V1.csv"
    if not all(path.is_file() for path in [summary_path, fit_path, selection_path]):
        pytest.skip("Expanded KiRHub training artifacts are optional local outputs")
    summary = json.loads(summary_path.read_text())
    fits = json.loads(fit_path.read_text())
    selection = pd.read_csv(selection_path)
    assert summary["status"] == "PASS"
    assert summary["checks"]["all_folds_reached_early_stopping"]
    assert len(fits) == 5
    assert all(row["epochs_ran"] < 600 for row in fits)
    assert selection["test_fold"].ne(selection["validation_fold"]).all()
    assert np.allclose(
        selection[["biomaster_weight", "expanded_adapter_weight", "dtiam_weight"]].sum(axis=1),
        1.0,
    )


def test_full_fit_candidate_is_kinase_only_and_fully_ranked() -> None:
    summary_path = FULL_FIT / "KIRHUB_EXPANDED_FULL_FIT_KINASE_CANDIDATE_SUMMARY_V1.json"
    score_path = FULL_FIT / "KIRHUB_FULL_FIT_720X108_KINASE_CANDIDATE_V1.csv.gz"
    if not summary_path.is_file() or not score_path.is_file():
        pytest.skip("Full-fit KiRHub kinase candidate is an optional local output")
    summary = json.loads(summary_path.read_text())
    score = pd.read_csv(score_path, low_memory=False)
    assert summary["status"] == "PASS"
    assert summary["fit_epochs"] == 190
    assert len(score) == 720 * 108
    assert score["ligand_inchikey"].nunique() == 720
    assert score["target_chembl_id"].nunique() == 108
    assert score.groupby("ligand_inchikey").size().eq(108).all()
    assert score["kirhub_kinase_candidate_rank_within_drug_108"].between(1, 108).all()

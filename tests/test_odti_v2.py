from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from biomaster.odti_v2 import (
    ODTIV2Config,
    RoutedInteractionRankerV2,
    censored_affinity_loss,
    ensemble_summary,
    expert_balance_loss,
    odti_v2_loss,
    within_group_rank_loss,
    within_group_interval_rank_loss,
    within_group_listwise_loss,
)
from biomaster.ofer_dti import OFERDTIConfig, OFERDTIModel, ofer_discovery_score, ofer_phase_a_loss
from scripts.train_biomaster_odti_v2 import (
    grouped_training_batches,
    dual_query_training_batches,
    infer_structure_group_dims,
    load_structure_features,
    load_target_token_features,
    target_token_normalization,
    prepare_arrays,
    validation_selection_value,
    load_local_graph_features,
    local_graph_batch,
)
from scripts.evaluate_biomaster_odti_v2 import progress_state, reusable_run
from scripts.train_biomaster_bindingdb_affinity_augmented_v1 import affinity_retrieval_metrics
from scripts.build_biomaster_odti_target_token_features_v1 import window_bounds
from scripts.build_biomaster_odti_structure_features_v1 import build_structure_context
from scripts.build_biomaster_odti_structure_features_v2 import (
    FEATURE_COLUMNS as STRUCTURE_V2_COLUMNS,
    FEATURE_GROUPS as STRUCTURE_V2_GROUPS,
    parse_pdb_ca,
    target_feature_row,
)


def make_inputs(batch: int = 10):
    return (
        torch.randn(batch, 2048),
        torch.randn(batch, 1024),
        torch.arange(batch) % 3,
        torch.randn(batch),
    )


def test_v2_structure_mask_is_exact_fallback() -> None:
    torch.manual_seed(7)
    model = RoutedInteractionRankerV2(
        family_count=3,
        config=ODTIV2Config(structure_input_dim=8, dropout=0.0),
    ).eval()
    drug, target, family, conplex = make_inputs()
    structure = torch.randn(10, 8)
    masked = model(drug, target, family, conplex, structure, torch.zeros(10))
    assert torch.equal(masked["final_logit"], masked["base_logit"])
    assert torch.equal(masked["structure_gate"], torch.zeros(10))

    enabled = model(drug, target, family, conplex, structure, torch.ones(10))
    assert torch.isfinite(enabled["final_logit"]).all()
    assert (enabled["structure_gate"] > 0).all()


def test_v2_without_structure_has_zero_residual() -> None:
    model = RoutedInteractionRankerV2(
        family_count=3,
        config=ODTIV2Config(structure_input_dim=0, dropout=0.0),
    ).eval()
    drug, target, family, conplex = make_inputs()
    result = model(drug, target, family, conplex)
    assert torch.equal(result["final_logit"], result["base_logit"])
    assert torch.equal(result["structure_gate"], torch.zeros(10))


def test_low_rank_film_interaction_is_identity_conditioned_at_initialization() -> None:
    torch.manual_seed(13)
    low_rank = RoutedInteractionRankerV2(
        family_count=3,
        config=ODTIV2Config(
            interaction_mode="low_rank_film",
            interaction_rank=24,
            dropout=0.0,
        ),
    ).eval()
    full = RoutedInteractionRankerV2(
        family_count=3,
        config=ODTIV2Config(interaction_mode="legacy_full", dropout=0.0),
    ).eval()
    drug, target, family, conplex = make_inputs()
    result = low_rank(drug, target, family, conplex)
    assert torch.equal(result["interaction_drug_embedding"], result["drug_embedding"])
    assert torch.equal(result["interaction_target_embedding"], result["target_embedding"])
    assert torch.isfinite(result["final_logit"]).all()
    low_parameters = sum(parameter.numel() for parameter in low_rank.parameters())
    full_parameters = sum(parameter.numel() for parameter in full.parameters())
    assert low_parameters < full_parameters - 5_000_000


def test_grouped_structure_encoder_and_enhanced_interaction_keep_exact_fallback() -> None:
    torch.manual_seed(17)
    config = ODTIV2Config(
        structure_input_dim=11,
        structure_group_dims=(2, 3, 4, 2),
        enhanced_structure_interaction=True,
        structure_gate_init_bias=-4.0,
        interaction_mode="low_rank_film",
        interaction_rank=16,
        dropout=0.0,
    )
    model = RoutedInteractionRankerV2(family_count=3, config=config).eval()
    drug, target, family, conplex = make_inputs()
    structure = torch.randn(10, 11)
    mask = torch.tensor([0, 1] * 5, dtype=torch.float32)
    result = model(drug, target, family, conplex, structure, mask)
    assert torch.equal(
        result["final_logit"][mask == 0], result["base_logit"][mask == 0]
    )
    assert torch.equal(
        result["structure_group_weights"][mask == 0], torch.zeros(5, 4)
    )
    assert torch.allclose(
        result["structure_group_weights"][mask == 1].sum(dim=1), torch.ones(5)
    )
    assert float(result["structure_gate"][mask == 1].mean().detach()) < 0.1


def test_structure_v2_group_contract_is_contiguous() -> None:
    dimensions = infer_structure_group_dims(STRUCTURE_V2_COLUMNS, "grouped")
    assert dimensions == tuple(len(columns) for columns in STRUCTURE_V2_GROUPS.values())
    assert sum(dimensions) == len(STRUCTURE_V2_COLUMNS)
    with pytest.raises(ValueError):
        infer_structure_group_dims(list(reversed(STRUCTURE_V2_COLUMNS)), "grouped")


def test_structure_v2_extracts_label_free_local_pocket_features(tmp_path: Path) -> None:
    pdb = tmp_path / "tiny.pdb"
    pdb.write_text(
        "ATOM      1  CA  LEU A   1       0.000   0.000   0.000  1.00 90.00           C  \n"
        "ATOM      2  CA  ASP A   2       4.000   0.000   0.000  1.00 80.00           C  \n"
        "ATOM      3  CA  PHE A   3       0.000   4.000   0.000  1.00 70.00           C  \n"
        "END\n"
    )
    parsed = parse_pdb_ca(pdb)
    assert len(parsed) == 3
    row = pd.Series(
        {
            "sequence_key": "SEQ_TEST",
            "pdb_path": str(pdb),
            "top_pocket_residue_ids": "A_1 A_2 A_3",
            "top_pocket_probability": 0.8,
            "top_pocket_score": 12.0,
            "p2rank26_top_pocket_volume": 300.0,
            "receptor_residue_count": 3,
            "sequence_match_status": "exact_match",
            "strict_structure_tier": "A_strict",
        }
    )
    features, audit = target_feature_row(row, None)
    assert audit["pdb_parsed"]
    assert audit["pocket_residue_count_mapped"] == 3
    assert features["chem_hydrophobic_fraction"] > 0
    assert features["chem_negative_fraction"] > 0
    assert features["chem_aromatic_fraction"] > 0
    assert features["geom_pocket_ca_radius_of_gyration"] > 0
    assert features["quality_pocket_plddt_mean"] == pytest.approx(0.8)


def test_local_graph_store_loader_and_batch_are_aligned(tmp_path: Path) -> None:
    root = tmp_path / "graph_store"
    root.mkdir()
    np.save(root / "LIGAND_ATOM_FEATURES_FLOAT16_V1.npy", np.zeros((3, 4), dtype=np.float16))
    np.save(root / "LIGAND_EDGE_INDEX_INT16_V1.npy", np.array([[0, 1], [1, 0]], dtype=np.int16))
    np.save(root / "LIGAND_EDGE_TYPE_UINT8_V1.npy", np.array([1, 1], dtype=np.uint8))
    np.save(root / "POCKET_ESM2_RESIDUE_FLOAT16_V1.npy", np.ones((2, 5), dtype=np.float16))
    np.save(root / "POCKET_RESIDUE_AUX_FLOAT16_V1.npy", np.ones((2, 3), dtype=np.float16))
    np.save(root / "POCKET_CA_COORD_FLOAT32_V1.npy", np.array([[0, 0, 0], [4, 0, 0]], dtype=np.float32))
    pd.DataFrame({
        "drug_feature_index": [0, 1], "node_offset": [0, 2], "node_count": [2, 1],
        "edge_offset": [0, 2], "edge_count": [2, 0], "graph_available": [True, False],
        "failure_reason": ["", "missing"],
    }).to_csv(root / "LIGAND_GRAPH_INDEX_V1.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "target_feature_index": [0, 1], "node_offset": [0, 2], "node_count": [2, 0],
        "graph_available": [True, False], "failure_reason": ["", "missing"],
    }).to_csv(root / "POCKET_GRAPH_INDEX_V1.csv.gz", index=False, compression="gzip")
    (root / "LOCAL_GRAPH_FEATURE_MANIFEST_V1.json").write_text(
        json.dumps({"status": "PASS", "label_dependency": "NONE"})
    )
    store = load_local_graph_features(str(root), 2, 2)
    assert store is not None
    batch = local_graph_batch(store, np.array([0, 1]), np.array([0, 1]), torch.device("cpu"))
    assert batch["ligand_atom_features"].shape == (2, 2, 4)
    assert batch["pocket_residue_features"].shape == (2, 2, 5)
    assert batch["local_pair_mask"].tolist() == [1.0, 0.0]
    assert batch["ligand_bond_type"][0, 0, 1].item() == 1
    assert batch["pocket_distance_bin"][0, 0, 1].item() == 1


def test_local_pair_branch_is_exactly_masked_when_graph_missing() -> None:
    torch.manual_seed(41)
    config = ODTIV2Config(
        drug_input_dim=8,
        target_input_dim=6,
        embedding_dim=16,
        hidden_dim=20,
        family_dim=4,
        expert_count=2,
        local_pair_atom_input_dim=4,
        local_pair_pocket_input_dim=5,
        local_pair_pocket_aux_dim=3,
        local_pair_hidden_dim=12,
        local_pair_heads=3,
        dropout=0.0,
    )
    model = RoutedInteractionRankerV2(family_count=3, config=config).eval()
    batch = 3
    common = {
        "ligand_atom_features": torch.randn(batch, 2, 4),
        "ligand_atom_mask": torch.ones(batch, 2),
        "ligand_bond_type": torch.zeros(batch, 2, 2, dtype=torch.long),
        "pocket_residue_features": torch.randn(batch, 2, 5),
        "pocket_residue_aux": torch.randn(batch, 2, 3),
        "pocket_residue_mask": torch.ones(batch, 2),
        "pocket_distance_bin": torch.zeros(batch, 2, 2, dtype=torch.long),
        "local_pair_mask": torch.tensor([1.0, 0.0, 0.0]),
    }
    output = model(
        torch.randn(batch, 8), torch.randn(batch, 6), torch.arange(batch) % 3,
        **common,
    )
    assert torch.equal(output["final_logit"][1:], output["base_logit"][1:])
    assert torch.equal(output["local_pair_gate"][1:], torch.zeros(2))
    assert output["local_pair_gate"][0] > 0


def test_v2_target_auxiliary_branch_is_masked_and_finite() -> None:
    torch.manual_seed(19)
    config = ODTIV2Config(target_aux_input_dim=13, dropout=0.0)
    model = RoutedInteractionRankerV2(family_count=3, config=config).eval()
    drug, target, family, conplex = make_inputs()
    aux = torch.randn(10, 13)
    base = model(drug, target, family, conplex, target_aux=torch.zeros_like(aux), target_aux_mask=torch.zeros(10))
    masked = model(drug, target, family, conplex, target_aux=aux, target_aux_mask=torch.zeros(10))
    enabled = model(drug, target, family, conplex, target_aux=aux, target_aux_mask=torch.ones(10))
    assert torch.equal(masked["final_logit"], base["final_logit"])
    assert torch.equal(masked["target_embedding"], masked["target_base_embedding"])
    assert torch.equal(masked["target_aux_gate"], torch.zeros(10))
    assert torch.isfinite(enabled["final_logit"]).all()
    assert (enabled["target_aux_gate"] > 0).all()


def test_v2_drug_auxiliary_branch_is_masked_and_finite() -> None:
    torch.manual_seed(21)
    config = ODTIV2Config(drug_aux_input_dim=13, dropout=0.0)
    model = RoutedInteractionRankerV2(family_count=3, config=config).eval()
    drug, target, family, conplex = make_inputs()
    aux = torch.randn(10, 13)
    base = model(drug, target, family, conplex)
    masked = model(
        drug,
        target,
        family,
        conplex,
        drug_aux=aux,
        drug_aux_mask=torch.zeros(10),
    )
    enabled = model(
        drug,
        target,
        family,
        conplex,
        drug_aux=aux,
        drug_aux_mask=torch.ones(10),
    )
    assert torch.equal(masked["final_logit"], base["final_logit"])
    assert torch.equal(masked["drug_embedding"], masked["drug_base_embedding"])
    assert torch.equal(masked["drug_aux_gate"], torch.zeros(10))
    assert torch.isfinite(enabled["final_logit"]).all()
    assert (enabled["drug_aux_gate"] > 0).all()


def test_v2_target_token_cross_attention_is_masked_and_finite() -> None:
    torch.manual_seed(23)
    config = ODTIV2Config(target_token_input_dim=11, target_token_heads=4, dropout=0.0)
    model = RoutedInteractionRankerV2(family_count=3, config=config).eval()
    drug, target, family, conplex = make_inputs()
    tokens = torch.randn(10, 5, 11)
    mask = torch.ones(10, 5)
    disabled = model(
        drug, target, family, conplex,
        target_tokens=tokens,
        target_token_mask=torch.zeros(10, 5),
    )
    baseline = model(drug, target, family, conplex)
    enabled = model(
        drug, target, family, conplex,
        target_tokens=tokens,
        target_token_mask=mask,
    )
    assert torch.equal(disabled["final_logit"], baseline["final_logit"])
    assert torch.equal(disabled["target_token_gate"], torch.zeros(10))
    assert torch.isfinite(enabled["final_logit"]).all()
    assert (enabled["target_token_gate"] > 0).all()
    assert float(enabled["target_token_gate"].mean().detach()) < 0.1


def test_target_token_window_and_normalization_are_deterministic() -> None:
    assert window_bounds(100, 50, 10) == [(0, 50), (40, 90), (50, 100)]
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    offsets = np.array([0, 5], dtype=np.int64)
    lengths = np.array([5, 5], dtype=np.int64)
    mean, std = target_token_normalization(
        features, offsets, lengths, np.array([0, 1]), max_length=5
    )
    assert np.allclose(mean, features.mean(axis=0))
    assert np.all(std > 0)


def test_target_token_loader_validates_pocket_mask_alignment(tmp_path: Path) -> None:
    feature_path = tmp_path / "tokens.npy"
    mask_path = tmp_path / "pocket.npy"
    index_path = tmp_path / "index.csv.gz"
    np.save(feature_path, np.zeros((7, 4), dtype=np.float16))
    np.save(mask_path, np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.uint8))
    pd.DataFrame(
        {
            "target_feature_index": [0, 1],
            "token_offset": [0, 4],
            "token_length": [4, 3],
        }
    ).to_csv(index_path, index=False, compression="gzip")
    features, offsets, lengths, pocket, source, index_source, pocket_source = load_target_token_features(
        str(feature_path), str(index_path), 2, 4, str(mask_path)
    )
    assert features.shape == (7, 4)
    assert offsets.tolist() == [0, 4]
    assert lengths.tolist() == [4, 3]
    assert pocket.tolist() == [0, 1, 0, 1, 0, 0, 1]
    assert source == str(feature_path)
    assert index_source == str(index_path)
    assert pocket_source == str(mask_path)


def test_v2_loss_has_all_components_and_backpropagates() -> None:
    torch.manual_seed(11)
    config = ODTIV2Config(
        structure_input_dim=4,
        drug_rank_weight=0.08,
        affinity_rank_weight=0.05,
        affinity_drug_rank_weight=0.03,
    )
    model = RoutedInteractionRankerV2(family_count=3, config=config)
    drug, target, family, conplex = make_inputs()
    result = model(
        drug, target, family, conplex, torch.randn(10, 4), torch.ones(10)
    )
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float32)
    losses = odti_v2_loss(
        result,
        labels,
        target_group=torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1]),
        drug_group=torch.tensor([0, 1, 0, 1, 2, 3, 0, 1, 2, 3]),
        affinity_lower=torch.tensor([7.0, 6.0, float("nan"), 5.0, 8.0, 6.0, 7.0, 7.0, 5.0, 6.0]),
        affinity_upper=torch.tensor([7.0, 6.0, float("nan"), 5.5, 8.0, 6.0, 7.0, 7.0, 5.0, 6.5]),
        affinity_observed=torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool),
        config=config,
    )
    assert set(losses) == {
        "total", "bce", "rank", "drug_rank", "expert_balance", "listwise", "affinity",
        "affinity_rank", "affinity_drug_rank", "observation", "contrastive",
    }
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["drug_rank"])
    assert losses["drug_rank"] > 0
    assert torch.isfinite(losses["expert_balance"])
    assert torch.isfinite(losses["affinity_rank"])
    assert losses["affinity_rank"] > 0
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_censored_affinity_loss_handles_exact_and_one_sided() -> None:
    prediction = torch.tensor([7.0, 5.0, 9.0, 4.0])
    lower = torch.tensor([7.0, 6.0, float("-inf"), float("-inf")])
    upper = torch.tensor([7.0, float("inf"), 8.0, 3.0])
    value = censored_affinity_loss(prediction, lower, upper)
    assert torch.isfinite(value)
    assert value > 0


def test_censored_affinity_loss_uses_predicted_variance() -> None:
    prediction = torch.tensor([7.0, 7.0], requires_grad=True)
    lower = torch.tensor([7.0, 7.0])
    upper = torch.tensor([7.0, 7.0])
    low_noise = torch.tensor([-5.0, -5.0], requires_grad=True)
    high_noise = torch.tensor([2.0, 2.0], requires_grad=True)
    low = censored_affinity_loss(prediction, lower, upper, log_variance=low_noise)
    high = censored_affinity_loss(prediction, lower, upper, log_variance=high_noise)
    assert torch.isfinite(low) and torch.isfinite(high)
    assert not torch.equal(low, high)
    (low + high).backward()
    assert low_noise.grad is not None
    assert high_noise.grad is not None


def test_pairwise_group_rank_loss_is_finite() -> None:
    logits = torch.tensor([2.0, 1.0, -1.0, -2.0, 1.5, -1.5])
    labels = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32)
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    value = within_group_rank_loss(logits, labels, groups, max_pairs=2)
    assert torch.isfinite(value)
    assert value > 0


def test_interval_affinity_rank_uses_only_unambiguous_orders() -> None:
    prediction = torch.tensor([7.0, 5.0, 4.0, 9.0], requires_grad=True)
    lower = torch.tensor([7.0, 5.0, 4.0, 8.0])
    upper = torch.tensor([7.0, 5.5, 4.0, float("inf")])
    groups = torch.tensor([0, 0, 0, 1])
    value = within_group_interval_rank_loss(
        prediction, lower, upper, groups, min_delta=0.5, margin=0.1
    )
    assert torch.isfinite(value)
    assert value > 0
    value.backward()
    assert prediction.grad is not None
    # Overlapping intervals and singleton groups have no invented order.
    zero = within_group_interval_rank_loss(
        torch.tensor([1.0, 2.0]),
        torch.tensor([5.0, 5.2]),
        torch.tensor([5.5, 5.6]),
        torch.tensor([0, 0]),
        min_delta=0.25,
    )
    assert torch.equal(zero, torch.tensor(0.0))


def test_bindingdb_affinity_metrics_keep_grey_zone_out_of_binary_contract() -> None:
    frame = pd.DataFrame(
        {
            "target_sequence_hash": ["a"] * 5 + ["b"] * 3,
            "mean_pchembl": [7.0, 6.2, 5.5, 4.8, 4.0, 7.0, 5.5, 4.0],
        }
    )
    score = frame["mean_pchembl"].to_numpy()
    result = affinity_retrieval_metrics(frame, score)
    assert result["targets"] == 2
    assert result["targets_with_strong_weak_metric"] == 2
    assert result["target_macro_spearman"] == pytest.approx(1.0)
    assert result["target_macro_strong_weak_auprc"] == pytest.approx(1.0)


def test_expert_balance_loss_is_zero_for_uniform_usage() -> None:
    gate = torch.full((4, 2), 0.5)
    assert torch.equal(expert_balance_loss(gate), torch.tensor(0.0))
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert expert_balance_loss(collapsed) > 0


def test_ofer_dual_process_masks_pre_event_activity_labels() -> None:
    torch.manual_seed(31)
    config = OFERDTIConfig(
        drug_input_dim=8,
        target_input_dim=6,
        latent_dim=12,
        bilinear_rank=4,
        hidden_dim=10,
        dropout=0.0,
    )
    model = OFERDTIModel(config).eval()
    drug = torch.randn(6, 8)
    target = torch.randn(6, 6)
    outputs = model(
        drug,
        target,
        torch.tensor([2014, 2015, 2016, 2017, 2018, 2018]),
    )
    losses = ofer_phase_a_loss(
        outputs,
        observation_event=torch.tensor([1, 0, 1, 1, 0, 1]),
        active_given_observed=torch.tensor([1, -1, 0, 1, -1, 0]),
        drug_group=torch.tensor([0, 0, 1, 1, 2, 2]),
        target_group=torch.tensor([0, 1, 0, 1, 0, 1]),
        config=config,
    )
    assert torch.isfinite(losses["total"])
    assert torch.equal(
        model.time_basis(torch.tensor([2000.0]), torch.tensor([0.0])),
        torch.zeros(1, 4),
    )
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_ofer_phase_a_frozen_variants_are_finite() -> None:
    torch.manual_seed(37)
    config = OFERDTIConfig(
        drug_input_dim=8,
        target_input_dim=6,
        latent_dim=12,
        bilinear_rank=4,
        hidden_dim=10,
        dropout=0.0,
    )
    model = OFERDTIModel(config).eval()
    drug = torch.randn(8, 8)
    target = torch.randn(8, 6)
    outputs = model(drug, target, torch.arange(2014, 2022))
    kwargs = dict(
        observation_event=torch.tensor([1, 0, 1, 1, 0, 1, 0, 1]),
        active_given_observed=torch.tensor([1, -1, 0, 1, -1, 0, -1, 1]),
        drug_group=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        target_group=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        sample_weight=torch.ones(8),
        sampling_role=torch.tensor([0, 1, 0, 2, 1, 0, 3, 0]),
    )
    for variant in [
        "OFER_FULL",
        "STATIC_OBSERVED_ONLY_BCE",
        "STATIC_FNML_STYLE",
        "DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD",
        "TIMESTAMP_SHUFFLED_OFER",
    ]:
        losses = ofer_phase_a_loss(outputs, variant=variant, config=config, **kwargs)
        assert torch.isfinite(losses["total"]), variant
    with pytest.raises(ValueError):
        ofer_phase_a_loss(outputs, variant="UNKNOWN", config=config, **kwargs)


def test_ofer_discovery_score_is_bounded_and_monotone_in_hazard() -> None:
    active = torch.tensor([0.0, 2.0])
    low_hazard = torch.full((2, 3), -4.0)
    high_hazard = torch.full((2, 3), 4.0)
    low = ofer_discovery_score(active, low_hazard)
    high = ofer_discovery_score(active, high_hazard)
    assert torch.all((low >= 0) & (low <= 1))
    assert torch.all((high >= 0) & (high <= 1))
    assert torch.all(high >= low)


def test_listwise_group_rank_loss_is_finite_and_zero_for_one_class() -> None:
    logits = torch.tensor([2.0, 1.0, -1.0, -2.0, 1.5, -1.5])
    labels = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.float32)
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    value = within_group_listwise_loss(logits, labels, groups)
    assert torch.isfinite(value)
    assert value > 0
    one_class = within_group_listwise_loss(
        logits[:2], labels[:2], torch.tensor([0, 0])
    )
    assert torch.equal(one_class, logits[:2].sum() * 0.0)


def test_grouped_training_batches_repeat_targets() -> None:
    data = pd.DataFrame(
        {
            "target_feature_index": [0, 0, 0, 1, 1, 1, 2, 2],
        }
    )
    positions = np.arange(len(data), dtype=np.int64)
    batches = grouped_training_batches(
        data=data,
        positions=positions,
        batch_size=4,
        seed=3,
        max_rows_per_target=2,
    )
    assert sum(len(batch) for batch in batches) == len(positions)
    assert all(len(batch) <= 4 for batch in batches)
    assert any(len(np.unique(data.iloc[batch]["target_feature_index"])) < len(batch) for batch in batches)


def test_dual_query_training_batches_cover_rows_without_duplicates() -> None:
    data = pd.DataFrame(
        {
            "target_feature_index": [0, 0, 1, 1, 2, 2, 0, 1, 2],
            "drug_feature_index": [0, 1, 0, 1, 0, 1, 2, 2, 2],
        }
    )
    positions = np.arange(len(data), dtype=np.int64)
    batches = dual_query_training_batches(
        positions,
        data,
        batch_size=4,
        seed=13,
        max_rows_per_target=3,
        max_rows_per_drug=3,
    )
    flattened = np.concatenate(batches)
    assert flattened.size == positions.size
    assert np.unique(flattened).size == positions.size
    assert set(flattened.tolist()) == set(positions.tolist())
    assert all(len(batch) <= 4 for batch in batches)


def test_family_vocab_is_train_only_with_unknown_bucket() -> None:
    data = pd.DataFrame(
        {
            "target_assay_family": ["enzyme", "enzyme", "new_family"],
            "target_feature_index": [0, 0, 1],
            "conplex_score": [0.1, 0.2, 0.3],
            "mean_pchembl": [7.0, 8.0, 9.0],
        }
    )
    target_features = np.ones((2, 4), dtype=np.float32)
    arrays = prepare_arrays(data, np.array([0, 1]), target_features)
    assert arrays["families"] == ["__UNK__", "enzyme"] or arrays["families"] == ["enzyme", "__UNK__"]
    assert arrays["family_index"][2] == arrays["families"].index("__UNK__")


def test_composite_selection_metric_prefers_retrieval_aware_score() -> None:
    row = {
        "micro_auprc": 0.5,
        "target_macro_auprc": 0.8,
        "drug_macro_auprc": 0.7,
    }
    assert validation_selection_value(row, "composite") == pytest.approx(0.63)


def test_structure_context_builder_is_pair_aligned_and_label_free(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.csv"
    target = tmp_path / "target.csv"
    atlas = tmp_path / "atlas.csv"
    pd.DataFrame(
        {
            "calibration_pair_id": ["p1", "p2", "p3"],
            "sequence_key": ["s1", "s1", "s2"],
            "binary_label": [1, 0, 1],
        }
    ).to_csv(pairs, index=False)
    pd.DataFrame(
        {
            "sequence_key": ["s1"],
            "pdb_path": ["receptor.pdb"],
            "top_pocket_score": [12.0],
            "top_pocket_probability": [0.8],
            "structure_bin": ["A_strict"],
            "sequence_match_status": ["exact_match"],
            "calibration_label": ["positive"],
        }
    ).to_csv(target, index=False)
    pd.DataFrame(
        {
            "sequence_key": ["s1"],
            "has_candidate_experimental_holo": [True],
            "best_holo_resolution": [2.2],
            "best_holo_coverage": [0.9],
        }
    ).to_csv(atlas, index=False)
    features, manifest = build_structure_context(pairs, target, atlas)
    assert features["calibration_pair_id"].tolist() == ["p1", "p2", "p3"]
    assert features["structure_mask"].tolist() == [1.0, 1.0, 0.0]
    assert "binary_label" not in features.columns
    assert "calibration_label" not in features.columns
    assert manifest["targets_with_structure_context"] == 1
    assert manifest["pairs_with_structure_context"] == 2


def test_structure_csv_loader_masks_missing_pairs(tmp_path: Path) -> None:
    path = tmp_path / "structure.csv"
    pd.DataFrame(
        {
            "calibration_pair_id": ["p1", "p3"],
            "f0": [1.0, 3.0],
            "f1": [2.0, 4.0],
            "structure_mask": [1.0, 0.0],
        }
    ).to_csv(path, index=False)
    features, mask, columns, source = load_structure_features(
        str(path), pd.Series(["p1", "p2", "p3"]), expected_dim=2
    )
    assert columns == ["f0", "f1"]
    assert source == str(path)
    assert np.allclose(features[0], [1.0, 2.0])
    assert np.allclose(features[1], [0.0, 0.0])
    assert np.allclose(features[2], [3.0, 4.0])
    assert np.allclose(mask, [1.0, 0.0, 0.0])


def test_ensemble_summary_is_pair_aligned() -> None:
    result = ensemble_summary(torch.tensor([[0.1, 0.4], [0.3, 0.2], [0.2, 0.3]]))
    assert torch.allclose(result["mean"], torch.tensor([0.2, 0.3]))
    assert torch.allclose(result["lower"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(result["upper"], torch.tensor([0.3, 0.4]))


def test_resume_existing_requires_matching_pass_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "S1__fold_0__seed_7"
    run_dir.mkdir(parents=True)
    (run_dir / "TEST_PREDICTIONS_V2.csv.gz").write_bytes(b"placeholder")
    (run_dir / "RUN_SUMMARY_V2.json").write_text(
        json.dumps({"status": "PASS", "protocol": "S1", "fold": 0, "seed": 7})
    )
    reused = reusable_run(tmp_path, "S1", 0, 7)
    assert reused is not None
    assert reused["resumed"] is True
    assert reusable_run(tmp_path, "S1", 0, 8) is None


def test_progress_state_normalizes_task_keys() -> None:
    tasks = [("S1", 0, 7), ("S1", 1, 7), ("S5", -1, 7)]
    values = {
        "S1": [{"protocol": "S1", "fold": 0, "seed": 7}],
        "S5": [],
    }
    state = progress_state(tasks, values, resumed_runs=1)
    assert state["completed"] == [["S1", 0, 7]]
    assert state["remaining"] == [["S1", 1, 7], ["S5", -1, 7]]

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from biomaster.comprehensive_balanced import (
    QueryFirstBatchSampler,
    QuerySamplingConfig,
)
from biomaster.odti_v2 import (
    ODTIV2Config,
    RoutedInteractionRankerV2,
    directional_retrieval_loss,
)
from scripts.train_biomaster_bidirectional_v6 import (
    configure_trainable_scope,
    cosine_decay_factor,
    pair_retention_loss,
    trainable_state_dict,
)


def _small_model(enabled: bool) -> RoutedInteractionRankerV2:
    return RoutedInteractionRankerV2(
        family_count=3,
        config=ODTIV2Config(
            drug_input_dim=8,
            target_input_dim=6,
            embedding_dim=16,
            hidden_dim=20,
            family_dim=4,
            expert_count=2,
            dropout=0.0,
            directional_heads_enabled=enabled,
            directional_hidden_dim=12,
            directional_dropout=0.0,
        ),
    )


def _forward(model: RoutedInteractionRankerV2) -> dict[str, torch.Tensor]:
    return model(
        torch.randn(8, 8),
        torch.randn(8, 6),
        torch.arange(8) % 3,
    )


def test_directional_heads_are_exact_zero_start_and_optional() -> None:
    torch.manual_seed(101)
    disabled = _small_model(False).eval()
    disabled_output = _forward(disabled)
    assert torch.equal(disabled_output["drug_to_target_logit"], disabled_output["final_logit"])
    assert torch.equal(disabled_output["target_to_drug_logit"], disabled_output["final_logit"])
    assert not any("_to_" in name and "head" in name for name in disabled.state_dict())

    torch.manual_seed(101)
    enabled = _small_model(True).eval()
    enabled_output = _forward(enabled)
    assert torch.equal(enabled_output["drug_to_target_logit"], enabled_output["final_logit"])
    assert torch.equal(enabled_output["target_to_drug_logit"], enabled_output["final_logit"])
    assert torch.equal(enabled_output["drug_to_target_residual"], torch.zeros(8))
    assert torch.equal(enabled_output["target_to_drug_residual"], torch.zeros(8))


def test_frozen_backbone_directional_step_changes_only_requested_head() -> None:
    torch.manual_seed(103)
    model = _small_model(True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.drug_to_target_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.drug_to_target_head.parameters(), lr=0.05)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    output = _forward(model)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = directional_retrieval_loss(
        output, labels, groups, "drug_to_target", listwise_weight=0.2
    )["total"]
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed
    assert all(name.startswith("drug_to_target_head.") for name in changed)
    assert not any(name.startswith("target_to_drug_head.") for name in changed)


def test_query_first_sampler_is_deterministic_balanced_and_observed_only() -> None:
    rows = []
    for query in range(4):
        for label in (0, 1):
            for repeat in range(3):
                rows.append(
                    {
                        "drug_feature_index": query,
                        "target_feature_index": query * 10 + label * 3 + repeat,
                        "binary_label": label,
                        "binary_observed": 1,
                        "murcko_scaffold": f"S{query}_{label}_{repeat}",
                    }
                )
    rows.append(
        {
            "drug_feature_index": 0,
            "target_feature_index": 999,
            "binary_label": 0,
            "binary_observed": 0,
            "murcko_scaffold": "UNKNOWN_MUST_NOT_BE_SAMPLED",
        }
    )
    data = pd.DataFrame(rows)
    config = QuerySamplingConfig(batch_size=16, steps_per_epoch=3, rows_per_query=4)
    sampler = QueryFirstBatchSampler(
        np.arange(len(data)), data, "drug_feature_index", config
    )
    first = sampler.batches(77)
    second = sampler.batches(77)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    for batch in first:
        assert len(batch) == 16
        assert len(data) - 1 not in set(batch)
        frame = data.iloc[batch]
        for _, group in frame.groupby("drug_feature_index"):
            assert set(group["binary_label"]) == {0, 1}
            assert int(group["binary_label"].sum()) * 2 == len(group)


def test_interaction_scope_keeps_entity_and_structure_towers_frozen() -> None:
    model = _small_model(True)
    heads, backbone = configure_trainable_scope(model, "interaction")
    assert heads and backbone
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert any(name.startswith("pair_trunk.") for name in trainable)
    assert any(name.startswith("bilinear.") for name in trainable)
    assert all(not name.startswith("drug_encoder.") for name in trainable)
    assert all(not name.startswith("target_encoder.") for name in trainable)
    assert all(not name.startswith("structure_") for name in trainable)
    assert set(trainable_state_dict(model)) == trainable


def test_pair_retention_backpropagates_to_pair_score() -> None:
    torch.manual_seed(107)
    model = _small_model(True)
    configure_trainable_scope(model, "interaction")
    output = _forward(model)
    teacher = {"final_logit": output["final_logit"].detach() + 0.5}
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    observed = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    losses = pair_retention_loss(
        output,
        teacher,
        labels,
        observed,
        bce_weight=0.25,
        distill_weight=0.5,
    )
    assert losses["pair_bce"].item() > 0
    assert losses["pair_distill"].item() > 0
    losses["total"].backward()
    assert model.pair_trunk[0].weight.grad is not None


def test_cosine_decay_preserves_optimizer_group_learning_rate_ratio() -> None:
    first = torch.nn.Parameter(torch.ones(()))
    second = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 1e-3},
            {"params": [second], "lr": 2e-5},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_decay_factor(epoch, 12),
    )
    previous = [group["lr"] for group in optimizer.param_groups]
    assert np.isclose(previous[0] / previous[1], 50.0)
    for _ in range(12):
        optimizer.step()
        scheduler.step()
        current = [group["lr"] for group in optimizer.param_groups]
        assert current[0] <= previous[0]
        assert current[1] <= previous[1]
        assert np.isclose(current[0] / current[1], 50.0)
        previous = current
    assert np.isclose(previous[0], 5e-5)
    assert np.isclose(previous[1], 1e-6)

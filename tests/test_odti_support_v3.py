"""Synthetic invariants for support-conditioned selection, without data files."""

import pytest
import torch
from torch.nn import functional as F

from biomaster.odti_support_v3 import (
    SupportInteractionRankerV3,
    SupportV3Config,
    drug_macro_pairwise_loss,
)
from biomaster.odti_v2 import ODTIV2Config, within_group_listwise_loss, within_group_rank_loss


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def model_and_batch(batch_size=3, support_count=4, enabled=True):
    torch.manual_seed(137)
    config = ODTIV2Config(
        drug_input_dim=16,
        target_input_dim=12,
        target_aux_input_dim=8,
        embedding_dim=12,
        hidden_dim=24,
        family_dim=4,
        expert_count=2,
        interaction_mode="low_rank_film",
        interaction_rank=6,
        dropout=0.0,
    )
    model = SupportInteractionRankerV3(
        config, 3, SupportV3Config(hidden_dim=16, support_enabled=enabled, dropout=0.0)
    ).eval()
    batch = {
        "drug": torch.randn(batch_size, 16),
        "target": torch.randn(batch_size, 12),
        "family": torch.arange(batch_size) % 3,
        "target_aux": torch.randn(batch_size, 8),
        "target_aux_mask": torch.ones(batch_size),
        "support_drug_features": torch.randn(batch_size, 2, support_count, 16),
        "support_similarity": torch.rand(batch_size, 2, support_count),
        "support_mask": torch.ones(batch_size, 2, support_count, dtype=torch.bool),
    }
    return model, batch


def test_absent_empty_and_disabled_support_exactly_use_global_base():
    model, batch = model_and_batch()
    batch["support_mask"].zero_()
    batch["support_drug_features"].fill_(float("nan"))
    batch["support_similarity"].fill_(float("nan"))
    output = model(**batch)
    assert torch.equal(output["final_logit"], output["global_base_logit"])
    assert torch.equal(output["final_logit"], output["base_logit"])
    assert not output["support_gate"].any()
    assert not output["support_attention"].any()
    no_support = {key: value for key, value in batch.items() if not key.startswith("support_")}
    assert torch.equal(model(**no_support)["final_logit"], output["final_logit"])
    empty = {**batch, **{key: batch[key][:, :, :0] for key in batch if key.startswith("support_")}}
    assert torch.equal(model(**empty)["final_logit"], output["final_logit"])
    disabled, data = model_and_batch(enabled=False)
    disabled_output = disabled(**data)
    assert torch.equal(disabled_output["final_logit"], disabled_output["global_base_logit"])


def test_padding_and_support_permutation_do_not_change_scores():
    model, batch = model_and_batch()
    batch["support_mask"][0, 0] = False
    batch["support_mask"][2, :, 2:] = False
    output = model(**batch)
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = {
        key: value[:, :, permutation] if key.startswith("support_") else value
        for key, value in batch.items()
    }
    torch.testing.assert_close(model(**permuted)["final_logit"], output["final_logit"])
    padded = dict(batch)
    for key in ("support_drug_features", "support_similarity", "support_mask"):
        value = batch[key]
        shape = list(value.shape)
        shape[2] = 3
        fill = torch.zeros(shape, dtype=value.dtype) if key == "support_mask" else torch.full(shape, float("nan"))
        padded[key] = torch.cat([value, fill], dim=2)
    padded_output = model(**padded)
    torch.testing.assert_close(padded_output["final_logit"], output["final_logit"])
    assert torch.isfinite(padded_output["final_logit"]).all()
    assert not padded_output["support_attention"][:, :, -3:].any()
    assert not output["support_attention"][0, 0].any()
    torch.testing.assert_close(output["support_attention"][0, 1].sum(), torch.tensor(1.0))


def test_batch_partition_and_candidate_order_do_not_change_scores():
    model, batch = model_and_batch()
    expected = model(**batch)["final_logit"]
    parts = [model(**{key: value[i : i + 1] for key, value in batch.items()})["final_logit"] for i in range(3)]
    torch.testing.assert_close(torch.cat(parts), expected, atol=1e-5, rtol=1e-5)
    order = torch.tensor([2, 0, 1])
    actual = model(**{key: value[order] for key, value in batch.items()})["final_logit"]
    torch.testing.assert_close(actual, expected[order], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scope", ["query", "class", "item"])
def test_explicit_support_dropout_masks(scope):
    model, batch = model_and_batch()
    shape = {"query": (3,), "class": (3, 2), "item": (3, 2, 4)}[scope]
    keep = torch.ones(shape, dtype=torch.bool)
    keep[0] = False
    output = model(**batch, support_keep_mask=keep)
    assert output["final_logit"][0] == output["base_logit"][0]
    assert not output["support_attention"][0].any()
    torch.testing.assert_close(output["support_gate"][1:], torch.full((2,), 0.5))


def test_valid_supports_share_encoder_and_receive_gradients_without_padding_nan():
    model, batch = model_and_batch()
    batch["support_mask"][0, 0, 0] = False
    batch["support_drug_features"][0, 0, 0] = float("nan")
    batch["support_drug_features"].requires_grad_()
    rows_encoded = []
    hook = model.base.drug_encoder.register_forward_hook(
        lambda module, inputs, output: rows_encoded.append(inputs[0].shape[0])
    )
    try:
        output = model(**batch)
    finally:
        hook.remove()
    assert rows_encoded == [3, int(batch["support_mask"].sum())]
    F.binary_cross_entropy_with_logits(output["final_logit"], torch.tensor([1.0, 0.0, 1.0])).backward()
    support_grad = batch["support_drug_features"].grad
    assert torch.isfinite(support_grad).all()
    assert not support_grad[0, 0, 0].any()
    assert support_grad[batch["support_mask"]].abs().sum() > 0
    for layer in (
        model.base.drug_encoder.net[0],
        model.base.target_encoder.net[0],
        model.base.pair_trunk[0],
        model.base.bilinear.drug,
        model.support_projection[0],
        model.support_attention_heads[0][0],
        model.support_attention_heads[1][0],
    ):
        assert layer.weight.requires_grad
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_local_representation_replaces_main_score_and_has_its_own_gradient():
    model, batch = model_and_batch()
    batch["support_mask"].zero_()
    available = torch.tensor([True, False, True])
    local = torch.randn(3, 16)
    local[1] = float("nan")
    local.requires_grad_()
    original = model(**batch, local_hidden=local, local_available=available)
    torch.testing.assert_close(original["final_logit"][available], model.local_head(local[available]).squeeze(-1))
    with torch.no_grad():
        model.base.shared_head.bias.add_(20.0)
    changed = model(**batch, local_hidden=local, local_available=available)
    assert torch.equal(changed["final_logit"][available], original["final_logit"][available])
    torch.testing.assert_close(changed["final_logit"][1] - original["final_logit"][1], torch.tensor(20.0))
    changed["final_logit"].sum().backward()
    assert torch.isfinite(local.grad).all()
    assert not local.grad[1].any()
    assert local.grad[available].abs().sum() > 0


def test_support_and_local_shape_contracts_are_explicit():
    model, batch = model_and_batch()
    missing = dict(batch)
    del missing["support_similarity"]
    with pytest.raises(ValueError, match="supplied together"):
        model(**missing)
    with pytest.raises(ValueError, match="local_hidden"):
        model(**batch, local_hidden=torch.randn(3, 17))
    with pytest.raises(ValueError, match="local_hidden"):
        model(**batch, local_available=torch.ones(3))
    with pytest.raises(ValueError, match="support_keep_mask"):
        model(**batch, support_keep_mask=torch.ones(3, 3))


def test_existing_pairwise_and_new_hard_loss_both_cover_each_positive():
    labels = torch.tensor([1, 1, 0])
    groups = torch.zeros(3, dtype=torch.long)
    observed = torch.ones(3, dtype=torch.bool)
    values, gradients = {}, {}
    for mode in ("pairwise", "hard", "legacy_pairwise", "legacy_listwise"):
        logits = torch.tensor([10.0, -5.0, 0.0], dtype=torch.float64, requires_grad=True)
        if mode == "legacy_pairwise":
            loss = within_group_rank_loss(logits, labels, groups)
        elif mode == "legacy_listwise":
            loss = within_group_listwise_loss(logits, labels, groups)
        else:
            loss = drug_macro_pairwise_loss(logits, labels, groups, observed, mode=mode)
        loss.backward()
        values[mode], gradients[mode] = loss.detach(), logits.grad
    torch.testing.assert_close(values["pairwise"], torch.tensor(2.5033803736941675, dtype=torch.float64))
    for mode in ("hard", "legacy_pairwise"):
        torch.testing.assert_close(values[mode], values["pairwise"])
        torch.testing.assert_close(gradients[mode], gradients["pairwise"])
    assert gradients["pairwise"][1] < -0.49
    assert gradients["legacy_listwise"][1].abs() < 1e-10


def test_hard_aggregation_changes_multi_negative_pressure_without_overflow():
    labels = torch.tensor([1, 0, 0])
    groups = torch.zeros(3, dtype=torch.long)
    observed = torch.ones(3, dtype=torch.bool)
    logits = torch.zeros(3, requires_grad=True)
    pairwise = drug_macro_pairwise_loss(logits, labels, groups, observed)
    hard = drug_macro_pairwise_loss(logits, labels, groups, observed, mode="hard")
    torch.testing.assert_close(pairwise, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(hard, torch.log(torch.tensor(3.0)))
    extreme = torch.tensor([-1000.0, 1000.0, 999.0], requires_grad=True)
    extreme_loss = drug_macro_pairwise_loss(extreme, labels, groups, observed, mode="hard")
    extreme_loss.backward()
    assert torch.isfinite(extreme_loss)
    assert torch.isfinite(extreme.grad).all()
    assert extreme.grad[1] > extreme.grad[2] > 0


@pytest.mark.parametrize("mode", ["pairwise", "hard"])
def test_unknown_rows_have_no_rank_gradient_and_drugs_are_equal_weight(mode):
    logits = torch.tensor([2.0, 0.0, -2.0, 0.0, 1.0, float("nan")], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, float("nan")])
    groups = torch.tensor([0, 0, 1, 1, 1, 1])
    observed = torch.tensor([True, True, True, True, True, False])
    total = drug_macro_pairwise_loss(logits, labels, groups, observed, mode=mode)
    first = drug_macro_pairwise_loss(logits[:2], labels[:2], groups[:2], observed[:2], mode=mode)
    second = drug_macro_pairwise_loss(logits[2:5], labels[2:5], groups[2:5], observed[2:5], mode=mode)
    torch.testing.assert_close(total, (first + second) / 2)
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[-1] == 0


def test_rank_handles_empty_observations_and_rejects_ambiguous_labels():
    logits = torch.tensor([float("nan"), 2.0], requires_grad=True)
    zero = drug_macro_pairwise_loss(logits, torch.tensor([float("nan"), -1.0]), torch.tensor([0, 1]), torch.zeros(2))
    zero.backward()
    assert zero == 0
    assert torch.equal(logits.grad, torch.zeros(2))
    with pytest.raises(ValueError, match="explicitly binary"):
        drug_macro_pairwise_loss(torch.zeros(2), torch.tensor([1.0, 0.5]), torch.zeros(2), torch.ones(2))
    with pytest.raises(ValueError, match="temperature"):
        drug_macro_pairwise_loss(torch.zeros(2), torch.tensor([1, 0]), torch.zeros(2), torch.ones(2), temperature=0)

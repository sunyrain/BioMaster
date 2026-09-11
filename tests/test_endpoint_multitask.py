import numpy as np
import pandas as pd
import pytest
import torch

from biomaster.endpoint_ablation import CyclingRows
from biomaster.endpoint_multitask import (MeasuredRankPool, query_rank_loss, endpoint_huber,
                                        ValidationPlateau, stream_state, restore_stream)


def test_query_sampling_never_invents_unobserved_pairs_or_uses_single_class_query():
    f = pd.DataFrame(dict(drug_feature_index=[10, 20, 30, 40], target_feature_index=[1, 1, 1, 2],
                         binary_label=[1, 0, 1, 1]))
    pool = MeasuredRankPool(f, 'target_feature_index')
    d, t = pool.sample(np.random.default_rng(4), queries=3, per_class=16)
    assert d.shape == (1, 2, 16) and np.all(t == 1)
    assert set(d[:, 0].ravel()) <= {10, 30}
    assert set(d[:, 1].ravel()) == {20}


def test_pairwise_ranking_moves_positive_up_and_negative_down():
    scores = torch.tensor([[[2., 3.], [-1., -2.]]], requires_grad=True)
    loss = query_rank_loss(scores)
    assert loss < query_rank_loss(scores.flip(1))
    loss.backward()
    assert (scores.grad[:, 0] < 0).all() and (scores.grad[:, 1] > 0).all()


def test_missing_regression_heads_are_not_zero_targets_or_nan_multiplied_masks():
    pred = torch.tensor([[1., float('nan')], [float('nan'), 0.]], requires_grad=True)
    value = torch.tensor([8., 6.]); endpoint = torch.tensor([0, 1])
    loss = endpoint_huber(pred, value, endpoint, torch.tensor([6., 5.]), torch.ones(2))
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 1] == 0 and pred.grad[1, 0] == 0
    with pytest.raises(ValueError):
        endpoint_huber(pred, torch.tensor([float('nan'), 6.]), endpoint, torch.zeros(2), torch.ones(2))


def test_regression_loss_balances_endpoints_instead_of_row_counts():
    pred = torch.zeros(4, 2)
    values = torch.tensor([2., 0., 0., 0.])
    ids = torch.tensor([0, 1, 1, 1])
    # Endpoint 0 Huber is 1.5; endpoint 1 is 0: mean is .75, not .375.
    assert endpoint_huber(pred, values, ids, torch.zeros(2), torch.ones(2)).item() == pytest.approx(.75)


def test_validation_plateau_waits_for_patience_and_learning_rate_reductions():
    rule = ValidationPlateau()
    assert rule.observe(.7, 30)['best']
    for i in range(1, 12):
        event = rule.observe(.7, 30)
        assert not event['converged']
        if i in [4, 8]:
            assert event['reduced_lr']
    assert rule.observe(.7, 30)['converged']


def test_validation_improvement_and_minimum_data_exposure_prevent_early_stop():
    rule = ValidationPlateau()
    rule.observe(.7, 0)
    for _ in range(13):
        assert not rule.observe(.7, 2)['converged']
    assert not rule.observe(.71, 30)['converged']
    assert rule.stale == 0 and rule.reductions_since_improvement == 0
    floor = ValidationPlateau(lr=5e-6)
    floor.observe(.7, 30)
    for _ in range(12):
        event = floor.observe(.7, 30)
    assert event['converged'] and floor.reductions_since_improvement == 0


def test_sampling_resume_preserves_future_exposure_across_epoch_boundary():
    source = CyclingRows(13, 5); source.take(11)
    frozen = stream_state(source)
    expected = source.take(31)
    resumed = CyclingRows(13, 999); restore_stream(resumed, frozen)
    np.testing.assert_array_equal(resumed.take(31), expected)
    assert resumed.passes == source.passes and resumed.offset == source.offset


def test_query_metrics_compare_within_query_and_report_real_topk_denominators():
    from scripts.run_endpoint_multitask_20260911 import QueryMetrics
    f = pd.DataFrame(dict(target_id=['A']*10+['B']*10,
                          binary_label=[1]*5+[0]*5+[1]*5+[0]*5))
    score = np.array([1]*5+[0]*5+[101]*5+[100]*5)
    metric = QueryMetrics(f, 'target_id').evaluate(score)
    assert metric['macro_ap'] == 1 and metric['p5'] == 1
    assert metric['queries'] == 2 and metric['p20_queries'] == 0 and metric['p20'] is None

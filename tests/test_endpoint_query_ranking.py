import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location('endpoint_query_ranking', Path(__file__).resolve().parents[1] / 'scripts/evaluate_endpoint_query_ranking_20260911.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_perfect_ranking_retrieves_both_positives_without_short_query_inflation():
    result = module.query_metrics(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.arange(10, 0, -1))
    assert result['ap'] == result['auroc'] == result['recall_at_5'] == result['ndcg_at_5'] == 1
    assert result['precision_at_5'] == .4
    assert np.isnan(result['recall_at_20'])


def test_reverse_ranking_loses_early_hits():
    result = module.query_metrics(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.arange(10))
    assert result['precision_at_5'] == result['recall_at_5'] == result['auroc'] == 0
    assert np.isclose(result['ap'], (1 / 9 + 2 / 10) / 2)
    assert result['reciprocal_rank'] == 1 / 9


def test_tied_scores_do_not_create_perfect_average_precision():
    result = module.query_metrics(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.ones(10))
    assert result['ap'] == .2
    assert result['auroc'] == .5

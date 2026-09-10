"""Hand-computable retrieval examples with unlabeled panel background."""

import numpy as np
import pytest

from biomaster.odti_v3_panel_metrics import positive_retrieval_metrics


def test_metrics_use_known_positive_ranks_and_skip_empty_queries():
    labels = np.zeros((3, 25), dtype=bool)
    labels[0, [0, 5]] = True  # Positive ranks 1 and 6.
    labels[1, 20] = True  # Positive rank 21.
    scores = np.broadcast_to(-np.arange(25), labels.shape)
    result = positive_retrieval_metrics(labels, scores, np.arange(25))
    assert result["queries"] == 2
    assert result["total_queries"] == 3
    assert result["skipped_queries_without_known_positive"] == 1
    assert result["candidate_targets"] == 25
    assert result["known_positive_pairs"] == 3
    assert result["unknown_background_is_measured_negative"] is False
    assert result["macro_positive_retrieval_ap"] == pytest.approx(((1 + 2 / 6) / 2 + 1 / 21) / 2)
    assert result["macro_mrr"] == pytest.approx((1 + 1 / 21) / 2)
    assert result["macro_recall_at_5"] == pytest.approx(0.25)
    assert result["macro_recall_at_20"] == pytest.approx(0.5)
    first_ndcg = (1 + 1 / np.log2(7)) / (1 + 1 / np.log2(3))
    assert result["macro_ndcg_at_20"] == pytest.approx(first_ndcg / 2)


def test_ties_are_broken_by_target_identity_and_not_column_order():
    labels = np.array([[True, False, True]])
    scores = np.zeros_like(labels, dtype=float)
    ids = np.array([30, 10, 20])
    expected = positive_retrieval_metrics(labels, scores, ids)
    # ID 10 background first; the two positives have ranks 2 and 3.
    assert expected["macro_positive_retrieval_ap"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert expected["macro_mrr"] == 0.5
    assert expected["macro_recall_at_5"] == expected["macro_recall_at_20"] == 1
    for order in [np.array([2, 0, 1]), np.array([1, 2, 0])]:
        actual = positive_retrieval_metrics(labels[:, order], scores[:, order], ids[order])
        assert actual == expected


def test_perfect_ranking_on_short_panel_is_one_with_string_ids():
    labels = np.array([[1, 1, 0, 0], [0, 0, 1, 0]])
    scores = np.array([[3, 2, 1, 0], [0, 0, 4, 0]])
    result = positive_retrieval_metrics(labels, scores, np.array(["T3", "T2", "T1", "T0"]))
    for key, value in result.items():
        if key.startswith("macro_"):
            assert value == 1


@pytest.mark.parametrize("num_queries", [0, 4])
def test_no_known_positives_has_undefined_metrics(num_queries):
    result = positive_retrieval_metrics(
        np.zeros((num_queries, 7), dtype=bool), np.zeros((num_queries, 7)), np.arange(7)
    )
    assert result["queries"] == 0
    assert result["skipped_queries_without_known_positive"] == num_queries
    assert all(value is None for key, value in result.items() if key.startswith("macro_"))


@pytest.mark.parametrize(
    "labels,scores,ids,error",
    [
        ([[1, 0]], [[1, np.nan]], [0, 1], "finite"),
        ([[1, -1]], [[1, 0]], [0, 1], "strictly binary"),
        ([[1, 0.5]], [[1, 0]], [0, 1], "strictly binary"),
        ([[1, 0]], [[1, 0]], [1, 1], "unique"),
        ([[1, 0]], [[1, 0]], [0, np.nan], "finite"),
        ([[1, 0]], [[1, 0]], ["T", ""], "nonempty"),
        ([[1, 0]], [[1]], [0, 1], "same"),
        ([[1, 0]], [[1, 0]], [0], "match"),
        ([[]], [[]], [], "nonempty"),
    ],
)
def test_invalid_metric_contracts_raise(labels, scores, ids, error):
    with pytest.raises(ValueError, match=error):
        positive_retrieval_metrics(np.asarray(labels), np.asarray(scores), np.asarray(ids))

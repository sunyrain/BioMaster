from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_biomaster_drug_centric_ranker_v1 import (
    binary_average_precision,
    paired_query_bootstrap,
    query_metric_table,
    select_robust_linear_ranker,
    within_query_percentile,
)


def test_within_query_percentile_is_query_local_and_monotone() -> None:
    frame = pd.DataFrame(
        {"drug": ["A", "A", "A", "B"], "score": [3.0, 1.0, 2.0, -5.0]}
    )
    result = within_query_percentile(frame, "score", "drug")
    assert result.tolist() == [1.0, 0.0, 0.5, 1.0]


def test_binary_average_precision_and_query_metrics() -> None:
    labels = np.array([1, 0, 1, 0], dtype=np.int8)
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert np.isclose(binary_average_precision(labels, scores), (1.0 + 2 / 3) / 2)
    frame = pd.DataFrame(
        {
            "parent_standard_inchi_key": ["A"] * 4,
            "binary_label": labels,
        }
    )
    table = query_metric_table(frame, scores)
    assert len(table) == 1
    assert np.isclose(table.loc[0, "average_precision"], (1.0 + 2 / 3) / 2)
    assert table.loc[0, "recall_at_5"] == 1.0


def test_robust_rank_selection_uses_only_supplied_validation_frame() -> None:
    rows = []
    features = []
    for drug in range(15):
        for label in [0, 1]:
            rows.append(
                {
                    "parent_standard_inchi_key": f"D{drug}",
                    "binary_label": label,
                }
            )
            features.append([2 * label - 1, 1 - 2 * label])
    frame = pd.DataFrame(rows)
    selected = select_robust_linear_ranker(
        frame,
        np.asarray(features, dtype=float),
        [(1.0, 0.0), (0.0, 1.0)],
        ["good", "bad"],
    )
    assert selected["weights"] == {"good": 1.0, "bad": 0.0}
    assert selected["worst_fold"] == 1.0


def test_paired_query_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "parent_standard_inchi_key": np.repeat(["A", "B", "C"], 4),
            "binary_label": np.tile([1, 0, 1, 0], 3),
        }
    )
    candidate = np.tile([0.9, 0.1, 0.8, 0.0], 3)
    reference = np.tile([0.1, 0.9, 0.0, 0.8], 3)
    first = paired_query_bootstrap(frame, candidate, reference, 1000, 17)
    second = paired_query_bootstrap(frame, candidate, reference, 1000, 17)
    assert first == second
    assert first["metrics"]["average_precision"]["difference_ci95_low"] > 0

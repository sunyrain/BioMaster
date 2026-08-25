from __future__ import annotations

import numpy as np
import pandas as pd

from biomaster.comprehensive_balanced import (
    BalancedSamplingConfig,
    balanced_training_batches,
    fuse_percentile_scores,
    sampling_audit,
    target_to_drug_metrics,
)


def _training_frame() -> pd.DataFrame:
    rows = []
    for target, positive, negative in [(0, 80, 8), (1, 10, 60), (2, 6, 6)]:
        for label, count in [(1, positive), (0, negative)]:
            for number in range(count):
                rows.append({
                    "target_feature_index": target,
                    "binary_label": label,
                    "binary_observed": 1,
                    "mean_pchembl": 7.0 if label else (4.5 if number % 2 else np.nan),
                    "any_explicit_inactive": int(label == 0 and number % 2 == 0),
                    "murcko_scaffold": f"T{target}_L{label}_S{number % 5}",
                })
    for target in range(3, 8):
        for number in range(9):
            rows.append({
                "target_feature_index": target,
                "binary_label": 0,
                "binary_observed": 0,
                "mean_pchembl": 5.0 + number / 10,
                "any_explicit_inactive": 0,
                "murcko_scaffold": f"A{target}_S{number % 3}",
            })
    return pd.DataFrame(rows)


def test_balanced_sampler_is_deterministic_and_balances_two_class_targets() -> None:
    data = _training_frame()
    config = BalancedSamplingConfig(
        batch_size=32,
        steps_per_epoch=5,
        rows_per_target=8,
        target_frequency_power=0.0,
        affinity_chunk_fraction=0.25,
    )
    positions = np.arange(len(data), dtype=np.int64)
    first = balanced_training_batches(positions, data, 17, config)
    second = balanced_training_batches(positions, data, 17, config)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(len(batch) == 32 for batch in first)
    for batch in first:
        part = data.iloc[batch]
        binary = part.loc[part["binary_observed"].eq(1)]
        for _, group in binary.groupby("target_feature_index"):
            assert int(group["binary_label"].sum()) == len(group) // 2
    audit = sampling_audit(first, data)
    assert audit["affinity_only_rows"] == 40
    assert 0.45 <= audit["binary_positive_fraction"] <= 0.55
    assert audit["binary_one_class_target_batch_row_fraction"] == 0.0


def test_target_to_drug_metrics_ranks_one_positive_per_query() -> None:
    frame = pd.DataFrame({
        "query_id": ["q1"] * 4 + ["q2"] * 4,
        "binary_label": [0, 1, 0, 0, 0, 0, 1, 0],
        "target_chembl_id": ["T1"] * 4 + ["T2"] * 4,
        "parent_standard_inchi_key": [f"D{x}" for x in range(8)],
    })
    binary = np.asarray([0.1, 0.9, 0.2, 0.0, 0.4, 0.3, 0.2, 0.1])
    affinity = np.asarray([0.0, 0.8, 0.1, 0.2, 0.3, 0.4, 0.2, 0.1])
    metrics, known = target_to_drug_metrics(frame, binary, affinity)
    assert metrics["queries"] == 2
    assert metrics["candidate_drugs"] == 4
    assert known.sort_values("query_id")["rank"].tolist() == [1, 3]
    assert metrics["median_rank"] == 2.0


def test_binary_shortlist_fusion_cannot_rescue_outside_candidate() -> None:
    binary = np.asarray([0.79, 0.80, 0.90, 1.00])
    affinity = np.asarray([1.00, 0.00, 0.50, 0.25])
    fused = fuse_percentile_scores(binary, affinity, "top20_refine", 0.5)
    assert fused[0] == binary[0]
    assert np.all(fused[1:] > fused[0])
    assert np.array_equal(
        fuse_percentile_scores(binary, affinity, "linear", 0.0), binary
    )


def test_fusion_rejects_invalid_shapes_and_weights() -> None:
    with np.testing.assert_raises(ValueError):
        fuse_percentile_scores(np.ones(2), np.ones(3))
    with np.testing.assert_raises(ValueError):
        fuse_percentile_scores(np.ones(2), np.ones(2), affinity_weight=1.1)

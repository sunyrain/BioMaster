from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_biomaster_direction_preserving_calibration_v1 import (
    direction_preserving_score,
    drug_rank_invariant,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_standard_inchi_key": ["D1"] * 3 + ["D2"] * 3,
            "dtiam_probability": [0.2, 0.8, 0.5, 0.7, 0.1, 0.4],
            "biomaster_stack_score": [0.9, 0.7, 0.8, 0.2, 0.4, 0.3],
        }
    )


def test_direction_preserving_score_keeps_every_within_drug_rank() -> None:
    frame = _frame()
    score = direction_preserving_score(
        frame, dtiam_bias_weight=1.75, biomaster_bias_weight=2.0
    )
    assert np.isfinite(score).all()
    assert drug_rank_invariant(frame, score)
    for _, positions in frame.groupby("parent_standard_inchi_key").groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        expected = np.argsort(-frame.iloc[index]["dtiam_probability"].to_numpy())
        observed = np.argsort(-score[index])
        assert np.array_equal(observed, expected)


def test_direction_preserving_score_rejects_missing_inputs() -> None:
    with pytest.raises(ValueError, match="missing calibration columns"):
        direction_preserving_score(
            _frame().drop(columns="biomaster_stack_score"), 1.0, 1.0
        )

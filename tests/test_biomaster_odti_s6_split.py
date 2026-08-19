from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_biomaster_odti_baselines_v1 import split_masks
from scripts.score_biomaster_recent_novel_targets_v1 import inference_arrays


def test_s6_masks_enforce_old_drug_and_target_double_cold_contract() -> None:
    data = pd.DataFrame(
        {
            "target_homology_cold_fold": [0, 1, 2, 2, 3, 4],
            "is_deployment_old_drug": [True, True, False, True, False, False],
            "has_deployment_old_drug_scaffold": [True, True, False, True, False, True],
        }
    )
    masks = split_masks(data, "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD", fold=0)
    np.testing.assert_array_equal(masks["test"], [True, False, False, False, False, False])
    np.testing.assert_array_equal(masks["valid"], [False, True, False, False, False, False])
    np.testing.assert_array_equal(masks["train"], [False, False, True, False, True, False])
    assert not (masks["train"] & masks["valid"]).any()
    assert not (masks["train"] & masks["test"]).any()


def test_external_inference_routes_unseen_family_to_checkpoint_unk() -> None:
    frame = pd.DataFrame(
        {
            "target_assay_family": ["enzyme", "family_absent_from_fit"],
            "conplex_score": [0.0, 0.0],
            "mean_pchembl": [np.nan, np.nan],
            "min_pchembl": [np.nan, np.nan],
            "max_pchembl": [np.nan, np.nan],
        }
    )
    checkpoint = {
        "families": ["enzyme", "__UNK__"],
        "normalization": {
            "target_mean": [0.0],
            "target_std": [1.0],
            "target_aux_mean": [0.0],
            "target_aux_std": [1.0],
            "conplex_mean": 0.0,
            "conplex_std": 1.0,
            "affinity_mean": 6.0,
            "affinity_std": 1.0,
            "structure_mean": [],
            "structure_std": [],
        },
    }
    arrays = inference_arrays(frame, checkpoint)
    np.testing.assert_array_equal(arrays["family_index"], [0, 1])

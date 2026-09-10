from __future__ import annotations

import pandas as pd

from scripts.build_biomaster_target_registry_745_feature_store_v1 import family_from_lane
from scripts.evaluate_biomaster_stage_a_720x745_v1 import metrics


def test_registry_assay_lanes_have_explicit_model_families() -> None:
    lanes = pd.Series([
        "ENZYME_BIOCHEMICAL",
        "KINASE_BIOCHEMICAL",
        "ION_CHANNEL_FUNCTIONAL",
        "TRANSPORTER_MEMBRANE_FUNCTIONAL",
        "EXTRACELLULAR_SPECIAL",
    ])
    assert family_from_lane(lanes).tolist() == [
        "enzyme",
        "kinase",
        "ion_channel",
        "transporter",
        "other_assayable",
    ]


def test_expanded_temporal_metrics_use_first_relevant_reciprocal_rank() -> None:
    frame = pd.DataFrame({
        "query_id": ["drug-a"] * 4,
        "binary_label": [0, 1, 0, 1],
        "score": [4.0, 3.0, 2.0, 1.0],
        "target_chembl_id": ["t1", "t2", "t3", "t4"],
    })
    result, positives = metrics(frame, "score")
    assert result["held_future_positives"] == 2
    assert result["macro_mrr"] == 0.5
    assert positives["rank"].tolist() == [2, 4]

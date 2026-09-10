import numpy as np
import pandas as pd

from biomaster.selectivity_training_v3 import (
    QueryBatchStream, observed_metrics, prepare_relations, query_microbatches,
)


def test_whole_entity_split_and_conflicting_duplicates():
    rows = [{"drug_feature_index": i, "target_feature_index": t,
             "binary_label": t % 2, "binary_observed": 1,
             "parent_standard_inchi_key": f"ENTITY_{i}", "target_assay_family": "enzyme"}
            for i in range(100) for t in range(3)]
    rows.append({**rows[0], "binary_label": 1})
    frame, audit = prepare_relations(pd.DataFrame(rows))
    assert frame.groupby("entity_key").split.nunique().max() == 1
    assert not ((frame.drug_feature_index == 0) & (frame.target_feature_index == 0)).any()
    assert audit["conflicting_observed_rows_excluded"] == 2
    shuffled, _ = prepare_relations(pd.DataFrame(rows).sample(frac=1, random_state=7))
    assert frame[["entity_key", "target_feature_index", "split"]].equals(
        shuffled[["entity_key", "target_feature_index", "split"]])


def test_standardization_alias_components_do_not_cross_holdout():
    rows = [{"drug_feature_index": i, "target_feature_index": 0,
             "binary_label": 1, "parent_standard_inchi_key": f"ENTITY_{i}",
             "target_assay_family": "enzyme"} for i in range(100)]
    rows.extend([{**rows[0], "parent_standard_inchi_key": "SALT", "target_feature_index": 1},
                 {**rows[1], "parent_standard_inchi_key": "SALT", "target_feature_index": 2}])
    frame, audit = prepare_relations(pd.DataFrame(rows))
    merged = frame.loc[frame.target_feature_index.isin([1, 2])]
    assert merged.entity_key.nunique() == 1
    assert merged.drug_feature_index.nunique() == 1
    assert merged.split.nunique() == 1
    assert audit["source_feature_entity_conflicts_merged"] == 2


def test_query_sampling_never_pads_short_query_or_crosses_fit_boundary():
    drug = np.array([0, 0, 1, 1, 1, 2, 2])
    labels = np.array([1, 0, 1, 0, 0, 1, 0])
    stream = QueryBatchStream(drug, labels, np.arange(5), queries_per_batch=2, max_items=16)
    batch = next(stream.batches(42))
    assert set(batch) == set(range(5))
    assert len(batch) == len(set(batch))
    chunks = list(query_microbatches(batch, drug, 2))
    assert all(len(np.unique(drug[c])) == 1 for c in chunks)
    assert len(chunks) == 2


def test_observed_metrics_exclude_one_class_queries_and_define_scope():
    result = observed_metrics(np.array([1, 0, 1]), np.array([2., 1., -1.]),
                              np.array([0, 0, 1]), np.array([0, 1, 0]))
    assert result["drug_two_class_queries"] == 1
    assert result["drug_macro_ap"] == 1.0
    assert result["drug_queries_with_more_than_20_candidates"] == 0
    assert "measured" in result["candidate_scope"]

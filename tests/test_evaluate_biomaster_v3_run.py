from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from biomaster.odti_support_data_v3 import SupportStore
from scripts.evaluate_biomaster_v3_run import (
    FEATURE_ORDER, evaluate_run, file_sha256, fixed_baseline_scores, paired_drug_ap_bootstrap, parser,
)


def baseline_artifact():
    return {"feature_order": FEATURE_ORDER, "scaler_mean": [0.] * 8, "scaler_scale": [1.] * 8,
            "coefficient": [[0., 1., 0., 0., 0., 0., 0., 0.]], "intercept": [0.]}


def test_fixed_coefficients_reproduce_training_pipeline_without_refitting():
    rng = np.random.default_rng(341)
    train = rng.random((80, 8), dtype=np.float32)
    labels = rng.integers(0, 2, size=80)
    scaler = StandardScaler().fit(train)
    classifier = LogisticRegression().fit(scaler.transform(train), labels)
    artifact = {"feature_order": FEATURE_ORDER, "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(), "coefficient": classifier.coef_.tolist(),
                "intercept": classifier.intercept_.tolist()}
    queries = rng.random((9, 8), dtype=np.float32)
    scores = fixed_baseline_scores(queries, artifact)
    np.testing.assert_array_equal(scores["positive_negative_logistic"], classifier.decision_function(scaler.transform(queries)))
    np.testing.assert_array_equal(scores["positive_max_tanimoto"], queries[:, 1])


def test_fixed_baseline_rejects_changed_feature_order_or_invalid_scale():
    artifact = baseline_artifact()
    artifact["feature_order"] = list(reversed(FEATURE_ORDER))
    with pytest.raises(ValueError, match="feature_order"):
        fixed_baseline_scores(np.ones((2, 8)), artifact)
    artifact = baseline_artifact()
    artifact["scaler_scale"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        fixed_baseline_scores(np.ones((2, 8)), artifact)


def test_paired_bootstrap_resamples_drugs_and_is_reproducible_under_row_shuffle():
    frame = pd.DataFrame({"drug_feature_index": [1, 1, 2, 2, 3], "binary_label": [0, 1, 0, 1, 1],
                          "v3_logit": [0., 1., 0., 1., 99.],
                          "positive_max_tanimoto": [0., 0., 0., 0., 0.],
                          "positive_negative_logistic": [0., 1., 1., 0., 0.]})
    report = paired_drug_ap_bootstrap(frame, samples=1000, seed=29)
    assert report["eligible_two_class_drugs"] == 2 and report["all_measured_drugs"] == 3
    assert report["comparisons"]["neural_minus_positive_max_tanimoto"]["mean_ap_difference"] == .5
    assert report["comparisons"]["neural_minus_positive_max_tanimoto"]["ci95"] == [.5, .5]
    assert report["comparisons"]["neural_minus_positive_negative_logistic"]["mean_ap_difference"] == .25
    assert report == paired_drug_ap_bootstrap(frame.sample(frac=1, random_state=18), samples=1000, seed=29)
    empty = paired_drug_ap_bootstrap(frame.iloc[-1:])
    assert empty["model_macro_ap"]["neural"] is None
    assert empty["comparisons"]["neural_minus_positive_max_tanimoto"]["ci95"] == [None, None]


def completed_run(tmp_path: Path):
    directory, cache = tmp_path / "run", tmp_path / "data"
    directory.mkdir()
    cache.mkdir()
    data = pd.DataFrame({"drug_feature_index": [0, 1, 2, 3, 4, 4, 5, 5],
                         "target_feature_index": [0, 0, 1, 1, 0, 1, 0, 1],
                         "binary_label": [1, 0, 1, 0, 1, 0, 0, 1],
                         "split": ["train"] * 4 + ["test"] * 4})
    features = np.vstack([np.eye(4, dtype=np.uint8), [[1, 0, 0, 0], [0, 0, 1, 0]]]).astype(np.uint8)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, features)
    store = SupportStore(features, data.drug_feature_index, data.target_feature_index, data.binary_label,
                         np.arange(4), backend="numpy", feature_path=feature_path)
    store.save(cache / "support_store")
    prepared = cache / "RELATIONS_V3.csv.gz"
    data.to_csv(prepared, index=False)
    manifest = {"status": "COMPLETE", "assignment_sha256": "assignment",
                "prepared_relations_sha256": file_sha256(prepared), "store_hash": store.store_hash,
                "identity": {"support_k": 2}}
    (cache / "DATA_MANIFEST_V3.json").write_text(json.dumps(manifest))
    summary = {"status": "completed", "selected_epoch": 1, "data_assignment_sha256": "assignment",
               "arguments": {"cache_dir": str(cache), "support_k": 2}}
    (directory / "RUN_SUMMARY_V3.json").write_text(json.dumps(summary))
    (directory / "BASELINES_V3.json").write_text(json.dumps(baseline_artifact()))
    np.savez_compressed(directory / "DENSE_TEST_PREDICTIONS_V3.npz", drug_indices=np.array([4, 5]),
                        target_indices=np.array([0, 1]), known_positive=np.eye(2, dtype=bool),
                        scores=np.array([[0., 1.], [1., 0.]]))
    measured = data.iloc[4:].copy()
    measured["v3_logit"] = [0., 1., 1., 0.]
    measured["positive_max_tanimoto"] = [1., 0., 0., 1.]
    measured["positive_negative_logistic"] = [1., 0., 0., 1.]
    measured.to_csv(directory / "TEST_PREDICTIONS_V3.csv.gz", index=False)
    return directory


def test_completed_run_evaluator_compares_dense_scores_with_same_support(tmp_path: Path, monkeypatch):
    directory = completed_run(tmp_path)
    # The evaluator must never fit a classifier, even though it reads test labels for metrics.
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **k: pytest.fail("no baseline refit allowed"))
    report = evaluate_run(directory, bootstrap_samples=1000, seed=7, save_scores=True)
    dense = report["dense_known_positive_retrieval"]
    assert dense["neural"]["macro_positive_retrieval_ap"] == .5
    for name in ["positive_max_tanimoto", "positive_negative_logistic"]:
        assert dense[name]["macro_positive_retrieval_ap"] == 1.
        assert not dense[name]["unknown_background_is_measured_negative"]
    assert not report["production_promoted"] and not report["baseline_fit_in_this_evaluation"]
    assert report["measured_test_paired_drug_bootstrap"]["comparisons"]["neural_minus_positive_max_tanimoto"]["ci95"] == [-.5, -.5]
    assert (directory / "DENSE_BASELINE_COMPARISON_V3.json").is_file()
    with np.load(directory / "DENSE_BASELINE_SCORES_V3.npz") as arrays:
        np.testing.assert_array_equal(arrays["positive_max_tanimoto"], np.eye(2))


def test_run_evaluator_rejects_incomplete_run_and_mismatched_fold(tmp_path: Path):
    directory = completed_run(tmp_path)
    path = directory / "RUN_SUMMARY_V3.json"
    summary = json.loads(path.read_text())
    summary["status"] = "training"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="completed"):
        evaluate_run(directory)
    summary["status"] = "completed"
    summary["data_assignment_sha256"] = "another-fold"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="assignment"):
        evaluate_run(directory)


def test_cli_requires_run_dir_and_positive_bootstrap_samples():
    with pytest.raises(SystemExit):
        parser().parse_args([])
    with pytest.raises(SystemExit):
        parser().parse_args(["--run-dir", "x", "--bootstrap-samples", "0"])
    assert parser().parse_args(["--run-dir", "x"]).bootstrap_samples == 1000

#!/usr/bin/env python3
"""Compare a completed V3 run with fixed same-evidence support baselines.

No classifier is fitted and no coefficient is selected in this evaluator.
Dense metrics describe retrieval of known positives over the saved target
panel. Observed-PN uncertainty resamples whole drugs with paired predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.odti_support_data_v3 import SupportStore
from biomaster.odti_v3_panel_metrics import positive_retrieval_metrics
from biomaster.selectivity_training_v3 import support_summary_features

BASELINE_NAMES = ("positive_max_tanimoto", "positive_negative_logistic")
FEATURE_ORDER = ["max_negative", "max_positive", "mean_negative", "mean_positive",
                 "has_negative", "has_positive", "max_delta", "mean_delta"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=positive_int, default=1000)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--save-scores", action="store_true", help="Also save dense neural/baseline matrices as an NPZ")
    return p


def fixed_baseline_scores(features: np.ndarray, artifact: dict) -> dict[str, np.ndarray]:
    """Apply the training-fitted StandardScaler and logistic decision function."""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER) or not np.isfinite(features).all():
        raise ValueError("Baseline features must be finite [rows, 8] support summaries")
    if artifact.get("feature_order") != FEATURE_ORDER:
        raise ValueError("Baseline feature_order differs from the trained support summary contract")
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    scale = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    coefficient = np.asarray(artifact["coefficient"], dtype=np.float64)
    intercept = np.asarray(artifact["intercept"], dtype=np.float64)
    if mean.shape != (8,) or scale.shape != (8,) or coefficient.shape != (1, 8) or intercept.shape != (1,):
        raise ValueError("Saved StandardScaler/logistic parameter shapes are invalid")
    if not all(np.isfinite(value).all() for value in [mean, scale, coefficient, intercept]) or (scale <= 0).any():
        raise ValueError("Saved baseline coefficients/scales must be finite and scales positive")
    # Match the training pipeline's float32 StandardScaler transform, including
    # intermediate cast behavior, before its float64 logistic decision function.
    scaled = features.astype(np.float32, copy=True)
    scaled -= mean.astype(scaled.dtype)
    scaled /= scale.astype(scaled.dtype)
    return {"positive_max_tanimoto": features[:, 1].astype(np.float64),
            "positive_negative_logistic": scaled @ coefficient[0] + intercept[0]}


def paired_drug_ap_bootstrap(frame: pd.DataFrame, *, samples: int = 1000, seed: int = 20260905) -> dict:
    """Paired percentile CI for observed-PN drug-macro AP differences.

    Only drugs with both measured classes have an AP comparison. Each bootstrap
    draw resamples these complete drug clusters, keeping all model predictions
    paired. It measures query sampling uncertainty for these fixed predictions.
    """
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    required = {"drug_feature_index", "binary_label", "v3_logit", *BASELINE_NAMES}
    if not required.issubset(frame):
        raise ValueError(f"Test prediction columns missing: {sorted(required - set(frame))}")
    if not np.isin(frame.binary_label, [0, 1]).all():
        raise ValueError("Observed-PN bootstrap accepts explicit binary labels only")
    model_names = ("neural", *BASELINE_NAMES)
    columns = ("v3_logit", *BASELINE_NAMES)
    if not np.isfinite(frame[list(columns)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Test model and baseline predictions must be finite")
    if frame.drug_feature_index.isna().any():
        raise ValueError("Drug cluster identifiers must be nonmissing")
    per_query, query_ids = [], []
    for drug, part in frame.groupby("drug_feature_index", sort=True):
        if part.binary_label.nunique() < 2:
            continue
        labels = part.binary_label.to_numpy()
        query_ids.append(int(drug))
        per_query.append([average_precision_score(labels, part[column].to_numpy()) for column in columns])
    ap = np.asarray(per_query, dtype=np.float64).reshape(-1, len(columns))
    result = {
        "metric": "observed_positive_negative_drug_macro_average_precision",
        "cluster_unit": "whole canonical drug entity; all predictions remain paired",
        "eligible_two_class_drugs": len(ap), "all_measured_drugs": int(frame.drug_feature_index.nunique()),
        "bootstrap_samples": int(samples), "bootstrap_seed": int(seed),
        "interval": "two-sided 95% percentile bootstrap interval",
        "uncertainty_scope": "drug sampling uncertainty for fixed predictions; excludes model-selection, seed and target-family uncertainty",
        "model_macro_ap": {name: float(ap[:, index].mean()) if len(ap) else None for index, name in enumerate(model_names)},
        "comparisons": {},
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ap), size=(samples, len(ap))) if len(ap) else np.empty((samples, 0), dtype=int)
    for index, name in enumerate(BASELINE_NAMES, start=1):
        differences = ap[:, 0] - ap[:, index]
        sampled = differences[draws].mean(axis=1) if len(ap) else None
        result["comparisons"][f"neural_minus_{name}"] = {
            "mean_ap_difference": float(differences.mean()) if len(ap) else None,
            "ci95": np.quantile(sampled, [0.025, 0.975]).tolist() if sampled is not None else [None, None],
            "drugs_neural_better": int((differences > 0).sum()),
            "drugs_equal": int((differences == 0).sum()),
            "drugs_neural_worse": int((differences < 0).sum()),
        }
    return result


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def evaluate_run(run_dir: str | Path, *, bootstrap_samples: int = 1000, seed: int = 20260905,
                 save_scores: bool = False) -> dict:
    directory = Path(run_dir).resolve()
    paths = {"summary": directory / "RUN_SUMMARY_V3.json", "baselines": directory / "BASELINES_V3.json",
             "dense_predictions": directory / "DENSE_TEST_PREDICTIONS_V3.npz",
             "measured_test_predictions": directory / "TEST_PREDICTIONS_V3.csv.gz"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Completed run artifact required: {path}")
    summary = json.loads(paths["summary"].read_text())
    if summary.get("status") != "completed":
        raise ValueError("Offline comparison requires a completed run")
    baseline_artifact = json.loads(paths["baselines"].read_text())
    # Validate parameter contract before expensive feature/support loading.
    fixed_baseline_scores(np.zeros((0, 8), np.float32), baseline_artifact)
    with np.load(paths["dense_predictions"], allow_pickle=False) as arrays:
        matrix, known_positive = arrays["scores"], arrays["known_positive"]
        drug_ids, target_ids = arrays["drug_indices"], arrays["target_indices"]
    neural_metrics = positive_retrieval_metrics(known_positive, matrix, target_ids)
    if drug_ids.ndim != 1 or len(drug_ids) != len(matrix) or len(np.unique(drug_ids)) != len(drug_ids):
        raise ValueError("Dense drug_indices must be unique and align with matrix rows")
    for values in [drug_ids, target_ids]:
        if values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError("Dense drug and target IDs must be nonnegative integer feature indices")
    cache_dir = _project_path(summary["arguments"]["cache_dir"])
    manifest = json.loads((cache_dir / "DATA_MANIFEST_V3.json").read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("assignment_sha256") != summary.get("data_assignment_sha256"):
        raise ValueError("Run data assignment differs from prepared support fold")
    prepared_path = cache_dir / "RELATIONS_V3.csv.gz"
    if file_sha256(prepared_path) != manifest["prepared_relations_sha256"]:
        raise ValueError("Prepared relation snapshot checksum mismatch")
    prepared = pd.read_csv(prepared_path, usecols=["drug_feature_index", "target_feature_index", "split"])
    test_drugs = prepared.loc[prepared.split.eq("test"), "drug_feature_index"].unique()
    if not np.isin(drug_ids, test_drugs).all():
        raise ValueError("Dense queries must use canonical drug indices in this test fold")
    if set(target_ids.tolist()) != set(prepared.target_feature_index.unique()):
        raise ValueError("Dense target panel must match the entire prepared target panel")
    store = SupportStore.load(cache_dir / "support_store")
    if store.store_hash != manifest["store_hash"]:
        raise ValueError("Support store differs from the saved experiment snapshot")
    k = int(summary["arguments"]["support_k"])
    if k != manifest["identity"]["support_k"]:
        raise ValueError("Run support k differs from its prepared cache")
    flat_drugs, flat_targets = np.repeat(drug_ids, len(target_ids)), np.tile(target_ids, len(drug_ids))
    support = store.cache_retrieve(cache_dir / f"dense_test_{len(drug_ids)}", flat_drugs, flat_targets, k=k)
    features = support_summary_features(support.similarities, support.mask)
    baseline_flat = fixed_baseline_scores(features, baseline_artifact)
    baseline_matrices = {name: scores.reshape(matrix.shape) for name, scores in baseline_flat.items()}
    dense_metrics = {"neural": neural_metrics, **{
        name: positive_retrieval_metrics(known_positive, scores, target_ids)
        for name, scores in baseline_matrices.items()
    }}
    measured = pd.read_csv(paths["measured_test_predictions"], low_memory=False)
    if "split" in measured and not measured.split.eq("test").all():
        raise ValueError("Measured prediction table contains rows outside the test split")
    if not np.isin(measured.drug_feature_index.to_numpy(), test_drugs).all():
        raise ValueError("Measured prediction drugs do not belong to this test fold")
    observed_bootstrap = paired_drug_ap_bootstrap(measured, samples=bootstrap_samples, seed=seed)
    result = {
        "status": "completed", "run_dir": str(directory), "selected_epoch": summary.get("selected_epoch"),
        "data_assignment_sha256": summary["data_assignment_sha256"], "support_store_hash": store.store_hash,
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "support_k": k, "baseline_fit_in_this_evaluation": False,
        "baseline_coefficients_source": "existing training-fitted BASELINES_V3.json; no refit or test-label weight selection",
        "dense_known_positive_retrieval": dense_metrics,
        "measured_test_paired_drug_bootstrap": observed_bootstrap,
        "comparison_scope": "development model comparison on the recorded test fold; not formal independent generalization evidence",
        "production_promoted": False,
    }
    if save_scores:
        destination = directory / "DENSE_BASELINE_SCORES_V3.npz"
        temporary = directory / "DENSE_BASELINE_SCORES_V3.pending.npz"
        np.savez_compressed(temporary, neural=matrix, known_positive=known_positive,
                            drug_indices=drug_ids, target_indices=target_ids, **baseline_matrices)
        temporary.replace(destination)
        result["dense_score_artifact"] = str(destination)
        result["dense_score_sha256"] = file_sha256(destination)
    output = directory / "DENSE_BASELINE_COMPARISON_V3.json"
    temporary = directory / "DENSE_BASELINE_COMPARISON_V3.pending.json"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"event": "offline_run_comparison_complete", "output": str(output),
                      "dense": dense_metrics, "observed_paired": observed_bootstrap}), flush=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    evaluate_run(args.run_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed, save_scores=args.save_scores)

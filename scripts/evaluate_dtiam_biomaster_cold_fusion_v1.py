#!/usr/bin/env python3
"""Validation-only DTIAM + BioMaster fusion for frozen S1--S4 protocols."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap, sha256  # noqa: E402
from run_biomaster_odti_baselines_v1 import (  # noqa: E402
    max_positive_tanimoto,
    metrics,
    split_masks,
    target_prior,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = BASE / "feature_store_v1/MORGAN2048_UINT8_V1.npy"
DTIAM = BASE / "public_retrained_v1/dtiam_same_data_compatible_v1"
BM_RUNS = BASE / "biomaster_odti_routed_ranker_v1"
BM_STACK = BM_RUNS / "cold_regime_stack_v1"
BM_MULTI = BM_RUNS / "multiseed_evaluation_v1"
OUT = BASE / "public_retrained_v1/dtiam_biomaster_cold_fusion_v1"
PROTOCOL_FOLDS = {
    "S1_SCAFFOLD_COLD_DRUG": list(range(5)),
    "S2_HOMOLOGY_COLD_TARGET": list(range(5)),
    "S3_STRICT_DOUBLE_COLD": list(range(5)),
    "S4_FIRST_SEEN_TEMPORAL_2023_2025": [-1],
}
FIVE_SEED_PROTOCOLS = {"S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"}
ALL_SEEDS = [20260813, 20260814, 20260815, 20260816, 20260817]
FEATURE_SETS = {
    "DTIAM_VALIDATION_CALIBRATION": ["dtiam_logit"],
    "DTIAM_BIOMASTER_SAME_DATA_FUSION": ["dtiam_logit", "biomaster_logit"],
    "DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION": [
        "dtiam_logit", "biomaster_logit", "train_positive_max_tanimoto",
        "target_train_prior", "target_has_train_positive_pool",
    ],
    "DTIAM_BIOMASTER_CONPLEX_EXTERNAL_ROUTED_FUSION": [
        "dtiam_logit", "biomaster_logit", "conplex_score",
        "train_positive_max_tanimoto", "target_train_prior",
        "target_has_train_positive_pool",
    ],
}


def bounded_logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability))


def sigmoid(values: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def dtiam_run(protocol: str, fold: int) -> Path:
    return DTIAM / f"{protocol}__fold_{fold}__OFFICIAL_DEFAULT_COMPAT_V1"


def passed_dtiam(protocol: str, fold: int) -> bool:
    path = dtiam_run(protocol, fold) / "RUN_SUMMARY_V1.json"
    return path.is_file() and json.loads(path.read_text()).get("status") == "PASS"


def load_biomaster_raw(protocol: str, fold: int, role: str) -> tuple[pd.DataFrame, list[int]]:
    seeds = ALL_SEEDS if protocol in FIVE_SEED_PROTOCOLS else [20260813]
    merged = None
    for seed in seeds:
        path = (
            BM_RUNS / f"{protocol}__fold_{fold}__seed_{seed}__CORE"
            / f"{role.upper()}_PREDICTIONS_V1.csv.gz"
        )
        frame = pd.read_csv(
            path,
            usecols=["calibration_pair_id", "biomaster_logit", "conplex_score"],
        ).rename(columns={
            "biomaster_logit": f"biomaster_logit_seed_{seed}",
            "conplex_score": f"conplex_score_seed_{seed}",
        })
        merged = frame if merged is None else merged.merge(
            frame, on="calibration_pair_id", validate="one_to_one"
        )
    assert merged is not None
    merged["biomaster_logit"] = merged[
        [f"biomaster_logit_seed_{seed}" for seed in seeds]
    ].mean(axis=1)
    conplex_columns = [f"conplex_score_seed_{seed}" for seed in seeds]
    if len(seeds) > 1 and not np.allclose(
        merged[conplex_columns].max(axis=1),
        merged[conplex_columns].min(axis=1),
        atol=1e-12,
    ):
        raise RuntimeError(f"ConPLex predictions differ by BioMaster seed for {protocol} fold {fold}")
    merged["conplex_score_external"] = merged[conplex_columns].mean(axis=1)
    return merged[[
        "calibration_pair_id", "biomaster_logit", "conplex_score_external"
    ]], seeds


def load_biomaster_stack(protocol: str, fold: int) -> tuple[pd.DataFrame, str]:
    multiseed = BM_MULTI / f"{protocol}_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz"
    if multiseed.is_file():
        frame = pd.read_csv(
            multiseed,
            usecols=["calibration_pair_id", "BIOMASTER_STACK_FIVE_SEED_MEAN"],
        ).rename(columns={"BIOMASTER_STACK_FIVE_SEED_MEAN": "biomaster_stack_score"})
        return frame, "FIVE_SEED_ENSEMBLE"
    path = (
        BM_STACK / f"{protocol}__fold_{fold}__seed_20260813__CORE"
        / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
    )
    frame = pd.read_csv(
        path, usecols=["calibration_pair_id", "COLD_REGIME_ROUTED_STACK"]
    ).rename(columns={"COLD_REGIME_ROUTED_STACK": "biomaster_stack_score"})
    return frame, "SINGLE_SEED_20260813"


def role_features(
    train: pd.DataFrame,
    role: pd.DataFrame,
    fingerprints: np.ndarray,
) -> pd.DataFrame:
    result = role[["calibration_pair_id"]].copy()
    similarity, _ = max_positive_tanimoto(train, role.reset_index(drop=True), fingerprints)
    result["train_positive_max_tanimoto"] = similarity
    result["target_train_prior"] = target_prior(train, role)
    positive_targets = set(train.loc[train["binary_label"].eq(1), "target_chembl_id"])
    result["target_has_train_positive_pool"] = role["target_chembl_id"].isin(
        positive_targets
    ).astype(np.int8).to_numpy()
    return result


def run_fold(
    data: pd.DataFrame,
    fingerprints: np.ndarray,
    protocol: str,
    fold: int,
) -> dict:
    run = dtiam_run(protocol, fold)
    masks = split_masks(data, protocol, fold)
    available = data["drug_feature_available"].to_numpy(dtype=bool)
    train = data.loc[masks["train"] & available].reset_index(drop=True)
    valid_role = data.loc[masks["valid"] & available].reset_index(drop=True)
    test_role = data.loc[masks["test"] & available].reset_index(drop=True)
    valid_external = role_features(train, valid_role, fingerprints)
    test_external = role_features(train, test_role, fingerprints)

    dtiam_valid = pd.read_csv(
        run / "VALIDATION_PREDICTIONS_V1.csv.gz",
        usecols=["calibration_pair_id", "dtiam_probability"],
    )
    dtiam_test = pd.read_csv(
        run / "TEST_PREDICTIONS_V1.csv.gz",
        usecols=["calibration_pair_id", "dtiam_probability"],
    )
    bm_valid, seeds = load_biomaster_raw(protocol, fold, "validation")
    bm_test, test_seeds = load_biomaster_raw(protocol, fold, "test")
    if seeds != test_seeds:
        raise RuntimeError("BioMaster validation/test seed mismatch")
    bm_stack, biomaster_evidence = load_biomaster_stack(protocol, fold)

    valid = valid_role.merge(dtiam_valid, on="calibration_pair_id", validate="one_to_one")
    valid = valid.merge(bm_valid, on="calibration_pair_id", validate="one_to_one")
    valid = valid.merge(valid_external, on="calibration_pair_id", validate="one_to_one")
    test = test_role.merge(dtiam_test, on="calibration_pair_id", validate="one_to_one")
    test = test.merge(bm_test, on="calibration_pair_id", validate="one_to_one")
    test = test.merge(test_external, on="calibration_pair_id", validate="one_to_one")
    test = test.merge(bm_stack, on="calibration_pair_id", validate="one_to_one")
    if len(valid) != len(dtiam_valid) or len(test) != len(dtiam_test):
        raise RuntimeError(f"Role alignment mismatch for {protocol} fold {fold}")
    valid["dtiam_logit"] = bounded_logit(valid["dtiam_probability"])
    test["dtiam_logit"] = bounded_logit(test["dtiam_probability"])
    valid["conplex_score"] = valid["conplex_score_external"]
    test["conplex_score"] = test["conplex_score_external"]
    test["biomaster_raw_score"] = sigmoid(test["biomaster_logit"])

    prediction = test[[
        "calibration_pair_id", "binary_label", "target_chembl_id",
        "parent_standard_inchi_key", "target_homology_cluster", "scaffold_group",
        "dtiam_probability", "biomaster_raw_score", "biomaster_stack_score",
    ]].copy()
    y_valid = valid["binary_label"].to_numpy(dtype=np.int8)
    metric_rows = []
    fitted = {}
    for model_name, columns in FEATURE_SETS.items():
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=2000, random_state=20260813
            ),
        )
        pipeline.fit(valid[columns].to_numpy(dtype=np.float64), y_valid)
        valid_score = pipeline.predict_proba(valid[columns].to_numpy(dtype=np.float64))[:, 1]
        test_score = pipeline.predict_proba(test[columns].to_numpy(dtype=np.float64))[:, 1]
        prediction[model_name] = test_score
        row = {
            "protocol": protocol,
            "fold": fold,
            "model": model_name,
            "features": ";".join(columns),
            "validation_rows": len(valid),
            "validation_micro_auprc": metrics(valid, valid_score)["micro_auprc"],
        }
        row.update(metrics(test, test_score))
        metric_rows.append(row)
        scaler = pipeline.named_steps["standardscaler"]
        logistic = pipeline.named_steps["logisticregression"]
        fitted[model_name] = {
            "features": columns,
            "scaler_mean": dict(zip(columns, scaler.mean_.tolist())),
            "scaler_scale": dict(zip(columns, scaler.scale_.tolist())),
            "standardized_coefficients": dict(zip(columns, logistic.coef_[0].tolist())),
            "intercept": float(logistic.intercept_[0]),
            "C": 0.1,
        }
    frozen = {
        "DTIAM_RAW": "dtiam_probability",
        f"BIOMASTER_RAW_{biomaster_evidence}": "biomaster_raw_score",
        f"BIOMASTER_STACK_{biomaster_evidence}": "biomaster_stack_score",
    }
    for model_name, column in frozen.items():
        row = {
            "protocol": protocol,
            "fold": fold,
            "model": model_name,
            "features": column,
            "validation_rows": np.nan,
            "validation_micro_auprc": np.nan,
        }
        row.update(metrics(test, prediction[column].to_numpy(dtype=np.float64)))
        metric_rows.append(row)

    destination = OUT / f"{protocol}__fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "DTIAM_BIOMASTER_COLD_FUSION_PREDICTIONS_V1.csv.gz"
    prediction.to_csv(prediction_path, index=False)
    checks = {
        "validation_and_test_disjoint": set(valid["calibration_pair_id"]).isdisjoint(
            test["calibration_pair_id"]
        ),
        "dtiam_exact_validation_coverage": len(valid) == len(dtiam_valid),
        "dtiam_exact_test_coverage": len(test) == len(dtiam_test),
        "all_scores_finite_bounded": bool(np.isfinite(
            prediction[list(FEATURE_SETS)].to_numpy(dtype=np.float64)
        ).all() and ((prediction[list(FEATURE_SETS)] >= 0.0) & (
            prediction[list(FEATURE_SETS)] <= 1.0
        )).all().all()),
        "test_labels_never_used_for_fitting": True,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": protocol,
        "fold": fold,
        "biomaster_seeds": seeds,
        "biomaster_evidence": biomaster_evidence,
        "fit_boundary": "Fixed C=0.1 balanced logistic fit on frozen validation only; test labels evaluation-only.",
        "same_data_boundary": "Same-data fusion excludes ConPLex and all public checkpoint scores; only frozen supervised models and training-derived routes are included.",
        "external_augmented_boundary": "ConPLex external route is reported separately because checkpoint-source overlap is unresolved.",
        "counts": {"validation": len(valid), "test": len(test)},
        "fitted_models": fitted,
        "metrics": metric_rows,
        "checks": checks,
        "prediction_sha256": sha256(prediction_path),
    }
    (destination / "RUN_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    return summary


def aggregate(protocols: list[str], iterations: int, seed: int) -> dict:
    coverage = []
    all_metrics = []
    all_bootstrap = []
    for protocol in protocols:
        completed = []
        predictions = []
        for fold in PROTOCOL_FOLDS[protocol]:
            destination = OUT / f"{protocol}__fold_{fold}"
            summary_path = destination / "RUN_SUMMARY_V1.json"
            prediction_path = destination / "DTIAM_BIOMASTER_COLD_FUSION_PREDICTIONS_V1.csv.gz"
            if not summary_path.is_file() or not prediction_path.is_file():
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "PASS":
                continue
            completed.append(fold)
            predictions.append(pd.read_csv(prediction_path))
        if not predictions:
            continue
        frame = pd.concat(predictions, ignore_index=True)
        if not frame["calibration_pair_id"].is_unique:
            raise RuntimeError(f"Fusion OOF pair overlap for {protocol}")
        model_columns = list(FEATURE_SETS) + [
            "dtiam_probability", "biomaster_raw_score", "biomaster_stack_score"
        ]
        for column in model_columns:
            row = {
                "protocol": protocol,
                "completed_folds": ";".join(map(str, completed)),
                "model": column,
            }
            row.update(metrics(frame, frame[column].to_numpy(dtype=np.float64)))
            all_metrics.append(row)
        if protocol == "S1_SCAFFOLD_COLD_DRUG":
            clusters = ["scaffold_group"]
        elif protocol == "S2_HOMOLOGY_COLD_TARGET":
            clusters = ["target_homology_cluster"]
        else:
            clusters = ["target_homology_cluster", "scaffold_group"]
        comparisons = [
            ("DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION", "biomaster_stack_score"),
            ("DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION", "dtiam_probability"),
            ("DTIAM_BIOMASTER_CONPLEX_EXTERNAL_ROUTED_FUSION", "biomaster_stack_score"),
            ("DTIAM_BIOMASTER_CONPLEX_EXTERNAL_ROUTED_FUSION", "DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION"),
        ]
        for cluster in clusters:
            for challenger, reference in comparisons:
                for metric_name in ["auprc", "auroc"]:
                    result = paired_cluster_bootstrap(
                        frame, challenger, reference, metric_name,
                        iterations, seed + len(all_bootstrap), cluster,
                    )
                    result["protocol"] = protocol
                    all_bootstrap.append(result)
        coverage.append({
            "protocol": protocol,
            "completed_folds": ";".join(map(str, completed)),
            "expected_folds": ";".join(map(str, PROTOCOL_FOLDS[protocol])),
            "complete": completed == PROTOCOL_FOLDS[protocol],
            "rows": len(frame),
        })
        frame.to_csv(OUT / f"{protocol}_POOLED_PREDICTIONS_V1.csv.gz", index=False)

    metric_frame = pd.DataFrame(all_metrics)
    bootstrap_frame = pd.DataFrame(all_bootstrap)
    coverage_frame = pd.DataFrame(coverage)
    metric_path = OUT / "DTIAM_BIOMASTER_COLD_FUSION_METRICS_V1.csv"
    bootstrap_path = OUT / "DTIAM_BIOMASTER_COLD_FUSION_BOOTSTRAP_V1.csv"
    coverage_path = OUT / "DTIAM_BIOMASTER_COLD_FUSION_COVERAGE_V1.csv"
    metric_frame.to_csv(metric_path, index=False)
    bootstrap_frame.to_csv(bootstrap_path, index=False)
    coverage_frame.to_csv(coverage_path, index=False)
    checks = {
        "at_least_one_protocol": len(coverage_frame) >= 1,
        "all_oof_scores_finite": bool(np.isfinite(
            metric_frame[["micro_auprc", "micro_auroc"]]
        ).all().all()),
        "all_bootstrap_intervals_finite": bool(np.isfinite(
            bootstrap_frame[["difference_ci95_low", "difference_ci95_high"]]
        ).all().all()),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "coverage": coverage,
        "checks": checks,
        "fit_boundary": "Per-fold fixed validation-only fusion; no test labels or KIRHub labels used for fitting.",
        "claim_policy": "Incomplete folds are diagnostic only; ConPLex external route cannot support a leakage-free same-data claim.",
        "artifacts": {
            "metrics_sha256": sha256(metric_path),
            "bootstrap_sha256": sha256(bootstrap_path),
            "coverage_sha256": sha256(coverage_path),
        },
    }
    (OUT / "DTIAM_BIOMASTER_COLD_FUSION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if len(protocols) == 1:
        protocol = protocols[0]
        metric_frame.to_csv(
            OUT / f"{protocol}_DTIAM_BIOMASTER_COLD_FUSION_METRICS_V1.csv", index=False
        )
        bootstrap_frame.to_csv(
            OUT / f"{protocol}_DTIAM_BIOMASTER_COLD_FUSION_BOOTSTRAP_V1.csv", index=False
        )
        coverage_frame.to_csv(
            OUT / f"{protocol}_DTIAM_BIOMASTER_COLD_FUSION_COVERAGE_V1.csv", index=False
        )
        (OUT / f"{protocol}_DTIAM_BIOMASTER_COLD_FUSION_SUMMARY_V1.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=list(PROTOCOL_FOLDS) + ["all"], default="all")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    data = pd.read_csv(PAIRS, low_memory=False)
    fingerprints = np.load(MORGAN, mmap_mode="r")
    protocols = list(PROTOCOL_FOLDS) if args.protocol == "all" else [args.protocol]
    OUT.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        requested = PROTOCOL_FOLDS[protocol] if args.fold == "all" else [int(args.fold)]
        for fold in requested:
            if not passed_dtiam(protocol, fold):
                print(json.dumps({"skip_missing_dtiam": protocol, "fold": fold}), flush=True)
                continue
            destination = OUT / f"{protocol}__fold_{fold}" / "RUN_SUMMARY_V1.json"
            if destination.is_file() and json.loads(destination.read_text()).get("status") == "PASS":
                print(json.dumps({"skip_existing_pass": protocol, "fold": fold}), flush=True)
                continue
            result = run_fold(data, fingerprints, protocol, fold)
            print(json.dumps({
                "status": result["status"], "protocol": protocol, "fold": fold,
            }), flush=True)
    summary = aggregate(protocols, args.iterations, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

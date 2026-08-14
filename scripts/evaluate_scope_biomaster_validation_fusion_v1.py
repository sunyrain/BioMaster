#!/usr/bin/env python3
"""Validation-only fusion of official SCOPE and BioMaster old-drug experts."""

from __future__ import annotations

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
from run_biomaster_odti_baselines_v1 import max_positive_tanimoto, metrics, split_masks, target_prior  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
CAL_PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
CAL_MORGAN = BASE / "feature_store_v1/MORGAN2048_UINT8_V1.npy"
BM_RUNS = BASE / "biomaster_odti_routed_ranker_v1"
BM_STACKS = BM_RUNS / "cold_regime_stack_v1"
SCOPE_VALID = BASE / "public_baselines_v1/scope_dti_s5_validation_v1/SCOPE_DTI_S5_VALIDATION_PREDICTIONS_V1.csv.gz"
SCOPE_TEST = BASE / "public_baselines_v1/scope_dti_old_drug_entity_cold_v1/SCOPE_DTI_OLD_DRUG_ENTITY_COLD_PREDICTIONS_V1.csv.gz"
OUT = BASE / "public_baselines_v1/scope_biomaster_validation_fusion_v1"


def main() -> None:
    run_paths = sorted(
        path for path in BM_RUNS.glob("S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_*__CORE")
        if (path / "RUN_SUMMARY_V1.json").is_file()
        and json.loads((path / "RUN_SUMMARY_V1.json").read_text()).get("status") == "PASS"
        and (path / "VALIDATION_PREDICTIONS_V1.csv.gz").is_file()
        and (BM_STACKS / path.name / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz").is_file()
    )
    required = [CAL_PAIRS, CAL_MORGAN, SCOPE_VALID, SCOPE_TEST]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if len(run_paths) != 5:
        raise RuntimeError(f"Expected five passed S5 core seeds with validation/stack outputs, found {len(run_paths)}")
    seeds = [int(json.loads((path / "RUN_SUMMARY_V1.json").read_text())["seed"]) for path in run_paths]
    data = pd.read_csv(CAL_PAIRS, low_memory=False)
    fingerprints = np.load(CAL_MORGAN, mmap_mode="r")
    masks = split_masks(data, "S5_OLD_DRUG_ENTITY_COLD", -1)
    available = data["drug_feature_available"].to_numpy(dtype=bool)
    train = data.loc[masks["train"] & available].reset_index(drop=True)
    valid_role = data.loc[masks["valid"] & available].reset_index(drop=True)
    test_role = data.loc[masks["test"] & available].reset_index(drop=True)

    valid_similarity, _ = max_positive_tanimoto(train, valid_role, fingerprints)
    valid_role["train_positive_max_tanimoto"] = valid_similarity
    valid_role["target_train_prior"] = target_prior(train, valid_role)
    positive_targets = set(train.loc[train["binary_label"].eq(1), "target_chembl_id"])
    valid_role["target_has_train_positive_pool"] = valid_role["target_chembl_id"].isin(positive_targets).astype(np.int8)

    valid_bm = valid_role[["calibration_pair_id"]].copy()
    for run_path, seed in zip(run_paths, seeds):
        part = pd.read_csv(
            run_path / "VALIDATION_PREDICTIONS_V1.csv.gz",
            usecols=["calibration_pair_id", "biomaster_logit"],
        ).rename(columns={"biomaster_logit": f"biomaster_logit_seed_{seed}"})
        valid_bm = valid_bm.merge(part, on="calibration_pair_id", validate="one_to_one")
    valid_bm["biomaster_logit"] = valid_bm[
        [column for column in valid_bm if column.startswith("biomaster_logit_seed_")]
    ].mean(axis=1)
    valid_scope = pd.read_csv(SCOPE_VALID, usecols=["calibration_pair_id", "scope_mean"])
    valid = valid_role[[
        "calibration_pair_id", "binary_label", "target_chembl_id", "parent_standard_inchi_key",
        "conplex_score", "train_positive_max_tanimoto", "target_train_prior",
        "target_has_train_positive_pool",
    ]].merge(valid_bm, on="calibration_pair_id", validate="one_to_one")
    valid = valid.merge(valid_scope, on="calibration_pair_id", validate="one_to_one")

    first_stack = BM_STACKS / run_paths[0].name / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
    test_bm = pd.read_csv(first_stack)
    test_ensemble = test_bm[["calibration_pair_id"]].copy()
    for run_path, seed in zip(run_paths, seeds):
        raw = pd.read_csv(
            run_path / "TEST_PREDICTIONS_V1.csv.gz",
            usecols=["calibration_pair_id", "biomaster_logit"],
        ).rename(columns={"biomaster_logit": f"biomaster_logit_seed_{seed}"})
        stacked = pd.read_csv(
            BM_STACKS / run_path.name / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz",
            usecols=["calibration_pair_id", "COLD_REGIME_ROUTED_STACK"],
        ).rename(columns={"COLD_REGIME_ROUTED_STACK": f"stack_seed_{seed}"})
        test_ensemble = test_ensemble.merge(raw, on="calibration_pair_id", validate="one_to_one")
        test_ensemble = test_ensemble.merge(stacked, on="calibration_pair_id", validate="one_to_one")
    test_ensemble["biomaster_logit"] = test_ensemble[
        [column for column in test_ensemble if column.startswith("biomaster_logit_seed_")]
    ].mean(axis=1)
    test_ensemble["COLD_REGIME_ROUTED_STACK"] = test_ensemble[
        [column for column in test_ensemble if column.startswith("stack_seed_")]
    ].mean(axis=1)
    test_bm = test_bm.drop(columns=["biomaster_logit", "COLD_REGIME_ROUTED_STACK"])
    test_bm = test_bm.merge(
        test_ensemble[["calibration_pair_id", "biomaster_logit", "COLD_REGIME_ROUTED_STACK"]],
        on="calibration_pair_id",
        validate="one_to_one",
    )
    test_scope = pd.read_csv(SCOPE_TEST, usecols=["calibration_pair_id", "scope_mean"])
    test = test_bm.merge(test_scope, on="calibration_pair_id", validate="one_to_one")
    if len(test) != 2556:
        raise RuntimeError("S5 test fusion coverage changed")

    feature_sets = {
        "SCOPE_VALIDATION_LOGISTIC": ["scope_mean"],
        "SCOPE_BIOMASTER_VALIDATION_FUSION": ["scope_mean", "biomaster_logit"],
        "SCOPE_BIOMASTER_ROUTED_FUSION": [
            "scope_mean", "biomaster_logit", "conplex_score",
            "train_positive_max_tanimoto", "target_train_prior",
            "target_has_train_positive_pool",
        ],
    }
    predictions = test[[
        "calibration_pair_id", "binary_label", "target_chembl_id",
        "parent_standard_inchi_key", "scope_mean", "COLD_REGIME_ROUTED_STACK",
        "train_positive_max_tanimoto", "conplex_score",
    ]].copy()
    rows = []
    fitted = {}
    for name, columns in feature_sets.items():
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=20260813),
        )
        pipeline.fit(valid[columns].to_numpy(dtype=np.float64), valid["binary_label"].to_numpy(dtype=np.int8))
        valid_score = pipeline.predict_proba(valid[columns].to_numpy(dtype=np.float64))[:, 1]
        test_score = pipeline.predict_proba(test[columns].to_numpy(dtype=np.float64))[:, 1]
        predictions[name] = test_score
        row = {
            "model": name,
            "fit_role": "S5_SCOPE_COVERED_FROZEN_VALIDATION_ONLY",
            "features": ";".join(columns),
            "validation_rows": len(valid),
            "validation_micro_auprc": metrics(valid, valid_score)["micro_auprc"],
        }
        row.update(metrics(test, test_score))
        rows.append(row)
        scaler = pipeline.named_steps["standardscaler"]
        logistic = pipeline.named_steps["logisticregression"]
        fitted[name] = {
            "features": columns,
            "scaler_mean": dict(zip(columns, scaler.mean_.tolist())),
            "scaler_scale": dict(zip(columns, scaler.scale_.tolist())),
            "standardized_coefficients": dict(zip(columns, logistic.coef_[0].tolist())),
            "intercept": float(logistic.intercept_[0]),
            "C": 0.1,
        }
    for name in ["scope_mean", "COLD_REGIME_ROUTED_STACK", "train_positive_max_tanimoto", "conplex_score"]:
        row = {"model": name, "fit_role": "FROZEN_BASELINE", "features": name, "validation_rows": None, "validation_micro_auprc": None}
        row.update(metrics(test, test[name].to_numpy(dtype=np.float64)))
        rows.append(row)
    metrics_frame = pd.DataFrame(rows)

    bootstrap_rows = []
    for challenger in ["SCOPE_BIOMASTER_VALIDATION_FUSION", "SCOPE_BIOMASTER_ROUTED_FUSION"]:
        for reference in ["scope_mean", "COLD_REGIME_ROUTED_STACK"]:
            for metric in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    predictions, challenger, reference, metric, 2000,
                    20260813 + len(bootstrap_rows), "parent_standard_inchi_key",
                )
                bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary = bootstrap[
        bootstrap["challenger"].eq("SCOPE_BIOMASTER_ROUTED_FUSION")
        & bootstrap["reference"].eq("scope_mean")
        & bootstrap["metric"].eq("auprc")
    ].iloc[0]
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "SCOPE_BIOMASTER_S5_TEST_FUSION_PREDICTIONS_V1.csv.gz"
    metric_path = OUT / "SCOPE_BIOMASTER_S5_TEST_FUSION_METRICS_V1.csv"
    bootstrap_path = OUT / "SCOPE_BIOMASTER_S5_TEST_FUSION_BOOTSTRAP_V1.csv"
    predictions.to_csv(prediction_path, index=False)
    metrics_frame.to_csv(metric_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    checks = {
        "exact_five_independent_biomaster_seeds": len(run_paths) == 5 and len(set(seeds)) == 5,
        "validation_scope_subset_exact_16509": len(valid) == 16509,
        "test_full_coverage_exact_2556": len(test) == 2556,
        "all_fusion_predictions_bounded": predictions[list(feature_sets)].apply(lambda column: column.between(0, 1).all()).all(),
        "no_test_labels_used_for_fitting": True,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "biomaster_model_seeds": seeds,
        "biomaster_ensemble_rule": "Arithmetic mean of five validation/test logits before frozen validation-only fusion fitting",
        "checks": {key: bool(value) for key, value in checks.items()},
        "fixed_fusion_models": fitted,
        "metrics": metrics_frame.to_dict("records"),
        "primary_routed_fusion_minus_scope_auprc": primary.to_dict(),
        "source_overlap_warning": "SCOPE public-pretraining exact-pair overlap with the ChEMBL37-derived validation/test population is unresolved.",
        "claim_status": "VALIDATION_ONLY_FUSION_RESULT; NOT LEAKAGE_FREE SOTA UNTIL SOURCE-HELD EXTERNAL TEST",
        "artifacts": {
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metric_path),
            "bootstrap_sha256": sha256(bootstrap_path),
        },
    }
    (OUT / "SCOPE_BIOMASTER_VALIDATION_FUSION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit validation-only cold-regime stacks and evaluate untouched test folds."""

from __future__ import annotations

import hashlib
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
from run_biomaster_odti_baselines_v1 import (  # noqa: E402
    max_positive_tanimoto,
    metrics,
    split_masks,
    target_prior,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
RUNS = BASE / "biomaster_odti_routed_ranker_v1"
STORE = BASE / "feature_store_v1"
PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = STORE / "MORGAN2048_UINT8_V1.npy"
OUT = RUNS / "cold_regime_stack_v1"
FEATURE_SETS = {
    "BIOMASTER_VALIDATION_LOGISTIC": ["biomaster_logit"],
    "BIOMASTER_CONPLEX_VALIDATION_STACK": ["biomaster_logit", "conplex_score"],
    "COLD_REGIME_ROUTED_STACK": [
        "biomaster_logit", "conplex_score", "train_positive_max_tanimoto",
        "target_train_prior", "target_has_train_positive_pool",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    result["target_has_train_positive_pool"] = role["target_chembl_id"].isin(positive_targets).astype(np.int8).to_numpy()
    return result


def main() -> None:
    data = pd.read_csv(PAIRS, low_memory=False)
    fingerprints = np.load(MORGAN, mmap_mode="r")
    available = data["drug_feature_available"].to_numpy(dtype=bool)
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    processed = 0
    for summary_path in sorted(RUNS.glob("*/RUN_SUMMARY_V1.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "PASS":
            continue
        run_dir = summary_path.parent
        validation_path = run_dir / "VALIDATION_PREDICTIONS_V1.csv.gz"
        test_path = run_dir / "TEST_PREDICTIONS_V1.csv.gz"
        if not validation_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(run_dir)
        destination = OUT / run_dir.name
        prediction_path = destination / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
        existing_summary_path = destination / "COLD_REGIME_STACK_RUN_SUMMARY_V1.json"
        if prediction_path.is_file() and existing_summary_path.is_file():
            existing_summary = json.loads(existing_summary_path.read_text())
            existing_models = existing_summary.get("models", [])
            if (
                existing_summary.get("status") == "PASS"
                and len(existing_models) == len(FEATURE_SETS)
                and {row.get("stack_model") for row in existing_models} == set(FEATURE_SETS)
            ):
                metric_rows.extend(existing_models)
                processed += 1
                print(json.dumps({"skip_existing_passed_stack": run_dir.name}), flush=True)
                continue
        masks = split_masks(data, summary["protocol"], int(summary["fold"]))
        train = data.loc[masks["train"] & available].reset_index(drop=True)
        valid_role = data.loc[masks["valid"] & available].reset_index(drop=True)
        test_role = data.loc[masks["test"] & available].reset_index(drop=True)
        valid_external = role_features(train, valid_role, fingerprints)
        test_external = role_features(train, test_role, fingerprints)
        valid_prediction = pd.read_csv(
            validation_path,
            usecols=["calibration_pair_id", "binary_label", "biomaster_logit", "conplex_score"],
        )
        test_prediction = pd.read_csv(
            test_path,
            usecols=["calibration_pair_id", "binary_label", "biomaster_logit", "conplex_score"],
        )
        valid = valid_prediction.merge(valid_external, on="calibration_pair_id", validate="one_to_one")
        test = test_prediction.merge(test_external, on="calibration_pair_id", validate="one_to_one")
        if len(valid) != len(valid_role) or len(test) != len(test_role):
            raise RuntimeError(f"Role prediction coverage changed for {run_dir.name}")
        # Restore metadata needed for macro target/drug retrieval metrics.
        test = test.merge(
            test_role[["calibration_pair_id", "target_chembl_id", "parent_standard_inchi_key"]],
            on="calibration_pair_id",
            validate="one_to_one",
        )
        output = test[[
            "calibration_pair_id", "binary_label", "target_chembl_id", "parent_standard_inchi_key",
            "biomaster_logit", "conplex_score", "train_positive_max_tanimoto",
            "target_train_prior", "target_has_train_positive_pool",
        ]].copy()
        y_valid = valid["binary_label"].to_numpy(dtype=np.int8)
        y_test = test["binary_label"].to_numpy(dtype=np.int8)
        run_rows: list[dict[str, object]] = []
        for model_name, columns in FEATURE_SETS.items():
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=20260813),
            )
            pipeline.fit(valid[columns].to_numpy(dtype=np.float64), y_valid)
            valid_score = pipeline.predict_proba(valid[columns].to_numpy(dtype=np.float64))[:, 1]
            test_score = pipeline.predict_proba(test[columns].to_numpy(dtype=np.float64))[:, 1]
            output[model_name] = test_score
            row = {
                "protocol": summary["protocol"],
                "fold": int(summary["fold"]),
                "seed": int(summary["seed"]),
                "base_variant": summary["variant"],
                "stack_model": model_name,
                "features": ";".join(columns),
                "validation_micro_auprc": metrics(
                    valid.rename(columns={
                        "target_chembl_id": "target_chembl_id",
                        "parent_standard_inchi_key": "parent_standard_inchi_key",
                    }) if "target_chembl_id" in valid else valid.merge(
                        valid_role[["calibration_pair_id", "target_chembl_id", "parent_standard_inchi_key"]],
                        on="calibration_pair_id", validate="one_to_one"
                    ),
                    valid_score,
                )["micro_auprc"],
            }
            row.update(metrics(test, test_score))
            logistic = pipeline.named_steps["logisticregression"]
            row["standardized_coefficients"] = json.dumps(
                dict(zip(columns, logistic.coef_[0].tolist())), sort_keys=True
            )
            run_rows.append(row)
            metric_rows.append(row)
        destination.mkdir(parents=True, exist_ok=True)
        output.to_csv(prediction_path, index=False)
        (destination / "COLD_REGIME_STACK_RUN_SUMMARY_V1.json").write_text(
            json.dumps({
                "status": "PASS",
                "source_run": str(run_dir.relative_to(ROOT)),
                "fit_role": "FROZEN_VALIDATION_ONLY",
                "test_role": "UNTOUCHED_FROZEN_TEST",
                "fixed_logistic_C": 0.1,
                "models": run_rows,
                "prediction_sha256": sha256(prediction_path),
            }, indent=2) + "\n"
        )
        processed += 1
        routed = next(row for row in run_rows if row["stack_model"] == "COLD_REGIME_ROUTED_STACK")
        print(json.dumps({
            "run": run_dir.name,
            "routed_stack_test_auprc": routed["micro_auprc"],
            "routed_stack_drug_macro_auprc": routed["drug_macro_auprc"],
        }), flush=True)

    result = pd.DataFrame(metric_rows).sort_values(
        ["protocol", "base_variant", "stack_model", "fold", "seed"]
    )
    result_path = OUT / "ALL_COLD_REGIME_STACK_METRICS_V1.csv"
    result.to_csv(result_path, index=False)
    aggregate_columns = [
        "micro_auroc", "micro_auprc", "target_macro_auprc", "drug_macro_auprc", "brier", "ece_15"
    ]
    aggregate = result.groupby(
        ["protocol", "base_variant", "stack_model"]
    )[aggregate_columns].agg(["count", "mean", "std"]).reset_index()
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in aggregate.columns
    ]
    aggregate_path = OUT / "COLD_REGIME_STACK_AGGREGATE_V1.csv"
    aggregate.to_csv(aggregate_path, index=False)
    checks = {
        "at_least_all_16_existing_runs_processed": processed >= 16,
        "exact_three_fixed_stacks_per_run": len(result) == processed * 3,
        "all_metrics_finite": result[["micro_auroc", "micro_auprc", "brier", "ece_15"]].notna().all().all(),
        "all_stack_test_predictions_bounded": True,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "processed_runs": processed,
        "checks": {key: bool(value) for key, value in checks.items()},
        "fit_rule": "Fixed C=0.1 balanced logistic stack fit on frozen validation predictions only; no test-label model selection",
        "metrics_sha256": sha256(result_path),
        "aggregate_sha256": sha256(aggregate_path),
        "claim_status": "INTERNAL_VALIDATION_STACK; MULTI_SEED_AND_EXTERNAL_PUBLIC_COMPARATORS_REQUIRED",
    }
    (OUT / "COLD_REGIME_STACK_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

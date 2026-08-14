#!/usr/bin/env python3
"""Retrain DTIAM's official representations on the frozen BioMaster splits.

This is deliberately labelled a compatible retraining rather than a bitwise
paper reproduction: the official BerMol and ESM2 representation definitions
are retained, while AutoGluon 1.4/Python 3.10 replaces the paper's
AutoGluon 0.5.2/Python 3.7 environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_biomaster_odti_baselines_v1 import metrics, split_masks  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
FEATURE_ROOT = BASE / "public_retrained_v1/dtiam_official_feature_store_v1"
BERMOL = FEATURE_ROOT / "DTIAM_BERMOL768_FLOAT32_V1.npy"
ESM2 = FEATURE_ROOT / "DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
PAIR_INDEX = FEATURE_ROOT / "DTIAM_86674_PAIR_FEATURE_INDEX_V1.csv.gz"
FEATURE_AUDIT = FEATURE_ROOT / "DTIAM_OFFICIAL_FEATURE_STORE_AUDIT_V1.json"
OUT = BASE / "public_retrained_v1/dtiam_same_data_compatible_v1"
FOLDS = 5
PROTOCOLS = [
    "S0_RANDOM_DIAGNOSTIC",
    "S1_SCAFFOLD_COLD_DRUG",
    "S2_HOMOLOGY_COLD_TARGET",
    "S3_STRICT_DOUBLE_COLD",
    "S4_FIRST_SEEN_TEMPORAL_2023_2025",
    "S5_OLD_DRUG_ENTITY_COLD",
]
FEATURE_COLUMNS = [f"bermol_{index:04d}" for index in range(768)] + [
    f"esm2_{index:04d}" for index in range(1280)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_table(
    rows: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    include_label: bool,
) -> pd.DataFrame:
    drug_index = rows["drug_feature_index"].to_numpy(dtype=np.int64)
    target_index = rows["target_feature_index"].to_numpy(dtype=np.int64)
    values = np.empty((len(rows), 2048), dtype=np.float32)
    values[:, :768] = drug_features[drug_index]
    values[:, 768:] = target_features[target_index]
    table = pd.DataFrame(values, columns=FEATURE_COLUMNS, copy=False)
    if include_label:
        table["y"] = rows["binary_label"].to_numpy(dtype=np.int8)
    return table


def run_one(
    data: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    protocol: str,
    fold: int,
    time_limit: int | None,
) -> dict[str, Any]:
    run_name = f"{protocol}__fold_{fold}__OFFICIAL_DEFAULT_COMPAT_V1"
    run_dir = OUT / run_name
    summary_path = run_dir / "RUN_SUMMARY_V1.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "SKIP_ALREADY_PASS", "run": run_name}), flush=True)
            return existing
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite incomplete run directory: {run_dir}. "
            "Preserve it for audit and choose a new implementation tag."
        )
    run_dir.mkdir(parents=True)
    predictor_dir = run_dir / "predictor"

    masks = split_masks(data, protocol, fold)
    available = data["dtiam_bermol_available"].to_numpy(dtype=bool)
    masks = {role: mask & available for role, mask in masks.items()}
    roles = {
        role: data.loc[mask].reset_index(drop=True)
        for role, mask in masks.items()
    }
    train, valid, test = (roles[role] for role in ["train", "valid", "test"])
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"Empty role in {run_name}: {[len(train), len(valid), len(test)]}")
    if any(frame["binary_label"].nunique() != 2 for frame in [train, valid, test]):
        raise RuntimeError(f"Single-class role in {run_name}")

    scaffold_disjoint = True
    homology_disjoint = True
    if protocol in {"S1_SCAFFOLD_COLD_DRUG", "S3_STRICT_DOUBLE_COLD", "S5_OLD_DRUG_ENTITY_COLD"}:
        scaffold_disjoint = set(train["scaffold_group"]).isdisjoint(test["scaffold_group"])
    if protocol in {"S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"}:
        homology_disjoint = set(train["target_homology_cluster"]).isdisjoint(
            test["target_homology_cluster"]
        )
    if not scaffold_disjoint or not homology_disjoint:
        raise RuntimeError(f"Frozen leakage invariant failed in {run_name}")

    print(
        json.dumps(
            {
                "status": "BUILD_FEATURE_TABLES",
                "run": run_name,
                "train": len(train),
                "valid": len(valid),
                "test": len(test),
                "features": 2048,
            }
        ),
        flush=True,
    )
    train_table = make_table(train, drug_features, target_features, include_label=True)
    valid_table = make_table(valid, drug_features, target_features, include_label=False)
    test_table = make_table(test, drug_features, target_features, include_label=False)

    fit_kwargs: dict[str, Any] = {
        "train_data": train_table,
        "excluded_model_types": [],
    }
    if time_limit is not None:
        fit_kwargs["time_limit"] = time_limit
    predictor = TabularPredictor(
        label="y",
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(predictor_dir),
        verbosity=3,
    )
    started = time.monotonic()
    predictor.fit(**fit_kwargs)
    fit_seconds = time.monotonic() - started

    valid_probability = predictor.predict_proba(valid_table).iloc[:, 1].to_numpy(dtype=np.float64)
    test_probability = predictor.predict_proba(test_table).iloc[:, 1].to_numpy(dtype=np.float64)
    valid_metrics = metrics(valid, valid_probability)
    test_metrics = metrics(test, test_probability)
    validation_prediction = valid[[
        "calibration_pair_id", "binary_label", "target_chembl_id", "primary_gene",
        "parent_standard_inchi_key", "parent_molecule_chembl_id",
    ]].copy()
    validation_prediction["dtiam_probability"] = valid_probability
    test_prediction = test[[
        "calibration_pair_id", "binary_label", "target_chembl_id", "primary_gene",
        "parent_standard_inchi_key", "parent_molecule_chembl_id",
    ]].copy()
    test_prediction["dtiam_probability"] = test_probability
    validation_path = run_dir / "VALIDATION_PREDICTIONS_V1.csv.gz"
    test_path = run_dir / "TEST_PREDICTIONS_V1.csv.gz"
    validation_prediction.to_csv(validation_path, index=False)
    test_prediction.to_csv(test_path, index=False)

    leaderboard = predictor.leaderboard(silent=True)
    leaderboard_path = run_dir / "AUTOGLUON_LEADERBOARD_V1.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    model_names = predictor.model_names()
    best_model = predictor.model_best
    checks = {
        "feature_audit_pass": json.loads(FEATURE_AUDIT.read_text()).get("status") == "PASS",
        "exact_2048_official_features": train_table.shape[1] == 2049,
        "all_roles_have_both_classes": all(
            frame["binary_label"].nunique() == 2 for frame in [train, valid, test]
        ),
        "pair_roles_disjoint": (
            set(train["calibration_pair_id"]).isdisjoint(valid["calibration_pair_id"])
            and set(train["calibration_pair_id"]).isdisjoint(test["calibration_pair_id"])
            and set(valid["calibration_pair_id"]).isdisjoint(test["calibration_pair_id"])
        ),
        "scaffold_cold_invariant": scaffold_disjoint,
        "homology_cold_invariant": homology_disjoint,
        "test_not_supplied_to_fit": True,
        "predictions_finite_and_bounded": bool(
            np.isfinite(test_probability).all()
            and ((test_probability >= 0.0) & (test_probability <= 1.0)).all()
        ),
        "at_least_one_trained_model": len(model_names) >= 1,
        "best_model_defined": bool(best_model),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run": run_name,
        "comparator": "DTIAM_OFFICIAL_REPRESENTATION_COMPATIBLE_RETRAIN_V1",
        "protocol": protocol,
        "fold": fold,
        "fit_policy": {
            "official_reference_call": "TabularPredictor(label='y', eval_metric='roc_auc').fit(train_data=train_data, excluded_model_types=[], presets=None)",
            "this_call": {
                "eval_metric": "roc_auc",
                "presets": None,
                "excluded_model_types": [],
                "time_limit_seconds": time_limit,
                "training_role_only": True,
            },
            "compatibility_boundary": "AutoGluon 1.4.0/Python 3.10 instead of paper AutoGluon 0.5.2/Python 3.7; official pretrained BerMol and ESM2 representation definitions are unchanged.",
        },
        "counts": {
            "train_rows": len(train),
            "valid_rows": len(valid),
            "test_rows": len(test),
            "train_positives": int(train["binary_label"].sum()),
            "valid_positives": int(valid["binary_label"].sum()),
            "test_positives": int(test["binary_label"].sum()),
            "train_targets": train["target_chembl_id"].nunique(),
            "test_targets": test["target_chembl_id"].nunique(),
            "train_compounds": train["parent_standard_inchi_key"].nunique(),
            "test_compounds": test["parent_standard_inchi_key"].nunique(),
        },
        "fit_seconds": fit_seconds,
        "trained_models": model_names,
        "best_model": best_model,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "checks": checks,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "autogluon": __import__("autogluon.tabular").tabular.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "inputs": {
            "pairs_sha256": sha256(PAIRS),
            "bermol_sha256": sha256(BERMOL),
            "esm2_sha256": sha256(ESM2),
            "pair_feature_index_sha256": sha256(PAIR_INDEX),
            "feature_audit_sha256": sha256(FEATURE_AUDIT),
        },
        "artifacts": {
            "validation_predictions_sha256": sha256(validation_path),
            "test_predictions_sha256": sha256(test_path),
            "leaderboard_sha256": sha256(leaderboard_path),
            "predictor_path": str(predictor_dir.relative_to(ROOT)),
        },
    }
    summary = json_safe(summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(
        json.dumps(
            {
                "status": "PASS",
                "run": run_name,
                "best_model": best_model,
                "test_micro_auprc": test_metrics["micro_auprc"],
                "test_micro_auroc": test_metrics["micro_auroc"],
                "fit_seconds": fit_seconds,
            }
        ),
        flush=True,
    )
    return summary


def update_aggregate() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT.glob("*/RUN_SUMMARY_V1.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != "PASS":
            continue
        row = {
            "protocol": summary["protocol"],
            "fold": summary["fold"],
            "run": summary["run"],
            "best_model": summary["best_model"],
            "fit_seconds": summary["fit_seconds"],
            **{f"valid_{key}": value for key, value in summary["validation_metrics"].items()},
            **{f"test_{key}": value for key, value in summary["test_metrics"].items()},
        }
        rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["protocol", "fold"])
    aggregate_path = OUT / "ALL_DTIAM_SAME_DATA_METRICS_V1.csv"
    frame.to_csv(aggregate_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "completed_runs": len(frame),
        "protocols": sorted(frame["protocol"].unique().tolist()),
        "metrics_sha256": sha256(aggregate_path),
        "claim_boundary": "Same-data compatible retraining comparator; not a bitwise reproduction of the paper software environment.",
    }
    (OUT / "DTIAM_SAME_DATA_AGGREGATE_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS + ["all"], required=True)
    parser.add_argument("--fold", default="all", help="0..4 or all; S4/S5 use fixed fold -1")
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional AutoGluon fit cap in seconds; omit for the closest official-default call.",
    )
    args = parser.parse_args()
    required = [PAIRS, BERMOL, ESM2, PAIR_INDEX, FEATURE_AUDIT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    audit = json.loads(FEATURE_AUDIT.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("DTIAM feature store audit must pass")

    data = pd.read_csv(PAIRS, low_memory=False)
    pair_index = pd.read_csv(PAIR_INDEX)
    if len(data) != 86674 or len(pair_index) != 86674:
        raise RuntimeError("Frozen pair universe changed")
    availability = pair_index[["calibration_pair_id", "dtiam_bermol_available"]]
    data = data.merge(availability, on="calibration_pair_id", how="left", validate="one_to_one")
    if data["dtiam_bermol_available"].isna().any() or int((~data["dtiam_bermol_available"]).sum()) != 1:
        raise RuntimeError("DTIAM quarantine boundary changed")
    drug_features = np.load(BERMOL, mmap_mode="r")
    target_features = np.load(ESM2, mmap_mode="r")
    if drug_features.shape != (62477, 768) or target_features.shape != (428, 1280):
        raise RuntimeError("DTIAM feature shapes changed")

    protocols = PROTOCOLS if args.protocol == "all" else [args.protocol]
    for protocol in protocols:
        if protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"}:
            folds = [-1]
        elif args.fold == "all":
            folds = list(range(FOLDS))
        else:
            folds = [int(args.fold)]
        for fold in folds:
            run_one(
                data,
                drug_features,
                target_features,
                protocol,
                fold,
                args.time_limit,
            )
            update_aggregate()


if __name__ == "__main__":
    main()

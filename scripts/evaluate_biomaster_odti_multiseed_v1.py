#!/usr/bin/env python3
"""Aggregate five model seeds and test frozen cold-regime ensembles."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
RUNS = BASE / "biomaster_odti_routed_ranker_v1"
STACKS = RUNS / "cold_regime_stack_v1"
BASELINES = BASE / "baseline_results_v1"
PAIR_META = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
SCOPE_S5 = (
    BASE / "public_baselines_v1/scope_dti_old_drug_entity_cold_v1"
    / "SCOPE_DTI_OLD_DRUG_ENTITY_COLD_PREDICTIONS_V1.csv.gz"
)
OUT = RUNS / "multiseed_evaluation_v1"
SEEDS = [20260813, 20260814, 20260815, 20260816, 20260817]
PROTOCOL_FOLDS = {
    "S2_HOMOLOGY_COLD_TARGET": list(range(5)),
    "S3_STRICT_DOUBLE_COLD": list(range(5)),
    "S5_OLD_DRUG_ENTITY_COLD": [-1],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_frame(protocol: str, fold: int, seed: int, meta: pd.DataFrame) -> pd.DataFrame:
    name = f"{protocol}__fold_{fold}__seed_{seed}__CORE"
    run_dir = RUNS / name
    summary_path = run_dir / "RUN_SUMMARY_V1.json"
    raw_path = run_dir / "TEST_PREDICTIONS_V1.csv.gz"
    stack_path = STACKS / name / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
    baseline_partition = "fixed_split" if protocol == "S5_OLD_DRUG_ENTITY_COLD" else f"fold_{fold}"
    baseline_path = BASELINES / protocol / baseline_partition / "BASELINE_TEST_PREDICTIONS_V1.csv.gz"
    required = [summary_path, raw_path, stack_path, baseline_path]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS" or summary.get("variant") != "core":
        raise RuntimeError(f"Invalid run summary: {summary_path}")
    raw = pd.read_csv(
        raw_path,
        usecols=[
            "calibration_pair_id", "binary_label", "target_chembl_id",
            "parent_standard_inchi_key", "biomaster_probability_calibrated",
        ],
    )
    stack = pd.read_csv(
        stack_path,
        usecols=["calibration_pair_id", "COLD_REGIME_ROUTED_STACK"],
    )
    baseline = pd.read_csv(
        baseline_path,
        usecols=[
            "calibration_pair_id", "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        ],
    )
    result = raw.merge(stack, on="calibration_pair_id", validate="one_to_one")
    result = result.merge(baseline, on="calibration_pair_id", validate="one_to_one")
    result = result.merge(meta, on="calibration_pair_id", how="left", validate="one_to_one")
    if result["target_homology_cluster"].isna().any():
        raise RuntimeError(f"Missing homology metadata in {name}")
    result["seed"] = seed
    result["fold"] = fold
    return result


def add_metric_rows(
    destination: list[dict[str, object]],
    frame: pd.DataFrame,
    protocol: str,
    evaluation: str,
    seed: int,
) -> None:
    score_columns = {
        "BIOMASTER_RAW_CORE": "biomaster_probability_calibrated",
        "BIOMASTER_COLD_REGIME_ROUTED_STACK": "COLD_REGIME_ROUTED_STACK",
        "CONPLEX_FROZEN_EXTERNAL": "CONPLEX_FROZEN_EXTERNAL",
        "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO": "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
    }
    if "scope_mean" in frame:
        score_columns["SCOPE_PUBLIC_CHECKPOINT_ENSEMBLE"] = "scope_mean"
    for model, column in score_columns.items():
        part = frame.dropna(subset=[column]).copy()
        row: dict[str, object] = {
            "protocol": protocol,
            "evaluation": evaluation,
            "seed": seed,
            "model": model,
            "score_column": column,
            "available_rows": int(len(part)),
        }
        row.update(metrics(part, part[column].to_numpy(dtype=np.float64)))
        destination.append(row)


def main() -> None:
    if not SCOPE_S5.is_file():
        raise FileNotFoundError(SCOPE_S5)
    meta = pd.read_csv(
        PAIR_META,
        usecols=["calibration_pair_id", "target_homology_cluster"],
        low_memory=False,
    )
    scope = pd.read_csv(SCOPE_S5, usecols=["calibration_pair_id", "scope_mean"])
    metric_rows: list[dict[str, object]] = []
    ensemble_frames: dict[str, pd.DataFrame] = {}
    total_runs = 0
    for protocol, folds in PROTOCOL_FOLDS.items():
        all_seed_frames: list[pd.DataFrame] = []
        for seed in SEEDS:
            seed_folds = [run_frame(protocol, fold, seed, meta) for fold in folds]
            seed_frame = pd.concat(seed_folds, ignore_index=True)
            if not seed_frame["calibration_pair_id"].is_unique:
                raise RuntimeError(f"OOF overlap for {protocol}, seed {seed}")
            if protocol == "S5_OLD_DRUG_ENTITY_COLD":
                seed_frame = seed_frame.merge(
                    scope, on="calibration_pair_id", how="left", validate="one_to_one"
                )
                if seed_frame["scope_mean"].isna().any():
                    raise RuntimeError("SCOPE S5 coverage changed")
            add_metric_rows(metric_rows, seed_frame, protocol, "PER_SEED_POOLED_OOF", seed)
            all_seed_frames.append(seed_frame)
            total_runs += len(folds)
        long = pd.concat(all_seed_frames, ignore_index=True)
        constant_columns = [
            "binary_label", "target_chembl_id", "parent_standard_inchi_key",
            "target_homology_cluster", "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        ]
        if protocol == "S5_OLD_DRUG_ENTITY_COLD":
            constant_columns.append("scope_mean")
        aggregate_spec: dict[str, str | list[str]] = {
            column: "first" for column in constant_columns
        }
        aggregate_spec.update({
            "biomaster_probability_calibrated": ["mean", "std"],
            "COLD_REGIME_ROUTED_STACK": ["mean", "std"],
        })
        ensemble = long.groupby("calibration_pair_id", as_index=False).agg(aggregate_spec)
        ensemble.columns = [
            column if isinstance(column, str) else "_".join(part for part in column if part)
            for column in ensemble.columns
        ]
        # Pandas returns tuples for mixed scalar/list aggregation; normalize all
        # names to a stable schema independent of its minor-version formatting.
        ensemble = ensemble.rename(columns={
            "calibration_pair_id_": "calibration_pair_id",
            "binary_label_first": "binary_label",
            "target_chembl_id_first": "target_chembl_id",
            "parent_standard_inchi_key_first": "parent_standard_inchi_key",
            "target_homology_cluster_first": "target_homology_cluster",
            "CONPLEX_FROZEN_EXTERNAL_first": "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO_first": "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
            "scope_mean_first": "scope_mean",
            "biomaster_probability_calibrated_mean": "BIOMASTER_RAW_FIVE_SEED_MEAN",
            "biomaster_probability_calibrated_std": "BIOMASTER_RAW_FIVE_SEED_STD",
            "COLD_REGIME_ROUTED_STACK_mean": "BIOMASTER_STACK_FIVE_SEED_MEAN",
            "COLD_REGIME_ROUTED_STACK_std": "BIOMASTER_STACK_FIVE_SEED_STD",
        })
        required_ensemble = [
            "calibration_pair_id", "binary_label", "target_chembl_id",
            "parent_standard_inchi_key", "target_homology_cluster",
            "BIOMASTER_RAW_FIVE_SEED_MEAN", "BIOMASTER_STACK_FIVE_SEED_MEAN",
        ]
        absent = [column for column in required_ensemble if column not in ensemble]
        if absent:
            raise RuntimeError(f"Aggregate schema failure for {protocol}: {absent}; {ensemble.columns.tolist()}")
        ensemble_frames[protocol] = ensemble
        ensemble_models = {
            "BIOMASTER_RAW_FIVE_SEED_ENSEMBLE": "BIOMASTER_RAW_FIVE_SEED_MEAN",
            "BIOMASTER_STACK_FIVE_SEED_ENSEMBLE": "BIOMASTER_STACK_FIVE_SEED_MEAN",
            "CONPLEX_FROZEN_EXTERNAL": "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO": "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        }
        if "scope_mean" in ensemble:
            ensemble_models["SCOPE_PUBLIC_CHECKPOINT_ENSEMBLE"] = "scope_mean"
        for model, score_column in ensemble_models.items():
            row: dict[str, object] = {
                "protocol": protocol,
                "evaluation": "FIVE_SEED_ENSEMBLE_POOLED_OOF",
                "seed": -1,
                "model": model,
                "score_column": score_column,
                "available_rows": int(len(ensemble)),
            }
            row.update(metrics(ensemble, ensemble[score_column].to_numpy(dtype=np.float64)))
            metric_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_path = OUT / "MULTISEED_METRICS_V1.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    variability = (
        metrics_frame[metrics_frame["evaluation"].eq("PER_SEED_POOLED_OOF")]
        .groupby(["protocol", "model"])[
            ["micro_auroc", "micro_auprc", "target_macro_auprc", "drug_macro_auprc"]
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    variability.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple) else str(column)
        for column in variability.columns
    ]
    variability_path = OUT / "MULTISEED_VARIABILITY_V1.csv"
    variability.to_csv(variability_path, index=False)

    bootstrap_rows: list[dict[str, object]] = []
    for protocol, ensemble in ensemble_frames.items():
        cluster = (
            "parent_standard_inchi_key"
            if protocol == "S5_OLD_DRUG_ENTITY_COLD"
            else "target_homology_cluster"
        )
        references = ["CONPLEX_FROZEN_EXTERNAL", "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"]
        if protocol == "S5_OLD_DRUG_ENTITY_COLD":
            references.append("scope_mean")
        for reference in references:
            for metric_name in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    ensemble,
                    "BIOMASTER_STACK_FIVE_SEED_MEAN",
                    reference,
                    metric_name,
                    2000,
                    20260813 + len(bootstrap_rows),
                    cluster,
                )
                row["protocol"] = protocol
                bootstrap_rows.append(row)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap_path = OUT / "MULTISEED_PAIRED_CLUSTER_BOOTSTRAP_V1.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    for protocol, ensemble in ensemble_frames.items():
        ensemble.to_csv(OUT / f"{protocol}_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz", index=False)

    primary = {}
    for protocol in PROTOCOL_FOLDS:
        reference = "scope_mean" if protocol == "S5_OLD_DRUG_ENTITY_COLD" else "CONPLEX_FROZEN_EXTERNAL"
        selected = bootstrap[
            bootstrap["protocol"].eq(protocol)
            & bootstrap["reference"].eq(reference)
            & bootstrap["metric"].eq("auprc")
        ].iloc[0]
        primary[protocol] = selected.to_dict()
    checks = {
        "exact_55_core_model_runs": total_runs == 55,
        "five_seed_metrics_each_protocol_model": (
            metrics_frame[metrics_frame["evaluation"].eq("PER_SEED_POOLED_OOF")]
            .groupby(["protocol", "model"])["seed"].nunique().eq(5).all()
        ),
        "all_metric_values_finite": np.isfinite(
            metrics_frame[["micro_auroc", "micro_auprc", "brier", "ece_15"]]
        ).all().all(),
        "all_bootstraps_complete": len(bootstrap) == 14,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_seeds": SEEDS,
        "core_model_runs": total_runs,
        "ensemble_rule": "Arithmetic mean of five independently trained frozen-protocol seed predictions",
        "stack_fit_boundary": "Each seed stack fit only on that seed's frozen validation predictions",
        "primary_auprc_comparisons": primary,
        "checks": {key: bool(value) for key, value in checks.items()},
        "artifacts": {
            "metrics_sha256": sha256(metrics_path),
            "variability_sha256": sha256(variability_path),
            "bootstrap_sha256": sha256(bootstrap_path),
        },
        "claim_status": "MULTI_SEED_INTERNAL_EVIDENCE; SCOPE SOURCE_OVERLAP REMAINS UNRESOLVED",
    }
    summary_path = OUT / "MULTISEED_EVALUATION_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

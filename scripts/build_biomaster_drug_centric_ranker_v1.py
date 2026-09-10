#!/usr/bin/env python3
"""Build and audit the drug-to-target production ranker.

The script keeps three claims separate:

1. ``independent_validation_rank`` is selected on frozen S5 validation without
   any DTIAM feature and evaluated once on the drug-entity-cold S5 test role.
2. ``independent_borda`` combines two BioMaster-owned frozen rankers (the
   validation-routed BioMaster stack and leakage-safe graph ranker) with an
   equal-weight within-drug Borda rule.  It does not require DTIAM at inference.
3. ``system_borda`` is the strongest deployment system.  It combines the
   S5-validation-selected DTIAM/BioMaster rank score, the existing routed
   DTIAM/BioMaster score, and the graph ranker using the same fixed Borda rule.

KIRHub is evaluation-only, but it has been inspected during earlier project
iterations.  Its results are therefore explicitly labelled post-audit rather
than used as a fresh confirmatory claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_biomaster_odti_baselines_v1 import (  # noqa: E402
    max_positive_tanimoto,
    metrics,
    split_masks,
    target_prior,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = BASE / "feature_store_v1/MORGAN2048_UINT8_V1.npy"
DTIAM_S5 = (
    BASE
    / "public_retrained_v1/dtiam_same_data_compatible_v1"
    / "S5_OLD_DRUG_ENTITY_COLD__fold_-1__OFFICIAL_DEFAULT_COMPAT_V1"
)
BIOMASTER_RUNS = BASE / "biomaster_odti_routed_ranker_v1"
BIOMASTER_S5_ENSEMBLE = (
    BIOMASTER_RUNS
    / "multiseed_evaluation_v1"
    / "S5_OLD_DRUG_ENTITY_COLD_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz"
)
DTIAM_DEPLOYMENT = (
    BASE
    / "public_retrained_v1/dtiam_720x384_deployment_v1"
    / "DTIAM_720X384_SCORES_V1.csv.gz"
)
BIOMASTER_DEPLOYMENT = (
    BASE
    / "biomaster_odti_deployment_v1"
    / "BIOMASTER_ODTI_720X384_SCORES_V1.csv.gz"
)
ROUTED_DEPLOYMENT = (
    BASE
    / "public_retrained_v1/dtiam_biomaster_scope_deployment_v1"
    / "DTIAM_BIOMASTER_SCOPE_720X384_V1.csv.gz"
)
V10_DEPLOYMENT = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1"
    / "leakage_safe_ranker_v10/OLD_DRUG_LEAKAGE_SAFE_ALL_276480_V10.csv.gz"
)
FULL_FIT_V6_DEPLOYMENT = (
    ROOT
    / "outputs/biomaster_bidirectional_v6_720x384"
    / "BIDIRECTIONAL_V6_FULL_FIT_720X384_SCORES.csv.gz"
)
FULL_FIT_V6_SUMMARY = (
    ROOT
    / "outputs/biomaster_bidirectional_v6_720x384"
    / "BIDIRECTIONAL_V6_FULL_FIT_720X384_SUMMARY.json"
)
KIRHUB = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1"
    / "leakage_safe_ranker_v10/external_evaluation"
    / "OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_V10.csv"
)
DEFAULT_OUT = BASE / "drug_centric_ranker_v1"
SEEDS = (20260813, 20260814, 20260815, 20260816, 20260817)
QUERY_COLUMN = "parent_standard_inchi_key"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_logit(values: Iterable[float]) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability))


def within_query_percentile(
    frame: pd.DataFrame, score_column: str, query_column: str
) -> pd.Series:
    """Return a 0--1 percentile with larger scores always better."""

    def transform(part: pd.Series) -> pd.Series:
        if len(part) == 1:
            return pd.Series(np.ones(1), index=part.index)
        rank = part.rank(method="average", ascending=True)
        return (rank - 1.0) / (len(part) - 1.0)

    return frame.groupby(query_column, sort=False)[score_column].transform(transform)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered.astype(bool)].mean())


def query_metric_table(
    frame: pd.DataFrame,
    score: np.ndarray,
    query_column: str = QUERY_COLUMN,
    ks: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    work = frame[[query_column, "binary_label"]].copy()
    work["score"] = np.asarray(score, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for query, part in work.groupby(query_column, sort=False):
        labels = part["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        values = part["score"].to_numpy(dtype=np.float64)
        order = np.argsort(-values, kind="stable")
        positives = int(labels.sum())
        row: dict[str, Any] = {
            query_column: str(query),
            # Use sklearn here so bootstrap point estimates use exactly the
            # same tie convention as the repository-wide metric function.
            "average_precision": float(average_precision_score(labels, values)),
        }
        for k in ks:
            top = labels[order[: min(k, len(labels))]]
            row[f"recall_at_{k}"] = float(top.sum() / positives)
            discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = float((top * discounts).sum())
            ideal_n = min(positives, len(top))
            idcg = float(discounts[:ideal_n].sum())
            row[f"ndcg_at_{k}"] = dcg / idcg if idcg else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def deterministic_query_fold(query: str, folds: int = 5) -> int:
    value = int(hashlib.sha256(str(query).encode("utf-8")).hexdigest()[:16], 16)
    return value % folds


def select_robust_linear_ranker(
    frame: pd.DataFrame,
    standardized: np.ndarray,
    weights: Iterable[tuple[float, ...]],
    feature_names: list[str],
) -> dict[str, Any]:
    """Select weights by mean drug AP minus across-fold instability."""

    if standardized.shape != (len(frame), len(feature_names)):
        raise ValueError("standardized feature matrix does not match frame/features")
    query_parts = []
    for query, index in frame.groupby(QUERY_COLUMN, sort=False).indices.items():
        positions = np.asarray(index, dtype=np.int64)
        labels = frame.iloc[positions]["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() != labels.max():
            query_parts.append(
                (str(query), positions, labels, deterministic_query_fold(str(query)))
            )
    if not query_parts:
        raise RuntimeError("validation contains no two-class drug queries")

    best: dict[str, Any] | None = None
    for raw_weight in weights:
        weight = np.asarray(raw_weight, dtype=np.float64)
        if weight.shape != (len(feature_names),):
            raise ValueError("weight width does not match feature width")
        score = standardized @ weight
        fold_values: list[list[float]] = [[] for _ in range(5)]
        for _, positions, labels, fold in query_parts:
            fold_values[fold].append(
                binary_average_precision(labels, score[positions])
            )
        fold_means = np.asarray(
            [np.mean(values) for values in fold_values if values], dtype=np.float64
        )
        if len(fold_means) < 2:
            raise RuntimeError("robust selection requires at least two populated folds")
        mean = float(fold_means.mean())
        std = float(fold_means.std(ddof=0))
        objective = mean - std
        candidate = {
            "weights": dict(zip(feature_names, weight.tolist(), strict=True)),
            "objective_mean_minus_fold_std": objective,
            "fold_mean_drug_auprc": fold_means.tolist(),
            "mean_fold_drug_auprc": mean,
            "fold_std": std,
            "worst_fold": float(fold_means.min()),
        }
        key = (
            objective,
            mean,
            -std,
            -float(np.abs(weight[1:]).sum()),
        )
        if best is None or key > best["_selection_key"]:
            best = {**candidate, "_selection_key": key}
    assert best is not None
    best.pop("_selection_key")
    return best


def paired_query_bootstrap(
    frame: pd.DataFrame,
    candidate_score: np.ndarray,
    reference_score: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    candidate = query_metric_table(frame, candidate_score)
    reference = query_metric_table(frame, reference_score)
    paired = candidate.merge(
        reference,
        on=QUERY_COLUMN,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(iterations, len(paired)))
    results: dict[str, Any] = {
        "query_column": QUERY_COLUMN,
        "queries_with_both_classes": int(len(paired)),
        "iterations": int(iterations),
        "seed": int(seed),
        "metrics": {},
    }
    for metric in [
        "average_precision",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_20",
    ]:
        difference = (
            paired[f"{metric}_candidate"].to_numpy(dtype=np.float64)
            - paired[f"{metric}_reference"].to_numpy(dtype=np.float64)
        )
        distribution = difference[indices].mean(axis=1)
        low, high = np.quantile(distribution, [0.025, 0.975])
        results["metrics"][metric] = {
            "candidate": float(paired[f"{metric}_candidate"].mean()),
            "reference": float(paired[f"{metric}_reference"].mean()),
            "observed_difference": float(difference.mean()),
            "difference_ci95_low": float(low),
            "difference_ci95_high": float(high),
            "probability_difference_gt_zero": float((distribution > 0).mean()),
            "ci95_excludes_zero_in_favor_of_candidate": bool(low > 0),
        }
    return results


def biomaster_validation_predictions() -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for seed in SEEDS:
        path = (
            BIOMASTER_RUNS
            / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}__CORE"
            / "VALIDATION_PREDICTIONS_V1.csv.gz"
        )
        current = pd.read_csv(
            path, usecols=["calibration_pair_id", "biomaster_logit"]
        ).rename(columns={"biomaster_logit": f"biomaster_logit_{seed}"})
        merged = (
            current
            if merged is None
            else merged.merge(current, on="calibration_pair_id", validate="one_to_one")
        )
    assert merged is not None
    columns = [f"biomaster_logit_{seed}" for seed in SEEDS]
    merged["biomaster_logit"] = merged[columns].mean(axis=1)
    return merged[["calibration_pair_id", "biomaster_logit"]]


def biomaster_test_predictions() -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for seed in SEEDS:
        path = (
            BIOMASTER_RUNS
            / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}__CORE"
            / "TEST_PREDICTIONS_V1.csv.gz"
        )
        current = pd.read_csv(
            path, usecols=["calibration_pair_id", "biomaster_logit"]
        ).rename(columns={"biomaster_logit": f"biomaster_logit_{seed}"})
        merged = (
            current
            if merged is None
            else merged.merge(current, on="calibration_pair_id", validate="one_to_one")
        )
    assert merged is not None
    columns = [f"biomaster_logit_{seed}" for seed in SEEDS]
    merged["biomaster_logit"] = merged[columns].mean(axis=1)
    stack = pd.read_csv(
        BIOMASTER_S5_ENSEMBLE,
        usecols=["calibration_pair_id", "BIOMASTER_STACK_FIVE_SEED_MEAN"],
    )
    return merged[["calibration_pair_id", "biomaster_logit"]].merge(
        stack, on="calibration_pair_id", validate="one_to_one"
    )


def frozen_s5_roles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(PAIRS, low_memory=False)
    fingerprints = np.load(MORGAN, mmap_mode="r")
    masks = split_masks(data, "S5_OLD_DRUG_ENTITY_COLD", -1)
    available = data["drug_feature_available"].to_numpy(dtype=bool)
    train, valid, test = [
        data.loc[masks[role] & available].reset_index(drop=True)
        for role in ["train", "valid", "test"]
    ]
    if [len(train), len(valid), len(test)] != [65276, 16668, 2556]:
        raise RuntimeError("frozen S5 role sizes changed")
    positive_targets = set(
        train.loc[train["binary_label"].eq(1), "target_chembl_id"].astype(str)
    )
    for frame in (valid, test):
        frame["train_positive_max_tanimoto"] = max_positive_tanimoto(
            train, frame, fingerprints
        )[0]
        frame["target_train_prior"] = target_prior(train, frame)
        frame["target_has_train_positive_pool"] = (
            frame["target_chembl_id"].astype(str).isin(positive_targets).astype(np.int8)
        )
    return train, valid, test


def standardization(
    valid: pd.DataFrame, columns: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    mean = valid[columns].mean(axis=0)
    scale = valid[columns].std(axis=0, ddof=0).replace(0.0, 1.0)
    return mean.to_dict(), scale.to_dict()


def apply_linear_ranker(
    frame: pd.DataFrame,
    feature_names: list[str],
    means: dict[str, float],
    scales: dict[str, float],
    weights: dict[str, float],
) -> np.ndarray:
    matrix = np.column_stack(
        [
            (frame[name].to_numpy(dtype=np.float64) - means[name]) / scales[name]
            for name in feature_names
        ]
    )
    vector = np.asarray([weights[name] for name in feature_names], dtype=np.float64)
    return matrix @ vector


def metric_row(name: str, frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    return {"model": name, **metrics(frame, np.asarray(score, dtype=np.float64))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--full-fit-deployment",
        default=str(FULL_FIT_V6_DEPLOYMENT),
        help="Directional FULL_FIT 720x384 score table to merge.",
    )
    parser.add_argument(
        "--full-fit-summary",
        default=str(FULL_FIT_V6_SUMMARY),
        help="Audit summary corresponding to --full-fit-deployment.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("bootstrap iterations must be at least 1000")
    full_fit_deployment_path = Path(args.full_fit_deployment).resolve()
    full_fit_summary_path = Path(args.full_fit_summary).resolve()
    required = [
        PAIRS,
        MORGAN,
        DTIAM_S5 / "VALIDATION_PREDICTIONS_V1.csv.gz",
        DTIAM_S5 / "TEST_PREDICTIONS_V1.csv.gz",
        BIOMASTER_S5_ENSEMBLE,
        DTIAM_DEPLOYMENT,
        BIOMASTER_DEPLOYMENT,
        ROUTED_DEPLOYMENT,
        V10_DEPLOYMENT,
        full_fit_deployment_path,
        full_fit_summary_path,
        KIRHUB,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    train, valid, test = frozen_s5_roles()
    dtiam_valid = pd.read_csv(
        DTIAM_S5 / "VALIDATION_PREDICTIONS_V1.csv.gz",
        usecols=["calibration_pair_id", "dtiam_probability"],
    )
    dtiam_test = pd.read_csv(
        DTIAM_S5 / "TEST_PREDICTIONS_V1.csv.gz",
        usecols=["calibration_pair_id", "dtiam_probability"],
    )
    valid = valid.merge(dtiam_valid, on="calibration_pair_id", validate="one_to_one")
    valid = valid.merge(
        biomaster_validation_predictions(),
        on="calibration_pair_id",
        validate="one_to_one",
    )
    test = test.merge(dtiam_test, on="calibration_pair_id", validate="one_to_one")
    test = test.merge(
        biomaster_test_predictions(), on="calibration_pair_id", validate="one_to_one"
    )
    valid["dtiam_logit"] = bounded_logit(valid["dtiam_probability"])
    test["dtiam_logit"] = bounded_logit(test["dtiam_probability"])

    independent_features = [
        "biomaster_logit",
        "train_positive_max_tanimoto",
        "target_train_prior",
    ]
    independent_mean, independent_scale = standardization(valid, independent_features)
    independent_z = np.column_stack(
        [
            (valid[name].to_numpy(dtype=np.float64) - independent_mean[name])
            / independent_scale[name]
            for name in independent_features
        ]
    )
    independent_grid = itertools.product(
        [1.0],
        np.arange(0.0, 3.0001, 0.25).tolist(),
        np.arange(-0.5, 0.5001, 0.25).tolist(),
    )
    independent_selection = select_robust_linear_ranker(
        valid, independent_z, independent_grid, independent_features
    )
    test["independent_validation_rank_score"] = apply_linear_ranker(
        test,
        independent_features,
        independent_mean,
        independent_scale,
        independent_selection["weights"],
    )

    system_features = [
        "dtiam_logit",
        "biomaster_logit",
        "train_positive_max_tanimoto",
        "target_train_prior",
    ]
    system_mean, system_scale = standardization(valid, system_features)
    system_z = np.column_stack(
        [
            (valid[name].to_numpy(dtype=np.float64) - system_mean[name])
            / system_scale[name]
            for name in system_features
        ]
    )
    system_grid = itertools.product(
        [1.0],
        np.arange(0.0, 1.5001, 0.25).tolist(),
        np.arange(0.0, 2.5001, 0.25).tolist(),
        np.arange(-0.5, 0.5001, 0.25).tolist(),
    )
    system_selection = select_robust_linear_ranker(
        valid, system_z, system_grid, system_features
    )
    test["system_validation_rank_score"] = apply_linear_ranker(
        test,
        system_features,
        system_mean,
        system_scale,
        system_selection["weights"],
    )

    s5_metrics = [
        metric_row("DTIAM_RAW", test, test["dtiam_probability"].to_numpy()),
        metric_row(
            "BIOMASTER_RAW_FIVE_SEED",
            test,
            test["biomaster_logit"].to_numpy(),
        ),
        metric_row(
            "BIOMASTER_STACK_FIVE_SEED",
            test,
            test["BIOMASTER_STACK_FIVE_SEED_MEAN"].to_numpy(),
        ),
        metric_row(
            "INDEPENDENT_VALIDATION_RANK",
            test,
            test["independent_validation_rank_score"].to_numpy(),
        ),
        metric_row(
            "SYSTEM_VALIDATION_RANK",
            test,
            test["system_validation_rank_score"].to_numpy(),
        ),
    ]
    s5_independent_bootstrap = paired_query_bootstrap(
        test,
        test["independent_validation_rank_score"].to_numpy(),
        test["dtiam_probability"].to_numpy(),
        args.bootstrap_iterations,
        args.seed,
    )
    s5_system_bootstrap = paired_query_bootstrap(
        test,
        test["system_validation_rank_score"].to_numpy(),
        test["dtiam_probability"].to_numpy(),
        args.bootstrap_iterations,
        args.seed + 1,
    )

    deployment = pd.read_csv(BIOMASTER_DEPLOYMENT, low_memory=False)
    deployment["biomaster_logit"] = deployment[
        "biomaster_ensemble_logit"
    ].to_numpy(dtype=np.float64)
    dtiam_deployment = pd.read_csv(
        DTIAM_DEPLOYMENT,
        usecols=["pairId", "dtiam_probability", "dtiam_logit"],
    ) if "dtiam_logit" in pd.read_csv(DTIAM_DEPLOYMENT, nrows=0).columns else pd.read_csv(
        DTIAM_DEPLOYMENT, usecols=["pairId", "dtiam_probability"]
    )
    if "dtiam_logit" not in dtiam_deployment:
        dtiam_deployment["dtiam_logit"] = bounded_logit(
            dtiam_deployment["dtiam_probability"]
        )
    routed = pd.read_csv(
        ROUTED_DEPLOYMENT,
        usecols=["pairId", "dtiam_biomaster_same_data_score"],
    )
    v10 = pd.read_csv(
        V10_DEPLOYMENT,
        usecols=["pairId", "old_drug_leakage_safe_score_v10"],
    )
    full_fit_summary = json.loads(full_fit_summary_path.read_text())
    if (
        full_fit_summary.get("status") != "PASS"
        or full_fit_summary.get("rows") != 720 * 384
        or len(full_fit_summary.get("checkpoints", [])) != 3
    ):
        raise RuntimeError("FULL_FIT directional checkpoint ensemble audit failed")
    full_fit_v6 = pd.read_csv(
        full_fit_deployment_path,
        usecols=[
            "pairId",
            "ensemble_pair_logit",
            "ensemble_drug_to_target_logit",
        ],
    )
    deployment = deployment.drop(
        columns=["old_drug_leakage_safe_score_v10"], errors="ignore"
    ).merge(dtiam_deployment, on="pairId", validate="one_to_one")
    deployment = deployment.merge(routed, on="pairId", validate="one_to_one")
    deployment = deployment.merge(v10, on="pairId", validate="one_to_one")
    deployment = deployment.merge(full_fit_v6, on="pairId", validate="one_to_one")
    if len(deployment) != 720 * 384:
        raise RuntimeError(f"deployment grid changed: {len(deployment)}")
    deployment["independent_validation_rank_score"] = apply_linear_ranker(
        deployment,
        independent_features,
        independent_mean,
        independent_scale,
        independent_selection["weights"],
    )
    deployment["system_validation_rank_score"] = apply_linear_ranker(
        deployment,
        system_features,
        system_mean,
        system_scale,
        system_selection["weights"],
    )
    percentile_sources = [
        "independent_validation_rank_score",
        "system_validation_rank_score",
        "biomaster_routed_stack_score",
        "dtiam_biomaster_same_data_score",
        "old_drug_leakage_safe_score_v10",
        "ensemble_drug_to_target_logit",
    ]
    for column in percentile_sources:
        deployment[f"{column}_percentile"] = within_query_percentile(
            deployment, column, "ligand_inchikey"
        )
    deployment["biomaster_independent_borda_score"] = deployment[
        [
            "biomaster_routed_stack_score_percentile",
            "old_drug_leakage_safe_score_v10_percentile",
        ]
    ].mean(axis=1)
    deployment["biomaster_independent_new_head_borda_score"] = deployment[
        [
            "independent_validation_rank_score_percentile",
            "old_drug_leakage_safe_score_v10_percentile",
        ]
    ].mean(axis=1)
    deployment["biomaster_system_borda_score"] = deployment[
        [
            "system_validation_rank_score_percentile",
            "dtiam_biomaster_same_data_score_percentile",
            "old_drug_leakage_safe_score_v10_percentile",
        ]
    ].mean(axis=1)
    # This diagnostic is deliberately not promoted: it tests whether the
    # FULL_FIT directional neural head and graph ranker are complementary
    # under the same fixed equal-Borda rule.
    deployment["biomaster_full_fit_v10_borda_score"] = deployment[
        [
            "ensemble_drug_to_target_logit_percentile",
            "old_drug_leakage_safe_score_v10_percentile",
        ]
    ].mean(axis=1)
    for column in [
        "old_drug_leakage_safe_score_v10",
        "ensemble_drug_to_target_logit",
        "biomaster_independent_borda_score",
        "biomaster_independent_new_head_borda_score",
        "biomaster_system_borda_score",
        "biomaster_full_fit_v10_borda_score",
    ]:
        deployment[f"{column}_rank_within_drug_384"] = deployment.groupby(
            "ligand_inchikey", sort=False
        )[column].rank(method="first", ascending=False).astype(np.int16)

    external = pd.read_csv(KIRHUB, low_memory=False)
    strict = external.loc[
        external["kirhub_frozen_unreported_scope"].fillna(False).astype(bool)
    ].copy()
    strict["binary_label"] = strict["kirhub_local_unreported_active"].astype(np.int8)
    strict[QUERY_COLUMN] = strict["ligand_inchikey"].astype(str)
    external_columns = [
        "pairId",
        "dtiam_probability",
        "biomaster_routed_stack_score",
        "old_drug_leakage_safe_score_v10",
        "ensemble_drug_to_target_logit",
        "biomaster_full_fit_v10_borda_score",
        "independent_validation_rank_score",
        "system_validation_rank_score",
        "biomaster_independent_borda_score",
        "biomaster_independent_new_head_borda_score",
        "biomaster_system_borda_score",
    ]
    strict = strict.drop(
        columns=[
            name
            for name in external_columns
            if name != "pairId" and name in strict.columns
        ]
    ).merge(deployment[external_columns], on="pairId", validate="one_to_one")
    if len(strict) != 2823 or int(strict["binary_label"].sum()) != 202:
        raise RuntimeError("strict KIRHub scope changed")
    external_models = {
        "DTIAM_RAW": "dtiam_probability",
        "BIOMASTER_ROUTED_STACK": "biomaster_routed_stack_score",
        "BIOMASTER_V10_GRAPH": "old_drug_leakage_safe_score_v10",
        "BIOMASTER_FULL_FIT_DIRECTIONAL": "ensemble_drug_to_target_logit",
        "BIOMASTER_FULL_FIT_V10_BORDA_DIAGNOSTIC": (
            "biomaster_full_fit_v10_borda_score"
        ),
        "INDEPENDENT_VALIDATION_RANK": "independent_validation_rank_score",
        "BIOMASTER_INDEPENDENT_BORDA": "biomaster_independent_borda_score",
        "BIOMASTER_INDEPENDENT_NEW_HEAD_BORDA": (
            "biomaster_independent_new_head_borda_score"
        ),
        "BIOMASTER_SYSTEM_BORDA": "biomaster_system_borda_score",
    }
    external_metrics = [
        metric_row(name, strict, strict[column].to_numpy())
        for name, column in external_models.items()
    ]
    external_bootstrap = {
        name: paired_query_bootstrap(
            strict,
            strict[column].to_numpy(),
            strict["dtiam_probability"].to_numpy(),
            args.bootstrap_iterations,
            args.seed + 100 + number,
        )
        for number, (name, column) in enumerate(
            {
                "BIOMASTER_INDEPENDENT_BORDA": "biomaster_independent_borda_score",
                "BIOMASTER_FULL_FIT_DIRECTIONAL": "ensemble_drug_to_target_logit",
                "BIOMASTER_INDEPENDENT_NEW_HEAD_BORDA": (
                    "biomaster_independent_new_head_borda_score"
                ),
                "BIOMASTER_SYSTEM_BORDA": "biomaster_system_borda_score",
            }.items()
        )
    }

    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
    top20_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_TOP20_V1.csv"
    system_top20_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_SYSTEM_TOP20_V1.csv"
    full_fit_top20_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_FULL_FIT_TOP20_V1.csv"
    s5_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_S5_TEST_V1.csv.gz"
    external_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_KIRHUB_V1.csv.gz"
    summary_output = output_dir / "BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json"
    keep = [
        "pairId",
        "ligand_inchikey",
        "drug_names",
        "target_chembl_id",
        "gene_symbol",
        "biomaster_independent_borda_score",
        "biomaster_independent_borda_score_rank_within_drug_384",
        "biomaster_independent_new_head_borda_score",
        "biomaster_independent_new_head_borda_score_rank_within_drug_384",
        "biomaster_system_borda_score",
        "biomaster_system_borda_score_rank_within_drug_384",
        "independent_validation_rank_score",
        "biomaster_routed_stack_score",
        "old_drug_leakage_safe_score_v10",
        "old_drug_leakage_safe_score_v10_rank_within_drug_384",
        "ensemble_drug_to_target_logit",
        "ensemble_drug_to_target_logit_rank_within_drug_384",
        "biomaster_full_fit_v10_borda_score",
        "dtiam_probability",
    ]
    deployment[keep].to_csv(deployment_output, index=False, compression="gzip")
    top20 = deployment.loc[
        deployment["biomaster_independent_borda_score_rank_within_drug_384"].le(20),
        keep,
    ].sort_values(
        ["ligand_inchikey", "biomaster_independent_borda_score_rank_within_drug_384"]
    )
    top20.to_csv(top20_output, index=False)
    system_top20 = deployment.loc[
        deployment["biomaster_system_borda_score_rank_within_drug_384"].le(20),
        keep,
    ].sort_values(
        ["ligand_inchikey", "biomaster_system_borda_score_rank_within_drug_384"]
    )
    system_top20.to_csv(system_top20_output, index=False)
    full_fit_top20 = deployment.loc[
        deployment["ensemble_drug_to_target_logit_rank_within_drug_384"].le(20),
        keep,
    ].sort_values(
        ["ligand_inchikey", "ensemble_drug_to_target_logit_rank_within_drug_384"]
    )
    full_fit_top20.to_csv(full_fit_top20_output, index=False)
    test[[
        "calibration_pair_id",
        "binary_label",
        "target_chembl_id",
        QUERY_COLUMN,
        "dtiam_probability",
        "biomaster_logit",
        "BIOMASTER_STACK_FIVE_SEED_MEAN",
        "independent_validation_rank_score",
        "system_validation_rank_score",
    ]].to_csv(s5_output, index=False, compression="gzip")
    strict.to_csv(external_output, index=False, compression="gzip")

    s5_by_name = {row["model"]: row for row in s5_metrics}
    external_by_name = {row["model"]: row for row in external_metrics}
    checks = {
        "frozen_s5_roles_exact": [len(train), len(valid), len(test)]
        == [65276, 16668, 2556],
        "no_s5_test_labels_used_for_selection": True,
        "independent_rank_has_no_dtiam_feature": "dtiam_logit"
        not in independent_features,
        "deployment_exact_720_by_384": len(deployment) == 720 * 384,
        "top20_exact_720_by_20": len(top20) == 720 * 20,
        "system_top20_exact_720_by_20": len(system_top20) == 720 * 20,
        "full_fit_top20_exact_720_by_20": len(full_fit_top20) == 720 * 20,
        "strict_kirhub_exact_2823_202": len(strict) == 2823
        and int(strict["binary_label"].sum()) == 202,
        "full_fit_three_checkpoint_ensemble_pass": (
            full_fit_summary.get("status") == "PASS"
            and len(full_fit_summary.get("checkpoints", [])) == 3
        ),
        "s5_independent_point_beats_dtiam_drug_macro_auprc": (
            s5_by_name["INDEPENDENT_VALIDATION_RANK"]["drug_macro_auprc"]
            > s5_by_name["DTIAM_RAW"]["drug_macro_auprc"]
        ),
        "external_independent_borda_point_beats_dtiam_drug_macro_auprc": (
            external_by_name["BIOMASTER_INDEPENDENT_BORDA"]["drug_macro_auprc"]
            > external_by_name["DTIAM_RAW"]["drug_macro_auprc"]
        ),
        "external_full_fit_directional_point_beats_dtiam_drug_macro_auprc": (
            external_by_name["BIOMASTER_FULL_FIT_DIRECTIONAL"]["drug_macro_auprc"]
            > external_by_name["DTIAM_RAW"]["drug_macro_auprc"]
        ),
        "external_system_borda_point_beats_dtiam_drug_macro_auprc": (
            external_by_name["BIOMASTER_SYSTEM_BORDA"]["drug_macro_auprc"]
            > external_by_name["DTIAM_RAW"]["drug_macro_auprc"]
        ),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "SCREEN_FAIL",
        "primary_direction": "OLD_DRUG_TO_384_TARGETS",
        "primary_metric": "drug_macro_auprc",
        "s5_fit_boundary": (
            "Weights and standardization are selected on frozen S5 validation. "
            "The 302 S5 test drugs and their labels are evaluation-only."
        ),
        "selection": {
            "independent_validation_rank": {
                "features": independent_features,
                "standardization_mean": independent_mean,
                "standardization_scale": independent_scale,
                **independent_selection,
            },
            "system_validation_rank": {
                "features": system_features,
                "standardization_mean": system_mean,
                "standardization_scale": system_scale,
                **system_selection,
            },
            "borda_rule": (
                "Convert every component to its percentile among the same drug's "
                "384 targets, then take an unweighted arithmetic mean."
            ),
        },
        "s5_test": {
            "rows": int(len(test)),
            "drugs": int(test[QUERY_COLUMN].nunique()),
            "two_class_drugs": int(
                test.groupby(QUERY_COLUMN)["binary_label"].nunique().eq(2).sum()
            ),
            "metrics": s5_metrics,
            "independent_rank_vs_dtiam_query_bootstrap": s5_independent_bootstrap,
            "system_rank_vs_dtiam_query_bootstrap": s5_system_bootstrap,
        },
        "strict_kirhub_post_audit": {
            "rows": int(len(strict)),
            "positives": int(strict["binary_label"].sum()),
            "drugs": int(strict[QUERY_COLUMN].nunique()),
            "two_class_drugs": int(
                strict.groupby(QUERY_COLUMN)["binary_label"].nunique().eq(2).sum()
            ),
            "metrics": external_metrics,
            "paired_query_bootstrap_vs_dtiam": external_bootstrap,
        },
        "full_fit_production_contract": {
            "status": full_fit_summary["status"],
            "rows": full_fit_summary["rows"],
            "drug_queries": full_fit_summary["drug_queries"],
            "candidate_targets": full_fit_summary["candidate_targets"],
            "checkpoints": full_fit_summary["checkpoints"],
            "primary_score": full_fit_summary["primary_score"],
            "claim_boundary": full_fit_summary["claim_boundary"],
        },
        "checks": checks,
        "promotion_decision": (
            "DRUG_CENTRIC_PRODUCTION_CANDIDATE_PENDING_NEW_UNTOUCHED_SOURCE"
            if all(checks.values())
            else "DO_NOT_PROMOTE"
        ),
        "claim_boundary": (
            "The independent validation head, FULL_FIT directional head, and system "
            "ranker may be reported only as frozen S5 or post-audit KIRHub point-"
            "estimate improvements over DTIAM. KIRHub has "
            "only 33 two-class drug queries and was inspected previously, so it is "
            "not a fresh confirmatory superiority test."
        ),
        "artifacts": {
            "deployment": str(deployment_output.relative_to(ROOT)),
            "top20": str(top20_output.relative_to(ROOT)),
            "system_top20": str(system_top20_output.relative_to(ROOT)),
            "full_fit_top20": str(full_fit_top20_output.relative_to(ROOT)),
            "s5_test": str(s5_output.relative_to(ROOT)),
            "kirhub": str(external_output.relative_to(ROOT)),
        },
        "inputs": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in required
        },
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

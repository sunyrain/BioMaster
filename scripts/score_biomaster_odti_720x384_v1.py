#!/usr/bin/env python3
"""Score the frozen 720 x 384 deployment matrix with BioMaster-ODTI V1."""

from __future__ import annotations

import glob
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_biomaster_odti_baselines_v1 import split_masks, target_prior  # noqa: E402
from train_biomaster_odti_routed_ranker_v1 import (  # noqa: E402
    RoutedInteractionRanker,
    infer,
    sigmoid,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
DEPLOY = BASE / "deployment_720x384_feature_store_v1"
DEPLOY_PAIRS = DEPLOY / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
DEPLOY_DRUG = DEPLOY / "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
DEPLOY_TARGET = DEPLOY / "PROJECT384_PROTBERT1024_FLOAT32_V1.npy"
DEPLOY_AUDIT = DEPLOY / "DEPLOYMENT_FEATURE_STORE_AUDIT_V1.json"
CAL_STORE = BASE / "feature_store_v1"
CAL_PAIRS = CAL_STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
CAL_DRUG = CAL_STORE / "MORGAN2048_UINT8_V1.npy"
RUNS = BASE / "biomaster_odti_routed_ranker_v1"
OUT = BASE / "biomaster_odti_deployment_v1"
STACK_COLUMNS = [
    "biomaster_ensemble_logit", "conplex_score", "train_positive_max_tanimoto",
    "target_train_prior", "target_has_train_positive_pool",
]


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = RoutedInteractionRanker(
        family_count=len(checkpoint["families"]),
        use_conplex=checkpoint["variant"] == "conplex_augmented",
        embedding_dim=architecture["embedding_dim"],
        hidden_dim=architecture["hidden_dim"],
        expert_count=architecture["experts"],
        dropout=architecture["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def inference_arrays(data: pd.DataFrame, checkpoint: dict[str, object]) -> dict[str, object]:
    normalization = checkpoint["normalization"]
    family_lookup = {name: index for index, name in enumerate(checkpoint["families"])}
    family_index = data["target_assay_family"].astype(str).map(family_lookup)
    if family_index.isna().any():
        raise RuntimeError(f"Unmapped target family: {data.loc[family_index.isna(), 'target_assay_family'].unique()}")
    return {
        "families": checkpoint["families"],
        "family_index": family_index.to_numpy(dtype=np.int64),
        "target_mean": np.asarray(normalization["target_mean"], dtype=np.float32),
        "target_std": np.asarray(normalization["target_std"], dtype=np.float32),
        "conplex": data["conplex_score"].to_numpy(dtype=np.float32),
        "conplex_mean": float(normalization["conplex_mean"]),
        "conplex_std": float(normalization["conplex_std"]),
        "affinity": pd.to_numeric(data["mean_pchembl"], errors="coerce").to_numpy(dtype=np.float32),
        "affinity_mean": float(normalization["affinity_mean"]),
        "affinity_std": float(normalization["affinity_std"]),
    }


def cross_store_max_tanimoto(
    train: pd.DataFrame,
    query: pd.DataFrame,
    train_fingerprints: np.ndarray,
    query_fingerprints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positives = train[train["binary_label"].eq(1)]
    pools = {
        target: np.unique(part["drug_feature_index"].to_numpy(dtype=np.int32))
        for target, part in positives.groupby("target_chembl_id", sort=False)
    }
    result = np.zeros(len(query), dtype=np.float32)
    has_pool = np.zeros(len(query), dtype=np.int8)
    for target, positions_raw in query.groupby("target_chembl_id", sort=False).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        reference_index = pools.get(target)
        if reference_index is None or not len(reference_index):
            continue
        has_pool[positions] = 1
        query_index = query.iloc[positions]["drug_feature_index"].to_numpy(dtype=np.int32)
        query_bits = np.asarray(query_fingerprints[query_index], dtype=np.float32)
        reference_bits = np.asarray(train_fingerprints[reference_index], dtype=np.float32)
        intersection = query_bits @ reference_bits.T
        union = query_bits.sum(axis=1, keepdims=True) + reference_bits.sum(axis=1)[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        result[positions] = similarity.max(axis=1)
    return result, has_pool


def add_external_features(
    train: pd.DataFrame,
    query: pd.DataFrame,
    train_fingerprints: np.ndarray,
    query_fingerprints: np.ndarray,
) -> pd.DataFrame:
    result = query.copy()
    similarity, has_pool = cross_store_max_tanimoto(
        train, result, train_fingerprints, query_fingerprints
    )
    result["train_positive_max_tanimoto"] = similarity
    result["target_has_train_positive_pool"] = has_pool
    result["target_train_prior"] = target_prior(train, result)
    return result


def score_checkpoint(
    data: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model, checkpoint = load_model(checkpoint_path, device)
    arrays = inference_arrays(data, checkpoint)
    positions = np.arange(len(data), dtype=np.int64)
    logits, _, _ = infer(
        model, positions, data, drug_features, target_features, arrays, device, 4096
    )
    probabilities = sigmoid(logits / float(checkpoint["temperature"]))
    del model
    torch.cuda.empty_cache()
    return logits, probabilities


def main() -> None:
    audit = json.loads(DEPLOY_AUDIT.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("Deployment feature store must pass")
    deploy = pd.read_csv(DEPLOY_PAIRS, low_memory=False)
    deploy["binary_label"] = 0
    deploy["mean_pchembl"] = np.nan
    calibration = pd.read_csv(CAL_PAIRS, low_memory=False)
    deploy_drug = np.load(DEPLOY_DRUG, mmap_mode="r")
    deploy_target = np.load(DEPLOY_TARGET, mmap_mode="r")
    calibration_drug = np.load(CAL_DRUG, mmap_mode="r")
    masks = split_masks(calibration, "S5_OLD_DRUG_ENTITY_COLD", -1)
    available = calibration["drug_feature_available"].to_numpy(dtype=bool)
    train = calibration.loc[masks["train"] & available].reset_index(drop=True)
    validation = calibration.loc[masks["valid"] & available].reset_index(drop=True)
    validation = add_external_features(train, validation, calibration_drug, calibration_drug)
    deploy = add_external_features(train, deploy, calibration_drug, deploy_drug)

    run_paths = sorted(
        path
        for path in RUNS.glob("S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_*__CORE")
        if (path / "RUN_SUMMARY_V1.json").is_file()
        and json.loads((path / "RUN_SUMMARY_V1.json").read_text()).get("status") == "PASS"
        and (path / "VALIDATION_PREDICTIONS_V1.csv.gz").is_file()
    )
    if not run_paths:
        raise RuntimeError("No passed S5 core checkpoint with validation predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deploy_logits = []
    deploy_probabilities = []
    validation_predictions = []
    seeds = []
    for run_path in run_paths:
        summary = json.loads((run_path / "RUN_SUMMARY_V1.json").read_text())
        seed = int(summary["seed"])
        seeds.append(seed)
        logits, probabilities = score_checkpoint(
            deploy, deploy_drug, deploy_target, run_path / "BEST_MODEL_V1.pt", device
        )
        deploy_logits.append(logits)
        deploy_probabilities.append(probabilities)
        valid = pd.read_csv(
            run_path / "VALIDATION_PREDICTIONS_V1.csv.gz",
            usecols=["calibration_pair_id", "biomaster_logit"],
        ).rename(columns={"biomaster_logit": f"logit_seed_{seed}"})
        validation_predictions.append(valid)
        print(json.dumps({"scored_seed": seed, "pairs": len(deploy)}), flush=True)

    validation_model = validation[["calibration_pair_id", "binary_label", "conplex_score"]].copy()
    for frame in validation_predictions:
        validation_model = validation_model.merge(frame, on="calibration_pair_id", validate="one_to_one")
    validation_model["biomaster_ensemble_logit"] = validation_model[
        [column for column in validation_model if column.startswith("logit_seed_")]
    ].mean(axis=1)
    validation_model = validation_model.merge(
        validation[[
            "calibration_pair_id", "train_positive_max_tanimoto", "target_train_prior",
            "target_has_train_positive_pool",
        ]],
        on="calibration_pair_id",
        validate="one_to_one",
    )
    stack = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=20260813),
    )
    stack.fit(
        validation_model[STACK_COLUMNS].to_numpy(dtype=np.float64),
        validation_model["binary_label"].to_numpy(dtype=np.int8),
    )
    deploy["biomaster_ensemble_logit"] = np.mean(np.stack(deploy_logits), axis=0)
    deploy["biomaster_ensemble_probability"] = np.mean(np.stack(deploy_probabilities), axis=0)
    deploy["biomaster_model_seed_std"] = np.std(np.stack(deploy_probabilities), axis=0)
    deploy["biomaster_routed_stack_score"] = stack.predict_proba(
        deploy[STACK_COLUMNS].to_numpy(dtype=np.float64)
    )[:, 1]
    for score, rank in [
        ("biomaster_routed_stack_score", "biomaster_routed_stack_rank_within_drug_384"),
        ("biomaster_ensemble_probability", "biomaster_ensemble_rank_within_drug_384"),
    ]:
        deploy[rank] = deploy.groupby("ligand_inchikey")[score].rank(
            method="first", ascending=False
        ).astype(np.int16)
    deploy["biomaster_routed_stack_percentile_within_drug_384"] = (
        1.0 - (deploy["biomaster_routed_stack_rank_within_drug_384"] - 1) / 383.0
    )
    deploy["biomaster_applicability_domain"] = np.select(
        [
            deploy["target_has_train_positive_pool"].eq(1) & deploy["train_positive_max_tanimoto"].ge(0.5),
            deploy["target_has_train_positive_pool"].eq(1),
        ],
        ["AD_HIGH_TARGET_POOL_AND_CHEMICAL_NEIGHBOR", "AD_MEDIUM_TARGET_POOL_ONLY"],
        default="AD_LOW_SEQUENCE_EXTRAPOLATION_TARGET",
    )
    deploy["biomaster_deployment_priority"] = np.select(
        [
            deploy["is_any_frozen_known_relationship"].fillna(False).astype(bool),
            deploy["biomaster_routed_stack_rank_within_drug_384"].le(20)
            & deploy["local_chembl37_unreported_pair"].fillna(False).astype(bool),
            deploy["biomaster_routed_stack_rank_within_drug_384"].le(50),
        ],
        ["KNOWN_RELATIONSHIP_CONTROL", "TOP20_LOCAL_UNREPORTED_REVIEW", "TOP50_EXPLORATORY_REVIEW"],
        default="HOLD_BEYOND_TOP50",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    score_path = OUT / "BIOMASTER_ODTI_720X384_SCORES_V1.csv.gz"
    top_path = OUT / "BIOMASTER_ODTI_TOP20_PER_OLD_DRUG_V1.csv.gz"
    application_path = OUT / "BIOMASTER_ODTI_INNOVATION_APPLICATION_CASES_V1.csv"
    output_columns = [column for column in deploy.columns if column not in ["binary_label", "mean_pchembl"]]
    deploy[output_columns].to_csv(score_path, index=False)
    deploy[deploy["biomaster_routed_stack_rank_within_drug_384"].le(20)][output_columns].to_csv(
        top_path, index=False
    )
    application = deploy[
        deploy["is_v8_mutation_application_pair"].fillna(False).astype(bool)
        | deploy["is_v8_database_gap_rediscovery_control"].fillna(False).astype(bool)
        | deploy["is_v8_prospective_unvalidated_case"].fillna(False).astype(bool)
    ].copy()
    application.sort_values(
        ["biomaster_routed_stack_rank_within_drug_384", "biomaster_routed_stack_score"],
        ascending=[True, False],
    ).to_csv(application_path, index=False)

    # Check consistency against the original S5 test for the shared 302 drugs x
    # measured target subset. Minor differences can occur only if active-moiety
    # SMILES differ between the two frozen universes, so report rather than hide.
    s5_test = None
    probability_columns = []
    for run_path, seed in zip(run_paths, seeds):
        column = f"biomaster_probability_seed_{seed}"
        probability_columns.append(column)
        part = pd.read_csv(
            run_path / "TEST_PREDICTIONS_V1.csv.gz",
            usecols=[
                "parent_standard_inchi_key", "target_chembl_id",
                "biomaster_probability_calibrated",
            ],
        ).rename(columns={"biomaster_probability_calibrated": column})
        if s5_test is None:
            s5_test = part
        else:
            s5_test = s5_test.merge(
                part,
                on=["parent_standard_inchi_key", "target_chembl_id"],
                validate="one_to_one",
            )
    if s5_test is None:
        raise RuntimeError("S5 test ensemble unexpectedly empty")
    s5_test["biomaster_test_ensemble_probability"] = s5_test[probability_columns].mean(axis=1)
    calibration_test_index = calibration.loc[masks["test"], [
        "parent_standard_inchi_key", "target_chembl_id", "drug_feature_index",
    ]].rename(columns={"drug_feature_index": "calibration_drug_feature_index"})
    s5_test = s5_test.merge(
        calibration_test_index,
        on=["parent_standard_inchi_key", "target_chembl_id"],
        validate="one_to_one",
    )
    shared = deploy.merge(
        s5_test,
        left_on=["ligand_inchikey", "target_chembl_id"],
        right_on=["parent_standard_inchi_key", "target_chembl_id"],
        how="inner",
        validate="one_to_one",
    )
    shared_difference = np.abs(
        shared["biomaster_ensemble_probability"] - shared["biomaster_test_ensemble_probability"]
    )
    same_fingerprint = np.asarray([
        np.array_equal(deploy_drug[int(deploy_index)], calibration_drug[int(calibration_index)])
        for deploy_index, calibration_index in zip(
            shared["drug_feature_index"], shared["calibration_drug_feature_index"]
        )
    ])
    same_fingerprint_difference = shared_difference[same_fingerprint]
    coefficients = stack.named_steps["logisticregression"]
    scaler = stack.named_steps["standardscaler"]
    checks = {
        "deployment_feature_store_pass": audit["status"] == "PASS",
        "exact_276480_scores": len(deploy) == 276480 and deploy["pairId"].is_unique,
        "exact_720_top20_sets": len(deploy[deploy["biomaster_routed_stack_rank_within_drug_384"].le(20)]) == 14400,
        "scores_finite_and_bounded": np.isfinite(deploy["biomaster_routed_stack_score"]).all() and deploy["biomaster_routed_stack_score"].between(0, 1).all(),
        "common_measured_active_target_pairs_exact_2235": len(shared) == 2235,
        "same_fingerprint_pairs_exact_2171": int(same_fingerprint.sum()) == 2171,
        "same_fingerprint_probability_mean_abs_diff_le0p001": float(same_fingerprint_difference.mean()) <= 0.001,
        "same_fingerprint_probability_q95_abs_diff_le0p006": float(np.quantile(same_fingerprint_difference, 0.95)) <= 0.006,
        "application_cases_exact_68": len(application) == 68,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": "BIOMASTER_ODTI_V1_OLD_DRUG_ENTITY_COLD_ENSEMBLE_AND_VALIDATION_ROUTED_STACK",
        "model_seeds": seeds,
        "model_seed_count": len(seeds),
        "fit_roles": {
            "base_model": "S5 training only; all old-drug entities and their scaffolds excluded",
            "stack": "S5 frozen validation only",
            "deployment": "720 old drugs x active 384 targets; no deployment labels used",
        },
        "counts": {
            "pairs": int(len(deploy)), "old_drugs": 720, "targets": 384,
            "top20_rows": int((deploy["biomaster_routed_stack_rank_within_drug_384"] <= 20).sum()),
            "mutation_application_pairs": int(deploy["is_v8_mutation_application_pair"].fillna(False).sum()),
            "database_gap_controls": int(deploy["is_v8_database_gap_rediscovery_control"].fillna(False).sum()),
            "prospective_unvalidated_cases": int(deploy["is_v8_prospective_unvalidated_case"].fillna(False).sum()),
        },
        "stack": {
            "features": STACK_COLUMNS,
            "standard_scaler_mean": dict(zip(STACK_COLUMNS, scaler.mean_.tolist())),
            "standard_scaler_scale": dict(zip(STACK_COLUMNS, scaler.scale_.tolist())),
            "standardized_coefficients": dict(zip(STACK_COLUMNS, coefficients.coef_[0].tolist())),
            "intercept": float(coefficients.intercept_[0]),
            "fixed_C": 0.1,
        },
        "shared_s5_test_reproduction": {
            "pairs": int(len(shared)),
            "base_probability_max_abs_difference": float(shared_difference.max()),
            "base_probability_mean_abs_difference": float(shared_difference.mean()),
            "same_fingerprint_pairs": int(same_fingerprint.sum()),
            "different_active_moiety_fingerprint_pairs": int((~same_fingerprint).sum()),
            "same_fingerprint_probability_mean_abs_difference": float(same_fingerprint_difference.mean()),
            "same_fingerprint_probability_q95_abs_difference": float(np.quantile(same_fingerprint_difference, 0.95)),
            "note": "321 of 2556 frozen S5 measured pairs target calibration targets outside the active 384 and are intentionally absent; 64 common pairs use different frozen active-moiety fingerprints.",
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "claim_status": "DEPLOYMENT_RANKING_NOT_PROSPECTIVE_VALIDATION; SCOPE_PUBLIC_CHECKPOINT_CURRENTLY_STRONGER_ON_S5_TEST",
    }
    summary_path = OUT / "BIOMASTER_ODTI_DEPLOYMENT_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps({key: bool(value) for key, value in checks.items()}, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

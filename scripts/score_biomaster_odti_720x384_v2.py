#!/usr/bin/env python3
"""Re-score the frozen 720 x 384 old-drug deployment matrix with ODTI V2.

The deployment universe is deliberately kept identical to the historical V1
scan: 720 old drugs, 384 project targets, and 276,480 Cartesian pairs.  No
deployment labels are used.  The S5 V2 checkpoints are loaded only for
inference; ranking is performed within each drug over the 384 targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2  # noqa: E402
from run_biomaster_odti_baselines_v1 import split_masks  # noqa: E402
from score_biomaster_odti_720x384_v1 import add_external_features  # noqa: E402
from train_biomaster_odti_v2 import predict  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
DEPLOY = BASE / "deployment_720x384_feature_store_v1"
CAL = BASE / "feature_store_v1"
DTIAM_DEPLOY = BASE / "public_retrained_v1/dtiam_deployment_feature_store_v1"
RUNS = BASE / "biomaster_odti_v2_s5_esm2_formal"
OUT = BASE / "biomaster_odti_deployment_v2_s5_esm2"

DEPLOY_PAIRS = DEPLOY / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
DEPLOY_DRUG = DEPLOY / "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
DEPLOY_TARGET = DEPLOY / "PROJECT384_PROTBERT1024_FLOAT32_V1.npy"
DEPLOY_TARGET_INDEX = DEPLOY / "PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz"
CAL_PAIRS = CAL / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
CAL_DRUG = CAL / "MORGAN2048_UINT8_V1.npy"
STRUCTURE = CAL / "ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
ESM2 = DTIAM_DEPLOY / "DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy"
ESM2_INDEX = DTIAM_DEPLOY / "DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz"
BERMOL = DTIAM_DEPLOY / "DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy"

SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]
TOP_K = [1, 5, 10, 20, 50, 100]


def _load_checkpoint(seed: int, device: torch.device):
    run = RUNS / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}"
    summary_path = run / "RUN_SUMMARY_V2.json"
    checkpoint_path = run / "BEST_MODEL_V2.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing V2 S5 checkpoint for seed {seed}: {run}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"V2 seed {seed} is not PASS")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ODTIV2Config(**checkpoint["config"])
    model = RoutedInteractionRankerV2(
        family_count=len(checkpoint["families"]),
        config=config,
        use_conplex=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, summary


def _structure_for_deployment(
    deploy: pd.DataFrame,
    calibration_pairs: pd.DataFrame,
    structure: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Broadcast target-constant structure context to deployment pairs.

    The frozen structure table is target-context only.  It is constant across
    all calibration pairs for a target, so taking the first row per target is
    label-blind and preserves the audited missing-modality fallback.  Targets
    absent from the calibration structure table receive mask=0 and zeros.
    """

    feature_columns = [
        c for c in structure.columns
        if c not in {"calibration_pair_id", "structure_mask"}
    ]
    pair_target = calibration_pairs[["calibration_pair_id", "target_chembl_id"]].drop_duplicates()
    target_structure = pair_target.merge(structure, on="calibration_pair_id", how="left")
    target_structure = target_structure.sort_values("calibration_pair_id").drop_duplicates(
        "target_chembl_id", keep="first"
    )
    target_structure = target_structure.set_index("target_chembl_id")
    target_structure.index = target_structure.index.astype(str)
    target_ids = deploy["target_chembl_id"].astype(str)
    # Vectorized reindexing is important here: the deployment matrix has
    # 276,480 rows, and a Python/pandas row lookup per pair makes the scorer
    # needlessly slow while doing no additional audit work.
    selected = target_structure.reindex(target_ids.to_numpy())
    raw = selected[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(raw).all(axis=1)
    values = np.zeros((len(deploy), len(feature_columns)), dtype=np.float32)
    values[valid] = raw[valid]
    if "structure_mask" in selected.columns:
        mask_values = pd.to_numeric(selected["structure_mask"], errors="coerce").to_numpy(dtype=np.float32)
        mask = np.where(valid & np.isfinite(mask_values) & (mask_values > 0), 1.0, 0.0).astype(np.float32)
    else:
        mask = valid.astype(np.float32)
    return values, mask, int((mask == 0).sum())


def _arrays(deploy: pd.DataFrame, checkpoint: dict[str, object]) -> dict[str, object]:
    normalization = checkpoint["normalization"]
    lookup = {str(name): i for i, name in enumerate(checkpoint["families"])}
    family = deploy["target_assay_family"].astype(str).map(lookup).fillna(lookup["__UNK__"])
    return {
        "families": checkpoint["families"],
        "family_index": family.to_numpy(dtype=np.int64),
        "target_mean": np.asarray(normalization["target_mean"], dtype=np.float32),
        "target_std": np.asarray(normalization["target_std"], dtype=np.float32),
        "drug_aux_mean": np.asarray(
            normalization.get("drug_aux_mean", np.zeros(0)), dtype=np.float32
        ),
        "drug_aux_std": np.asarray(
            normalization.get("drug_aux_std", np.ones(0)), dtype=np.float32
        ),
        "target_aux_mean": np.asarray(normalization["target_aux_mean"], dtype=np.float32),
        "target_aux_std": np.asarray(normalization["target_aux_std"], dtype=np.float32),
        "target_token_mean": np.zeros(0, dtype=np.float32),
        "target_token_std": np.ones(0, dtype=np.float32),
        "conplex": deploy["conplex_score"].to_numpy(dtype=np.float32),
        "conplex_mean": float(normalization["conplex_mean"]),
        "conplex_std": float(normalization["conplex_std"]),
        "affinity": deploy["mean_pchembl"].to_numpy(dtype=np.float32),
        "affinity_lower": deploy["min_pchembl"].to_numpy(dtype=np.float32),
        "affinity_upper": deploy["max_pchembl"].to_numpy(dtype=np.float32),
        "affinity_mean": float(normalization["affinity_mean"]),
        "affinity_std": float(normalization["affinity_std"]),
        "structure_mean": np.asarray(normalization["structure_mean"], dtype=np.float32),
        "structure_std": np.asarray(normalization["structure_std"], dtype=np.float32),
    }


def _known_recall(frame: pd.DataFrame, rank_column: str) -> dict[str, object]:
    known = frame[frame["is_any_frozen_known_relationship"].fillna(False).astype(bool)]
    result: dict[str, object] = {
        "known_relationships": int(len(known)),
        "rank_column": rank_column,
        "top_k": {},
    }
    for k in TOP_K:
        recovered = int((known[rank_column] <= k).sum())
        result["top_k"][str(k)] = {
            "recovered": recovered,
            "total": int(len(known)),
            "recall": float(recovered / len(known)) if len(known) else None,
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deploy = pd.read_csv(DEPLOY_PAIRS, low_memory=False)
    calibration = pd.read_csv(CAL_PAIRS, low_memory=False)
    structure = pd.read_csv(STRUCTURE, low_memory=False)
    target_index = pd.read_csv(DEPLOY_TARGET_INDEX, low_memory=False)
    esm_index = pd.read_csv(ESM2_INDEX, low_memory=False)
    deploy_drug = np.load(DEPLOY_DRUG, mmap_mode="r")
    deploy_target = np.load(DEPLOY_TARGET, mmap_mode="r")
    calibration_drug = np.load(CAL_DRUG, mmap_mode="r")
    target_aux = np.load(ESM2, mmap_mode="r")
    deploy_drug_aux = np.load(BERMOL, mmap_mode="r")

    expected = {"pairs": 276480, "drugs": 720, "targets": 384}
    if len(deploy) != expected["pairs"]:
        raise RuntimeError(f"unexpected deployment rows: {len(deploy)}")
    if deploy["ligand_inchikey"].nunique() != expected["drugs"]:
        raise RuntimeError("unexpected deployment drug count")
    if deploy["target_chembl_id"].nunique() != expected["targets"]:
        raise RuntimeError("unexpected deployment target count")
    if target_index["target_feature_index"].nunique() != expected["targets"]:
        raise RuntimeError("unexpected deployment target feature index")
    if esm_index["target_feature_index"].nunique() != expected["targets"]:
        raise RuntimeError("ESM2 deployment feature index is not 384-target aligned")
    if target_aux.shape != (expected["targets"], 1280):
        raise RuntimeError(f"unexpected ESM2 deployment shape: {target_aux.shape}")
    if deploy_drug_aux.shape != (expected["drugs"], 768):
        raise RuntimeError(f"unexpected BERMOL deployment shape: {deploy_drug_aux.shape}")

    # These columns are intentionally missing-label placeholders.  The V2
    # inference path needs the columns for its auxiliary heads, but no
    # deployment label is read or used.
    deploy["binary_label"] = 0
    deploy["mean_pchembl"] = np.nan
    deploy["min_pchembl"] = np.nan
    deploy["max_pchembl"] = np.nan
    structure_values, structure_mask, missing_structure_rows = _structure_for_deployment(
        deploy, calibration, structure
    )

    # Add the same label-blind applicability covariates used in the V1
    # deployment artifact.  They are reported for candidate triage, not fed
    # into the pure V2 ranking score.
    masks = split_masks(calibration, "S5_OLD_DRUG_ENTITY_COLD", -1)
    available = calibration["drug_feature_available"].to_numpy(dtype=bool)
    train = calibration.loc[masks["train"] & available].reset_index(drop=True)
    deploy = add_external_features(train, deploy, calibration_drug, deploy_drug)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = np.arange(len(deploy), dtype=np.int64)
    mean_probabilities: list[np.ndarray] = []
    mean_logits: list[np.ndarray] = []
    seed_spreads: list[np.ndarray] = []
    checkpoint_metadata = []
    for seed in SEEDS:
        model, checkpoint, summary = _load_checkpoint(seed, device)
        arrays = _arrays(deploy, checkpoint)
        drug_aux_dim = int(checkpoint["config"].get("drug_aux_input_dim", 0))
        drug_aux = deploy_drug_aux if drug_aux_dim > 0 else None
        if drug_aux is not None and drug_aux.shape[1] != drug_aux_dim:
            raise RuntimeError(
                f"deployment BERMOL width {drug_aux.shape[1]} != checkpoint {drug_aux_dim}"
            )
        drug_cache = torch.from_numpy(np.asarray(deploy_drug, dtype=np.float32)).to(device)
        drug_aux_cache = None
        if drug_aux is not None:
            drug_aux_cache = torch.from_numpy(
                ((np.asarray(drug_aux, dtype=np.float32) - arrays["drug_aux_mean"])
                 / arrays["drug_aux_std"])
            ).to(device)
        target_cache = torch.from_numpy(
            ((np.asarray(deploy_target, dtype=np.float32) - arrays["target_mean"])
             / arrays["target_std"])
        ).to(device)
        aux_cache = torch.from_numpy(
            ((np.asarray(target_aux, dtype=np.float32) - arrays["target_aux_mean"])
             / arrays["target_aux_std"])
        ).to(device)
        predicted = predict(
            model,
            positions,
            deploy,
            deploy_drug,
            drug_aux,
            deploy_target,
            target_aux,
            None,
            None,
            None,
            None,
            int(checkpoint.get("target_token_max_len", 1022)),
            None,
            None,
            structure_values,
            structure_mask,
            arrays,
            device,
            4096,
            drug_aux_available=np.ones(expected["drugs"], dtype=np.float32) if drug_aux is not None else None,
            drug_feature_cache=drug_cache,
            drug_aux_feature_cache=drug_aux_cache,
            target_feature_cache=target_cache,
            target_aux_feature_cache=aux_cache,
        )
        temperature = float(checkpoint["temperature"])
        logits = np.asarray(predicted["final_logit"], dtype=np.float64)
        probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
        mean_logits.append(logits)
        mean_probabilities.append(probabilities)
        checkpoint_metadata.append({
            "seed": int(seed),
            "temperature": temperature,
            "run_summary": str(RUNS / f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}" / "RUN_SUMMARY_V2.json"),
            "test_micro_auprc": summary["test_metrics"]["micro_auprc"],
        })
        print(json.dumps({"scored_seed": seed, "pairs": len(deploy), "device": str(device)}), flush=True)
        del model, drug_cache, drug_aux_cache, target_cache, aux_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probability_matrix = np.stack(mean_probabilities, axis=0)
    logit_matrix = np.stack(mean_logits, axis=0)
    deploy["biomaster_v2_ensemble_probability"] = probability_matrix.mean(axis=0)
    deploy["biomaster_v2_ensemble_logit"] = logit_matrix.mean(axis=0)
    deploy["biomaster_v2_model_seed_std"] = probability_matrix.std(axis=0)
    deploy["biomaster_v2_ensemble_rank_within_drug_384"] = deploy.groupby("ligand_inchikey")[
        "biomaster_v2_ensemble_probability"
    ].rank(method="first", ascending=False).astype(np.int16)
    deploy["biomaster_v2_ensemble_percentile_within_drug_384"] = (
        1.0 - (deploy["biomaster_v2_ensemble_rank_within_drug_384"] - 1) / 383.0
    )
    deploy["biomaster_v2_applicability_domain"] = np.select(
        [
            deploy["target_has_train_positive_pool"].eq(1)
            & deploy["train_positive_max_tanimoto"].ge(0.5),
            deploy["target_has_train_positive_pool"].eq(1),
        ],
        ["AD_HIGH_TARGET_POOL_AND_CHEMICAL_NEIGHBOR", "AD_MEDIUM_TARGET_POOL_ONLY"],
        default="AD_LOW_SEQUENCE_EXTRAPOLATION_TARGET",
    )
    deploy["biomaster_v2_deployment_priority"] = np.select(
        [
            deploy["is_any_frozen_known_relationship"].fillna(False).astype(bool),
            deploy["biomaster_v2_ensemble_rank_within_drug_384"].le(20)
            & deploy["local_chembl37_unreported_pair"].fillna(False).astype(bool),
            deploy["biomaster_v2_ensemble_rank_within_drug_384"].le(50),
        ],
        ["KNOWN_RELATIONSHIP_CONTROL", "TOP20_LOCAL_UNREPORTED_REVIEW", "TOP50_EXPLORATORY_REVIEW"],
        default="HOLD_BEYOND_TOP50",
    )

    score_columns = [c for c in deploy.columns if c not in {"binary_label", "mean_pchembl", "min_pchembl", "max_pchembl"}]
    score_path = OUT / "BIOMASTER_ODTI_720X384_SCORES_V2_ESM2.csv.gz"
    top_path = OUT / "BIOMASTER_ODTI_TOP20_PER_OLD_DRUG_V2_ESM2.csv.gz"
    deploy[score_columns].to_csv(score_path, index=False, compression="gzip")
    deploy[deploy["biomaster_v2_ensemble_rank_within_drug_384"] <= 20][score_columns].to_csv(
        top_path, index=False, compression="gzip"
    )

    v2_recall = _known_recall(deploy, "biomaster_v2_ensemble_rank_within_drug_384")
    v1_path = BASE / "biomaster_odti_deployment_v1/BIOMASTER_ODTI_720X384_SCORES_V1.csv.gz"
    v1_recall = None
    if v1_path.is_file():
        v1 = pd.read_csv(v1_path, usecols=[
            "is_any_frozen_known_relationship",
            "biomaster_routed_stack_rank_within_drug_384",
        ], low_memory=False)
        v1_recall = _known_recall(v1, "biomaster_routed_stack_rank_within_drug_384")

    counts = {
        "pairs": int(len(deploy)),
        "old_drugs": int(deploy["ligand_inchikey"].nunique()),
        "targets": int(deploy["target_chembl_id"].nunique()),
        "known_relationship_controls": int(deploy["is_any_frozen_known_relationship"].sum()),
        "top20_rows": int((deploy["biomaster_v2_ensemble_rank_within_drug_384"] <= 20).sum()),
        "top20_local_unreported_rows": int(
            (
                (deploy["biomaster_v2_ensemble_rank_within_drug_384"] <= 20)
                & deploy["local_chembl37_unreported_pair"].fillna(False).astype(bool)
            ).sum()
        ),
        "top50_rows": int((deploy["biomaster_v2_ensemble_rank_within_drug_384"] <= 50).sum()),
        "ad_high_rows": int((deploy["biomaster_v2_applicability_domain"] == "AD_HIGH_TARGET_POOL_AND_CHEMICAL_NEIGHBOR").sum()),
        "ad_medium_rows": int((deploy["biomaster_v2_applicability_domain"] == "AD_MEDIUM_TARGET_POOL_ONLY").sum()),
        "ad_low_rows": int((deploy["biomaster_v2_applicability_domain"] == "AD_LOW_SEQUENCE_EXTRAPOLATION_TARGET").sum()),
    }
    summary = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "status": "PASS",
        "model": "BIOMASTER_ODTI_V2_ESM2_5SEED_DEPLOYMENT_ENSEMBLE",
        "deployment_space": {
            "pairs": expected["pairs"],
            "old_drugs": expected["drugs"],
            "targets": expected["targets"],
            "ranking": "within each old drug over 384 targets",
        },
        "counts": counts,
        "seeds": SEEDS,
        "checkpoint_metadata": checkpoint_metadata,
        "feature_sources": {
            "deployment_pairs": str(DEPLOY_PAIRS),
            "deployment_drug_features": str(DEPLOY_DRUG),
            "deployment_drug_aux_features": str(BERMOL),
            "deployment_target_features": str(DEPLOY_TARGET),
            "deployment_target_aux_features": str(ESM2),
            "structure_context": str(STRUCTURE),
        },
        "structure_context": {
            "feature_dim": int(structure_values.shape[1]),
            "deployment_rows_with_mask_zero": missing_structure_rows,
            "deployment_rows_with_mask_one": int(structure_mask.sum()),
            "missing_targets_are_fallback": True,
        },
        "v2_known_relationship_rediscovery_recall": v2_recall,
        "v1_known_relationship_rediscovery_recall": v1_recall,
        "artifacts": {
            "scores": str(score_path),
            "top20": str(top_path),
        },
        "claim_status": "DEPLOYMENT_RESCORING; KNOWN-CONTROL REDISCOVERY ONLY; NOVEL PAIR RECALL REQUIRES WET-LAB VALIDATION",
    }
    (OUT / "BIOMASTER_ODTI_DEPLOYMENT_V2_ESM2_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

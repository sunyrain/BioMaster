#!/usr/bin/env python3
"""Refit the expanded functional adapter and score the 720x108 kinase panel.

The refit epoch is fixed to the median grouped-OOF best epoch.  Three seeds are
averaged.  This creates a kinase-only candidate artifact and does not replace
the general 720x384 production ranking.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_kirhub_functional_adapter_v1 import (  # noqa: E402
    FunctionalInhibitionAdapter,
    rank_pairs,
    set_seed,
)


EXPANDED = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_pretrained_features_v1"
TRAINED = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_functional_adapter_v1"
DATA = EXPANDED / "KIRHUB_EXPANDED_FUNCTIONAL_PAIRS_V1.csv.gz"
DRUG_FEATURES = EXPANDED / "KIRHUB_EXPANDED_DRUG92_BERMOL768_FLOAT32_V1.npy"
TARGET_FEATURES = EXPANDED / "KIRHUB_EXPANDED_TARGET345_ESM2_1280_FLOAT32_V1.npy"
FIT_HISTORY = TRAINED / "KIRHUB_EXPANDED_FIT_HISTORY_V1.json"
OOF_CHECKPOINT = TRAINED / "KIRHUB_EXPANDED_GROUPED_FOLD_CHECKPOINTS_V1.pt"
SELECTION = TRAINED / "KIRHUB_EXPANDED_CROSSFIT_SELECTION_V1.csv"
DEPLOY = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1"
DEPLOY_DRUG_INDEX = DEPLOY / "DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz"
DEPLOY_DRUG_FEATURES = DEPLOY / "DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy"
DEPLOY_TARGET_INDEX = DEPLOY / "DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz"
DEPLOY_TARGET_FEATURES = DEPLOY / "DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy"
BASELINE = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
OUT = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_full_fit_kinase_candidate_v1"
SEEDS = [20260831, 20260847, 20260873]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_tensors(
    frame: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    drug = torch.from_numpy(np.asarray(
        drug_features[frame["drug_feature_index"].to_numpy(dtype=np.int64)], dtype=np.float32
    )).to(device)
    target = torch.from_numpy(np.asarray(
        target_features[frame["target_feature_index"].to_numpy(dtype=np.int64)], dtype=np.float32
    )).to(device)
    prior = torch.zeros((len(frame), 4), dtype=torch.float32, device=device)
    continuous = torch.from_numpy(
        frame["inhibition_fraction_1uM"].to_numpy(dtype=np.float32)
    ).to(device)
    binary = torch.from_numpy(
        frame["strong_inhibition_label"].to_numpy(dtype=np.float32)
    ).to(device)
    counts = frame.groupby("ligand_inchikey")["ligand_inchikey"].transform("size").to_numpy()
    weight = (1.0 / counts).astype(np.float32)
    weight /= weight.mean()
    return drug, target, prior, continuous, binary, torch.from_numpy(weight).to(device)


def refit_seed(
    frame: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    arguments: dict,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    set_seed(seed)
    current = full_tensors(frame, drug_features, target_features, device)
    high, low, contrast = rank_pairs(
        frame, np.arange(len(frame)), int(arguments["max_rank_pairs_per_drug"]), seed
    )
    high_t = torch.from_numpy(high).to(device)
    low_t = torch.from_numpy(low).to(device)
    contrast_t = torch.from_numpy(contrast).to(device)
    model = FunctionalInhibitionAdapter(
        use_priors=False, projection_dim=int(arguments["projection_dim"]),
        hidden_dim=int(arguments["hidden_dim"]), dropout=float(arguments["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(arguments["learning_rate"]),
        weight_decay=float(arguments["weight_decay"]),
    )
    drug, target, prior, continuous, binary, sample_weight = current
    weighted_positive = float((sample_weight * binary).sum().cpu())
    weighted_negative = float((sample_weight * (1 - binary)).sum().cpu())
    positive_weight = torch.tensor(
        weighted_negative / max(weighted_positive, 1e-8), device=device
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logit = model(drug, target, prior)
        prediction = torch.sigmoid(logit)
        regression = (
            torch.nn.functional.smooth_l1_loss(
                prediction, continuous, reduction="none", beta=0.10
            ) * sample_weight
        ).mean()
        binary_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logit, binary, reduction="none", pos_weight=positive_weight
            ) * sample_weight
        ).mean()
        rank_rows = torch.nn.functional.softplus(-(logit[high_t] - logit[low_t]))
        ranking = (rank_rows * contrast_t).sum() / contrast_t.sum().clamp_min(1e-8)
        loss = (
            regression + float(arguments["binary_weight"]) * binary_loss
            + float(arguments["rank_weight"]) * ranking
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append({
                "epoch": epoch, "loss": float(loss.detach().cpu()),
                "regression_loss": float(regression.detach().cpu()),
                "binary_loss": float(binary_loss.detach().cpu()),
                "rank_loss": float(ranking.detach().cpu()),
            })
    return copy.deepcopy({
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }), history


@torch.no_grad()
def score_panel(
    model: torch.nn.Module,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    drug_indices: np.ndarray,
    target_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(drug_indices), batch_size):
        stop = min(len(drug_indices), start + batch_size)
        drug = torch.from_numpy(np.asarray(
            drug_features[drug_indices[start:stop]], dtype=np.float32
        )).to(device)
        target = torch.from_numpy(np.asarray(
            target_features[target_indices[start:stop]], dtype=np.float32
        )).to(device)
        prior = torch.zeros((stop - start, 4), dtype=torch.float32, device=device)
        values.append(torch.sigmoid(model(drug, target, prior)).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    global TRAINED, FIT_HISTORY, OOF_CHECKPOINT, SELECTION, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-dir", default=str(TRAINED))
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    TRAINED = Path(args.trained_dir).resolve()
    FIT_HISTORY = TRAINED / "KIRHUB_EXPANDED_FIT_HISTORY_V1.json"
    OOF_CHECKPOINT = TRAINED / "KIRHUB_EXPANDED_GROUPED_FOLD_CHECKPOINTS_V1.pt"
    SELECTION = TRAINED / "KIRHUB_EXPANDED_CROSSFIT_SELECTION_V1.csv"
    OUT = Path(args.out_dir).resolve()
    required = [
        DATA, DRUG_FEATURES, TARGET_FEATURES, FIT_HISTORY, OOF_CHECKPOINT, SELECTION,
        DEPLOY_DRUG_INDEX, DEPLOY_DRUG_FEATURES, DEPLOY_TARGET_INDEX,
        DEPLOY_TARGET_FEATURES, BASELINE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA required")
    frame = pd.read_csv(DATA, low_memory=False)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    target_features = np.load(TARGET_FEATURES, mmap_mode="r")
    histories = json.loads(FIT_HISTORY.read_text())
    fit_epochs = int(np.median([row["best_epoch"] for row in histories]))
    package = torch.load(OOF_CHECKPOINT, map_location="cpu", weights_only=False)
    arguments = package["arguments"]
    states, full_histories = {}, {}
    for seed in SEEDS:
        state, history = refit_seed(
            frame, drug_features, target_features, arguments,
            fit_epochs, seed, device,
        )
        states[str(seed)] = state
        full_histories[str(seed)] = history

    deploy_drugs = pd.read_csv(DEPLOY_DRUG_INDEX).sort_values("drug_feature_index")
    deploy_targets = pd.read_csv(DEPLOY_TARGET_INDEX).sort_values("target_feature_index")
    kinase_targets = deploy_targets[deploy_targets["assay_lane"].eq("KINASE_BIOCHEMICAL")].copy()
    deploy_drug_features = np.load(DEPLOY_DRUG_FEATURES, mmap_mode="r")
    deploy_target_features = np.load(DEPLOY_TARGET_FEATURES, mmap_mode="r")
    grid = deploy_drugs[[
        "drug_feature_index", "ligand_inchikey", "drug_names"
    ]].assign(_key=1).merge(
        kinase_targets[[
            "target_feature_index", "target_chembl_id", "uniprot_accession", "gene_symbol"
        ]].assign(_key=1), on="_key", how="inner"
    ).drop(columns="_key")
    drug_indices = grid["drug_feature_index"].to_numpy(dtype=np.int64)
    target_indices = grid["target_feature_index"].to_numpy(dtype=np.int64)
    seed_scores = []
    for seed in SEEDS:
        model = FunctionalInhibitionAdapter(
            use_priors=False, projection_dim=int(arguments["projection_dim"]),
            hidden_dim=int(arguments["hidden_dim"]), dropout=float(arguments["dropout"]),
        ).to(device)
        model.load_state_dict(states[str(seed)])
        seed_scores.append(score_panel(
            model, deploy_drug_features, deploy_target_features,
            drug_indices, target_indices, device,
        ))
    grid["kirhub_functional_full_fit_score"] = np.mean(seed_scores, axis=0)
    grid["kirhub_functional_seed_std"] = np.std(seed_scores, axis=0)

    baseline = pd.read_csv(BASELINE, low_memory=False)[[
        "ligand_inchikey", "target_chembl_id", "old_drug_leakage_safe_score_v10",
        "dtiam_probability",
    ]]
    grid = grid.merge(
        baseline, on=["ligand_inchikey", "target_chembl_id"], how="left",
        validate="one_to_one",
    )
    selections = pd.read_csv(SELECTION)
    weights = {
        "biomaster": float(selections["biomaster_weight"].median()),
        "functional": float(selections["expanded_adapter_weight"].median()),
        "dtiam": float(selections["dtiam_weight"].median()),
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise RuntimeError(f"Median fusion weights do not sum to one: {weights}")
    for column in [
        "kirhub_functional_full_fit_score", "old_drug_leakage_safe_score_v10",
        "dtiam_probability",
    ]:
        grid[column + "_percentile_within_drug_108"] = grid.groupby(
            "ligand_inchikey"
        )[column].rank(pct=True, method="average")
    grid["kirhub_kinase_candidate_fusion_score"] = (
        weights["biomaster"] * grid["old_drug_leakage_safe_score_v10_percentile_within_drug_108"]
        + weights["functional"] * grid["kirhub_functional_full_fit_score_percentile_within_drug_108"]
        + weights["dtiam"] * grid["dtiam_probability_percentile_within_drug_108"]
    )
    grid["kirhub_kinase_candidate_rank_within_drug_108"] = grid.groupby(
        "ligand_inchikey"
    )["kirhub_kinase_candidate_fusion_score"].rank(method="min", ascending=False)
    training_drugs = set(frame["ligand_inchikey"])
    training_accessions = set(frame["uniprot_accession"])
    measured_pairs = set(zip(frame["ligand_inchikey"], frame["uniprot_accession"]))
    grid["kirhub_training_drug"] = grid["ligand_inchikey"].isin(training_drugs)
    grid["kirhub_training_target_accession"] = grid["uniprot_accession"].isin(training_accessions)
    grid["kirhub_measured_training_pair"] = [
        (drug, accession) in measured_pairs
        for drug, accession in zip(grid["ligand_inchikey"], grid["uniprot_accession"])
    ]

    checks = {
        "median_oof_best_epoch_valid": 1 <= fit_epochs <= int(arguments["epochs"]),
        "three_seed_full_fit": len(states) == 3,
        "exact_720x108_grid": len(grid) == 720 * 108,
        "all_scores_finite": np.isfinite(grid[[
            "kirhub_functional_full_fit_score", "kirhub_functional_seed_std",
            "kirhub_kinase_candidate_fusion_score",
        ]]).all().all(),
        "all_ranks_1_to_108": grid["kirhub_kinase_candidate_rank_within_drug_108"].between(1, 108).all(),
        "each_drug_has_108_targets": grid.groupby("ligand_inchikey").size().eq(108).all(),
        "median_weights_nonnegative_sum_one": (
            all(value >= 0.0 for value in weights.values())
            and np.isclose(sum(weights.values()), 1.0)
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    score_path = OUT / "KIRHUB_FULL_FIT_720X108_KINASE_CANDIDATE_V1.csv.gz"
    checkpoint_path = OUT / "KIRHUB_EXPANDED_FUNCTIONAL_FULL_FIT_3SEED_V1.pt"
    history_path = OUT / "KIRHUB_EXPANDED_FUNCTIONAL_FULL_FIT_HISTORY_V1.json"
    grid.to_csv(score_path, index=False, compression="gzip")
    torch.save({
        "states": states, "seeds": SEEDS, "fit_epochs": fit_epochs,
        "architecture_arguments": arguments, "fusion_weights": weights,
        "scope": "kinase-only 720x108 candidate; not general 720x384 production",
        "data_sha256": sha256(DATA),
    }, checkpoint_path)
    history_path.write_text(json.dumps(full_histories, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "fit_epochs": fit_epochs, "seeds": SEEDS,
        "fusion_weights": weights,
        "counts": {
            "drugs": 720, "kinase_targets": 108, "pairs": len(grid),
            "kirhub_training_drugs": int(grid.loc[grid["kirhub_training_drug"], "ligand_inchikey"].nunique()),
            "kirhub_training_targets_in_panel": int(grid.loc[
                grid["kirhub_training_target_accession"], "uniprot_accession"
            ].nunique()),
            "measured_training_pairs_in_panel": int(grid["kirhub_measured_training_pair"].sum()),
        },
        "checks": checks,
        "claim_boundary": (
            "Candidate kinase-only full-fit scoring artifact. OOF metrics, not these in-sample "
            "scores, remain the performance evidence. Do not merge this rank directly with "
            "non-kinase targets or call it untouched validation."
        ),
        "inputs_sha256": {path.name: sha256(path) for path in required},
        "artifacts": {
            score_path.name: sha256(score_path), checkpoint_path.name: sha256(checkpoint_path),
            history_path.name: sha256(history_path),
        },
    }
    summary_path = OUT / "KIRHUB_EXPANDED_FULL_FIT_KINASE_CANDIDATE_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train BioMaster-ODTI's first old-drug-oriented routed interaction ranker.

This is the first fully trainable model in the frozen benchmark program.  It
uses label-free Morgan and ProtBert inputs, an interaction-aware dual encoder,
family-conditioned mixture-of-experts routing, target-balanced BCE, a
within-target ranking loss, and a quantitative pChEMBL auxiliary objective.
The optional ``conplex_augmented`` variant adds the frozen external ConPLex
score as a separately auditable residual branch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_biomaster_odti_baselines_v1 import (  # noqa: E402
    PROTOCOLS,
    metrics,
    sha256,
    split_masks,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
STORE = BASE / "feature_store_v1"
PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = STORE / "MORGAN2048_UINT8_V1.npy"
PROTBERT = STORE / "PROTBERT1024_FLOAT32_V1.npy"
FEATURE_AUDIT = STORE / "FEATURE_STORE_AUDIT_V1.json"
BASELINE_METRICS = BASE / "baseline_results_v1/ALL_BASELINE_METRICS_V1.csv"
OUT = BASE / "biomaster_odti_routed_ranker_v1"


class RoutedInteractionRanker(nn.Module):
    def __init__(
        self,
        family_count: int,
        use_conplex: bool,
        embedding_dim: int = 192,
        hidden_dim: int = 256,
        expert_count: int = 6,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.use_conplex = use_conplex
        self.drug_encoder = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.family_embedding = nn.Embedding(family_count, 24)
        pair_width = embedding_dim * 4 + 24
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.shared_head = nn.Linear(hidden_dim, 1)
        self.expert_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(expert_count)])
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + 24, 96),
            nn.GELU(),
            nn.Linear(96, expert_count),
        )
        self.affinity_head = nn.Linear(hidden_dim, 1)
        self.dot_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        if use_conplex:
            self.conplex_weight = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
            self.conplex_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        drug: torch.Tensor,
        target: torch.Tensor,
        family: torch.Tensor,
        conplex: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drug_encoded = self.drug_encoder(drug)
        target_encoded = self.target_encoder(target)
        family_encoded = self.family_embedding(family)
        pair = torch.cat(
            [
                drug_encoded,
                target_encoded,
                drug_encoded * target_encoded,
                torch.abs(drug_encoded - target_encoded),
                family_encoded,
            ],
            dim=1,
        )
        hidden = self.pair_trunk(pair)
        experts = torch.cat([head(hidden) for head in self.expert_heads], dim=1)
        gate = torch.softmax(self.gate(torch.cat([target_encoded, family_encoded], dim=1)), dim=1)
        routed = (gate * experts).sum(dim=1)
        normalized_drug = torch.nn.functional.normalize(drug_encoded, dim=1)
        normalized_target = torch.nn.functional.normalize(target_encoded, dim=1)
        dot = (normalized_drug * normalized_target).sum(dim=1)
        logit = self.shared_head(hidden).squeeze(1) + routed + self.dot_scale.exp() * dot
        if self.use_conplex:
            logit = logit + self.conplex_weight * conplex + self.conplex_bias
        affinity = self.affinity_head(hidden).squeeze(1)
        return logit, affinity, gate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60, 60)
    return 1.0 / (1.0 + np.exp(-value))


def target_class_weights(data: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    selected = data.iloc[positions][["target_feature_index", "binary_label"]].copy()
    counts = selected.groupby(["target_feature_index", "binary_label"]).size()
    raw = np.asarray(
        [1.0 / counts.loc[(target, label)] for target, label in selected.itertuples(index=False)],
        dtype=np.float32,
    )
    return raw / raw.mean()


def prepare_arrays(
    data: pd.DataFrame,
    train_positions: np.ndarray,
    target_features: np.ndarray,
) -> dict[str, object]:
    families = sorted(data["target_assay_family"].astype(str).unique())
    family_lookup = {name: index for index, name in enumerate(families)}
    family_index = data["target_assay_family"].astype(str).map(family_lookup).to_numpy(dtype=np.int64)
    train_target_indices = np.unique(data.iloc[train_positions]["target_feature_index"].to_numpy(dtype=np.int64))
    train_target_features = np.asarray(target_features[train_target_indices], dtype=np.float32)
    target_mean = train_target_features.mean(axis=0).astype(np.float32)
    target_std = train_target_features.std(axis=0).astype(np.float32)
    target_std[target_std < 1e-6] = 1.0
    conplex = data["conplex_score"].to_numpy(dtype=np.float32)
    conplex_mean = float(conplex[train_positions].mean())
    conplex_std = float(conplex[train_positions].std())
    if conplex_std < 1e-8:
        conplex_std = 1.0
    affinity = pd.to_numeric(data["mean_pchembl"], errors="coerce").to_numpy(dtype=np.float32)
    affinity_train = affinity[train_positions]
    affinity_mean = float(np.nanmean(affinity_train))
    affinity_std = float(np.nanstd(affinity_train))
    if affinity_std < 1e-8:
        affinity_std = 1.0
    return {
        "families": families,
        "family_index": family_index,
        "target_mean": target_mean,
        "target_std": target_std,
        "conplex": conplex,
        "conplex_mean": conplex_mean,
        "conplex_std": conplex_std,
        "affinity": affinity,
        "affinity_mean": affinity_mean,
        "affinity_std": affinity_std,
    }


def batch_inputs(
    positions: np.ndarray,
    data: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    arrays: dict[str, object],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    part = data.iloc[positions]
    drug_index = part["drug_feature_index"].to_numpy(dtype=np.int64)
    target_index = part["target_feature_index"].to_numpy(dtype=np.int64)
    drug = torch.from_numpy(np.asarray(drug_features[drug_index], dtype=np.float32)).to(device)
    target_np = np.asarray(target_features[target_index], dtype=np.float32)
    target_np = (target_np - arrays["target_mean"]) / arrays["target_std"]
    target = torch.from_numpy(target_np).to(device)
    family = torch.from_numpy(np.asarray(arrays["family_index"])[positions]).to(device)
    conplex_np = (np.asarray(arrays["conplex"])[positions] - arrays["conplex_mean"]) / arrays["conplex_std"]
    conplex = torch.from_numpy(conplex_np.astype(np.float32)).to(device)
    label = torch.from_numpy(part["binary_label"].to_numpy(dtype=np.float32)).to(device)
    affinity_np = (np.asarray(arrays["affinity"])[positions] - arrays["affinity_mean"]) / arrays["affinity_std"]
    affinity = torch.from_numpy(affinity_np.astype(np.float32)).to(device)
    target_group = torch.from_numpy(target_index.copy()).to(device)
    return drug, target, family, conplex, label, affinity, target_group


def within_target_rank_loss(
    logits: torch.Tensor, labels: torch.Tensor, target_group: torch.Tensor
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for target in torch.unique(target_group):
        selected = target_group.eq(target)
        positive = logits[selected & labels.gt(0.5)]
        negative = logits[selected & labels.lt(0.5)]
        if positive.numel() and negative.numel():
            losses.append(torch.nn.functional.softplus(negative.mean() - positive.mean()))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def infer(
    model: nn.Module,
    positions: np.ndarray,
    data: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    arrays: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    affinities: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    for start in range(0, len(positions), batch_size):
        batch = positions[start : start + batch_size]
        drug, target, family, conplex, _, _, _ = batch_inputs(
            batch, data, drug_features, target_features, arrays, device
        )
        logit, affinity, gate = model(drug, target, family, conplex)
        logits.append(logit.cpu().numpy())
        affinities.append(affinity.cpu().numpy())
        gates.append(gate.cpu().numpy())
    return np.concatenate(logits), np.concatenate(affinities), np.concatenate(gates)


def temperature_scale(validation_logits: np.ndarray, validation_y: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        scaled = validation_logits / math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - validation_y * scaled))

    fit = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not fit.success:
        return 1.0
    return float(math.exp(fit.x))


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    feature_audit = json.loads(FEATURE_AUDIT.read_text())
    if feature_audit.get("status") != "PASS":
        raise RuntimeError("Feature store must pass")
    data = pd.read_csv(PAIRS, low_memory=False)
    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    fold = -1 if args.protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"} else args.fold
    masks = split_masks(data, args.protocol, fold)
    available = data["drug_feature_available"].to_numpy(dtype=bool)
    train_positions, valid_positions, test_positions = [
        np.flatnonzero(masks[name] & available) for name in ["train", "valid", "test"]
    ]
    arrays = prepare_arrays(data, train_positions, target_features)
    weights = target_class_weights(data, train_positions)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RoutedInteractionRanker(
        family_count=len(arrays["families"]),
        use_conplex=args.variant == "conplex_augmented",
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        expert_count=args.experts,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 20)
    best_ap = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    no_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_positions))
        loss_sum = bce_sum = rank_sum = affinity_sum = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            local = order[start : start + args.batch_size]
            positions = train_positions[local]
            drug, target, family, conplex, label, affinity, target_group = batch_inputs(
                positions, data, drug_features, target_features, arrays, device
            )
            row_weight = torch.from_numpy(weights[local]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logit, affinity_prediction, _ = model(drug, target, family, conplex)
            smoothed_label = label * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            bce_per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, smoothed_label, reduction="none"
            )
            bce = (bce_per_row * row_weight).mean()
            ranking = within_target_rank_loss(logit, label, target_group)
            affinity_available = torch.isfinite(affinity)
            if affinity_available.any():
                affinity_loss = torch.nn.functional.huber_loss(
                    affinity_prediction[affinity_available], affinity[affinity_available], delta=1.0
                )
            else:
                affinity_loss = logit.sum() * 0.0
            loss = bce + args.rank_weight * ranking + args.affinity_weight * affinity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            bce_sum += float(bce.detach())
            rank_sum += float(ranking.detach())
            affinity_sum += float(affinity_loss.detach())
            steps += 1
        scheduler.step()
        validation_logits, _, _ = infer(
            model, valid_positions, data, drug_features, target_features, arrays, device, args.inference_batch_size
        )
        validation_y = data.iloc[valid_positions]["binary_label"].to_numpy(dtype=np.int8)
        validation_ap = float(average_precision_score(validation_y, validation_logits))
        row = {
            "epoch": epoch,
            "loss": loss_sum / steps,
            "bce": bce_sum / steps,
            "rank_loss": rank_sum / steps,
            "affinity_loss": affinity_sum / steps,
            "valid_micro_auprc": validation_ap,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_ap > best_ap + args.min_delta:
            best_ap = validation_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device)
    validation_logits, _, _ = infer(
        model, valid_positions, data, drug_features, target_features, arrays, device, args.inference_batch_size
    )
    validation_y = data.iloc[valid_positions]["binary_label"].to_numpy(dtype=np.int8)
    temperature = temperature_scale(validation_logits, validation_y)
    test_logits, test_affinity, test_gate = infer(
        model, test_positions, data, drug_features, target_features, arrays, device, args.inference_batch_size
    )
    test_probability = sigmoid(test_logits / temperature)
    test = data.iloc[test_positions].reset_index(drop=True)
    result_metrics = metrics(test, test_probability)

    variant = args.variant.upper()
    run_name = f"{args.protocol}__fold_{fold}__seed_{args.seed}__{variant}"
    run_dir = OUT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "BEST_MODEL_V1.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "RoutedInteractionRanker",
            "variant": args.variant,
            "families": arrays["families"],
            "normalization": {
                "target_mean": arrays["target_mean"],
                "target_std": arrays["target_std"],
                "conplex_mean": arrays["conplex_mean"],
                "conplex_std": arrays["conplex_std"],
                "affinity_mean": arrays["affinity_mean"],
                "affinity_std": arrays["affinity_std"],
            },
            "architecture": {
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "experts": args.experts,
                "dropout": args.dropout,
            },
            "temperature": temperature,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(run_dir / "TRAINING_HISTORY_V1.csv", index=False)
    predictions = test[[
        "calibration_pair_id", "sequence_key", "target_chembl_id", "primary_gene",
        "target_assay_family", "parent_standard_inchi_key", "parent_molecule_chembl_id",
        "binary_label", "mean_pchembl", "conplex_score",
    ]].copy()
    predictions["biomaster_logit"] = test_logits
    predictions["biomaster_probability_calibrated"] = test_probability
    predictions["pchembl_aux_prediction"] = test_affinity * arrays["affinity_std"] + arrays["affinity_mean"]
    predictions["route_expert_index"] = test_gate.argmax(axis=1)
    predictions["route_max_probability"] = test_gate.max(axis=1)
    prediction_path = run_dir / "TEST_PREDICTIONS_V1.csv.gz"
    predictions.to_csv(prediction_path, index=False)

    comparator_rows: list[dict[str, object]] = []
    if BASELINE_METRICS.is_file():
        baselines = pd.read_csv(BASELINE_METRICS)
        comparator = baselines[
            baselines["protocol"].eq(args.protocol)
            & pd.to_numeric(baselines["fold"]).eq(fold)
        ]
        comparator_rows = comparator[["model", "micro_auroc", "micro_auprc", "target_macro_auprc", "drug_macro_auprc"]].to_dict("records")
    checks = {
        "feature_store_pass": feature_audit["status"] == "PASS",
        "train_valid_test_nonempty": all(len(x) > 0 for x in [train_positions, valid_positions, test_positions]),
        "test_has_both_classes": test["binary_label"].nunique() == 2,
        "best_checkpoint_selected_by_validation_only": best_epoch > 0,
        "calibration_temperature_fit_on_validation_only": np.isfinite(temperature) and temperature > 0,
        "all_test_predictions_finite": np.isfinite(test_probability).all(),
        "all_test_predictions_bounded": ((test_probability >= 0) & (test_probability <= 1)).all(),
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": "BIOMASTER_ODTI_ROUTED_INTERACTION_RANKER_V1",
        "variant": args.variant,
        "protocol": args.protocol,
        "fold": fold,
        "seed": args.seed,
        "device": str(device),
        "split_counts": {
            "train": int(len(train_positions)),
            "valid": int(len(valid_positions)),
            "test": int(len(test_positions)),
            "quarantined_total_pairs": int((~available).sum()),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_micro_auprc": best_ap,
            "epochs_completed": len(history),
            "temperature": temperature,
        },
        "objectives": {
            "target_class_balanced_bce": 1.0,
            "within_target_rank_weight": args.rank_weight,
            "pchembl_huber_aux_weight": args.affinity_weight,
        },
        "test_metrics": result_metrics,
        "same_split_baselines": comparator_rows,
        "checks": {key: bool(value) for key, value in checks.items()},
        "artifacts": {
            "checkpoint_sha256": sha256(checkpoint),
            "predictions_sha256": sha256(prediction_path),
        },
        "claim_status": "EXPERIMENTAL_RESULT_ONLY_NOT_SOTA_UNTIL_MULTI_SEED_ALL_FOLDS_EXTERNAL_COMPARATORS",
    }
    result_path = run_dir / "RUN_SUMMARY_V1.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps({
        "status": result["status"],
        "run": run_name,
        "best_epoch": best_epoch,
        "test_micro_auroc": result_metrics["micro_auroc"],
        "test_micro_auprc": result_metrics["micro_auprc"],
        "target_macro_auprc": result_metrics["target_macro_auprc"],
        "drug_macro_auprc": result_metrics["drug_macro_auprc"],
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--variant", choices=["core", "conplex_augmented"], default="core")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.12)
    parser.add_argument("--affinity-weight", type=float, default=0.06)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.fold not in range(5):
        raise ValueError("--fold must be in 0..4")
    train(args)


if __name__ == "__main__":
    main()

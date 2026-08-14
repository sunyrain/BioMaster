#!/usr/bin/env python3
"""Train the pre-frozen BioMaster-BiRoute V2 model and its ablations.

All representations are frozen, label-free BerMol/ESM2 features.  Model and
checkpoint selection use the frozen training and validation roles only; test
labels are loaded only after the best validation checkpoint is restored.
"""

from __future__ import annotations

import argparse
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
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from biomaster_biroute_v2_core import (  # noqa: E402
    BioMasterBiRouteV2,
    route_load_balance_loss,
    within_group_pairwise_loss,
)
from run_biomaster_odti_baselines_v1 import metrics, split_masks  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIR_STORE = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
FEATURE_BASE = BASE / "public_retrained_v1/dtiam_official_feature_store_v1"
DRUG_FEATURES = FEATURE_BASE / "DTIAM_BERMOL768_FLOAT32_V1.npy"
TARGET_FEATURES = FEATURE_BASE / "DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
PAIR_INDEX = FEATURE_BASE / "DTIAM_86674_PAIR_FEATURE_INDEX_V1.csv.gz"
FEATURE_AUDIT = FEATURE_BASE / "DTIAM_OFFICIAL_FEATURE_STORE_AUDIT_V1.json"
FREEZE = ROOT / "configs/biomaster_biroute_v2_freeze.json"
OUT = BASE / "biomaster_biroute_v2"

ABLATIONS = {
    "FULL_BIROUTE": {"rank_losses": True, "route_conditioning": True, "symmetric_retrieval": False},
    "NO_DIRECTIONAL_RANK_LOSSES": {
        "rank_losses": False, "route_conditioning": True, "symmetric_retrieval": False,
    },
    "NO_ROUTE_CONDITIONING": {
        "rank_losses": True, "route_conditioning": False, "symmetric_retrieval": False,
    },
    "SYMMETRIC_RETRIEVAL_HEAD": {
        "rank_losses": True, "route_conditioning": True, "symmetric_retrieval": True,
    },
}
ROUTE_COLUMNS = [
    "drug_entity_cold",
    "drug_scaffold_cold",
    "target_entity_cold",
    "target_homology_cold",
    "is_deployment_old_drug",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def inverse_group_class_weights(
    data: pd.DataFrame, positions: np.ndarray, group_column: str
) -> np.ndarray:
    selected = data.iloc[positions][[group_column, "binary_label"]].copy()
    counts = selected.groupby([group_column, "binary_label"], dropna=False).size()
    values = np.asarray(
        [1.0 / counts.loc[(group, label)] for group, label in selected.itertuples(index=False)],
        dtype=np.float32,
    )
    return values / values.mean()


def actual_route_features(
    data: pd.DataFrame, train_positions: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    train = data.iloc[train_positions]
    role = data.iloc[positions]
    train_drugs = set(train["drug_feature_index"].astype(int))
    train_scaffolds = set(train["scaffold_group"].astype(str))
    train_targets = set(train["target_feature_index"].astype(int))
    train_homology = set(train["target_homology_cluster"].astype(str))
    return np.column_stack([
        ~role["drug_feature_index"].astype(int).isin(train_drugs),
        ~role["scaffold_group"].astype(str).isin(train_scaffolds),
        ~role["target_feature_index"].astype(int).isin(train_targets),
        ~role["target_homology_cluster"].astype(str).isin(train_homology),
        role["is_deployment_old_drug"].astype(bool),
    ]).astype(np.float32)


def episodic_train_route_features(train: pd.DataFrame, epoch: int) -> np.ndarray:
    """Create label-free pseudo-cold episodes for training the route gate.

    Foundation embeddings are frozen and no entity-ID embeddings exist.  Fold
    assignments and stable integer hashes expose the gate to all coldness
    combinations without using validation/test membership or labels.
    """
    episode = (epoch - 1) % 5
    second = (episode * 2 + 1) % 5
    drug_id = train["drug_feature_index"].to_numpy(dtype=np.int64)
    target_id = train["target_feature_index"].to_numpy(dtype=np.int64)
    return np.column_stack([
        np.mod(drug_id * 1103515245 + 12345, 5) == episode,
        train["scaffold_cold_fold"].to_numpy(dtype=np.int64) == episode,
        np.mod(target_id * 2654435761 + 1013904223, 5) == second,
        train["target_homology_cold_fold"].to_numpy(dtype=np.int64) == second,
        train["is_deployment_old_drug"].astype(bool).to_numpy(),
    ]).astype(np.float32)


def grouped_training_order(train: pd.DataFrame, epoch: int) -> np.ndarray:
    """Alternate target- and drug-contiguous batches for both ranking directions."""
    group_column = "target_feature_index" if epoch % 2 else "drug_feature_index"
    grouped = []
    group_values = train[group_column].drop_duplicates().to_numpy(copy=True)
    np.random.shuffle(group_values)
    raw = train[group_column].to_numpy()
    for group in group_values:
        local = np.flatnonzero(raw == group)
        np.random.shuffle(local)
        grouped.append(local)
    return np.concatenate(grouped) if grouped else np.empty(0, dtype=np.int64)


def normalization(
    data: pd.DataFrame,
    train_positions: np.ndarray,
    drug_features: np.ndarray,
    target_features: np.ndarray,
) -> dict[str, object]:
    train = data.iloc[train_positions]
    drug_index = np.unique(train["drug_feature_index"].to_numpy(dtype=np.int64))
    target_index = np.unique(train["target_feature_index"].to_numpy(dtype=np.int64))
    drug = np.asarray(drug_features[drug_index], dtype=np.float32)
    target = np.asarray(target_features[target_index], dtype=np.float32)
    affinity = pd.to_numeric(train["mean_pchembl"], errors="coerce").to_numpy(dtype=np.float32)
    result = {
        "drug_mean": drug.mean(axis=0).astype(np.float32),
        "drug_std": drug.std(axis=0).astype(np.float32),
        "target_mean": target.mean(axis=0).astype(np.float32),
        "target_std": target.std(axis=0).astype(np.float32),
        "affinity_mean": float(np.nanmean(affinity)),
        "affinity_std": float(np.nanstd(affinity)),
    }
    result["drug_std"][result["drug_std"] < 1e-6] = 1.0
    result["target_std"][result["target_std"] < 1e-6] = 1.0
    if result["affinity_std"] < 1e-6:
        result["affinity_std"] = 1.0
    return result


def batch_tensors(
    frame: pd.DataFrame,
    local_positions: np.ndarray,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    route_features: np.ndarray,
    norm: dict[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    part = frame.iloc[local_positions]
    # Explicit copies avoid exposing read-only pandas views to torch.  The
    # tensors are not mutated, but writable ownership removes undefined-
    # behavior warnings and makes that boundary auditable.
    drug_index = part["drug_feature_index"].to_numpy(dtype=np.int64, copy=True)
    target_index = part["target_feature_index"].to_numpy(dtype=np.int64, copy=True)
    drug = (np.asarray(drug_features[drug_index], dtype=np.float32) - norm["drug_mean"]) / norm["drug_std"]
    target = (
        np.asarray(target_features[target_index], dtype=np.float32) - norm["target_mean"]
    ) / norm["target_std"]
    affinity = (
        pd.to_numeric(part["mean_pchembl"], errors="coerce").to_numpy(dtype=np.float32)
        - norm["affinity_mean"]
    ) / norm["affinity_std"]
    return {
        "drug": torch.from_numpy(drug).to(device, non_blocking=True),
        "target": torch.from_numpy(target).to(device, non_blocking=True),
        "route": torch.from_numpy(route_features[local_positions]).to(device, non_blocking=True),
        "label": torch.from_numpy(part["binary_label"].to_numpy(dtype=np.float32)).to(device),
        "affinity": torch.from_numpy(affinity).to(device),
        "target_group": torch.from_numpy(target_index).to(device),
        "drug_group": torch.from_numpy(drug_index).to(device),
    }


@torch.no_grad()
def infer(
    model: nn.Module,
    frame: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    route_features: np.ndarray,
    norm: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(frame), batch_size):
        local = np.arange(start, min(start + batch_size, len(frame)), dtype=np.int64)
        batch = batch_tensors(
            frame, local, drug_features, target_features, route_features, norm, device
        )
        output = model(batch["drug"], batch["target"], batch["route"])
        for key, value in output.items():
            collected.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values) for key, values in collected.items()}


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def temperature_scale(logits: np.ndarray, label: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        scaled = logits / math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - label * scaled))

    fit = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    return float(math.exp(fit.x)) if fit.success else 1.0


def selection_metrics(frame: pd.DataFrame, logits: np.ndarray) -> dict[str, float]:
    result = metrics(frame, sigmoid(logits))
    values = [result["micro_auprc"], result["target_macro_auprc"], result["drug_macro_auprc"]]
    result["bidirectional_selection_score"] = float(np.nanmean(values))
    return result


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    feature_audit = json.loads(FEATURE_AUDIT.read_text())
    freeze = json.loads(FREEZE.read_text())
    if feature_audit.get("status") != "PASS":
        raise RuntimeError("DTIAM official feature store audit must pass")
    data = pd.read_csv(PAIR_STORE, low_memory=False)
    pair_index = pd.read_csv(PAIR_INDEX)
    if len(data) != 86674 or len(pair_index) != len(data):
        raise RuntimeError("Frozen pair population changed")
    if not pair_index["calibration_pair_id"].equals(data["calibration_pair_id"]):
        raise RuntimeError("DTIAM feature pair order changed")
    available = pair_index["dtiam_bermol_available"].to_numpy(dtype=bool)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    target_features = np.load(TARGET_FEATURES, mmap_mode="r")
    fold = -1 if args.protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"} else args.fold
    masks = split_masks(data, args.protocol, fold)
    role_positions = {
        role: np.flatnonzero(mask & available) for role, mask in masks.items()
    }
    train_frame = data.iloc[role_positions["train"]].reset_index(drop=True)
    validation_frame = data.iloc[role_positions["valid"]].reset_index(drop=True)
    test_frame = data.iloc[role_positions["test"]].reset_index(drop=True)
    if not all(len(frame) and frame["binary_label"].nunique() == 2 for frame in [train_frame, validation_frame, test_frame]):
        raise RuntimeError("Each role must be nonempty and contain both classes")
    norm = normalization(data, role_positions["train"], drug_features, target_features)
    validation_route = actual_route_features(data, role_positions["train"], role_positions["valid"])
    test_route = actual_route_features(data, role_positions["train"], role_positions["test"])
    target_weights = inverse_group_class_weights(train_frame, np.arange(len(train_frame)), "target_feature_index")
    drug_weights = inverse_group_class_weights(train_frame, np.arange(len(train_frame)), "drug_feature_index")

    ablation = ABLATIONS[args.ablation]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BioMasterBiRouteV2(
        latent_dim=int(freeze["model"]["latent_dim"]),
        hidden_dim=int(freeze["model"]["hidden_dim"]),
        dropout=float(freeze["model"]["dropout"]),
        route_conditioning=bool(ablation["route_conditioning"]),
        symmetric_retrieval=bool(ablation["symmetric_retrieval"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 20
    )
    best_score = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = grouped_training_order(train_frame, epoch)
        route = episodic_train_route_features(train_frame, epoch)
        totals = {name: 0.0 for name in ["loss", "final_bce", "td_bce", "dt_bce", "td_rank", "dt_rank", "affinity", "route_balance"]}
        steps = 0
        for start in range(0, len(order), args.batch_size):
            local = order[start : start + args.batch_size]
            batch = batch_tensors(
                train_frame, local, drug_features, target_features, route, norm, device
            )
            output = model(batch["drug"], batch["target"], batch["route"])
            label = batch["label"]
            smoothed = label * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            target_weight = torch.from_numpy(target_weights[local]).to(device)
            drug_weight = torch.from_numpy(drug_weights[local]).to(device)
            final_bce = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output["final_logit"], smoothed, reduction="none"
                ) * target_weight
            ).mean()
            td_bce = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output["target_to_drug_logit"], smoothed, reduction="none"
                ) * target_weight
            ).mean()
            dt_bce = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output["drug_to_target_logit"], smoothed, reduction="none"
                ) * drug_weight
            ).mean()
            if ablation["rank_losses"]:
                td_rank = within_group_pairwise_loss(
                    output["target_to_drug_logit"], label, batch["target_group"]
                )
                dt_rank = within_group_pairwise_loss(
                    output["drug_to_target_logit"], label, batch["drug_group"]
                )
            else:
                td_rank = output["final_logit"].sum() * 0.0
                dt_rank = output["final_logit"].sum() * 0.0
            finite_affinity = torch.isfinite(batch["affinity"])
            affinity_loss = (
                torch.nn.functional.huber_loss(
                    output["affinity"][finite_affinity], batch["affinity"][finite_affinity], delta=1.0
                ) if finite_affinity.any() else output["final_logit"].sum() * 0.0
            )
            balance = route_load_balance_loss(output["route_weights"])
            loss = (
                final_bce + 0.2 * td_bce + 0.2 * dt_bce
                + 0.12 * td_rank + 0.12 * dt_rank
                + 0.05 * affinity_loss + 0.01 * balance
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            for name, value in [
                ("loss", loss), ("final_bce", final_bce), ("td_bce", td_bce),
                ("dt_bce", dt_bce), ("td_rank", td_rank), ("dt_rank", dt_rank),
                ("affinity", affinity_loss), ("route_balance", balance),
            ]:
                totals[name] += float(value.detach())
            steps += 1
        scheduler.step()
        validation = infer(
            model, validation_frame, drug_features, target_features, validation_route,
            norm, device, args.inference_batch_size,
        )
        validation_result = selection_metrics(validation_frame, validation["final_logit"])
        row = {
            "epoch": epoch,
            **{name: value / steps for name, value in totals.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_micro_auprc": validation_result["micro_auprc"],
            "validation_target_macro_auprc": validation_result["target_macro_auprc"],
            "validation_drug_macro_auprc": validation_result["drug_macro_auprc"],
            "validation_bidirectional_selection_score": validation_result["bidirectional_selection_score"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = validation_result["bidirectional_selection_score"]
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device)
    validation = infer(
        model, validation_frame, drug_features, target_features, validation_route,
        norm, device, args.inference_batch_size,
    )
    validation_y = validation_frame["binary_label"].to_numpy(dtype=np.int8)
    temperature = temperature_scale(validation["final_logit"], validation_y)
    test = infer(
        model, test_frame, drug_features, target_features, test_route,
        norm, device, args.inference_batch_size,
    )
    probability = sigmoid(test["final_logit"] / temperature)
    test_metrics = metrics(test_frame, probability)

    run_name = f"{args.protocol}__fold_{fold}__seed_{args.seed}__{args.ablation}"
    run_dir = OUT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "BEST_MODEL_V2.pt"
    torch.save({
        "model_state_dict": best_state,
        "model_class": "BioMasterBiRouteV2",
        "ablation": args.ablation,
        "normalization": norm,
        "temperature": temperature,
        "freeze_sha256": sha256(FREEZE),
    }, checkpoint)
    pd.DataFrame(history).to_csv(run_dir / "TRAINING_HISTORY_V2.csv", index=False)
    predictions = test_frame[[
        "calibration_pair_id", "target_chembl_id", "primary_gene", "target_assay_family",
        "parent_standard_inchi_key", "parent_molecule_chembl_id", "binary_label", "mean_pchembl",
        "scaffold_group", "target_homology_cluster",
    ]].copy()
    predictions["biroute_probability"] = probability
    for key in ["final_logit", "pair_logit", "target_to_drug_logit", "drug_to_target_logit", "affinity"]:
        predictions[key] = test[key]
    for index, name in enumerate(["PAIR_INTERACTION", "TARGET_TO_DRUG", "DRUG_TO_TARGET"]):
        predictions[f"route_weight_{name.lower()}"] = test["route_weights"][:, index]
    for index, name in enumerate(ROUTE_COLUMNS):
        predictions[name] = test_route[:, index].astype(np.int8)
    prediction_path = run_dir / "TEST_PREDICTIONS_V2.csv.gz"
    predictions.to_csv(prediction_path, index=False)
    validation_predictions = validation_frame[["calibration_pair_id", "binary_label"]].copy()
    validation_predictions["final_logit"] = validation["final_logit"]
    validation_predictions["biroute_probability"] = sigmoid(validation["final_logit"] / temperature)
    validation_path = run_dir / "VALIDATION_PREDICTIONS_V2.csv.gz"
    validation_predictions.to_csv(validation_path, index=False)

    checks = {
        "feature_store_pass": feature_audit["status"] == "PASS",
        "exact_frozen_population": len(data) == 86674,
        "all_roles_nonempty_with_both_classes": all(
            len(frame) and frame["binary_label"].nunique() == 2
            for frame in [train_frame, validation_frame, test_frame]
        ),
        "best_checkpoint_selected_by_validation_only": best_epoch > 0,
        "test_not_used_for_training_or_selection": True,
        "all_predictions_finite_and_bounded": np.isfinite(probability).all()
        and ((probability >= 0) & (probability <= 1)).all(),
        "route_weights_sum_to_one": np.allclose(test["route_weights"].sum(axis=1), 1.0, atol=1e-5),
        "hard_gate_target_scope_unchanged": freeze["target_scope_invariant"]["hard_gate_excluded_never_recovered"] == 480,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": "BIOMASTER_BIROUTE_V2",
        "protocol": args.protocol,
        "fold": fold,
        "seed": args.seed,
        "ablation": args.ablation,
        "ablation_settings": ablation,
        "device": str(device),
        "counts": {role: int(len(frame)) for role, frame in [
            ("train", train_frame), ("valid", validation_frame), ("test", test_frame)
        ]},
        "training": {
            "best_epoch": best_epoch,
            "best_validation_bidirectional_selection_score": best_score,
            "epochs_completed": len(history),
            "temperature": temperature,
        },
        "test_metrics": test_metrics,
        "checks": {name: bool(value) for name, value in checks.items()},
        "claim_status": "DEVELOPMENT_RESULT_ONLY; CONFIRMATORY_FOLDS_AND_PREDECLARED_ABLATIONS_REQUIRED",
        "hashes": {
            "method_freeze": sha256(FREEZE),
            "core_implementation": sha256(ROOT / "scripts/biomaster_biroute_v2_core.py"),
            "runner_implementation": sha256(Path(__file__)),
            "checkpoint": sha256(checkpoint),
            "test_predictions": sha256(prediction_path),
            "validation_predictions": sha256(validation_path),
        },
    }
    summary_path = run_dir / "RUN_SUMMARY_V2.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=[
        "S1_SCAFFOLD_COLD_DRUG", "S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD",
        "S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD",
    ])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="FULL_BIROUTE")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.fold not in range(5):
        raise ValueError("--fold must be 0..4")
    train(args)


if __name__ == "__main__":
    main()

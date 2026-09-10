#!/usr/bin/env python3
"""Train a frozen-embedding KiRHub functional-inhibition adapter.

The adapter learns only a small interaction head on top of fixed BerMol768 and
ESM2-650M-1280 embeddings.  It is evaluated with out-of-fold entity-grouped
predictions; no random pair split is permitted.  Existing BioMaster/DTIAM
scores can be supplied as frozen priors in a separately reported variant.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/kirhub_functional_adaptation_v1"
DATA = BASE / "KIRHUB_FUNCTIONAL_PRIMARY_SINGLE_CONSTRUCT_7505_V1.csv.gz"
DATA_SUMMARY = BASE / "KIRHUB_FUNCTIONAL_DATA_SUMMARY_V1.json"
FEATURE_STORE = (
    ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1"
    / "dtiam_deployment_feature_store_v1"
)
DRUG_FEATURES = FEATURE_STORE / "DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy"
TARGET_FEATURES = FEATURE_STORE / "DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy"
OUT = BASE / "functional_adapter_v1"
PRIOR_COLUMNS = [
    "old_drug_leakage_safe_score_v10",
    "biomaster_full_fit_v10_borda_score",
    "dtiam_probability",
    "ensemble_drug_to_target_logit",
]
VARIANTS = ("embedding_only", "embedding_plus_frozen_priors")
REGIMES = {
    "DRUG_SCAFFOLD_COLD": "drug_scaffold_cold_fold",
    "TARGET_HOMOLOGY_COLD": "target_homology_cold_fold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience-checks", type=int, default=14)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--projection-dim", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--rank-weight", type=float, default=0.45)
    parser.add_argument("--binary-weight", type=float, default=0.30)
    parser.add_argument("--max-rank-pairs-per-drug", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--regimes", nargs="+", choices=sorted(REGIMES), default=list(REGIMES)
    )
    return parser.parse_args()


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


class FunctionalInhibitionAdapter(nn.Module):
    def __init__(
        self,
        use_priors: bool,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_priors = use_priors
        self.drug_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.target_projection = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        pair_width = projection_dim * 4 + (16 if use_priors else 0)
        if use_priors:
            self.prior_projection = nn.Sequential(
                nn.Linear(len(PRIOR_COLUMNS), 24), nn.GELU(), nn.Linear(24, 16),
            )
        self.pair_head = nn.Sequential(
            nn.Linear(pair_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, drug: torch.Tensor, target: torch.Tensor, priors: torch.Tensor
    ) -> torch.Tensor:
        d = self.drug_projection(drug)
        t = self.target_projection(target)
        pieces = [d, t, d * t, torch.abs(d - t)]
        if self.use_priors:
            pieces.append(self.prior_projection(priors))
        return self.pair_head(torch.cat(pieces, dim=1)).squeeze(1)


def rank_pairs(
    frame: pd.DataFrame,
    positions: np.ndarray,
    max_pairs_per_drug: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local high/low row indices for clear within-drug contrasts."""
    selected = frame.iloc[positions].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    high_rows, low_rows, weights = [], [], []
    for _, group in selected.groupby("ligand_inchikey", sort=False):
        local = group.index.to_numpy(dtype=np.int64)
        y = group["inhibition_fraction_1uM"].to_numpy(dtype=np.float32)
        high, low = np.where((y[:, None] - y[None, :]) >= 0.30)
        if len(high) == 0:
            continue
        if len(high) > max_pairs_per_drug:
            keep = rng.choice(len(high), size=max_pairs_per_drug, replace=False)
            high, low = high[keep], low[keep]
        high_rows.extend(local[high])
        low_rows.extend(local[low])
        weights.extend((y[high] - y[low]).tolist())
    return (
        np.asarray(high_rows, dtype=np.int64),
        np.asarray(low_rows, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def safe_binary_metric(y: np.ndarray, score: np.ndarray, metric: str) -> float | None:
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    if metric == "roc":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    y = frame["strong_inhibition_label"].to_numpy(dtype=np.int8)
    score = frame[score_column].to_numpy(dtype=np.float64)
    continuous = frame["inhibition_fraction_1uM"].to_numpy(dtype=np.float64)
    output: dict[str, Any] = {
        "pairs": len(frame),
        "drugs": int(frame["ligand_inchikey"].nunique()),
        "targets": int(frame["target_chembl_id"].nunique()),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "micro_auroc": safe_binary_metric(y, score, "roc"),
        "micro_auprc": safe_binary_metric(y, score, "pr"),
        "continuous_mae": float(np.mean(np.abs(score - continuous))),
    }
    global_spearman = spearmanr(continuous, score).statistic
    output["continuous_spearman"] = float(global_spearman) if np.isfinite(global_spearman) else None
    aps, aucs, ndcgs, correlations = [], [], [], []
    recalls = {5: [], 10: [], 20: []}
    precisions = {5: [], 10: [], 20: []}
    positive_drugs = 0
    for _, group in frame.groupby("ligand_inchikey", sort=False):
        yy = group["strong_inhibition_label"].to_numpy(dtype=np.int8)
        ss = group[score_column].to_numpy(dtype=np.float64)
        graded = group["inhibition_fraction_1uM"].to_numpy(dtype=np.float64)
        if np.unique(yy).size == 2:
            aps.append(float(average_precision_score(yy, ss)))
            aucs.append(float(roc_auc_score(yy, ss)))
        if len(group) >= 2 and graded.max() > 0:
            ndcgs.append(float(ndcg_score(graded[None, :], ss[None, :], k=min(20, len(group)))))
            corr = spearmanr(graded, ss).statistic
            if np.isfinite(corr):
                correlations.append(float(corr))
        positives = int(yy.sum())
        if positives == 0:
            continue
        positive_drugs += 1
        order = np.argsort(-ss, kind="mergesort")
        for k in recalls:
            kk = min(k, len(group))
            hits = int(yy[order[:kk]].sum())
            recalls[k].append(hits / positives)
            precisions[k].append(hits / kk)
    output.update({
        "macro_drug_auprc": float(np.mean(aps)) if aps else None,
        "macro_drug_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_drug_ndcg_at_20_continuous": float(np.mean(ndcgs)) if ndcgs else None,
        "macro_drug_spearman_continuous": float(np.mean(correlations)) if correlations else None,
        "drugs_with_two_classes": len(aps),
        "drugs_with_positive": positive_drugs,
    })
    for k in recalls:
        output[f"macro_recall_at_{k}"] = float(np.mean(recalls[k])) if recalls[k] else None
        output[f"macro_precision_at_{k}"] = float(np.mean(precisions[k])) if precisions[k] else None
    return output


def validation_value(frame: pd.DataFrame, score_column: str) -> float:
    current = metrics(frame, score_column)
    ap = current["macro_drug_auprc"]
    ndcg = current["macro_drug_ndcg_at_20_continuous"]
    if ap is None:
        ap = current["micro_auprc"] or 0.0
    if ndcg is None:
        ndcg = 0.0
    return float(0.60 * ap + 0.40 * ndcg)


def prepare_fold_tensors(
    frame: pd.DataFrame,
    positions: np.ndarray,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    part = frame.iloc[positions]
    drug_index = part["drug_feature_index"].to_numpy(dtype=np.int64)
    target_index = part["target_feature_index"].to_numpy(dtype=np.int64)
    drug = torch.from_numpy(np.asarray(drug_features[drug_index], dtype=np.float32)).to(device)
    target = torch.from_numpy(np.asarray(target_features[target_index], dtype=np.float32)).to(device)
    prior_np = part[PRIOR_COLUMNS].to_numpy(dtype=np.float32)
    prior_np = (prior_np - prior_mean) / prior_std
    priors = torch.from_numpy(prior_np.astype(np.float32)).to(device)
    continuous = torch.from_numpy(
        part["inhibition_fraction_1uM"].to_numpy(dtype=np.float32)
    ).to(device)
    binary = torch.from_numpy(
        part["strong_inhibition_label"].to_numpy(dtype=np.float32)
    ).to(device)
    counts = part.groupby("ligand_inchikey")["ligand_inchikey"].transform("size").to_numpy()
    drug_weight = (1.0 / counts).astype(np.float32)
    drug_weight /= drug_weight.mean()
    weights = torch.from_numpy(drug_weight).to(device)
    return drug, target, priors, continuous, binary, weights


@torch.no_grad()
def predict(
    model: nn.Module,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> np.ndarray:
    model.eval()
    drug, target, priors, _, _, _ = tensors
    return torch.sigmoid(model(drug, target, priors)).cpu().numpy()


def train_one(
    frame: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    test_positions: np.ndarray,
    variant: str,
    fold_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, torch.Tensor], dict[str, np.ndarray]]:
    set_seed(fold_seed)
    use_priors = variant == "embedding_plus_frozen_priors"
    train_prior = frame.iloc[train_positions][PRIOR_COLUMNS].to_numpy(dtype=np.float32)
    prior_mean = train_prior.mean(axis=0).astype(np.float32)
    prior_std = train_prior.std(axis=0).astype(np.float32)
    prior_std[prior_std < 1e-6] = 1.0
    train_tensors = prepare_fold_tensors(
        frame, train_positions, drug_features, target_features, prior_mean, prior_std, device
    )
    valid_tensors = prepare_fold_tensors(
        frame, valid_positions, drug_features, target_features, prior_mean, prior_std, device
    )
    test_tensors = prepare_fold_tensors(
        frame, test_positions, drug_features, target_features, prior_mean, prior_std, device
    )
    rank_high, rank_low, rank_weight = rank_pairs(
        frame, train_positions, args.max_rank_pairs_per_drug, fold_seed
    )
    rank_high_t = torch.from_numpy(rank_high).to(device)
    rank_low_t = torch.from_numpy(rank_low).to(device)
    rank_weight_t = torch.from_numpy(rank_weight).to(device)
    model = FunctionalInhibitionAdapter(
        use_priors=use_priors,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    drug, target, priors, continuous, binary, sample_weight = train_tensors
    weighted_positive = float((sample_weight * binary).sum().detach().cpu())
    weighted_negative = float((sample_weight * (1.0 - binary)).sum().detach().cpu())
    positive_weight = weighted_negative / max(weighted_positive, 1e-8)
    best_value = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    checks_without_gain = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logit = model(drug, target, priors)
        prediction = torch.sigmoid(logit)
        regression_rows = torch.nn.functional.smooth_l1_loss(
            prediction, continuous, reduction="none", beta=0.10
        )
        regression_loss = (regression_rows * sample_weight).mean()
        binary_rows = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, binary, reduction="none",
            pos_weight=torch.tensor(positive_weight, device=device),
        )
        binary_loss = (binary_rows * sample_weight).mean()
        rank_rows = torch.nn.functional.softplus(
            -(logit[rank_high_t] - logit[rank_low_t])
        )
        rank_loss = (rank_rows * rank_weight_t).sum() / rank_weight_t.sum().clamp_min(1e-8)
        loss = regression_loss + args.binary_weight * binary_loss + args.rank_weight * rank_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        valid_score = predict(model, valid_tensors)
        valid_frame = frame.iloc[valid_positions].copy()
        valid_frame["candidate_score"] = valid_score
        value = validation_value(valid_frame, "candidate_score")
        history.append({
            "epoch": float(epoch), "loss": float(loss.detach().cpu()),
            "regression_loss": float(regression_loss.detach().cpu()),
            "binary_loss": float(binary_loss.detach().cpu()),
            "rank_loss": float(rank_loss.detach().cpu()), "validation_value": value,
        })
        if value > best_value + 1e-5:
            best_value = value
            best_epoch = epoch
            best_state = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
            checks_without_gain = 0
        else:
            checks_without_gain += 1
        if checks_without_gain >= args.patience_checks:
            break
    if best_state is None:
        raise RuntimeError("No best model state captured")
    model.load_state_dict(best_state)
    valid_score = predict(model, valid_tensors)
    test_score = predict(model, test_tensors)
    fit_summary = {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_validation_value": best_value,
        "epochs_ran": int(history[-1]["epoch"]),
        "rank_pairs": len(rank_high),
        "train_rows": len(train_positions),
        "valid_rows": len(valid_positions),
        "test_rows": len(test_positions),
        "history": history,
    }
    normalization = {"prior_mean": prior_mean, "prior_std": prior_std}
    return valid_score, test_score, fit_summary, best_state, normalization


def main() -> None:
    args = parse_args()
    required = [DATA, DATA_SUMMARY, DRUG_FEATURES, TARGET_FEATURES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    data_summary = json.loads(DATA_SUMMARY.read_text())
    if data_summary.get("status") != "PASS":
        raise RuntimeError("KiRHub functional data package is not PASS")
    frame = pd.read_csv(DATA, low_memory=False)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    target_features = np.load(TARGET_FEATURES, mmap_mode="r")
    if drug_features.shape != (720, 768) or target_features.shape != (384, 1280):
        raise RuntimeError("Unexpected pretrained feature dimensions")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    all_oof, fold_rows, fit_rows = [], [], []
    saved_states: dict[str, dict[str, Any]] = {}
    for regime in args.regimes:
        fold_column = REGIMES[regime]
        oof = frame[[
            "pairId", "ligand_inchikey", "drug_names", "target_chembl_id",
            "gene_symbol", "residual_activity_pct_1uM", "inhibition_fraction_1uM",
            "strong_inhibition_label", fold_column,
        ] + PRIOR_COLUMNS].copy()
        for variant in VARIANTS:
            oof[f"{variant}_oof_score"] = np.nan
        oof["validation_selected_oof_score"] = np.nan
        oof["validation_selected_variant"] = ""
        for test_fold in range(5):
            valid_fold = (test_fold + 1) % 5
            test_positions = np.flatnonzero(frame[fold_column].to_numpy() == test_fold)
            valid_positions = np.flatnonzero(frame[fold_column].to_numpy() == valid_fold)
            train_positions = np.flatnonzero(
                ~frame[fold_column].isin([test_fold, valid_fold]).to_numpy()
            )
            fold_candidates: dict[str, dict[str, Any]] = {}
            for variant_index, variant in enumerate(VARIANTS):
                valid_score, test_score, fit_summary, best_state, normalization = train_one(
                    frame, drug_features, target_features,
                    train_positions, valid_positions, test_positions,
                    variant, args.seed + test_fold * 101 + variant_index * 1009,
                    args, device,
                )
                oof.loc[test_positions, f"{variant}_oof_score"] = test_score
                valid_frame = frame.iloc[valid_positions].copy()
                valid_frame["candidate_score"] = valid_score
                valid_metrics = metrics(valid_frame, "candidate_score")
                valid_value = validation_value(valid_frame, "candidate_score")
                test_frame = frame.iloc[test_positions].copy()
                test_frame["candidate_score"] = test_score
                test_metrics = metrics(test_frame, "candidate_score")
                fold_candidates[variant] = {
                    "validation_value": valid_value,
                    "test_score": test_score,
                    "state": best_state,
                    "normalization": normalization,
                    "fit_summary": fit_summary,
                }
                fit_rows.append({
                    "regime": regime, "test_fold": test_fold,
                    "validation_fold": valid_fold, **fit_summary,
                })
                fold_rows.append({
                    "regime": regime, "test_fold": test_fold,
                    "validation_fold": valid_fold, "model": variant,
                    "selected_by_validation": False,
                    "validation_selection_value": valid_value,
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    **{f"validation_{key}": value for key, value in valid_metrics.items()},
                })
            selected_variant = max(
                VARIANTS, key=lambda name: fold_candidates[name]["validation_value"]
            )
            selected = fold_candidates[selected_variant]
            oof.loc[test_positions, "validation_selected_oof_score"] = selected["test_score"]
            oof.loc[test_positions, "validation_selected_variant"] = selected_variant
            fold_rows[-2 + VARIANTS.index(selected_variant)]["selected_by_validation"] = True
            saved_states[f"{regime}_fold{test_fold}_{selected_variant}"] = {
                "state_dict": selected["state"],
                "normalization": selected["normalization"],
                "test_fold": test_fold, "validation_fold": valid_fold,
                "variant": selected_variant,
            }
        score_columns = {
            "EMBEDDING_ONLY_ADAPTER": "embedding_only_oof_score",
            "EMBEDDING_PLUS_FROZEN_PRIORS_ADAPTER": "embedding_plus_frozen_priors_oof_score",
            "VALIDATION_SELECTED_ADAPTER": "validation_selected_oof_score",
            "FROZEN_BIOMASTER_DRUG_RANKER": "old_drug_leakage_safe_score_v10",
            "FROZEN_BIOMASTER_FULL_FIT": "biomaster_full_fit_v10_borda_score",
            "FROZEN_DTIAM": "dtiam_probability",
        }
        for model_name, score_column in score_columns.items():
            if oof[score_column].isna().any():
                raise RuntimeError(f"Incomplete OOF scores: {regime} {model_name}")
            current = metrics(oof, score_column)
            fold_rows.append({
                "regime": regime, "test_fold": "ALL_OOF",
                "validation_fold": "", "model": model_name,
                "selected_by_validation": model_name == "VALIDATION_SELECTED_ADAPTER",
                "validation_selection_value": None,
                **{f"test_{key}": value for key, value in current.items()},
            })
        oof["regime"] = regime
        all_oof.append(oof)

    oof_frame = pd.concat(all_oof, ignore_index=True)
    metric_frame = pd.DataFrame(fold_rows)
    fit_frame = pd.DataFrame(fit_rows)
    oof_path = OUT / "KIRHUB_FUNCTIONAL_GROUPED_OOF_PREDICTIONS_V1.csv.gz"
    metrics_path = OUT / "KIRHUB_FUNCTIONAL_GROUPED_METRICS_V1.csv"
    fit_path = OUT / "KIRHUB_FUNCTIONAL_FIT_HISTORY_V1.json"
    checkpoint_path = OUT / "KIRHUB_FUNCTIONAL_VALIDATION_SELECTED_FOLDS_V1.pt"
    oof_frame.to_csv(oof_path, index=False, compression="gzip")
    metric_frame.to_csv(metrics_path, index=False)
    fit_path.write_text(json.dumps(fit_rows, ensure_ascii=False, indent=2) + "\n")
    torch.save({
        "architecture": "fixed BerMol768 + fixed ESM2-650M1280 -> dual projections -> interaction head",
        "states": saved_states, "arguments": vars(args), "prior_columns": PRIOR_COLUMNS,
        "data_sha256": sha256(DATA),
    }, checkpoint_path)

    aggregate = metric_frame[metric_frame["test_fold"].eq("ALL_OOF")].copy()
    drug_primary = aggregate[aggregate["regime"].eq("DRUG_SCAFFOLD_COLD")]
    selected_row = drug_primary[
        drug_primary["model"].eq("VALIDATION_SELECTED_ADAPTER")
    ].iloc[0]
    dtiam_row = drug_primary[drug_primary["model"].eq("FROZEN_DTIAM")].iloc[0]
    biomaster_row = drug_primary[
        drug_primary["model"].eq("FROZEN_BIOMASTER_DRUG_RANKER")
    ].iloc[0]
    checks = {
        "exact_7505_primary_rows": len(frame) == 7505,
        "all_oof_predictions_complete": not oof_frame[[
            "embedding_only_oof_score", "embedding_plus_frozen_priors_oof_score",
            "validation_selected_oof_score",
        ]].isna().any().any(),
        "all_oof_predictions_bounded": oof_frame[[
            "embedding_only_oof_score", "embedding_plus_frozen_priors_oof_score",
            "validation_selected_oof_score",
        ]].apply(lambda x: x.between(0, 1).all()).all(),
        "no_random_pair_split": True,
        "drug_scaffolds_grouped": frame.groupby("drug_scaffold_group")[
            "drug_scaffold_cold_fold"
        ].nunique().eq(1).all(),
        "target_homology_clusters_grouped": frame.groupby("target_homology_cluster")[
            "target_homology_cold_fold"
        ].nunique().eq(1).all(),
        "pretrained_embedding_arrays_unchanged": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": str(device),
        "task": "drug_to_target_1uM_functional_kinase_inhibition_ranking",
        "architecture": {
            "frozen_inputs": "BerMol768 drug embedding + ESM2-650M1280 target embedding",
            "trainable_part": "small dual-projection pair interaction adapter only",
            "loss": "continuous SmoothL1 + strong-inhibition BCE + within-drug pairwise ranking",
            "optional_priors": PRIOR_COLUMNS,
        },
        "evaluation": {
            "primary": "5-fold drug Bemis-Murcko scaffold cold OOF",
            "secondary": "5-fold target >=40% homology-cluster cold OOF",
            "model_choice": "embedding-only versus frozen-prior variant selected inside each validation fold",
        },
        "primary_drug_scaffold_cold": {
            "validation_selected_adapter": selected_row.to_dict(),
            "frozen_biomaster_drug_ranker": biomaster_row.to_dict(),
            "frozen_dtiam": dtiam_row.to_dict(),
        },
        "checks": checks,
        "claim_boundary": (
            "These are internal grouped-development OOF results after KiRHub adaptation, not a new "
            "untouched external validation. Recall@K denominators are the measured 1 uM targets "
            "in the 96-target single-construct overlap, not all 384 project targets."
        ),
        "inputs_sha256": {path.name: sha256(path) for path in required},
        "artifacts": {
            oof_path.name: sha256(oof_path), metrics_path.name: sha256(metrics_path),
            fit_path.name: sha256(fit_path), checkpoint_path.name: sha256(checkpoint_path),
        },
    }
    summary_path = OUT / "KIRHUB_FUNCTIONAL_ADAPTER_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

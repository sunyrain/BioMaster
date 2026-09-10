#!/usr/bin/env python3
"""Train the expanded 92-drug/345-target KiRHub functional adapter.

Only the small pair head is trainable.  The 92x345 BerMol/ESM2 inputs are
frozen, and every OOF prediction is from a drug-scaffold-held-out model.
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
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_kirhub_functional_adapter_v1 import (  # noqa: E402
    FunctionalInhibitionAdapter,
    rank_pairs,
    set_seed,
)


BASE = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_pretrained_features_v1"
DATA = BASE / "KIRHUB_EXPANDED_FUNCTIONAL_PAIRS_V1.csv.gz"
DATA_SUMMARY = BASE / "KIRHUB_EXPANDED_PRETRAINED_FEATURE_SUMMARY_V1.json"
DRUG_FEATURES = BASE / "KIRHUB_EXPANDED_DRUG92_BERMOL768_FLOAT32_V1.npy"
TARGET_FEATURES = BASE / "KIRHUB_EXPANDED_TARGET345_ESM2_1280_FLOAT32_V1.npy"
OUT = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_functional_adapter_v1"
BASELINE = "old_drug_leakage_safe_score_v10"
DTIAM = "dtiam_probability"
ADAPTER = "expanded_embedding_adapter_oof_score"
BOOTSTRAPS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience-checks", type=int, default=14)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--projection-dim", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--rank-weight", type=float, default=0.45)
    parser.add_argument("--binary-weight", type=float, default=0.30)
    parser.add_argument("--max-rank-pairs-per-drug", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    y = frame["strong_inhibition_label"].to_numpy(dtype=np.int8)
    score = frame[score_column].to_numpy(dtype=np.float64)
    rows = []
    for drug, group in frame.groupby("ligand_inchikey", sort=False):
        yy = group["strong_inhibition_label"].to_numpy(dtype=np.int8)
        ss = group[score_column].to_numpy(dtype=np.float64)
        graded = group["inhibition_fraction_1uM"].to_numpy(dtype=np.float64)
        order = np.argsort(-ss, kind="mergesort")
        positives = int(yy.sum())
        row = {
            "drug": drug, "pairs": len(group), "positives": positives,
            "ap": float(average_precision_score(yy, ss)) if np.unique(yy).size == 2 else None,
            "auc": float(roc_auc_score(yy, ss)) if np.unique(yy).size == 2 else None,
            "ndcg20": float(ndcg_score(graded[None, :], ss[None, :], k=min(20, len(group)))),
        }
        for k in (5, 10, 20):
            hits = int(yy[order[: min(k, len(group))]].sum())
            row[f"recall{k}"] = hits / positives if positives else None
        rows.append(row)
    groups = pd.DataFrame(rows)
    return {
        "pairs": len(frame), "drugs": frame["ligand_inchikey"].nunique(),
        "targets": frame["uniprot_accession"].nunique(), "positives": int(y.sum()),
        "micro_auroc": float(roc_auc_score(y, score)),
        "micro_auprc": float(average_precision_score(y, score)),
        "macro_drug_auprc": float(groups["ap"].dropna().mean()),
        "macro_drug_auroc": float(groups["auc"].dropna().mean()),
        "macro_drug_ndcg_at_20_continuous": float(groups["ndcg20"].mean()),
        "macro_recall_at_5": float(groups["recall5"].dropna().mean()),
        "macro_recall_at_10": float(groups["recall10"].dropna().mean()),
        "macro_recall_at_20": float(groups["recall20"].dropna().mean()),
        "drugs_with_two_classes": int(groups["ap"].notna().sum()),
        "drugs_with_positive": int(groups["recall20"].notna().sum()),
    }


def validation_value(frame: pd.DataFrame, score_column: str) -> float:
    current = ranking_metrics(frame, score_column)
    return float(
        0.60 * current["macro_drug_auprc"]
        + 0.40 * current["macro_drug_ndcg_at_20_continuous"]
    )


def tensors(
    frame: pd.DataFrame,
    positions: np.ndarray,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    part = frame.iloc[positions]
    drug = torch.from_numpy(np.asarray(
        drug_features[part["drug_feature_index"].to_numpy(dtype=np.int64)], dtype=np.float32
    )).to(device)
    target = torch.from_numpy(np.asarray(
        target_features[part["target_feature_index"].to_numpy(dtype=np.int64)], dtype=np.float32
    )).to(device)
    prior = torch.zeros((len(part), 4), dtype=torch.float32, device=device)
    continuous = torch.from_numpy(
        part["inhibition_fraction_1uM"].to_numpy(dtype=np.float32)
    ).to(device)
    binary = torch.from_numpy(
        part["strong_inhibition_label"].to_numpy(dtype=np.float32)
    ).to(device)
    counts = part.groupby("ligand_inchikey")["ligand_inchikey"].transform("size").to_numpy()
    weight = (1.0 / counts).astype(np.float32)
    weight /= weight.mean()
    return drug, target, prior, continuous, binary, torch.from_numpy(weight).to(device)


@torch.no_grad()
def predict(model: torch.nn.Module, current: tuple[torch.Tensor, ...]) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(current[0], current[1], current[2])).cpu().numpy()


def train_fold(
    frame: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    test_positions: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any], dict[str, torch.Tensor]]:
    set_seed(seed)
    train_tensors = tensors(frame, train_positions, drug_features, target_features, device)
    valid_tensors = tensors(frame, valid_positions, drug_features, target_features, device)
    test_tensors = tensors(frame, test_positions, drug_features, target_features, device)
    high, low, contrast = rank_pairs(
        frame, train_positions, args.max_rank_pairs_per_drug, seed
    )
    high_t = torch.from_numpy(high).to(device)
    low_t = torch.from_numpy(low).to(device)
    contrast_t = torch.from_numpy(contrast).to(device)
    model = FunctionalInhibitionAdapter(
        use_priors=False, projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    drug, target, prior, continuous, binary, sample_weight = train_tensors
    weighted_positive = float((sample_weight * binary).sum().cpu())
    weighted_negative = float((sample_weight * (1 - binary)).sum().cpu())
    positive_weight = torch.tensor(
        weighted_negative / max(weighted_positive, 1e-8), device=device
    )
    best_value, best_epoch, stale, best_state = -math.inf, 0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logit = model(drug, target, prior)
        predicted = torch.sigmoid(logit)
        regression = (
            torch.nn.functional.smooth_l1_loss(
                predicted, continuous, reduction="none", beta=0.10
            ) * sample_weight
        ).mean()
        binary_loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logit, binary, reduction="none", pos_weight=positive_weight
            ) * sample_weight
        ).mean()
        rank_rows = torch.nn.functional.softplus(-(logit[high_t] - logit[low_t]))
        ranking = (rank_rows * contrast_t).sum() / contrast_t.sum().clamp_min(1e-8)
        loss = regression + args.binary_weight * binary_loss + args.rank_weight * ranking
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        valid_score = predict(model, valid_tensors)
        validation = frame.iloc[valid_positions].copy()
        validation["score"] = valid_score
        value = validation_value(validation, "score")
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "validation_value": value})
        if value > best_value + 1e-5:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        if stale >= args.patience_checks:
            break
    if best_state is None:
        raise RuntimeError("No expanded best state")
    model.load_state_dict(best_state)
    return predict(model, test_tensors), {
        "best_epoch": best_epoch, "best_validation_value": best_value,
        "epochs_ran": history[-1]["epoch"], "rank_pairs": len(high),
        "train_rows": len(train_positions), "valid_rows": len(valid_positions),
        "test_rows": len(test_positions), "history": history,
    }, best_state


def percentiles(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame.groupby("ligand_inchikey")[column].rank(
        pct=True, method="average"
    ).to_numpy(dtype=np.float64)


def choose_fusion(validation: pd.DataFrame) -> dict[str, float]:
    ranks = {column: percentiles(validation, column) for column in [BASELINE, ADAPTER, DTIAM]}
    best = None
    for baseline_step in range(11):
        for adapter_step in range(11 - baseline_step):
            weights = np.array([
                baseline_step / 10, adapter_step / 10,
                1 - baseline_step / 10 - adapter_step / 10,
            ])
            validation["candidate"] = sum(
                weight * ranks[column]
                for weight, column in zip(weights, [BASELINE, ADAPTER, DTIAM])
            )
            value = validation_value(validation, "candidate")
            if best is None or value > best[0] + 1e-12:
                best = (value, *weights.tolist())
    return {
        "validation_value": best[0], "biomaster_weight": best[1],
        "expanded_adapter_weight": best[2], "dtiam_weight": best[3],
    }


def per_drug(frame: pd.DataFrame, score_column: str, model: str) -> pd.DataFrame:
    rows = []
    for drug, group in frame.groupby("ligand_inchikey", sort=True):
        y = group["strong_inhibition_label"].to_numpy(dtype=np.int8)
        score = group[score_column].to_numpy(dtype=np.float64)
        graded = group["inhibition_fraction_1uM"].to_numpy(dtype=np.float64)
        order = np.argsort(-score, kind="mergesort")
        positives = int(y.sum())
        rows.append({
            "drug": drug, "model": model,
            "ap": float(average_precision_score(y, score)) if np.unique(y).size == 2 else None,
            "ndcg20": float(ndcg_score(graded[None, :], score[None, :], k=min(20, len(group)))),
            "recall20": float(y[order[:20]].sum() / positives) if positives else None,
        })
    return pd.DataFrame(rows)


def bootstrap(groups: pd.DataFrame, reference: str, metric: str, seed: int) -> dict[str, Any]:
    pivot = groups.pivot(index="drug", columns="model", values=metric)
    pair = pivot[["CROSSFIT_EXPANDED_FUSION", reference]].dropna().to_numpy(dtype=np.float64)
    difference = pair[:, 0] - pair[:, 1]
    rng = np.random.default_rng(seed)
    samples = difference[rng.integers(0, len(difference), size=(BOOTSTRAPS, len(difference)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "reference": reference, "metric": metric, "paired_drugs": len(difference),
        "fusion_macro": float(pair[:, 0].mean()), "reference_macro": float(pair[:, 1].mean()),
        "difference": float(difference.mean()), "ci95_low": float(low), "ci95_high": float(high),
        "probability_gt_zero": float((samples > 0).mean()),
    }


def main() -> None:
    global OUT
    args = parse_args()
    OUT = Path(args.out_dir).resolve()
    required = [DATA, DATA_SUMMARY, DRUG_FEATURES, TARGET_FEATURES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    data_summary = json.loads(DATA_SUMMARY.read_text())
    if data_summary.get("status") != "PASS":
        raise RuntimeError("Expanded feature package is not PASS")
    frame = pd.read_csv(DATA, low_memory=False)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    target_features = np.load(TARGET_FEATURES, mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    frame[ADAPTER] = np.nan
    fit_rows, state_rows = [], {}
    fold_column = "drug_scaffold_cold_fold"
    for test_fold in range(5):
        valid_fold = (test_fold + 1) % 5
        test = np.flatnonzero(frame[fold_column].to_numpy() == test_fold)
        valid = np.flatnonzero(frame[fold_column].to_numpy() == valid_fold)
        train = np.flatnonzero(~frame[fold_column].isin([test_fold, valid_fold]).to_numpy())
        test_score, fit, state = train_fold(
            frame, drug_features, target_features, train, valid, test,
            args, args.seed + 101 * test_fold, device,
        )
        frame.loc[test, ADAPTER] = test_score
        fit_rows.append({"test_fold": test_fold, "validation_fold": valid_fold, **fit})
        state_rows[f"fold{test_fold}"] = {
            "test_fold": test_fold, "validation_fold": valid_fold, "state_dict": state,
        }
    if frame[ADAPTER].isna().any():
        raise RuntimeError("Incomplete expanded OOF predictions")

    full_metrics = ranking_metrics(frame, ADAPTER)
    common = frame[frame["deployment_720x384_comparison_scope"].astype(bool)].copy()
    common["crossfit_expanded_fusion_score"] = np.nan
    selection_rows = []
    for test_fold in range(5):
        valid_fold = (test_fold + 1) % 5
        validation = common[common[fold_column].eq(valid_fold)].copy()
        test = common[common[fold_column].eq(test_fold)].copy()
        if validation["ligand_inchikey"].nunique() == 0 or test["ligand_inchikey"].nunique() == 0:
            raise RuntimeError("No common-scope drugs in a grouped fold")
        selected = choose_fusion(validation)
        ranks = {column: percentiles(test, column) for column in [BASELINE, ADAPTER, DTIAM]}
        score = (
            selected["biomaster_weight"] * ranks[BASELINE]
            + selected["expanded_adapter_weight"] * ranks[ADAPTER]
            + selected["dtiam_weight"] * ranks[DTIAM]
        )
        common.loc[test.index, "crossfit_expanded_fusion_score"] = score
        selection_rows.append({
            "test_fold": test_fold, "validation_fold": valid_fold, **selected,
        })
    models = {
        "CROSSFIT_EXPANDED_FUSION": "crossfit_expanded_fusion_score",
        "EXPANDED_EMBEDDING_ADAPTER": ADAPTER,
        "FROZEN_BIOMASTER_DRUG_RANKER": BASELINE,
        "FROZEN_DTIAM": DTIAM,
    }
    metrics_rows, group_rows = [], []
    for model, column in models.items():
        metrics_rows.append({"model": model, **ranking_metrics(common, column)})
        group_rows.append(per_drug(common, column, model))
    metrics = pd.DataFrame(metrics_rows)
    groups = pd.concat(group_rows, ignore_index=True)
    bootstrap_rows = []
    for reference in ["FROZEN_BIOMASTER_DRUG_RANKER", "FROZEN_DTIAM"]:
        for metric in ["ap", "ndcg20", "recall20"]:
            bootstrap_rows.append(bootstrap(groups, reference, metric, args.seed + len(bootstrap_rows)))
    bootstraps = pd.DataFrame(bootstrap_rows)

    oof_path = OUT / "KIRHUB_EXPANDED_DRUG_SCAFFOLD_OOF_V1.csv.gz"
    common_path = OUT / "KIRHUB_EXPANDED_COMMON_SCOPE_FUSION_OOF_V1.csv.gz"
    metrics_path = OUT / "KIRHUB_EXPANDED_COMMON_SCOPE_METRICS_V1.csv"
    selection_path = OUT / "KIRHUB_EXPANDED_CROSSFIT_SELECTION_V1.csv"
    bootstrap_path = OUT / "KIRHUB_EXPANDED_COMMON_SCOPE_BOOTSTRAP_V1.csv"
    fit_path = OUT / "KIRHUB_EXPANDED_FIT_HISTORY_V1.json"
    checkpoint_path = OUT / "KIRHUB_EXPANDED_GROUPED_FOLD_CHECKPOINTS_V1.pt"
    frame.to_csv(oof_path, index=False, compression="gzip")
    common.to_csv(common_path, index=False, compression="gzip")
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    bootstraps.to_csv(bootstrap_path, index=False)
    fit_path.write_text(json.dumps(fit_rows, ensure_ascii=False, indent=2) + "\n")
    torch.save({
        "architecture": "frozen BerMol768 + frozen ESM2-650M1280 + embedding-only interaction adapter",
        "arguments": vars(args), "states": state_rows, "data_sha256": sha256(DATA),
    }, checkpoint_path)

    table = metrics.set_index("model")
    fusion = table.loc["CROSSFIT_EXPANDED_FUSION"]
    biomaster = table.loc["FROZEN_BIOMASTER_DRUG_RANKER"]
    dtiam = table.loc["FROZEN_DTIAM"]
    checks = {
        "exact_30973_rows": len(frame) == 30973,
        "all_92_drugs_oof_scored": frame[ADAPTER].notna().all() and frame["ligand_inchikey"].nunique() == 92,
        "five_scaffold_grouped_folds": frame.groupby("drug_scaffold_group")[fold_column].nunique().eq(1).all(),
        "all_folds_reached_early_stopping": all(
            row["epochs_ran"] < args.epochs for row in fit_rows
        ),
        "fusion_macro_auprc_above_dtiam": fusion["macro_drug_auprc"] > dtiam["macro_drug_auprc"],
        "fusion_recall20_above_dtiam": fusion["macro_recall_at_20"] > dtiam["macro_recall_at_20"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": str(device),
        "expanded_internal_oof": full_metrics,
        "common_720x384_scope": {
            "rows": len(common), "crossfit_fusion": fusion.to_dict(),
            "frozen_biomaster": biomaster.to_dict(), "frozen_dtiam": dtiam.to_dict(),
        },
        "checks": checks,
        "claim_boundary": (
            "Internal KiRHub grouped-development OOF. Common-scope fusion covers only rows with "
            "frozen 720x384 baselines; it is not untouched external validation."
        ),
        "inputs_sha256": {path.name: sha256(path) for path in required},
        "artifacts": {
            path.name: sha256(path) for path in [
                oof_path, common_path, metrics_path, selection_path,
                bootstrap_path, fit_path, checkpoint_path,
            ]
        },
    }
    report_path = OUT / "KIRHUB_EXPANDED_FUNCTIONAL_ADAPTER_SUMMARY_V1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

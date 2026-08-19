#!/usr/bin/env python3
"""Evaluate leakage-resistant baselines on every frozen BioMaster-ODTI split."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
STORE = BASE / "feature_store_v1"
PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = STORE / "MORGAN2048_UINT8_V1.npy"
FEATURE_AUDIT = STORE / "FEATURE_STORE_AUDIT_V1.json"
OUT = BASE / "baseline_results_v1"
FOLDS = 5
PROTOCOLS = [
    "S0_RANDOM_DIAGNOSTIC",
    "S1_SCAFFOLD_COLD_DRUG",
    "S2_HOMOLOGY_COLD_TARGET",
    "S3_STRICT_DOUBLE_COLD",
    "S4_FIRST_SEEN_TEMPORAL_2023_2025",
    "S5_OLD_DRUG_ENTITY_COLD",
    "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_masks(data: pd.DataFrame, protocol: str, fold: int) -> dict[str, np.ndarray]:
    if protocol == "S0_RANDOM_DIAGNOSTIC":
        column = "random_pair_fold"
        return {
            "train": ~data[column].isin([fold, (fold + 1) % FOLDS]).to_numpy(),
            "valid": data[column].eq((fold + 1) % FOLDS).to_numpy(),
            "test": data[column].eq(fold).to_numpy(),
        }
    if protocol == "S1_SCAFFOLD_COLD_DRUG":
        column = "scaffold_cold_fold"
        return {
            "train": ~data[column].isin([fold, (fold + 1) % FOLDS]).to_numpy(),
            "valid": data[column].eq((fold + 1) % FOLDS).to_numpy(),
            "test": data[column].eq(fold).to_numpy(),
        }
    if protocol == "S2_HOMOLOGY_COLD_TARGET":
        column = "target_homology_cold_fold"
        return {
            "train": ~data[column].isin([fold, (fold + 1) % FOLDS]).to_numpy(),
            "valid": data[column].eq((fold + 1) % FOLDS).to_numpy(),
            "test": data[column].eq(fold).to_numpy(),
        }
    if protocol == "S3_STRICT_DOUBLE_COLD":
        drug_fold = data["scaffold_cold_fold"]
        target_fold = data["target_homology_cold_fold"]
        valid_fold = (fold + 1) % FOLDS
        return {
            "train": (~drug_fold.isin([fold, valid_fold]) & ~target_fold.isin([fold, valid_fold])).to_numpy(),
            "valid": (drug_fold.eq(valid_fold) & target_fold.eq(valid_fold)).to_numpy(),
            "test": (drug_fold.eq(fold) & target_fold.eq(fold)).to_numpy(),
        }
    if protocol == "S4_FIRST_SEEN_TEMPORAL_2023_2025":
        eligible = data["temporal_role"].eq("TRAIN_POOL_THROUGH_2022")
        # Validation remains pre-2023 and is scaffold-disjoint from fitting rows.
        valid = eligible & data["scaffold_cold_fold"].eq(0)
        return {
            "train": (eligible & ~valid).to_numpy(),
            "valid": valid.to_numpy(),
            "test": data["temporal_role"].eq("TEST_FIRST_SEEN_2023_2025").to_numpy(),
        }
    if protocol == "S5_OLD_DRUG_ENTITY_COLD":
        eligible = ~data["has_deployment_old_drug_scaffold"].astype(bool)
        valid = eligible & data["scaffold_cold_fold"].eq(0)
        return {
            "train": (eligible & ~valid).to_numpy(),
            "valid": valid.to_numpy(),
            "test": data["is_deployment_old_drug"].astype(bool).to_numpy(),
        }
    if protocol == "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD":
        # Deployment-aligned target-to-drug retrieval.  The queried old-drug
        # entities and every one of their observed Murcko scaffolds are absent
        # from fitting, while validation/test target homology components are
        # also disjoint from fitting and from one another.  Unlike S3, the
        # complete old-drug slice of each target-cold fold is evaluated rather
        # than only the accidental scaffold-fold intersection.
        target_fold = data["target_homology_cold_fold"]
        valid_fold = (fold + 1) % FOLDS
        old_drug = data["is_deployment_old_drug"].astype(bool)
        eligible_train = ~data["has_deployment_old_drug_scaffold"].astype(bool)
        return {
            "train": (
                eligible_train & ~target_fold.isin([fold, valid_fold])
            ).to_numpy(),
            "valid": (old_drug & target_fold.eq(valid_fold)).to_numpy(),
            "test": (old_drug & target_fold.eq(fold)).to_numpy(),
        }
    raise ValueError(protocol)


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    result = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if selected.any():
            result += selected.mean() * abs(float(y[selected].mean()) - float(probability[selected].mean()))
    return float(result) if total else float("nan")


def grouped_binary_metrics(
    frame: pd.DataFrame, score: np.ndarray, group: str, prefix: str
) -> dict[str, float | int | None]:
    aucs: list[float] = []
    aps: list[float] = []
    recalls = {5: [], 10: [], 20: []}
    ndcgs = {5: [], 10: [], 20: []}
    evaluated = 0
    work = frame[[group, "binary_label"]].copy()
    work["score"] = score
    for _, part in work.groupby(group, sort=False):
        y = part["binary_label"].to_numpy(dtype=np.int8)
        s = part["score"].to_numpy(dtype=np.float64)
        if y.min() == y.max():
            continue
        evaluated += 1
        aucs.append(float(roc_auc_score(y, s)))
        aps.append(float(average_precision_score(y, s)))
        order = np.argsort(-s, kind="stable")
        positives = int(y.sum())
        for k in recalls:
            top = y[order[: min(k, len(y))]]
            recalls[k].append(float(top.sum() / positives))
            discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = float((top * discounts).sum())
            ideal_n = min(positives, len(top))
            idcg = float(discounts[:ideal_n].sum())
            ndcgs[k].append(dcg / idcg if idcg else 0.0)
    result: dict[str, float | int | None] = {
        f"{prefix}_groups_with_both_classes": evaluated,
        f"{prefix}_macro_auroc": float(np.mean(aucs)) if aucs else None,
        f"{prefix}_macro_auprc": float(np.mean(aps)) if aps else None,
    }
    for k in recalls:
        result[f"{prefix}_macro_recall_at_{k}"] = float(np.mean(recalls[k])) if recalls[k] else None
        result[f"{prefix}_macro_ndcg_at_{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else None
    return result


def metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float | int | None]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    raw = np.asarray(score, dtype=np.float64)
    # Ranking metrics use the unaltered score. Probabilistic metrics require a
    # bounded value; ConPLex and Tanimoto are already in [0,1].
    probability = np.clip(raw, 0.0, 1.0)
    result: dict[str, float | int | None] = {
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "micro_auroc": float(roc_auc_score(y, raw)),
        "micro_auprc": float(average_precision_score(y, raw)),
        "brier": float(brier_score_loss(y, probability)),
        "ece_15": ece(y, probability),
    }
    result.update(grouped_binary_metrics(frame, raw, "target_chembl_id", "target"))
    result.update(grouped_binary_metrics(frame, raw, "parent_standard_inchi_key", "drug"))
    return result


def target_prior(train: pd.DataFrame, test: pd.DataFrame, strength: float = 10.0) -> np.ndarray:
    global_prior = float(train["binary_label"].mean())
    grouped = train.groupby("target_chembl_id")["binary_label"].agg(["sum", "count"])
    prior = (grouped["sum"] + strength * global_prior) / (grouped["count"] + strength)
    return test["target_chembl_id"].map(prior).fillna(global_prior).to_numpy(dtype=np.float64)


def max_positive_tanimoto(
    train: pd.DataFrame, test: pd.DataFrame, fingerprints: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    train_positive = train[train["binary_label"].eq(1)]
    pools = {
        target: np.unique(part["drug_feature_index"].to_numpy(dtype=np.int32))
        for target, part in train_positive.groupby("target_chembl_id", sort=False)
    }
    result = np.zeros(len(test), dtype=np.float32)
    target_without_positive_pool = 0
    rows_without_positive_pool = 0
    for target, row_positions in test.groupby("target_chembl_id", sort=False).indices.items():
        positions = np.asarray(row_positions, dtype=np.int64)
        reference_index = pools.get(target)
        if reference_index is None or len(reference_index) == 0:
            target_without_positive_pool += 1
            rows_without_positive_pool += len(positions)
            continue
        query_index = test.iloc[positions]["drug_feature_index"].to_numpy(dtype=np.int32)
        query = fingerprints[query_index].astype(np.float32, copy=False)
        reference = fingerprints[reference_index].astype(np.float32, copy=False)
        intersection = query @ reference.T
        union = query.sum(axis=1, keepdims=True) + reference.sum(axis=1)[None, :] - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        result[positions] = similarity.max(axis=1)
    return result, {
        "test_targets_without_train_positive_pool": int(target_without_positive_pool),
        "test_rows_without_train_positive_pool": int(rows_without_positive_pool),
    }


def run_one(data: pd.DataFrame, fingerprints: np.ndarray, protocol: str, fold: int) -> list[dict[str, object]]:
    masks = split_masks(data, protocol, fold)
    masks = {name: mask & data["drug_feature_available"].to_numpy(dtype=bool) for name, mask in masks.items()}
    train, valid, test = (data.loc[masks[name]].reset_index(drop=True) for name in ["train", "valid", "test"])
    if train.empty or valid.empty or test.empty or test["binary_label"].nunique() != 2:
        raise RuntimeError(f"Invalid split {protocol} fold {fold}: {len(train)}, {len(valid)}, {len(test)}")
    if protocol in {"S1_SCAFFOLD_COLD_DRUG", "S3_STRICT_DOUBLE_COLD", "S5_OLD_DRUG_ENTITY_COLD"}:
        if not set(train["scaffold_group"]).isdisjoint(test["scaffold_group"]):
            raise RuntimeError(f"Scaffold leakage in {protocol} fold {fold}")
    if protocol in {"S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"}:
        if not set(train["target_homology_cluster"]).isdisjoint(test["target_homology_cluster"]):
            raise RuntimeError(f"Target homology leakage in {protocol} fold {fold}")

    global_prior = float(train["binary_label"].mean())
    scores: dict[str, np.ndarray] = {
        "GLOBAL_TRAIN_PREVALENCE": np.full(len(test), global_prior, dtype=np.float64),
        "TARGET_TRAIN_EMPIRICAL_BAYES_PRIOR": target_prior(train, test),
        "CONPLEX_FROZEN_EXTERNAL": test["conplex_score"].to_numpy(dtype=np.float64),
    }
    similarity, similarity_audit = max_positive_tanimoto(train, test, fingerprints)
    scores["TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"] = similarity
    rows: list[dict[str, object]] = []
    prediction = test[[
        "calibration_pair_id", "sequence_key", "target_chembl_id", "primary_gene",
        "parent_standard_inchi_key", "parent_molecule_chembl_id", "binary_label",
    ]].copy()
    for model, score in scores.items():
        prediction[model] = score
        row: dict[str, object] = {
            "protocol": protocol,
            "fold": fold,
            "model": model,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
            "train_targets": int(train["target_chembl_id"].nunique()),
            "test_targets": int(test["target_chembl_id"].nunique()),
            "train_compounds": int(train["parent_standard_inchi_key"].nunique()),
            "test_compounds": int(test["parent_standard_inchi_key"].nunique()),
        }
        row.update(metrics(test, score))
        row.update(similarity_audit if model == "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO" else {})
        rows.append(row)
    run_dir = OUT / protocol / (f"fold_{fold}" if fold >= 0 else "fixed_split")
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(run_dir / "BASELINE_TEST_PREDICTIONS_V1.csv.gz", index=False)
    (run_dir / "BASELINE_METRICS_V1.json").write_text(
        json.dumps({"protocol": protocol, "fold": fold, "models": rows}, indent=2) + "\n"
    )
    print(json.dumps({
        "protocol": protocol,
        "fold": fold,
        "test_rows": len(test),
        "conplex_auprc": rows[2]["micro_auprc"],
        "similarity_auprc": rows[3]["micro_auprc"],
    }))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS + ["all"], default="all")
    parser.add_argument("--fold", default="all", help="0..4 or all; fixed protocols ignore this")
    args = parser.parse_args()
    audit = json.loads(FEATURE_AUDIT.read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("Feature store audit must pass before benchmarking")
    data = pd.read_csv(PAIRS, low_memory=False)
    fingerprints = np.load(MORGAN, mmap_mode="r")
    protocols = PROTOCOLS if args.protocol == "all" else [args.protocol]
    all_rows: list[dict[str, object]] = []
    for protocol in protocols:
        if protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"}:
            folds = [-1]
        elif args.fold == "all":
            folds = list(range(FOLDS))
        else:
            folds = [int(args.fold)]
        for fold in folds:
            all_rows.extend(run_one(data, fingerprints, protocol, fold))
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(all_rows)
    metrics_path = OUT / "ALL_BASELINE_METRICS_V1.csv"
    # Merge with existing runs when a subset was requested.
    if metrics_path.is_file() and (args.protocol != "all" or args.fold != "all"):
        old = pd.read_csv(metrics_path)
        keys = set(zip(metrics_frame["protocol"], metrics_frame["fold"], metrics_frame["model"]))
        old = old[~old.apply(lambda row: (row["protocol"], row["fold"], row["model"]) in keys, axis=1)]
        metrics_frame = pd.concat([old, metrics_frame], ignore_index=True)
    metrics_frame.sort_values(["protocol", "fold", "model"]).to_csv(metrics_path, index=False)
    checks = {
        "all_requested_runs_have_four_models": len(all_rows) % 4 == 0,
        "all_metrics_finite": metrics_frame[["micro_auroc", "micro_auprc", "brier", "ece_15"]].notna().all().all(),
        "all_test_sets_have_both_classes": (metrics_frame["positives"] > 0).all() and (metrics_frame["positives"] < metrics_frame["rows"]).all(),
        "feature_store_pass": audit["status"] == "PASS",
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope_note": "ConPLex is a frozen external-pretrained comparator with source-overlap not yet excluded; it is not evidence of BioMaster SOTA.",
        "checks": {key: bool(value) for key, value in checks.items()},
        "metrics_sha256": sha256(metrics_path),
        "requested_protocol": args.protocol,
        "requested_fold": args.fold,
        "completed_model_evaluations": int(len(all_rows)),
    }
    (OUT / "BASELINE_BENCHMARK_SUMMARY_V1.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

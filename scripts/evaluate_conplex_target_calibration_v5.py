#!/usr/bin/env python3
"""Evaluate ConPLEx per target against ChEMBL positives and weak/inactive pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs/current_production_package_v2/conplex_target_calibration_v5"
DEFAULT_LABELS = DEFAULT_DIR / "CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz"
DEFAULT_PREDICTIONS = DEFAULT_DIR / "CHEMBL37_CONPLEX_CALIBRATION_PREDICTIONS_V5.tsv"
DEFAULT_OUT = DEFAULT_DIR / "evaluation"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fold(scaffold: str, folds: int = 5) -> int:
    value = scaffold or "NO_SCAFFOLD"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % folds


def ef_at_fraction(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    n = len(labels)
    positives = int(labels.sum())
    if n == 0 or positives == 0:
        return float("nan")
    size = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-scores, kind="mergesort")
    top_rate = labels[order[:size]].mean()
    return float(top_rate / labels.mean())


def recall_at_budget(labels: np.ndarray, scores: np.ndarray, budget: int) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    return float(labels[order[: min(budget, len(order))]].sum() / positives)


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (
            probabilities <= right if right == 1.0 else probabilities < right
        )
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value)


def bedroc(labels: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    ordered = np.column_stack([scores, labels])[np.argsort(-scores, kind="mergesort")]
    return float(CalcBEDROC(ordered, 1, alpha))


def fingerprints(smiles: pd.Series) -> list[Any | None]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    values: list[Any | None] = []
    for value in smiles.fillna("").astype(str):
        molecule = Chem.MolFromSmiles(value)
        values.append(generator.GetFingerprint(molecule) if molecule is not None else None)
    return values


def scaffold_holdout_similarity(group: pd.DataFrame) -> np.ndarray:
    fps = fingerprints(group["model_ligand_smiles"])
    labels = group["binary_label"].to_numpy(dtype=int)
    folds = group["murcko_scaffold"].fillna("").astype(str).map(stable_fold).to_numpy()
    result = np.full(len(group), np.nan, dtype=float)
    for fold in sorted(set(folds)):
        train_positive = [
            fps[idx]
            for idx in range(len(group))
            if folds[idx] != fold and labels[idx] == 1 and fps[idx] is not None
        ]
        if not train_positive:
            continue
        for idx in np.flatnonzero(folds == fold):
            if fps[idx] is not None:
                result[idx] = max(DataStructs.BulkTanimotoSimilarity(fps[idx], train_positive))
    return result


def metrics(labels: np.ndarray, scores: np.ndarray, prefix: str = "") -> dict[str, float]:
    valid = np.isfinite(scores)
    y = labels[valid]
    s = scores[valid]
    output: dict[str, float] = {f"{prefix}n": float(len(y))}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return output
    clipped = np.clip(s, 0, 1)
    output.update(
        {
            f"{prefix}prevalence": float(y.mean()),
            f"{prefix}pr_auc": float(average_precision_score(y, s)),
            f"{prefix}roc_auc": float(roc_auc_score(y, s)),
            f"{prefix}ef_1pct": ef_at_fraction(y, s, 0.01),
            f"{prefix}ef_5pct": ef_at_fraction(y, s, 0.05),
            f"{prefix}bedroc_alpha20": bedroc(y, s, 20.0),
            f"{prefix}recall_at_24": recall_at_budget(y, s, 24),
            f"{prefix}recall_at_48": recall_at_budget(y, s, 48),
            f"{prefix}brier_descriptive": float(brier_score_loss(y, clipped)),
            f"{prefix}ece_descriptive": ece(y, clipped),
        }
    )
    return output


def random_baseline(labels: np.ndarray, prefix: str = "random_") -> dict[str, float]:
    """Analytic ranking baseline for a random ordering at the observed prevalence."""
    if len(labels) == 0:
        return {f"{prefix}n": 0.0}
    prevalence = float(labels.mean())
    return {
        f"{prefix}n": float(len(labels)),
        f"{prefix}prevalence": prevalence,
        f"{prefix}pr_auc": prevalence,
        f"{prefix}roc_auc": 0.5,
        f"{prefix}ef_1pct": 1.0,
        f"{prefix}ef_5pct": 1.0,
    }


def bootstrap_ap_difference(
    labels: np.ndarray,
    conplex: np.ndarray,
    similarity: np.ndarray,
    seed: int,
    iterations: int = 300,
) -> tuple[float, float, float]:
    valid = np.isfinite(conplex) & np.isfinite(similarity)
    y = labels[valid]
    left = conplex[valid]
    right = similarity[valid]
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if len(positives) < 2 or len(negatives) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sample = np.concatenate(
            [
                rng.choice(positives, size=len(positives), replace=True),
                rng.choice(negatives, size=len(negatives), replace=True),
            ]
        )
        differences.append(
            average_precision_score(y[sample], left[sample])
            - average_precision_score(y[sample], right[sample])
        )
    values = np.asarray(differences)
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_target(group: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    group = group.copy().reset_index(drop=True)
    group["ligand_similarity_scaffold_holdout"] = scaffold_holdout_similarity(group)
    labels = group["binary_label"].to_numpy(dtype=int)
    conplex = pd.to_numeric(group["conplex_score"], errors="coerce").to_numpy(dtype=float)
    similarity = group["ligand_similarity_scaffold_holdout"].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "sequence_key": group["sequence_key"].iloc[0],
        "primary_gene": group["primary_gene"].iloc[0],
        "target_assay_family": group["target_assay_family"].iloc[0],
        "positive_n": int(labels.sum()),
        "negative_n": int((labels == 0).sum()),
        "scaffold_n": int(group["murcko_scaffold"].nunique()),
    }
    row.update(metrics(labels, conplex, "conplex_"))
    row.update(metrics(labels, similarity, "similarity_"))
    row.update(random_baseline(labels))
    seed = int(hashlib.sha256(row["sequence_key"].encode()).hexdigest()[:8], 16)
    mean, lower, upper = bootstrap_ap_difference(labels, conplex, similarity, seed)
    row["conplex_minus_similarity_ap_bootstrap_mean"] = mean
    row["conplex_minus_similarity_ap_ci95_low"] = lower
    row["conplex_minus_similarity_ap_ci95_high"] = upper

    years = pd.to_numeric(group.get("min_document_year", pd.Series(np.nan, index=group.index)), errors="coerce")
    temporal = years.ge(2023).to_numpy()
    row["post2022_pair_n"] = int(temporal.sum())
    row["post2022_positive_n"] = int(labels[temporal].sum())
    row["post2022_negative_n"] = int((labels[temporal] == 0).sum())
    if temporal.sum():
        row.update(metrics(labels[temporal], conplex[temporal], "post2022_conplex_"))

    enough = row["positive_n"] >= 20 and row["negative_n"] >= 20
    better = np.isfinite(lower) and lower > 0
    useful_early = row.get("conplex_ef_5pct", 0) > row.get("random_ef_5pct", 1.0)
    row["conplex_target_use_status_v5"] = "T4_insufficient_positive_negative_data"
    if enough:
        row["conplex_target_use_status_v5"] = "T3_descriptive_not_better_than_similarity"
    if enough and better and useful_early:
        row["conplex_target_use_status_v5"] = "T1_target_calibrated_retrieval_supported"
    elif enough and row.get("conplex_pr_auc", 0) > row.get("conplex_prevalence", 1):
        row["conplex_target_use_status_v5"] = "T2_signal_without_baseline_superiority"
    return row, group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for path in [args.labels, args.predictions]:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = pd.read_csv(args.labels, low_memory=False)
    predictions = pd.read_csv(
        args.predictions,
        sep="\t",
        header=None,
        names=["calibration_pair_id", "prediction_sequence_key", "conplex_score"],
    )
    if predictions["calibration_pair_id"].duplicated().any():
        raise ValueError("Duplicate ConPLEx predictions")
    data = labels.merge(predictions, on="calibration_pair_id", how="left", validate="one_to_one")
    mismatch = data["sequence_key"].astype(str).ne(data["prediction_sequence_key"].astype(str))
    if mismatch.any() or data["conplex_score"].isna().any():
        raise ValueError("ConPLEx prediction mapping is incomplete or mismatched")

    target_rows: list[dict[str, Any]] = []
    annotated: list[pd.DataFrame] = []
    for _, group in data.groupby("sequence_key", sort=True):
        row, target_data = evaluate_target(group)
        target_rows.append(row)
        annotated.append(target_data)
    targets = pd.DataFrame(target_rows)
    if "conplex_pr_auc" not in targets:
        targets["conplex_pr_auc"] = np.nan
    targets = targets.sort_values(
        ["conplex_target_use_status_v5", "conplex_pr_auc", "primary_gene"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    all_pairs = pd.concat(annotated, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output_dir / "CONPLEX_TARGET_CALIBRATION_METRICS_V5.csv", index=False)
    all_pairs.to_csv(
        args.output_dir / "CONPLEX_CALIBRATION_PAIRS_SCORED_V5.csv.gz",
        index=False,
        compression="gzip",
    )
    status_counts = targets["conplex_target_use_status_v5"].value_counts().to_dict()
    summary = {
        "status": "passed",
        "created_utc": now(),
        "pair_rows": int(len(data)),
        "target_rows": int(len(targets)),
        "positive_rows": int(data["binary_label"].sum()),
        "negative_rows": int((data["binary_label"] == 0).sum()),
        "target_use_status": {str(key): int(value) for key, value in status_counts.items()},
        "targets_with_both_classes": int(((targets["positive_n"] > 0) & (targets["negative_n"] > 0)).sum()),
        "targets_with_20_each": int(((targets["positive_n"] >= 20) & (targets["negative_n"] >= 20)).sum()),
        "labels_sha256": sha256(args.labels),
        "predictions_sha256": sha256(args.predictions),
        "interpretation": "Target-specific retrospective calibration; post-2022 slice reduces but does not prove absence of all training overlap.",
    }
    (args.output_dir / "CONPLEX_TARGET_CALIBRATION_SUMMARY_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

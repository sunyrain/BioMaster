#!/usr/bin/env python3
"""Train scaffold-holdout target-specific Morgan logistic models on ChEMBL 37."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    ROOT
    / "outputs/current_production_package_v2/conplex_target_calibration_v5_official"
    / "CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz"
)
DRUGS = ROOT / "configs/project_drugs_v4.csv"
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/target_qsar_calibration_v5"
FP_SIZE = 2048
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(smiles: str) -> Any | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    return GENERATOR.GetFingerprint(molecule) if molecule is not None else None


def dense(fps: list[Any]) -> np.ndarray:
    matrix = np.zeros((len(fps), FP_SIZE), dtype=np.float32)
    for index, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, matrix[index])
    return matrix


def ef5(labels: np.ndarray, scores: np.ndarray) -> float:
    if not len(labels) or labels.sum() == 0:
        return np.nan
    size = max(1, int(np.ceil(len(labels) * 0.05)))
    order = np.argsort(-scores, kind="mergesort")
    return float(labels[order[:size]].mean() / labels.mean())


def paired_bootstrap_ap(
    labels: np.ndarray, left: np.ndarray, right: np.ndarray, seed: int, iterations: int = 200
) -> tuple[float, float, float]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(positives) < 2 or len(negatives) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sample = np.concatenate(
            [
                rng.choice(positives, size=len(positives), replace=True),
                rng.choice(negatives, size=len(negatives), replace=True),
            ]
        )
        values.append(
            average_precision_score(labels[sample], left[sample])
            - average_precision_score(labels[sample], right[sample])
        )
    array = np.asarray(values)
    return float(array.mean()), float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def train_one(group: pd.DataFrame, drug_frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    group = group.reset_index(drop=True).copy()
    labels = group["binary_label"].to_numpy(dtype=int)
    fps = [fingerprint(value) for value in group["model_ligand_smiles"]]
    valid = np.array([item is not None for item in fps])
    group = group.loc[valid].reset_index(drop=True)
    labels = labels[valid]
    fps = [item for item in fps if item is not None]
    scaffolds = group["murcko_scaffold"].fillna("").replace("", "NO_SCAFFOLD").astype(str).to_numpy()
    unique_scaffolds = np.unique(scaffolds)
    folds = min(5, len(unique_scaffolds))
    oof_qsar = np.full(len(group), np.nan)
    oof_similarity = np.full(len(group), np.nan)
    if folds >= 2:
        splitter = GroupKFold(n_splits=folds)
        for train_index, test_index in splitter.split(np.zeros(len(group)), labels, groups=scaffolds):
            y_train = labels[train_index]
            if len(np.unique(y_train)) < 2:
                continue
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=20260719,
            )
            model.fit(dense([fps[index] for index in train_index]), y_train)
            oof_qsar[test_index] = model.predict_proba(dense([fps[index] for index in test_index]))[:, 1]
            positive_fps = [fps[index] for index in train_index if labels[index] == 1]
            for index in test_index:
                oof_similarity[index] = max(DataStructs.BulkTanimotoSimilarity(fps[index], positive_fps))

    usable = np.isfinite(oof_qsar) & np.isfinite(oof_similarity)
    y = labels[usable]
    q = oof_qsar[usable]
    s = oof_similarity[usable]
    prevalence = float(y.mean()) if len(y) else np.nan
    row: dict[str, Any] = {
        "sequence_key": group["sequence_key"].iloc[0],
        "primary_gene": group["primary_gene"].iloc[0],
        "target_assay_family": group["target_assay_family"].iloc[0],
        "positive_n": int(labels.sum()),
        "negative_n": int((labels == 0).sum()),
        "scaffold_n": int(len(unique_scaffolds)),
        "oof_n": int(len(y)),
        "prevalence": prevalence,
    }
    if len(y) and len(np.unique(y)) == 2:
        row.update(
            {
                "qsar_oof_pr_auc": float(average_precision_score(y, q)),
                "similarity_oof_pr_auc": float(average_precision_score(y, s)),
                "qsar_oof_roc_auc": float(roc_auc_score(y, q)),
                "similarity_oof_roc_auc": float(roc_auc_score(y, s)),
                "qsar_oof_ef_5pct": ef5(y, q),
                "similarity_oof_ef_5pct": ef5(y, s),
            }
        )
        seed = int(hashlib.sha256(row["sequence_key"].encode()).hexdigest()[:8], 16)
        mean, lower, upper = paired_bootstrap_ap(y, q, s, seed)
        row["qsar_minus_similarity_ap_mean"] = mean
        row["qsar_minus_similarity_ap_ci95_low"] = lower
        row["qsar_minus_similarity_ap_ci95_high"] = upper
    else:
        for column in [
            "qsar_oof_pr_auc",
            "similarity_oof_pr_auc",
            "qsar_oof_roc_auc",
            "similarity_oof_roc_auc",
            "qsar_oof_ef_5pct",
            "similarity_oof_ef_5pct",
            "qsar_minus_similarity_ap_mean",
            "qsar_minus_similarity_ap_ci95_low",
            "qsar_minus_similarity_ap_ci95_high",
        ]:
            row[column] = np.nan

    min_year = pd.to_numeric(group.get("min_document_year", pd.Series(np.nan, index=group.index)), errors="coerce")
    max_year = pd.to_numeric(group.get("max_document_year", pd.Series(np.nan, index=group.index)), errors="coerce")
    temporal_train = max_year.le(2022).to_numpy()
    temporal_test = min_year.ge(2023).to_numpy()
    row["temporal_train_n"] = int(temporal_train.sum())
    row["temporal_test_n"] = int(temporal_test.sum())
    row["temporal_test_positive_n"] = int(labels[temporal_test].sum())
    row["temporal_test_negative_n"] = int((labels[temporal_test] == 0).sum())
    temporal_columns = [
        "temporal_qsar_pr_auc",
        "temporal_similarity_pr_auc",
        "temporal_qsar_roc_auc",
        "temporal_similarity_roc_auc",
        "temporal_qsar_ef_5pct",
        "temporal_similarity_ef_5pct",
    ]
    for column in temporal_columns:
        row[column] = np.nan
    temporal_evaluable = (
        temporal_train.sum() >= 40
        and temporal_test.sum() >= 10
        and labels[temporal_train].sum() >= 20
        and (labels[temporal_train] == 0).sum() >= 20
        and labels[temporal_test].sum() >= 5
        and (labels[temporal_test] == 0).sum() >= 5
    )
    if temporal_evaluable:
        temporal_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=20260719,
        )
        temporal_model.fit(
            dense([fps[index] for index in np.flatnonzero(temporal_train)]), labels[temporal_train]
        )
        temporal_qsar = temporal_model.predict_proba(
            dense([fps[index] for index in np.flatnonzero(temporal_test)])
        )[:, 1]
        train_positive_fps = [fps[index] for index in np.flatnonzero(temporal_train) if labels[index] == 1]
        temporal_similarity = np.array(
            [
                max(DataStructs.BulkTanimotoSimilarity(fps[index], train_positive_fps))
                for index in np.flatnonzero(temporal_test)
            ]
        )
        temporal_y = labels[temporal_test]
        row.update(
            {
                "temporal_qsar_pr_auc": float(average_precision_score(temporal_y, temporal_qsar)),
                "temporal_similarity_pr_auc": float(
                    average_precision_score(temporal_y, temporal_similarity)
                ),
                "temporal_qsar_roc_auc": float(roc_auc_score(temporal_y, temporal_qsar)),
                "temporal_similarity_roc_auc": float(
                    roc_auc_score(temporal_y, temporal_similarity)
                ),
                "temporal_qsar_ef_5pct": ef5(temporal_y, temporal_qsar),
                "temporal_similarity_ef_5pct": ef5(temporal_y, temporal_similarity),
            }
        )
    row["temporal_validation_status_v5"] = "not_evaluable"
    temporal_contradiction = False
    if temporal_evaluable:
        temporal_contradiction = (
            row["temporal_qsar_pr_auc"] + 0.02 < row["temporal_similarity_pr_auc"]
            or row["temporal_qsar_roc_auc"] < 0.50
        )
        row["temporal_validation_status_v5"] = (
            "contradicts_scaffold_result" if temporal_contradiction else "supports_or_noninferior"
        )

    enough = row["positive_n"] >= 20 and row["negative_n"] >= 20 and row["oof_n"] >= 40
    qsar_supported = (
        enough
        and row["qsar_minus_similarity_ap_ci95_low"] > 0
        and row["qsar_oof_ef_5pct"] > 1.0
        and not temporal_contradiction
    )
    similarity_supported = enough and row["similarity_oof_pr_auc"] > prevalence
    row["target_ligand_model_status_v5"] = "T4_insufficient_data"
    if similarity_supported:
        row["target_ligand_model_status_v5"] = "T2_similarity_supported"
    if enough and not similarity_supported:
        row["target_ligand_model_status_v5"] = "T3_no_reliable_ligand_signal"
    if qsar_supported:
        row["target_ligand_model_status_v5"] = "T1_qsar_beats_similarity"

    final_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=20260719,
    )
    final_model.fit(dense(fps), labels)
    drug_fps = drug_frame["_fingerprint"].tolist()
    predictions = final_model.predict_proba(dense(drug_fps))[:, 1]
    positive_fps = [fp for fp, label in zip(fps, labels) if label == 1]
    all_fps = fps
    max_positive_similarity = np.array(
        [max(DataStructs.BulkTanimotoSimilarity(fp, positive_fps)) for fp in drug_fps]
    )
    max_any_similarity = np.array(
        [max(DataStructs.BulkTanimotoSimilarity(fp, all_fps)) for fp in drug_fps]
    )
    active_scaffolds = set(group.loc[labels == 1, "murcko_scaffold"].fillna("").astype(str))
    exact_active_smiles = set(group.loc[labels == 1, "model_ligand_smiles"].astype(str))
    output = drug_frame.drop(columns="_fingerprint").copy()
    output["sequence_key"] = row["sequence_key"]
    output["primary_gene"] = row["primary_gene"]
    output["target_assay_family"] = row["target_assay_family"]
    output["target_ligand_model_status_v5"] = row["target_ligand_model_status_v5"]
    output["temporal_validation_status_v5"] = row["temporal_validation_status_v5"]
    output["target_qsar_probability_v5"] = predictions
    output["max_known_active_similarity_v5"] = max_positive_similarity
    output["max_known_compound_similarity_v5"] = max_any_similarity
    output["same_known_active_scaffold_v5"] = output["murcko_scaffold"].astype(str).isin(active_scaffolds)
    output["exact_known_active_smiles_v5"] = output["model_ligand_smiles"].astype(str).isin(exact_active_smiles)
    output["ligand_applicability_v5"] = "remote_lt_0.40"
    output.loc[max_positive_similarity >= 0.40, "ligand_applicability_v5"] = "moderate_0.40_0.69"
    output.loc[max_positive_similarity >= 0.70, "ligand_applicability_v5"] = "near_known_active_ge_0.70"
    output.loc[output["exact_known_active_smiles_v5"], "ligand_applicability_v5"] = "exact_known_active"
    return row, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for path in [args.labels, DRUGS]:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = pd.read_csv(args.labels, low_memory=False)
    labels = labels[labels["calibration_label"].isin(["positive", "negative_or_inactive"])].copy()
    labels["binary_label"] = labels["calibration_label"].eq("positive").astype(int)
    drugs = pd.read_csv(DRUGS, low_memory=False)
    drugs["_fingerprint"] = drugs["model_ligand_smiles"].map(fingerprint)
    if drugs["_fingerprint"].isna().any():
        raise ValueError("Project drug fingerprint failure")
    # Use the same standardized scaffold definition as the ChEMBL calibration labels.
    from rdkit.Chem.Scaffolds import MurckoScaffold

    drugs["murcko_scaffold"] = [
        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=False)
        or smiles
        for smiles in drugs["model_ligand_smiles"]
    ]

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for _, group in labels.groupby("sequence_key", sort=True):
        if group["binary_label"].nunique() < 2:
            continue
        row, predictions = train_one(group, drugs)
        metric_rows.append(row)
        prediction_rows.append(predictions)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["target_ligand_model_status_v5", "qsar_oof_pr_auc", "primary_gene"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "TARGET_QSAR_SCAFFOLD_HOLDOUT_METRICS_V5.csv"
    predictions_path = args.output_dir / "PROJECT_DRUG_TARGET_QSAR_PREDICTIONS_V5.csv.gz"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    status = metrics["target_ligand_model_status_v5"].value_counts().to_dict()
    summary = {
        "status": "passed",
        "created_utc": now(),
        "evaluated_targets": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "project_drugs": int(len(drugs)),
        "target_status": {str(key): int(value) for key, value in status.items()},
        "method": "fixed Morgan radius-2 2048-bit balanced logistic regression; 5-fold Murcko scaffold OOF",
        "interpretation": "Known-chemistry exploitation lane only; exact known actives are controls, not discoveries.",
        "inputs": {str(path): sha256(path) for path in [args.labels, DRUGS]},
    }
    (args.output_dir / "TARGET_QSAR_CALIBRATION_SUMMARY_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

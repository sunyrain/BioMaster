#!/usr/bin/env python3
"""Audit existing Top3000 model outputs against exact ChEMBL 37 pair labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = (
    ROOT
    / "outputs/current_production_package_v2/chembl37_target_calibration_api_v5"
    / "PROJECT_TARGET_CHEMBL37_API_NUMERIC_PAIRS_V5.csv.gz"
)
TOP3000 = ROOT / "outputs/current_production_package_v2/formal_full_universe_v4/refined_top3000_v4_complete.csv"
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/existing_model_chembl37_audit_v5"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ef5(labels: np.ndarray, scores: np.ndarray) -> float:
    size = max(1, int(np.ceil(len(labels) * 0.05)))
    order = np.argsort(-scores, kind="mergesort")
    return float(labels[order[:size]].mean() / labels.mean())


def metric_row(frame: pd.DataFrame, group_type: str, group_value: str) -> dict[str, Any]:
    labels = frame["binary_label"].to_numpy(dtype=int)
    row: dict[str, Any] = {
        "group_type": group_type,
        "group_value": group_value,
        "pair_n": int(len(frame)),
        "positive_n": int(labels.sum()),
        "negative_n": int((labels == 0).sum()),
        "prevalence": float(labels.mean()) if len(labels) else np.nan,
    }
    for source, column in {
        "conplex": "conplex_score",
        "boltz_affinity": "boltz_affinity_probability_refined",
        "boltz_confidence": "boltz_confidence_score_refined",
    }.items():
        scores = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(scores)
        y = labels[valid]
        s = scores[valid]
        row[f"{source}_n"] = int(len(y))
        if len(y) and len(np.unique(y)) == 2:
            row[f"{source}_pr_auc"] = float(average_precision_score(y, s))
            row[f"{source}_roc_auc"] = float(roc_auc_score(y, s))
            row[f"{source}_ef_5pct"] = ef5(y, s)
        else:
            row[f"{source}_pr_auc"] = np.nan
            row[f"{source}_roc_auc"] = np.nan
            row[f"{source}_ef_5pct"] = np.nan
    return row


def collapse_labels(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["sequence_key", "parent_molecule_chembl_id"]
    if not frame.duplicated(keys).any():
        return frame.copy()
    rows = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        row = group.iloc[0].copy()
        labels = set(group["calibration_label"].dropna().astype(str))
        if "conflicting_exclude" in labels or {"positive", "negative_or_inactive"}.issubset(labels):
            row["calibration_label"] = "conflicting_exclude"
        elif "positive" in labels:
            row["calibration_label"] = "positive"
        elif "negative_or_inactive" in labels:
            row["calibration_label"] = "negative_or_inactive"
        else:
            row["calibration_label"] = "grey_or_unresolved"
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--top3000", type=Path, default=TOP3000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for path in [args.pairs, args.top3000]:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = collapse_labels(pd.read_csv(args.pairs, low_memory=False))
    top = pd.read_csv(args.top3000, low_memory=False)
    if top.duplicated(["sequence_key", "drug_chembl_id"]).any():
        raise ValueError("Top3000 contains duplicate sequence_key/drug pairs")
    label_columns = [
        "sequence_key",
        "parent_molecule_chembl_id",
        "calibration_label",
        "min_pchembl",
        "max_pchembl",
        "mean_pchembl",
        "activity_rows",
        "assay_count",
        "document_count",
        "min_document_year",
        "max_document_year",
    ]
    available = [column for column in label_columns if column in labels]
    merged = top.merge(
        labels[available],
        left_on=["sequence_key", "drug_chembl_id"],
        right_on=["sequence_key", "parent_molecule_chembl_id"],
        how="left",
        validate="one_to_one",
    )
    merged["chembl37_exact_pair_overlap"] = merged["calibration_label"].notna()
    evaluation = merged[
        merged["calibration_label"].isin(["positive", "negative_or_inactive"])
    ].copy()
    evaluation["binary_label"] = evaluation["calibration_label"].eq("positive").astype(int)

    metrics: list[dict[str, Any]] = []
    if len(evaluation):
        metrics.append(metric_row(evaluation, "global_selected_top3000", "all"))
        for family, group in evaluation.groupby("target_assay_family_v2", dropna=False):
            if group["binary_label"].nunique() == 2:
                metrics.append(metric_row(group, "assay_family", str(family)))
        for target, group in evaluation.groupby("sequence_key", dropna=False):
            if group["binary_label"].nunique() == 2:
                metrics.append(metric_row(group, "target", str(target)))
    metric_table = pd.DataFrame(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = args.output_dir / "TOP3000_CHEMBL37_EXACT_PAIR_OVERLAP_V5.csv.gz"
    metrics_path = args.output_dir / "TOP3000_EXISTING_MODEL_DISCRIMINATION_V5.csv"
    merged.to_csv(overlap_path, index=False, compression="gzip")
    metric_table.to_csv(metrics_path, index=False)
    counts = merged["calibration_label"].fillna("no_exact_record").value_counts().to_dict()
    summary = {
        "status": "passed_diagnostic_only",
        "created_utc": now(),
        "top3000_rows": int(len(top)),
        "exact_overlap_rows": int(merged["chembl37_exact_pair_overlap"].sum()),
        "label_counts": {str(key): int(value) for key, value in counts.items()},
        "evaluation_rows": int(len(evaluation)),
        "evaluation_targets": int(evaluation["sequence_key"].nunique()),
        "targets_with_both_labels": int(
            evaluation.groupby("sequence_key")["binary_label"].nunique().eq(2).sum()
        ) if len(evaluation) else 0,
        "limitations": [
            "Top3000 was preselected by ConPLEx and feasibility filters, so this is not an unbiased benchmark.",
            "ChEMBL absence is unlabeled, not negative.",
            "Model-training overlap is not fully excluded.",
        ],
        "inputs": {
            str(args.pairs): sha256(args.pairs),
            str(args.top3000): sha256(args.top3000),
        },
    }
    (args.output_dir / "TOP3000_EXISTING_MODEL_CHEMBL37_AUDIT_SUMMARY_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

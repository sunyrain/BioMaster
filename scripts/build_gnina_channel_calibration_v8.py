#!/usr/bin/env python3
"""Freeze target-wise GNINA channel admission before discovery docking."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/affinity_experiment_package_v8.yaml"


def stable_seed(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return np.nan, np.nan
    positive_samples = positive[
        rng.integers(0, len(positive), size=(repeats, len(positive)))
    ]
    negative_samples = negative[
        rng.integers(0, len(negative), size=(repeats, len(negative)))
    ]
    comparisons = positive_samples[:, :, None] - negative_samples[:, None, :]
    estimates = (comparisons > 0).mean(axis=(1, 2)) + 0.5 * (
        comparisons == 0
    ).mean(axis=(1, 2))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def metric_summary(
    group: pd.DataFrame,
    score_column: str,
    prefix: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    labels = group["control_class"].eq("positive").astype(int).to_numpy()
    scores = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores)
    if valid.sum() < 4 or len(np.unique(labels[valid])) != 2:
        return {
            f"auroc_{prefix}_v8": np.nan,
            f"average_precision_{prefix}_v8": np.nan,
            f"auroc_ci_low_{prefix}_v8": np.nan,
            f"auroc_ci_high_{prefix}_v8": np.nan,
        }
    labels = labels[valid]
    scores = scores[valid]
    low, high = bootstrap_auc(labels, scores, repeats, seed)
    return {
        f"auroc_{prefix}_v8": float(roc_auc_score(labels, scores)),
        f"average_precision_{prefix}_v8": float(average_precision_score(labels, scores)),
        f"auroc_ci_low_{prefix}_v8": low,
        f"auroc_ci_high_{prefix}_v8": high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {key: ROOT / value for key, value in config["inputs"].items()}
    output_dir = ROOT / config["outputs"]["directory"] / "target_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    controls = pd.read_csv(inputs["gnina_control_scores"], low_memory=False)
    receptors = pd.read_csv(inputs["receptors"], low_memory=False)
    legacy = pd.read_csv(inputs["gnina_legacy_metrics"], low_memory=False)
    calibration = config["calibration"]
    repeats = int(calibration["bootstrap_repeats"])

    rows: list[dict[str, Any]] = []
    for sequence_key, group in controls.groupby("sequence_key", sort=True):
        positive = int(group["control_class"].eq("positive").sum())
        negative = int(group["control_class"].eq("negative").sum())
        baseline = positive / max(positive + negative, 1)
        row: dict[str, Any] = {
            "sequence_key": sequence_key,
            "primary_gene": group["primary_gene"].iloc[0],
            "target_assay_family": group["target_assay_family"].iloc[0],
            "positive_controls_v8": positive,
            "negative_controls_v8": negative,
            "class_prevalence_v8": baseline,
        }
        row.update(
            metric_summary(
                group,
                "best_cnn_affinity",
                "cnn_affinity",
                repeats,
                stable_seed(f"{sequence_key}:cnn"),
            )
        )
        row.update(
            metric_summary(
                group,
                "score_vina_directional",
                "vina_affinity",
                repeats,
                stable_seed(f"{sequence_key}:vina"),
            )
        )
        rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics = metrics.merge(
        receptors[
            [
                "sequence_key",
                "docking_receptor_source",
                "selected_pdb_id",
                "validated_experimental_holo",
                "selection_status",
            ]
        ],
        on="sequence_key",
        how="left",
        validate="one_to_one",
    ).merge(
        legacy[
            [
                "sequence_key",
                "auroc_locked_consensus",
                "average_precision_locked_consensus",
                "calibration_pass",
                "calibration_strong",
            ]
        ].rename(
            columns={
                "calibration_pass": "legacy_locked_consensus_pass_v2",
                "calibration_strong": "legacy_locked_consensus_strong_v2",
            }
        ),
        on="sequence_key",
        how="left",
        validate="one_to_one",
    )

    evaluable = (
        metrics["positive_controls_v8"].ge(int(calibration["minimum_positive"]))
        & metrics["negative_controls_v8"].ge(int(calibration["minimum_negative"]))
    )
    metrics["calibration_evaluable_v8"] = evaluable
    for channel in ["cnn_affinity", "vina_affinity"]:
        metrics[f"{channel}_pass_v8"] = (
            evaluable
            & metrics[f"auroc_{channel}_v8"].ge(float(calibration["auroc_minimum"]))
            & metrics[f"average_precision_{channel}_v8"].ge(
                metrics["class_prevalence_v8"]
                + float(calibration["average_precision_lift_minimum"])
            )
        )
        metrics[f"{channel}_strong_v8"] = (
            metrics[f"{channel}_pass_v8"]
            & metrics[f"auroc_ci_low_{channel}_v8"].ge(
                float(calibration["bootstrap_strong_lower_bound"])
            )
        )

    cnn_pass = metrics["cnn_affinity_pass_v8"]
    vina_pass = metrics["vina_affinity_pass_v8"]
    cnn_strong = metrics["cnn_affinity_strong_v8"]
    vina_strong = metrics["vina_affinity_strong_v8"]
    metrics["target_admission_tier_v8"] = "T0_not_admitted"
    metrics.loc[cnn_pass ^ vina_pass, "target_admission_tier_v8"] = "T4_single_pass"
    metrics.loc[(cnn_strong ^ vina_strong) & ~(cnn_pass & vina_pass), "target_admission_tier_v8"] = (
        "T3_single_strong"
    )
    metrics.loc[cnn_pass & vina_pass, "target_admission_tier_v8"] = "T2_dual_pass"
    metrics.loc[cnn_strong & vina_strong, "target_admission_tier_v8"] = "T1_dual_strong"
    metrics["target_admitted_v8"] = metrics["target_admission_tier_v8"].ne("T0_not_admitted")
    metrics["calibration_interpretation_v8"] = (
        "Admission is target-specific discrimination on ChEMBL controls; it is not a binder probability."
    )

    output = output_dir / "GNINA_TARGET_CHANNEL_CALIBRATION_V8.csv"
    metrics.to_csv(output, index=False)
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets_with_control_scores": int(len(metrics)),
        "targets_evaluable": int(metrics["calibration_evaluable_v8"].sum()),
        "cnn_affinity_pass_targets": int(metrics["cnn_affinity_pass_v8"].sum()),
        "vina_affinity_pass_targets": int(metrics["vina_affinity_pass_v8"].sum()),
        "dual_pass_targets": int((cnn_pass & vina_pass).sum()),
        "admitted_union_targets": int(metrics["target_admitted_v8"].sum()),
        "admission_tier_counts": metrics["target_admission_tier_v8"].value_counts().to_dict(),
        "receptor_source_by_admission": pd.crosstab(
            metrics["target_admission_tier_v8"], metrics["docking_receptor_source"]
        ).to_dict(),
        "rule": {
            "minimum_positive": int(calibration["minimum_positive"]),
            "minimum_negative": int(calibration["minimum_negative"]),
            "auroc_minimum": float(calibration["auroc_minimum"]),
            "average_precision_lift_minimum": float(
                calibration["average_precision_lift_minimum"]
            ),
            "strong_bootstrap_lower_bound": float(
                calibration["bootstrap_strong_lower_bound"]
            ),
            "combination": "channel-wise OR for target admission; no weighted score",
        },
        "output": str(output.relative_to(ROOT)),
    }
    summary_path = output_dir / "GNINA_TARGET_CHANNEL_CALIBRATION_V8_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

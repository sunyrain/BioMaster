#!/usr/bin/env python3
"""Audit paired two-seed ESM-C residual results against matching E0 runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/biomaster_odti_pretrained_formal_audit_v1"
SEEDS = [20260816, 20260817]


def aggregate(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frames.append(
            pd.read_csv(
                path,
                usecols=[
                    "calibration_pair_id",
                    "binary_label",
                    "v2_probability_calibrated",
                ],
                low_memory=False,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return frame.groupby(
        ["calibration_pair_id", "binary_label"], as_index=False
    ).agg(score=("v2_probability_calibrated", "mean"))


def paired_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    clusters: pd.DataFrame,
    repeats: int = 500,
    seed: int = 20260819,
) -> pd.DataFrame:
    frame = candidate.merge(
        baseline,
        on=["calibration_pair_id", "binary_label"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).merge(clusters, on="calibration_pair_id", validate="one_to_one")
    rows = []
    rng = np.random.default_rng(seed)
    for cluster_column in ["target_homology_cluster", "scaffold_group"]:
        values = frame[cluster_column].astype(str).to_numpy()
        unique = pd.unique(values)
        members = {name: np.flatnonzero(values == name) for name in unique}
        deltas = {"micro_auprc": [], "micro_auroc": []}
        for _ in range(repeats):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([members[name] for name in sampled])
            part = frame.iloc[indices]
            y = part["binary_label"].to_numpy(dtype=np.int8)
            if y.min() == y.max():
                continue
            deltas["micro_auprc"].append(
                average_precision_score(y, part["score_candidate"])
                - average_precision_score(y, part["score_baseline"])
            )
            deltas["micro_auroc"].append(
                roc_auc_score(y, part["score_candidate"])
                - roc_auc_score(y, part["score_baseline"])
            )
        for metric, values_metric in deltas.items():
            values_metric = np.asarray(values_metric, dtype=np.float64)
            low, high = np.quantile(values_metric, [0.025, 0.975])
            rows.append(
                {
                    "cluster_column": cluster_column,
                    "metric": metric,
                    "bootstrap_replicates": int(len(values_metric)),
                    "mean_delta_candidate_minus_e0": float(values_metric.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                }
            )
    return pd.DataFrame(rows)


def cell_paths(fold: int, protocol: str, seed: int, candidate: bool) -> Path:
    if protocol == "S3_STRICT_DOUBLE_COLD":
        base = (
            ROOT / "outputs/biomaster_odti_esmc_formal_s3_20260819"
            if candidate
            else ROOT / "outputs/odti_unified_champion_s3_20260817"
        )
        name = f"S3_STRICT_DOUBLE_COLD__fold_{fold}__seed_{seed}"
    else:
        base = (
            ROOT / "outputs/biomaster_odti_esmc_formal_s5_20260819"
            if candidate
            else ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s5_esm2_formal"
        )
        name = f"S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}"
    return base / name / "TEST_PREDICTIONS_V2.csv.gz"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(
        ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz",
        usecols=["calibration_pair_id", "target_homology_cluster", "scaffold_group"],
        low_memory=False,
    )
    metrics_rows = []
    bootstrap_rows = []
    for protocol in ["S3_STRICT_DOUBLE_COLD", "S5_OLD_DRUG_ENTITY_COLD"]:
        if protocol.startswith("S3"):
            candidate_paths = [cell_paths(fold, protocol, seed, True) for fold in range(5) for seed in SEEDS]
            baseline_paths = [cell_paths(fold, protocol, seed, False) for fold in range(5) for seed in SEEDS]
        else:
            candidate_paths = [cell_paths(-1, protocol, seed, True) for seed in SEEDS]
            baseline_paths = [cell_paths(-1, protocol, seed, False) for seed in SEEDS]
        if not all(path.is_file() for path in candidate_paths + baseline_paths):
            raise FileNotFoundError("missing paired prediction artifact")
        candidate = aggregate(candidate_paths)
        baseline = aggregate(baseline_paths)
        paired = candidate.merge(
            baseline,
            on=["calibration_pair_id", "binary_label"],
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        y = paired["binary_label"].to_numpy(dtype=np.int8)
        candidate_ap = float(average_precision_score(y, paired["score_candidate"]))
        baseline_ap = float(average_precision_score(y, paired["score_baseline"]))
        candidate_roc = float(roc_auc_score(y, paired["score_candidate"]))
        baseline_roc = float(roc_auc_score(y, paired["score_baseline"]))
        metrics_rows.extend(
            [
                {"protocol": protocol, "model": "esmc_candidate", "rows": len(paired), "micro_auprc": candidate_ap, "micro_auroc": candidate_roc},
                {"protocol": protocol, "model": "e0_matching_seeds", "rows": len(paired), "micro_auprc": baseline_ap, "micro_auroc": baseline_roc},
            ]
        )
        bootstrap = paired_bootstrap(candidate, baseline, frozen)
        bootstrap.insert(0, "protocol", protocol)
        bootstrap_rows.append(bootstrap)
    metrics_path = OUT / "PRETRAINED_RESIDUAL_FORMAL_PAIRED_METRICS_V1.csv"
    bootstrap_path = OUT / "PRETRAINED_RESIDUAL_FORMAL_PAIRED_BOOTSTRAP_V1.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(bootstrap_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol": "ESMC_600M_SEPARATE_TARGET_RESIDUAL_PAIRED_FORMAL_2SEED",
        "seeds": SEEDS,
        "metrics": metrics_rows,
        "bootstrap": pd.concat(bootstrap_rows, ignore_index=True).to_dict(orient="records"),
        "artifacts": {"metrics": str(metrics_path), "bootstrap": str(bootstrap_path)},
        "claim_status": "PAIRED_INTERNAL_2SEED; NOT_CHAMPION_PROMOTION; EXTERNAL_GATE_PENDING",
    }
    (OUT / "PRETRAINED_RESIDUAL_FORMAL_AUDIT_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

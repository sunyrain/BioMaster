#!/usr/bin/env python3
"""Pair-aligned V2 versus V1 comparison with cold-start cluster bootstrap."""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_biomaster_odti_baselines_v1 import metrics


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
V1_ROOT = ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_routed_ranker_v1"
V2_ROOT = ROOT / "outputs/old_drug_target_sota_v1"


def load_v1(paths: list[str]) -> tuple[pd.DataFrame, list[int]]:
    frames = []
    seeds: set[int] = set()
    for path in paths:
        match = re.search(r"__seed_(\d+)__", path)
        if not match:
            raise ValueError(f"cannot parse V1 seed from {path}")
        seed = int(match.group(1))
        seeds.add(seed)
        frame = pd.read_csv(path, low_memory=False)
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no V1 prediction files")
    all_rows = pd.concat(frames, ignore_index=True)
    key = "calibration_pair_id"
    if all_rows.duplicated([key, "seed"]).any():
        raise RuntimeError("V1 has duplicate pair/seed predictions")
    grouped = (
        all_rows.groupby(key, as_index=False)
        .agg(
            v1_score=("biomaster_probability_calibrated", "mean"),
            v1_seed_count=("seed", "nunique"),
            binary_label=("binary_label", "first"),
            target_chembl_id=("target_chembl_id", "first"),
            parent_standard_inchi_key=("parent_standard_inchi_key", "first"),
        )
    )
    return grouped, sorted(seeds)


def bootstrap_clusters(
    frame: pd.DataFrame,
    cluster_column: str,
    left: str,
    right: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    clusters = frame[cluster_column].dropna().astype(str).unique()
    if len(clusters) < 2:
        raise ValueError(f"not enough clusters for {cluster_column}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    memberships = {
        cluster: np.flatnonzero(frame[cluster_column].astype(str).to_numpy() == cluster)
        for cluster in clusters
    }

    def safe_binary_metric(y: np.ndarray, score: np.ndarray, metric: str) -> float:
        if y.size == 0 or np.min(y) == np.max(y):
            return float("nan")
        if metric == "micro_auroc":
            return float(roc_auc_score(y, score))
        if metric == "micro_auprc":
            return float(average_precision_score(y, score))
        raise ValueError(metric)

    for replicate in range(repeats):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([memberships[cluster] for cluster in sampled])
        subset = frame.iloc[indices]
        y = subset["binary_label"].to_numpy(dtype=np.int8)
        metric_scores = {
            "micro_auroc": (
                safe_binary_metric(y, subset[left].to_numpy(dtype=np.float64), "micro_auroc"),
                safe_binary_metric(y, subset[right].to_numpy(dtype=np.float64), "micro_auroc"),
            ),
            "micro_auprc": (
                safe_binary_metric(y, subset[left].to_numpy(dtype=np.float64), "micro_auprc"),
                safe_binary_metric(y, subset[right].to_numpy(dtype=np.float64), "micro_auprc"),
            ),
        }
        for metric, (left_value, right_value) in metric_scores.items():
            if not np.isfinite(left_value) or not np.isfinite(right_value):
                continue
            rows.append(
                {
                    "cluster_column": cluster_column,
                    "replicate": replicate,
                    "metric": metric,
                    "v2_minus_v1": float(left_value) - float(right_value),
                }
            )
    return pd.DataFrame(rows)


def summarize_bootstrap(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cluster, metric), part in table.groupby(["cluster_column", "metric"], sort=True):
        values = part["v2_minus_v1"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "cluster_column": cluster,
                "metric": metric,
                "bootstrap_replicates": int(len(values)),
                "mean_delta": float(values.mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "ci_excludes_zero": bool(np.quantile(values, 0.025) > 0 or np.quantile(values, 0.975) < 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=["S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"],
        default="S3_STRICT_DOUBLE_COLD",
    )
    parser.add_argument("--v2")
    parser.add_argument("--v1-glob")
    parser.add_argument("--pairs", default=str(PAIRS))
    parser.add_argument("--out-dir")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    suffix = "s2" if args.protocol.startswith("S2_") else "s3"
    v2_path = (
        Path(args.v2)
        if args.v2
        else V2_ROOT / f"biomaster_odti_v2_v21_{suffix}_formal/{args.protocol}_V2_SEED_MEAN_PREDICTIONS.csv.gz"
    )
    v1_glob = args.v1_glob or str(
        V1_ROOT / f"{args.protocol}__fold_*__seed_*__CORE/TEST_PREDICTIONS_V1.csv.gz"
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else V2_ROOT / f"biomaster_odti_v2_v21_{suffix}_formal/paired_v1_v2_audit"
    )
    pairs_path = Path(args.pairs)
    if not v2_path.is_file() or not pairs_path.is_file():
        raise FileNotFoundError("V2 aggregate or frozen pair table is missing")
    v1_paths = sorted(glob.glob(v1_glob))
    v1, v1_seeds = load_v1(v1_paths)
    v2 = pd.read_csv(v2_path, low_memory=False)
    required_v2 = {"calibration_pair_id", "score_mean", "binary_label"}
    if not required_v2.issubset(v2.columns):
        raise ValueError(f"V2 aggregate missing columns: {sorted(required_v2 - set(v2.columns))}")
    v2 = v2.rename(columns={"score_mean": "v2_score"})
    pairs = pd.read_csv(pairs_path, low_memory=False)[
        ["calibration_pair_id", "scaffold_group", "target_homology_cluster"]
    ]
    frame = v2.merge(v1, on="calibration_pair_id", how="inner", suffixes=("", "_v1"))
    frame = frame.merge(pairs, on="calibration_pair_id", how="left", validate="one_to_one")
    if frame.empty:
        raise RuntimeError("V1 and V2 have no pair intersection")
    if frame["v1_seed_count"].min() < 5:
        raise RuntimeError("V1 pair-aligned comparison is missing one or more of five seeds")
    if frame["seed_count"].min() < 5:
        raise RuntimeError("V2 pair-aligned comparison is missing one or more of five seeds")
    frame["binary_label"] = frame["binary_label"].astype(np.int8)

    metric_rows = []
    for name, score in [("V1", "v1_score"), ("V2", "v2_score")]:
        result = metrics(frame, frame[score].to_numpy(dtype=np.float64))
        result.update({"model": name, "pair_count": int(len(frame))})
        metric_rows.append(result)
    metrics_table = pd.DataFrame(metric_rows)

    bootstrap_tables = []
    for index, cluster in enumerate(["target_homology_cluster", "scaffold_group"]):
        bootstrap_tables.append(
            bootstrap_clusters(
                frame,
                cluster,
                "v2_score",
                "v1_score",
                args.bootstrap_repeats,
                args.seed + index,
            )
        )
    bootstrap_raw = pd.concat(bootstrap_tables, ignore_index=True)
    bootstrap_summary = summarize_bootstrap(bootstrap_raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "S2" if args.protocol.startswith("S2_") else "S3"
    frame.to_csv(out_dir / f"{prefix}_V1_V2_PAIR_ALIGNED_PREDICTIONS.csv.gz", index=False, compression="gzip")
    metrics_table.to_csv(out_dir / f"{prefix}_V1_V2_METRICS.csv", index=False)
    bootstrap_raw.to_csv(out_dir / f"{prefix}_V1_V2_CLUSTER_BOOTSTRAP_RAW.csv.gz", index=False, compression="gzip")
    bootstrap_summary.to_csv(out_dir / f"{prefix}_V1_V2_CLUSTER_BOOTSTRAP_SUMMARY.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol": args.protocol,
        "pair_count": int(len(frame)),
        "v1_seeds": v1_seeds,
        "v2_seed_count_min": int(frame["seed_count"].min()),
        "v2_seed_count_max": int(frame["seed_count"].max()),
        "bootstrap_repeats_per_cluster": args.bootstrap_repeats,
        "metrics": metrics_table.to_dict(orient="records"),
        "bootstrap_summary": bootstrap_summary.to_dict(orient="records"),
        "claim_status": "PAIRED_INTERNAL_EVIDENCE; NO_EXTERNAL_OR_PROSPECTIVE_CLAIM",
    }
    (out_dir / f"{prefix}_V1_V2_COMPARISON_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

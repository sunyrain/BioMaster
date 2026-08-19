#!/usr/bin/env python3
"""Pair-aligned bootstrap comparison between two V2 aggregate suites.

This is used for architecture ablations such as ProtBERT-only versus pooled
ESM2, or pooled ESM2 versus residue-token cross-attention.  The two inputs
must be produced by the V2 evaluator and contain pair-aligned ``score_mean``
and ``seed_count`` columns.  Cluster bootstrap is performed on the frozen
scaffold and target-homology groups rather than resampling individual rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_biomaster_odti_baselines_v1 import metrics


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"


def load_suite(path: Path, name: str, min_seeds: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"calibration_pair_id", "score_mean", "binary_label", "seed_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} aggregate is missing columns: {sorted(missing)}")
    if frame.duplicated("calibration_pair_id").any():
        raise ValueError(f"{name} aggregate contains duplicate pair ids")
    if int(frame["seed_count"].min()) < min_seeds:
        raise ValueError(
            f"{name} aggregate has seed_count<{min_seeds} for at least one pair"
        )
    return frame[["calibration_pair_id", "score_mean", "binary_label", "seed_count"]].rename(
        columns={"score_mean": f"{name}_score", "seed_count": f"{name}_seed_count"}
    )


def bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    left_score: str,
    right_score: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    cluster_values = frame[cluster_column].astype(str).to_numpy()
    clusters = pd.unique(cluster_values)
    if len(clusters) < 2:
        raise ValueError(f"not enough clusters for {cluster_column}")
    memberships = {
        cluster: np.flatnonzero(cluster_values == cluster)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(repeats):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([memberships[cluster] for cluster in sampled])
        subset = frame.iloc[indices]
        labels = subset["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        for metric_name, scorer in [
            ("micro_auroc", roc_auc_score),
            ("micro_auprc", average_precision_score),
        ]:
            left = float(scorer(labels, subset[left_score].to_numpy(dtype=np.float64)))
            right = float(scorer(labels, subset[right_score].to_numpy(dtype=np.float64)))
            rows.append(
                {
                    "cluster_column": cluster_column,
                    "replicate": replicate,
                    "metric": metric_name,
                    "left_minus_right": left - right,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="left V2 aggregate CSV.gz")
    parser.add_argument("--right", required=True, help="right V2 aggregate CSV.gz")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--pairs", default=str(PAIRS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    if args.min_seeds < 1:
        raise ValueError("--min-seeds must be positive")
    left = load_suite(Path(args.left), args.left_name, args.min_seeds)
    right = load_suite(Path(args.right), args.right_name, args.min_seeds)
    frame = left.merge(right, on="calibration_pair_id", how="inner", validate="one_to_one")
    if frame.empty:
        raise RuntimeError("the two V2 suites have no pair intersection")
    if not frame["binary_label_x"].eq(frame["binary_label_y"]).all():
        raise RuntimeError("the two suites disagree on binary labels")
    frame = frame.rename(columns={"binary_label_x": "binary_label"}).drop(columns=["binary_label_y"])
    pairs = pd.read_csv(args.pairs, low_memory=False)[
        [
            "calibration_pair_id",
            "target_chembl_id",
            "parent_standard_inchi_key",
            "scaffold_group",
            "target_homology_cluster",
        ]
    ]
    frame = frame.merge(pairs, on="calibration_pair_id", how="left", validate="one_to_one")
    if frame[["scaffold_group", "target_homology_cluster"]].isna().any().any():
        raise RuntimeError("missing frozen cluster assignments in paired frame")

    left_score = f"{args.left_name}_score"
    right_score = f"{args.right_name}_score"
    metric_rows = []
    for name, score in [(args.left_name, left_score), (args.right_name, right_score)]:
        row = metrics(frame, frame[score].to_numpy(dtype=np.float64))
        row.update({"model": name, "pair_count": int(len(frame))})
        metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows)
    raw = pd.concat(
        [
            bootstrap(frame, cluster, left_score, right_score, args.bootstrap_repeats, args.seed + i)
            for i, cluster in enumerate(["target_homology_cluster", "scaffold_group"])
        ],
        ignore_index=True,
    )
    summary_rows = []
    for (cluster, metric_name), part in raw.groupby(["cluster_column", "metric"], sort=True):
        values = part["left_minus_right"].to_numpy(dtype=np.float64)
        low, high = np.quantile(values, [0.025, 0.975])
        summary_rows.append(
            {
                "cluster_column": cluster,
                "metric": metric_name,
                "bootstrap_replicates": int(len(values)),
                "mean_delta": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    bootstrap_summary = pd.DataFrame(summary_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "V2_ABLATION_PAIR_ALIGNED.csv.gz", index=False, compression="gzip")
    metric_table.to_csv(out_dir / "V2_ABLATION_METRICS.csv", index=False)
    raw.to_csv(out_dir / "V2_ABLATION_BOOTSTRAP_RAW.csv.gz", index=False, compression="gzip")
    bootstrap_summary.to_csv(out_dir / "V2_ABLATION_BOOTSTRAP_SUMMARY.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "left": str(args.left),
        "right": str(args.right),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "pair_count": int(len(frame)),
        "bootstrap_repeats_per_cluster": int(args.bootstrap_repeats),
        "metrics": metric_table.to_dict(orient="records"),
        "bootstrap_summary": bootstrap_summary.to_dict(orient="records"),
        "claim_status": "PAIRED_INTERNAL_ABLATION; NO_EXTERNAL_OR_PROSPECTIVE_CLAIM",
    }
    (out_dir / "V2_ABLATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

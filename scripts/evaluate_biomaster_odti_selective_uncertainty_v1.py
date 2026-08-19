#!/usr/bin/env python3
"""Evaluate seed-ensemble uncertainty and selective risk for BioMaster-ODTI.

The score is not interpreted as a calibrated binding probability.  Instead,
independent seed predictions are pair-aligned and their spread is used for
selective prediction and wet-lab triage:

* exploitation: low-spread pairs retained at a chosen coverage;
* exploration: high-spread pairs for information-gain experiments.

No test labels are used to fit a threshold.  Labels are only read to report
post-hoc risk/coverage metrics.
"""

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


def parse_seed(path: str) -> int:
    match = re.search(r"__seed_(\d+)", path)
    if not match:
        raise ValueError(f"cannot parse seed from prediction path: {path}")
    return int(match.group(1))


def load_seed_predictions(paths: list[str], score_column: str) -> tuple[pd.DataFrame, list[int]]:
    if not paths:
        raise FileNotFoundError("no prediction files matched --run-glob")
    frames: list[pd.DataFrame] = []
    seeds: list[int] = []
    required = {
        "calibration_pair_id",
        "binary_label",
        "target_chembl_id",
        "parent_standard_inchi_key",
        score_column,
    }
    for path in paths:
        seed = parse_seed(path)
        frame = pd.read_csv(path, low_memory=False)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame.duplicated("calibration_pair_id").any():
            raise ValueError(f"duplicate pair ids in {path}")
        frame = frame[list(required)].copy()
        frame["seed"] = seed
        frames.append(frame)
        seeds.append(seed)
    all_rows = pd.concat(frames, ignore_index=True)
    if all_rows.duplicated(["calibration_pair_id", "seed"]).any():
        raise ValueError("duplicate pair/seed rows across input files")
    grouped = (
        all_rows.groupby("calibration_pair_id", as_index=False)
        .agg(
            binary_label=("binary_label", "first"),
            target_chembl_id=("target_chembl_id", "first"),
            parent_standard_inchi_key=("parent_standard_inchi_key", "first"),
            score_mean=(score_column, "mean"),
            score_std=(score_column, "std"),
            score_min=(score_column, "min"),
            score_max=(score_column, "max"),
            seed_count=("seed", "nunique"),
        )
    )
    grouped["score_std"] = grouped["score_std"].fillna(0.0)
    grouped["uncertainty"] = grouped["score_std"]
    grouped["confidence"] = 1.0 / (1.0 + grouped["uncertainty"])
    return grouped, sorted(set(seeds))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def selective_table(frame: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    ordered = frame.sort_values(["uncertainty", "calibration_pair_id"], kind="mergesort")
    labels = ordered["binary_label"].to_numpy(dtype=np.int8)
    scores = ordered["score_mean"].to_numpy(dtype=np.float64)
    uncertainties = ordered["uncertainty"].to_numpy(dtype=np.float64)
    rows = []
    for coverage in coverages:
        count = max(1, int(np.ceil(len(ordered) * coverage)))
        y = labels[:count]
        s = scores[:count]
        rows.append(
            {
                "coverage": float(count / len(ordered)),
                "retained_rows": int(count),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "micro_auroc": safe_auc(y, s),
                "micro_auprc": float(average_precision_score(y, s)) if y.size else float("nan"),
                "mean_uncertainty": float(uncertainties[:count].mean()),
                "uncertainty_quantile_cut": float(uncertainties[count - 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--score-column", default="v2_probability_calibrated")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.run_glob))
    frame, seeds = load_seed_predictions(paths, args.score_column)
    if len(seeds) < args.min_seeds or int(frame["seed_count"].min()) < args.min_seeds:
        raise ValueError(
            f"need at least {args.min_seeds} seeds for every pair; got seeds={seeds}, "
            f"min_pair_seed_count={frame['seed_count'].min()}"
        )
    coverages = [round(value, 2) for value in np.linspace(0.10, 1.00, 10)]
    selective = selective_table(frame, coverages)
    full = metrics(frame, frame["score_mean"].to_numpy(dtype=np.float64))
    frame["exploitation_rank"] = frame["uncertainty"].rank(method="first", ascending=True)
    frame["exploration_rank"] = frame["uncertainty"].rank(method="first", ascending=False)
    frame = frame.sort_values("exploration_rank")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "PAIR_ALIGNED_ENSEMBLE_UNCERTAINTY.csv.gz", index=False, compression="gzip")
    selective.to_csv(out_dir / "SELECTIVE_RISK_COVERAGE.csv", index=False)
    frame.head(1000).to_csv(out_dir / "EXPLORATION_TOP1000_HIGH_SPREAD.csv.gz", index=False, compression="gzip")
    frame.sort_values("exploitation_rank").head(1000).to_csv(
        out_dir / "EXPLOITATION_TOP1000_LOW_SPREAD.csv.gz", index=False, compression="gzip"
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "prediction_files": paths,
        "seeds": seeds,
        "pair_count": int(len(frame)),
        "min_pair_seed_count": int(frame["seed_count"].min()),
        "full_metrics": full,
        "uncertainty_definition": "std_across_independent_seed_probabilities",
        "selective_risk_path": str(out_dir / "SELECTIVE_RISK_COVERAGE.csv"),
        "exploitation_path": str(out_dir / "EXPLOITATION_TOP1000_LOW_SPREAD.csv.gz"),
        "exploration_path": str(out_dir / "EXPLORATION_TOP1000_HIGH_SPREAD.csv.gz"),
        "claim_status": "UNCERTAINTY_TRIAGE_ONLY; NOT_A_CALIBRATED_BINDING_PROBABILITY",
    }
    (out_dir / "UNCERTAINTY_EVALUATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

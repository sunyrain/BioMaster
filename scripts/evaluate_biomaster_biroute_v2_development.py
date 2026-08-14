#!/usr/bin/env python3
"""Development-fold ablation and paired-bootstrap audit for BiRoute V2.

This evaluator is explicitly retrospective on predeclared development folds.
It may decide whether the frozen method is worth carrying forward, but it may
not alter the method or support a confirmatory/SOTA claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
RUNS = BASE / "biomaster_biroute_v2"
REFERENCE = (
    BASE / "public_retrained_v1/dtiam_same_data_comparison_v1/"
    "S3_STRICT_DOUBLE_COLD_ALIGNED_PREDICTIONS_V1.csv.gz"
)
FREEZE = ROOT / "configs/biomaster_biroute_v2_freeze.json"
OUT = RUNS / "development_audit_v2"
ABLATIONS = [
    "FULL_BIROUTE",
    "NO_DIRECTIONAL_RANK_LOSSES",
    "NO_ROUTE_CONDITIONING",
    "SYMMETRIC_RETRIEVAL_HEAD",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold(fold: int, seed: int) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference["fold"].eq(fold)].copy()
    required = [
        "calibration_pair_id", "binary_label", "dtiam_probability", "biomaster_stack_score",
        "CONPLEX_FROZEN_EXTERNAL", "scaffold_group", "target_homology_cluster",
    ]
    frame = reference[required]
    for ablation in ABLATIONS:
        path = RUNS / (
            f"S3_STRICT_DOUBLE_COLD__fold_{fold}__seed_{seed}__{ablation}/"
            "TEST_PREDICTIONS_V2.csv.gz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        prediction = pd.read_csv(path, usecols=["calibration_pair_id", "binary_label", "biroute_probability"])
        if not prediction["binary_label"].equals(
            prediction["calibration_pair_id"].map(
                frame.set_index("calibration_pair_id")["binary_label"]
            )
        ):
            raise RuntimeError(f"Label alignment changed for {path}")
        frame = frame.merge(
            prediction[["calibration_pair_id", "biroute_probability"]].rename(
                columns={"biroute_probability": ablation}
            ),
            on="calibration_pair_id", validate="one_to_one",
        )
    if not frame["calibration_pair_id"].is_unique:
        raise RuntimeError("Development fold pairs are not unique")
    frame["fold"] = fold
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text())
    development = set(freeze["scope"]["development_folds"])
    if not set(args.folds).issubset(development):
        raise RuntimeError("This evaluator may only inspect predeclared development folds")
    frames = [load_fold(fold, args.seed) for fold in args.folds]
    pooled = pd.concat(frames, ignore_index=True)
    if not pooled["calibration_pair_id"].is_unique:
        raise RuntimeError("Development fold pair sets overlap")
    OUT.mkdir(parents=True, exist_ok=True)

    score_rows = []
    columns = ABLATIONS + ["dtiam_probability", "biomaster_stack_score", "CONPLEX_FROZEN_EXTERNAL"]
    y = pooled["binary_label"].to_numpy(dtype=np.int8)
    for column in columns:
        values = pooled[column].to_numpy(dtype=np.float64)
        score_rows.append({
            "model": column,
            "folds": ";".join(map(str, args.folds)),
            "rows": len(pooled),
            "micro_auprc": float(average_precision_score(y, values)),
            "micro_auroc": float(roc_auc_score(y, values)),
        })
    comparisons = []
    references = [
        "NO_DIRECTIONAL_RANK_LOSSES", "NO_ROUTE_CONDITIONING", "SYMMETRIC_RETRIEVAL_HEAD",
        "dtiam_probability", "biomaster_stack_score", "CONPLEX_FROZEN_EXTERNAL",
    ]
    for reference in references:
        for cluster_index, cluster_column in enumerate(["target_homology_cluster", "scaffold_group"]):
            row = paired_cluster_bootstrap(
                pooled, "FULL_BIROUTE", reference, "auprc", args.iterations,
                args.seed + len(comparisons) * 100 + cluster_index,
                cluster_column=cluster_column,
            )
            row["folds"] = ";".join(map(str, args.folds))
            row["audit_role"] = "RETROSPECTIVE_DEVELOPMENT_ONLY"
            comparisons.append(row)
    score_frame = pd.DataFrame(score_rows).sort_values("micro_auprc", ascending=False)
    comparison_frame = pd.DataFrame(comparisons)
    score_path = OUT / "BIROUTE_V2_DEVELOPMENT_SCORES.csv"
    comparison_path = OUT / "BIROUTE_V2_DEVELOPMENT_PAIRED_BOOTSTRAP.csv"
    aligned_path = OUT / "BIROUTE_V2_DEVELOPMENT_ALIGNED_PREDICTIONS.csv.gz"
    score_frame.to_csv(score_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)
    pooled.to_csv(aligned_path, index=False)
    ablation_support = {}
    for reference in ABLATIONS[1:]:
        rows = comparison_frame[comparison_frame["reference"].eq(reference)]
        ablation_support[reference] = {
            "full_biroute_better_on_both_cluster_axes": bool(
                len(rows) == 2 and rows["ci95_excludes_zero_in_favor_of_challenger"].all()
            ),
            "minimum_observed_difference": float(rows["observed_difference"].min()),
        }
    checks = {
        "only_predeclared_development_folds": set(args.folds).issubset(development),
        "all_four_predeclared_models_aligned": all(column in pooled for column in ABLATIONS),
        "pair_ids_unique": pooled["calibration_pair_id"].is_unique,
        "all_bootstraps_complete": len(comparison_frame) == len(references) * 2,
        "all_scores_finite": np.isfinite(score_frame[["micro_auprc", "micro_auroc"]]).all().all(),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "folds": args.folds,
        "rows": int(len(pooled)),
        "primary_model": "FULL_BIROUTE",
        "scores": score_frame.to_dict("records"),
        "predeclared_ablation_support": ablation_support,
        "checks": {name: bool(value) for name, value in checks.items()},
        "decision_boundary": (
            "Retrospective development evidence only. Confirmatory folds 3-4 remain untouched; "
            "no architecture or hyperparameter may be changed after promotion."
        ),
        "claim_status": "NO_CONFIRMATORY_OR_SOTA_CLAIM",
        "hashes": {
            "method_freeze": sha256(FREEZE),
            "scores": sha256(score_path),
            "bootstrap": sha256(comparison_path),
            "aligned_predictions": sha256(aligned_path),
        },
    }
    summary_path = OUT / "BIROUTE_V2_DEVELOPMENT_AUDIT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

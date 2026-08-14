#!/usr/bin/env python3
"""Paired cluster-bootstrap tests for frozen BioMaster-ODTI cold splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
MODEL = BASE / "biomaster_odti_routed_ranker_v1"
BASELINE = BASE / "baseline_results_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
OUT = MODEL / "paired_bootstrap_v1"
PROTOCOLS = ["S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold(protocol: str, fold: int, seed: int) -> pd.DataFrame:
    run = MODEL / f"{protocol}__fold_{fold}__seed_{seed}__CORE/TEST_PREDICTIONS_V1.csv.gz"
    baseline = BASELINE / protocol / f"fold_{fold}/BASELINE_TEST_PREDICTIONS_V1.csv.gz"
    if not run.is_file() or not baseline.is_file():
        raise FileNotFoundError([str(run), str(baseline)])
    neural = pd.read_csv(
        run,
        usecols=["calibration_pair_id", "binary_label", "biomaster_probability_calibrated"],
    )
    comparison = pd.read_csv(
        baseline,
        usecols=[
            "calibration_pair_id", "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        ],
    )
    frame = neural.merge(comparison, on="calibration_pair_id", how="inner", validate="one_to_one")
    frame["protocol"] = protocol
    frame["fold"] = fold
    return frame


def score(y: np.ndarray, values: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(y, values))
    if metric == "auroc":
        return float(roc_auc_score(y, values))
    raise ValueError(metric)


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    challenger: str,
    reference: str,
    metric: str,
    iterations: int,
    seed: int,
    cluster_column: str = "target_homology_cluster",
) -> dict[str, object]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    challenger_score = frame[challenger].to_numpy(dtype=np.float64)
    reference_score = frame[reference].to_numpy(dtype=np.float64)
    observed_challenger = score(y, challenger_score, metric)
    observed_reference = score(y, reference_score, metric)
    clusters = frame[cluster_column].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    positions = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    generator = np.random.default_rng(seed)
    differences: list[float] = []
    skipped = 0
    for _ in range(iterations):
        sampled = generator.choice(unique_clusters, size=len(unique_clusters), replace=True)
        index = np.concatenate([positions[cluster] for cluster in sampled])
        sampled_y = y[index]
        if sampled_y.min() == sampled_y.max():
            skipped += 1
            continue
        differences.append(
            score(sampled_y, challenger_score[index], metric)
            - score(sampled_y, reference_score[index], metric)
        )
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < iterations * 0.95:
        raise RuntimeError("Too many invalid bootstrap resamples")
    return {
        "metric": metric,
        "challenger": challenger,
        "reference": reference,
        "rows": int(len(frame)),
        "clusters": int(len(unique_clusters)),
        "cluster_column": cluster_column,
        "challenger_score": observed_challenger,
        "reference_score": observed_reference,
        "observed_difference": observed_challenger - observed_reference,
        "bootstrap_iterations_requested": iterations,
        "bootstrap_iterations_valid": int(len(values)),
        "bootstrap_iterations_skipped": skipped,
        "difference_ci95_low": float(np.quantile(values, 0.025)),
        "difference_ci95_high": float(np.quantile(values, 0.975)),
        "probability_difference_gt_zero": float((values > 0).mean()),
        "ci95_excludes_zero_in_favor_of_challenger": bool(np.quantile(values, 0.025) > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--model-seed", type=int, default=20260813)
    args = parser.parse_args()
    pair_meta = pd.read_csv(
        PAIRS, usecols=["calibration_pair_id", "target_homology_cluster"], low_memory=False
    )
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        folds = []
        for fold in range(5):
            frame = load_fold(protocol, fold, args.model_seed)
            frame = frame.merge(pair_meta, on="calibration_pair_id", how="left", validate="one_to_one")
            if frame["target_homology_cluster"].isna().any():
                raise RuntimeError("Missing homology cluster")
            folds.append(frame)
            coverage_rows.append({
                "protocol": protocol,
                "evaluation": f"FOLD_{fold}",
                "rows": len(frame),
                "unique_pairs": frame["calibration_pair_id"].nunique(),
                "clusters": frame["target_homology_cluster"].nunique(),
            })
            for reference in ["CONPLEX_FROZEN_EXTERNAL", "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"]:
                for metric in ["auprc", "auroc"]:
                    row = paired_cluster_bootstrap(
                        frame,
                        "biomaster_probability_calibrated",
                        reference,
                        metric,
                        args.iterations,
                        args.seed + fold * 100 + len(results),
                    )
                    row.update({"protocol": protocol, "evaluation": f"FOLD_{fold}", "fold": fold})
                    results.append(row)
        oof = pd.concat(folds, ignore_index=True)
        if not oof["calibration_pair_id"].is_unique:
            raise RuntimeError(f"OOF pairs overlap for {protocol}")
        coverage_rows.append({
            "protocol": protocol,
            "evaluation": "POOLED_OOF",
            "rows": len(oof),
            "unique_pairs": oof["calibration_pair_id"].nunique(),
            "clusters": oof["target_homology_cluster"].nunique(),
        })
        for reference in ["CONPLEX_FROZEN_EXTERNAL", "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"]:
            for metric in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    oof,
                    "biomaster_probability_calibrated",
                    reference,
                    metric,
                    args.iterations,
                    args.seed + 10000 + len(results),
                )
                row.update({"protocol": protocol, "evaluation": "POOLED_OOF", "fold": -1})
                results.append(row)

    result_frame = pd.DataFrame(results)
    coverage = pd.DataFrame(coverage_rows)
    result_path = OUT / "PAIRED_CLUSTER_BOOTSTRAP_RESULTS_V1.csv"
    coverage_path = OUT / "PAIRED_OOF_COVERAGE_V1.csv"
    result_frame.to_csv(result_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    s2_conplex = result_frame[
        result_frame["protocol"].eq("S2_HOMOLOGY_COLD_TARGET")
        & result_frame["evaluation"].eq("POOLED_OOF")
        & result_frame["reference"].eq("CONPLEX_FROZEN_EXTERNAL")
        & result_frame["metric"].eq("auprc")
    ].iloc[0]
    s3_conplex = result_frame[
        result_frame["protocol"].eq("S3_STRICT_DOUBLE_COLD")
        & result_frame["evaluation"].eq("POOLED_OOF")
        & result_frame["reference"].eq("CONPLEX_FROZEN_EXTERNAL")
        & result_frame["metric"].eq("auprc")
    ].iloc[0]
    checks = {
        "all_fold_pair_sets_unique": (coverage["rows"] == coverage["unique_pairs"]).all(),
        "expected_ten_folds_and_two_pooled_rows": len(coverage) == 12,
        "all_bootstrap_results_complete": len(result_frame) == 48,
        "all_ci_bounds_finite": np.isfinite(result_frame[["difference_ci95_low", "difference_ci95_high"]]).all().all(),
        "s2_pooled_auprc_ci_excludes_zero_vs_conplex": bool(s2_conplex["ci95_excludes_zero_in_favor_of_challenger"]),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "bootstrap_unit": "target_homology_cluster; clusters sampled with replacement and all member rows retained",
        "model_seed": args.model_seed,
        "bootstrap_seed": args.seed,
        "iterations": args.iterations,
        "checks": {key: bool(value) for key, value in checks.items()},
        "primary_s2_pooled_vs_conplex_auprc": s2_conplex.to_dict(),
        "primary_s3_pooled_vs_conplex_auprc": s3_conplex.to_dict(),
        "artifacts": {
            "results_sha256": sha256(result_path),
            "coverage_sha256": sha256(coverage_path),
        },
        "claim_status": "PAIRED_INTERNAL_EVIDENCE_ONLY; PUBLIC_SAME_SPLIT_MODELS_AND_MULTIPLE_MODEL_SEEDS_STILL_REQUIRED",
    }
    summary_path = OUT / "PAIRED_CLUSTER_BOOTSTRAP_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

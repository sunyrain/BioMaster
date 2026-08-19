#!/usr/bin/env python3
"""Generic paired cluster-bootstrap for a completed V2 S1/S2/S3 aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DEFAULT_DTIAM = {
    "S1_SCAFFOLD_COLD_DRUG": BASE / "public_retrained_v1/dtiam_same_data_comparison_v1/S1_SCAFFOLD_COLD_DRUG_ALIGNED_PREDICTIONS_V1.csv.gz",
    "S2_HOMOLOGY_COLD_TARGET": BASE / "public_retrained_v1/dtiam_same_data_comparison_v1/S2_HOMOLOGY_COLD_TARGET_ALIGNED_PREDICTIONS_V1.csv.gz",
    "S3_STRICT_DOUBLE_COLD": BASE / "public_retrained_v1/dtiam_same_data_comparison_v1/S3_STRICT_DOUBLE_COLD_ALIGNED_PREDICTIONS_V1.csv.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=list(DEFAULT_DTIAM), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dtiam-predictions", default=None)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    protocol = args.protocol
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "paired_bootstrap_v1"
    prediction_path = model_dir / f"{protocol}_V2_SEED_MEAN_PREDICTIONS.csv.gz"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    v2 = pd.read_csv(prediction_path, low_memory=False).rename(columns={"score_mean": "V2_SEED_MEAN"})
    folds = []
    for fold in range(5):
        path = BASE / f"baseline_results_v1/{protocol}/fold_{fold}/BASELINE_TEST_PREDICTIONS_V1.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        folds.append(pd.read_csv(path, low_memory=False))
    baseline = pd.concat(folds, ignore_index=True)
    if not baseline["calibration_pair_id"].is_unique:
        raise RuntimeError("baseline OOF pair ids overlap")
    references = [
        "TARGET_TRAIN_EMPIRICAL_BAYES_PRIOR",
        "CONPLEX_FROZEN_EXTERNAL",
        "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
    ]
    frame = v2.merge(
        baseline[["calibration_pair_id", *references]],
        on="calibration_pair_id",
        how="inner",
        validate="one_to_one",
    )
    dtiam_path = Path(args.dtiam_predictions) if args.dtiam_predictions else DEFAULT_DTIAM[protocol]
    if dtiam_path.is_file():
        dtiam = pd.read_csv(dtiam_path, low_memory=False)
        dtiam_score = next(
            (column for column in ["dtiam_probability", "DTIAM_probability", "score_mean"] if column in dtiam.columns),
            None,
        )
        if dtiam_score is not None:
            dtiam = dtiam[["calibration_pair_id", dtiam_score]].rename(columns={dtiam_score: "DTIAM_SAME_DATA"})
            frame = frame.merge(dtiam, on="calibration_pair_id", how="inner", validate="one_to_one")
            references.append("DTIAM_SAME_DATA")
    meta = pd.read_csv(
        PAIRS,
        usecols=["calibration_pair_id", "scaffold_group", "target_homology_cluster"],
        low_memory=False,
    )
    frame = frame.merge(meta, on="calibration_pair_id", how="left", validate="one_to_one")
    checks = {
        "pair_ids_unique": bool(frame["calibration_pair_id"].is_unique),
        "all_clusters_present": bool(frame[["scaffold_group", "target_homology_cluster"]].notna().all().all()),
        "scores_finite": bool(np.isfinite(frame[["V2_SEED_MEAN", *references]].to_numpy(dtype=np.float64)).all()),
        "five_seed_coverage": bool(v2["seed_count"].eq(5).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    results = []
    for cluster_column in ["scaffold_group", "target_homology_cluster"]:
        for reference in references:
            for metric in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    frame,
                    "V2_SEED_MEAN",
                    reference,
                    metric,
                    args.iterations,
                    args.seed + len(results),
                    cluster_column,
                )
                row.update({"protocol": protocol})
                results.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_path = out_dir / f"{protocol}_V2_PAIRED_CLUSTER_BOOTSTRAP_RESULTS_V1.csv"
    aligned_path = out_dir / f"{protocol}_V2_PAIR_ALIGNED_PREDICTIONS_V1.csv.gz"
    result_frame.to_csv(result_path, index=False)
    frame.to_csv(aligned_path, index=False, compression="gzip")
    primary = result_frame[
        result_frame["cluster_column"].eq("target_homology_cluster")
        & result_frame["metric"].eq("auprc")
    ]
    final_checks = {
        **checks,
        "all_bootstrap_rows_complete": len(result_frame) == len(references) * 4,
        "all_ci_finite": bool(np.isfinite(result_frame[["difference_ci95_low", "difference_ci95_high"]].to_numpy()).all()),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(final_checks.values()) else "FAIL",
        "protocol": protocol,
        "rows": int(len(frame)),
        "iterations": args.iterations,
        "references": references,
        "checks": {key: bool(value) for key, value in final_checks.items()},
        "primary_target_cluster_auprc": primary.to_dict(orient="records"),
        "artifacts": {
            "results_sha256": sha256(result_path),
            "aligned_predictions_sha256": sha256(aligned_path),
        },
        "claim_status": "PAIRED_INTERNAL_CLUSTER_BOOTSTRAP; EXTERNAL_AND_PROSPECTIVE_GATES_REMAIN",
    }
    summary_path = out_dir / f"{protocol}_V2_PAIRED_CLUSTER_BOOTSTRAP_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

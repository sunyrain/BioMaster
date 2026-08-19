#!/usr/bin/env python3
"""Paired cluster bootstrap for the completed S1 V2 multiseed ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
MODEL = BASE / "biomaster_odti_v2_v21_s1_formal"
PREDICTION = MODEL / "S1_SCAFFOLD_COLD_DRUG_V2_SEED_MEAN_PREDICTIONS.csv.gz"
BASELINES = BASE / "baseline_results_v1/S1_SCAFFOLD_COLD_DRUG"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DTIAM = BASE / "public_retrained_v1/dtiam_same_data_comparison_v1/S1_SCAFFOLD_COLD_DRUG_ALIGNED_PREDICTIONS_V1.csv.gz"
OUT = MODEL / "paired_bootstrap_v1"
ITERATIONS = 2000
SEED = 20260817
REFERENCES = [
    "TARGET_TRAIN_EMPIRICAL_BAYES_PRIOR",
    "CONPLEX_FROZEN_EXTERNAL",
    "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
    "DTIAM_SAME_DATA",
]
BASELINE_REFERENCES = [reference for reference in REFERENCES if reference != "DTIAM_SAME_DATA"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(MODEL))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    prediction = model_dir / "S1_SCAFFOLD_COLD_DRUG_V2_SEED_MEAN_PREDICTIONS.csv.gz"
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "paired_bootstrap_v1"
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    v2 = pd.read_csv(prediction, low_memory=False).rename(columns={"score_mean": "V2_SEED_MEAN"})
    baseline_frames = [
        pd.read_csv(BASELINES / f"fold_{fold}/BASELINE_TEST_PREDICTIONS_V1.csv.gz", low_memory=False)
        for fold in range(5)
    ]
    baseline = pd.concat(baseline_frames, ignore_index=True)
    if not baseline["calibration_pair_id"].is_unique:
        raise RuntimeError("S1 baseline OOF pair ids overlap")
    meta = pd.read_csv(
        PAIRS,
        usecols=["calibration_pair_id", "scaffold_group", "target_homology_cluster"],
        low_memory=False,
    )
    dtiam = pd.read_csv(
        DTIAM,
        usecols=["calibration_pair_id", "dtiam_probability"],
        low_memory=False,
    ).rename(columns={"dtiam_probability": "DTIAM_SAME_DATA"})
    if not dtiam["calibration_pair_id"].is_unique:
        raise RuntimeError("DTIAM same-data pair ids overlap")
    frame = v2.merge(
        baseline[["calibration_pair_id", *BASELINE_REFERENCES]],
        on="calibration_pair_id",
        how="inner",
        validate="one_to_one",
    ).merge(dtiam, on="calibration_pair_id", how="inner", validate="one_to_one")
    frame = frame.merge(meta, on="calibration_pair_id", how="left", validate="one_to_one")
    checks = {
        "exact_86673_pairs": len(frame) == 86673,
        "pair_ids_unique": frame["calibration_pair_id"].is_unique,
        "five_seed_coverage": bool(v2["seed_count"].eq(5).all()),
        "all_clusters_present": bool(
            frame[["scaffold_group", "target_homology_cluster"]].notna().all().all()
        ),
        "scores_finite": bool(
            np.isfinite(frame[["V2_SEED_MEAN", *REFERENCES]].to_numpy(dtype=np.float64)).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    results: list[dict[str, object]] = []
    for cluster_column in ["scaffold_group", "target_homology_cluster"]:
        for reference in REFERENCES:
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
                row.update({"protocol": "S1_SCAFFOLD_COLD_DRUG"})
                results.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_path = out_dir / "S1_V2_PAIRED_CLUSTER_BOOTSTRAP_RESULTS_V1.csv"
    aligned_path = out_dir / "S1_V2_PAIR_ALIGNED_PREDICTIONS_V1.csv.gz"
    result_frame.to_csv(result_path, index=False)
    frame.to_csv(aligned_path, index=False, compression="gzip")

    def primary(reference: str, cluster: str = "scaffold_group") -> dict[str, object]:
        selected = result_frame[
            result_frame["reference"].eq(reference)
            & result_frame["cluster_column"].eq(cluster)
            & result_frame["metric"].eq("auprc")
        ]
        if len(selected) != 1:
            raise RuntimeError((reference, cluster, len(selected)))
        return selected.iloc[0].to_dict()

    primary_rows = {reference: primary(reference) for reference in REFERENCES}
    final_checks = {
        **checks,
        "all_bootstrap_rows_complete": len(result_frame) == 16,
        "all_ci_finite": bool(
            np.isfinite(
                result_frame[["difference_ci95_low", "difference_ci95_high"]].to_numpy()
            ).all()
        ),
        "scaffold_ci_positive_vs_conplex": bool(
            primary_rows["CONPLEX_FROZEN_EXTERNAL"]["difference_ci95_low"] > 0
        ),
        "scaffold_ci_positive_vs_similarity": bool(
            primary_rows["TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"]["difference_ci95_low"] > 0
        ),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(final_checks.values()) else "FAIL",
        "protocol": "S1_SCAFFOLD_COLD_DRUG",
        "model": "BIOMASTER_ODTI_V2_5SEED_MEAN",
        "rows": int(len(frame)),
        "iterations": args.iterations,
        "bootstrap_axes": ["scaffold_group", "target_homology_cluster"],
        "checks": {key: bool(value) for key, value in final_checks.items()},
        "primary_scaffold_auprc": primary_rows,
        "artifacts": {
            "results_sha256": sha256(result_path),
            "aligned_predictions_sha256": sha256(aligned_path),
        },
        "claim_status": "MULTISEED_INTERNAL_SCAFFOLD_COLD_EVIDENCE; EXTERNAL_AND_W1_GATES_REMAIN",
    }
    summary_path = out_dir / "S1_V2_PAIRED_CLUSTER_BOOTSTRAP_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(final_checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

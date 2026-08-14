#!/usr/bin/env python3
"""Five-seed development audit for the BiRoute directional ranking component."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


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
OUT = RUNS / "multiseed_development_audit_v2"
SEEDS = [20260813, 20260814, 20260815, 20260816, 20260817]
FOLDS = [0, 1, 2]
VARIANTS = ["FULL_BIROUTE", "NO_DIRECTIONAL_RANK_LOSSES"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(fold: int, seed: int, variant: str) -> Path:
    return RUNS / (
        f"S3_STRICT_DOUBLE_COLD__fold_{fold}__seed_{seed}__{variant}/"
        "TEST_PREDICTIONS_V2.csv.gz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text())
    if freeze["scope"]["development_folds"] != FOLDS:
        raise RuntimeError("Predeclared development folds changed")
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference["fold"].isin(FOLDS)][[
        "calibration_pair_id", "binary_label", "fold", "dtiam_probability",
        "biomaster_stack_score", "scaffold_group", "target_homology_cluster",
    ]].copy()
    if len(reference) != 10531 or not reference["calibration_pair_id"].is_unique:
        raise RuntimeError("Expected exact 10,531-row development OOF population")
    aligned = reference.copy()
    score_rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            pieces = []
            for fold in FOLDS:
                path = prediction_path(fold, seed, variant)
                if not path.is_file():
                    raise FileNotFoundError(path)
                piece = pd.read_csv(
                    path, usecols=["calibration_pair_id", "binary_label", "biroute_probability"]
                )
                piece["fold"] = fold
                pieces.append(piece)
            prediction = pd.concat(pieces, ignore_index=True)
            if len(prediction) != len(reference) or not prediction["calibration_pair_id"].is_unique:
                raise RuntimeError(f"Prediction coverage changed for {seed} {variant}")
            expected_label = prediction["calibration_pair_id"].map(
                reference.set_index("calibration_pair_id")["binary_label"]
            )
            if not prediction["binary_label"].reset_index(drop=True).equals(
                expected_label.reset_index(drop=True)
            ):
                raise RuntimeError(f"Label alignment changed for {seed} {variant}")
            column = f"{variant}__seed_{seed}"
            aligned = aligned.merge(
                prediction[["calibration_pair_id", "biroute_probability"]].rename(
                    columns={"biroute_probability": column}
                ),
                on="calibration_pair_id", validate="one_to_one",
            )
            y = aligned["binary_label"].to_numpy(dtype=np.int8)
            score_rows.append({
                "seed": seed,
                "variant": variant,
                "rows": len(prediction),
                "pooled_development_auprc": float(
                    average_precision_score(y, aligned[column].to_numpy(dtype=np.float64))
                ),
            })
    for variant in VARIANTS:
        columns = [f"{variant}__seed_{seed}" for seed in SEEDS]
        aligned[f"{variant}__FIVE_SEED_MEAN"] = aligned[columns].mean(axis=1)
    score_frame = pd.DataFrame(score_rows)
    pivot = score_frame.pivot(index="seed", columns="variant", values="pooled_development_auprc")
    pivot["full_minus_no_rank"] = pivot["FULL_BIROUTE"] - pivot["NO_DIRECTIONAL_RANK_LOSSES"]
    y = aligned["binary_label"].to_numpy(dtype=np.int8)
    ensemble_scores = {
        variant: float(average_precision_score(y, aligned[f"{variant}__FIVE_SEED_MEAN"]))
        for variant in VARIANTS
    }
    comparisons = []
    for reference_column in [
        "NO_DIRECTIONAL_RANK_LOSSES__FIVE_SEED_MEAN",
        "dtiam_probability",
        "biomaster_stack_score",
    ]:
        for axis_index, cluster_column in enumerate(["target_homology_cluster", "scaffold_group"]):
            row = paired_cluster_bootstrap(
                aligned,
                "FULL_BIROUTE__FIVE_SEED_MEAN",
                reference_column,
                "auprc",
                args.iterations,
                20260813 + len(comparisons) * 100 + axis_index,
                cluster_column=cluster_column,
            )
            row["audit_role"] = "RETROSPECTIVE_DEVELOPMENT_MULTI_SEED"
            comparisons.append(row)
    comparison_frame = pd.DataFrame(comparisons)
    component_rows = comparison_frame[
        comparison_frame["reference"].eq("NO_DIRECTIONAL_RANK_LOSSES__FIVE_SEED_MEAN")
    ]
    component_supported = bool(
        (pivot["full_minus_no_rank"] > 0).sum() >= 4
        and len(component_rows) == 2
        and component_rows["ci95_excludes_zero_in_favor_of_challenger"].all()
    )
    full_vs_dtiam = comparison_frame[comparison_frame["reference"].eq("dtiam_probability")]
    full_v2_promotion = bool(
        component_supported
        and len(full_vs_dtiam) == 2
        and full_vs_dtiam["ci95_excludes_zero_in_favor_of_challenger"].all()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    score_path = OUT / "BIROUTE_V2_MULTI_SEED_SCORES.csv"
    comparison_path = OUT / "BIROUTE_V2_MULTI_SEED_PAIRED_BOOTSTRAP.csv"
    aligned_path = OUT / "BIROUTE_V2_MULTI_SEED_ALIGNED_PREDICTIONS.csv.gz"
    score_frame.to_csv(score_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)
    aligned.to_csv(aligned_path, index=False)
    checks = {
        "exact_five_seeds": sorted(score_frame["seed"].unique()) == SEEDS,
        "exact_two_variants_each_seed": len(score_frame) == len(SEEDS) * len(VARIANTS),
        "exact_development_oof_population": len(aligned) == 10531 and aligned["calibration_pair_id"].is_unique,
        "all_scores_finite": np.isfinite(score_frame["pooled_development_auprc"]).all(),
        "all_six_bootstraps_complete": len(comparison_frame) == 6,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "folds": FOLDS,
        "seeds": SEEDS,
        "rows": len(aligned),
        "per_seed": pivot.reset_index().to_dict("records"),
        "five_seed_ensemble_auprc": ensemble_scores,
        "directional_ranking_component_supported_in_development": component_supported,
        "full_v2_promotion_to_confirmatory": full_v2_promotion,
        "decision": (
            "ADVANCE_FULL_V2_TO_CONFIRMATORY_FOLDS"
            if full_v2_promotion
            else (
                "DO_NOT_ADVANCE_FULL_V2; CARRY_DIRECTIONAL_RANKING_COMPONENT_TO_NEW_PREDECLARED_VERSION"
                if component_supported
                else "DO_NOT_ADVANCE_FULL_V2; DIRECTIONAL_RANKING_COMPONENT_NOT_STABLE"
            )
        ),
        "checks": {name: bool(value) for name, value in checks.items()},
        "claim_status": "DEVELOPMENT_COMPONENT_EVIDENCE_ONLY; CONFIRMATORY_FOLDS_UNTOUCHED",
        "hashes": {
            "method_freeze": sha256(FREEZE),
            "scores": sha256(score_path),
            "bootstrap": sha256(comparison_path),
            "aligned_predictions": sha256(aligned_path),
        },
    }
    summary_path = OUT / "BIROUTE_V2_MULTI_SEED_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

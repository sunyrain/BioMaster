#!/usr/bin/env python3
"""Compare same-data DTIAM retraining to frozen BioMaster/public baselines."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap, sha256  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
DTIAM = BASE / "public_retrained_v1/dtiam_same_data_compatible_v1"
BM_MULTI = BASE / "biomaster_odti_routed_ranker_v1/multiseed_evaluation_v1"
BM_STACK = BASE / "biomaster_odti_routed_ranker_v1/cold_regime_stack_v1"
BASELINES = BASE / "baseline_results_v1"
PAIR_META = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
SCOPE = (
    BASE / "public_baselines_v1/scope_dti_old_drug_entity_cold_v1"
    / "SCOPE_DTI_OLD_DRUG_ENTITY_COLD_PREDICTIONS_V1.csv.gz"
)
OUT = BASE / "public_retrained_v1/dtiam_same_data_comparison_v1"
PROTOCOL_FOLDS = {
    "S1_SCAFFOLD_COLD_DRUG": list(range(5)),
    "S2_HOMOLOGY_COLD_TARGET": list(range(5)),
    "S3_STRICT_DOUBLE_COLD": list(range(5)),
    "S4_FIRST_SEEN_TEMPORAL_2023_2025": [-1],
    "S5_OLD_DRUG_ENTITY_COLD": [-1],
}


def load_dtiam_fold(protocol: str, fold: int) -> pd.DataFrame | None:
    run = DTIAM / f"{protocol}__fold_{fold}__OFFICIAL_DEFAULT_COMPAT_V1"
    summary_path = run / "RUN_SUMMARY_V1.json"
    prediction_path = run / "TEST_PREDICTIONS_V1.csv.gz"
    if not summary_path.is_file() or not prediction_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS":
        return None
    frame = pd.read_csv(prediction_path)
    frame["fold"] = fold
    return frame


def load_biomaster(protocol: str, completed: list[int]) -> tuple[pd.DataFrame, str]:
    """Load the strongest frozen BioMaster comparator without refitting it.

    Five-seed ensembles exist for S2/S3/S5. S1/S4 predate that expansion, so
    their audited seed-20260813 validation-only routed stacks are retained and
    labelled explicitly as single-seed evidence.
    """
    multiseed_path = BM_MULTI / f"{protocol}_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz"
    if multiseed_path.is_file():
        usecols = [
            "calibration_pair_id", "BIOMASTER_RAW_FIVE_SEED_MEAN",
            "BIOMASTER_STACK_FIVE_SEED_MEAN", "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        ] + (["scope_mean"] if protocol == "S5_OLD_DRUG_ENTITY_COLD" else [])
        frame = pd.read_csv(multiseed_path, usecols=usecols).rename(columns={
            "BIOMASTER_RAW_FIVE_SEED_MEAN": "biomaster_raw_score",
            "BIOMASTER_STACK_FIVE_SEED_MEAN": "biomaster_stack_score",
        })
        return frame, "FIVE_SEED_ENSEMBLE"

    parts = []
    for fold in completed:
        stack_path = BM_STACK / (
            f"{protocol}__fold_{fold}__seed_20260813__CORE"
            "/COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
        )
        baseline_fold = f"fold_{fold}" if fold >= 0 else "fixed_split"
        baseline_path = BASELINES / protocol / baseline_fold / "BASELINE_TEST_PREDICTIONS_V1.csv.gz"
        if not stack_path.is_file() or not baseline_path.is_file():
            raise FileNotFoundError([stack_path, baseline_path])
        stack = pd.read_csv(stack_path, usecols=[
            "calibration_pair_id", "BIOMASTER_VALIDATION_LOGISTIC", "COLD_REGIME_ROUTED_STACK",
        ])
        baseline = pd.read_csv(baseline_path, usecols=[
            "calibration_pair_id", "CONPLEX_FROZEN_EXTERNAL",
            "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        ])
        parts.append(stack.merge(baseline, on="calibration_pair_id", how="inner", validate="one_to_one"))
    frame = pd.concat(parts, ignore_index=True).rename(columns={
        "BIOMASTER_VALIDATION_LOGISTIC": "biomaster_raw_score",
        "COLD_REGIME_ROUTED_STACK": "biomaster_stack_score",
    })
    if not frame["calibration_pair_id"].is_unique:
        raise RuntimeError(f"BioMaster comparator pair overlap for {protocol}")
    return frame, "SINGLE_SEED_20260813"


def load_protocol(
    protocol: str,
    folds: list[int],
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int], str]:
    parts = []
    completed = []
    for fold in folds:
        part = load_dtiam_fold(protocol, fold)
        if part is not None:
            parts.append(part)
            completed.append(fold)
    if not parts:
        raise FileNotFoundError(f"No passed DTIAM runs for {protocol}")
    dtiam = pd.concat(parts, ignore_index=True)
    if not dtiam["calibration_pair_id"].is_unique:
        raise RuntimeError(f"DTIAM test pair overlap for {protocol}")

    bm, biomaster_evidence = load_biomaster(protocol, completed)
    frame = dtiam.merge(bm, on="calibration_pair_id", how="inner", validate="one_to_one")
    frame = frame.merge(meta, on="calibration_pair_id", how="left", validate="one_to_one")
    if len(frame) != len(dtiam):
        raise RuntimeError(f"BioMaster/DTIAM coverage mismatch for {protocol}: {len(frame)} != {len(dtiam)}")
    if frame[["target_homology_cluster", "scaffold_group"]].isna().any().any():
        raise RuntimeError(f"Missing cluster metadata for {protocol}")
    return frame, completed, biomaster_evidence


def metric_rows(
    frame: pd.DataFrame,
    protocol: str,
    completed: list[int],
    biomaster_evidence: str,
) -> list[dict[str, object]]:
    models = {
        "DTIAM_OFFICIAL_REPRESENTATION_COMPAT_RETRAIN": "dtiam_probability",
        f"BIOMASTER_RAW_{biomaster_evidence}": "biomaster_raw_score",
        f"BIOMASTER_STACK_{biomaster_evidence}": "biomaster_stack_score",
        "CONPLEX_FROZEN_EXTERNAL": "CONPLEX_FROZEN_EXTERNAL",
        "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO": "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
    }
    if "scope_mean" in frame:
        models["SCOPE_PUBLIC_CHECKPOINT_ENSEMBLE"] = "scope_mean"
    rows = []
    for model, column in models.items():
        row: dict[str, object] = {
            "protocol": protocol,
            "evaluation": "FIXED_TEST" if completed == [-1] else "POOLED_COMPLETED_OOF",
            "completed_folds": ";".join(map(str, completed)),
            "model": model,
            "score_column": column,
        }
        row.update(metrics(frame, frame[column].to_numpy(dtype=np.float64)))
        rows.append(row)
    return rows


def bootstrap_rows(
    frame: pd.DataFrame,
    protocol: str,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    if protocol == "S1_SCAFFOLD_COLD_DRUG":
        cluster_columns = ["scaffold_group"]
    elif protocol == "S2_HOMOLOGY_COLD_TARGET":
        cluster_columns = ["target_homology_cluster"]
    elif protocol == "S3_STRICT_DOUBLE_COLD":
        # Report both axes; a robust strict-double claim should agree under both.
        cluster_columns = ["target_homology_cluster", "scaffold_group"]
    elif protocol == "S4_FIRST_SEEN_TEMPORAL_2023_2025":
        # Temporal test is fixed; report dependence sensitivity on both axes.
        cluster_columns = ["target_homology_cluster", "scaffold_group"]
    else:
        cluster_columns = ["parent_standard_inchi_key"]
    comparisons = [
        ("biomaster_stack_score", "dtiam_probability"),
        ("dtiam_probability", "CONPLEX_FROZEN_EXTERNAL"),
        ("dtiam_probability", "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"),
    ]
    if "scope_mean" in frame:
        comparisons.append(("dtiam_probability", "scope_mean"))
    rows = []
    for cluster_column in cluster_columns:
        for challenger, reference in comparisons:
            for metric_name in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    frame,
                    challenger,
                    reference,
                    metric_name,
                    iterations,
                    seed + len(rows),
                    cluster_column,
                )
                row["protocol"] = protocol
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=list(PROTOCOL_FOLDS) + ["all"], default="all")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    meta = pd.read_csv(
        PAIR_META,
        usecols=["calibration_pair_id", "target_homology_cluster", "scaffold_group"],
        low_memory=False,
    )
    protocols = list(PROTOCOL_FOLDS) if args.protocol == "all" else [args.protocol]
    all_metrics = []
    all_bootstrap = []
    coverage = []
    OUT.mkdir(parents=True, exist_ok=True)
    for protocol in protocols:
        try:
            frame, completed, biomaster_evidence = load_protocol(
                protocol, PROTOCOL_FOLDS[protocol], meta
            )
        except FileNotFoundError:
            if args.protocol == "all":
                continue
            raise
        all_metrics.extend(metric_rows(frame, protocol, completed, biomaster_evidence))
        all_bootstrap.extend(bootstrap_rows(frame, protocol, args.iterations, args.seed + len(all_bootstrap)))
        expected = PROTOCOL_FOLDS[protocol]
        coverage.append({
            "protocol": protocol,
            "completed_folds": ";".join(map(str, completed)),
            "expected_folds": ";".join(map(str, expected)),
            "complete": completed == expected,
            "biomaster_evidence": biomaster_evidence,
            "rows": len(frame),
            "unique_pairs": frame["calibration_pair_id"].nunique(),
        })
        frame.to_csv(OUT / f"{protocol}_ALIGNED_PREDICTIONS_V1.csv.gz", index=False)
    if not all_metrics:
        raise RuntimeError("No passed DTIAM run available for comparison")

    metric_frame = pd.DataFrame(all_metrics)
    bootstrap_frame = pd.DataFrame(all_bootstrap)
    coverage_frame = pd.DataFrame(coverage)
    metric_path = OUT / "DTIAM_SAME_DATA_COMPARISON_METRICS_V1.csv"
    bootstrap_path = OUT / "DTIAM_SAME_DATA_PAIRED_BOOTSTRAP_V1.csv"
    coverage_path = OUT / "DTIAM_SAME_DATA_COMPARISON_COVERAGE_V1.csv"
    metric_frame.to_csv(metric_path, index=False)
    bootstrap_frame.to_csv(bootstrap_path, index=False)
    coverage_frame.to_csv(coverage_path, index=False)

    primary = bootstrap_frame[
        bootstrap_frame["challenger"].eq("biomaster_stack_score")
        & bootstrap_frame["reference"].eq("dtiam_probability")
        & bootstrap_frame["metric"].eq("auprc")
    ].to_dict("records")
    checks = {
        "all_aligned_pair_sets_unique": bool((coverage_frame["rows"] == coverage_frame["unique_pairs"]).all()),
        "all_core_metrics_finite": bool(np.isfinite(metric_frame[["micro_auroc", "micro_auprc"]]).all().all()),
        "all_bootstrap_ci_finite": bool(np.isfinite(
            bootstrap_frame[["difference_ci95_low", "difference_ci95_high"]]
        ).all().all()),
        "at_least_one_protocol_compared": len(coverage_frame) >= 1,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "coverage": coverage_frame.to_dict("records"),
        "primary_biomaster_stack_minus_dtiam_auprc": primary,
        "bootstrap_policy": "S1 drug scaffold; S2 target-homology cluster; S3 and S4 reported independently by target-homology cluster and drug scaffold; S5 old-drug entity.",
        "comparison_boundary": "DTIAM uses official BerMol/ESM2 representations and same frozen supervised data/splits, but AutoGluon 1.4 compatibility environment rather than paper 0.5.2.",
        "claim_policy": "Incomplete OOF coverage is diagnostic only. Full SOTA evidence requires all expected folds and source-held external validation.",
        "artifacts": {
            "metrics_sha256": sha256(metric_path),
            "bootstrap_sha256": sha256(bootstrap_path),
            "coverage_sha256": sha256(coverage_path),
        },
    }
    (OUT / "DTIAM_SAME_DATA_COMPARISON_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if len(protocols) == 1:
        protocol = protocols[0]
        metric_frame.to_csv(
            OUT / f"{protocol}_DTIAM_SAME_DATA_COMPARISON_METRICS_V1.csv", index=False
        )
        bootstrap_frame.to_csv(
            OUT / f"{protocol}_DTIAM_SAME_DATA_PAIRED_BOOTSTRAP_V1.csv", index=False
        )
        coverage_frame.to_csv(
            OUT / f"{protocol}_DTIAM_SAME_DATA_COMPARISON_COVERAGE_V1.csv", index=False
        )
        (OUT / f"{protocol}_DTIAM_SAME_DATA_COMPARISON_SUMMARY_V1.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

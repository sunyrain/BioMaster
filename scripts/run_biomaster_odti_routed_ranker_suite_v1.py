#!/usr/bin/env python3
"""Run and aggregate the BioMaster-ODTI routed-ranker experiment suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/train_biomaster_odti_routed_ranker_v1.py"
OUT = ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_routed_ranker_v1"
BASELINES = ROOT / "outputs/old_drug_target_sota_v1/baseline_results_v1/ALL_BASELINE_METRICS_V1.csv"


def jobs(suite: str) -> list[tuple[str, int]]:
    if suite == "pilot":
        return [
            ("S1_SCAFFOLD_COLD_DRUG", 0),
            ("S2_HOMOLOGY_COLD_TARGET", 0),
            ("S3_STRICT_DOUBLE_COLD", 0),
            ("S5_OLD_DRUG_ENTITY_COLD", -1),
        ]
    if suite == "strict_all_folds":
        result = []
        for protocol in [
            "S1_SCAFFOLD_COLD_DRUG",
            "S2_HOMOLOGY_COLD_TARGET",
            "S3_STRICT_DOUBLE_COLD",
        ]:
            result.extend((protocol, fold) for fold in range(5))
        result.extend([
            ("S4_FIRST_SEEN_TEMPORAL_2023_2025", -1),
            ("S5_OLD_DRUG_ENTITY_COLD", -1),
        ])
        return result
    if suite == "cold_confirm":
        return [
            (protocol, fold)
            for protocol in ["S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"]
            for fold in range(5)
        ]
    if suite == "old_drug_confirm":
        return [("S5_OLD_DRUG_ENTITY_COLD", -1)]
    raise ValueError(suite)


def summary_path(protocol: str, fold: int, seed: int, variant: str) -> Path:
    name = f"{protocol}__fold_{fold}__seed_{seed}__{variant.upper()}"
    return OUT / name / "RUN_SUMMARY_V1.json"


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(OUT.glob("*/RUN_SUMMARY_V1.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "PASS":
            continue
        row = {
            "model": result["model"],
            "variant": result["variant"],
            "protocol": result["protocol"],
            "fold": result["fold"],
            "seed": result["seed"],
            "best_epoch": result["training"]["best_epoch"],
            "best_validation_micro_auprc": result["training"]["best_validation_micro_auprc"],
        }
        row.update(result["test_metrics"])
        row["summary_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, frame
    frame = frame.sort_values(["protocol", "variant", "fold", "seed"])
    metric_columns = [
        "micro_auroc", "micro_auprc", "target_macro_auroc", "target_macro_auprc",
        "drug_macro_auroc", "drug_macro_auprc", "brier", "ece_15",
    ]
    grouped = frame.groupby(["protocol", "variant"], dropna=False)[metric_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().add_suffix("_std")
    count = grouped.size().rename("runs")
    aggregate_frame = pd.concat([count, mean, std], axis=1).reset_index()
    return frame, aggregate_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["pilot", "cold_confirm", "old_drug_confirm", "strict_all_folds"], default="pilot")
    parser.add_argument("--seeds", default="20260813")
    parser.add_argument("--variants", default="core,conplex_augmented")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    variants = [value for value in args.variants.split(",") if value]
    OUT.mkdir(parents=True, exist_ok=True)
    requested = completed = skipped = 0
    for protocol, fold in jobs(args.suite):
        for seed in seeds:
            for variant in variants:
                requested += 1
                path = summary_path(protocol, fold, seed, variant)
                if path.is_file() and not args.force:
                    existing = json.loads(path.read_text())
                    if existing.get("status") == "PASS":
                        skipped += 1
                        print(json.dumps({"skip_passed": str(path.relative_to(ROOT))}), flush=True)
                        continue
                command = [
                    sys.executable, str(TRAIN), "--protocol", protocol,
                    "--fold", str(0 if fold < 0 else fold), "--seed", str(seed),
                    "--variant", variant, "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
                completed += 1
    runs, aggregate_frame = aggregate()
    run_path = OUT / "ALL_ROUTED_RANKER_RUNS_V1.csv"
    aggregate_path = OUT / "ROUTED_RANKER_AGGREGATE_V1.csv"
    runs.to_csv(run_path, index=False)
    aggregate_frame.to_csv(aggregate_path, index=False)

    baseline_context = []
    if BASELINES.is_file():
        baseline = pd.read_csv(BASELINES)
        baseline_table = (
            baseline[baseline["model"].isin(["CONPLEX_FROZEN_EXTERNAL", "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"])]
            .groupby(["protocol", "model"])[["micro_auroc", "micro_auprc"]]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        baseline_table.columns = [
            "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
            for column in baseline_table.columns
        ]
        baseline_context = baseline_table.to_dict("records")
    checks = {
        "all_requested_jobs_accounted_for": requested == completed + skipped,
        "all_collected_runs_pass": not runs.empty,
        "aggregate_written": aggregate_path.is_file(),
        "baseline_context_available": bool(baseline_context),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "suite": args.suite,
        "seeds": seeds,
        "variants": variants,
        "requested_jobs": requested,
        "completed_jobs": completed,
        "skipped_existing_passed_jobs": skipped,
        "total_collected_passed_runs": int(len(runs)),
        "checks": {key: bool(value) for key, value in checks.items()},
        "baseline_context": baseline_context,
        "claim_status": "NO_SOTA_CLAIM_UNTIL_FIVE_SEEDS_PUBLIC_BASELINES_AND_PAIRED_BOOTSTRAP_COMPLETE",
    }
    (OUT / "ROUTED_RANKER_SUITE_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

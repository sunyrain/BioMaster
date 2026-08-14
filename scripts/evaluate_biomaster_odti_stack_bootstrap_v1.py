#!/usr/bin/env python3
"""Paired group bootstrap for the validation-only cold-regime stack."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import (  # noqa: E402
    paired_cluster_bootstrap,
    sha256,
)


BASE = ROOT / "outputs/old_drug_target_sota_v1"
STACK = BASE / "biomaster_odti_routed_ranker_v1/cold_regime_stack_v1"
PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
OUT = STACK / "paired_bootstrap_v1"
MODEL_SCORE = "COLD_REGIME_ROUTED_STACK"
REFERENCES = ["conplex_score", "train_positive_max_tanimoto"]
ITERATIONS = 1000
SEED = 20260813


def load(protocol: str, fold: int) -> pd.DataFrame:
    run_name = f"{protocol}__fold_{fold}__seed_20260813__CORE"
    path = STACK / run_name / "COLD_REGIME_STACK_TEST_PREDICTIONS_V1.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    meta = pd.read_csv(
        PAIRS,
        usecols=["calibration_pair_id", "target_homology_cluster", "scaffold_group"],
        low_memory=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []

    # Five-fold protocols: fold test sets are disjoint, so a pooled OOF test is
    # valid after exact-pair uniqueness is checked.
    for protocol in ["S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD"]:
        folds = []
        for fold in range(5):
            frame = load(protocol, fold).merge(meta, on="calibration_pair_id", validate="one_to_one")
            folds.append(frame)
            coverage.append({
                "protocol": protocol, "evaluation": f"FOLD_{fold}", "rows": len(frame),
                "unique_pairs": frame["calibration_pair_id"].nunique(),
                "bootstrap_groups": frame["target_homology_cluster"].nunique(),
                "bootstrap_group_column": "target_homology_cluster",
            })
            for reference in REFERENCES:
                for metric in ["auprc", "auroc"]:
                    row = paired_cluster_bootstrap(
                        frame, MODEL_SCORE, reference, metric, ITERATIONS,
                        SEED + fold * 100 + len(results), "target_homology_cluster",
                    )
                    row.update({"protocol": protocol, "evaluation": f"FOLD_{fold}", "fold": fold})
                    results.append(row)
        pooled = pd.concat(folds, ignore_index=True)
        if not pooled["calibration_pair_id"].is_unique:
            raise RuntimeError(f"Overlapping OOF pairs for {protocol}")
        coverage.append({
            "protocol": protocol, "evaluation": "POOLED_OOF", "rows": len(pooled),
            "unique_pairs": pooled["calibration_pair_id"].nunique(),
            "bootstrap_groups": pooled["target_homology_cluster"].nunique(),
            "bootstrap_group_column": "target_homology_cluster",
        })
        for reference in REFERENCES:
            for metric in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    pooled, MODEL_SCORE, reference, metric, ITERATIONS,
                    SEED + 10000 + len(results), "target_homology_cluster",
                )
                row.update({"protocol": protocol, "evaluation": "POOLED_OOF", "fold": -1})
                results.append(row)

    # Fixed old-drug entity-cold application test: resample complete drug
    # entities to preserve within-drug target-ranking dependence.
    protocol = "S5_OLD_DRUG_ENTITY_COLD"
    old = load(protocol, -1).merge(meta, on="calibration_pair_id", validate="one_to_one")
    coverage.append({
        "protocol": protocol, "evaluation": "FIXED_TEST", "rows": len(old),
        "unique_pairs": old["calibration_pair_id"].nunique(),
        "bootstrap_groups": old["parent_standard_inchi_key"].nunique(),
        "bootstrap_group_column": "parent_standard_inchi_key",
    })
    for reference in REFERENCES:
        for metric in ["auprc", "auroc"]:
            row = paired_cluster_bootstrap(
                old, MODEL_SCORE, reference, metric, ITERATIONS,
                SEED + 20000 + len(results), "parent_standard_inchi_key",
            )
            row.update({"protocol": protocol, "evaluation": "FIXED_TEST", "fold": -1})
            results.append(row)

    frame = pd.DataFrame(results)
    coverage_frame = pd.DataFrame(coverage)
    result_path = OUT / "COLD_REGIME_STACK_PAIRED_BOOTSTRAP_RESULTS_V1.csv"
    coverage_path = OUT / "COLD_REGIME_STACK_BOOTSTRAP_COVERAGE_V1.csv"
    frame.to_csv(result_path, index=False)
    coverage_frame.to_csv(coverage_path, index=False)

    def primary(protocol: str, evaluation: str, reference: str) -> dict[str, object]:
        selected = frame[
            frame["protocol"].eq(protocol)
            & frame["evaluation"].eq(evaluation)
            & frame["reference"].eq(reference)
            & frame["metric"].eq("auprc")
        ]
        if len(selected) != 1:
            raise RuntimeError((protocol, evaluation, reference, len(selected)))
        return selected.iloc[0].to_dict()

    s2 = primary("S2_HOMOLOGY_COLD_TARGET", "POOLED_OOF", "conplex_score")
    s3 = primary("S3_STRICT_DOUBLE_COLD", "POOLED_OOF", "conplex_score")
    s5 = primary("S5_OLD_DRUG_ENTITY_COLD", "FIXED_TEST", "train_positive_max_tanimoto")
    checks = {
        "all_evaluation_pair_sets_unique": (coverage_frame["rows"] == coverage_frame["unique_pairs"]).all(),
        "all_bootstrap_intervals_finite": np.isfinite(frame[["difference_ci95_low", "difference_ci95_high"]]).all().all(),
        "s2_auprc_ci_excludes_zero_vs_conplex": bool(s2["ci95_excludes_zero_in_favor_of_challenger"]),
        "s3_auprc_ci_excludes_zero_vs_conplex": bool(s3["ci95_excludes_zero_in_favor_of_challenger"]),
        "old_drug_auprc_observed_above_similarity": float(s5["observed_difference"]) > 0,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "iterations": ITERATIONS,
        "checks": {key: bool(value) for key, value in checks.items()},
        "s2_pooled_oof_vs_conplex_auprc": s2,
        "s3_pooled_oof_vs_conplex_auprc": s3,
        "s5_fixed_old_drug_vs_similarity_auprc": s5,
        "artifacts": {
            "results_sha256": sha256(result_path),
            "coverage_sha256": sha256(coverage_path),
        },
        "claim_status": "SINGLE_MODEL_SEED_INTERNAL_RESULT; MULTI_SEED_AND_PUBLIC_COMPARATORS_REQUIRED",
    }
    summary_path = OUT / "COLD_REGIME_STACK_PAIRED_BOOTSTRAP_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

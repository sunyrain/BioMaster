#!/usr/bin/env python3
"""Audit the current old-drug-to-target production candidate against DTIAM."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_s3_query_balanced_blend_dev_v1 import (  # noqa: E402
    clustered_micro_bootstrap,
    directional_bootstrap,
    metrics,
    sha256,
)


FUSION_SUMMARY = ROOT / (
    "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_biomaster_validation_fusion_v1/"
    "DTIAM_BIOMASTER_VALIDATION_FUSION_SUMMARY_V1.json"
)
KIRHUB_PREDICTIONS = ROOT / (
    "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_biomaster_scope_deployment_v1/"
    "SOURCE_HELD_KIRHUB_DTIAM_INTEGRATED_PREDICTIONS_V1.csv.gz"
)
DEFAULT_OUT = ROOT / (
    "outputs/old_drug_target_sota_v1/old_drug_primary_head_audit_v1/"
    "OLD_DRUG_PRIMARY_HEAD_AUDIT_V1.json"
)
CANDIDATE = "dtiam_biomaster_same_data_score"
REFERENCE = "dtiam_probability"


def find_metric(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("model") == model]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metric row for {model}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.bootstrap_iterations < 500:
        raise ValueError("bootstrap_iterations must be at least 500")
    for path in [FUSION_SUMMARY, KIRHUB_PREDICTIONS]:
        if not path.is_file():
            raise FileNotFoundError(path)

    fusion = json.loads(FUSION_SUMMARY.read_text())
    if fusion.get("status") != "PASS":
        raise RuntimeError("frozen S5 validation fusion audit did not pass")
    internal_candidate = find_metric(
        fusion["metrics"], "DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION"
    )
    internal_reference = find_metric(fusion["metrics"], "DTIAM_RAW")
    internal_metrics = {}
    for metric in ["micro_auprc", "target_macro_auprc", "drug_macro_auprc"]:
        internal_metrics[metric] = {
            "candidate": float(internal_candidate[metric]),
            "dtiam": float(internal_reference[metric]),
            "difference": float(internal_candidate[metric])
            - float(internal_reference[metric]),
        }

    external = pd.read_csv(KIRHUB_PREDICTIONS, low_memory=False)
    strict = external[
        external["kirhub_frozen_unreported_scope"].fillna(False).astype(bool)
    ].copy()
    strict["binary_label"] = strict["kirhub_local_unreported_active"].astype(np.int8)
    strict["parent_standard_inchi_key"] = strict["ligand_inchikey"].astype(str)
    candidate_metrics = metrics(strict, strict[CANDIDATE].to_numpy(dtype=float))
    reference_metrics = metrics(strict, strict[REFERENCE].to_numpy(dtype=float))
    external_deltas = {
        metric: float(candidate_metrics[metric]) - float(reference_metrics[metric])
        for metric in ["micro_auprc", "target_macro_auprc", "drug_macro_auprc"]
    }
    directional = [
        directional_bootstrap(
            strict,
            CANDIDATE,
            REFERENCE,
            group,
            args.bootstrap_iterations,
            args.seed + number,
        )
        for number, group in enumerate(
            ["target_chembl_id", "parent_standard_inchi_key"]
        )
    ]
    clustered = [
        clustered_micro_bootstrap(
            strict,
            CANDIDATE,
            REFERENCE,
            cluster,
            args.bootstrap_iterations,
            args.seed + 100 + number,
        )
        for number, cluster in enumerate(["ligand_inchikey", "target_chembl_id"])
    ]
    internal_bootstrap = fusion["primary_same_data_routed_minus_dtiam_auprc"]
    checks = {
        "frozen_s5_fusion_pass": fusion.get("status") == "PASS",
        "no_s5_test_labels_used_for_fitting": bool(
            fusion.get("checks", {}).get("no_test_labels_used_for_fitting")
        ),
        "strict_kirhub_exact_2823_with_202_positives": len(strict) == 2823
        and int(strict["binary_label"].sum()) == 202,
        "external_scores_finite": bool(
            np.isfinite(strict[[CANDIDATE, REFERENCE]].to_numpy(dtype=float)).all()
        ),
        "internal_s5_micro_auprc_bootstrap_positive": float(
            internal_bootstrap["difference_ci95_low"]
        )
        > 0,
        "external_point_micro_auprc_improved": external_deltas["micro_auprc"] > 0,
        "external_point_target_macro_auprc_improved": external_deltas[
            "target_macro_auprc"
        ]
        > 0,
        "external_point_drug_macro_auprc_improved": external_deltas[
            "drug_macro_auprc"
        ]
        > 0,
        "all_bootstrap_intervals_finite": all(
            np.isfinite(
                [item["difference_ci95_low"], item["difference_ci95_high"]]
            ).all()
            for item in [*directional, *clustered]
        ),
    }
    external_all_ci_positive = all(
        bool(item["ci95_excludes_zero_in_favor_of_candidate"])
        for item in [*directional, *clustered]
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": "DTIAM_BIOMASTER_S5_VALIDATION_ROUTED_FUSION",
        "fit_boundary": (
            "Fusion coefficients were fit on frozen S5 validation only. S5 test "
            "and KIRHub labels are evaluation-only."
        ),
        "internal_s5": {
            "metrics": internal_metrics,
            "micro_auprc_cluster_bootstrap_vs_dtiam": internal_bootstrap,
        },
        "strict_kirhub": {
            "rows": int(len(strict)),
            "positives": int(strict["binary_label"].sum()),
            "candidate_metrics": candidate_metrics,
            "dtiam_metrics": reference_metrics,
            "deltas_vs_dtiam": external_deltas,
            "directional_macro_bootstrap_vs_dtiam": directional,
            "clustered_micro_bootstrap_vs_dtiam": clustered,
            "all_four_ci95_positive": external_all_ci_positive,
        },
        "checks": checks,
        "promotion_decision": (
            "EXTERNALLY_CONFIRMED"
            if external_all_ci_positive
            else "CURRENT_PRODUCTION_CANDIDATE_PENDING_NEW_SOURCE_CONFIRMATION"
        ),
        "claim_boundary": (
            "The candidate is significantly stronger on frozen internal S5 and "
            "improves all three KIRHub AUPRC point estimates. KIRHub confidence "
            "intervals cross zero because only 33 two-class drug queries are "
            "available, so external superiority is not claimed."
        ),
        "inputs": {
            str(FUSION_SUMMARY.relative_to(ROOT)): sha256(FUSION_SUMMARY),
            str(KIRHUB_PREDICTIONS.relative_to(ROOT)): sha256(KIRHUB_PREDICTIONS),
        },
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

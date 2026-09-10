#!/usr/bin/env python3
"""Validate a drug-direction-preserving BioMaster/DTIAM calibration layer.

The method keeps DTIAM's within-drug target order exactly and changes only the
constant offset assigned to each drug query.  The offset combines label-free
drug-level means from DTIAM and BioMaster.  Consequently drug-macro ranking is
an exact non-inferiority invariant, while cross-drug micro and target-query
ranking can improve.

Coefficients are selected only on frozen folds 0--2.  Folds 3--4 are scored
once with locked coefficients and are used only for confirmation metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_s3_query_balanced_blend_dev_v1 import (  # noqa: E402
    clustered_micro_bootstrap,
    directional_bootstrap,
    fast_macro_ap,
    group_indices,
    metrics,
    sha256,
)


SOURCE = ROOT / (
    "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_biomaster_cold_fusion_v1"
)
DEFAULT_OUT = ROOT / (
    "outputs/old_drug_target_sota_v1/"
    "direction_preserving_calibration_v1"
)
PROTOCOLS = (
    "S2_HOMOLOGY_COLD_TARGET",
    "S3_STRICT_DOUBLE_COLD",
)
DEVELOPMENT_FOLDS = (0, 1, 2)
CONFIRMATION_FOLDS = (3, 4)
TEMPORAL_PROTOCOL = "S4_FIRST_SEEN_TEMPORAL_2023_2025"
TEMPORAL_SOURCE = SOURCE / f"{TEMPORAL_PROTOCOL}_POOLED_PREDICTIONS_V1.csv.gz"
DEPLOYMENT_DTIAM = SOURCE.parent / (
    "dtiam_720x384_deployment_v1/DTIAM_720X384_SCORES_V1.csv.gz"
)
DEPLOYMENT_BIOMASTER = ROOT / (
    "outputs/old_drug_target_sota_v1/biomaster_odti_deployment_v1/"
    "BIOMASTER_ODTI_720X384_SCORES_V1.csv.gz"
)
KIRHUB = ROOT / (
    "outputs/evidence_routing_compute_execution_20260808_v1/"
    "leakage_safe_ranker_v10/external_evaluation/"
    "OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_V10.csv"
)
DTIAM = "dtiam_probability"
BIOMASTER = "biomaster_stack_score"
CANDIDATE = "direction_preserving_score"
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 5, "mtime": 0}


def _logit_standardize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    logits = np.log(values / (1.0 - values))
    scale = float(logits.std())
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("score vector has no finite variation")
    return (logits - float(logits.mean())) / scale


def direction_preserving_score(
    frame: pd.DataFrame,
    dtiam_bias_weight: float,
    biomaster_bias_weight: float,
) -> np.ndarray:
    """Return a score with DTIAM order preserved within every drug query."""

    required = {DTIAM, BIOMASTER, "parent_standard_inchi_key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")
    dtiam = _logit_standardize(frame[DTIAM].to_numpy(dtype=np.float64))
    biomaster = _logit_standardize(frame[BIOMASTER].to_numpy(dtype=np.float64))
    drug = frame["parent_standard_inchi_key"].astype(str).to_numpy()
    dtiam_mean = pd.Series(dtiam).groupby(drug, sort=False).transform("mean").to_numpy()
    biomaster_mean = (
        pd.Series(biomaster).groupby(drug, sort=False).transform("mean").to_numpy()
    )
    return (
        dtiam
        - dtiam_mean
        + float(dtiam_bias_weight) * dtiam_mean
        + float(biomaster_bias_weight) * biomaster_mean
    )


def drug_rank_invariant(frame: pd.DataFrame, candidate: np.ndarray) -> bool:
    """Check exact equality of DTIAM and candidate ranks within each drug."""

    work = frame[["parent_standard_inchi_key", DTIAM]].copy()
    work[CANDIDATE] = np.asarray(candidate, dtype=np.float64)
    reference_rank = work.groupby("parent_standard_inchi_key", sort=False)[DTIAM].rank(
        method="average", ascending=False
    )
    candidate_rank = work.groupby("parent_standard_inchi_key", sort=False)[CANDIDATE].rank(
        method="average", ascending=False
    )
    return bool(np.array_equal(reference_rank.to_numpy(), candidate_rank.to_numpy()))


def objective(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    micro = float(average_precision_score(labels, score))
    target = fast_macro_ap(
        labels, score, group_indices(frame["target_chembl_id"].to_numpy())
    )[0]
    drug = fast_macro_ap(
        labels, score, group_indices(frame["parent_standard_inchi_key"].to_numpy())
    )[0]
    return {
        "micro_auprc": micro,
        "target_macro_auprc": float(target),
        "drug_macro_auprc": float(drug),
        "equal_weight_objective": float((micro + target + drug) / 3.0),
    }


def tune(frame: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Coarse-to-fine development-only coefficient search."""

    records: list[dict[str, float | str]] = []

    def evaluate(stage: str, left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
        stage_rows = []
        for dtiam_weight, biomaster_weight in product(left, right):
            score = direction_preserving_score(frame, dtiam_weight, biomaster_weight)
            row = {
                "stage": stage,
                "dtiam_bias_weight": float(dtiam_weight),
                "biomaster_bias_weight": float(biomaster_weight),
                **objective(frame, score),
            }
            records.append(row)
            stage_rows.append(row)
        best = sorted(
            stage_rows,
            key=lambda row: (
                -float(row["equal_weight_objective"]),
                abs(float(row["dtiam_bias_weight"]))
                + abs(float(row["biomaster_bias_weight"])),
                float(row["dtiam_bias_weight"]),
                float(row["biomaster_bias_weight"]),
            ),
        )[0]
        return float(best["dtiam_bias_weight"]), float(best["biomaster_bias_weight"])

    coarse = np.arange(-1.0, 5.0001, 1.0)
    coarse_dtiam, coarse_biomaster = evaluate("coarse", coarse, coarse)
    fine_dtiam = np.arange(coarse_dtiam - 1.0, coarse_dtiam + 1.0001, 0.25)
    fine_biomaster = np.arange(
        coarse_biomaster - 1.0, coarse_biomaster + 1.0001, 0.25
    )
    locked_dtiam, locked_biomaster = evaluate("fine", fine_dtiam, fine_biomaster)
    return locked_dtiam, locked_biomaster, pd.DataFrame(records)


def source_path(protocol: str, fold: int) -> Path:
    return SOURCE / (
        f"{protocol}__fold_{fold}/"
        "DTIAM_BIOMASTER_COLD_FUSION_PREDICTIONS_V1.csv.gz"
    )


def load_scope(protocol: str, folds: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = []
    hashes = {}
    for fold in folds:
        path = source_path(protocol, fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        part = pd.read_csv(path, low_memory=False)
        part["fold"] = fold
        parts.append(part)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    frame = pd.concat(parts, ignore_index=True)
    if frame["calibration_pair_id"].duplicated().any():
        raise RuntimeError("scope contains duplicate pair identifiers")
    return frame, hashes


def evaluate_protocol(
    protocol: str,
    out_dir: Path,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    development, development_hashes = load_scope(protocol, DEVELOPMENT_FOLDS)
    confirmation, confirmation_hashes = load_scope(protocol, CONFIRMATION_FOLDS)
    if set(development["calibration_pair_id"]) & set(confirmation["calibration_pair_id"]):
        raise RuntimeError("development and confirmation pairs overlap")

    dtiam_weight, biomaster_weight, tuning = tune(development)
    development[CANDIDATE] = direction_preserving_score(
        development, dtiam_weight, biomaster_weight
    )
    # Coefficients are already locked.  This computation accesses prediction
    # columns and drug identifiers only; confirmation labels are not inputs.
    confirmation[CANDIDATE] = direction_preserving_score(
        confirmation, dtiam_weight, biomaster_weight
    )

    model_metrics = {
        "DTIAM": metrics(confirmation, confirmation[DTIAM].to_numpy(dtype=float)),
        "BIOMASTER_STACK": metrics(
            confirmation, confirmation[BIOMASTER].to_numpy(dtype=float)
        ),
        "DIRECTION_PRESERVING": metrics(
            confirmation, confirmation[CANDIDATE].to_numpy(dtype=float)
        ),
    }
    candidate = model_metrics["DIRECTION_PRESERVING"]
    reference = model_metrics["DTIAM"]
    rank_invariant = drug_rank_invariant(
        confirmation, confirmation[CANDIDATE].to_numpy(dtype=float)
    )
    drug_metric_delta = float(candidate["drug_macro_auprc"]) - float(
        reference["drug_macro_auprc"]
    )

    directional = [
        directional_bootstrap(
            confirmation,
            CANDIDATE,
            DTIAM,
            group,
            bootstrap_iterations,
            seed + number,
        )
        for number, group in enumerate(
            ["target_chembl_id", "parent_standard_inchi_key"]
        )
    ]
    clustered = [
        clustered_micro_bootstrap(
            confirmation,
            CANDIDATE,
            DTIAM,
            cluster,
            bootstrap_iterations,
            seed + 100 + number,
        )
        for number, cluster in enumerate(["target_homology_cluster", "scaffold_group"])
    ]

    token = protocol.split("_", 1)[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = out_dir / f"{token}_DEVELOPMENT_TUNING_V1.csv"
    development_path = out_dir / f"{token}_DEVELOPMENT_SCORES_V1.csv.gz"
    confirmation_path = out_dir / f"{token}_CONFIRMATION_SCORES_V1.csv.gz"
    tuning.to_csv(tuning_path, index=False)
    development.to_csv(development_path, index=False, compression=GZIP_COMPRESSION)
    confirmation.to_csv(confirmation_path, index=False, compression=GZIP_COMPRESSION)

    checks = {
        "development_folds_only_0_2": set(development["fold"]) == set(DEVELOPMENT_FOLDS),
        "confirmation_folds_only_3_4": set(confirmation["fold"]) == set(CONFIRMATION_FOLDS),
        "development_confirmation_pairs_disjoint": not bool(
            set(development["calibration_pair_id"])
            & set(confirmation["calibration_pair_id"])
        ),
        "confirmation_scores_finite": bool(
            np.isfinite(confirmation[CANDIDATE].to_numpy(dtype=float)).all()
        ),
        "dtiam_within_drug_rank_exactly_preserved": rank_invariant,
        "drug_macro_auprc_exactly_preserved": abs(drug_metric_delta) < 1e-12,
        "confirmation_micro_auprc_improved": float(candidate["micro_auprc"])
        > float(reference["micro_auprc"]),
        "confirmation_target_macro_auprc_improved": float(
            candidate["target_macro_auprc"]
        )
        > float(reference["target_macro_auprc"]),
    }
    return {
        "protocol": protocol,
        "development_folds": list(DEVELOPMENT_FOLDS),
        "confirmation_folds": list(CONFIRMATION_FOLDS),
        "locked_coefficients": {
            "dtiam_drug_bias_weight": dtiam_weight,
            "biomaster_drug_bias_weight": biomaster_weight,
        },
        "confirmation_metrics": model_metrics,
        "confirmation_deltas_vs_dtiam": {
            "micro_auprc": float(candidate["micro_auprc"])
            - float(reference["micro_auprc"]),
            "target_macro_auprc": float(candidate["target_macro_auprc"])
            - float(reference["target_macro_auprc"]),
            "drug_macro_auprc": drug_metric_delta,
        },
        "directional_bootstrap_vs_dtiam": directional,
        "clustered_micro_bootstrap_vs_dtiam": clustered,
        "checks": {key: bool(value) for key, value in checks.items()},
        "inputs": {**development_hashes, **confirmation_hashes},
        "artifacts": {
            "tuning": str(tuning_path.relative_to(ROOT)),
            "tuning_sha256": sha256(tuning_path),
            "development_scores": str(development_path.relative_to(ROOT)),
            "development_scores_sha256": sha256(development_path),
            "confirmation_scores": str(confirmation_path.relative_to(ROOT)),
            "confirmation_scores_sha256": sha256(confirmation_path),
        },
    }


def evaluate_temporal_transfer(
    locked_results: list[dict[str, Any]],
    out_dir: Path,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Apply already-locked S2/S3 coefficients to S4 without temporal tuning."""

    if not TEMPORAL_SOURCE.is_file():
        raise FileNotFoundError(TEMPORAL_SOURCE)
    temporal = pd.read_csv(TEMPORAL_SOURCE, low_memory=False)
    if temporal["calibration_pair_id"].duplicated().any():
        raise RuntimeError("temporal transfer scope contains duplicate pair identifiers")

    reference = metrics(temporal, temporal[DTIAM].to_numpy(dtype=float))
    biomaster = metrics(temporal, temporal[BIOMASTER].to_numpy(dtype=float))
    presets: list[dict[str, Any]] = []
    score_columns: list[str] = []
    checks: dict[str, bool] = {
        "temporal_source_unique": not bool(
            temporal["calibration_pair_id"].duplicated().any()
        ),
        "temporal_labels_not_used_for_coefficient_selection": True,
    }


    for number, locked in enumerate(locked_results):
        source_protocol = str(locked["protocol"])
        token = source_protocol.split("_", 1)[0]
        coefficients = locked["locked_coefficients"]
        column = f"direction_preserving_{token.lower()}_locked_score"
        score_columns.append(column)
        temporal[column] = direction_preserving_score(
            temporal,
            float(coefficients["dtiam_drug_bias_weight"]),
            float(coefficients["biomaster_drug_bias_weight"]),
        )
        candidate = metrics(temporal, temporal[column].to_numpy(dtype=float))
        rank_invariant = drug_rank_invariant(
            temporal, temporal[column].to_numpy(dtype=float)
        )
        drug_delta = float(candidate["drug_macro_auprc"]) - float(
            reference["drug_macro_auprc"]
        )
        checks.update(
            {
                f"{token}_scores_finite": bool(
                    np.isfinite(temporal[column].to_numpy(dtype=float)).all()
                ),
                f"{token}_dtiam_within_drug_rank_exactly_preserved": rank_invariant,
                f"{token}_drug_macro_auprc_exactly_preserved": abs(drug_delta) < 1e-12,
                f"{token}_temporal_micro_auprc_improved": float(
                    candidate["micro_auprc"]
                )
                > float(reference["micro_auprc"]),
                f"{token}_temporal_target_macro_auprc_improved": float(
                    candidate["target_macro_auprc"]
                )
                > float(reference["target_macro_auprc"]),
            }
        )
        presets.append(
            {
                "coefficient_source_protocol": source_protocol,
                "coefficients": coefficients,
                "metrics": candidate,
                "deltas_vs_dtiam": {
                    "micro_auprc": float(candidate["micro_auprc"])
                    - float(reference["micro_auprc"]),
                    "target_macro_auprc": float(candidate["target_macro_auprc"])
                    - float(reference["target_macro_auprc"]),
                    "drug_macro_auprc": drug_delta,
                },
                "directional_bootstrap_vs_dtiam": [
                    directional_bootstrap(
                        temporal,
                        column,
                        DTIAM,
                        group,
                        bootstrap_iterations,
                        seed + number * 100 + group_number,
                    )
                    for group_number, group in enumerate(
                        ["target_chembl_id", "parent_standard_inchi_key"]
                    )
                ],
                "clustered_micro_bootstrap_vs_dtiam": [
                    clustered_micro_bootstrap(
                        temporal,
                        column,
                        DTIAM,
                        cluster,
                        bootstrap_iterations,
                        seed + number * 100 + 10 + cluster_number,
                    )
                    for cluster_number, cluster in enumerate(
                        ["target_homology_cluster", "scaffold_group"]
                    )
                ],
            }
        )

    score_path = out_dir / "S4_LOCKED_TRANSFER_SCORES_V1.csv.gz"
    temporal.to_csv(score_path, index=False, compression=GZIP_COMPRESSION)
    return {
        "protocol": TEMPORAL_PROTOCOL,
        "selection_contract": (
            "Both S2- and S3-derived coefficient presets were locked before S4 "
            "evaluation. S4 labels were used for evaluation only and neither "
            "preset was selected or retuned on S4."
        ),
        "rows": int(len(temporal)),
        "reference_metrics": {"DTIAM": reference, "BIOMASTER_STACK": biomaster},
        "locked_preset_transfers": presets,
        "checks": checks,
        "input": {
            str(TEMPORAL_SOURCE.relative_to(ROOT)): sha256(TEMPORAL_SOURCE),
        },
        "artifact": {
            "scores": str(score_path.relative_to(ROOT)),
            "scores_sha256": sha256(score_path),
            "score_columns": score_columns,
        },
    }


def evaluate_dense_deployment_stress(
    locked_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stress-test query offsets on the fixed 720 x 384 candidate universe."""

    required = [DEPLOYMENT_DTIAM, DEPLOYMENT_BIOMASTER, KIRHUB]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    dtiam = pd.read_csv(
        DEPLOYMENT_DTIAM,
        usecols=[
            "pairId",
            "ligand_inchikey",
            "target_chembl_id",
            "dtiam_probability",
        ],
    )
    biomaster = pd.read_csv(
        DEPLOYMENT_BIOMASTER,
        usecols=["pairId", "biomaster_routed_stack_score"],
    )
    dense = dtiam.merge(biomaster, on="pairId", validate="one_to_one").rename(
        columns={
            "ligand_inchikey": "parent_standard_inchi_key",
            "biomaster_routed_stack_score": BIOMASTER,
        }
    )
    preset_parameters = [
        (
            str(item["protocol"]),
            float(item["locked_coefficients"]["dtiam_drug_bias_weight"]),
            float(item["locked_coefficients"]["biomaster_drug_bias_weight"]),
        )
        for item in locked_results
    ]
    preset_parameters.append(
        (
            "ARITHMETIC_MEAN_OF_S2_S3_LOCKED",
            float(np.mean([item[1] for item in preset_parameters])),
            float(np.mean([item[2] for item in preset_parameters])),
        )
    )
    candidate_columns: list[tuple[str, str]] = []
    rank_checks: dict[str, bool] = {}
    for number, (name, dtiam_weight, biomaster_weight) in enumerate(preset_parameters):
        column = f"dense_candidate_{number}"
        dense[column] = direction_preserving_score(
            dense, dtiam_weight, biomaster_weight
        )
        candidate_columns.append((name, column))
        rank_checks[f"{name}_full_384_target_rank_preserved"] = drug_rank_invariant(
            dense, dense[column].to_numpy(dtype=float)
        )

    kirhub = pd.read_csv(KIRHUB, low_memory=False)
    score_columns = [column for _, column in candidate_columns]
    external = dense[
        [
            "pairId",
            "parent_standard_inchi_key",
            "target_chembl_id",
            DTIAM,
            *score_columns,
        ]
    ].merge(
        kirhub.drop(
            columns=[
                column
                for column in [DTIAM, "target_chembl_id"]
                if column in kirhub
            ]
        ),
        on="pairId",
        validate="one_to_one",
    )
    strict = external[
        external["kirhub_frozen_unreported_scope"].fillna(False).astype(bool)
    ].copy()
    strict["binary_label"] = strict["kirhub_local_unreported_active"].astype(np.int8)
    reference = metrics(strict, strict[DTIAM].to_numpy(dtype=float))
    candidates = []
    for name, column in candidate_columns:
        candidate = metrics(strict, strict[column].to_numpy(dtype=float))
        candidates.append(
            {
                "preset": name,
                "metrics": candidate,
                "deltas_vs_dtiam": {
                    metric: float(candidate[metric]) - float(reference[metric])
                    for metric in [
                        "micro_auprc",
                        "target_macro_auprc",
                        "drug_macro_auprc",
                    ]
                },
            }
        )
    any_dense_improvement = any(
        item["deltas_vs_dtiam"]["micro_auprc"] > 0
        and item["deltas_vs_dtiam"]["target_macro_auprc"] > 0
        for item in candidates
    )
    checks = {
        "exact_720x384_unique_pairs": len(dense) == 720 * 384
        and dense["pairId"].is_unique,
        "strict_kirhub_exact_2823_with_202_positives": len(strict) == 2823
        and int(strict["binary_label"].sum()) == 202,
        "all_dense_scores_finite": bool(
            np.isfinite(dense[score_columns].to_numpy(dtype=float)).all()
        ),
        **rank_checks,
    }
    return {
        "scope": "720_OLD_DRUGS_X_384_TARGETS_THEN_STRICT_KIRHUB",
        "reference_metrics": {"DTIAM": reference},
        "candidate_metrics": candidates,
        "checks": checks,
        "promotion_gate": (
            "PASS_DENSE_GENERALIZATION"
            if any_dense_improvement
            else "FAIL_REJECT_CANDIDATE_SET_DEPENDENT_CALIBRATION"
        ),
        "interpretation": (
            "Within-drug rank preservation remains exact, but drug-level offsets "
            "estimated from a complete 384-target universe shift relative to the "
            "sparse benchmark candidate sets. A failed gate forbids deployment of "
            "this calibration layer."
        ),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    if args.bootstrap_iterations < 500:
        raise ValueError("bootstrap_iterations must be at least 500")
    out_dir = Path(args.out_dir).resolve()
    results = [
        evaluate_protocol(
            protocol,
            out_dir,
            args.bootstrap_iterations,
            args.seed + 1000 * number,
        )
        for number, protocol in enumerate(PROTOCOLS)
    ]
    temporal_transfer = evaluate_temporal_transfer(
        results,
        out_dir,
        args.bootstrap_iterations,
        args.seed + 10000,
    )
    deployment_stress = evaluate_dense_deployment_stress(results)
    checks = {
        f"{item['protocol']}::{name}": value
        for item in results
        for name, value in item["checks"].items()
    }
    checks.update(
        {
            f"{TEMPORAL_PROTOCOL}::{name}": value
            for name, value in temporal_transfer["checks"].items()
        }
    )
    checks.update(
        {
            f"DENSE_DEPLOYMENT_STRESS::{name}": value
            for name, value in deployment_stress["checks"].items()
        }
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "method": "BIOMASTER_DTIAM_DIRECTION_PRESERVING_QUERY_CALIBRATION_V1",
        "selection_contract": (
            "Coefficients use frozen folds 0-2 only. Folds 3-4 are confirmation-only. "
            "The candidate score is a monotonic DTIAM within-drug residual plus "
            "label-free DTIAM/BioMaster drug-level offsets."
        ),
        "results": results,
        "temporal_locked_transfer": temporal_transfer,
        "dense_deployment_stress": deployment_stress,
        "promotion_decision": (
            "REJECT_DIRECTION_PRESERVING_CALIBRATION_FOR_PRODUCTION"
            if deployment_stress["promotion_gate"].startswith("FAIL")
            else "ELIGIBLE_FOR_PRODUCTION_REVIEW"
        ),
        "checks": checks,
        "claim_boundary": (
            "This is a performance-oriented heterogeneous ensemble and calibrated "
            "query layer, not a standalone BioMaster backbone comparison. Drug-macro "
            "non-inferiority is exact by construction. S2/S3 confirmation remains "
            "retrospective internal evidence; S4 is a coefficient-locked temporal "
            "transfer audit, not a prospective experimental validation. The dense "
            "deployment stress gate is authoritative for promotion and currently "
            "rejects this candidate-set-dependent calibration."
        ),
    }
    summary_path = out_dir / "DIRECTION_PRESERVING_CALIBRATION_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

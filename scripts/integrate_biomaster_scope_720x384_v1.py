#!/usr/bin/env python3
"""Integrate BioMaster and SCOPE deployment experts and run source-held KIRHub audit."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_biomaster_odti_paired_bootstrap_v1 import paired_cluster_bootstrap  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
BM = BASE / "biomaster_odti_deployment_v1/BIOMASTER_ODTI_720X384_SCORES_V1.csv.gz"
SCOPE = BASE / "public_baselines_v1/scope_dti_720x384_v1/SCOPE_DTI_720X384_PREDICTIONS_V1.csv.gz"
FUSION = BASE / "public_baselines_v1/scope_biomaster_validation_fusion_v1/SCOPE_BIOMASTER_VALIDATION_FUSION_SUMMARY_V1.json"
KIRHUB = (
    ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/leakage_safe_ranker_v10"
    / "external_evaluation/OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_V10.csv"
)
OUT = BASE / "biomaster_scope_integrated_deployment_v1"
FUSION_NAME = "SCOPE_BIOMASTER_ROUTED_FUSION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logistic(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60, 60)
    return 1.0 / (1.0 + np.exp(-value))


def main() -> None:
    required = [BM, SCOPE, FUSION, KIRHUB]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    bm = pd.read_csv(BM, low_memory=False)
    scope = pd.read_csv(SCOPE, low_memory=False)
    frozen = json.loads(FUSION.read_text())
    if frozen.get("status") != "PASS":
        raise RuntimeError("Frozen validation-only SCOPE fusion must pass")
    parameters = frozen["fixed_fusion_models"][FUSION_NAME]
    frame = bm.merge(
        scope[["pairId", "scope_target_available", "scope_mean", "scope_std"]],
        on="pairId",
        how="left",
        validate="one_to_one",
    )
    if len(frame) != 276480:
        raise RuntimeError("Deployment matrix changed")
    frame["biomaster_logit"] = frame["biomaster_ensemble_logit"]
    features = parameters["features"]
    covered = frame["scope_mean"].notna().to_numpy()
    standardized = np.empty((covered.sum(), len(features)), dtype=np.float64)
    for index, feature in enumerate(features):
        standardized[:, index] = (
            frame.loc[covered, feature].to_numpy(dtype=np.float64)
            - float(parameters["scaler_mean"][feature])
        ) / float(parameters["scaler_scale"][feature])
    coefficient = np.asarray(
        [parameters["standardized_coefficients"][feature] for feature in features], dtype=np.float64
    )
    fusion_score = logistic(standardized @ coefficient + float(parameters["intercept"]))
    frame["biomaster_scope_fusion_score"] = np.nan
    frame.loc[covered, "biomaster_scope_fusion_score"] = fusion_score
    frame["integrated_old_drug_target_score"] = frame["biomaster_scope_fusion_score"].fillna(
        frame["biomaster_routed_stack_score"]
    )
    frame["integrated_model_route"] = np.where(
        covered,
        "SCOPE_BIOMASTER_VALIDATION_ROUTED_FUSION",
        "BIOMASTER_SEQUENCE_CHEMISTRY_FALLBACK_SCOPE_TARGET_MISSING",
    )
    frame["integrated_rank_within_old_drug_384"] = frame.groupby("ligand_inchikey")[
        "integrated_old_drug_target_score"
    ].rank(method="first", ascending=False).astype(np.int16)
    frame["integrated_percentile_within_old_drug_384"] = (
        1.0 - (frame["integrated_rank_within_old_drug_384"] - 1) / 383.0
    )
    frame["scope_biomaster_absolute_disagreement"] = np.where(
        covered,
        np.abs(frame["scope_mean"] - frame["biomaster_ensemble_probability"]),
        np.nan,
    )
    frame["integrated_uncertainty_tier"] = np.select(
        [
            ~covered,
            frame["scope_biomaster_absolute_disagreement"].ge(0.5)
            | frame["scope_std"].ge(0.20)
            | frame["biomaster_model_seed_std"].ge(0.20),
            frame["scope_biomaster_absolute_disagreement"].ge(0.25)
            | frame["scope_std"].ge(0.10)
            | frame["biomaster_model_seed_std"].ge(0.10),
        ],
        ["U3_SCOPE_TARGET_MISSING", "U2_HIGH_DISAGREEMENT", "U1_MODERATE_DISAGREEMENT"],
        default="U0_LOW_DISAGREEMENT",
    )
    frame["integrated_priority"] = np.select(
        [
            frame["is_any_frozen_known_relationship"].fillna(False).astype(bool),
            frame["integrated_rank_within_old_drug_384"].le(20)
            & frame["local_chembl37_unreported_pair"].fillna(False).astype(bool)
            & frame["integrated_uncertainty_tier"].isin(["U0_LOW_DISAGREEMENT", "U1_MODERATE_DISAGREEMENT"]),
            frame["integrated_rank_within_old_drug_384"].le(20),
            frame["integrated_rank_within_old_drug_384"].le(50),
        ],
        [
            "KNOWN_RELATIONSHIP_CONTROL", "TOP20_LOCAL_UNREPORTED_LOW_MODERATE_UNCERTAINTY",
            "TOP20_REVIEW_HIGH_OR_MISSING_MODALITY_UNCERTAINTY", "TOP50_EXPLORATORY_REVIEW",
        ],
        default="HOLD_BEYOND_TOP50",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    score_path = OUT / "BIOMASTER_SCOPE_INTEGRATED_720X384_V1.csv.gz"
    top_path = OUT / "BIOMASTER_SCOPE_INTEGRATED_TOP20_PER_OLD_DRUG_V1.csv.gz"
    application_path = OUT / "BIOMASTER_SCOPE_INNOVATION_APPLICATION_CASES_68_V1.csv"
    frame.to_csv(score_path, index=False)
    frame[frame["integrated_rank_within_old_drug_384"].le(20)].to_csv(top_path, index=False)
    application = frame[
        frame["is_v8_mutation_application_pair"].fillna(False).astype(bool)
        | frame["is_v8_database_gap_rediscovery_control"].fillna(False).astype(bool)
        | frame["is_v8_prospective_unvalidated_case"].fillna(False).astype(bool)
    ].copy()
    application.sort_values(
        ["integrated_rank_within_old_drug_384", "integrated_old_drug_target_score"],
        ascending=[True, False],
    ).to_csv(application_path, index=False)

    # Source-held external biochemical audit. KIRHub labels were never used in
    # representation learning, base fitting, validation stacking, or routing.
    external = pd.read_csv(KIRHUB, low_memory=False)
    external = external.merge(
        frame[[
            "pairId", "integrated_old_drug_target_score", "biomaster_scope_fusion_score",
            "biomaster_routed_stack_score", "scope_mean", "integrated_model_route",
        ]],
        on="pairId",
        how="left",
        validate="one_to_one",
    )
    if external["integrated_old_drug_target_score"].isna().any():
        raise RuntimeError("KIRHub external pairs not fully covered")
    external_rows = []
    slices = {
        "ALL_KIRHUB_WT_ACTIVE": (np.ones(len(external), dtype=bool), "kirhub_wt_active_le30pct_residual"),
        "STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE": (
            external["kirhub_frozen_unreported_scope"].fillna(False).astype(bool).to_numpy(),
            "kirhub_local_unreported_active",
        ),
    }
    model_scores = {
        "BIOMASTER_SCOPE_VALIDATION_ROUTED_FUSION": "integrated_old_drug_target_score",
        "SCOPE_PUBLIC_CHECKPOINT": "scope_mean",
        "BIOMASTER_INTERNAL_ROUTED_STACK": "biomaster_routed_stack_score",
        "V10_PREVIOUS_LEAKAGE_SAFE": "old_drug_leakage_safe_score_v10",
        "CONPLEX_DRUG_CENTRIC": "conplex_percentile_within_ligand_384",
    }
    bootstrap_rows = []
    for slice_name, (mask, label_column) in slices.items():
        part = external.loc[mask].copy()
        part["binary_label"] = part[label_column].fillna(False).astype(np.int8)
        # The benchmark metric helper uses the canonical ChEMBL drug key.  KIRHub
        # is keyed by the same chemical entity under the project-facing name.
        part["parent_standard_inchi_key"] = part["ligand_inchikey"]
        for model_name, score_column in model_scores.items():
            # SCOPE lacks four of the 384 project targets.  Report that public
            # checkpoint only on its genuinely supported rows; the integrated
            # route remains evaluable on all rows through the frozen BioMaster
            # sequence/chemistry fallback.
            model_part = part.dropna(subset=[score_column]).copy()
            row = {
                "evaluation_slice": slice_name,
                "model": model_name,
                "score_column": score_column,
                "available_rows": int(len(model_part)),
                "unavailable_rows": int(len(part) - len(model_part)),
            }
            row.update(metrics(model_part, model_part[score_column].to_numpy(dtype=np.float64)))
            external_rows.append(row)
        for reference in ["old_drug_leakage_safe_score_v10", "scope_mean"]:
            comparison = part.dropna(
                subset=["integrated_old_drug_target_score", reference]
            ).copy()
            for metric_name in ["auprc", "auroc"]:
                row = paired_cluster_bootstrap(
                    comparison,
                    "integrated_old_drug_target_score",
                    reference,
                    metric_name,
                    2000,
                    20260813 + len(bootstrap_rows),
                    "ligand_inchikey",
                )
                row["evaluation_slice"] = slice_name
                bootstrap_rows.append(row)
    external_metrics = pd.DataFrame(external_rows)
    external_bootstrap = pd.DataFrame(bootstrap_rows)
    external_path = OUT / "SOURCE_HELD_KIRHUB_INTEGRATED_PREDICTIONS_V1.csv.gz"
    external_metric_path = OUT / "SOURCE_HELD_KIRHUB_METRICS_V1.csv"
    external_bootstrap_path = OUT / "SOURCE_HELD_KIRHUB_PAIRED_BOOTSTRAP_V1.csv"
    external.to_csv(external_path, index=False)
    external_metrics.to_csv(external_metric_path, index=False)
    external_bootstrap.to_csv(external_bootstrap_path, index=False)

    strict_metric = external_metrics[
        external_metrics["evaluation_slice"].eq("STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE")
        & external_metrics["model"].eq("BIOMASTER_SCOPE_VALIDATION_ROUTED_FUSION")
    ].iloc[0].to_dict()
    strict_v10 = external_metrics[
        external_metrics["evaluation_slice"].eq("STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE")
        & external_metrics["model"].eq("V10_PREVIOUS_LEAKAGE_SAFE")
    ].iloc[0].to_dict()
    strict_v10_bootstrap = external_bootstrap[
        external_bootstrap["evaluation_slice"].eq("STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE")
        & external_bootstrap["reference"].eq("old_drug_leakage_safe_score_v10")
        & external_bootstrap["metric"].eq("auprc")
    ].iloc[0].to_dict()
    strict_scope_bootstrap = external_bootstrap[
        external_bootstrap["evaluation_slice"].eq("STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE")
        & external_bootstrap["reference"].eq("scope_mean")
        & external_bootstrap["metric"].eq("auprc")
    ].iloc[0].to_dict()
    checks = {
        "exact_720x384_unique_integrated_scores": len(frame) == 276480 and frame["pairId"].is_unique,
        "exact_14400_top20_rows": int(frame["integrated_rank_within_old_drug_384"].le(20).sum()) == 14400,
        "fusion_covered_273600_and_fallback_2880": int(covered.sum()) == 273600 and int((~covered).sum()) == 2880,
        "all_integrated_scores_finite_bounded": np.isfinite(frame["integrated_old_drug_target_score"]).all() and frame["integrated_old_drug_target_score"].between(0, 1).all(),
        "exact_68_application_cases": len(application) == 68,
        "kirhub_exact_8058_external_pairs": len(external) == 8058,
        "strict_unreported_exact_2823_pairs_202_positives": int(slices["STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE"][0].sum()) == 2823 and int(external.loc[slices["STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE"][0], "kirhub_local_unreported_active"].sum()) == 202,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": "BIOMASTER_SCOPE_INTEGRATED_OLD_DRUG_TARGET_V1",
        "frozen_fusion_source": str(FUSION.relative_to(ROOT)),
        "routing": {
            "scope_biomaster_fusion_pairs": int(covered.sum()),
            "biomaster_sequence_chemistry_fallback_pairs": int((~covered).sum()),
            "scope_missing_project_targets": sorted(frame.loc[~covered, "target_chembl_id"].unique()),
        },
        "counts": {
            "pairs": int(len(frame)), "old_drugs": 720, "targets": 384,
            "mutation_application_pairs": int(frame["is_v8_mutation_application_pair"].fillna(False).sum()),
            "database_gap_controls": int(frame["is_v8_database_gap_rediscovery_control"].fillna(False).sum()),
            "prospective_unvalidated_cases": int(frame["is_v8_prospective_unvalidated_case"].fillna(False).sum()),
        },
        "source_held_kirhub": {
            "all_pairs": int(len(external)),
            "scope_supported_pairs": int(external["scope_mean"].notna().sum()),
            "scope_unsupported_pairs": int(external["scope_mean"].isna().sum()),
            "strict_frozen_unreported_pairs": 2823,
            "strict_frozen_unreported_positives": 202,
            "integrated_metrics": strict_metric,
            "previous_v10_metrics": strict_v10,
            "integrated_minus_v10_micro_auprc_bootstrap": strict_v10_bootstrap,
            "integrated_minus_scope_micro_auprc_bootstrap": strict_scope_bootstrap,
            "interpretation": "Integrated fusion significantly exceeds SCOPE on strict KIRHub AUPRC, but is statistically indistinguishable from frozen V10 and has mixed target- versus drug-centric macro behavior.",
            "label_use_boundary": "KIRHub labels used only for final audit; never for model/stack fitting or routing",
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "claim_status": "MULTI_SEED_COMPLETE; SOURCE_HELD_KIRHUB_MIXED_AND_NO_OVERALL_V10_WIN; NO_FULL_SOTA_CLAIM; SAME_DATA_PUBLIC_RETRAINING_REQUIRED",
        "artifacts": {
            "integrated_scores_sha256": sha256(score_path),
            "top20_sha256": sha256(top_path),
            "application_cases_sha256": sha256(application_path),
            "external_predictions_sha256": sha256(external_path),
            "external_metrics_sha256": sha256(external_metric_path),
            "external_bootstrap_sha256": sha256(external_bootstrap_path),
        },
    }
    summary_path = OUT / "BIOMASTER_SCOPE_INTEGRATED_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps({key: bool(value) for key, value in checks.items()}, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

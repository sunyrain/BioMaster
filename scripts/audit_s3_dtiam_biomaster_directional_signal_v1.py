#!/usr/bin/env python3
"""Compare DTIAM and BioMaster pair signal under identical cold-protocol query groups."""

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
COMPARISON = (
    ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_same_data_comparison_v1"
)
FUSION = (
    ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_biomaster_cold_fusion_v1"
)
PROTOCOLS = {
    "S1_SCAFFOLD_COLD_DRUG": "s1",
    "S2_HOMOLOGY_COLD_TARGET": "s2",
    "S3_STRICT_DOUBLE_COLD": "s3",
    "S4_FIRST_SEEN_TEMPORAL_2023_2025": "s4",
    "S5_OLD_DRUG_ENTITY_COLD": "s5",
}
MODELS = {
    "DTIAM_OFFICIAL_DEFAULT_COMPAT": "dtiam_probability",
    "BIOMASTER_STACK_FIVE_SEED_MEAN": "biomaster_stack_score",
    "BIOMASTER_RAW_FIVE_SEED_MEAN": "biomaster_raw_score",
}
SCOPES = {
    "ALL_FOLDS_0_4": [0, 1, 2, 3, 4],
    "DEVELOPMENT_FOLDS_0_2": [0, 1, 2],
    "CONFIRMATORY_FOLDS_3_4": [3, 4],
}
FIXED_TEST_SCOPES = {
    "FROZEN_TEST": [-1],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    series = pd.Series(np.arange(len(groups)), index=groups)
    return [part.to_numpy(dtype=np.int64) for _, part in series.groupby(level=0, sort=False)]


def macro_ap(y: np.ndarray, score: np.ndarray, positions: list[np.ndarray]) -> tuple[float, int]:
    values = []
    for index in positions:
        labels = y[index]
        if labels.min() == labels.max():
            continue
        values.append(float(average_precision_score(labels, score[index])))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def fast_macro_ap(y: np.ndarray, score: np.ndarray, positions: list[np.ndarray]) -> float:
    total = 0.0
    evaluated = 0
    for index in positions:
        labels = y[index]
        positives = int(labels.sum())
        if positives == 0 or positives == len(labels):
            continue
        ranked = labels[np.argsort(-score[index], kind="stable")]
        precision = np.cumsum(ranked, dtype=float) / np.arange(1, len(ranked) + 1)
        total += float(precision[ranked.astype(bool)].sum() / positives)
        evaluated += 1
    return total / evaluated if evaluated else float("nan")


def permute_within(
    score: np.ndarray, positions: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    result = score.copy()
    for index in positions:
        if len(index) > 1:
            result[index] = score[rng.permutation(index)]
    return result


def intervals(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
    }


def one_model(
    frame: pd.DataFrame,
    model: str,
    column: str,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    score = frame[column].to_numpy(dtype=float)
    target_index = group_indices(frame["target_chembl_id"].to_numpy())
    drug_index = group_indices(frame["parent_standard_inchi_key"].to_numpy())
    target_ap, target_n = macro_ap(y, score, target_index)
    drug_ap, drug_n = macro_ap(y, score, drug_index)
    rng = np.random.default_rng(seed)
    target_null = []
    drug_null = []
    for _ in range(permutations):
        target_null.append(fast_macro_ap(y, permute_within(score, target_index, rng), target_index))
        drug_null.append(fast_macro_ap(y, permute_within(score, drug_index, rng), drug_index))
    target_interval = intervals(target_null)
    drug_interval = intervals(drug_null)
    return {
        "model": model,
        "score_column": column,
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "micro_auroc": float(roc_auc_score(y, score)),
        "micro_auprc": float(average_precision_score(y, score)),
        "target_macro_auprc": target_ap,
        "target_groups_with_both_classes": target_n,
        "target_null_mean": target_interval["mean"],
        "target_null_ci95_low": target_interval["ci95_low"],
        "target_null_ci95_high": target_interval["ci95_high"],
        "target_query_pair_signal_supported": bool(target_ap > target_interval["ci95_high"]),
        "drug_macro_auprc": drug_ap,
        "drug_groups_with_both_classes": drug_n,
        "drug_null_mean": drug_interval["mean"],
        "drug_null_ci95_low": drug_interval["ci95_low"],
        "drug_null_ci95_high": drug_interval["ci95_high"],
        "drug_query_pair_signal_supported": bool(drug_ap > drug_interval["ci95_high"]),
    }


def paired_directional_bootstrap(
    frame: pd.DataFrame,
    challenger: str,
    reference: str,
    group_column: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    differences = []
    for _, part in frame.groupby(group_column, sort=False, observed=True):
        labels = part["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        differences.append(
            float(average_precision_score(labels, part[challenger]))
            - float(average_precision_score(labels, part[reference]))
        )
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "challenger": challenger,
        "reference": reference,
        "direction_group": group_column,
        "groups": int(len(values)),
        "observed_macro_auprc_difference": float(values.mean()),
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "probability_difference_gt_zero": float((sampled > 0).mean()),
        "ci95_excludes_zero_in_favor_of_challenger": bool(low > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="S3_STRICT_DOUBLE_COLD")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.permutations < 50 or args.bootstrap_iterations < 500:
        raise ValueError("Insufficient audit iterations")
    prefix = PROTOCOLS[args.protocol]
    input_path = COMPARISON / f"{args.protocol}_ALIGNED_PREDICTIONS_V1.csv.gz"
    out = ROOT / f"outputs/old_drug_target_sota_v1/query_shortcut_dtiam_biomaster_{prefix}_audit_v1"
    frame = pd.read_csv(input_path, low_memory=False)
    if args.protocol in {"S1_SCAFFOLD_COLD_DRUG", "S4_FIRST_SEEN_TEMPORAL_2023_2025"}:
        models = {
            "DTIAM_OFFICIAL_DEFAULT_COMPAT": "dtiam_probability",
            "BIOMASTER_STACK_SINGLE_SEED_20260813": "biomaster_stack_score",
            "BIOMASTER_RAW_SINGLE_SEED_20260813": "biomaster_raw_score",
        }
        biomaster_stack_model = "BIOMASTER_STACK_SINGLE_SEED_20260813"
    elif args.protocol == "S5_OLD_DRUG_ENTITY_COLD":
        models = {
            "DTIAM_OFFICIAL_DEFAULT_COMPAT": "dtiam_probability",
            "BIOMASTER_STACK_FIVE_SEED_MEAN": "BIOMASTER_STACK_FIVE_SEED_MEAN",
            "BIOMASTER_RAW_FIVE_SEED_MEAN": "BIOMASTER_RAW_FIVE_SEED_MEAN",
        }
        biomaster_stack_model = "BIOMASTER_STACK_FIVE_SEED_MEAN"
    else:
        models = dict(MODELS)
        biomaster_stack_model = "BIOMASTER_STACK_FIVE_SEED_MEAN"
    if args.protocol == "S5_OLD_DRUG_ENTITY_COLD":
        fusion_path = (
            ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/"
            "dtiam_biomaster_validation_fusion_v1/"
            "DTIAM_BIOMASTER_S5_FUSION_PREDICTIONS_V1.csv.gz"
        )
    else:
        fusion_path = FUSION / f"{args.protocol}_POOLED_PREDICTIONS_V1.csv.gz"
    if fusion_path.is_file():
        fusion = pd.read_csv(
            fusion_path,
            usecols=["calibration_pair_id", "DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION"],
            low_memory=False,
        )
        frame = frame.merge(fusion, on="calibration_pair_id", how="left", validate="one_to_one")
        if frame["DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION"].notna().all():
            models["VALIDATION_ONLY_SAME_DATA_ROUTED_FUSION"] = (
                "DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION"
            )
    required = {
        "calibration_pair_id", "binary_label", "target_chembl_id",
        "parent_standard_inchi_key", "fold", *models.values(),
    }
    absent = sorted(required - set(frame.columns))
    if absent:
        raise RuntimeError(f"Missing columns: {absent}")

    rows = []
    bootstraps = []
    scopes = (
        FIXED_TEST_SCOPES
        if args.protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"}
        else SCOPES
    )
    for scope_number, (scope, folds) in enumerate(scopes.items()):
        part = frame[frame["fold"].isin(folds)].copy()
        for model_number, (model, column) in enumerate(models.items()):
            row = one_model(
                part, model, column, args.permutations,
                args.seed + 1009 * model_number + 100003 * scope_number,
            )
            row.update({"scope": scope, "folds": ";".join(map(str, folds))})
            rows.append(row)
        biomaster_stack_column = models[biomaster_stack_model]
        comparisons = [("dtiam_probability", biomaster_stack_column)]
        if "DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION" in part:
            comparisons.extend([
                ("DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION", "dtiam_probability"),
                ("DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION", biomaster_stack_column),
            ])
        for comparison_number, (challenger, reference) in enumerate(comparisons):
            for direction_number, group in enumerate(
                ["parent_standard_inchi_key", "target_chembl_id"]
            ):
                result = paired_directional_bootstrap(
                    part, challenger, reference, group,
                    args.bootstrap_iterations,
                    args.seed + 500003 * scope_number + 1009 * comparison_number + direction_number,
                )
                result.update({"scope": scope, "folds": ";".join(map(str, folds))})
                bootstraps.append(result)

    result = pd.DataFrame(rows).sort_values(["scope", "model"])
    bootstrap = pd.DataFrame(bootstraps).sort_values(["scope", "direction_group"])
    out.mkdir(parents=True, exist_ok=True)
    protocol_token = args.protocol.split("_", 1)[0]
    result_path = out / f"{protocol_token}_DTIAM_BIOMASTER_DIRECTIONAL_SIGNAL_RESULTS_V1.csv"
    bootstrap_path = out / f"{protocol_token}_DTIAM_VS_BIOMASTER_DIRECTIONAL_BOOTSTRAP_V1.csv"
    result.to_csv(result_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    primary_scope_name = next(iter(scopes))
    all_scope = result[result["scope"].eq(primary_scope_name)]
    dtiam = all_scope[all_scope["model"].eq("DTIAM_OFFICIAL_DEFAULT_COMPAT")].iloc[0]
    biomaster = all_scope[all_scope["model"].eq(biomaster_stack_model)].iloc[0]
    checks = {
        "identical_pair_population": bool(frame["calibration_pair_id"].is_unique),
        "all_scopes_and_models_present": len(result) == len(scopes) * len(models),
        "all_numeric_outputs_finite": bool(
            np.isfinite(result.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
            and np.isfinite(bootstrap.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        ),
    }
    findings = {
        "dtiam_supports_target_query_pair_signal_all_folds": bool(
            dtiam["target_query_pair_signal_supported"]
        ),
        "dtiam_supports_drug_query_pair_signal_all_folds": bool(
            dtiam["drug_query_pair_signal_supported"]
        ),
        "biomaster_supports_target_query_pair_signal_all_folds": bool(
            biomaster["target_query_pair_signal_supported"]
        ),
        "biomaster_supports_drug_query_pair_signal_all_folds": bool(
            biomaster["drug_query_pair_signal_supported"]
        ),
        "model_deficiency_not_protocol_impossibility": bool(
            dtiam["drug_query_pair_signal_supported"]
            and not biomaster["drug_query_pair_signal_supported"]
        ),
    }
    routed = all_scope[
        all_scope["model"].eq("VALIDATION_ONLY_SAME_DATA_ROUTED_FUSION")
    ]
    if not routed.empty:
        findings.update({
            "routed_fusion_supports_target_query_pair_signal_all_folds": bool(
                routed.iloc[0]["target_query_pair_signal_supported"]
            ),
            "routed_fusion_supports_drug_query_pair_signal_all_folds": bool(
                routed.iloc[0]["drug_query_pair_signal_supported"]
            ),
        })
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": args.protocol,
        "purpose": "same-pair diagnostic separating directional pair ranking from query-group score priors",
        "permutations": args.permutations,
        "bootstrap_iterations": args.bootstrap_iterations,
        "checks": {key: bool(value) for key, value in checks.items()},
        "scientific_findings": findings,
        "primary_scope": primary_scope_name,
        "all_folds_primary_rows": all_scope.to_dict("records"),
        "claim_boundary": (
            "This diagnoses current-model directional signal on an already evaluated frozen cold-protocol population; "
            "it does not tune EQIR and does not establish SOTA."
        ),
        "artifacts": {
            "results": str(result_path.relative_to(ROOT)),
            "results_sha256": sha256(result_path),
            "bootstrap": str(bootstrap_path.relative_to(ROOT)),
            "bootstrap_sha256": sha256(bootstrap_path),
            "input": str(input_path.relative_to(ROOT)),
            "input_sha256": sha256(input_path),
        },
    }
    summary_path = out / f"{protocol_token}_DTIAM_BIOMASTER_DIRECTIONAL_SIGNAL_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

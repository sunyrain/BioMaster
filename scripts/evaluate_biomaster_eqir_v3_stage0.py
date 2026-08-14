#!/usr/bin/env python3
"""Run the preregistered V3 Stage-0 structure utility screen on S3 folds 0--2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
AUDIT = BASE / "v3_label_blind_structure_audit_v1"
SCORES = AUDIT / "evaluation_v1/V3_LABEL_BLIND_GNINA_PAIR_SCORES_V1.csv.gz"
EXTRACTION = AUDIT / "evaluation_v1/V3_LABEL_BLIND_GNINA_EXTRACTION_SUMMARY_V1.json"
LEDGER = AUDIT / "V3_LABEL_BLIND_EFFECTIVE_EVALUATION_LEDGER_V1.csv.gz"
PREDICTIONS = BASE / "biomaster_odti_routed_ranker_v1"
OUT = BASE / "biomaster_eqir_v3_stage0"
FOLDS = [0, 1, 2]
SEEDS = [20260813, 20260814, 20260815, 20260816, 20260817]
ALPHAS = np.round(np.arange(0.0, 4.0001, 0.05), 8)
STRUCTURE_COLUMN = "primary_cnn_affinity_target_ecdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def prediction_path(prediction_root: Path, fold: int, seed: int, role: str) -> Path:
    name = "VALIDATION_PREDICTIONS_V1.csv.gz" if role == "VALID" else "TEST_PREDICTIONS_V1.csv.gz"
    return prediction_root / (
        f"S3_STRICT_DOUBLE_COLD__fold_{fold}__seed_{seed}__CORE/{name}"
    )


def macro_ap(frame: pd.DataFrame, scores: np.ndarray, group_column: str) -> tuple[float, int]:
    work = frame[[group_column, "binary_label"]].copy()
    work["score"] = scores
    values = []
    for _, group in work.groupby(group_column, sort=False):
        labels = group["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        values.append(float(average_precision_score(labels, group["score"])))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def fit_alpha(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    base_logit = frame["biomaster_logit"].to_numpy(dtype=float)
    increment = frame[STRUCTURE_COLUMN].to_numpy(dtype=float) - 0.5
    best: dict[str, float] | None = None
    for alpha in ALPHAS:
        score = sigmoid(base_logit + float(alpha) * increment)
        micro = float(average_precision_score(labels, score))
        target_macro = macro_ap(frame, score, "target_chembl_id")[0]
        drug_macro = macro_ap(frame, score, "parent_molecule_chembl_id")[0]
        objective = float((micro + target_macro + drug_macro) / 3.0)
        row = {
            "alpha": float(alpha),
            "micro_auprc": micro,
            "target_macro_auprc": target_macro,
            "drug_macro_auprc": drug_macro,
            "equal_weight_three_metric_objective": objective,
        }
        if best is None or objective > best["equal_weight_three_metric_objective"] + 1e-12:
            best = row
    if best is None or not np.isfinite(list(best.values())).all():
        raise RuntimeError("No finite Stage-0 alpha candidate")
    return best


def grouped_bootstrap(
    frame: pd.DataFrame, group_column: str, iterations: int, seed: int
) -> dict[str, float | int]:
    groups = frame[group_column].astype(str)
    unique = groups.unique()
    indices = {group: np.flatnonzero(groups.to_numpy() == group) for group in unique}
    rng = np.random.default_rng(seed)
    differences = []
    labels_all = frame["binary_label"].to_numpy(dtype=np.int8)
    base_all = frame["base_ensemble"].to_numpy(dtype=float)
    fused_all = frame["fused_ensemble"].to_numpy(dtype=float)
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        labels = labels_all[selected]
        if labels.min() == labels.max():
            continue
        differences.append(
            average_precision_score(labels, fused_all[selected])
            - average_precision_score(labels, base_all[selected])
        )
    return {
        "group_column": group_column,
        "groups": int(len(unique)),
        "iterations_requested": int(iterations),
        "iterations_valid": int(len(differences)),
        "difference_mean": float(np.mean(differences)),
        "difference_ci95_low": float(np.quantile(differences, 0.025)),
        "difference_ci95_high": float(np.quantile(differences, 0.975)),
    }


def grouped_direction_metrics(frame: pd.DataFrame, group_column: str, score_column: str) -> dict[str, Any]:
    aps = []
    aucs = []
    for _, group in frame.groupby(group_column, sort=False):
        labels = group["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        scores = group[score_column].to_numpy(dtype=float)
        aps.append(average_precision_score(labels, scores))
        aucs.append(roc_auc_score(labels, scores))
    return {
        "evaluable_groups": len(aps),
        "macro_auprc": float(np.mean(aps)) if aps else None,
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
    }


def directional_group_bootstrap(
    frame: pd.DataFrame, group_column: str, iterations: int, seed: int
) -> dict[str, Any]:
    differences = []
    for _, group in frame.groupby(group_column, sort=False):
        labels = group["binary_label"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        differences.append(
            float(average_precision_score(labels, group["fused_ensemble"]))
            - float(average_precision_score(labels, group["base_ensemble"]))
        )
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        raise RuntimeError(f"No evaluable directional groups for {group_column}")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "group_column": group_column,
        "groups": int(len(values)),
        "observed_macro_auprc_difference": float(values.mean()),
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "probability_difference_gt_zero": float((boot > 0).mean()),
    }


def fast_macro_ap(labels: np.ndarray, scores: np.ndarray, positions: list[np.ndarray]) -> float:
    total = 0.0
    evaluated = 0
    for index in positions:
        y = labels[index]
        positives = int(y.sum())
        if positives == 0 or positives == len(y):
            continue
        ranked = y[np.argsort(-scores[index], kind="stable")]
        precision = np.cumsum(ranked, dtype=float) / np.arange(1, len(ranked) + 1)
        total += float(precision[ranked.astype(bool)].sum() / positives)
        evaluated += 1
    return total / evaluated if evaluated else float("nan")


def within_query_permutation_null(
    frame: pd.DataFrame, group_column: str, iterations: int, seed: int
) -> dict[str, Any]:
    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    scores = frame["fused_ensemble"].to_numpy(dtype=float)
    positions = [
        np.asarray(index, dtype=np.int64)
        for index in frame.groupby(group_column, sort=False).indices.values()
    ]
    observed = fast_macro_ap(labels, scores, positions)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(iterations):
        shuffled = scores.copy()
        for index in positions:
            if len(index) > 1:
                shuffled[index] = scores[rng.permutation(index)]
        null.append(fast_macro_ap(labels, shuffled, positions))
    values = np.asarray(null, dtype=float)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "group_column": group_column,
        "iterations": int(iterations),
        "observed_macro_auprc": float(observed),
        "null_mean": float(values.mean()),
        "null_ci95_low": float(low),
        "null_ci95_high": float(high),
        "pair_signal_supported": bool(observed > high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--extraction-summary", type=Path, default=EXTRACTION)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--prediction-root", type=Path, default=PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.iterations < 500 or args.permutations < 50:
        raise ValueError("Stage-0 directional audit requires >=500 bootstraps and >=50 permutations")
    for path in [args.scores, args.extraction_summary, args.ledger]:
        if not path.exists():
            raise FileNotFoundError(path)
    extraction = json.loads(args.extraction_summary.read_text())
    if extraction["partial"] and not args.allow_partial:
        raise RuntimeError("Stage-0 evaluation is locked until all label-blind GNINA jobs complete")
    scores = pd.read_csv(args.scores)
    ledger = pd.read_csv(args.ledger)
    evidence = ledger.merge(
        scores, on=["target_chembl_id", "parent_molecule_chembl_id"], how="inner", validate="one_to_one"
    )
    if evidence[STRUCTURE_COLUMN].isna().any():
        raise RuntimeError("Missing primary target-local structure rank")

    run_rows: list[dict[str, Any]] = []
    test_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for fold in FOLDS:
            role_column = f"s3_fold_{fold}_role"
            role_data = {}
            for role in ["VALID", "TEST"]:
                path = prediction_path(args.prediction_root, fold, seed, role)
                if not path.exists():
                    raise FileNotFoundError(path)
                predictions = pd.read_csv(path)
                selected = evidence[evidence[role_column].eq(role)]
                frame = predictions.merge(
                    selected[[
                        "target_chembl_id", "parent_molecule_chembl_id", STRUCTURE_COLUMN,
                        "target_homology_cluster", "scaffold_group",
                    ]],
                    on=["target_chembl_id", "parent_molecule_chembl_id"],
                    how="inner",
                    validate="one_to_one",
                )
                expected = len(selected)
                if not args.allow_partial and len(frame) != expected:
                    raise RuntimeError(
                        f"Coverage mismatch seed={seed} fold={fold} role={role}: {len(frame)} != {expected}"
                    )
                role_data[role] = frame
            selection = fit_alpha(role_data["VALID"])
            alpha = selection["alpha"]
            test = role_data["TEST"].copy()
            test["base_probability"] = sigmoid(test["biomaster_logit"].to_numpy(dtype=float))
            test["fused_probability"] = sigmoid(
                test["biomaster_logit"].to_numpy(dtype=float)
                + alpha * (test[STRUCTURE_COLUMN].to_numpy(dtype=float) - 0.5)
            )
            base_metrics = metrics(test["binary_label"].to_numpy(), test["base_probability"].to_numpy())
            fused_metrics = metrics(test["binary_label"].to_numpy(), test["fused_probability"].to_numpy())
            run_rows.append({
                "seed": seed,
                "fold": fold,
                "validation_rows": len(role_data["VALID"]),
                "test_rows": len(test),
                "selected_alpha": alpha,
                "validation_fused_auprc": selection["micro_auprc"],
                "validation_fused_target_macro_auprc": selection["target_macro_auprc"],
                "validation_fused_drug_macro_auprc": selection["drug_macro_auprc"],
                "validation_three_metric_objective": selection["equal_weight_three_metric_objective"],
                "test_base_auprc": base_metrics["auprc"],
                "test_fused_auprc": fused_metrics["auprc"],
                "test_auprc_difference": fused_metrics["auprc"] - base_metrics["auprc"],
                "test_base_auroc": base_metrics["auroc"],
                "test_fused_auroc": fused_metrics["auroc"],
            })
            test["seed"] = seed
            test["fold"] = fold
            test["selected_alpha"] = alpha
            test_frames.append(test)
    runs = pd.DataFrame(run_rows)
    all_tests = pd.concat(test_frames, ignore_index=True)

    per_seed_rows = []
    for seed, frame in all_tests.groupby("seed", sort=True):
        base = metrics(frame["binary_label"].to_numpy(), frame["base_probability"].to_numpy())
        fused = metrics(frame["binary_label"].to_numpy(), frame["fused_probability"].to_numpy())
        per_seed_rows.append({
            "seed": int(seed),
            "rows": len(frame),
            "base_auprc": base["auprc"],
            "fused_auprc": fused["auprc"],
            "auprc_difference": fused["auprc"] - base["auprc"],
            "base_auroc": base["auroc"],
            "fused_auroc": fused["auroc"],
        })
    per_seed = pd.DataFrame(per_seed_rows)

    key_columns = [
        "calibration_pair_id", "binary_label", "target_chembl_id", "parent_molecule_chembl_id",
        "target_homology_cluster", "scaffold_group", "fold", STRUCTURE_COLUMN,
    ]
    ensemble = all_tests.groupby(key_columns, as_index=False).agg(
        base_ensemble=("base_probability", "mean"),
        fused_ensemble=("fused_probability", "mean"),
    )
    ensemble_base = metrics(ensemble["binary_label"].to_numpy(), ensemble["base_ensemble"].to_numpy())
    ensemble_fused = metrics(ensemble["binary_label"].to_numpy(), ensemble["fused_ensemble"].to_numpy())
    bootstrap_rows = [
        grouped_bootstrap(ensemble, "target_homology_cluster", args.iterations, 2026081301),
        grouped_bootstrap(ensemble, "scaffold_group", args.iterations, 2026081302),
    ]
    bootstrap = pd.DataFrame(bootstrap_rows)
    directional_rows = [
        directional_group_bootstrap(
            ensemble, "target_chembl_id", args.iterations, 2026081303
        ),
        directional_group_bootstrap(
            ensemble, "parent_molecule_chembl_id", args.iterations, 2026081304
        ),
    ]
    directional = pd.DataFrame(directional_rows)
    permutation_nulls = {
        "target_query": within_query_permutation_null(
            ensemble, "target_chembl_id", args.permutations, 2026081305
        ),
        "drug_query": within_query_permutation_null(
            ensemble, "parent_molecule_chembl_id", args.permutations, 2026081306
        ),
    }
    positive_seeds = int(per_seed["auprc_difference"].gt(0).sum())
    dual_ci_pass = bool(bootstrap["difference_ci95_low"].gt(0).all())
    target_direction = directional[
        directional["group_column"].eq("target_chembl_id")
    ].iloc[0]
    drug_direction = directional[
        directional["group_column"].eq("parent_molecule_chembl_id")
    ].iloc[0]
    target_noninferiority_margin = -0.002
    target_direction_noninferior = bool(
        target_direction["observed_macro_auprc_difference"] >= 0
        and target_direction["difference_ci95_low"] >= target_noninferiority_margin
    )
    drug_direction_increment = bool(drug_direction["difference_ci95_low"] > 0)
    drug_query_signal = bool(permutation_nulls["drug_query"]["pair_signal_supported"])
    stage0_pass = bool(
        positive_seeds >= 4
        and ensemble_fused["auprc"] > ensemble_base["auprc"]
        and dual_ci_pass
        and target_direction_noninferior
        and drug_direction_increment
        and drug_query_signal
        and not extraction["partial"]
    )
    decision = (
        "ADVANCE_TO_EQIR_STAGE1_FIVE_SEED_ABLATIONS"
        if stage0_pass
        else "STOP_STRUCTURE_ALGORITHM_LINE_STAGE0_INCREMENT_NOT_ESTABLISHED"
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "stage": "V3_STAGE0_LABEL_BLIND_STRUCTURE_UTILITY_SCREEN",
        "partial": bool(extraction["partial"]),
        "protocol": "S3_STRICT_DOUBLE_COLD",
        "development_folds": FOLDS,
        "confirmatory_folds_untouched": [3, 4],
        "structure_feature": STRUCTURE_COLUMN,
        "alpha_grid": {"minimum": 0.0, "maximum": 4.0, "step": 0.05},
        "alpha_selection_objective": "equal-weight mean of validation micro, target-macro and drug-macro AUPRC",
        "per_seed": per_seed.to_dict("records"),
        "positive_seed_count": positive_seeds,
        "five_seed_ensemble": {
            "rows": len(ensemble),
            "base_auprc": ensemble_base["auprc"],
            "fused_auprc": ensemble_fused["auprc"],
            "auprc_difference": ensemble_fused["auprc"] - ensemble_base["auprc"],
            "base_auroc": ensemble_base["auroc"],
            "fused_auroc": ensemble_fused["auroc"],
        },
        "grouped_bootstrap": bootstrap_rows,
        "directional_group_bootstrap": directional_rows,
        "within_query_permutation_nulls": permutation_nulls,
        "target_centric": {
            "base": grouped_direction_metrics(ensemble, "target_chembl_id", "base_ensemble"),
            "fused": grouped_direction_metrics(ensemble, "target_chembl_id", "fused_ensemble"),
        },
        "drug_centric_diagnostic": {
            "base": grouped_direction_metrics(
                ensemble, "parent_molecule_chembl_id", "base_ensemble"
            ),
            "fused": grouped_direction_metrics(
                ensemble, "parent_molecule_chembl_id", "fused_ensemble"
            ),
        },
        "promotion_gate": {
            "at_least_4_of_5_positive": positive_seeds >= 4,
            "ensemble_gain_positive": ensemble_fused["auprc"] > ensemble_base["auprc"],
            "both_grouped_bootstrap_ci95_low_above_zero": dual_ci_pass,
            "target_macro_difference_point_nonnegative_and_ci95_low_at_least_minus_0_002": target_direction_noninferior,
            "drug_macro_difference_ci95_low_above_zero": drug_direction_increment,
            "fused_drug_query_pair_signal_above_permutation_ci95_high": drug_query_signal,
            "full_queue_complete": not extraction["partial"],
        },
        "decision": decision,
        "algorithm_innovation_claim": "NOT_ESTABLISHED_BY_STAGE0_ALONE",
        "source_hashes": {
            "scores": sha256(args.scores),
            "extraction_summary": sha256(args.extraction_summary),
            "evaluation_ledger": sha256(args.ledger),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "EQIR_V3_STAGE0_PER_FOLD_RESULTS_V1.csv", index=False)
    per_seed.to_csv(args.output_dir / "EQIR_V3_STAGE0_PER_SEED_RESULTS_V1.csv", index=False)
    bootstrap.to_csv(args.output_dir / "EQIR_V3_STAGE0_GROUPED_BOOTSTRAP_V1.csv", index=False)
    directional.to_csv(
        args.output_dir / "EQIR_V3_STAGE0_DIRECTIONAL_BOOTSTRAP_V1.csv", index=False
    )
    ensemble.to_csv(
        args.output_dir / "EQIR_V3_STAGE0_ENSEMBLE_PREDICTIONS_V1.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 5, "mtime": 0},
    )
    (args.output_dir / "EQIR_V3_STAGE0_SUMMARY_V1.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

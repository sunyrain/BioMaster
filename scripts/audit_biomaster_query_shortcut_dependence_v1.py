#!/usr/bin/env python3
"""Diagnose whether apparent DTI performance is pair signal or query-group prior."""

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
BASE = ROOT / "outputs/old_drug_target_sota_v1"
PRED = BASE / "biomaster_odti_routed_ranker_v1/multiseed_evaluation_v1"
OUT = BASE / "query_shortcut_dependence_audit_v1"
FILES = {
    "S2_HOMOLOGY_COLD_TARGET": PRED / "S2_HOMOLOGY_COLD_TARGET_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz",
    "S3_STRICT_DOUBLE_COLD": PRED / "S3_STRICT_DOUBLE_COLD_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz",
    "S5_OLD_DRUG_ENTITY_COLD": PRED / "S5_OLD_DRUG_ENTITY_COLD_FIVE_SEED_ENSEMBLE_PREDICTIONS_V1.csv.gz",
}
ID_COLUMNS = {
    "calibration_pair_id",
    "binary_label",
    "target_chembl_id",
    "parent_standard_inchi_key",
    "target_homology_cluster",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def macro_ap(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> tuple[float, int]:
    frame = pd.DataFrame({"y": y, "score": score, "group": groups})
    values: list[float] = []
    for _, part in frame.groupby("group", sort=False, observed=True):
        labels = part["y"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        values.append(float(average_precision_score(labels, part["score"].to_numpy(dtype=float))))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def fast_macro_ap(
    y: np.ndarray, score: np.ndarray, indices: list[np.ndarray]
) -> tuple[float, int]:
    """Macro AP for continuous ranking scores without sklearn/groupby overhead."""
    total = 0.0
    evaluated = 0
    for positions in indices:
        labels = y[positions]
        positives = int(labels.sum())
        if positives == 0 or positives == len(labels):
            continue
        ranked = labels[np.argsort(-score[positions], kind="stable")]
        precision = np.cumsum(ranked, dtype=float) / np.arange(1, len(ranked) + 1)
        total += float(precision[ranked.astype(bool)].sum() / positives)
        evaluated += 1
    return (total / evaluated if evaluated else float("nan"), evaluated)


def metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float | int]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    target_ap, target_n = macro_ap(y, score, frame["target_chembl_id"].to_numpy())
    drug_ap, drug_n = macro_ap(y, score, frame["parent_standard_inchi_key"].to_numpy())
    return {
        "micro_auroc": float(roc_auc_score(y, score)),
        "micro_auprc": float(average_precision_score(y, score)),
        "target_macro_auprc": target_ap,
        "target_groups_with_both_classes": target_n,
        "drug_macro_auprc": drug_ap,
        "drug_groups_with_both_classes": drug_n,
    }


def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    series = pd.Series(np.arange(len(groups)), index=groups)
    return [part.to_numpy(dtype=np.int64) for _, part in series.groupby(level=0, sort=False)]


def permute_within(score: np.ndarray, indices: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    shuffled = score.copy()
    for positions in indices:
        if len(positions) > 1:
            shuffled[positions] = score[rng.permutation(positions)]
    return shuffled


def quantiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(data)),
        "ci95_low": float(np.quantile(data, 0.025)),
        "ci95_high": float(np.quantile(data, 0.975)),
    }


def score_columns(frame: pd.DataFrame) -> list[str]:
    permitted = {
        "CONPLEX_FROZEN_EXTERNAL",
        "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO",
        "scope_mean",
        "BIOMASTER_RAW_FIVE_SEED_MEAN",
        "BIOMASTER_STACK_FIVE_SEED_MEAN",
    }
    result = []
    for column in frame.columns:
        if column in ID_COLUMNS or column.endswith("_STD") or column not in permitted:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            result.append(column)
    return result


def run_one(protocol: str, path: Path, iterations: int, seed: int) -> list[dict[str, object]]:
    frame = pd.read_csv(path, low_memory=False)
    if not frame["calibration_pair_id"].is_unique:
        raise RuntimeError(f"Duplicate pair IDs in {path}")
    if frame["binary_label"].nunique() != 2:
        raise RuntimeError(f"Both labels required in {path}")
    target = frame["target_chembl_id"].to_numpy()
    drug = frame["parent_standard_inchi_key"].to_numpy()
    target_indices = group_indices(target)
    drug_indices = group_indices(drug)
    rows: list[dict[str, object]] = []

    for model_number, column in enumerate(score_columns(frame)):
        score = frame[column].to_numpy(dtype=float)
        if not np.isfinite(score).all():
            raise RuntimeError(f"Non-finite score in {protocol}/{column}")
        raw = metrics(frame, score)
        target_mean = pd.Series(score).groupby(target, sort=False).transform("mean").to_numpy()
        drug_mean = pd.Series(score).groupby(drug, sort=False).transform("mean").to_numpy()
        y = frame["binary_label"].to_numpy(dtype=np.int8)
        target_mean_micro_ap = float(average_precision_score(y, target_mean))
        drug_mean_micro_ap = float(average_precision_score(y, drug_mean))
        target_centered_micro_ap = float(average_precision_score(y, score - target_mean))
        drug_centered_micro_ap = float(average_precision_score(y, score - drug_mean))

        rng = np.random.default_rng(seed + 1009 * model_number + 100_003 * list(FILES).index(protocol))
        target_perm_micro_ap: list[float] = []
        target_perm_target_macro_ap: list[float] = []
        drug_perm_micro_ap: list[float] = []
        drug_perm_drug_macro_ap: list[float] = []
        for _ in range(iterations):
            shuffled_target = permute_within(score, target_indices, rng)
            target_perm_micro_ap.append(float(average_precision_score(y, shuffled_target)))
            target_perm_target_macro_ap.append(float(fast_macro_ap(y, shuffled_target, target_indices)[0]))

            shuffled_drug = permute_within(score, drug_indices, rng)
            drug_perm_micro_ap.append(float(average_precision_score(y, shuffled_drug)))
            drug_perm_drug_macro_ap.append(float(fast_macro_ap(y, shuffled_drug, drug_indices)[0]))

        target_perm_micro = quantiles(target_perm_micro_ap)
        target_perm_directional = quantiles(target_perm_target_macro_ap)
        drug_perm_micro = quantiles(drug_perm_micro_ap)
        drug_perm_directional = quantiles(drug_perm_drug_macro_ap)
        rows.append(
            {
                "protocol": protocol,
                "model": column,
                "rows": len(frame),
                "positives": int(frame["binary_label"].sum()),
                "prevalence": float(frame["binary_label"].mean()),
                **{f"raw_{key}": value for key, value in raw.items()},
                "target_mean_only_micro_auprc": target_mean_micro_ap,
                "drug_mean_only_micro_auprc": drug_mean_micro_ap,
                "target_centered_micro_auprc": target_centered_micro_ap,
                "drug_centered_micro_auprc": drug_centered_micro_ap,
                **{f"within_target_permutation_micro_auprc_{key}": value for key, value in target_perm_micro.items()},
                **{f"within_target_permutation_target_macro_auprc_{key}": value for key, value in target_perm_directional.items()},
                **{f"within_drug_permutation_micro_auprc_{key}": value for key, value in drug_perm_micro.items()},
                **{f"within_drug_permutation_drug_macro_auprc_{key}": value for key, value in drug_perm_directional.items()},
                "target_query_pair_signal_supported": bool(
                    raw["target_macro_auprc"] > target_perm_directional["ci95_high"]
                ),
                "drug_query_pair_signal_supported": bool(
                    raw["drug_macro_auprc"] > drug_perm_directional["ci95_high"]
                ),
                "micro_score_exceeds_both_group_preserving_nulls": bool(
                    raw["micro_auprc"] > target_perm_micro["ci95_high"]
                    and raw["micro_auprc"] > drug_perm_micro["ci95_high"]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.iterations < 20:
        raise ValueError("At least 20 permutations are required")

    all_rows: list[dict[str, object]] = []
    for protocol, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        all_rows.extend(run_one(protocol, path, args.iterations, args.seed))
    result = pd.DataFrame(all_rows).sort_values(["protocol", "model"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "QUERY_SHORTCUT_DEPENDENCE_RESULTS_V1.csv"
    result.to_csv(result_path, index=False)

    primary = result[result["model"].eq("BIOMASTER_STACK_FIVE_SEED_MEAN")]
    checks = {
        "three_protocols_present": set(result["protocol"]) == set(FILES),
        "all_primary_protocols_present": set(primary["protocol"]) == set(FILES),
        "all_scores_and_null_intervals_finite": np.isfinite(
            result.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all(),
    }
    findings = {
        "all_primary_target_query_pair_signal_supported": bool(
            primary["target_query_pair_signal_supported"].all()
        ),
        "all_primary_drug_query_pair_signal_supported_when_evaluable": bool(
            primary.loc[
                primary["raw_drug_groups_with_both_classes"].gt(0),
                "drug_query_pair_signal_supported",
            ].all()
        ),
        "primary_protocols_without_supported_drug_query_pair_signal": primary.loc[
            ~primary["drug_query_pair_signal_supported"], "protocol"
        ].tolist(),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "purpose": "diagnostic only: separate pair-level ranking signal from target/drug query-group priors without fitting on test labels",
        "iterations": args.iterations,
        "seed": args.seed,
        "transforms_read_test_labels": False,
        "interpretation": {
            "within_target_permutation": "preserves each target's score distribution and between-target score prior while destroying drug ranking within target",
            "within_drug_permutation": "preserves each drug's score distribution and between-drug score prior while destroying target ranking within drug",
            "mean_only": "retains only label-blind group-level score prior",
            "centered": "removes the label-blind group-level mean and retains within-query ranking",
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "scientific_findings": findings,
        "primary_biomaster": primary.to_dict(orient="records"),
        "artifacts": {
            "results": str(result_path.relative_to(ROOT)),
            "results_sha256": sha256(result_path),
            "input_sha256": {protocol: sha256(path) for protocol, path in FILES.items()},
        },
        "claim_boundary": "This audit can falsify query-pair signal but cannot establish SOTA or causal debiasing; grouped bootstrap and same-data public baselines remain required.",
    }
    summary_path = OUT / "QUERY_SHORTCUT_DEPENDENCE_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-fit a query-balanced DTIAM/BioMaster blend on cold-protocol folds 0--2."""

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
BASE = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_biomaster_cold_fusion_v1"
PROTOCOLS = {
    "S2_HOMOLOGY_COLD_TARGET": "S2",
    "S3_STRICT_DOUBLE_COLD": "S3",
}
FOLDS = [0, 1, 2]
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.025), 3)
DTIAM = "dtiam_probability"
BIOMASTER = "biomaster_stack_score"
CANDIDATE = "query_balanced_blend"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(protocol: str, fold: int) -> Path:
    return BASE / (
        f"{protocol}__fold_{fold}/"
        "DTIAM_BIOMASTER_COLD_FUSION_PREDICTIONS_V1.csv.gz"
    )


def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    series = pd.Series(np.arange(len(groups)), index=groups)
    return [part.to_numpy(dtype=np.int64) for _, part in series.groupby(level=0, sort=False)]


def fast_macro_ap(y: np.ndarray, score: np.ndarray, positions: list[np.ndarray]) -> tuple[float, int]:
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
    return (total / evaluated if evaluated else float("nan"), evaluated)


def exact_macro_ap(y: np.ndarray, score: np.ndarray, positions: list[np.ndarray]) -> tuple[float, int]:
    values = []
    for index in positions:
        labels = y[index]
        if labels.min() == labels.max():
            continue
        values.append(float(average_precision_score(labels, score[index])))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float | int]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    target = group_indices(frame["target_chembl_id"].to_numpy())
    drug = group_indices(frame["parent_standard_inchi_key"].to_numpy())
    target_ap, target_n = exact_macro_ap(y, score, target)
    drug_ap, drug_n = exact_macro_ap(y, score, drug)
    return {
        "micro_auroc": float(roc_auc_score(y, score)),
        "micro_auprc": float(average_precision_score(y, score)),
        "target_macro_auprc": target_ap,
        "target_groups_with_both_classes": target_n,
        "drug_macro_auprc": drug_ap,
        "drug_groups_with_both_classes": drug_n,
    }


def tune(train_parts: list[pd.DataFrame]) -> tuple[float, list[dict[str, float]]]:
    frame = pd.concat(train_parts, ignore_index=True)
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    dtiam = frame[DTIAM].to_numpy(dtype=float)
    biomaster = frame[BIOMASTER].to_numpy(dtype=float)
    target = group_indices(frame["target_chembl_id"].to_numpy())
    drug = group_indices(frame["parent_standard_inchi_key"].to_numpy())
    rows = []
    for weight in WEIGHTS:
        score = float(weight) * dtiam + (1.0 - float(weight)) * biomaster
        micro = float(average_precision_score(y, score))
        target_macro = fast_macro_ap(y, score, target)[0]
        drug_macro = fast_macro_ap(y, score, drug)[0]
        rows.append({
            "dtiam_weight": float(weight),
            "biomaster_weight": float(1.0 - weight),
            "micro_auprc": micro,
            "target_macro_auprc": target_macro,
            "drug_macro_auprc": drug_macro,
            "equal_weight_three_metric_objective": float((micro + target_macro + drug_macro) / 3.0),
        })
    best = sorted(
        rows,
        key=lambda row: (-row["equal_weight_three_metric_objective"], row["dtiam_weight"]),
    )[0]
    return float(best["dtiam_weight"]), rows


def permutation_null(
    frame: pd.DataFrame,
    score_column: str,
    group_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float | bool]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    score = frame[score_column].to_numpy(dtype=float)
    positions = group_indices(frame[group_column].to_numpy())
    observed = fast_macro_ap(y, score, positions)[0]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(iterations):
        shuffled = score.copy()
        for index in positions:
            if len(index) > 1:
                shuffled[index] = score[rng.permutation(index)]
        null.append(fast_macro_ap(y, shuffled, positions)[0])
    low, high = np.quantile(np.asarray(null), [0.025, 0.975])
    return {
        "observed_macro_auprc": float(observed),
        "null_mean": float(np.mean(null)),
        "null_ci95_low": float(low),
        "null_ci95_high": float(high),
        "pair_signal_supported": bool(observed > high),
    }


def directional_bootstrap(
    frame: pd.DataFrame,
    challenger: str,
    reference: str,
    group_column: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    differences = []
    for _, part in frame.groupby(group_column, sort=False, observed=True):
        y = part["binary_label"].to_numpy(dtype=np.int8)
        if y.min() == y.max():
            continue
        differences.append(
            float(average_precision_score(y, part[challenger]))
            - float(average_precision_score(y, part[reference]))
        )
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "reference": reference,
        "direction_group": group_column,
        "groups": int(len(values)),
        "observed_macro_auprc_difference": float(values.mean()),
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "probability_difference_gt_zero": float((boot > 0).mean()),
        "ci95_excludes_zero_in_favor_of_candidate": bool(low > 0),
    }


def clustered_micro_bootstrap(
    frame: pd.DataFrame,
    challenger: str,
    reference: str,
    cluster_column: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    challenger_score = frame[challenger].to_numpy(dtype=float)
    reference_score = frame[reference].to_numpy(dtype=float)
    groups = frame[cluster_column].astype(str).to_numpy()
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([positions[group] for group in sampled])
        labels = y[index]
        if labels.min() == labels.max():
            continue
        differences.append(
            float(average_precision_score(labels, challenger_score[index]))
            - float(average_precision_score(labels, reference_score[index]))
        )
    values = np.asarray(differences, dtype=float)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "reference": reference,
        "cluster_column": cluster_column,
        "clusters": int(len(unique)),
        "observed_micro_auprc_difference": float(
            average_precision_score(y, challenger_score) - average_precision_score(y, reference_score)
        ),
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "probability_difference_gt_zero": float((values > 0).mean()),
        "ci95_excludes_zero_in_favor_of_candidate": bool(low > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="S3_STRICT_DOUBLE_COLD")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if args.permutations < 50 or args.bootstrap_iterations < 500:
        raise ValueError("Insufficient audit iterations")

    folds = {}
    hashes = {}
    for fold in FOLDS:
        path = source_path(args.protocol, fold)
        part = pd.read_csv(path, low_memory=False)
        part["fold"] = fold
        folds[fold] = part
        hashes[str(path.relative_to(ROOT))] = sha256(path)

    held_parts = []
    tuning_rows = []
    selected = {}
    for held_fold in FOLDS:
        weight, rows = tune([folds[fold] for fold in FOLDS if fold != held_fold])
        selected[str(held_fold)] = weight
        for row in rows:
            tuning_rows.append({"held_fold": held_fold, **row})
        held = folds[held_fold].copy()
        held[CANDIDATE] = weight * held[DTIAM] + (1.0 - weight) * held[BIOMASTER]
        held["selected_dtiam_weight"] = weight
        held_parts.append(held)
    crossfit = pd.concat(held_parts, ignore_index=True).sort_values("calibration_pair_id")
    if not crossfit["calibration_pair_id"].is_unique:
        raise RuntimeError("Cross-fit development pairs overlap")

    model_metrics = {
        "DTIAM": metrics(crossfit, crossfit[DTIAM].to_numpy(dtype=float)),
        "BIOMASTER_STACK": metrics(crossfit, crossfit[BIOMASTER].to_numpy(dtype=float)),
        "QUERY_BALANCED_BLEND": metrics(crossfit, crossfit[CANDIDATE].to_numpy(dtype=float)),
    }
    nulls = {
        "target_query": permutation_null(
            crossfit, CANDIDATE, "target_chembl_id", args.permutations, args.seed
        ),
        "drug_query": permutation_null(
            crossfit, CANDIDATE, "parent_standard_inchi_key", args.permutations, args.seed + 1
        ),
    }
    directional = []
    clustered = []
    for reference_number, reference in enumerate([DTIAM, BIOMASTER]):
        for direction_number, group in enumerate(
            ["target_chembl_id", "parent_standard_inchi_key"]
        ):
            directional.append(
                directional_bootstrap(
                    crossfit, CANDIDATE, reference, group, args.bootstrap_iterations,
                    args.seed + 100 * reference_number + direction_number,
                )
            )
        for cluster_number, cluster in enumerate(["target_homology_cluster", "scaffold_group"]):
            clustered.append(
                clustered_micro_bootstrap(
                    crossfit, CANDIDATE, reference, cluster, args.bootstrap_iterations,
                    args.seed + 1000 * reference_number + cluster_number,
                )
            )

    protocol_token = PROTOCOLS[args.protocol]
    out = ROOT / f"outputs/old_drug_target_sota_v1/{protocol_token.lower()}_query_balanced_blend_dev_v1"
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = out / f"{protocol_token}_DEV_FOLDS_0_2_QUERY_BALANCED_BLEND_PREDICTIONS_V1.csv.gz"
    tuning_path = out / f"{protocol_token}_DEV_FOLDS_0_2_QUERY_BALANCED_BLEND_TUNING_V1.csv"
    directional_path = out / f"{protocol_token}_QUERY_BALANCED_BLEND_DIRECTIONAL_BOOTSTRAP_V1.csv"
    clustered_path = out / f"{protocol_token}_QUERY_BALANCED_BLEND_CLUSTERED_MICRO_BOOTSTRAP_V1.csv"
    crossfit.to_csv(
        prediction_path, index=False,
        compression={"method": "gzip", "compresslevel": 5, "mtime": 0},
    )
    pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)
    pd.DataFrame(directional).to_csv(directional_path, index=False)
    pd.DataFrame(clustered).to_csv(clustered_path, index=False)

    candidate = model_metrics["QUERY_BALANCED_BLEND"]
    dtiam = model_metrics["DTIAM"]
    biomaster = model_metrics["BIOMASTER_STACK"]
    gates = {
        "micro_exceeds_both_references": candidate["micro_auprc"] > max(
            dtiam["micro_auprc"], biomaster["micro_auprc"]
        ),
        "target_macro_exceeds_both_references": candidate["target_macro_auprc"] > max(
            dtiam["target_macro_auprc"], biomaster["target_macro_auprc"]
        ),
        "drug_macro_exceeds_both_references": candidate["drug_macro_auprc"] > max(
            dtiam["drug_macro_auprc"], biomaster["drug_macro_auprc"]
        ),
        "target_query_pair_signal_supported": nulls["target_query"]["pair_signal_supported"],
        "drug_query_pair_signal_supported": nulls["drug_query"]["pair_signal_supported"],
        "all_directional_bootstrap_lowers_above_zero": all(
            row["ci95_excludes_zero_in_favor_of_candidate"] for row in directional
        ),
        "all_clustered_micro_bootstrap_lowers_above_zero": all(
            row["ci95_excludes_zero_in_favor_of_candidate"] for row in clustered
        ),
    }
    integrity = {
        "only_folds_0_2_read": set(crossfit["fold"].unique()) == set(FOLDS),
        "unique_crossfit_pairs": bool(crossfit["calibration_pair_id"].is_unique),
        "selected_weights_in_grid": all(value in set(WEIGHTS) for value in selected.values()),
        "all_scores_finite": bool(
            np.isfinite(crossfit[[DTIAM, BIOMASTER, CANDIDATE]].to_numpy(dtype=float)).all()
        ),
    }
    decision = (
        "ADVANCE_AS_STRONG_NONNOVEL_BASELINE"
        if all(gates.values())
        else "RETAIN_AS_NONNOVEL_BASELINE; DO_NOT_USE_AS_EQIR_CORE"
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(integrity.values()) else "FAIL",
        "protocol": args.protocol,
        "scope": f"{protocol_token} development folds 0-2 only; each held fold scored by a weight selected on the other two folds",
        "selection_objective": "equal-weight mean of micro AUPRC, target-macro AUPRC and drug-macro AUPRC",
        "weight_grid": {"minimum": 0.0, "maximum": 1.0, "step": 0.025},
        "selected_dtiam_weight_by_held_fold": selected,
        "metrics": model_metrics,
        "candidate_query_permutation_nulls": nulls,
        "directional_paired_bootstrap": directional,
        "clustered_micro_bootstrap": clustered,
        "integrity_checks": {key: bool(value) for key, value in integrity.items()},
        "scientific_gates": {key: bool(value) for key, value in gates.items()},
        "decision": decision,
        "novelty_boundary": (
            "A convex validation/cross-fold score blend is a strong calibration baseline, not an algorithm novelty. "
            "EQIR must exceed it under the same directional and clustered gates."
        ),
        "artifacts": {
            "predictions": str(prediction_path.relative_to(ROOT)),
            "predictions_sha256": sha256(prediction_path),
            "tuning": str(tuning_path.relative_to(ROOT)),
            "tuning_sha256": sha256(tuning_path),
            "directional_bootstrap": str(directional_path.relative_to(ROOT)),
            "directional_bootstrap_sha256": sha256(directional_path),
            "clustered_micro_bootstrap": str(clustered_path.relative_to(ROOT)),
            "clustered_micro_bootstrap_sha256": sha256(clustered_path),
            "input_sha256": hashes,
        },
    }
    summary_path = out / f"{protocol_token}_QUERY_BALANCED_BLEND_DEV_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(integrity, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post-freeze external evaluation of the V10 leakage-safe old-drug ranker.

KiRHub is a labelled external benchmark previously used by the project. BindingDB
is a positive-only retrospective audit. Neither source is read until the frozen
V10 prediction hash has been verified.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V10 = RUN / "leakage_safe_ranker_v10"
PREDICTIONS = V10 / "OLD_DRUG_LEAKAGE_SAFE_ALL_276480_V10.csv.gz"
MODEL_SUMMARY = V10 / "OLD_DRUG_LEAKAGE_SAFE_SUMMARY_V10.json"
V8_EXTERNAL = (
    RUN / "old_drug_advanced_ranker_v8/external_evaluation/"
    "OLD_DRUG_ADVANCED_RANKER_EXTERNAL_KIRHUB_V8.csv"
)
BINDINGDB = RUN / "final_evidence_routing_v9/BINDINGDB_PAIR_POSITIVE_RETRIEVAL_V9.csv"
OUT = V10 / "external_evaluation"

SEED = 20260813
BOOTSTRAPS = 1000
PERMUTATIONS = 10000

SCORES = {
    "V10_LEAKAGE_SAFE_KNOWN_GRAPH": "old_drug_leakage_safe_score_v10",
    "V8_PREVIOUS_GRAPH_FUSION": "old_drug_advanced_ranker_score_v8",
    "V7_NEW_TARGET_RANKER": "v7_new_target_score",
    "V7_RELATION_RANKER": "v7_relation_score",
    "FROZEN_DTA_CONSENSUS": "dta_cross_target_consensus_score",
    "CONPLEX_DRUG_CENTRIC": "conplex_percentile_within_ligand_384",
    "DRUGCLIP_DRUG_CENTRIC": "drugclip_percentile_within_ligand_382",
}

BINDING_RANKS = {
    "V10_LEAKAGE_SAFE_KNOWN_GRAPH": "old_drug_leakage_safe_rank_within_drug_384_v10",
    "V8_PREVIOUS_GRAPH_FUSION": "old_drug_advanced_rank_within_drug_384_v8",
    "V7_NEW_TARGET_RANKER": "old_drug_new_target_rank_within_drug_384_v7",
    "FROZEN_DTA_CONSENSUS": "dta_consensus_rank_within_drug_384_v7",
    "CONPLEX_DRUG_CENTRIC": "conplex_rank_within_ligand_384",
    "DRUGCLIP_DRUG_CENTRIC": "drugclip_rank_within_ligand_382",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieval_metrics(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    x = frame[["kirhub_compound", score, label]].dropna().copy()
    y = x[label].astype(int)
    result: dict[str, Any] = {
        "pairs": int(len(x)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "micro_auroc": float(roc_auc_score(y, x[score])) if y.nunique() == 2 else None,
        "micro_auprc": float(average_precision_score(y, x[score])) if y.nunique() == 2 else None,
    }
    aucs, aps = [], []
    recall = {5: [], 10: [], 20: []}
    precision = {5: [], 10: [], 20: []}
    for _, group in x.groupby("kirhub_compound", sort=False):
        yy = group[label].astype(int)
        if yy.nunique() == 2:
            aucs.append(roc_auc_score(yy, group[score]))
            aps.append(average_precision_score(yy, group[score]))
        positives = int(yy.sum())
        if not positives:
            continue
        ranked = group.sort_values(score, ascending=False, kind="mergesort")
        for k in recall:
            kk = min(k, len(ranked))
            hits = int(ranked.head(kk)[label].sum())
            recall[k].append(hits / positives)
            precision[k].append(hits / kk)
    result["macro_drug_auroc"] = float(np.mean(aucs)) if aucs else None
    result["macro_drug_auprc"] = float(np.mean(aps)) if aps else None
    result["drugs_with_two_classes"] = len(aucs)
    for k in recall:
        result[f"macro_recall_at_{k}"] = float(np.mean(recall[k])) if recall[k] else None
        result[f"macro_precision_at_{k}"] = float(np.mean(precision[k])) if precision[k] else None
    return result


def kirhub_cluster_bootstrap(frame: pd.DataFrame, label: str, slice_name: str) -> pd.DataFrame:
    compounds = np.asarray(sorted(frame["kirhub_compound"].unique()), dtype=object)
    compound_index = {compound: index for index, compound in enumerate(compounds)}
    compound_codes = frame["kirhub_compound"].map(compound_index).to_numpy(dtype=int)
    y = frame[label].astype(int).to_numpy()
    rng = np.random.default_rng(SEED)
    metric_names = ("micro_auroc", "micro_auprc", "macro_recall_at_20")
    point = {
        (name, metric): float(retrieval_metrics(frame, score, label)[metric])
        for name, score in SCORES.items() for metric in metric_names
    }
    score_arrays = {name: frame[score].to_numpy(dtype=float) for name, score in SCORES.items()}
    recall20: dict[str, np.ndarray] = {}
    positive_drug = np.zeros(len(compounds), dtype=bool)
    for name, score in SCORES.items():
        per_drug = np.full(len(compounds), np.nan, dtype=float)
        for compound, group in frame.groupby("kirhub_compound", sort=False):
            index = compound_index[compound]
            positives = int(group[label].sum())
            if positives:
                positive_drug[index] = True
                ranked = group.sort_values(score, ascending=False, kind="mergesort")
                per_drug[index] = float(ranked.head(min(20, len(ranked)))[label].sum() / positives)
        recall20[name] = per_drug

    samples = {(name, metric): [] for name in SCORES for metric in metric_names}
    differences = {
        (baseline, metric): []
        for baseline in SCORES if baseline != "V10_LEAKAGE_SAFE_KNOWN_GRAPH"
        for metric in metric_names
    }
    for _ in range(BOOTSTRAPS):
        counts = rng.multinomial(len(compounds), np.full(len(compounds), 1.0 / len(compounds)))
        row_weights = counts[compound_codes].astype(float)
        replicate: dict[tuple[str, str], float] = {}
        for name in SCORES:
            values = {
                "micro_auroc": float(roc_auc_score(y, score_arrays[name], sample_weight=row_weights)),
                "micro_auprc": float(average_precision_score(y, score_arrays[name], sample_weight=row_weights)),
                "macro_recall_at_20": float(
                    np.average(recall20[name][positive_drug], weights=counts[positive_drug])
                ),
            }
            for metric, value in values.items():
                samples[(name, metric)].append(value)
                replicate[(name, metric)] = value
        for baseline in SCORES:
            if baseline == "V10_LEAKAGE_SAFE_KNOWN_GRAPH":
                continue
            for metric in metric_names:
                differences[(baseline, metric)].append(
                    replicate[("V10_LEAKAGE_SAFE_KNOWN_GRAPH", metric)]
                    - replicate[(baseline, metric)]
                )

    rows = []
    for name in SCORES:
        for metric in metric_names:
            values = np.asarray(samples[(name, metric)])
            rows.append({
                "evaluation_slice": slice_name,
                "comparison": name,
                "metric": metric,
                "point_estimate": point[(name, metric)],
                "bootstrap_mean": float(values.mean()),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "cluster_unit": "kirhub_compound",
                "bootstrap_replicates": BOOTSTRAPS,
                "seed": SEED,
            })
    for baseline in SCORES:
        if baseline == "V10_LEAKAGE_SAFE_KNOWN_GRAPH":
            continue
        for metric in metric_names:
            values = np.asarray(differences[(baseline, metric)])
            rows.append({
                "evaluation_slice": slice_name,
                "comparison": f"V10_LEAKAGE_SAFE_KNOWN_GRAPH_MINUS_{baseline}",
                "metric": metric,
                "point_estimate": point[("V10_LEAKAGE_SAFE_KNOWN_GRAPH", metric)] - point[(baseline, metric)],
                "bootstrap_mean": float(values.mean()),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "cluster_unit": "kirhub_compound",
                "bootstrap_replicates": BOOTSTRAPS,
                "seed": SEED,
            })
    return pd.DataFrame(rows)


def positive_only_metrics(frame: pd.DataFrame, rank_column: str) -> dict[str, Any]:
    ranks = frame[rank_column].astype(float)
    denominator = np.where(rank_column.endswith("_382"), 381.0, 383.0)
    percentile = 1.0 - (ranks.to_numpy() - 1.0) / denominator
    return {
        "pairs": int(len(frame)),
        "drugs": int(frame["ligand_inchikey"].nunique()),
        "median_rank": float(ranks.median()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "mean_rank_percentile": float(np.mean(percentile)),
        **{f"recall_at_{k}": float(ranks.le(k).mean()) for k in (5, 10, 20, 50)},
    }


def bindingdb_permutation(frame: pd.DataFrame, observed: dict[str, Any]) -> pd.DataFrame:
    """Random-rank null preserving the number of positives per drug."""
    group_sizes = frame.groupby("ligand_inchikey", sort=True).size().to_numpy(dtype=int)
    rng = np.random.default_rng(SEED)
    null = {"mean_reciprocal_rank": [], "mean_rank_percentile": [], "recall_at_20": []}
    for _ in range(PERMUTATIONS):
        ranks = np.concatenate([
            rng.choice(np.arange(1, 385), size=size, replace=False) for size in group_sizes
        ]).astype(float)
        null["mean_reciprocal_rank"].append(float(np.mean(1.0 / ranks)))
        null["mean_rank_percentile"].append(float(np.mean(1.0 - (ranks - 1.0) / 383.0)))
        null["recall_at_20"].append(float(np.mean(ranks <= 20)))
    rows = []
    for metric, values_list in null.items():
        values = np.asarray(values_list)
        value = float(observed[metric])
        rows.append({
            "model_name": "V10_LEAKAGE_SAFE_KNOWN_GRAPH",
            "metric": metric,
            "observed": value,
            "null_mean": float(values.mean()),
            "null_ci95_lower": float(np.quantile(values, 0.025)),
            "null_ci95_upper": float(np.quantile(values, 0.975)),
            "one_sided_permutation_p": float((1 + np.sum(values >= value)) / (PERMUTATIONS + 1)),
            "permutation_replicates": PERMUTATIONS,
            "seed": SEED,
        })
    return pd.DataFrame(rows)


def main() -> None:
    pre_external = [PREDICTIONS, MODEL_SUMMARY]
    missing = [str(path) for path in pre_external if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    summary = json.loads(MODEL_SUMMARY.read_text())
    expected = summary["outputs"][str(PREDICTIONS.relative_to(ROOT))]
    if summary["status"] != "PASS" or sha256(PREDICTIONS) != expected:
        raise RuntimeError("V10 predictions are not a valid frozen artifact")
    if "No KiRHub or BindingDB file is read" not in summary["external_firewall"]:
        raise RuntimeError("V10 training firewall declaration is missing")

    # External files are intentionally touched only after the frozen hash check.
    external_inputs = [V8_EXTERNAL, BINDINGDB]
    missing = [str(path) for path in external_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PREDICTIONS, low_memory=False)

    kirhub = pd.read_csv(V8_EXTERNAL, low_memory=False).merge(
        pred[["pairId", "old_drug_leakage_safe_score_v10", "old_drug_leakage_safe_rank_within_drug_384_v10"]],
        on="pairId", how="left", validate="one_to_one",
    )
    if len(kirhub) != 8058 or kirhub[list(SCORES.values())].isna().any().any():
        raise RuntimeError("KiRHub merge incomplete")
    kirhub["v10_rank_within_measured_kirhub_targets"] = kirhub.groupby("kirhub_compound")[
        "old_drug_leakage_safe_score_v10"
    ].rank(method="min", ascending=False)

    slices = [
        ("ALL_KIRHUB_ACTIVE", kirhub, "kirhub_wt_active_le30pct_residual"),
        (
            "STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE",
            kirhub[kirhub["kirhub_frozen_unreported_scope"].astype(bool)].copy(),
            "kirhub_local_unreported_active",
        ),
    ]
    metric_rows, bootstrap_tables = [], []
    for slice_name, frame, label in slices:
        for model_name, score in SCORES.items():
            metric_rows.append({
                "evaluation_slice": slice_name,
                "model_name": model_name,
                "score_column": score,
                **retrieval_metrics(frame, score, label),
            })
        bootstrap_tables.append(kirhub_cluster_bootstrap(frame, label, slice_name))
    kirhub_metrics = pd.DataFrame(metric_rows)
    kirhub_bootstrap = pd.concat(bootstrap_tables, ignore_index=True)

    binding = pd.read_csv(BINDINGDB, low_memory=False).merge(
        pred[["pairId", "old_drug_leakage_safe_score_v10", "old_drug_leakage_safe_rank_within_drug_384_v10"]],
        on="pairId", how="left", validate="one_to_one",
    )
    if binding["old_drug_leakage_safe_rank_within_drug_384_v10"].isna().any():
        raise RuntimeError("BindingDB merge incomplete")
    binding_slices = {
        "ALL_EXACT_BINDINGDB_PAIRS": np.ones(len(binding), dtype=bool),
        "SINGLE_CHAIN_QUANTITATIVE_SUPPORT_LE10UM": binding[
            "bindingdb_strict_single_chain_support_le10um_any"
        ].astype(bool),
        "STRICT_CH37_UNREPORTED_SINGLE_CHAIN_SUPPORT_LE10UM": binding[
            "strict_positive_only_evaluation_v9"
        ].astype(bool),
    }
    binding_rows = []
    for slice_name, mask in binding_slices.items():
        frame = binding.loc[mask].copy()
        for model_name, rank_column in BINDING_RANKS.items():
            binding_rows.append({
                "evaluation_slice": slice_name,
                "model_name": model_name,
                "rank_column": rank_column,
                **positive_only_metrics(frame, rank_column),
            })
    binding_metrics = pd.DataFrame(binding_rows)
    strict_binding = binding.loc[binding_slices["STRICT_CH37_UNREPORTED_SINGLE_CHAIN_SUPPORT_LE10UM"]].copy()
    v10_binding = positive_only_metrics(strict_binding, BINDING_RANKS["V10_LEAKAGE_SAFE_KNOWN_GRAPH"])
    binding_permutation = bindingdb_permutation(strict_binding, v10_binding)

    kirhub_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_V10.csv"
    kirhub_metrics_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_METRICS_V10.csv"
    kirhub_bootstrap_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_BOOTSTRAP_V10.csv"
    binding_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_BINDINGDB_RETRIEVAL_V10.csv"
    binding_metrics_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_BINDINGDB_METRICS_V10.csv"
    binding_permutation_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_BINDINGDB_PERMUTATION_V10.csv"
    kirhub.to_csv(kirhub_path, index=False)
    kirhub_metrics.to_csv(kirhub_metrics_path, index=False)
    kirhub_bootstrap.to_csv(kirhub_bootstrap_path, index=False)
    binding.to_csv(binding_path, index=False)
    binding_metrics.to_csv(binding_metrics_path, index=False)
    binding_permutation.to_csv(binding_permutation_path, index=False)

    strict_k = kirhub_metrics[
        kirhub_metrics["evaluation_slice"].eq("STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE")
    ]
    v10_k = strict_k[strict_k["model_name"].eq("V10_LEAKAGE_SAFE_KNOWN_GRAPH")].iloc[0]
    v8_k = strict_k[strict_k["model_name"].eq("V8_PREVIOUS_GRAPH_FUSION")].iloc[0]
    v10_v8_diff = kirhub_bootstrap[
        kirhub_bootstrap["evaluation_slice"].eq("STRICT_FROZEN_UNREPORTED_KIRHUB_ACTIVE")
        & kirhub_bootstrap["comparison"].eq(
            "V10_LEAKAGE_SAFE_KNOWN_GRAPH_MINUS_V8_PREVIOUS_GRAPH_FUSION"
        )
    ]
    checks = {
        "v10_frozen_hash_verified_before_external_reads": sha256(PREDICTIONS) == expected,
        "exactly_8058_kirhub_pairs": len(kirhub) == 8058,
        "strict_kirhub_exactly_2823_pairs_202_positives": (
            len(slices[1][1]) == 2823 and int(slices[1][1][slices[1][2]].sum()) == 202
        ),
        "all_seven_kirhub_models_complete": kirhub[list(SCORES.values())].notna().all().all(),
        "all_1000_drug_cluster_bootstraps_complete": kirhub_bootstrap["bootstrap_replicates"].eq(BOOTSTRAPS).all(),
        "bindingdb_strict_exactly_11_pairs_8_drugs": len(strict_binding) == 11 and strict_binding["ligand_inchikey"].nunique() == 8,
        "bindingdb_v10_ranks_complete": binding["old_drug_leakage_safe_rank_within_drug_384_v10"].notna().all(),
        "all_10000_bindingdb_permutations_complete": binding_permutation["permutation_replicates"].eq(PERMUTATIONS).all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    output_paths = [
        kirhub_path, kirhub_metrics_path, kirhub_bootstrap_path,
        binding_path, binding_metrics_path, binding_permutation_path,
    ]
    external_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "model": summary["selected_architecture"],
        "external_firewall": "Frozen V10 prediction SHA-256 was verified before KiRHub or BindingDB was opened.",
        "strict_kirhub": {
            "pairs": 2823,
            "positives": 202,
            "v10": {key: float(v10_k[key]) for key in [
                "micro_auroc", "micro_auprc", "macro_drug_auroc", "macro_drug_auprc",
                "macro_recall_at_5", "macro_recall_at_10", "macro_recall_at_20",
            ]},
            "v8_previous": {key: float(v8_k[key]) for key in [
                "micro_auroc", "micro_auprc", "macro_drug_auroc", "macro_drug_auprc",
                "macro_recall_at_5", "macro_recall_at_10", "macro_recall_at_20",
            ]},
            "v10_minus_v8_bootstrap": {
                row["metric"]: {
                    "point": float(row["point_estimate"]),
                    "ci95": [float(row["ci95_lower"]), float(row["ci95_upper"])],
                }
                for _, row in v10_v8_diff.iterrows()
            },
        },
        "strict_bindingdb_positive_only": {
            **v10_binding,
            "permutation_p": {
                row["metric"]: float(row["one_sided_permutation_p"])
                for _, row in binding_permutation.iterrows()
            },
        },
        "claim_boundaries": [
            "KiRHub was used in earlier project versions: this is post-freeze for V10, not never-seen for the project.",
            "BindingDB is positive-only and previously inspected in V9: it estimates retrieval, not AUROC or specificity.",
            "No external result was used to reselect or refit the V10 model.",
        ],
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [PREDICTIONS, MODEL_SUMMARY, V8_EXTERNAL, BINDINGDB]
        },
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
    }
    summary_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_SUMMARY_V10.json"
    summary_path.write_text(json.dumps(external_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(external_summary, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

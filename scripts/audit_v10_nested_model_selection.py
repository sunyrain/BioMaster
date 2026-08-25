#!/usr/bin/env python3
"""Nested drug-grouped audit of V10 architecture selection optimism."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from train_v10_leakage_safe_ranker import BASE_FEATURES, et, fit_model, hgb, metrics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
PAIR = RUN / "final_evidence_routing_v9/PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V9.csv.gz"
GRAPH = RUN / "candidate_edge_excluded_graph_v10/CANDIDATE_EDGE_EXCLUDED_GRAPH_FEATURES_720_X_384_V10.csv.gz"
SUMMARY = RUN / "leakage_safe_ranker_v10/OLD_DRUG_LEAKAGE_SAFE_SUMMARY_V10.json"
FIXED_OOF = RUN / "leakage_safe_ranker_v10/OLD_DRUG_LEAKAGE_SAFE_FULL384_OOF_V10.csv.gz"
OUT = RUN / "leakage_safe_ranker_v10/nested_selection_audit"
SEED = 20260813
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAPS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_models(base: list[str], known: list[str]):
    return {
        "ET_BASELINE_SAFE": et(base),
        "HGB_BASELINE_SAFE": hgb(base),
        "ET_KNOWN_GRAPH_FUSION_SAFE": et(base + known),
        "HGB_KNOWN_GRAPH_FUSION_SAFE": hgb(base + known),
        "HGB_KNOWN_GRAPH_ONLY_SAFE": hgb(known),
    }


def select_from_metrics(table: pd.DataFrame) -> str:
    return str(table.sort_values(
        ["macro_recall_at_10", "macro_drug_auprc", "macro_recall_at_20", "micro_auprc"],
        ascending=False, kind="mergesort",
    ).iloc[0]["model_name"])


def paired_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    drugs = sorted(set(frame.loc[frame["retrieval_secondary_positive"].astype(bool), "ligand_inchikey"]))
    metric_names = ("drug_auroc", "drug_auprc", "recall_at_20")
    per_drug = {metric: [] for metric in metric_names}
    from sklearn.metrics import average_precision_score, roc_auc_score

    for drug in drugs:
        group = frame[frame["ligand_inchikey"].eq(drug)]
        y = group["retrieval_secondary_positive"].astype(int)
        positives = int(y.sum())
        per_drug["drug_auroc"].append(
            roc_auc_score(y, group["nested_selection_oof_score_v10"])
            - roc_auc_score(y, group["fixed_winner_oof_score_v10"])
        )
        per_drug["drug_auprc"].append(
            average_precision_score(y, group["nested_selection_oof_score_v10"])
            - average_precision_score(y, group["fixed_winner_oof_score_v10"])
        )
        per_drug["recall_at_20"].append(
            group.nlargest(20, "nested_selection_oof_score_v10")["retrieval_secondary_positive"].sum() / positives
            - group.nlargest(20, "fixed_winner_oof_score_v10")["retrieval_secondary_positive"].sum() / positives
        )
    rng = np.random.default_rng(SEED)
    rows = []
    for metric, values_list in per_drug.items():
        values = np.asarray(values_list)
        samples = np.empty(BOOTSTRAPS)
        for index in range(BOOTSTRAPS):
            draw = rng.integers(0, len(values), len(values))
            samples[index] = values[draw].mean()
        rows.append({
            "comparison": "NESTED_SELECTION_MINUS_FIXED_WINNER_OOF", "metric": metric,
            "point_estimate": float(values.mean()), "bootstrap_mean": float(samples.mean()),
            "ci95_lower": float(np.quantile(samples, .025)),
            "ci95_upper": float(np.quantile(samples, .975)),
            "cluster_unit": "ligand_inchikey", "bootstrap_replicates": BOOTSTRAPS,
            "seed": SEED,
        })
    return pd.DataFrame(rows)


def main() -> None:
    required = [PAIR, GRAPH, SUMMARY, FIXED_OOF]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    frozen = json.loads(SUMMARY.read_text())
    if frozen["status"] != "PASS":
        raise RuntimeError("Frozen V10 package is not PASS")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = [
        "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol",
        "is_any_frozen_known_relationship", "chembl37_pair_record_class",
    ]
    data = pd.read_csv(PAIR, usecols=meta + BASE_FEATURES, low_memory=False)
    graph = pd.read_csv(GRAPH, low_memory=False)
    known_features = [column for column in graph.columns if column.startswith("known_graph_")]
    data = data.merge(graph[["pairId"] + known_features], on="pairId", how="left", validate="one_to_one")
    known = data["is_any_frozen_known_relationship"].astype(bool)
    positive = data["chembl37_pair_record_class"].eq("K1_STRICT_BINDING_POSITIVE") & ~known
    negative = data["chembl37_pair_record_class"].eq("K3_STRICT_GREY_NEGATIVE_OR_INACTIVE") & ~known
    labeled = data[positive | negative].copy().reset_index(drop=True)
    labeled["ranker_label"] = positive[positive | negative].astype(int).to_numpy()
    retrieval_label = positive.astype(int).to_numpy()
    candidates = candidate_models(BASE_FEATURES, known_features)

    outer = GroupKFold(n_splits=OUTER_FOLDS)
    outer_splits = list(outer.split(labeled, labeled["ranker_label"], labeled["ligand_inchikey"]))
    nested_scores = np.full(len(data), np.nan, dtype=float)
    selection_rows, inner_metric_rows = [], []
    for outer_fold, (outer_train_index, outer_test_index) in enumerate(outer_splits):
        outer_train = labeled.iloc[outer_train_index].reset_index(drop=True)
        outer_test_drugs = set(labeled.iloc[outer_test_index]["ligand_inchikey"])
        inner = GroupKFold(n_splits=INNER_FOLDS)
        inner_splits = list(inner.split(
            outer_train, outer_train["ranker_label"], outer_train["ligand_inchikey"]
        ))
        current_metrics = []
        for model_name, (template, features) in candidates.items():
            inner_scores = np.full(len(data), np.nan, dtype=float)
            inner_development_drugs: set[str] = set()
            for inner_train_index, inner_test_index in inner_splits:
                model = fit_model(
                    clone(template), outer_train.iloc[inner_train_index][features],
                    outer_train.iloc[inner_train_index]["ranker_label"].to_numpy(),
                )
                test_drugs = set(outer_train.iloc[inner_test_index]["ligand_inchikey"])
                inner_development_drugs.update(test_drugs)
                mask = data["ligand_inchikey"].isin(test_drugs).to_numpy()
                inner_scores[mask] = model.predict_proba(data.loc[mask, features])[:, 1]
            mask = data["ligand_inchikey"].isin(inner_development_drugs).to_numpy()
            frame = data.loc[mask, ["pairId", "ligand_inchikey"]].copy()
            frame["label"] = retrieval_label[mask]
            frame["score"] = inner_scores[mask]
            positive_drugs = set(frame.loc[frame["label"].astype(bool), "ligand_inchikey"])
            result = metrics(
                frame[frame["ligand_inchikey"].isin(positive_drugs)], "score", "label"
            )
            row = {"outer_fold": outer_fold, "model_name": model_name, **result}
            inner_metric_rows.append(row)
            current_metrics.append(row)
        selected = select_from_metrics(pd.DataFrame(current_metrics))
        template, features = candidates[selected]
        final_outer = fit_model(
            clone(template), outer_train[features], outer_train["ranker_label"].to_numpy()
        )
        outer_mask = data["ligand_inchikey"].isin(outer_test_drugs).to_numpy()
        nested_scores[outer_mask] = final_outer.predict_proba(data.loc[outer_mask, features])[:, 1]
        selection_rows.append({
            "outer_fold": outer_fold, "selected_model": selected,
            "outer_train_labeled_pairs": len(outer_train),
            "outer_train_positive_pairs": int(outer_train["ranker_label"].sum()),
            "outer_test_drugs": len(outer_test_drugs), "outer_scored_pairs": int(outer_mask.sum()),
            "inner_folds": INNER_FOLDS,
        })

    evaluated = np.isfinite(nested_scores)
    nested = data.loc[evaluated, [
        "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol"
    ]].copy()
    nested["retrieval_secondary_positive"] = retrieval_label[evaluated]
    nested["nested_selection_oof_score_v10"] = nested_scores[evaluated]
    fixed = pd.read_csv(FIXED_OOF, low_memory=False)
    fixed = fixed[fixed["model_name"].eq(frozen["selected_architecture"])][[
        "pairId", "oof_score_v10"
    ]].rename(columns={"oof_score_v10": "fixed_winner_oof_score_v10"})
    nested = nested.merge(fixed, on="pairId", how="left", validate="one_to_one")
    positive_drugs = set(nested.loc[
        nested["retrieval_secondary_positive"].astype(bool), "ligand_inchikey"
    ])
    evaluation = nested[nested["ligand_inchikey"].isin(positive_drugs)]
    nested_metric = metrics(evaluation, "nested_selection_oof_score_v10", "retrieval_secondary_positive")
    fixed_metric = metrics(evaluation, "fixed_winner_oof_score_v10", "retrieval_secondary_positive")
    bootstrap = paired_bootstrap(evaluation)

    score_path = OUT / "OLD_DRUG_NESTED_MODEL_SELECTION_OOF_V10.csv.gz"
    inner_path = OUT / "OLD_DRUG_NESTED_MODEL_SELECTION_INNER_METRICS_V10.csv"
    choices_path = OUT / "OLD_DRUG_NESTED_MODEL_SELECTION_OUTER_CHOICES_V10.csv"
    bootstrap_path = OUT / "OLD_DRUG_NESTED_MODEL_SELECTION_BOOTSTRAP_V10.csv"
    nested.to_csv(score_path, index=False, compression="gzip")
    pd.DataFrame(inner_metric_rows).to_csv(inner_path, index=False)
    choices = pd.DataFrame(selection_rows)
    choices.to_csv(choices_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    checks = {
        "frozen_v10_dependency_pass": frozen["status"] == "PASS",
        "exact_5_outer_and_4_inner_folds": len(selection_rows) == 5 and all(
            row["inner_folds"] == 4 for row in selection_rows
        ),
        "only_five_leakage_safe_candidates": set(
            pd.DataFrame(inner_metric_rows)["model_name"]
        ) == set(candidates),
        "all_nested_scores_complete_for_288_drugs": (
            evaluated.sum() == 288 * 384 and nested["ligand_inchikey"].nunique() == 288
        ),
        "exact_620_positive_179_evaluation_drugs": (
            int(nested["retrieval_secondary_positive"].sum()) == 620 and len(positive_drugs) == 179
        ),
        "fixed_winner_scores_merge_complete": nested["fixed_winner_oof_score_v10"].notna().all(),
        "all_1000_paired_drug_bootstraps_complete": bootstrap[
            "bootstrap_replicates"
        ].eq(BOOTSTRAPS).all(),
        "no_external_benchmark_read": all(
            token not in " ".join(str(path).lower() for path in required)
            for token in ("kirhub", "bindingdb")
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    output_paths = [score_path, inner_path, choices_path, bootstrap_path]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "protocol": (
            "Five outer drug-group folds. Within each outer training partition, four inner drug-group folds "
            "select among the five predeclared leakage-safe architectures. The selected architecture is refit "
            "on the outer training partition and evaluated only on outer-held-out drugs."
        ),
        "outer_model_choices": choices["selected_model"].value_counts().to_dict(),
        "nested_selection_oof_metrics": nested_metric,
        "fixed_full_oof_selected_winner_metrics": fixed_metric,
        "nested_minus_fixed_paired_drug_bootstrap": bootstrap.to_dict(orient="records"),
        "interpretation": (
            "Nested selection is the less optimistic internal performance estimate. The separately frozen full-data "
            "architecture remains the deployment fit; external results remain untouched."
        ),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
    }
    summary_path = OUT / "OLD_DRUG_NESTED_MODEL_SELECTION_SUMMARY_V10.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

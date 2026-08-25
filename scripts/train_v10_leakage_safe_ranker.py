#!/usr/bin/env python3
"""Train V10 using only secondary-label-independent graph features for selection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
PAIR = RUN / "final_evidence_routing_v9/PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V9.csv.gz"
GRAPH = RUN / "candidate_edge_excluded_graph_v10/CANDIDATE_EDGE_EXCLUDED_GRAPH_FEATURES_720_X_384_V10.csv.gz"
GRAPH_SUMMARY = RUN / "candidate_edge_excluded_graph_v10/CANDIDATE_EDGE_EXCLUDED_GRAPH_SUMMARY_V10.json"
V8_SUMMARY = RUN / "old_drug_advanced_ranker_v8/OLD_DRUG_ADVANCED_RANKER_SUMMARY_V8.json"
OUT = RUN / "leakage_safe_ranker_v10"
SEED = 20260813
FOLDS = 5

BASE_FEATURES = [
    "conplex_score", "conplex_percentile_within_target",
    "conplex_percentile_within_ligand_384", "drugclip_cosine_mean",
    "drugclip_sixfold_std", "drugclip_percentile_within_target",
    "drugclip_percentile_within_ligand_382",
    "conplex_percentile_within_ligand_same_pocket_source",
    "drugclip_percentile_within_ligand_same_pocket_source",
    "dta_target_percentile_disagreement", "dta_ligand_percentile_disagreement_384_382",
    "dta_cross_target_consensus_score", "dta_target_top10pct_concordant",
    "dta_drug_centric_top10pct_concordant_384", "dta_bidirectional_top10pct_concordant_384",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_weights(y: np.ndarray) -> np.ndarray:
    count0, count1 = max(1, int((y == 0).sum())), max(1, int((y == 1).sum()))
    return np.where(y == 1, len(y) / (2 * count1), len(y) / (2 * count0)).astype(float)


def fit_model(model: Pipeline, x: pd.DataFrame, y: np.ndarray) -> Pipeline:
    if isinstance(model.named_steps["model"], HistGradientBoostingClassifier):
        model.fit(x, y, model__sample_weight=balanced_weights(y))
    else:
        model.fit(x, y)
    return model


def metrics(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    y = frame[label].astype(int)
    output: dict[str, Any] = {
        "pairs": len(frame), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "micro_auroc": float(roc_auc_score(y, frame[score])) if y.nunique() == 2 else None,
        "micro_auprc": float(average_precision_score(y, frame[score])) if y.nunique() == 2 else None,
    }
    aucs, aps = [], []
    recall = {5: [], 10: [], 20: []}
    precision = {5: [], 10: [], 20: []}
    for _, group in frame.groupby("ligand_inchikey", sort=False):
        yy = group[label].astype(int)
        if yy.nunique() == 2:
            aucs.append(roc_auc_score(yy, group[score]))
            aps.append(average_precision_score(yy, group[score]))
        positives = int(yy.sum())
        if not positives:
            continue
        ranked = group.sort_values(score, ascending=False, kind="mergesort")
        for k in recall:
            hits = int(ranked.head(min(k, len(ranked)))[label].sum())
            recall[k].append(hits / positives)
            precision[k].append(hits / min(k, len(ranked)))
    output["macro_drug_auroc"] = float(np.mean(aucs)) if aucs else None
    output["macro_drug_auprc"] = float(np.mean(aps)) if aps else None
    output["drugs_with_two_classes"] = len(aucs)
    for k in recall:
        output[f"macro_recall_at_{k}"] = float(np.mean(recall[k])) if recall[k] else None
        output[f"macro_precision_at_{k}"] = float(np.mean(precision[k])) if precision[k] else None
    return output


def et(features: list[str], estimators: int = 600) -> tuple[Pipeline, list[str]]:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=estimators, min_samples_leaf=3, max_features=0.65,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        )),
    ]), features


def hgb(features: list[str]) -> tuple[Pipeline, list[str]]:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=350, max_leaf_nodes=15,
            min_samples_leaf=18, l2_regularization=2.0, random_state=SEED,
        )),
    ]), features


def main() -> None:
    required = [PAIR, GRAPH, GRAPH_SUMMARY, V8_SUMMARY]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    graph_summary = json.loads(GRAPH_SUMMARY.read_text())
    if graph_summary.get("status") != "PASS":
        raise RuntimeError("Corrected graph package is not PASS")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = [
        "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol",
        "is_any_frozen_known_relationship", "chembl37_pair_record_class",
    ]
    data = pd.read_csv(PAIR, usecols=meta + BASE_FEATURES, low_memory=False)
    graph = pd.read_csv(GRAPH, low_memory=False)
    graph_features = [column for column in graph.columns if column != "pairId"]
    known_features = [column for column in graph_features if column.startswith("known_graph_")]
    corrected_all_features = graph_features
    data = data.merge(graph, on="pairId", how="left", validate="one_to_one")
    known = data["is_any_frozen_known_relationship"].astype(bool)
    positive = data["chembl37_pair_record_class"].eq("K1_STRICT_BINDING_POSITIVE") & ~known
    negative = data["chembl37_pair_record_class"].eq("K3_STRICT_GREY_NEGATIVE_OR_INACTIVE") & ~known
    labeled = data[positive | negative].copy().reset_index(drop=True)
    labeled["ranker_label"] = positive[positive | negative].astype(int).to_numpy()
    positive_count = data.loc[positive].groupby("ligand_inchikey").size()
    single_drugs = set(positive_count[positive_count.eq(1)].index)
    multi_drugs = set(positive_count[positive_count.gt(1)].index)

    feature_sets = {
        "BASELINE": BASE_FEATURES,
        "KNOWN_FUSION": BASE_FEATURES + known_features,
        "KNOWN_GRAPH_ONLY": known_features,
        "CORRECTED_ALL_FUSION": BASE_FEATURES + corrected_all_features,
    }
    candidates: dict[str, tuple[Pipeline, list[str], bool]] = {
        "ET_BASELINE_SAFE": (*et(feature_sets["BASELINE"]), True),
        "HGB_BASELINE_SAFE": (*hgb(feature_sets["BASELINE"]), True),
        "ET_KNOWN_GRAPH_FUSION_SAFE": (*et(feature_sets["KNOWN_FUSION"]), True),
        "HGB_KNOWN_GRAPH_FUSION_SAFE": (*hgb(feature_sets["KNOWN_FUSION"]), True),
        "HGB_KNOWN_GRAPH_ONLY_SAFE": (*hgb(feature_sets["KNOWN_GRAPH_ONLY"]), True),
        "ET_CORRECTED_ALL_GRAPH_TRANSDUCTIVE_CONTROL": (*et(feature_sets["CORRECTED_ALL_FUSION"]), False),
        "HGB_CORRECTED_ALL_GRAPH_TRANSDUCTIVE_CONTROL": (*hgb(feature_sets["CORRECTED_ALL_FUSION"]), False),
    }
    splitter = GroupKFold(n_splits=FOLDS)
    splits = list(splitter.split(labeled, labeled["ranker_label"], labeled["ligand_inchikey"]))
    retrieval_label = positive.astype(int).to_numpy()
    score_by_model: dict[str, np.ndarray] = {}
    metrics_rows, oof_rows = [], []
    for model_name, (template, features, selection_eligible) in candidates.items():
        scores = np.full(len(data), np.nan, dtype=np.float64)
        for train_index, test_index in splits:
            model = fit_model(
                clone(template), labeled.iloc[train_index][features],
                labeled.iloc[train_index]["ranker_label"].to_numpy(),
            )
            test_drugs = set(labeled.iloc[test_index]["ligand_inchikey"])
            mask = data["ligand_inchikey"].isin(test_drugs).to_numpy()
            scores[mask] = model.predict_proba(data.loc[mask, features])[:, 1]
        score_by_model[model_name] = scores
        evaluated = np.isfinite(scores)
        frame = data.loc[evaluated, [
            "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol"
        ]].copy()
        frame["retrieval_secondary_positive"] = retrieval_label[evaluated]
        frame["oof_score_v10"] = scores[evaluated]
        frame["model_name"] = model_name
        frame["selection_eligible"] = selection_eligible
        oof_rows.append(frame)
        slices = {
            "ALL_POSITIVE_DRUGS": frame[frame["ligand_inchikey"].isin(single_drugs | multi_drugs)],
            "SINGLE_SECONDARY_POSITIVE_DRUGS": frame[frame["ligand_inchikey"].isin(single_drugs)],
            "MULTI_SECONDARY_POSITIVE_DRUGS": frame[frame["ligand_inchikey"].isin(multi_drugs)],
        }
        for slice_name, current in slices.items():
            metrics_rows.append({
                "model_name": model_name, "selection_eligible": selection_eligible,
                "evaluation_slice": slice_name,
                **metrics(current, "oof_score_v10", "retrieval_secondary_positive"),
            })
    metric_table = pd.DataFrame(metrics_rows)
    selection = metric_table[
        metric_table["selection_eligible"].astype(bool)
        & metric_table["evaluation_slice"].eq("ALL_POSITIVE_DRUGS")
    ].sort_values(
        ["macro_recall_at_10", "macro_drug_auprc", "macro_recall_at_20", "micro_auprc"],
        ascending=False, kind="mergesort",
    )
    winner = str(selection.iloc[0]["model_name"])
    template, winner_features, eligible = candidates[winner]
    if not eligible:
        raise RuntimeError("Transductive control cannot win model selection")
    final_model = fit_model(template, labeled[winner_features], labeled["ranker_label"].to_numpy())
    final_scores = final_model.predict_proba(data[winner_features])[:, 1]
    predictions = data[["pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol"]].copy()
    predictions["old_drug_leakage_safe_score_v10"] = final_scores
    predictions["old_drug_leakage_safe_rank_within_drug_384_v10"] = predictions.groupby(
        "ligand_inchikey"
    )["old_drug_leakage_safe_score_v10"].rank(method="min", ascending=False)
    prediction_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_ALL_276480_V10.csv.gz"
    metrics_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_OOF_METRICS_V10.csv"
    oof_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_FULL384_OOF_V10.csv.gz"
    model_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_MODEL_V10.joblib"
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    metric_table.to_csv(metrics_path, index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(oof_path, index=False, compression="gzip")
    joblib.dump({
        "model": final_model, "features": winner_features, "architecture": winner,
        "selection_eligible": True, "seed": SEED,
    }, model_path)

    selected_all = selection.iloc[0]
    selected_single = metric_table[
        metric_table["model_name"].eq(winner)
        & metric_table["evaluation_slice"].eq("SINGLE_SECONDARY_POSITIVE_DRUGS")
    ].iloc[0]
    v8_summary = json.loads(V8_SUMMARY.read_text())
    checks = {
        "corrected_graph_package_pass": graph_summary["status"] == "PASS",
        "exact_620_positive_4358_negative_labels": int(positive.sum()) == 620 and int(negative.sum()) == 4358,
        "exact_80_single_and_99_multi_positive_drugs": len(single_drugs) == 80 and len(multi_drugs) == 99,
        "five_drug_grouped_folds": len(splits) == 5,
        "all_candidate_oof_scores_complete": all(np.isfinite(scores).sum() == 288 * 384 for scores in score_by_model.values()),
        "transductive_controls_excluded_from_selection": not metric_table.loc[
            metric_table["model_name"].str.contains("TRANSDUCTIVE"), "selection_eligible"
        ].astype(bool).any(),
        "selected_model_uses_no_secondary_or_union_graph_features": not any(
            feature.startswith(("secondary_graph_", "all_positive_graph_")) for feature in winner_features
        ),
        "all_276480_pairs_scored": len(predictions) == 276480 and predictions["old_drug_leakage_safe_score_v10"].notna().all(),
        "each_drug_has_384_valid_ranks": (
            predictions.groupby("ligand_inchikey").size().eq(384).all()
            and predictions["old_drug_leakage_safe_rank_within_drug_384_v10"].between(1, 384).all()
        ),
        "kirhub_and_bindingdb_not_read": all(
            token not in " ".join(str(path).lower() for path in required)
            for token in ["kirhub", "bindingdb"]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "task": "old-drug secondary/new-target ranking with secondary-label-independent model selection",
        "selected_architecture": winner,
        "selected_features": winner_features,
        "selection_criterion": "eligible-only drug-grouped OOF Recall@10, then macro AUPRC, Recall@20, micro AUPRC",
        "selection_eligible_models": [name for name, (_, _, eligible) in candidates.items() if eligible],
        "transductive_diagnostic_controls": [name for name, (_, _, eligible) in candidates.items() if not eligible],
        "selected_oof_all_positive_drugs": {
            key: float(selected_all[key]) for key in [
                "micro_auroc", "micro_auprc", "macro_drug_auroc", "macro_drug_auprc",
                "macro_recall_at_5", "macro_recall_at_10", "macro_recall_at_20",
            ]
        },
        "selected_oof_single_positive_drugs": {
            key: float(selected_single[key]) for key in [
                "micro_auroc", "micro_auprc", "macro_drug_auroc", "macro_drug_auprc",
                "macro_recall_at_5", "macro_recall_at_10", "macro_recall_at_20",
            ]
        },
        "v8_internal_metric_correction": {
            "previous_selected_architecture": v8_summary["selected_architecture"],
            "previous_reported_oof": v8_summary["selected_internal_oof_metrics"],
            "previous_metric_validity": (
                "Not valid as candidate-edge-excluded OOF because six degree features contained the candidate adjacency indicator."
            ),
        },
        "external_firewall": "No KiRHub or BindingDB file is read; predictions and model are frozen before evaluation.",
        "deployment_scope": (
            "Old-drug target-profile expansion using only frozen known/MoA graph context for eligible models; "
            "not novel-drug cold start and not a binding probability."
        ),
        "software": {"sklearn": sklearn.__version__, "seed": SEED},
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [prediction_path, metrics_path, oof_path, model_path]
        },
    }
    summary_path = OUT / "OLD_DRUG_LEAKAGE_SAFE_SUMMARY_V10.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

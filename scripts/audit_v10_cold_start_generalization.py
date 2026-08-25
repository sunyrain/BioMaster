#!/usr/bin/env python3
"""Drug-, target-, and double-label-cold generalization audit for frozen V10 design."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
PAIR = RUN / "final_evidence_routing_v9/PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V9.csv.gz"
GRAPH = RUN / "candidate_edge_excluded_graph_v10/CANDIDATE_EDGE_EXCLUDED_GRAPH_FEATURES_720_X_384_V10.csv.gz"
RANKER = RUN / "leakage_safe_ranker_v10/OLD_DRUG_LEAKAGE_SAFE_MODEL_V10.joblib"
RANKER_SUMMARY = RUN / "leakage_safe_ranker_v10/OLD_DRUG_LEAKAGE_SAFE_SUMMARY_V10.json"
OUT = RUN / "leakage_safe_ranker_v10/generalization_audit"

SEED = 20260813
FOLDS = 5
BOOTSTRAPS = 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_weights(y: np.ndarray) -> np.ndarray:
    count0, count1 = max(1, int((y == 0).sum())), max(1, int((y == 1).sum()))
    return np.where(y == 1, len(y) / (2 * count1), len(y) / (2 * count0)).astype(float)


def fit_model(template: Any, x: pd.DataFrame, y: np.ndarray) -> Any:
    model = clone(template)
    if isinstance(model.named_steps["model"], HistGradientBoostingClassifier):
        model.fit(x, y, model__sample_weight=balanced_weights(y))
    else:
        model.fit(x, y)
    return model


def assign_group_folds(labeled: pd.DataFrame, all_units: list[str], group_column: str) -> dict[str, int]:
    """GroupKFold labelled units; balance unlabelled units without using outcomes."""
    splitter = GroupKFold(n_splits=FOLDS)
    assignment: dict[str, int] = {}
    for fold, (_, test_index) in enumerate(
        splitter.split(labeled, labeled["ranker_label"], labeled[group_column])
    ):
        for unit in labeled.iloc[test_index][group_column].unique():
            if unit in assignment and assignment[unit] != fold:
                raise RuntimeError(f"Group {unit} assigned to multiple folds")
            assignment[str(unit)] = fold
    counts = np.bincount(list(assignment.values()), minlength=FOLDS).astype(int)
    for unit in sorted(set(all_units) - set(assignment)):
        fold = int(np.argmin(counts))
        assignment[str(unit)] = fold
        counts[fold] += 1
    if len(assignment) != len(set(all_units)):
        raise RuntimeError(f"Incomplete {group_column} fold assignment")
    return assignment


def score_drug_cold(
    data: pd.DataFrame, labeled: pd.DataFrame, features: list[str], template: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.full(len(data), np.nan, dtype=float)
    audits = []
    for fold in range(FOLDS):
        train = labeled[labeled["drug_fold"] != fold]
        test_mask = data["drug_fold"].eq(fold).to_numpy()
        model = fit_model(template, train[features], train["ranker_label"].to_numpy())
        scores[test_mask] = model.predict_proba(data.loc[test_mask, features])[:, 1]
        audits.append({
            "protocol": "DRUG_LABEL_COLD", "drug_fold": fold, "target_fold": -1,
            "train_pairs": len(train), "train_positives": int(train["ranker_label"].sum()),
            "score_pairs": int(test_mask.sum()),
            "heldout_drugs": int(data.loc[test_mask, "ligand_inchikey"].nunique()),
            "heldout_targets": 0,
            "train_test_drug_identity_overlap": 0,
            "train_test_target_identity_overlap": -1,
        })
    return scores, audits


def score_target_cold(
    data: pd.DataFrame, labeled: pd.DataFrame, features: list[str], template: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.full(len(data), np.nan, dtype=float)
    audits = []
    for fold in range(FOLDS):
        train = labeled[labeled["target_fold"] != fold]
        test_mask = data["target_fold"].eq(fold).to_numpy()
        model = fit_model(template, train[features], train["ranker_label"].to_numpy())
        scores[test_mask] = model.predict_proba(data.loc[test_mask, features])[:, 1]
        audits.append({
            "protocol": "TARGET_LABEL_COLD", "drug_fold": -1, "target_fold": fold,
            "train_pairs": len(train), "train_positives": int(train["ranker_label"].sum()),
            "score_pairs": int(test_mask.sum()),
            "heldout_drugs": 0,
            "heldout_targets": int(data.loc[test_mask, "target_chembl_id"].nunique()),
            "train_test_drug_identity_overlap": -1,
            "train_test_target_identity_overlap": 0,
        })
    return scores, audits


def score_double_cold(
    data: pd.DataFrame, labeled: pd.DataFrame, features: list[str], template: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scores = np.full(len(data), np.nan, dtype=float)
    audits = []
    for drug_fold in range(FOLDS):
        for target_fold in range(FOLDS):
            train = labeled[
                labeled["drug_fold"].ne(drug_fold)
                & labeled["target_fold"].ne(target_fold)
            ]
            test_mask = (
                data["drug_fold"].eq(drug_fold) & data["target_fold"].eq(target_fold)
            ).to_numpy()
            model = fit_model(template, train[features], train["ranker_label"].to_numpy())
            scores[test_mask] = model.predict_proba(data.loc[test_mask, features])[:, 1]
            audits.append({
                "protocol": "DOUBLE_LABEL_COLD", "drug_fold": drug_fold,
                "target_fold": target_fold,
                "train_pairs": len(train), "train_positives": int(train["ranker_label"].sum()),
                "score_pairs": int(test_mask.sum()),
                "heldout_drugs": int(data.loc[test_mask, "ligand_inchikey"].nunique()),
                "heldout_targets": int(data.loc[test_mask, "target_chembl_id"].nunique()),
                "train_test_drug_identity_overlap": 0,
                "train_test_target_identity_overlap": 0,
            })
    return scores, audits


def retrieval_metrics(frame: pd.DataFrame, score: str, label: str) -> dict[str, Any]:
    positive_drugs = set(frame.loc[frame[label].astype(bool), "ligand_inchikey"])
    x = frame[frame["ligand_inchikey"].isin(positive_drugs)].copy()
    y = x[label].astype(int)
    result: dict[str, Any] = {
        "pairs": int(len(x)), "positives": int(y.sum()), "drugs": len(positive_drugs),
        "targets": int(x["target_chembl_id"].nunique()), "prevalence": float(y.mean()),
        "micro_auroc": float(roc_auc_score(y, x[score])),
        "micro_auprc": float(average_precision_score(y, x[score])),
    }
    aucs, aps = [], []
    recall = {5: [], 10: [], 20: []}
    precision = {5: [], 10: [], 20: []}
    for _, group in x.groupby("ligand_inchikey", sort=False):
        yy = group[label].astype(int)
        aucs.append(roc_auc_score(yy, group[score]))
        aps.append(average_precision_score(yy, group[score]))
        positives = int(yy.sum())
        ranked = group.sort_values(score, ascending=False, kind="mergesort")
        for k in recall:
            kk = min(k, len(ranked))
            hits = int(ranked.head(kk)[label].sum())
            recall[k].append(hits / positives)
            precision[k].append(hits / kk)
    result["macro_drug_auroc"] = float(np.mean(aucs))
    result["macro_drug_auprc"] = float(np.mean(aps))
    for k in recall:
        result[f"macro_recall_at_{k}"] = float(np.mean(recall[k]))
        result[f"macro_precision_at_{k}"] = float(np.mean(precision[k]))
    return result


def paired_drug_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    positive_drugs = sorted(set(frame.loc[frame["retrieval_secondary_positive"].astype(bool), "ligand_inchikey"]))
    metric_names = ("drug_auroc", "drug_auprc", "recall_at_20")
    protocols = {
        "DRUG_LABEL_COLD": "drug_label_cold_score_v10",
        "TARGET_LABEL_COLD": "target_label_cold_score_v10",
        "DOUBLE_LABEL_COLD": "double_label_cold_score_v10",
    }
    values: dict[str, dict[str, np.ndarray]] = {
        protocol: {metric: np.zeros(len(positive_drugs)) for metric in metric_names}
        for protocol in protocols
    }
    for index, drug in enumerate(positive_drugs):
        group = frame[frame["ligand_inchikey"].eq(drug)]
        y = group["retrieval_secondary_positive"].astype(int)
        positives = int(y.sum())
        for protocol, score in protocols.items():
            values[protocol]["drug_auroc"][index] = roc_auc_score(y, group[score])
            values[protocol]["drug_auprc"][index] = average_precision_score(y, group[score])
            ranked = group.sort_values(score, ascending=False, kind="mergesort")
            values[protocol]["recall_at_20"][index] = ranked.head(20)[
                "retrieval_secondary_positive"
            ].sum() / positives
    rng = np.random.default_rng(SEED)
    rows = []
    for comparison, left, right in [
        ("TARGET_MINUS_DRUG", "TARGET_LABEL_COLD", "DRUG_LABEL_COLD"),
        ("DOUBLE_MINUS_DRUG", "DOUBLE_LABEL_COLD", "DRUG_LABEL_COLD"),
        ("DOUBLE_MINUS_TARGET", "DOUBLE_LABEL_COLD", "TARGET_LABEL_COLD"),
    ]:
        for metric in metric_names:
            per_drug_difference = values[left][metric] - values[right][metric]
            samples = np.empty(BOOTSTRAPS, dtype=float)
            for replicate in range(BOOTSTRAPS):
                draw = rng.integers(0, len(positive_drugs), size=len(positive_drugs))
                samples[replicate] = float(per_drug_difference[draw].mean())
            rows.append({
                "evaluation_slice": "ALL_SECONDARY_POSITIVE_DRUGS_FULL_384",
                "comparison": comparison, "metric": metric,
                "point_estimate": float(per_drug_difference.mean()),
                "bootstrap_mean": float(samples.mean()),
                "ci95_lower": float(np.quantile(samples, 0.025)),
                "ci95_upper": float(np.quantile(samples, 0.975)),
                "cluster_unit": "ligand_inchikey", "bootstrap_replicates": BOOTSTRAPS,
                "seed": SEED,
            })
    return pd.DataFrame(rows)


def main() -> None:
    required = [PAIR, GRAPH, RANKER, RANKER_SUMMARY]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    summary = json.loads(RANKER_SUMMARY.read_text())
    bundle = joblib.load(RANKER)
    if summary["status"] != "PASS" or bundle["architecture"] != summary["selected_architecture"]:
        raise RuntimeError("V10 frozen design is not valid")
    features = list(bundle["features"])
    if any(feature.startswith(("secondary_graph_", "all_positive_graph_")) for feature in features):
        raise RuntimeError("Selected cold-start model cannot use secondary-derived graph features")
    if sha256(RANKER) != summary["outputs"][str(RANKER.relative_to(ROOT))]:
        raise RuntimeError("Frozen V10 model hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)

    meta = [
        "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol",
        "is_any_frozen_known_relationship", "chembl37_pair_record_class",
    ]
    pair = pd.read_csv(PAIR, usecols=meta, low_memory=False)
    graph = pd.read_csv(GRAPH, usecols=["pairId"] + features, low_memory=False)
    data = pair.merge(graph, on="pairId", how="left", validate="one_to_one")
    known = data["is_any_frozen_known_relationship"].astype(bool)
    positive = data["chembl37_pair_record_class"].eq("K1_STRICT_BINDING_POSITIVE") & ~known
    negative = data["chembl37_pair_record_class"].eq("K3_STRICT_GREY_NEGATIVE_OR_INACTIVE") & ~known
    labeled = data[positive | negative].copy().reset_index(drop=True)
    labeled["ranker_label"] = positive[positive | negative].astype(int).to_numpy()

    drug_assignment = assign_group_folds(
        labeled, data["ligand_inchikey"].unique().tolist(), "ligand_inchikey"
    )
    target_assignment = assign_group_folds(
        labeled, data["target_chembl_id"].unique().tolist(), "target_chembl_id"
    )
    data["drug_fold"] = data["ligand_inchikey"].map(drug_assignment).astype(int)
    data["target_fold"] = data["target_chembl_id"].map(target_assignment).astype(int)
    # Candidate-edge exclusion makes the degree one smaller only on a known
    # edge's own row. The entity-level seed degree is therefore the maximum
    # across all candidate rows for that entity.
    data["known_graph_drug_seed_degree_stable"] = data.groupby("ligand_inchikey")[
        "known_graph_drug_seed_degree"
    ].transform("max")
    data["known_graph_target_seed_degree_stable"] = data.groupby("target_chembl_id")[
        "known_graph_target_seed_degree"
    ].transform("max")
    labeled["drug_fold"] = labeled["ligand_inchikey"].map(drug_assignment).astype(int)
    labeled["target_fold"] = labeled["target_chembl_id"].map(target_assignment).astype(int)

    template = bundle["model"]
    drug_scores, drug_audit = score_drug_cold(data, labeled, features, template)
    target_scores, target_audit = score_target_cold(data, labeled, features, template)
    double_scores, double_audit = score_double_cold(data, labeled, features, template)
    data["drug_label_cold_score_v10"] = drug_scores
    data["target_label_cold_score_v10"] = target_scores
    data["double_label_cold_score_v10"] = double_scores
    data["retrieval_secondary_positive"] = positive.astype(int).to_numpy()

    positive_count = data.loc[positive].groupby("ligand_inchikey").size()
    single_drugs = set(positive_count[positive_count.eq(1)].index)
    multi_drugs = set(positive_count[positive_count.gt(1)].index)
    target_zero = data["known_graph_target_seed_degree_stable"].eq(0)
    drug_zero = data["known_graph_drug_seed_degree_stable"].eq(0)
    slices = {
        "ALL_SECONDARY_POSITIVE_DRUGS_FULL_384": data,
        "SINGLE_SECONDARY_POSITIVE_DRUGS_FULL_384": data[data["ligand_inchikey"].isin(single_drugs)],
        "MULTI_SECONDARY_POSITIVE_DRUGS_FULL_384": data[data["ligand_inchikey"].isin(multi_drugs)],
        "ZERO_KNOWN_TARGET_DEGREE_POOL_199": data[target_zero],
        "DOUBLE_ZERO_KNOWN_DEGREE_POOL_199": data[target_zero & drug_zero],
    }
    protocols = {
        "DRUG_LABEL_COLD": "drug_label_cold_score_v10",
        "TARGET_LABEL_COLD": "target_label_cold_score_v10",
        "DOUBLE_LABEL_COLD": "double_label_cold_score_v10",
    }
    metric_rows = []
    for protocol, score in protocols.items():
        for slice_name, frame in slices.items():
            metric_rows.append({
                "protocol": protocol, "evaluation_slice": slice_name, "score_column": score,
                **retrieval_metrics(frame, score, "retrieval_secondary_positive"),
            })
    metric_table = pd.DataFrame(metric_rows)
    fold_audit = pd.DataFrame(drug_audit + target_audit + double_audit)
    bootstrap = paired_drug_bootstrap(data)

    score_path = OUT / "OLD_DRUG_COLD_START_SCORES_276480_V10.csv.gz"
    metrics_path = OUT / "OLD_DRUG_COLD_START_METRICS_V10.csv"
    folds_path = OUT / "OLD_DRUG_COLD_START_FOLD_AUDIT_V10.csv"
    assignment_path = OUT / "OLD_DRUG_COLD_START_FOLD_ASSIGNMENTS_V10.csv"
    bootstrap_path = OUT / "OLD_DRUG_COLD_START_PAIRED_BOOTSTRAP_V10.csv"
    data[[
        "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol",
        "drug_fold", "target_fold", "known_graph_drug_seed_degree",
        "known_graph_target_seed_degree", "known_graph_drug_seed_degree_stable",
        "known_graph_target_seed_degree_stable", "retrieval_secondary_positive",
        "drug_label_cold_score_v10", "target_label_cold_score_v10", "double_label_cold_score_v10",
    ]].to_csv(score_path, index=False, compression="gzip")
    metric_table.to_csv(metrics_path, index=False)
    fold_audit.to_csv(folds_path, index=False)
    pd.concat([
        pd.DataFrame({"unit_type": "drug", "unit_id": list(drug_assignment), "fold": list(drug_assignment.values())}),
        pd.DataFrame({"unit_type": "target", "unit_id": list(target_assignment), "fold": list(target_assignment.values())}),
    ], ignore_index=True).to_csv(assignment_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    lookup = metric_table.set_index(["protocol", "evaluation_slice"])
    all_slice = "ALL_SECONDARY_POSITIVE_DRUGS_FULL_384"
    zero_slice = "ZERO_KNOWN_TARGET_DEGREE_POOL_199"
    double_zero_slice = "DOUBLE_ZERO_KNOWN_DEGREE_POOL_199"
    summary_metrics = {}
    for protocol in protocols:
        summary_metrics[protocol] = {}
        for label, slice_name in [
            ("all_384", all_slice), ("zero_known_target_degree_199", zero_slice),
            ("double_zero_known_degree_199", double_zero_slice),
        ]:
            row = lookup.loc[(protocol, slice_name)]
            summary_metrics[protocol][label] = {
                key: int(row[key]) if key in {"pairs", "positives", "drugs", "targets"} else float(row[key])
                for key in [
                    "pairs", "positives", "drugs", "targets", "prevalence", "micro_auroc",
                    "micro_auprc", "macro_drug_auroc", "macro_drug_auprc",
                    "macro_recall_at_10", "macro_recall_at_20",
                ]
            }
    checks = {
        "frozen_v10_model_hash_verified": sha256(RANKER) == summary["outputs"][str(RANKER.relative_to(ROOT))],
        "selected_features_secondary_label_independent": not any(
            feature.startswith(("secondary_graph_", "all_positive_graph_")) for feature in features
        ),
        "exact_720_drugs_384_targets_276480_pairs": (
            data["ligand_inchikey"].nunique() == 720
            and data["target_chembl_id"].nunique() == 384 and len(data) == 276480
        ),
        "exact_620_positive_4358_negative_labels": int(positive.sum()) == 620 and int(negative.sum()) == 4358,
        "all_three_protocol_scores_complete": data[list(protocols.values())].notna().all().all(),
        "exact_5_drug_5_target_and_25_double_models": (
            len(drug_audit) == 5 and len(target_audit) == 5 and len(double_audit) == 25
        ),
        "double_cold_has_zero_train_test_identity_overlap": (
            fold_audit.loc[fold_audit["protocol"].eq("DOUBLE_LABEL_COLD"), [
                "train_test_drug_identity_overlap", "train_test_target_identity_overlap"
            ]].eq(0).all().all()
        ),
        "exact_199_zero_known_degree_targets": data.loc[target_zero, "target_chembl_id"].nunique() == 199,
        "zero_target_pool_exact_181_positives_90_drugs": (
            int((positive & target_zero).sum()) == 181
            and data.loc[positive & target_zero, "ligand_inchikey"].nunique() == 90
        ),
        "double_zero_pool_exact_29_positives_19_drugs": (
            int((positive & target_zero & drug_zero).sum()) == 29
            and data.loc[positive & target_zero & drug_zero, "ligand_inchikey"].nunique() == 19
        ),
        "no_external_benchmark_read": all(
            token not in " ".join(str(path).lower() for path in required)
            for token in ("kirhub", "bindingdb")
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    output_paths = [score_path, metrics_path, folds_path, assignment_path, bootstrap_path]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "architecture": bundle["architecture"], "features": features,
        "protocol_definitions": {
            "DRUG_LABEL_COLD": "Every scored drug identity is absent from the supervised secondary-label training fold.",
            "TARGET_LABEL_COLD": "Every scored target identity is absent from the supervised secondary-label training fold.",
            "DOUBLE_LABEL_COLD": "For each of 25 drug-fold x target-fold blocks, both scored identities are absent from supervised training.",
        },
        "cold_scope_boundary": (
            "Cold means no project secondary K1/K3 label for held-out identities. Frozen pretrained DTA representations, "
            "chemical/protein similarities, and known/MoA graph context remain available. Zero-degree slices explicitly "
            "test targets and drugs without direct known-graph seeds."
        ),
        "summary_metrics": summary_metrics,
        "paired_drug_bootstrap_differences": bootstrap.to_dict(orient="records"),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
    }
    summary_path = OUT / "OLD_DRUG_COLD_START_SUMMARY_V10.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

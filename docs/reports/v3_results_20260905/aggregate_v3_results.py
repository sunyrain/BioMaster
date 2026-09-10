"""Recompute the completed V3 matrix; no fitting or checkpoint selection."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from biomaster.odti_v3_panel_metrics import positive_retrieval_metrics

OUT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs/biomaster_v3_20260905/experiments"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_query_ap(scores, labels, targets):
    result = []
    for score, label in zip(scores, labels):
        ranks = np.flatnonzero(label[np.lexsort((targets, -score))]) + 1
        result.append(np.mean(np.arange(1, len(ranks) + 1) / ranks))
    return np.asarray(result)


def main():
    status = json.loads((RUNS / "EXPERIMENT_STATUS_V3.json").read_text())
    assert status["status"] == "completed" and status["completed_jobs"] == 12
    frame = pd.read_csv(RUNS / "EXPERIMENT_RESULTS_V3.csv").sort_values(["variant", "seed"])
    assert len(frame) == 12 and frame.groupby("variant").size().eq(2).all()
    first = RUNS / "D_support_rank/seed_20260905"
    reference = np.load(first / "DENSE_BASELINE_SCORES_V3.npz")
    labels, drugs, targets = (reference[key] for key in ["known_positive", "drug_indices", "target_indices"])
    relations = pd.read_csv(ROOT / "outputs/biomaster_v3_20260905/data/RELATIONS_V3.csv.gz")
    train = relations[relations.split.eq("train")]
    target_train = train.groupby("target_feature_index").binary_label.agg(["mean", "sum", "count"]).reindex(targets)
    prior_valid = target_train["count"].notna().to_numpy()
    histories, ap, rows, priors, sources = {}, {}, [], [], {}
    base_metrics = json.loads((first / "BASELINES_V3.json").read_text())["metrics"]
    for _, record in frame.iterrows():
        name, seed = record["variant"], int(record["seed"])
        directory = RUNS / name / f"seed_{seed}"
        summary = json.loads((directory / "RUN_SUMMARY_V3.json").read_text())
        history = json.loads((directory / "TRAINING_HISTORY_V3.json").read_text())
        assert summary["status"] == "completed" and len(history) == 6
        assert all(epoch["coverage_unique_rows"] == len(train) for epoch in history)
        exposure = [epoch["exposure_sha256"] for epoch in history]
        if seed in histories:
            assert exposure == histories[seed]
        histories[seed] = exposure
        arrays = np.load(directory / "DENSE_TEST_PREDICTIONS_V3.npz")
        for key in ["known_positive", "drug_indices", "target_indices"]:
            np.testing.assert_array_equal(arrays[key], reference[key])
        scores = arrays["scores"]
        metrics = positive_retrieval_metrics(labels, scores, targets)
        for metric, value in metrics.items():
            if metric.startswith("macro_"):
                np.testing.assert_allclose(value, summary["dense_test"][metric], atol=1e-12, rtol=0)
        ap.setdefault(name, []).append(per_query_ap(scores, labels, targets))
        # Descriptive association only: predictions are also a function of drugs.
        priors.append({"variant": name, "seed": seed,
            "targets_with_training_relations": int(prior_valid.sum()),
            **{f"target_mean_score_spearman_train_{column}": float(spearmanr(scores.mean(axis=0)[prior_valid], target_train[column].to_numpy()[prior_valid]).statistic)
               for column in ["mean", "sum", "count"]}})
        for filename in ["RUN_SUMMARY_V3.json", "TRAINING_HISTORY_V3.json", "DENSE_TEST_PREDICTIONS_V3.npz"]:
            path = directory / filename
            sources[str(path.relative_to(ROOT))] = digest(path)
    numeric = ["test_measured_drug_macro_ap", "test_dense_macro_positive_retrieval_ap", "test_dense_macro_recall_at_5",
               "test_dense_macro_recall_at_20", "seconds", "peak_gpu_allocated_gib"]
    for name, group in frame.groupby("variant", sort=True):
        row = {"variant": name, "seeds": 2, "selected_epochs": ",".join(group.selected_epoch.astype(str))}
        row.update({f"{column}_{stat}": float(getattr(group[column], stat)())
                    for column in numeric for stat in ["mean", "min", "max"]})
        rows.append(row)
    for name in ["positive_max_tanimoto", "positive_negative_logistic"]:
        metrics = positive_retrieval_metrics(labels, reference[name], targets)
        ap[name] = [per_query_ap(reference[name], labels, targets)]
        rows.append({"variant": name, "seeds": 0,
                     "test_measured_drug_macro_ap_mean": base_metrics["test"][name]["drug_macro_ap"],
                     "test_dense_macro_positive_retrieval_ap_mean": metrics["macro_positive_retrieval_ap"],
                     "test_dense_macro_recall_at_5_mean": metrics["macro_recall_at_5"],
                     "test_dense_macro_recall_at_20_mean": metrics["macro_recall_at_20"]})
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUT / "AGGREGATE_RESULTS_V3.csv", index=False)
    pd.DataFrame(priors).to_csv(OUT / "TARGET_PRIOR_ASSOCIATIONS_V3.csv", index=False)
    averaged_ap = {name: np.mean(values, axis=0) for name, values in ap.items()}
    pd.DataFrame({"drug_feature_index": drugs, **averaged_ap}).to_csv(OUT / "DENSE_PER_DRUG_AP_V3.csv", index=False)
    draws = np.random.default_rng(20260905).integers(len(drugs), size=(10000, len(drugs)))
    comparisons = {}
    for left, right in [("B_support_pair", "A_global_pair"), ("D_support_rank", "C_global_rank"),
                        ("C_global_rank", "A_global_pair"), ("D_support_rank", "B_support_pair"),
                        ("E_local_rank", "C_global_rank"), ("F_local_support_rank", "D_support_rank"),
                        ("B_support_pair", "positive_max_tanimoto"), ("B_support_pair", "positive_negative_logistic")]:
        delta = averaged_ap[left] - averaged_ap[right]
        comparisons[f"{left}_minus_{right}"] = {
            "mean_delta": float(delta.mean()), "query_bootstrap_ci95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
            "better_queries": int((delta > 1e-12).sum()), "tied_queries": int((np.abs(delta) <= 1e-12).sum()),
            "worse_queries": int((delta < -1e-12).sum())}
    audit = {
        "status": "all_12_completed_artifacts_verified", "completion_utc": status["updated_utc"],
        "each_run_epochs": 6, "each_epoch_coverage_rows": len(train), "same_seed_exposure_hashes_match_all_variants": True,
        "same_dense_queries_labels_and_targets_all_variants": True,
        "all_saved_dense_metrics_recomputed_match": True,
        "dense_queries": len(drugs), "dense_targets": len(targets), "known_positive_pairs": int(labels.sum()),
        "panel_targets_without_training_relations": int((~prior_valid).sum()),
        "dense_known_positive_pairs_without_target_training_relations": int(labels[:, ~prior_valid].sum()),
        "reported_variant_mean": "mean of two separately evaluated seeds; not an ensemble score",
        "bootstrap_scope": "10000 paired whole-query resamples of per-query AP averaged across the two fixed seed models; excludes initialization, family and model-selection uncertainty",
        "comparisons": comparisons, "source_sha256": sources,
    }
    (OUT / "RESULTS_VERIFICATION_V3.json").write_text(json.dumps(audit, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["Positive-neighbor", "PN logistic", "A: Global", "B: Global + support", "C: Global + rank",
                 "D: Global + support + rank", "E: Local + rank", "F: Local + support + rank"]
        keys = ["positive_max_tanimoto", "positive_negative_logistic"] + sorted(frame.variant.unique())
        indexed = aggregate.set_index("variant").loc[keys]
        colors = ["#6b7280", "#9ca3af", "#3b82f6", "#3b82f6", "#3b82f6", "#3b82f6", "#e87955", "#e87955"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for axis, metric, title in zip(axes, ["test_measured_drug_macro_ap", "test_dense_macro_positive_retrieval_ap"],
                                        ["Measured positive / negative AP (772 drugs)", "Full-panel known-positive AP (64 drugs x 495 targets)"]):
            values = indexed[f"{metric}_mean"].to_numpy()
            axis.barh(np.arange(8), values, color=colors, height=.6)
            for i, value in enumerate(values):
                low, high = indexed.iloc[i].get(f"{metric}_min"), indexed.iloc[i].get(f"{metric}_max")
                axis.text(max(value, high if pd.notna(high) else value) + .015, i, f"{value:.4f}", va="center", fontsize=9)
                if pd.notna(low) and pd.notna(high):
                    axis.plot([low, high], [i, i], color="#111827", linewidth=2.5)
            axis.set_xlim(0, 1.12)
            axis.set_xticks([0, .25, .5, .75, 1])
            axis.set_title(title, fontsize=11, pad=12)
            axis.set_xlabel("Macro average precision")
            axis.grid(axis="x", alpha=.15)
            axis.set_axisbelow(True)
            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
        axes[0].set_yticks(np.arange(8), names)
        axes[0].invert_yaxis()
        fig.suptitle("BioMaster V3: completed 6-epoch experiments", fontsize=14)
        fig.text(.5, .01, "Neural bars: mean of 2 seeds; black segments: seed range (not CI). Unknown panel pairs are retrieval background.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .96))
        fig.savefig(OUT / "V3_RESULTS_COMPARISON.png", dpi=180)
        fig.savefig(OUT / "V3_RESULTS_COMPARISON.svg")
        plt.close(fig)
    except ImportError:
        pass
    print(aggregate[["variant", "test_measured_drug_macro_ap_mean", "test_dense_macro_positive_retrieval_ap_mean", "test_dense_macro_recall_at_20_mean"]].to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()

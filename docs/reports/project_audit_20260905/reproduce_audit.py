"""Read-only audit of existing BioMaster data, score artifacts and losses.

Run from the repository root with the existing odti environment:
    python docs/reports/project_audit_20260905/reproduce_audit.py
Only writes diagnostics next to this script; never trains or changes a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT / "outputs/retrain_20260901/comprehensive_training_v1"
os.environ["BIOMASTER_COMPREHENSIVE_PACKAGE"] = str(PACKAGE)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from biomaster.comprehensive_balanced import (  # noqa: E402
    BalancedSamplingConfig, balanced_training_batches,
)
from biomaster.odti_v2 import (  # noqa: E402
    _multi_positive_info_nce, censored_affinity_loss, within_group_listwise_loss,
)
from train_biomaster_comprehensive_full_fit_v1 import exact_keys, split_positions  # noqa: E402
from train_biomaster_comprehensive_balanced_v2 import optimized_splits  # noqa: E402


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    relation_path = PACKAGE / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
    data = pd.read_csv(relation_path, low_memory=False)
    data["exact_pair_key"] = exact_keys(data)
    raw, _ = split_positions(data)
    deployment_dir = ROOT / "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1"
    deployment = pd.read_csv(deployment_dir / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz", low_memory=False)
    target_index = pd.read_csv(deployment_dir / "PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz")
    target_map = dict(zip(
        target_index.sequence.map(lambda v: hashlib.sha256(v.encode()).hexdigest()),
        target_index.target_chembl_id,
    ))
    positions, split_audit = optimized_splits(data, raw, deployment, target_map)
    full = data.iloc[raw["full_fit"]]
    fit = data.iloc[positions["eval_fit"]]
    source = data.iloc[raw["source"]]
    binary = full[full.binary_observed.eq(1)]
    grouped = binary.groupby("drug_feature_index").binary_label.agg(["size", "nunique"])
    target_counts = binary.groupby("target_feature_index").size().sort_values(ascending=False)
    source_crossed = source[source.min_document_year.le(2022) & source.max_document_year.gt(2022)]
    fit_crossed = fit[fit.min_document_year.le(2022) & fit.max_document_year.gt(2022)]
    columns = ["source_kind", "target_chembl_id", "parent_molregno", "parent_molecule_chembl_id",
               "parent_standard_inchi_key", "min_document_year", "max_document_year",
               "mean_pchembl", "min_pchembl", "max_pchembl", "calibration_label", "assay_ids"]
    fit_crossed[columns].to_csv(OUT / "temporal_cross_cutoff_relations.csv", index=False)
    result = {
        "scope": "existing 20260901 artifacts; diagnostic only; not a new test or model selection",
        "relations_file": str(relation_path.relative_to(ROOT)),
        "relations_sha256": hashlib.sha256(relation_path.read_bytes()).hexdigest(),
        "split_audit": split_audit,
        "supervision": {
            "full_fit_eligible_rows": len(full), "binary_unique_rows": len(binary),
            "binary_drug_feature_entities": len(grouped),
            "singleton_binary_drugs": int(grouped['size'].eq(1).sum()),
            "two_class_drugs": int(grouped['nunique'].eq(2).sum()),
            "top10_target_fraction": target_counts.iloc[:10].sum() / len(binary),
            "top50_target_fraction": target_counts.iloc[:50].sum() / len(binary),
            "source_endpoint_counts": source.standard_types.fillna("MISSING").value_counts().to_dict(),
            "source_numeric_interval_rows": int(source.max_pchembl.gt(source.min_pchembl).sum()),
            "source_negative_rows_without_numeric_pchembl": int((source.binary_label.eq(0) & source.mean_pchembl.isna()).sum()),
        },
        "temporal": {
            "source_pre2023_rows": int(source.min_document_year.le(2022).sum()),
            "source_pre2023_with_future_records": len(source_crossed),
            "actual_eval_fit_with_future_records": len(fit_crossed),
            "actual_eval_fit_with_2024_or_later_records": int(fit_crossed.max_document_year.ge(2024).sum()),
            "note": "Cross-cutoff evidence confirms use of post-cutoff aggregate information; label-change count needs raw activity reaggregation.",
        },
    }
    # Reproduce one actual FULL_FIT seed's row exposure, without neural training.
    summary_path = ROOT / "outputs/retrain_20260901/comprehensive_balanced_full_fit_v2/FULL_FIT_2026_COMPREHENSIVE_BALANCED__seed_20260816/FULL_FIT_RUN_SUMMARY_V2.json"
    summary = json.loads(summary_path.read_text())
    config = BalancedSamplingConfig(**summary["sampler"]["full_fit"])
    seen = set()
    exposure = []
    for epoch in range(1, summary["selection"]["best_epoch"] + 1):
        batches = balanced_training_batches(raw["full_fit"], data, 20260816 + epoch, config)
        emitted = np.concatenate(batches)
        seen.update(emitted.tolist())
        exposure.append({"epoch": epoch, "unique_rows_this_epoch": len(np.unique(emitted)), "unique_rows_cumulative": len(seen)})
        print(json.dumps({"sampling": exposure[-1]}), flush=True)
    result["sampling_exposure_seed20260816"] = {
        "coverage_sweep": summary["sampler"]["coverage_sweep"],
        "eligible_rows": len(full), "seen_rows": len(seen), "never_drawn_rows": len(full) - len(seen),
        "epochs": exposure,
    }
    result["backbone_config"] = summary["model"]["config"]
    result["backbone_selection"] = summary["selection"]
    # InfoNCE: only d0-t0 and d1-t1 are observed. Off-diagonals are unknown.
    similarity = torch.tensor([[0.2, 0.8], [0.4, 0.3]], requires_grad=True)
    loss = _multi_positive_info_nce(similarity, torch.eye(2, dtype=torch.bool), 0.1)
    loss.backward()
    result["unknown_contrastive_gradient_example"] = {
        "similarity": similarity.detach().tolist(), "positive_mask": [[1, 0], [0, 1]],
        "loss": loss.item(), "gradient": similarity.grad.tolist(),
        "interpretation": "Positive off-diagonal derivatives make gradient descent decrease unobserved interaction similarity.",
    }
    # The listwise term can ignore a weak positive once one positive dominates.
    logits = torch.tensor([10.0, -10.0, 0.0], requires_grad=True)
    loss = within_group_listwise_loss(logits, torch.tensor([1., 1., 0.]), torch.zeros(3, dtype=torch.long))
    loss.backward()
    result["listwise_positive_mass_example"] = {
        "positive_positive_negative_logits": logits.detach().tolist(),
        "loss": loss.item(), "gradient": logits.grad.tolist(),
        "note": "Illustrates only the listwise component; pairwise/BCE terms still supervise the weak positive.",
    }
    interval_examples = []
    for prediction in [0.0, 10.0, -10.0]:
        pred = torch.tensor([prediction], requires_grad=True)
        loss = censored_affinity_loss(pred, torch.tensor([1.]), torch.tensor([2.]), log_variance=torch.tensor([0.]))
        loss.backward()
        interval_examples.append({"prediction": prediction, "lower": 1., "upper": 2., "loss": loss.item(), "gradient": pred.grad.item()})
    result["finite_interval_tail_gradient_examples"] = interval_examples
    # Existing S5 scores: describe baselines and candidate size, without fitting.
    s5_dir = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1"
    s5 = pd.read_csv(s5_dir / "BIOMASTER_DRUG_TO_TARGET_S5_TEST_V1.csv.gz")
    two_class = [g for _, g in s5.groupby("parent_standard_inchi_key") if g.binary_label.nunique() == 2]
    sizes = np.array([len(g) for g in two_class])
    baseline_columns = ["train_positive_max_tanimoto", "target_train_prior", "biomaster_logit", "independent_validation_rank_score", "dtiam_probability"]
    expected_ap = []
    for g in two_class:
        n, m = len(g), int(g.binary_label.sum())
        harmonic_per_n = (1.0 / np.arange(1, n + 1)).sum() / n
        expected_ap.append(harmonic_per_n + (m - 1) / (n - 1) * (1 - harmonic_per_n))
    result["s5_diagnostics"] = {
        "rows": len(s5), "all_drugs": s5.parent_standard_inchi_key.nunique(), "two_class_drugs": len(two_class),
        "two_class_query_size_quantiles": {str(q): float(np.quantile(sizes, q)) for q in [0, .25, .5, .75, 1]},
        "two_class_queries_at_most_20_candidates": int((sizes <= 20).sum()),
        "random_order_expected_drug_macro_recall20": float(np.minimum(20 / sizes, 1).mean()),
        "random_order_expected_drug_macro_ap": float(np.mean(expected_ap)),
        "existing_score_drug_macro_ap": {c: float(np.mean([average_precision_score(g.binary_label, g[c]) for g in two_class])) for c in baseline_columns if c in s5},
        "existing_bootstrap": json.loads((s5_dir / "BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json").read_text())["s5_test"]["independent_rank_vs_dtiam_query_bootstrap"],
    }
    # Dependency audit of the expanded KIRHub adapter's second-stage fusion.
    result["kirhub_fusion_fold_dependencies"] = []
    for outer_test in range(5):
        fusion_valid = (outer_test + 1) % 5
        generating_model_valid = (fusion_valid + 1) % 5
        generating_model_train = sorted(set(range(5)) - {fusion_valid, generating_model_valid})
        result["kirhub_fusion_fold_dependencies"].append({
            "outer_test": outer_test, "fusion_validation": fusion_valid,
            "validation_score_model_train_folds": generating_model_train,
            "outer_test_influences_fusion_validation_scores": outer_test in generating_model_train,
        })
    path = OUT / "AUDIT_DIAGNOSTICS.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n")
    print(str(path), flush=True)


if __name__ == "__main__":
    main()

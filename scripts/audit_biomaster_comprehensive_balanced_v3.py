#!/usr/bin/env python3
"""Audit comprehensive FULL_FIT rankings across fixed relation cohorts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from score_biomaster_deployment_augmented_720x384_v1 import CAL_PAIRS  # noqa: E402
from train_biomaster_comprehensive_balanced_v2 import DEPLOY_TARGET_INDEX, RELATIONS  # noqa: E402


SCORE_PATHS = {
    "capped_v1": ROOT / (
        "outputs/biomaster_deployment_augmented_720x384_v1/"
        "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
    ),
    "raw_comprehensive_v1": ROOT / (
        "outputs/biomaster_comprehensive_full_fit_720x384_v1/"
        "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
    ),
    "balanced_linear045_v2": ROOT / (
        "outputs/biomaster_comprehensive_balanced_full_fit_720x384_v2/"
        "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
    ),
    "balanced_binary_v3": ROOT / (
        "outputs/biomaster_comprehensive_balanced_full_fit_720x384_v3/"
        "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
    ),
    "binary_selected_v4": ROOT / (
        "outputs/biomaster_comprehensive_binary_selected_720x384_v4/"
        "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
    ),
    "consensus_gated_v5": ROOT / (
        "outputs/biomaster_comprehensive_consensus_720x384_v5/"
        "CONSENSUS_FULL_FIT_720X384_SCORES_V5.csv.gz"
    ),
}
OUT = ROOT / "outputs/biomaster_comprehensive_consensus_720x384_v5"


def target_by_sequence() -> dict[str, str]:
    index = pd.read_csv(DEPLOY_TARGET_INDEX, low_memory=False)
    index["sequence_sha256"] = index["sequence"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return dict(zip(
        index["sequence_sha256"].astype(str),
        index["target_chembl_id"].astype(str),
        strict=True,
    ))


def cohort_relations(frozen_drugs: set[str]) -> pd.DataFrame:
    relations = pd.read_csv(RELATIONS, low_memory=False)
    observed = pd.to_numeric(
        relations["binary_observed"], errors="coerce"
    ).fillna(0).eq(1)
    relations = relations.loc[
        relations["source_kind"].eq("chembl37_comprehensive") & observed
    ].copy()
    relations["deployment_target_chembl_id"] = relations["sequence_sha256"].astype(str).map(
        target_by_sequence()
    )
    relations = relations.loc[
        relations["deployment_target_chembl_id"].notna()
        & relations["parent_standard_inchi_key"].astype(str).isin(frozen_drugs)
    ].copy()
    relations["deployment_pair_key"] = (
        relations["deployment_target_chembl_id"].astype(str)
        + "__"
        + relations["parent_standard_inchi_key"].astype(str)
    )
    relations = relations.sort_values("calibration_pair_id").drop_duplicates(
        "deployment_pair_key"
    )
    capped = pd.read_csv(CAL_PAIRS, low_memory=False)
    capped_keys = set(
        capped["target_chembl_id"].astype(str)
        + "__"
        + capped["parent_standard_inchi_key"].astype(str)
    )
    relations["cohort"] = (
        np.where(relations["deployment_pair_key"].isin(capped_keys), "capped_", "newly_added_")
        + np.where(relations["binary_label"].eq(1), "positive", "negative")
    )
    return relations


def main() -> None:
    missing = [str(path) for path in [RELATIONS, DEPLOY_TARGET_INDEX, CAL_PAIRS, *SCORE_PATHS.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in SCORE_PATHS.items()}
    frozen_drugs = set(frames["consensus_gated_v5"]["ligand_inchikey"].astype(str))
    relations = cohort_relations(frozen_drugs)
    for name, frame in frames.items():
        lookup = frame.set_index(
            frame["target_chembl_id"].astype(str)
            + "__"
            + frame["ligand_inchikey"].astype(str)
        )["ensemble_rank_within_target_720"]
        relations[name] = lookup.reindex(relations["deployment_pair_key"]).to_numpy(dtype=float)
        if relations[name].isna().any():
            raise RuntimeError(f"{name} did not map every cohort relation")

    rows: list[dict[str, object]] = []
    for cohort, part in relations.groupby("cohort", sort=True):
        for model in SCORE_PATHS:
            ranks = part[model].to_numpy(dtype=float)
            rows.append({
                "cohort": cohort,
                "rows": int(len(part)),
                "model": model,
                "mean_rank": float(ranks.mean()),
                "median_rank": float(np.median(ranks)),
                "hit_at_10": float(np.mean(ranks <= 10)),
                "hit_at_36": float(np.mean(ranks <= 36)),
                "hit_at_72": float(np.mean(ranks <= 72)),
            })
    comparison = pd.DataFrame(rows)
    overall = []
    labels = relations["binary_label"].to_numpy(dtype=np.int8)
    for model in SCORE_PATHS:
        score = 1.0 - (relations[model].to_numpy(dtype=float) - 1.0) / 720.0
        overall.append({
            "model": model,
            "rows": int(len(relations)),
            "positive_rows": int(labels.sum()),
            "negative_rows": int((labels == 0).sum()),
            "auprc": float(average_precision_score(labels, score)),
            "auroc": float(roc_auc_score(labels, score)),
        })
    overall_frame = pd.DataFrame(overall)

    comparison_path = OUT / "COMPREHENSIVE_RELATION_COHORT_COMPARISON_V5.csv"
    overall_path = OUT / "COMPREHENSIVE_RELATION_OVERALL_METRICS_V5.csv"
    mapped_path = OUT / "COMPREHENSIVE_RELATION_MAPPED_RANKS_V5.csv.gz"
    comparison.to_csv(comparison_path, index=False)
    overall_frame.to_csv(overall_path, index=False)
    relations[[
        "calibration_pair_id",
        "deployment_target_chembl_id",
        "parent_standard_inchi_key",
        "binary_label",
        "cohort",
        *SCORE_PATHS,
    ]].to_csv(mapped_path, index=False, compression="gzip")
    expected_counts = {
        "capped_negative": 1925,
        "capped_positive": 310,
        "newly_added_negative": 3177,
        "newly_added_positive": 580,
    }
    actual_counts = relations["cohort"].value_counts().sort_index().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"cohort membership changed: {actual_counts}")
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "COMPREHENSIVE_CONSENSUS_FULL_FIT_720X384_AUDIT_V5",
        "cohort_counts": actual_counts,
        "ranking_universe": "720 frozen old drugs independently within each target",
        "score_paths": {name: str(path.relative_to(ROOT)) for name, path in SCORE_PATHS.items()},
        "artifacts": {
            "cohort_comparison": str(comparison_path.relative_to(ROOT)),
            "overall_metrics": str(overall_path.relative_to(ROOT)),
            "mapped_ranks": str(mapped_path.relative_to(ROOT)),
        },
    }
    summary_path = OUT / "COMPREHENSIVE_RELATION_AUDIT_SUMMARY_V5.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(comparison.to_string(index=False))
    print(overall_frame.to_string(index=False))
    print(json.dumps({"status": "PASS", "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

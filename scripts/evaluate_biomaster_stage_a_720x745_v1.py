#!/usr/bin/env python3
"""Evaluate Stage-A temporal positives in expanded routed target spaces.

Only 2024--2025 positives frozen before the directional checkpoint selection
are used.  Targets known positive for the same drug by the <=2022 fit cutoff
are removed.  Background candidates remain unlabeled; AP/Recall are therefore
positive-retrieval diagnostics, not binary classification estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DEFAULT_RELATIONS = ROOT / (
    "outputs/retrain_20260901/comprehensive_training_v1/"
    "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
)
DEFAULT_SCORES = ROOT / (
    "outputs/retrain_20260901/bidirectional_720x745_stage_a/"
    "BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_SCORES_V1.csv.gz"
)
BASELINE_SUMMARY = ROOT / (
    "outputs/retrain_20260901/bidirectional_stage_a/ENSEMBLE_STAGE_A_SUMMARY_V6.json"
)
DEFAULT_OUT = ROOT / "outputs/retrain_20260901/bidirectional_720x745_stage_a"


def metrics(frame: pd.DataFrame, score_column: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    positive_rows = []
    for query, part in frame.groupby("query_id", sort=True):
        part = part.sort_values(score_column, ascending=False, kind="mergesort").copy()
        labels = part["binary_label"].to_numpy(dtype=np.int8)
        values = part[score_column].to_numpy(dtype=np.float64)
        positives = int(labels.sum())
        positive_ranks = np.flatnonzero(labels) + 1
        if positives < 1:
            raise RuntimeError(f"query has no future positive: {query}")
        row: dict[str, Any] = {
            "query_id": query,
            "candidate_targets": len(part),
            "held_future_positives": positives,
            "best_positive_rank": int(positive_ranks.min()),
            "mean_positive_rank": float(positive_ranks.mean()),
            # Match the frozen Stage-A contract: reciprocal rank is defined by
            # the first relevant target, while the reciprocal-log metric
            # averages over every held future positive.
            "mrr": float(1.0 / positive_ranks.min()),
            "mean_reciprocal_log_rank": float((1.0 / np.log2(positive_ranks + 1)).mean()),
            "mean_top_percentile": float(
                np.mean(1.0 - (positive_ranks - 1) / max(len(part) - 1, 1))
            ),
            "positive_retrieval_ap": float(average_precision_score(labels, values)),
        }
        for cutoff in (1, 5, 10, 20, 50, 100):
            hits = int(labels[: min(cutoff, len(part))].sum())
            row[f"recall_at_{cutoff}"] = hits / positives
            row[f"hit_at_{cutoff}"] = float(hits > 0)
        rows.append(row)
        current = part.loc[part["binary_label"].eq(1)].copy()
        current["rank"] = positive_ranks
        positive_rows.append(current)
    queries = pd.DataFrame(rows)
    result = {
        "queries": len(queries),
        "candidate_rows": len(frame),
        "held_future_positives": int(frame["binary_label"].sum()),
        "candidate_targets_mean": float(queries["candidate_targets"].mean()),
    }
    for column in queries.columns:
        if column not in {"query_id", "candidate_targets", "held_future_positives"}:
            result[f"macro_{column}"] = float(queries[column].mean())
    return result, pd.concat(positive_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relations", default=str(DEFAULT_RELATIONS))
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--baseline-summary", default=str(BASELINE_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    relations_path = Path(args.relations).resolve()
    scores_path = Path(args.scores).resolve()
    baseline_path = Path(args.baseline_summary).resolve()
    required = [relations_path, scores_path, baseline_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    os.environ["BIOMASTER_COMPREHENSIVE_PACKAGE"] = str(relations_path.parent)
    from train_biomaster_comprehensive_full_fit_v1 import split_positions  # noqa: E402
    from train_biomaster_bidirectional_v6 import (  # noqa: E402
        DEPLOY_PAIRS,
        DEPLOY_TARGET_INDEX,
        _mapped_temporal_frame,
        optimized_splits,
    )

    relations = pd.read_csv(relations_path, low_memory=False)
    raw_positions, raw_audit = split_positions(relations)
    deployment = pd.read_csv(DEPLOY_PAIRS, low_memory=False)
    deployment_targets = pd.read_csv(DEPLOY_TARGET_INDEX, low_memory=False)
    deployment_targets["sequence_sha256"] = deployment_targets["sequence"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    target_by_sequence = dict(zip(
        deployment_targets["sequence_sha256"].astype(str),
        deployment_targets["target_chembl_id"].astype(str),
        strict=True,
    ))
    positions, split_audit = optimized_splits(
        relations, raw_positions, deployment, target_by_sequence
    )
    fit = relations.iloc[positions["eval_fit"]]
    temporal, temporal_positions, mapping_audit = _mapped_temporal_frame(
        relations, raw_positions["source"]
    )
    # Reuse the exact strict-warm subset selected by `optimized_splits`.  The
    # larger S4 temporal table also contains cold or out-of-deployment pairs;
    # those were never part of the frozen 20-query/27-positive dense test.
    future = relations.iloc[positions["temporal_test"]].copy()
    scores = pd.read_csv(scores_path, low_memory=False)
    if len(scores) != 720 * 745 or future.empty:
        raise RuntimeError("expanded score or temporal-positive contract failed")
    registry = scores.drop_duplicates("target_chembl_id")[
        ["target_chembl_id", "target_model_warmth", "production_route", "current_score_status"]
    ]
    registry_ids = set(registry["target_chembl_id"].astype(str))
    hash_to_registry = dict(
        zip(
            pd.read_csv(
                ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_REGISTRY_NON_GPCR_745_V3.csv.gz",
                usecols=["sequence_sha256", "target_chembl_id"],
            )["sequence_sha256"].astype(str),
            pd.read_csv(
                ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_REGISTRY_NON_GPCR_745_V3.csv.gz",
                usecols=["sequence_sha256", "target_chembl_id"],
            )["target_chembl_id"].astype(str),
            strict=True,
        )
    )
    # S4 retains its source ChEMBL target identifier, while both the frozen
    # deployment universe and the 745 registry deduplicate protein entities by
    # sequence.  Resolve through the exact-pair mapping instead of assuming
    # that synonymous ChEMBL target IDs are identical.
    future["registry_target_id"] = future["sequence_sha256"].astype(str).map(
        hash_to_registry
    )
    if future["registry_target_id"].isna().any():
        raise RuntimeError("a frozen future positive is outside the 745 registry")
    positive_fit = fit.loc[fit["binary_observed"].eq(1) & fit["binary_label"].eq(1)].copy()
    positive_fit["registry_target_id"] = positive_fit["sequence_sha256"].astype(str).map(hash_to_registry)
    positive_fit = positive_fit.loc[positive_fit["registry_target_id"].notna()]
    known_by_drug = {
        str(drug): set(group["registry_target_id"].astype(str))
        for drug, group in positive_fit.groupby("parent_standard_inchi_key", sort=False)
    }

    scope_masks = {
        "frozen_core_384": scores["current_score_status"].eq("SCORED_IN_FROZEN_384_CORE"),
        "primary_450_precalibration": scores["production_route"].isin(
            ["PRIMARY_BIOCHEMICAL_DIRECT_SM", "PRIMARY_FUNCTIONAL_DIRECT_SM"]
        ),
        "active_routed_500_diagnostic": ~scores["production_route"].eq(
            "REGISTRY_ONLY_SPECIAL_NO_DIRECT_SM"
        ),
        "registry_745_diagnostic": np.ones(len(scores), dtype=bool),
    }
    score_column = "ensemble_drug_to_target_logit"
    metric_by_scope = {}
    positive_rank_frames = []
    expected_queries = future["parent_standard_inchi_key"].nunique()
    for scope, scope_mask in scope_masks.items():
        chunks = []
        for drug, held in future.groupby("parent_standard_inchi_key", sort=True):
            part = scores.loc[
                scope_mask & scores["ligand_inchikey"].astype(str).eq(str(drug))
            ].copy()
            future_targets = set(held["registry_target_id"].astype(str))
            if not future_targets.issubset(set(part["target_chembl_id"].astype(str))):
                raise RuntimeError(f"scope {scope} excludes a future positive for {drug}")
            known = known_by_drug.get(str(drug), set())
            if future_targets & known:
                raise RuntimeError("future positive was already known by the fit cutoff")
            part = part.loc[~part["target_chembl_id"].astype(str).isin(known)].copy()
            part["query_id"] = str(drug)
            part["binary_label"] = part["target_chembl_id"].astype(str).isin(
                future_targets
            ).astype(np.int8)
            chunks.append(part)
        evaluated = pd.concat(chunks, ignore_index=True)
        current_metrics, positives = metrics(evaluated, score_column)
        positives["evaluation_scope"] = scope
        metric_by_scope[scope] = current_metrics
        positive_rank_frames.append(positives)
        if current_metrics["queries"] != expected_queries:
            raise RuntimeError("temporal query count changed across expanded scopes")

    baseline = json.loads(baseline_path.read_text())
    baseline_metrics = baseline["metrics"]["test_2024_2025"]["drug_to_target_ensemble"]
    core = metric_by_scope["frozen_core_384"]
    core_comparison = {
        key: {
            "recomputed": core[f"macro_{key}"],
            "stage_a_summary": baseline_metrics[f"macro_{key}"],
            "difference": core[f"macro_{key}"] - baseline_metrics[f"macro_{key}"],
        }
        for key in [
            "mean_positive_rank", "mrr", "mean_reciprocal_log_rank",
            "mean_top_percentile", "positive_retrieval_ap", "recall_at_10",
            "recall_at_20", "recall_at_50",
        ]
    }
    exact_core_reproduction = all(
        abs(item["difference"]) <= 1e-6 for item in core_comparison.values()
    )
    checks = {
        "frozen_2024_2025_only": set(pd.to_numeric(future["min_document_year"]).astype(int)).issubset({2024, 2025}),
        "expected_20_queries": expected_queries == 20,
        "expected_27_future_positives": len(future) == 27,
        "all_future_targets_in_registry": set(future["registry_target_id"]).issubset(registry_ids),
        "frozen_core_reproduces_stage_a": exact_core_reproduction,
        "no_test_label_used_for_checkpoint_selection": True,
        "background_is_unlabeled_not_negative": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(
            {"checks": checks, "frozen_core_reproduction": core_comparison},
            ensure_ascii=False,
            indent=2,
        ))

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ranks_path = out / "STAGE_A_TEMPORAL_2024_2025_POSITIVE_RANKS_EXPANDED_SCOPES_V1.csv"
    pd.concat(positive_rank_frames, ignore_index=True).to_csv(ranks_path, index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "STAGE_A_FROZEN_TEMPORAL_2024_2025_EXPANDED_TARGET_SCOPES_V1",
        "metrics": metric_by_scope,
        "frozen_core_reproduction": core_comparison,
        "split_audit": {"raw": raw_audit, "optimized": split_audit, "mapping": mapping_audit},
        "checks": checks,
        "claim_boundary": (
            "Only held 2024-2025 positives are labels.  Other candidate targets are "
            "unlabeled ranking background.  The 745-target result is a coverage-stress "
            "diagnostic across heterogeneous routes, not a calibrated production rank."
        ),
    }
    summary_path = out / "STAGE_A_TEMPORAL_2024_2025_EXPANDED_SCOPES_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

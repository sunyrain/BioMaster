#!/usr/bin/env python3
"""Stage-A training for the BioMaster bidirectional retrieval heads.

The audited comprehensive V5 evaluation checkpoint is the immutable pair
backbone.  Two zero-start residual heads are trained with query-first batches:
drug -> target is the production direction and target -> drug is auxiliary.
Only explicit binary observations enter either directional objective.

Epoch selection uses the mapped 2023 portion of the frozen S4 first-seen set.
The 2024--2025 portion is scored exactly once after selection.  This script is
therefore a head-only screen, not a FULL_FIT production refit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.comprehensive_balanced import (  # noqa: E402
    QueryFirstBatchSampler,
    QuerySamplingConfig,
)
from biomaster.odti_v2 import (  # noqa: E402
    ODTIV2Config,
    RoutedInteractionRankerV2,
    directional_retrieval_loss,
)
from train_biomaster_comprehensive_balanced_v2 import (  # noqa: E402
    DEPLOY_DRUG,
    DEPLOY_PAIRS,
    DEPLOY_TARGET,
    DEPLOY_TARGET_AUX,
    DEPLOY_TARGET_INDEX,
    DRUG_FEATURE_INDEX,
    external_arrays,
    optimized_splits,
)
from score_biomaster_deployment_augmented_720x384_v1 import augmented_structure  # noqa: E402
from train_biomaster_comprehensive_full_fit_v1 import (  # noqa: E402
    ESM2,
    MORGAN,
    PACKAGE_MANIFEST,
    PROTBERT,
    RELATIONS,
    exact_keys,
    split_positions,
    target_structure_arrays,
)
from train_biomaster_odti_v2 import predict, set_seed, tensor_batch  # noqa: E402


S4_PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DEFAULT_REFERENCE = ROOT / (
    "outputs/biomaster_comprehensive_balanced_full_fit_v2/"
    "FULL_FIT_2026_COMPREHENSIVE_BALANCED__seed_20260816/"
    "EVALUATION_BEST_MODEL_COMPREHENSIVE_BALANCED_V2.pt"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_bidirectional_v6_stage_a"


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _grouped_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
    group_column: str,
    prefix: str,
) -> dict[str, float | int | None]:
    aucs: list[float] = []
    aps: list[float] = []
    recalls = {5: [], 10: [], 20: []}
    ndcgs = {5: [], 10: [], 20: []}
    work = frame[[group_column, "binary_label"]].copy()
    work["score"] = np.asarray(score, dtype=np.float64)
    for _, part in work.groupby(group_column, sort=False):
        labels = part["binary_label"].to_numpy(dtype=np.int8)
        values = part["score"].to_numpy(dtype=np.float64)
        if labels.min() == labels.max():
            continue
        aucs.append(float(roc_auc_score(labels, values)))
        aps.append(float(average_precision_score(labels, values)))
        order = np.argsort(-values, kind="stable")
        positives = int(labels.sum())
        for cutoff in recalls:
            top = labels[order[: min(cutoff, len(labels))]]
            recalls[cutoff].append(float(top.sum() / positives))
            discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
            dcg = float((top * discounts).sum())
            ideal = min(positives, len(top))
            denominator = float(discounts[:ideal].sum())
            ndcgs[cutoff].append(dcg / denominator if denominator else 0.0)
    result: dict[str, float | int | None] = {
        f"{prefix}_groups_with_both_classes": int(len(aucs)),
        f"{prefix}_macro_auroc": float(np.mean(aucs)) if aucs else None,
        f"{prefix}_macro_auprc": float(np.mean(aps)) if aps else None,
    }
    for cutoff in recalls:
        result[f"{prefix}_macro_recall_at_{cutoff}"] = (
            float(np.mean(recalls[cutoff])) if recalls[cutoff] else None
        )
        result[f"{prefix}_macro_ndcg_at_{cutoff}"] = (
            float(np.mean(ndcgs[cutoff])) if ndcgs[cutoff] else None
        )
    return result


def retrieval_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "micro_auroc": float(roc_auc_score(labels, values)),
        "micro_auprc": float(average_precision_score(labels, values)),
    }
    result.update(
        _grouped_metrics(
            frame, values, "target_chembl_id", "target"
        )
    )
    result.update(
        _grouped_metrics(
            frame, values, "parent_standard_inchi_key", "drug"
        )
    )
    return result


def drug_to_target_dense_metrics(
    frame: pd.DataFrame, score: np.ndarray
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Rank held future positives among an unlabeled target candidate set."""

    work = frame[[
        "query_id",
        "ligand_inchikey",
        "drug_names",
        "target_chembl_id",
        "gene_symbol",
        "binary_label",
        "candidate_status",
    ]].copy()
    work["score"] = np.asarray(score, dtype=np.float64)
    work["rank"] = work.groupby("query_id", sort=False)["score"].rank(
        method="min", ascending=False
    ).astype(int)
    query_rows: list[dict[str, Any]] = []
    known_rows = []
    for query, part in work.groupby("query_id", sort=False):
        labels = part["binary_label"].to_numpy(dtype=np.int8)
        values = part["score"].to_numpy(dtype=np.float64)
        positives = int(labels.sum())
        if positives < 1:
            raise RuntimeError("dense drug query has no held future positive")
        order = np.argsort(-values, kind="stable")
        ordered = labels[order]
        positive_ranks = np.flatnonzero(ordered == 1) + 1
        discounts = 1.0 / np.log2(np.arange(2, len(ordered) + 2))
        row: dict[str, Any] = {
            "query_id": str(query),
            "candidate_targets": int(len(part)),
            "held_future_positives": positives,
            "best_positive_rank": int(positive_ranks.min()),
            "mean_positive_rank": float(positive_ranks.mean()),
            "mrr": float(1.0 / positive_ranks.min()),
            "mean_reciprocal_log_rank": float(
                np.mean(1.0 / np.log2(positive_ranks + 1.0))
            ),
            "mean_top_percentile": float(
                np.mean(1.0 - (positive_ranks - 1) / max(len(part) - 1, 1))
            ),
            # AP is a positive-retrieval diagnostic; background candidates are
            # unlabeled rather than asserted biochemical negatives.
            "positive_retrieval_ap": float(average_precision_score(labels, values)),
        }
        for cutoff in (1, 5, 10, 20, 50):
            top = ordered[: min(cutoff, len(ordered))]
            row[f"recall_at_{cutoff}"] = float(top.sum() / positives)
            row[f"hit_at_{cutoff}"] = float(top.sum() > 0)
            dcg = float((top * discounts[: len(top)]).sum())
            ideal = min(positives, len(top))
            denominator = float(discounts[:ideal].sum())
            row[f"ndcg_at_{cutoff}"] = dcg / denominator if denominator else 0.0
        query_rows.append(row)
        known_rows.append(part.loc[part["binary_label"].eq(1)])
    queries = pd.DataFrame(query_rows)
    metric_columns = [
        column for column in queries.columns if column not in {"query_id", "candidate_targets", "held_future_positives"}
    ]
    metrics: dict[str, Any] = {
        "queries": int(len(queries)),
        "candidate_rows": int(len(work)),
        "held_future_positives": int(work["binary_label"].sum()),
        "candidate_targets_mean": float(queries["candidate_targets"].mean()),
    }
    metrics.update(
        {f"macro_{column}": float(queries[column].mean()) for column in metric_columns}
    )
    return metrics, pd.concat(known_rows, ignore_index=True)


def build_drug_to_target_frame(
    positives: pd.DataFrame,
    deployment: pd.DataFrame,
    deployment_structure: np.ndarray,
    deployment_structure_mask: np.ndarray,
    target_by_sequence: dict[str, str],
    known_targets_by_drug: dict[str, set[str]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build cutoff-clean old-drug -> 384-target prospective queries."""

    future = positives.loc[positives["binary_label"].eq(1)].copy()
    future["deployment_target_chembl_id"] = future["sequence_sha256"].astype(str).map(
        target_by_sequence
    )
    if future["deployment_target_chembl_id"].isna().any():
        raise RuntimeError("a held future target is absent from the 384-target universe")
    chunks = []
    structures = []
    masks = []
    for drug, held in future.groupby("parent_standard_inchi_key", sort=True):
        candidates = deployment.loc[
            deployment["ligand_inchikey"].astype(str).eq(str(drug))
        ].copy()
        if len(candidates) != 384:
            raise RuntimeError(f"old-drug query does not have 384 targets: {drug}")
        future_targets = set(held["deployment_target_chembl_id"].astype(str))
        known_targets = known_targets_by_drug.get(str(drug), set())
        if future_targets & known_targets:
            raise RuntimeError("future positive was already positive by the fit cutoff")
        retained = ~candidates["target_chembl_id"].astype(str).isin(known_targets)
        candidates = candidates.loc[retained].copy()
        original = candidates.index.to_numpy(dtype=np.int64)
        candidates["query_id"] = str(drug)
        candidates["parent_standard_inchi_key"] = str(drug)
        candidates["binary_label"] = candidates["target_chembl_id"].astype(str).isin(
            future_targets
        ).astype(np.int8)
        if int(candidates["binary_label"].sum()) != len(future_targets):
            raise RuntimeError("held future positives did not map exactly once")
        candidates["candidate_status"] = np.where(
            candidates["binary_label"].eq(1),
            "HELD_FUTURE_POSITIVE",
            "UNLABELED_AT_CUTOFF",
        )
        candidates["calibration_pair_id"] = (
            "D2T::" + str(drug) + "::" + candidates["target_feature_index"].astype(str)
        )
        candidates["binary_observed"] = 0
        candidates["mean_pchembl"] = np.nan
        candidates["min_pchembl"] = np.nan
        candidates["max_pchembl"] = np.nan
        chunks.append(candidates)
        structures.append(deployment_structure[original])
        masks.append(deployment_structure_mask[original])
    if not chunks:
        raise RuntimeError("dense drug-to-target frame has no query")
    return (
        pd.concat(chunks, ignore_index=True),
        np.concatenate(structures).astype(np.float32),
        np.concatenate(masks).astype(np.float32),
    )


def _arrays_from_checkpoint(
    data: pd.DataFrame, checkpoint: dict[str, Any]
) -> dict[str, object]:
    normalization = checkpoint["normalization"]
    families = [str(value) for value in checkpoint["families"]]
    family_lookup = {name: index for index, name in enumerate(families)}
    if "__UNK__" not in family_lookup:
        raise RuntimeError("reference checkpoint has no __UNK__ family")
    family_index = (
        data["target_assay_family"]
        .fillna("__UNK__")
        .astype(str)
        .map(family_lookup)
        .fillna(family_lookup["__UNK__"])
        .to_numpy(dtype=np.int64)
    )
    affinity = pd.to_numeric(data["mean_pchembl"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    lower = pd.to_numeric(data["min_pchembl"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    upper = pd.to_numeric(data["max_pchembl"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    return {
        **normalization,
        "families": families,
        "family_index": family_index,
        "drug_aux_mean": np.zeros(0, dtype=np.float32),
        "drug_aux_std": np.ones(0, dtype=np.float32),
        "conplex": pd.to_numeric(data["conplex_score"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32),
        "affinity": affinity,
        "affinity_lower": lower,
        "affinity_upper": upper,
    }


def _mapped_temporal_frame(
    data: pd.DataFrame, source_positions: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    old = pd.read_csv(S4_PAIRS, low_memory=False)
    old = old.loc[old["temporal_role"].eq("TEST_FIRST_SEEN_2023_2025")].copy()
    old["exact_pair_key"] = exact_keys(old)
    source = data.iloc[source_positions][["exact_pair_key"]].copy()
    if source["exact_pair_key"].duplicated().any():
        raise RuntimeError("comprehensive source exact pair keys are not unique")
    lookup = pd.Series(source.index.to_numpy(dtype=np.int64), index=source["exact_pair_key"])
    mapped = old["exact_pair_key"].map(lookup)
    missing = old.loc[mapped.isna(), [
        "calibration_pair_id", "target_chembl_id", "parent_standard_inchi_key", "binary_label"
    ]]
    kept = old.loc[mapped.notna()].copy().reset_index(drop=True)
    positions = mapped.loc[mapped.notna()].to_numpy(dtype=np.int64)
    comprehensive_labels = data.iloc[positions]["binary_label"].to_numpy(dtype=np.int8)
    if not np.array_equal(
        comprehensive_labels, kept["binary_label"].to_numpy(dtype=np.int8)
    ):
        raise RuntimeError("mapped S4 labels disagree with comprehensive relations")
    kept["comprehensive_position"] = positions
    audit = {
        "s4_temporal_rows": int(len(old)),
        "mapped_rows": int(len(kept)),
        "quarantined_unmapped_rows": int(len(missing)),
        "quarantined_pairs": missing.to_dict("records"),
        "by_year": {
            str(int(year)): int(count)
            for year, count in kept.groupby("min_document_year").size().items()
        },
    }
    return kept, positions, audit


def _model_forward(
    model: RoutedInteractionRankerV2, values: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(
        values["drug"],
        values["target"],
        values["family"],
        values["conplex"],
        values["structure"],
        values["structure_mask"],
        target_aux=values["target_aux"],
    )


def _tensor_values(
    batch: np.ndarray,
    data: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    target_aux: np.ndarray,
    structure_features: np.ndarray,
    structure_mask: np.ndarray,
    arrays: dict[str, object],
    config: ODTIV2Config,
    device: torch.device,
    drug_cache: torch.Tensor,
    target_cache: torch.Tensor,
    target_aux_cache: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values = tensor_batch(
        batch,
        data,
        drug_features,
        None,
        target_features,
        target_aux,
        None,
        None,
        None,
        None,
        config.target_token_max_len,
        None,
        None,
        structure_features,
        structure_mask,
        arrays,
        device,
        drug_feature_cache=drug_cache,
        target_feature_cache=target_cache,
        target_aux_feature_cache=target_aux_cache,
    )
    values["binary_observed"] = torch.from_numpy(
        data.iloc[batch]["binary_observed"].to_numpy(dtype=np.float32).copy()
    ).to(device)
    return values


def _predict(
    model: RoutedInteractionRankerV2,
    positions: np.ndarray,
    data: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    target_aux: np.ndarray,
    structure_features: np.ndarray,
    structure_mask: np.ndarray,
    arrays: dict[str, object],
    config: ODTIV2Config,
    device: torch.device,
    batch_size: int,
    drug_cache: torch.Tensor,
    target_cache: torch.Tensor,
    target_aux_cache: torch.Tensor,
) -> dict[str, np.ndarray]:
    return predict(
        model,
        positions,
        data,
        drug_features,
        None,
        target_features,
        target_aux,
        None,
        None,
        None,
        None,
        config.target_token_max_len,
        None,
        None,
        structure_features,
        structure_mask,
        arrays,
        device,
        batch_size,
        drug_feature_cache=drug_cache,
        target_feature_cache=target_cache,
        target_aux_feature_cache=target_aux_cache,
    )


def _evaluate(
    model: RoutedInteractionRankerV2,
    frame: pd.DataFrame,
    positions: np.ndarray,
    **prediction_kwargs: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    prediction = _predict(model, positions, **prediction_kwargs)
    return (
        {
            "pair": retrieval_metrics(frame, prediction["final_logit"]),
            "drug_to_target": retrieval_metrics(
                frame, prediction["drug_to_target_logit"]
            ),
            "target_to_drug": retrieval_metrics(
                frame, prediction["target_to_drug_logit"]
            ),
        },
        prediction,
    )


def _predict_external(
    model: RoutedInteractionRankerV2,
    frame: pd.DataFrame,
    structure: np.ndarray,
    structure_mask: np.ndarray,
    arrays: dict[str, object],
    config: ODTIV2Config,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    target_aux: np.ndarray,
    device: torch.device,
    batch_size: int,
    drug_cache: torch.Tensor,
    target_cache: torch.Tensor,
    target_aux_cache: torch.Tensor,
) -> dict[str, np.ndarray]:
    return predict(
        model,
        np.arange(len(frame), dtype=np.int64),
        frame,
        drug_features,
        None,
        target_features,
        target_aux,
        None,
        None,
        None,
        None,
        config.target_token_max_len,
        None,
        None,
        structure,
        structure_mask,
        arrays,
        device,
        batch_size,
        drug_feature_cache=drug_cache,
        target_feature_cache=target_cache,
        target_aux_feature_cache=target_aux_cache,
    )


def _evaluate_dense(
    model: RoutedInteractionRankerV2,
    frame: pd.DataFrame,
    **prediction_kwargs: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    prediction = _predict_external(model, frame, **prediction_kwargs)
    metrics: dict[str, dict[str, Any]] = {}
    known: dict[str, pd.DataFrame] = {}
    for name, key in (
        ("pair", "final_logit"),
        ("drug_to_target", "drug_to_target_logit"),
        ("target_to_drug", "target_to_drug_logit"),
    ):
        metrics[name], known[name] = drug_to_target_dense_metrics(frame, prediction[key])
    return metrics, prediction, known


def _selection_value(metrics: dict[str, dict[str, Any]]) -> float:
    primary = metrics["drug_to_target"]
    return (
        0.35 * float(primary["macro_ndcg_at_20"])
        + 0.25 * float(primary["macro_recall_at_20"])
        + 0.20 * float(primary["macro_mean_top_percentile"])
        + 0.20 * float(primary["macro_mean_reciprocal_log_rank"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--reference-checkpoint", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rows-per-query", type=int, default=16)
    parser.add_argument("--d2t-steps", type=int, default=48)
    parser.add_argument("--t2d-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--bce-weight", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.patience < 1:
        raise ValueError("invalid early-stopping contract")
    required = [
        Path(args.reference_checkpoint),
        RELATIONS,
        MORGAN,
        PROTBERT,
        ESM2,
        PACKAGE_MANIFEST,
        DRUG_FEATURE_INDEX,
        DEPLOY_PAIRS,
        DEPLOY_DRUG,
        DEPLOY_TARGET,
        DEPLOY_TARGET_AUX,
        DEPLOY_TARGET_INDEX,
        S4_PAIRS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(PACKAGE_MANIFEST.read_text()).get("status") != "PASS":
        raise RuntimeError("comprehensive package manifest is not PASS")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(
        args.reference_checkpoint, map_location="cpu", weights_only=False
    )
    contract = checkpoint.get("full_fit_contract", {})
    if contract.get("evaluation_training_cutoff_year_inclusive") != 2022:
        raise RuntimeError("reference checkpoint is not the <=2022 evaluation model")

    data = pd.read_csv(RELATIONS, low_memory=False)
    data["exact_pair_key"] = exact_keys(data)
    drug_index = pd.read_csv(
        DRUG_FEATURE_INDEX,
        usecols=["drug_feature_index", "murcko_scaffold"],
        low_memory=False,
    ).set_index("drug_feature_index")
    data["murcko_scaffold"] = (
        drug_index.loc[
            data["drug_feature_index"].to_numpy(dtype=np.int64), "murcko_scaffold"
        ]
        .reset_index(drop=True)
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    deployment = pd.read_csv(DEPLOY_PAIRS, low_memory=False)
    deployment_target_index = pd.read_csv(DEPLOY_TARGET_INDEX, low_memory=False)
    deployment_target_index["sequence_sha256"] = deployment_target_index["sequence"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    target_by_sequence = dict(
        zip(
            deployment_target_index["sequence_sha256"].astype(str),
            deployment_target_index["target_chembl_id"].astype(str),
            strict=True,
        )
    )
    raw_positions, raw_audit = split_positions(data)
    positions, split_audit = optimized_splits(
        data, raw_positions, deployment, target_by_sequence
    )
    eval_fit = positions["eval_fit"]
    temporal, temporal_positions, mapping_audit = _mapped_temporal_frame(
        data, raw_positions["source"]
    )
    years = pd.to_numeric(temporal["min_document_year"], errors="raise")
    dev_mask = years.eq(2023).to_numpy()
    test_mask = years.isin([2024, 2025]).to_numpy()
    dev_frame = temporal.loc[dev_mask].reset_index(drop=True)
    test_frame = temporal.loc[test_mask].reset_index(drop=True)
    dev_positions = temporal_positions[dev_mask]
    test_positions = temporal_positions[test_mask]

    fit_frame = data.iloc[eval_fit]
    warm_drugs = set(fit_frame["parent_standard_inchi_key"].astype(str))
    warm_targets = set(fit_frame["target_chembl_id"].astype(str))
    for frame in (dev_frame, test_frame):
        frame["drug_warm"] = frame["parent_standard_inchi_key"].astype(str).isin(warm_drugs)
        frame["target_warm"] = frame["target_chembl_id"].astype(str).isin(warm_targets)
        frame["cold_regime"] = np.select(
            [
                frame["drug_warm"] & frame["target_warm"],
                ~frame["drug_warm"] & frame["target_warm"],
                frame["drug_warm"] & ~frame["target_warm"],
            ],
            ["double_warm", "drug_cold", "target_cold"],
            default="double_cold",
        )

    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux = np.load(ESM2, mmap_mode="r")
    structure_features, structure_mask, structure_columns, _, structure_audit = (
        target_structure_arrays(data, len(target_features))
    )
    deployment_structure, deployment_structure_mask, deployment_structure_audit = (
        augmented_structure(deployment)
    )
    fit_positive = fit_frame.loc[
        fit_frame["binary_observed"].eq(1) & fit_frame["binary_label"].eq(1)
    ].copy()
    fit_positive["deployment_target_chembl_id"] = fit_positive[
        "sequence_sha256"
    ].astype(str).map(target_by_sequence)
    fit_positive = fit_positive.loc[
        fit_positive["deployment_target_chembl_id"].notna()
        & fit_positive["parent_standard_inchi_key"].astype(str).isin(
            set(deployment["ligand_inchikey"].astype(str))
        )
    ]
    known_targets_by_drug = {
        str(drug): set(group["deployment_target_chembl_id"].astype(str))
        for drug, group in fit_positive.groupby("parent_standard_inchi_key", sort=False)
    }
    dense_dev_frame, dense_dev_structure, dense_dev_mask = build_drug_to_target_frame(
        data.iloc[positions["temporal_dev"]],
        deployment,
        deployment_structure,
        deployment_structure_mask,
        target_by_sequence,
        known_targets_by_drug,
    )
    dense_test_frame, dense_test_structure, dense_test_mask = build_drug_to_target_frame(
        data.iloc[positions["temporal_test"]],
        deployment,
        deployment_structure,
        deployment_structure_mask,
        target_by_sequence,
        known_targets_by_drug,
    )
    arrays = _arrays_from_checkpoint(data, checkpoint)
    config = ODTIV2Config(
        **{
            **checkpoint["config"],
            "directional_heads_enabled": True,
            "directional_hidden_dim": 64,
            "directional_dropout": 0.0,
        }
    )
    model = RoutedInteractionRankerV2(
        len(checkpoint["families"]), config, use_conplex=False
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith(("drug_to_target_head.", "target_to_drug_head."))
        for key in incompatible.missing_keys
    ):
        raise RuntimeError(f"reference checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for head in (model.drug_to_target_head, model.target_to_drug_head):
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    model.to(device)

    drug_cache = torch.from_numpy(np.asarray(drug_features, dtype=np.float32)).to(device)
    target_cache = torch.from_numpy(
        (np.asarray(target_features, dtype=np.float32) - arrays["target_mean"])
        / arrays["target_std"]
    ).to(device)
    target_aux_cache = torch.from_numpy(
        (np.asarray(target_aux, dtype=np.float32) - arrays["target_aux_mean"])
        / arrays["target_aux_std"]
    ).to(device)
    deploy_drug = np.load(DEPLOY_DRUG, mmap_mode="r")
    deploy_target = np.load(DEPLOY_TARGET, mmap_mode="r")
    deploy_target_aux = np.load(DEPLOY_TARGET_AUX, mmap_mode="r")
    dense_dev_arrays = external_arrays(dense_dev_frame, arrays)
    dense_test_arrays = external_arrays(dense_test_frame, arrays)
    deploy_drug_cache = torch.from_numpy(
        np.asarray(deploy_drug, dtype=np.float32)
    ).to(device)
    deploy_target_cache = torch.from_numpy(
        (np.asarray(deploy_target, dtype=np.float32) - arrays["target_mean"])
        / arrays["target_std"]
    ).to(device)
    deploy_target_aux_cache = torch.from_numpy(
        (np.asarray(deploy_target_aux, dtype=np.float32) - arrays["target_aux_mean"])
        / arrays["target_aux_std"]
    ).to(device)
    prediction_kwargs = {
        "data": data,
        "drug_features": drug_features,
        "target_features": target_features,
        "target_aux": target_aux,
        "structure_features": structure_features,
        "structure_mask": structure_mask,
        "arrays": arrays,
        "config": config,
        "device": device,
        "batch_size": args.inference_batch_size,
        "drug_cache": drug_cache,
        "target_cache": target_cache,
        "target_aux_cache": target_aux_cache,
    }
    baseline_metrics, baseline_prediction = _evaluate(
        model, dev_frame, dev_positions, **prediction_kwargs
    )
    if not (
        np.array_equal(
            baseline_prediction["drug_to_target_logit"],
            baseline_prediction["final_logit"],
        )
        and np.array_equal(
            baseline_prediction["target_to_drug_logit"],
            baseline_prediction["final_logit"],
        )
    ):
        raise RuntimeError("zero-start directional heads changed the V5 baseline")
    dense_dev_prediction_kwargs = {
        "structure": dense_dev_structure,
        "structure_mask": dense_dev_mask,
        "arrays": dense_dev_arrays,
        "config": config,
        "drug_features": deploy_drug,
        "target_features": deploy_target,
        "target_aux": deploy_target_aux,
        "device": device,
        "batch_size": args.inference_batch_size,
        "drug_cache": deploy_drug_cache,
        "target_cache": deploy_target_cache,
        "target_aux_cache": deploy_target_aux_cache,
    }
    baseline_dense_metrics, baseline_dense_prediction, _ = _evaluate_dense(
        model, dense_dev_frame, **dense_dev_prediction_kwargs
    )
    if not np.array_equal(
        baseline_dense_prediction["drug_to_target_logit"],
        baseline_dense_prediction["final_logit"],
    ):
        raise RuntimeError("zero-start head changed dense D->T baseline")
    baseline_value = _selection_value(baseline_dense_metrics)

    d2t_sampling = QuerySamplingConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.d2t_steps,
        rows_per_query=args.rows_per_query,
        query_frequency_power=0.0,
    )
    t2d_sampling = QuerySamplingConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.t2d_steps,
        rows_per_query=args.rows_per_query,
        query_frequency_power=0.25,
    )
    d2t_sampler = QueryFirstBatchSampler(
        eval_fit, data, "drug_feature_index", d2t_sampling
    )
    t2d_sampler = QueryFirstBatchSampler(
        eval_fit, data, "target_feature_index", t2d_sampling
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.learning_rate / 20
    )
    best_value = baseline_value
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(("drug_to_target_head.", "target_to_drug_head."))
    }
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "selection_value": baseline_value,
            **{
                f"dev_dense_{head}_{name}": value
                for head, metrics in baseline_dense_metrics.items()
                for name, value in metrics.items()
            },
            **{
                f"dev_{head}_{name}": value
                for head, metrics in baseline_metrics.items()
                for name, value in metrics.items()
            },
        }
    ]
    no_improvement = 0
    stop_reason = "MAX_EPOCHS"
    for epoch in range(1, args.max_epochs + 1):
        model.eval()
        model.drug_to_target_head.train()
        model.target_to_drug_head.train()
        d2t_batches = d2t_sampler.batches(args.seed + 1000 * epoch)
        t2d_batches = t2d_sampler.batches(args.seed + 1000 * epoch + 1)
        schedule = [
            ("drug_to_target", batch) for batch in d2t_batches
        ] + [("target_to_drug", batch) for batch in t2d_batches]
        random.Random(args.seed + epoch).shuffle(schedule)
        components: dict[str, float] = {}
        for direction, batch in schedule:
            values = _tensor_values(
                batch,
                data,
                drug_features,
                target_features,
                target_aux,
                structure_features,
                structure_mask,
                arrays,
                config,
                device,
                drug_cache,
                target_cache,
                target_aux_cache,
            )
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, values)
            group = values["drug_group"] if direction == "drug_to_target" else values["target_group"]
            losses = directional_retrieval_loss(
                output,
                values["labels"],
                group,
                direction,
                values["binary_observed"],
                bce_weight=args.bce_weight,
                rank_weight=args.rank_weight,
                listwise_weight=args.listwise_weight,
                residual_weight=args.residual_weight,
                max_pairs=config.rank_max_pairs,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip,
            )
            optimizer.step()
            for name, value in losses.items():
                key = f"{direction}_{name}"
                components[key] = components.get(key, 0.0) + float(value.detach())
        scheduler.step()
        dev_metrics, _ = _evaluate(model, dev_frame, dev_positions, **prediction_kwargs)
        dense_dev_metrics, _, _ = _evaluate_dense(
            model, dense_dev_frame, **dense_dev_prediction_kwargs
        )
        selection = _selection_value(dense_dev_metrics)
        row = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selection_value": selection,
            **{
                f"loss_{name}": value
                / (args.d2t_steps if name.startswith("drug_to_target") else args.t2d_steps)
                for name, value in components.items()
            },
            **{
                f"dev_dense_{head}_{name}": value
                for head, metrics in dense_dev_metrics.items()
                for name, value in metrics.items()
            },
            **{
                f"dev_{head}_{name}": value
                for head, metrics in dev_metrics.items()
                for name, value in metrics.items()
            },
        }
        history.append(row)
        print(json.dumps(row, default=json_safe), flush=True)
        if selection > best_value + args.min_delta:
            best_value = selection
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name.startswith(("drug_to_target_head.", "target_to_drug_head."))
            }
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch >= args.min_epochs and no_improvement >= args.patience:
            stop_reason = "DEVELOPMENT_EARLY_STOPPING"
            break

    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    model.eval()
    best_dev_metrics, best_dev_prediction = _evaluate(
        model, dev_frame, dev_positions, **prediction_kwargs
    )
    best_dense_dev_metrics, best_dense_dev_prediction, best_dense_dev_known = (
        _evaluate_dense(model, dense_dev_frame, **dense_dev_prediction_kwargs)
    )
    test_metrics, test_prediction = _evaluate(
        model, test_frame, test_positions, **prediction_kwargs
    )
    dense_test_prediction_kwargs = {
        **dense_dev_prediction_kwargs,
        "structure": dense_test_structure,
        "structure_mask": dense_test_mask,
        "arrays": dense_test_arrays,
    }
    dense_test_metrics, dense_test_prediction, dense_test_known = _evaluate_dense(
        model, dense_test_frame, **dense_test_prediction_kwargs
    )
    pair_unchanged = np.array_equal(
        baseline_prediction["final_logit"], best_dev_prediction["final_logit"]
    )
    target_aux_delta = (
        float(best_dev_metrics["target_to_drug"]["target_macro_auprc"])
        - float(baseline_metrics["pair"]["target_macro_auprc"])
    )
    primary_delta = best_value - baseline_value
    gates = {
        "pair_backbone_exactly_unchanged": pair_unchanged,
        "d2t_dense_development_selection_improved": primary_delta > args.min_delta,
        "t2d_target_macro_auprc_not_degraded_gt_0_01": target_aux_delta >= -0.01,
        "selected_trained_epoch": best_epoch > 0,
    }
    stage_a_pass = all(gates.values())

    run_dir = Path(args.out_dir).resolve() / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "STAGE_A_TRAINING_HISTORY_V6.csv", index=False)
    for name, frame, prediction in (
        ("DEVELOPMENT_2023", dev_frame, best_dev_prediction),
        ("TEST_2024_2025", test_frame, test_prediction),
    ):
        output = frame.copy()
        for key in (
            "final_logit",
            "drug_to_target_logit",
            "target_to_drug_logit",
            "drug_to_target_residual",
            "target_to_drug_residual",
        ):
            output[key] = prediction[key]
        output.to_csv(run_dir / f"{name}_PREDICTIONS_V6.csv.gz", index=False, compression="gzip")
    for name, frame, prediction, known in (
        (
            "DENSE_D2T_DEVELOPMENT_2023",
            dense_dev_frame,
            best_dense_dev_prediction,
            best_dense_dev_known,
        ),
        (
            "DENSE_D2T_TEST_2024_2025",
            dense_test_frame,
            dense_test_prediction,
            dense_test_known,
        ),
    ):
        output = frame.copy()
        for key in (
            "final_logit",
            "drug_to_target_logit",
            "target_to_drug_logit",
            "drug_to_target_residual",
            "target_to_drug_residual",
        ):
            output[key] = prediction[key]
        output.to_csv(
            run_dir / f"{name}_ALL_CANDIDATES_V6.csv.gz",
            index=False,
            compression="gzip",
        )
        for head, known_frame in known.items():
            known_frame.to_csv(
                run_dir / f"{name}_{head.upper()}_HELD_POSITIVE_RANKS_V6.csv",
                index=False,
            )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "RoutedInteractionRankerV2",
            "config": config.__dict__,
            "families": checkpoint["families"],
            "normalization": checkpoint["normalization"],
            "base_checkpoint": str(Path(args.reference_checkpoint).resolve()),
            "stage_a_contract": {
                "fit_cutoff_year_inclusive": 2022,
                "selection_year": 2023,
                "untouched_test_years": [2024, 2025],
                "selection_task": "OLD_DRUG_TO_384_TARGETS_AFTER_REMOVING_PRE_2023_KNOWN_TARGETS",
                "backbone_frozen": True,
                "selected_epoch": best_epoch,
                "stage_a_pass": stage_a_pass,
            },
        },
        run_dir / "STAGE_A_BEST_BIDIRECTIONAL_V6.pt",
    )
    regime_metrics: dict[str, Any] = {}
    for regime, subset in test_frame.groupby("cold_regime", sort=True):
        index = subset.index.to_numpy(dtype=np.int64)
        if subset["binary_label"].nunique() == 2:
            regime_metrics[str(regime)] = {
                "pair": retrieval_metrics(subset, test_prediction["final_logit"][index]),
                "drug_to_target": retrieval_metrics(
                    subset, test_prediction["drug_to_target_logit"][index]
                ),
                "target_to_drug": retrieval_metrics(
                    subset, test_prediction["target_to_drug_logit"][index]
                ),
            }
        else:
            regime_metrics[str(regime)] = {
                "rows": int(len(subset)),
                "positives": int(subset["binary_label"].sum()),
                "metrics_available": False,
            }
    summary = {
        "status": "PASS" if stage_a_pass else "SCREEN_FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_BIDIRECTIONAL_V6_STAGE_A_FROZEN_BACKBONE",
        "seed": args.seed,
        "device": str(device),
        "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()),
        "base_checkpoint_contract": contract,
        "data": {
            "evaluation_fit_rows": int(len(eval_fit)),
            "evaluation_fit_two_class_drugs": int(
                fit_frame.loc[fit_frame["binary_observed"].eq(1)]
                .groupby("drug_feature_index")["binary_label"]
                .nunique()
                .eq(2)
                .sum()
            ),
            "evaluation_fit_two_class_targets": int(
                fit_frame.loc[fit_frame["binary_observed"].eq(1)]
                .groupby("target_feature_index")["binary_label"]
                .nunique()
                .eq(2)
                .sum()
            ),
            "temporal_mapping": mapping_audit,
            "development_regimes": dev_frame["cold_regime"].value_counts().to_dict(),
            "test_regimes": test_frame["cold_regime"].value_counts().to_dict(),
            "dense_d2t_development": {
                "queries": int(dense_dev_frame["query_id"].nunique()),
                "candidate_rows": int(len(dense_dev_frame)),
                "held_future_positives": int(dense_dev_frame["binary_label"].sum()),
            },
            "dense_d2t_test": {
                "queries": int(dense_test_frame["query_id"].nunique()),
                "candidate_rows": int(len(dense_test_frame)),
                "held_future_positives": int(dense_test_frame["binary_label"].sum()),
            },
        },
        "samplers": {
            "drug_to_target": {
                **d2t_sampling.to_dict(),
                "eligible_two_class_queries": d2t_sampler.query_count,
            },
            "target_to_drug": {
                **t2d_sampling.to_dict(),
                "eligible_two_class_queries": t2d_sampler.query_count,
            },
        },
        "loss": {
            "bce_weight": args.bce_weight,
            "rank_weight": args.rank_weight,
            "listwise_weight": args.listwise_weight,
            "residual_weight": args.residual_weight,
        },
        "selection": {
            "task": "old drug -> 384 targets, pre-2023 known targets removed",
            "formula": "0.35*macro_ndcg_at_20 + 0.25*macro_recall_at_20 + 0.20*macro_mean_top_percentile + 0.20*macro_mean_reciprocal_log_rank",
            "baseline_value": baseline_value,
            "best_value": best_value,
            "delta": primary_delta,
            "best_epoch": best_epoch,
            "epochs_completed": len(history) - 1,
            "stop_reason": stop_reason,
        },
        "gates": gates,
        "baseline_dense_d2t_development_metrics": baseline_dense_metrics,
        "best_dense_d2t_development_metrics": best_dense_dev_metrics,
        "untouched_dense_d2t_test_metrics": dense_test_metrics,
        "baseline_development_metrics": baseline_metrics,
        "best_development_metrics": best_dev_metrics,
        "untouched_test_metrics": test_metrics,
        "untouched_test_metrics_by_cold_regime": regime_metrics,
        "split_audit": {**raw_audit, **split_audit},
        "structure": {
            **structure_audit,
            "deployment": deployment_structure_audit,
            "columns": structure_columns,
        },
        "claim_boundary": (
            "Only explicit binary observations through 2022 train the two residual heads. "
            "The V5 pair backbone is frozen. Dense old-drug-to-target ranking of 2023 "
            "future positives selects the head checkpoint after pre-cutoff known targets "
            "are removed; 2024-2025 is evaluated once after selection. Unlabeled candidate "
            "targets are ranking background, not asserted biochemical negatives. This "
            "artifact is not a FULL_FIT model."
        ),
    }
    (run_dir / "STAGE_A_SUMMARY_V6.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), flush=True)


if __name__ == "__main__":
    main()

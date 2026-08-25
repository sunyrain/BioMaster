#!/usr/bin/env python3
"""Select, test, and refit a deployment-augmented BioMaster checkpoint.

The evaluation fit uses every feature-resolved frozen row except deterministic
BindingDB exact-target-cold development/test targets, plus only the recovered
ChEMBL relations assigned to ``train``.  Epoch selection uses the recovered
strict-double-warm development relations and an exact-target-cold BindingDB
development fold.  Tests are evaluated once after selection.

After testing, a fresh deployment model is refit for the selected epoch count
on all available frozen, BindingDB, and recovered historical relations.  The
evaluation checkpoint and its untouched-test predictions remain separate from
the final FULL_FIT deployment checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2, odti_v2_loss
from train_biomaster_bindingdb_affinity_augmented_v1 import (
    affinity_retrieval_metrics,
    build_combined_data,
)
from train_biomaster_odti_v2 import (
    grouped_training_batches,
    load_structure_features,
    predict,
    prepare_arrays,
    set_seed,
    sigmoid,
    temperature_scale,
    tensor_batch,
)


AUGMENT = ROOT / "outputs/biomaster_deployment_augmentation_v1"
RECOVERED = AUGMENT / "RECOVERED_CHEMBL37_RELATIONS_V1.csv.gz"
MORGAN = AUGMENT / "MORGAN2048_UINT8_DEPLOYMENT_AUGMENTED_V1.npy"
PROTBERT = AUGMENT / "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy"
ESM2 = AUGMENT / "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy"
STRUCTURE = AUGMENT / "ODTI_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz"
TARGET_STRUCTURE = AUGMENT / "TARGET_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz"
AUGMENT_MANIFEST = AUGMENT / "DEPLOYMENT_AUGMENTATION_MANIFEST_V1.json"
REFERENCE_CHECKPOINT = ROOT / (
    "outputs/biomaster_bindingdb_interval_rank_multiseed_v2/w08/"
    "S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_20260816/"
    "BEST_MODEL_BINDINGDB_AFFINITY_V1.pt"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_deployment_augmented_full_fit_v1"
AFFINITY_WEIGHT = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def classification_metrics(labels: np.ndarray, logits: np.ndarray, temperature: float = 1.0) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    logits = np.asarray(logits, dtype=np.float64)
    probability = sigmoid(logits / max(float(temperature), 1e-6))
    return {
        "rows": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auprc": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
    }


def bdb_fused_metrics(frame: pd.DataFrame, prediction: dict[str, np.ndarray]) -> tuple[dict[str, float | int], np.ndarray]:
    work = frame[["target_sequence_hash"]].copy()
    work["binary"] = prediction["final_logit"]
    work["affinity"] = prediction["affinity"]
    binary_rank = work.groupby("target_sequence_hash", sort=False)["binary"].rank(method="average", pct=True)
    affinity_rank = work.groupby("target_sequence_hash", sort=False)["affinity"].rank(method="average", pct=True)
    fused = (1.0 - AFFINITY_WEIGHT) * binary_rank.to_numpy() + AFFINITY_WEIGHT * affinity_rank.to_numpy()
    result = affinity_retrieval_metrics(frame, fused)
    return result, fused


def bdb_selection(metrics: dict[str, float | int]) -> float:
    return (
        0.50 * float(metrics["target_macro_strong_weak_auprc"])
        + 0.25 * float(metrics["target_macro_ndcg"])
        + 0.25 * (float(metrics["target_macro_spearman"]) + 1.0) / 2.0
    )


def make_arrays(
    data: pd.DataFrame,
    fit_positions: np.ndarray,
    target_features: np.ndarray,
    target_aux: np.ndarray,
    structure_features: np.ndarray,
    structure_mask: np.ndarray,
    target_token_max_len: int,
) -> dict[str, object]:
    arrays = prepare_arrays(
        data, fit_positions, target_features, target_aux,
        None, None, None, target_token_max_len,
    )
    active = fit_positions[structure_mask[fit_positions] > 0]
    if len(active):
        mean = np.asarray(structure_features[active].mean(axis=0), dtype=np.float32)
        std = np.asarray(structure_features[active].std(axis=0), dtype=np.float32)
        std[std < 1e-6] = 1.0
    else:
        mean = np.zeros(structure_features.shape[1], dtype=np.float32)
        std = np.ones(structure_features.shape[1], dtype=np.float32)
    arrays["structure_mean"] = mean
    arrays["structure_std"] = std
    return arrays


def predict_positions(
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
) -> dict[str, np.ndarray]:
    return predict(
        model, positions, data, drug_features, None, target_features, target_aux,
        None, None, None, None, config.target_token_max_len, None, None,
        structure_features, structure_mask, arrays, device, batch_size,
    )


def build_retrieval_frame(
    recovered: pd.DataFrame,
    role: str,
    target_structure: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    positives = recovered.loc[
        recovered["augmentation_role"].eq(role) & recovered["binary_label"].eq(1)
    ].copy()
    targets = recovered.sort_values("calibration_pair_id").drop_duplicates("sequence_sha256").copy()
    if targets["target_feature_index"].duplicated().any():
        raise RuntimeError("recovered retrieval target index is not sequence-unique")
    target_columns = [
        "target_feature_index", "target_chembl_id", "target_assay_family",
        "primary_gene", "query_accession", "sequence_sha256",
    ]
    chunks = []
    for query_number, positive in enumerate(positives.itertuples(index=False)):
        chunk = targets[target_columns].copy()
        chunk["query_id"] = str(positive.calibration_pair_id)
        chunk["query_number"] = query_number
        chunk["parent_standard_inchi_key"] = str(positive.parent_standard_inchi_key)
        chunk["drug_feature_index"] = int(positive.drug_feature_index)
        chunk["known_target_feature_index"] = int(positive.target_feature_index)
        chunk["binary_label"] = chunk["target_feature_index"].eq(int(positive.target_feature_index)).astype(np.int8)
        chunk["calibration_pair_id"] = (
            "RETRIEVAL::" + str(positive.calibration_pair_id) + "::"
            + chunk["target_feature_index"].astype(str)
        )
        chunk["conplex_score"] = 0.0
        chunk["mean_pchembl"] = np.nan
        chunk["min_pchembl"] = np.nan
        chunk["max_pchembl"] = np.nan
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    structure_columns = [
        column for column in target_structure.columns
        if column not in {"target_feature_index", "target_chembl_id", "sequence_sha256", "structure_mask", "structure_route"}
    ]
    indexed = target_structure.set_index("target_feature_index")
    target_indices = frame["target_feature_index"].to_numpy(dtype=np.int64)
    values = indexed.loc[target_indices, structure_columns].to_numpy(dtype=np.float32)
    mask = indexed.loc[target_indices, "structure_mask"].to_numpy(dtype=np.float32)
    return frame, values, mask


def retrieval_metrics(
    frame: pd.DataFrame,
    prediction: dict[str, np.ndarray],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    work = frame[[
        "query_id", "query_number", "parent_standard_inchi_key",
        "target_feature_index", "known_target_feature_index", "binary_label",
    ]].copy()
    work["binary_score"] = prediction["final_logit"]
    work["affinity_score"] = prediction["affinity"]
    work["binary_rank_unit"] = work.groupby("query_id", sort=False)["binary_score"].rank(method="average", pct=True)
    work["affinity_rank_unit"] = work.groupby("query_id", sort=False)["affinity_score"].rank(method="average", pct=True)
    work["fusion_score"] = (
        (1.0 - AFFINITY_WEIGHT) * work["binary_rank_unit"]
        + AFFINITY_WEIGHT * work["affinity_rank_unit"]
    )
    work["rank"] = work.groupby("query_id", sort=False)["fusion_score"].rank(method="min", ascending=False).astype(int)
    known = work.loc[work["binary_label"].eq(1)].copy()
    if known["query_id"].duplicated().any() or len(known) != work["query_id"].nunique():
        raise RuntimeError("retrieval positive alignment failed")
    universe = int(work.groupby("query_id").size().iloc[0])
    known["top_percentile"] = 1.0 - (known["rank"] - 1) / max(universe - 1, 1)
    metrics = {
        "queries": int(len(known)),
        "candidate_targets": universe,
        "mean_rank": float(known["rank"].mean()),
        "median_rank": float(known["rank"].median()),
        "mrr": float((1.0 / known["rank"]).mean()),
        "hit_at_1": float(known["rank"].le(1).mean()),
        "hit_at_5": float(known["rank"].le(5).mean()),
        "hit_at_10": float(known["rank"].le(10).mean()),
        "mean_top_percentile": float(known["top_percentile"].mean()),
    }
    return metrics, known


def train_epoch(
    model: RoutedInteractionRankerV2,
    optimizer: torch.optim.Optimizer,
    positions: np.ndarray,
    epoch: int,
    seed: int,
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
    max_rows_per_target: int,
    gradient_clip: float,
    drug_feature_cache: torch.Tensor | None = None,
    target_feature_cache: torch.Tensor | None = None,
    target_aux_feature_cache: torch.Tensor | None = None,
) -> dict[str, float | int]:
    model.train()
    batches = grouped_training_batches(
        positions, data, batch_size, seed + epoch, max_rows_per_target
    )
    emitted = sum(len(batch) for batch in batches)
    if emitted != len(positions):
        raise RuntimeError(f"epoch sampler emitted {emitted} of {len(positions)} rows")
    component: dict[str, float] = {}
    for batch in batches:
        values = tensor_batch(
            batch, data, drug_features, None, target_features, target_aux,
            None, None, None, None, config.target_token_max_len, None, None,
            structure_features, structure_mask, arrays, device,
            drug_feature_cache=drug_feature_cache,
            target_feature_cache=target_feature_cache,
            target_aux_feature_cache=target_aux_feature_cache,
        )
        values["binary_observed"] = torch.from_numpy(
            data.iloc[batch]["binary_observed"].to_numpy(dtype=np.float32).copy()
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            values["drug"], values["target"], values["family"], values["conplex"],
            values["structure"], values["structure_mask"], target_aux=values["target_aux"],
        )
        losses = odti_v2_loss(
            output, values["labels"], values["target_group"], values["drug_group"],
            affinity_lower=values["affinity_lower"],
            affinity_upper=values["affinity_upper"],
            affinity_observed=values["affinity_observed"],
            binary_observed=values["binary_observed"], config=config,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        for name, value in losses.items():
            component[name] = component.get(name, 0.0) + float(value.detach())
    return {
        "batches": len(batches),
        "rows_emitted": emitted,
        **{f"loss_{name}": value / max(len(batches), 1) for name, value in component.items()},
    }


def checkpoint_payload(
    state: dict[str, torch.Tensor],
    config: ODTIV2Config,
    arrays: dict[str, object],
    temperature: float,
    contract: dict[str, object],
) -> dict[str, object]:
    return {
        "model_state_dict": state,
        "model_class": "RoutedInteractionRankerV2",
        "config": config.__dict__,
        "families": arrays["families"],
        "normalization": {
            "target_mean": arrays["target_mean"],
            "target_std": arrays["target_std"],
            "target_aux_mean": arrays["target_aux_mean"],
            "target_aux_std": arrays["target_aux_std"],
            "target_token_mean": arrays["target_token_mean"],
            "target_token_std": arrays["target_token_std"],
            "conplex_mean": arrays["conplex_mean"],
            "conplex_std": arrays["conplex_std"],
            "affinity_mean": arrays["affinity_mean"],
            "affinity_std": arrays["affinity_std"],
            "structure_mean": arrays["structure_mean"],
            "structure_std": arrays["structure_std"],
        },
        "temperature": float(temperature),
        "full_fit_contract": contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--max-rows-per-target", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--bindingdb-cutoff-year", type=int, default=2026)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.patience < 1:
        raise ValueError("invalid epoch/patience contract")
    required = [
        RECOVERED, MORGAN, PROTBERT, ESM2, STRUCTURE, TARGET_STRUCTURE,
        AUGMENT_MANIFEST, REFERENCE_CHECKPOINT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    manifest = json.loads(AUGMENT_MANIFEST.read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError("augmentation manifest is not PASS")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    current, _, _, _, _, assembly_counts = build_combined_data(
        "S5_OLD_DRUG_ENTITY_COLD", -1, True,
        bindingdb_cutoff_year=args.bindingdb_cutoff_year,
        bindingdb_holdout_fold=-1,
    )
    current["augmentation_role"] = "current"
    current["augmentation_source"] = np.where(
        current["binary_observed"].astype(bool), "FROZEN_CHEMBL37_86674", "BINDINGDB_DIRECT_KI_KD"
    )
    recovered = pd.read_csv(RECOVERED, low_memory=False)
    recovered["binary_observed"] = 1
    recovered["affinity_observed"] = 1
    recovered["target_sequence_hash"] = recovered["sequence_sha256"].astype(str)
    recovered["external_scaffold_group"] = recovered["scaffold_group"].fillna("").astype(str)
    recovered["external_target_hash"] = recovered["sequence_sha256"].astype(str)
    recovered["supplement_train_eligible"] = False
    recovered["bindingdb_target_cold_valid"] = False
    recovered["bindingdb_publication_year"] = np.nan
    recovered["date_min"] = ""
    recovered["date_max"] = ""
    data = pd.concat([current, recovered], ignore_index=True, sort=False)
    n_current = len(current)
    recovered_positions = np.arange(n_current, len(data), dtype=np.int64)
    rec_role = data.iloc[recovered_positions]["augmentation_role"].astype(str).to_numpy()
    rec_train = recovered_positions[rec_role == "train"]
    rec_dev = recovered_positions[rec_role == "dev"]
    rec_test = recovered_positions[rec_role == "test"]

    feature_available = data["drug_feature_available"].astype(bool).to_numpy()
    current_positions = np.arange(n_current, dtype=np.int64)
    current_feature_positions = current_positions[feature_available[:n_current]]
    external_mask = ~current["binary_observed"].astype(bool).to_numpy()
    external = current.loc[external_mask].copy()
    recovered_hashes = set(recovered["sequence_sha256"].astype(str))
    stats = external.groupby("target_sequence_hash").agg(
        rows=("mean_pchembl", "size"), minimum=("mean_pchembl", "min"),
        maximum=("mean_pchembl", "max"), target_feature_index=("target_feature_index", "first"),
    )
    eligible = stats.loc[
        stats["target_feature_index"].astype(int).ge(428)
        & stats["rows"].ge(5)
        & stats["minimum"].le(5.0)
        & stats["maximum"].ge(6.0)
        & ~stats.index.astype(str).isin(recovered_hashes)
    ].copy()
    eligible["fold"] = [int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5 for value in eligible.index]
    bdb_dev_hashes = set(eligible.index[eligible["fold"].eq(0)].astype(str))
    bdb_test_hashes = set(eligible.index[eligible["fold"].eq(1)].astype(str))
    bdb_dev_mask = current["target_sequence_hash"].astype(str).isin(bdb_dev_hashes).to_numpy() & external_mask
    bdb_test_mask = current["target_sequence_hash"].astype(str).isin(bdb_test_hashes).to_numpy() & external_mask
    bdb_dev = np.flatnonzero(bdb_dev_mask).astype(np.int64)
    bdb_test = np.flatnonzero(bdb_test_mask).astype(np.int64)
    held_bdb_mask = bdb_dev_mask | bdb_test_mask
    eval_fit_current = current_feature_positions[~held_bdb_mask[current_feature_positions]]
    eval_fit = np.concatenate([eval_fit_current, rec_train]).astype(np.int64)
    full_fit = np.concatenate([current_feature_positions, recovered_positions]).astype(np.int64)

    fit_drugs = set(data.iloc[eval_fit]["parent_standard_inchi_key"].astype(str))
    fit_targets = set(data.iloc[eval_fit]["target_feature_index"].astype(int))
    if set(data.iloc[np.concatenate([rec_dev, rec_test])]["parent_standard_inchi_key"].astype(str)) - fit_drugs:
        raise RuntimeError("recovered held-out drug is not warm in evaluation fit")
    if set(data.iloc[np.concatenate([rec_dev, rec_test])]["target_feature_index"].astype(int)) - fit_targets:
        raise RuntimeError("recovered held-out target is not warm in evaluation fit")
    fit_target_hashes = set(data.iloc[eval_fit]["target_sequence_hash"].astype(str))
    if (bdb_dev_hashes | bdb_test_hashes) & fit_target_hashes:
        raise RuntimeError("BindingDB exact target leaked into evaluation fitting positions")

    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux = np.load(ESM2, mmap_mode="r")
    if data["drug_feature_index"].max() >= len(drug_features) or data["target_feature_index"].max() >= len(target_features):
        raise RuntimeError("augmented pair feature index exceeds feature matrices")
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ODTIV2Config(**reference["config"])
    if config.structure_input_dim != 19 or config.interaction_mode != "low_rank_film":
        raise RuntimeError("reference checkpoint architecture contract failed")
    structure_features, structure_mask, structure_columns, structure_source = load_structure_features(
        str(STRUCTURE), data["calibration_pair_id"], config.structure_input_dim
    )
    target_structure = pd.read_csv(TARGET_STRUCTURE, low_memory=False)
    dev_retrieval_frame, dev_retrieval_structure, dev_retrieval_mask = build_retrieval_frame(
        recovered, "dev", target_structure
    )
    test_retrieval_frame, test_retrieval_structure, test_retrieval_mask = build_retrieval_frame(
        recovered, "test", target_structure
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Stage 1: validation-selected evaluation fit.
    eval_arrays = make_arrays(
        data, eval_fit, target_features, target_aux, structure_features,
        structure_mask, config.target_token_max_len,
    )
    model = RoutedInteractionRankerV2(len(eval_arrays["families"]), config, use_conplex=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.learning_rate / 20
    )
    history: list[dict[str, object]] = []
    best_value = -np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    stop_reason = "MAX_EPOCHS"
    for epoch in range(1, args.max_epochs + 1):
        train_row = train_epoch(
            model, optimizer, eval_fit, epoch, args.seed, data, drug_features,
            target_features, target_aux, structure_features, structure_mask,
            eval_arrays, config, device, args.batch_size, args.max_rows_per_target,
            args.gradient_clip,
        )
        scheduler.step()
        model.eval()
        dev_prediction = predict_positions(
            model, rec_dev, data, drug_features, target_features, target_aux,
            structure_features, structure_mask, eval_arrays, config, device,
            args.inference_batch_size,
        )
        dev_class = classification_metrics(
            data.iloc[rec_dev]["binary_label"].to_numpy(), dev_prediction["final_logit"]
        )
        bdb_dev_prediction = predict_positions(
            model, bdb_dev, data, drug_features, target_features, target_aux,
            structure_features, structure_mask, eval_arrays, config, device,
            args.inference_batch_size,
        )
        bdb_dev_metrics, _ = bdb_fused_metrics(data.iloc[bdb_dev], bdb_dev_prediction)
        retrieval_dev_prediction = predict(
            model, np.arange(len(dev_retrieval_frame), dtype=np.int64), dev_retrieval_frame,
            drug_features, None, target_features, target_aux, None, None, None, None,
            config.target_token_max_len, None, None, dev_retrieval_structure,
            dev_retrieval_mask, eval_arrays, device, args.inference_batch_size,
        )
        retrieval_dev_metrics, _ = retrieval_metrics(dev_retrieval_frame, retrieval_dev_prediction)
        selection = (
            0.35 * float(retrieval_dev_metrics["mean_top_percentile"])
            + 0.20 * float(retrieval_dev_metrics["hit_at_10"])
            + 0.25 * float(dev_class["auprc"])
            + 0.20 * bdb_selection(bdb_dev_metrics)
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selection_value": selection,
            **train_row,
            **{f"dev_relation_{key}": value for key, value in dev_class.items()},
            **{f"dev_retrieval_{key}": value for key, value in retrieval_dev_metrics.items()},
            **{f"dev_bdb_{key}": value for key, value in bdb_dev_metrics.items()},
        }
        history.append(row)
        print(json.dumps({"seed": args.seed, **row}, default=json_safe), flush=True)
        if selection > best_value + args.min_delta:
            best_value = selection
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch >= args.min_epochs and no_improvement >= args.patience:
            stop_reason = "DEVELOPMENT_EARLY_STOPPING"
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("development selection failed")
    model.load_state_dict(best_state)
    model.to(device).eval()

    # One-time untouched evaluation.
    dev_best_prediction = predict_positions(
        model, rec_dev, data, drug_features, target_features, target_aux,
        structure_features, structure_mask, eval_arrays, config, device,
        args.inference_batch_size,
    )
    temperature = temperature_scale(
        dev_best_prediction["final_logit"],
        data.iloc[rec_dev]["binary_label"].to_numpy(dtype=np.int8),
    )
    test_prediction = predict_positions(
        model, rec_test, data, drug_features, target_features, target_aux,
        structure_features, structure_mask, eval_arrays, config, device,
        args.inference_batch_size,
    )
    test_class = classification_metrics(
        data.iloc[rec_test]["binary_label"].to_numpy(), test_prediction["final_logit"], temperature
    )
    train_rec_prediction = predict_positions(
        model, rec_train, data, drug_features, target_features, target_aux,
        structure_features, structure_mask, eval_arrays, config, device,
        args.inference_batch_size,
    )
    train_rec_class = classification_metrics(
        data.iloc[rec_train]["binary_label"].to_numpy(), train_rec_prediction["final_logit"], temperature
    )
    bdb_test_prediction = predict_positions(
        model, bdb_test, data, drug_features, target_features, target_aux,
        structure_features, structure_mask, eval_arrays, config, device,
        args.inference_batch_size,
    )
    bdb_test_metrics, bdb_test_fused = bdb_fused_metrics(data.iloc[bdb_test], bdb_test_prediction)
    retrieval_test_prediction = predict(
        model, np.arange(len(test_retrieval_frame), dtype=np.int64), test_retrieval_frame,
        drug_features, None, target_features, target_aux, None, None, None, None,
        config.target_token_max_len, None, None, test_retrieval_structure,
        test_retrieval_mask, eval_arrays, device, args.inference_batch_size,
    )
    retrieval_test_metrics, retrieval_test_known = retrieval_metrics(
        test_retrieval_frame, retrieval_test_prediction
    )

    out_root = Path(args.out_dir).resolve()
    run_dir = out_root / f"FULL_FIT_2026__seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_checkpoint = run_dir / "EVALUATION_BEST_MODEL_BINDINGDB_AFFINITY_V1.pt"
    evaluation_contract = {
        "mode": "VALIDATION_SELECTED_UNTOUCHED_TEST_EVALUATION",
        "selected_epoch": best_epoch,
        "maximum_epochs": args.max_epochs,
        "stop_reason": stop_reason,
        "recovered_dev_test_excluded": True,
        "bindingdb_target_cold_dev_test_excluded": True,
    }
    torch.save(
        checkpoint_payload(best_state, config, eval_arrays, temperature, evaluation_contract),
        evaluation_checkpoint,
    )
    pd.DataFrame(history).to_csv(run_dir / "EVALUATION_TRAINING_HISTORY_V1.csv", index=False)
    relation_test = data.iloc[rec_test][[
        "calibration_pair_id", "sequence_key", "target_chembl_id", "primary_gene",
        "parent_standard_inchi_key", "parent_molecule_name", "binary_label",
        "mean_pchembl", "augmentation_role",
    ]].copy()
    relation_test["raw_logit"] = test_prediction["final_logit"]
    relation_test["calibrated_probability"] = sigmoid(test_prediction["final_logit"] / temperature)
    relation_test.to_csv(run_dir / "DOUBLE_WARM_RELATION_TEST_PREDICTIONS_V1.csv", index=False)
    retrieval_test_known.to_csv(run_dir / "DOUBLE_WARM_RETRIEVAL_TEST_POSITIVE_RANKS_V1.csv", index=False)
    bdb_test_frame = data.iloc[bdb_test][[
        "calibration_pair_id", "target_sequence_hash", "parent_standard_inchi_key",
        "mean_pchembl", "min_pchembl", "max_pchembl",
    ]].copy()
    bdb_test_frame["binary_score"] = bdb_test_prediction["final_logit"]
    bdb_test_frame["affinity_score"] = bdb_test_prediction["affinity"]
    bdb_test_frame["fused_rank_score"] = bdb_test_fused
    bdb_test_frame.to_csv(run_dir / "BINDINGDB_TARGET_COLD_TEST_PREDICTIONS_V1.csv.gz", index=False, compression="gzip")

    # Stage 2: fresh deployment refit on every historical relation using the
    # validation-selected epoch count.  This model is never used for the test
    # metrics above.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(args.seed)
    full_arrays = make_arrays(
        data, full_fit, target_features, target_aux, structure_features,
        structure_mask, config.target_token_max_len,
    )
    final_model = RoutedInteractionRankerV2(len(full_arrays["families"]), config, use_conplex=False).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer, T_max=args.max_epochs, eta_min=args.learning_rate / 20
    )
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        row = train_epoch(
            final_model, final_optimizer, full_fit, epoch, args.seed, data,
            drug_features, target_features, target_aux, structure_features,
            structure_mask, full_arrays, config, device, args.batch_size,
            args.max_rows_per_target, args.gradient_clip,
        )
        final_scheduler.step()
        row = {"epoch": epoch, "learning_rate": float(final_scheduler.get_last_lr()[0]), **row}
        refit_history.append(row)
        print(json.dumps({"seed": args.seed, "refit": True, **row}, default=json_safe), flush=True)
    final_state = {name: value.detach().cpu().clone() for name, value in final_model.state_dict().items()}
    final_checkpoint = run_dir / "FULL_FIT_MODEL_BINDINGDB_AFFINITY_V1.pt"
    full_contract = {
        "protocol": "FULL_FIT_2026_DEPLOYMENT_AUGMENTED_V1",
        "training_mode": "FRESH_REFIT_ALL_HISTORICAL_RELATIONS_AT_SELECTED_EPOCH",
        "selected_epoch": best_epoch,
        "scheduler_t_max": args.max_epochs,
        "conplex_enabled": False,
        "structure_input_dim": config.structure_input_dim,
        "evaluation_checkpoint_separate": True,
        "evaluation_test_labels_used_for_epoch_selection": False,
        "external_current_new_relation_labels_used": False,
    }
    torch.save(
        checkpoint_payload(final_state, config, full_arrays, 1.0, full_contract),
        final_checkpoint,
    )
    pd.DataFrame(refit_history).to_csv(run_dir / "FULL_FIT_TRAINING_HISTORY_V1.csv", index=False)

    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "FULL_FIT_2026_DEPLOYMENT_AUGMENTED_V1",
        "seed": args.seed,
        "device": str(device),
        "new_relation_labels_used_for_training_or_selection": False,
        "selection": {
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_value": best_value,
            "stop_reason": stop_reason,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "scheduler_t_max": args.max_epochs,
        },
        "fit_counts": {
            "evaluation_fit_rows": len(eval_fit),
            "evaluation_current_rows": len(eval_fit_current),
            "evaluation_recovered_train_rows": len(rec_train),
            "full_refit_rows": len(full_fit),
            "current_feature_available_rows": len(current_feature_positions),
            "recovered_all_rows": len(recovered_positions),
            "recovered_drugs": recovered["parent_standard_inchi_key"].nunique(),
            "unique_drugs_full_refit": data.iloc[full_fit]["parent_standard_inchi_key"].nunique(),
            "unique_targets_full_refit": data.iloc[full_fit]["target_feature_index"].nunique(),
            "warm_frozen_720_before": manifest["counts"]["frozen_720_warm_before"],
            "warm_frozen_720_after": manifest["counts"]["warm_after_all_recovered_relations"],
        },
        "split_counts": {
            "double_warm_dev_rows": len(rec_dev),
            "double_warm_test_rows": len(rec_test),
            "bindingdb_target_cold_dev_rows": len(bdb_dev),
            "bindingdb_target_cold_dev_targets": len(bdb_dev_hashes),
            "bindingdb_target_cold_test_rows": len(bdb_test),
            "bindingdb_target_cold_test_targets": len(bdb_test_hashes),
        },
        "test_metrics": {
            "double_warm_relation": test_class,
            "double_warm_positive_retrieval": retrieval_test_metrics,
            "bindingdb_exact_target_cold": bdb_test_metrics,
            "recovered_train_relation_for_overfit_comparison": train_rec_class,
            "train_test_auprc_gap": float(train_rec_class["auprc"] - test_class["auprc"]),
            "train_test_auroc_gap": float(train_rec_class["auroc"] - test_class["auroc"]),
        },
        "model": {
            "parameter_count": int(sum(value.numel() for value in final_model.parameters())),
            "config": config.__dict__,
            "conplex_enabled": False,
            "structure_columns": structure_columns,
            "structure_source": structure_source,
        },
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [RECOVERED, MORGAN, PROTBERT, ESM2, STRUCTURE, TARGET_STRUCTURE, AUGMENT_MANIFEST, REFERENCE_CHECKPOINT]
        },
        "artifacts": {
            "evaluation_checkpoint": str(evaluation_checkpoint.relative_to(ROOT)),
            "evaluation_checkpoint_sha256": sha256(evaluation_checkpoint),
            "evaluation_state_sha256": state_sha256(best_state),
            "checkpoint": str(final_checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(final_checkpoint),
            "state_sha256": state_sha256(final_state),
        },
        "claim_boundary": (
            "Test metrics come only from the validation-selected evaluation checkpoint. "
            "The deployment FULL_FIT checkpoint is a fresh post-test refit on all historical relations."
        ),
        "assembly_counts": assembly_counts,
    }
    summary_path = run_dir / "FULL_FIT_RUN_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe))


if __name__ == "__main__":
    main()

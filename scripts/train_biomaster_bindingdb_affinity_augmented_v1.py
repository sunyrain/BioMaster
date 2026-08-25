#!/usr/bin/env python3
"""Paired exploratory training with the 9,778-row BindingDB Ki/Kd batch.

The frozen ChEMBL37 rows retain all binary/ranking/contrastive supervision.
BindingDB rows are appended only to the affinity task, with
``binary_observed=0``.  For S3, only supplemental rows whose exact target is
already in the fold's internal training entity set and whose ligand scaffold
is outside validation/test are admitted.  For S5, deployment-old-drug
scaffolds are excluded.  Validation and test rows are always the frozen
ChEMBL37 roles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, ndcg_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2, odti_v2_loss  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics, split_masks  # noqa: E402
from train_biomaster_odti_v2 import (  # noqa: E402
    fast_validation_metrics,
    build_target_token_cache,
    grouped_training_batches,
    dual_query_training_batches,
    load_target_token_features,
    load_structure_features,
    predict,
    prepare_arrays,
    set_seed,
    sigmoid,
    temperature_scale,
    tensor_batch,
    validation_selection_value,
)

BASE = ROOT / "outputs/old_drug_target_sota_v1"
STORE = BASE / "feature_store_v1"
PACKAGE = ROOT / "outputs/biomaster_bindingdb_affinity_feature_package_v1"
PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = PACKAGE / "MORGAN2048_UINT8_COMBINED_V1.npy"
PROTBERT = PACKAGE / "PROTBERT1024_FLOAT32_COMBINED_V1.npy"
ESM2 = PACKAGE / "ESM2_650M_1280_FLOAT32_COMBINED_V1.npy"
TOKEN_PACKAGE = ROOT / "outputs/biomaster_bindingdb_target_token_feature_package_v1"
TOKEN_FEATURES = TOKEN_PACKAGE / "ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy"
TOKEN_INDEX = TOKEN_PACKAGE / "ESM2_650M_RESIDUE_INDEX_COMBINED_V1.csv.gz"
TOKEN_MASK = TOKEN_PACKAGE / "ESM2_650M_POCKET_MASK_UINT8_COMBINED_V1.npy"
SUPPLEMENT = PACKAGE / "BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz"
SUPPLEMENT_METADATA = ROOT / (
    "outputs/biomaster_bindingdb_full_training_subset_v1/"
    "BINDINGDB_HIGH_CONFIDENCE_DIRECT_KI_KD_PAIRS_V1.csv.gz"
)
STRUCTURE = STORE / "ODTI_STRUCTURE_CONTEXT_V1.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def murcko(smiles: object) -> str:
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) if molecule is not None else ""
    except Exception:
        return ""


def build_combined_data(
    protocol: str,
    fold: int,
    include_supplement: bool,
    bindingdb_cutoff_year: int = 2022,
    bindingdb_holdout_fold: int = -1,
    bindingdb_holdout_fold_count: int = 5,
    bindingdb_min_validation_pairs: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    internal = pd.read_csv(PAIRS, low_memory=False)
    internal["binary_observed"] = 1
    internal["affinity_observed"] = 1
    internal["target_sequence_hash"] = ""
    internal["external_scaffold_group"] = internal["murcko_scaffold"].fillna("").astype(str)
    internal["external_target_hash"] = ""
    internal_masks = split_masks(internal, protocol, fold)
    available = internal["drug_feature_available"].to_numpy(dtype=bool)
    train_internal = np.flatnonzero(internal_masks["train"] & available)
    valid_internal = np.flatnonzero(internal_masks["valid"] & available)
    test_internal = np.flatnonzero(internal_masks["test"] & available)

    if not include_supplement:
        positions = np.arange(len(internal), dtype=np.int64)
        return internal, positions[internal_masks["train"] & available], positions[internal_masks["valid"] & available], positions[internal_masks["test"] & available], np.zeros(0, dtype=np.int64), {
            "supplement_rows_total": 0,
            "supplement_rows_train": 0,
            "supplement_rows_rejected": 0,
            "train_internal_rows": int(len(train_internal)),
        }

    external = pd.read_csv(SUPPLEMENT, low_memory=False)
    metadata = pd.read_csv(
        SUPPLEMENT_METADATA,
        usecols=["target_sequence_hash", "ligand_inchikey", "date_min", "date_max"],
        low_memory=False,
    ).rename(columns={"ligand_inchikey": "parent_standard_inchi_key"})
    if metadata.duplicated(["target_sequence_hash", "parent_standard_inchi_key"]).any():
        raise RuntimeError("BindingDB pair metadata key is not unique")
    external = external.merge(
        metadata,
        on=["target_sequence_hash", "parent_standard_inchi_key"],
        how="left",
        validate="one_to_one",
    )
    if external["date_max"].isna().any():
        raise RuntimeError("BindingDB feature rows are missing publication metadata")
    publication_year = pd.to_datetime(external["date_max"], errors="coerce").dt.year
    if publication_year.isna().any():
        raise RuntimeError("BindingDB publication date could not be parsed")
    external["bindingdb_publication_year"] = publication_year.astype(int)
    rows_before_cutoff = len(external)
    external = external.loc[
        external["bindingdb_publication_year"].le(int(bindingdb_cutoff_year))
    ].copy()
    external["murcko_scaffold"] = external["model_ligand_smiles"].map(murcko)
    external["external_scaffold_group"] = external["murcko_scaffold"]
    # Build strict entity-compatible supplemental eligibility.  Exact target
    # compatibility is intentionally used instead of a loose sequence-family
    # guess for S3, avoiding hidden target-homology leakage.
    train_target_indices = set(internal.iloc[train_internal]["target_feature_index"].astype(int))
    valid_test_scaffolds = set(
        internal.iloc[np.concatenate([valid_internal, test_internal])]["murcko_scaffold"].fillna("").astype(str)
    )
    deployment_scaffolds = set(
        internal.loc[internal["has_deployment_old_drug_scaffold"].astype(bool), "murcko_scaffold"].fillna("").astype(str)
    )
    if protocol == "S3_STRICT_DOUBLE_COLD":
        eligible = external["target_feature_index"].astype(int).isin(train_target_indices)
        eligible &= ~external["murcko_scaffold"].isin(valid_test_scaffolds)
    elif protocol == "S5_OLD_DRUG_ENTITY_COLD":
        eligible = ~external["murcko_scaffold"].isin(deployment_scaffolds)
    else:
        eligible = np.ones(len(external), dtype=bool)
    external["supplement_train_eligible"] = np.asarray(eligible, dtype=bool)
    external["bindingdb_target_cold_valid"] = False
    heldout_target_hashes: set[str] = set()
    if bindingdb_holdout_fold >= 0:
        if bindingdb_holdout_fold_count < 2:
            raise ValueError("bindingdb_holdout_fold_count must be at least two")
        if bindingdb_holdout_fold >= bindingdb_holdout_fold_count:
            raise ValueError("bindingdb_holdout_fold is outside the configured folds")
        # Appended targets have no ChEMBL37 training row.  Holding out all
        # rows for selected exact sequence hashes therefore gives a genuine
        # target-cold affinity validation, while the S5 scaffold exclusion
        # keeps every deployment old drug cold as well.
        external_target_cold = external["target_feature_index"].astype(int).ge(428)
        target_stats = external.loc[external_target_cold].groupby("target_sequence_hash").agg(
            rows=("mean_pchembl", "size"),
            minimum=("mean_pchembl", "min"),
            maximum=("mean_pchembl", "max"),
        )
        eligible_validation_targets = set(
            target_stats.index[
                target_stats["rows"].ge(int(bindingdb_min_validation_pairs))
                & target_stats["minimum"].le(5.0)
                & target_stats["maximum"].ge(6.0)
            ].astype(str)
        )
        heldout_target_hashes = {
            target_hash
            for target_hash in eligible_validation_targets
            if int(hashlib.sha256(target_hash.encode("utf-8")).hexdigest()[:8], 16)
            % int(bindingdb_holdout_fold_count)
            == int(bindingdb_holdout_fold)
        }
        heldout = external["target_sequence_hash"].astype(str).isin(heldout_target_hashes)
        external.loc[heldout & external["supplement_train_eligible"], "bindingdb_target_cold_valid"] = True
        # Exclude every row of a held-out exact target, including rows made
        # ineligible for evaluation by the deployment-scaffold guard.
        external.loc[heldout, "supplement_train_eligible"] = False
    external["binary_observed"] = 0
    external["affinity_observed"] = 1
    # Internal metric/query columns are retained for output compatibility;
    # supplemental rows never enter valid/test.
    for column in internal.columns:
        if column not in external.columns:
            if column in {"binary_label", "drug_feature_available"}:
                external[column] = 0
            elif column in {"mean_pchembl", "min_pchembl", "max_pchembl", "conplex_score"}:
                external[column] = 0.0
            else:
                external[column] = ""
    external = external[
        internal.columns.tolist()
        + [
            "supplement_train_eligible",
            "bindingdb_target_cold_valid",
            "bindingdb_publication_year",
            "date_min",
            "date_max",
        ]
    ]
    internal["supplement_train_eligible"] = False
    internal["bindingdb_target_cold_valid"] = False
    internal["bindingdb_publication_year"] = np.nan
    internal["date_min"] = ""
    internal["date_max"] = ""
    combined = pd.concat([internal, external], ignore_index=True, sort=False)
    n_internal = len(internal)
    ext_pos = np.arange(n_internal, len(combined), dtype=np.int64)
    supplement_train = ext_pos[combined.iloc[ext_pos]["supplement_train_eligible"].to_numpy(dtype=bool)]
    supplement_valid = ext_pos[combined.iloc[ext_pos]["bindingdb_target_cold_valid"].to_numpy(dtype=bool)]
    train_positions = np.concatenate([train_internal, supplement_train]).astype(np.int64)
    valid_positions = valid_internal.astype(np.int64)
    test_positions = test_internal.astype(np.int64)
    counts = {
        "supplement_rows_total": int(len(external)),
        "supplement_rows_train": int(len(supplement_train)),
        "supplement_rows_rejected": int(len(external) - len(supplement_train)),
        "supplement_rows_before_publication_cutoff": int(rows_before_cutoff),
        "supplement_rows_after_publication_cutoff": int(len(external)),
        "supplement_publication_cutoff_year": int(bindingdb_cutoff_year),
        "supplement_target_cold_valid_rows": int(len(supplement_valid)),
        "supplement_target_cold_valid_targets": int(len(heldout_target_hashes)),
        "train_internal_rows": int(len(train_internal)),
    }
    return combined, train_positions, valid_positions, test_positions, supplement_valid, counts


def affinity_retrieval_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float | int]:
    """Continuous and strong-vs-weak target-wise BindingDB validation."""

    work = frame[["target_sequence_hash", "mean_pchembl"]].copy()
    work["score"] = np.asarray(score, dtype=np.float64)
    spearman: list[float] = []
    ndcg: list[float] = []
    auprc: list[float] = []
    for _, part in work.groupby("target_sequence_hash", sort=False):
        if len(part) < 3 or part["mean_pchembl"].nunique() < 2:
            continue
        correlation = part[["mean_pchembl", "score"]].corr(method="spearman").iloc[0, 1]
        if np.isfinite(correlation):
            spearman.append(float(correlation))
        relevance = np.maximum(part["mean_pchembl"].to_numpy(dtype=np.float64) - 4.0, 0.0)
        ndcg.append(float(ndcg_score(relevance.reshape(1, -1), part["score"].to_numpy().reshape(1, -1))))
        binary = np.where(part["mean_pchembl"].ge(6.0), 1, np.where(part["mean_pchembl"].le(5.0), 0, -1))
        keep = binary >= 0
        if keep.sum() >= 2 and np.unique(binary[keep]).size == 2:
            auprc.append(float(average_precision_score(binary[keep], part["score"].to_numpy()[keep])))
    return {
        "rows": int(len(work)),
        "targets": int(work["target_sequence_hash"].nunique()),
        "targets_with_continuous_metric": int(len(spearman)),
        "target_macro_spearman": float(np.mean(spearman)) if spearman else 0.0,
        "target_macro_ndcg": float(np.mean(ndcg)) if ndcg else 0.0,
        "targets_with_strong_weak_metric": int(len(auprc)),
        "target_macro_strong_weak_auprc": float(np.mean(auprc)) if auprc else 0.0,
    }


def train_one(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    if not all(path.is_file() for path in [PAIRS, MORGAN, PROTBERT, ESM2, SUPPLEMENT]):
        raise FileNotFoundError("BindingDB feature package is incomplete; build it first")
    fold = -1 if args.protocol == "S5_OLD_DRUG_ENTITY_COLD" else args.fold
    data, train_positions, valid_positions, test_positions, bindingdb_valid_positions, supplement_counts = build_combined_data(
        args.protocol, fold, not args.no_supplement,
        args.bindingdb_cutoff_year,
        args.bindingdb_holdout_fold,
        args.bindingdb_holdout_fold_count,
        args.bindingdb_min_validation_pairs,
    )
    if any(len(value) == 0 for value in [train_positions, valid_positions, test_positions]):
        raise RuntimeError(f"empty split: train={len(train_positions)} valid={len(valid_positions)} test={len(test_positions)}")
    if data.iloc[valid_positions]["binary_label"].nunique() < 2 or data.iloc[test_positions]["binary_label"].nunique() < 2:
        raise RuntimeError("frozen validation/test roles must contain both binary classes")
    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux_features = np.load(ESM2, mmap_mode="r")
    target_token_features = None
    target_token_offsets = None
    target_token_lengths = None
    target_token_pocket_mask_features = None
    target_token_dim = None
    target_token_max_len = int(args.target_token_max_len)
    target_token_source = None
    target_token_index_source = None
    target_token_pocket_mask_source = None
    if args.use_target_tokens:
        token_paths = [TOKEN_FEATURES, TOKEN_INDEX, TOKEN_MASK]
        if not all(path.is_file() for path in token_paths):
            raise FileNotFoundError(
                "target token package is incomplete; build it before --use-target-tokens"
            )
        (
            target_token_features,
            target_token_offsets,
            target_token_lengths,
            target_token_pocket_mask_features,
            target_token_source,
            target_token_index_source,
            target_token_pocket_mask_source,
        ) = load_target_token_features(
            str(TOKEN_FEATURES),
            str(TOKEN_INDEX),
            len(target_features),
            1280,
            str(TOKEN_MASK),
        )
        target_token_dim = int(target_token_features.shape[1])
    structure_features, structure_mask, structure_columns, structure_path = load_structure_features(
        str(STRUCTURE), data["calibration_pair_id"], 19
    )
    structure_train_rows = train_positions[structure_mask[train_positions] > 0]
    if len(structure_train_rows):
        structure_mean = np.asarray(structure_features[structure_train_rows].mean(axis=0), dtype=np.float32)
        structure_std = np.asarray(structure_features[structure_train_rows].std(axis=0), dtype=np.float32)
        structure_std[structure_std < 1e-6] = 1.0
    else:
        structure_mean = np.zeros(19, dtype=np.float32)
        structure_std = np.ones(19, dtype=np.float32)
    arrays = prepare_arrays(
        data,
        train_positions,
        target_features,
        target_aux_features,
        target_token_features,
        target_token_offsets,
        target_token_lengths,
        target_token_max_len,
    )
    arrays["structure_mean"] = structure_mean
    arrays["structure_std"] = structure_std
    config = ODTIV2Config(
        target_aux_input_dim=1280,
        target_token_input_dim=(0 if target_token_features is None else int(target_token_dim)),
        target_token_heads=int(args.target_token_heads),
        target_token_max_len=target_token_max_len,
        target_token_gate_init_bias=(
            float(args.target_token_gate_init_bias)
            if target_token_features is not None
            else -4.0
        ),
        structure_input_dim=19,
        interaction_mode=args.interaction_mode,
        interaction_rank=args.interaction_rank,
        film_scale=args.film_scale,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        expert_count=args.experts,
        dropout=args.dropout,
        rank_weight=args.rank_weight,
        affinity_weight=args.affinity_weight,
        affinity_rank_weight=args.affinity_rank_weight,
        affinity_drug_rank_weight=args.affinity_drug_rank_weight,
        affinity_rank_min_delta=args.affinity_rank_min_delta,
        affinity_rank_margin=args.affinity_rank_margin,
        affinity_rank_max_pairs=args.affinity_rank_max_pairs,
        observation_weight=0.0,
        contrastive_weight=args.contrastive_weight,
        rank_max_pairs=args.rank_max_pairs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = RoutedInteractionRankerV2(family_count=len(arrays["families"]), config=config, use_conplex=False).to(device)
    target_token_cache = None
    target_token_mask_cache = None
    if target_token_features is not None and args.cache_target_tokens:
        cache_np, cache_mask_np = build_target_token_cache(
            target_token_features,
            target_token_offsets,
            target_token_lengths,
            np.asarray(arrays["target_token_mean"], dtype=np.float32),
            np.asarray(arrays["target_token_std"], dtype=np.float32),
            target_token_max_len,
            target_token_pocket_mask_features,
        )
        target_token_cache = torch.from_numpy(cache_np).to(device)
        target_token_mask_cache = torch.from_numpy(cache_mask_np).to(device)
        del cache_np, cache_mask_np
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 20)
    best_selection, best_epoch, best_state, no_improvement = -1.0, -1, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.batch_sampler == "dual_query":
            batches = dual_query_training_batches(
                train_positions, data, args.batch_size, args.seed + epoch,
                args.max_rows_per_target, args.max_rows_per_drug,
            )
        else:
            batches = grouped_training_batches(
                train_positions, data, args.batch_size, args.seed + epoch,
                args.max_rows_per_target,
            )
        component_sums: dict[str, float] = {}
        loss_sum = 0.0
        for positions in batches:
            values = tensor_batch(
                positions, data, drug_features, None, target_features, target_aux_features,
                target_token_features, target_token_offsets, target_token_lengths,
                target_token_pocket_mask_features, target_token_max_len,
                target_token_cache, target_token_mask_cache, structure_features, structure_mask,
                arrays, device,
            )
            values["binary_observed"] = torch.from_numpy(
                data.iloc[positions]["binary_observed"].to_numpy(dtype=np.float32).copy()
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                values["drug"], values["target"], values["family"], values["conplex"],
                values["structure"], values["structure_mask"], target_aux=values["target_aux"],
                target_tokens=values["target_tokens"], target_token_mask=values["target_token_mask"],
            )
            losses = odti_v2_loss(
                outputs, values["labels"], values["target_group"], values["drug_group"],
                affinity_lower=values["affinity_lower"], affinity_upper=values["affinity_upper"],
                affinity_observed=values["affinity_observed"], binary_observed=values["binary_observed"],
                config=config,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_sum += float(losses["total"].detach())
            for name, value in losses.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
        scheduler.step()
        validation = predict(
            model, valid_positions, data, drug_features, None, target_features, target_aux_features,
            target_token_features, target_token_offsets, target_token_lengths,
            target_token_pocket_mask_features, target_token_max_len,
            target_token_cache, target_token_mask_cache, structure_features, structure_mask, arrays,
            device, args.inference_batch_size,
        )
        valid_frame = data.iloc[valid_positions].reset_index(drop=True)
        validation_metrics = fast_validation_metrics(valid_frame, sigmoid(validation["final_logit"]))
        bindingdb_validation_metrics: dict[str, float | int] = {}
        if len(bindingdb_valid_positions):
            bindingdb_prediction = predict(
                model, bindingdb_valid_positions, data, drug_features, None,
                target_features, target_aux_features, target_token_features,
                target_token_offsets, target_token_lengths,
                target_token_pocket_mask_features, target_token_max_len,
                target_token_cache, target_token_mask_cache, structure_features,
                structure_mask, arrays, device, args.inference_batch_size,
            )
            bindingdb_validation_metrics = affinity_retrieval_metrics(
                data.iloc[bindingdb_valid_positions], bindingdb_prediction["affinity"]
            )
        if args.selection_metric == "composite_with_bindingdb_target_cold":
            if not bindingdb_validation_metrics:
                raise RuntimeError("target-cold BindingDB validation is required for this selection metric")
            binary_selection = validation_selection_value(validation_metrics, "composite")
            affinity_selection = 0.5 * float(bindingdb_validation_metrics["target_macro_strong_weak_auprc"]) + 0.25 * float(bindingdb_validation_metrics["target_macro_ndcg"]) + 0.25 * (float(bindingdb_validation_metrics["target_macro_spearman"]) + 1.0) / 2.0
            selection = 0.65 * binary_selection + 0.35 * affinity_selection
        else:
            selection = validation_selection_value(validation_metrics, args.selection_metric)
        history.append({"epoch": epoch, "loss": loss_sum / max(len(batches), 1), **{f"loss_{k}": v / max(len(batches), 1) for k, v in component_sums.items()}, "valid_selection_value": selection, **{f"valid_{k}": v for k, v in validation_metrics.items()}, **{f"valid_bindingdb_{k}": v for k, v in bindingdb_validation_metrics.items()}})
        if selection > best_selection + args.min_delta:
            best_selection, best_epoch = selection, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    valid_pred = predict(
        model, valid_positions, data, drug_features, None, target_features, target_aux_features,
        target_token_features, target_token_offsets, target_token_lengths,
        target_token_pocket_mask_features, target_token_max_len,
        target_token_cache, target_token_mask_cache, structure_features, structure_mask, arrays,
        device, args.inference_batch_size,
    )
    temperature = temperature_scale(valid_pred["final_logit"], data.iloc[valid_positions]["binary_label"].to_numpy(dtype=np.int8))
    test_pred = predict(
        model, test_positions, data, drug_features, None, target_features, target_aux_features,
        target_token_features, target_token_offsets, target_token_lengths,
        target_token_pocket_mask_features, target_token_max_len,
        target_token_cache, target_token_mask_cache, structure_features, structure_mask, arrays,
        device, args.inference_batch_size,
    )
    test_frame = data.iloc[test_positions].reset_index(drop=True)
    probability = sigmoid(test_pred["final_logit"] / temperature)
    result_metrics = metrics(test_frame, probability)
    final_bindingdb_metrics: dict[str, float | int] = {}
    if len(bindingdb_valid_positions):
        bindingdb_final_prediction = predict(
            model, bindingdb_valid_positions, data, drug_features, None,
            target_features, target_aux_features, target_token_features,
            target_token_offsets, target_token_lengths,
            target_token_pocket_mask_features, target_token_max_len,
            target_token_cache, target_token_mask_cache, structure_features,
            structure_mask, arrays, device, args.inference_batch_size,
        )
        final_bindingdb_metrics = affinity_retrieval_metrics(
            data.iloc[bindingdb_valid_positions], bindingdb_final_prediction["affinity"]
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"{args.protocol}__fold_{fold}__seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "BEST_MODEL_BINDINGDB_AFFINITY_V1.pt"
    torch.save({
        "model_state_dict": best_state,
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
        "temperature": temperature,
        "supplement_counts": supplement_counts,
        "target_token_source": target_token_source,
        "target_token_index_source": target_token_index_source,
        "target_token_pocket_mask_source": target_token_pocket_mask_source,
    }, checkpoint)
    pd.DataFrame(history).to_csv(run_dir / "TRAINING_HISTORY_BINDINGDB_AFFINITY_V1.csv", index=False)
    predictions = test_frame[["calibration_pair_id", "sequence_key", "target_chembl_id", "primary_gene", "target_assay_family", "parent_standard_inchi_key", "parent_molecule_chembl_id", "binary_label", "mean_pchembl", "conplex_score"]].copy()
    for key, value in test_pred.items():
        predictions[f"v2_{key}"] = value
    predictions["v2_probability_calibrated"] = probability
    pred_path = run_dir / "TEST_PREDICTIONS_BINDINGDB_AFFINITY_V1.csv.gz"
    predictions.to_csv(pred_path, index=False)
    bindingdb_validation_path = None
    if len(bindingdb_valid_positions):
        bindingdb_validation_path = run_dir / "BINDINGDB_TARGET_COLD_VALIDATION_PREDICTIONS_V1.csv.gz"
        bindingdb_frame = data.iloc[bindingdb_valid_positions][[
            "calibration_pair_id", "target_sequence_hash", "parent_standard_inchi_key",
            "mean_pchembl", "min_pchembl", "max_pchembl", "bindingdb_publication_year",
        ]].copy()
        for key, value in bindingdb_final_prediction.items():
            bindingdb_frame[f"v2_{key}"] = value
        bindingdb_frame.to_csv(bindingdb_validation_path, index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "fold": fold,
        "seed": args.seed,
        "device": str(device),
        "model": "BIOMASTER_ODTI_V2_POOLED_ESM2_WITH_BINDINGDB_AFFINITY_ONLY",
        "target_token_auxiliary": {"enabled": bool(target_token_features is not None), "feature_dim": int(target_token_dim or 0), "max_len": int(target_token_max_len), "source": target_token_source, "index_source": target_token_index_source, "pocket_mask_source": target_token_pocket_mask_source},
        "split_counts": {"train_internal": int(supplement_counts["train_internal_rows"]), "train_total": int(len(train_positions)), "valid": int(len(valid_positions)), "test": int(len(test_positions))},
        "supplement": supplement_counts | {"enabled": not args.no_supplement, "pairs_sha256": sha256(SUPPLEMENT)},
        "training": {"best_epoch": best_epoch, "epochs_completed": len(history), "best_validation_selection_value": best_selection, "temperature": temperature, "selection_metric": args.selection_metric},
        "test_metrics": result_metrics,
        "bindingdb_target_cold_validation_metrics": final_bindingdb_metrics,
        "checks": {"test_predictions_finite": bool(np.isfinite(probability).all()), "test_predictions_bounded": bool(((probability >= 0) & (probability <= 1)).all()), "bindingdb_binary_masked": True, "frozen_test_role": True},
        "artifacts": {"checkpoint_sha256": sha256(checkpoint), "predictions_sha256": sha256(pred_path), "bindingdb_validation_predictions_sha256": (sha256(bindingdb_validation_path) if bindingdb_validation_path else None)},
        "claim_status": "EXPLORATORY_PAIRED_SCREEN; NOT_SOTA_OR_CHAMPION_PROMOTION",
    }
    (run_dir / "RUN_SUMMARY_BINDINGDB_AFFINITY_V1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["S3_STRICT_DOUBLE_COLD", "S5_OLD_DRUG_ENTITY_COLD"], required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--interaction-mode", choices=["legacy_full", "low_rank_film"], default="legacy_full")
    parser.add_argument("--interaction-rank", type=int, default=48)
    parser.add_argument("--film-scale", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.12)
    parser.add_argument("--affinity-weight", type=float, default=0.06)
    parser.add_argument("--affinity-rank-weight", type=float, default=0.0)
    parser.add_argument("--affinity-drug-rank-weight", type=float, default=0.0)
    parser.add_argument("--affinity-rank-min-delta", type=float, default=0.25)
    parser.add_argument("--affinity-rank-margin", type=float, default=0.10)
    parser.add_argument("--affinity-rank-max-pairs", type=int, default=4096)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--rank-max-pairs", type=int, default=4096)
    parser.add_argument("--max-rows-per-target", type=int, default=16)
    parser.add_argument("--max-rows-per-drug", type=int, default=16)
    parser.add_argument("--batch-sampler", choices=["target", "dual_query"], default="target")
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--selection-metric", choices=["composite", "micro_auprc", "target_macro_auprc", "drug_macro_auprc", "composite_with_bindingdb_target_cold"], default="composite")
    parser.add_argument("--bindingdb-cutoff-year", type=int, default=2022)
    parser.add_argument("--bindingdb-holdout-fold", type=int, default=-1)
    parser.add_argument("--bindingdb-holdout-fold-count", type=int, default=5)
    parser.add_argument("--bindingdb-min-validation-pairs", type=int, default=5)
    parser.add_argument("--out-dir", default="outputs/biomaster_bindingdb_affinity_augmented_screen_v1")
    parser.add_argument("--no-supplement", action="store_true")
    parser.add_argument("--use-target-tokens", action="store_true", help="enable residue-level ESM2 cross-attention branch")
    parser.add_argument("--cache-target-tokens", action="store_true", help="materialize all normalized residue tokens on GPU; can exceed memory for long targets")
    parser.add_argument("--target-token-max-len", type=int, default=1022)
    parser.add_argument("--target-token-heads", type=int, default=4)
    parser.add_argument("--target-token-gate-init-bias", type=float, default=-4.0)
    parser.add_argument("--cpu", action="store_true")
    train_one(parser.parse_args())


if __name__ == "__main__":
    main()

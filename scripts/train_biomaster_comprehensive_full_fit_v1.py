#!/usr/bin/env python3
"""Train validation-selected and FULL_FIT BioMaster models on uncapped ChEMBL37.

The comprehensive ChEMBL table is authoritative for every feature-resolved
positive/negative relation.  Recovered deployment rows are retained as an
independent anti-overfit evaluation: exact dev/test relation keys are removed
from the evaluation fit, but are restored for the post-evaluation FULL_FIT
refit.  BindingDB target-cold development/test targets are likewise excluded
from evaluation fitting.  No test metric selects an epoch.
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2
from train_biomaster_deployment_augmented_v1 import (
    bdb_fused_metrics,
    bdb_selection,
    build_retrieval_frame,
    checkpoint_payload,
    classification_metrics,
    json_safe,
    make_arrays,
    predict_positions,
    retrieval_metrics,
    sha256,
    state_sha256,
    train_epoch,
)
from train_biomaster_odti_v2 import predict, set_seed, sigmoid, temperature_scale


PACKAGE = ROOT / "outputs/biomaster_comprehensive_training_v1"
RELATIONS = PACKAGE / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
MORGAN = PACKAGE / "MORGAN2048_UINT8_COMPREHENSIVE_V1.npy"
PACKAGE_MANIFEST = PACKAGE / "COMPREHENSIVE_TRAINING_MANIFEST_V1.json"
AUGMENT = ROOT / "outputs/biomaster_deployment_augmentation_v1"
RECOVERED = AUGMENT / "RECOVERED_CHEMBL37_RELATIONS_V1.csv.gz"
PROTBERT = AUGMENT / "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy"
ESM2 = AUGMENT / "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy"
TARGET_STRUCTURE = AUGMENT / "TARGET_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz"
BASE_STORE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1"
BASE_PAIRS = BASE_STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
BASE_STRUCTURE = BASE_STORE / "ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
REFERENCE_CHECKPOINT = ROOT / (
    "outputs/biomaster_bindingdb_interval_rank_multiseed_v2/w08/"
    "S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_20260816/"
    "BEST_MODEL_BINDINGDB_AFFINITY_V1.pt"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_comprehensive_full_fit_v1"


def target_structure_arrays(
    data: pd.DataFrame,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, dict[str, int]]:
    """Expand target-constant pocket context to all comprehensive pair rows."""

    base_pairs = pd.read_csv(
        BASE_PAIRS, usecols=["calibration_pair_id", "target_feature_index"], low_memory=False
    )
    base_structure = pd.read_csv(BASE_STRUCTURE, low_memory=False)
    feature_columns = [
        column for column in base_structure.columns
        if column not in {"calibration_pair_id", "structure_mask"}
    ]
    if len(feature_columns) != 19:
        raise RuntimeError(f"expected 19 structure columns, found {len(feature_columns)}")
    base = base_pairs.merge(
        base_structure, on="calibration_pair_id", how="left", validate="one_to_one"
    )
    for column in feature_columns + ["structure_mask"]:
        if base.groupby("target_feature_index", sort=False)[column].nunique(dropna=False).max() > 1:
            raise RuntimeError(f"base structure feature is not target-constant: {column}")
    base = base.sort_values("calibration_pair_id").drop_duplicates("target_feature_index")
    augmented = pd.read_csv(TARGET_STRUCTURE, low_memory=False)
    missing = set(feature_columns + ["target_feature_index", "structure_mask"]) - set(augmented.columns)
    if missing:
        raise RuntimeError(f"target structure table is missing columns: {sorted(missing)}")
    if augmented["target_feature_index"].duplicated().any():
        raise RuntimeError("augmented target structure indices are not unique")

    values = np.zeros((target_count, len(feature_columns)), dtype=np.float32)
    mask = np.zeros(target_count, dtype=np.float32)
    for frame in (base, augmented):
        indices = frame["target_feature_index"].to_numpy(dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= target_count):
            raise RuntimeError("target structure index exceeds target embedding matrix")
        values[indices] = frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        mask[indices] = frame["structure_mask"].fillna(0.0).to_numpy(dtype=np.float32)
    pair_indices = data["target_feature_index"].to_numpy(dtype=np.int64)
    counts = {
        "target_count": int(target_count),
        "base_structure_targets": int(base["target_feature_index"].nunique()),
        "augmented_structure_targets": int(augmented["target_feature_index"].nunique()),
        "available_structure_targets": int((mask > 0).sum()),
        "pair_rows_with_structure": int((mask[pair_indices] > 0).sum()),
    }
    return values[pair_indices], mask[pair_indices], feature_columns, augmented, counts


def exact_keys(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["target_chembl_id"].fillna("").astype(str)
        + "__"
        + frame["parent_standard_inchi_key"].fillna("").astype(str)
    )


def split_positions(data: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    kinds = data["source_kind"].fillna("").astype(str)
    source = np.flatnonzero(kinds.eq("chembl37_comprehensive").to_numpy()).astype(np.int64)
    bdb = np.flatnonzero(kinds.eq("bindingdb_affinity_only").to_numpy()).astype(np.int64)
    recovered = np.flatnonzero(kinds.eq("chembl37_recovered").to_numpy()).astype(np.int64)
    if not (len(source) > 426000 and len(bdb) > 9000 and len(recovered) > 900):
        raise RuntimeError("comprehensive source partitions failed row-count contract")

    recovered_frame = data.iloc[recovered]
    roles = recovered_frame["augmentation_role"].astype(str).to_numpy()
    rec_train = recovered[roles == "train"]
    rec_dev = recovered[roles == "dev"]
    rec_test = recovered[roles == "test"]
    rec_train_unique = rec_train[
        ~data.iloc[rec_train]["duplicate_of_comprehensive"].astype(bool).to_numpy()
    ]
    rec_full_unique = recovered[
        ~data.iloc[recovered]["duplicate_of_comprehensive"].astype(bool).to_numpy()
    ]
    held_relation_keys = set(data.iloc[np.concatenate([rec_dev, rec_test])]["exact_pair_key"].astype(str))

    source_eval = source[
        ~data.iloc[source]["exact_pair_key"].astype(str).isin(held_relation_keys).to_numpy()
    ]
    bdb_frame = data.iloc[bdb]
    recovered_hashes = set(recovered_frame["target_sequence_hash"].astype(str))
    stats = bdb_frame.groupby("target_sequence_hash").agg(
        rows=("mean_pchembl", "size"),
        minimum=("mean_pchembl", "min"),
        maximum=("mean_pchembl", "max"),
        target_feature_index=("target_feature_index", "first"),
    )
    eligible = stats.loc[
        stats["target_feature_index"].astype(int).ge(428)
        & stats["rows"].ge(5)
        & stats["minimum"].le(5.0)
        & stats["maximum"].ge(6.0)
        & ~stats.index.astype(str).isin(recovered_hashes)
    ].copy()
    eligible["fold"] = [
        int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 5
        for value in eligible.index
    ]
    bdb_dev_hashes = set(eligible.index[eligible["fold"].eq(0)].astype(str))
    bdb_test_hashes = set(eligible.index[eligible["fold"].eq(1)].astype(str))
    bdb_hash = bdb_frame["target_sequence_hash"].astype(str)
    bdb_dev = bdb[bdb_hash.isin(bdb_dev_hashes).to_numpy()]
    bdb_test = bdb[bdb_hash.isin(bdb_test_hashes).to_numpy()]
    held_bdb_hashes = bdb_dev_hashes | bdb_test_hashes
    bdb_eval = bdb[
        (~bdb_hash.isin(held_bdb_hashes) & ~bdb_frame["exact_pair_key"].astype(str).isin(held_relation_keys)).to_numpy()
    ]

    eval_fit = np.concatenate([source_eval, bdb_eval, rec_train_unique]).astype(np.int64)
    full_fit = np.concatenate([source, bdb, rec_full_unique]).astype(np.int64)
    held_rec = np.concatenate([rec_dev, rec_test]).astype(np.int64)
    fit_drugs = set(data.iloc[eval_fit]["parent_standard_inchi_key"].astype(str))
    fit_targets = set(data.iloc[eval_fit]["target_feature_index"].astype(int))
    if set(data.iloc[held_rec]["parent_standard_inchi_key"].astype(str)) - fit_drugs:
        raise RuntimeError("held-out recovered drug is not warm in evaluation fit")
    if set(data.iloc[held_rec]["target_feature_index"].astype(int)) - fit_targets:
        raise RuntimeError("held-out recovered target is not warm in evaluation fit")
    if held_bdb_hashes & set(data.iloc[eval_fit]["target_sequence_hash"].astype(str)):
        raise RuntimeError("BindingDB exact target leaked into evaluation fit")
    if held_relation_keys & set(data.iloc[eval_fit]["exact_pair_key"].astype(str)):
        raise RuntimeError("recovered dev/test exact relation leaked into evaluation fit")

    positions = {
        "source": source,
        "bdb": bdb,
        "recovered": recovered,
        "rec_train": rec_train,
        "rec_dev": rec_dev,
        "rec_test": rec_test,
        "bdb_dev": bdb_dev,
        "bdb_test": bdb_test,
        "eval_fit": eval_fit,
        "full_fit": full_fit,
    }
    audit = {
        "source_rows": len(source),
        "bindingdb_rows": len(bdb),
        "recovered_rows": len(recovered),
        "recovered_train_rows": len(rec_train),
        "recovered_train_unique_rows_added": len(rec_train_unique),
        "recovered_dev_rows": len(rec_dev),
        "recovered_test_rows": len(rec_test),
        "recovered_duplicates_not_repeated_full_fit": int(len(recovered) - len(rec_full_unique)),
        "held_relation_keys": len(held_relation_keys),
        "source_rows_excluded_for_relation_evaluation": int(len(source) - len(source_eval)),
        "bindingdb_target_cold_dev_rows": len(bdb_dev),
        "bindingdb_target_cold_dev_targets": len(bdb_dev_hashes),
        "bindingdb_target_cold_test_rows": len(bdb_test),
        "bindingdb_target_cold_test_targets": len(bdb_test_hashes),
        "evaluation_fit_rows": len(eval_fit),
        "full_fit_rows": len(full_fit),
        "full_fit_unique_drugs": data.iloc[full_fit]["drug_feature_index"].nunique(),
        "full_fit_unique_targets": data.iloc[full_fit]["target_feature_index"].nunique(),
    }
    return positions, audit


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
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.patience < 1:
        raise ValueError("invalid epoch/patience contract")
    required = [
        RELATIONS, MORGAN, PACKAGE_MANIFEST, RECOVERED, PROTBERT, ESM2,
        TARGET_STRUCTURE, BASE_PAIRS, BASE_STRUCTURE, REFERENCE_CHECKPOINT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    package_manifest = json.loads(PACKAGE_MANIFEST.read_text())
    if package_manifest.get("status") != "PASS":
        raise RuntimeError("comprehensive package manifest is not PASS")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = pd.read_csv(RELATIONS, low_memory=False)
    data["exact_pair_key"] = exact_keys(data)
    positions, split_audit = split_positions(data)
    recovered = data.iloc[positions["recovered"]].copy()
    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux = np.load(ESM2, mmap_mode="r")
    if data["drug_feature_index"].max() >= len(drug_features):
        raise RuntimeError("drug feature index exceeds comprehensive Morgan matrix")
    if data["target_feature_index"].max() >= len(target_features):
        raise RuntimeError("target feature index exceeds augmented embedding matrix")
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ODTIV2Config(**reference["config"])
    if config.structure_input_dim != 19 or config.interaction_mode != "low_rank_film":
        raise RuntimeError("reference architecture contract failed")
    structure_features, structure_mask, structure_columns, target_structure, structure_audit = (
        target_structure_arrays(data, len(target_features))
    )
    dev_retrieval_frame, dev_retrieval_structure, dev_retrieval_mask = build_retrieval_frame(
        recovered, "dev", target_structure
    )
    test_retrieval_frame, test_retrieval_structure, test_retrieval_mask = build_retrieval_frame(
        recovered, "test", target_structure
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    eval_fit = positions["eval_fit"]
    rec_train = positions["rec_train"]
    rec_dev = positions["rec_dev"]
    rec_test = positions["rec_test"]
    bdb_dev = positions["bdb_dev"]
    bdb_test = positions["bdb_test"]
    eval_arrays = make_arrays(
        data, eval_fit, target_features, target_aux, structure_features,
        structure_mask, config.target_token_max_len,
    )
    # The uncapped matrix is I/O-bound if 2,048-bit rows are repeatedly read
    # from a memmap.  A numerically identical float32 cache fits comfortably
    # on the deployment GPU and materially shortens every epoch.
    drug_feature_cache = torch.from_numpy(
        np.asarray(drug_features, dtype=np.float32)
    ).to(device)
    eval_target_feature_cache = torch.from_numpy(
        (np.asarray(target_features, dtype=np.float32) - eval_arrays["target_mean"])
        / eval_arrays["target_std"]
    ).to(device)
    eval_target_aux_feature_cache = torch.from_numpy(
        (np.asarray(target_aux, dtype=np.float32) - eval_arrays["target_aux_mean"])
        / eval_arrays["target_aux_std"]
    ).to(device)
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
            drug_feature_cache=drug_feature_cache,
            target_feature_cache=eval_target_feature_cache,
            target_aux_feature_cache=eval_target_aux_feature_cache,
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
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
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

    dev_best_prediction = predict_positions(
        model, rec_dev, data, drug_features, target_features, target_aux,
        structure_features, structure_mask, eval_arrays, config, device,
        args.inference_batch_size,
    )
    temperature = temperature_scale(
        dev_best_prediction["final_logit"], data.iloc[rec_dev]["binary_label"].to_numpy(dtype=np.int8)
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
    run_dir = out_root / f"FULL_FIT_2026_COMPREHENSIVE__seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_checkpoint = run_dir / "EVALUATION_BEST_MODEL_COMPREHENSIVE_V1.pt"
    evaluation_contract = {
        "mode": "VALIDATION_SELECTED_UNTOUCHED_TEST_EVALUATION",
        "selected_epoch": best_epoch,
        "maximum_epochs": args.max_epochs,
        "stop_reason": stop_reason,
        "recovered_dev_test_exact_relations_excluded": True,
        "bindingdb_target_cold_dev_test_excluded": True,
        "chembl37_per_target_label_cap": None,
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
    retrieval_test_known.to_csv(
        run_dir / "DOUBLE_WARM_RETRIEVAL_TEST_POSITIVE_RANKS_V1.csv", index=False
    )
    bdb_test_frame = data.iloc[bdb_test][[
        "calibration_pair_id", "target_sequence_hash", "parent_standard_inchi_key",
        "mean_pchembl", "min_pchembl", "max_pchembl",
    ]].copy()
    bdb_test_frame["binary_score"] = bdb_test_prediction["final_logit"]
    bdb_test_frame["affinity_score"] = bdb_test_prediction["affinity"]
    bdb_test_frame["fused_rank_score"] = bdb_test_fused
    bdb_test_frame.to_csv(
        run_dir / "BINDINGDB_TARGET_COLD_TEST_PREDICTIONS_V1.csv.gz",
        index=False, compression="gzip",
    )

    # Fresh post-test refit.  The test labels above do not affect this epoch count.
    del model
    del eval_target_feature_cache, eval_target_aux_feature_cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(args.seed)
    full_fit = positions["full_fit"]
    full_arrays = make_arrays(
        data, full_fit, target_features, target_aux, structure_features,
        structure_mask, config.target_token_max_len,
    )
    full_target_feature_cache = torch.from_numpy(
        (np.asarray(target_features, dtype=np.float32) - full_arrays["target_mean"])
        / full_arrays["target_std"]
    ).to(device)
    full_target_aux_feature_cache = torch.from_numpy(
        (np.asarray(target_aux, dtype=np.float32) - full_arrays["target_aux_mean"])
        / full_arrays["target_aux_std"]
    ).to(device)
    final_model = RoutedInteractionRankerV2(
        len(full_arrays["families"]), config, use_conplex=False
    ).to(device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer, T_max=args.max_epochs, eta_min=args.learning_rate / 20
    )
    refit_history: list[dict[str, object]] = []
    for epoch in range(1, best_epoch + 1):
        row = train_epoch(
            final_model, final_optimizer, full_fit, epoch, args.seed, data,
            drug_features, target_features, target_aux, structure_features,
            structure_mask, full_arrays, config, device, args.batch_size,
            args.max_rows_per_target, args.gradient_clip,
            drug_feature_cache=drug_feature_cache,
            target_feature_cache=full_target_feature_cache,
            target_aux_feature_cache=full_target_aux_feature_cache,
        )
        final_scheduler.step()
        row = {"epoch": epoch, "learning_rate": float(final_scheduler.get_last_lr()[0]), **row}
        refit_history.append(row)
        print(json.dumps({"seed": args.seed, "refit": True, **row}, default=json_safe), flush=True)
    final_state = {
        name: value.detach().cpu().clone() for name, value in final_model.state_dict().items()
    }
    final_checkpoint = run_dir / "FULL_FIT_MODEL_COMPREHENSIVE_V1.pt"
    full_contract = {
        "protocol": "FULL_FIT_2026_UNCAPPED_CHEMBL37_COMPREHENSIVE_V1",
        "training_mode": "FRESH_REFIT_ALL_HISTORICAL_RELATIONS_AT_SELECTED_EPOCH",
        "selected_epoch": best_epoch,
        "scheduler_t_max": args.max_epochs,
        "chembl37_per_target_label_cap": None,
        "chembl37_feature_resolved_rows": split_audit["source_rows"],
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
        "protocol": "FULL_FIT_2026_UNCAPPED_CHEMBL37_COMPREHENSIVE_V1",
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
        "split_and_fit_counts": split_audit,
        "structure": structure_audit,
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
        },
        "sources": {
            str(path.relative_to(ROOT)): sha256(path) for path in required
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
            "Test metrics are from the validation-selected evaluation checkpoint only. "
            "The FULL_FIT checkpoint is a fresh post-test refit on all feature-resolved historical relations."
        ),
    }
    summary_path = run_dir / "FULL_FIT_RUN_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), flush=True)


if __name__ == "__main__":
    main()

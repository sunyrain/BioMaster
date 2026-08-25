#!/usr/bin/env python3
"""Train BioMaster on all relations with balanced gradients and aligned retrieval.

Model selection is deliberately retrospective and label-blind with respect to
the current external showcase cases.  ChEMBL relations first reported through
2022 form the evaluation fit; 2023 double-warm positives select the epoch in a
target-to-720-old-drugs retrieval task, and 2024--2025 positives remain an
untouched temporal test.  A fresh FULL_FIT refit then makes every eligible
ChEMBL/BindingDB/recovered relation available through balanced sampling.
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

from biomaster.comprehensive_balanced import (  # noqa: E402
    BalancedSamplingConfig,
    balanced_training_batches,
    sampling_audit,
    target_to_drug_metrics,
)
from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2, odti_v2_loss  # noqa: E402
from score_biomaster_deployment_augmented_720x384_v1 import augmented_structure  # noqa: E402
from train_biomaster_comprehensive_full_fit_v1 import (  # noqa: E402
    BASE_PAIRS,
    BASE_STRUCTURE,
    ESM2,
    MORGAN,
    PACKAGE_MANIFEST,
    PROTBERT,
    RECOVERED,
    REFERENCE_CHECKPOINT,
    RELATIONS,
    TARGET_STRUCTURE,
    exact_keys,
    split_positions,
    target_structure_arrays,
)
from train_biomaster_deployment_augmented_v1 import (  # noqa: E402
    bdb_fused_metrics,
    bdb_selection,
    checkpoint_payload,
    classification_metrics,
    json_safe,
    make_arrays,
    predict_positions,
    sha256,
    state_sha256,
    train_epoch,
)
from train_biomaster_odti_v2 import predict, set_seed, sigmoid, temperature_scale, tensor_batch  # noqa: E402


DEPLOY = ROOT / "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1"
DEPLOY_PAIRS = DEPLOY / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
DEPLOY_DRUG = DEPLOY / "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
DEPLOY_TARGET = DEPLOY / "PROJECT384_PROTBERT1024_FLOAT32_V1.npy"
DEPLOY_TARGET_INDEX = DEPLOY / "PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz"
DEPLOY_TARGET_AUX = ROOT / (
    "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy"
)
DRUG_FEATURE_INDEX = (
    ROOT / "outputs/biomaster_comprehensive_training_v1/DRUG_FEATURE_INDEX_COMPREHENSIVE_V1.csv.gz"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_comprehensive_balanced_full_fit_v2"
# Negative-aware post-training calibration selected the binary head for the
# primary target-to-720 retrieval objective.  The affinity head is still
# trained and evaluated independently.
AFFINITY_WEIGHT = 0.0


def balanced_arrays(
    data: pd.DataFrame,
    fit_positions: np.ndarray,
    target_features: np.ndarray,
    target_aux: np.ndarray,
    structure_features: np.ndarray,
    structure_mask: np.ndarray,
    target_token_max_len: int,
) -> tuple[dict[str, object], dict[str, float | int]]:
    """Use unique-target and target/label-balanced normalization statistics."""

    arrays = make_arrays(
        data,
        fit_positions,
        target_features,
        target_aux,
        structure_features,
        structure_mask,
        target_token_max_len,
    )
    frame = data.iloc[fit_positions].copy()
    available = frame.loc[structure_mask[fit_positions] > 0].drop_duplicates(
        "target_feature_index"
    )
    if len(available):
        absolute = available.index.to_numpy(dtype=np.int64)
        values = np.asarray(structure_features[absolute], dtype=np.float32)
        structure_mean = values.mean(axis=0).astype(np.float32)
        structure_std = values.std(axis=0).astype(np.float32)
        structure_std[structure_std < 1e-6] = 1.0
        arrays["structure_mean"] = structure_mean
        arrays["structure_std"] = structure_std

    affinity = pd.to_numeric(frame["mean_pchembl"], errors="coerce")
    binary_observed = pd.to_numeric(frame["binary_observed"], errors="coerce").fillna(0).eq(1)
    labels = pd.to_numeric(frame["binary_label"], errors="coerce").fillna(-1).astype(int)
    normalizer = frame[["target_feature_index"]].copy()
    normalizer["affinity"] = affinity
    normalizer["stratum"] = np.where(binary_observed, "B" + labels.astype(str), "AFFINITY_ONLY")
    group_moments = (
        normalizer.loc[normalizer["affinity"].notna()]
        .groupby(["target_feature_index", "stratum"], sort=False)["affinity"]
        .agg(["mean", lambda values: float(np.square(values).mean())])
    )
    group_moments.columns = ["mean", "second"]
    if len(group_moments):
        affinity_mean = float(group_moments["mean"].mean())
        affinity_variance = max(float(group_moments["second"].mean()) - affinity_mean**2, 0.0)
        affinity_std = max(float(np.sqrt(affinity_variance)), 1e-6)
        arrays["affinity_mean"] = affinity_mean
        arrays["affinity_std"] = affinity_std
    audit = {
        "structure_normalization_unique_targets": int(len(available)),
        "affinity_normalization_target_strata": int(len(group_moments)),
        "affinity_mean": float(arrays["affinity_mean"]),
        "affinity_std": float(arrays["affinity_std"]),
    }
    return arrays, audit


def optimized_splits(
    data: pd.DataFrame,
    raw_positions: dict[str, np.ndarray],
    deployment: pd.DataFrame,
    deployment_target_by_sequence: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Create strict temporal selection/test sets and warm relation controls."""

    source = raw_positions["source"]
    source_frame = data.iloc[source].copy()
    year = pd.to_numeric(source_frame["min_document_year"], errors="coerce")
    source_pre = source[year.le(2022).to_numpy()]
    pre_frame = data.iloc[source_pre]
    warm_drugs = set(pre_frame["parent_standard_inchi_key"].astype(str))
    warm_target_sequences = set(pre_frame["sequence_sha256"].astype(str))
    deployment_drugs = set(deployment["ligand_inchikey"].astype(str))
    deployment_target_sequences = set(deployment_target_by_sequence)

    future_eligible = (
        source_frame["binary_label"].eq(1)
        & source_frame["parent_standard_inchi_key"].astype(str).isin(deployment_drugs)
        & source_frame["sequence_sha256"].astype(str).isin(deployment_target_sequences)
        & source_frame["parent_standard_inchi_key"].astype(str).isin(warm_drugs)
        & source_frame["sequence_sha256"].astype(str).isin(warm_target_sequences)
    )
    temporal_dev = source[(future_eligible & year.eq(2023)).to_numpy()]
    temporal_test = source[(future_eligible & year.isin([2024, 2025])).to_numpy()]
    if len(temporal_dev) < 10 or len(temporal_test) < 20:
        raise RuntimeError("strict temporal target-to-drug split is too small")

    recovered = raw_positions["recovered"]
    recovered_frame = data.iloc[recovered]
    held_recovered = np.concatenate([raw_positions["rec_dev"], raw_positions["rec_test"]])
    held_recovered_keys = set(data.iloc[held_recovered]["exact_pair_key"].astype(str))
    source_pre = source_pre[
        ~data.iloc[source_pre]["exact_pair_key"].astype(str).isin(held_recovered_keys).to_numpy()
    ]
    rec_train = raw_positions["rec_train"]
    rec_train_frame = data.iloc[rec_train]
    rec_train_year = pd.to_numeric(rec_train_frame["min_document_year"], errors="coerce")
    rec_train_unique = rec_train[
        rec_train_year.le(2022).to_numpy()
        & ~rec_train_frame["duplicate_of_comprehensive"].astype(bool).to_numpy()
    ]
    eval_fit = np.concatenate([source_pre, rec_train_unique]).astype(np.int64)
    eval_drugs = set(data.iloc[eval_fit]["parent_standard_inchi_key"].astype(str))
    eval_targets = set(data.iloc[eval_fit]["target_feature_index"].astype(int))

    temporal_keys = set(
        data.iloc[np.concatenate([temporal_dev, temporal_test])]["exact_pair_key"].astype(str)
    )

    def warm_recovered(values: np.ndarray) -> np.ndarray:
        frame = data.iloc[values]
        keep = (
            frame["parent_standard_inchi_key"].astype(str).isin(eval_drugs)
            & frame["target_feature_index"].astype(int).isin(eval_targets)
            & frame["sequence_sha256"].astype(str).isin(deployment_target_sequences)
            & ~frame["exact_pair_key"].astype(str).isin(temporal_keys)
        )
        return values[keep.to_numpy()]

    rec_dev = warm_recovered(raw_positions["rec_dev"])
    rec_test = warm_recovered(raw_positions["rec_test"])
    if data.iloc[rec_dev]["binary_label"].nunique() < 2:
        raise RuntimeError("warm recovered development set lost a binary class")
    if set(data.iloc[eval_fit]["exact_pair_key"].astype(str)) & (
        temporal_keys | set(data.iloc[np.concatenate([rec_dev, rec_test])]["exact_pair_key"].astype(str))
    ):
        raise RuntimeError("an evaluation relation leaked into temporal fit")

    positions = {
        **raw_positions,
        "eval_fit": eval_fit,
        "temporal_dev": temporal_dev,
        "temporal_test": temporal_test,
        "rec_dev_warm": rec_dev,
        "rec_test_warm": rec_test,
    }
    audit = {
        "evaluation_cutoff_year_inclusive": 2022,
        "evaluation_source_rows": int(len(source_pre)),
        "evaluation_recovered_unique_train_rows": int(len(rec_train_unique)),
        "evaluation_fit_rows": int(len(eval_fit)),
        "evaluation_bindingdb_rows": 0,
        "source_rows_with_missing_year_excluded": int(year.isna().sum()),
        "source_rows_after_cutoff_excluded": int(year.gt(2022).sum()),
        "temporal_dev_rows_2023": int(len(temporal_dev)),
        "temporal_dev_targets": int(data.iloc[temporal_dev]["target_chembl_id"].nunique()),
        "temporal_test_rows_2024_2025": int(len(temporal_test)),
        "temporal_test_targets": int(data.iloc[temporal_test]["target_chembl_id"].nunique()),
        "warm_recovered_dev_rows": int(len(rec_dev)),
        "warm_recovered_test_rows": int(len(rec_test)),
        "full_fit_rows": int(len(raw_positions["full_fit"])),
    }
    return positions, audit


def build_target_to_drug_frame(
    positives: pd.DataFrame,
    deployment: pd.DataFrame,
    deployment_structure: np.ndarray,
    deployment_structure_mask: np.ndarray,
    deployment_target_by_sequence: dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Cross each held positive target with the frozen 720-drug library."""

    positives = positives.loc[positives["binary_label"].eq(1)].drop_duplicates(
        ["target_chembl_id", "parent_standard_inchi_key"]
    )
    chunks = []
    structures = []
    masks = []
    for positive in positives.itertuples(index=False):
        sequence_hash = str(positive.sequence_sha256)
        deployment_target_id = deployment_target_by_sequence.get(sequence_hash)
        if deployment_target_id is None:
            raise RuntimeError(f"held target sequence is absent from deployment: {sequence_hash}")
        selected = deployment["target_chembl_id"].astype(str).eq(deployment_target_id)
        candidates = deployment.loc[selected].copy()
        if len(candidates) != 720:
            raise RuntimeError(f"deployment target does not have 720 drugs: {deployment_target_id}")
        hits = candidates["ligand_inchikey"].astype(str).eq(str(positive.parent_standard_inchi_key))
        if int(hits.sum()) != 1:
            raise RuntimeError("held positive does not map once into the 720-drug universe")
        candidates["query_id"] = str(positive.calibration_pair_id)
        candidates["source_target_chembl_id"] = str(positive.target_chembl_id)
        candidates["parent_standard_inchi_key"] = candidates["ligand_inchikey"].astype(str)
        candidates["calibration_pair_id"] = (
            "T2D::" + str(positive.calibration_pair_id) + "::" + candidates["drug_feature_index"].astype(str)
        )
        candidates["binary_label"] = hits.astype(np.int8)
        candidates["binary_observed"] = 1
        candidates["mean_pchembl"] = np.nan
        candidates["min_pchembl"] = np.nan
        candidates["max_pchembl"] = np.nan
        candidates["conplex_score"] = 0.0
        chunks.append(candidates)
        original = candidates.index.to_numpy(dtype=np.int64)
        structures.append(deployment_structure[original])
        masks.append(deployment_structure_mask[original])
    if not chunks:
        raise RuntimeError("target-to-drug retrieval frame has no queries")
    return (
        pd.concat(chunks, ignore_index=True),
        np.concatenate(structures).astype(np.float32),
        np.concatenate(masks).astype(np.float32),
    )


def external_arrays(frame: pd.DataFrame, normalization: dict[str, object]) -> dict[str, object]:
    families = [str(value) for value in normalization["families"]]
    lookup = {name: index for index, name in enumerate(families)}
    family = frame["target_assay_family"].astype(str).map(lookup).fillna(lookup["__UNK__"])
    return {
        "families": families,
        "family_index": family.to_numpy(dtype=np.int64),
        "drug_aux_mean": np.zeros(0, dtype=np.float32),
        "drug_aux_std": np.ones(0, dtype=np.float32),
        "target_mean": normalization["target_mean"],
        "target_std": normalization["target_std"],
        "target_aux_mean": normalization["target_aux_mean"],
        "target_aux_std": normalization["target_aux_std"],
        "target_token_mean": normalization["target_token_mean"],
        "target_token_std": normalization["target_token_std"],
        "conplex": frame["conplex_score"].to_numpy(dtype=np.float32),
        "conplex_mean": normalization["conplex_mean"],
        "conplex_std": normalization["conplex_std"],
        "affinity": frame["mean_pchembl"].to_numpy(dtype=np.float32),
        "affinity_lower": frame["min_pchembl"].to_numpy(dtype=np.float32),
        "affinity_upper": frame["max_pchembl"].to_numpy(dtype=np.float32),
        "affinity_mean": normalization["affinity_mean"],
        "affinity_std": normalization["affinity_std"],
        "structure_mean": normalization["structure_mean"],
        "structure_std": normalization["structure_std"],
    }


def train_balanced_epoch(
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
    sampling: BalancedSamplingConfig,
    gradient_clip: float,
    drug_feature_cache: torch.Tensor,
    target_feature_cache: torch.Tensor,
    target_aux_feature_cache: torch.Tensor,
) -> dict[str, object]:
    model.train()
    batches = balanced_training_batches(positions, data, seed + epoch, sampling)
    component: dict[str, float] = {}
    for batch in batches:
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
            drug_feature_cache=drug_feature_cache,
            target_feature_cache=target_feature_cache,
            target_aux_feature_cache=target_aux_feature_cache,
        )
        values["binary_observed"] = torch.from_numpy(
            data.iloc[batch]["binary_observed"].to_numpy(dtype=np.float32).copy()
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            values["drug"],
            values["target"],
            values["family"],
            values["conplex"],
            values["structure"],
            values["structure_mask"],
            target_aux=values["target_aux"],
        )
        losses = odti_v2_loss(
            output,
            values["labels"],
            values["target_group"],
            values["drug_group"],
            affinity_lower=values["affinity_lower"],
            affinity_upper=values["affinity_upper"],
            affinity_observed=values["affinity_observed"],
            binary_observed=values["binary_observed"],
            config=config,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        for name, value in losses.items():
            component[name] = component.get(name, 0.0) + float(value.detach())
    audit = sampling_audit(batches, data)
    return {
        **audit,
        **{f"loss_{name}": value / len(batches) for name, value in component.items()},
    }


def target_to_drug_prediction(
    model: RoutedInteractionRankerV2,
    frame: pd.DataFrame,
    structure: np.ndarray,
    structure_mask: np.ndarray,
    arrays: dict[str, object],
    config: ODTIV2Config,
    deploy_drug: np.ndarray,
    deploy_target: np.ndarray,
    deploy_target_aux: np.ndarray,
    device: torch.device,
    batch_size: int,
    drug_cache: torch.Tensor,
    target_cache: torch.Tensor,
    target_aux_cache: torch.Tensor,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    prediction = predict(
        model,
        np.arange(len(frame), dtype=np.int64),
        frame,
        deploy_drug,
        None,
        deploy_target,
        deploy_target_aux,
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
    return target_to_drug_metrics(
        frame,
        prediction["final_logit"],
        prediction["affinity"],
        AFFINITY_WEIGHT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps-per-epoch", type=int, default=128)
    parser.add_argument("--rows-per-target", type=int, default=16)
    parser.add_argument("--coverage-sweep", action="store_true")
    parser.add_argument("--coverage-max-rows-per-target", type=int, default=16)
    parser.add_argument("--target-frequency-power", type=float, default=0.5)
    parser.add_argument("--affinity-chunk-fraction", type=float, default=0.09375)
    parser.add_argument("--inactive-only-negative-fraction", type=float, default=0.25)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.patience < 1:
        raise ValueError("invalid epoch/patience contract")
    if args.coverage_max_rows_per_target < 1:
        raise ValueError("coverage-max-rows-per-target must be positive")
    sampling_full = BalancedSamplingConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        rows_per_target=args.rows_per_target,
        target_frequency_power=args.target_frequency_power,
        affinity_chunk_fraction=args.affinity_chunk_fraction,
        inactive_only_negative_fraction=args.inactive_only_negative_fraction,
    )
    sampling_full.validate()
    sampling_eval = BalancedSamplingConfig(
        **{**sampling_full.to_dict(), "affinity_chunk_fraction": 0.0}
    )
    required = [
        RELATIONS,
        MORGAN,
        PACKAGE_MANIFEST,
        RECOVERED,
        PROTBERT,
        ESM2,
        TARGET_STRUCTURE,
        BASE_PAIRS,
        BASE_STRUCTURE,
        REFERENCE_CHECKPOINT,
        DEPLOY_PAIRS,
        DEPLOY_DRUG,
        DEPLOY_TARGET,
        DEPLOY_TARGET_INDEX,
        DEPLOY_TARGET_AUX,
        DRUG_FEATURE_INDEX,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(PACKAGE_MANIFEST.read_text()).get("status") != "PASS":
        raise RuntimeError("comprehensive package manifest is not PASS")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = pd.read_csv(RELATIONS, low_memory=False)
    data["exact_pair_key"] = exact_keys(data)
    # V1 wrote standardized scaffolds to the drug index but did not propagate
    # them to the relation table.  Hydrate by the immutable feature index so an
    # existing expensive fingerprint package remains reusable.
    drug_index = pd.read_csv(
        DRUG_FEATURE_INDEX,
        usecols=["drug_feature_index", "murcko_scaffold"],
        low_memory=False,
    ).set_index("drug_feature_index")
    scaffold = drug_index.loc[
        data["drug_feature_index"].to_numpy(dtype=np.int64), "murcko_scaffold"
    ].reset_index(drop=True)
    data["murcko_scaffold"] = scaffold.fillna("").astype(str).to_numpy()
    if data["murcko_scaffold"].eq("").any():
        raise RuntimeError("standardized scaffold hydration failed")
    deployment = pd.read_csv(DEPLOY_PAIRS, low_memory=False)
    deployment_target_index = pd.read_csv(DEPLOY_TARGET_INDEX, low_memory=False)
    deployment_target_index["sequence_sha256"] = deployment_target_index["sequence"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    if deployment_target_index["sequence_sha256"].duplicated().any():
        raise RuntimeError("deployment target sequences are not unique")
    deployment_target_by_sequence = dict(zip(
        deployment_target_index["sequence_sha256"].astype(str),
        deployment_target_index["target_chembl_id"].astype(str),
        strict=True,
    ))
    raw_positions, raw_split_audit = split_positions(data)
    positions, optimized_split_audit = optimized_splits(
        data, raw_positions, deployment, deployment_target_by_sequence
    )
    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux = np.load(ESM2, mmap_mode="r")
    deploy_drug = np.load(DEPLOY_DRUG, mmap_mode="r")
    deploy_target = np.load(DEPLOY_TARGET, mmap_mode="r")
    deploy_target_aux = np.load(DEPLOY_TARGET_AUX, mmap_mode="r")
    reference = torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = ODTIV2Config(**reference["config"])
    if config.structure_input_dim != 19 or config.interaction_mode != "low_rank_film":
        raise RuntimeError("reference architecture contract failed")
    structure_features, structure_mask, structure_columns, _, structure_audit = (
        target_structure_arrays(data, len(target_features))
    )
    deployment_structure, deployment_structure_mask, deployment_structure_audit = (
        augmented_structure(deployment)
    )

    temporal_dev_frame, temporal_dev_structure, temporal_dev_mask = build_target_to_drug_frame(
        data.iloc[positions["temporal_dev"]],
        deployment,
        deployment_structure,
        deployment_structure_mask,
        deployment_target_by_sequence,
    )
    temporal_test_frame, temporal_test_structure, temporal_test_mask = build_target_to_drug_frame(
        data.iloc[positions["temporal_test"]],
        deployment,
        deployment_structure,
        deployment_structure_mask,
        deployment_target_by_sequence,
    )
    recovered_dev_frame, recovered_dev_structure, recovered_dev_mask = build_target_to_drug_frame(
        data.iloc[positions["rec_dev_warm"]],
        deployment,
        deployment_structure,
        deployment_structure_mask,
        deployment_target_by_sequence,
    )
    recovered_test_frame, recovered_test_structure, recovered_test_mask = build_target_to_drug_frame(
        data.iloc[positions["rec_test_warm"]],
        deployment,
        deployment_structure,
        deployment_structure_mask,
        deployment_target_by_sequence,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    eval_fit = positions["eval_fit"]
    eval_arrays, eval_normalization_audit = balanced_arrays(
        data,
        eval_fit,
        target_features,
        target_aux,
        structure_features,
        structure_mask,
        config.target_token_max_len,
    )
    temporal_dev_arrays = external_arrays(temporal_dev_frame, eval_arrays)
    temporal_test_arrays = external_arrays(temporal_test_frame, eval_arrays)
    recovered_dev_arrays = external_arrays(recovered_dev_frame, eval_arrays)
    recovered_test_arrays = external_arrays(recovered_test_frame, eval_arrays)

    drug_cache = torch.from_numpy(np.asarray(drug_features, dtype=np.float32)).to(device)
    target_cache = torch.from_numpy(
        (np.asarray(target_features, dtype=np.float32) - eval_arrays["target_mean"])
        / eval_arrays["target_std"]
    ).to(device)
    target_aux_cache = torch.from_numpy(
        (np.asarray(target_aux, dtype=np.float32) - eval_arrays["target_aux_mean"])
        / eval_arrays["target_aux_std"]
    ).to(device)
    deploy_drug_cache = torch.from_numpy(np.asarray(deploy_drug, dtype=np.float32)).to(device)
    deploy_target_cache = torch.from_numpy(
        (np.asarray(deploy_target, dtype=np.float32) - eval_arrays["target_mean"])
        / eval_arrays["target_std"]
    ).to(device)
    deploy_target_aux_cache = torch.from_numpy(
        (np.asarray(deploy_target_aux, dtype=np.float32) - eval_arrays["target_aux_mean"])
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
        coverage_row: dict[str, object] = {}
        if args.coverage_sweep:
            coverage_row = train_epoch(
                model,
                optimizer,
                eval_fit,
                epoch,
                args.seed,
                data,
                drug_features,
                target_features,
                target_aux,
                structure_features,
                structure_mask,
                eval_arrays,
                config,
                device,
                args.batch_size,
                args.coverage_max_rows_per_target,
                args.gradient_clip,
                drug_cache,
                target_cache,
                target_aux_cache,
            )
        balanced_row = train_balanced_epoch(
            model,
            optimizer,
            eval_fit,
            epoch,
            args.seed,
            data,
            drug_features,
            target_features,
            target_aux,
            structure_features,
            structure_mask,
            eval_arrays,
            config,
            device,
            sampling_eval,
            args.gradient_clip,
            drug_cache,
            target_cache,
            target_aux_cache,
        )
        train_row = {
            **{f"coverage_{key}": value for key, value in coverage_row.items()},
            **{f"balanced_{key}": value for key, value in balanced_row.items()},
            "optimizer_steps_this_epoch": int(
                int(coverage_row.get("batches", 0)) + int(balanced_row["batches"])
            ),
        }
        scheduler.step()
        model.eval()
        temporal_dev_metrics, _ = target_to_drug_prediction(
            model,
            temporal_dev_frame,
            temporal_dev_structure,
            temporal_dev_mask,
            temporal_dev_arrays,
            config,
            deploy_drug,
            deploy_target,
            deploy_target_aux,
            device,
            args.inference_batch_size,
            deploy_drug_cache,
            deploy_target_cache,
            deploy_target_aux_cache,
        )
        recovered_dev_metrics, _ = target_to_drug_prediction(
            model,
            recovered_dev_frame,
            recovered_dev_structure,
            recovered_dev_mask,
            recovered_dev_arrays,
            config,
            deploy_drug,
            deploy_target,
            deploy_target_aux,
            device,
            args.inference_batch_size,
            deploy_drug_cache,
            deploy_target_cache,
            deploy_target_aux_cache,
        )
        rec_dev_prediction = predict_positions(
            model,
            positions["rec_dev_warm"],
            data,
            drug_features,
            target_features,
            target_aux,
            structure_features,
            structure_mask,
            eval_arrays,
            config,
            device,
            args.inference_batch_size,
        )
        rec_dev_class = classification_metrics(
            data.iloc[positions["rec_dev_warm"]]["binary_label"].to_numpy(),
            rec_dev_prediction["final_logit"],
        )
        bdb_dev_prediction = predict_positions(
            model,
            positions["bdb_dev"],
            data,
            drug_features,
            target_features,
            target_aux,
            structure_features,
            structure_mask,
            eval_arrays,
            config,
            device,
            args.inference_batch_size,
        )
        bdb_dev_metrics, _ = bdb_fused_metrics(
            data.iloc[positions["bdb_dev"]], bdb_dev_prediction
        )
        selection = (
            0.50 * float(temporal_dev_metrics["mean_top_percentile"])
            + 0.20 * float(temporal_dev_metrics["mean_reciprocal_log_rank"])
            + 0.15 * float(recovered_dev_metrics["mean_top_percentile"])
            + 0.10 * float(rec_dev_class["auprc"])
            + 0.05 * bdb_selection(bdb_dev_metrics)
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selection_value": selection,
            **train_row,
            **{f"dev_temporal_t2d_{key}": value for key, value in temporal_dev_metrics.items()},
            **{f"dev_recovered_t2d_{key}": value for key, value in recovered_dev_metrics.items()},
            **{f"dev_relation_{key}": value for key, value in rec_dev_class.items()},
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

    rec_dev_best = predict_positions(
        model,
        positions["rec_dev_warm"],
        data,
        drug_features,
        target_features,
        target_aux,
        structure_features,
        structure_mask,
        eval_arrays,
        config,
        device,
        args.inference_batch_size,
    )
    temperature = temperature_scale(
        rec_dev_best["final_logit"],
        data.iloc[positions["rec_dev_warm"]]["binary_label"].to_numpy(dtype=np.int8),
    )
    rec_test_prediction = predict_positions(
        model,
        positions["rec_test_warm"],
        data,
        drug_features,
        target_features,
        target_aux,
        structure_features,
        structure_mask,
        eval_arrays,
        config,
        device,
        args.inference_batch_size,
    )
    rec_test_class = classification_metrics(
        data.iloc[positions["rec_test_warm"]]["binary_label"].to_numpy(),
        rec_test_prediction["final_logit"],
        temperature,
    )
    temporal_test_metrics, temporal_test_known = target_to_drug_prediction(
        model,
        temporal_test_frame,
        temporal_test_structure,
        temporal_test_mask,
        temporal_test_arrays,
        config,
        deploy_drug,
        deploy_target,
        deploy_target_aux,
        device,
        args.inference_batch_size,
        deploy_drug_cache,
        deploy_target_cache,
        deploy_target_aux_cache,
    )
    recovered_test_metrics, recovered_test_known = target_to_drug_prediction(
        model,
        recovered_test_frame,
        recovered_test_structure,
        recovered_test_mask,
        recovered_test_arrays,
        config,
        deploy_drug,
        deploy_target,
        deploy_target_aux,
        device,
        args.inference_batch_size,
        deploy_drug_cache,
        deploy_target_cache,
        deploy_target_aux_cache,
    )
    bdb_test_prediction = predict_positions(
        model,
        positions["bdb_test"],
        data,
        drug_features,
        target_features,
        target_aux,
        structure_features,
        structure_mask,
        eval_arrays,
        config,
        device,
        args.inference_batch_size,
    )
    bdb_test_metrics, bdb_test_fused = bdb_fused_metrics(
        data.iloc[positions["bdb_test"]], bdb_test_prediction
    )

    out_root = Path(args.out_dir).resolve()
    run_dir = out_root / f"FULL_FIT_2026_COMPREHENSIVE_BALANCED__seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_checkpoint = run_dir / "EVALUATION_BEST_MODEL_COMPREHENSIVE_BALANCED_V2.pt"
    evaluation_contract = {
        "mode": "STRICT_TEMPORAL_TARGET_TO_720_DRUG_SELECTION",
        "evaluation_training_cutoff_year_inclusive": 2022,
        "selected_epoch": best_epoch,
        "selected_optimizer_steps": int(sum(
            int(row["optimizer_steps_this_epoch"]) for row in history[:best_epoch]
        )),
        "selection_external_current_case_labels_used": False,
        "temporal_test_labels_used_for_selection": False,
        "sampler": sampling_eval.to_dict(),
        "coverage_sweep": args.coverage_sweep,
        "coverage_max_rows_per_target": args.coverage_max_rows_per_target,
        "selection_affinity_weight": AFFINITY_WEIGHT,
    }
    torch.save(
        checkpoint_payload(best_state, config, eval_arrays, temperature, evaluation_contract),
        evaluation_checkpoint,
    )
    pd.DataFrame(history).to_csv(run_dir / "EVALUATION_TRAINING_HISTORY_V2.csv", index=False)
    temporal_test_known.to_csv(run_dir / "TEMPORAL_2024_2025_TARGET_TO_DRUG_RANKS_V2.csv", index=False)
    recovered_test_known.to_csv(run_dir / "RECOVERED_TARGET_TO_DRUG_TEST_RANKS_V2.csv", index=False)
    relation_test = data.iloc[positions["rec_test_warm"]][[
        "calibration_pair_id",
        "target_chembl_id",
        "primary_gene",
        "parent_standard_inchi_key",
        "parent_molecule_name",
        "binary_label",
        "mean_pchembl",
    ]].copy()
    relation_test["raw_logit"] = rec_test_prediction["final_logit"]
    relation_test["calibrated_probability"] = sigmoid(
        rec_test_prediction["final_logit"] / temperature
    )
    relation_test.to_csv(run_dir / "DOUBLE_WARM_RELATION_TEST_PREDICTIONS_V2.csv", index=False)
    bdb_test_frame = data.iloc[positions["bdb_test"]][[
        "calibration_pair_id",
        "target_sequence_hash",
        "parent_standard_inchi_key",
        "mean_pchembl",
        "min_pchembl",
        "max_pchembl",
    ]].copy()
    bdb_test_frame["binary_score"] = bdb_test_prediction["final_logit"]
    bdb_test_frame["affinity_score"] = bdb_test_prediction["affinity"]
    bdb_test_frame["fused_rank_score"] = bdb_test_fused
    bdb_test_frame.to_csv(
        run_dir / "BINDINGDB_TARGET_COLD_TEST_PREDICTIONS_V2.csv.gz",
        index=False,
        compression="gzip",
    )

    del model, target_cache, target_aux_cache, deploy_target_cache, deploy_target_aux_cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(args.seed)
    full_fit = positions["full_fit"]
    full_arrays, full_normalization_audit = balanced_arrays(
        data,
        full_fit,
        target_features,
        target_aux,
        structure_features,
        structure_mask,
        config.target_token_max_len,
    )
    full_target_cache = torch.from_numpy(
        (np.asarray(target_features, dtype=np.float32) - full_arrays["target_mean"])
        / full_arrays["target_std"]
    ).to(device)
    full_target_aux_cache = torch.from_numpy(
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
        coverage_row = {}
        if args.coverage_sweep:
            coverage_row = train_epoch(
                final_model,
                final_optimizer,
                full_fit,
                epoch,
                args.seed,
                data,
                drug_features,
                target_features,
                target_aux,
                structure_features,
                structure_mask,
                full_arrays,
                config,
                device,
                args.batch_size,
                args.coverage_max_rows_per_target,
                args.gradient_clip,
                drug_cache,
                full_target_cache,
                full_target_aux_cache,
            )
        balanced_row = train_balanced_epoch(
            final_model,
            final_optimizer,
            full_fit,
            epoch,
            args.seed,
            data,
            drug_features,
            target_features,
            target_aux,
            structure_features,
            structure_mask,
            full_arrays,
            config,
            device,
            sampling_full,
            args.gradient_clip,
            drug_cache,
            full_target_cache,
            full_target_aux_cache,
        )
        row = {
            **{f"coverage_{key}": value for key, value in coverage_row.items()},
            **{f"balanced_{key}": value for key, value in balanced_row.items()},
            "optimizer_steps_this_epoch": int(
                int(coverage_row.get("batches", 0)) + int(balanced_row["batches"])
            ),
        }
        final_scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": float(final_scheduler.get_last_lr()[0]),
            **row,
        }
        refit_history.append(record)
        print(json.dumps({"seed": args.seed, "refit": True, **record}, default=json_safe), flush=True)
    final_state = {
        name: value.detach().cpu().clone() for name, value in final_model.state_dict().items()
    }
    final_checkpoint = run_dir / "FULL_FIT_MODEL_COMPREHENSIVE_BALANCED_V2.pt"
    full_contract = {
        "protocol": "FULL_FIT_2026_COMPREHENSIVE_BALANCED_V2",
        "training_mode": "FRESH_REFIT_ALL_RELATIONS_BALANCED_SAMPLING",
        "selected_epoch": best_epoch,
        "selected_optimizer_steps": int(sum(
            int(row["optimizer_steps_this_epoch"]) for row in refit_history
        )),
        "all_feature_resolved_relations_eligible": True,
        "per_target_label_cap": None,
        "sampler": sampling_full.to_dict(),
        "coverage_sweep": args.coverage_sweep,
        "coverage_max_rows_per_target": args.coverage_max_rows_per_target,
        "selection_affinity_weight": AFFINITY_WEIGHT,
        "conplex_enabled": False,
        "external_current_new_relation_labels_used": False,
    }
    torch.save(
        checkpoint_payload(final_state, config, full_arrays, 1.0, full_contract),
        final_checkpoint,
    )
    pd.DataFrame(refit_history).to_csv(run_dir / "FULL_FIT_TRAINING_HISTORY_V2.csv", index=False)

    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "FULL_FIT_2026_COMPREHENSIVE_BALANCED_V2",
        "seed": args.seed,
        "device": str(device),
        "new_relation_labels_used_for_training_or_selection": False,
        "selection": {
            "best_epoch": best_epoch,
            "best_optimizer_steps": int(sum(
                int(row["optimizer_steps_this_epoch"]) for row in history[:best_epoch]
            )),
            "epochs_completed": len(history),
            "best_value": best_value,
            "stop_reason": stop_reason,
            "weights": {
                "temporal_target_to_drug_mean_percentile": 0.50,
                "temporal_target_to_drug_reciprocal_log_rank": 0.20,
                "recovered_target_to_drug_mean_percentile": 0.15,
                "warm_relation_auprc": 0.10,
                "bindingdb_target_cold": 0.05,
            },
        },
        "sampler": {
            "evaluation": sampling_eval.to_dict(),
            "full_fit": sampling_full.to_dict(),
            "coverage_sweep": args.coverage_sweep,
            "coverage_max_rows_per_target": args.coverage_max_rows_per_target,
            "evaluation_first_epoch_audit": {
                key: value for key, value in history[0].items()
                if key in {
                    "batches",
                    "rows_emitted",
                    "unique_rows_emitted",
                    "unique_targets_emitted",
                    "binary_positive_fraction",
                    "binary_one_class_target_batch_row_fraction",
                    "binary_target_gini",
                    "binary_effective_targets",
                    "unique_scaffolds_emitted",
                }
                or key.startswith("coverage_")
                or key.startswith("balanced_")
            },
        },
        "splits": {**raw_split_audit, **optimized_split_audit},
        "normalization": {
            "evaluation": eval_normalization_audit,
            "full_fit": full_normalization_audit,
        },
        "structure": {
            **structure_audit,
            "deployment": deployment_structure_audit,
            "normalization_is_unique_target_weighted": True,
            "columns": structure_columns,
        },
        "test_metrics": {
            "strict_temporal_2024_2025_target_to_720_drugs": temporal_test_metrics,
            "recovered_double_warm_target_to_720_drugs": recovered_test_metrics,
            "recovered_double_warm_relation": rec_test_class,
            "bindingdb_exact_target_cold": bdb_test_metrics,
        },
        "model": {
            "parameter_count": int(sum(value.numel() for value in final_model.parameters())),
            "config": config.__dict__,
            "conplex_enabled": False,
        },
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "artifacts": {
            "evaluation_checkpoint": str(evaluation_checkpoint.relative_to(ROOT)),
            "evaluation_checkpoint_sha256": sha256(evaluation_checkpoint),
            "evaluation_state_sha256": state_sha256(best_state),
            "checkpoint": str(final_checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(final_checkpoint),
            "state_sha256": state_sha256(final_state),
        },
        "claim_boundary": (
            "Epoch selection uses only <=2022 fitting data, 2023 temporal development, "
            "frozen recovered development, and BindingDB development metrics. 2024-2025 "
            "temporal positives, frozen recovered test rows, and current external case labels "
            "do not select the epoch. FULL_FIT is a fresh balanced refit on all eligible rows."
        ),
    }
    summary_path = run_dir / "FULL_FIT_RUN_SUMMARY_V2.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), flush=True)


if __name__ == "__main__":
    main()

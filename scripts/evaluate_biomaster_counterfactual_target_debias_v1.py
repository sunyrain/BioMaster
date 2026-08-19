#!/usr/bin/env python3
"""Evaluate label-free counterfactual target marginalization for retrieval.

For each S2 target-cold checkpoint and old drug, estimate its generic drug
prior by averaging the model score over only that checkpoint's fitting-target
homology folds.  Subtracting a validation-selected fraction of this marginal
from the actual pair score suppresses target-invariant drug popularity.  No
external-panel labels participate in fitting or selection.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_biomaster_recent_target_retrieval_heads_v1 import (  # noqa: E402
    SEEDS,
    grouped_metrics,
    load_model,
)
from score_biomaster_recent_novel_targets_v1 import (  # noqa: E402
    DRUG_FEATURES,
    DRUG_INDEX,
    FREEZE,
    TARGET_ESM2,
    TARGET_INDEX,
    TARGET_PROTBERT,
    TRAIN_PAIRS,
    TRAIN_TARGET_ESM2,
    TRAIN_TARGET_INDEX,
    TRAIN_TARGET_PROTBERT,
    build_pair_frame,
    inference_arrays,
)
from train_biomaster_odti_v2 import predict  # noqa: E402


OUT = ROOT / "outputs/biomaster_recent_target_strengthening_v1"
PRIOR_OUT = OUT / "S2_OLD_DRUG_COUNTERFACTUAL_TARGET_PRIORS_V1.csv.gz"
INTERNAL_OUT = OUT / "S6_COUNTERFACTUAL_DEBIAS_INTERNAL_METRICS_V1.csv"
EXTERNAL_OUT = OUT / "RECENT_TARGET_COUNTERFACTUAL_DEBIAS_SCORES_V1.csv.gz"
POSITIVES_OUT = OUT / "RECENT_TARGET_COUNTERFACTUAL_DEBIAS_POSITIVES_V1.csv"
SUMMARY_OUT = OUT / "COUNTERFACTUAL_TARGET_DEBIAS_SUMMARY_V1.json"
ASSIGNMENTS = (
    ROOT / "outputs/old_drug_target_sota_v1/benchmark_splits_v1"
    / "CHEMBL37_86674_FROZEN_SPLIT_ASSIGNMENTS_V1.csv.gz"
)
RUNS = ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s2_esm2_full"
PREFIX = "S2_HOMOLOGY_COLD_TARGET"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_unit(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    work = frame[["target_chembl_id"]].copy()
    work["__score"] = values
    return (
        work.groupby("target_chembl_id", sort=False)["__score"]
        .rank(method="average", pct=True, ascending=True)
        .to_numpy(dtype=np.float64)
    )


def checkpoint(fold: int, seed: int) -> Path:
    return (
        RUNS / f"{PREFIX}__fold_{fold}__seed_{seed}" / "BEST_MODEL_V2.pt"
    )


def reference_target_metadata() -> pd.DataFrame:
    pairs = pd.read_csv(
        TRAIN_PAIRS,
        usecols=[
            "target_feature_index",
            "target_chembl_id",
            "target_assay_family",
            "target_homology_cold_fold",
        ],
        low_memory=False,
    ).drop_duplicates()
    conflict = pairs.groupby("target_feature_index").agg(
        targets=("target_chembl_id", "nunique"),
        families=("target_assay_family", "nunique"),
        folds=("target_homology_cold_fold", "nunique"),
    )
    if (conflict > 1).any().any() or pairs["target_feature_index"].duplicated().any():
        raise RuntimeError("training target metadata is not one row per feature index")
    index = pd.read_csv(TRAIN_TARGET_INDEX, low_memory=False)
    output = index[["target_feature_index"]].merge(
        pairs, on="target_feature_index", how="left", validate="one_to_one"
    )
    if output.isna().any().any() or len(output) != 428:
        raise RuntimeError("reference target metadata is incomplete")
    return output.sort_values("target_feature_index").reset_index(drop=True)


def pair_frame_for_reference(targets: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for target in targets.to_dict("records"):
        chunk = drugs[["drug_feature_index", "ligand_inchikey", "drug_names"]].copy()
        chunk["target_feature_index"] = int(target["target_feature_index"])
        chunk["target_chembl_id"] = str(target["target_chembl_id"])
        chunk["target_assay_family"] = str(target["target_assay_family"])
        chunk["binary_label"] = 0
        chunk["conplex_score"] = 0.0
        chunk["mean_pchembl"] = np.nan
        chunk["min_pchembl"] = np.nan
        chunk["max_pchembl"] = np.nan
        chunks.append(chunk)
    output = pd.concat(chunks, ignore_index=True)
    if len(output) != len(targets) * 720:
        raise RuntimeError("counterfactual reference Cartesian product changed")
    return output


def score_checkpoint(
    model: torch.nn.Module,
    checkpoint_data: dict[str, object],
    frame: pd.DataFrame,
    drug_features: np.ndarray,
    target_features: np.ndarray,
    target_aux: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = inference_arrays(frame, checkpoint_data)
    structure_dim = int(checkpoint_data["config"].get("structure_input_dim", 0))
    structure = np.zeros((len(frame), structure_dim), dtype=np.float32)
    structure_mask = np.zeros(len(frame), dtype=np.float32)
    positions = np.arange(len(frame), dtype=np.int64)
    drug_cache = torch.from_numpy(np.array(drug_features, dtype=np.float32, copy=True)).to(device)
    target_cache = torch.from_numpy(
        (np.array(target_features, dtype=np.float32, copy=True) - arrays["target_mean"])
        / arrays["target_std"]
    ).to(device)
    target_aux_cache = torch.from_numpy(
        (np.array(target_aux, dtype=np.float32, copy=True) - arrays["target_aux_mean"])
        / arrays["target_aux_std"]
    ).to(device)
    output = predict(
        model, positions, frame, drug_features, None, target_features, target_aux,
        None, None, None, None, int(checkpoint_data.get("target_token_max_len", 1022)),
        None, None, structure, structure_mask, arrays, device, 4096,
        drug_feature_cache=drug_cache,
        target_feature_cache=target_cache,
        target_aux_feature_cache=target_aux_cache,
    )
    base = np.asarray(output["base_logit"], dtype=np.float64)
    affinity = (
        np.asarray(output["affinity"], dtype=np.float64)
        * float(checkpoint_data["normalization"]["affinity_std"])
        + float(checkpoint_data["normalization"]["affinity_mean"])
    )
    del drug_cache, target_cache, target_aux_cache, output
    return base, affinity


def build_counterfactual_priors_and_external() -> tuple[pd.DataFrame, pd.DataFrame]:
    freeze = json.loads(FREEZE.read_text())
    drug_index = pd.read_csv(DRUG_INDEX).sort_values("drug_feature_index").reset_index(drop=True)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    reference_meta = reference_target_metadata()
    train_target_features = np.load(TRAIN_TARGET_PROTBERT, mmap_mode="r")
    train_target_aux = np.load(TRAIN_TARGET_ESM2, mmap_mode="r")
    external_target_index = pd.read_csv(TARGET_INDEX)
    external_target_features = np.load(TARGET_PROTBERT, mmap_mode="r")
    external_target_aux = np.load(TARGET_ESM2, mmap_mode="r")
    external = build_pair_frame(freeze, external_target_index, drug_index)
    # Reuse the internal metric helper's generic target grouping column; the
    # external proteins deliberately have no ChEMBL target identifier.
    external["target_chembl_id"] = external["uniprot_accession"].astype(str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prior_rows = []
    external_parts = []
    for fold in range(5):
        valid_fold = (fold + 1) % 5
        reference_targets = reference_meta[
            ~reference_meta["target_homology_cold_fold"].isin([fold, valid_fold])
        ].copy()
        reference = pair_frame_for_reference(reference_targets, drug_index)
        for seed in SEEDS:
            path = checkpoint(fold, int(seed))
            model, checkpoint_data = load_model(path, device)
            reference_base, reference_affinity = score_checkpoint(
                model, checkpoint_data, reference, drug_features,
                train_target_features, train_target_aux, device,
            )
            reference_scored = reference[["drug_feature_index", "ligand_inchikey"]].copy()
            reference_scored["base"] = reference_base
            reference_scored["affinity"] = reference_affinity
            marginal = reference_scored.groupby(
                ["drug_feature_index", "ligand_inchikey"], as_index=False
            ).agg(
                counterfactual_base_mean=("base", "mean"),
                counterfactual_base_std=("base", "std"),
                counterfactual_affinity_mean=("affinity", "mean"),
                counterfactual_affinity_std=("affinity", "std"),
            )
            marginal["fold"] = fold
            marginal["seed"] = int(seed)
            marginal["reference_target_count"] = len(reference_targets)
            prior_rows.append(marginal)
            external_base, external_affinity = score_checkpoint(
                model, checkpoint_data, external, drug_features,
                external_target_features, external_target_aux, device,
            )
            part = external[
                ["uniprot_accession", "target_chembl_id", "drug_feature_index", "ligand_inchikey"]
            ].copy()
            part["fold"] = fold
            part["seed"] = int(seed)
            part["actual_base"] = external_base
            part["actual_affinity"] = external_affinity
            part = part.merge(
                marginal[
                    [
                        "drug_feature_index", "ligand_inchikey",
                        "counterfactual_base_mean", "counterfactual_affinity_mean",
                    ]
                ],
                on=["drug_feature_index", "ligand_inchikey"],
                how="left",
                validate="many_to_one",
            )
            external_parts.append(part)
            del model, reference_base, reference_affinity, external_base, external_affinity
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({
                "fold": fold, "seed": int(seed),
                "reference_targets": len(reference_targets), "device": str(device),
            }), flush=True)
    return pd.concat(prior_rows, ignore_index=True), pd.concat(external_parts, ignore_index=True)


def internal_frames(priors: pd.DataFrame) -> list[pd.DataFrame]:
    drug_map = priors[["drug_feature_index", "ligand_inchikey"]].drop_duplicates()
    frames = []
    for fold in range(5):
        seed_parts = []
        metadata = None
        for seed in SEEDS:
            run = RUNS / f"{PREFIX}__fold_{fold}__seed_{seed}"
            prediction = pd.read_csv(run / "TEST_PREDICTIONS_V2.csv.gz", low_memory=False)
            prediction = prediction.merge(
                drug_map,
                left_on="parent_standard_inchi_key",
                right_on="ligand_inchikey",
                how="inner",
                validate="many_to_one",
            )
            if metadata is None:
                metadata = prediction[
                    [
                        "calibration_pair_id", "target_chembl_id", "drug_feature_index",
                        "ligand_inchikey", "binary_label", "mean_pchembl",
                    ]
                ].copy()
            checkpoint_data = torch.load(
                run / "BEST_MODEL_V2.pt", map_location="cpu", weights_only=False
            )
            actual_affinity = (
                prediction["v2_affinity"].to_numpy(dtype=np.float64)
                * float(checkpoint_data["normalization"]["affinity_std"])
                + float(checkpoint_data["normalization"]["affinity_mean"])
            )
            marginal = priors[(priors["fold"] == fold) & (priors["seed"] == seed)]
            part = prediction[["calibration_pair_id", "drug_feature_index"]].copy()
            part[f"base_{seed}"] = prediction["v2_base_logit"].to_numpy(dtype=np.float64)
            part[f"affinity_{seed}"] = actual_affinity
            part = part.merge(
                marginal[
                    ["drug_feature_index", "counterfactual_base_mean", "counterfactual_affinity_mean"]
                ],
                on="drug_feature_index", how="left", validate="many_to_one",
            ).rename(
                columns={
                    "counterfactual_base_mean": f"base_prior_{seed}",
                    "counterfactual_affinity_mean": f"affinity_prior_{seed}",
                }
            )
            seed_parts.append(part.drop(columns="drug_feature_index"))
        merged = metadata
        for part in seed_parts:
            merged = merged.merge(part, on="calibration_pair_id", how="inner", validate="one_to_one")
        merged["base"] = merged[[f"base_{seed}" for seed in SEEDS]].mean(axis=1)
        merged["base_prior"] = merged[[f"base_prior_{seed}" for seed in SEEDS]].mean(axis=1)
        merged["affinity"] = merged[[f"affinity_{seed}" for seed in SEEDS]].mean(axis=1)
        merged["affinity_prior"] = merged[[f"affinity_prior_{seed}" for seed in SEEDS]].mean(axis=1)
        merged["fold"] = fold
        frames.append(merged)
    return frames


def select_and_evaluate(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, float]]:
    all_folds = pd.concat(frames, ignore_index=True)
    development = all_folds[all_folds["fold"].isin([0, 1, 2])]
    candidates = []
    alphas = np.arange(0.0, 2.01, 0.25)
    for binary_alpha, affinity_alpha, binary_weight_units in itertools.product(
        alphas, alphas, range(5)
    ):
        binary = development["base"].to_numpy() - binary_alpha * development["base_prior"].to_numpy()
        affinity = development["affinity"].to_numpy() - affinity_alpha * development["affinity_prior"].to_numpy()
        binary_rank = rank_unit(development, binary)
        affinity_rank = rank_unit(development, affinity)
        binary_weight = binary_weight_units / 4.0
        score = binary_weight * binary_rank + (1.0 - binary_weight) * affinity_rank
        metric = grouped_metrics(development, score)["target_macro_auprc"]
        candidates.append((float(metric), binary_weight, float(binary_alpha), float(affinity_alpha)))
    candidates.sort(key=lambda row: (row[0], row[1], -row[2], -row[3]), reverse=True)
    objective, binary_weight, binary_alpha, affinity_alpha = candidates[0]
    selected = {
        "binary_weight": binary_weight,
        "affinity_weight": 1.0 - binary_weight,
        "binary_prior_alpha": binary_alpha,
        "affinity_prior_alpha": affinity_alpha,
        "development_target_macro_auprc": objective,
    }
    rows = []
    for partition, folds in [
        ("DEVELOPMENT_FOLDS_0_2", [0, 1, 2]),
        ("HELDOUT_FOLDS_3_4", [3, 4]),
        ("ALL_FOLDS_REPORT_ONLY", [0, 1, 2, 3, 4]),
    ]:
        frame = all_folds[all_folds["fold"].isin(folds)]
        raw_binary = frame["base"].to_numpy()
        debiased_binary = raw_binary - binary_alpha * frame["base_prior"].to_numpy()
        raw_affinity = frame["affinity"].to_numpy()
        debiased_affinity = raw_affinity - affinity_alpha * frame["affinity_prior"].to_numpy()
        scores = {
            "RAW_BINARY": raw_binary,
            "COUNTERFACTUAL_BINARY": debiased_binary,
            "RAW_AFFINITY": raw_affinity,
            "COUNTERFACTUAL_AFFINITY": debiased_affinity,
            "SELECTED_COUNTERFACTUAL_FUSION": (
                binary_weight * rank_unit(frame, debiased_binary)
                + (1.0 - binary_weight) * rank_unit(frame, debiased_affinity)
            ),
        }
        for name, score in scores.items():
            rows.append({"partition": partition, "score": name, **grouped_metrics(frame, score)})
    return pd.DataFrame(rows), selected


def external_scores(
    external_parts: pd.DataFrame,
    selected: dict[str, float],
) -> pd.DataFrame:
    group_columns = ["uniprot_accession", "target_chembl_id", "drug_feature_index", "ligand_inchikey"]
    frame = external_parts.groupby(group_columns, as_index=False).agg(
        actual_base=("actual_base", "mean"),
        base_prior=("counterfactual_base_mean", "mean"),
        actual_affinity=("actual_affinity", "mean"),
        affinity_prior=("counterfactual_affinity_mean", "mean"),
    )
    frame["counterfactual_binary"] = (
        frame["actual_base"] - selected["binary_prior_alpha"] * frame["base_prior"]
    )
    frame["counterfactual_affinity"] = (
        frame["actual_affinity"] - selected["affinity_prior_alpha"] * frame["affinity_prior"]
    )
    frame["binary_rank_unit"] = rank_unit(frame, frame["counterfactual_binary"].to_numpy())
    frame["affinity_rank_unit"] = rank_unit(frame, frame["counterfactual_affinity"].to_numpy())
    frame["counterfactual_fusion"] = (
        selected["binary_weight"] * frame["binary_rank_unit"]
        + selected["affinity_weight"] * frame["affinity_rank_unit"]
    )
    for column in [
        "actual_base", "counterfactual_binary", "actual_affinity",
        "counterfactual_affinity", "counterfactual_fusion",
    ]:
        frame[f"{column}_rank_720"] = (
            frame.groupby("uniprot_accession")[column]
            .rank(method="min", ascending=False)
            .astype(np.int16)
        )
    return frame


def main() -> None:
    required = [
        FREEZE, DRUG_INDEX, DRUG_FEATURES, TARGET_INDEX, TARGET_PROTBERT, TARGET_ESM2,
        TRAIN_TARGET_INDEX, TRAIN_TARGET_PROTBERT, TRAIN_TARGET_ESM2, TRAIN_PAIRS, ASSIGNMENTS,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    priors, external_parts = build_counterfactual_priors_and_external()
    priors.to_csv(PRIOR_OUT, index=False, compression="gzip")
    frames = internal_frames(priors)
    internal, selected = select_and_evaluate(frames)
    internal.to_csv(INTERNAL_OUT, index=False)
    external = external_scores(external_parts, selected)
    freeze = json.loads(FREEZE.read_text())
    target_index = pd.read_csv(TARGET_INDEX)
    drug_index = pd.read_csv(DRUG_INDEX)
    metadata = build_pair_frame(freeze, target_index, drug_index)[
        [
            "uniprot_accession", "drug_feature_index", "gene_symbol", "drug_names",
            "is_literature_experimental_positive",
        ]
    ]
    external = external.merge(
        metadata,
        on=["uniprot_accession", "drug_feature_index"],
        how="left", validate="one_to_one",
    )
    external.to_csv(EXTERNAL_OUT, index=False, compression="gzip")
    positives = external[external["is_literature_experimental_positive"]].copy()
    positive_columns = [
        "uniprot_accession", "gene_symbol", "drug_names", "ligand_inchikey",
        *[column for column in external.columns if column.endswith("_rank_720")],
    ]
    positives[positive_columns].to_csv(POSITIVES_OUT, index=False)
    pivot = external.pivot(
        index="drug_feature_index", columns="uniprot_accession", values="counterfactual_fusion"
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "method": "COUNTERFACTUAL_TARGET_MARGINALIZATION_V1",
        "selection_status": "INTERNAL_S6_DEVELOPMENT_FOLDS_ONLY",
        "selected": selected,
        "internal_metrics": internal.to_dict("records"),
        "external_status": "POST_HOC_DEVELOPMENT_DIAGNOSTIC_NOT_BLIND_VALIDATION",
        "external_positive_ranks": positives[positive_columns].to_dict("records"),
        "external_cross_target_fusion_spearman": float(pivot.corr(method="spearman").iloc[0, 1]),
        "claim_boundary": (
            "The marginal is label-free at scoring time but learned model parameters still use ChEMBL labels. "
            "The two literature controls were not used for selection and remain retrospective weak-binding cases."
        ),
        "artifacts": {
            "priors": rel(PRIOR_OUT),
            "internal_metrics": rel(INTERNAL_OUT),
            "external_scores": rel(EXTERNAL_OUT),
            "external_positive_ranks": rel(POSITIVES_OUT),
        },
    }
    report["artifact_sha256"] = {
        rel(path): sha256(path)
        for path in [PRIOR_OUT, INTERNAL_OUT, EXTERNAL_OUT, POSITIVES_OUT]
    }
    SUMMARY_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "selected": selected,
        "external_positive_ranks": report["external_positive_ranks"],
        "external_cross_target_fusion_spearman": report["external_cross_target_fusion_spearman"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose new-target x old-drug retrieval heads without external-label tuning.

Fusion weights are selected only on S2 out-of-fold predictions restricted to
the S6 old-drug test slice (development folds 0--2).  Folds 3--4 are an
internal untouched evaluation.  The two literature controls are read only
after the internal choice and are explicitly post-hoc development evidence.
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
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2  # noqa: E402
from score_biomaster_recent_novel_targets_v1 import (  # noqa: E402
    DRUG_FEATURES,
    DRUG_INDEX,
    FREEZE,
    TARGET_ESM2,
    TARGET_INDEX,
    TARGET_PROTBERT,
    build_pair_frame,
    inference_arrays,
)
from train_biomaster_odti_v2 import predict  # noqa: E402


OUT = ROOT / "outputs/biomaster_recent_target_strengthening_v1"
SCORES = OUT / "RECENT_TARGET_HEAD_DIAGNOSTIC_SCORES_V1.csv.gz"
POSITIVES = OUT / "RECENT_TARGET_HEAD_DIAGNOSTIC_POSITIVES_V1.csv"
INTERNAL = OUT / "S6_INTERNAL_HEAD_AND_FUSION_METRICS_V1.csv"
SUMMARY = OUT / "RECENT_TARGET_HEAD_DIAGNOSTIC_SUMMARY_V1.json"
ASSIGNMENTS = (
    ROOT / "outputs/old_drug_target_sota_v1/benchmark_splits_v1"
    / "CHEMBL37_86674_FROZEN_SPLIT_ASSIGNMENTS_V1.csv.gz"
)
CONPLEX_MODEL = ROOT / "third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt"

SUITES = {
    "S2_TARGET_COLD_25": {
        "root": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s2_esm2_full",
        "prefix": "S2_HOMOLOGY_COLD_TARGET",
        "folds": range(5),
    },
    "S3_DOUBLE_COLD_25": {
        "root": ROOT / "outputs/odti_unified_champion_s3_20260817",
        "prefix": "S3_STRICT_DOUBLE_COLD",
        "folds": range(5),
    },
    "S5_OLD_DRUG_COLD_5": {
        "root": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s5_esm2_formal",
        "prefix": "S5_OLD_DRUG_ENTITY_COLD",
        "folds": [-1],
    },
}
SEEDS = range(20260816, 20260821)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def rank_unit(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Map a score to [0,1] within each target; larger remains better."""

    return (
        frame.groupby("target_chembl_id", sort=False)[column]
        .rank(method="average", pct=True, ascending=True)
        .to_numpy(dtype=np.float64)
    )


def grouped_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, object]:
    y = frame["binary_label"].to_numpy(dtype=np.int8)
    aps: list[float] = []
    aucs: list[float] = []
    for _, part in frame.assign(__score=score).groupby("target_chembl_id", sort=False):
        labels = part["binary_label"].to_numpy(dtype=np.int8)
        if labels.size < 2 or labels.min() == labels.max():
            continue
        values = part["__score"].to_numpy(dtype=np.float64)
        aps.append(float(average_precision_score(labels, values)))
        aucs.append(float(roc_auc_score(labels, values)))
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "target_groups_with_both_classes": len(aps),
        "target_macro_auprc": float(np.mean(aps)) if aps else None,
        "target_macro_auroc": float(np.mean(aucs)) if aucs else None,
    }
    if y.min() != y.max():
        result["micro_auprc"] = float(average_precision_score(y, score))
        result["micro_auroc"] = float(roc_auc_score(y, score))
    else:
        result["micro_auprc"] = None
        result["micro_auroc"] = None
    return result


def checkpoint_path(suite: dict[str, object], fold: int, seed: int) -> Path:
    run = suite["root"] / f"{suite['prefix']}__fold_{fold}__seed_{seed}"
    path = run / "BEST_MODEL_V2.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(path: Path, device: torch.device) -> tuple[RoutedInteractionRankerV2, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ODTIV2Config(**checkpoint["config"])
    use_conplex = "conplex_weight" in checkpoint["model_state_dict"]
    model = RoutedInteractionRankerV2(
        family_count=len(checkpoint["families"]), config=config, use_conplex=use_conplex
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def conplex_scores(drug: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact SimpleCoembeddingNoSigmoid ReLU + cosine forward pass."""

    state = torch.load(CONPLEX_MODEL, map_location="cpu", weights_only=True)
    drug_tensor = torch.from_numpy(np.asarray(drug, dtype=np.float32))
    target_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))
    drug_projection = torch.relu(
        torch.nn.functional.linear(
            drug_tensor, state["drug_projector.0.weight"], state["drug_projector.0.bias"]
        )
    )
    target_projection = torch.relu(
        torch.nn.functional.linear(
            target_tensor, state["target_projector.0.weight"], state["target_projector.0.bias"]
        )
    )
    drug_projection = torch.nn.functional.normalize(drug_projection, dim=1)
    target_projection = torch.nn.functional.normalize(target_projection, dim=1)
    return (target_projection @ drug_projection.T).numpy().reshape(-1)


def internal_s6_diagnostic() -> tuple[pd.DataFrame, tuple[float, float, float], dict[str, object]]:
    """Select a label-blind-at-external-panel rank fusion on S6 development folds."""

    assignments = pd.read_csv(
        ASSIGNMENTS,
        usecols=["parent_standard_inchi_key", "is_deployment_old_drug"],
        low_memory=False,
    )
    old_keys = set(
        assignments.loc[
            assignments["is_deployment_old_drug"].astype(bool), "parent_standard_inchi_key"
        ].astype(str)
    )
    suite = SUITES["S2_TARGET_COLD_25"]
    fold_frames: list[pd.DataFrame] = []
    for fold in range(5):
        seed_frames = []
        for seed in SEEDS:
            run = suite["root"] / f"{suite['prefix']}__fold_{fold}__seed_{seed}"
            prediction = pd.read_csv(run / "TEST_PREDICTIONS_V2.csv.gz", low_memory=False)
            checkpoint = torch.load(run / "BEST_MODEL_V2.pt", map_location="cpu", weights_only=False)
            affinity = (
                prediction["v2_affinity"].to_numpy(dtype=np.float64)
                * float(checkpoint["normalization"]["affinity_std"])
                + float(checkpoint["normalization"]["affinity_mean"])
            )
            seed_frames.append(
                pd.DataFrame(
                    {
                        "calibration_pair_id": prediction["calibration_pair_id"].astype(str),
                        f"probability_{seed}": prediction["v2_probability_calibrated"],
                        f"affinity_{seed}": affinity,
                    }
                )
            )
            if seed == min(SEEDS):
                metadata = prediction[
                    [
                        "calibration_pair_id",
                        "target_chembl_id",
                        "parent_standard_inchi_key",
                        "binary_label",
                        "mean_pchembl",
                        "conplex_score",
                    ]
                ].copy()
        merged = metadata
        for seed_frame in seed_frames:
            merged = merged.merge(seed_frame, on="calibration_pair_id", how="inner", validate="one_to_one")
        merged = merged[merged["parent_standard_inchi_key"].astype(str).isin(old_keys)].copy()
        merged["probability"] = merged[[f"probability_{seed}" for seed in SEEDS]].mean(axis=1)
        merged["affinity"] = merged[[f"affinity_{seed}" for seed in SEEDS]].mean(axis=1)
        merged["binary_rank_unit"] = rank_unit(merged, "probability")
        merged["affinity_rank_unit"] = rank_unit(merged, "affinity")
        merged["conplex_rank_unit"] = rank_unit(merged, "conplex_score")
        merged["fold"] = fold
        fold_frames.append(merged)
    all_folds = pd.concat(fold_frames, ignore_index=True)
    development = all_folds[all_folds["fold"].isin([0, 1, 2])]
    choices = []
    for units in itertools.product(range(5), repeat=3):
        if sum(units) != 4:
            continue
        weights = tuple(value / 4.0 for value in units)
        score = (
            weights[0] * development["binary_rank_unit"].to_numpy()
            + weights[1] * development["affinity_rank_unit"].to_numpy()
            + weights[2] * development["conplex_rank_unit"].to_numpy()
        )
        value = grouped_metrics(development, score)["target_macro_auprc"]
        choices.append((float(value), weights))
    # Deterministic conservative tie-break: prefer the existing binary head,
    # then affinity, then the external ConPLex component.
    choices.sort(key=lambda item: (item[0], item[1][0], item[1][1]), reverse=True)
    selected_value, selected = choices[0]
    rows: list[dict[str, object]] = []
    for partition, selected_folds in [
        ("DEVELOPMENT_FOLDS_0_2", [0, 1, 2]),
        ("HELDOUT_FOLDS_3_4", [3, 4]),
        ("ALL_FOLDS_REPORT_ONLY", [0, 1, 2, 3, 4]),
    ]:
        frame = all_folds[all_folds["fold"].isin(selected_folds)]
        heads = {
            "BINARY": frame["binary_rank_unit"].to_numpy(),
            "AFFINITY": frame["affinity_rank_unit"].to_numpy(),
            "CONPLEX": frame["conplex_rank_unit"].to_numpy(),
            "INTERNAL_SELECTED_FUSION": (
                selected[0] * frame["binary_rank_unit"].to_numpy()
                + selected[1] * frame["affinity_rank_unit"].to_numpy()
                + selected[2] * frame["conplex_rank_unit"].to_numpy()
            ),
        }
        for name, score in heads.items():
            rows.append({"partition": partition, "head": name, **grouped_metrics(frame, score)})
    metrics_frame = pd.DataFrame(rows)
    selection = {
        "selection_population": "S6 slice of S2 OOF predictions, folds 0-2 only",
        "objective": "target_macro_auprc",
        "grid_step": 0.25,
        "weights": {"binary": selected[0], "affinity": selected[1], "conplex": selected[2]},
        "development_objective_value": selected_value,
        "heldout_folds": [3, 4],
    }
    return metrics_frame, selected, selection


def external_diagnostic(selected: tuple[float, float, float]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    freeze = json.loads(FREEZE.read_text())
    target_index = pd.read_csv(TARGET_INDEX)
    drug_index = pd.read_csv(DRUG_INDEX)
    frame = build_pair_frame(freeze, target_index, drug_index)
    drug_features = np.load(DRUG_FEATURES, mmap_mode="r")
    target_protbert = np.load(TARGET_PROTBERT, mmap_mode="r")
    target_esm2 = np.load(TARGET_ESM2, mmap_mode="r")
    frame["conplex_official"] = conplex_scores(drug_features, target_protbert)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suite_metadata = []
    positions = np.arange(len(frame), dtype=np.int64)
    for name, suite in SUITES.items():
        probabilities: list[np.ndarray] = []
        affinities: list[np.ndarray] = []
        fallback = []
        checkpoints = []
        for fold in suite["folds"]:
            for seed in SEEDS:
                path = checkpoint_path(suite, int(fold), int(seed))
                model, checkpoint = load_model(path, device)
                arrays = inference_arrays(frame, checkpoint)
                structure_dim = int(checkpoint["config"].get("structure_input_dim", 0))
                structure = np.zeros((len(frame), structure_dim), dtype=np.float32)
                structure_mask = np.zeros(len(frame), dtype=np.float32)
                drug_cache = torch.from_numpy(np.asarray(drug_features, dtype=np.float32)).to(device)
                target_cache = torch.from_numpy(
                    (np.asarray(target_protbert, dtype=np.float32) - arrays["target_mean"])
                    / arrays["target_std"]
                ).to(device)
                target_aux_cache = torch.from_numpy(
                    (np.asarray(target_esm2, dtype=np.float32) - arrays["target_aux_mean"])
                    / arrays["target_aux_std"]
                ).to(device)
                output = predict(
                    model, positions, frame, drug_features, None, target_protbert, target_esm2,
                    None, None, None, None, int(checkpoint.get("target_token_max_len", 1022)),
                    None, None, structure, structure_mask, arrays, device, 4096,
                    drug_feature_cache=drug_cache,
                    target_feature_cache=target_cache,
                    target_aux_feature_cache=target_aux_cache,
                )
                final = np.asarray(output["final_logit"], dtype=np.float64)
                base = np.asarray(output["base_logit"], dtype=np.float64)
                probability = 1.0 / (
                    1.0 + np.exp(-np.clip(final / float(checkpoint["temperature"]), -60.0, 60.0))
                )
                affinity = (
                    np.asarray(output["affinity"], dtype=np.float64)
                    * float(checkpoint["normalization"]["affinity_std"])
                    + float(checkpoint["normalization"]["affinity_mean"])
                )
                probabilities.append(probability)
                affinities.append(affinity)
                fallback.append(float(np.max(np.abs(final - base))))
                checkpoints.append(rel(path))
                del model, drug_cache, target_cache, target_aux_cache, output
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        frame[f"{name}__binary"] = np.mean(np.stack(probabilities), axis=0)
        frame[f"{name}__affinity_pchembl"] = np.mean(np.stack(affinities), axis=0)
        frame[f"{name}__binary_rank_unit"] = (
            frame.groupby("uniprot_accession")[f"{name}__binary"]
            .rank(method="average", pct=True, ascending=True)
        )
        frame[f"{name}__affinity_rank_unit"] = (
            frame.groupby("uniprot_accession")[f"{name}__affinity_pchembl"]
            .rank(method="average", pct=True, ascending=True)
        )
        suite_metadata.append(
            {
                "suite": name,
                "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints,
                "max_abs_final_minus_base_logit": max(fallback),
            }
        )
    frame["conplex_rank_unit"] = (
        frame.groupby("uniprot_accession")["conplex_official"]
        .rank(method="average", pct=True, ascending=True)
    )
    # The selected fusion is transferred unchanged from internal S6.  S2 is
    # the only existing suite whose OOF predictions support that selection.
    frame["internal_selected_fusion"] = (
        selected[0] * frame["S2_TARGET_COLD_25__binary_rank_unit"]
        + selected[1] * frame["S2_TARGET_COLD_25__affinity_rank_unit"]
        + selected[2] * frame["conplex_rank_unit"]
    )
    score_columns = [
        "conplex_official",
        "internal_selected_fusion",
        *[column for column in frame.columns if "__binary" in column or "__affinity" in column],
    ]
    for column in score_columns:
        frame[f"{column}__rank_720"] = (
            frame.groupby("uniprot_accession")[column]
            .rank(method="min", ascending=False)
            .astype(np.int16)
        )
    return frame, suite_metadata


def main() -> None:
    required = [
        FREEZE, TARGET_INDEX, TARGET_PROTBERT, TARGET_ESM2, DRUG_INDEX,
        DRUG_FEATURES, ASSIGNMENTS, CONPLEX_MODEL,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)
    internal_metrics, selected, selection = internal_s6_diagnostic()
    internal_metrics.to_csv(INTERNAL, index=False)
    frame, suite_metadata = external_diagnostic(selected)
    frame.to_csv(SCORES, index=False, compression="gzip")
    positives = frame[frame["is_literature_experimental_positive"]].copy()
    rank_columns = [column for column in frame.columns if column.endswith("__rank_720")]
    positive_columns = [
        "uniprot_accession", "gene_symbol", "drug_names", "ligand_inchikey",
        "experimental_positive_old_drug_name", *rank_columns,
    ]
    positives[positive_columns].to_csv(POSITIVES, index=False)
    target_correlations = {}
    for name in SUITES:
        pivot = frame.pivot(
            index="drug_feature_index", columns="uniprot_accession", values=f"{name}__binary"
        )
        target_correlations[name] = float(pivot.corr(method="spearman").iloc[0, 1])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "analysis_status": "POST_HOC_DEVELOPMENT_DIAGNOSTIC_NOT_BLIND_EXTERNAL_VALIDATION",
        "internal_selection": selection,
        "internal_metrics": internal_metrics.to_dict("records"),
        "external_positive_ranks": positives[positive_columns].to_dict("records"),
        "cross_target_binary_rank_spearman": target_correlations,
        "checkpoint_suites": suite_metadata,
        "interpretation_boundary": [
            "Binary heads target the frozen strong-activity contract (positive mean pChEMBL >= 6).",
            "Affinity heads rank predicted pChEMBL and are exploratory for weak micromolar binding.",
            "The literature Kd values map to about pKd 5.04 and 4.77, so neither pair is a clean strong-activity positive under the training contract.",
            "Unknown target-drug pairs are not verified negatives; ranks are retrieval hypotheses.",
        ],
        "artifacts": {
            "internal_metrics": rel(INTERNAL),
            "external_scores": rel(SCORES),
            "external_positive_ranks": rel(POSITIVES),
        },
    }
    report["artifact_sha256"] = {
        rel(path): sha256(path) for path in [INTERNAL, SCORES, POSITIVES]
    }
    SUMMARY.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "selected_weights": selection["weights"],
        "external_positive_ranks": report["external_positive_ranks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

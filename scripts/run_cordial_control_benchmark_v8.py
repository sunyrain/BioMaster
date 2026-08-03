#!/usr/bin/env python3
"""Benchmark official CORDIAL weights on the project GNINA control poses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CORDIAL = ROOT / ".external/CORDIAL"
BASE = ROOT / "outputs/affinity_first_remote_discovery_v1"
DEFAULT_OUT = (
    ROOT
    / "outputs/affinity_experiment_package_v8/cordial_control_benchmark"
)
CONTROL_SCORES = (
    BASE
    / "gnina_target_calibration_v2/GNINA_TARGET_CALIBRATION_LIGAND_SCORES_V2.csv.gz"
)
CONTROL_POSE_DIR = BASE / "gnina_target_calibration_v2/targets"
RECEPTORS = (
    BASE
    / "experimental_holo_validation_v2/TARGET_DOCKING_RECEPTOR_SELECTION_463_V2.csv"
)
TARGET_CALIBRATION = (
    ROOT
    / "outputs/affinity_experiment_package_v8/target_calibration/"
    "GNINA_TARGET_CHANNEL_CALIBRATION_V8.csv"
)
WEIGHTS = (
    CORDIAL
    / "weights/full.cordial.v2b.conv1d-k7c4-k3c1-nomix.attn-row_ah2-col_ah1-ff4-2x."
    "mlp-256-256-mishx2.1-9-1.bcel-lte.model"
)
NORMALIZATION = CORDIAL / "resources/normalization/full.train.norm.pkl"
RDLogger.DisableLog("rdApp.*")


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def select_targets(calibration: pd.DataFrame, limit: int) -> list[str]:
    admitted = calibration[calibration["target_admitted_v8"].astype(bool)].copy()
    tier_order = {
        "T1_dual_strong": 0,
        "T2_dual_pass": 1,
        "T3_single_strong": 2,
        "T4_single_pass": 3,
    }
    admitted["tier_order"] = admitted["target_admission_tier_v8"].map(tier_order)
    admitted["receptor_group"] = np.where(
        admitted["docking_receptor_source"].eq("experimental_holo"),
        "experimental_holo",
        "alphafold_fallback",
    )
    admitted = admitted.sort_values(
        ["tier_order", "receptor_group", "target_assay_family", "primary_gene"],
        kind="mergesort",
    )
    if limit <= 0 or limit >= len(admitted):
        return admitted["sequence_key"].tolist()
    strata = [
        (tier, source)
        for tier in tier_order
        for source in ["experimental_holo", "alphafold_fallback"]
        if (
            admitted["target_admission_tier_v8"].eq(tier)
            & admitted["receptor_group"].eq(source)
        ).any()
    ]
    quota = max(1, limit // len(strata))
    selected: list[str] = []
    for tier, source in strata:
        group = admitted[
            admitted["target_admission_tier_v8"].eq(tier)
            & admitted["receptor_group"].eq(source)
        ]
        family_first = group.drop_duplicates("target_assay_family", keep="first")
        ordered = pd.concat(
            [
                family_first,
                group[~group["sequence_key"].isin(family_first["sequence_key"])],
            ],
            ignore_index=True,
        )
        selected.extend(ordered.head(quota)["sequence_key"].tolist())
    for key in admitted["sequence_key"]:
        if len(selected) >= limit:
            break
        if key not in selected:
            selected.append(key)
    return selected[:limit]


def best_control_poses(
    sequence_key: str, requested_pairs: set[str]
) -> dict[str, Chem.Mol]:
    source = CONTROL_POSE_DIR / sequence_key / "docked.sdf"
    if not source.exists():
        return {}
    best: dict[str, tuple[float, float, Chem.Mol]] = {}
    supplier = Chem.SDMolSupplier(
        str(source), sanitize=False, removeHs=False, strictParsing=False
    )
    for molecule in supplier:
        if molecule is None:
            continue
        pair_id = clean(
            molecule.GetProp("control_pair_id")
            if molecule.HasProp("control_pair_id")
            else molecule.GetProp("_Name")
            if molecule.HasProp("_Name")
            else ""
        )
        if pair_id not in requested_pairs:
            continue
        cnn = (
            float(molecule.GetProp("CNNaffinity"))
            if molecule.HasProp("CNNaffinity")
            else -np.inf
        )
        vina = (
            -float(molecule.GetProp("minimizedAffinity"))
            if molecule.HasProp("minimizedAffinity")
            else -np.inf
        )
        prior = best.get(pair_id)
        if prior is None or (cnn, vina) > (prior[0], prior[1]):
            best[pair_id] = (cnn, vina, molecule)
    return {key: value[2] for key, value in best.items()}


def prepare_inputs(
    output_dir: Path, limit_targets: int
) -> pd.DataFrame:
    controls = pd.read_csv(CONTROL_SCORES, low_memory=False)
    receptors = pd.read_csv(RECEPTORS, low_memory=False).drop_duplicates("sequence_key")
    calibration = pd.read_csv(TARGET_CALIBRATION, low_memory=False)
    target_keys = select_targets(calibration, limit_targets)
    controls = controls[controls["sequence_key"].isin(target_keys)].copy()
    controls = controls.sort_values(
        ["sequence_key", "control_class", "control_pair_id"], kind="mergesort"
    )
    receptor_paths = receptors.set_index("sequence_key")[
        "docking_receptor_path"
    ].to_dict()
    receptor_sources = receptors.set_index("sequence_key")[
        "docking_receptor_source"
    ].to_dict()

    ligand_dir = output_dir / "ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sequence_key, group in controls.groupby("sequence_key", sort=True):
        pair_ids = set(group["control_pair_id"].astype(str))
        poses = best_control_poses(sequence_key, pair_ids)
        protein_path = Path(clean(receptor_paths.get(sequence_key))).resolve()
        if not protein_path.exists():
            continue
        for _, row in group.iterrows():
            pair_id = clean(row["control_pair_id"])
            molecule = poses.get(pair_id)
            if molecule is None:
                continue
            ligand_path = ligand_dir / f"{pair_id}.sdf"
            writer = Chem.SDWriter(str(ligand_path))
            molecule.SetProp("_Name", pair_id)
            writer.write(molecule)
            writer.close()
            rows.append(
                {
                    "cordial_input_index_v8": len(rows),
                    "control_pair_id": pair_id,
                    "sequence_key": sequence_key,
                    "primary_gene": row["primary_gene"],
                    "docking_receptor_source": clean(
                        receptor_sources.get(sequence_key)
                    ),
                    "control_class": row["control_class"],
                    "best_cnn_affinity": row["best_cnn_affinity"],
                    "score_vina_directional": row["score_vina_directional"],
                    "ligand_path": str(ligand_path.resolve()),
                    "protein_path": str(protein_path),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "CORDIAL_CONTROL_INPUT_MANIFEST_V8.csv"
    manifest.to_csv(manifest_path, index=False)
    pair_file = output_dir / "cordial_pairs.csv"
    pair_file.write_text(
        "".join(
            f"{row.ligand_path};{row.protein_path}\n"
            for row in manifest.itertuples(index=False)
        ),
        encoding="utf-8",
    )
    return manifest


def parse_probability_vector(value: Any) -> list[float]:
    text = clean(value).strip("[]")
    values = [float(token) for token in text.split()]
    if len(values) != 8:
        raise ValueError(f"Expected 8 CORDIAL thresholds, got {len(values)}")
    return values


def evaluate(
    manifest: pd.DataFrame, prediction_path: Path, output_dir: Path
) -> dict[str, Any]:
    predictions = pd.read_csv(prediction_path)
    predictions["cordial_input_index_v8"] = pd.to_numeric(
        predictions["Original_Index"], errors="raise"
    ).astype(int)
    vectors = predictions["Predicted_Probabilities"].map(parse_probability_vector)
    for threshold in range(1, 9):
        predictions[f"cordial_probability_pchembl_ge_{threshold}_v8"] = vectors.map(
            lambda values, index=threshold - 1: values[index]
        )
    probability_columns = [
        f"cordial_probability_pchembl_ge_{threshold}_v8"
        for threshold in range(1, 9)
    ]
    predictions["cordial_expected_ordinal_score_v8"] = predictions[
        probability_columns
    ].sum(axis=1)
    merged = manifest.merge(
        predictions[
            ["cordial_input_index_v8", "cordial_expected_ordinal_score_v8"]
            + probability_columns
        ],
        on="cordial_input_index_v8",
        how="inner",
        validate="one_to_one",
    )
    merged.to_csv(output_dir / "CORDIAL_CONTROL_PREDICTIONS_V8.csv", index=False)

    metrics = []
    for sequence_key, group in merged.groupby("sequence_key", sort=True):
        positive = int(group["control_class"].eq("positive").sum())
        negative = int(group["control_class"].eq("negative").sum())
        if positive < 8 or negative < 8:
            continue
        labels = group["control_class"].eq("positive").astype(int)
        baseline = positive / (positive + negative)
        row = {
            "sequence_key": sequence_key,
            "primary_gene": group["primary_gene"].iloc[0],
            "docking_receptor_source": group["docking_receptor_source"].iloc[0],
            "positive_controls": positive,
            "negative_controls": negative,
            "class_prevalence": baseline,
        }
        for name, column in [
            ("cordial", "cordial_expected_ordinal_score_v8"),
            ("cnn_affinity", "best_cnn_affinity"),
            ("vina_affinity", "score_vina_directional"),
        ]:
            scores = pd.to_numeric(group[column], errors="coerce")
            row[f"auroc_{name}_v8"] = float(roc_auc_score(labels, scores))
            row[f"average_precision_{name}_v8"] = float(
                average_precision_score(labels, scores)
            )
        metrics.append(row)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(output_dir / "CORDIAL_TARGET_METRICS_V8.csv", index=False)
    source_metrics = {}
    for source, group in metric_frame.groupby("docking_receptor_source", sort=True):
        source_metrics[source] = {
            "targets": int(len(group)),
            "cordial_median_auroc": float(group["auroc_cordial_v8"].median()),
            "cnn_median_auroc": float(group["auroc_cnn_affinity_v8"].median()),
            "vina_median_auroc": float(group["auroc_vina_affinity_v8"].median()),
        }
    return {
        "control_predictions": int(len(merged)),
        "targets_evaluable": int(len(metric_frame)),
        "cordial_median_auroc": float(metric_frame["auroc_cordial_v8"].median()),
        "cnn_median_auroc_same_subset": float(
            metric_frame["auroc_cnn_affinity_v8"].median()
        ),
        "vina_median_auroc_same_subset": float(
            metric_frame["auroc_vina_affinity_v8"].median()
        ),
        "cordial_auroc_ge_0_65_targets": int(
            metric_frame["auroc_cordial_v8"].ge(0.65).sum()
        ),
        "cordial_beats_both_gnina_targets": int(
            (
                metric_frame["auroc_cordial_v8"]
                > metric_frame[["auroc_cnn_affinity_v8", "auroc_vina_affinity_v8"]].max(
                    axis=1
                )
            ).sum()
        ),
        "metrics_by_receptor_source": source_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--limit-targets", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists() or not NORMALIZATION.exists():
        raise FileNotFoundError("Official CORDIAL weights or normalization data are missing")

    started = time.time()
    manifest = prepare_inputs(output_dir, args.limit_targets)
    if args.prepare_only:
        print(json.dumps({"prepared_controls": len(manifest)}, indent=2))
        return
    command = [
        "python",
        str(CORDIAL / "run_protocols.py"),
        "--inference",
        "--device",
        str(args.device),
        "--batch_size",
        "16",
        "--num_workers",
        "0",
        "--num_feature_computation_workers",
        "1",
        "--input_ligand_protein_pair_file",
        str(output_dir / "cordial_pairs.csv"),
        "--load_model",
        str(WEIGHTS),
        "--load_normalization_data_pkl",
        str(NORMALIZATION),
        "--cache_dir",
        str(output_dir / "cache"),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(CORDIAL) + ":" + environment.get("PYTHONPATH", "")
    )
    log_path = output_dir / "cordial_inference.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"CORDIAL failed; inspect {log_path}")
    prediction_path = output_dir / "inference_results_predictions.csv"
    metrics = evaluate(manifest, prediction_path, output_dir)
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": time.time() - started,
        "requested_target_limit": int(args.limit_targets),
        "prepared_controls": int(len(manifest)),
        "device": str(args.device),
        "score_definition": (
            "sum of eight official probabilities for pChEMBL thresholds >=1..8; "
            "frozen before target-level evaluation"
        ),
        **metrics,
    }
    summary_path = output_dir / "CORDIAL_CONTROL_BENCHMARK_V8_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

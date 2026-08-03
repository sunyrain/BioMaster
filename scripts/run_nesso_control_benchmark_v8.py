#!/usr/bin/env python3
"""Benchmark official Nesso-1 on the same 20-target controls used for CORDIAL."""

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
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/affinity_experiment_package_v8"
DEFAULT_OUT = BASE / "nesso_control_benchmark_stratified20"
REFERENCE = BASE / "cordial_control_benchmark_stratified20/CORDIAL_CONTROL_INPUT_MANIFEST_V8.csv"
CONTROL_MANIFEST = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/target_docking_calibration_v2/"
    "GNINA_TARGET_CALIBRATION_CONTROLS_V2.csv.gz"
)
UNIVERSE = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/"
    "PHYSICAL_PAIR_UNIVERSE_334749_HOMOLOGY_AUDITED_V1.csv.gz"
)
NESSO = ROOT / ".venvs/nesso/bin/nesso"
CACHE = ROOT / ".cache/nesso"


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def prepare_inputs(output_dir: Path) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE, low_memory=False)
    controls = pd.read_csv(
        CONTROL_MANIFEST,
        usecols=[
            "control_pair_id",
            "canonical_control_smiles",
            "max_pchembl",
            "min_pchembl",
        ],
        low_memory=False,
    ).drop_duplicates("control_pair_id")
    sequences = pd.read_csv(
        UNIVERSE, usecols=["sequence_key", "sequence"], low_memory=False
    ).drop_duplicates("sequence_key")
    manifest = (
        reference.merge(
            controls,
            on="control_pair_id",
            how="left",
            validate="one_to_one",
        )
        .merge(sequences, on="sequence_key", how="left", validate="many_to_one")
        .copy()
    )
    required = [
        "control_pair_id",
        "sequence_key",
        "control_class",
        "canonical_control_smiles",
        "sequence",
    ]
    if manifest[required].fillna("").astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Nesso benchmark manifest has missing required values")
    if manifest["control_pair_id"].duplicated().any():
        raise ValueError("Nesso benchmark control IDs are not unique")

    molecules = manifest["canonical_control_smiles"].map(Chem.MolFromSmiles)
    if molecules.isna().any():
        raise ValueError("Nesso benchmark contains invalid control SMILES")
    manifest["heavy_atom_count_v8"] = molecules.map(lambda mol: mol.GetNumHeavyAtoms())
    manifest["molecular_weight_v8"] = molecules.map(Descriptors.MolWt)
    manifest["rotatable_bonds_v8"] = molecules.map(Lipinski.NumRotatableBonds)

    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    yaml_paths: list[str] = []
    for row in manifest.itertuples(index=False):
        record_id = clean(row.control_pair_id)
        path = input_dir / f"{record_id}.yaml"
        payload = {
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": clean(row.sequence),
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": clean(row.canonical_control_smiles),
                    }
                },
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        yaml_paths.append(str(path.resolve()))
    manifest["nesso_input_yaml_v8"] = yaml_paths
    manifest.to_csv(output_dir / "NESSO_CONTROL_INPUT_MANIFEST_V8.csv", index=False)
    return manifest


def run_nesso(
    output_dir: Path,
    gpu: int,
    workers: int,
    override: bool,
) -> None:
    command = [
        str(NESSO),
        "predict",
        str((output_dir / "inputs").resolve()),
        "--out_dir",
        str(output_dir.resolve()),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--num_workers",
        str(workers),
        "--precision",
        "bf16-mixed",
        "--recycling_steps",
        "5",
        "--no_kernels",
        "--require_affinity",
    ]
    if override:
        command.append("--override")
    environment = os.environ.copy()
    environment["NESSO_CACHE"] = str(CACHE)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = output_dir / "nesso_inference.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Nesso failed; inspect {log_path}")


def collect_predictions(manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        pair_id = clean(row.control_pair_id)
        path = output_dir / "predictions" / pair_id / "affinity.json"
        if not path.exists():
            rows.append(
                {
                    "control_pair_id": pair_id,
                    "nesso_prediction_status_v8": "missing",
                }
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "control_pair_id": pair_id,
                    "nesso_prediction_status_v8": "completed",
                    **{f"nesso_{key}_v8": value for key, value in payload.items()},
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "control_pair_id": pair_id,
                    "nesso_prediction_status_v8": (
                        f"parse_error:{type(exc).__name__}:{exc}"
                    ),
                }
            )
    predictions = pd.DataFrame(rows)
    merged = manifest.merge(
        predictions,
        on="control_pair_id",
        how="left",
        validate="one_to_one",
    )
    merged["nesso_affinity_directional_v8"] = -pd.to_numeric(
        merged["nesso_affinity_pred_value_v8"], errors="coerce"
    )
    merged.to_csv(output_dir / "NESSO_CONTROL_PREDICTIONS_V8.csv", index=False)
    return merged


def evaluate(predictions: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    channels = [
        ("nesso_binder_probability", "nesso_affinity_probability_binary_v8"),
        ("nesso_affinity_directional", "nesso_affinity_directional_v8"),
        ("gnina_cnn_affinity", "best_cnn_affinity"),
        ("gnina_vina_affinity", "score_vina_directional"),
        ("heavy_atom_count", "heavy_atom_count_v8"),
        ("molecular_weight", "molecular_weight_v8"),
        ("rotatable_bonds", "rotatable_bonds_v8"),
    ]
    for sequence_key, group in predictions.groupby("sequence_key", sort=True):
        positive = int(group["control_class"].eq("positive").sum())
        negative = int(group["control_class"].eq("negative").sum())
        if positive < 8 or negative < 8:
            continue
        labels = group["control_class"].eq("positive").astype(int)
        row: dict[str, Any] = {
            "sequence_key": sequence_key,
            "primary_gene": group["primary_gene"].iloc[0],
            "docking_receptor_source": group["docking_receptor_source"].iloc[0],
            "positive_controls": positive,
            "negative_controls": negative,
            "class_prevalence": positive / (positive + negative),
        }
        for name, column in channels:
            scores = pd.to_numeric(group[column], errors="coerce")
            valid = scores.notna()
            if valid.sum() < 16 or labels[valid].nunique() != 2:
                row[f"auroc_{name}_v8"] = np.nan
                row[f"average_precision_{name}_v8"] = np.nan
                continue
            row[f"auroc_{name}_v8"] = float(
                roc_auc_score(labels[valid], scores[valid])
            )
            row[f"average_precision_{name}_v8"] = float(
                average_precision_score(labels[valid], scores[valid])
            )
        row["spearman_nesso_binder_vs_heavy_atom_v8"] = float(
            pd.to_numeric(group["nesso_affinity_probability_binary_v8"], errors="coerce").corr(
                pd.to_numeric(group["heavy_atom_count_v8"], errors="coerce"), method="spearman"
            )
        )
        metrics.append(row)
    frame = pd.DataFrame(metrics)
    frame.to_csv(output_dir / "NESSO_TARGET_METRICS_V8.csv", index=False)

    nesso_best = frame[
        [
            "auroc_nesso_binder_probability_v8",
            "auroc_nesso_affinity_directional_v8",
        ]
    ].max(axis=1)
    gnina_best = frame[
        [
            "auroc_gnina_cnn_affinity_v8",
            "auroc_gnina_vina_affinity_v8",
        ]
    ].max(axis=1)
    size_best = frame[
        [
            "auroc_heavy_atom_count_v8",
            "auroc_molecular_weight_v8",
        ]
    ].max(axis=1)
    source_metrics: dict[str, Any] = {}
    for source, group in frame.groupby("docking_receptor_source", sort=True):
        source_metrics[source] = {
            "targets": int(len(group)),
            "nesso_binder_median_auroc": float(
                group["auroc_nesso_binder_probability_v8"].median()
            ),
            "nesso_affinity_median_auroc": float(
                group["auroc_nesso_affinity_directional_v8"].median()
            ),
            "gnina_cnn_median_auroc": float(
                group["auroc_gnina_cnn_affinity_v8"].median()
            ),
            "gnina_vina_median_auroc": float(
                group["auroc_gnina_vina_affinity_v8"].median()
            ),
            "heavy_atom_median_auroc": float(
                group["auroc_heavy_atom_count_v8"].median()
            ),
            "molecular_weight_median_auroc": float(
                group["auroc_molecular_weight_v8"].median()
            ),
        }
    return {
        "control_predictions": int(
            predictions["nesso_prediction_status_v8"].eq("completed").sum()
        ),
        "targets_evaluable": int(len(frame)),
        "nesso_binder_probability_median_auroc": float(
            frame["auroc_nesso_binder_probability_v8"].median()
        ),
        "nesso_affinity_directional_median_auroc": float(
            frame["auroc_nesso_affinity_directional_v8"].median()
        ),
        "cnn_median_auroc_same_subset": float(
            frame["auroc_gnina_cnn_affinity_v8"].median()
        ),
        "vina_median_auroc_same_subset": float(
            frame["auroc_gnina_vina_affinity_v8"].median()
        ),
        "heavy_atom_count_median_auroc": float(
            frame["auroc_heavy_atom_count_v8"].median()
        ),
        "molecular_weight_median_auroc": float(
            frame["auroc_molecular_weight_v8"].median()
        ),
        "nesso_binder_vs_heavy_atom_median_spearman": float(
            frame["spearman_nesso_binder_vs_heavy_atom_v8"].median()
        ),
        "nesso_any_channel_auroc_ge_0_65_targets": int(nesso_best.ge(0.65).sum()),
        "nesso_best_channel_beats_both_gnina_targets": int(
            (nesso_best > gnina_best).sum()
        ),
        "nesso_best_channel_beats_size_baseline_targets": int(
            (nesso_best > size_best).sum()
        ),
        "metrics_by_receptor_source": source_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    if not NESSO.exists():
        raise FileNotFoundError(NESSO)
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = prepare_inputs(output_dir)
    if args.prepare_only:
        print(json.dumps({"prepared_controls": len(manifest)}, indent=2))
        return
    if not args.evaluate_only:
        run_nesso(output_dir, args.gpu, args.workers, args.override)
    predictions = collect_predictions(manifest, output_dir)
    metrics = evaluate(predictions, output_dir)
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": time.time() - started,
        "requested_controls": int(len(manifest)),
        "requested_targets": int(manifest["sequence_key"].nunique()),
        "model": "Nesso-1 v1.0.0 official code and weights",
        "primary_channel": "affinity_probability_binary",
        "secondary_channel": (
            "-affinity_pred_value, where affinity_pred_value is "
            "log10(IC50/uM) and lower is stronger"
        ),
        "interpretation": (
            "Local historical-control discrimination only; training overlap is "
            "unknown and outputs are not prospective binder probabilities."
        ),
        **metrics,
    }
    summary_path = output_dir / "NESSO_CONTROL_BENCHMARK_V8_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

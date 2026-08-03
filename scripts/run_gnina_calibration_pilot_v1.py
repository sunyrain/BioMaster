#!/usr/bin/env python3
"""Run a five-target GNINA positive/negative calibration pilot on two GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/affinity_first_remote_discovery_v1"
GNINA = Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2")
CUDNN = ROOT / ".conda_envs/boltz2/lib/python3.11/site-packages/nvidia/cudnn/lib"


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def make_3d_molecule(smiles: str, name: str, properties: dict[str, Any], seed: int) -> Chem.Mol | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxIterations = 1000
    status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=300)
    except Exception:
        pass
    molecule = Chem.RemoveHs(molecule)
    molecule.SetProp("_Name", name)
    for key, value in properties.items():
        molecule.SetProp(key, str(value))
    return molecule


def prepare_target_job(group: pd.DataFrame, target: pd.Series, directory: Path) -> dict[str, Any]:
    sequence_key = clean(target["sequence_key"])
    job_dir = directory / sequence_key
    job_dir.mkdir(parents=True, exist_ok=True)
    receptor = job_dir / "receptor.pdbqt"
    receptor_source = clean(target["pdb_path"])
    receptor_log = job_dir / "receptor_prepare.log"
    with receptor_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            ["obabel", receptor_source, "-O", str(receptor), "-xr"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not receptor.exists() or receptor.stat().st_size == 0:
        raise RuntimeError(f"Receptor preparation failed for {sequence_key}")

    ligand_input = job_dir / "controls.sdf"
    writer = Chem.SDWriter(str(ligand_input))
    prepared = 0
    for index, row in group.iterrows():
        molecule = make_3d_molecule(
            clean(row["canonical_control_smiles"]),
            clean(row["control_pair_id"]),
            {
                "control_class": clean(row["control_class"]),
                "sequence_key": sequence_key,
                "target_control_rank": row["target_control_rank"],
            },
            20260720 + int(index) % 100000,
        )
        if molecule is not None:
            writer.write(molecule)
            prepared += 1
    writer.close()
    if prepared < 6:
        raise RuntimeError(f"Only {prepared} controls prepared for {sequence_key}")
    return {
        "sequence_key": sequence_key,
        "primary_gene": clean(target["primary_gene"]),
        "target_assay_family": clean(target["target_assay_family"]),
        "receptor": str(receptor),
        "ligand_input": str(ligand_input),
        "output": str(job_dir / "docked.sdf"),
        "log": str(job_dir / "gnina.log"),
        "stdout": str(job_dir / "gnina.stdout"),
        "center_x": float(target["top_pocket_center_x"]),
        "center_y": float(target["top_pocket_center_y"]),
        "center_z": float(target["top_pocket_center_z"]),
        "prepared_controls": prepared,
    }


def run_job(job: dict[str, Any], gpu: int, exhaustiveness: int, modes: int) -> dict[str, Any]:
    command = [
        str(GNINA),
        "-r", job["receptor"],
        "-l", job["ligand_input"],
        "-o", job["output"],
        "--log", job["log"],
        "--center_x", str(job["center_x"]),
        "--center_y", str(job["center_y"]),
        "--center_z", str(job["center_z"]),
        "--size_x", "22", "--size_y", "22", "--size_z", "22",
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(modes),
        "--cnn_scoring", "rescore",
        "--seed", str(20260720 + gpu),
        "--device", "0",
        "--cpu", "6",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["LD_LIBRARY_PATH"] = str(CUDNN) + ":" + environment.get("LD_LIBRARY_PATH", "")
    started = time.time()
    with Path(job["stdout"]).open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, env=environment, check=False)
    result = dict(job)
    result["gpu"] = gpu
    result["return_code"] = completed.returncode
    result["runtime_seconds"] = time.time() - started
    result["completed"] = completed.returncode == 0 and Path(job["output"]).exists()
    return result


def parse_poses(job: dict[str, Any]) -> list[dict[str, Any]]:
    if not job["completed"]:
        return []
    rows = []
    supplier = Chem.SDMolSupplier(job["output"], removeHs=False)
    for pose_index, molecule in enumerate(supplier):
        if molecule is None:
            continue
        props = molecule.GetPropsAsDict()
        rows.append(
            {
                "sequence_key": job["sequence_key"],
                "primary_gene": job["primary_gene"],
                "target_assay_family": job["target_assay_family"],
                "control_pair_id": molecule.GetProp("_Name") if molecule.HasProp("_Name") else "",
                "control_class": clean(props.get("control_class")),
                "pose_index": pose_index,
                "minimized_affinity": props.get("minimizedAffinity"),
                "cnn_score": props.get("CNNscore"),
                "cnn_affinity": props.get("CNNaffinity"),
                "cnn_vs": props.get("CNN_VS"),
                "cnn_affinity_variance": props.get("CNNaffinity_variance"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controls",
        default=str(BASE / "target_docking_calibration_v1/GNINA_TARGET_CALIBRATION_CONTROLS_V1.csv.gz"),
    )
    parser.add_argument(
        "--readiness",
        default=str(BASE / "target_docking_calibration_v1/TARGET_DOCKING_CALIBRATION_READINESS_463_V1.csv"),
    )
    parser.add_argument("--out-dir", default=str(BASE / "gnina_calibration_pilot_v1"))
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--modes", type=int, default=3)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = pd.read_csv(args.controls, low_memory=False).fillna("")
    readiness = pd.read_csv(args.readiness, low_memory=False).fillna("")
    readiness = readiness[
        readiness["calibration_ready_8x8"].astype(bool)
        & readiness["structure_bin"].isin(["A_strict_overlapping_pocket", "B_strict_supported_overlap"])
    ].copy()
    readiness["receptor_size"] = readiness["sequence_key"].map(
        controls.groupby("sequence_key")["sequence_key"].size()
    ).fillna(999999)
    selected_targets = []
    families = ["enzyme", "kinase", "nuclear_epigenetic", "transporter", "ion_channel"]
    for family in families:
        group = readiness[readiness["target_assay_family"].eq(family)]
        if len(group):
            selected_targets.append(group.sort_values("primary_gene", kind="mergesort").iloc[0])
    if len(selected_targets) < args.targets:
        used = {row["sequence_key"] for row in selected_targets}
        for _, row in readiness.iterrows():
            if row["sequence_key"] not in used:
                selected_targets.append(row)
                used.add(row["sequence_key"])
            if len(selected_targets) >= args.targets:
                break
    selected_targets = selected_targets[: args.targets]

    jobs = []
    for target in selected_targets:
        group = controls[controls["sequence_key"].eq(target["sequence_key"])].copy()
        group = group[group["target_control_rank"].le(args.per_class)]
        jobs.append(prepare_target_job(group, target, out_dir))
    pd.DataFrame(jobs).to_csv(out_dir / "GNINA_CALIBRATION_PILOT_JOBS_V1.csv", index=False)

    started = time.time()
    completed_jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_job, job, index % 2, args.exhaustiveness, args.modes): job
            for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            completed_jobs.append(future.result())
    elapsed = time.time() - started

    pose_rows = []
    for job in completed_jobs:
        pose_rows.extend(parse_poses(job))
    poses = pd.DataFrame(pose_rows)
    poses.to_csv(out_dir / "GNINA_CALIBRATION_PILOT_POSES_V1.csv.gz", index=False, compression="gzip")
    ligand = poses.groupby(
        ["sequence_key", "primary_gene", "target_assay_family", "control_pair_id", "control_class"],
        as_index=False,
    ).agg(
        best_cnn_affinity=("cnn_affinity", "max"),
        best_cnn_score=("cnn_score", "max"),
        best_cnn_vs=("cnn_vs", "max"),
        best_vina_affinity=("minimized_affinity", "min"),
        pose_count=("pose_index", "size"),
    )
    ligand.to_csv(out_dir / "GNINA_CALIBRATION_PILOT_LIGAND_SCORES_V1.csv", index=False)
    metrics = []
    for sequence_key, group in ligand.groupby("sequence_key"):
        labels = group["control_class"].eq("positive").astype(int)
        row = {
            "sequence_key": sequence_key,
            "primary_gene": group["primary_gene"].iloc[0],
            "target_assay_family": group["target_assay_family"].iloc[0],
            "ligands": len(group),
            "positive": int(labels.sum()),
            "negative": int((1 - labels).sum()),
        }
        for column in ["best_cnn_affinity", "best_cnn_score", "best_cnn_vs"]:
            row[f"auroc_{column}"] = float(roc_auc_score(labels, group[column]))
        row["auroc_best_vina_affinity"] = float(roc_auc_score(labels, -group["best_vina_affinity"]))
        row["best_metric"] = max(
            [
                "best_cnn_affinity",
                "best_cnn_score",
                "best_cnn_vs",
                "best_vina_affinity",
            ],
            key=lambda name: row[f"auroc_{name}"],
        )
        row["best_auroc"] = row[f"auroc_{row['best_metric']}"]
        row["pilot_calibration_pass"] = row["best_auroc"] >= 0.70
        metrics.append(row)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(out_dir / "GNINA_CALIBRATION_PILOT_TARGET_METRICS_V1.csv", index=False)
    completed_ligands = int(len(ligand))
    summary = {
        "status": "passed" if all(job["completed"] for job in completed_jobs) else "partial",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets": len(jobs),
        "completed_targets": sum(job["completed"] for job in completed_jobs),
        "completed_ligands": completed_ligands,
        "elapsed_seconds": elapsed,
        "observed_wall_seconds_per_ligand_two_gpu": elapsed / max(completed_ligands, 1),
        "projected_9260_control_wall_hours_two_gpu": elapsed / max(completed_ligands, 1) * 9260 / 3600,
        "target_calibration_passes": int(metrics_frame["pilot_calibration_pass"].sum()),
        "target_metrics": metrics_frame.to_dict(orient="records"),
        "warning": (
            "Pilot uses AlphaFold/P2Rank receptors, four positives and four negatives per target, "
            "exhaustiveness 4. Formal calibration must use 8-12 per class, exhaustiveness 8, and "
            "validated experimental holo structures where available."
        ),
    }
    (out_dir / "GNINA_CALIBRATION_PILOT_V1_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

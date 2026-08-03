#!/usr/bin/env python3
"""Run resumable, target-calibrated GNINA docking on all selected ChEMBL controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/affinity_first_remote_discovery_v1"
DEFAULT_CONTROLS = BASE / "target_docking_calibration_v2/GNINA_TARGET_CALIBRATION_CONTROLS_V2.csv.gz"
DEFAULT_RECEPTORS = BASE / "experimental_holo_validation_v2/TARGET_DOCKING_RECEPTOR_SELECTION_463_V2.csv"
DEFAULT_OUT = BASE / "gnina_target_calibration_v2"
GNINA = Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2")
CUDNN = ROOT / ".conda_envs/boltz2/lib/python3.11/site-packages/nvidia/cudnn/lib"
RDLogger.DisableLog("rdApp.*")


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def stable_seed(text: str) -> int:
    return 1 + int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:7], 16) % 2_000_000_000


def make_3d_molecule(smiles: str, name: str, properties: dict[str, Any]) -> Chem.Mol | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = stable_seed(name)
    params.maxIterations = 500
    params.timeout = 15
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=150)
        else:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=150)
    except Exception:
        pass
    molecule = Chem.RemoveHs(molecule)
    molecule.SetProp("_Name", name)
    for key, value in properties.items():
        molecule.SetProp(str(key), str(value))
    return molecule


def prepare_job(group: pd.DataFrame, target: pd.Series, out_dir: Path) -> dict[str, Any]:
    sequence_key = clean(target["sequence_key"])
    job_dir = out_dir / "targets" / sequence_key
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"
    receptor_source = Path(clean(target["docking_receptor_path"]))
    if not receptor_source.exists():
        raise RuntimeError(f"receptor missing: {sequence_key}: {receptor_source}")
    box_values = [
        float(target[column])
        for column in ["box_center_x", "box_center_y", "box_center_z", "box_size_x", "box_size_y", "box_size_z"]
    ]
    if not all(math.isfinite(value) for value in box_values):
        raise RuntimeError(f"docking box missing or non-finite: {sequence_key}")
    receptor = job_dir / "receptor.pdbqt"
    receptor_log = job_dir / "receptor_prepare.log"
    if not receptor.exists() or receptor.stat().st_size < 100:
        with receptor_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                ["obabel", str(receptor_source), "-O", str(receptor), "-xr"],
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0 or not receptor.exists() or receptor.stat().st_size < 100:
            raise RuntimeError(f"receptor preparation failed: {sequence_key}")

    ligand_input = job_dir / "controls.sdf"
    failures = []
    prepared = 0
    writer = Chem.SDWriter(str(ligand_input))
    for _, row in group.sort_values(["control_class", "target_control_rank"], kind="mergesort").iterrows():
        pair_id = clean(row["control_pair_id"])
        molecule = make_3d_molecule(
            clean(row["canonical_control_smiles"]), pair_id,
            {
                "control_pair_id": pair_id,
                "control_class": clean(row["control_class"]),
                "sequence_key": sequence_key,
                "target_control_rank": row["target_control_rank"],
                "parent_molecule_chembl_id": clean(row.get("parent_molecule_chembl_id")),
            },
        )
        if molecule is None:
            failures.append(pair_id)
            continue
        writer.write(molecule)
        prepared += 1
    writer.close()
    if prepared < 2:
        raise RuntimeError(f"insufficient prepared controls: {sequence_key}: {prepared}")
    output = job_dir / "docked.sdf"
    return {
        "sequence_key": sequence_key,
        "primary_gene": clean(target.get("primary_gene")),
        "target_assay_family": clean(target.get("target_assay_family")),
        "receptor_source": clean(target.get("docking_receptor_source")),
        "selected_pdb_id": clean(target.get("selected_pdb_id")),
        "receptor": str(receptor),
        "ligand_input": str(ligand_input),
        "output": str(output),
        "log": str(job_dir / "gnina.log"),
        "stdout": str(job_dir / "gnina.stdout"),
        "status_path": str(status_path),
        "center_x": box_values[0],
        "center_y": box_values[1],
        "center_z": box_values[2],
        "size_x": box_values[3],
        "size_y": box_values[4],
        "size_z": box_values[5],
        "requested_controls": int(len(group)),
        "prepared_controls": prepared,
        "conformer_failures": ";".join(failures),
    }


def prior_job_complete(job: dict[str, Any]) -> dict[str, Any] | None:
    status_path = Path(job["status_path"])
    output = Path(job["output"])
    if not status_path.exists() or not output.exists() or output.stat().st_size < 100:
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("completed") and int(status.get("return_code", -1)) == 0:
            result = dict(job)
            result.update(status)
            result["resumed"] = True
            return result
    except Exception:
        return None
    return None


def run_job(job: dict[str, Any], gpu: int, exhaustiveness: int, modes: int, cpus: int) -> dict[str, Any]:
    resumed = prior_job_complete(job)
    if resumed:
        return resumed
    for key in ["output", "log", "stdout"]:
        Path(job[key]).unlink(missing_ok=True)
    command = [
        str(GNINA), "-r", job["receptor"], "-l", job["ligand_input"], "-o", job["output"],
        "--log", job["log"],
        "--center_x", str(job["center_x"]), "--center_y", str(job["center_y"]), "--center_z", str(job["center_z"]),
        "--size_x", str(job["size_x"]), "--size_y", str(job["size_y"]), "--size_z", str(job["size_z"]),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(modes),
        "--cnn_scoring", "rescore", "--seed", str(stable_seed(job["sequence_key"])), "--cpu", str(cpus),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["LD_LIBRARY_PATH"] = str(CUDNN) + ":" + environment.get("LD_LIBRARY_PATH", "")
    started = time.time()
    with Path(job["stdout"]).open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, env=environment, check=False)
    result = dict(job)
    result.update(
        {
            "gpu": gpu,
            "return_code": int(completed.returncode),
            "runtime_seconds": time.time() - started,
            "completed": completed.returncode == 0 and Path(job["output"]).exists() and Path(job["output"]).stat().st_size > 100,
            "resumed": False,
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    status_payload = {key: result[key] for key in ["gpu", "return_code", "runtime_seconds", "completed", "completed_utc"]}
    Path(job["status_path"]).write_text(json.dumps(status_payload, indent=2) + "\n", encoding="utf-8")
    return result


def run_gpu_queue(
    jobs: list[dict[str, Any]], gpu: int, exhaustiveness: int, modes: int, cpus: int
) -> list[dict[str, Any]]:
    """Run one serial queue assigned to a specific GPU."""
    results = []
    for index, job in enumerate(jobs, start=1):
        try:
            results.append(run_job(job, gpu, exhaustiveness, modes, cpus))
        except Exception as exc:
            failed = dict(job)
            failed.update(
                {
                    "gpu": gpu,
                    "completed": False,
                    "return_code": -999,
                    "runtime_seconds": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            results.append(failed)
        if index % 10 == 0 or index == len(jobs):
            print(f"gpu={gpu} completed_local_targets={index}/{len(jobs)}", flush=True)
    return results


def balanced_worker_queues(
    jobs: list[dict[str, Any]], gpu_ids: list[int], jobs_per_gpu: int
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    worker_gpu = {
        worker: gpu
        for worker, gpu in enumerate(
            gpu for gpu in gpu_ids for _ in range(jobs_per_gpu)
        )
    }
    queues = {worker: [] for worker in worker_gpu}
    loads = {worker: 0 for worker in worker_gpu}
    for job in sorted(jobs, key=lambda row: int(row["prepared_controls"]), reverse=True):
        worker = min(loads, key=lambda item: loads[item])
        queues[worker].append(job)
        loads[worker] += int(job["prepared_controls"])
    gpu_loads = {
        gpu: sum(loads[worker] for worker in loads if worker_gpu[worker] == gpu)
        for gpu in gpu_ids
    }
    print(
        f"worker_control_loads={loads} gpu_control_loads={gpu_loads}",
        flush=True,
    )
    return queues, worker_gpu


def parse_poses(job: dict[str, Any]) -> list[dict[str, Any]]:
    if not job.get("completed"):
        return []
    rows = []
    supplier = Chem.SDMolSupplier(job["output"], removeHs=False)
    for pose_index, molecule in enumerate(supplier):
        if molecule is None:
            continue
        props = molecule.GetPropsAsDict()
        pair_id = clean(props.get("control_pair_id")) or (molecule.GetProp("_Name") if molecule.HasProp("_Name") else "")
        rows.append(
            {
                "sequence_key": job["sequence_key"],
                "primary_gene": job["primary_gene"],
                "target_assay_family": job["target_assay_family"],
                "receptor_source": job["receptor_source"],
                "selected_pdb_id": job["selected_pdb_id"],
                "control_pair_id": pair_id,
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


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        estimates.append(roc_auc_score(sampled_labels, scores[indices]))
    if not estimates:
        return math.nan, math.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def aggregate_and_score(completed_jobs: list[dict[str, Any]], out_dir: Path, bootstrap_repeats: int) -> dict[str, Any]:
    pose_rows = []
    for job in completed_jobs:
        pose_rows.extend(parse_poses(job))
    poses = pd.DataFrame(pose_rows)
    poses_path = out_dir / "GNINA_TARGET_CALIBRATION_POSES_V2.csv.gz"
    poses.to_csv(poses_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    if poses.empty:
        raise RuntimeError("No GNINA poses were parsed")
    numeric = ["cnn_affinity", "cnn_score", "cnn_vs", "minimized_affinity"]
    for column in numeric:
        poses[column] = pd.to_numeric(poses[column], errors="coerce")
    ligand = poses.groupby(
        ["sequence_key", "primary_gene", "target_assay_family", "receptor_source", "selected_pdb_id", "control_pair_id", "control_class"],
        as_index=False, dropna=False,
    ).agg(
        best_cnn_affinity=("cnn_affinity", "max"), best_cnn_score=("cnn_score", "max"),
        best_cnn_vs=("cnn_vs", "max"), best_vina_affinity=("minimized_affinity", "min"), pose_count=("pose_index", "size"),
    )
    ligand["score_vina_directional"] = -ligand["best_vina_affinity"]
    ranked = []
    for _, group in ligand.groupby("sequence_key", sort=False):
        group = group.copy()
        percentile_columns = []
        for column in ["best_cnn_affinity", "best_cnn_score", "best_cnn_vs", "score_vina_directional"]:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().sum() >= 2:
                fill = values.fillna(values.min() - 1.0).to_numpy()
                rank_column = f"target_percentile_{column}"
                group[rank_column] = (rankdata(fill, method="average") - 1) / max(len(fill) - 1, 1)
                percentile_columns.append(rank_column)
        group["locked_consensus_score"] = group[percentile_columns].mean(axis=1) if percentile_columns else math.nan
        ranked.append(group)
    ligand = pd.concat(ranked, ignore_index=True)
    ligand_path = out_dir / "GNINA_TARGET_CALIBRATION_LIGAND_SCORES_V2.csv.gz"
    ligand.to_csv(ligand_path, index=False, compression={"method": "gzip", "compresslevel": 5})

    metric_rows = []
    metric_columns = {
        "cnn_affinity": "best_cnn_affinity", "cnn_score": "best_cnn_score", "cnn_vs": "best_cnn_vs",
        "vina_affinity": "score_vina_directional", "locked_consensus": "locked_consensus_score",
    }
    for sequence_key, group in ligand.groupby("sequence_key"):
        labels = group["control_class"].eq("positive").astype(int).to_numpy()
        row = {
            "sequence_key": sequence_key, "primary_gene": group["primary_gene"].iloc[0],
            "target_assay_family": group["target_assay_family"].iloc[0], "receptor_source": group["receptor_source"].iloc[0],
            "selected_pdb_id": group["selected_pdb_id"].iloc[0], "ligands": len(group),
            "positive": int(labels.sum()), "negative": int((1 - labels).sum()),
        }
        for metric, column in metric_columns.items():
            scores = pd.to_numeric(group[column], errors="coerce").to_numpy()
            valid = np.isfinite(scores)
            if valid.sum() >= 4 and len(np.unique(labels[valid])) == 2:
                row[f"auroc_{metric}"] = float(roc_auc_score(labels[valid], scores[valid]))
                row[f"average_precision_{metric}"] = float(average_precision_score(labels[valid], scores[valid]))
            else:
                row[f"auroc_{metric}"] = math.nan
                row[f"average_precision_{metric}"] = math.nan
        primary_scores = pd.to_numeric(group["locked_consensus_score"], errors="coerce").to_numpy()
        valid = np.isfinite(primary_scores)
        if valid.sum() >= 16 and len(np.unique(labels[valid])) == 2:
            low, high = bootstrap_auc(labels[valid], primary_scores[valid], bootstrap_repeats, stable_seed(sequence_key))
        else:
            low = high = math.nan
        row["locked_consensus_auroc_ci_low"] = low
        row["locked_consensus_auroc_ci_high"] = high
        baseline_ap = labels.mean() if len(labels) else math.nan
        row["calibration_evaluable_8x8"] = row["positive"] >= 8 and row["negative"] >= 8
        row["calibration_pass"] = bool(
            row["calibration_evaluable_8x8"]
            and row.get("auroc_locked_consensus", math.nan) >= 0.65
            and row.get("average_precision_locked_consensus", math.nan) >= baseline_ap + 0.10
        )
        row["calibration_strong"] = bool(row["calibration_pass"] and low >= 0.50)
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "GNINA_TARGET_CALIBRATION_METRICS_V2.csv"
    metrics.to_csv(metrics_path, index=False)
    return {
        "pose_rows": int(len(poses)), "ligand_rows": int(len(ligand)), "targets_with_scores": int(len(metrics)),
        "targets_evaluable_8x8": int(metrics["calibration_evaluable_8x8"].sum()),
        "targets_calibration_pass": int(metrics["calibration_pass"].sum()),
        "targets_calibration_strong": int(metrics["calibration_strong"].sum()),
        "outputs": {"poses": str(poses_path.relative_to(ROOT)), "ligand_scores": str(ligand_path.relative_to(ROOT)), "target_metrics": str(metrics_path.relative_to(ROOT))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", default=str(DEFAULT_CONTROLS))
    parser.add_argument("--receptors", default=str(DEFAULT_RECEPTORS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--modes", type=int, default=5)
    parser.add_argument("--cpus-per-job", type=int, default=6)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--limit-targets", type=int, default=0)
    args = parser.parse_args()
    if not GNINA.exists():
        raise FileNotFoundError(GNINA)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    controls = pd.read_csv(args.controls, low_memory=False).fillna("")
    receptors = pd.read_csv(args.receptors, low_memory=False).fillna("")
    controls = controls[controls["control_class"].isin(["positive", "negative"])].copy()
    target_keys = sorted(set(controls["sequence_key"]) & set(receptors["sequence_key"]))
    if args.limit_targets > 0:
        target_keys = target_keys[: args.limit_targets]
    receptor_index = receptors.set_index("sequence_key", drop=False)

    jobs = []
    preparation_failures = []
    with ThreadPoolExecutor(max_workers=args.prepare_workers) as executor:
        preparation_futures = {
            executor.submit(
                prepare_job,
                controls[controls["sequence_key"].eq(sequence_key)].copy(),
                receptor_index.loc[sequence_key].copy(),
                out_dir,
            ): sequence_key
            for sequence_key in target_keys
        }
        for index, future in enumerate(as_completed(preparation_futures), start=1):
            sequence_key = preparation_futures[future]
            try:
                jobs.append(future.result())
            except Exception as exc:
                preparation_failures.append({"sequence_key": sequence_key, "error": f"{type(exc).__name__}: {exc}"})
            if index % 50 == 0 or index == len(preparation_futures):
                print(f"prepared_targets={index}/{len(target_keys)} jobs={len(jobs)} failures={len(preparation_failures)}", flush=True)
    jobs.sort(key=lambda row: row["sequence_key"])
    pd.DataFrame(jobs).to_csv(out_dir / "GNINA_TARGET_CALIBRATION_JOBS_V2.csv", index=False)
    pd.DataFrame(preparation_failures).to_csv(out_dir / "GNINA_TARGET_CALIBRATION_PREPARATION_FAILURES_V2.csv", index=False)

    gpu_ids = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("No GPU IDs supplied")
    started = time.time()
    completed_jobs = []
    worker_queues, worker_gpu = balanced_worker_queues(
        jobs, gpu_ids, args.jobs_per_gpu
    )
    with ThreadPoolExecutor(max_workers=len(worker_queues)) as executor:
        pending = {
            executor.submit(
                run_gpu_queue,
                queue,
                worker_gpu[worker],
                args.exhaustiveness,
                args.modes,
                args.cpus_per_job,
            ): worker
            for worker, queue in worker_queues.items()
        }
        for future in as_completed(pending):
            worker = pending[future]
            gpu = worker_gpu[worker]
            try:
                completed_jobs.extend(future.result())
            except Exception as exc:
                print(
                    f"worker_queue_failed worker={worker} gpu={gpu} "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )
            print(
                f"worker_queue_complete worker={worker} gpu={gpu} "
                f"collected_targets={len(completed_jobs)}/{len(jobs)}",
                flush=True,
            )
    elapsed = time.time() - started
    job_frame = pd.DataFrame(completed_jobs)
    job_frame.to_csv(out_dir / "GNINA_TARGET_CALIBRATION_JOB_RESULTS_V2.csv", index=False)
    completed = [job for job in completed_jobs if job.get("completed")]
    aggregate = aggregate_and_score(completed, out_dir, args.bootstrap_repeats) if completed else {}
    summary = {
        "status": "passed" if len(completed) == len(jobs) and not preparation_failures else "partial",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_control_rows": int(len(controls[controls["sequence_key"].isin(target_keys)])),
        "requested_targets": len(target_keys), "prepared_targets": len(jobs), "preparation_failures": len(preparation_failures),
        "completed_targets": len(completed), "failed_docking_targets": len(jobs) - len(completed),
        "elapsed_seconds": elapsed, "exhaustiveness": args.exhaustiveness, "num_modes": args.modes,
        "gpus": gpu_ids, "jobs_per_gpu": args.jobs_per_gpu,
        "locked_metric": "mean within-target percentile across CNNaffinity, CNNscore, CNN_VS and directional Vina",
        "calibration_rule": "At least 8 positive and 8 negative controls; locked consensus AUROC >=0.65; average precision >= class prevalence +0.10. Strong additionally requires bootstrap AUROC 95% CI lower bound >=0.50.",
        **aggregate,
    }
    (out_dir / "GNINA_TARGET_CALIBRATION_V2_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run resumable GNINA docking for the frozen v8 remote discovery queue."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/affinity_experiment_package_v8.yaml"
GNINA = Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2")
CUDNN = ROOT / ".conda_envs/boltz2/lib/python3.11/site-packages/nvidia/cudnn/lib"
CALIBRATION_TARGETS = (
    ROOT / "outputs/affinity_first_remote_discovery_v1/gnina_target_calibration_v2/targets"
)
RDLogger.DisableLog("rdApp.*")


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def stable_seed(text: str) -> int:
    return 1 + int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:7], 16) % 2_000_000_000


def make_3d_molecule(smiles: str) -> Chem.Mol | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = stable_seed(smiles)
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
    return Chem.RemoveHs(molecule)


def build_conformer_cache(
    queue: pd.DataFrame, workers: int
) -> tuple[dict[str, Chem.Mol], pd.DataFrame]:
    smiles_values = sorted(set(queue["model_ligand_smiles"].map(clean)) - {""})
    molecules: dict[str, Chem.Mol] = {}
    audit_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(make_3d_molecule, smiles): smiles for smiles in smiles_values}
        for index, future in enumerate(as_completed(futures), start=1):
            smiles = futures[future]
            try:
                molecule = future.result()
                status = "prepared" if molecule is not None else "embed_failed"
            except Exception as exc:
                molecule = None
                status = f"error:{type(exc).__name__}:{exc}"
            if molecule is not None:
                molecules[smiles] = molecule
            audit_rows.append({"model_ligand_smiles": smiles, "conformer_status_v8": status})
            if index % 100 == 0 or index == len(futures):
                print(
                    f"prepared_unique_ligands={index}/{len(futures)} "
                    f"success={len(molecules)}",
                    flush=True,
                )
    return molecules, pd.DataFrame(audit_rows)


def prepare_job(
    group: pd.DataFrame,
    conformers: dict[str, Chem.Mol],
    output_dir: Path,
) -> dict[str, Any]:
    sequence_key = clean(group["sequence_key"].iloc[0])
    job_dir = output_dir / "targets" / sequence_key
    job_dir.mkdir(parents=True, exist_ok=True)
    receptor = CALIBRATION_TARGETS / sequence_key / "receptor.pdbqt"
    if not receptor.exists() or receptor.stat().st_size < 100:
        raise RuntimeError(f"calibration receptor missing: {sequence_key}: {receptor}")

    box_columns = [
        "box_center_x",
        "box_center_y",
        "box_center_z",
        "box_size_x",
        "box_size_y",
        "box_size_z",
    ]
    box_values = [float(group[column].iloc[0]) for column in box_columns]
    if not all(math.isfinite(value) for value in box_values):
        raise RuntimeError(f"non-finite docking box: {sequence_key}")

    ligand_input = job_dir / "candidates.sdf"
    failures: list[str] = []
    prepared = 0
    writer = Chem.SDWriter(str(ligand_input))
    for _, row in group.sort_values("gnina_discovery_queue_rank_v8", kind="mergesort").iterrows():
        pair_id = clean(row["physical_pair_id"])
        smiles = clean(row["model_ligand_smiles"])
        template = conformers.get(smiles)
        if template is None:
            failures.append(pair_id)
            continue
        molecule = Chem.Mol(template)
        molecule.SetProp("_Name", pair_id)
        for key in [
            "physical_pair_id",
            "sequence_key",
            "primary_gene",
            "gnina_discovery_queue_rank_v8",
            "target_admission_tier_v8",
        ]:
            molecule.SetProp(key, clean(row.get(key)))
        writer.write(molecule)
        prepared += 1
    writer.close()
    if prepared < 1:
        raise RuntimeError(f"no candidate conformers: {sequence_key}")
    return {
        "sequence_key": sequence_key,
        "primary_gene": clean(group["primary_gene"].iloc[0]),
        "target_assay_family": clean(group["target_assay_family_v2"].iloc[0]),
        "target_admission_tier_v8": clean(group["target_admission_tier_v8"].iloc[0]),
        "receptor_source": clean(group["docking_receptor_source"].iloc[0]),
        "selected_pdb_id": clean(group["selected_pdb_id"].iloc[0]),
        "receptor": str(receptor),
        "ligand_input": str(ligand_input),
        "output": str(job_dir / "docked.sdf"),
        "log": str(job_dir / "gnina.log"),
        "stdout": str(job_dir / "gnina.stdout"),
        "status_path": str(job_dir / "status.json"),
        "center_x": box_values[0],
        "center_y": box_values[1],
        "center_z": box_values[2],
        "size_x": box_values[3],
        "size_y": box_values[4],
        "size_z": box_values[5],
        "requested_candidates": int(len(group)),
        "prepared_candidates": prepared,
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
    # A killed GNINA process can leave a syntactically valid but incomplete SDF.
    for key in ["output", "log", "stdout"]:
        Path(job[key]).unlink(missing_ok=True)
    command = [
        str(GNINA),
        "-r",
        job["receptor"],
        "-l",
        job["ligand_input"],
        "-o",
        job["output"],
        "--log",
        job["log"],
        "--center_x",
        str(job["center_x"]),
        "--center_y",
        str(job["center_y"]),
        "--center_z",
        str(job["center_z"]),
        "--size_x",
        str(job["size_x"]),
        "--size_y",
        str(job["size_y"]),
        "--size_z",
        str(job["size_z"]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(modes),
        "--cnn_scoring",
        "rescore",
        "--seed",
        str(stable_seed(f"v8:{job['sequence_key']}")),
        "--cpu",
        str(cpus),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["LD_LIBRARY_PATH"] = str(CUDNN) + ":" + environment.get("LD_LIBRARY_PATH", "")
    started = time.time()
    with Path(job["stdout"]).open("w", encoding="utf-8") as stdout:
        completed = subprocess.run(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    result = dict(job)
    result.update(
        {
            "gpu": gpu,
            "return_code": int(completed.returncode),
            "runtime_seconds": time.time() - started,
            "completed": completed.returncode == 0
            and Path(job["output"]).exists()
            and Path(job["output"]).stat().st_size > 100,
            "resumed": False,
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    status_payload = {
        key: result[key]
        for key in ["gpu", "return_code", "runtime_seconds", "completed", "completed_utc"]
    }
    Path(job["status_path"]).write_text(
        json.dumps(status_payload, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_gpu_queue(
    jobs: list[dict[str, Any]], gpu: int, exhaustiveness: int, modes: int, cpus: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
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
        completed_candidates = sum(
            int(row.get("prepared_candidates", 0))
            for row in results
            if row.get("completed")
        )
        print(
            f"gpu={gpu} completed_targets={index}/{len(jobs)} "
            f"completed_candidates={completed_candidates}",
            flush=True,
        )
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
    for job in sorted(jobs, key=lambda row: int(row["prepared_candidates"]), reverse=True):
        worker = min(loads, key=lambda item: loads[item])
        queues[worker].append(job)
        loads[worker] += int(job["prepared_candidates"])
    gpu_loads = {
        gpu: sum(loads[worker] for worker in loads if worker_gpu[worker] == gpu)
        for gpu in gpu_ids
    }
    print(f"worker_candidate_loads={loads} gpu_candidate_loads={gpu_loads}", flush=True)
    return queues, worker_gpu


def parse_completed_jobs(jobs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    pose_rows: list[dict[str, Any]] = []
    for job in jobs:
        if not job.get("completed"):
            continue
        supplier = Chem.SDMolSupplier(job["output"], removeHs=False)
        pair_pose_counts: dict[str, int] = {}
        for molecule in supplier:
            if molecule is None:
                continue
            props = molecule.GetPropsAsDict()
            pair_id = clean(props.get("physical_pair_id")) or clean(
                molecule.GetProp("_Name") if molecule.HasProp("_Name") else ""
            )
            pose_index = pair_pose_counts.get(pair_id, 0)
            pair_pose_counts[pair_id] = pose_index + 1
            pose_rows.append(
                {
                    "physical_pair_id": pair_id,
                    "sequence_key": job["sequence_key"],
                    "primary_gene": job["primary_gene"],
                    "target_assay_family": job["target_assay_family"],
                    "target_admission_tier_v8": job["target_admission_tier_v8"],
                    "receptor_source": job["receptor_source"],
                    "selected_pdb_id": job["selected_pdb_id"],
                    "pose_index_within_pair": pose_index,
                    "minimized_affinity": props.get("minimizedAffinity"),
                    "cnn_score": props.get("CNNscore"),
                    "cnn_affinity": props.get("CNNaffinity"),
                    "cnn_vs": props.get("CNN_VS"),
                    "cnn_affinity_variance": props.get("CNNaffinity_variance"),
                }
            )
    poses = pd.DataFrame(pose_rows)
    pose_path = output_dir / "GNINA_REMOTE_DISCOVERY_POSES_V8.csv.gz"
    poses.to_csv(pose_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    if poses.empty:
        return {"pose_rows": 0, "pair_rows": 0}
    for column in ["minimized_affinity", "cnn_score", "cnn_affinity", "cnn_vs"]:
        poses[column] = pd.to_numeric(poses[column], errors="coerce")
    pairs = poses.groupby(
        [
            "physical_pair_id",
            "sequence_key",
            "primary_gene",
            "target_assay_family",
            "target_admission_tier_v8",
            "receptor_source",
            "selected_pdb_id",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        best_cnn_affinity=("cnn_affinity", "max"),
        best_cnn_score=("cnn_score", "max"),
        best_cnn_vs=("cnn_vs", "max"),
        best_vina_affinity=("minimized_affinity", "min"),
        pose_count=("pose_index_within_pair", "size"),
    )
    pairs["score_vina_directional"] = -pairs["best_vina_affinity"]
    pair_path = output_dir / "GNINA_REMOTE_DISCOVERY_PAIR_SCORES_V8.csv.gz"
    pairs.to_csv(pair_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    return {
        "pose_rows": int(len(poses)),
        "pair_rows": int(len(pairs)),
        "pose_output": str(pose_path.relative_to(ROOT)),
        "pair_output": str(pair_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit-targets", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--compress-raw-after-complete", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = ROOT / config["outputs"]["directory"]
    queue_path = base / "discovery_queue/GNINA_REMOTE_DISCOVERY_QUEUE_V8.csv.gz"
    output_dir = Path(args.out_dir).resolve() if args.out_dir else base / "gnina_discovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not GNINA.exists():
        raise FileNotFoundError(GNINA)

    queue = pd.read_csv(queue_path, low_memory=False)
    if args.limit_targets > 0:
        target_keys = sorted(queue["sequence_key"].unique())[: args.limit_targets]
        queue = queue[queue["sequence_key"].isin(target_keys)].copy()
    if args.limit_pairs > 0:
        queue = queue.head(args.limit_pairs).copy()
    conformers, conformer_audit = build_conformer_cache(
        queue, int(config["discovery_docking"]["prepare_workers"])
    )
    conformer_audit.to_csv(output_dir / "GNINA_DISCOVERY_CONFORMER_AUDIT_V8.csv", index=False)

    jobs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sequence_key, group in queue.groupby("sequence_key", sort=True):
        try:
            jobs.append(prepare_job(group.copy(), conformers, output_dir))
        except Exception as exc:
            failures.append(
                {"sequence_key": sequence_key, "error": f"{type(exc).__name__}: {exc}"}
            )
    pd.DataFrame(jobs).to_csv(output_dir / "GNINA_DISCOVERY_JOBS_V8.csv", index=False)
    pd.DataFrame(failures).to_csv(
        output_dir / "GNINA_DISCOVERY_PREPARATION_FAILURES_V8.csv", index=False
    )

    gpu_ids = [int(value) for value in config["discovery_docking"]["gpus"]]
    jobs_per_gpu = int(config["discovery_docking"].get("jobs_per_gpu", 1))
    queues, worker_gpu = balanced_worker_queues(jobs, gpu_ids, jobs_per_gpu)
    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = {
            executor.submit(
                run_gpu_queue,
                worker_jobs,
                worker_gpu[worker],
                int(config["discovery_docking"]["exhaustiveness"]),
                int(config["discovery_docking"]["num_modes"]),
                int(config["discovery_docking"]["cpus_per_job"]),
            ): worker
            for worker, worker_jobs in queues.items()
        }
        for future in as_completed(futures):
            results.extend(future.result())
    elapsed = time.time() - started
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "GNINA_DISCOVERY_JOB_RESULTS_V8.csv", index=False)
    completed = [row for row in results if row.get("completed")]
    aggregate = parse_completed_jobs(completed, output_dir)

    if args.compress_raw_after_complete and len(completed) == len(jobs):
        for job in completed:
            for key in ["ligand_input", "output"]:
                source = Path(job[key])
                target = source.with_suffix(source.suffix + ".gz")
                if source.exists() and not target.exists():
                    with source.open("rb") as input_handle, gzip.open(
                        target, "wb", compresslevel=5
                    ) as output_handle:
                        shutil.copyfileobj(input_handle, output_handle)
                    source.unlink()

    summary = {
        "status": "passed" if len(completed) == len(jobs) and not failures else "partial",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_pairs": int(len(queue)),
        "requested_targets": int(queue["sequence_key"].nunique()),
        "unique_ligands": int(queue["model_ligand_smiles"].nunique()),
        "prepared_targets": int(len(jobs)),
        "preparation_failures": int(len(failures)),
        "completed_targets": int(len(completed)),
        "failed_targets": int(len(jobs) - len(completed)),
        "elapsed_seconds": elapsed,
        "exhaustiveness": int(config["discovery_docking"]["exhaustiveness"]),
        "num_modes": int(config["discovery_docking"]["num_modes"]),
        "gpus": gpu_ids,
        "jobs_per_gpu": jobs_per_gpu,
        "interpretation": (
            "Raw GNINA scores are not comparable across targets; "
            "use v8 control-normalized evidence only."
        ),
        **aggregate,
    }
    summary_path = output_dir / "GNINA_REMOTE_DISCOVERY_V8_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

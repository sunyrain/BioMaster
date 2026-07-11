#!/usr/bin/env python3
"""Recover incomplete Boltz-2 rows one at a time and repair signed run metadata.

The formal queue deliberately uses small batches for throughput. Very long
proteins can still exhaust GPU memory and leave otherwise valid rows in the
same batch complete. This utility reruns only incomplete rows in fresh Boltz
processes, using the original batch seed and model parameters, then copies the
validated prediction directory back into the signed batch run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_boltz2_batched_queue import (
    completed_prediction_stems,
    read_json,
    result_output_provenance,
    summarize_status,
    write_json,
)

def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prediction_dir(run_dir: Path, stem: str) -> Path | None:
    matches = [path for path in run_dir.rglob(stem) if path.is_dir() and path.parent.name == "predictions"]
    return matches[0] if len(matches) == 1 else None


def run_gpu_queue(
    gpu: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boltz_bin = (ROOT / args.boltz_bin).resolve()
    cache_dir = (ROOT / args.cache_dir).resolve()
    recovery_root = run_dir / "recovery_runs"
    log_root = run_dir / "recovery_logs"
    tmp_dir = (ROOT / args.tmp_dir / f"gpu_{gpu}").resolve()
    recovery_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        stem = str(row["boltzStem"])
        yaml_path = Path(str(row["yamlPath"])).resolve()
        output_dir = recovery_root / stem
        log_path = log_root / f"{stem}.log"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        cmd = [
            str(boltz_bin),
            "predict",
            str(yaml_path),
            "--out_dir",
            str(output_dir),
            "--cache",
            str(cache_dir),
            "--model",
            "boltz2",
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--recycling_steps",
            str(args.recycling_steps),
            "--sampling_steps",
            str(args.sampling_steps),
            "--diffusion_samples",
            str(args.diffusion_samples),
            "--sampling_steps_affinity",
            str(args.sampling_steps_affinity),
            "--diffusion_samples_affinity",
            str(args.diffusion_samples_affinity),
            "--seed",
            str(int(row["seed"])),
            "--num_workers",
            str(args.num_workers),
            "--preprocessing-threads",
            str(args.preprocessing_threads),
            "--override",
        ]
        if args.no_kernels:
            cmd.append("--no_kernels")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["BOLTZ_CACHE"] = str(cache_dir)
        env["TMPDIR"] = str(tmp_dir)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"=== recovery gpu={gpu} start={utc_now()} stem={stem} ===\n")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        complete = stem in completed_prediction_stems(output_dir)
        records.append(
            {
                "pairId": str(row["pairId"]),
                "boltzStem": stem,
                "batch": str(row["batch"]),
                "seed": int(row["seed"]),
                "gpu": str(gpu),
                "returnCode": int(result.returncode),
                "completed": bool(complete),
                "elapsedSec": round(time.time() - started, 3),
                "yamlPath": str(yaml_path),
                "outputDir": str(output_dir),
                "logPath": str(log_path),
                "timeUtc": utc_now(),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--boltz-bin", default=".conda_envs/boltz2/bin/boltz")
    parser.add_argument("--cache-dir", default="outputs/boltz2_structure_affinity_v1/boltz_cache")
    parser.add_argument("--tmp-dir", default=".tmp/bv4")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--diffusion-samples", type=int, default=2)
    parser.add_argument("--sampling-steps-affinity", type=int, default=50)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preprocessing-threads", type=int, default=2)
    parser.add_argument("--no-kernels", action="store_true", default=True)
    parser.add_argument("--allow-kernels", dest="no_kernels", action="store_false")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    run_dir = Path(args.run_dir).resolve()
    provenance_path = run_dir / "result_provenance.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    provenance = pd.read_csv(provenance_path, low_memory=False).fillna("")
    manifest["boltzStem"] = manifest["yamlFile"].astype(str).map(lambda value: Path(value).stem)
    manifest["yamlPath"] = manifest["yamlPath"].astype(str).map(
        lambda value: str((ROOT / value).resolve()) if not Path(value).is_absolute() else value
    )
    incomplete = ~provenance["resultCompletedVerified"].astype(str).str.lower().isin({"true", "1", "1.0"})
    todo = provenance.loc[incomplete, ["pairId", "boltzStem", "batch", "seed"]].merge(
        manifest[["pairId", "yamlPath"]], on="pairId", how="left", validate="one_to_one"
    )
    if todo.empty:
        print(json.dumps({"status": "nothing_to_recover", "rows": 0}, ensure_ascii=False))
        return 0
    if todo["yamlPath"].astype(str).eq("").any():
        raise RuntimeError("One or more incomplete rows lack a signed YAML path")

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    assignments: dict[str, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    ordered = todo.sort_values(["boltzStem", "pairId"]).to_dict(orient="records")
    for index, row in enumerate(ordered):
        assignments[gpus[index % len(gpus)]].append(row)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_gpu_queue, gpu, rows, args, run_dir) for gpu, rows in assignments.items() if rows]
        recovery_records = [record for future in futures for record in future.result()]

    recovery = pd.DataFrame(recovery_records).sort_values(["batch", "boltzStem"])
    recovery_csv = run_dir / "recovery_rows.csv"
    recovery.to_csv(recovery_csv, index=False)
    recovery_json = run_dir / "recovery_summary.json"

    for record in recovery_records:
        if not record["completed"]:
            continue
        stem = str(record["boltzStem"])
        source = prediction_dir(Path(record["outputDir"]), stem)
        batch = str(record["batch"])
        target = run_dir / "batch_runs" / batch / f"boltz_results_{batch}" / "predictions" / stem
        if source is None:
            raise RuntimeError(f"Validated recovery prediction directory not found for {stem}")
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    affected_batches = sorted(set(todo["batch"].astype(str)))
    for batch in affected_batches:
        status_path = run_dir / "status" / f"{batch}.json"
        batch_run_dir = run_dir / "batch_runs" / batch
        batch_input_dir = run_dir / "batch_inputs" / batch
        previous_status = read_json(status_path)
        expected_stems = {path.stem for path in batch_input_dir.glob("*.yaml")}
        completed_stems = completed_prediction_stems(batch_run_dir)
        missing_stems = sorted(expected_stems - completed_stems)
        previous_status.update(
            {
                "status": "success" if not missing_stems else "partial_success",
                "returnCode": 0 if not missing_stems else 1,
                "completedRowsVerified": len(expected_stems) - len(missing_stems),
                "missingRowsVerified": len(missing_stems),
                "missingStems": missing_stems,
                "recoveryApplied": True,
                "recoveryMethod": "single_input_same_batch_seed",
                "recoveryRowsCsv": str(recovery_csv),
                "recoveryTimeUtc": utc_now(),
            }
        )
        write_json(status_path, previous_status)
        batch_provenance_path = run_dir / "provenance" / f"{batch}.csv"
        batch_provenance = pd.read_csv(batch_provenance_path, low_memory=False).fillna("")
        for index, row in batch_provenance.iterrows():
            stem = str(row["boltzStem"])
            output = result_output_provenance(batch_run_dir, stem)
            for key, value in output.items():
                batch_provenance.at[index, key] = value
            batch_provenance.at[index, "resultCompletedVerified"] = stem in completed_stems
            if stem in set(todo["boltzStem"].astype(str)):
                batch_provenance.at[index, "recoveryApplied"] = True
                batch_provenance.at[index, "recoveryMethod"] = "single_input_same_batch_seed"
        batch_provenance.to_csv(batch_provenance_path, index=False)

    summary = summarize_status(run_dir)
    completed_count = int(recovery["completed"].astype(bool).sum())
    payload = {
        "createdUtc": utc_now(),
        "attemptedRows": int(len(recovery)),
        "completedRows": completed_count,
        "failedRows": int(len(recovery) - completed_count),
        "affectedBatches": affected_batches,
        "runStatus": summary,
        "recoveryRowsCsv": str(recovery_csv),
    }
    write_json(recovery_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if completed_count == len(recovery) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a Boltz-2 YAML queue in resumable GPU batches.

This runner is intentionally operational rather than scientific: it shards an
existing Boltz YAML input package into small directories, runs one Boltz process
per GPU, and writes per-batch status files so long queues can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def prepare_batches(
    root: Path,
    args: argparse.Namespace,
    run_parameter_signature: str,
) -> list[dict[str, Any]]:
    manifest_path = root / args.input_manifest
    input_dir = root / args.input_dir
    out_dir = root / args.out_dir
    batch_root = out_dir / "batch_inputs"
    run_root = out_dir / "batch_runs"
    log_root = out_dir / "logs"
    batch_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    required = {"pairId", "yamlFile", "yamlSha256", "inputSignatureSha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Signed Boltz manifest is missing columns: {sorted(missing)}")
    manifest["_rank"] = pd.to_numeric(manifest["externalQueueRank"], errors="coerce").fillna(999999)
    manifest = manifest.sort_values("_rank").head(args.top_n).copy() if args.top_n > 0 else manifest.sort_values("_rank").copy()
    records = manifest.to_dict(orient="records")
    batches: list[dict[str, Any]] = []
    for batch_index, rows in enumerate(chunked(records, args.batch_size), start=1):
        batch_name = f"batch_{batch_index:04d}"
        batch_input_dir = batch_root / batch_name
        batch_run_dir = run_root / batch_name
        batch_status = out_dir / "status" / f"{batch_name}.json"
        batch_input_dir.mkdir(parents=True, exist_ok=True)
        batch_run_dir.mkdir(parents=True, exist_ok=True)

        row_provenance = [
            {
                "pairId": str(row["pairId"]),
                "yamlFile": str(row["yamlFile"]),
                "yamlSha256": str(row["yamlSha256"]),
                "inputSignatureSha256": str(row["inputSignatureSha256"]),
            }
            for row in rows
        ]
        seed = int(args.seed_base) + batch_index
        batch_input_signature = payload_sha256(
            {
                "batchIndex": batch_index,
                "runParameterSignature": run_parameter_signature,
                "seed": seed,
                "rows": row_provenance,
            }
        )
        previous = read_json(batch_status)
        previous_signature = str(previous.get("batchInputSignature") or "")
        if previous and previous_signature != batch_input_signature:
            if not args.force:
                raise RuntimeError(
                    f"Existing {batch_name} has a different input signature; use a new --out-dir or --force"
                )
            shutil.rmtree(batch_input_dir, ignore_errors=True)
            shutil.rmtree(batch_run_dir, ignore_errors=True)
            batch_status.unlink(missing_ok=True)
            (log_root / f"{batch_name}.log").unlink(missing_ok=True)
            batch_input_dir.mkdir(parents=True, exist_ok=True)
            batch_run_dir.mkdir(parents=True, exist_ok=True)

        yaml_files: list[str] = []
        for row in rows:
            yaml_file = Path(str(row["yamlFile"]))
            source = yaml_file if yaml_file.is_absolute() else input_dir / yaml_file.name
            if not source.exists():
                source = root / str(row["yamlFile"])
            if not source.exists():
                raise FileNotFoundError(f"Missing YAML for {row.get('pairId')}: {row.get('yamlFile')}")
            dest = batch_input_dir / source.name
            if dest.exists() and file_sha256(dest) != str(row["yamlSha256"]):
                dest.unlink()
            if not dest.exists():
                try:
                    dest.symlink_to(source.resolve())
                except OSError:
                    shutil.copy2(source, dest)
            if file_sha256(dest) != str(row["yamlSha256"]):
                raise RuntimeError(f"Batch YAML hash mismatch: {dest}")
            yaml_files.append(dest.name)

        expected_stems = {Path(name).stem for name in yaml_files}
        completed_outputs = completed_prediction_stems(
            batch_run_dir,
            required_models=max(1, int(args.diffusion_samples)),
        )
        provenance_path = out_dir / "provenance" / f"{batch_name}.csv"
        accepted_missing = (
            set(str(value) for value in previous.get("missingStems", []))
            if args.accept_partial and previous.get("status") == "partial_success"
            else set()
        )
        provenance_ok = False
        if provenance_path.exists():
            previous_provenance = pd.read_csv(provenance_path, low_memory=False).fillna("")
            provenance_required = {"pairId", "batchInputSignature", "resultCompletedVerified"}
            if provenance_required.issubset(previous_provenance.columns):
                completed_flags = previous_provenance[
                    "resultCompletedVerified"
                ].astype(str).str.lower().isin({"true", "1", "1.0"})
                provenance_stems = previous_provenance["yamlFile"].map(
                    lambda value: Path(str(value)).stem
                )
                allowed_incomplete = provenance_stems.isin(accepted_missing)
                provenance_ok = (
                    len(previous_provenance) == len(rows)
                    and set(previous_provenance["pairId"].astype(str))
                    == {str(row["pairId"]) for row in rows}
                    and previous_provenance["batchInputSignature"].astype(str).eq(batch_input_signature).all()
                    and (completed_flags | allowed_incomplete).all()
                )
        completed = (
            previous.get("status") in (
                {"success", "partial_success"} if args.accept_partial else {"success"}
            )
            and previous_signature == batch_input_signature
            and (expected_stems - accepted_missing).issubset(completed_outputs)
            and provenance_ok
            and not args.force
        )
        batches.append(
            {
                "batch": batch_name,
                "batchIndex": batch_index,
                "rows": len(rows),
                "firstRank": int(float(rows[0]["externalQueueRank"])),
                "lastRank": int(float(rows[-1]["externalQueueRank"])),
                "inputDir": str(batch_input_dir),
                "runDir": str(batch_run_dir),
                "statusPath": str(batch_status),
                "provenancePath": str(provenance_path),
                "logPath": str(log_root / f"{batch_name}.log"),
                "yamlFiles": yaml_files,
                "rowProvenance": row_provenance,
                "seed": seed,
                "runParameterSignature": run_parameter_signature,
                "batchInputSignature": batch_input_signature,
                "skip": completed,
            }
        )
    pd.DataFrame(batches).drop(columns=["yamlFiles", "rowProvenance"]).to_csv(
        out_dir / "batch_plan.csv", index=False
    )
    return batches


def worker(gpu_id: str, task_queue: Queue, root: str, args_dict: dict[str, Any]) -> None:
    root_path = Path(root)
    args = argparse.Namespace(**args_dict)
    boltz_bin = root_path / args.boltz_bin
    cache_dir = root_path / args.cache_dir
    tmp_dir = root_path / args.tmp_dir / f"gpu_{gpu_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    while True:
        batch = task_queue.get()
        if batch is None:
            return
        if batch.get("skip"):
            continue

        status_path = Path(batch["statusPath"])
        log_path = Path(batch["logPath"])
        started = time.time()
        free_before = disk_free_gb(root_path)
        if free_before < args.min_free_gb:
            write_json(
                status_path,
                {
                    "status": "stopped_low_disk",
                    "gpu": gpu_id,
                    "batch": batch["batch"],
                    "freeGb": round(free_before, 3),
                    "minFreeGb": args.min_free_gb,
                    "timeUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            return

        cmd = [
            str(boltz_bin),
            "predict",
            batch["inputDir"],
            "--out_dir",
            batch["runDir"],
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
            str(batch["seed"]),
            "--num_workers",
            str(args.num_workers),
            "--preprocessing-threads",
            str(args.preprocessing_threads),
            "--override",
        ]
        if args.no_kernels:
            cmd.append("--no_kernels")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["BOLTZ_CACHE"] = str(cache_dir)
        env["TMPDIR"] = str(tmp_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {batch['batch']} gpu={gpu_id} start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} rows={batch['rows']} ranks={batch['firstRank']}-{batch['lastRank']} ===\n")
            log.write(" ".join(cmd) + "\n")
            log.flush()
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=root_path)

        elapsed = time.time() - started
        expected_stems = {Path(name).stem for name in batch["yamlFiles"]}
        completed_stems = completed_prediction_stems(
            Path(batch["runDir"]),
            required_models=max(1, int(args.diffusion_samples)),
        )
        missing_stems = sorted(expected_stems - completed_stems)
        if result.returncode != 0:
            batch_status = "failed"
        elif missing_stems:
            batch_status = "partial_success"
        else:
            batch_status = "success"
        status = {
            "status": batch_status,
            "returnCode": int(result.returncode),
            "gpu": gpu_id,
            "batch": batch["batch"],
            "rows": int(batch["rows"]),
            "firstRank": int(batch["firstRank"]),
            "lastRank": int(batch["lastRank"]),
            "elapsedSec": round(elapsed, 3),
            "freeGbBefore": round(free_before, 3),
            "freeGbAfter": round(disk_free_gb(root_path), 3),
            "logPath": str(log_path),
            "runDir": batch["runDir"],
            "completedRowsVerified": int(len(expected_stems) - len(missing_stems)),
            "missingRowsVerified": int(len(missing_stems)),
            "missingStems": missing_stems,
            "seed": int(batch["seed"]),
            "seedScheme": args.seed_scheme,
            "runParameterSignature": batch["runParameterSignature"],
            "batchInputSignature": batch["batchInputSignature"],
            "timeUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json(status_path, status)
        provenance = []
        for row in batch["rowProvenance"]:
            stem = Path(row["yamlFile"]).stem
            output_provenance = result_output_provenance(Path(batch["runDir"]), stem)
            provenance.append(
                {
                    **row,
                    **output_provenance,
                    "boltzStem": stem,
                    "batch": batch["batch"],
                    "batchInputSignature": batch["batchInputSignature"],
                    "runParameterSignature": batch["runParameterSignature"],
                    "seed": int(batch["seed"]),
                    "resultCompletedVerified": stem in completed_stems,
                }
            )
        provenance_dir = status_path.parent.parent / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(provenance).to_csv(provenance_dir / f"{batch['batch']}.csv", index=False)
        if batch_status != "success" and args.stop_on_failure:
            return


def summarize_status(out_dir: Path) -> dict[str, Any]:
    status_dir = out_dir / "status"
    records = [read_json(path) for path in sorted(status_dir.glob("batch_*.json"))]
    records = [record for record in records if record]
    counts: dict[str, int] = {}
    rows_by_status: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        rows_by_status[status] = rows_by_status.get(status, 0) + int(record.get("rows", 0) or 0)
    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statusFiles": len(records),
        "batchStatusCounts": counts,
        "rowsByStatus": rows_by_status,
        "freeGb": round(disk_free_gb(ROOT), 3),
    }
    write_json(out_dir / "run_status_summary.json", summary)
    if records:
        pd.DataFrame(records).to_csv(out_dir / "batch_status.csv", index=False)
    provenance_paths = sorted((out_dir / "provenance").glob("batch_*.csv"))
    if provenance_paths:
        provenance = pd.concat(
            [pd.read_csv(path, low_memory=False).fillna("") for path in provenance_paths],
            ignore_index=True,
        )
        if provenance["pairId"].duplicated().any():
            raise RuntimeError("Duplicate pairId rows in Boltz result provenance")
        provenance.to_csv(out_dir / "result_provenance.csv", index=False)
        summary["provenanceRows"] = int(len(provenance))
        summary["provenanceCompletedRows"] = int(
            provenance["resultCompletedVerified"].astype(str).str.lower().isin({"true", "1", "1.0"}).sum()
        )
        write_json(out_dir / "run_status_summary.json", summary)
    return summary


def completed_prediction_stems(run_dir: Path, required_models: int = 1) -> set[str]:
    confidence = {
        path.name.replace("confidence_", "").replace("_model_0.json", ""): path
        for path in run_dir.rglob("confidence_*_model_0.json")
    }
    affinity = {
        path.name.replace("affinity_", "").replace(".json", ""): path
        for path in run_dir.rglob("affinity_*.json")
    }
    model_paths: list[dict[str, Path]] = []
    for model_index in range(required_models):
        suffix = f"_model_{model_index}.cif"
        model_paths.append(
            {
                path.name.removesuffix(suffix): path
                for path in run_dir.rglob(f"*{suffix}")
            }
        )
    complete: set[str] = set()
    candidate_stems = set(confidence) & set(affinity)
    for paths in model_paths:
        candidate_stems &= set(paths)
    for stem in candidate_stems:
        confidence_json = read_json(confidence[stem])
        affinity_json = read_json(affinity[stem])
        values = [
            confidence_json.get("confidence_score"),
            confidence_json.get("ligand_iptm"),
            confidence_json.get("complex_iplddt"),
            affinity_json.get("affinity_probability_binary"),
        ]
        try:
            valid = all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in values)
        except (TypeError, ValueError):
            valid = False
        if valid and all(paths[stem].stat().st_size > 0 for paths in model_paths):
            complete.add(stem)
    return complete


def result_output_provenance(run_dir: Path, stem: str) -> dict[str, Any]:
    patterns = {
        "confidence": f"confidence_{stem}_model_0.json",
        "affinity": f"affinity_{stem}.json",
        "cifModel0": f"{stem}_model_0.cif",
        "cifModel1": f"{stem}_model_1.cif",
    }
    output: dict[str, Any] = {}
    for label, name in patterns.items():
        matches = list(run_dir.rglob(name))
        path = matches[0] if len(matches) == 1 else None
        output[f"{label}Path"] = str(path) if path else ""
        output[f"{label}Sha256"] = file_sha256(path) if path else ""
    return output


def model_environment_provenance(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    boltz_bin = root / args.boltz_bin
    python_bin = boltz_bin.parent / "python"
    environment: dict[str, Any] = {"boltzBin": str(boltz_bin)}
    command = [
        str(python_bin),
        "-c",
        (
            "import importlib.metadata,json,sys,torch;"
            "print(json.dumps({'boltz':importlib.metadata.version('boltz'),'torch':torch.__version__,"
            "'cuda':torch.version.cuda,'python':sys.version}))"
        ),
    ]
    try:
        environment.update(json.loads(subprocess.check_output(command, text=True, cwd=root).strip()))
    except Exception as exc:  # noqa: BLE001
        environment["environmentError"] = str(exc)
    cache_dir = root / args.cache_dir
    artifacts = []
    for path in sorted(cache_dir.glob("*.ckpt")):
        artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    environment["modelArtifacts"] = artifacts
    environment["modelArtifactBundleSha256"] = payload_sha256(artifacts)
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", default="outputs/final_1000_funnel_v1/boltz_pre3000_input_package/boltz2_input_manifest.csv")
    parser.add_argument("--input-dir", default="outputs/final_1000_funnel_v1/boltz_pre3000_input_package/inputs")
    parser.add_argument("--out-dir", default="outputs/final_1000_funnel_v1/boltz_pre3000_full_run")
    parser.add_argument("--top-n", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--boltz-bin", default=".conda_envs/boltz2/bin/boltz")
    parser.add_argument("--cache-dir", default="outputs/boltz2_structure_affinity_v1/boltz_cache")
    parser.add_argument("--tmp-dir", default=".tmp/boltz_pre3000")
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--sampling-steps-affinity", type=int, default=10)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=20260710)
    parser.add_argument("--seed-scheme", choices=["batch_index_offset"], default="batch_index_offset")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preprocessing-threads", type=int, default=2)
    parser.add_argument("--no-kernels", action="store_true", default=True)
    parser.add_argument("--allow-kernels", dest="no_kernels", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--accept-partial",
        action="store_true",
        help=(
            "On resume, accept a signed partial_success batch when every non-missing stem "
            "is complete and each missing stem is explicitly recorded in provenance."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    run_parameters = {
        "model": "boltz2",
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "sampling_steps_affinity": args.sampling_steps_affinity,
        "diffusion_samples_affinity": args.diffusion_samples_affinity,
        "no_kernels": args.no_kernels,
        "seed_base": args.seed_base,
        "seed_scheme": args.seed_scheme,
    }
    run_signature = hashlib.sha256(
        json.dumps(run_parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / args.input_manifest
    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    manifest["_rank"] = pd.to_numeric(manifest["externalQueueRank"], errors="coerce").fillna(999999)
    selected_manifest = (
        manifest.sort_values("_rank").head(args.top_n).copy()
        if args.top_n > 0
        else manifest.sort_values("_rank").copy()
    )
    manifest_bundle = [
        {
            "pairId": str(row["pairId"]),
            "yamlSha256": str(row["yamlSha256"]),
            "inputSignatureSha256": str(row["inputSignatureSha256"]),
        }
        for _, row in selected_manifest.iterrows()
    ]
    batches = prepare_batches(ROOT, args, run_signature)
    pending = [batch for batch in batches if not batch.get("skip")]
    plan = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputManifest": args.input_manifest,
        "inputDir": args.input_dir,
        "outDir": args.out_dir,
        "topN": args.top_n,
        "batchSize": args.batch_size,
        "totalBatches": len(batches),
        "pendingBatches": len(pending),
        "skippedBatches": len(batches) - len(pending),
        "gpus": [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()],
        "freeGbAtStart": round(disk_free_gb(ROOT), 3),
        "runParameters": run_parameters,
        "runParameterSignature": run_signature,
        "inputManifestSha256": file_sha256(manifest_path),
        "inputBundleSha256": payload_sha256(manifest_bundle),
        "batchPlanSha256": file_sha256(out_dir / "batch_plan.csv"),
        "modelEnvironment": model_environment_provenance(ROOT, args),
    }
    write_json(out_dir / "run_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.plan_only:
        summarize_status(out_dir)
        return 0

    q: Queue = Queue()
    for batch in pending:
        q.put(batch)
    args_dict = vars(args)
    processes: list[Process] = []
    gpus = plan["gpus"]
    for _gpu in gpus:
        q.put(None)
    for gpu in gpus:
        process = Process(target=worker, args=(gpu, q, str(ROOT), args_dict), daemon=False)
        process.start()
        processes.append(process)
    exit_code = 0
    try:
        while any(process.is_alive() for process in processes):
            time.sleep(30)
            summary = summarize_status(out_dir)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                exit_code = int(process.exitcode or 1)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        summarize_status(out_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

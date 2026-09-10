#!/usr/bin/env python3
"""Run a fixed V3A four-cell and V3B local-main matrix sequentially on one GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_biomaster_selectivity_v3 import write_json


def source_fingerprints():
    paths = sorted(set((ROOT / "biomaster").glob("*v3*.py")) | {ROOT / "biomaster/odti_v2.py"} |
                   {ROOT / "scripts" / name for name in ["train_biomaster_selectivity_v3.py", "prepare_biomaster_selectivity_v3.py"]})
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_jobs(args):
    variants = [("D_support_rank", "both", 1., False), ("A_global_pair", "none", 0., False),
                ("B_support_pair", "both", 0., False), ("C_global_rank", "none", 1., False)]
    if not args.skip_local:
        variants += [("E_local_rank", "none", 1., True), ("F_local_support_rank", "both", 1., True)]
    jobs = []
    # Complete the V3A cells across seeds before evaluating the local-main replacement.
    for name, support, rank, local in variants:
        for seed in args.seeds:
            out = args.out / name / f"seed_{seed}"
            command = [sys.executable, "-u", str(ROOT / "scripts/train_biomaster_selectivity_v3.py"),
                       "--out", str(out), "--cache-dir", str(args.cache_dir), "--seed", str(seed),
                       "--epochs", str(args.epochs), "--support", support, "--rank-weight", str(rank),
                       "--batch-size", "512", "--micro-batch-size", "64" if local else "256",
                       "--eval-batch-size", "64" if local else "256", "--precision", "fp32",
                       "--dense-test-drugs", str(args.dense_test_drugs), "--log-every", "50"]
            if local:
                command.extend(["--local", "--local-pocket-store", str(ROOT / "outputs/biomaster_v3_20260905/local_pockets/POCKET_REGIONS_V3.json")])
            jobs.append({"name": name, "seed": seed, "out": str(out), "command": command,
                         "status": "pending", "log": str(out / "TRAINING_V3.log")})
    return jobs


def collect_results(jobs, out):
    import pandas as pd
    rows = []
    exposures = {}
    for job in jobs:
        path = Path(job["out"]) / "RUN_SUMMARY_V3.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        row = {"variant": job["name"], "seed": job["seed"], "selected_epoch": result["selected_epoch"],
               "validation_drug_macro_ap": result["best_validation_drug_macro_ap"],
               "test_measured_drug_macro_ap": result["test"]["drug_macro_ap"],
               "test_measured_micro_ap": result["test"]["micro_ap"],
               "peak_gpu_allocated_gib": result["peak_gpu_allocated_gib"],
               "seconds": result["training_and_evaluation_seconds"]}
        for name, value in (result.get("dense_test") or {}).items():
            if name.startswith("macro_"):
                row[f"test_dense_{name}"] = value
        comparison_path = path.parent / "DENSE_BASELINE_COMPARISON_V3.json"
        if comparison_path.exists():
            row["dense_baseline_comparison"] = str(comparison_path)
        rows.append(row)
        history = json.loads((path.parent / "TRAINING_HISTORY_V3.json").read_text())
        hashes = tuple(x["exposure_sha256"] for x in history)
        key = (job["seed"], len(history))
        if key in exposures and exposures[key] != hashes:
            raise RuntimeError("paired experiments used different training relation/query exposure")
        exposures[key] = hashes
    if rows:
        pd.DataFrame(rows).to_csv(out / "EXPERIMENT_RESULTS_V3.csv", index=False)
    return rows


def evaluate_completed(out):
    """Report matched baselines after selection; never feed test scores back."""
    if not (out / "DENSE_TEST_PREDICTIONS_V3.npz").exists():
        return
    if (out / "DENSE_BASELINE_COMPARISON_V3.json").exists():
        return
    with (out / "BASELINE_EVALUATION_V3.log").open("a") as log:
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_biomaster_v3_run.py"),
                        "--run-dir", str(out), "--save-scores"], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "EXPERIMENT_MANIFEST_V3.json"
    current_source = source_fingerprints()
    if manifest.exists():
        if not args.resume:
            raise FileExistsError("experiment matrix already exists; use --resume")
        saved = json.loads(manifest.read_text())
        if saved["source_sha256"] != current_source:
            raise ValueError("source changed since matrix launch; use a new experiment directory")
        if saved["epochs"] != args.epochs or saved["seeds"] != args.seeds or saved["skip_local"] != args.skip_local:
            raise ValueError("experiment matrix configuration changed")
        if saved["cache_dir"] != str(args.cache_dir.resolve()) or saved["dense_test_drugs"] != args.dense_test_drugs:
            raise ValueError("experiment data snapshot or dense panel configuration changed")
        jobs = saved["jobs"]
    else:
        jobs = build_jobs(args)
        write_json(manifest, {"protocol": "V3A_SUPPORT_X_D2T_PLUS_V3B_LOCAL_MAIN",
                   "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": current_source,
                   "epochs": args.epochs, "seeds": args.seeds, "skip_local": args.skip_local,
                   "cache_dir": str(args.cache_dir.resolve()), "dense_test_drugs": args.dense_test_drugs,
                   "selection": "validation measured drug-macro AP; no test-based job selection",
                   "jobs": jobs, "production_promotion": False})
    process = None

    def save_status(status, active=None, error=None):
        write_json(args.out / "EXPERIMENT_STATUS_V3.json", {"status": status, "pid": os.getpid(),
                   "updated_utc": datetime.now(timezone.utc).isoformat(), "active": active,
                   "completed_jobs": sum(j["status"] == "completed" for j in jobs), "total_jobs": len(jobs),
                   "jobs": jobs, "error": error})

    def stop(signum, frame):
        if process is not None and process.poll() is None:
            process.terminate()
        save_status("stopped", error=f"signal {signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for job in jobs:
            out = Path(job["out"])
            if (out / "RUN_SUMMARY_V3.json").exists():
                evaluate_completed(out)
                job["status"] = "completed"
                continue
            if source_fingerprints() != current_source:
                raise RuntimeError("training source changed during the matrix; stopping before next job")
            out.mkdir(parents=True, exist_ok=True)
            command = list(job["command"])
            if (out / "LAST_MODEL_V3.pt").exists():
                command.append("--resume")
            job["status"] = "running"
            print(json.dumps({"event": "launch", "variant": job["name"], "seed": job["seed"], "out": str(out)}), flush=True)
            with open(job["log"], "a", buffering=1) as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = process.pid
                save_status("running", active={"name": job["name"], "seed": job["seed"], "pid": process.pid})
                code = process.wait()
            if code:
                job["status"] = "failed"
                job["exit_code"] = code
                raise RuntimeError(f"{job['name']} seed {job['seed']} failed with exit {code}; see {job['log']}")
            job["status"] = "completed"
            evaluate_completed(out)
            collect_results(jobs, args.out)
            save_status("running")
        collect_results(jobs, args.out)
        save_status("completed")
        print(json.dumps({"event": "matrix_completed", "jobs": len(jobs)}), flush=True)
    except Exception as exc:
        save_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/biomaster_v3_20260905/experiments")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/biomaster_v3_20260905/data")
    p.add_argument("--seeds", type=int, nargs="+", default=[20260905, 20260906])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--dense-test-drugs", type=int, default=64)
    p.add_argument("--skip-local", action="store_true")
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())

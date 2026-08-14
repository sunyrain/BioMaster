#!/usr/bin/env python3
"""Run prepared AMBER screening systems across local CUDA devices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--equilibration-ns", type=float, default=0.5)
    parser.add_argument("--production-ns", type=float, default=5.0)
    parser.add_argument("--report-ps", type=float, default=20.0)
    parser.add_argument("--seed-base", type=int, default=20260840)
    parser.add_argument("--rank", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-name", default="openmm_5ns")
    return parser.parse_args()


def run_one(row: dict[str, object], gpu: str, args: argparse.Namespace) -> dict[str, object]:
    workdir = Path(str(row["workdir"]))
    system = workdir / "system"
    output = workdir / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "run_metadata.json"
    final_checkpoint = output / "final.chk"
    status_path = output / "queue_status.json"
    if metadata.is_file() and final_checkpoint.is_file():
        previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
        result = {
            **previous,
            "status": "SUCCESS",
            "pair_id": row["pair_id"],
            "drug": row.get("drug", ""),
            "target": row.get("target", ""),
            "md_screen_rank": int(row["md_screen_rank"]),
            "gpu": gpu,
            "workdir": str(workdir),
            "output_dir": str(output),
            "skipped": True,
        }
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    prmtop = system / "complex.prmtop"
    inpcrd = system / "complex.inpcrd"
    if not prmtop.is_file() or not inpcrd.is_file() or not prmtop.stat().st_size or not inpcrd.stat().st_size:
        result = {
            "status": "TECHNICAL_INCOMPLETE", "pair_id": row["pair_id"], "gpu": gpu,
            "workdir": str(workdir), "error": "missing_or_empty_prmtop_inpcrd",
        }
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        return result

    command = [
        str(ROOT / ".conda_envs/md_openmm/bin/python"),
        str(ROOT / "md/run_openmm_amber.py"),
        "--prmtop", str(prmtop), "--inpcrd", str(inpcrd),
        "--output-dir", str(output), "--device-index", "0",
        "--seed", str(args.seed_base + int(row["md_screen_rank"])),
        "--equilibration-ns", str(args.equilibration_ns),
        "--production-ns", str(args.production_ns),
        "--report-ps", str(args.report_ps),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.time()
    with (output / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    result = {
        "status": "SUCCESS" if completed.returncode == 0 and metadata.is_file() else "FAILED",
        "pair_id": row["pair_id"], "drug": row.get("drug", ""),
        "target": row.get("target", ""), "md_screen_rank": int(row["md_screen_rank"]),
        "gpu": gpu, "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "workdir": str(workdir), "output_dir": str(output), "skipped": False,
    }
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    args.preparation_root = args.preparation_root.resolve()
    summary_path = args.preparation_root / "preparation_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path, low_memory=False)
    ready = summary[summary["status"].eq("READY")].sort_values("md_screen_rank").copy()
    if args.rank:
        ready = ready[ready["md_screen_rank"].isin(args.rank)].copy()
    if args.limit > 0:
        ready = ready.head(args.limit).copy()
    if ready.empty:
        raise RuntimeError("No READY MD system is available")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    queues = [[] for _ in gpus]
    for index, row in enumerate(ready.to_dict(orient="records")):
        queues[index % len(gpus)].append(row)

    results: list[dict[str, object]] = []
    def worker(gpu: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        local = []
        for row in rows:
            result = run_one(row, gpu, args)
            local.append(result)
            print(
                f"gpu={gpu} rank={row['md_screen_rank']} pair={row['pair_id']} "
                f"status={result['status']}", flush=True
            )
        return local

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, rows) for gpu, rows in zip(gpus, queues)]
        for future in as_completed(futures):
            results.extend(future.result())
    results = sorted(results, key=lambda item: int(item.get("md_screen_rank", 999999)))
    summary_stem = f"{args.output_name}_queue_summary"
    pd.DataFrame(results).to_csv(args.preparation_root / f"{summary_stem}.csv", index=False)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_ready_systems": len(ready),
        "status_counts": pd.Series([row["status"] for row in results]).value_counts().to_dict(),
        "gpus": gpus,
        "equilibration_ns": args.equilibration_ns,
        "production_ns": args.production_ns,
        "output_name": args.output_name,
        "requested_ranks": args.rank,
        "limit": args.limit,
    }
    (args.preparation_root / f"{summary_stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

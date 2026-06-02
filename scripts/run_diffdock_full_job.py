from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


CONFIDENCE_RE = re.compile(r"rank1_confidence([-+]?\d+(?:\.\d+)?)\.sdf$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_job(job_index: Path, job_id: int) -> dict[str, str]:
    for row in read_csv(job_index):
        if int(row["job_id"]) == job_id:
            return row
    raise ValueError(f"job_id not found in {job_index}: {job_id}")


def output_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_job_lock(lock_path: Path, job_id: int) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "job_id": job_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        ensure_ascii=False,
    )
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                pid = int(lock_data.get("pid", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                pid = 0
            if pid and not process_alive(pid):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            return False
        try:
            os.write(fd, (payload + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return True
    return False


def release_job_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def write_diffdock_config(
    default_config: Path,
    destination: Path,
    samples_per_complex: int,
    inference_steps: int,
    actual_steps: int,
    batch_size: int,
) -> None:
    with default_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "samples_per_complex": samples_per_complex,
            "inference_steps": inference_steps,
            "actual_steps": actual_steps,
            "batch_size": batch_size,
            "model_dir": "./workdir/v1.1/score_model",
            "confidence_model_dir": "./workdir/v1.1/confidence_model",
            "ckpt": "best_ema_inference_epoch_model.pt",
            "confidence_ckpt": "best_model_epoch75.pt",
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def parse_confidence_file(complex_dir: Path) -> tuple[float | None, Path | None]:
    matches: list[tuple[float, Path]] = []
    for path in complex_dir.glob("rank1_confidence*.sdf"):
        match = CONFIDENCE_RE.match(path.name)
        if match:
            matches.append((float(match.group(1)), path))
    if not matches:
        return None, None
    matches.sort(key=lambda item: item[1].name)
    return matches[0]


def collect_scores(
    chunk_csv: Path,
    out_dir: Path,
    score_csv: Path,
    source_chunk: str,
    sdf_retention: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for input_row in read_csv(chunk_csv):
        complex_name = input_row["complex_name"]
        complex_dir = out_dir / complex_name
        rank1_sdf = complex_dir / "rank1.sdf"
        confidence, confidence_sdf = parse_confidence_file(complex_dir)

        status = "completed" if confidence is not None else "missing_output"
        retained_rank1 = rank1_sdf if rank1_sdf.exists() else None
        retained_confidence = confidence_sdf if confidence_sdf and confidence_sdf.exists() else None

        if sdf_retention == "rank1_confidence":
            if rank1_sdf.exists():
                rank1_sdf.unlink()
                retained_rank1 = None
        elif sdf_retention == "none":
            if rank1_sdf.exists():
                rank1_sdf.unlink()
            if confidence_sdf and confidence_sdf.exists():
                confidence_sdf.unlink()
            retained_rank1 = None
            retained_confidence = None

        rows.append(
            {
                "pair_id": complex_name,
                "complex_name": complex_name,
                "diffdock_confidence": "" if confidence is None else f"{confidence:.2f}",
                "docking_score": "",
                "rank1_sdf_path": "" if retained_rank1 is None else str(retained_rank1),
                "confidence_sdf_path": "" if retained_confidence is None else str(retained_confidence),
                "source_chunk": source_chunk,
                "status": status,
                "error": "" if status == "completed" else "rank1_confidence_sdf_missing",
            }
        )

    fieldnames = [
        "pair_id",
        "complex_name",
        "diffdock_confidence",
        "docking_score",
        "rank1_sdf_path",
        "confidence_sdf_path",
        "source_chunk",
        "status",
        "error",
    ]
    write_csv(score_csv, fieldnames, rows)

    completed = sum(1 for row in rows if row["status"] == "completed")
    summary = {
        "score_csv": str(score_csv),
        "rows": len(rows),
        "completed": completed,
        "missing_output": len(rows) - completed,
        "sdf_retention": sdf_retention,
    }
    score_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def run_job(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    job_index = resolve(root, args.job_index)
    job = load_job(job_index, args.job_id)

    chunk_csv = resolve(root, job["chunk_csv"])
    out_dir = resolve(root, job["out_dir"])
    score_csv = resolve(root, job["score_csv"])
    log_file = resolve(root, job["log_file"])
    diffdock_dir = resolve(root, args.diffdock_dir)
    config_path = log_file.with_suffix(".config.yaml")
    lock_path = score_csv.parent / f"{score_csv.name}.lock"

    expected_rows = int(job["row_count"])
    if not args.force and output_count(score_csv) >= expected_rows:
        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "status": "already_scored",
                    "score_csv": str(score_csv),
                    "rows": expected_rows,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if not acquire_job_lock(lock_path, args.job_id):
        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "status": "already_running",
                    "lock_file": str(lock_path),
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        if not args.force and output_count(score_csv) >= expected_rows:
            print(
                json.dumps(
                    {
                        "job_id": args.job_id,
                        "status": "already_scored",
                        "score_csv": str(score_csv),
                        "rows": expected_rows,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        free_gb = shutil.disk_usage(root).free / (1024**3)
        if free_gb < args.min_free_gb:
            print(
                json.dumps(
                    {
                        "job_id": args.job_id,
                        "status": "blocked_low_disk",
                        "free_gb": round(free_gb, 2),
                        "min_free_gb": args.min_free_gb,
                    },
                    ensure_ascii=False,
                )
            )
            return 75

        write_diffdock_config(
            default_config=diffdock_dir / "default_inference_args.yaml",
            destination=config_path,
            samples_per_complex=args.samples_per_complex,
            inference_steps=args.inference_steps,
            actual_steps=args.actual_steps,
            batch_size=args.batch_size,
        )

        command = [
            sys.executable,
            "inference.py",
            "--config",
            str(config_path),
            "--protein_ligand_csv",
            str(chunk_csv),
            "--out_dir",
            str(out_dir),
            "--loglevel",
            args.loglevel,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        env.setdefault("MKL_NUM_THREADS", str(args.omp_threads))
        env.setdefault("OPENBLAS_NUM_THREADS", str(args.omp_threads))

        log_file.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with log_file.open("w", encoding="utf-8") as log_handle:
            log_handle.write(" ".join(command) + "\n")
            log_handle.flush()
            process = subprocess.run(
                command,
                cwd=diffdock_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed_seconds = round(time.time() - start, 2)

        summary = collect_scores(
            chunk_csv=chunk_csv,
            out_dir=out_dir,
            score_csv=score_csv,
            source_chunk=str(chunk_csv),
            sdf_retention=args.sdf_retention,
        )
        summary.update(
            {
                "job_id": args.job_id,
                "returncode": process.returncode,
                "elapsed_seconds": elapsed_seconds,
                "log_file": str(log_file),
                "cuda_device": str(args.cuda_device),
            }
        )
        print(json.dumps(summary, ensure_ascii=False))
        return process.returncode
    finally:
        release_job_lock(lock_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one full DiffDock chunk and collect rank1 confidence scores.")
    parser.add_argument("--job-index", default="outputs/report_scale/diffdock_full_run/diffdock_full_job_index.csv")
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--diffdock-dir", default="third_party/DiffDock")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--samples-per-complex", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--actual-steps", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sdf-retention", choices=["all", "rank1_confidence", "none"], default="rank1_confidence")
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--loglevel", default="INFO")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())

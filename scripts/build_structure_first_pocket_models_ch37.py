#!/usr/bin/env python3
"""Model the recovered no-pocket targets that lack exact AF structures.

Long proteins are covered by overlapping sequence windows.  The resulting
Boltz-2 monomer structures are fragment hypotheses, not replacements for an
experimentally validated full-length state.  P2Rank and fpocket results are
mapped back to canonical target residue numbers so downstream routing can keep
the evidence boundary explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
EXCLUSION_AUDIT = (
    ROOT / "outputs/final_target_package_ch37/FINAL_TARGET_EXCLUSION_AUDIT_888.csv"
)
OUTDIR = (
    ROOT
    / "outputs/no_experimental_pocket_prediction_ch37_v1/structure_first_models_v1"
)
BOLTZ = ROOT / ".conda_envs/boltz2/bin/boltz"
BOLTZ_CACHE = Path("/root/autodl-tmp/boltz_cache")
P2RANK = ROOT / "tools/p2rank_2.6-alpha/prank"
POCKET_ENV = ROOT / ".conda_envs/pocket_tools"
FPOCKET = POCKET_ENV / "bin/fpocket"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def window_ranges(length: int, window: int, overlap: int) -> list[tuple[int, int]]:
    if length <= window:
        return [(1, length)]
    if not 0 < overlap < window:
        raise ValueError("Overlap must be positive and smaller than the window")
    count = math.ceil((length - window) / (window - overlap)) + 1
    last_start = length - window
    starts = [round(index * last_start / (count - 1)) for index in range(count)]
    ranges = [(start + 1, start + window) for start in starts]
    if ranges[0][0] != 1 or ranges[-1][1] != length:
        raise RuntimeError("Fragment windows do not cover the sequence termini")
    for left, right in zip(ranges, ranges[1:]):
        if right[0] > left[1] + 1:
            raise RuntimeError("Fragment windows contain a sequence gap")
    return ranges


def write_yaml(path: Path, sequence: str) -> None:
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "- protein:\n"
        "    id: A\n"
        f"    sequence: {sequence}\n"
        "    msa: empty\n",
        encoding="utf-8",
    )


def prepare(outdir: Path, window: int, overlap: int) -> None:
    universe = pd.read_csv(UNIVERSE, low_memory=False)
    exclusion = pd.read_csv(EXCLUSION_AUDIT, low_memory=False)
    recovered = exclusion[
        exclusion["first_exclusion_reason"].eq("EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET")
    ]
    if len(recovered) != 46:
        raise RuntimeError(f"Expected 46 recovered no-pocket targets, observed {len(recovered)}")
    targets = universe[
        universe["target_chembl_id"].isin(recovered["target_chembl_id"])
        & ~universe["af_exact_sequence_model"].fillna(False).astype(bool)
    ].copy()
    if len(targets) != 2:
        raise RuntimeError(f"Expected two recovered targets without exact AF structures, observed {len(targets)}")
    inputs = outdir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rank = 0
    for target in targets.sort_values("target_chembl_id").to_dict(orient="records"):
        canonical = clean(target["sequence"])
        if len(canonical) != int(target["sequence_length"]):
            raise RuntimeError(f"Sequence length mismatch for {target['target_chembl_id']}")
        model_sequence = canonical.replace("U", "C")
        substitutions = ";".join(
            f"U{index + 1}C_FOR_STRUCTURE_MODEL"
            for index, residue in enumerate(canonical)
            if residue == "U"
        )
        for fragment_index, (start, end) in enumerate(
            window_ranges(len(model_sequence), window, overlap), start=1
        ):
            rank += 1
            stem = (
                f"{target['target_chembl_id']}_{target['gene_symbol']}_"
                f"{start:04d}_{end:04d}"
            )
            yaml_path = inputs / f"{stem}.yaml"
            fragment_sequence = model_sequence[start - 1 : end]
            write_yaml(yaml_path, fragment_sequence)
            yaml_hash = file_sha256(yaml_path)
            signature = payload_sha256(
                {
                    "target": target["target_chembl_id"],
                    "canonical_sequence_sha256": target["sequence_sha256"],
                    "fragment_start": start,
                    "fragment_end": end,
                    "model_sequence": fragment_sequence,
                    "substitutions": substitutions,
                    "model": "boltz2",
                    "msa": "empty",
                }
            )
            rows.append(
                {
                    "externalQueueRank": rank,
                    "pairId": stem,
                    "target_chembl_id": target["target_chembl_id"],
                    "gene_symbol": target["gene_symbol"],
                    "uniprot_accession": target["uniprot_accession"],
                    "target_class_l1": target["target_class_l1"],
                    "assay_lane": target["assay_lane"],
                    "evidence_class": target["evidence_class"],
                    "canonical_sequence_length": len(canonical),
                    "canonical_sequence_sha256": target["sequence_sha256"],
                    "fragment_index": fragment_index,
                    "fragment_start": start,
                    "fragment_end": end,
                    "fragment_length": len(fragment_sequence),
                    "fragment_overlap_target": overlap,
                    "model_sequence_substitutions": substitutions,
                    "yamlFile": str(yaml_path.resolve()),
                    "yamlSha256": yaml_hash,
                    "inputSignatureSha256": signature,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest_path = outdir / "STRUCTURE_FIRST_FRAGMENT_MANIFEST_V1.csv"
    manifest.to_csv(manifest_path, index=False)
    by_target = manifest.groupby(["target_chembl_id", "gene_symbol"], as_index=False).agg(
        fragments=("pairId", "count"),
        sequence_length=("canonical_sequence_length", "first"),
        min_start=("fragment_start", "min"),
        max_end=("fragment_end", "max"),
    )
    write_json(
        outdir / "STRUCTURE_FIRST_FRAGMENT_PREPARATION_SUMMARY_V1.json",
        {
            "created_utc": utc_now(),
            "status": "PASS",
            "targets": int(manifest["target_chembl_id"].nunique()),
            "fragments": int(len(manifest)),
            "window_length": window,
            "target_overlap": overlap,
            "target_fragment_counts": {
                row.gene_symbol: int(row.fragments) for row in by_target.itertuples(index=False)
            },
            "nonstandard_residue_policy": "selenocysteine U is represented by C for structure modelling and recorded explicitly",
            "msa_policy": "explicit empty MSA; fragment models are hypothesis-level structural evidence",
            "manifest": str(manifest_path.resolve()),
        },
    )


def find_prediction(output: Path, stem: str) -> tuple[Path | None, Path | None]:
    structures = list(output.rglob(f"{stem}_model_0.pdb"))
    confidences = list(output.rglob(f"confidence_{stem}_model_0.json"))
    return (
        structures[0] if len(structures) == 1 and structures[0].stat().st_size > 0 else None,
        confidences[0] if len(confidences) == 1 and confidences[0].stat().st_size > 0 else None,
    )


def run_boltz_one(
    row: dict[str, Any],
    gpu: str,
    outdir: Path,
    sampling_steps: int,
    recycling_steps: int,
    force: bool,
) -> dict[str, Any]:
    stem = str(row["pairId"])
    output = outdir / "boltz_runs" / stem
    output.mkdir(parents=True, exist_ok=True)
    structure, confidence = find_prediction(output, stem)
    if structure and confidence and not force:
        return {
            "pairId": stem,
            "target_chembl_id": row["target_chembl_id"],
            "status": "REUSED_COMPLETE",
            "return_code": 0,
            "gpu": gpu,
            "elapsed_seconds": 0.0,
            "structure_path": str(structure.resolve()),
            "confidence_path": str(confidence.resolve()),
        }
    command = [
        str(BOLTZ),
        "predict",
        str(row["yamlFile"]),
        "--out_dir",
        str(output),
        "--cache",
        str(BOLTZ_CACHE),
        "--model",
        "boltz2",
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--recycling_steps",
        str(recycling_steps),
        "--sampling_steps",
        str(sampling_steps),
        "--diffusion_samples",
        "1",
        "--seed",
        str(20260890 + int(row["externalQueueRank"])),
        "--num_workers",
        "0",
        "--preprocessing-threads",
        "1",
        "--output_format",
        "pdb",
        "--no_kernels",
    ]
    if force:
        command.append("--override")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "BOLTZ_CACHE": str(BOLTZ_CACHE),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TMPDIR": str((outdir / "tmp" / f"gpu_{gpu}").resolve()),
        }
    )
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    log = outdir / "logs" / f"{stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {utc_now()} gpu={gpu} ===\n")
        handle.write(" ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    structure, confidence = find_prediction(output, stem)
    status = "SUCCESS" if completed.returncode == 0 and structure and confidence else "FAILED"
    return {
        "pairId": stem,
        "target_chembl_id": row["target_chembl_id"],
        "status": status,
        "return_code": int(completed.returncode),
        "gpu": gpu,
        "elapsed_seconds": round(time.time() - started, 3),
        "structure_path": str(structure.resolve()) if structure else "",
        "confidence_path": str(confidence.resolve()) if confidence else "",
        "log_path": str(log.resolve()),
    }


def boltz_worker(
    gpu: str,
    queue: mp.Queue,
    result_queue: mp.Queue,
    outdir: str,
    sampling_steps: int,
    recycling_steps: int,
    force: bool,
) -> None:
    while True:
        row = queue.get()
        if row is None:
            return
        try:
            result = run_boltz_one(
                row,
                gpu,
                Path(outdir),
                sampling_steps,
                recycling_steps,
                force,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "pairId": row["pairId"],
                "target_chembl_id": row["target_chembl_id"],
                "status": "FAILED_EXCEPTION",
                "return_code": -1,
                "gpu": gpu,
                "elapsed_seconds": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result_queue.put(result)


def run_boltz(
    outdir: Path,
    gpus: list[str],
    sampling_steps: int,
    recycling_steps: int,
    force: bool,
    targets: list[str],
    limit: int,
) -> None:
    if not BOLTZ.is_file() or not BOLTZ_CACHE.is_dir():
        raise FileNotFoundError("Boltz executable or model cache is missing")
    manifest = pd.read_csv(outdir / "STRUCTURE_FIRST_FRAGMENT_MANIFEST_V1.csv")
    if targets:
        manifest = manifest[
            manifest["target_chembl_id"].isin(targets)
            | manifest["gene_symbol"].isin(targets)
        ].copy()
    if limit > 0:
        manifest = manifest.head(limit).copy()
    records = manifest.to_dict(orient="records")
    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    for row in records:
        task_queue.put(row)
    processes: list[mp.Process] = []
    for gpu in gpus:
        task_queue.put(None)
        process = mp.Process(
            target=boltz_worker,
            args=(
                gpu,
                task_queue,
                result_queue,
                str(outdir),
                sampling_steps,
                recycling_steps,
                force,
            ),
        )
        process.start()
        processes.append(process)
    results: list[dict[str, Any]] = []
    for completed_count in range(1, len(records) + 1):
        result = result_queue.get()
        results.append(result)
        print(
            f"Boltz fragment {completed_count}/{len(records)} "
            f"{result['pairId']} {result['status']}",
            flush=True,
        )
    for process in processes:
        process.join()
    current = pd.DataFrame(results)
    log_path = outdir / "STRUCTURE_FIRST_BOLTZ_RUN_LOG_V1.csv"
    if log_path.is_file() and (targets or limit > 0):
        previous = pd.read_csv(log_path)
        previous = previous[~previous["pairId"].isin(current["pairId"])]
        current = pd.concat([previous, current], ignore_index=True)
    current.sort_values("pairId").to_csv(log_path, index=False)
    write_json(
        outdir / "STRUCTURE_FIRST_BOLTZ_RUN_SUMMARY_V1.json",
        {
            "created_utc": utc_now(),
            "requested_this_run": len(records),
            "status_counts_this_run": pd.DataFrame(results)["status"].value_counts().to_dict(),
            "total_logged_fragments": int(len(current)),
            "total_completed_fragments": int(
                current["status"].isin(["SUCCESS", "REUSED_COMPLETE"]).sum()
            ),
            "sampling_steps": sampling_steps,
            "recycling_steps": recycling_steps,
            "msa": "empty",
            "gpus": gpus,
        },
    )


def run_fpocket_one(task: tuple[str, str]) -> dict[str, Any]:
    pair_id, input_text = task
    input_path = Path(input_text)
    output = input_path.parent / f"{input_path.stem}_out"
    info = output / f"{input_path.stem}_info.txt"
    if info.is_file():
        return {"pairId": pair_id, "status": "REUSED_COMPLETE", "return_code": 0}
    environment = os.environ.copy()
    environment["PATH"] = str(POCKET_ENV / "bin") + os.pathsep + environment.get("PATH", "")
    environment["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [str(FPOCKET), "-f", input_path.name],
        cwd=input_path.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=900,
    )
    return {
        "pairId": pair_id,
        "status": "SUCCESS" if completed.returncode == 0 and info.is_file() else "FAILED",
        "return_code": int(completed.returncode),
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
    }


def run_pockets(outdir: Path, threads: int, workers: int) -> None:
    run_log = pd.read_csv(outdir / "STRUCTURE_FIRST_BOLTZ_RUN_LOG_V1.csv")
    completed = run_log[run_log["status"].isin(["SUCCESS", "REUSED_COMPLETE"])].copy()
    if completed.empty:
        raise RuntimeError("No completed Boltz fragment structures")
    pocket_inputs = outdir / "pocket_inputs"
    p2rank_output = outdir / "p2rank"
    pocket_inputs.mkdir(parents=True, exist_ok=True)
    p2rank_output.mkdir(parents=True, exist_ok=True)
    input_rows: list[dict[str, Any]] = []
    for row in completed.to_dict(orient="records"):
        source = Path(row["structure_path"])
        destination = pocket_inputs / f"{row['pairId']}.pdb"
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != source.resolve():
                destination.unlink()
        if not destination.exists():
            destination.symlink_to(source.resolve())
        input_rows.append(
            {
                "pairId": row["pairId"],
                "structure_path": str(source.resolve()),
                # Keep the output-local symlink path.  Resolving it here makes
                # both tools write beside the Boltz source and breaks the
                # package-local parser/provenance boundary.
                "pocket_input_path": str(destination.absolute()),
                "structure_sha256": file_sha256(source),
            }
        )
    pocket_manifest = pd.DataFrame(input_rows)
    pocket_manifest.to_csv(outdir / "STRUCTURE_FIRST_POCKET_INPUT_MANIFEST_V1.csv", index=False)

    expected = [p2rank_output / f"{Path(row['pocket_input_path']).name}_predictions.csv" for row in input_rows]
    missing = [row for row, path in zip(input_rows, expected) if not path.is_file()]
    p2rank_return_code = 0
    if missing:
        dataset = outdir / "STRUCTURE_FIRST_P2RANK_PENDING_V1.ds"
        dataset.write_text(
            "\n".join(row["pocket_input_path"] for row in missing) + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["PATH"] = str(POCKET_ENV / "bin") + os.pathsep + environment.get("PATH", "")
        command = [
            str(P2RANK),
            "predict",
            "-c",
            "alphafold",
            "-threads",
            str(threads),
            "-visualizations",
            "0",
            "-o",
            str(p2rank_output),
            str(dataset),
        ]
        completed_p2 = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        p2rank_return_code = int(completed_p2.returncode)
        (outdir / "STRUCTURE_FIRST_P2RANK_RUN_V1.log").write_text(
            completed_p2.stdout, encoding="utf-8"
        )

    tasks = [(row["pairId"], row["pocket_input_path"]) for row in input_rows]
    fpocket_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_fpocket_one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            fpocket_results.append(future.result())
            if count % 5 == 0 or count == len(futures):
                print(f"fragment fpocket {count}/{len(futures)}", flush=True)
    fpocket_log = pd.DataFrame(fpocket_results).sort_values("pairId")
    fpocket_log.to_csv(outdir / "STRUCTURE_FIRST_FPOCKET_RUN_LOG_V1.csv", index=False)
    p2rank_present = sum(path.is_file() for path in expected)
    fpocket_completed = int(
        fpocket_log["status"].isin(["SUCCESS", "REUSED_COMPLETE"]).sum()
    )
    write_json(
        outdir / "STRUCTURE_FIRST_POCKET_RUN_SUMMARY_V1.json",
        {
            "created_utc": utc_now(),
            "structures": len(input_rows),
            "p2rank_predictions_present": p2rank_present,
            "p2rank_return_code": p2rank_return_code,
            "fpocket_completed": fpocket_completed,
            "status": "PASS" if p2rank_present == len(input_rows) and fpocket_completed == len(input_rows) else "FAIL",
        },
    )


def parse_fpocket_info(path: Path) -> list[dict[str, Any]]:
    pockets: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    if not path.is_file():
        return pockets
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Pocket ") and line.endswith(":"):
            if current:
                pockets.append(current)
            current = {"rank": int(line.replace("Pocket", "").replace(":", "").strip())}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = (
            key.strip().lower().replace(".", "").replace("-", "").replace(" ", "_")
        )
        try:
            current[key] = float(value.strip())
        except ValueError:
            current[key] = value.strip()
    if current:
        pockets.append(current)
    return pockets


def fpocket_residues(path: Path, offset: int) -> tuple[list[int], list[int]]:
    local: set[int] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                local.add(int(line[22:26]))
            except ValueError:
                continue
    return sorted(local), sorted(residue + offset for residue in local)


def p2rank_rows(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        frame = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")
    except pd.errors.ParserError:
        return []
    frame.columns = [str(column).strip() for column in frame.columns]
    if "rank" not in frame.columns:
        return []
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame = frame[frame["rank"].notna()]
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        local = []
        for token in clean(row.get("residue_ids")).split():
            try:
                local.append(int(token.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        rows.append(
            {
                "rank": int(row["rank"]),
                "score": pd.to_numeric(row.get("score"), errors="coerce"),
                "probability": pd.to_numeric(row.get("probability"), errors="coerce"),
                "local_residue_ids": sorted(set(local)),
                "canonical_residue_ids": sorted({residue + offset for residue in local}),
            }
        )
    return rows


def mean_ca_bfactor(path: Path) -> float | None:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            try:
                values.append(float(line[60:66]))
            except ValueError:
                continue
    return sum(values) / len(values) if values else None


def parse(outdir: Path) -> None:
    manifest = pd.read_csv(outdir / "STRUCTURE_FIRST_FRAGMENT_MANIFEST_V1.csv")
    boltz_log = pd.read_csv(outdir / "STRUCTURE_FIRST_BOLTZ_RUN_LOG_V1.csv")
    merged = manifest.merge(
        boltz_log[["pairId", "status", "structure_path", "confidence_path"]],
        on="pairId",
        how="left",
        validate="one_to_one",
    )
    candidate_rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        completed = clean(row.get("status")) in {"SUCCESS", "REUSED_COMPLETE"}
        structure = Path(clean(row.get("structure_path"))) if completed else None
        input_path = outdir / "pocket_inputs" / f"{row['pairId']}.pdb"
        p2_path = outdir / "p2rank" / f"{input_path.name}_predictions.csv"
        offset = int(row["fragment_start"]) - 1
        p2rows = p2rank_rows(p2_path, offset) if completed else []
        fp_output = input_path.parent / f"{input_path.stem}_out"
        fp_info = fp_output / f"{input_path.stem}_info.txt"
        fprows = parse_fpocket_info(fp_info) if completed else []
        for pocket in p2rows:
            candidate_rows.append(
                {
                    "target_chembl_id": row["target_chembl_id"],
                    "gene_symbol": row["gene_symbol"],
                    "pairId": row["pairId"],
                    "fragment_start": row["fragment_start"],
                    "fragment_end": row["fragment_end"],
                    "method": "P2RANK",
                    "rank": pocket["rank"],
                    "score": pocket["score"],
                    "probability_or_druggability": pocket["probability"],
                    "volume_A3": None,
                    "local_residue_ids": ";".join(map(str, pocket["local_residue_ids"])),
                    "canonical_residue_ids": ";".join(map(str, pocket["canonical_residue_ids"])),
                }
            )
        for pocket in fprows:
            rank = int(pocket["rank"])
            atom_file = fp_output / "pockets" / f"pocket{rank}_atm.pdb"
            local, canonical = fpocket_residues(atom_file, offset)
            candidate_rows.append(
                {
                    "target_chembl_id": row["target_chembl_id"],
                    "gene_symbol": row["gene_symbol"],
                    "pairId": row["pairId"],
                    "fragment_start": row["fragment_start"],
                    "fragment_end": row["fragment_end"],
                    "method": "FPOCKET",
                    "rank": rank,
                    "score": pocket.get("score"),
                    "probability_or_druggability": pocket.get("druggability_score"),
                    "volume_A3": pocket.get("volume"),
                    "local_residue_ids": ";".join(map(str, local)),
                    "canonical_residue_ids": ";".join(map(str, canonical)),
                }
            )
        fragment_rows.append(
            {
                **{key: row[key] for key in manifest.columns},
                "boltz_status": row.get("status"),
                "structure_path": clean(row.get("structure_path")),
                "fragment_mean_plddt": mean_ca_bfactor(structure) if structure else None,
                "p2rank_pocket_count": len(p2rows),
                "p2rank_top_probability": p2rows[0]["probability"] if p2rows else None,
                "fpocket_pocket_count": len(fprows),
                "fpocket_max_druggability": max(
                    (float(pocket.get("druggability_score", 0) or 0) for pocket in fprows),
                    default=None,
                ),
            }
        )
    fragments = pd.DataFrame(fragment_rows)
    candidates = pd.DataFrame(candidate_rows)
    target_rows: list[dict[str, Any]] = []
    for target_id, group in fragments.groupby("target_chembl_id"):
        expected = len(group)
        completed = int(group["boltz_status"].isin(["SUCCESS", "REUSED_COMPLETE"]).sum())
        p2_positive = int(group["p2rank_pocket_count"].gt(0).sum())
        fp_positive = int(group["fpocket_pocket_count"].gt(0).sum())
        both = int(
            (group["p2rank_pocket_count"].gt(0) & group["fpocket_pocket_count"].gt(0)).sum()
        )
        mean_plddt = pd.to_numeric(
            group["fragment_mean_plddt"], errors="coerce"
        ).mean()
        if completed != expected or both == 0:
            evidence_tier = "PF4_FRAGMENT_POCKET_INCOMPLETE"
        elif mean_plddt >= 70:
            evidence_tier = "PF1_HIGH_CONFIDENCE_FRAGMENT_DUAL_METHOD"
        elif mean_plddt >= 50:
            evidence_tier = "PF2_MODERATE_CONFIDENCE_FRAGMENT_DUAL_METHOD"
        else:
            evidence_tier = "PF3_LOW_CONFIDENCE_FRAGMENT_POCKET_HYPOTHESES"
        first = group.iloc[0]
        target_rows.append(
            {
                "target_chembl_id": target_id,
                "gene_symbol": first["gene_symbol"],
                "uniprot_accession": first["uniprot_accession"],
                "target_class_l1": first["target_class_l1"],
                "assay_lane": first["assay_lane"],
                "evidence_class": first["evidence_class"],
                "canonical_sequence_length": int(first["canonical_sequence_length"]),
                "fragments_expected": expected,
                "fragments_completed": completed,
                "full_sequence_covered": bool(
                    group["fragment_start"].min() == 1
                    and group["fragment_end"].max() == first["canonical_sequence_length"]
                    and completed == expected
                ),
                "p2rank_positive_fragments": p2_positive,
                "fpocket_positive_fragments": fp_positive,
                "dual_method_positive_fragments": both,
                "mean_fragment_plddt": mean_plddt,
                "structure_strategy": "OVERLAPPING_SEQUENCE_FRAGMENT_BOLTZ2_EMPTY_MSA",
                "pocket_evidence_tier": evidence_tier,
                "next_compute_action_zh": (
                    "按规范残基编号合并重叠片段口袋；结合功能域、膜拓扑和已知机制筛选后进入局部构象ensemble"
                    if completed == expected
                    else "补齐失败片段后再进行口袋路由"
                ),
            }
        )
    targets = pd.DataFrame(target_rows).sort_values("target_chembl_id")
    fragments.to_csv(outdir / "STRUCTURE_FIRST_FRAGMENT_RESULTS_V1.csv", index=False)
    candidates.to_csv(
        outdir / "STRUCTURE_FIRST_POCKET_CANDIDATES_CANONICAL_V1.csv.gz",
        index=False,
        compression="gzip",
    )
    targets.to_csv(outdir / "STRUCTURE_FIRST_TARGET_POCKET_SUMMARY_2_V1.csv", index=False)
    all_complete = bool(
        len(targets) == 2
        and targets["full_sequence_covered"].all()
        and targets["dual_method_positive_fragments"].gt(0).all()
    )
    summary = {
        "created_utc": utc_now(),
        "status": "PASS" if all_complete else "INCOMPLETE",
        "targets": int(len(targets)),
        "fragments": int(len(fragments)),
        "completed_fragments": int(
            fragments["boltz_status"].isin(["SUCCESS", "REUSED_COMPLETE"]).sum()
        ),
        "p2rank_pocket_candidates": int(
            (candidates["method"] == "P2RANK").sum() if "method" in candidates else 0
        ),
        "fpocket_pocket_candidates": int(
            (candidates["method"] == "FPOCKET").sum() if "method" in candidates else 0
        ),
        "target_evidence_tiers": targets["pocket_evidence_tier"].value_counts().to_dict(),
        "boundaries": [
            "Fragment models provide local pocket hypotheses and do not establish a full-length conformational state.",
            "Empty-MSA Boltz-2 structures are lower-confidence than exact-sequence AFDB or experimental structures.",
            "Canonical residue mapping preserves sequence location but not inter-fragment geometry.",
        ],
    }
    write_json(outdir / "STRUCTURE_FIRST_TARGET_POCKET_SUMMARY_V1.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run-boltz", "run-pockets", "parse", "all"])
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--p2rank-threads", type=int, default=16)
    parser.add_argument("--fpocket-workers", type=int, default=8)
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    if args.mode in {"prepare", "all"}:
        prepare(outdir, args.window, args.overlap)
    if args.mode in {"run-boltz", "all"}:
        run_boltz(
            outdir,
            [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()],
            args.sampling_steps,
            args.recycling_steps,
            args.force,
            args.target,
            args.limit,
        )
    if args.mode in {"run-pockets", "all"}:
        run_pockets(outdir, args.p2rank_threads, args.fpocket_workers)
    if args.mode in {"parse", "all"}:
        parse(outdir)


if __name__ == "__main__":
    main()

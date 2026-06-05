from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


SDF_PROP_RE = re.compile(r">\s*<([^>]+)>\s*\n([^\n]*)", re.MULTILINE)
STDOUT_FIELD_PATTERNS = {
    "CNNscore": re.compile(r"CNNscore\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "CNNaffinity": re.compile(r"CNNaffinity\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    "minimizedAffinity": re.compile(r"Affinity\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
}


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def fmt(value: Any, digits: int = 2) -> str:
    parsed = number(value)
    return "NA" if parsed is None else f"{parsed:.{digits}f}"


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def abs_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def is_known(row: dict[str, Any]) -> bool:
    return truthy(row.get("knownDrugTargetPair"))


def is_novel(row: dict[str, Any]) -> bool:
    novelty = str(row.get("noveltyClass") or "").lower()
    return "new_pair" in novelty or "model_priority" in novelty


def safe_token(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return cleaned[:120] or "item"


def path_digest(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]


def resolve_binary(value: str) -> str:
    path = Path(value)
    if path.exists():
        return str(path)
    found = shutil.which(value)
    return found or value


def gnina_runtime_env(cpu_threads: int = 0, hide_cuda: bool = False) -> tuple[dict[str, str], str]:
    lib_paths = [
        "/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/usr/local/cuda/lib64",
    ]
    existing = [path for path in lib_paths if Path(path).exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    value = ":".join(existing + ([current] if current else []))
    env = os.environ.copy()
    if value:
        env["LD_LIBRARY_PATH"] = value
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if cpu_threads > 0:
        thread_value = str(cpu_threads)
        env["OMP_NUM_THREADS"] = thread_value
        env["OPENBLAS_NUM_THREADS"] = thread_value
        env["MKL_NUM_THREADS"] = thread_value
        env["VECLIB_MAXIMUM_THREADS"] = thread_value
        env["NUMEXPR_NUM_THREADS"] = thread_value
    return env, value


def check_gnina_runtime(gnina: str) -> dict[str, Any]:
    path = Path(gnina)
    resolved = str(path) if path.exists() else (shutil.which(gnina) or gnina)
    executable_exists = bool(Path(resolved).exists() and os.access(resolved, os.X_OK))
    if not executable_exists:
        return {
            "binaryAvailable": False,
            "runtimeReady": False,
            "version": "",
            "runtimeError": f"gnina executable not found or not executable: {gnina}",
            "ldLibraryPath": "",
        }
    env, ld_path = gnina_runtime_env()
    try:
        result = subprocess.run(
            [resolved, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 - audit runtime failure rather than hiding it.
        return {
            "binaryAvailable": True,
            "runtimeReady": False,
            "version": "",
            "runtimeError": f"{type(exc).__name__}: {exc}",
            "ldLibraryPath": ld_path,
        }
    version = (result.stdout or result.stderr or "").strip().splitlines()[0] if (result.stdout or result.stderr) else ""
    return {
        "binaryAvailable": True,
        "runtimeReady": result.returncode == 0,
        "version": version,
        "runtimeError": "" if result.returncode == 0 else (result.stderr or result.stdout or "")[:700],
        "ldLibraryPath": ld_path,
    }


def parse_gnina_output(output_sdf: Path, stdout: str, stderr: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if output_sdf.exists() and output_sdf.stat().st_size > 0:
        text = output_sdf.read_text(encoding="utf-8", errors="replace")
        for key, value in SDF_PROP_RE.findall(text):
            clean_key = key.strip()
            clean_value = value.strip()
            if clean_key in {"CNNscore", "CNNaffinity", "CNN_VS", "minimizedAffinity"}:
                metrics[clean_key] = clean_value
    combined = "\n".join([stdout or "", stderr or ""])
    for key, pattern in STDOUT_FIELD_PATTERNS.items():
        if key not in metrics:
            match = pattern.search(combined)
            if match:
                metrics[key] = match.group(1)
    return {
        "gninaCnnScore": round(number(metrics.get("CNNscore")) or 0.0, 6)
        if number(metrics.get("CNNscore")) is not None
        else "",
        "gninaCnnAffinity": round(number(metrics.get("CNNaffinity")) or 0.0, 6)
        if number(metrics.get("CNNaffinity")) is not None
        else "",
        "gninaMinimizedAffinity": round(number(metrics.get("minimizedAffinity")) or 0.0, 6)
        if number(metrics.get("minimizedAffinity")) is not None
        else "",
    }


def classify_gnina(row: dict[str, Any]) -> tuple[float, str, str]:
    if row.get("gninaStatus") != "ok":
        return 0.0, "D_gnina_not_scored", row.get("gninaError") or "gnina_not_scored"

    cnn_score = number(row.get("gninaCnnScore"))
    cnn_affinity = number(row.get("gninaCnnAffinity"))
    standard_tier = str(row.get("standardPoseValidationTier") or "")
    pose_quality_tier = str(row.get("poseQualityTier") or "")
    pose_supported = standard_tier.startswith(("A_", "B_")) or pose_quality_tier.startswith(("A_", "B_"))

    if cnn_score is None and cnn_affinity is None:
        return 0.0, "D_gnina_unparsed", "gnina_cnn_metrics_unparsed"

    score_component = 100.0 * max(0.0, min(1.0, cnn_score or 0.0))
    affinity_component = 0.0
    if cnn_affinity is not None:
        # GNINA CNNaffinity is a pK-like score; 5 to 9 maps to 0 to 100.
        affinity_component = max(0.0, min(100.0, ((cnn_affinity - 5.0) / 4.0) * 100.0))
    composite = 0.55 * score_component + 0.45 * affinity_component if cnn_affinity is not None else score_component

    if pose_supported and (cnn_score or 0.0) >= 0.70 and (cnn_affinity or 0.0) >= 6.5:
        return max(composite, 85.0), "A_gnina_cnn_structural_support", "pose_supported_high_cnn_score_high_cnn_affinity"
    if pose_supported and (cnn_score or 0.0) >= 0.50 and (cnn_affinity or 0.0) >= 5.5:
        return max(composite, 70.0), "B_gnina_cnn_structural_support", "pose_supported_moderate_cnn_signal"
    if (cnn_score or 0.0) >= 0.50 or (cnn_affinity or 0.0) >= 5.5:
        return max(composite, 55.0), "C_gnina_cnn_signal_pose_review", "cnn_signal_requires_pose_or_context_review"
    if pose_supported:
        return max(composite, 40.0), "C_gnina_pose_supported_weak_cnn", "pose_supported_but_cnn_signal_weak"
    return max(composite, 20.0), "D_gnina_low_support_review", "weak_cnn_signal_or_pose_validation_conflict"


def base_row(root: Path, row: pd.Series) -> dict[str, Any]:
    ligand = abs_path(root, row.get("confidenceSdfPath"))
    receptor = abs_path(root, row.get("receptorPdbPath"))
    ligand_pdbqt = abs_path(root, row.get("ligandPdbqtPath"))
    receptor_pdbqt = abs_path(root, row.get("receptorPdbqtPath"))
    input_ready = bool(ligand and ligand.exists() and receptor and receptor.exists())
    return {
        "externalQueueRank": row.get("externalQueueRank", ""),
        "direction": row.get("direction", ""),
        "pairId": row.get("pairId", ""),
        "drugId": row.get("drugId", ""),
        "drug": row.get("drug", ""),
        "target": row.get("target", ""),
        "protein": row.get("protein", ""),
        "proteinName": row.get("proteinName", ""),
        "knownDrugTargetPair": row.get("knownDrugTargetPair", ""),
        "noveltyClass": row.get("noveltyClass", ""),
        "structureConfidenceTier": row.get("structureConfidenceTier", ""),
        "targetDruggabilityTier": row.get("targetDruggabilityTier", ""),
        "poseQualityTier": row.get("poseQualityTier", ""),
        "standardPoseValidationTier": row.get("standardPoseValidationTier", ""),
        "vinaScoreKcalMol": row.get("vinaScoreKcalMol", ""),
        "vinaOptimizedScoreKcalMol": row.get("vinaOptimizedScoreKcalMol", ""),
        "vinaConsensusTier": row.get("vinaConsensusTier", ""),
        "confidenceSdfPath": str(ligand) if ligand else "",
        "receptorPdbPath": str(receptor) if receptor else "",
        "ligandPdbqtPath": str(ligand_pdbqt) if ligand_pdbqt else "",
        "receptorPdbqtPath": str(receptor_pdbqt) if receptor_pdbqt else "",
        "gninaInputReady": input_ready,
    }


def output_path(row: dict[str, Any], out_dir: Path) -> Path:
    source = Path(row["confidenceSdfPath"])
    return out_dir / "gnina_outputs" / f"{safe_token(row['pairId'])}_{path_digest(source)}.sdf"


def run_row(row: dict[str, Any], args: argparse.Namespace, gnina: str, out_dir: Path) -> dict[str, Any]:
    out_sdf = output_path(row, out_dir)
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    if not row["gninaInputReady"]:
        merged = {**row, "gninaStatus": "not_ready", "gninaError": "ligand_sdf_or_receptor_pdb_missing", "gninaOutputSdfPath": ""}
        score, tier, reason = classify_gnina(merged)
        return {**merged, "gninaConsensusScore": round(score, 4), "gninaConsensusTier": tier, "gninaConsensusReason": reason}

    if args.skip_existing and out_sdf.exists() and out_sdf.stat().st_size > 0:
        parsed = parse_gnina_output(out_sdf, "", "")
        status = "ok" if parsed.get("gninaCnnScore") != "" or parsed.get("gninaCnnAffinity") != "" else "failed"
        merged = {
            **row,
            **parsed,
            "gninaStatus": status,
            "gninaError": "" if status == "ok" else "cached_output_unparsed",
            "gninaReturnCode": 0 if status == "ok" else "",
            "gninaRuntimeSec": 0.0,
            "gninaCpuAffinity": args.cpu_affinity,
            "gninaOutputSdfPath": str(out_sdf),
            "gninaStdoutPreview": "",
            "gninaStderrPreview": "",
        }
        score, tier, reason = classify_gnina(merged)
        return {**merged, "gninaConsensusScore": round(score, 4), "gninaConsensusTier": tier, "gninaConsensusReason": reason}

    command = [
        gnina,
        "--score_only",
        "--cnn_scoring",
        args.cnn_scoring,
        "-r",
        row["receptorPdbPath"],
        "-l",
        row["confidenceSdfPath"],
        "--autobox_ligand",
        row["confidenceSdfPath"],
        "--autobox_add",
        str(args.autobox_add),
        "-o",
        str(out_sdf),
    ]
    if args.no_gpu:
        command.append("--no_gpu")
    if args.cpu_affinity:
        command = ["taskset", "-c", args.cpu_affinity] + command
    start = time.time()
    try:
        env, _ = gnina_runtime_env(args.cpu_threads, args.hide_cuda or args.no_gpu)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        merged = {
            **row,
            "gninaStatus": "timeout",
            "gninaError": "timeout",
            "gninaReturnCode": 124,
            "gninaRuntimeSec": round(time.time() - start, 4),
            "gninaCpuAffinity": args.cpu_affinity,
            "gninaOutputSdfPath": str(out_sdf) if out_sdf.exists() else "",
            "gninaStdoutPreview": (exc.stdout or "")[:700],
            "gninaStderrPreview": (exc.stderr or "")[:700],
        }
        score, tier, reason = classify_gnina(merged)
        return {**merged, "gninaConsensusScore": round(score, 4), "gninaConsensusTier": tier, "gninaConsensusReason": reason}

    parsed = parse_gnina_output(out_sdf, result.stdout, result.stderr)
    status = "ok" if result.returncode == 0 and (parsed.get("gninaCnnScore") != "" or parsed.get("gninaCnnAffinity") != "") else "failed"
    merged = {
        **row,
        **parsed,
        "gninaStatus": status,
        "gninaError": "" if status == "ok" else (result.stderr or result.stdout)[:700],
        "gninaReturnCode": result.returncode,
        "gninaRuntimeSec": round(time.time() - start, 4),
        "gninaCpuAffinity": args.cpu_affinity,
        "gninaOutputSdfPath": str(out_sdf) if out_sdf.exists() else "",
        "gninaStdoutPreview": result.stdout[:700],
        "gninaStderrPreview": result.stderr[:700],
    }
    score, tier, reason = classify_gnina(merged)
    return {**merged, "gninaConsensusScore": round(score, 4), "gninaConsensusTier": tier, "gninaConsensusReason": reason}


def not_ready_rows(rows: list[dict[str, Any]], status: str, error: str) -> list[dict[str, Any]]:
    audit_rows = []
    for row in rows:
        merged = {
            **row,
            "gninaStatus": status,
            "gninaError": error,
            "gninaReturnCode": "",
            "gninaRuntimeSec": "",
            "gninaOutputSdfPath": "",
            "gninaCnnScore": "",
            "gninaCnnAffinity": "",
            "gninaMinimizedAffinity": "",
            "gninaStdoutPreview": "",
            "gninaStderrPreview": "",
        }
        score, tier, reason = classify_gnina(merged)
        audit_rows.append({**merged, "gninaConsensusScore": round(score, 4), "gninaConsensusTier": tier, "gninaConsensusReason": reason})
    return audit_rows


def direction_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, group in audit_df.groupby("direction", dropna=False):
        scored = int(sum(group["gninaStatus"].astype(str) == "ok"))
        ready = int(sum(group["gninaInputReady"].apply(truthy)))
        supported = int(sum(group["gninaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
        rows.append(
            {
                "direction": direction,
                "rows": int(len(group)),
                "inputReadyRows": ready,
                "gninaScoredRows": scored,
                "gninaScoredPct": round(pct(scored, len(group)), 4),
                "gninaSupportedRows": supported,
                "gninaSupportedPct": round(pct(supported, len(group)), 4),
                "knownRows": int(sum(group.apply(lambda item: is_known(item.to_dict()), axis=1))),
                "novelRows": int(sum(group.apply(lambda item: is_novel(item.to_dict()), axis=1))),
                "medianGninaCnnScore": (
                    round(pd.to_numeric(group["gninaCnnScore"], errors="coerce").median(), 4) if scored else ""
                ),
                "medianGninaCnnAffinity": (
                    round(pd.to_numeric(group["gninaCnnAffinity"], errors="coerce").median(), 4) if scored else ""
                ),
                "tierCounts": json.dumps(dict(Counter(group["gninaConsensusTier"].astype(str))), ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values("direction")


def summarize(
    audit_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    source_rows: int,
    args: argparse.Namespace,
    gnina_path: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    scored = int(sum(audit_df["gninaStatus"].astype(str) == "ok"))
    ready = int(sum(audit_df["gninaInputReady"].apply(truthy)))
    supported = int(sum(audit_df["gninaConsensusTier"].astype(str).str.startswith(("A_", "B_"))))
    status_counts = dict(Counter(audit_df["gninaStatus"].astype(str)))
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": f"GNINA CNN score-only structural rescoring over Top{len(audit_df)} external structural queue rows.",
        "source": args.source,
        "sourceRows": int(source_rows),
        "candidateRows": int(len(audit_df)),
        "auditOnly": bool(args.audit_only),
        "gninaBinary": gnina_path,
        "gninaBinaryAvailable": bool(runtime.get("binaryAvailable")),
        "gninaRuntimeReady": bool(runtime.get("runtimeReady")),
        "gninaVersion": runtime.get("version", ""),
        "gninaRuntimeError": runtime.get("runtimeError", ""),
        "gninaLdLibraryPath": runtime.get("ldLibraryPath", ""),
        "gninaHideCuda": bool(args.hide_cuda or args.no_gpu),
        "gninaCpuThreads": int(args.cpu_threads),
        "gninaCpuAffinity": args.cpu_affinity,
        "inputReadyRows": ready,
        "inputReadyPct": round(pct(ready, len(audit_df)), 4),
        "gninaScoredRows": scored,
        "gninaScoredPct": round(pct(scored, len(audit_df)), 4),
        "gninaSupportedRows": supported,
        "gninaSupportedPct": round(pct(supported, len(audit_df)), 4),
        "knownRows": int(sum(audit_df.apply(lambda row: is_known(row.to_dict()), axis=1))),
        "novelRows": int(sum(audit_df.apply(lambda row: is_novel(row.to_dict()), axis=1))),
        "medianGninaCnnScore": (
            round(pd.to_numeric(audit_df["gninaCnnScore"], errors="coerce").median(), 4) if scored else ""
        ),
        "medianGninaCnnAffinity": (
            round(pd.to_numeric(audit_df["gninaCnnAffinity"], errors="coerce").median(), 4) if scored else ""
        ),
        "statusCounts": status_counts,
        "tierCounts": dict(Counter(audit_df["gninaConsensusTier"].astype(str))),
        "directionRows": direction_df.to_dict(orient="records"),
        "methodNote": "GNINA is intended here as a CNN rescoring layer over existing DiffDock ligand poses and receptor structures. It is an independent structural consensus stress test, not an experimental affinity measurement. If auditOnly is true, rows record execution_pending rather than model results.",
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    lines = [
        "# GNINA CNN Rescoring Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["scope"],
        "",
        summary["methodNote"],
        "",
        "## Headline Results",
        "",
        f"- GNINA binary available: {summary['gninaBinaryAvailable']} (`{summary['gninaBinary']}`).",
        f"- GNINA runtime ready: {summary['gninaRuntimeReady']} ({summary.get('gninaVersion', '')}).",
        f"- GNINA LD_LIBRARY_PATH: `{summary.get('gninaLdLibraryPath', '')}`.",
        f"- GNINA CPU affinity: `{summary.get('gninaCpuAffinity', '') or 'unset'}`; CPU thread env cap: {summary.get('gninaCpuThreads', 0)}.",
        f"- Input-ready rows: {summary['inputReadyRows']}/{summary['candidateRows']} ({fmt(summary['inputReadyPct'])}%).",
        f"- GNINA-scored rows: {summary['gninaScoredRows']}/{summary['candidateRows']} ({fmt(summary['gninaScoredPct'])}%).",
        f"- A/B GNINA-supported rows: {summary['gninaSupportedRows']}/{summary['candidateRows']} ({fmt(summary['gninaSupportedPct'])}%).",
        f"- Median CNNscore: {fmt(summary['medianGninaCnnScore'], 4)}.",
        f"- Median CNNaffinity: {fmt(summary['medianGninaCnnAffinity'], 4)}.",
        f"- Status counts: {summary['statusCounts']}.",
        f"- Tier counts: {summary['tierCounts']}.",
        "",
        "## Direction Summary",
        "",
        "| Direction | Rows | Scored | A/B Supported | Median CNNscore | Median CNNaffinity | Tier counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in direction_df.iterrows():
        lines.append(
            f"| {row['direction']} | {row['rows']} | {row['gninaScoredRows']} | "
            f"{row['gninaSupportedRows']} | {fmt(row['medianGninaCnnScore'], 4)} | "
            f"{fmt(row['medianGninaCnnAffinity'], 4)} | {row['tierCounts']} |"
        )
    lines.extend(
        [
            "",
            "## Top Queue Rows",
            "",
            "| Queue rank | Direction | Pair | Drug | Target | Status | CNNscore | CNNaffinity | Tier | Reason |",
            "| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in audit_df.head(20).iterrows():
        lines.append(
            f"| {row.get('externalQueueRank', '')} | {row.get('direction', '')} | {row.get('pairId', '')} | "
            f"{row.get('drug', '')} | {row.get('target', '')} | {row.get('gninaStatus', '')} | "
            f"{fmt(row.get('gninaCnnScore'), 4)} | {fmt(row.get('gninaCnnAffinity'), 4)} | "
            f"{row.get('gninaConsensusTier', '')} | {row.get('gninaConsensusReason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/B rows require favorable GNINA CNN signal together with existing pose-quality support.",
            "- C rows are reviewable but need expert pose or context review before promotion.",
            "- D rows should not be promoted from GNINA alone; runtime statuses mean the model was not executed.",
            "",
            "## Output Files",
            "",
            "- Candidate audit: `outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_candidate_audit.csv`",
            "- Direction summary: `outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_direction_summary.csv`",
            "- Summary JSON: `outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_summary.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_path = root / args.source
    df = pd.read_csv(source_path, low_memory=False).fillna("")
    selected = df.sort_values("externalQueueRank").copy()
    if args.top_n > 0:
        selected = selected.head(args.top_n).copy()
    base_rows = [base_row(root, row) for _, row in selected.iterrows()]

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gnina = resolve_binary(args.gnina)
    runtime = check_gnina_runtime(gnina)
    audit_rows: list[dict[str, Any]]
    if args.audit_only:
        status = "execution_pending" if runtime["runtimeReady"] else ("binary_missing" if not runtime["binaryAvailable"] else "runtime_not_ready")
        error = (
            "gnina runtime ready; scoring intentionally not executed in audit-only mode"
            if runtime["runtimeReady"]
            else runtime["runtimeError"]
        )
        audit_rows = not_ready_rows(base_rows, status, error)
    elif not runtime["runtimeReady"]:
        status = "binary_missing" if not runtime["binaryAvailable"] else "runtime_not_ready"
        audit_rows = not_ready_rows(base_rows, status, runtime["runtimeError"])
    else:
        audit_rows = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(run_row, row, args, gnina, out_dir) for row in base_rows]
            for index, future in enumerate(as_completed(futures), start=1):
                audit_rows.append(future.result())
                if args.progress_every and index % args.progress_every == 0:
                    print(json.dumps({"processed": index}, ensure_ascii=False))

    audit_df = pd.DataFrame(audit_rows).sort_values("externalQueueRank").fillna("")
    direction_df = direction_summary(audit_df)
    summary = summarize(audit_df, direction_df, len(df), args, gnina, runtime)

    paths = {
        "audit": out_dir / "gnina_cnn_rescoring_candidate_audit.csv",
        "direction": out_dir / "gnina_cnn_rescoring_direction_summary.csv",
        "summary": out_dir / "gnina_cnn_rescoring_summary.json",
        "md": out_dir / "GNINA_CNN_RESCORING_AUDIT.md",
    }
    audit_df.to_csv(paths["audit"], index=False)
    direction_df.to_csv(paths["direction"], index=False)
    write_json(paths["summary"], summary)
    paths["md"].write_text(markdown(summary, direction_df, audit_df), encoding="utf-8")
    return {"summary": summary, **{key: str(value.relative_to(root)) for key, value in paths.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or audit GNINA CNN score-only rescoring over the external Top100 queue.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/sota_validation/external_sota_model_inputs/gnina_top100_rescoring_queue.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/gnina_cnn_rescoring")
    parser.add_argument("--gnina", default="/root/autodl-tmp/tools/gnina/gnina.1.3.2")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--autobox-add", type=float, default=4.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cnn-scoring", default="rescore")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--hide-cuda", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cpu-affinity", default="", help="Optional taskset CPU list, for example 44-47, used to cap GNINA CPU placement.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build(root, args)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

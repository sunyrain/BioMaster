from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import time
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEPMap_MANIFEST_URL = "https://depmap.org/portal/api/download/files"
REQUIRED_FILENAMES = ["CRISPRGeneDependency.csv", "Model.csv"]
OPTIONAL_FILENAMES = ["CRISPRGeneEffect.csv"]

MODEL_ID_CANDIDATES = ["ModelID", "DepMap_ID", "DepMapID", "model_id"]
LINEAGE_CANDIDATES = ["OncotreeLineage", "Lineage", "lineage", "PrimaryDisease", "OncotreePrimaryDisease"]
MODEL_NAME_CANDIDATES = ["StrippedCellLineName", "CellLineName", "ModelName", "CCLEName"]


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        if math.isnan(float(value)) or math.isinf(float(value)):
            return "NA"
    except Exception:
        return "NA"
    return f"{float(value):.{digits}f}"


def pct_str(value: float | int | None) -> str:
    return "NA" if value is None else f"{float(value):.2f}%"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def depmap_gene_symbol(column_name: str) -> str:
    text = str(column_name or "").strip()
    match = re.match(r"^(.+?)\s+\(\d+\)$", text)
    if match:
        text = match.group(1)
    return normalize_symbol(text)


def numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def first_column(columns: list[str], candidates: list[str]) -> str:
    for col in candidates:
        if col in columns:
            return col
    return columns[0]


def fetch_depmap_manifest() -> list[dict[str, str]]:
    with urllib.request.urlopen(DEPMap_MANIFEST_URL, timeout=60) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def choose_release(rows: list[dict[str, str]], release: str | None) -> tuple[str, str, dict[str, dict[str, str]]]:
    if release:
        release_rows = [row for row in rows if row.get("release") == release]
        missing = [name for name in REQUIRED_FILENAMES if not any(row.get("filename") == name for row in release_rows)]
        if missing:
            raise ValueError(f"Release {release!r} is missing required DepMap files: {missing}")
    else:
        by_release: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_release[row.get("release", "")].append(row)
        candidates = []
        for rel, group in by_release.items():
            filenames = {row.get("filename") for row in group}
            if all(name in filenames for name in REQUIRED_FILENAMES):
                release_date = max(row.get("release_date", "") for row in group)
                candidates.append((release_date, rel, group))
        if not candidates:
            raise ValueError("Could not find a DepMap release containing the required CRISPR dependency and Model files.")
        _, release, release_rows = sorted(candidates, reverse=True)[0]

    release_date = max((row.get("release_date", "") for row in release_rows), default="")
    by_file = {
        row["filename"]: row
        for row in release_rows
        if row.get("filename") in set(REQUIRED_FILENAMES + OPTIONAL_FILENAMES)
    }
    return release, release_date, by_file


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def urllib_download(url: str, tmp: Path, progress_mb: int = 100) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "BioMaster-DepMap-audit/1.0"})
    started = time.time()
    next_report = progress_mb * 1024 * 1024
    downloaded = 0
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[download] {tmp.name}: {downloaded / 1024 / 1024:.1f} MB at "
                    f"{downloaded / 1024 / 1024 / elapsed:.2f} MB/s",
                    flush=True,
                )
                next_report += progress_mb * 1024 * 1024


def curl_download(url: str, tmp: Path) -> None:
    if not shutil.which("curl"):
        raise RuntimeError("curl is not available")
    command = [
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "60",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--output",
        str(tmp),
        url,
    ]
    subprocess.run(command, check=True)


def download_file(
    url: str,
    dest: Path,
    expected_md5: str = "",
    force: bool = False,
    progress_mb: int = 100,
    download_tool: str = "auto",
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if not expected_md5 or md5(dest) == expected_md5:
            return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"[download] {dest.name} -> {dest}", flush=True)
    if download_tool not in {"auto", "curl", "urllib"}:
        raise ValueError(f"Unknown download tool: {download_tool}")
    if download_tool in {"auto", "curl"} and shutil.which("curl"):
        curl_download(url, tmp)
    else:
        urllib_download(url, tmp, progress_mb)
    if expected_md5:
        observed = md5(tmp)
        if observed != expected_md5:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"MD5 mismatch for {dest}: expected {expected_md5}, observed {observed}")
    tmp.replace(dest)


def prepare_depmap_files(
    root: Path,
    data_dir: Path,
    release: str | None,
    force_download: bool,
    include_gene_effect: bool,
    download_tool: str,
) -> dict[str, Any]:
    rows = fetch_depmap_manifest()
    selected_release, release_date, by_file = choose_release(rows, release)
    release_dir = data_dir / slug(selected_release)
    files: dict[str, str] = {}
    file_meta: dict[str, dict[str, str]] = {}
    filenames = REQUIRED_FILENAMES + (OPTIONAL_FILENAMES if include_gene_effect else [])
    for filename in filenames:
        meta = by_file[filename]
        dest = release_dir / filename
        download_file(meta["url"], dest, meta.get("md5_hash", ""), force_download, download_tool=download_tool)
        files[filename] = str(dest)
        file_meta[filename] = {
            "release": meta.get("release", ""),
            "releaseDate": meta.get("release_date", ""),
            "filename": filename,
            "md5": meta.get("md5_hash", ""),
            "path": str(dest),
            "sizeBytes": dest.stat().st_size if dest.exists() else 0,
        }
    if not include_gene_effect:
        for filename in OPTIONAL_FILENAMES:
            meta = by_file.get(filename)
            dest = release_dir / filename
            if meta and dest.exists() and (not meta.get("md5_hash") or md5(dest) == meta.get("md5_hash")):
                files[filename] = str(dest)
                file_meta[filename] = {
                    "release": meta.get("release", ""),
                    "releaseDate": meta.get("release_date", ""),
                    "filename": filename,
                    "md5": meta.get("md5_hash", ""),
                    "path": str(dest),
                    "sizeBytes": dest.stat().st_size,
                    "usedBecauseAlreadyCached": True,
                }
    return {
        "release": selected_release,
        "releaseDate": release_date,
        "manifestUrl": DEPMap_MANIFEST_URL,
        "files": files,
        "fileMeta": file_meta,
        "includeGeneEffect": include_gene_effect or "CRISPRGeneEffect.csv" in files,
    }


def probe_url(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "BioMaster-DepMap-audit/1.0"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "probeStatus": "reachable",
                "httpStatus": int(response.status),
                "contentLength": response.headers.get("Content-Length", ""),
                "elapsedSeconds": round(time.time() - started, 3),
            }
    except HTTPError as exc:
        return {
            "probeStatus": "http_error",
            "httpStatus": int(exc.code),
            "contentLength": exc.headers.get("Content-Length", "") if exc.headers else "",
            "elapsedSeconds": round(time.time() - started, 3),
            "error": str(exc),
        }
    except (TimeoutError, URLError, OSError) as exc:
        return {
            "probeStatus": "connection_error",
            "httpStatus": "",
            "contentLength": "",
            "elapsedSeconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def local_file_state(dest: Path, expected_md5: str) -> dict[str, Any]:
    part = dest.with_suffix(dest.suffix + ".part")
    complete_exists = dest.exists()
    part_exists = part.exists()
    complete_size = dest.stat().st_size if complete_exists else 0
    part_size = part.stat().st_size if part_exists else 0
    md5_status = "not_checked"
    if complete_exists and expected_md5:
        try:
            md5_status = "pass" if md5(dest) == expected_md5 else "fail"
        except OSError as exc:
            md5_status = f"error:{type(exc).__name__}"
    elif complete_exists:
        md5_status = "no_expected_md5"
    if complete_exists and md5_status in {"pass", "no_expected_md5"}:
        status = "complete_cached"
    elif complete_exists:
        status = "complete_cache_md5_mismatch"
    elif part_exists:
        status = "partial_download_only"
    else:
        status = "missing_local_cache"
    return {
        "localStatus": status,
        "completePath": str(dest),
        "completeExists": complete_exists,
        "completeSizeBytes": complete_size,
        "partialPath": str(part),
        "partialExists": part_exists,
        "partialSizeBytes": part_size,
        "md5Status": md5_status,
    }


def build_data_access_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    source_path = root / args.source
    out_dir = root / args.out_dir
    data_dir = root / args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_candidates(source_path)
    oncology = candidates[candidates["direction"].astype(str).str.lower().eq("oncology")].copy()
    target_counts_df = target_counts(candidates)
    target_scope_path = out_dir / "depmap_oncology_target_scope.csv"
    target_counts_df.to_csv(target_scope_path, index=False)

    manifest_error = ""
    manifest_rows: list[dict[str, str]] = []
    selected_release = ""
    release_date = ""
    by_file: dict[str, dict[str, str]] = {}
    started = time.time()
    try:
        manifest_rows = fetch_depmap_manifest()
        selected_release, release_date, by_file = choose_release(manifest_rows, args.release)
    except Exception as exc:  # noqa: BLE001 - this is an access audit, so record the exact failure.
        manifest_error = f"{type(exc).__name__}: {exc}"

    release_dir = data_dir / slug(selected_release or args.release or "unknown_release")
    file_rows: list[dict[str, Any]] = []
    for filename in REQUIRED_FILENAMES + OPTIONAL_FILENAMES:
        meta = by_file.get(filename, {})
        dest = release_dir / filename
        row = {
            "filename": filename,
            "required": filename in REQUIRED_FILENAMES,
            "release": selected_release,
            "releaseDate": release_date,
            "manifestPresent": bool(meta),
            "urlHost": urlparse(meta.get("url", "")).netloc,
            "md5": meta.get("md5_hash", ""),
        }
        row.update(local_file_state(dest, meta.get("md5_hash", "")))
        if args.probe_urls and meta.get("url"):
            row.update(probe_url(meta["url"], args.probe_timeout))
        else:
            row.update({"probeStatus": "not_run", "httpStatus": "", "contentLength": "", "elapsedSeconds": "", "error": ""})
        file_rows.append(row)

    file_audit = pd.DataFrame(file_rows)
    file_audit_path = out_dir / "depmap_required_file_access_audit.csv"
    file_audit.to_csv(file_audit_path, index=False)

    required_rows = file_audit[file_audit["required"].astype(bool)] if not file_audit.empty else pd.DataFrame()
    required_manifest_present = int(required_rows["manifestPresent"].sum()) if not required_rows.empty else 0
    required_complete = int((required_rows["localStatus"] == "complete_cached").sum()) if not required_rows.empty else 0
    required_partial = int(required_rows["partialExists"].astype(bool).sum()) if not required_rows.empty else 0
    all_required_ready = required_complete == len(REQUIRED_FILENAMES)
    calculation_readiness = "ready_to_score" if all_required_ready else "not_ready_external_matrix_unavailable"
    summary = {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "DepMap data-access and oncology-candidate scope audit; no full matrix download is attempted in this mode.",
        "source": str(source_path),
        "sourceRows": int(len(candidates)),
        "oncologyCandidateRows": int(len(oncology)),
        "uniqueOncologyTargets": int(oncology["target"].nunique()),
        "uniqueOncologyDrugs": int(oncology["drug"].nunique()),
        "uniqueOncologyPairs": int(oncology["pairId"].nunique()),
        "manifestUrl": DEPMap_MANIFEST_URL,
        "manifestFetchStatus": "ok" if not manifest_error else "failed",
        "manifestError": manifest_error,
        "manifestRows": int(len(manifest_rows)),
        "manifestElapsedSeconds": round(time.time() - started, 3),
        "selectedRelease": selected_release,
        "selectedReleaseDate": release_date,
        "requiredFiles": REQUIRED_FILENAMES,
        "optionalFiles": OPTIONAL_FILENAMES,
        "requiredFilesPresentInManifest": required_manifest_present,
        "requiredFilesCompleteCached": required_complete,
        "requiredFilesPartialCached": required_partial,
        "calculationReadiness": calculation_readiness,
        "canComputeDependencyScoresNow": bool(all_required_ready),
        "targetScopeOutput": str(target_scope_path),
        "fileAccessAuditOutput": str(file_audit_path),
        "methodNote": (
            "This audit only verifies candidate scope, DepMap manifest availability, and local/cache readiness. "
            "A missing or slow DepMap matrix is a data-access gap, not evidence that oncology targets lack "
            "cancer dependency."
        ),
    }
    write_json(out_dir / "depmap_data_access_summary.json", summary)
    write_access_markdown(out_dir / "DEPMAP_DATA_ACCESS_AUDIT.md", summary, file_audit)
    return summary


def write_access_markdown(out_path: Path, summary: dict[str, Any], file_audit: pd.DataFrame) -> None:
    lines = [
        "# DepMap Data Access Audit",
        "",
        f"Generated: {summary['createdUtc']}",
        "",
        "## Purpose",
        "",
        "This audit records whether the oncology dependency layer can be computed from local DepMap files. It does not treat missing files or slow downloads as negative biological evidence.",
        "",
        "## Candidate Scope",
        "",
        f"- Candidate matrix: `{summary['source']}`.",
        f"- Total candidate rows: {summary['sourceRows']}.",
        f"- Oncology rows: {summary['oncologyCandidateRows']}.",
        f"- Unique oncology targets: {summary['uniqueOncologyTargets']}.",
        f"- Unique oncology drugs: {summary['uniqueOncologyDrugs']}.",
        f"- Unique oncology drug-target pairs: {summary['uniqueOncologyPairs']}.",
        "",
        "## DepMap Access State",
        "",
        f"- Manifest URL: `{summary['manifestUrl']}`.",
        f"- Manifest fetch status: {summary['manifestFetchStatus']}.",
        f"- Selected release: {summary['selectedRelease']} ({summary['selectedReleaseDate']}).",
        f"- Required files present in manifest: {summary['requiredFilesPresentInManifest']}/{len(REQUIRED_FILENAMES)}.",
        f"- Required files complete in local cache: {summary['requiredFilesCompleteCached']}/{len(REQUIRED_FILENAMES)}.",
        f"- Required files with partial cache: {summary['requiredFilesPartialCached']}/{len(REQUIRED_FILENAMES)}.",
        f"- Calculation readiness: {summary['calculationReadiness']}.",
        "",
        "## Required File Audit",
        "",
        "| File | Required | Manifest | Local Status | Complete Bytes | Partial Bytes | Probe |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in file_audit.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.filename),
                    str(bool(row.required)),
                    str(bool(row.manifestPresent)),
                    str(row.localStatus),
                    str(int(row.completeSizeBytes)),
                    str(int(row.partialSizeBytes)),
                    str(row.probeStatus),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- DepMap is an oncology target-dependency layer, not a drug-target binding or docking model.",
            "- It asks whether cancer cell lines depend on a target after CRISPR loss-of-function perturbation.",
            "- The current audit is ready to become a scoring run once `CRISPRGeneDependency.csv` and `Model.csv` are fully cached and pass MD5 checks.",
            "- Until those matrices are present, the correct conclusion is a data-access gap rather than a negative dependency result.",
            "",
            "## Output Files",
            "",
            "- `depmap_data_access_summary.json`: machine-readable access and scope summary.",
            "- `depmap_required_file_access_audit.csv`: file-level manifest/cache/probe status.",
            "- `depmap_oncology_target_scope.csv`: oncology target scope and candidate counts.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"direction", "target", "pairId", "drug", "protein"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Candidate matrix is missing required columns: {sorted(missing)}")
    df["targetNorm"] = df["target"].map(normalize_symbol)
    return df


def read_depmap_matrix(path: Path, target_symbols: set[str]) -> tuple[pd.DataFrame, str, dict[str, list[str]]]:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    model_id_col = first_column(columns, MODEL_ID_CANDIDATES)
    symbol_to_columns: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        if col == model_id_col:
            continue
        symbol = depmap_gene_symbol(col)
        if symbol in target_symbols:
            symbol_to_columns[symbol].append(col)
    usecols = [model_id_col] + sorted({col for cols in symbol_to_columns.values() for col in cols})
    matrix = pd.read_csv(path, usecols=usecols)
    matrix = matrix.rename(columns={model_id_col: "ModelID"})
    matrix["ModelID"] = matrix["ModelID"].astype(str)
    return matrix, model_id_col, dict(symbol_to_columns)


def read_model_metadata(path: Path) -> pd.DataFrame:
    model = pd.read_csv(path, dtype=str).fillna("")
    if model.empty:
        return pd.DataFrame(columns=["ModelID", "cellLineName", "lineage", "primaryDisease", "subtype"])
    model_id_col = first_column(list(model.columns), MODEL_ID_CANDIDATES)
    lineage_col = next((col for col in LINEAGE_CANDIDATES if col in model.columns), "")
    name_col = next((col for col in MODEL_NAME_CANDIDATES if col in model.columns), "")
    primary_col = "OncotreePrimaryDisease" if "OncotreePrimaryDisease" in model.columns else lineage_col
    subtype_col = "OncotreeSubtype" if "OncotreeSubtype" in model.columns else ""
    out = pd.DataFrame(
        {
            "ModelID": model[model_id_col].astype(str),
            "cellLineName": model[name_col].astype(str) if name_col else "",
            "lineage": model[lineage_col].astype(str) if lineage_col else "",
            "primaryDisease": model[primary_col].astype(str) if primary_col else "",
            "subtype": model[subtype_col].astype(str) if subtype_col else "",
        }
    )
    out["lineage"] = out["lineage"].replace("", "unclassified")
    return out.drop_duplicates("ModelID")


def collapse_gene_values(matrix: pd.DataFrame, columns: list[str], mode: str) -> pd.Series:
    if not columns:
        return pd.Series(float("nan"), index=matrix.index, dtype=float)
    values = matrix[columns].apply(pd.to_numeric, errors="coerce")
    if len(columns) == 1:
        return values.iloc[:, 0]
    if mode == "min":
        return values.min(axis=1, skipna=True)
    if mode == "max":
        return values.max(axis=1, skipna=True)
    return values.mean(axis=1, skipna=True)


def quantile(values: pd.Series, q: float) -> float | None:
    values = numeric_series(values).dropna()
    if values.empty:
        return None
    return round(float(values.quantile(q)), 4)


def mean(values: pd.Series) -> float | None:
    values = numeric_series(values).dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def median(values: pd.Series) -> float | None:
    values = numeric_series(values).dropna()
    if values.empty:
        return None
    return round(float(values.median()), 4)


def min_value(values: pd.Series) -> float | None:
    values = numeric_series(values).dropna()
    if values.empty:
        return None
    return round(float(values.min()), 4)


def max_value(values: pd.Series) -> float | None:
    values = numeric_series(values).dropna()
    if values.empty:
        return None
    return round(float(values.max()), 4)


def top_lineage_string(lineage_rows: pd.DataFrame, limit: int = 6) -> str:
    if lineage_rows.empty:
        return ""
    rows = lineage_rows.sort_values(["dependentModels", "dependentPct", "lineage"], ascending=[False, False, True]).head(limit)
    parts = []
    for row in rows.itertuples(index=False):
        if int(row.dependentModels) <= 0:
            continue
        parts.append(f"{row.lineage}:{int(row.dependentModels)}/{int(row.profiledModels)} ({float(row.dependentPct):.1f}%)")
    return "; ".join(parts)


def classify_tier(
    matched: bool,
    profiled_models: int,
    dependent_pct: float,
    strong_dependent_pct: float,
    max_dependency: float | None,
    min_effect: float | None,
    p10_effect: float | None,
) -> str:
    if not matched:
        return "U_no_depmap_gene_match"
    if profiled_models <= 0:
        return "U_no_depmap_profiled_models"
    max_dep = max_dependency if max_dependency is not None else 0.0
    has_effect = min_effect is not None or p10_effect is not None
    min_eff = min_effect if min_effect is not None else 0.0
    p10_eff = p10_effect if p10_effect is not None else 0.0
    if strong_dependent_pct >= 5.0 or dependent_pct >= 15.0 or (has_effect and p10_eff <= -0.75):
        return "A_recurrent_cancer_dependency"
    if dependent_pct >= 5.0 or max_dep >= 0.8 or (has_effect and min_eff <= -1.0):
        return "B_subset_cancer_dependency"
    if dependent_pct > 0.0 or max_dep >= 0.5 or (has_effect and min_eff <= -0.5):
        return "C_weak_or_rare_dependency"
    return "D_no_depmap_dependency_signal"


TIER_SCORE = {
    "A_recurrent_cancer_dependency": 94.0,
    "B_subset_cancer_dependency": 82.0,
    "C_weak_or_rare_dependency": 62.0,
    "D_no_depmap_dependency_signal": 38.0,
    "U_no_depmap_profiled_models": 50.0,
    "U_no_depmap_gene_match": 50.0,
}

TIER_ADJUSTMENT = {
    "A_recurrent_cancer_dependency": 2.8,
    "B_subset_cancer_dependency": 1.4,
    "C_weak_or_rare_dependency": 0.2,
    "D_no_depmap_dependency_signal": -1.0,
    "U_no_depmap_profiled_models": 0.0,
    "U_no_depmap_gene_match": 0.0,
}


def score_for_tier(tier: str, dependent_pct: float, strong_dependent_pct: float) -> float:
    base = TIER_SCORE.get(tier, 50.0)
    bonus = min(4.0, dependent_pct / 20.0 + strong_dependent_pct / 10.0)
    if tier.startswith(("A_", "B_", "C_")):
        return round(max(0.0, min(100.0, base + bonus)), 4)
    return base


def adjustment_for_tier(tier: str, dependent_pct: float, strong_dependent_pct: float) -> float:
    base = TIER_ADJUSTMENT.get(tier, 0.0)
    if tier.startswith(("A_", "B_")):
        base += min(0.7, dependent_pct / 100.0 + strong_dependent_pct / 50.0)
    return round(base, 4)


def reason_for_tier(
    tier: str,
    profiled_models: int,
    dependent_models: int,
    dependent_pct: float,
    strong_dependent_models: int,
    max_dependency: float | None,
    min_effect: float | None,
    top_lineages: str,
) -> str:
    if tier == "U_no_depmap_gene_match":
        return "No exact DepMap gene-symbol column was found; treat as data coverage gap, not negative biology."
    if tier == "U_no_depmap_profiled_models":
        return "DepMap gene column was found, but no profiled cell-line values were available."
    support = (
        f"{dependent_models}/{profiled_models} profiled cancer models meet dependency probability >=0.5 "
        f"({dependent_pct:.2f}%); {strong_dependent_models} meet >=0.9; "
        f"max dependency probability {fmt(max_dependency, 3)}."
    )
    if min_effect is not None:
        support += f" Minimum Chronos gene effect {fmt(min_effect, 3)}."
    if top_lineages:
        support += f" Top dependent lineages: {top_lineages}."
    if tier == "A_recurrent_cancer_dependency":
        return "Recurrent DepMap cancer dependency signal. " + support
    if tier == "B_subset_cancer_dependency":
        return "Subset or lineage-specific DepMap cancer dependency signal. " + support
    if tier == "C_weak_or_rare_dependency":
        return "Weak or rare DepMap dependency signal; suitable for review rather than prioritization alone. " + support
    return "No material DepMap dependency signal under the current thresholds. " + support


def build_target_audit(
    target_symbols: set[str],
    effect: pd.DataFrame,
    dependency: pd.DataFrame,
    model_meta: pd.DataFrame,
    effect_cols: dict[str, list[str]],
    dep_cols: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    meta = model_meta.set_index("ModelID", drop=False)
    for symbol in sorted(target_symbols):
        e_cols = effect_cols.get(symbol, [])
        d_cols = dep_cols.get(symbol, [])
        matched = bool(e_cols or d_cols)
        base_index = effect.index if not effect.empty else dependency.index
        e_series = collapse_gene_values(effect, e_cols, "min") if e_cols else pd.Series(float("nan"), index=base_index)
        d_series = collapse_gene_values(dependency, d_cols, "max") if d_cols else pd.Series(float("nan"), index=dependency.index)
        values = pd.DataFrame(
            {
                "ModelID": effect["ModelID"].astype(str)
                if "ModelID" in effect and not effect.empty
                else dependency["ModelID"].astype(str),
                "geneEffect": e_series,
            }
        )
        if "ModelID" in dependency:
            dep_values = pd.DataFrame({"ModelID": dependency["ModelID"].astype(str), "dependencyProbability": d_series})
            values = values.merge(dep_values, on="ModelID", how="outer")
        else:
            values["dependencyProbability"] = float("nan")
        values = values.merge(model_meta, on="ModelID", how="left")
        values["lineage"] = values["lineage"].fillna("unclassified").replace("", "unclassified")
        values["geneEffect"] = pd.to_numeric(values["geneEffect"], errors="coerce")
        values["dependencyProbability"] = pd.to_numeric(values["dependencyProbability"], errors="coerce")
        profiled = values[values["geneEffect"].notna() | values["dependencyProbability"].notna()].copy()
        profiled_models = int(len(profiled))
        dependent_mask = (profiled["dependencyProbability"] >= 0.5) | (profiled["geneEffect"] <= -0.5)
        strong_mask = (profiled["dependencyProbability"] >= 0.9) | (profiled["geneEffect"] <= -1.0)
        dependent_models = int(dependent_mask.sum())
        strong_dependent_models = int(strong_mask.sum())
        dependent_pct = round(pct(dependent_models, profiled_models), 4)
        strong_dependent_pct = round(pct(strong_dependent_models, profiled_models), 4)

        for lineage, group in profiled.groupby("lineage", dropna=False):
            lineage_dependent = (group["dependencyProbability"] >= 0.5) | (group["geneEffect"] <= -0.5)
            lineage_strong = (group["dependencyProbability"] >= 0.9) | (group["geneEffect"] <= -1.0)
            lineage_rows.append(
                {
                    "target": symbol,
                    "lineage": lineage or "unclassified",
                    "profiledModels": int(len(group)),
                    "dependentModels": int(lineage_dependent.sum()),
                    "dependentPct": round(pct(int(lineage_dependent.sum()), len(group)), 4),
                    "strongDependentModels": int(lineage_strong.sum()),
                    "strongDependentPct": round(pct(int(lineage_strong.sum()), len(group)), 4),
                    "medianGeneEffect": median(group["geneEffect"]),
                    "minGeneEffect": min_value(group["geneEffect"]),
                    "maxDependencyProbability": max_value(group["dependencyProbability"]),
                }
            )

        lineage_df = pd.DataFrame([row for row in lineage_rows if row["target"] == symbol])
        top_lineages = top_lineage_string(lineage_df)
        max_dependency = max_value(profiled["dependencyProbability"])
        min_effect = min_value(profiled["geneEffect"])
        p10_effect = quantile(profiled["geneEffect"], 0.10)
        tier = classify_tier(
            matched,
            profiled_models,
            dependent_pct,
            strong_dependent_pct,
            max_dependency,
            min_effect,
            p10_effect,
        )
        rows.append(
            {
                "target": symbol,
                "depmapMatchedGeneFlag": bool(matched),
                "depmapEffectColumns": ";".join(e_cols),
                "depmapDependencyColumns": ";".join(d_cols),
                "profiledModels": profiled_models,
                "dependentModels": dependent_models,
                "dependentPct": dependent_pct,
                "strongDependentModels": strong_dependent_models,
                "strongDependentPct": strong_dependent_pct,
                "medianGeneEffect": median(profiled["geneEffect"]),
                "p10GeneEffect": p10_effect,
                "minGeneEffect": min_effect,
                "meanDependencyProbability": mean(profiled["dependencyProbability"]),
                "medianDependencyProbability": median(profiled["dependencyProbability"]),
                "maxDependencyProbability": max_dependency,
                "topDependentLineages": top_lineages,
                "depmapDependencyTier": tier,
                "depmapDependencyPositiveFlag": tier.startswith(("A_", "B_")),
                "depmapDependencyScore": score_for_tier(tier, dependent_pct, strong_dependent_pct),
                "depmapDependencyAdjustment": adjustment_for_tier(tier, dependent_pct, strong_dependent_pct),
                "depmapDependencyReason": reason_for_tier(
                    tier,
                    profiled_models,
                    dependent_models,
                    dependent_pct,
                    strong_dependent_models,
                    max_dependency,
                    min_effect,
                    top_lineages,
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(lineage_rows)


def rank_column(df: pd.DataFrame) -> str:
    for col in ["sotaContextRankGlobal", "sotaReadyRankGlobal", "finalRankGlobal", "rank"]:
        if col in df.columns:
            return col
    return ""


def oncology_rank_column(df: pd.DataFrame) -> str:
    for col in ["sotaContextRankWithinDirection", "sotaReadyRankWithinDirection", "finalRankWithinDirection", "rank"]:
        if col in df.columns:
            return col
    return rank_column(df)


def topn(df: pd.DataFrame, n: int, rank_col: str) -> pd.DataFrame:
    if rank_col and rank_col in df.columns:
        ranks = pd.to_numeric(df[rank_col], errors="coerce")
        return df[ranks <= n].copy()
    return df.head(n).copy()


def build_candidate_outputs(
    candidates: pd.DataFrame,
    target_audit: pd.DataFrame,
    out_dir: Path,
    final_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = candidates.merge(target_audit, on="target", how="left")
    is_oncology = merged["direction"].astype(str).str.lower().eq("oncology")
    dep_cols = [
        "depmapMatchedGeneFlag",
        "depmapDependencyTier",
        "depmapDependencyPositiveFlag",
        "depmapDependencyScore",
        "depmapDependencyAdjustment",
        "depmapDependencyReason",
        "profiledModels",
        "dependentModels",
        "dependentPct",
        "strongDependentModels",
        "strongDependentPct",
        "medianGeneEffect",
        "p10GeneEffect",
        "minGeneEffect",
        "meanDependencyProbability",
        "medianDependencyProbability",
        "maxDependencyProbability",
        "topDependentLineages",
    ]
    for col in dep_cols:
        if col not in merged.columns:
            merged[col] = ""

    base_score_col = "sotaContextScore" if "sotaContextScore" in merged.columns else "sotaReadyScore" if "sotaReadyScore" in merged.columns else "finalPriorityScore"
    base_score = pd.to_numeric(merged[base_score_col], errors="coerce").fillna(0.0)
    adjustment = pd.to_numeric(merged["depmapDependencyAdjustment"], errors="coerce").fillna(0.0)
    merged["sotaDepmapOncologyScore"] = base_score
    merged.loc[is_oncology, "sotaDepmapOncologyScore"] = (base_score + adjustment).clip(0, 100)
    merged["sotaDepmapOncologyTier"] = "non_oncology_not_scored"
    merged.loc[is_oncology, "sotaDepmapOncologyTier"] = merged.loc[is_oncology, "depmapDependencyTier"].fillna("U_no_depmap_gene_match")
    merged["sotaDepmapOncologyAction"] = "not_applicable_non_oncology"
    positive = merged["depmapDependencyTier"].astype(str).str.startswith(("A_", "B_"))
    weak = merged["depmapDependencyTier"].astype(str).str.startswith("C_")
    no_signal = merged["depmapDependencyTier"].astype(str).str.startswith("D_")
    gap = merged["depmapDependencyTier"].astype(str).str.startswith("U_")
    merged.loc[is_oncology & positive, "sotaDepmapOncologyAction"] = "oncology_dependency_supported_review"
    merged.loc[is_oncology & weak, "sotaDepmapOncologyAction"] = "oncology_weak_dependency_review"
    merged.loc[is_oncology & no_signal, "sotaDepmapOncologyAction"] = "oncology_no_dependency_signal_review"
    merged.loc[is_oncology & gap, "sotaDepmapOncologyAction"] = "oncology_depmap_coverage_gap"

    sort_cols = ["sotaDepmapOncologyScore"]
    ascending = [False]
    rcol = rank_column(merged)
    if rcol:
        sort_cols.append(rcol)
        ascending.append(True)
    merged = merged.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    merged.insert(0, "sotaDepmapOncologyRankGlobal", range(1, len(merged) + 1))
    merged["sotaDepmapOncologyRankWithinDirection"] = (
        merged.groupby("direction")["sotaDepmapOncologyScore"].rank(method="first", ascending=False).astype(int)
    )

    oncology = merged[is_oncology].copy()
    shortlist = oncology.sort_values(
        ["depmapDependencyPositiveFlag", "sotaDepmapOncologyScore", oncology_rank_column(oncology)],
        ascending=[False, False, True],
    ).head(300)
    rank_col = oncology_rank_column(oncology)
    review_base = oncology.copy()
    if rank_col and rank_col in review_base.columns:
        review_base = review_base[pd.to_numeric(review_base[rank_col], errors="coerce").fillna(10**9) <= 300]
    review_queue = review_base[
        review_base["depmapDependencyTier"].astype(str).str.startswith(("C_", "D_", "U_"))
    ].sort_values([rank_col if rank_col else "sotaDepmapOncologyScore"], ascending=[True] if rank_col else [False])

    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    oncology.to_csv(out_dir / "depmap_oncology_candidate_audit.csv", index=False)
    merged.to_csv(final_dir / "final_priority_depmap_oncology_matrix.csv", index=False)
    shortlist.to_csv(final_dir / "final_priority_depmap_oncology_top300_expert_shortlist.csv", index=False)
    review_queue.to_csv(final_dir / "final_priority_depmap_oncology_review_queue.csv", index=False)
    return merged, shortlist, review_queue


def target_counts(candidates: pd.DataFrame) -> pd.DataFrame:
    oncology = candidates[candidates["direction"].astype(str).str.lower().eq("oncology")].copy()
    counts = (
        oncology.groupby("target", as_index=False)
        .agg(
            oncologyCandidateRows=("pairId", "count"),
            uniqueOncologyDrugs=("drug", "nunique"),
            bestSotaContextRankWithinDirection=(
                "sotaContextRankWithinDirection" if "sotaContextRankWithinDirection" in oncology.columns else "rank",
                "min",
            ),
            bestFinalRankWithinDirection=("finalRankWithinDirection" if "finalRankWithinDirection" in oncology.columns else "rank", "min"),
            bestSotaContextScore=("sotaContextScore" if "sotaContextScore" in oncology.columns else "finalPriorityScore", "max"),
            knownDrugTargetRows=("knownDrugTargetPair", "sum") if "knownDrugTargetPair" in oncology.columns else ("pairId", "count"),
        )
        .reset_index(drop=True)
    )
    return counts


def build_summary(
    source_path: Path,
    candidates: pd.DataFrame,
    target_audit: pd.DataFrame,
    lineage_audit: pd.DataFrame,
    merged: pd.DataFrame,
    shortlist: pd.DataFrame,
    review_queue: pd.DataFrame,
    depmap_meta: dict[str, Any],
) -> dict[str, Any]:
    is_oncology = merged["direction"].astype(str).str.lower().eq("oncology")
    oncology = merged[is_oncology].copy()
    unique_targets = int(oncology["target"].nunique())
    matched_targets = int(target_audit["depmapMatchedGeneFlag"].sum()) if "depmapMatchedGeneFlag" in target_audit else 0
    positive_targets = int(target_audit["depmapDependencyPositiveFlag"].sum()) if "depmapDependencyPositiveFlag" in target_audit else 0
    candidate_positive = int(oncology["depmapDependencyPositiveFlag"].fillna(False).astype(bool).sum())
    candidate_matched = int(oncology["depmapMatchedGeneFlag"].fillna(False).astype(bool).sum())
    rank_col = oncology_rank_column(oncology)
    top100 = topn(oncology, 100, rank_col)
    return {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "DepMap CRISPR oncology dependency audit over oncology rows in the final SOTA candidate matrix.",
        "source": str(source_path),
        "sourceRows": int(len(candidates)),
        "oncologyCandidateRows": int(len(oncology)),
        "uniqueOncologyTargets": unique_targets,
        "depmapRelease": depmap_meta["release"],
        "depmapReleaseDate": depmap_meta["releaseDate"],
        "depmapManifestUrl": depmap_meta["manifestUrl"],
        "depmapFiles": depmap_meta["fileMeta"],
        "depmapMatchedTargets": matched_targets,
        "depmapMatchedTargetsPct": round(pct(matched_targets, unique_targets), 4),
        "depmapDependencyPositiveTargets": positive_targets,
        "depmapDependencyPositiveTargetsPct": round(pct(positive_targets, unique_targets), 4),
        "depmapMatchedCandidateRows": candidate_matched,
        "depmapMatchedCandidateRowsPct": round(pct(candidate_matched, len(oncology)), 4),
        "depmapDependencyPositiveCandidateRows": candidate_positive,
        "depmapDependencyPositiveCandidateRowsPct": round(pct(candidate_positive, len(oncology)), 4),
        "candidateTierCounts": dict(Counter(oncology["depmapDependencyTier"].fillna("U_no_depmap_gene_match"))),
        "targetTierCounts": dict(Counter(target_audit["depmapDependencyTier"].fillna("U_no_depmap_gene_match"))),
        "top100Oncology": {
            "rankColumn": rank_col,
            "rows": int(len(top100)),
            "matchedRows": int(top100["depmapMatchedGeneFlag"].fillna(False).astype(bool).sum()) if len(top100) else 0,
            "positiveRows": int(top100["depmapDependencyPositiveFlag"].fillna(False).astype(bool).sum()) if len(top100) else 0,
            "tierCounts": dict(Counter(top100["depmapDependencyTier"].fillna("U_no_depmap_gene_match"))) if len(top100) else {},
            "knownRows": int(pd.to_numeric(top100.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).sum()) if len(top100) and "knownDrugTargetPair" in top100 else 0,
            "novelRows": int((top100.get("strictNovelPairFlag", False).fillna(False).astype(bool)).sum()) if len(top100) and "strictNovelPairFlag" in top100 else 0,
        },
        "lineageRows": int(len(lineage_audit)),
        "shortlistRows": int(len(shortlist)),
        "reviewQueueRows": int(len(review_queue)),
        "methodNote": (
            "CRISPRGeneDependency is treated as dependency probability. If CRISPRGeneEffect is locally "
            "available or explicitly requested, Chronos loss-of-function effect metrics are added as an "
            "optional stricter signal. This is an oncology target-context audit: it supports target "
            "vulnerability in cancer cell lines, but it does not by itself prove drug binding, clinical "
            "efficacy, or safety."
        ),
    }


def write_markdown(out_path: Path, summary: dict[str, Any], target_audit: pd.DataFrame) -> None:
    top_targets = target_audit.sort_values(
        ["depmapDependencyPositiveFlag", "depmapDependencyScore", "dependentPct"],
        ascending=[False, False, False],
    ).head(12)
    lines = [
        "# DepMap Oncology Dependency Audit",
        "",
        f"Generated: {summary['createdUtc']}",
        "",
        "## Purpose",
        "",
        "This audit adds an orthogonal cancer dependency layer to the existing drug-target prioritization. It asks whether each oncology candidate target shows CRISPR loss-of-function dependency in DepMap cancer cell-line data.",
        "",
        "## Data And Scope",
        "",
        f"- Candidate matrix: `{summary['source']}`.",
        f"- DepMap release: {summary['depmapRelease']} ({summary['depmapReleaseDate']}).",
        "- DepMap files: `CRISPRGeneDependency.csv` and `Model.csv`; `CRISPRGeneEffect.csv` is optional and used when cached or explicitly requested.",
        f"- Oncology candidate rows: {summary['oncologyCandidateRows']}; unique oncology targets: {summary['uniqueOncologyTargets']}.",
        "",
        "## Headline Results",
        "",
        f"- DepMap gene-symbol matched targets: {summary['depmapMatchedTargets']}/{summary['uniqueOncologyTargets']} ({pct_str(summary['depmapMatchedTargetsPct'])}).",
        f"- Dependency-positive targets: {summary['depmapDependencyPositiveTargets']}/{summary['uniqueOncologyTargets']} ({pct_str(summary['depmapDependencyPositiveTargetsPct'])}).",
        f"- Dependency-positive oncology candidate rows: {summary['depmapDependencyPositiveCandidateRows']}/{summary['oncologyCandidateRows']} ({pct_str(summary['depmapDependencyPositiveCandidateRowsPct'])}).",
        f"- Top100 oncology dependency-positive rows: {summary['top100Oncology']['positiveRows']}/{summary['top100Oncology']['rows']}.",
        f"- Candidate tier counts: {summary['candidateTierCounts']}.",
        "",
        "## Representative Dependency-Supported Targets",
        "",
        "| Target | Tier | Profiled Models | Dependent Models | Dependent % | Max Dependency | Min Gene Effect | Top Dependent Lineages |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in top_targets.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.target),
                    str(row.depmapDependencyTier),
                    str(int(row.profiledModels)),
                    str(int(row.dependentModels)),
                    fmt(row.dependentPct),
                    fmt(row.maxDependencyProbability, 3),
                    fmt(row.minGeneEffect, 3),
                    str(row.topDependentLineages)[:220],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A/B tiers mean the target has recurrent or subset cancer-cell dependency evidence and is better supported for oncology expert review.",
            "- C tier means weak or rare dependency and should be treated as review evidence rather than a standalone prioritization signal.",
            "- D tier means DepMap did not support material cancer dependency under the current thresholds.",
            "- U tier means coverage is missing; it is not negative biological evidence.",
            "",
            "## Output Files",
            "",
            "- `depmap_oncology_candidate_audit.csv`: oncology candidate-level dependency annotations.",
            "- `depmap_oncology_target_audit.csv`: target-level DepMap dependency metrics.",
            "- `depmap_oncology_lineage_summary.csv`: target-by-lineage dependency summaries.",
            "- `final_priority_depmap_oncology_matrix.csv`: full candidate matrix with oncology dependency columns.",
            "- `final_priority_depmap_oncology_top300_expert_shortlist.csv`: dependency-aware oncology shortlist.",
            "- `final_priority_depmap_oncology_review_queue.csv`: high-priority oncology candidates needing dependency-context review.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    source_path = root / args.source
    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    data_dir = root / args.data_dir

    candidates = read_candidates(source_path)
    oncology = candidates[candidates["direction"].astype(str).str.lower().eq("oncology")].copy()
    target_symbols = set(oncology["targetNorm"].dropna().astype(str))
    if not target_symbols:
        raise ValueError("No oncology targets found in the source candidate matrix.")

    depmap_meta = prepare_depmap_files(
        root,
        data_dir,
        args.release,
        args.force_download,
        args.include_gene_effect,
        args.download_tool,
    )
    if "CRISPRGeneEffect.csv" in depmap_meta["files"]:
        effect, _, effect_cols = read_depmap_matrix(Path(depmap_meta["files"]["CRISPRGeneEffect.csv"]), target_symbols)
    else:
        effect = pd.DataFrame({"ModelID": []})
        effect_cols = {}
    dependency, _, dep_cols = read_depmap_matrix(Path(depmap_meta["files"]["CRISPRGeneDependency.csv"]), target_symbols)
    model_meta = read_model_metadata(Path(depmap_meta["files"]["Model.csv"]))

    target_audit, lineage_audit = build_target_audit(target_symbols, effect, dependency, model_meta, effect_cols, dep_cols)
    counts = target_counts(candidates)
    target_audit = counts.merge(target_audit, on="target", how="right")
    target_audit = target_audit.sort_values(
        ["depmapDependencyPositiveFlag", "depmapDependencyScore", "dependentPct", "bestSotaContextScore"],
        ascending=[False, False, False, False],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    target_audit.to_csv(out_dir / "depmap_oncology_target_audit.csv", index=False)
    lineage_audit.sort_values(["target", "dependentModels", "dependentPct"], ascending=[True, False, False]).to_csv(
        out_dir / "depmap_oncology_lineage_summary.csv", index=False
    )

    merged, shortlist, review_queue = build_candidate_outputs(candidates, target_audit, out_dir, final_dir)
    summary = build_summary(source_path, candidates, target_audit, lineage_audit, merged, shortlist, review_queue, depmap_meta)
    write_json(out_dir / "depmap_oncology_dependency_summary.json", summary)
    write_markdown(out_dir / "DEPMAP_ONCOLOGY_DEPENDENCY_AUDIT.md", summary, target_audit)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DepMap CRISPR oncology dependency audit for final SOTA candidates.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--source",
        default="outputs/sota_validation/final_prioritization/final_priority_sota_context_matrix.csv",
        help="Candidate matrix to annotate.",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/depmap_oncology_dependency")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--data-dir", default="data/external/depmap")
    parser.add_argument("--release", default=None, help="Exact DepMap release name. Defaults to the latest complete public release.")
    parser.add_argument("--force-download", action="store_true", help="Redownload DepMap files even if local MD5 checks pass.")
    parser.add_argument("--include-gene-effect", action="store_true", help="Also download/read CRISPRGeneEffect.csv. This is stricter but larger and slower.")
    parser.add_argument("--download-tool", choices=["auto", "curl", "urllib"], default="auto", help="Downloader for DepMap files.")
    parser.add_argument(
        "--data-access-audit-only",
        action="store_true",
        help="Write candidate-scope and DepMap manifest/cache readiness outputs without downloading full matrices.",
    )
    parser.add_argument("--probe-urls", action="store_true", help="Probe DepMap file URLs with HEAD requests during access audit.")
    parser.add_argument("--probe-timeout", type=int, default=20, help="Per-file URL probe timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_data_access_audit(args) if args.data_access_audit_only else run(args)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

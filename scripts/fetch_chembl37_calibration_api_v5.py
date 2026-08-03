#!/usr/bin/env python3
"""Fetch an interim ChEMBL 37 numeric calibration set through the official API."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from scripts.extract_chembl37_target_calibration_v5 import (
    ANCHORS,
    CONFIG,
    ROOT,
    TARGETS,
    classify_pairs,
    clean,
    coverage_table,
    load_project_targets,
    now,
    sha256,
    split_ids,
)


API = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/chembl37_target_calibration_api_v5"
_THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(_THREAD_LOCAL, "session"):
        value = requests.Session()
        value.headers.update({"User-Agent": "BioMaster-calibrated-pipeline-v5/1.0"})
        _THREAD_LOCAL.session = value
    return _THREAD_LOCAL.session


def get_json(endpoint: str, params: dict[str, Any], retries: int = 7) -> dict[str, Any]:
    url = f"{API}/{endpoint}.json"
    for attempt in range(retries):
        try:
            response = session().get(url, params=params, timeout=90)
            if response.status_code == 200:
                return response.json()
            if response.status_code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
        except (requests.RequestException, ValueError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f"Request failed: {url} {params}: {error}") from error
        time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Request exhausted: {url} {params}")


def fetch_pages(endpoint: str, params: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    limit = 1000
    while True:
        query = dict(params)
        query.update({"limit": limit, "offset": offset})
        payload = get_json(endpoint, query)
        rows.extend(payload.get(key, []))
        meta = payload.get("page_meta", {})
        total = int(meta.get("total_count", len(rows)))
        if len(rows) >= total or not meta.get("next"):
            break
        offset += limit
    return rows


def target_ids(row: pd.Series) -> list[str]:
    values = split_ids(row.get("chembl_moa_target_chembl_ids", ""))
    return sorted(set(values))


def fetch_target(row: pd.Series, cache_dir: Path, force: bool = False) -> dict[str, Any]:
    sequence_key = clean(row["sequence_key"])
    cache_path = cache_dir / f"{sequence_key}.json.gz"
    if cache_path.is_file() and not force:
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    all_assays: dict[str, dict[str, Any]] = {}
    activities: dict[int, dict[str, Any]] = {}
    ids = target_ids(row)
    for target_id in ids:
        assays = fetch_pages(
            "assay",
            {
                "target_chembl_id": target_id,
                "confidence_score": 9,
                "assay_type": "B",
            },
            "assays",
        )
        all_assays.update({clean(assay.get("assay_chembl_id")): assay for assay in assays})
        numeric = fetch_pages(
            "activity",
            {
                "target_chembl_id": target_id,
                "assay_type": "B",
                "standard_type__in": "Ki,Kd,IC50",
                "pchembl_value__isnull": False,
            },
            "activities",
        )
        for activity in numeric:
            assay_id = clean(activity.get("assay_chembl_id"))
            if assay_id not in all_assays:
                continue
            if clean(activity.get("standard_relation")) != "=":
                continue
            if int(activity.get("potential_duplicate") or 0) != 0:
                continue
            if clean(activity.get("data_validity_comment")) not in {"", "Manually validated"}:
                continue
            activities[int(activity["activity_id"])] = activity

    payload = {
        "sequence_key": sequence_key,
        "primary_gene": clean(row["primary_gene"]),
        "target_assay_family": clean(row["target_assay_family"]),
        "query_accession": clean(row["query_accession"]),
        "target_chembl_ids": ids,
        "assays": list(all_assays.values()),
        "activities": list(activities.values()),
        "api_limitation": "numeric_equal_relation_only; explicit inactive text is added by the authoritative SQLite extraction",
    }
    temporary = cache_path.with_suffix(".json.gz.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    temporary.replace(cache_path)
    return payload


def aggregate(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        assay_map = {
            clean(item.get("assay_chembl_id")): item
            for item in payload["assays"]
        }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for activity in payload["activities"]:
            parent = clean(activity.get("parent_molecule_chembl_id")) or clean(
                activity.get("molecule_chembl_id")
            )
            if parent:
                grouped.setdefault(parent, []).append(activity)
        for parent, activities in grouped.items():
            pchembl = pd.to_numeric(
                pd.Series([item.get("pchembl_value") for item in activities]), errors="coerce"
            ).dropna()
            assay_ids = sorted({clean(item.get("assay_chembl_id")) for item in activities} - {""})
            documents = sorted({clean(item.get("document_chembl_id")) for item in activities} - {""})
            relationship = sorted(
                {
                    clean(assay_map.get(assay_id, {}).get("relationship_type"))
                    for assay_id in assay_ids
                }
                - {""}
            )
            exemplar = activities[0]
            rows.append(
                {
                    "sequence_key": payload["sequence_key"],
                    "primary_gene": payload["primary_gene"],
                    "target_assay_family": payload["target_assay_family"],
                    "query_accession": payload["query_accession"],
                    "target_chembl_id": clean(exemplar.get("target_chembl_id")),
                    "parent_molregno": parent,
                    "activity_rows": len(activities),
                    "assay_count": len(assay_ids),
                    "document_count": len(documents),
                    "min_pchembl": float(pchembl.min()),
                    "max_pchembl": float(pchembl.max()),
                    "mean_pchembl": float(pchembl.mean()),
                    "numeric_rows": int(len(pchembl)),
                    "any_explicit_inactive": 0,
                    "min_document_year": pd.to_numeric(
                        pd.Series([item.get("document_year") for item in activities]), errors="coerce"
                    ).min(),
                    "max_document_year": pd.to_numeric(
                        pd.Series([item.get("document_year") for item in activities]), errors="coerce"
                    ).max(),
                    "standard_types": ",".join(
                        sorted({clean(item.get("standard_type")) for item in activities} - {""})
                    ),
                    "relationship_types": ",".join(relationship),
                    "assay_ids": ",".join(assay_ids),
                    "doc_ids": ",".join(documents),
                    "parent_molecule_chembl_id": parent,
                    "parent_molecule_name": clean(exemplar.get("molecule_pref_name")),
                    "parent_canonical_smiles": clean(exemplar.get("canonical_smiles")),
                    "parent_standard_inchi_key": "",
                }
            )
    return pd.DataFrame(rows)


def api_mapping(project: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in project.iterrows():
        ids = target_ids(row)
        if not ids:
            rows.append(
                {
                    **row.to_dict(),
                    "target_chembl_id": None,
                    "is_direct_component": False,
                    "anchor_target_id_matches": False,
                }
            )
        for target_id in ids:
            rows.append(
                {
                    **row.to_dict(),
                    "target_chembl_id": target_id,
                    "is_direct_component": True,
                    "anchor_target_id_matches": True,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "api_cache"
    cache_dir.mkdir(exist_ok=True)

    status = get_json("status", {})
    if status.get("chembl_db_version") != "ChEMBL_37":
        raise RuntimeError(f"Expected ChEMBL_37 API, got {status}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    project = load_project_targets()
    if args.max_targets:
        project = project.head(args.max_targets).copy()

    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_target, row, cache_dir, args.force): clean(row["sequence_key"])
            for _, row in project.iterrows()
        }
        completed = 0
        for future in as_completed(futures):
            key = futures[future]
            try:
                payloads.append(future.result())
            except Exception as error:  # keep a resumable failure manifest
                failures.append({"sequence_key": key, "error": str(error)})
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed={completed}/{len(futures)} failures={len(failures)}", flush=True)

    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "API_FETCH_FAILURES_V5.csv", index=False)
        raise RuntimeError(f"ChEMBL API failures: {len(failures)}; rerun resumes from cache")

    pairs = classify_pairs(aggregate(payloads), config)
    mapped = api_mapping(project)
    coverage = coverage_table(project, mapped, pairs, config)
    mapped.to_csv(args.output_dir / "PROJECT_TARGET_CHEMBL37_API_MAPPING_V5.csv", index=False)
    pairs.to_csv(
        args.output_dir / "PROJECT_TARGET_CHEMBL37_API_NUMERIC_PAIRS_V5.csv.gz",
        index=False,
        compression="gzip",
    )
    coverage.to_csv(args.output_dir / "PROJECT_TARGET_CALIBRATION_API_COVERAGE_V5.csv", index=False)
    summary = {
        "status": "passed_interim_numeric_api",
        "created_utc": now(),
        "api_status": status,
        "project_targets": int(len(project)),
        "cached_target_payloads": int(len(payloads)),
        "pair_rows": int(len(pairs)),
        "positive_pairs": int(pairs["calibration_label"].eq("positive").sum()),
        "negative_pairs": int(pairs["calibration_label"].eq("negative_or_inactive").sum()),
        "grey_pairs": int(pairs["calibration_label"].eq("grey_or_unresolved").sum()),
        "conflicting_pairs": int(pairs["calibration_label"].eq("conflicting_exclude").sum()),
        "target_tiers": {
            str(key): int(value)
            for key, value in coverage["calibration_tier_v5"].value_counts().items()
        },
        "limitation": "Explicit inactive text is not included in the API interim set; the SQLite extraction is authoritative.",
        "inputs": {
            str(CONFIG.relative_to(ROOT)): sha256(CONFIG),
            str(TARGETS.relative_to(ROOT)): sha256(TARGETS),
            str(ANCHORS.relative_to(ROOT)): sha256(ANCHORS),
        },
    }
    (args.output_dir / "CHEMBL37_API_CALIBRATION_SUMMARY_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

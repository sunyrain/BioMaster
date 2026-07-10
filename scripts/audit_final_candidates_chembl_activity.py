#!/usr/bin/env python3
"""Audit exact active-moiety/target activity records through the ChEMBL API."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
MOLECULE_API = "https://www.ebi.ac.uk/chembl/api/data/molecule"
ASSAY_API = "https://www.ebi.ac.uk/chembl/api/data/assay"
CACHE_SCHEMA = "chembl_exact_pair_audit_v3"


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def split_ids(value: Any) -> list[str]:
    return sorted({token for token in re.split(r"[;,|\s]+", clean(value)) if token.startswith("CHEMBL")})


def fetch(molecule_id: str, target_id: str, retries: int = 4) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    offset = 0
    limit = 1000
    while True:
        params = {
            "molecule_chembl_id": molecule_id,
            "target_chembl_id": target_id,
            "limit": limit,
            "offset": offset,
        }
        payload: dict[str, Any] | None = None
        for attempt in range(retries):
            try:
                response = requests.get(API, params=params, timeout=60)
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 == retries:
                    return {"ok": False, "activities": records, "error": str(exc)}
                time.sleep(1.5 * (attempt + 1))
        if payload is None:
            return {"ok": False, "activities": records, "error": "unreachable"}
        page = payload.get("activities", [])
        records.extend(page)
        page_meta = payload.get("page_meta") or {}
        total = int(page_meta.get("total_count") or len(records))
        if not page or len(records) >= total or not page_meta.get("next"):
            return {"ok": True, "activities": records, "error": ""}
        offset += limit


def fetch_molecule_hierarchy(molecule_id: str, retries: int = 4) -> dict[str, Any]:
    """Resolve parent and salt/child IDs so activity lookup is active-moiety aware."""

    last_error = ""
    record: dict[str, Any] | None = None
    for attempt in range(retries):
        try:
            response = requests.get(f"{MOLECULE_API}/{molecule_id}.json", timeout=60)
            response.raise_for_status()
            record = response.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    if record is None:
        return {"ok": False, "molecule_ids": [molecule_id], "error": last_error}
    hierarchy = record.get("molecule_hierarchy") or {}
    parent_id = clean(hierarchy.get("parent_chembl_id")) or molecule_id
    children: list[str] = []
    for attempt in range(retries):
        try:
            response = requests.get(
                f"{MOLECULE_API}.json",
                params={
                    "molecule_hierarchy__parent_chembl_id": parent_id,
                    "limit": 1000,
                },
                timeout=60,
            )
            response.raise_for_status()
            children = [
                clean(item.get("molecule_chembl_id"))
                for item in response.json().get("molecules", [])
                if clean(item.get("molecule_chembl_id"))
            ]
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    else:
        return {
            "ok": False,
            "molecule_ids": sorted({molecule_id, parent_id}),
            "error": last_error,
        }
    return {
        "ok": True,
        "molecule_ids": sorted({molecule_id, parent_id, *children}),
        "parent_chembl_id": parent_id,
        "error": "",
    }


def fetch_assay_metadata(assay_id: str, retries: int = 4) -> dict[str, Any]:
    last_error = ""
    for attempt in range(retries):
        try:
            response = requests.get(f"{ASSAY_API}/{assay_id}.json", timeout=60)
            response.raise_for_status()
            payload = response.json()
            return {
                "ok": True,
                "confidence_score": payload.get("confidence_score"),
                "relationship_type": payload.get("relationship_type"),
                "target_chembl_id": payload.get("target_chembl_id"),
                "assay_variant_accession": payload.get("assay_variant_accession"),
                "assay_variant_mutation": payload.get("assay_variant_mutation"),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    return {"ok": False, "error": last_error}


def summarize(
    records: list[dict[str, Any]],
    assay_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def numeric(values: list[Any]) -> list[float]:
        parsed = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").dropna()
        return parsed.astype(float).tolist()

    pchembl = numeric([record.get("pchembl_value") for record in records])
    binding = [record for record in records if clean(record.get("assay_type")).upper() == "B"]
    binding_pchembl = numeric([record.get("pchembl_value") for record in binding])
    years = sorted(
        {int(record["document_year"]) for record in records if clean(record.get("document_year"))}
    )
    documents = sorted({clean(record.get("document_chembl_id")) for record in records if clean(record.get("document_chembl_id"))})
    assay_metadata = assay_metadata or {}
    raw_strong_records = [
        record
        for record in binding
        if (value := pd.to_numeric(record.get("pchembl_value"), errors="coerce")) == value
        and float(value) >= 5.0
    ]

    def high_quality(record: dict[str, Any]) -> bool:
        assay = assay_metadata.get(clean(record.get("assay_chembl_id")), {})
        confidence = pd.to_numeric(assay.get("confidence_score"), errors="coerce")
        validity = clean(record.get("data_validity_comment"))
        relation = clean(record.get("standard_relation")) or clean(record.get("relation"))
        variant = clean(record.get("assay_variant_mutation")) or clean(
            assay.get("assay_variant_mutation")
        )
        standard_flag = clean(record.get("standard_flag"))
        return (
            bool(assay.get("ok"))
            and confidence == confidence
            and float(confidence) >= 8.0
            and not validity
            and relation == "="
            and not variant
            and standard_flag in {"1", "1.0", "True", "true"}
        )

    strong_quality = [record for record in raw_strong_records if high_quality(record)]
    manual_strong = [record for record in raw_strong_records if not high_quality(record)]
    if strong_quality:
        status = "exact_binding_activity_pchembl_ge_5"
    elif manual_strong:
        status = "manual_exact_binding_review"
    elif binding:
        status = "exact_binding_record_without_strong_standardized_potency"
    elif records:
        status = "exact_nonbinding_activity_record"
    else:
        status = "no_exact_chembl_activity_record"
    return {
        "chembl_exact_activity_status": status,
        "chembl_exact_activity_count": len(records),
        "chembl_exact_binding_count": len(binding),
        "chembl_exact_binding_pchembl_ge_5_count": len(strong_quality),
        "chembl_exact_raw_binding_pchembl_ge_5_count": len(raw_strong_records),
        "chembl_exact_manual_binding_review_count": len(manual_strong),
        "chembl_exact_max_pchembl": max(pchembl) if pchembl else None,
        "chembl_exact_max_binding_pchembl": max(binding_pchembl) if binding_pchembl else None,
        "chembl_exact_document_years": ";".join(map(str, years[:30])),
        "chembl_exact_document_ids": ";".join(documents[:30]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument(
        "--anchors",
        default="outputs/chembl_moa_enhanced_information_package_v1/chembl_moa_anchor_gene_table_v2.csv",
    )
    parser.add_argument("--drug-library", default="data/processed/drug_library_active_moiety_v4.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh-failed", action="store_true", default=True)
    parser.add_argument("--no-refresh-failed", dest="refresh_failed", action="store_false")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, low_memory=False).fillna("")
    anchors = pd.read_csv(args.anchors, low_memory=False).fillna("")
    drugs = pd.read_csv(args.drug_library, low_memory=False).fillna("")
    target_ids = anchors.drop_duplicates("gene").set_index("gene")["chembl_moa_target_chembl_ids"].to_dict()
    active_to_ids = (
        drugs.groupby("model_ligand_smiles")["drug_id"]
        .apply(lambda values: sorted({re.sub(r"__.*$", "", clean(value)) for value in values if clean(value)}))
        .to_dict()
    )

    cache_path = Path(args.cache)
    cache: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    cache["__meta__"] = {"schema": CACHE_SCHEMA}

    def needs_query(key: str) -> bool:
        return key not in cache or (
            args.refresh_failed
            and isinstance(cache.get(key), dict)
            and not bool(cache[key].get("ok"))
        )
    row_specs: list[tuple[list[str], list[str]]] = []
    for _, row in candidates.iterrows():
        ligand = clean(row.get("model_ligand_smiles")) or clean(row.get("active_moiety_smiles"))
        molecule_ids = active_to_ids.get(ligand, [re.sub(r"__.*$", "", clean(row.get("drug_chembl_id")))])
        targets = split_ids(target_ids.get(clean(row.get("primary_gene")), ""))
        row_specs.append((molecule_ids, targets))

    hierarchy_keys = {
        f"hierarchy::{molecule_id}": molecule_id
        for molecule_ids, _ in row_specs
        for molecule_id in molecule_ids
        if needs_query(f"hierarchy::{molecule_id}")
    }
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_molecule_hierarchy, molecule_id): key
            for key, molecule_id in hierarchy_keys.items()
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()

    requests_needed: dict[str, tuple[str, str]] = {}
    row_queries: list[list[str]] = []
    row_molecule_ids: list[list[str]] = []
    row_hierarchy_errors: list[list[str]] = []
    for base_ids, targets in row_specs:
        molecule_ids: set[str] = set()
        hierarchy_errors: list[str] = []
        if not targets:
            hierarchy_errors.append("missing_target_chembl_id")
        for base_id in base_ids:
            hierarchy = cache.get(
                f"hierarchy::{base_id}",
                {"ok": False, "molecule_ids": [base_id], "error": "missing_hierarchy_cache"},
            )
            molecule_ids.update(hierarchy.get("molecule_ids") or [base_id])
            if not hierarchy.get("ok"):
                hierarchy_errors.append(clean(hierarchy.get("error")) or "hierarchy_query_failed")
        keys = []
        for molecule_id in sorted(molecule_ids):
            for target_id in targets:
                key = f"{molecule_id}__{target_id}"
                keys.append(key)
                if needs_query(key):
                    requests_needed[key] = (molecule_id, target_id)
        row_queries.append(keys)
        row_molecule_ids.append(sorted(molecule_ids))
        row_hierarchy_errors.append(hierarchy_errors)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch, molecule_id, target_id): key
            for key, (molecule_id, target_id) in requests_needed.items()
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()

    assay_ids = sorted(
        {
            clean(record.get("assay_chembl_id"))
            for keys in row_queries
            for key in keys
            for record in cache.get(key, {}).get("activities", [])
            if clean(record.get("assay_chembl_id"))
        }
    )
    assay_keys = {
        f"assay::{assay_id}": assay_id
        for assay_id in assay_ids
        if needs_query(f"assay::{assay_id}")
    }
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_assay_metadata, assay_id): key
            for key, assay_id in assay_keys.items()
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    annotations = []
    for keys, molecule_ids, hierarchy_errors in zip(
        row_queries, row_molecule_ids, row_hierarchy_errors, strict=True
    ):
        records = []
        errors = list(hierarchy_errors)
        for key in keys:
            result = cache.get(key, {"ok": False, "activities": [], "error": "missing_cache"})
            records.extend(result.get("activities", []))
            if not result.get("ok"):
                errors.append(clean(result.get("error")))
        # The same activity can be returned through parent/salt-equivalent IDs.
        unique = {str(record.get("activity_id")): record for record in records}
        assay_meta = {
            assay_id: cache.get(f"assay::{assay_id}", {"ok": False, "error": "missing_assay_cache"})
            for assay_id in {
                clean(record.get("assay_chembl_id")) for record in unique.values()
                if clean(record.get("assay_chembl_id"))
            }
        }
        assay_errors = [
            clean(value.get("error")) or "assay_metadata_query_failed"
            for value in assay_meta.values()
            if not value.get("ok")
        ]
        errors.extend(assay_errors)
        annotation = summarize(list(unique.values()), assay_meta)
        annotation["chembl_activity_query_ok"] = not errors
        annotation["chembl_activity_query_errors"] = ";".join(error for error in errors if error)
        annotation["chembl_hierarchy_query_ok"] = not hierarchy_errors
        annotation["chembl_molecule_ids_queried"] = ";".join(molecule_ids)
        annotation["chembl_assay_metadata_query_ok"] = not assay_errors
        annotations.append(annotation)
    annotation_df = pd.DataFrame(annotations, index=candidates.index)
    result = pd.concat([candidates, annotation_df], axis=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "rows": len(result),
        "new_api_queries": len(requests_needed),
        "new_hierarchy_queries": len(hierarchy_keys),
        "new_assay_metadata_queries": len(assay_keys),
        "cache_schema": CACHE_SCHEMA,
        "query_failures": int((~result["chembl_activity_query_ok"]).sum()),
        "status_counts": result["chembl_exact_activity_status"].value_counts().to_dict(),
        "warning": "ChEMBL assay records are exact-pair evidence, but assay type and standardized potency must be reviewed before calling direct binding.",
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

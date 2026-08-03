#!/usr/bin/env python3
"""Build a cached PDBe experimental/holo structure atlas for project targets."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "outputs" / "affinity_first_remote_discovery_v1" / "TARGET_MODEL_READINESS_463_V1.csv"
DEFAULT_OUT = ROOT / "outputs" / "affinity_first_remote_discovery_v1" / "experimental_structure_atlas_v1"
PDBe = "https://www.ebi.ac.uk/pdbe/api"
COMMON_ARTIFACTS = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "ACT", "ACE", "FMT",
    "CL", "BR", "IOD", "NA", "K", "MG", "CA", "ZN", "MN", "CO", "NI", "CU", "FE", "CD",
    "NH4", "NO3", "CO3", "CIT", "TRS", "MES", "HEP", "BME", "DMS", "IPA", "MPD", "NAG",
    "MAN", "BGC", "GLC", "FUC", "GAL", "SIA", "MSE",
}


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def get_json(url: str, retries: int = 4) -> tuple[int, Any]:
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 404:
                return 404, {}
            response.raise_for_status()
            return response.status_code, response.json()
        except Exception:
            if attempt + 1 == retries:
                return 599, {}
            time.sleep(1.0 + attempt)
    return 599, {}


def best_structures(accessions: list[str]) -> tuple[str, list[dict[str, Any]], int]:
    for accession in accessions:
        status, payload = get_json(f"{PDBe}/mappings/best_structures/{accession}")
        rows = payload.get(accession, []) if isinstance(payload, dict) else []
        if rows:
            return accession, rows, status
    return "", [], 404


def ligand_rows(pdb_id: str) -> list[dict[str, Any]]:
    status, payload = get_json(f"{PDBe}/pdb/entry/ligand_monomers/{pdb_id}")
    if status != 200 or not isinstance(payload, dict):
        return []
    return payload.get(pdb_id.lower(), payload.get(pdb_id.upper(), [])) or []


def is_candidate_ligand(row: dict[str, Any]) -> bool:
    code = clean(row.get("chem_comp_id")).upper()
    try:
        weight = float(row.get("weight") or 0.0)
    except (TypeError, ValueError):
        weight = 0.0
    return code not in COMMON_ARTIFACTS and 120.0 <= weight <= 1200.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-entries-per-target", type=int, default=20)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.targets, low_memory=False).fillna("")

    target_results: dict[str, tuple[str, list[dict[str, Any]], int]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for _, row in targets.iterrows():
            sequence_key = clean(row["sequence_key"])
            candidates = []
            for value in [row.get("representative_protein_id"), row.get("anchor_canonical_uniprot")]:
                accession = clean(value)
                if accession and accession not in candidates:
                    candidates.append(accession)
                base = accession.split("-")[0]
                if base and base not in candidates:
                    candidates.append(base)
            futures[executor.submit(best_structures, candidates)] = sequence_key
        for future in as_completed(futures):
            target_results[futures[future]] = future.result()

    entry_ids: set[str] = set()
    target_entry_rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        sequence_key = clean(target["sequence_key"])
        matched_accession, structures, api_status = target_results.get(sequence_key, ("", [], 599))
        sorted_structures = sorted(
            structures,
            key=lambda row: (
                -(float(row.get("coverage") or 0.0)),
                float(row.get("resolution") or 99.0),
            ),
        )[: args.max_entries_per_target]
        for structure in sorted_structures:
            pdb_id = clean(structure.get("pdb_id")).lower()
            if not pdb_id:
                continue
            entry_ids.add(pdb_id)
            target_entry_rows.append(
                {
                    "sequence_key": sequence_key,
                    "primary_gene": clean(target.get("primary_gene")),
                    "matched_uniprot": matched_accession,
                    "pdb_id": pdb_id,
                    "chain_id": clean(structure.get("chain_id")),
                    "experimental_method": clean(structure.get("experimental_method")),
                    "resolution": structure.get("resolution"),
                    "coverage": structure.get("coverage"),
                    "unp_start": structure.get("unp_start"),
                    "unp_end": structure.get("unp_end"),
                    "api_status": api_status,
                }
            )

    ligands_by_entry: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(ligand_rows, pdb_id): pdb_id for pdb_id in sorted(entry_ids)}
        for future in as_completed(futures):
            pdb_id = futures[future]
            rows = future.result()
            ligands_by_entry[pdb_id] = rows
            (cache_dir / f"{pdb_id}_ligands.json").write_text(
                json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    entry_frame = pd.DataFrame(target_entry_rows)
    if entry_frame.empty:
        entry_frame = pd.DataFrame(columns=["sequence_key", "primary_gene", "pdb_id"])
    entry_frame["entry_ligand_count"] = entry_frame["pdb_id"].map(
        lambda pdb_id: len(ligands_by_entry.get(clean(pdb_id), []))
    )
    entry_frame["entry_candidate_small_molecule_count"] = entry_frame["pdb_id"].map(
        lambda pdb_id: sum(is_candidate_ligand(row) for row in ligands_by_entry.get(clean(pdb_id), []))
    )
    entry_frame["entry_has_candidate_small_molecule"] = entry_frame[
        "entry_candidate_small_molecule_count"
    ].gt(0)
    entry_frame["experimental_structure_download_url"] = entry_frame["pdb_id"].map(
        lambda pdb_id: f"https://files.rcsb.org/download/{str(pdb_id).upper()}.cif.gz"
    )

    ligand_output_rows: list[dict[str, Any]] = []
    for pdb_id, rows in ligands_by_entry.items():
        for row in rows:
            ligand_output_rows.append(
                {
                    "pdb_id": pdb_id,
                    "chem_comp_id": clean(row.get("chem_comp_id")),
                    "chem_comp_name": clean(row.get("chem_comp_name")),
                    "weight": row.get("weight"),
                    "chain_id": clean(row.get("chain_id")),
                    "author_residue_number": row.get("author_residue_number"),
                    "candidate_small_molecule": is_candidate_ligand(row),
                }
            )
    ligand_frame = pd.DataFrame(ligand_output_rows)

    per_target_rows = []
    for _, target in targets.iterrows():
        sequence_key = clean(target["sequence_key"])
        group = entry_frame[entry_frame["sequence_key"].eq(sequence_key)]
        holo = group[group["entry_has_candidate_small_molecule"]]
        per_target_rows.append(
            {
                "sequence_key": sequence_key,
                "primary_gene": clean(target.get("primary_gene")),
                "representative_protein_id": clean(target.get("representative_protein_id")),
                "structure_bin": clean(target.get("structure_bin")),
                "experimental_entry_count_top20": int(len(group)),
                "candidate_holo_entry_count_top20": int(len(holo)),
                "has_candidate_experimental_holo": bool(len(holo)),
                "best_holo_pdb_id": clean(holo.iloc[0]["pdb_id"]) if len(holo) else "",
                "best_holo_resolution": holo.iloc[0]["resolution"] if len(holo) else "",
                "best_holo_coverage": holo.iloc[0]["coverage"] if len(holo) else "",
                "holo_validation_status": (
                    "entry_level_candidate_ligand_pending_chain_distance_validation" if len(holo) else "no_candidate_holo_in_top20"
                ),
            }
        )
    per_target = pd.DataFrame(per_target_rows)
    target_output = out_dir / "TARGET_EXPERIMENTAL_STRUCTURE_ATLAS_463_V1.csv"
    entry_output = out_dir / "TARGET_EXPERIMENTAL_ENTRY_CANDIDATES_V1.csv"
    ligand_output = out_dir / "EXPERIMENTAL_ENTRY_LIGANDS_V1.csv"
    per_target.to_csv(target_output, index=False)
    entry_frame.to_csv(entry_output, index=False)
    ligand_frame.to_csv(ligand_output, index=False)
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets": int(len(per_target)),
        "targets_with_experimental_entries_top20": int(per_target["experimental_entry_count_top20"].gt(0).sum()),
        "targets_with_entry_level_candidate_holo": int(per_target["has_candidate_experimental_holo"].sum()),
        "unique_experimental_entries": int(entry_frame["pdb_id"].nunique()),
        "candidate_holo_entry_rows": int(entry_frame["entry_has_candidate_small_molecule"].sum()),
        "warning": (
            "Candidate holo means the PDB entry contains a non-artifact 120-1200 Da ligand. "
            "Ligand-to-target-chain distance and binding-site equivalence remain pending."
        ),
        "outputs": {
            "target_atlas": str(target_output.relative_to(ROOT)),
            "entry_candidates": str(entry_output.relative_to(ROOT)),
            "entry_ligands": str(ligand_output.relative_to(ROOT)),
        },
    }
    (out_dir / "EXPERIMENTAL_STRUCTURE_ATLAS_V1_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

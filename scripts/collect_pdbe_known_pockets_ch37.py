#!/usr/bin/env python3
"""Collect residue-level experimental ligand pockets from the current PDBe API."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
OUTDIR = ROOT / "outputs/chembl37_known_pocket_atlas/pdbe"
RAW = OUTDIR / "raw_by_uniprot"
API = "https://www.ebi.ac.uk/pdbe/api"
THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    if not hasattr(THREAD_LOCAL, "session"):
        value = requests.Session()
        value.headers.update({"User-Agent": "BioMaster-known-pocket-atlas/1.0"})
        THREAD_LOCAL.session = value
    return THREAD_LOCAL.session


def fetch_json(url: str, timeout: int, retries: int = 4) -> tuple[int, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session().get(url, timeout=timeout)
            if response.status_code == 404:
                return 404, {}
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.status_code, response.json()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def cache_path(accession: str) -> Path:
    return RAW / f"{accession}.json.gz"


def fetch_accession(accession: str, timeout: int, force: bool) -> dict[str, Any]:
    path = cache_path(accession)
    if path.is_file() and not force:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    result: dict[str, Any] = {"accession": accession, "status": "", "error": ""}
    try:
        ligand_status, ligand_sites = fetch_json(
            f"{API}/uniprot/ligand_sites/{accession}", timeout
        )
        structure_status, best_structures = fetch_json(
            f"{API}/uniprot/best_structures/{accession}", timeout
        )
        result.update({
            "status": "completed",
            "ligand_sites_http_status": ligand_status,
            "best_structures_http_status": structure_status,
            "ligand_sites": ligand_sites,
            "best_structures": best_structures,
        })
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)
    return result


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ligand_class(additional: dict[str, Any], minimum_atoms: int) -> str:
    if bool(additional.get("isSolvent")):
        return "SOLVENT_OR_ADDITIVE"
    atoms = pd.to_numeric(additional.get("numAtoms"), errors="coerce")
    chembl = clean(additional.get("chemblId"))
    drugbank = clean(additional.get("drugBankId"))
    cofactor = clean(additional.get("coFactorId"))
    if chembl or drugbank:
        return "DRUG_MAPPED"
    if cofactor:
        return "COFACTOR"
    if pd.notna(atoms) and float(atoms) >= minimum_atoms:
        return "DRUGLIKE_UNMAPPED"
    if pd.notna(atoms) and float(atoms) >= 3:
        return "SMALL_FUNCTIONAL_OR_FRAGMENT"
    return "ION_OR_TINY_FRAGMENT"


def best_structure_index(payload: dict[str, Any], accession: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload.get(accession, []):
        pdb_id = clean(row.get("pdb_id")).lower()
        chain = clean(row.get("chain_id"))
        entity = clean(row.get("entity_id"))
        output[(pdb_id, entity, chain)] = row
    return output


def flatten_one(
    target: dict[str, Any], payload: dict[str, Any], minimum_atoms: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accession = target["uniprot_accession"]
    ligand_root = payload.get("ligand_sites", {}).get(accession, {})
    structures_root = payload.get("best_structures", {})
    pdbe_sequence = clean(ligand_root.get("sequence"))
    sequence_exact = bool(pdbe_sequence and hashlib.sha256(pdbe_sequence.encode("ascii")).hexdigest() == target["sequence_sha256"])
    structure_index = best_structure_index(structures_root, accession)
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for ligand in ligand_root.get("data", []) or []:
        ligand_id = clean(ligand.get("accession"))
        additional = ligand.get("additionalData") or {}
        classification = ligand_class(additional, minimum_atoms)
        for residue in ligand.get("residues", []) or []:
            position = int(residue["startIndex"])
            amino_acid = clean(residue.get("startCode"))
            for interaction in residue.get("interactingPDBEntries", []) or []:
                pdb_id = clean(interaction.get("pdbId")).lower()
                entity_id = clean(interaction.get("entityId"))
                chain_text = clean(interaction.get("chainIds"))
                chains = [value.strip() for value in chain_text.replace(";", ",").split(",") if value.strip()] or [""]
                for chain in chains:
                    key = (pdb_id, entity_id, chain, ligand_id)
                    if key not in grouped:
                        grouped[key] = {
                            "target_chembl_id": target["target_chembl_id"],
                            "gene_symbol": target["gene_symbol"],
                            "uniprot_accession": accession,
                            "pdb_id": pdb_id,
                            "entity_id": entity_id,
                            "chain_id": chain,
                            "ligand_id": ligand_id,
                            "ligand_name": clean(ligand.get("name")),
                            "ligand_class": classification,
                            "ligand_num_atoms": additional.get("numAtoms"),
                            "ligand_chembl_id": clean(additional.get("chemblId")),
                            "ligand_drugbank_id": clean(additional.get("drugBankId")),
                            "ligand_cofactor_id": clean(additional.get("coFactorId")),
                            "ligand_is_solvent": bool(additional.get("isSolvent")),
                            "ligand_significance": additional.get("significance"),
                            "ligand_scaffold": clean(additional.get("scaffoldId")),
                            "pdbe_sequence_exact": sequence_exact,
                            "residues": {},
                        }
                    grouped[key]["residues"][position] = amino_acid
    rows: list[dict[str, Any]] = []
    for (pdb_id, entity_id, chain, ligand_id), row in grouped.items():
        structure = structure_index.get((pdb_id, entity_id, chain), {})
        positions = sorted(row.pop("residues"))
        amino_acids = []
        # Recover residue codes from the ligand record grouping without making the output nested.
        for ligand in ligand_root.get("data", []) or []:
            if clean(ligand.get("accession")) != ligand_id:
                continue
            by_position = {
                int(value["startIndex"]): clean(value.get("startCode"))
                for value in ligand.get("residues", []) or []
            }
            amino_acids = [by_position.get(value, "") for value in positions]
            break
        row.update({
            "source": "PDBE",
            "pocket_instance_id": f"PDBE:{accession}:{pdb_id}:{entity_id}:{chain}:{ligand_id}",
            "uniprot_residue_positions": ";".join(map(str, positions)),
            "uniprot_residue_codes": ";".join(amino_acids),
            "binding_residue_count": len(positions),
            "experimental_method": clean(structure.get("experimental_method")),
            "resolution": structure.get("resolution"),
            "structure_tax_id": structure.get("tax_id"),
            "structure_coverage": structure.get("coverage"),
            "preferred_assembly_id": structure.get("preferred_assembly_id"),
        })
        rows.append(row)
    summary = {
        "target_chembl_id": target["target_chembl_id"],
        "gene_symbol": target["gene_symbol"],
        "uniprot_accession": accession,
        "collection_status": payload.get("status"),
        "collection_error": payload.get("error", ""),
        "pdbe_sequence_available": bool(pdbe_sequence),
        "pdbe_sequence_exact": sequence_exact,
        "pdbe_ligand_pocket_instances": len(rows),
        "pdbe_unique_pdbs": len({row["pdb_id"] for row in rows}),
        "pdbe_unique_ligands": len({row["ligand_id"] for row in rows}),
        "pdbe_drug_mapped_instances": sum(row["ligand_class"] == "DRUG_MAPPED" for row in rows),
        "pdbe_druglike_instances": sum(row["ligand_class"] in {"DRUG_MAPPED", "DRUGLIKE_UNMAPPED"} for row in rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--minimum-atoms", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    master = pd.read_csv(MASTER, low_memory=False)
    if len(master) != 888:
        raise RuntimeError("Expected 888 targets")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    targets = master[[
        "target_chembl_id", "gene_symbol", "uniprot_accession", "sequence_sha256"
    ]].to_dict(orient="records")
    payloads: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {
            executor.submit(fetch_accession, row["uniprot_accession"], args.timeout, args.force): row
            for row in targets
        }
        for index, future in enumerate(as_completed(jobs), start=1):
            row = jobs[future]
            payload = future.result()
            payloads[row["uniprot_accession"]] = payload
            if index % 25 == 0 or index == len(jobs):
                print(f"PDBe {index}/{len(jobs)}", flush=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target in targets:
        rows, summary = flatten_one(
            target, payloads[target["uniprot_accession"]], args.minimum_atoms
        )
        all_rows.extend(rows)
        summaries.append(summary)
    pockets = pd.DataFrame(all_rows)
    coverage = pd.DataFrame(summaries)
    pockets.to_csv(OUTDIR / "PDBE_EXPERIMENTAL_LIGAND_POCKET_INSTANCES.csv.gz", index=False, compression="gzip")
    coverage.to_csv(OUTDIR / "PDBE_TARGET_COVERAGE_888.csv", index=False)
    summary = {
        "targets": 888,
        "collection_status_counts": coverage["collection_status"].value_counts(dropna=False).to_dict(),
        "targets_with_any_ligand_pocket": int(coverage["pdbe_ligand_pocket_instances"].gt(0).sum()),
        "targets_with_druglike_pocket": int(coverage["pdbe_druglike_instances"].gt(0).sum()),
        "pocket_instances": int(len(pockets)),
        "unique_pdbs": int(pockets["pdb_id"].nunique()) if len(pockets) else 0,
        "unique_ligands": int(pockets["ligand_id"].nunique()) if len(pockets) else 0,
        "sequence_exact_targets": int(coverage["pdbe_sequence_exact"].sum()),
    }
    (OUTDIR / "PDBE_COLLECTION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

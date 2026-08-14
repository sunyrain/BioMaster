#!/usr/bin/env python3
"""Collect project-relevant scPDB interaction sites and map them to UniProt residues."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PDBE = ROOT / "outputs/chembl37_known_pocket_atlas/pdbe/PDBE_EXPERIMENTAL_LIGAND_POCKET_INSTANCES.csv.gz"
ENTRY_LIST = ROOT / "downloads/known_pocket_sources_v1/scPDB_entries.lst"
OUTDIR = ROOT / "outputs/chembl37_known_pocket_atlas/scpdb"
CACHE = OUTDIR / "raw_interactions"
BASE = "https://drugdesign.unistra.fr/scPDB/ressources/2016/entries"
THREAD_LOCAL = threading.local()

# Matches both protein and ligand residue tokens in lines such as
# "|ALA  15-A ... |GSP 538-A".
RESIDUE_TOKEN = re.compile(r"\|\s*([A-Za-z0-9]{1,4})\s+(-?\d+[A-Za-z]?)\s*-\s*([^\s|]+)")
IFP_TOKEN = re.compile(r"\|([^|]+)")


def session() -> requests.Session:
    if not hasattr(THREAD_LOCAL, "session"):
        value = requests.Session()
        value.headers.update({"User-Agent": "BioMaster-known-pocket-atlas/1.0"})
        THREAD_LOCAL.session = value
    return THREAD_LOCAL.session


def get_text(url: str, timeout: int, retries: int = 4) -> tuple[int, str]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session().get(url, timeout=timeout)
            if response.status_code == 404:
                return 404, ""
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.status_code, response.text
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.25 * (attempt + 1))
    assert error is not None
    raise error


def fetch_entry(entry: str, timeout: int, force: bool) -> dict[str, object]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{entry}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, object] = {"entry": entry, "status": "", "error": ""}
    try:
        ints_status, ints = get_text(f"{BASE}/{entry}/intsifp.txt", timeout)
        ifp_status, ifp = get_text(f"{BASE}/{entry}/IFP.txt", timeout)
        output.update({
            "status": "completed",
            "ints_http_status": ints_status,
            "ifp_http_status": ifp_status,
            "intsifp": ints,
            "ifp": ifp,
        })
    except Exception as exc:
        output.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return output


def parse_interactions(payload: dict[str, object]) -> dict[str, object]:
    protein_residues: set[tuple[str, str, str]] = set()
    ligand_tokens: list[tuple[str, str, str]] = []
    interaction_types: set[str] = set()
    for line in str(payload.get("intsifp", "")).splitlines():
        tokens = RESIDUE_TOKEN.findall(line)
        if len(tokens) < 2:
            continue
        protein, ligand = tokens[0], tokens[-1]
        protein_residues.add((protein[2], protein[1], protein[0].upper()))
        ligand_tokens.append((ligand[2], ligand[1], ligand[0].upper()))
        interaction_types.add(line.split("\t", 1)[0].strip())

    # IFP first line represents the complete 6.5 A site and is a useful fallback
    # when no atom-level interaction was assigned.
    site_residues: set[tuple[str, str, str]] = set()
    ifp_lines = str(payload.get("ifp", "")).splitlines()
    if ifp_lines:
        for value in IFP_TOKEN.findall(ifp_lines[0]):
            match = re.match(r"\s*([^\s])\s+([A-Za-z0-9]{1,4})(-?\d+[A-Za-z]?)\s*$", value)
            if match:
                site_residues.add((match.group(1), match.group(3), match.group(2).upper()))

    if ligand_tokens:
        ligand_counts = pd.Series(ligand_tokens).value_counts()
        ligand_chain, ligand_resnum, ligand_id = ligand_counts.index[0]
    else:
        ligand_chain = ligand_resnum = ligand_id = ""
    receptor_chains = sorted({value[0] for value in protein_residues or site_residues})
    return {
        "receptor_chains": ";".join(receptor_chains),
        "ligand_id": ligand_id,
        "ligand_chain": ligand_chain,
        "ligand_pdb_resnum": ligand_resnum,
        "interaction_residues_pdb": ";".join(
            f"{chain}:{name}{number}" for chain, number, name in sorted(protein_residues)
        ),
        "interaction_residue_count": len(protein_residues),
        "site_residues_pdb": ";".join(
            f"{chain}:{name}{number}" for chain, number, name in sorted(site_residues)
        ),
        "site_residue_count": len(site_residues),
        "interaction_types": ";".join(sorted(interaction_types)),
    }


def map_to_pdbe(entries: pd.DataFrame, pdbe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_pdb = {key: value for key, value in pdbe.groupby("pdb_id", sort=False)}
    for record in entries.to_dict(orient="records"):
        candidates = by_pdb.get(record["pdb_id"])
        if candidates is None:
            continue
        receptor_chains = set(str(record["receptor_chains"]).split(";")) - {""}
        ligand_id = str(record["ligand_id"])
        matched = candidates[
            candidates["chain_id"].astype(str).isin(receptor_chains)
            & candidates["ligand_id"].astype(str).str.upper().eq(ligand_id.upper())
        ]
        mapping_level = "EXACT_CHAIN_LIGAND"
        if matched.empty:
            matched = candidates[candidates["chain_id"].astype(str).isin(receptor_chains)]
            mapping_level = "CHAIN_ONLY"
        if matched.empty:
            mapping_level = "UNRESOLVED_PROJECT_PDB"
            rows.append({**record, "scpdb_mapping_level": mapping_level})
            continue
        for _, match in matched.iterrows():
            rows.append({
                **record,
                "scpdb_mapping_level": mapping_level,
                "target_chembl_id": match["target_chembl_id"],
                "gene_symbol": match["gene_symbol"],
                "uniprot_accession": match["uniprot_accession"],
                "pdbe_pocket_instance_id": match["pocket_instance_id"],
                "uniprot_residue_positions": match["uniprot_residue_positions"] if mapping_level == "EXACT_CHAIN_LIGAND" else "",
                "binding_residue_count": match["binding_residue_count"] if mapping_level == "EXACT_CHAIN_LIGAND" else pd.NA,
                "pdbe_ligand_class": match["ligand_class"] if mapping_level == "EXACT_CHAIN_LIGAND" else "",
                "experimental_method": match["experimental_method"],
                "resolution": match["resolution"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not ENTRY_LIST.is_file():
        raise FileNotFoundError(ENTRY_LIST)
    pdbe = pd.read_csv(PDBE, low_memory=False)
    project_pdbs = set(pdbe["pdb_id"].astype(str).str.lower())
    all_entries = [value.strip() for value in ENTRY_LIST.read_text().splitlines() if value.strip()]
    selected = [value for value in all_entries if value.split("_", 1)[0].lower() in project_pdbs]
    payloads: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(fetch_entry, value, args.timeout, args.force): value for value in selected}
        for index, future in enumerate(as_completed(jobs), start=1):
            payloads[jobs[future]] = future.result()
            if index % 100 == 0 or index == len(jobs):
                print(f"scPDB {index}/{len(jobs)}", flush=True)
    parsed = []
    for entry in selected:
        payload = payloads[entry]
        parsed.append({
            "source": "SCPDB_2017",
            "scpdb_entry": entry,
            "pdb_id": entry.split("_", 1)[0].lower(),
            "site_number": entry.split("_", 1)[1] if "_" in entry else "",
            "collection_status": payload.get("status"),
            "collection_error": payload.get("error"),
            **parse_interactions(payload),
        })
    parsed_frame = pd.DataFrame(parsed)
    mapped = map_to_pdbe(parsed_frame, pdbe)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    parsed_frame.to_csv(OUTDIR / "SCPDB_PROJECT_ENTRIES_PARSED.csv.gz", index=False, compression="gzip")
    mapped.to_csv(OUTDIR / "SCPDB_PROJECT_POCKET_MAPPINGS.csv.gz", index=False, compression="gzip")
    summary = {
        "scpdb_all_entries": len(all_entries),
        "project_pdb_overlap_entries": len(selected),
        "collection_status_counts": parsed_frame["collection_status"].value_counts(dropna=False).to_dict(),
        "entries_with_interactions": int(parsed_frame["interaction_residue_count"].gt(0).sum()),
        "entries_with_site_residues": int(parsed_frame["site_residue_count"].gt(0).sum()),
        "mapping_level_counts": mapped["scpdb_mapping_level"].value_counts(dropna=False).to_dict() if len(mapped) else {},
        "exact_mapped_targets": int(mapped.loc[mapped["scpdb_mapping_level"].eq("EXACT_CHAIN_LIGAND"), "uniprot_accession"].nunique()) if len(mapped) else 0,
    }
    (OUTDIR / "SCPDB_COLLECTION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

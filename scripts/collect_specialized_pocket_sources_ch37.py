#!/usr/bin/env python3
"""Collect BioLiP2, KLIFS, and GPCRdb annotations for the official 888 targets."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
BIOLIP = ROOT / "downloads/known_pocket_sources_v1/BioLiP.txt.gz"
OUTDIR = ROOT / "outputs/chembl37_known_pocket_atlas/specialized_sources"
KLIFS = "https://klifs.net/api"
GPCRDB = "https://gpcrdb.org/services"


def request_json(url: str, params: Any = None, timeout: int = 120, retries: int = 4) -> Any:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url, params=params, timeout=timeout,
                headers={"User-Agent": "BioMaster-known-pocket-atlas/1.0"},
            )
            if response.status_code == 404:
                return None
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    assert error is not None
    raise error


def parse_biolip(project_accessions: set[str]) -> pd.DataFrame:
    if not BIOLIP.is_file():
        raise FileNotFoundError(BIOLIP)
    columns = [
        "pdb_id", "receptor_chain", "resolution", "binding_site_code", "ligand_id",
        "ligand_chain", "ligand_serial", "binding_residues_pdb", "binding_residues_chain_index",
        "catalytic_residues_pdb", "catalytic_residues_chain_index", "ec_number", "go_terms",
        "affinity_manual", "affinity_moad", "affinity_pdbbind_cn", "affinity_bindingdb",
        "uniprot_accession", "pubmed_id", "ligand_auth_seq_id", "receptor_observed_sequence",
    ]
    rows: list[list[str]] = []
    with gzip.open(BIOLIP, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 21:
                continue
            accession_field = fields[17].strip()
            accessions = {
                token.strip() for token in accession_field.replace(";", ",").split(",") if token.strip()
            }
            matched = sorted(accessions & project_accessions)
            for accession in matched:
                row = fields[:21]
                row[17] = accession
                rows.append(row)
    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output
    output["source"] = "BIOLIP2"
    output["resolution"] = pd.to_numeric(output["resolution"], errors="coerce")
    output["binding_residue_count"] = output["binding_residues_chain_index"].fillna("").map(
        lambda value: len(str(value).split())
    )
    missing_affinity_tokens = {"", "-", "none", "nan", "na", "n/a"}
    output["has_affinity"] = output[[
        "affinity_manual", "affinity_moad", "affinity_pdbbind_cn", "affinity_bindingdb"
    ]].fillna("").apply(
        lambda row: any(str(value).strip().lower() not in missing_affinity_tokens for value in row),
        axis=1,
    )
    output["biolip_record_id"] = (
        "BIOLIP2:" + output["uniprot_accession"] + ":" + output["pdb_id"].str.lower()
        + ":" + output["receptor_chain"] + ":" + output["ligand_id"]
        + ":" + output["binding_site_code"]
    )
    return output


def collect_klifs(project_accessions: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kinases = pd.DataFrame(request_json(f"{KLIFS}/kinase_information", {"species": "HUMAN"}))
    project = kinases[kinases["uniprot"].isin(project_accessions)].copy()
    structures: list[dict[str, Any]] = []
    ids = project["kinase_ID"].astype(int).tolist()
    for start in range(0, len(ids), 25):
        # KLIFS documents an array here, but the production API accepts a
        # comma-separated value and rejects repeated query parameters.
        params = {"kinase_ID": ",".join(map(str, ids[start:start + 25]))}
        structures.extend(request_json(f"{KLIFS}/structures_list", params) or [])
    structure_frame = pd.DataFrame(structures)
    if not structure_frame.empty:
        structure_frame = structure_frame.merge(
            project[["kinase_ID", "uniprot", "name", "HGNC", "family", "group"]].rename(
                columns={"uniprot": "uniprot_accession", "name": "klifs_kinase_name"}
            ),
            on="kinase_ID", how="left", validate="many_to_one", suffixes=("", "_target"),
        )
        structure_frame["source"] = "KLIFS"
        structure_frame["pdb"] = structure_frame["pdb"].astype(str).str.lower()
        structure_frame["resolution"] = pd.to_numeric(structure_frame["resolution"], errors="coerce")
        structure_frame["quality_score"] = pd.to_numeric(structure_frame["quality_score"], errors="coerce")
        structure_frame["klifs_structure_id"] = "KLIFS:" + structure_frame["structure_ID"].astype(str)
    return project, structure_frame


def fetch_gpcr_proteins(entry_names: list[str], workers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {
            executor.submit(request_json, f"{GPCRDB}/protein/{name}/"): name for name in entry_names
        }
        for future in as_completed(jobs):
            value = future.result()
            if isinstance(value, dict):
                rows.append(value)
    return pd.DataFrame(rows)


def fetch_gpcr_interactions(pdb_ids: list[str], workers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {
            executor.submit(request_json, f"{GPCRDB}/structure/{pdb_id}/interaction/"): pdb_id
            for pdb_id in pdb_ids
        }
        for index, future in enumerate(as_completed(jobs), start=1):
            values = future.result()
            if isinstance(values, list):
                rows.extend(values)
            if index % 100 == 0 or index == len(jobs):
                print(f"GPCRdb interactions {index}/{len(jobs)}", flush=True)
    return pd.DataFrame(rows)


def collect_gpcrdb(
    project_accessions: set[str], workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structures_raw = request_json(f"{GPCRDB}/structure/") or []
    structures = pd.json_normalize(structures_raw, sep="_")
    entry_names = sorted(set(structures["protein"].dropna().astype(str)))
    proteins = fetch_gpcr_proteins(entry_names, workers)
    protein_accession = proteins.set_index("entry_name")["accession"].to_dict()
    structures["uniprot_accession"] = structures["protein"].map(protein_accession)
    structures = structures[structures["uniprot_accession"].isin(project_accessions)].copy()
    structures["pdb_code"] = structures["pdb_code"].astype(str).str.lower()
    structures["source"] = "GPCRDB"

    interactions = fetch_gpcr_interactions(
        sorted(structures["pdb_code"].str.upper().unique()), workers
    )
    pockets: list[dict[str, Any]] = []
    if not interactions.empty:
        interactions["pdb_code"] = interactions["pdb_code"].astype(str).str.lower()
        mapping = structures.drop_duplicates("pdb_code").set_index("pdb_code").to_dict(orient="index")
        for (pdb_id, ligand_name), group in interactions.groupby(["pdb_code", "ligand_name"], dropna=False):
            meta = mapping.get(pdb_id, {})
            residues = group[["sequence_number", "amino_acid", "display_generic_number"]].drop_duplicates()
            positions = sorted(pd.to_numeric(residues["sequence_number"], errors="coerce").dropna().astype(int).unique())
            interaction_types = sorted(set(group["interaction_type"].dropna().astype(str)))
            pockets.append({
                "source": "GPCRDB",
                "gpcrdb_pocket_id": f"GPCRDB:{meta.get('uniprot_accession', '')}:{pdb_id}:{ligand_name}",
                "uniprot_accession": meta.get("uniprot_accession"),
                "pdb_id": pdb_id,
                "preferred_chain": meta.get("preferred_chain"),
                "protein_entry_name": meta.get("protein"),
                "ligand_name": ligand_name,
                "uniprot_residue_positions": ";".join(map(str, positions)),
                "binding_residue_count": len(positions),
                "generic_residue_numbers": ";".join(sorted(set(residues["display_generic_number"].dropna().astype(str)))),
                "interaction_types": ";".join(interaction_types),
                "state": meta.get("state"),
                "resolution": meta.get("resolution"),
                "experimental_method": meta.get("type"),
                "publication_date": meta.get("publication_date"),
                "publication": meta.get("publication"),
            })
    return proteins, structures, pd.DataFrame(pockets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER, low_memory=False)
    project_accessions = set(master["uniprot_accession"].astype(str))

    biolip = parse_biolip(project_accessions)
    biolip.to_csv(OUTDIR / "BIOLIP2_PROJECT_INTERACTION_SITES.csv.gz", index=False, compression="gzip")

    klifs_kinases, klifs_structures = collect_klifs(project_accessions)
    klifs_kinases.to_csv(OUTDIR / "KLIFS_PROJECT_KINASES.csv", index=False)
    klifs_structures.to_csv(OUTDIR / "KLIFS_PROJECT_STRUCTURES.csv.gz", index=False, compression="gzip")

    gpcr_proteins, gpcr_structures, gpcr_pockets = collect_gpcrdb(project_accessions, args.workers)
    gpcr_proteins.to_csv(OUTDIR / "GPCRDB_PROTEIN_MAPPING.csv", index=False)
    gpcr_structures.to_csv(OUTDIR / "GPCRDB_PROJECT_STRUCTURES.csv.gz", index=False, compression="gzip")
    gpcr_pockets.to_csv(OUTDIR / "GPCRDB_PROJECT_INTERACTION_POCKETS.csv.gz", index=False, compression="gzip")

    summary = {
        "biolip2_records": int(len(biolip)),
        "biolip2_targets": int(biolip["uniprot_accession"].nunique()) if len(biolip) else 0,
        "biolip2_pdbs": int(biolip["pdb_id"].nunique()) if len(biolip) else 0,
        "klifs_project_kinase_records": int(len(klifs_kinases)),
        "klifs_project_kinases": int(klifs_kinases["uniprot"].nunique()) if len(klifs_kinases) else 0,
        "klifs_project_structures": int(len(klifs_structures)),
        "klifs_targets_with_structures": int(klifs_structures["uniprot_accession"].nunique()) if len(klifs_structures) else 0,
        "gpcrdb_project_structures": int(len(gpcr_structures)),
        "gpcrdb_targets_with_structures": int(gpcr_structures["uniprot_accession"].nunique()) if len(gpcr_structures) else 0,
        "gpcrdb_interaction_pockets": int(len(gpcr_pockets)),
        "gpcrdb_targets_with_interaction_pockets": int(gpcr_pockets["uniprot_accession"].nunique()) if len(gpcr_pockets) else 0,
    }
    (OUTDIR / "SPECIALIZED_SOURCE_COLLECTION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

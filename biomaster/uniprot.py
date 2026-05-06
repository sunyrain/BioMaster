from __future__ import annotations

import csv
import io
import re
from typing import Iterator

import requests


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
HUMAN_REVIEWED_QUERY = "(proteome:UP000005640) AND (reviewed:true)"
UNIPROT_FIELDS = "accession,id,protein_name,gene_primary,organism_name,length,sequence"


def _next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
    return match.group(1) if match else None


def iter_human_reviewed_proteins(limit: int | None = None, batch_size: int = 500) -> Iterator[dict[str, str]]:
    params = {
        "query": HUMAN_REVIEWED_QUERY,
        "fields": UNIPROT_FIELDS,
        "format": "tsv",
        "size": str(batch_size),
    }
    url: str | None = UNIPROT_SEARCH_URL
    produced = 0
    while url:
        response = requests.get(url, params=params if url == UNIPROT_SEARCH_URL else None, timeout=60)
        response.raise_for_status()
        reader = csv.DictReader(io.StringIO(response.text), delimiter="\t")
        for record in reader:
            yield normalize_search_record(record)
            produced += 1
            if limit is not None and produced >= limit:
                return
        params = None
        url = _next_link(response.headers.get("Link"))


def normalize_search_record(record: dict[str, str]) -> dict[str, str]:
    return {
        "protein_id": record.get("Entry", ""),
        "entry_name": record.get("Entry Name", ""),
        "protein_name": record.get("Protein names", ""),
        "gene_name": record.get("Gene Names (primary)", ""),
        "organism": record.get("Organism", ""),
        "length": record.get("Length", ""),
        "sequence": record.get("Sequence", ""),
    }


def fetch_uniprot_entry(accession: str, timeout: int = 30) -> dict[str, str]:
    response = requests.get(UNIPROT_ENTRY_URL.format(accession=accession), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    protein_name = ""
    description = payload.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    if recommended:
        protein_name = recommended.get("fullName", {}).get("value", "")
    if not protein_name and description.get("submissionNames"):
        protein_name = description["submissionNames"][0].get("fullName", {}).get("value", "")
    genes = payload.get("genes") or []
    gene_name = ""
    if genes:
        gene_name = genes[0].get("geneName", {}).get("value", "")
    organism_payload = payload.get("organism", {})
    organism = organism_payload.get("scientificName", "")
    common = organism_payload.get("commonName", "")
    if common:
        organism = f"{organism} ({common})"
    sequence_payload = payload.get("sequence", {})
    return {
        "protein_id": payload.get("primaryAccession", accession),
        "entry_name": payload.get("uniProtkbId", ""),
        "protein_name": protein_name,
        "gene_name": gene_name,
        "organism": organism,
        "length": str(sequence_payload.get("length", "")),
        "sequence": sequence_payload.get("value", ""),
    }


def fetch_alphafold_prediction(accession: str, timeout: int = 30) -> dict[str, str]:
    response = requests.get(ALPHAFOLD_API_URL.format(accession=accession), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return {}
    item = payload[0]
    return {
        "alphafold_pdb_url": item.get("pdbUrl", ""),
        "alphafold_cif_url": item.get("cifUrl", ""),
        "alphafold_bcif_url": item.get("bcifUrl", ""),
        "alphafold_global_plddt": str(item.get("globalMetricValue", "")),
        "alphafold_model_created_date": item.get("modelCreatedDate", ""),
        "alphafold_latest_version": str(item.get("latestVersion", "")),
        "sequence": item.get("uniprotSequence", ""),
    }


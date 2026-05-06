from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

EXTRA_FIELDS = [
    "structure_source",
    "chembl_pref_name",
    "chembl_first_approval",
    "chembl_max_phase",
    "pubchem_match_method",
    "pubchem_lookup_status",
    "chembl_lookup_status",
]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "UNKNOWN"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def request_json(session: requests.Session, url: str, *, method: str = "GET", data: dict[str, str] | None = None, timeout: int = 45) -> dict[str, Any] | None:
    for attempt in range(3):
        try:
            if method == "POST":
                response = session.post(url, data=data, timeout=timeout)
            else:
                response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1.0 + attempt)
    return None


def request_text(session: requests.Session, url: str, *, method: str = "GET", data: dict[str, str] | None = None, timeout: int = 45) -> tuple[int | None, str]:
    for attempt in range(3):
        try:
            if method == "POST":
                response = session.post(url, data=data, timeout=timeout)
            else:
                response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return response.status_code, ""
            response.raise_for_status()
            return response.status_code, response.text
        except Exception as exc:
            if attempt == 2:
                return None, str(exc)
            time.sleep(1.0 + attempt)
    return None, ""


def fetch_chembl(session: requests.Session, chembl_id: str) -> dict[str, Any] | None:
    if not chembl_id:
        return None
    return request_json(session, CHEMBL_MOLECULE_URL.format(chembl_id=chembl_id))


def lookup_pubchem_cid_by_smiles(session: requests.Session, smiles: str) -> tuple[str, str]:
    if not smiles:
        return "", "missing_smiles"
    url = f"{PUBCHEM_BASE_URL}/compound/smiles/cids/TXT"
    status, text = request_text(session, url, method="POST", data={"smiles": smiles})
    if status != 200 or not text.strip():
        return "", f"smiles_lookup_failed:{status or text[:120]}"
    cid = text.strip().splitlines()[0].strip()
    return cid, "smiles"


def lookup_pubchem_cid_by_name(session: requests.Session, name: str) -> tuple[str, str]:
    if not name:
        return "", "missing_name"
    url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(name)}/cids/TXT"
    status, text = request_text(session, url)
    if status != 200 or not text.strip():
        return "", f"name_lookup_failed:{status or text[:120]}"
    cid = text.strip().splitlines()[0].strip()
    return cid, "name"


def fetch_pubchem_properties(session: requests.Session, cid: str) -> dict[str, Any]:
    if not cid:
        return {}
    properties = "CanonicalSMILES,IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight"
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/{properties}/JSON"
    payload = request_json(session, url)
    if not payload:
        return {}
    rows = ((payload.get("PropertyTable") or {}).get("Properties") or [])
    return rows[0] if rows else {}


def write_sdf_from_chembl(row: dict[str, str], chembl_payload: dict[str, Any] | None, sdf_dir: Path) -> tuple[str, str]:
    structures = (chembl_payload or {}).get("molecule_structures") or {}
    molfile = structures.get("molfile") or ""
    if not molfile.strip():
        return "", ""
    drug_id = row.get("drug_id") or row.get("chembl_id") or row.get("drug_name") or "UNKNOWN"
    target = sdf_dir / f"{slugify(str(drug_id))}.sdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = molfile.rstrip() + "\n$$$$\n"
    target.write_text(text, encoding="utf-8")
    return str(target), "ChEMBL API molfile"


def enrich_row(session: requests.Session, row: dict[str, str], sdf_dir: Path, pubchem_name_fallback: bool) -> dict[str, Any]:
    enriched: dict[str, Any] = dict(row)
    chembl_id = (row.get("chembl_id") or "").strip()
    chembl_payload = fetch_chembl(session, chembl_id)
    chembl_status = "ok" if chembl_payload else "not_found_or_failed"
    structures = (chembl_payload or {}).get("molecule_structures") or {}
    props = (chembl_payload or {}).get("molecule_properties") or {}

    if structures.get("canonical_smiles"):
        enriched["canonical_smiles"] = structures["canonical_smiles"]
    if structures.get("standard_inchi_key") and not enriched.get("inchikey"):
        enriched["inchikey"] = structures["standard_inchi_key"]
    if props.get("full_molformula") and not enriched.get("molecular_formula"):
        enriched["molecular_formula"] = props["full_molformula"]
    if props.get("full_mwt") and not enriched.get("molecular_weight"):
        enriched["molecular_weight"] = props["full_mwt"]

    sdf_path, structure_source = write_sdf_from_chembl(row, chembl_payload, sdf_dir)
    if sdf_path and not enriched.get("sdf_path"):
        enriched["sdf_path"] = sdf_path

    cid = (row.get("pubchem_cid") or "").strip()
    pubchem_match_method = "existing" if cid else ""
    pubchem_status = "existing" if cid else ""
    if not cid:
        lookup_smiles = enriched.get("canonical_smiles") or row.get("canonical_smiles") or row.get("isomeric_smiles") or ""
        cid, pubchem_match_method = lookup_pubchem_cid_by_smiles(session, str(lookup_smiles))
        pubchem_status = "ok" if cid else pubchem_match_method
    if not cid and pubchem_name_fallback:
        cid, pubchem_match_method = lookup_pubchem_cid_by_name(session, row.get("drug_name") or "")
        pubchem_status = "ok" if cid else pubchem_match_method
    if cid:
        enriched["pubchem_cid"] = cid
        pubchem_props = fetch_pubchem_properties(session, cid)
        if pubchem_props.get("InChIKey"):
            enriched["inchikey"] = pubchem_props["InChIKey"]
        if pubchem_props.get("MolecularFormula"):
            enriched["molecular_formula"] = pubchem_props["MolecularFormula"]
        if pubchem_props.get("MolecularWeight"):
            enriched["molecular_weight"] = str(pubchem_props["MolecularWeight"])
        if pubchem_props.get("CanonicalSMILES"):
            enriched["canonical_smiles"] = pubchem_props["CanonicalSMILES"]
        if pubchem_props.get("IsomericSMILES"):
            enriched["isomeric_smiles"] = pubchem_props["IsomericSMILES"]

    existing_source = enriched.get("source") or ""
    sources = [item for item in [existing_source, "ChEMBL API" if chembl_payload else "", "PubChem PUG REST" if cid else ""] if item]
    enriched["source"] = ";".join(dict.fromkeys(sources))
    enriched["structure_source"] = structure_source
    enriched["chembl_pref_name"] = (chembl_payload or {}).get("pref_name") or ""
    enriched["chembl_first_approval"] = (chembl_payload or {}).get("first_approval") or ""
    enriched["chembl_max_phase"] = (chembl_payload or {}).get("max_phase") or ""
    enriched["pubchem_match_method"] = pubchem_match_method
    enriched["pubchem_lookup_status"] = pubchem_status
    enriched["chembl_lookup_status"] = chembl_status
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich BioMaster drug library with ChEMBL and PubChem structure IDs.")
    parser.add_argument("--drugs", default="outputs/report_scale/drug_library.csv")
    parser.add_argument("--out", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--metadata-out", default="data/processed/drug_library_pubchem_chembl_mapped.metadata.json")
    parser.add_argument("--sdf-dir", default="data/processed/ligands_sdf_chembl")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--pubchem-name-fallback", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    fields, rows = read_rows(Path(args.drugs))
    if args.limit is not None:
        rows = rows[: args.limit]
    out_fields = fields + [field for field in EXTRA_FIELDS if field not in fields]
    session = requests.Session()
    session.headers.update({"User-Agent": "BioMaster/0.1 (research data integration)"})

    enriched_rows: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for index, row in enumerate(rows, start=1):
        enriched_rows.append(enrich_row(session, row, Path(args.sdf_dir), args.pubchem_name_fallback))
        if args.sleep:
            time.sleep(args.sleep)
        if index % 100 == 0:
            print(f"processed {index}/{len(rows)}", flush=True)

    write_rows(Path(args.out), out_fields, enriched_rows)
    def nonempty(field: str) -> int:
        return sum(1 for row in enriched_rows if str(row.get(field) or "").strip())

    metadata = {
        "created_utc": started,
        "input_csv": args.drugs,
        "output_csv": args.out,
        "sdf_dir": args.sdf_dir,
        "rows": len(enriched_rows),
        "nonempty_pubchem_cid": nonempty("pubchem_cid"),
        "nonempty_inchikey": nonempty("inchikey"),
        "nonempty_molecular_formula": nonempty("molecular_formula"),
        "nonempty_sdf_path": nonempty("sdf_path"),
        "nonempty_chembl_pref_name": nonempty("chembl_pref_name"),
        "chembl_ok": sum(1 for row in enriched_rows if row.get("chembl_lookup_status") == "ok"),
        "pubchem_ok": sum(1 for row in enriched_rows if row.get("pubchem_lookup_status") == "ok"),
        "source_urls": {
            "chembl_api": "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json",
            "pubchem_pug_rest": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
        },
        "notes": "Structures are filled from ChEMBL molecule molfile when available. PubChem CIDs are matched primarily by structure SMILES, not by FDA product name; FDA approval validation remains a separate task.",
    }
    write_rows_path = Path(args.metadata_out)
    write_rows_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import requests


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PROPERTIES = "SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight"


def fetch_compound_properties(identifier: str, namespace: str = "name", timeout: int = 30) -> dict[str, str]:
    encoded = quote(str(identifier), safe="")
    url = f"{PUBCHEM_BASE}/compound/{namespace}/{encoded}/property/{PUBCHEM_PROPERTIES}/JSON"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    properties = payload["PropertyTable"]["Properties"][0]
    return {
        "pubchem_cid": str(properties.get("CID", "")),
        "isomeric_smiles": str(properties.get("SMILES", "")),
        "canonical_smiles": str(properties.get("ConnectivitySMILES", properties.get("SMILES", ""))),
        "inchikey": str(properties.get("InChIKey", "")),
        "molecular_formula": str(properties.get("MolecularFormula", "")),
        "molecular_weight": str(properties.get("MolecularWeight", "")),
    }


def download_sdf(cid: str, output_path: str | Path, timeout: int = 60) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PUBCHEM_BASE}/compound/cid/{quote(str(cid), safe='')}/SDF"
    response = requests.get(url, params={"record_type": "3d"}, timeout=timeout)
    if response.status_code == 404:
        response = requests.get(url, params={"record_type": "2d"}, timeout=timeout)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


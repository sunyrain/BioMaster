#!/usr/bin/env python3
"""Create the model ligand library used consistently by ConPLEx and Boltz."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_106k_to_1000_physics_funnel import standardize_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--output", default="data/processed/drug_library_active_moiety_v4.csv")
    parser.add_argument("--summary", default="data/processed/drug_library_active_moiety_v4.summary.json")
    args = parser.parse_args()
    data = pd.read_csv(args.input, low_memory=False).fillna("")
    standardized = data["canonical_smiles"].map(standardize_smiles)
    data["canonical_smiles_original"] = data["canonical_smiles"]
    data["canonical_smiles_rdkit"] = [row["canonical_smiles_rdkit"] for row in standardized]
    data["active_moiety_smiles"] = [row["active_moiety_smiles"] for row in standardized]
    data["murcko_scaffold"] = [row["murcko_scaffold"] for row in standardized]
    data["model_ligand_smiles"] = data["active_moiety_smiles"]
    data["model_ligand_standardization"] = "largest_RDKit_fragment_then_Uncharger"
    if data["model_ligand_smiles"].astype(str).eq("").any():
        raise ValueError("Some FDA rows do not have a usable model_ligand_smiles")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    summary = {
        "rows": int(len(data)),
        "unique_drug_ids": int(data["drug_id"].nunique()),
        "unique_model_ligands": int(data["model_ligand_smiles"].nunique()),
        "rows_changed_from_canonical": int(
            (data["model_ligand_smiles"] != data["canonical_smiles_original"]).sum()
        ),
        "method": "RDKit canonicalization, largest fragment selection, and Uncharger neutralization.",
        "warning": "This is a structure representation standard, not pharmacokinetic active-metabolite normalization.",
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

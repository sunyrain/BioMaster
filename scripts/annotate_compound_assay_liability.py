#!/usr/bin/env python3
"""Add reproducible compound-level assay-liability annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def catalog(kind: FilterCatalogParams.FilterCatalogs) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(kind)
    return FilterCatalog(params)


PAINS = catalog(FilterCatalogParams.FilterCatalogs.PAINS)
BRENK = catalog(FilterCatalogParams.FilterCatalogs.BRENK)
NIH = catalog(FilterCatalogParams.FilterCatalogs.NIH)
METALS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 55, 56, 57,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
}


def descriptions(filter_catalog: FilterCatalog, molecule: Chem.Mol) -> str:
    return ";".join(sorted({entry.GetDescription() for entry in filter_catalog.GetMatches(molecule)}))


def molecule_annotations(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return {
            "liability_rdkit_parse_ok": False,
            "liability_pains_alerts": "",
            "liability_brenk_alerts": "",
            "liability_nih_alerts": "",
            "liability_formal_charge": None,
            "liability_molecular_weight": None,
            "liability_heavy_atoms": None,
            "liability_metal_atom_count": None,
            "assay_interference_review": True,
            "brenk_developability_review": True,
            "severe_compound_liability": True,
            "compound_liability_notes": "rdkit_parse_failure",
        }
    pains = descriptions(PAINS, molecule)
    brenk = descriptions(BRENK, molecule)
    nih = descriptions(NIH, molecule)
    charge = int(Chem.GetFormalCharge(molecule))
    mw = float(Descriptors.MolWt(molecule))
    heavy = int(molecule.GetNumHeavyAtoms())
    metal_count = sum(atom.GetAtomicNum() in METALS for atom in molecule.GetAtoms())
    severe_notes: list[str] = []
    if abs(charge) >= 3:
        severe_notes.append("absolute_formal_charge_ge_3")
    if mw > 1000:
        severe_notes.append("molecular_weight_gt_1000")
    if heavy < 6:
        severe_notes.append("too_few_heavy_atoms")
    if metal_count:
        severe_notes.append("metal_containing_structure")
    # Brenk is a broad medicinal-chemistry/developability catalog.  It is
    # retained for review but is not, by itself, treated as an assay-
    # interference signal for an already approved drug.
    review = bool(pains or nih or abs(charge) >= 2)
    notes = list(severe_notes)
    if pains:
        notes.append("PAINS_alert")
    if brenk:
        notes.append("Brenk_alert")
    if nih:
        notes.append("NIH_alert")
    if abs(charge) == 2:
        notes.append("absolute_formal_charge_2")
    return {
        "liability_rdkit_parse_ok": True,
        "liability_pains_alerts": pains,
        "liability_brenk_alerts": brenk,
        "liability_nih_alerts": nih,
        "liability_formal_charge": charge,
        "liability_molecular_weight": mw,
        "liability_heavy_atoms": heavy,
        "liability_metal_atom_count": metal_count,
        "assay_interference_review": review,
        "brenk_developability_review": bool(brenk),
        "severe_compound_liability": bool(severe_notes),
        "compound_liability_notes": ";".join(notes),
    }


def annotate(data: pd.DataFrame) -> pd.DataFrame:
    smiles = data.get("active_moiety_smiles", pd.Series("", index=data.index)).map(clean)
    fallback = data.get("canonical_smiles_rdkit", data.get("canonical_smiles", pd.Series("", index=data.index))).map(clean)
    smiles = smiles.where(smiles.ne(""), fallback)
    unique = {value: molecule_annotations(value) for value in sorted(set(smiles))}
    annotations = pd.DataFrame([unique[value] for value in smiles], index=data.index)
    out = data.copy()
    out["liability_input_smiles"] = smiles
    for column in annotations.columns:
        out[column] = annotations[column]
    names = out.get("drug_names", out.get("fda_generic_name", pd.Series("", index=out.index))).astype(str)
    out["multi_product_label_review"] = names.str.contains(";", regex=False, na=False)
    out["composite_drug_id_review"] = out.get(
        "drug_chembl_id", pd.Series("", index=out.index)
    ).astype(str).str.contains("__", regex=False, na=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    data = pd.read_csv(args.input, low_memory=False).fillna("")
    result = annotate(data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "rows": int(len(result)),
        "unique_compounds": int(result["liability_input_smiles"].nunique()),
        "assay_interference_review_rows": int(result["assay_interference_review"].sum()),
        "severe_compound_liability_rows": int(result["severe_compound_liability"].sum()),
        "pains_rows": int(result["liability_pains_alerts"].astype(str).ne("").sum()),
        "brenk_rows": int(result["liability_brenk_alerts"].astype(str).ne("").sum()),
        "brenk_developability_review_rows": int(result["brenk_developability_review"].sum()),
        "nih_rows": int(result["liability_nih_alerts"].astype(str).ne("").sum()),
        "multi_product_label_review_rows": int(result["multi_product_label_review"].sum()),
        "warning": "Structural alerts are assay-risk annotations, not proof that an approved drug is inactive.",
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

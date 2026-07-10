#!/usr/bin/env python3
"""Audit two-sample Boltz ligand-pose stability after protein alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from Bio.PDB import MMCIFParser, Superimposer


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def model1_path(model0: Path) -> Path:
    return Path(str(model0).replace("_model_0.cif", "_model_1.cif"))


def parse(path: Path):
    return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))[0]


def protein_ca(model) -> dict[tuple[str, int, str], Any]:
    atoms: dict[tuple[str, int, str], Any] = {}
    for chain in model:
        if chain.id != "A":
            continue
        for residue in chain:
            if "CA" in residue:
                atoms[(chain.id, int(residue.id[1]), residue.resname)] = residue["CA"]
    return atoms


def pocket_residue_ids_from_yaml(path: Path | None) -> set[int]:
    if path is None or not path.is_file():
        return set()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    residues: set[int] = set()
    for constraint in payload.get("constraints") or []:
        pocket = constraint.get("pocket") if isinstance(constraint, dict) else None
        for contact in (pocket or {}).get("contacts") or []:
            if isinstance(contact, (list, tuple)) and len(contact) >= 2 and str(contact[0]) == "A":
                residues.add(int(contact[1]))
    return residues


def ligand_atoms(model) -> list[Any]:
    atoms = []
    for chain in model:
        if chain.id == "A":
            continue
        atoms.extend(atom for atom in chain.get_atoms() if atom.element != "H")
    return atoms


def interface_residues(model, ligand: list[Any], cutoff: float = 5.0) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    ligand_coords = np.asarray([atom.coord for atom in ligand], dtype=float)
    if ligand_coords.size == 0:
        return result
    for chain in model:
        if chain.id != "A":
            continue
        for residue in chain:
            coords = np.asarray([atom.coord for atom in residue.get_atoms() if atom.element != "H"], dtype=float)
            if coords.size and np.min(np.linalg.norm(coords[:, None, :] - ligand_coords[None, :, :], axis=2)) <= cutoff:
                result.add((int(residue.id[1]), residue.resname))
    return result


def symmetry_corrected_ligand_rmsd(
    ligand0: list[Any],
    ligand1: list[Any],
    ligand_smiles: str,
) -> tuple[float, float, str, int]:
    coords0 = np.asarray([atom.coord for atom in ligand0], dtype=float)
    coords1 = np.asarray([atom.coord for atom in ligand1], dtype=float)
    raw = float(math.sqrt(np.mean(np.sum((coords0 - coords1) ** 2, axis=1))))
    if not ligand_smiles:
        return raw, raw, "atom_order_no_smiles", 1
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(ligand_smiles)
        if molecule is None or molecule.GetNumHeavyAtoms() != len(ligand0):
            return raw, raw, "atom_order_smiles_count_mismatch", 1
        expected_elements = [atom.GetSymbol().upper() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
        observed_elements = [str(atom.element).upper() for atom in ligand0]
        if expected_elements != observed_elements:
            return raw, raw, "atom_order_smiles_element_order_mismatch", 1
        permutations = molecule.GetSubstructMatches(
            molecule,
            uniquify=False,
            useChirality=True,
            maxMatches=10000,
        )
        valid = [
            match
            for match in permutations
            if len(match) == len(ligand1)
            and [str(ligand1[index].element).upper() for index in match] == observed_elements
        ]
        if not valid:
            return raw, raw, "atom_order_no_valid_automorphism", 1
        corrected = min(
            float(math.sqrt(np.mean(np.sum((coords0 - coords1[np.asarray(match)]) ** 2, axis=1))))
            for match in valid
        )
        return corrected, raw, "rdkit_graph_automorphism", len(valid)
    except Exception:  # noqa: BLE001
        return raw, raw, "atom_order_symmetry_error", 1


def audit_pair(
    path0: Path,
    pocket_residue_ids: set[int] | None = None,
    ligand_smiles: str = "",
) -> dict[str, Any]:
    path1 = model1_path(path0)
    if not path0.exists() or not path1.exists():
        return {"pose_stability_completed": False, "pose_stability_reason": "missing_model_0_or_model_1"}
    try:
        first = parse(path0)
        second = parse(path1)
        ca0 = protein_ca(first)
        ca1 = protein_ca(second)
        common = sorted(set(ca0) & set(ca1))
        if len(common) < 20:
            return {"pose_stability_completed": False, "pose_stability_reason": "fewer_than_20_common_CA"}
        global_superimposer = Superimposer()
        global_superimposer.set_atoms([ca0[key] for key in common], [ca1[key] for key in common])
        input_pocket_common = [key for key in common if key[1] in (pocket_residue_ids or set())]
        alignment_keys = input_pocket_common if len(input_pocket_common) >= 3 else common
        alignment_mode = "input_pocket_ca" if len(input_pocket_common) >= 3 else "whole_protein_ca_fallback"
        superimposer = Superimposer()
        superimposer.set_atoms([ca0[key] for key in alignment_keys], [ca1[key] for key in alignment_keys])
        superimposer.apply(list(second.get_atoms()))
        ligand0 = ligand_atoms(first)
        ligand1 = ligand_atoms(second)
        if not ligand0 or len(ligand0) != len(ligand1):
            return {"pose_stability_completed": False, "pose_stability_reason": "ligand_atom_count_mismatch"}
        coords0 = np.asarray([atom.coord for atom in ligand0], dtype=float)
        coords1 = np.asarray([atom.coord for atom in ligand1], dtype=float)
        rmsd, raw_rmsd, rmsd_method, automorphism_count = symmetry_corrected_ligand_rmsd(
            ligand0, ligand1, ligand_smiles
        )
        centroid = float(np.linalg.norm(coords0.mean(axis=0) - coords1.mean(axis=0)))
        interface0 = interface_residues(first, ligand0)
        interface1 = interface_residues(second, ligand1)
        union = interface0 | interface1
        jaccard = float(len(interface0 & interface1) / len(union)) if union else 0.0
        if rmsd <= 3.0 and centroid <= 3.0 and jaccard >= 0.40:
            stability = "A_stable_conditional_pose"
        elif rmsd <= 6.0 and centroid <= 6.0 and jaccard >= 0.20:
            stability = "B_moderate_conditional_pose"
        else:
            stability = "C_unstable_conditional_pose"
        return {
            "pose_stability_completed": True,
            "pose_stability_reason": "",
            "pose_protein_alignment_rmsd": float(superimposer.rms),
            "pose_protein_global_alignment_rmsd": float(global_superimposer.rms),
            "pose_alignment_mode": alignment_mode,
            "pose_input_pocket_ca_count": int(len(input_pocket_common)),
            "pose_common_ca_count": int(len(common)),
            "pose_ligand_heavy_atom_count": int(len(ligand0)),
            "pose_ligand_rmsd": rmsd,
            "pose_ligand_raw_atom_order_rmsd": raw_rmsd,
            "pose_ligand_rmsd_method": rmsd_method,
            "pose_ligand_automorphism_count": int(automorphism_count),
            "pose_ligand_centroid_distance": centroid,
            "pose_interface_residue_jaccard": jaccard,
            "pose_interface_residue_count_model0": int(len(interface0)),
            "pose_interface_residue_count_model1": int(len(interface1)),
            "pose_stability_tier": stability,
        }
    except Exception as exc:  # noqa: BLE001
        return {"pose_stability_completed": False, "pose_stability_reason": f"parse_or_alignment_error:{exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path-column", default="boltz_cif_path_refined")
    parser.add_argument("--yaml-path-column", default="")
    parser.add_argument("--smiles-column", default="model_ligand_smiles")
    parser.add_argument("--summary")
    args = parser.parse_args()
    data = pd.read_csv(args.input, low_memory=False).fillna("")
    if args.path_column not in data.columns:
        raise ValueError(f"Missing path column: {args.path_column}")
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    yaml_values = (
        data[args.yaml_path_column].map(clean)
        if args.yaml_path_column and args.yaml_path_column in data.columns
        else pd.Series("", index=data.index)
    )
    smiles_values = (
        data[args.smiles_column].map(clean)
        if args.smiles_column and args.smiles_column in data.columns
        else pd.Series("", index=data.index)
    )
    for value, yaml_value, ligand_smiles in zip(
        data[args.path_column].map(clean), yaml_values, smiles_values, strict=True
    ):
        if not value:
            rows.append({"pose_stability_completed": False, "pose_stability_reason": "missing_model_0_path"})
            continue
        cache_key = f"{value}__{yaml_value}__{ligand_smiles}"
        if cache_key not in cache:
            residues = pocket_residue_ids_from_yaml(Path(yaml_value)) if yaml_value else set()
            cache[cache_key] = audit_pair(Path(value), residues, ligand_smiles=ligand_smiles)
        rows.append(cache[cache_key])
    annotations = pd.DataFrame(rows, index=data.index)
    result = data.copy()
    for column in annotations.columns:
        result[column] = annotations[column]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    tiers = result.get("pose_stability_tier", pd.Series("", index=result.index)).value_counts().to_dict()
    summary = {
        "rows": int(len(result)),
        "completed": int(result["pose_stability_completed"].astype(bool).sum()),
        "tiers": tiers,
        "median_ligand_rmsd": float(pd.to_numeric(result.get("pose_ligand_rmsd"), errors="coerce").median()),
        "median_interface_jaccard": float(
            pd.to_numeric(result.get("pose_interface_residue_jaccard"), errors="coerce").median()
        ),
        "warning": "Pocket contacts were supplied to Boltz; stability is conditional pose reproducibility, not blind pocket recovery.",
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

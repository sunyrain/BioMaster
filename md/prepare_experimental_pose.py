#!/usr/bin/env python3
"""Map a Boltz ligand pose onto an experimental receptor structure.

The output is a cleaned experimental receptor PDB and an RDKit SDF whose
bonding comes from the audited input SMILES while coordinates come from the
aligned Boltz prediction. It is a preparation aid, not an automatic receptor
protonation or biological-assembly decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBIO, PDBParser, Select
from openmm import unit
from openmm.app import PDBxFile
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HID": "H", "HIE": "H", "HIP": "H",
}


class ProteinChainSelect(Select):
    def __init__(self, chain_id: str) -> None:
        self.chain_id = chain_id

    def accept_chain(self, chain) -> bool:  # noqa: ANN001
        return chain.id == self.chain_id

    def accept_residue(self, residue) -> bool:  # noqa: ANN001
        return residue.id[0] == " " and residue.resname in AA3_TO_1


def experimental_chain(path: Path, chain_id: str):  # noqa: ANN201
    structure = PDBParser(QUIET=True).get_structure("experimental", path)
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Experimental chain {chain_id!r} is absent")
    chain = model[chain_id]
    residues = [
        residue
        for residue in chain
        if residue.id[0] == " " and residue.resname in AA3_TO_1 and "CA" in residue
    ]
    if not residues:
        raise ValueError("No standard protein residues with CA atoms found")
    return structure, residues


def assign_histidines_from_experimental_hydrogens(residues) -> dict[str, str]:  # noqa: ANN001
    assignments = {}
    for residue in residues:
        if residue.resname != "HIS":
            continue
        atom_names = {atom.name for atom in residue}
        has_hd1 = "HD1" in atom_names
        has_he2 = "HE2" in atom_names
        if has_hd1 and has_he2:
            amber_name = "HIP"
        elif has_hd1:
            amber_name = "HID"
        elif has_he2:
            amber_name = "HIE"
        else:
            assignments[str(residue.id[1])] = "UNRESOLVED_NO_EXPERIMENTAL_HYDROGEN"
            continue
        assignments[str(residue.id[1])] = amber_name
        residue.resname = amber_name
    return assignments


def apply_histidine_overrides(residues, overrides: list[str]) -> dict[str, str]:  # noqa: ANN001
    parsed = {}
    by_id = {str(residue.id[1]): residue for residue in residues}
    for item in overrides:
        residue_id, separator, amber_name = item.partition(":")
        amber_name = amber_name.upper()
        if not separator or amber_name not in {"HID", "HIE", "HIP"}:
            raise ValueError(f"Invalid histidine override {item!r}; expected RESID:HID|HIE|HIP")
        if residue_id not in by_id or by_id[residue_id].resname not in {"HIS", "HID", "HIE", "HIP"}:
            raise ValueError(f"Histidine override residue {residue_id!r} is absent or not histidine")
        by_id[residue_id].resname = amber_name
        parsed[residue_id] = amber_name
    return parsed


def boltz_entities(path: Path):  # noqa: ANN201
    cif = PDBxFile(str(path))
    positions = np.asarray(
        cif.positions.value_in_unit(unit.angstrom), dtype=float
    )
    protein_residues = []
    ligand_atoms = []
    for residue in cif.topology.residues():
        atoms = list(residue.atoms())
        if residue.name in AA3_TO_1 and any(atom.name == "CA" for atom in atoms):
            protein_residues.append(residue)
        elif residue.name.startswith("LIG"):
            ligand_atoms.extend(atoms)
    if not protein_residues or not ligand_atoms:
        raise ValueError("Boltz CIF must contain a protein and a LIG residue")
    return cif, positions, protein_residues, ligand_atoms


def aligned_ca_pairs(exp_residues, boltz_residues):  # noqa: ANN001, ANN201
    exp_seq = "".join(AA3_TO_1[residue.resname] for residue in exp_residues)
    boltz_seq = "".join(AA3_TO_1[residue.name] for residue in boltz_residues)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(exp_seq, boltz_seq)[0]
    pairs = []
    for (exp_start, exp_end), (boltz_start, boltz_end) in zip(
        alignment.aligned[0], alignment.aligned[1], strict=True
    ):
        length = min(exp_end - exp_start, boltz_end - boltz_start)
        pairs.extend((exp_start + offset, boltz_start + offset) for offset in range(length))
    return exp_seq, boltz_seq, alignment.score, pairs


def kabsch(moving: np.ndarray, fixed: np.ndarray):  # noqa: ANN201
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    covariance = (moving - moving_center).T @ (fixed - fixed_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = fixed_center - moving_center @ rotation
    fitted = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - fixed) ** 2, axis=1))))
    return rotation, translation, rmsd


def ligand_with_audited_graph(
    elements: list[str], coordinates: np.ndarray, smiles: str
) -> tuple[Chem.Mol, str]:
    audited = Chem.MolFromSmiles(smiles)
    if audited is None:
        raise ValueError("Ligand SMILES could not be parsed")
    audited_elements = [atom.GetSymbol() for atom in audited.GetAtoms()]
    if audited_elements == elements:
        audited_conformer = Chem.Conformer(audited.GetNumAtoms())
        for index, xyz in enumerate(coordinates):
            audited_conformer.SetAtomPosition(index, xyz.tolist())
        audited.AddConformer(audited_conformer)
        inferred_smiles = "atom_order_verified_against_audited_smiles"
        audited = Chem.AddHs(audited, addCoords=True)
        if AllChem.UFFHasAllMoleculeParams(audited):
            force_field = AllChem.UFFGetMoleculeForceField(audited)
            for index in range(len(elements)):
                force_field.AddFixedPoint(index)
            force_field.Initialize()
            force_field.Minimize(maxIts=500)
        return audited, inferred_smiles

    editable = Chem.RWMol()
    for element in elements:
        editable.AddAtom(Chem.Atom(element))
    inferred = editable.GetMol()
    conformer = Chem.Conformer(len(elements))
    for index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(index, xyz.tolist())
    inferred.AddConformer(conformer)
    rdDetermineBonds.DetermineBonds(inferred, charge=0)
    Chem.SanitizeMol(inferred)

    match = inferred.GetSubstructMatch(audited, useChirality=False)
    if len(match) != audited.GetNumAtoms():
        reverse = audited.GetSubstructMatch(inferred, useChirality=False)
        if len(reverse) != inferred.GetNumAtoms():
            raise ValueError(
                "Coordinate-inferred ligand graph does not match the audited SMILES: "
                f"inferred={Chem.MolToSmiles(inferred)} audited={Chem.MolToSmiles(audited)}"
            )
        match = tuple(reverse.index(index) for index in range(audited.GetNumAtoms()))

    audited_conformer = Chem.Conformer(audited.GetNumAtoms())
    inferred_conformer = inferred.GetConformer()
    for audited_index, inferred_index in enumerate(match):
        point = inferred_conformer.GetAtomPosition(inferred_index)
        audited_conformer.SetAtomPosition(audited_index, point)
    audited.RemoveAllConformers()
    audited.AddConformer(audited_conformer)
    audited = Chem.AddHs(audited, addCoords=True)
    if AllChem.UFFHasAllMoleculeParams(audited):
        force_field = AllChem.UFFGetMoleculeForceField(audited)
        for index in range(Chem.RemoveHs(audited).GetNumAtoms()):
            force_field.AddFixedPoint(index)
        force_field.Initialize()
        force_field.Minimize(maxIts=500)
    return audited, Chem.MolToSmiles(inferred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental-pdb", required=True, type=Path)
    parser.add_argument("--experimental-chain", default="A")
    parser.add_argument("--boltz-cif", required=True, type=Path)
    parser.add_argument("--ligand-smiles", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument(
        "--alignment-residue-ids",
        default="",
        help=(
            "Optional 1-based Boltz protein-sequence residue IDs used for local pocket "
            "alignment, e.g. '121,122,179'"
        ),
    )
    parser.add_argument(
        "--histidine-override",
        action="append",
        default=[],
        help="Repeatable experimental residue override, for example 279:HID",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    structure, exp_residues = experimental_chain(args.experimental_pdb, args.experimental_chain)
    histidine_assignments = assign_histidines_from_experimental_hydrogens(exp_residues)
    histidine_overrides = apply_histidine_overrides(exp_residues, args.histidine_override)
    _, boltz_positions, boltz_residues, ligand_atoms = boltz_entities(args.boltz_cif)
    exp_seq, boltz_seq, alignment_score, pairs = aligned_ca_pairs(exp_residues, boltz_residues)

    requested_alignment_ids = {
        int(value) for value in re.findall(r"\d+", args.alignment_residue_ids)
    }
    fit_pairs = pairs
    alignment_mode = "whole_protein_ca"
    if requested_alignment_ids:
        fit_pairs = [
            pair
            for pair in pairs
            if int(boltz_residues[pair[1]].id) in requested_alignment_ids
        ]
        if len(fit_pairs) < 3:
            raise ValueError(
                f"Only {len(fit_pairs)} requested pocket residues aligned; at least 3 are required"
            )
        alignment_mode = "requested_pocket_ca"

    exp_ca = np.asarray([exp_residues[i]["CA"].coord for i, _ in fit_pairs], dtype=float)
    boltz_ca = []
    for _, index in fit_pairs:
        ca = next(atom for atom in boltz_residues[index].atoms() if atom.name == "CA")
        boltz_ca.append(boltz_positions[ca.index])
    boltz_ca = np.asarray(boltz_ca, dtype=float)
    rotation, translation, alignment_rmsd = kabsch(boltz_ca, exp_ca)

    ligand_coordinates = np.asarray(
        [boltz_positions[atom.index] for atom in ligand_atoms], dtype=float
    )
    ligand_coordinates = ligand_coordinates @ rotation + translation
    receptor_heavy_coordinates = np.asarray(
        [
            atom.coord
            for residue in exp_residues
            for atom in residue
            if getattr(atom, "element", "") != "H"
        ],
        dtype=float,
    )
    receptor_ligand_distances = np.linalg.norm(
        ligand_coordinates[:, None, :] - receptor_heavy_coordinates[None, :, :], axis=2
    )
    minimum_receptor_ligand_distance = float(receptor_ligand_distances.min())
    severe_clash_count = int(np.sum(receptor_ligand_distances < 1.5))
    elements = [atom.element.symbol for atom in ligand_atoms]
    ligand, inferred_smiles = ligand_with_audited_graph(
        elements, ligand_coordinates, args.ligand_smiles
    )
    ligand.SetProp("_Name", args.pair_id)
    ligand.SetProp("source_pose", str(args.boltz_cif))
    ligand.SetProp("experimental_receptor", str(args.experimental_pdb))

    receptor_path = args.output_dir / "receptor_experimental_clean.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(receptor_path), ProteinChainSelect(args.experimental_chain))
    ligand_path = args.output_dir / "ligand_aligned.sdf"
    writer = Chem.SDWriter(str(ligand_path))
    writer.write(ligand)
    writer.close()

    audit = {
        "pair_id": args.pair_id,
        "experimental_pdb": str(args.experimental_pdb.resolve()),
        "experimental_chain": args.experimental_chain,
        "boltz_cif": str(args.boltz_cif.resolve()),
        "experimental_sequence_length": len(exp_seq),
        "boltz_sequence_length": len(boltz_seq),
        "aligned_ca_count": len(pairs),
        "fitted_ca_count": len(fit_pairs),
        "alignment_mode": alignment_mode,
        "requested_alignment_residue_ids": sorted(requested_alignment_ids),
        "alignment_score": alignment_score,
        "alignment_ca_rmsd_angstrom": alignment_rmsd,
        "ligand_heavy_atoms": len(elements),
        "minimum_receptor_ligand_heavy_distance_angstrom": minimum_receptor_ligand_distance,
        "severe_receptor_ligand_clash_count_lt_1p5a": severe_clash_count,
        "histidine_assignments_from_experimental_hydrogens": histidine_assignments,
        "histidine_overrides": histidine_overrides,
        "audited_smiles": Chem.MolToSmiles(Chem.RemoveHs(ligand), isomericSmiles=True),
        "coordinate_inferred_smiles": inferred_smiles,
        "interpretation": (
            "The ligand coordinates are a Boltz conditional pose mapped onto an experimental "
            "receptor. This does not validate binding or receptor protonation."
        ),
    }
    (args.output_dir / "pose_mapping_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare deterministic project ligand and P2Rank-pocket LMDBs for DrugCLIP."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "affinity_first_remote_discovery_v1"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def write_lmdb(records: list[dict[str, Any]], path: Path) -> None:
    path.unlink(missing_ok=True)
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=max(1024**3, len(records) * 8 * 1024**2),
    )
    with env.begin(write=True) as transaction:
        for index, record in enumerate(records):
            transaction.put(str(index).encode("ascii"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
    env.sync()
    env.close()


def ligand_record(smiles: str, identifier: str, seed: int) -> tuple[dict[str, Any] | None, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, "rdkit_parse_failure"
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    params.maxIterations = 1000
    status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(molecule, params)
    if status != 0:
        return None, "conformer_generation_failure"
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
    except Exception:
        pass
    molecule = Chem.RemoveHs(molecule)
    conformer = molecule.GetConformer()
    record = {
        "coordinates": [np.asarray(conformer.GetPositions(), dtype=np.float32)],
        "atoms": [atom.GetSymbol() for atom in molecule.GetAtoms()],
        "smi": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "name": identifier,
        "IDs": identifier,
        "subset": "FDA_active_moiety_v4",
    }
    return record, "ok"


def parse_residue_ids(value: Any) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for token in clean(value).split():
        match = re.fullmatch(r"([^_]+)_(-?\d+)", token)
        if match:
            result.add((match.group(1), int(match.group(2))))
    return result


def pocket_record(row: pd.Series) -> tuple[dict[str, Any] | None, str, int, int]:
    pdb_path = Path(clean(row.get("pdb_path")))
    residue_ids = parse_residue_ids(row.get("top_pocket_residue_ids"))
    if not pdb_path.exists():
        return None, "missing_receptor_pdb", 0, 0
    if not residue_ids:
        return None, "missing_pocket_residue_ids", 0, 0
    try:
        structure = PDBParser(QUIET=True).get_structure(clean(row["sequence_key"]), str(pdb_path))
    except Exception as error:
        return None, f"pdb_parse_failure:{type(error).__name__}", 0, 0
    atoms: list[str] = []
    coordinates: list[np.ndarray] = []
    matched_residues = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                key = (clean(chain.id), int(residue.id[1]))
                if key not in residue_ids:
                    continue
                matched_residues += 1
                for atom in residue.get_atoms():
                    element = clean(getattr(atom, "element", "")).upper()
                    if not element or element == "H":
                        continue
                    atoms.append(element)
                    coordinates.append(np.asarray(atom.coord, dtype=np.float32))
        break
    if not atoms:
        return None, "no_atoms_for_pocket_residue_ids", matched_residues, 0
    record = {
        "pocket": clean(row["sequence_key"]),
        "pocket_atoms": atoms,
        "pocket_coordinates": coordinates,
    }
    return record, "ok", matched_residues, len(atoms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ligands",
        default=str(DEFAULT_INPUT_DIR / "LIGAND_MODEL_READINESS_723_V1.csv"),
    )
    parser.add_argument(
        "--targets",
        default=str(DEFAULT_INPUT_DIR / "TARGET_MODEL_READINESS_463_V1.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_INPUT_DIR / "drugclip_inputs_v1"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ligands = pd.read_csv(args.ligands, low_memory=False).fillna("")
    targets = pd.read_csv(args.targets, low_memory=False).fillna("")
    targets = targets[targets["structure_bin"].isin(
        ["A_strict_overlapping_pocket", "B_strict_supported_overlap"]
    )].copy()

    ligand_records: list[dict[str, Any]] = []
    ligand_manifest: list[dict[str, Any]] = []
    for source_index, row in ligands.reset_index(drop=True).iterrows():
        smiles = clean(row["model_ligand_smiles"])
        identifier = "LIG_" + __import__("hashlib").sha1(smiles.encode("utf-8")).hexdigest()[:12]
        record, status = ligand_record(smiles, identifier, 20260720 + source_index)
        lmdb_index = len(ligand_records) if record is not None else None
        if record is not None:
            ligand_records.append(record)
        ligand_manifest.append(
            {
                "source_index": source_index,
                "lmdb_index": lmdb_index,
                "ligand_id": identifier,
                "model_ligand_smiles": smiles,
                "preparation_status": status,
            }
        )

    pocket_records: list[dict[str, Any]] = []
    pocket_manifest: list[dict[str, Any]] = []
    for _, row in targets.sort_values("sequence_key", kind="mergesort").iterrows():
        record, status, matched_residues, atom_count = pocket_record(row)
        lmdb_index = len(pocket_records) if record is not None else None
        if record is not None:
            pocket_records.append(record)
        pocket_manifest.append(
            {
                "lmdb_index": lmdb_index,
                "sequence_key": clean(row["sequence_key"]),
                "primary_gene": clean(row.get("primary_gene")),
                "structure_bin": clean(row.get("structure_bin")),
                "pdb_path": clean(row.get("pdb_path")),
                "pocket_residue_count_requested": len(parse_residue_ids(row.get("top_pocket_residue_ids"))),
                "pocket_residue_count_matched": matched_residues,
                "pocket_atom_count": atom_count,
                "preparation_status": status,
            }
        )

    ligand_lmdb = out_dir / "project723_ligands.lmdb"
    pocket_lmdb = out_dir / "project308_strict_pockets.lmdb"
    write_lmdb(ligand_records, ligand_lmdb)
    write_lmdb(pocket_records, pocket_lmdb)
    ligand_manifest_path = out_dir / "PROJECT723_LIGAND_LMDB_MANIFEST_V1.csv"
    pocket_manifest_path = out_dir / "PROJECT308_POCKET_LMDB_MANIFEST_V1.csv"
    pd.DataFrame(ligand_manifest).to_csv(ligand_manifest_path, index=False)
    pd.DataFrame(pocket_manifest).to_csv(pocket_manifest_path, index=False)
    summary = {
        "status": "passed" if len(ligand_records) >= 715 and len(pocket_records) >= 300 else "failed",
        "created_utc": now_utc(),
        "ligand_source_rows": int(len(ligands)),
        "ligand_lmdb_rows": int(len(ligand_records)),
        "ligand_failures": int(len(ligands) - len(ligand_records)),
        "strict_target_rows": int(len(targets)),
        "pocket_lmdb_rows": int(len(pocket_records)),
        "pocket_failures": int(len(targets) - len(pocket_records)),
        "pocket_atom_count": pd.Series([row["pocket_atom_count"] for row in pocket_manifest]).describe().to_dict(),
        "outputs": {
            "ligand_lmdb": str(ligand_lmdb.relative_to(ROOT)),
            "pocket_lmdb": str(pocket_lmdb.relative_to(ROOT)),
            "ligand_manifest": str(ligand_manifest_path.relative_to(ROOT)),
            "pocket_manifest": str(pocket_manifest_path.relative_to(ROOT)),
        },
    }
    summary_path = out_dir / "DRUGCLIP_INPUT_PREPARATION_V1_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "passed":
        raise RuntimeError("DrugCLIP input preparation contract failed")


if __name__ == "__main__":
    main()

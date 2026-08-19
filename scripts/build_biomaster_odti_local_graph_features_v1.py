#!/usr/bin/env python3
"""Build label-free ligand-atom and pocket-residue graph feature stores.

The stores are entity aligned, not pair/label derived:

* ligand nodes and bonds come only from the frozen canonical SMILES index;
* pocket nodes use the audited exact-sequence pocket mask, ESM2 residue
  representations and receptor C-alpha coordinates;
* no activity, affinity, split, docking score or positive/negative control is
  read.

Variable-size graphs are written as concatenated arrays plus dense entity
indices.  The trainer pads only the graphs present in a minibatch, avoiding a
62k-drug global padding tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1"
TOKEN_STORE = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1"
DEFAULT_DRUG_INDEX = STORE / "DRUG_FEATURE_INDEX_V1.csv.gz"
DEFAULT_TARGET_INDEX = STORE / "TARGET_FEATURE_INDEX_V1.csv.gz"
DEFAULT_TARGET_STRUCTURE = ROOT / "outputs/affinity_first_remote_discovery_v1/DTA_STAGE1_REMOTE_STRICT_STRUCTURE_V1.csv.gz"
DEFAULT_ESM2 = TOKEN_STORE / "DTIAM_ESM2_T33_650M_RESIDUE_FLOAT16_V1.npy"
DEFAULT_ESM2_INDEX = TOKEN_STORE / "DTIAM_ESM2_T33_650M_RESIDUE_INDEX_V1.csv.gz"
DEFAULT_POCKET_MASK = TOKEN_STORE / "DTIAM_ESM2_T33_650M_POCKET_MASK_UINT8_V1.npy"
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_local_graph_features_v1"


ATOM_FEATURE_DIM = 32
POCKET_AUX_DIM = 23
ELEMENTS = [6, 7, 8, 16, 15, 9, 17, 35, 53, 5, 14]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atom_features(atom: Chem.Atom) -> np.ndarray:
    value = np.zeros(ATOM_FEATURE_DIM, dtype=np.float32)
    atomic_number = int(atom.GetAtomicNum())
    element_index = ELEMENTS.index(atomic_number) if atomic_number in ELEMENTS else 11
    value[element_index] = 1.0
    degree = min(int(atom.GetDegree()), 5)
    value[12 + degree] = 1.0
    hybridizations = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    ]
    hybridization = atom.GetHybridization()
    hybrid_index = hybridizations.index(hybridization) if hybridization in hybridizations else 5
    value[18 + hybrid_index] = 1.0
    value[24] = float(np.clip(atom.GetFormalCharge(), -4, 4)) / 4.0
    value[25] = float(atom.GetIsAromatic())
    value[26] = float(atom.IsInRing())
    value[27] = min(float(atom.GetTotalNumHs()), 4.0) / 4.0
    value[28] = min(float(atom.GetTotalValence()), 8.0) / 8.0
    value[29] = min(float(atom.GetMass()), 250.0) / 250.0
    value[30] = float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    value[31] = float(atom.GetNoImplicit())
    return value


def bond_type(bond: Chem.Bond) -> int:
    kind = bond.GetBondType()
    if kind == Chem.BondType.SINGLE:
        return 1
    if kind == Chem.BondType.DOUBLE:
        return 2
    if kind == Chem.BondType.TRIPLE:
        return 3
    if kind == Chem.BondType.AROMATIC:
        return 4
    return 5


def parse_pdb_ca(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    residues: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if line[16:17] not in {" ", "A"}:
                continue
            try:
                key = (line[21:22].strip() or " ", int(line[22:26]))
                coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                bfactor = float(line[60:66])
            except ValueError:
                continue
            residues.setdefault(key, {"coord": coord, "bfactor": bfactor})
    return residues


def first_target_rows(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    required = {"sequence_key", "pdb_path", "sequence_match_status"}
    if not required.issubset(table.columns):
        raise ValueError(f"target structure table is missing {sorted(required - set(table.columns))}")
    columns = [
        "sequence_key", "pdb_path", "sequence_match_status",
        "top_pocket_residue_ids", "receptor_pdb_sha256",
    ]
    work = table[[column for column in columns if column in table.columns]].copy()
    work["__coverage"] = work.notna().sum(axis=1)
    work = work.sort_values(["sequence_key", "__coverage"], ascending=[True, False])
    return work.drop_duplicates("sequence_key", keep="first").drop(columns="__coverage")


def build_ligands(drug_index_path: Path, out_dir: Path, max_atoms: int) -> dict[str, Any]:
    drugs = pd.read_csv(drug_index_path, low_memory=False).sort_values("drug_feature_index")
    if not np.array_equal(
        drugs["drug_feature_index"].to_numpy(dtype=np.int64),
        np.arange(len(drugs), dtype=np.int64),
    ):
        raise ValueError("drug_feature_index must be dense and sorted")
    node_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    edge_type_chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    node_offset = 0
    edge_offset = 0
    for item in drugs.itertuples(index=False):
        index = int(item.drug_feature_index)
        smiles = str(item.model_ligand_smiles)
        molecule = Chem.MolFromSmiles(smiles)
        reason = None
        if molecule is None:
            reason = "rdkit_parse_failed"
        elif molecule.GetNumAtoms() < 1:
            reason = "zero_atoms"
        elif molecule.GetNumAtoms() > max_atoms:
            reason = f"atom_count_exceeds_{max_atoms}"
        if reason is not None:
            rows.append({
                "drug_feature_index": index,
                "node_offset": node_offset, "node_count": 0,
                "edge_offset": edge_offset, "edge_count": 0,
                "graph_available": False, "failure_reason": reason,
            })
            failures.append({"drug_feature_index": index, "reason": reason})
            continue
        nodes = np.stack([atom_features(atom) for atom in molecule.GetAtoms()])
        edge_index: list[tuple[int, int]] = []
        edge_types: list[int] = []
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            kind = bond_type(bond)
            edge_index.extend([(left, right), (right, left)])
            edge_types.extend([kind, kind])
        edges = (
            np.asarray(edge_index, dtype=np.int16)
            if edge_index else np.zeros((0, 2), dtype=np.int16)
        )
        types = np.asarray(edge_types, dtype=np.uint8)
        node_chunks.append(nodes.astype(np.float16))
        edge_chunks.append(edges)
        edge_type_chunks.append(types)
        rows.append({
            "drug_feature_index": index,
            "node_offset": node_offset, "node_count": len(nodes),
            "edge_offset": edge_offset, "edge_count": len(edges),
            "graph_available": True, "failure_reason": "",
        })
        node_offset += len(nodes)
        edge_offset += len(edges)
    nodes = np.concatenate(node_chunks, axis=0) if node_chunks else np.zeros((0, ATOM_FEATURE_DIM), np.float16)
    edges = np.concatenate(edge_chunks, axis=0) if edge_chunks else np.zeros((0, 2), np.int16)
    edge_types = np.concatenate(edge_type_chunks, axis=0) if edge_type_chunks else np.zeros(0, np.uint8)
    node_path = out_dir / "LIGAND_ATOM_FEATURES_FLOAT16_V1.npy"
    edge_path = out_dir / "LIGAND_EDGE_INDEX_INT16_V1.npy"
    edge_type_path = out_dir / "LIGAND_EDGE_TYPE_UINT8_V1.npy"
    index_path = out_dir / "LIGAND_GRAPH_INDEX_V1.csv.gz"
    np.save(node_path, nodes)
    np.save(edge_path, edges)
    np.save(edge_type_path, edge_types)
    pd.DataFrame(rows).to_csv(index_path, index=False, compression="gzip")
    counts = pd.Series([row["node_count"] for row in rows if row["graph_available"]])
    return {
        "entity_count": len(rows),
        "available_count": int(sum(bool(row["graph_available"]) for row in rows)),
        "quarantined_count": len(failures),
        "quarantine_reason_counts": pd.Series([item["reason"] for item in failures]).value_counts().to_dict(),
        "total_nodes": int(len(nodes)), "total_directed_edges": int(len(edges)),
        "atom_feature_dim": ATOM_FEATURE_DIM, "max_atoms": max_atoms,
        "node_count_quantiles": {
            str(q): float(counts.quantile(q)) for q in [0.5, 0.9, 0.95, 0.99, 1.0]
        },
        "artifacts": {
            "nodes": {"path": str(node_path), "sha256": sha256(node_path)},
            "edges": {"path": str(edge_path), "sha256": sha256(edge_path)},
            "edge_types": {"path": str(edge_type_path), "sha256": sha256(edge_type_path)},
            "index": {"path": str(index_path), "sha256": sha256(index_path)},
        },
    }


def build_pockets(
    target_index_path: Path,
    target_structure_path: Path,
    esm2_path: Path,
    esm2_index_path: Path,
    pocket_mask_path: Path,
    out_dir: Path,
    max_residues: int,
) -> dict[str, Any]:
    targets = pd.read_csv(target_index_path, low_memory=False).sort_values("target_feature_index")
    token_index = pd.read_csv(esm2_index_path, low_memory=False).sort_values("target_feature_index")
    if len(targets) != len(token_index):
        raise ValueError("target and token indices have different lengths")
    if not np.array_equal(
        targets["target_feature_index"].to_numpy(dtype=np.int64),
        np.arange(len(targets), dtype=np.int64),
    ):
        raise ValueError("target_feature_index must be dense and sorted")
    structure = first_target_rows(target_structure_path).set_index("sequence_key")
    esm2 = np.load(esm2_path, mmap_mode="r")
    pocket_mask = np.load(pocket_mask_path, mmap_mode="r")
    if esm2.ndim != 2 or pocket_mask.shape != (esm2.shape[0],):
        raise ValueError("ESM2 residue matrix and pocket mask are misaligned")
    esm_chunks: list[np.ndarray] = []
    aux_chunks: list[np.ndarray] = []
    coord_chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    offset = 0
    for target, token in zip(targets.itertuples(index=False), token_index.itertuples(index=False)):
        target_id = int(target.target_feature_index)
        sequence_key = str(target.sequence_key)
        start, length = int(token.token_offset), int(token.token_length)
        local_mask = np.asarray(pocket_mask[start : start + length], dtype=bool)
        positions = np.flatnonzero(local_mask)
        reason = None
        pdb_path = None
        if sequence_key not in structure.index:
            reason = "no_structure_row"
        else:
            row = structure.loc[sequence_key]
            if str(row.get("sequence_match_status", "")) != "exact_match":
                reason = "sequence_not_exact"
            else:
                pdb_path = Path(str(row.get("pdb_path", "")))
                if not pdb_path.is_file():
                    reason = "pdb_missing"
        if reason is None and len(positions) == 0:
            reason = "empty_pocket_mask"
        if reason is None and len(positions) > max_residues:
            reason = f"pocket_residue_count_exceeds_{max_residues}"
        residue_map = parse_pdb_ca(pdb_path) if reason is None and pdb_path else {}
        chain = "A"
        mapped = [(position, residue_map.get((chain, int(position) + 1))) for position in positions]
        if reason is None and any(value is None for _, value in mapped):
            reason = "pocket_coordinate_mapping_incomplete"
        if reason is not None:
            rows.append({
                "target_feature_index": target_id, "sequence_key": sequence_key,
                "node_offset": offset, "node_count": 0,
                "graph_available": False, "failure_reason": reason,
            })
            failures.append({"target_feature_index": target_id, "reason": reason})
            continue
        sequence = str(target.protein_sequence)
        esm_values = np.asarray(esm2[start : start + length][positions], dtype=np.float16)
        aux = np.zeros((len(positions), POCKET_AUX_DIM), dtype=np.float32)
        coords = np.zeros((len(positions), 3), dtype=np.float32)
        for local, (position, value) in enumerate(mapped):
            aa = sequence[int(position)] if int(position) < len(sequence) else "X"
            if aa in AMINO_ACIDS:
                aux[local, AMINO_ACIDS.index(aa)] = 1.0
            aux[local, 20] = float(value["bfactor"]) / 100.0
            aux[local, 21] = float(position) / max(len(sequence) - 1, 1)
            aux[local, 22] = math.sin(float(position) / max(len(sequence), 1) * math.pi)
            coords[local] = np.asarray(value["coord"], dtype=np.float32)
        esm_chunks.append(esm_values)
        aux_chunks.append(aux.astype(np.float16))
        coord_chunks.append(coords)
        rows.append({
            "target_feature_index": target_id, "sequence_key": sequence_key,
            "node_offset": offset, "node_count": len(positions),
            "graph_available": True, "failure_reason": "",
        })
        offset += len(positions)
    esm_nodes = np.concatenate(esm_chunks, axis=0) if esm_chunks else np.zeros((0, esm2.shape[1]), np.float16)
    aux_nodes = np.concatenate(aux_chunks, axis=0) if aux_chunks else np.zeros((0, POCKET_AUX_DIM), np.float16)
    coords = np.concatenate(coord_chunks, axis=0) if coord_chunks else np.zeros((0, 3), np.float32)
    esm_path_out = out_dir / "POCKET_ESM2_RESIDUE_FLOAT16_V1.npy"
    aux_path = out_dir / "POCKET_RESIDUE_AUX_FLOAT16_V1.npy"
    coord_path = out_dir / "POCKET_CA_COORD_FLOAT32_V1.npy"
    index_path = out_dir / "POCKET_GRAPH_INDEX_V1.csv.gz"
    np.save(esm_path_out, esm_nodes)
    np.save(aux_path, aux_nodes)
    np.save(coord_path, coords)
    pd.DataFrame(rows).to_csv(index_path, index=False, compression="gzip")
    counts = pd.Series([row["node_count"] for row in rows if row["graph_available"]])
    return {
        "entity_count": len(rows),
        "available_count": int(sum(bool(row["graph_available"]) for row in rows)),
        "quarantined_count": len(failures),
        "quarantine_reason_counts": pd.Series([item["reason"] for item in failures]).value_counts().to_dict(),
        "total_nodes": int(len(esm_nodes)), "esm_feature_dim": int(esm2.shape[1]),
        "aux_feature_dim": POCKET_AUX_DIM, "max_residues": max_residues,
        "node_count_quantiles": {
            str(q): float(counts.quantile(q)) for q in [0.5, 0.9, 0.95, 0.99, 1.0]
        },
        "artifacts": {
            "esm_nodes": {"path": str(esm_path_out), "sha256": sha256(esm_path_out)},
            "aux_nodes": {"path": str(aux_path), "sha256": sha256(aux_path)},
            "coordinates": {"path": str(coord_path), "sha256": sha256(coord_path)},
            "index": {"path": str(index_path), "sha256": sha256(index_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-index", default=str(DEFAULT_DRUG_INDEX))
    parser.add_argument("--target-index", default=str(DEFAULT_TARGET_INDEX))
    parser.add_argument("--target-structure", default=str(DEFAULT_TARGET_STRUCTURE))
    parser.add_argument("--esm2", default=str(DEFAULT_ESM2))
    parser.add_argument("--esm2-index", default=str(DEFAULT_ESM2_INDEX))
    parser.add_argument("--pocket-mask", default=str(DEFAULT_POCKET_MASK))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--max-pocket-residues", type=int, default=128)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "drug_index": Path(args.drug_index),
        "target_index": Path(args.target_index),
        "target_structure": Path(args.target_structure),
        "esm2": Path(args.esm2),
        "esm2_index": Path(args.esm2_index),
        "pocket_mask": Path(args.pocket_mask),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    ligand = build_ligands(inputs["drug_index"], out_dir, args.max_atoms)
    pocket = build_pockets(
        inputs["target_index"], inputs["target_structure"], inputs["esm2"],
        inputs["esm2_index"], inputs["pocket_mask"], out_dir,
        args.max_pocket_residues,
    )
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "route": "LABEL_FREE_ENTITY_ALIGNED_LIGAND_AND_POCKET_GRAPHS_V1",
        "label_dependency": "NONE",
        "pair_pose_available": False,
        "cross_distance_contract": (
            "intra-ligand bonds and intra-pocket CA distances only; no ligand-pocket "
            "cross-distance is created without a jointly posed complex"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "ligand": ligand,
        "pocket": pocket,
    }
    manifest_path = out_dir / "LOCAL_GRAPH_FEATURE_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

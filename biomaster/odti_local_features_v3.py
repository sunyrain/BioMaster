"""Lazy stereochemistry-aware molecular graphs and complete sequence regions.

Target alignment is by SHA256 of the full amino-acid sequence. Sequence windows
average contiguous residue bins so every available residue contributes; they
must not be described as pockets. Truncated residue caches are rejected.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRUG_INDEX = ROOT / "outputs/retrain_20260901/comprehensive_training_v1/DRUG_FEATURE_INDEX_COMPREHENSIVE_V1.csv.gz"
DEFAULT_TARGET_INDICES = (
    ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz",
    ROOT / "outputs/biomaster_bindingdb_affinity_feature_package_v1/BINDINGDB_NEW_TARGET_FEATURE_INDEX_V1.csv.gz",
    ROOT / "outputs/biomaster_deployment_augmentation_v1/NEW_TARGET_FEATURE_INDEX_V1.csv.gz",
)
DEFAULT_RESIDUE_ARRAY = ROOT / "outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy"
DEFAULT_RESIDUE_INDEX = ROOT / "outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_INDEX_COMBINED_V1.csv.gz"
ATOM_FEATURE_DIM = 40
FEATURE_VERSION = "SEQUENCE_OR_P2RANK_REGIONS_STEREO_V3_2"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256("".join(str(sequence).split()).upper().encode()).hexdigest()


def _table(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (list, tuple)):
        return pd.concat([_table(part) for part in value], ignore_index=True)
    return pd.read_csv(value, low_memory=False)


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def target_maps(table: pd.DataFrame) -> tuple[dict[int, str], dict[str, str], dict[str, int]]:
    ids: dict[int, str] = {}
    aliases: dict[str, str] = {}
    lengths: dict[str, int] = {}
    for row in table.to_dict("records"):
        sequence = _text(row.get("protein_sequence", "")) or _text(row.get("sequence", ""))
        sequence = "".join(sequence.split()).upper()
        declared = next((_text(row.get(key, "")) for key in ("sequence_sha256", "target_sequence_hash", "external_target_hash") if _text(row.get(key, ""))), "")
        digest = sequence_sha256(sequence) if sequence else declared
        if not digest or len(digest) != 64:
            raise ValueError("target index requires full sequence or its SHA256")
        if sequence and declared and declared != digest:
            raise ValueError("declared sequence SHA256 does not match sequence")
        index = int(row["target_feature_index"])
        if index in ids and ids[index] != digest:
            raise ValueError(f"conflicting sequence for target_feature_index={index}")
        ids[index] = digest
        length = len(sequence) if sequence else int(row.get("sequence_length", 0))
        if digest in lengths and lengths[digest] != length:
            raise ValueError("conflicting sequence lengths for the same hash")
        lengths[digest] = length
        for alias in (digest, _text(row.get("sequence_key", ""))):
            if alias:
                if alias in aliases and aliases[alias] != digest:
                    raise ValueError("ambiguous sequence alias")
                aliases[alias] = digest
    return ids, aliases, lengths


def molecular_graph(smiles: str, max_atoms: int = 128) -> dict[str, Any]:
    """Full molecular graph; oversized molecules fall back, never truncate."""
    molecule = Chem.MolFromSmiles(str(smiles))
    reason = ""
    if molecule is None:
        reason = "invalid_smiles"
    elif molecule.GetNumAtoms() == 0:
        reason = "empty_molecule"
    elif molecule.GetNumAtoms() > max_atoms:
        reason = "atom_count_exceeds_limit"
    if reason:
        return {"atom_features": np.zeros((0, ATOM_FEATURE_DIM), np.float32), "bond_type": np.zeros((0, 0), np.int64), "bond_stereo": np.zeros((0, 0), np.int64), "available": False, "reason": reason}
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    elements = [6, 7, 8, 16, 15, 9, 17, 35, 53, 5, 14]
    hybridizations = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
    n = molecule.GetNumAtoms()
    features = np.zeros((n, ATOM_FEATURE_DIM), np.float32)
    for atom in molecule.GetAtoms():
        f = features[atom.GetIdx()]
        z = atom.GetAtomicNum()
        f[elements.index(z) if z in elements else 11] = 1
        f[12 + min(atom.GetDegree(), 5)] = 1
        h = atom.GetHybridization()
        f[18 + (hybridizations.index(h) if h in hybridizations else 5)] = 1
        f[24:30] = [np.clip(atom.GetFormalCharge(), -4, 4) / 4, atom.GetIsAromatic(), atom.IsInRing(), min(atom.GetTotalNumHs(), 4) / 4, min(atom.GetTotalValence(), 8) / 8, min(atom.GetMass(), 250) / 250]
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        if cip in ("R", "S", "r", "s"):
            f[30 + ("R", "S", "r", "s").index(cip)] = 1
        f[34] = atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and not cip
        f[35] = atom.HasProp("_ChiralityPossible") and not cip
        f[36] = min(atom.GetIsotope(), 250) / 250
        f[37] = min(atom.GetNumRadicalElectrons(), 4) / 4
        f[38] = atom.GetNoImplicit()
        f[39] = 1
    types = np.zeros((n, n), np.int64)
    stereos = np.zeros((n, n), np.int64)
    kinds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    stereo_kinds = [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOANY]
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        types[i, j] = types[j, i] = kinds.index(bond.GetBondType()) + 1 if bond.GetBondType() in kinds else 5
        stereo = bond.GetStereo()
        stereos[i, j] = stereos[j, i] = stereo_kinds.index(stereo) + 1 if stereo in stereo_kinds else (0 if stereo == Chem.BondStereo.STEREONONE else 6)
    return {"atom_features": features, "bond_type": types, "bond_stereo": stereos, "available": True, "reason": ""}


def sequence_regions(residues: np.ndarray, max_regions: int = 4, tokens_per_region: int = 32) -> dict[str, Any]:
    """Average contiguous bins over the FULL sequence, retaining coverage spans."""
    if residues.ndim != 2 or min(max_regions, tokens_per_region) < 1:
        raise ValueError("expected rank-two residues and positive region limits")
    length, width = residues.shape
    features = np.zeros((max_regions, tokens_per_region, width), np.float16)
    positions = np.zeros((max_regions, tokens_per_region, 3), np.float32)
    mask = np.zeros((max_regions, tokens_per_region), bool)
    spans = np.full((max_regions, tokens_per_region, 2), -1, np.int64)
    if length:
        if not np.isfinite(residues).all():
            raise ValueError("nonfinite residue features")
        region_count = min(max_regions, max(1, int(np.ceil(length / tokens_per_region))))
        region_edges = np.linspace(0, length, region_count + 1, dtype=np.int64)
        for region, (start, end) in enumerate(zip(region_edges[:-1], region_edges[1:])):
            count = min(tokens_per_region, int(end - start))
            edges = np.linspace(start, end, count + 1, dtype=np.int64)
            for token, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                features[region, token] = np.asarray(residues[left:right], np.float32).mean(0)
                positions[region, token] = [left / length, right / length, (left + right) / (2 * length)]
                spans[region, token] = [left, right]
                mask[region, token] = True
    return {"residue_features": features, "residue_positions": positions, "residue_mask": mask, "region_mask": mask.any(-1), "coverage_spans": spans, "sequence_length": length, "available": bool(length), "region_kind": "SEQUENCE_COVERAGE_WINDOWS"}


def pocket_sequence_regions(residues: np.ndarray, pockets: list[dict[str, Any]],
                            max_regions: int = 4, tokens_per_region: int = 32) -> dict[str, Any]:
    """Up to three selected pockets plus one FULL-sequence coverage region.

    Pockets contain noncontiguous canonical residue indices. Their bins retain
    explicit index membership; ``coverage_spans`` is deliberately -1 there,
    because min/max would incorrectly suggest every intervening residue was
    represented. Only the full-sequence region uses continuous coverage spans.
    """
    if not pockets or max_regions < 2 or len(residues) == 0:
        return sequence_regions(residues, max_regions, tokens_per_region)
    length, width = residues.shape
    result = sequence_regions(np.zeros((0, width), residues.dtype), max_regions, tokens_per_region)
    selected = pockets[:min(3, max_regions - 1)]
    membership: list[list[list[int]]] = [[[] for _ in range(tokens_per_region)] for _ in range(max_regions)]
    types = ["PADDING"] * max_regions
    for region, pocket in enumerate(selected):
        indices = np.asarray(pocket["residue_indices"], dtype=np.int64)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= length) or len(np.unique(indices)) != len(indices):
            raise ValueError("pocket indices outside complete sequence or duplicated")
        indices = np.sort(indices)
        count = min(tokens_per_region, len(indices))
        edges = np.linspace(0, len(indices), count + 1, dtype=np.int64)
        for token, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            chosen = indices[left:right]
            values = np.asarray(residues[chosen], np.float32)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite pocket residue features")
            result["residue_features"][region, token] = values.mean(0)
            result["residue_positions"][region, token] = [chosen[0] / length, (chosen[-1] + 1) / length, (chosen.mean() + 0.5) / length]
            result["residue_mask"][region, token] = True
            membership[region][token] = chosen.tolist()
        types[region] = "P2RANK_PREDICTED_POCKET"
    full = sequence_regions(residues, 1, tokens_per_region)
    full_index = max_regions - 1
    for key in ("residue_features", "residue_positions", "residue_mask", "coverage_spans"):
        result[key][full_index] = full[key][0]
    types[full_index] = "FULL_SEQUENCE_COVERAGE"
    result.update(region_mask=result["residue_mask"].any(-1), sequence_length=length,
                  available=True, region_kind="P2RANK_POCKETS_AND_FULL_SEQUENCE",
                  region_types=types, pocket_residue_index_bins=membership,
                  selected_pockets=selected, full_sequence_region_index=full_index)
    return result


class LocalFeatureStore:
    def __init__(self, drug_index: Any = DEFAULT_DRUG_INDEX, target_index: Any = None,
                 residue_array: Any = DEFAULT_RESIDUE_ARRAY, residue_index: Any = DEFAULT_RESIDUE_INDEX,
                 residue_target_index: Any = None, *, max_regions: int = 4, tokens_per_region: int = 32,
                 max_atoms: int = 128, cache_size: int = 8192, cache_dir: str | Path | None = None,
                 pocket_store: str | Path | None = None) -> None:
        if min(max_regions, tokens_per_region, max_atoms, cache_size) < 1:
            raise ValueError("local feature limits must be positive")
        self.max_regions, self.tokens_per_region, self.max_atoms, self.cache_size = max_regions, tokens_per_region, max_atoms, cache_size
        drugs = _table(drug_index)
        smiles_column = "model_ligand_smiles" if "model_ligand_smiles" in drugs else "input_smiles"
        if drugs["drug_feature_index"].duplicated().any():
            raise ValueError("duplicate drug indices")
        self.smiles = dict(zip(drugs.drug_feature_index.astype(int), drugs[smiles_column].astype(str)))
        targets = _table(DEFAULT_TARGET_INDICES if target_index is None else target_index)
        self.target_hashes, aliases, self.sequence_lengths = target_maps(targets)
        source_ids, source_aliases, source_lengths = target_maps(targets if residue_target_index is None else _table(residue_target_index))
        aliases.update(source_aliases)
        self.residues = np.load(residue_array, mmap_mode="r") if isinstance(residue_array, (str, Path)) else np.asarray(residue_array)
        if self.residues.ndim != 2:
            raise ValueError("residue array must be [total_residues,embedding_width]")
        self.residue_dim = int(self.residues.shape[1])
        self.residue_rows: dict[str, tuple[int, int]] = {}
        for row in _table(residue_index).to_dict("records"):
            key = _text(row.get("sequence_sha256", "")) or _text(row.get("target_sequence_hash", "")) or _text(row.get("sequence_key", ""))
            digest = aliases.get(key, key if len(key) == 64 else "")
            if not digest and not key and "target_feature_index" in row:
                # Explicit source ID lookup, never implicit dataframe row order.
                digest = source_ids.get(int(row["target_feature_index"]), "")
            if not digest:
                raise ValueError(f"cannot resolve residue sequence identity: {key}")
            start, length = int(row["token_offset"]), int(row["token_length"])
            expected = self.sequence_lengths.get(digest, source_lengths.get(digest, int(row.get("sequence_length", length))))
            if length != expected:
                raise ValueError("truncated residue cache cannot satisfy full-sequence coverage")
            if start < 0 or length < 0 or start + length > len(self.residues):
                raise ValueError("residue offset outside array")
            if digest in self.residue_rows and self.residue_rows[digest] != (start, length):
                raise ValueError("duplicate ambiguous residue sequence")
            self.residue_rows[digest] = (start, length)
        self.drug_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.target_cache: dict[str, dict[str, Any]] = {}
        self.pocket_store_path = Path(pocket_store).resolve() if pocket_store is not None else None
        self.pocket_targets: dict[str, Any] = {}
        if self.pocket_store_path is not None:
            from biomaster.odti_pockets_v3 import load_pocket_store
            payload = load_pocket_store(self.pocket_store_path)
            self.pocket_targets = payload["targets"]
            for digest, item in self.pocket_targets.items():
                if digest in self.sequence_lengths and item["sequence_length"] != self.sequence_lengths[digest]:
                    raise ValueError("pocket cache and target index sequence lengths disagree")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def drug(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index in self.drug_cache:
            self.drug_cache.move_to_end(index)
            return self.drug_cache[index]
        if index not in self.smiles:
            raise KeyError(f"unknown drug_feature_index {index}")
        cache_key = hashlib.sha256(f"{FEATURE_VERSION}|{self.max_atoms}|{self.smiles[index]}".encode()).hexdigest()
        path = self.cache_dir / f"drug_{cache_key}.npz" if self.cache_dir else None
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as data:
                graph = {key: data[key] for key in ("atom_features", "bond_type", "bond_stereo")}
                graph.update(available=bool(data["available"]), reason=str(data["reason"]))
        else:
            graph = molecular_graph(self.smiles[index], self.max_atoms)
            if path is not None:
                np.savez_compressed(path, **graph)
        self.drug_cache[index] = graph
        if len(self.drug_cache) > self.cache_size:
            self.drug_cache.popitem(last=False)
        return graph

    def target(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index not in self.target_hashes:
            raise KeyError(f"unknown target_feature_index {index}")
        digest = self.target_hashes[index]
        if digest not in self.target_cache:
            row = self.residue_rows.get(digest)
            residues = self.residues[row[0]:row[0] + row[1]] if row else np.zeros((0, self.residue_dim), np.float16)
            pockets = self.pocket_targets.get(digest, {}).get("pockets", [])
            item = pocket_sequence_regions(residues, pockets, self.max_regions, self.tokens_per_region)
            item.update(sequence_sha256=digest, reason="" if row else "no_complete_residue_cache", expected_sequence_length=self.sequence_lengths[digest])
            self.target_cache[digest] = item
        return self.target_cache[digest]

    def batch(self, drug_ids: Sequence[int], target_ids: Sequence[int], device: Any = None) -> dict[str, torch.Tensor]:
        if len(drug_ids) != len(target_ids) or not len(drug_ids):
            raise ValueError("drug and target IDs must be nonempty and equal length")
        graphs = [self.drug(int(i)) for i in drug_ids]
        targets = [self.target(int(i)) for i in target_ids]
        count = len(graphs)
        atoms = max(1, max(len(graph["atom_features"]) for graph in graphs))
        arrays = {"atom_features": np.zeros((count, atoms, ATOM_FEATURE_DIM), np.float32), "bond_type": np.zeros((count, atoms, atoms), np.int64), "bond_stereo": np.zeros((count, atoms, atoms), np.int64), "atom_mask": np.zeros((count, atoms), bool)}
        for i, graph in enumerate(graphs):
            n = len(graph["atom_features"])
            arrays["atom_features"][i, :n] = graph["atom_features"]
            arrays["bond_type"][i, :n, :n] = graph["bond_type"]
            arrays["bond_stereo"][i, :n, :n] = graph["bond_stereo"]
            arrays["atom_mask"][i, :n] = True
        for key in ("residue_features", "residue_positions", "residue_mask", "region_mask"):
            arrays[key] = np.stack([target[key] for target in targets])
        result = {}
        for key, value in arrays.items():
            tensor = torch.from_numpy(value)
            if tensor.is_floating_point():
                tensor = tensor.float()
            result[key] = tensor.to(device) if device is not None else tensor
        return result

    def coverage_report(self) -> dict[str, Any]:
        available = [i for i, digest in self.target_hashes.items() if digest in self.residue_rows]
        pocket_available = [i for i in available if self.target_hashes[i] in self.pocket_targets and self.max_regions >= 2]
        return {"feature_version": FEATURE_VERSION, "region_kind": "P2RANK_POCKETS_WITH_FULL_SEQUENCE_AND_SEQUENCE_FALLBACK" if self.pocket_store_path else "SEQUENCE_COVERAGE_WINDOWS", "target_count": len(self.target_hashes), "targets_with_complete_residues": len(available), "targets_with_pocket_and_complete_residues": len(pocket_available), "targets_using_sequence_only_fallback": len(available) - len(pocket_available), "pocket_store": str(self.pocket_store_path) if self.pocket_store_path else None, "missing_target_ids": sorted(set(self.target_hashes) - set(available)), "drug_count": len(self.smiles), "max_regions": self.max_regions, "tokens_per_region": self.tokens_per_region, "max_atoms": self.max_atoms, "residue_dim": self.residue_dim, "cached_drug_count": len(self.drug_cache), "cached_drug_failures": {str(i): graph["reason"] for i, graph in self.drug_cache.items() if not graph["available"]}, "complete_sequence_coverage_for_available_targets": True, "uses_complex_pose": False}

#!/usr/bin/env python3
"""Train the strengthened BioMaster-ODTI V2 ranker.

V2 preserves the frozen S1--S5 split contract used by V1, but adds:

* a bilinear drug--target interaction term;
* an optional structure/pocket residual branch with exact missing-modality
  fallback;
* a multi-task loss with contrastive alignment and optional observation labels;
* interval-aware auxiliary affinity supervision.

The optional structure CSV is deliberately simple and auditable.  It must
contain ``calibration_pair_id`` and numeric feature columns.  An optional
``structure_mask`` column marks rows with valid structure evidence; missing
rows receive zeros and mask=0, so the final score falls back exactly to the
sequence/chemistry base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2, odti_v2_loss  # noqa: E402
from run_biomaster_odti_baselines_v1 import metrics, split_masks  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
STORE = BASE / "feature_store_v1"
PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
MORGAN = STORE / "MORGAN2048_UINT8_V1.npy"
PROTBERT = STORE / "PROTBERT1024_FLOAT32_V1.npy"
FEATURE_AUDIT = STORE / "FEATURE_STORE_AUDIT_V1.json"
DEFAULT_TARGET_AUX = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
DEFAULT_DRUG_AUX = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_BERMOL768_FLOAT32_V1.npy"
DRUG_AUX_INDEX = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_BERMOL_DRUG_INDEX_V1.csv.gz"
DEFAULT_LOCAL_GRAPH_DIR = ROOT / "outputs/biomaster_odti_local_graph_features_v1"
DEFAULT_OUT = BASE / "biomaster_odti_v2"


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_partial_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | None,
) -> dict[str, object]:
    """Load shape-compatible weights while keeping optional branches additive."""

    if not checkpoint_path:
        return {"enabled": False, "path": None, "loaded": 0, "skipped": 0}
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("init checkpoint does not contain a state dictionary")
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "enabled": True,
        "path": str(path),
        "sha256": sha256(path),
        "loaded": len(compatible),
        "skipped": len(state) - len(compatible),
        "source_config": payload.get("config"),
    }


def set_local_only_trainable(
    model: torch.nn.Module,
    enabled: bool,
) -> list[str]:
    """Freeze the pretrained trunk and expose only local-pair parameters."""

    if not enabled:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return []
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("local_")
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError("local-only training requested but local branch is disabled")
    return trainable


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a validation-only temperature without introducing a scipy dependency."""

    candidates = np.exp(np.linspace(-2.0, 2.0, 81))
    losses = []
    labels_float = labels.astype(np.float64)
    for temperature in candidates:
        scaled = np.clip(logits / temperature, -60.0, 60.0)
        losses.append(float(np.mean(np.logaddexp(0.0, scaled) - labels_float * scaled)))
    return float(candidates[int(np.argmin(losses))])


def load_structure_features(
    path: str | None,
    pair_ids: pd.Series,
    expected_dim: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], str | None]:
    """Load an optional pair-aligned structure feature table."""

    if not path:
        dim = int(expected_dim or 0)
        return (
            np.zeros((len(pair_ids), dim), dtype=np.float32),
            np.zeros(len(pair_ids), dtype=np.float32),
            [],
            None,
        )
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    table = pd.read_csv(source, low_memory=False)
    if "calibration_pair_id" not in table.columns:
        raise ValueError("structure feature CSV requires calibration_pair_id")
    if table["calibration_pair_id"].duplicated().any():
        raise ValueError("structure feature CSV has duplicate calibration_pair_id")
    feature_columns = [
        column
        for column in table.columns
        if column not in {"calibration_pair_id", "structure_mask"}
    ]
    if not feature_columns:
        raise ValueError("structure feature CSV has no numeric feature columns")
    numeric = table[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("structure feature CSV contains non-numeric or missing values")
    if expected_dim is not None and len(feature_columns) != expected_dim:
        raise ValueError(
            f"structure feature width {len(feature_columns)} != expected {expected_dim}"
        )
    indexed = table.set_index("calibration_pair_id")
    numeric_indexed = numeric.set_index(table["calibration_pair_id"])
    requested = pair_ids.astype(str)
    output = np.zeros((len(requested), len(feature_columns)), dtype=np.float32)
    mask = np.zeros(len(requested), dtype=np.float32)
    present = requested.isin(indexed.index).to_numpy()
    if present.any():
        present_ids = requested[present]
        output[present] = numeric_indexed.loc[present_ids].to_numpy(dtype=np.float32)
        if "structure_mask" in indexed.columns:
            mask[present] = pd.to_numeric(
                indexed.loc[present_ids, "structure_mask"], errors="raise"
            ).to_numpy(dtype=np.float32)
        else:
            mask[present] = 1.0
    return output, np.clip(mask, 0.0, 1.0), feature_columns, str(source)


def infer_structure_group_dims(
    feature_columns: list[str],
    mode: str,
) -> tuple[int, ...]:
    """Infer contiguous semantic slices from a V2 pocket-context store."""

    if mode == "flat" or not feature_columns:
        return ()
    if mode != "grouped":
        raise ValueError("structure encoder mode must be flat or grouped")
    prefixes = ("quality_", "chem_", "geom_", "consensus_")
    dimensions = tuple(
        sum(column.startswith(prefix) for column in feature_columns)
        for prefix in prefixes
    )
    if any(value < 1 for value in dimensions) or sum(dimensions) != len(feature_columns):
        raise ValueError(
            "grouped structure encoder requires quality_/chem_/geom_/consensus_ columns"
        )
    expected = [
        column
        for prefix in prefixes
        for column in feature_columns
        if column.startswith(prefix)
    ]
    if expected != feature_columns:
        raise ValueError("grouped structure columns must be contiguous and ordered")
    return dimensions


def _boolean_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    return (
        series.fillna("").astype(str).str.strip().str.lower()
        .isin({"1", "true", "yes", "y"})
        .to_numpy(dtype=bool)
    )


def load_local_graph_features(
    directory: str | None,
    drug_count: int,
    target_count: int,
) -> dict[str, object] | None:
    """Load the label-free concatenated ligand/pocket graph feature store."""

    if not directory:
        return None
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    names = {
        "ligand_nodes": "LIGAND_ATOM_FEATURES_FLOAT16_V1.npy",
        "ligand_edges": "LIGAND_EDGE_INDEX_INT16_V1.npy",
        "ligand_edge_types": "LIGAND_EDGE_TYPE_UINT8_V1.npy",
        "ligand_index": "LIGAND_GRAPH_INDEX_V1.csv.gz",
        "pocket_nodes": "POCKET_ESM2_RESIDUE_FLOAT16_V1.npy",
        "pocket_aux": "POCKET_RESIDUE_AUX_FLOAT16_V1.npy",
        "pocket_coords": "POCKET_CA_COORD_FLOAT32_V1.npy",
        "pocket_index": "POCKET_GRAPH_INDEX_V1.csv.gz",
        "manifest": "LOCAL_GRAPH_FEATURE_MANIFEST_V1.json",
    }
    paths = {name: root / filename for name, filename in names.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("status") != "PASS" or manifest.get("label_dependency") != "NONE":
        raise ValueError("local graph feature manifest must be label-free and PASS")
    ligand_index = pd.read_csv(paths["ligand_index"], low_memory=False).sort_values(
        "drug_feature_index"
    ).reset_index(drop=True)
    pocket_index = pd.read_csv(paths["pocket_index"], low_memory=False).sort_values(
        "target_feature_index"
    ).reset_index(drop=True)
    if len(ligand_index) < drug_count or not np.array_equal(
        ligand_index["drug_feature_index"].to_numpy(dtype=np.int64)[:drug_count],
        np.arange(drug_count, dtype=np.int64),
    ):
        raise ValueError("ligand graph index is not dense/aligned")
    if len(pocket_index) < target_count or not np.array_equal(
        pocket_index["target_feature_index"].to_numpy(dtype=np.int64)[:target_count],
        np.arange(target_count, dtype=np.int64),
    ):
        raise ValueError("pocket graph index is not dense/aligned")
    ligand_nodes = np.load(paths["ligand_nodes"], mmap_mode="r")
    ligand_edges = np.load(paths["ligand_edges"], mmap_mode="r")
    ligand_edge_types = np.load(paths["ligand_edge_types"], mmap_mode="r")
    pocket_nodes = np.load(paths["pocket_nodes"], mmap_mode="r")
    pocket_aux = np.load(paths["pocket_aux"], mmap_mode="r")
    pocket_coords = np.load(paths["pocket_coords"], mmap_mode="r")
    if ligand_nodes.ndim != 2 or ligand_edges.ndim != 2 or ligand_edges.shape[1] != 2:
        raise ValueError("ligand graph arrays have invalid shapes")
    if ligand_edge_types.shape != (len(ligand_edges),):
        raise ValueError("ligand edge types are not aligned")
    if pocket_nodes.ndim != 2 or pocket_aux.ndim != 2 or pocket_coords.ndim != 2:
        raise ValueError("pocket graph arrays must be rank-2")
    if not (len(pocket_nodes) == len(pocket_aux) == len(pocket_coords)):
        raise ValueError("pocket node arrays are not aligned")
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_path": str(paths["manifest"]),
        "ligand_nodes": ligand_nodes,
        "ligand_edges": ligand_edges,
        "ligand_edge_types": ligand_edge_types,
        "ligand_node_offsets": ligand_index["node_offset"].to_numpy(dtype=np.int64)[:drug_count],
        "ligand_node_counts": ligand_index["node_count"].to_numpy(dtype=np.int64)[:drug_count],
        "ligand_edge_offsets": ligand_index["edge_offset"].to_numpy(dtype=np.int64)[:drug_count],
        "ligand_edge_counts": ligand_index["edge_count"].to_numpy(dtype=np.int64)[:drug_count],
        "ligand_available": _boolean_array(ligand_index["graph_available"].iloc[:drug_count]),
        "pocket_nodes": pocket_nodes,
        "pocket_aux": pocket_aux,
        "pocket_coords": pocket_coords,
        "pocket_offsets": pocket_index["node_offset"].to_numpy(dtype=np.int64)[:target_count],
        "pocket_counts": pocket_index["node_count"].to_numpy(dtype=np.int64)[:target_count],
        "pocket_available": _boolean_array(pocket_index["graph_available"].iloc[:target_count]),
        "atom_dim": int(ligand_nodes.shape[1]),
        "pocket_dim": int(pocket_nodes.shape[1]),
        "pocket_aux_dim": int(pocket_aux.shape[1]),
    }


def local_graph_batch(
    store: dict[str, object] | None,
    drug_index: np.ndarray,
    target_index: np.ndarray,
    device: torch.device,
    pocket_mean: np.ndarray | None = None,
    pocket_std: np.ndarray | None = None,
) -> dict[str, torch.Tensor | None]:
    if store is None:
        return {
            "ligand_atom_features": None, "ligand_atom_mask": None,
            "ligand_bond_type": None, "pocket_residue_features": None,
            "pocket_residue_aux": None, "pocket_residue_mask": None,
            "pocket_distance_bin": None, "local_pair_mask": None,
        }
    batch_size = len(drug_index)
    ligand_counts = np.asarray(store["ligand_node_counts"])[drug_index]
    pocket_counts = np.asarray(store["pocket_counts"])[target_index]
    max_atoms = max(int(ligand_counts.max(initial=0)), 1)
    max_residues = max(int(pocket_counts.max(initial=0)), 1)
    atom_dim = int(store["atom_dim"])
    pocket_dim = int(store["pocket_dim"])
    pocket_aux_dim = int(store["pocket_aux_dim"])
    atom = np.zeros((batch_size, max_atoms, atom_dim), dtype=np.float32)
    atom_mask = np.zeros((batch_size, max_atoms), dtype=np.float32)
    bond = np.zeros((batch_size, max_atoms, max_atoms), dtype=np.uint8)
    pocket = np.zeros((batch_size, max_residues, pocket_dim), dtype=np.float32)
    pocket_aux = np.zeros((batch_size, max_residues, pocket_aux_dim), dtype=np.float32)
    pocket_mask = np.zeros((batch_size, max_residues), dtype=np.float32)
    distance_bin = np.zeros((batch_size, max_residues, max_residues), dtype=np.uint8)
    ligand_available = np.asarray(store["ligand_available"])[drug_index]
    pocket_available = np.asarray(store["pocket_available"])[target_index]
    pair_available = ligand_available & pocket_available
    ligand_nodes = store["ligand_nodes"]
    ligand_edges = store["ligand_edges"]
    ligand_edge_types = store["ligand_edge_types"]
    pocket_nodes = store["pocket_nodes"]
    pocket_aux_nodes = store["pocket_aux"]
    pocket_coords = store["pocket_coords"]
    for row, (drug_id, target_id) in enumerate(zip(drug_index, target_index)):
        if not pair_available[row]:
            continue
        atom_count = int(ligand_counts[row])
        atom_start = int(np.asarray(store["ligand_node_offsets"])[drug_id])
        atom[row, :atom_count] = np.asarray(
            ligand_nodes[atom_start : atom_start + atom_count], dtype=np.float32
        )
        atom_mask[row, :atom_count] = 1.0
        edge_count = int(np.asarray(store["ligand_edge_counts"])[drug_id])
        edge_start = int(np.asarray(store["ligand_edge_offsets"])[drug_id])
        if edge_count:
            edge = np.asarray(
                ligand_edges[edge_start : edge_start + edge_count], dtype=np.int64
            )
            kinds = np.asarray(
                ligand_edge_types[edge_start : edge_start + edge_count], dtype=np.uint8
            )
            bond[row, edge[:, 0], edge[:, 1]] = kinds
        residue_count = int(pocket_counts[row])
        residue_start = int(np.asarray(store["pocket_offsets"])[target_id])
        values = np.asarray(
            pocket_nodes[residue_start : residue_start + residue_count], dtype=np.float32
        )
        if pocket_mean is not None and pocket_std is not None:
            values = (values - pocket_mean) / pocket_std
        pocket[row, :residue_count] = values
        pocket_aux[row, :residue_count] = np.asarray(
            pocket_aux_nodes[residue_start : residue_start + residue_count], dtype=np.float32
        )
        pocket_mask[row, :residue_count] = 1.0
        coords = np.asarray(
            pocket_coords[residue_start : residue_start + residue_count], dtype=np.float32
        )
        delta = coords[:, None, :] - coords[None, :, :]
        distance = np.sqrt(np.square(delta).sum(axis=2))
        bins = np.zeros((residue_count, residue_count), dtype=np.uint8)
        bins[(distance > 0) & (distance <= 4.0)] = 1
        bins[(distance > 4.0) & (distance <= 8.0)] = 2
        bins[(distance > 8.0) & (distance <= 12.0)] = 3
        bins[(distance > 12.0) & (distance <= 16.0)] = 4
        distance_bin[row, :residue_count, :residue_count] = bins
    return {
        "ligand_atom_features": torch.from_numpy(atom).to(device),
        "ligand_atom_mask": torch.from_numpy(atom_mask).to(device),
        "ligand_bond_type": torch.from_numpy(bond).to(device=device, dtype=torch.long),
        "pocket_residue_features": torch.from_numpy(pocket).to(device),
        "pocket_residue_aux": torch.from_numpy(pocket_aux).to(device),
        "pocket_residue_mask": torch.from_numpy(pocket_mask).to(device),
        "pocket_distance_bin": torch.from_numpy(distance_bin).to(device=device, dtype=torch.long),
        "local_pair_mask": torch.from_numpy(pair_available.astype(np.float32)).to(device),
    }


def local_pocket_normalization(
    store: dict[str, object] | None,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if store is None:
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    width = int(store["pocket_dim"])
    total = 0
    summed = np.zeros(width, dtype=np.float64)
    squared = np.zeros(width, dtype=np.float64)
    available = np.asarray(store["pocket_available"])
    offsets = np.asarray(store["pocket_offsets"])
    counts = np.asarray(store["pocket_counts"])
    nodes = store["pocket_nodes"]
    for target_index in np.unique(target_indices):
        if not available[target_index]:
            continue
        start = int(offsets[target_index])
        count = int(counts[target_index])
        value = np.asarray(nodes[start : start + count], dtype=np.float32)
        summed += value.sum(axis=0, dtype=np.float64)
        squared += np.square(value, dtype=np.float32).sum(axis=0, dtype=np.float64)
        total += count
    if total < 1:
        return np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    mean = summed / total
    variance = np.maximum(squared / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def load_drug_aux_availability(
    path: Path | None,
    drug_count: int,
) -> tuple[np.ndarray, str | None]:
    """Load a feature-aligned availability mask for an optional drug branch."""

    if path is None:
        return np.ones(drug_count, dtype=np.float32), None
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, low_memory=False).sort_values("drug_feature_index")
    if len(table) < drug_count:
        raise ValueError("drug auxiliary index does not cover every drug feature index")
    indices = table["drug_feature_index"].to_numpy(dtype=np.int64)[:drug_count]
    if not np.array_equal(indices, np.arange(drug_count, dtype=np.int64)):
        raise ValueError("drug auxiliary index is not dense and aligned")
    column = "dtiam_bermol_available" if "dtiam_bermol_available" in table.columns else "feature_available"
    if column not in table.columns:
        raise ValueError("drug auxiliary index has no availability column")
    values = pd.Series(table[column].iloc[:drug_count]).fillna(False).astype(bool).to_numpy(dtype=np.float32)
    return values, str(path)


def load_target_token_features(
    features_path: str | None,
    index_path: str | None,
    target_count: int,
    expected_dim: int | None,
    pocket_mask_path: str | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
    str | None,
    str | None,
]:
    """Load a concatenated residue-token feature store and dense target index."""

    if not features_path:
        return None, None, None, None, None, None, None
    feature_path = Path(features_path)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not index_path:
        raise ValueError("--target-token-index is required with --target-token-features")
    index_file = Path(index_path)
    if not index_file.is_file():
        raise FileNotFoundError(index_file)
    features = np.load(feature_path, mmap_mode="r")
    if features.ndim != 2:
        raise ValueError("target token features must be a rank-2 concatenated matrix")
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"target token width {features.shape[1]} != expected {expected_dim}"
        )
    index = pd.read_csv(index_file, low_memory=False).sort_values(
        "target_feature_index"
    ).reset_index(drop=True)
    required = {"target_feature_index", "token_offset", "token_length"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"target token index is missing columns: {sorted(missing)}")
    if len(index) < target_count:
        raise ValueError("target token index does not cover every target feature index")
    dense = index["target_feature_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(dense[:target_count], np.arange(target_count, dtype=np.int64)):
        raise ValueError("target token index must be dense and aligned to target_feature_index")
    offsets = index["token_offset"].to_numpy(dtype=np.int64)[:target_count]
    lengths = index["token_length"].to_numpy(dtype=np.int64)[:target_count]
    if (offsets < 0).any() or (lengths <= 0).any() or (offsets + lengths > features.shape[0]).any():
        raise ValueError("target token offsets/lengths exceed the feature matrix")
    pocket_mask = None
    pocket_source = None
    if pocket_mask_path:
        pocket_path = Path(pocket_mask_path)
        if not pocket_path.is_file():
            raise FileNotFoundError(pocket_path)
        pocket_mask = np.load(pocket_path, mmap_mode="r")
        if pocket_mask.ndim != 1 or pocket_mask.shape[0] != features.shape[0]:
            raise ValueError("target pocket mask must be a rank-1 vector aligned to token rows")
        pocket_mask = np.asarray(pocket_mask, dtype=np.float32)
        pocket_source = str(pocket_path)
    return features, offsets, lengths, pocket_mask, str(feature_path), str(index_file), pocket_source


def target_token_normalization(
    features: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    target_indices: np.ndarray,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only token mean/std with bounded per-target windows."""

    if max_length < 1:
        raise ValueError("target token max length must be positive")
    width = int(features.shape[1])
    total = 0
    summed = np.zeros(width, dtype=np.float64)
    squared = np.zeros(width, dtype=np.float64)
    for target_index in np.unique(target_indices):
        length = min(int(lengths[target_index]), max_length)
        values = np.asarray(
            features[int(offsets[target_index]) : int(offsets[target_index]) + length],
            dtype=np.float32,
        )
        summed += values.sum(axis=0, dtype=np.float64)
        squared += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
        total += length
    if total < 1:
        return np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    mean = summed / total
    variance = np.maximum(squared / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def build_target_token_cache(
    features: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    max_length: int,
    pocket_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize bounded normalized token tensors for fast repeated batches."""

    if max_length < 1:
        raise ValueError("target token max length must be positive")
    target_count = int(len(offsets))
    width = int(features.shape[1])
    cache = np.zeros((target_count, max_length, width), dtype=np.float32)
    mask = np.zeros((target_count, max_length), dtype=np.float32)
    for target_index in range(target_count):
        length = min(int(lengths[target_index]), max_length)
        start = int(offsets[target_index])
        values = np.asarray(features[start : start + length], dtype=np.float32)
        cache[target_index, :length] = (values - mean) / std
        if pocket_mask is None:
            mask[target_index, :length] = 1.0
        else:
            mask[target_index, :length] = np.asarray(
                pocket_mask[start : start + length], dtype=np.float32
            ).clip(0.0, 1.0)
    return cache, mask


def prepare_arrays(
    data: pd.DataFrame,
    train_positions: np.ndarray,
    target_features: np.ndarray,
    target_aux_features: np.ndarray | None = None,
    target_token_features: np.ndarray | None = None,
    target_token_offsets: np.ndarray | None = None,
    target_token_lengths: np.ndarray | None = None,
    target_token_max_len: int = 1022,
    drug_aux_features: np.ndarray | None = None,
    target_extra_features: np.ndarray | None = None,
) -> dict[str, object]:
    # Build categorical vocabularies from fitting rows only.  An explicit UNK
    # bucket keeps strict target-cold/family-cold evaluation well-defined.
    families = sorted(data.iloc[train_positions]["target_assay_family"].astype(str).unique())
    if "__UNK__" not in families:
        families.append("__UNK__")
    family_lookup = {name: index for index, name in enumerate(families)}
    family_index = (
        data["target_assay_family"].astype(str).map(family_lookup).fillna(family_lookup["__UNK__"])
        .to_numpy(dtype=np.int64)
    )
    train_target_indices = np.unique(data.iloc[train_positions]["target_feature_index"].to_numpy(dtype=np.int64))
    if drug_aux_features is not None:
        if "drug_feature_index" not in data.columns:
            raise ValueError("drug auxiliary features require drug_feature_index in the pair table")
        train_drug_indices = np.unique(data.iloc[train_positions]["drug_feature_index"].to_numpy(dtype=np.int64))
        train_drug_aux_features = np.asarray(drug_aux_features[train_drug_indices], dtype=np.float32)
        drug_aux_mean = train_drug_aux_features.mean(axis=0).astype(np.float32)
        drug_aux_std = train_drug_aux_features.std(axis=0).astype(np.float32)
        drug_aux_std[drug_aux_std < 1e-6] = 1.0
    else:
        drug_aux_mean = np.zeros(0, dtype=np.float32)
        drug_aux_std = np.ones(0, dtype=np.float32)
    train_target_features = np.asarray(target_features[train_target_indices], dtype=np.float32)
    target_mean = train_target_features.mean(axis=0).astype(np.float32)
    target_std = train_target_features.std(axis=0).astype(np.float32)
    target_std[target_std < 1e-6] = 1.0
    if target_aux_features is not None:
        train_aux_features = np.asarray(target_aux_features[train_target_indices], dtype=np.float32)
        target_aux_mean = train_aux_features.mean(axis=0).astype(np.float32)
        target_aux_std = train_aux_features.std(axis=0).astype(np.float32)
        target_aux_std[target_aux_std < 1e-6] = 1.0
    else:
        target_aux_mean = np.zeros(0, dtype=np.float32)
        target_aux_std = np.ones(0, dtype=np.float32)
    if target_extra_features is not None:
        train_extra_features = np.asarray(
            target_extra_features[train_target_indices], dtype=np.float32
        )
        target_extra_mean = train_extra_features.mean(axis=0).astype(np.float32)
        target_extra_std = train_extra_features.std(axis=0).astype(np.float32)
        target_extra_std[target_extra_std < 1e-6] = 1.0
    else:
        target_extra_mean = np.zeros(0, dtype=np.float32)
        target_extra_std = np.ones(0, dtype=np.float32)
    if target_token_features is not None:
        if target_token_offsets is None or target_token_lengths is None:
            raise ValueError("target token offsets and lengths are required")
        target_token_mean, target_token_std = target_token_normalization(
            target_token_features,
            target_token_offsets,
            target_token_lengths,
            train_target_indices,
            target_token_max_len,
        )
    else:
        target_token_mean = np.zeros(0, dtype=np.float32)
        target_token_std = np.ones(0, dtype=np.float32)
    conplex = data["conplex_score"].to_numpy(dtype=np.float32)
    conplex_mean = float(conplex[train_positions].mean())
    conplex_std = float(conplex[train_positions].std())
    if conplex_std < 1e-8:
        conplex_std = 1.0
    affinity = pd.to_numeric(data["mean_pchembl"], errors="coerce").to_numpy(dtype=np.float32)
    lower_column = "min_pchembl" if "min_pchembl" in data.columns else "mean_pchembl"
    upper_column = "max_pchembl" if "max_pchembl" in data.columns else "mean_pchembl"
    affinity_lower = pd.to_numeric(data[lower_column], errors="coerce").to_numpy(dtype=np.float32)
    affinity_upper = pd.to_numeric(data[upper_column], errors="coerce").to_numpy(dtype=np.float32)
    affinity_lower = np.where(np.isfinite(affinity_lower), affinity_lower, affinity)
    affinity_upper = np.where(np.isfinite(affinity_upper), affinity_upper, affinity)
    swap = np.isfinite(affinity_lower) & np.isfinite(affinity_upper) & (affinity_lower > affinity_upper)
    affinity_lower, affinity_upper = (
        np.where(swap, affinity_upper, affinity_lower),
        np.where(swap, affinity_lower, affinity_upper),
    )
    finite = np.isfinite(affinity[train_positions])
    if finite.any():
        affinity_mean = float(np.nanmean(affinity[train_positions]))
        affinity_std = float(np.nanstd(affinity[train_positions]))
    else:
        affinity_mean, affinity_std = 0.0, 1.0
    if affinity_std < 1e-8:
        affinity_std = 1.0
    return {
        "families": families,
        "family_index": family_index,
        "drug_aux_mean": drug_aux_mean,
        "drug_aux_std": drug_aux_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_aux_mean": target_aux_mean,
        "target_aux_std": target_aux_std,
        "target_extra_mean": target_extra_mean,
        "target_extra_std": target_extra_std,
        "target_token_mean": target_token_mean,
        "target_token_std": target_token_std,
        "conplex": conplex,
        "conplex_mean": conplex_mean,
        "conplex_std": conplex_std,
        "affinity": affinity,
        "affinity_lower": affinity_lower,
        "affinity_upper": affinity_upper,
        "affinity_mean": affinity_mean,
        "affinity_std": affinity_std,
    }


def grouped_training_batches(
    positions: np.ndarray,
    data: pd.DataFrame,
    batch_size: int,
    seed: int,
    max_rows_per_target: int = 16,
) -> list[np.ndarray]:
    """Create deterministic target-aware minibatches.

    Random row batches often contain only one example for a target, making the
    target-ranking and target-side contrastive objectives inactive.  We first
    make small chunks within each target and then pack chunks into ordinary
    minibatches.  The result retains row-level stochasticity while ensuring
    repeated target entities are present in most batches.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_rows_per_target < 1:
        raise ValueError("max_rows_per_target must be positive")
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    target_values = data.iloc[positions]["target_feature_index"].to_numpy(dtype=np.int64)
    for target in np.unique(target_values):
        group = positions[target_values == target].copy()
        rng.shuffle(group)
        for start in range(0, len(group), max_rows_per_target):
            groups.append(group[start : start + max_rows_per_target])
    rng.shuffle(groups)
    batches: list[np.ndarray] = []
    current: list[np.ndarray] = []
    current_size = 0
    for group in groups:
        if current and current_size + len(group) > batch_size:
            batches.append(np.concatenate(current).astype(np.int64, copy=False))
            current = []
            current_size = 0
        current.append(group)
        current_size += len(group)
    if current:
        batches.append(np.concatenate(current).astype(np.int64, copy=False))
    return batches


def dual_query_training_batches(
    positions: np.ndarray,
    data: pd.DataFrame,
    batch_size: int,
    seed: int,
    max_rows_per_target: int = 16,
    max_rows_per_drug: int = 16,
) -> list[np.ndarray]:
    """Build deterministic batches with both target and drug neighborhoods.

    A target-only sampler makes the within-drug objective silently inactive
    whenever a batch contains at most one row for each ligand.  This sampler
    assigns every row to one of the two query directions for that epoch,
    chunks target-assigned rows by target and drug-assigned rows by drug, then
    mixes both chunk types into ordinary batches.  Every input row is emitted
    exactly once per epoch.

    The sampler is intentionally opt-in: the frozen V2 default remains the
    target-aware sampler, while the dual mode is used for the old-drug query
    ablation and can be promoted only after paired multiseed evidence.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_rows_per_target < 1 or max_rows_per_drug < 1:
        raise ValueError("group caps must be positive")
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return []
    rng = np.random.default_rng(seed)
    target_values = data.iloc[positions]["target_feature_index"].to_numpy(dtype=np.int64)
    drug_values = data.iloc[positions]["drug_feature_index"].to_numpy(dtype=np.int64)

    # Each row is assigned to exactly one query direction for this epoch.
    # Target-assigned rows are chunked by target and drug-assigned rows by
    # drug.  Mixing the resulting chunks gives both objectives repeated-group
    # context while keeping the construction linear in the number of rows.
    direction = rng.integers(0, 2, size=len(positions), dtype=np.int8)
    chunks: list[np.ndarray] = []

    def add_chunks(group_values: np.ndarray, row_mask: np.ndarray, cap: int) -> None:
        selected_positions = positions[row_mask]
        selected_groups = group_values[row_mask]
        for group in np.unique(selected_groups):
            rows = selected_positions[selected_groups == group].copy()
            rng.shuffle(rows)
            for start in range(0, len(rows), cap):
                chunks.append(rows[start : start + cap])

    add_chunks(target_values, direction == 0, max_rows_per_target)
    add_chunks(drug_values, direction == 1, max_rows_per_drug)
    rng.shuffle(chunks)

    batches: list[np.ndarray] = []
    current: list[np.ndarray] = []
    current_size = 0
    for chunk in chunks:
        if current and current_size + len(chunk) > batch_size:
            batches.append(np.concatenate(current).astype(np.int64, copy=False))
            current = []
            current_size = 0
        current.append(chunk)
        current_size += len(chunk)
    if current:
        batches.append(np.concatenate(current).astype(np.int64, copy=False))
    return batches


def validation_selection_value(
    metric_row: dict[str, float | int | None],
    selection_metric: str,
) -> float:
    """Return a validation score aligned with the deployment objective."""

    def finite(name: str) -> float:
        value = metric_row.get(name)
        return float(value) if value is not None and np.isfinite(value) else 0.0

    if selection_metric == "micro_auprc":
        return finite("micro_auprc")
    if selection_metric == "target_macro_auprc":
        return finite("target_macro_auprc")
    if selection_metric == "drug_macro_auprc":
        return finite("drug_macro_auprc")
    if selection_metric == "composite":
        return (
            0.50 * finite("micro_auprc")
            + 0.30 * finite("target_macro_auprc")
            + 0.20 * finite("drug_macro_auprc")
        )
    raise ValueError(f"unknown selection metric: {selection_metric}")


def fast_validation_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    """Validation-only metrics used inside the epoch loop.

    Full deployment metrics include AUROC, recall@k, NDCG and calibration for
    every target and drug.  Recomputing all of them at every epoch makes large
    S1/S2 validation roles unnecessarily slow.  Checkpoint selection only needs
    the three AUPRC terms used by the composite objective; the complete metric
    suite is still computed once on the final test role.
    """

    score = np.asarray(score, dtype=np.float64)
    labels = frame["binary_label"].to_numpy(dtype=np.int8)

    def grouped_ap(column: str) -> float:
        values: list[float] = []
        # Do not copy the full 80+ column pair table at every epoch.  The
        # validation objective only needs the query key, binary label and
        # score; restricting columns avoids a severe CPU/pandas tail that can
        # dominate formal runs while leaving the metric exactly unchanged.
        work = frame[[column, "binary_label"]].copy()
        work["__score"] = score
        for _, part in work.groupby(column, sort=False):
            y = part["binary_label"].to_numpy(dtype=np.int8)
            if y.size == 0 or y.min() == y.max():
                continue
            values.append(float(average_precision_score(y, part["__score"].to_numpy(dtype=np.float64))))
        return float(np.mean(values)) if values else 0.0

    return {
        "micro_auprc": float(average_precision_score(labels, score)),
        "target_macro_auprc": grouped_ap("target_chembl_id"),
        "drug_macro_auprc": grouped_ap("parent_standard_inchi_key"),
    }


def tensor_batch(
    positions: np.ndarray,
    data: pd.DataFrame,
    drug_features: np.ndarray,
    drug_aux_features: np.ndarray | None,
    target_features: np.ndarray,
    target_aux_features: np.ndarray | None,
    target_token_features: np.ndarray | None,
    target_token_offsets: np.ndarray | None,
    target_token_lengths: np.ndarray | None,
    target_token_pocket_mask_features: np.ndarray | None,
    target_token_max_len: int,
    target_token_cache: torch.Tensor | None,
    target_token_mask_cache: torch.Tensor | None,
    structure_features: np.ndarray,
    structure_mask: np.ndarray,
    arrays: dict[str, object],
    device: torch.device,
    observation_column: str | None = None,
    drug_aux_available: np.ndarray | None = None,
    drug_feature_cache: torch.Tensor | None = None,
    drug_aux_feature_cache: torch.Tensor | None = None,
    target_feature_cache: torch.Tensor | None = None,
    target_aux_feature_cache: torch.Tensor | None = None,
    target_extra_feature_cache: torch.Tensor | None = None,
    target_extra_features: np.ndarray | None = None,
    local_graph_store: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    part = data.iloc[positions]
    drug_index = part["drug_feature_index"].to_numpy(dtype=np.int64)
    target_index = part["target_feature_index"].to_numpy(dtype=np.int64)
    if drug_feature_cache is not None:
        drug_ids = torch.from_numpy(drug_index.copy()).to(device=device, dtype=torch.long)
        drug = drug_feature_cache.index_select(0, drug_ids)
    else:
        drug = torch.from_numpy(np.asarray(drug_features[drug_index], dtype=np.float32)).to(device)
    if drug_aux_features is not None:
        if drug_aux_feature_cache is not None:
            drug_ids = torch.from_numpy(drug_index.copy()).to(device=device, dtype=torch.long)
            drug_aux = drug_aux_feature_cache.index_select(0, drug_ids)
        else:
            drug_aux_np = np.asarray(drug_aux_features[drug_index], dtype=np.float32)
            drug_aux_np = (drug_aux_np - arrays["drug_aux_mean"]) / arrays["drug_aux_std"]
            drug_aux = torch.from_numpy(drug_aux_np).to(device)
        if drug_aux_available is None:
            drug_aux_mask_np = np.ones(len(positions), dtype=np.float32)
        else:
            drug_aux_mask_np = np.asarray(drug_aux_available[drug_index], dtype=np.float32)
        drug_aux_mask = torch.from_numpy(drug_aux_mask_np).to(device)
    else:
        drug_aux = None
        drug_aux_mask = None
    if target_feature_cache is not None:
        target_ids = torch.from_numpy(target_index.copy()).to(device=device, dtype=torch.long)
        target = target_feature_cache.index_select(0, target_ids)
    else:
        target_np = np.asarray(target_features[target_index], dtype=np.float32)
        target_np = (target_np - arrays["target_mean"]) / arrays["target_std"]
        target = torch.from_numpy(target_np).to(device)
    if target_aux_features is not None:
        if target_aux_feature_cache is not None:
            target_ids = torch.from_numpy(target_index.copy()).to(device=device, dtype=torch.long)
            target_aux = target_aux_feature_cache.index_select(0, target_ids)
        else:
            target_aux_np = np.asarray(target_aux_features[target_index], dtype=np.float32)
            target_aux_np = (target_aux_np - arrays["target_aux_mean"]) / arrays["target_aux_std"]
            target_aux = torch.from_numpy(target_aux_np).to(device)
    else:
        target_aux = None
    if target_extra_features is not None:
        if target_extra_feature_cache is not None:
            target_ids = torch.from_numpy(target_index.copy()).to(
                device=device, dtype=torch.long
            )
            target_extra = target_extra_feature_cache.index_select(0, target_ids)
        else:
            target_extra_np = np.asarray(
                target_extra_features[target_index], dtype=np.float32
            )
            target_extra_np = (
                target_extra_np - arrays["target_extra_mean"]
            ) / arrays["target_extra_std"]
            target_extra = torch.from_numpy(target_extra_np).to(device)
    else:
        target_extra = None
    if target_token_features is not None:
        if target_token_offsets is None or target_token_lengths is None:
            raise ValueError("target token offsets and lengths are required")
        if target_token_cache is not None and target_token_mask_cache is not None:
            target_ids = torch.as_tensor(
                target_index.copy(), device=device, dtype=torch.long
            )
            target_tokens = target_token_cache.index_select(0, target_ids)
            target_token_mask = target_token_mask_cache.index_select(0, target_ids)
        else:
            clipped_lengths = np.minimum(
                target_token_lengths[target_index], int(target_token_max_len)
            ).astype(np.int64)
            max_length = int(clipped_lengths.max())
            token_np = np.zeros(
                (len(positions), max_length, target_token_features.shape[1]), dtype=np.float32
            )
            token_mask_np = np.zeros((len(positions), max_length), dtype=np.float32)
            token_mean = np.asarray(arrays["target_token_mean"], dtype=np.float32)
            token_std = np.asarray(arrays["target_token_std"], dtype=np.float32)
            for row, target_id in enumerate(target_index):
                length = int(clipped_lengths[row])
                start = int(target_token_offsets[target_id])
                values = np.asarray(target_token_features[start : start + length], dtype=np.float32)
                token_np[row, :length] = (values - token_mean) / token_std
                if target_token_pocket_mask_features is None:
                    token_mask_np[row, :length] = 1.0
                else:
                    token_mask_np[row, :length] = np.asarray(
                        target_token_pocket_mask_features[start : start + length], dtype=np.float32
                    ).clip(0.0, 1.0)
            target_tokens = torch.from_numpy(token_np).to(device)
            target_token_mask = torch.from_numpy(token_mask_np).to(device)
    else:
        target_tokens = None
        target_token_mask = None
    family = torch.from_numpy(np.asarray(arrays["family_index"])[positions]).to(device)
    conplex_np = (np.asarray(arrays["conplex"])[positions] - arrays["conplex_mean"]) / arrays["conplex_std"]
    affinity_np = (np.asarray(arrays["affinity"])[positions] - arrays["affinity_mean"]) / arrays["affinity_std"]
    lower_raw = np.asarray(arrays["affinity_lower"])[positions]
    upper_raw = np.asarray(arrays["affinity_upper"])[positions]
    lower_np = np.where(
        np.isfinite(lower_raw),
        (lower_raw - arrays["affinity_mean"]) / arrays["affinity_std"],
        -np.inf,
    ).astype(np.float32)
    upper_np = np.where(
        np.isfinite(upper_raw),
        (upper_raw - arrays["affinity_mean"]) / arrays["affinity_std"],
        np.inf,
    ).astype(np.float32)
    structure_np = np.asarray(structure_features[positions], dtype=np.float32)
    if structure_np.shape[1] > 0:
        structure_mean = arrays.get("structure_mean", np.zeros(structure_np.shape[1], dtype=np.float32))
        structure_std = arrays.get("structure_std", np.ones(structure_np.shape[1], dtype=np.float32))
        structure_np = (structure_np - structure_mean) / structure_std
    result = {
        "drug": drug,
        "drug_aux": drug_aux,
        "drug_aux_mask": drug_aux_mask,
        "target": target,
        "target_aux": target_aux,
        "target_extra": target_extra,
        "target_tokens": target_tokens,
        "target_token_mask": target_token_mask,
        "family": family,
        "conplex": torch.from_numpy(conplex_np.astype(np.float32)).to(device),
        "structure": torch.from_numpy(structure_np.astype(np.float32)).to(device),
        "structure_mask": torch.from_numpy(structure_mask[positions]).to(device),
        "labels": torch.from_numpy(
            part["binary_label"].to_numpy(dtype=np.float32).copy()
        ).to(device),
        "affinity": torch.from_numpy(affinity_np.astype(np.float32)).to(device),
        "affinity_lower": torch.from_numpy(lower_np).to(device),
        "affinity_upper": torch.from_numpy(upper_np).to(device),
        "target_group": torch.from_numpy(target_index.copy()).to(device),
        "drug_group": torch.from_numpy(drug_index.copy()).to(device),
        "affinity_observed": torch.from_numpy(np.isfinite(lower_raw) | np.isfinite(upper_raw)).to(device),
    }
    result.update(
        local_graph_batch(
            local_graph_store,
            drug_index,
            target_index,
            device,
            np.asarray(arrays.get("local_pocket_mean"), dtype=np.float32)
            if local_graph_store is not None else None,
            np.asarray(arrays.get("local_pocket_std"), dtype=np.float32)
            if local_graph_store is not None else None,
        )
    )
    if observation_column:
        if observation_column not in part.columns:
            raise ValueError(f"observation column not found: {observation_column}")
        observed = pd.to_numeric(part[observation_column], errors="raise").to_numpy(dtype=np.float32)
        result["observed_labels"] = torch.from_numpy(observed).to(device)
    return result


def predict(
    model: RoutedInteractionRankerV2,
    positions: np.ndarray,
    data: pd.DataFrame,
    drug_features: np.ndarray,
    drug_aux_features: np.ndarray | None,
    target_features: np.ndarray,
    target_aux_features: np.ndarray | None,
    target_token_features: np.ndarray | None,
    target_token_offsets: np.ndarray | None,
    target_token_lengths: np.ndarray | None,
    target_token_pocket_mask_features: np.ndarray | None,
    target_token_max_len: int,
    target_token_cache: torch.Tensor | None,
    target_token_mask_cache: torch.Tensor | None,
    structure_features: np.ndarray,
    structure_mask: np.ndarray,
    arrays: dict[str, object],
    device: torch.device,
    batch_size: int,
    drug_aux_available: np.ndarray | None = None,
    drug_feature_cache: torch.Tensor | None = None,
    drug_aux_feature_cache: torch.Tensor | None = None,
    target_feature_cache: torch.Tensor | None = None,
    target_aux_feature_cache: torch.Tensor | None = None,
    target_extra_feature_cache: torch.Tensor | None = None,
    target_extra_features: np.ndarray | None = None,
    local_graph_store: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            values = tensor_batch(
                batch, data, drug_features, drug_aux_features, target_features,
                target_aux_features,
                target_token_features, target_token_offsets, target_token_lengths,
                target_token_pocket_mask_features,
                target_token_max_len, target_token_cache, target_token_mask_cache,
                structure_features, structure_mask, arrays, device,
                drug_aux_available=drug_aux_available,
                drug_feature_cache=drug_feature_cache,
                drug_aux_feature_cache=drug_aux_feature_cache,
                target_feature_cache=target_feature_cache,
                target_aux_feature_cache=target_aux_feature_cache,
                target_extra_feature_cache=target_extra_feature_cache,
                target_extra_features=target_extra_features,
                local_graph_store=local_graph_store,
            )
            result = model(
                values["drug"], values["target"], values["family"], values["conplex"],
                values["structure"], values["structure_mask"],
                drug_aux=values["drug_aux"], drug_aux_mask=values["drug_aux_mask"],
                target_aux=values["target_aux"],
                target_extra=values["target_extra"],
                target_tokens=values["target_tokens"],
                target_token_mask=values["target_token_mask"],
                ligand_atom_features=values["ligand_atom_features"],
                ligand_atom_mask=values["ligand_atom_mask"],
                ligand_bond_type=values["ligand_bond_type"],
                pocket_residue_features=values["pocket_residue_features"],
                pocket_residue_aux=values["pocket_residue_aux"],
                pocket_residue_mask=values["pocket_residue_mask"],
                pocket_distance_bin=values["pocket_distance_bin"],
                local_pair_mask=values["local_pair_mask"],
            )
            for key, value in result.items():
                if value.ndim == 1:
                    outputs.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    feature_audit = json.loads(FEATURE_AUDIT.read_text())
    if feature_audit.get("status") != "PASS":
        raise RuntimeError("feature store audit must pass")
    data = pd.read_csv(PAIRS, low_memory=False)
    drug_features = np.load(MORGAN, mmap_mode="r")
    drug_aux_features = None
    drug_aux_source = None
    drug_aux_availability = None
    drug_aux_index_source = None
    if getattr(args, "drug_aux_features", None):
        drug_aux_source_path = Path(args.drug_aux_features)
        if not drug_aux_source_path.is_file():
            raise FileNotFoundError(drug_aux_source_path)
        drug_aux_features = np.load(drug_aux_source_path, mmap_mode="r")
        if drug_aux_features.ndim != 2 or drug_aux_features.shape[0] < len(drug_features):
            raise ValueError(
                "drug auxiliary feature matrix must be rank-2 and cover every drug feature index"
            )
        if args.drug_aux_dim is not None and drug_aux_features.shape[1] != args.drug_aux_dim:
            raise ValueError(
                f"drug auxiliary width {drug_aux_features.shape[1]} != expected {args.drug_aux_dim}"
            )
        if not np.isfinite(np.asarray(drug_aux_features[: len(drug_features)])).all():
            raise ValueError("drug auxiliary feature matrix contains non-finite values")
        drug_aux_source = str(drug_aux_source_path)
        configured_index = getattr(args, "drug_aux_index", None)
        if configured_index:
            drug_aux_index_path: Path | None = Path(configured_index)
        elif drug_aux_source_path.resolve() == DEFAULT_DRUG_AUX.resolve():
            # Preserve the historical BERMol behavior without silently applying
            # its availability mask to an unrelated molecular representation.
            drug_aux_index_path = DRUG_AUX_INDEX
        else:
            drug_aux_index_path = None
        drug_aux_availability, drug_aux_index_source = load_drug_aux_availability(
            drug_aux_index_path, len(drug_features)
        )
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux_features = None
    target_aux_source = None
    if args.target_aux_features:
        target_aux_source_path = Path(args.target_aux_features)
        if not target_aux_source_path.is_file():
            raise FileNotFoundError(target_aux_source_path)
        target_aux_features = np.load(target_aux_source_path, mmap_mode="r")
        if target_aux_features.ndim != 2 or target_aux_features.shape[0] < len(target_features):
            raise ValueError(
                "target auxiliary feature matrix must be rank-2 and cover every target feature index"
            )
        if args.target_aux_dim is not None and target_aux_features.shape[1] != args.target_aux_dim:
            raise ValueError(
                f"target auxiliary width {target_aux_features.shape[1]} != expected {args.target_aux_dim}"
            )
        if not np.isfinite(np.asarray(target_aux_features[: len(target_features)])).all():
            raise ValueError("target auxiliary feature matrix contains non-finite values")
        target_aux_source = str(target_aux_source_path)
    target_extra_features = None
    target_extra_source = None
    if getattr(args, "target_extra_features", None):
        target_extra_source_path = Path(args.target_extra_features)
        if not target_extra_source_path.is_file():
            raise FileNotFoundError(target_extra_source_path)
        target_extra_features = np.load(target_extra_source_path, mmap_mode="r")
        if (
            target_extra_features.ndim != 2
            or target_extra_features.shape[0] < len(target_features)
        ):
            raise ValueError(
                "target extra feature matrix must be rank-2 and cover every target feature index"
            )
        expected_extra_dim = getattr(args, "target_extra_dim", None)
        if (
            expected_extra_dim is not None
            and target_extra_features.shape[1] != expected_extra_dim
        ):
            raise ValueError(
                f"target extra width {target_extra_features.shape[1]} "
                f"!= expected {expected_extra_dim}"
            )
        if not np.isfinite(
            np.asarray(target_extra_features[: len(target_features)])
        ).all():
            raise ValueError("target extra feature matrix contains non-finite values")
        target_extra_source = str(target_extra_source_path)
    target_token_features = None
    target_token_offsets = None
    target_token_lengths = None
    target_token_pocket_mask_features = None
    target_token_source = None
    target_token_index_source = None
    target_token_pocket_mask_source = None
    target_token_dim = None
    target_token_max_len = int(getattr(args, "target_token_max_len", 1022))
    if getattr(args, "target_token_features", None):
        (
            target_token_features,
            target_token_offsets,
            target_token_lengths,
            target_token_pocket_mask_features,
            target_token_source,
            target_token_index_source,
            target_token_pocket_mask_source,
        ) = load_target_token_features(
            args.target_token_features,
            getattr(args, "target_token_index", None),
            len(target_features),
            getattr(args, "target_token_dim", None),
            getattr(args, "target_token_pocket_mask", None),
        )
        target_token_dim = int(target_token_features.shape[1])
    local_graph_store = load_local_graph_features(
        getattr(args, "local_graph_features", None),
        len(drug_features),
        len(target_features),
    )
    effective_batch_size = int(args.batch_size)
    effective_inference_batch_size = int(args.inference_batch_size)
    if target_token_features is not None:
        # Raw token tensors are [batch, length, 1280]. Keep a conservative
        # bound so long-protein batches remain auditable on a single GPU.
        effective_batch_size = min(effective_batch_size, 512)
        effective_inference_batch_size = min(effective_inference_batch_size, 512)
    if local_graph_store is not None:
        # Dense atom/residue attention is bounded per minibatch; this cap keeps
        # the worst-case A×P attention tensor predictable on a single GPU.
        effective_batch_size = min(effective_batch_size, 256)
        effective_inference_batch_size = min(effective_inference_batch_size, 256)
    fold = -1 if args.protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"} else args.fold
    masks = split_masks(data, args.protocol, fold)
    available = data["drug_feature_available"].to_numpy(dtype=bool)
    train_positions, valid_positions, test_positions = [
        np.flatnonzero(masks[name] & available) for name in ["train", "valid", "test"]
    ]
    if args.max_rows > 0:
        train_positions = train_positions[: args.max_rows]
        valid_positions = valid_positions[: args.max_rows]
        test_positions = test_positions[: args.max_rows]
    if any(len(value) == 0 for value in [train_positions, valid_positions, test_positions]):
        raise RuntimeError("train/valid/test split is empty after --max-rows truncation")
    valid_class_count = int(data.iloc[valid_positions]["binary_label"].nunique())
    test_class_count = int(data.iloc[test_positions]["binary_label"].nunique())
    if valid_class_count < 2 or test_class_count < 2:
        raise RuntimeError(
            "validation and test roles must contain both classes; "
            f"got valid_classes={valid_class_count}, test_classes={test_class_count}. "
            "Increase --max-rows or use the complete frozen role."
        )
    arrays = prepare_arrays(
        data,
        train_positions,
        target_features,
        target_aux_features,
        target_token_features,
        target_token_offsets,
        target_token_lengths,
        target_token_max_len,
        drug_aux_features=drug_aux_features,
        target_extra_features=target_extra_features,
    )
    local_pocket_mean, local_pocket_std = local_pocket_normalization(
        local_graph_store,
        data.iloc[train_positions]["target_feature_index"].to_numpy(dtype=np.int64),
    )
    arrays["local_pocket_mean"] = local_pocket_mean
    arrays["local_pocket_std"] = local_pocket_std
    structure_features, structure_mask, structure_columns, structure_path = load_structure_features(
        args.structure_features, data["calibration_pair_id"], args.structure_dim
    )
    structure_group_dims = infer_structure_group_dims(
        structure_columns, getattr(args, "structure_encoder", "flat")
    )
    structure_train_rows = train_positions[structure_mask[train_positions] > 0]
    if structure_features.shape[1] > 0 and len(structure_train_rows) > 0:
        structure_mean = np.asarray(
            structure_features[structure_train_rows].mean(axis=0), dtype=np.float32
        )
        structure_std = np.asarray(
            structure_features[structure_train_rows].std(axis=0), dtype=np.float32
        )
        structure_std[structure_std < 1e-6] = 1.0
    else:
        structure_mean = np.zeros(structure_features.shape[1], dtype=np.float32)
        structure_std = np.ones(structure_features.shape[1], dtype=np.float32)
    arrays["structure_mean"] = structure_mean
    arrays["structure_std"] = structure_std
    config = ODTIV2Config(
        drug_aux_input_dim=(0 if drug_aux_features is None else int(drug_aux_features.shape[1])),
        drug_aux_gate_init_bias=args.drug_aux_gate_init_bias,
        structure_input_dim=structure_features.shape[1],
        interaction_mode=getattr(args, "interaction_mode", "legacy_full"),
        interaction_rank=int(getattr(args, "interaction_rank", 48)),
        film_scale=float(getattr(args, "film_scale", 0.10)),
        structure_group_dims=structure_group_dims,
        enhanced_structure_interaction=bool(
            getattr(args, "enhanced_structure_interaction", False)
        ),
        structure_gate_init_bias=getattr(args, "structure_gate_init_bias", None),
        local_pair_atom_input_dim=(
            0 if local_graph_store is None else int(local_graph_store["atom_dim"])
        ),
        local_pair_pocket_input_dim=(
            0 if local_graph_store is None else int(local_graph_store["pocket_dim"])
        ),
        local_pair_pocket_aux_dim=(
            0 if local_graph_store is None else int(local_graph_store["pocket_aux_dim"])
        ),
        local_pair_hidden_dim=int(getattr(args, "local_pair_hidden_dim", 96)),
        local_pair_layers=int(getattr(args, "local_pair_layers", 2)),
        local_pair_heads=int(getattr(args, "local_pair_heads", 4)),
        local_pair_gate_init_bias=getattr(args, "local_pair_gate_init_bias", -4.0),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        expert_count=args.experts,
        dropout=args.dropout,
        rank_weight=args.rank_weight,
        drug_rank_weight=args.drug_rank_weight,
        expert_balance_weight=args.expert_balance_weight,
        listwise_weight=args.listwise_weight,
        affinity_weight=args.affinity_weight,
        observation_weight=args.observation_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        rank_max_pairs=args.rank_max_pairs,
        target_aux_input_dim=(0 if target_aux_features is None else int(target_aux_features.shape[1])),
        target_aux_gate_init_bias=args.target_aux_gate_init_bias,
        target_extra_input_dim=(
            0 if target_extra_features is None else int(target_extra_features.shape[1])
        ),
        target_extra_gate_init_bias=getattr(args, "target_extra_gate_init_bias", -4.0),
        target_token_input_dim=(0 if target_token_features is None else int(target_token_dim)),
        target_token_heads=int(getattr(args, "target_token_heads", 4)),
        target_token_max_len=target_token_max_len,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    drug_feature_cache = None
    drug_aux_feature_cache = None
    target_feature_cache = None
    target_aux_feature_cache = None
    target_extra_feature_cache = None
    if getattr(args, "cache_dense_features", False):
        # The dense feature store is small enough for a 24--48 GB GPU
        # (Morgan-2048 for the indexed ligands is roughly 0.5 GB in fp32).
        # Keeping it on device removes repeated memmap -> NumPy -> CUDA copies
        # from every batch while preserving the exact feature values.
        drug_feature_cache = torch.from_numpy(
            np.asarray(drug_features, dtype=np.float32)
        ).to(device)
        if drug_aux_features is not None:
            drug_aux_feature_cache = torch.from_numpy(
                ((np.asarray(drug_aux_features, dtype=np.float32) - arrays["drug_aux_mean"])
                 / arrays["drug_aux_std"])
            ).to(device)
        target_feature_cache = torch.from_numpy(
            ((np.asarray(target_features, dtype=np.float32) - arrays["target_mean"])
             / arrays["target_std"])
        ).to(device)
        if target_aux_features is not None:
            target_aux_feature_cache = torch.from_numpy(
                ((np.asarray(target_aux_features, dtype=np.float32) - arrays["target_aux_mean"])
                 / arrays["target_aux_std"])
            ).to(device)
        if target_extra_features is not None:
            target_extra_feature_cache = torch.from_numpy(
                (
                    (
                        np.asarray(target_extra_features, dtype=np.float32)
                        - arrays["target_extra_mean"]
                    )
                    / arrays["target_extra_std"]
                )
            ).to(device)
    target_token_cache = None
    target_token_mask_cache = None
    if target_token_features is not None:
        cache_np, cache_mask_np = build_target_token_cache(
            target_token_features,
            target_token_offsets,
            target_token_lengths,
            np.asarray(arrays["target_token_mean"], dtype=np.float32),
            np.asarray(arrays["target_token_std"], dtype=np.float32),
            target_token_max_len,
            target_token_pocket_mask_features,
        )
        target_token_cache = torch.from_numpy(cache_np).to(device)
        target_token_mask_cache = torch.from_numpy(cache_mask_np).to(device)
        del cache_np, cache_mask_np
    model = RoutedInteractionRankerV2(
        family_count=len(arrays["families"]), config=config, use_conplex=args.use_conplex
    ).to(device)
    init_checkpoint_info = load_partial_checkpoint(
        model, getattr(args, "init_checkpoint", None)
    )
    freeze_base_epochs = int(getattr(args, "freeze_base_epochs", 0))
    if freeze_base_epochs < 0:
        raise ValueError("freeze_base_epochs must be non-negative")
    local_only_trainable = set_local_only_trainable(
        model, freeze_base_epochs > 0
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 20
    )
    best_selection = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    no_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if freeze_base_epochs > 0 and epoch == freeze_base_epochs + 1:
            set_local_only_trainable(model, False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - freeze_base_epochs, 1),
                eta_min=args.learning_rate / 20,
            )
        model.train()
        if args.random_batches:
            order = np.random.permutation(len(train_positions))
            batches = [
                train_positions[order[start : start + effective_batch_size]]
                for start in range(0, len(order), effective_batch_size)
            ]
        else:
            if args.batch_sampler == "dual_query":
                batches = dual_query_training_batches(
                    train_positions,
                    data,
                    effective_batch_size,
                    args.seed + epoch,
                    args.max_rows_per_target,
                    args.max_rows_per_drug,
                )
            else:
                batches = grouped_training_batches(
                    train_positions,
                    data,
                    effective_batch_size,
                    args.seed + epoch,
                    args.max_rows_per_target,
                )
        loss_sum = 0.0
        steps = 0
        component_sums: dict[str, float] = {}
        for positions in batches:
            values = tensor_batch(
                positions, data, drug_features, drug_aux_features, target_features,
                target_aux_features,
                target_token_features, target_token_offsets, target_token_lengths,
                target_token_pocket_mask_features,
                target_token_max_len, target_token_cache, target_token_mask_cache,
                structure_features, structure_mask, arrays, device,
                args.observation_column,
                drug_aux_available=drug_aux_availability,
                drug_feature_cache=drug_feature_cache,
                drug_aux_feature_cache=drug_aux_feature_cache,
                target_feature_cache=target_feature_cache,
                target_aux_feature_cache=target_aux_feature_cache,
                target_extra_feature_cache=target_extra_feature_cache,
                target_extra_features=target_extra_features,
                local_graph_store=local_graph_store,
            )
            optimizer.zero_grad(set_to_none=True)
            result = model(
                values["drug"], values["target"], values["family"], values["conplex"],
                values["structure"], values["structure_mask"],
                drug_aux=values["drug_aux"], drug_aux_mask=values["drug_aux_mask"],
                target_aux=values["target_aux"],
                target_extra=values["target_extra"],
                target_tokens=values["target_tokens"],
                target_token_mask=values["target_token_mask"],
                ligand_atom_features=values["ligand_atom_features"],
                ligand_atom_mask=values["ligand_atom_mask"],
                ligand_bond_type=values["ligand_bond_type"],
                pocket_residue_features=values["pocket_residue_features"],
                pocket_residue_aux=values["pocket_residue_aux"],
                pocket_residue_mask=values["pocket_residue_mask"],
                pocket_distance_bin=values["pocket_distance_bin"],
                local_pair_mask=values["local_pair_mask"],
            )
            losses = odti_v2_loss(
                result,
                values["labels"],
                values["target_group"],
                values["drug_group"],
                affinity_lower=values["affinity_lower"],
                affinity_upper=values["affinity_upper"],
                affinity_observed=values["affinity_observed"],
                observed_labels=values.get("observed_labels"),
                config=config,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_sum += float(losses["total"].detach())
            for name, value in losses.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
            steps += 1
        scheduler.step()
        validation = predict(
            model, valid_positions, data, drug_features, drug_aux_features, target_features,
            target_aux_features,
            target_token_features, target_token_offsets, target_token_lengths,
            target_token_pocket_mask_features,
            target_token_max_len, target_token_cache, target_token_mask_cache,
            structure_features, structure_mask, arrays, device,
            effective_inference_batch_size,
            drug_aux_available=drug_aux_availability,
            drug_feature_cache=drug_feature_cache,
            drug_aux_feature_cache=drug_aux_feature_cache,
            target_feature_cache=target_feature_cache,
            target_aux_feature_cache=target_aux_feature_cache,
            target_extra_feature_cache=target_extra_feature_cache,
            target_extra_features=target_extra_features,
            local_graph_store=local_graph_store,
        )
        validation_frame = data.iloc[valid_positions].reset_index(drop=True)
        validation_probability = sigmoid(validation["final_logit"])
        validation_metrics = fast_validation_metrics(validation_frame, validation_probability)
        selection_value = validation_selection_value(validation_metrics, args.selection_metric)
        history.append({
            "epoch": epoch,
            "loss": loss_sum / max(steps, 1),
            **{
                f"loss_{name}": value / max(steps, 1)
                for name, value in component_sums.items()
            },
            "valid_micro_auprc": float(validation_metrics["micro_auprc"]),
            "valid_target_macro_auprc": float(validation_metrics["target_macro_auprc"] or 0.0),
            "valid_drug_macro_auprc": float(validation_metrics["drug_macro_auprc"] or 0.0),
            "valid_selection_value": float(selection_value),
            "structure_rows": float(structure_mask[train_positions].sum()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if selection_value > best_selection + args.min_delta:
            best_selection = selection_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation = predict(
        model, valid_positions, data, drug_features, drug_aux_features, target_features,
        target_aux_features,
        target_token_features, target_token_offsets, target_token_lengths,
        target_token_pocket_mask_features,
        target_token_max_len, target_token_cache, target_token_mask_cache,
        structure_features, structure_mask, arrays, device,
        effective_inference_batch_size,
        drug_aux_available=drug_aux_availability,
        drug_feature_cache=drug_feature_cache,
        drug_aux_feature_cache=drug_aux_feature_cache,
        target_feature_cache=target_feature_cache,
        target_aux_feature_cache=target_aux_feature_cache,
        target_extra_feature_cache=target_extra_feature_cache,
        target_extra_features=target_extra_features,
        local_graph_store=local_graph_store,
    )
    validation_labels = data.iloc[valid_positions]["binary_label"].to_numpy(dtype=np.int8)
    temperature = temperature_scale(validation["final_logit"], validation_labels)
    test = predict(
        model, test_positions, data, drug_features, drug_aux_features, target_features,
        target_aux_features,
        target_token_features, target_token_offsets, target_token_lengths,
        target_token_pocket_mask_features,
        target_token_max_len, target_token_cache, target_token_mask_cache,
        structure_features, structure_mask, arrays, device,
        effective_inference_batch_size,
        drug_aux_available=drug_aux_availability,
        drug_feature_cache=drug_feature_cache,
        drug_aux_feature_cache=drug_aux_feature_cache,
        target_feature_cache=target_feature_cache,
        target_aux_feature_cache=target_aux_feature_cache,
        target_extra_feature_cache=target_extra_feature_cache,
        target_extra_features=target_extra_features,
        local_graph_store=local_graph_store,
    )
    test_frame = data.iloc[test_positions].reset_index(drop=True)
    test_probability = sigmoid(test["final_logit"] / temperature)
    result_metrics = metrics(test_frame, test_probability)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.protocol}__fold_{fold}__seed_{args.seed}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "BEST_MODEL_V2.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "RoutedInteractionRankerV2",
            "config": config.__dict__,
            "init_checkpoint": init_checkpoint_info,
            "freeze_base_epochs": freeze_base_epochs,
            "local_only_trainable_parameters": sorted(local_only_trainable),
            "families": arrays["families"],
            "drug_aux_source": drug_aux_source,
            "drug_aux_sha256": (
                sha256(Path(drug_aux_source)) if drug_aux_source is not None else None
            ),
            "drug_aux_index_source": drug_aux_index_source,
            "drug_aux_index_sha256": (
                sha256(Path(drug_aux_index_source))
                if drug_aux_index_source is not None else None
            ),
            "drug_aux_input_dim": int(config.drug_aux_input_dim),
            "target_aux_source": target_aux_source,
            "target_aux_sha256": (
                sha256(Path(target_aux_source)) if target_aux_source is not None else None
            ),
            "target_aux_input_dim": int(config.target_aux_input_dim),
            "target_extra_source": target_extra_source,
            "target_extra_sha256": (
                sha256(Path(target_extra_source))
                if target_extra_source is not None
                else None
            ),
            "target_extra_input_dim": int(config.target_extra_input_dim),
            "target_token_source": target_token_source,
            "target_token_index_source": target_token_index_source,
            "target_token_pocket_mask_source": target_token_pocket_mask_source,
            "target_token_sha256": (
                sha256(Path(target_token_source)) if target_token_source is not None else None
            ),
            "target_token_index_sha256": (
                sha256(Path(target_token_index_source))
                if target_token_index_source is not None else None
            ),
            "target_token_pocket_mask_sha256": (
                sha256(Path(target_token_pocket_mask_source))
                if target_token_pocket_mask_source is not None else None
            ),
            "target_token_input_dim": int(config.target_token_input_dim),
            "target_token_max_len": int(config.target_token_max_len),
            "structure_columns": structure_columns,
            "structure_source": structure_path,
            "normalization": {
                "drug_aux_mean": arrays["drug_aux_mean"],
                "drug_aux_std": arrays["drug_aux_std"],
                "target_mean": arrays["target_mean"],
                "target_std": arrays["target_std"],
                "target_aux_mean": arrays["target_aux_mean"],
                "target_aux_std": arrays["target_aux_std"],
                "target_extra_mean": arrays["target_extra_mean"],
                "target_extra_std": arrays["target_extra_std"],
                "target_token_mean": arrays["target_token_mean"],
                "target_token_std": arrays["target_token_std"],
                "conplex_mean": arrays["conplex_mean"],
                "conplex_std": arrays["conplex_std"],
                "affinity_mean": arrays["affinity_mean"],
                "affinity_std": arrays["affinity_std"],
                "structure_mean": arrays["structure_mean"],
                "structure_std": arrays["structure_std"],
            },
            "temperature": temperature,
            "selection_metric": args.selection_metric,
            "best_validation_selection_value": best_selection,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(run_dir / "TRAINING_HISTORY_V2.csv", index=False)
    predictions = test_frame[[
        "calibration_pair_id", "sequence_key", "target_chembl_id", "primary_gene",
        "target_assay_family", "parent_standard_inchi_key", "parent_molecule_chembl_id",
        "binary_label", "mean_pchembl", "conplex_score",
    ]].copy()
    for key, value in test.items():
        predictions[f"v2_{key}"] = value
    predictions["v2_probability_calibrated"] = test_probability
    prediction_path = run_dir / "TEST_PREDICTIONS_V2.csv.gz"
    predictions.to_csv(prediction_path, index=False)
    checks = {
        "feature_store_pass": feature_audit["status"] == "PASS",
        "train_valid_test_nonempty": all(len(x) > 0 for x in [train_positions, valid_positions, test_positions]),
        "test_has_both_classes": test_frame["binary_label"].nunique() == 2,
        "best_checkpoint_validation_selected": best_epoch > 0,
        "temperature_validation_only": np.isfinite(temperature) and temperature > 0,
        "test_predictions_finite": np.isfinite(test_probability).all(),
        "test_predictions_bounded": ((test_probability >= 0) & (test_probability <= 1)).all(),
        "structure_fallback_rows_exact": bool(
            np.allclose(
                test["final_logit"][structure_mask[test_positions] == 0],
                test["base_logit"][structure_mask[test_positions] == 0],
            )
        ),
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": "BIOMASTER_ODTI_ROUTED_INTERACTION_RANKER_V2",
        "protocol": args.protocol,
        "fold": fold,
        "seed": args.seed,
        "device": str(device),
        "split_counts": {
            "train": int(len(train_positions)),
            "valid": int(len(valid_positions)),
            "test": int(len(test_positions)),
            "quarantined_total_pairs": int((~available).sum()),
        },
        "structure": {
            "source": structure_path,
            "feature_columns": structure_columns,
            "feature_dim": int(structure_features.shape[1]),
            "encoder": getattr(args, "structure_encoder", "flat"),
            "group_dims": list(config.structure_group_dims),
            "enhanced_pair_interaction": bool(
                config.enhanced_structure_interaction
            ),
            "train_rows_masked_on": int((structure_mask[train_positions] > 0).sum()),
            "valid_rows_masked_on": int((structure_mask[valid_positions] > 0).sum()),
            "test_rows_masked_on": int((structure_mask[test_positions] > 0).sum()),
        },
        "drug_auxiliary": {
            "source": drug_aux_source,
            "sha256": (
                sha256(Path(drug_aux_source)) if drug_aux_source is not None else None
            ),
            "index_source": drug_aux_index_source,
            "index_sha256": (
                sha256(Path(drug_aux_index_source))
                if drug_aux_index_source is not None else None
            ),
            "feature_dim": int(config.drug_aux_input_dim),
            "available_drugs": int(drug_aux_availability.sum()) if drug_aux_availability is not None else 0,
            "enabled": bool(drug_aux_features is not None),
        },
        "target_auxiliary": {
            "source": target_aux_source,
            "sha256": (
                sha256(Path(target_aux_source)) if target_aux_source is not None else None
            ),
            "feature_dim": int(config.target_aux_input_dim),
            "enabled": bool(target_aux_features is not None),
        },
        "target_extra_auxiliary": {
            "source": target_extra_source,
            "sha256": (
                sha256(Path(target_extra_source))
                if target_extra_source is not None
                else None
            ),
            "feature_dim": int(config.target_extra_input_dim),
            "enabled": bool(target_extra_features is not None),
        },
        "target_token_auxiliary": {
            "source": target_token_source,
            "index_source": target_token_index_source,
            "pocket_mask_source": target_token_pocket_mask_source,
            "sha256": (
                sha256(Path(target_token_source)) if target_token_source is not None else None
            ),
            "index_sha256": (
                sha256(Path(target_token_index_source))
                if target_token_index_source is not None else None
            ),
            "pocket_mask_sha256": (
                sha256(Path(target_token_pocket_mask_source))
                if target_token_pocket_mask_source is not None else None
            ),
            "feature_dim": int(config.target_token_input_dim),
            "max_len": int(config.target_token_max_len),
            "enabled": bool(target_token_features is not None),
        },
        "local_graph_auxiliary": {
            "source": getattr(args, "local_graph_features", None),
            "enabled": bool(local_graph_store is not None),
            "manifest": (
                local_graph_store.get("manifest_path") if local_graph_store is not None else None
            ),
            "atom_dim": int(config.local_pair_atom_input_dim),
            "pocket_dim": int(config.local_pair_pocket_input_dim),
            "pocket_aux_dim": int(config.local_pair_pocket_aux_dim),
            "available_ligands": (
                int(np.asarray(local_graph_store["ligand_available"]).sum())
                if local_graph_store is not None else 0
            ),
            "available_targets": (
                int(np.asarray(local_graph_store["pocket_available"]).sum())
                if local_graph_store is not None else 0
            ),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_selection_value": best_selection,
            "selection_metric": args.selection_metric,
            "epochs_completed": len(history),
            "temperature": temperature,
            "requested_batch_size": int(args.batch_size),
            "effective_batch_size": effective_batch_size,
            "requested_inference_batch_size": int(args.inference_batch_size),
                "effective_inference_batch_size": effective_inference_batch_size,
                "cache_dense_features": bool(getattr(args, "cache_dense_features", False)),
            "batch_sampler": "random" if args.random_batches else args.batch_sampler,
            "max_rows_per_target": int(args.max_rows_per_target),
            "max_rows_per_drug": int(args.max_rows_per_drug),
            "init_checkpoint": init_checkpoint_info,
            "freeze_base_epochs": freeze_base_epochs,
            "local_only_parameter_count": int(
                sum(
                    parameter.numel()
                    for name, parameter in model.named_parameters()
                    if name.startswith("local_")
                )
            ),
        },
        "interaction": {
            "mode": config.interaction_mode,
            "rank": int(config.interaction_rank),
            "film_scale": float(config.film_scale),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "local_pair_enabled": bool(config.local_pair_atom_input_dim > 0),
            "local_pair_hidden_dim": int(config.local_pair_hidden_dim),
            "local_pair_layers": int(config.local_pair_layers),
            "local_pair_heads": int(config.local_pair_heads),
        },
        "test_metrics": result_metrics,
        "checks": {key: bool(value) for key, value in checks.items()},
        "artifacts": {
            "checkpoint_sha256": sha256(checkpoint),
            "predictions_sha256": sha256(prediction_path),
        },
        "claim_status": "V2_IMPLEMENTATION_SMOKE_OR_EXPERIMENTAL; NOT_SOTA_UNTIL_FROZEN_MULTISEED_EXTERNAL_GATES",
    }
    checks_json = {key: bool(value) for key, value in checks.items()}
    summary_path = run_dir / "RUN_SUMMARY_V2.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks_json, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.12)
    parser.add_argument(
        "--drug-rank-weight",
        type=float,
        default=0.0,
        help="optional within-drug pairwise ranking weight for old-drug retrieval",
    )
    parser.add_argument(
        "--expert-balance-weight",
        type=float,
        default=0.0,
        help="optional anti-collapse regularizer for routed expert usage",
    )
    parser.add_argument("--listwise-weight", type=float, default=0.0)
    parser.add_argument("--rank-max-pairs", type=int, default=4096)
    parser.add_argument("--affinity-weight", type=float, default=0.06)
    parser.add_argument("--observation-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--structure-features")
    parser.add_argument("--structure-dim", type=int)
    parser.add_argument(
        "--structure-encoder",
        choices=["flat", "grouped"],
        default="flat",
        help="flat legacy encoder or semantic quality/chemistry/geometry/consensus pooling",
    )
    parser.add_argument(
        "--enhanced-structure-interaction",
        action="store_true",
        help="use drug-target-structure pair, tri-linear and symmetric difference features",
    )
    parser.add_argument(
        "--structure-gate-init-bias",
        type=float,
        default=None,
        help="optional conservative initialization for the structure residual gate",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=["legacy_full", "low_rank_film"],
        default="legacy_full",
    )
    parser.add_argument("--interaction-rank", type=int, default=48)
    parser.add_argument("--film-scale", type=float, default=0.10)
    parser.add_argument(
        "--local-graph-features",
        default=None,
        help="label-free concatenated ligand/pocket graph feature directory",
    )
    parser.add_argument("--local-pair-hidden-dim", type=int, default=96)
    parser.add_argument("--local-pair-layers", type=int, default=2)
    parser.add_argument("--local-pair-heads", type=int, default=4)
    parser.add_argument("--local-pair-gate-init-bias", type=float, default=-4.0)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="optional shape-compatible checkpoint used to warm-start additive branches",
    )
    parser.add_argument(
        "--freeze-base-epochs",
        type=int,
        default=0,
        help="train only local_* parameters for the first N epochs, then unfreeze",
    )
    parser.add_argument(
        "--drug-aux-features",
        default=None,
        help="optional drug-aligned molecular pretraining feature matrix (.npy)",
    )
    parser.add_argument("--drug-aux-dim", type=int, default=None)
    parser.add_argument(
        "--drug-aux-index",
        default=None,
        help=(
            "optional dense drug auxiliary index CSV containing drug_feature_index "
            "and feature_available; required when an auxiliary store has missing rows"
        ),
    )
    parser.add_argument(
        "--drug-aux-gate-init-bias",
        type=float,
        default=None,
        help="optional molecular residual gate bias; unset preserves zero-input behavior",
    )
    parser.add_argument(
        "--target-aux-features",
        default=None,
        help="optional target-aligned auxiliary feature matrix (.npy), e.g. audited ESM2-650M features",
    )
    parser.add_argument("--target-aux-dim", type=int, default=None)
    parser.add_argument(
        "--target-aux-gate-init-bias",
        type=float,
        default=None,
        help="optional pooled target-aux residual gate bias; unset preserves audited V2 initialization",
    )
    parser.add_argument(
        "--target-extra-features",
        default=None,
        help=(
            "optional second target-aligned PLM matrix added as a separately gated "
            "residual after --target-aux-features"
        ),
    )
    parser.add_argument("--target-extra-dim", type=int, default=None)
    parser.add_argument(
        "--target-extra-gate-init-bias",
        type=float,
        default=-4.0,
        help="near-zero residual initialization for a newly screened target PLM",
    )
    parser.add_argument("--target-token-features")
    parser.add_argument("--target-token-index")
    parser.add_argument("--target-token-pocket-mask")
    parser.add_argument("--target-token-dim", type=int, default=None)
    parser.add_argument("--target-token-heads", type=int, default=4)
    parser.add_argument("--target-token-max-len", type=int, default=1022)
    parser.add_argument("--use-conplex", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=["composite", "micro_auprc", "target_macro_auprc", "drug_macro_auprc"],
        default="composite",
    )
    parser.add_argument(
        "--random-batches",
        action="store_true",
        help="disable the default target-aware minibatch sampler",
    )
    parser.add_argument(
        "--batch-sampler",
        choices=["target", "dual_query"],
        default="target",
        help="target-aware default or dual target+drug query neighborhoods",
    )
    parser.add_argument("--max-rows-per-target", type=int, default=16)
    parser.add_argument("--max-rows-per-drug", type=int, default=16)
    parser.add_argument(
        "--observation-column",
        help="optional binary observation/recorded-pair column for the propensity head",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cache-dense-features",
        action="store_true",
        help="cache dense Morgan/target features on device to reduce formal-run I/O overhead",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed audit for the label-free local graph feature store."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.store)
    manifest_path = root / "LOCAL_GRAPH_FEATURE_MANIFEST_V1.json"
    checks: dict[str, bool] = {}
    failures: list[str] = []
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checks["manifest_pass"] = manifest.get("status") == "PASS"
    checks["label_free"] = manifest.get("label_dependency") == "NONE"
    checks["no_cross_pose"] = manifest.get("pair_pose_available") is False
    required = [
        "LIGAND_ATOM_FEATURES_FLOAT16_V1.npy",
        "LIGAND_EDGE_INDEX_INT16_V1.npy",
        "LIGAND_EDGE_TYPE_UINT8_V1.npy",
        "LIGAND_GRAPH_INDEX_V1.csv.gz",
        "POCKET_ESM2_RESIDUE_FLOAT16_V1.npy",
        "POCKET_RESIDUE_AUX_FLOAT16_V1.npy",
        "POCKET_CA_COORD_FLOAT32_V1.npy",
        "POCKET_GRAPH_INDEX_V1.csv.gz",
    ]
    checks["all_artifacts_present"] = all((root / name).is_file() for name in required)
    if checks["all_artifacts_present"]:
        ligand_nodes = np.load(root / required[0], mmap_mode="r")
        ligand_edges = np.load(root / required[1], mmap_mode="r")
        ligand_edge_types = np.load(root / required[2], mmap_mode="r")
        pocket_nodes = np.load(root / required[4], mmap_mode="r")
        pocket_aux = np.load(root / required[5], mmap_mode="r")
        pocket_coords = np.load(root / required[6], mmap_mode="r")
        ligand_index = pd.read_csv(root / required[3], low_memory=False)
        pocket_index = pd.read_csv(root / required[7], low_memory=False)
        checks["ligand_shapes"] = (
            ligand_nodes.ndim == 2 and ligand_edges.ndim == 2 and ligand_edges.shape[1] == 2
            and ligand_edge_types.shape == (len(ligand_edges),)
        )
        checks["pocket_shapes"] = (
            pocket_nodes.ndim == 2 and pocket_aux.ndim == 2 and pocket_coords.shape == (len(pocket_nodes), 3)
            and len(pocket_aux) == len(pocket_nodes)
        )
        checks["finite_arrays"] = all(
            np.isfinite(np.asarray(value, dtype=np.float32)).all()
            for value in [ligand_nodes, pocket_nodes, pocket_aux, pocket_coords]
        )
        checks["dense_ligand_index"] = np.array_equal(
            ligand_index["drug_feature_index"].to_numpy(dtype=np.int64),
            np.arange(len(ligand_index), dtype=np.int64),
        )
        checks["dense_pocket_index"] = np.array_equal(
            pocket_index["target_feature_index"].to_numpy(dtype=np.int64),
            np.arange(len(pocket_index), dtype=np.int64),
        )
        checks["offsets_in_bounds"] = (
            (ligand_index["node_offset"] + ligand_index["node_count"] <= len(ligand_nodes)).all()
            and (ligand_index["edge_offset"] + ligand_index["edge_count"] <= len(ligand_edges)).all()
            and (pocket_index["node_offset"] + pocket_index["node_count"] <= len(pocket_nodes)).all()
        )
        checks["available_nonempty"] = (
            (ligand_index.loc[ligand_index.graph_available.astype(bool), "node_count"] > 0).all()
            and (pocket_index.loc[pocket_index.graph_available.astype(bool), "node_count"] > 0).all()
        )
    else:
        for name in ["ligand_shapes", "pocket_shapes", "finite_arrays", "dense_ligand_index", "dense_pocket_index", "offsets_in_bounds", "available_nonempty"]:
            checks[name] = False
    checks = {name: bool(value) for name, value in checks.items()}
    for name, value in checks.items():
        if not value:
            failures.append(name)
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "manifest_sha256": sha256(manifest_path),
        "store": str(root),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

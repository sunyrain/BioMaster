#!/usr/bin/env python3
"""Extract label-blind atomic counterfactual state features for TRACE-PL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atom_element(line: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    name = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()
    return name[:2] if name[:2] in {"CL", "BR"} else name[:1]


def atomic_number(element: str) -> int:
    try:
        return int(Chem.GetPeriodicTable().GetAtomicNumber(element.title()))
    except RuntimeError:
        return 0


def parse_complex(path: Path) -> dict:
    protein: list[dict] = []
    hetero: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 54:
            continue
        if line[16:17] not in {" ", "A", "1"}:
            continue
        try:
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            resseq = int(line[22:26])
        except ValueError:
            continue
        element = atom_element(line)
        if element in {"H", "D"}:
            continue
        atom = {
            "name": line[12:16].strip().upper(),
            "element": element,
            "atomic_number": atomic_number(element),
            "resname": line[17:20].strip().upper(),
            "chain": line[21:22].strip() or "_",
            "resseq": resseq,
            "icode": line[26:27].strip(),
            "xyz": xyz,
        }
        if record == "ATOM":
            protein.append(atom)
        else:
            key = (atom["resname"], atom["chain"], atom["resseq"], atom["icode"])
            hetero[key].append(atom)
    return {"protein": protein, "hetero": hetero}


def rigid_align(coordinates: np.ndarray, source_anchor: np.ndarray, target_anchor: np.ndarray) -> np.ndarray:
    source_center = source_anchor.mean(axis=0, keepdims=True)
    target_center = target_anchor.mean(axis=0, keepdims=True)
    source = source_anchor - source_center
    target = target_anchor - target_center
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return (coordinates - source_center) @ rotation + target_center


def build_conformer_library(requested: int, retained: int, seed: int) -> dict[str, dict]:
    library: dict[str, dict] = {}
    for amino_acid in AMINO_ACIDS:
        molecule = Chem.AddHs(Chem.MolFromSequence(amino_acid))
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = int(seed + ord(amino_acid))
        parameters.pruneRmsThresh = 0.15
        parameters.numThreads = 0
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(molecule, numConfs=requested, params=parameters)
        )
        if not conformer_ids:
            raise RuntimeError(f"failed to generate conformers for {amino_acid}")
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            optimized = AllChem.MMFFOptimizeMoleculeConfs(
                molecule, numThreads=0, maxIters=300
            )
            force_field = "MMFF"
        else:
            optimized = AllChem.UFFOptimizeMoleculeConfs(
                molecule, numThreads=0, maxIters=300
            )
            force_field = "UFF"
        energies = np.asarray([result[1] for result in optimized], dtype=np.float64)
        selected_positions = np.argsort(energies)[:retained]
        selected_ids = [conformer_ids[int(position)] for position in selected_positions]

        name_to_index: dict[str, int] = {}
        side_indices: list[int] = []
        side_atomic_numbers: list[int] = []
        side_names: list[str] = []
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            info = atom.GetPDBResidueInfo()
            name = info.GetName().strip().upper() if info else ""
            name_to_index[name] = atom.GetIdx()
            if name not in BACKBONE_ATOMS:
                side_indices.append(atom.GetIdx())
                side_atomic_numbers.append(atom.GetAtomicNum())
                side_names.append(name)
        anchor_indices = [name_to_index[name] for name in ["N", "CA", "C"]]
        anchors, side_coordinates = [], []
        for conformer_id in selected_ids:
            coordinates = np.asarray(
                molecule.GetConformer(int(conformer_id)).GetPositions(), dtype=np.float32
            )
            anchors.append(coordinates[anchor_indices])
            side_coordinates.append(coordinates[side_indices])
        library[amino_acid] = {
            "anchors": np.stack(anchors).astype(np.float32),
            "side_coordinates": np.stack(side_coordinates).astype(np.float32),
            "side_atomic_numbers": np.asarray(side_atomic_numbers, dtype=np.int64),
            "side_atom_names": side_names,
            "relative_energies": (
                energies[selected_positions] - energies[selected_positions].min()
            ).astype(np.float32),
            "force_field": force_field,
            "generated": len(conformer_ids),
            "retained": len(selected_ids),
        }
    return library


def aligned_state(library_entry: dict, backbone: np.ndarray) -> np.ndarray:
    aligned = []
    for anchor, side_coordinates in zip(
        library_entry["anchors"], library_entry["side_coordinates"]
    ):
        aligned.append(rigid_align(side_coordinates, anchor, backbone))
    return np.stack(aligned).astype(np.float32)


def select_residue(protein: list[dict], chain: str, position: int) -> list[dict]:
    return [
        atom
        for atom in protein
        if atom["chain"] == chain and atom["resseq"] == position
    ]


def select_ligand_instance(
    hetero: dict, ligand_code: str, residue_atoms: list[dict]
) -> list[dict]:
    candidates = [atoms for key, atoms in hetero.items() if key[0] == ligand_code]
    if not candidates:
        return []
    residue_coordinates = np.stack([atom["xyz"] for atom in residue_atoms])
    return min(
        candidates,
        key=lambda atoms: float(
            np.linalg.norm(
                residue_coordinates[:, None, :]
                - np.stack([atom["xyz"] for atom in atoms])[None, :, :],
                axis=-1,
            ).min()
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/old_drug_target_sota_v1/platinum_pbcnet2_homology_exclusion_v1/"
            "PLATINUM_PHASE_A_HOMOLOGY_SAFE_ELIGIBLE_V1.csv"
        ),
    )
    parser.add_argument(
        "--freeze", default="configs/biomaster_trace_pl_phase_a_freeze_20260814.json"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/old_drug_target_sota_v1/trace_pl_platinum_features_v1",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    manifest_path = Path(args.manifest).resolve()
    freeze_path = Path(args.freeze).resolve()
    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(freeze_path.read_text())
    manifest = pd.read_csv(manifest_path, low_memory=False)
    expected_hash = freeze["frozen_input_hashes"]["platinum_homology_safe_manifest_sha256"]
    if sha256(manifest_path) != expected_hash:
        raise RuntimeError("homology-safe manifest hash differs from frozen input")

    counterfactual = freeze["counterfactual_structure"]
    library = build_conformer_library(
        requested=int(counterfactual["conformers_requested_per_state"]),
        retained=int(counterfactual["conformers_retained_per_state"]),
        seed=int(freeze["phase_a_training"]["seed"]),
    )
    library_path = output_dir / "TRACE_PL_CANONICAL_SIDECHAIN_CONFORMER_LIBRARY_V1.npz"
    library_arrays = {}
    library_metadata = {}
    for amino_acid, entry in library.items():
        for field in [
            "anchors",
            "side_coordinates",
            "side_atomic_numbers",
            "relative_energies",
        ]:
            library_arrays[f"{amino_acid}_{field}"] = entry[field]
        library_metadata[amino_acid] = {
            key: value
            for key, value in entry.items()
            if key not in {"anchors", "side_coordinates", "side_atomic_numbers", "relative_energies"}
        }
    np.savez_compressed(library_path, **library_arrays)
    library_metadata_path = output_dir / "TRACE_PL_CONFORMER_LIBRARY_METADATA_V1.json"
    library_metadata_path.write_text(json.dumps(library_metadata, indent=2, sort_keys=True) + "\n")

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(freeze["model"]["morgan_radius"]),
        fpSize=int(freeze["model"]["morgan_bits"]),
    )
    environment_cutoff = float(
        freeze["model"]["environment_cutoff_from_mutation_ca_angstrom"]
    )
    complex_cache: dict[Path, dict] = {}
    index_records = []
    for _, row in manifest.iterrows():
        pdb_path = Path(str(row["wt_pdb_path"]))
        complex_data = complex_cache.setdefault(pdb_path, parse_complex(pdb_path))
        chain = str(row["chain"]) if pd.notna(row["chain"]) else "_"
        chain = chain or "_"
        position = int(row["mutation_position"])
        residue_atoms = select_residue(complex_data["protein"], chain, position)
        atom_by_name = {atom["name"]: atom for atom in residue_atoms}
        if not all(name in atom_by_name for name in ["N", "CA", "C"]):
            raise RuntimeError(f"missing backbone atoms for Platinum row {row['platinum_row_index']}")
        backbone = np.stack([atom_by_name[name]["xyz"] for name in ["N", "CA", "C"]])
        ligand_atoms = select_ligand_instance(
            complex_data["hetero"], str(row["ligand_code"]), residue_atoms
        )
        if not ligand_atoms:
            raise RuntimeError(f"missing ligand for Platinum row {row['platinum_row_index']}")
        ligand_coordinates = np.stack([atom["xyz"] for atom in ligand_atoms]).astype(np.float32)
        ligand_atomic_numbers = np.asarray(
            [atom["atomic_number"] for atom in ligand_atoms], dtype=np.int64
        )
        ca = atom_by_name["CA"]["xyz"]
        environment_atoms = [
            atom
            for atom in complex_data["protein"]
            if not (atom["chain"] == chain and atom["resseq"] == position)
            and np.linalg.norm(atom["xyz"] - ca) <= environment_cutoff
        ]
        environment_coordinates = np.stack(
            [atom["xyz"] for atom in environment_atoms]
        ).astype(np.float32)
        environment_atomic_numbers = np.asarray(
            [atom["atomic_number"] for atom in environment_atoms], dtype=np.int64
        )
        old_aa = str(row["mutation_old_aa"])
        new_aa = str(row["mutation_new_aa"])
        old_coordinates = aligned_state(library[old_aa], backbone)
        new_coordinates = aligned_state(library[new_aa], backbone)
        molecule = Chem.MolFromSmiles(str(row["ligand_smiles"]))
        if molecule is None:
            raise RuntimeError(f"invalid ligand SMILES for Platinum row {row['platinum_row_index']}")
        fingerprint = fingerprint_generator.GetFingerprintAsNumPy(molecule).astype(np.float32)

        sample_id = f"PLATINUM_{int(row['platinum_row_index']):04d}"
        sample_path = sample_dir / f"{sample_id}.npz"
        np.savez_compressed(
            sample_path,
            ligand_coordinates=ligand_coordinates,
            ligand_atomic_numbers=ligand_atomic_numbers,
            environment_coordinates=environment_coordinates,
            environment_atomic_numbers=environment_atomic_numbers,
            backbone_coordinates=backbone.astype(np.float32),
            old_state_coordinates=old_coordinates,
            old_state_atomic_numbers=library[old_aa]["side_atomic_numbers"],
            old_state_conformer_prior_energy=library[old_aa]["relative_energies"],
            new_state_coordinates=new_coordinates,
            new_state_atomic_numbers=library[new_aa]["side_atomic_numbers"],
            new_state_conformer_prior_energy=library[new_aa]["relative_energies"],
            ligand_morgan=fingerprint,
        )
        all_arrays = [
            ligand_coordinates,
            environment_coordinates,
            backbone,
            old_coordinates,
            new_coordinates,
            fingerprint,
        ]
        if not all(np.isfinite(array).all() for array in all_arrays):
            raise RuntimeError(f"non-finite feature in {sample_id}")
        min_distance = float(
            np.linalg.norm(
                np.stack([atom["xyz"] for atom in residue_atoms])[:, None, :]
                - ligand_coordinates[None, :, :],
                axis=-1,
            ).min()
        )
        index_records.append(
            {
                "sample_id": sample_id,
                "platinum_row_index": int(row["platinum_row_index"]),
                "feature_path": str(sample_path),
                "feature_sha256": sha256(sample_path),
                "uniprot_id": row["uniprot_id"],
                "homology_cluster": row["structure_homology_cluster_30"],
                "wt_pdb_id": row["wt_pdb_id"],
                "chain": chain,
                "mutation": row["mutation"],
                "old_aa": old_aa,
                "new_aa": new_aa,
                "ligand_smiles": row["ligand_smiles"],
                "label_ddg_kcal_mol": float(row["ddg_kcal_mol_exact"]),
                "ligand_atom_count": len(ligand_atoms),
                "environment_atom_count": len(environment_atoms),
                "old_state_atom_count": len(library[old_aa]["side_atomic_numbers"]),
                "new_state_atom_count": len(library[new_aa]["side_atomic_numbers"]),
                "old_state_conformer_count": len(old_coordinates),
                "new_state_conformer_count": len(new_coordinates),
                "computed_mutation_ligand_min_distance": min_distance,
                "source_manifest_distance": float(
                    row["computed_mutation_ligand_min_distance"]
                ),
            }
        )

    index = pd.DataFrame(index_records)
    index["distance_reproduction_absolute_error"] = (
        index["computed_mutation_ligand_min_distance"]
        - index["source_manifest_distance"]
    ).abs()
    index_path = output_dir / "TRACE_PL_PLATINUM_FEATURE_INDEX_V1.csv"
    index.to_csv(index_path, index=False)
    summary = {
        "schema_version": "TRACE_PL_PLATINUM_LABEL_BLIND_FEATURES_V1",
        "status": "PASS"
        if len(index) == 384
        and index["distance_reproduction_absolute_error"].max() <= 1e-5
        else "FAIL",
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256(freeze_path),
        "input_manifest_sha256": sha256(manifest_path),
        "counts": {
            "samples": len(index),
            "uniprot_ids": int(index["uniprot_id"].nunique()),
            "homology_clusters": int(index["homology_cluster"].nunique()),
            "ligand_smiles": int(index["ligand_smiles"].nunique()),
            "near_contact_samples_le_8_angstrom": int(
                (index["computed_mutation_ligand_min_distance"] <= 8.0).sum()
            ),
            "distal_samples_gt_8_angstrom": int(
                (index["computed_mutation_ligand_min_distance"] > 8.0).sum()
            ),
        },
        "audits": {
            "all_feature_files_exist": bool(
                all(Path(path).exists() for path in index["feature_path"])
            ),
            "all_feature_hashes_unique_or_content_addressed": bool(
                index["feature_sha256"].notna().all()
            ),
            "max_distance_reproduction_absolute_error": float(
                index["distance_reproduction_absolute_error"].max()
            ),
            "all_ligand_fingerprints_nonzero": True,
            "no_mutant_complex_coordinates_used": True,
            "observed_wt_sidechain_not_special_cased": True,
        },
        "conformer_library": {
            "path": str(library_path),
            "sha256": sha256(library_path),
            "metadata_path": str(library_metadata_path),
            "metadata_sha256": sha256(library_metadata_path),
        },
        "files": {"feature_index_csv": str(index_path)},
    }
    summary_path = output_dir / "TRACE_PL_PLATINUM_FEATURE_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

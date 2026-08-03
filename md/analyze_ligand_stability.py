#!/usr/bin/env python3
"""Summarize short protein-ligand MD after protein-backbone alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array, minimize_vectors
from MDAnalysis.lib.mdamath import make_whole


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
    return rotation, translation


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def summary_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "final": float(array[-1]),
    }


def reconstruct_complex(protein, ligand, dimensions) -> None:  # noqa: ANN001
    """Make both fragments whole and place the ligand in the protein's nearest image."""
    make_whole(protein)
    make_whole(ligand)
    displacement = ligand.center_of_geometry() - protein.center_of_geometry()
    nearest = minimize_vectors(displacement, dimensions)
    ligand.positions += nearest - displacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology",
        type=Path,
        help="Generic topology (for example AMBER prmtop or GROMACS tpr)",
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        help="Optional initial coordinates; omit when the topology contains them",
    )
    parser.add_argument("--prmtop", type=Path, help="Legacy alias for --topology")
    parser.add_argument("--inpcrd", type=Path, help="Legacy alias for --coordinates")
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--equilibration-frames", type=int, default=0)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--contact-cutoff-a", type=float, default=4.0)
    parser.add_argument(
        "--disable-pbc-reconstruction",
        action="store_true",
        help="Do not unwrap and center the protein-ligand complex",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = args.topology or args.prmtop
    coordinates = args.coordinates or args.inpcrd
    if topology is None:
        raise ValueError("Provide --topology (or the legacy --prmtop argument)")
    universe_args = [str(topology)]
    if coordinates is not None:
        universe_args.append(str(coordinates))
    universe_args.append(str(args.trajectory))
    universe = mda.Universe(*universe_args)
    backbone = universe.select_atoms("protein and backbone")
    ligand = universe.select_atoms(f"resname {args.ligand_resname} and not name H*")
    protein_heavy = universe.select_atoms("protein and not name H*")
    protein = universe.select_atoms("protein")
    if not len(backbone) or not len(ligand):
        raise ValueError("Protein backbone or ligand selection is empty")

    universe.trajectory[0]
    if not args.disable_pbc_reconstruction:
        reconstruct_complex(protein, ligand, universe.trajectory.ts.dimensions)
    reference_backbone = backbone.positions.copy()
    reference_ligand = ligand.positions.copy()
    initial_distances = distance_array(reference_ligand, protein_heavy.positions)
    initial_contact_residues = {
        protein_heavy[j].resindex
        for _, j in zip(*np.where(initial_distances <= args.contact_cutoff_a), strict=True)
    }

    ligand_rmsd = []
    ligand_com_displacement = []
    backbone_rmsd = []
    contact_fraction = []
    contact_residue_count = []
    minimum_heavy_distance = []
    analyzed_frames = 0
    for frame_index, _ in enumerate(universe.trajectory):
        if frame_index < args.equilibration_frames:
            continue
        if not args.disable_pbc_reconstruction:
            reconstruct_complex(protein, ligand, universe.trajectory.ts.dimensions)
        rotation, translation = kabsch(backbone.positions, reference_backbone)
        aligned_backbone = backbone.positions @ rotation + translation
        aligned_ligand = ligand.positions @ rotation + translation
        aligned_protein = protein_heavy.positions @ rotation + translation
        distances = distance_array(aligned_ligand, aligned_protein)
        contacts = {
            protein_heavy[j].resindex
            for _, j in zip(*np.where(distances <= args.contact_cutoff_a), strict=True)
        }
        ligand_rmsd.append(rmsd(aligned_ligand, reference_ligand))
        ligand_com_displacement.append(
            float(np.linalg.norm(aligned_ligand.mean(axis=0) - reference_ligand.mean(axis=0)))
        )
        backbone_rmsd.append(rmsd(aligned_backbone, reference_backbone))
        contact_residue_count.append(float(len(contacts)))
        union = initial_contact_residues | contacts
        contact_fraction.append(
            float(len(initial_contact_residues & contacts) / len(union)) if union else 1.0
        )
        minimum_heavy_distance.append(float(distances.min()))
        analyzed_frames += 1

    result = {
        "trajectory": str(args.trajectory.resolve()),
        "total_frames": len(universe.trajectory),
        "skipped_equilibration_frames": args.equilibration_frames,
        "analyzed_frames": analyzed_frames,
        "alignment_selection": "protein and backbone",
        "pbc_reconstruction": not args.disable_pbc_reconstruction,
        "ligand_selection": f"resname {args.ligand_resname} and not name H*",
        "initial_contact_residue_count": len(initial_contact_residues),
        "ligand_heavy_atom_rmsd_angstrom": summary_stats(ligand_rmsd),
        "ligand_centroid_displacement_angstrom": summary_stats(ligand_com_displacement),
        "protein_backbone_rmsd_angstrom": summary_stats(backbone_rmsd),
        "contact_residue_jaccard": summary_stats(contact_fraction),
        "contact_residue_count": summary_stats(contact_residue_count),
        "minimum_ligand_protein_heavy_distance_angstrom": summary_stats(minimum_heavy_distance),
        "interpretation_limit": (
            "Short MD can identify immediate instability but cannot establish binding affinity. "
            "The reference pose is a Boltz conditional pose mapped to an experimental receptor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

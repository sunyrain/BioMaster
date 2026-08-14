#!/usr/bin/env python3
"""Build a solvated ff19SB/GAFF2 AMBER protein-ligand system."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from openmm.app import AmberInpcrdFile, AmberPrmtopFile


def bounded_environment() -> dict[str, str]:
    """Keep nested quantum/BLAS work within the queue's declared CPU share."""
    environment = os.environ.copy()
    threads = environment.get("BIOMASTER_MD_PREP_THREADS", "2")
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_MAX_THREADS": threads,
        }
    )
    return environment


def executable(name: str) -> str:
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.is_file():
        return str(candidate)
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Required AmberTools executable is absent: {name}")
    return path


def run(command: list[str], log_path: Path, cwd: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=bounded_environment(),
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}: {command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor-pdb", required=True, type=Path)
    parser.add_argument("--ligand-sdf", required=True, type=Path)
    parser.add_argument("--ligand-net-charge", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--water-padding-a", type=float, default=12.0)
    parser.add_argument("--water-model", choices=["opc", "tip3p"], default="opc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receptor = args.output_dir / "receptor_amber.pdb"
    ligand = args.output_dir / "ligand_aligned.sdf"
    if args.receptor_pdb.resolve() != receptor.resolve():
        shutil.copy2(args.receptor_pdb, receptor)
    if args.ligand_sdf.resolve() != ligand.resolve():
        shutil.copy2(args.ligand_sdf, ligand)

    run(
        [
            executable("antechamber"), "-i", ligand.name, "-fi", "sdf",
            "-o", "ligand_gaff2.mol2", "-fo", "mol2", "-c", "bcc",
            "-at", "gaff2", "-nc", str(args.ligand_net_charge), "-rn", "LIG", "-pf", "y",
        ],
        args.output_dir / "antechamber.log",
        args.output_dir,
    )
    run(
        [
            executable("parmchk2"), "-i", "ligand_gaff2.mol2", "-f", "mol2",
            "-o", "ligand_gaff2.frcmod", "-s", "gaff2",
        ],
        args.output_dir / "parmchk2.log",
        args.output_dir,
    )

    water_source = "leaprc.water.opc" if args.water_model == "opc" else "leaprc.water.tip3p"
    water_box = "OPCBOX" if args.water_model == "opc" else "TIP3PBOX"
    leap_text = f"""source leaprc.protein.ff19SB
source leaprc.gaff2
source {water_source}
loadamberparams ligand_gaff2.frcmod
LIG = loadmol2 ligand_gaff2.mol2
REC = loadpdb receptor_amber.pdb
COM = combine {{REC LIG}}
check COM
solvateoct COM {water_box} {args.water_padding_a:.3f}
addionsrand COM Na+ 0
addionsrand COM Cl- 0
check COM
savepdb COM solvated_complex.pdb
saveamberparm COM complex.prmtop complex.inpcrd
quit
"""
    (args.output_dir / "tleap.in").write_text(leap_text, encoding="utf-8")
    run(
        [executable("tleap"), "-f", "tleap.in"],
        args.output_dir / "tleap.log",
        args.output_dir,
    )

    prmtop = AmberPrmtopFile(str(args.output_dir / "complex.prmtop"))
    inpcrd = AmberInpcrdFile(str(args.output_dir / "complex.inpcrd"))
    ligand_residues = [residue for residue in prmtop.topology.residues() if residue.name == "LIG"]
    audit = {
        "receptor_pdb": str(args.receptor_pdb.resolve()),
        "ligand_sdf": str(args.ligand_sdf.resolve()),
        "ligand_net_charge": args.ligand_net_charge,
        "force_fields": {"protein": "ff19SB", "ligand": "GAFF2/AM1-BCC"},
        "water_model": args.water_model,
        "water_padding_angstrom": args.water_padding_a,
        "atoms": prmtop.topology.getNumAtoms(),
        "residues": prmtop.topology.getNumResidues(),
        "ligand_residues": len(ligand_residues),
        "ligand_atoms": len(list(ligand_residues[0].atoms())) if ligand_residues else 0,
        "periodic_box_present": inpcrd.boxVectors is not None,
    }
    if len(ligand_residues) != 1 or inpcrd.boxVectors is None:
        raise RuntimeError(f"Built system failed topology audit: {audit}")
    (args.output_dir / "system_build_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

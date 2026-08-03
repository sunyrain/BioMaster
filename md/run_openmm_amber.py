#!/usr/bin/env python3
"""Run restrained equilibration and production MD from audited AMBER inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openmm import CustomExternalForce, LangevinMiddleIntegrator, MonteCarloBarostat, Platform
from openmm import unit
from openmm.app import (
    AmberInpcrdFile,
    AmberPrmtopFile,
    CheckpointReporter,
    DCDReporter,
    HBonds,
    PME,
    Simulation,
    StateDataReporter,
)


WATER_AND_IONS = {
    "HOH",
    "WAT",
    "TIP3",
    "NA",
    "CL",
    "K",
    "MG",
    "CA",
    "ZN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenMM CUDA MD from a pre-audited AMBER prmtop/inpcrd pair."
    )
    parser.add_argument("--prmtop", type=Path, required=True)
    parser.add_argument("--inpcrd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--equilibration-ns", type=float, default=1.0)
    parser.add_argument("--production-ns", type=float, default=50.0)
    parser.add_argument("--report-ps", type=float, default=100.0)
    parser.add_argument("--restraint-k", type=float, default=1000.0)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def steps_for_ns(ns: float, timestep_fs: float) -> int:
    return int(round(ns * 1_000_000.0 / timestep_fs))


def interval_for_ps(ps: float, timestep_fs: float) -> int:
    return max(1, int(round(ps * 1000.0 / timestep_fs)))


def add_heavy_atom_restraints(system, topology, positions, force_constant: float):
    force = CustomExternalForce(
        "k*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    force.addGlobalParameter(
        "k", force_constant * unit.kilojoule_per_mole / unit.nanometer**2
    )
    for parameter in ("x0", "y0", "z0"):
        force.addPerParticleParameter(parameter)
    restrained = 0
    for atom in topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        if atom.residue.name.upper() in WATER_AND_IONS:
            continue
        xyz = positions[atom.index].value_in_unit(unit.nanometer)
        force.addParticle(atom.index, xyz)
        restrained += 1
    system.addForce(force)
    return force, restrained


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.prmtop, args.inpcrd):
        if not path.is_file():
            raise FileNotFoundError(path)

    prmtop = AmberPrmtopFile(str(args.prmtop))
    inpcrd = AmberInpcrdFile(str(args.inpcrd))
    if inpcrd.boxVectors is None:
        raise ValueError("Periodic box vectors are required for production MD")

    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5e-4,
    )
    system.addForce(
        MonteCarloBarostat(
            args.pressure_bar * unit.bar,
            args.temperature_k * unit.kelvin,
            25,
        )
    )
    restraint, restrained_atoms = add_heavy_atom_restraints(
        system, prmtop.topology, inpcrd.positions, args.restraint_k
    )
    integrator = LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        args.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    platform = Platform.getPlatformByName("CUDA")
    properties = {
        "DeviceIndex": str(args.device_index),
        "Precision": "mixed",
        "DeterministicForces": "true",
    }
    simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    report_interval = interval_for_ps(args.report_ps, args.timestep_fs)
    resumed = args.resume_checkpoint is not None
    simulation.reporters.extend(
        [
            StateDataReporter(
                str(args.output_dir / "state.csv"),
                report_interval,
                append=resumed,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                separator=",",
            ),
            DCDReporter(
                str(args.output_dir / "trajectory.dcd"),
                report_interval,
                append=resumed,
            ),
            CheckpointReporter(str(args.output_dir / "state.chk"), report_interval),
        ]
    )

    if resumed:
        simulation.loadCheckpoint(str(args.resume_checkpoint))
        simulation.context.setParameter("k", 0.0)
    else:
        simulation.minimizeEnergy(maxIterations=10000)
        simulation.context.setVelocitiesToTemperature(
            args.temperature_k * unit.kelvin, args.seed
        )
        simulation.step(steps_for_ns(args.equilibration_ns, args.timestep_fs))
        simulation.context.setParameter("k", 0.0)
        simulation.context.setVelocitiesToTemperature(
            args.temperature_k * unit.kelvin, args.seed + 1
        )

    production_steps = steps_for_ns(args.production_ns, args.timestep_fs)
    simulation.step(production_steps)
    simulation.saveCheckpoint(str(args.output_dir / "final.chk"))
    simulation.saveState(str(args.output_dir / "final_state.xml"))
    metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prmtop": str(args.prmtop.resolve()),
        "inpcrd": str(args.inpcrd.resolve()),
        "platform": platform.getName(),
        "platform_properties": properties,
        "seed": args.seed,
        "temperature_k": args.temperature_k,
        "pressure_bar": args.pressure_bar,
        "timestep_fs": args.timestep_fs,
        "equilibration_ns": 0.0 if resumed else args.equilibration_ns,
        "production_ns": args.production_ns,
        "report_ps": args.report_ps,
        "restrained_heavy_atoms": restrained_atoms,
        "equilibration_restraint_k_kj_mol_nm2": args.restraint_k,
        "resumed": resumed,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

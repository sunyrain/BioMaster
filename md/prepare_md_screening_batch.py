#!/usr/bin/env python3
"""Prepare audited GROMACS packages for a short-MD screening manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import pandas as pd
import parmed as pmd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MDP_FILES = ("em.mdp", "nvt.mdp.template", "npt.mdp", "production_5ns.mdp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--water-padding-a", type=float, default=12.0)
    parser.add_argument("--max-system-atoms", type=int, default=200_000)
    parser.add_argument("--build-timeout-minutes", type=int, default=30)
    parser.add_argument("--keep-large-intermediates", action="store_true")
    return parser.parse_args()


def run(command: list[str], log: Path, cwd: Path | None = None) -> None:
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {command}; see {log}"
        )


def run_reduce(input_pdb: Path, output_pdb: Path, stderr_log: Path) -> None:
    with output_pdb.open("w", encoding="utf-8") as output, stderr_log.open(
        "w", encoding="utf-8"
    ) as errors:
        completed = subprocess.run(
            ["reduce", "-BUILD", str(input_pdb)],
            stdout=output,
            stderr=errors,
            check=False,
            env=os.environ.copy(),
        )
    output_text = output_pdb.read_text(encoding="utf-8", errors="replace")
    if "ATOM" not in output_text or not output_text.rstrip().endswith("END"):
        raise RuntimeError(
            f"Reduce did not produce a complete PDB (exit {completed.returncode}); "
            f"see {stderr_log}"
        )


def prepare_one(
    row: dict[str, object],
    output_root: Path,
    water_padding_a: float,
    max_system_atoms: int,
    build_timeout_minutes: int,
    keep_large_intermediates: bool,
) -> dict[str, object]:
    rank = int(row["md_screen_rank"])
    pair_id = str(row["pair_id"])
    item = output_root / f"{rank:03d}_{pair_id}"
    prep = item / "prep"
    system = item / "system"
    package = item / "remote_package"
    status_path = item / "preparation_status.json"
    item.mkdir(parents=True, exist_ok=True)
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("status") == "READY" and (package / "complex.top").is_file():
            return previous

    status: dict[str, object] = {
        "md_screen_rank": rank,
        "final384_rank": int(row["final384_rank"]),
        "pair_id": pair_id,
        "drug": str(row["fda_generic_name"]),
        "target": str(row["primary_gene"]),
        "status": "FAILED",
        "workdir": str(item.resolve()),
    }
    try:
        prep.mkdir(exist_ok=True)
        system.mkdir(exist_ok=True)
        package.mkdir(exist_ok=True)
        run(
            [
                sys.executable,
                str(PROJECT_ROOT / "md" / "prepare_experimental_pose.py"),
                "--experimental-pdb", str(row["pdb_path"]),
                "--experimental-chain", "A",
                "--boltz-cif", str(row["boltz_cif_path_refined"]),
                "--ligand-smiles", str(row["active_moiety_smiles"]),
                "--pair-id", pair_id,
                "--alignment-residue-ids", str(row["top_pocket_residue_ids"]),
                "--output-dir", str(prep),
            ],
            item / "pose_preparation.log",
        )
        pose_audit = json.loads((prep / "pose_mapping_audit.json").read_text())
        alignment_rmsd = float(pose_audit["alignment_ca_rmsd_angstrom"])
        if alignment_rmsd > 3.5:
            raise RuntimeError(f"Boltz-to-receptor alignment RMSD is {alignment_rmsd:.2f} A")
        severe_clashes = int(pose_audit["severe_receptor_ligand_clash_count_lt_1p5a"])
        minimum_distance = float(
            pose_audit["minimum_receptor_ligand_heavy_distance_angstrom"]
        )
        if severe_clashes > 4 or minimum_distance < 1.0:
            raise RuntimeError(
                f"Mapped pose has {severe_clashes} clashes below 1.5 A and a "
                f"minimum distance of {minimum_distance:.2f} A"
            )

        run_reduce(
            prep / "receptor_experimental_clean.pdb",
            prep / "receptor_reduce_h.pdb",
            item / "reduce.stderr.log",
        )
        run(
            [
                "pdb4amber", "-i", str(prep / "receptor_reduce_h.pdb"),
                "-o", str(prep / "receptor_amber.pdb"),
                "-l", str(prep / "pdb4amber.log"),
            ],
            item / "pdb4amber.stdout.log",
        )
        run(
            [
                "timeout", "--kill-after=30s", f"{build_timeout_minutes}m",
                sys.executable,
                str(PROJECT_ROOT / "md" / "build_amber_system.py"),
                "--receptor-pdb", str(prep / "receptor_amber.pdb"),
                "--ligand-sdf", str(prep / "ligand_aligned.sdf"),
                "--ligand-net-charge", str(int(row["ligand_formal_charge"])),
                "--output-dir", str(system),
                "--water-padding-a", str(water_padding_a),
                "--water-model", "opc",
            ],
            item / "system_build.log",
        )
        system_audit = json.loads((system / "system_build_audit.json").read_text())
        atoms = int(system_audit["atoms"])
        if atoms > max_system_atoms:
            raise RuntimeError(f"System has {atoms} atoms; cap is {max_system_atoms}")

        structure = pmd.load_file(
            str(system / "complex.prmtop"), str(system / "complex.inpcrd")
        )
        structure.save(str(package / "complex.top"), overwrite=True)
        structure.save(str(package / "complex.gro"), overwrite=True)
        run(
            [
                sys.executable,
                str(PROJECT_ROOT / "md" / "fix_gromacs_charge_rounding.py"),
                "--topology", str(package / "complex.top"),
                "--audit", str(package / "charge_rounding_audit.json"),
            ],
            item / "charge_rounding_fix.log",
        )
        charge_audit = json.loads(
            (package / "charge_rounding_audit.json").read_text(encoding="utf-8")
        )
        for filename in MDP_FILES:
            shutil.copy2(PROJECT_ROOT / "md" / "gromacs" / filename, package / filename)
        shutil.copy2(
            PROJECT_ROOT / "md" / "slurm" / "gromacs_screen_array_cpu.slurm",
            package / "gromacs_screen_array_cpu.slurm",
        )
        package_audit = {
            **status,
            "status": "READY",
            "alignment_ca_rmsd_angstrom": alignment_rmsd,
            "minimum_receptor_ligand_heavy_distance_angstrom": minimum_distance,
            "initial_clash_count_lt_1p5a": severe_clashes,
            "atoms": atoms,
            "ligand_formal_charge": int(row["ligand_formal_charge"]),
            "md_context_tier": str(row["md_context_tier"]),
            "gromacs_total_charge_after_e": charge_audit["total_charge_after_e"],
            "gromacs_rounding_correction_e": charge_audit["rounding_correction_e"],
            "remote_package": str(package.resolve()),
        }
        (package / "package_audit.json").write_text(
            json.dumps(package_audit, indent=2) + "\n", encoding="utf-8"
        )
        if not keep_large_intermediates:
            for filename in ("solvated_complex.pdb", "complex.prmtop", "complex.inpcrd"):
                (system / filename).unlink(missing_ok=True)
        status = package_audit
    except Exception as error:  # noqa: BLE001
        status["error"] = str(error)
        status["traceback"] = traceback.format_exc()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def write_summary(results: list[dict[str, object]], output_root: Path) -> None:
    results = sorted(results, key=lambda item: int(item["md_screen_rank"]))
    fields = [
        "md_screen_rank", "final384_rank", "pair_id", "drug", "target",
        "status", "atoms", "alignment_ca_rmsd_angstrom", "ligand_formal_charge",
        "md_context_tier", "remote_package", "error", "workdir",
    ]
    with (output_root / "preparation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    counts = pd.Series([item["status"] for item in results]).value_counts().to_dict()
    (output_root / "preparation_summary.json").write_text(
        json.dumps({"total": len(results), "status_counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = manifest.to_dict(orient="records")
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                prepare_one,
                row,
                args.output_root,
                args.water_padding_a,
                args.max_system_atoms,
                args.build_timeout_minutes,
                args.keep_large_intermediates,
            ): int(row["md_screen_rank"])
            for row in rows
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{len(results)}/{len(rows)}] rank={result['md_screen_rank']} "
                f"pair={result['pair_id']} status={result['status']}",
                flush=True,
            )
            write_summary(results, args.output_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage READY short-MD packages into a reproducible Slurm submission wave."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = (
    "complex.gro", "complex.top", "em.mdp", "npt.mdp", "nvt.mdp.template",
    "production_5ns.mdp", "package_audit.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--exclude-summary", action="append", type=Path, default=[])
    return parser.parse_args()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")
    excluded: set[str] = set()
    for path in args.exclude_summary:
        with path.open(newline="", encoding="utf-8") as handle:
            excluded.update(row["pair_id"] for row in csv.DictReader(handle))

    candidate_by_pair = {}
    for root in args.preparation_root:
        for status_path in root.glob("*/preparation_status.json"):
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "READY" or status["pair_id"] in excluded:
                continue
            package = status_path.parent / "remote_package"
            if not all((package / filename).is_file() for filename in PACKAGE_FILES):
                continue
            candidate = (int(status["md_screen_rank"]), status, package)
            previous = candidate_by_pair.get(status["pair_id"])
            if previous is None or candidate[0] < previous[0]:
                candidate_by_pair[status["pair_id"]] = candidate
    candidates = sorted(candidate_by_pair.values(), key=lambda item: item[0])
    if len(candidates) < args.count:
        raise RuntimeError(f"Only {len(candidates)} READY packages; requested {args.count}")

    args.output_root.mkdir(parents=True)
    cases_root = args.output_root / "cases"
    cases_root.mkdir()
    rows = []
    manifest_lines = []
    for task_index, (md_rank, status, package) in enumerate(candidates[: args.count], start=1):
        pair_id = str(status["pair_id"])
        case_name = f"{task_index:03d}_{pair_id}"
        case = cases_root / case_name
        case.mkdir()
        charge_audit = package / "charge_rounding_audit.json"
        if not charge_audit.is_file():
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "md" / "fix_gromacs_charge_rounding.py"),
                    "--topology", str(package / "complex.top"),
                    "--audit", str(charge_audit),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        for filename in (*PACKAGE_FILES, "charge_rounding_audit.json"):
            hardlink_or_copy(package / filename, case / filename)
        system_audit_path = package.parent / "system" / "system_build_audit.json"
        system_audit = (
            json.loads(system_audit_path.read_text(encoding="utf-8"))
            if system_audit_path.is_file()
            else {}
        )
        remote_workdir = f"{args.remote_base.rstrip('/')}/cases/{case_name}"
        seed = 2026071300 + md_rank
        manifest_lines.append(
            f"{md_rank}\t{pair_id}\t{remote_workdir}\t{seed}\n"
        )
        rows.append(
            {
                "array_task_id": task_index,
                "md_screen_rank": md_rank,
                "final384_rank": status["final384_rank"],
                "pair_id": pair_id,
                "drug": status["drug"],
                "target": status["target"],
                "atoms": status["atoms"],
                "alignment_ca_rmsd_angstrom": status["alignment_ca_rmsd_angstrom"],
                "md_context_tier": status.get("md_context_tier", ""),
                "water_model": system_audit.get("water_model", ""),
                "water_padding_angstrom": system_audit.get("water_padding_angstrom", ""),
                "remote_workdir": remote_workdir,
                "seed": seed,
            }
        )
    (args.output_root / "manifest.tsv").write_text(
        "".join(manifest_lines), encoding="utf-8"
    )
    with (args.output_root / "submission_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(
        PROJECT_ROOT / "md" / "slurm" / "gromacs_screen_array_cpu.slurm",
        args.output_root / "gromacs_screen_array_cpu.slurm",
    )
    (args.output_root / "submission_audit.json").write_text(
        json.dumps(
            {
                "count": len(rows),
                "unique_pairs": len({row["pair_id"] for row in rows}),
                "unique_drugs": len({row["drug"] for row in rows}),
                "unique_targets": len({row["target"] for row in rows}),
                "remote_base": args.remote_base,
                "max_concurrent_nodes": 5,
                "protocol": "EM -> NVT -> NPT -> 5 ns production",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json.loads((args.output_root / "submission_audit.json").read_text()), indent=2))


if __name__ == "__main__":
    main()

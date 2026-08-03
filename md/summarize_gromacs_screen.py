#!/usr/bin/env python3
"""Summarize GROMACS short-MD screening XVG files without third-party packages."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence


def read_xvg(path: Path) -> List[List[float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] in "#@":
                continue
            rows.append([float(value) for value in stripped.split()])
    if not rows:
        raise ValueError("No numeric data in {}".format(path))
    return rows


def quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def describe(rows: Sequence[Sequence[float]], column: int, start_ps: float, scale: float = 1.0) -> Dict[str, float]:
    selected = [row[column] * scale for row in rows if row[0] >= start_ps]
    if not selected:
        raise ValueError("No rows at or after {} ps".format(start_ps))
    ordered = sorted(selected)
    tail_count = max(1, int(math.ceil(len(selected) * 0.2)))
    return {
        "n": len(selected),
        "min": min(selected),
        "q25": quantile(ordered, 0.25),
        "median": statistics.median(selected),
        "q75": quantile(ordered, 0.75),
        "max": max(selected),
        "mean": statistics.mean(selected),
        "final": selected[-1],
        "last_20pct_median": statistics.median(selected[-tail_count:]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--global-index", required=True, type=int)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--context-tier", required=True)
    parser.add_argument("--equilibration-ps", type=float, default=500.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = args.analysis_dir
    start = args.equilibration_ps
    ligand = read_xvg(directory / "ligand_rmsd_nm.xvg")
    backbone = read_xvg(directory / "backbone_rmsd_nm.xvg")
    distance = read_xvg(directory / "min_distance_nm.xvg")
    contacts = read_xvg(directory / "contact_atom_pairs.xvg")
    thermo = read_xvg(directory / "thermodynamics.xvg")

    result = {
        "global_submission_index": args.global_index,
        "pair_id": args.pair_id,
        "context_tier": args.context_tier,
        "equilibration_excluded_ps": start,
        "ligand_rmsd_angstrom": describe(ligand, 1, start, 10.0),
        "protein_backbone_rmsd_angstrom": describe(backbone, 1, start, 10.0),
        "minimum_heavy_atom_distance_angstrom": describe(distance, 1, start, 10.0),
        "heavy_atom_contact_pairs_4A": describe(contacts, 1, start),
        "temperature_K": describe(thermo, 1, start),
        "pressure_bar": describe(thermo, 2, start),
        "density_kg_m3": describe(thermo, 3, start),
        "interpretation_limit": (
            "Five-nanosecond pose-retention triage; not an affinity estimate or "
            "experimental validation. Context tier B omits known biological context."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correct tiny AMBER-to-GROMACS charge-rounding residuals in a topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SECTION = re.compile(r"^\s*\[\s*([^]]+)\s*]\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    return parser.parse_args()


def data_tokens(line: str) -> list[str]:
    return line.split(";", 1)[0].split()


def parse_topology(lines: list[str]):  # noqa: ANN201
    section = ""
    current_molecule = ""
    expect_molecule_name = False
    atom_charges: dict[str, list[tuple[int, float]]] = {}
    molecule_counts: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = SECTION.match(line)
        if match:
            section = match.group(1).strip().lower()
            expect_molecule_name = section == "moleculetype"
            continue
        tokens = data_tokens(line)
        if not tokens:
            continue
        if expect_molecule_name:
            current_molecule = tokens[0]
            atom_charges.setdefault(current_molecule, [])
            expect_molecule_name = False
        elif section == "atoms" and current_molecule and len(tokens) >= 7:
            atom_charges[current_molecule].append((index, float(tokens[6])))
        elif section == "molecules" and len(tokens) >= 2:
            molecule_counts[tokens[0]] = int(tokens[1])
    return atom_charges, molecule_counts


def total_charge(
    atom_charges: dict[str, list[tuple[int, float]]], molecule_counts: dict[str, int]
) -> float:
    return sum(
        sum(charge for _, charge in atom_charges[name]) * count
        for name, count in molecule_counts.items()
    )


def replace_charge(line: str, new_charge: float) -> str:
    body, separator, comment = line.partition(";")
    tokens = body.split()
    tokens[6] = f"{new_charge:.8f}"
    rebuilt = " ".join(tokens)
    if separator and "qtot" not in comment:
        rebuilt += f"   ;{comment.rstrip()}"
    return rebuilt + "\n"


def main() -> None:
    args = parse_args()
    lines = args.topology.read_text(encoding="utf-8").splitlines(keepends=True)
    atom_charges, molecule_counts = parse_topology(lines)
    before = total_charge(atom_charges, molecule_counts)
    if abs(before) > args.tolerance:
        raise RuntimeError(
            f"Topology charge {before:.8f} exceeds rounding tolerance {args.tolerance}"
        )
    correction = 0.0
    corrected_molecule = ""
    corrected_line = -1
    if abs(before) > 5e-8:
        single_copy = [
            name
            for name, count in molecule_counts.items()
            if count == 1 and atom_charges.get(name)
        ]
        if not single_copy:
            raise RuntimeError("No single-copy molecule is available for rounding correction")
        corrected_molecule = single_copy[0]
        corrected_line, old_charge = atom_charges[corrected_molecule][0]
        correction = -before
        lines[corrected_line] = replace_charge(
            lines[corrected_line], old_charge + correction
        )
        args.topology.write_text("".join(lines), encoding="utf-8")

    checked_lines = args.topology.read_text(encoding="utf-8").splitlines(keepends=True)
    checked_charges, checked_counts = parse_topology(checked_lines)
    after = total_charge(checked_charges, checked_counts)
    if abs(after) > 5e-7:
        raise RuntimeError(f"Charge correction left a residual of {after:.8f}")
    result = {
        "topology": str(args.topology.resolve()),
        "total_charge_before_e": before,
        "rounding_correction_e": correction,
        "corrected_molecule": corrected_molecule,
        "corrected_line_1_based": corrected_line + 1 if corrected_line >= 0 else None,
        "total_charge_after_e": after,
        "interpretation": "Numerical text-rounding correction; not a protonation-state change.",
    }
    if args.audit:
        args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

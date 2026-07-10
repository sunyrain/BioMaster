from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.audit_boltz_pose_stability import (
    pocket_residue_ids_from_yaml,
    symmetry_corrected_ligand_rmsd,
)


def test_pocket_residue_ids_are_read_from_boltz_yaml(tmp_path: Path) -> None:
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
version: 1
constraints:
  - pocket:
      binder: B
      contacts:
        - [A, 10]
        - [A, 25]
        - [C, 99]
""",
        encoding="utf-8",
    )

    assert pocket_residue_ids_from_yaml(path) == {10, 25}


def test_ligand_rmsd_is_corrected_for_graph_symmetry() -> None:
    class Atom:
        def __init__(self, x: float) -> None:
            self.coord = np.asarray([x, 0.0, 0.0])
            self.element = "C"

    corrected, raw, method, permutations = symmetry_corrected_ligand_rmsd(
        [Atom(0.0), Atom(10.0)],
        [Atom(10.0), Atom(0.0)],
        "CC",
    )

    assert raw == 10.0
    assert corrected == 0.0
    assert method == "rdkit_graph_automorphism"
    assert permutations >= 2

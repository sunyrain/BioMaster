#!/usr/bin/env python3
"""Audit conditional poses before target-specific metal/cofactor MD preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser


ROOT = Path(__file__).resolve().parents[1]


def context_receptor_path(protocol_path: Path) -> Path:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    configured = str(
        protocol.get("files", {}).get("receptor_prepared_with_context_pdb", "")
    )
    if not configured:
        raise ValueError("protocol_has_no_receptor_prepared_with_context_pdb")
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def observed_hetero_residues(pdb_path: Path) -> list[str]:
    structure = PDBParser(QUIET=True).get_structure("receptor", pdb_path)
    model = next(structure.get_models())
    observed = []
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":
                continue
            name = residue.resname.strip().upper()
            if name not in {"HOH", "WAT"}:
                observed.append(f"{name}:{chain.id}:{residue.id[1]}")
    return sorted(set(observed))


def audit_one(row: dict[str, object], output_root: Path) -> dict[str, object]:
    rank = int(row["md_screen_rank"])
    pair_id = str(row["pair_id"])
    item = output_root / f"{rank:03d}_{pair_id}"
    prep = item / "prep"
    item.mkdir(parents=True, exist_ok=True)
    prep.mkdir(exist_ok=True)
    status: dict[str, object] = {
        "md_screen_rank": rank,
        "final384_rank": int(row["final384_rank"]),
        "pair_id": pair_id,
        "drug": str(row["fda_generic_name"]),
        "target": str(row["primary_gene"]),
        "target_chembl_id": str(row["target_chembl_id"]),
        "required_context_residues": str(row["required_context_residues"]),
        "status": "TECHNICAL_FAILURE",
        "workdir": str(item.resolve()),
    }
    try:
        command = [
            sys.executable, str(ROOT / "md/prepare_experimental_pose.py"),
            "--experimental-pdb", str(row["pdb_path"]),
            "--experimental-chain", str(row.get("experimental_chain", "A")),
            "--boltz-cif", str(row["boltz_cif_path_refined"]),
            "--ligand-smiles", str(row["active_moiety_smiles"]),
            "--pair-id", pair_id,
            "--alignment-residue-ids", str(
                row.get("boltz_pocket_residue_ids", row.get("top_pocket_residue_ids", ""))
            ),
            "--output-dir", str(prep),
        ]
        with (item / "pose_preparation.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            raise RuntimeError(f"pose_preparation_exit_{completed.returncode}")
        pose = json.loads((prep / "pose_mapping_audit.json").read_text())
        rmsd = float(pose["alignment_ca_rmsd_angstrom"])
        clashes = int(pose["severe_receptor_ligand_clash_count_lt_1p5a"])
        minimum_distance = float(pose["minimum_receptor_ligand_heavy_distance_angstrom"])
        context_pdb = context_receptor_path(Path(str(row["protocol_json"])))
        observed = observed_hetero_residues(context_pdb)
        required = {
            value.strip().upper()
            for value in str(row["required_context_residues"]).split(";")
            if value.strip()
        }
        observed_names = {value.split(":", 1)[0] for value in observed}
        status.update(
            {
                "alignment_ca_rmsd_angstrom": rmsd,
                "minimum_receptor_ligand_heavy_distance_angstrom": minimum_distance,
                "initial_clash_count_lt_1p5a": clashes,
                "observed_hetero_residues": ";".join(observed),
                "context_receptor_pdb": str(context_pdb),
                "required_context_observed": required.issubset(observed_names),
            }
        )
        if rmsd > 3.5:
            status["status"] = "STATE_ALIGNMENT_FAIL"
        elif clashes > 4 or minimum_distance < 1.0:
            status["status"] = "POSE_CLASH_FAIL"
        elif not required.issubset(observed_names):
            status["status"] = "REQUIRED_CONTEXT_NOT_OBSERVED"
        else:
            pre_pose_status = str(row.get("md_pre_pose_status", ""))
            if pre_pose_status == "PROVISIONAL_STANDARD":
                status["status"] = "POSE_GATE_PASS_STANDARD_MD_AUTHORIZED"
            else:
                status["status"] = "POSE_GATE_PASS_SPECIALIZED_PARAMETERIZATION_REQUIRED"
    except Exception as error:  # noqa: BLE001
        status["error"] = str(error)
        status["traceback"] = traceback.format_exc()
    (item / "specialized_pose_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = pd.read_csv(
        args.manifest.resolve(), low_memory=False, keep_default_na=False
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(audit_one, row, output_root): int(row["md_screen_rank"])
            for row in manifest.to_dict(orient="records")
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"rank={result['md_screen_rank']} pair={result['pair_id']} "
                f"status={result['status']}", flush=True
            )
    results = sorted(results, key=lambda row: int(row["md_screen_rank"]))
    pd.DataFrame(results).to_csv(output_root / "specialized_pose_audit.csv", index=False)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "status_counts": pd.Series([row["status"] for row in results]).value_counts().to_dict(),
        "interpretation": (
            "Pose-gate pass authorizes only the parameterization route declared in the input "
            "manifest; it is not an MD result or binding claim."
        ),
    }
    (output_root / "specialized_pose_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

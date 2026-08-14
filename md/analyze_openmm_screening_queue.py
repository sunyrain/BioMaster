#!/usr/bin/env python3
"""Analyze all completed OpenMM screening trajectories with a frozen pose rubric."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_class(summary: dict[str, object]) -> str:
    rmsd = summary["ligand_heavy_atom_rmsd_angstrom"]
    centroid = summary["ligand_centroid_displacement_angstrom"]
    contacts = summary["contact_residue_jaccard"]
    minimum = summary["minimum_ligand_protein_heavy_distance_angstrom"]
    backbone = summary["protein_backbone_rmsd_angstrom"]
    if (
        rmsd["median"] <= 2.5
        and rmsd["final"] <= 3.5
        and centroid["final"] <= 3.5
        and contacts["median"] >= 0.50
        and backbone["median"] <= 3.0
    ):
        return "MD_A_POSE_RETAINED"
    if (
        rmsd["median"] <= 4.0
        and rmsd["final"] <= 5.0
        and centroid["final"] <= 5.0
        and contacts["median"] >= 0.30
        and backbone["median"] <= 4.5
    ):
        return "MD_B_POSE_RELAXED_BUT_RETAINED"
    if (
        rmsd["median"] <= 6.0
        and centroid["final"] <= 7.0
        and contacts["median"] >= 0.15
        and minimum["median"] <= 4.0
    ):
        return "MD_C_CONTACT_RETAINED_POSE_SHIFTED"
    return "MD_D_UNSTABLE_OR_CONTACT_LOST"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--queue-summary-name", default="openmm_queue_summary.csv")
    parser.add_argument("--run-output-name", default="openmm_5ns")
    parser.add_argument(
        "--evidence-filename", default="C4_STANDARD_OPENMM_5NS_POSE_EVIDENCE_V1.csv"
    )
    parser.add_argument(
        "--summary-filename", default="C4_STANDARD_OPENMM_5NS_ANALYSIS_SUMMARY_V1.json"
    )
    args = parser.parse_args()
    preparation_root = args.preparation_root.resolve()
    output_dir = (args.output_dir or preparation_root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    queue_path = preparation_root / args.queue_summary_name
    if not queue_path.is_file():
        raise FileNotFoundError(queue_path)
    queue = pd.read_csv(queue_path, low_memory=False)
    queue = queue[queue["status"].eq("SUCCESS")].copy()
    records = []
    for row in queue.sort_values("md_screen_rank").itertuples(index=False):
        workdir = Path(str(row.workdir))
        trajectory = workdir / args.run_output_name / "trajectory.dcd"
        metadata_path = workdir / args.run_output_name / "run_metadata.json"
        if not trajectory.is_file() or not metadata_path.is_file():
            records.append(
                {
                    "md_screen_rank": int(row.md_screen_rank),
                    "pair_id": row.pair_id,
                    "analysis_status": "TECHNICAL_INCOMPLETE",
                    "analysis_error": "missing_trajectory_or_metadata",
                }
            )
            continue
        metadata = json.loads(metadata_path.read_text())
        topology = workdir / "system/complex.prmtop"
        coordinates = workdir / "system/complex.inpcrd"
        report_ps = float(metadata["report_ps"])
        equilibration_ns = float(metadata["equilibration_ns"])
        equilibration_frames = int(round(equilibration_ns * 1000.0 / report_ps))
        summary_path = output_dir / f"{int(row.md_screen_rank):03d}_{row.pair_id}.json"
        command = [
            str(ROOT / ".conda_envs/md_openmm/bin/python"),
            str(ROOT / "md/analyze_ligand_stability.py"),
            "--topology", str(topology),
            "--coordinates", str(coordinates),
            "--trajectory", str(trajectory),
            "--equilibration-frames", str(equilibration_frames),
            "--output", str(summary_path),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            records.append(
                {
                    "md_screen_rank": int(row.md_screen_rank),
                    "pair_id": row.pair_id,
                    "analysis_status": "FAILED",
                    "analysis_error": completed.stderr[-1000:],
                }
            )
            continue
        summary = json.loads(summary_path.read_text())
        records.append(
            {
                "md_screen_rank": int(row.md_screen_rank),
                "pair_id": row.pair_id,
                "drug": getattr(row, "drug", ""),
                "target": getattr(row, "target", ""),
                "analysis_status": "PASS",
                "md_pose_retention_class": pose_class(summary),
                "ligand_rmsd_median_A": summary["ligand_heavy_atom_rmsd_angstrom"]["median"],
                "ligand_rmsd_final_A": summary["ligand_heavy_atom_rmsd_angstrom"]["final"],
                "ligand_centroid_final_A": summary["ligand_centroid_displacement_angstrom"]["final"],
                "contact_jaccard_median": summary["contact_residue_jaccard"]["median"],
                "contact_jaccard_final": summary["contact_residue_jaccard"]["final"],
                "backbone_rmsd_median_A": summary["protein_backbone_rmsd_angstrom"]["median"],
                "minimum_heavy_distance_median_A": summary[
                    "minimum_ligand_protein_heavy_distance_angstrom"
                ]["median"],
                "analyzed_frames": summary["analyzed_frames"],
                "reference_coordinates": summary["reference_coordinates"],
                "reference_box_source": summary.get("reference_box_source", ""),
                "pbc_reconstruction": bool(summary.get("pbc_reconstruction", False)),
                "topology_sha256": sha256_file(topology),
                "coordinates_sha256": sha256_file(coordinates),
                "trajectory_sha256": sha256_file(trajectory),
                "run_metadata_sha256": sha256_file(metadata_path),
                "analysis_sha256": sha256_file(summary_path),
                "trajectory_bytes": trajectory.stat().st_size,
                "topology_path": str(topology),
                "coordinates_path": str(coordinates),
                "trajectory_path": str(trajectory),
                "run_metadata_path": str(metadata_path),
                "analysis_path": str(summary_path),
            }
        )
    evidence = pd.DataFrame(records).sort_values("md_screen_rank")
    output = output_dir / args.evidence_filename
    evidence.to_csv(output, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": len(queue),
        "analysis_status_counts": evidence["analysis_status"].value_counts().to_dict(),
        "pose_class_counts": evidence.get(
            "md_pose_retention_class", pd.Series(dtype=str)
        ).value_counts().to_dict(),
        "interpretation": (
            "5 ns single-seed retention of a pre-positioned conditional pose, measured "
            "against the prepared initial complex after excluding equilibration; not affinity."
        ),
        "queue_summary_name": args.queue_summary_name,
        "run_output_name": args.run_output_name,
        "output": str(output),
    }
    (output_dir / args.summary_filename).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Route C4 nonstandard cofactors, metals, and ligands to explicit MD protocols."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/c4_md_execution_v2"


def route(row: pd.Series) -> tuple[str, str, str]:
    if row["status"] != "POSE_GATE_PASS_SPECIALIZED_PARAMETERIZATION_REQUIRED":
        return (
            "S0_STOP_INITIAL_POSE_CLASH",
            "STOPPED",
            "Initial mapped pose fails the predeclared clash gate; parameterization cannot rescue it.",
        )
    context = str(row.get("required_context_residues", "")).upper()
    elements = str(row.get("specialized_ligand_elements", "")).upper()
    if elements == "B" and context:
        return (
            "S1_JOINT_METAL_AND_BORON_PARAMETERIZATION",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "QM-derived boron ligand parameters plus target-specific metal-center treatment are both required.",
        )
    if elements == "B":
        return (
            "S2_BORON_LIGAND_QM_PARAMETERIZATION",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "Generic GAFF2/AM1-BCC is not accepted for the boronic acid pharmacophore; derive and audit bespoke parameters.",
        )
    if "HEM" in context:
        return (
            "S3_VALIDATED_HEME_STATE_PROTOCOL",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "Freeze heme oxidation/protonation and axial-ligation state and use a validated heme parameter set.",
        )
    if any(item in context for item in ["FAD", "FMN"]):
        return (
            "S4_VALIDATED_FLAVIN_PROTOCOL",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "Retain the experimental flavin state and use a validated FAD/FMN parameter set.",
        )
    if "SAM" in context:
        return (
            "S5_METAL_COFACTOR_JOINT_PROTOCOL",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "Retain SAM and the catalytic metal together; validate both cofactor and coordination geometry.",
        )
    if any(item in context for item in ["ZN", "FE2"]):
        return (
            "S6_MCPB_METAL_CENTER_PROTOCOL",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "Build a target-specific bonded or validated nonbonded metal-center model; generic ion parameters are insufficient.",
        )
    if any(item in context for item in ["MG", "CA", "NA"]):
        return (
            "S7_ION_COORDINATION_RESTRAINT_PROTOCOL",
            "READY_FOR_SPECIALIZED_PARAMETERIZATION",
            "Use validated ion parameters with predeclared coordination restraints and monitor geometry explicitly.",
        )
    return (
        "S8_MANUAL_NONSTANDARD_CONTEXT_REVIEW",
        "MANUAL_REVIEW",
        "A nonstandard context was detected but no validated automated branch is frozen.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=BASE / "C4_MD_EXECUTION_MANIFEST_SPECIALIZED_CONTEXT_V2.csv",
    )
    parser.add_argument(
        "--pose-audit", type=Path,
        default=BASE / "specialized_pose_audit/specialized_pose_audit.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "specialized_parameterization",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest.resolve(), keep_default_na=False, low_memory=False)
    audit = pd.read_csv(args.pose_audit.resolve(), keep_default_na=False, low_memory=False)
    data = manifest.merge(
        audit[
            [
                "pair_id", "status", "alignment_ca_rmsd_angstrom",
                "minimum_receptor_ligand_heavy_distance_angstrom",
                "initial_clash_count_lt_1p5a", "observed_hetero_residues",
                "context_receptor_pdb", "required_context_observed",
            ]
        ],
        on="pair_id", how="left", validate="one_to_one",
    )
    decisions = data.apply(route, axis=1, result_type="expand")
    data[["specialized_protocol_route", "specialized_compute_status", "route_reason"]] = decisions
    data["mandatory_validation"] = (
        "parameter completeness; net charge; retained cofactor/metal; coordination geometry; "
        "initial energy; 0.1 ns restrained smoke test before 5 ns production"
    )
    data["result_scope"] = (
        "Specialized short-MD pose stress only; no affinity inference and no comparison with standard-protocol energies."
    )
    output = output_dir / "C4_SPECIALIZED_PARAMETERIZATION_ROUTES_V1.csv"
    data.to_csv(output, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(data),
        "pose_gate_pass": int(
            data["status"].eq("POSE_GATE_PASS_SPECIALIZED_PARAMETERIZATION_REQUIRED").sum()
        ),
        "pose_gate_stop": int(data["status"].eq("POSE_CLASH_FAIL").sum()),
        "route_counts": data["specialized_protocol_route"].value_counts().to_dict(),
        "status_counts": data["specialized_compute_status"].value_counts().to_dict(),
        "boundary": (
            "These rows are not sent through the generic protein+GAFF2 path; each must "
            "pass target-specific parameter and coordination validation first."
        ),
        "output": str(output),
    }
    (output_dir / "C4_SPECIALIZED_PARAMETERIZATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

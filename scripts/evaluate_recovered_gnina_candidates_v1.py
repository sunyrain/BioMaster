#!/usr/bin/env python3
"""Evaluate recovered candidate docking against each target's frozen control distributions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/recovered_gnina_candidate_docking_v1"
CANDIDATES = OUT / "RECOVERED_GNINA_CANDIDATE_PANEL_V1.csv"
JOBS = OUT / "RECOVERED_GNINA_CANDIDATE_JOBS_V1.csv"
CONTROL_EVIDENCE = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_CONTROL_LIGAND_EVIDENCE_V1.csv.gz"
QUALIFICATION = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_PREDICTED_POCKET_METRICS_23_V1.csv"
RDLogger.DisableLog("rdApp.*")


def parse_output(target: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for pose_index, molecule in enumerate(Chem.SDMolSupplier(str(output), removeHs=False, sanitize=False)):
        if molecule is None:
            continue
        props = molecule.GetPropsAsDict()
        rows.append({
            "target_chembl_id": target,
            "candidate_pair_id": molecule.GetProp("_Name") if molecule.HasProp("_Name") else "",
            "pose_row_index": pose_index,
            "cnn_score": props.get("CNNscore"), "cnn_affinity": props.get("CNNaffinity"),
            "cnn_vs": props.get("CNN_VS"), "vina_affinity": props.get("minimizedAffinity"),
        })
    poses = pd.DataFrame(rows)
    if poses.empty:
        return poses, poses
    numeric = ["cnn_score", "cnn_affinity", "cnn_vs", "vina_affinity"]
    poses[numeric] = poses[numeric].apply(pd.to_numeric, errors="coerce")
    primary_indices = poses.groupby("candidate_pair_id")["cnn_score"].idxmax()
    primary = poses.loc[primary_indices].rename(columns={
        "pose_row_index": "primary_pose_row_index", "cnn_score": "primary_pose_cnn_score",
        "cnn_affinity": "primary_cnn_affinity", "cnn_vs": "primary_cnn_vs",
        "vina_affinity": "primary_pose_vina_affinity",
    })
    primary = primary[[
        "target_chembl_id", "candidate_pair_id", "primary_pose_row_index", "primary_pose_cnn_score",
        "primary_cnn_affinity", "primary_cnn_vs", "primary_pose_vina_affinity",
    ]]
    best_vina = poses.groupby("candidate_pair_id", as_index=False)["vina_affinity"].min().rename(
        columns={"vina_affinity": "best_vina_affinity"}
    )
    counts = poses.groupby("candidate_pair_id", as_index=False).size().rename(columns={"size": "pose_count"})
    ligands = primary.merge(best_vina, on="candidate_pair_id", validate="one_to_one").merge(
        counts, on="candidate_pair_id", validate="one_to_one"
    )
    ligands["vina_directional"] = -ligands["best_vina_affinity"]
    return poses, ligands


def main() -> None:
    candidates = pd.read_csv(CANDIDATES)
    jobs = pd.read_csv(JOBS, dtype=str).fillna("")
    pose_frames = []
    ligand_frames = []
    for job in jobs.itertuples(index=False):
        output = Path(job.output_sdf)
        status_path = output.parent / "run_status.json"
        if not status_path.is_file() or not json.loads(status_path.read_text(encoding="utf-8")).get("completed"):
            continue
        poses, ligands = parse_output(job.target_chembl_id, output)
        if len(ligands):
            pose_frames.append(poses)
            ligand_frames.append(ligands)
    if not ligand_frames:
        raise RuntimeError("No completed candidate docking output")
    poses = pd.concat(pose_frames, ignore_index=True)
    scores = pd.concat(ligand_frames, ignore_index=True)
    evidence = candidates.merge(scores, on=["target_chembl_id", "candidate_pair_id"], validate="one_to_one")
    controls = pd.read_csv(CONTROL_EVIDENCE)
    qualification = pd.read_csv(QUALIFICATION)[
        ["target_chembl_id", "predicted_pocket_qualification", "auroc_primary_cnn_affinity", "auroc_secondary_vina"]
    ]

    calibration_rows = []
    for target, group in evidence.groupby("target_chembl_id", sort=True):
        target_controls = controls[controls["target_chembl_id"].eq(target)].copy()
        positive_controls = target_controls[pd.to_numeric(target_controls["binary_label"]).eq(1)]
        all_cnn = pd.to_numeric(target_controls["primary_cnn_affinity"], errors="coerce").dropna().to_numpy()
        all_vina = pd.to_numeric(target_controls["vina_directional"], errors="coerce").dropna().to_numpy()
        positive_cnn = pd.to_numeric(positive_controls["primary_cnn_affinity"], errors="coerce").dropna().to_numpy()
        positive_vina = pd.to_numeric(positive_controls["vina_directional"], errors="coerce").dropna().to_numpy()
        for _, row in group.iterrows():
            cnn = float(row["primary_cnn_affinity"])
            vina = float(row["vina_directional"])
            calibration_rows.append({
                "candidate_pair_id": row["candidate_pair_id"],
                "cnn_affinity_percentile_vs_all_controls": float(np.mean(all_cnn <= cnn)),
                "vina_percentile_vs_all_controls": float(np.mean(all_vina <= vina)),
                "cnn_affinity_percentile_vs_positive_controls": float(np.mean(positive_cnn <= cnn)),
                "vina_percentile_vs_positive_controls": float(np.mean(positive_vina <= vina)),
                "positive_control_median_cnn_affinity": float(np.median(positive_cnn)),
                "positive_control_median_vina_directional": float(np.median(positive_vina)),
            })
    evidence = evidence.merge(pd.DataFrame(calibration_rows), on="candidate_pair_id", validate="one_to_one")
    evidence = evidence.merge(qualification, on="target_chembl_id", validate="many_to_one")
    evidence["gnina_control_calibrated_support"] = (
        evidence["cnn_affinity_percentile_vs_all_controls"].ge(0.75)
        & evidence["vina_percentile_vs_all_controls"].ge(0.75)
        & evidence["primary_cnn_affinity"].ge(evidence["positive_control_median_cnn_affinity"])
        & evidence["vina_directional"].ge(evidence["positive_control_median_vina_directional"])
    )
    concordant = evidence["candidate_route"].eq("TWO_MODEL_TARGET_TOP10PCT_CONCORDANT")
    support = evidence["gnina_control_calibrated_support"]
    evidence["candidate_triage_status"] = np.select(
        [concordant & support, concordant & ~support, ~concordant & support],
        [
            "COMPUTATIONAL_TRIAGE_PASS_REQUIRES_EXPERIMENT",
            "NO_GNINA_CONTROL_CALIBRATED_SUPPORT",
            "ORTHOGONAL_RESCUE_SIGNAL_MODEL_DISAGREEMENT_REVIEW",
        ],
        default="MODEL_DISAGREEMENT_NO_ORTHOGONAL_SUPPORT",
    )
    evidence["cnn_affinity_rank_within_candidate_target"] = evidence.groupby("target_chembl_id")["primary_cnn_affinity"].rank(
        ascending=False, method="min"
    )
    evidence["vina_rank_within_candidate_target"] = evidence.groupby("target_chembl_id")["vina_directional"].rank(
        ascending=False, method="min"
    )
    evidence = evidence.sort_values(
        ["target_chembl_id", "candidate_triage_status", "cnn_affinity_rank_within_candidate_target"],
        kind="mergesort",
    )

    poses_path = OUT / "RECOVERED_GNINA_CANDIDATE_POSES_V1.csv.gz"
    evidence_path = OUT / "RECOVERED_GNINA_CANDIDATE_EVIDENCE_V1.csv"
    poses.to_csv(poses_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    evidence.to_csv(evidence_path, index=False)
    formal_pass = evidence[evidence["candidate_triage_status"].eq("COMPUTATIONAL_TRIAGE_PASS_REQUIRES_EXPERIMENT")]
    diagnostic = evidence[evidence["candidate_triage_status"].eq("ORTHOGONAL_RESCUE_SIGNAL_MODEL_DISAGREEMENT_REVIEW")]
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS" if len(evidence) == len(candidates) else "INCOMPLETE",
        "targets": evidence["target_chembl_id"].nunique(), "candidate_pairs": len(evidence), "pose_rows": len(poses),
        "formal_computational_triage_pass_pairs": len(formal_pass),
        "formal_pass_targets": int(formal_pass["target_chembl_id"].nunique()),
        "model_disagreement_orthogonal_rescue_pairs": len(diagnostic),
        "triage_status_counts": {str(k): int(v) for k, v in evidence["candidate_triage_status"].value_counts().items()},
        "formal_pass_candidates": formal_pass[["target_chembl_id", "gene_symbol", "drug_names", "ligand_inchikey"]].to_dict("records"),
        "interpretation": "A computational triage pass requires a strongly control-qualified predicted pocket, two-model DTA concordance, and candidate CNN-affinity plus Vina scores at or above the target positive-control medians and the 75th percentile of all controls. It is not binding evidence.",
        "outputs": {"poses": str(poses_path), "candidate_evidence": str(evidence_path)},
    }
    (OUT / "RECOVERED_GNINA_CANDIDATE_EVALUATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Candidate evaluation is incomplete")


if __name__ == "__main__":
    main()

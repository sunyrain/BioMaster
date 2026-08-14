#!/usr/bin/env python3
"""Evaluate 12x12 GNINA controls for the recovered predicted-pocket program."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1"
CONTROLS = OUT / "RECOVERED_GNINA_CONTROL_PANEL_23_X_24_V1.csv.gz"
JOBS = OUT / "RECOVERED_GNINA_POCKET_VALIDATION_JOBS_23_V1.csv"
RDLogger.DisableLog("rdApp.*")


def seed(value: str) -> int:
    return 1 + int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            values.append(roc_auc_score(labels[indices], scores[indices]))
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def permutation_p(labels: np.ndarray, scores: np.ndarray, observed: float, repeats: int, rng: np.random.Generator) -> float:
    exceed = 0
    for _ in range(repeats):
        exceed += roc_auc_score(rng.permutation(labels), scores) >= observed
    return float((exceed + 1) / (repeats + 1))


def bh_adjust(values: pd.Series) -> pd.Series:
    output = pd.Series(math.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    n = len(valid)
    adjusted = valid.to_numpy() * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def parse_target(target: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for pose_index, molecule in enumerate(Chem.SDMolSupplier(str(output), removeHs=False, sanitize=False)):
        if molecule is None:
            continue
        props = molecule.GetPropsAsDict()
        rows.append({
            "target_chembl_id": target,
            "control_pair_id": molecule.GetProp("_Name") if molecule.HasProp("_Name") else "",
            "pose_row_index": pose_index,
            "cnn_score": props.get("CNNscore"),
            "cnn_affinity": props.get("CNNaffinity"),
            "cnn_vs": props.get("CNN_VS"),
            "vina_affinity": props.get("minimizedAffinity"),
        })
    poses = pd.DataFrame(rows)
    if poses.empty:
        return poses, poses
    score_columns = ["cnn_score", "cnn_affinity", "cnn_vs", "vina_affinity"]
    poses[score_columns] = poses[score_columns].apply(pd.to_numeric, errors="coerce")
    primary_indices = poses.groupby("control_pair_id")["cnn_score"].idxmax()
    primary = poses.loc[primary_indices].rename(columns={
        "pose_row_index": "primary_pose_row_index", "cnn_score": "primary_pose_cnn_score",
        "cnn_affinity": "primary_cnn_affinity", "cnn_vs": "primary_cnn_vs",
        "vina_affinity": "primary_pose_vina_affinity",
    })
    primary = primary[[
        "target_chembl_id", "control_pair_id", "primary_pose_row_index", "primary_pose_cnn_score",
        "primary_cnn_affinity", "primary_cnn_vs", "primary_pose_vina_affinity",
    ]]
    best_vina = poses.groupby("control_pair_id", as_index=False)["vina_affinity"].min().rename(
        columns={"vina_affinity": "best_vina_affinity"}
    )
    pose_count = poses.groupby("control_pair_id", as_index=False).size().rename(columns={"size": "pose_count"})
    ligand = primary.merge(best_vina, on="control_pair_id", validate="one_to_one").merge(
        pose_count, on="control_pair_id", validate="one_to_one"
    )
    ligand["vina_directional"] = -ligand["best_vina_affinity"]
    return poses, ligand


def main() -> None:
    controls = pd.read_csv(CONTROLS, dtype=str).fillna("")
    jobs = pd.read_csv(JOBS, dtype=str).fillna("")
    pose_frames = []
    ligand_frames = []
    completed_targets = []
    for job in jobs.itertuples(index=False):
        output = Path(job.output_sdf)
        status_path = output.parent / "run_status.json"
        if not output.is_file() or not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status.get("completed"):
            continue
        poses, ligands = parse_target(job.target_chembl_id, output)
        if not ligands.empty:
            pose_frames.append(poses)
            ligand_frames.append(ligands)
            completed_targets.append(job.target_chembl_id)
    if not ligand_frames:
        raise RuntimeError("No completed recovered GNINA outputs")
    poses = pd.concat(pose_frames, ignore_index=True)
    ligand_scores = pd.concat(ligand_frames, ignore_index=True)
    evidence = controls.merge(
        ligand_scores, on=["target_chembl_id", "control_pair_id"], how="inner", validate="one_to_one"
    )

    metrics = []
    for target, group in evidence.groupby("target_chembl_id", sort=True):
        labels = pd.to_numeric(group["binary_label"], errors="raise").to_numpy(dtype=int)
        primary = pd.to_numeric(group["primary_cnn_affinity"], errors="coerce").to_numpy(dtype=float)
        vina = pd.to_numeric(group["vina_directional"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(primary) & np.isfinite(vina)
        labels, primary, vina = labels[valid], primary[valid], vina[valid]
        positive = int(labels.sum())
        negative = int(len(labels) - positive)
        rng = np.random.default_rng(seed(target))
        primary_auc = float(roc_auc_score(labels, primary))
        vina_auc = float(roc_auc_score(labels, vina))
        ci_low, ci_high = bootstrap_auc(labels, primary, 2000, rng)
        positive_scores = primary[labels == 1]
        negative_scores = primary[labels == 0]
        mw = mannwhitneyu(positive_scores, negative_scores, alternative="greater")
        job = jobs[jobs["target_chembl_id"].eq(target)].iloc[0]
        metrics.append({
            "target_chembl_id": target, "gene_symbol": job["gene_symbol"],
            "computed_pocket_evidence": job["computed_pocket_evidence"],
            "receptor_preparation_method": job["receptor_preparation_method"],
            "rows": len(labels), "positive": positive, "negative": negative,
            "formally_evaluable_12x12": positive >= 12 and negative >= 12,
            "auroc_primary_cnn_affinity": primary_auc,
            "average_precision_primary_cnn_affinity": float(average_precision_score(labels, primary)),
            "auroc_secondary_vina": vina_auc,
            "average_precision_secondary_vina": float(average_precision_score(labels, vina)),
            "median_primary_cnn_affinity_positive": float(np.median(positive_scores)),
            "median_primary_cnn_affinity_negative": float(np.median(negative_scores)),
            "median_primary_cnn_affinity_delta": float(np.median(positive_scores) - np.median(negative_scores)),
            "bootstrap_auroc_ci_low": ci_low, "bootstrap_auroc_ci_high": ci_high,
            "permutation_p_one_sided": permutation_p(labels, primary, primary_auc, 5000, rng),
            "mannwhitney_p_one_sided": float(mw.pvalue),
        })
    metrics = pd.DataFrame(metrics)
    metrics["permutation_bh_q"] = bh_adjust(metrics["permutation_p_one_sided"])
    metrics["primary_gate_pass"] = (
        metrics["formally_evaluable_12x12"]
        & metrics["auroc_primary_cnn_affinity"].ge(0.65)
        & metrics["average_precision_primary_cnn_affinity"].ge(0.60)
        & metrics["permutation_bh_q"].le(0.10)
    )
    metrics["primary_gate_strong"] = metrics["primary_gate_pass"] & metrics["bootstrap_auroc_ci_low"].gt(0.50)
    metrics["secondary_vina_direction_consistent"] = metrics["auroc_secondary_vina"].ge(0.55)
    metrics["predicted_pocket_qualification"] = np.select(
        [
            metrics["primary_gate_strong"] & metrics["secondary_vina_direction_consistent"],
            metrics["primary_gate_pass"],
            metrics["auroc_primary_cnn_affinity"].ge(0.60),
        ],
        [
            "QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION",
            "QUALIFIED_PRIMARY_ONLY_SECONDARY_REVIEW",
            "MARGINAL_NOT_QUALIFIED",
        ],
        default="FAILED_CONTROL_SEPARATION",
    )
    metrics.loc[
        metrics["receptor_preparation_method"].str.contains("FALLBACK", na=False)
        & metrics["predicted_pocket_qualification"].str.startswith("QUALIFIED"),
        "predicted_pocket_qualification",
    ] += "_RECEPTOR_FALLBACK_REVIEW"

    poses_path = OUT / "RECOVERED_GNINA_CONTROL_POSES_V1.csv.gz"
    evidence_path = OUT / "RECOVERED_GNINA_CONTROL_LIGAND_EVIDENCE_V1.csv.gz"
    metrics_path = OUT / "RECOVERED_GNINA_PREDICTED_POCKET_METRICS_23_V1.csv"
    poses.to_csv(poses_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    evidence.to_csv(evidence_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "status": "PASS" if len(set(completed_targets)) == len(jobs) else "PARTIAL",
        "completed_targets": len(set(completed_targets)), "expected_targets": len(jobs),
        "ligand_rows": len(evidence), "pose_rows": len(poses),
        "primary_gate_pass": int(metrics["primary_gate_pass"].sum()),
        "primary_gate_strong": int(metrics["primary_gate_strong"].sum()),
        "secondary_vina_consistent": int(metrics["secondary_vina_direction_consistent"].sum()),
        "qualification_counts": {str(k): int(v) for k, v in metrics["predicted_pocket_qualification"].value_counts().items()},
        "qualification_policy": "Frozen before full evaluation: 12x12 controls, CNN-affinity AUROC >=0.65, AP >=0.60, BH q<=0.10; strong additionally requires bootstrap AUROC lower CI >0.50. Vina AUROC >=0.55 is an orthogonal direction check.",
        "interpretation": "Retrospective docking control separation can qualify a predicted pocket for candidate docking, but never upgrades it to experimental-pocket evidence or establishes binding.",
        "outputs": {"poses": str(poses_path), "ligand_evidence": str(evidence_path), "metrics": str(metrics_path)},
    }
    (OUT / "RECOVERED_GNINA_POCKET_VALIDATION_EVALUATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

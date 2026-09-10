#!/usr/bin/env python3
"""Evaluate completed GNINA calibration jobs under the frozen Gate-A/Gate-B rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/gnina_calibration_338_v1"
MANIFEST = OUT / "GNINA_CALIBRATION_EXECUTION_MANIFEST_LOCKED_V1.csv.gz"
JOBS = OUT / "execution_inputs_prepared_v1/GNINA_CALIBRATION_JOBS_V1.csv"
RDLogger.DisableLog("rdApp.*")


def clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def seed(value: str) -> int:
    return 1 + int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000


def parse_target(target: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for pose_index, molecule in enumerate(Chem.SDMolSupplier(str(output), removeHs=False, sanitize=False)):
        if molecule is None:
            continue
        props = molecule.GetPropsAsDict()
        pair_id = molecule.GetProp("_Name") if molecule.HasProp("_Name") else clean(props.get("control_pair_id"))
        rows.append({
            "target_chembl_id": target, "control_pair_id": pair_id,
            "pose_row_index": pose_index, "cnn_score": props.get("CNNscore"),
            "cnn_affinity": props.get("CNNaffinity"), "vina_affinity": props.get("minimizedAffinity"),
        })
    poses = pd.DataFrame(rows)
    if poses.empty:
        return poses, poses
    for column in ["cnn_score", "cnn_affinity", "vina_affinity"]:
        poses[column] = pd.to_numeric(poses[column], errors="coerce")
    primary_indices = poses.groupby("control_pair_id")["cnn_score"].idxmax()
    primary = poses.loc[primary_indices, [
        "target_chembl_id", "control_pair_id", "pose_row_index", "cnn_score", "cnn_affinity", "vina_affinity"
    ]].rename(columns={
        "pose_row_index": "primary_pose_row_index", "cnn_score": "primary_pose_cnn_score",
        "cnn_affinity": "primary_cnn_affinity", "vina_affinity": "primary_pose_vina_affinity",
    })
    vina = poses.groupby("control_pair_id", as_index=False)["vina_affinity"].min().rename(
        columns={"vina_affinity": "best_vina_affinity"}
    )
    counts = poses.groupby("control_pair_id", as_index=False).size().rename(columns={"size": "pose_count"})
    ligand = primary.merge(vina, on="control_pair_id", how="left", validate="one_to_one").merge(
        counts, on="control_pair_id", how="left", validate="one_to_one"
    )
    ligand["vina_directional"] = -ligand["best_vina_affinity"]
    return poses, ligand


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) == 2:
            values.append(roc_auc_score(labels[idx], scores[idx]))
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values else (math.nan, math.nan)


def permutation_p(labels: np.ndarray, scores: np.ndarray, observed: float, repeats: int, rng: np.random.Generator) -> float:
    exceed = 0
    for _ in range(repeats):
        permuted = rng.permutation(labels)
        exceed += roc_auc_score(permuted, scores) >= observed
    return float((exceed + 1) / (repeats + 1))


def bh_adjust(values: pd.Series) -> pd.Series:
    output = pd.Series(math.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return output
    n = len(valid)
    adjusted = (valid.to_numpy() * n / np.arange(1, n + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output.loc[valid.index] = np.minimum(adjusted, 1.0)
    return output


def metric_row(target: str, group: pd.DataFrame, scope: str, bootstrap: int, permutations: int) -> dict:
    labels = group["control_class"].eq("positive").astype(int).to_numpy()
    primary = pd.to_numeric(group["primary_cnn_affinity"], errors="coerce").to_numpy()
    vina = pd.to_numeric(group["vina_directional"], errors="coerce").to_numpy()
    valid = np.isfinite(primary)
    labels = labels[valid]
    primary = primary[valid]
    vina = vina[valid]
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    row = {
        "target_chembl_id": target, "scope": scope, "rows": len(labels),
        "positive": positive, "negative": negative,
        "formally_evaluable": (positive >= 8 and negative >= 8) if scope == "GATE_A" else (positive >= 4 and negative >= 4),
    }
    if positive == 0 or negative == 0:
        return row
    rng = np.random.default_rng(seed(target + scope))
    row["auroc_primary_cnn_affinity"] = float(roc_auc_score(labels, primary))
    row["average_precision_primary_cnn_affinity"] = float(average_precision_score(labels, primary))
    row["auroc_secondary_vina"] = float(roc_auc_score(labels, vina))
    row["average_precision_secondary_vina"] = float(average_precision_score(labels, vina))
    primary_order = np.argsort(-primary)
    vina_order = np.argsort(-vina)
    row["bedroc_alpha20_primary_cnn_affinity"] = float(
        CalcBEDROC(labels[primary_order].reshape(-1, 1), 0, 20.0)
    )
    row["bedroc_alpha20_secondary_vina"] = float(
        CalcBEDROC(labels[vina_order].reshape(-1, 1), 0, 20.0)
    )
    low, high = bootstrap_auc(labels, primary, bootstrap, rng)
    row["bootstrap_auroc_ci_low"] = low
    row["bootstrap_auroc_ci_high"] = high
    row["permutation_p_one_sided"] = permutation_p(
        labels, primary, row["auroc_primary_cnn_affinity"], permutations, rng
    )
    top_n = max(1, int(math.ceil(0.05 * len(labels))))
    row["ef_5pct"] = float(labels[primary_order[:top_n]].mean() / labels.mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--jobs", type=Path, default=JOBS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--label", default="current")
    parser.add_argument("--output-dir", type=Path, default=OUT / "evaluation")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, low_memory=False)
    jobs = pd.read_csv(args.jobs)
    pose_frames = []
    ligand_frames = []
    completed_targets = []
    for row in jobs.itertuples(index=False):
        output = Path(str(row.output_sdf))
        status_path = output.parent / "run_status.json"
        if not output.exists() or not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not status.get("completed"):
            continue
        poses, ligand = parse_target(str(row.target_chembl_id), output)
        if not ligand.empty:
            pose_frames.append(poses)
            ligand_frames.append(ligand)
            completed_targets.append(str(row.target_chembl_id))
    poses = pd.concat(pose_frames, ignore_index=True) if pose_frames else pd.DataFrame()
    ligands = pd.concat(ligand_frames, ignore_index=True) if ligand_frames else pd.DataFrame()
    if ligands.empty:
        raise RuntimeError("No completed GNINA outputs")
    evidence = manifest.merge(
        ligands, on=["target_chembl_id", "control_pair_id"], how="inner", validate="one_to_one"
    )
    metrics = []
    for target, group in evidence.groupby("target_chembl_id", sort=True):
        gate_a = group[group["calibration_split"].eq("LOCKED_EVALUATION")]
        metrics.append(metric_row(target, gate_a, "GATE_A", args.bootstrap, args.permutations))
        if "gate_b_evaluation_member" in group.columns:
            gate_b = group[group["gate_b_evaluation_member"].fillna(False).astype(bool)]
            if len(gate_b):
                metrics.append(
                    metric_row(target, gate_b, "GATE_B", args.bootstrap, args.permutations)
                )
    metrics = pd.DataFrame(metrics)
    metrics["permutation_bh_q"] = math.nan
    for scope, indices in metrics.groupby("scope").groups.items():
        metrics.loc[indices, "permutation_bh_q"] = bh_adjust(metrics.loc[indices, "permutation_p_one_sided"])
    metrics["primary_gate_pass_raw"] = (
        metrics["formally_evaluable"].fillna(False).astype(bool)
        & metrics["auroc_primary_cnn_affinity"].ge(0.65)
        & metrics["permutation_bh_q"].le(0.10)
        & (metrics["scope"].eq("GATE_B") | metrics["average_precision_primary_cnn_affinity"].ge(0.60))
    )
    gate_a_pass = metrics.loc[
        metrics["scope"].eq("GATE_A"), ["target_chembl_id", "primary_gate_pass_raw"]
    ].set_index("target_chembl_id")["primary_gate_pass_raw"]
    metrics["gate_a_dependency_pass"] = metrics["target_chembl_id"].map(gate_a_pass).fillna(False)
    metrics["primary_gate_pass"] = metrics["primary_gate_pass_raw"] & (
        metrics["scope"].eq("GATE_A") | metrics["gate_a_dependency_pass"]
    )
    metrics["primary_gate_strong"] = metrics["primary_gate_pass"] & metrics["bootstrap_auroc_ci_low"].gt(0.50)
    base = args.output_dir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    suffix = args.label.upper()
    poses_path = base / f"GNINA_CALIBRATION_POSES_{suffix}_V1.csv.gz"
    evidence_path = base / f"GNINA_CALIBRATION_LIGAND_EVIDENCE_{suffix}_V1.csv.gz"
    metrics_path = base / f"GNINA_CALIBRATION_TARGET_METRICS_{suffix}_V1.csv"
    poses.to_csv(poses_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    evidence.to_csv(evidence_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "completed_targets": len(set(completed_targets)), "ligand_rows": len(evidence),
        "pose_rows": len(poses),
        "gate_a_formally_evaluable": int(metrics.query("scope == 'GATE_A'")["formally_evaluable"].sum()),
        "gate_a_pass": int(metrics.query("scope == 'GATE_A'")["primary_gate_pass"].sum()),
        "gate_b_formally_evaluable": int(metrics.query("scope == 'GATE_B'")["formally_evaluable"].sum()),
        "gate_b_pass": int(metrics.query("scope == 'GATE_B'")["primary_gate_pass"].sum()),
        "partial_evaluation": len(set(completed_targets)) < manifest["target_chembl_id"].nunique(),
        "outputs": {"poses": str(poses_path), "ligand_evidence": str(evidence_path), "target_metrics": str(metrics_path)},
    }
    path = base / f"GNINA_CALIBRATION_EVALUATION_SUMMARY_{suffix}_V1.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

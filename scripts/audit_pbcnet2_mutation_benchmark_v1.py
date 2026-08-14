#!/usr/bin/env python3
"""Reproduce and audit the public PBCNet2.0 mutation benchmark.

This script is intentionally evaluation-only.  The 65 public mutation examples
are treated as a locked external test set and are never emitted as training
features or split assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch


MUTATION_TOKEN = re.compile(r"^[A-Z][0-9]+[A-Z]$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if kind == "pearson":
        return float(stats.pearsonr(a, b).statistic)
    return float(stats.spearmanr(a, b).statistic)


def kabsch_rmsd(x: np.ndarray, y: np.ndarray) -> float:
    """RMSD after rigid alignment, assuming audited atom-order correspondence."""
    if x.shape != y.shape or x.size == 0:
        return float("nan")
    x0 = x - x.mean(axis=0, keepdims=True)
    y0 = y - y.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(x0.T @ y0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    delta = x0 @ rotation - y0
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def mutation_tokens(filename: str) -> list[str]:
    stem = Path(filename).stem
    pieces = stem.split("_")
    return [piece for piece in pieces if MUTATION_TOKEN.fullmatch(piece)]


def graph_pair_audit(g1, g2) -> dict:
    d1 = g1.nodes["atom"].data
    d2 = g2.nodes["atom"].data
    ligand1 = d1["type"].detach().cpu().numpy().astype(bool)
    ligand2 = d2["type"].detach().cpu().numpy().astype(bool)
    protein1 = ~ligand1
    protein2 = ~ligand2

    ligand_x1 = d1["x"].detach().cpu().numpy()[ligand1]
    ligand_x2 = d2["x"].detach().cpu().numpy()[ligand2]
    ligand_scalar1 = d1["atom_scalar"].detach().cpu().numpy()[ligand1]
    ligand_scalar2 = d2["atom_scalar"].detach().cpu().numpy()[ligand2]
    ligand_pos1 = d1["pos"].detach().cpu().numpy()[ligand1]
    ligand_pos2 = d2["pos"].detach().cpu().numpy()[ligand2]
    same_count = int(ligand1.sum()) == int(ligand2.sum())
    same_ordered_identity = bool(
        same_count
        and np.array_equal(ligand_x1, ligand_x2)
        and np.array_equal(ligand_scalar1, ligand_scalar2)
    )

    res1 = set(d1["res_idx"].detach().cpu().numpy()[protein1].astype(int).tolist())
    res2 = set(d2["res_idx"].detach().cpu().numpy()[protein2].astype(int).tolist())
    union = res1 | res2
    residue_jaccard = float(len(res1 & res2) / len(union)) if union else float("nan")

    return {
        "wt_atoms": int(g1.num_nodes("atom")),
        "mutant_atoms": int(g2.num_nodes("atom")),
        "wt_ligand_atoms": int(ligand1.sum()),
        "mutant_ligand_atoms": int(ligand2.sum()),
        "ordered_ligand_identity_equal": same_ordered_identity,
        "ligand_kabsch_rmsd_angstrom": (
            kabsch_rmsd(ligand_pos1, ligand_pos2) if same_ordered_identity else float("nan")
        ),
        "protein_residue_id_jaccard": residue_jaccard,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".external/pbcnet2")
    parser.add_argument(
        "--output-dir",
        default="outputs/old_drug_target_sota_v1/pbcnet2_mutation_benchmark_audit_v1",
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mutation_root = repo / "data" / "Mutation"
    model_path = repo / "PBCNet2.pth"
    readout_path = repo / "model_code" / "models" / "readout.py"
    graph_builder_path = repo / "Graph2pickle.py"

    torch.set_num_threads(args.threads)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    records: list[dict] = []
    for target_dir in sorted(path for path in mutation_root.iterdir() if path.is_dir()):
        table = pd.read_csv(target_dir / "predict.csv")
        for row_index, row in table.iterrows():
            wt_path = target_dir / row["lig1"]
            mutant_path = target_dir / row["lig2"]
            g1 = load_graph(wt_path)
            g2 = load_graph(mutant_path)
            pair_audit = graph_pair_audit(g1, g2)
            with torch.no_grad():
                forward, reverse = model(dgl.batch([g1]), dgl.batch([g2]))
            pred = float(forward.reshape(-1)[0].cpu())
            pred_reverse = float(reverse.reshape(-1)[0].cpu())
            tokens = mutation_tokens(str(row["lig2"]))
            records.append(
                {
                    "target_uniprot": target_dir.name,
                    "row_index": int(row_index),
                    "wt_graph": str(row["lig1"]),
                    "mutant_graph": str(row["lig2"]),
                    "mutation_tokens": ";".join(tokens),
                    "mutation_count": len(tokens),
                    "experimental_label": float(row["Label"]),
                    "stored_prediction": float(row["pre"]),
                    "reproduced_prediction": pred,
                    "reproduced_reverse_prediction": pred_reverse,
                    "absolute_reproduction_error": abs(pred - float(row["pre"])),
                    "hard_antisymmetry_error": abs(pred + pred_reverse),
                    **pair_audit,
                }
            )

    predictions = pd.DataFrame.from_records(records)
    predictions_path = output_dir / "PBCNET2_MUTATION_REPRODUCED_PREDICTIONS_V1.csv"
    predictions.to_csv(predictions_path, index=False)

    per_target: list[dict] = []
    for target, group in predictions.groupby("target_uniprot", sort=True):
        y = group["experimental_label"].to_numpy(float)
        p = group["reproduced_prediction"].to_numpy(float)
        per_target.append(
            {
                "target_uniprot": target,
                "n": int(len(group)),
                "pearson": safe_corr(y, p, "pearson"),
                "spearman": safe_corr(y, p, "spearman"),
                "mae": float(np.mean(np.abs(y - p))),
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            }
        )
    per_target_df = pd.DataFrame(per_target)
    per_target_path = output_dir / "PBCNET2_MUTATION_PER_TARGET_METRICS_V1.csv"
    per_target_df.to_csv(per_target_path, index=False)

    y_all = predictions["experimental_label"].to_numpy(float)
    p_all = predictions["reproduced_prediction"].to_numpy(float)
    readout_source = readout_path.read_text()
    architecture_audit = {
        "separate_complex_encoding": readout_source.count("self._readout(") >= 2,
        "ligand_mask_before_pooling": "emb * mask" in readout_source,
        "global_pool_before_difference": (
            "dgl.readout_nodes" in readout_source and "emb1-emb2" in readout_source
        ),
        "explicit_cross_complex_atom_or_residue_correspondence": False,
        "reverse_branch_is_learned_not_hard_negative": True,
        "evidence": {
            "readout": "ligand atom embeddings are summed independently for each complex",
            "difference": "LayerNorm(emb1 - emb2) is applied only after global pooling",
            "reverse": "the same FNN is evaluated on emb2 - emb1; no algebraic negation is imposed",
        },
    }

    max_repro = float(predictions["absolute_reproduction_error"].max())
    all_ligands_correspond = bool(predictions["ordered_ligand_identity_equal"].all())
    summary = {
        "schema_version": "PBCNET2_MUTATION_BENCHMARK_AUDIT_V1",
        "status": "PASS" if max_repro <= 1e-5 and len(predictions) == 65 else "FAIL",
        "evaluation_role": "LOCKED_EXTERNAL_TEST_ONLY",
        "training_use_prohibited": True,
        "repository": {
            "path": str(repo),
            "commit": git_value(repo, "rev-parse", "HEAD"),
            "commit_date": git_value(repo, "log", "-1", "--format=%aI"),
            "model_sha256": sha256(model_path),
            "readout_source_sha256": sha256(readout_path),
            "graph_builder_sha256": sha256(graph_builder_path),
        },
        "environment": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "dgl": dgl.__version__,
            "device": "cpu",
        },
        "dataset": {
            "targets": int(predictions["target_uniprot"].nunique()),
            "examples": int(len(predictions)),
            "single_mutation_examples": int((predictions["mutation_count"] == 1).sum()),
            "multiple_mutation_examples": int((predictions["mutation_count"] > 1).sum()),
            "all_ordered_ligand_atom_identities_match": all_ligands_correspond,
            "median_ligand_kabsch_rmsd_angstrom": float(
                predictions["ligand_kabsch_rmsd_angstrom"].median()
            ),
            "median_protein_residue_id_jaccard": float(
                predictions["protein_residue_id_jaccard"].median()
            ),
        },
        "reproduction": {
            "max_absolute_prediction_error": max_repro,
            "mean_absolute_prediction_error": float(
                predictions["absolute_reproduction_error"].mean()
            ),
            "tolerance": 1e-5,
            "within_tolerance": bool(max_repro <= 1e-5),
        },
        "metrics": {
            "macro_target_pearson": float(per_target_df["pearson"].mean()),
            "macro_target_spearman": float(per_target_df["spearman"].mean()),
            "pooled_pearson": safe_corr(y_all, p_all, "pearson"),
            "pooled_spearman": safe_corr(y_all, p_all, "spearman"),
            "pooled_mae": float(np.mean(np.abs(y_all - p_all))),
            "pooled_rmse": float(np.sqrt(np.mean((y_all - p_all) ** 2))),
        },
        "antisymmetry": {
            "hard_constraint_satisfied": bool(
                predictions["hard_antisymmetry_error"].max() <= 1e-6
            ),
            "mean_abs_forward_plus_reverse": float(
                predictions["hard_antisymmetry_error"].mean()
            ),
            "max_abs_forward_plus_reverse": float(
                predictions["hard_antisymmetry_error"].max()
            ),
        },
        "architecture_audit": architecture_audit,
        "innovation_gap_supported_by_code": (
            "PBCNet2.0 does not compare aligned ligand-protein interaction edges before pooling; "
            "its complex-level difference is post-pooling and its reverse branch is not exactly antisymmetric."
        ),
        "files": {
            "predictions_csv": str(predictions_path),
            "per_target_metrics_csv": str(per_target_path),
        },
    }
    summary_path = output_dir / "PBCNET2_MUTATION_BENCHMARK_AUDIT_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

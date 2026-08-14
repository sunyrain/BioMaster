#!/usr/bin/env python3
"""Diagnose the preregistered TRACE-PL Phase-A mechanism failure.

This audit reads only Platinum development OOF predictions.  It never opens
the locked PBCNet2 mutation labels or predictions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FEATURE_INDEX = ROOT / "outputs/old_drug_target_sota_v1/trace_pl_platinum_features_v1/TRACE_PL_PLATINUM_FEATURE_INDEX_V1.csv"
OOF = ROOT / "outputs/old_drug_target_sota_v1/trace_pl_phase_a_v1/TRACE_PL_PHASE_A_OOF_PREDICTIONS_V1.csv"
PHASE_A_SUMMARY = ROOT / "outputs/old_drug_target_sota_v1/trace_pl_phase_a_v1/TRACE_PL_PHASE_A_SUMMARY_V1.json"
OUT = ROOT / "outputs/old_drug_target_sota_v1/trace_pl_failure_audit_v1"

PRIMARY_PROTOCOL = "homology_cluster_cold"
MODELS = [
    "trace_full",
    "direct_pair",
    "trace_single_conformer",
    "trace_no_atomic_contact",
    "trace_shuffled_ligand_contact",
    "extra_trees_nonatomic",
    "train_mean",
    "zero_ddg",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_corr(y: np.ndarray, p: np.ndarray, kind: str) -> float | None:
    if len(y) < 3 or np.ptp(p) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    value = stats.pearsonr(y, p).statistic if kind == "pearson" else stats.spearmanr(y, p).statistic
    return float(value) if np.isfinite(value) else None


def metric_row(frame: pd.DataFrame, model: str, distance_bin: str) -> dict[str, object]:
    y = frame["label"].to_numpy(dtype=float)
    p = frame["prediction"].to_numpy(dtype=float)
    error = p - y
    return {
        "model": model,
        "distance_bin": distance_bin,
        "n": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": finite_corr(y, p, "pearson"),
        "spearman": finite_corr(y, p, "spearman"),
        "label_mean": float(np.mean(y)),
        "label_std": float(np.std(y)),
        "positive_fraction": float(np.mean(y > 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURE_INDEX)
    oof = pd.read_csv(OOF)
    phase = json.loads(PHASE_A_SUMMARY.read_text(encoding="utf-8"))

    primary = oof[oof["protocol"].eq(PRIMARY_PROTOCOL) & oof["model"].isin(MODELS)].copy()
    primary = primary.merge(
        features[
            [
                "sample_id",
                "uniprot_id",
                "wt_pdb_id",
                "chain",
                "mutation",
                "old_aa",
                "new_aa",
                "ligand_smiles",
                "ligand_atom_count",
                "environment_atom_count",
                "old_state_conformer_count",
                "new_state_conformer_count",
                "computed_mutation_ligand_min_distance",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    # Three official Platinum records have no UniProt mapping.  Preserve them
    # under a structure-chain identity rather than silently dropping them.
    primary["uniprot_id"] = primary["uniprot_id"].fillna(
        "UNMAPPED__" + primary["wt_pdb_id"].astype(str) + "__" + primary["chain"].astype(str)
    )
    required = [
        "label",
        "prediction",
        "homology_cluster",
        "mutation",
        "ligand_smiles",
        "computed_mutation_ligand_min_distance",
    ]
    if primary[required].isna().any().any():
        missing_columns = primary[required].columns[primary[required].isna().any()].tolist()
        raise ValueError(f"Unexpected missing merged fields: {missing_columns}")

    primary["distance_bin"] = pd.cut(
        primary["computed_mutation_ligand_min_distance"],
        bins=[-np.inf, 4.0, 6.0, 8.0, np.inf],
        labels=["LE_4A", "GT_4_LE_6A", "GT_6_LE_8A", "GT_8A"],
        right=True,
    ).astype(str)

    metric_rows: list[dict[str, object]] = []
    for model in MODELS:
        model_frame = primary[primary["model"].eq(model)]
        metric_rows.append(metric_row(model_frame, model, "ALL"))
        for distance_bin, group in model_frame.groupby("distance_bin", observed=True, sort=False):
            metric_rows.append(metric_row(group, model, str(distance_bin)))
    metrics = pd.DataFrame(metric_rows)
    metrics_path = OUT / "TRACE_PL_FAILURE_METRICS_BY_DISTANCE_V1.csv"
    metrics.to_csv(metrics_path, index=False)

    wide = primary.pivot(index="sample_id", columns="model", values="prediction")
    labels = primary.drop_duplicates("sample_id").set_index("sample_id")["label"]
    if wide.shape[0] != len(features) or not wide.index.equals(labels.index):
        wide = wide.sort_index()
        labels = labels.reindex(wide.index)
    full = wide["trace_full"].to_numpy(dtype=float)
    sensitivity_rows = []
    for comparator in [
        "direct_pair",
        "trace_single_conformer",
        "trace_no_atomic_contact",
        "trace_shuffled_ligand_contact",
    ]:
        other = wide[comparator].to_numpy(dtype=float)
        sensitivity_rows.append(
            {
                "reference": "trace_full",
                "comparator": comparator,
                "prediction_pearson": finite_corr(full, other, "pearson"),
                "mean_absolute_prediction_difference": float(np.mean(np.abs(full - other))),
                "root_mean_square_prediction_difference": float(np.sqrt(np.mean((full - other) ** 2))),
                "full_minus_comparator_mean_squared_error": float(
                    np.mean((full - labels.to_numpy(dtype=float)) ** 2)
                    - np.mean((other - labels.to_numpy(dtype=float)) ** 2)
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_path = OUT / "TRACE_PL_FAILURE_VARIANT_SENSITIVITY_V1.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    unique = primary[primary["model"].eq("trace_full")].copy()
    cluster_counts = unique["homology_cluster"].value_counts()
    protein_counts = unique["uniprot_id"].value_counts()
    ligand_counts = unique["ligand_smiles"].value_counts()
    fold_metrics = (
        primary[primary["model"].isin(["trace_full", "direct_pair", "train_mean"])]
        .groupby(["fold", "model"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "n": len(group),
                    "rmse": float(np.sqrt(np.mean((group["prediction"] - group["label"]) ** 2))),
                    "pearson": finite_corr(
                        group["label"].to_numpy(dtype=float),
                        group["prediction"].to_numpy(dtype=float),
                        "pearson",
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    fold_path = OUT / "TRACE_PL_FAILURE_FOLD_CONCENTRATION_V1.csv"
    fold_metrics.to_csv(fold_path, index=False)

    primary_full = metrics[(metrics["model"].eq("trace_full")) & (metrics["distance_bin"].eq("ALL"))].iloc[0]
    direct_full = metrics[(metrics["model"].eq("direct_pair")) & (metrics["distance_bin"].eq("ALL"))].iloc[0]
    summary = {
        "schema_version": "TRACE_PL_FAILURE_AUDIT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "Platinum development OOF only; PBCNet2 external labels remain unopened",
        "decision": "TRACE_PL_PHASE_A_REJECTED; DO_NOT_OPEN_PBC65",
        "counts": {
            "samples": int(len(unique)),
            "uniprot_ids": int(unique["uniprot_id"].nunique()),
            "homology_clusters": int(unique["homology_cluster"].nunique()),
            "ligands": int(unique["ligand_smiles"].nunique()),
            "distance_le_8A": int((unique["computed_mutation_ligand_min_distance"] <= 8.0).sum()),
            "distance_gt_8A": int((unique["computed_mutation_ligand_min_distance"] > 8.0).sum()),
            "singleton_homology_clusters": int((cluster_counts == 1).sum()),
            "singleton_proteins": int((protein_counts == 1).sum()),
            "singleton_ligands": int((ligand_counts == 1).sum()),
        },
        "concentration": {
            "largest_homology_cluster_fraction": float(cluster_counts.iloc[0] / len(unique)),
            "largest_protein_fraction": float(protein_counts.iloc[0] / len(unique)),
            "largest_ligand_fraction": float(ligand_counts.iloc[0] / len(unique)),
        },
        "primary_outcome": {
            "trace_full_rmse": float(primary_full["rmse"]),
            "trace_full_pearson": float(primary_full["pearson"]),
            "direct_pair_rmse": float(direct_full["rmse"]),
            "direct_pair_pearson": float(direct_full["pearson"]),
            "bootstrap_trace_full_minus_direct_pair_rmse": phase["bootstrap_comparisons"][
                "trace_full_vs_direct_pair"
            ],
        },
        "mechanism_diagnosis": {
            "rotamer_ensemble_supported": False,
            "atomic_contact_supported": False,
            "true_ligand_contact_supported": False,
            "shared_state_potential_supported": False,
            "interpretation": (
                "The canonical free-amino-acid conformer field is not an adequate proxy for a "
                "protein-conditioned mutant side-chain ensemble. Its imposed potential-difference "
                "structure loses predictive signal relative to the capacity-matched direct paired head."
            ),
        },
        "integrity": {
            "external_test_opened": False,
            "phase_a_integrity_passed": bool(phase["development_gates"]["integrity"]),
            "all_expected_models_present": bool(set(MODELS).issubset(set(primary["model"]))),
            "one_prediction_per_sample_model": bool(
                primary.groupby(["sample_id", "model"]).size().eq(1).all()
            ),
        },
        "inputs": {
            str(FEATURE_INDEX.relative_to(ROOT)): sha256(FEATURE_INDEX),
            str(OOF.relative_to(ROOT)): sha256(OOF),
            str(PHASE_A_SUMMARY.relative_to(ROOT)): sha256(PHASE_A_SUMMARY),
        },
        "artifacts": {
            str(metrics_path.relative_to(ROOT)): sha256(metrics_path),
            str(sensitivity_path.relative_to(ROOT)): sha256(sensitivity_path),
            str(fold_path.relative_to(ROOT)): sha256(fold_path),
        },
    }
    summary_path = OUT / "TRACE_PL_FAILURE_AUDIT_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

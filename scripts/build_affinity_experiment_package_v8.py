#!/usr/bin/env python3
"""Build the v8 target-calibrated discovery reserve and measurement package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski
from sklearn.linear_model import HuberRegressor
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/affinity_experiment_package_v8.yaml"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"1", "true", "yes", "y"})


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def empirical_percentile(value: float, reference: np.ndarray) -> float:
    reference = np.sort(reference[np.isfinite(reference)])
    if not len(reference) or not np.isfinite(value):
        return np.nan
    left = np.searchsorted(reference, value, side="left")
    right = np.searchsorted(reference, value, side="right")
    return float((left + right) / (2.0 * len(reference)))

PROPERTY_COLUMNS_V8 = [
    "ligand_heavy_atoms_v8",
    "ligand_rotatable_bonds_v8",
    "ligand_formal_charge_v8",
]


def ligand_properties(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(clean(smiles))
    if molecule is None:
        return {
            "ligand_heavy_atoms_v8": np.nan,
            "ligand_rotatable_bonds_v8": np.nan,
            "ligand_formal_charge_v8": np.nan,
        }
    return {
        "ligand_heavy_atoms_v8": int(molecule.GetNumHeavyAtoms()),
        "ligand_rotatable_bonds_v8": int(Lipinski.NumRotatableBonds(molecule)),
        "ligand_formal_charge_v8": int(Chem.GetFormalCharge(molecule)),
    }


def add_candidate_properties(frame: pd.DataFrame) -> pd.DataFrame:
    unique = frame[["model_ligand_smiles"]].drop_duplicates().copy()
    properties = pd.DataFrame(
        [ligand_properties(value) for value in unique["model_ligand_smiles"]]
    )
    properties["model_ligand_smiles"] = unique["model_ligand_smiles"].values
    return frame.merge(
        properties, on="model_ligand_smiles", how="left", validate="many_to_one"
    )


def add_size_adjusted_scores(
    discovery: pd.DataFrame,
    control_scores: pd.DataFrame,
    controls_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_columns = [
        "control_pair_id",
        "control_heavy_atoms",
        "control_rotatable_bonds",
        "control_formal_charge",
    ]
    manifest = controls_manifest[manifest_columns].drop_duplicates("control_pair_id")
    controls = control_scores.merge(
        manifest, on="control_pair_id", how="left", validate="one_to_one"
    ).rename(
        columns={
            "control_heavy_atoms": "ligand_heavy_atoms_v8",
            "control_rotatable_bonds": "ligand_rotatable_bonds_v8",
            "control_formal_charge": "ligand_formal_charge_v8",
        }
    )
    candidates = add_candidate_properties(discovery)
    if controls[PROPERTY_COLUMNS_V8].isna().any().any():
        raise ValueError("Control ligand properties are incomplete")
    if candidates[PROPERTY_COLUMNS_V8].isna().any().any():
        raise ValueError("Discovery ligand properties are incomplete")

    center_columns = {
        column: f"{column}_target_median"
        for column in PROPERTY_COLUMNS_V8
    }
    centers = (
        controls.groupby("sequence_key", as_index=False)[PROPERTY_COLUMNS_V8]
        .median()
        .rename(columns=center_columns)
    )
    controls = controls.merge(centers, on="sequence_key", how="left", validate="many_to_one")
    candidates = candidates.merge(
        centers, on="sequence_key", how="left", validate="many_to_one"
    )
    parameters: dict[str, Any] = {
        "fit_population": "negative controls only",
        "features": PROPERTY_COLUMNS_V8,
        "model": "HuberRegressor on target-centered ligand properties",
        "channels": {},
    }
    control_x = np.column_stack(
        [
            pd.to_numeric(controls[column], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(
                controls[center_columns[column]], errors="coerce"
            ).to_numpy(dtype=float)
            for column in PROPERTY_COLUMNS_V8
        ]
    )
    candidate_x = np.column_stack(
        [
            pd.to_numeric(candidates[column], errors="coerce").to_numpy(dtype=float)
            - pd.to_numeric(
                candidates[center_columns[column]], errors="coerce"
            ).to_numpy(dtype=float)
            for column in PROPERTY_COLUMNS_V8
        ]
    )
    negative = controls["control_class"].eq("negative").to_numpy()
    for raw_column, adjusted_column in [
        ("best_cnn_affinity", "cnn_affinity_size_adjusted_v8"),
        ("score_vina_directional", "vina_affinity_size_adjusted_v8"),
    ]:
        raw = pd.to_numeric(controls[raw_column], errors="coerce").to_numpy(dtype=float)
        target_center = (
            controls.groupby("sequence_key")[raw_column]
            .transform("median")
            .to_numpy(dtype=float)
        )
        train = negative & np.isfinite(raw) & np.isfinite(control_x).all(axis=1)
        model = HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=1000)
        model.fit(control_x[train], (raw - target_center)[train])
        controls[adjusted_column] = raw - model.predict(control_x)
        candidate_raw = pd.to_numeric(
            candidates[raw_column], errors="coerce"
        ).to_numpy(dtype=float)
        candidates[adjusted_column] = candidate_raw - model.predict(candidate_x)
        parameters["channels"][raw_column] = {
            "intercept": float(model.intercept_),
            "coefficients": {
                column: float(value)
                for column, value in zip(PROPERTY_COLUMNS_V8, model.coef_)
            },
            "training_rows": int(train.sum()),
        }
    return candidates, controls, parameters


def normalize_against_controls(
    discovery: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    control_groups = {key: group.copy() for key, group in controls.groupby("sequence_key")}
    rows: list[dict[str, Any]] = []
    for sequence_key, group in discovery.groupby("sequence_key", sort=False):
        reference = control_groups.get(sequence_key)
        if reference is None:
            for pair_id in group["physical_pair_id"]:
                rows.append({"physical_pair_id": pair_id})
            continue
        positive = reference[reference["control_class"].eq("positive")]
        negative = reference[reference["control_class"].eq("negative")]
        references = {
            "cnn_negative": pd.to_numeric(
                negative["best_cnn_affinity"], errors="coerce"
            ).to_numpy(dtype=float),
            "cnn_positive": pd.to_numeric(
                positive["best_cnn_affinity"], errors="coerce"
            ).to_numpy(dtype=float),
            "vina_negative": pd.to_numeric(
                negative["score_vina_directional"], errors="coerce"
            ).to_numpy(dtype=float),
            "vina_positive": pd.to_numeric(
                positive["score_vina_directional"], errors="coerce"
            ).to_numpy(dtype=float),
            "cnn_adjusted_negative": pd.to_numeric(
                negative["cnn_affinity_size_adjusted_v8"], errors="coerce"
            ).to_numpy(dtype=float),
            "cnn_adjusted_positive": pd.to_numeric(
                positive["cnn_affinity_size_adjusted_v8"], errors="coerce"
            ).to_numpy(dtype=float),
            "vina_adjusted_negative": pd.to_numeric(
                negative["vina_affinity_size_adjusted_v8"], errors="coerce"
            ).to_numpy(dtype=float),
            "vina_adjusted_positive": pd.to_numeric(
                positive["vina_affinity_size_adjusted_v8"], errors="coerce"
            ).to_numpy(dtype=float),
        }
        for _, row in group.iterrows():
            cnn = float(row["best_cnn_affinity"])
            vina = float(row["score_vina_directional"])
            cnn_adjusted = float(row["cnn_affinity_size_adjusted_v8"])
            vina_adjusted = float(row["vina_affinity_size_adjusted_v8"])
            rows.append(
                {
                    "physical_pair_id": row["physical_pair_id"],
                    "cnn_negative_percentile_v8": empirical_percentile(
                        cnn, references["cnn_negative"]
                    ),
                    "cnn_positive_percentile_v8": empirical_percentile(
                        cnn, references["cnn_positive"]
                    ),
                    "vina_negative_percentile_v8": empirical_percentile(
                        vina, references["vina_negative"]
                    ),
                    "vina_positive_percentile_v8": empirical_percentile(
                        vina, references["vina_positive"]
                    ),
                    "cnn_size_adjusted_negative_percentile_v8": empirical_percentile(
                        cnn_adjusted, references["cnn_adjusted_negative"]
                    ),
                    "cnn_size_adjusted_positive_percentile_v8": empirical_percentile(
                        cnn_adjusted, references["cnn_adjusted_positive"]
                    ),
                    "vina_size_adjusted_negative_percentile_v8": empirical_percentile(
                        vina_adjusted, references["vina_adjusted_negative"]
                    ),
                    "vina_size_adjusted_positive_percentile_v8": empirical_percentile(
                        vina_adjusted, references["vina_adjusted_positive"]
                    ),
                    "control_positive_n_v8": int(len(positive)),
                    "control_negative_n_v8": int(len(negative)),
                }
            )
    return discovery.merge(
        pd.DataFrame(rows), on="physical_pair_id", how="left", validate="one_to_one"
    )


def assign_pair_tiers(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    rules = config["pair_evidence"]
    q90 = float(rules["strong_negative_percentile"])
    q95 = float(rules["exceptional_negative_percentile"])
    q50 = float(rules["noncontradictory_negative_percentile"])
    positive_floor = float(rules["positive_distribution_floor"])
    size_adjusted_floor = float(rules["size_adjusted_negative_percentile_floor"])

    cnn_valid = as_bool(data["cnn_affinity_pass_v8"])
    vina_valid = as_bool(data["vina_affinity_pass_v8"])
    cnn_neg = pd.to_numeric(data["cnn_negative_percentile_v8"], errors="coerce")
    vina_neg = pd.to_numeric(data["vina_negative_percentile_v8"], errors="coerce")
    cnn_pos = pd.to_numeric(data["cnn_positive_percentile_v8"], errors="coerce")
    vina_pos = pd.to_numeric(data["vina_positive_percentile_v8"], errors="coerce")
    cnn_adjusted_neg = pd.to_numeric(
        data["cnn_size_adjusted_negative_percentile_v8"], errors="coerce"
    )
    vina_adjusted_neg = pd.to_numeric(
        data["vina_size_adjusted_negative_percentile_v8"], errors="coerce"
    )
    cnn_not_size_artifact = cnn_adjusted_neg.ge(size_adjusted_floor)
    vina_not_size_artifact = vina_adjusted_neg.ge(size_adjusted_floor)
    dual = cnn_valid & vina_valid

    dual_both_strong = (
        dual
        & cnn_neg.ge(q90)
        & vina_neg.ge(q90)
        & cnn_pos.ge(positive_floor)
        & vina_pos.ge(positive_floor)
        & cnn_not_size_artifact
        & vina_not_size_artifact
    )
    dual_exceptional = (
        dual
        & (
            (
                cnn_neg.ge(q95)
                & vina_neg.ge(0.75)
                & cnn_not_size_artifact
            )
            | (
                vina_neg.ge(q95)
                & cnn_neg.ge(0.75)
                & vina_not_size_artifact
            )
        )
        & (cnn_pos.ge(positive_floor) | vina_pos.ge(positive_floor))
    )
    dual_one_strong = (
        dual
        & (
            (cnn_neg.ge(q90) & vina_neg.ge(q50) & cnn_not_size_artifact)
            | (vina_neg.ge(q90) & cnn_neg.ge(q50) & vina_not_size_artifact)
        )
        & (cnn_pos.ge(positive_floor) | vina_pos.ge(positive_floor))
    )
    single_cnn = cnn_valid & ~vina_valid
    single_vina = vina_valid & ~cnn_valid
    single_exceptional = (
        (
            single_cnn
            & cnn_neg.ge(q95)
            & cnn_pos.ge(positive_floor)
            & cnn_not_size_artifact
        )
        | (
            single_vina
            & vina_neg.ge(q95)
            & vina_pos.ge(positive_floor)
            & vina_not_size_artifact
        )
    )
    single_strong = (
        (
            single_cnn
            & cnn_neg.ge(q90)
            & cnn_pos.ge(positive_floor)
            & cnn_not_size_artifact
        )
        | (
            single_vina
            & vina_neg.ge(q90)
            & vina_pos.ge(positive_floor)
            & vina_not_size_artifact
        )
    )

    data["pair_evidence_tier_v8"] = "R_below_target_calibrated_floor"
    data.loc[dual_one_strong | single_strong, "pair_evidence_tier_v8"] = (
        "P3_single_channel_strong_noncontradictory"
    )
    data.loc[dual_exceptional | single_exceptional, "pair_evidence_tier_v8"] = (
        "P2_exceptional_primary_channel"
    )
    data.loc[dual_both_strong, "pair_evidence_tier_v8"] = "P1_dual_channel_strong"
    tier_order = {
        "P1_dual_channel_strong": 0,
        "P2_exceptional_primary_channel": 1,
        "P3_single_channel_strong_noncontradictory": 2,
        "R_below_target_calibrated_floor": 9,
    }
    data["pair_evidence_order_v8"] = data["pair_evidence_tier_v8"].map(tier_order)
    data["pair_primary_negative_percentile_v8"] = np.where(
        dual,
        np.minimum(cnn_neg, vina_neg),
        np.where(cnn_valid, cnn_neg, vina_neg),
    )
    data["pair_best_negative_percentile_v8"] = np.nanmax(
        np.column_stack([cnn_neg.to_numpy(), vina_neg.to_numpy()]), axis=1
    )
    data["pair_primary_positive_percentile_v8"] = np.where(
        dual,
        np.minimum(cnn_pos, vina_pos),
        np.where(cnn_valid, cnn_pos, vina_pos),
    )
    data["pair_best_size_adjusted_negative_percentile_v8"] = np.nanmax(
        np.column_stack([cnn_adjusted_neg.to_numpy(), vina_adjusted_neg.to_numpy()]), axis=1
    )
    data["target_calibrated_pair_supported_v8"] = data[
        "pair_evidence_tier_v8"
    ].str.startswith("P")
    return data


def merge_context(
    queue: pd.DataFrame,
    scores: pd.DataFrame,
    master_path: Path,
    current1000_path: Path,
) -> pd.DataFrame:
    data = queue.merge(
        scores,
        on=["physical_pair_id", "sequence_key", "primary_gene"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_score"),
    )
    master_columns = [
        "physical_pair_id",
        "any_known_fda_target_pair",
        "any_family_or_rediscovery_risk_v2",
        "severe_compound_liability",
        "same_known_active_scaffold_v5",
        "exact_known_active_smiles_v5",
        "max_known_active_similarity_v5",
        "expanded_exact_known_target_v1",
        "expanded_target_homology_risk_v1",
        "remote_pair_homology_audited_v1",
        "dta_stage1_strict_homology_audited_v1",
        "source_label_targets",
        "source_action_types",
    ]
    master_columns = ["physical_pair_id"] + [
        column for column in master_columns
        if column != "physical_pair_id" and column not in data.columns
    ]
    master = pd.read_csv(master_path, usecols=master_columns, low_memory=False)
    data = data.merge(master, on="physical_pair_id", how="left", validate="one_to_one")

    current_columns = [
        "model_ligand_smiles",
        "sequence_key",
        "v5_rank",
        "boltz_support_tier_refined",
        "boltz_affinity_probability_refined",
        "pose_stability_tier",
        "agent_feasibility_grade",
        "agent_verdict",
        "agent_literature_class",
        "agent_primary_disease",
        "review_verdict_v6",
    ]
    header = pd.read_csv(current1000_path, nrows=0).columns
    available = [column for column in current_columns if column in header]
    current = pd.read_csv(current1000_path, usecols=available, low_memory=False)
    current = current.drop_duplicates(["model_ligand_smiles", "sequence_key"])
    current["in_previous_final1000_v8"] = True
    data = data.merge(
        current,
        on=["model_ligand_smiles", "sequence_key"],
        how="left",
        validate="one_to_one",
    )
    data["in_previous_final1000_v8"] = as_bool(data["in_previous_final1000_v8"])
    return data


def validate_remote_contract(frame: pd.DataFrame) -> None:
    boolean_false = [
        "any_known_fda_target_pair",
        "any_family_or_rediscovery_risk_v2",
        "severe_compound_liability",
        "same_known_active_scaffold_v5",
        "exact_known_active_smiles_v5",
        "expanded_exact_known_target_v1",
        "expanded_target_homology_risk_v1",
    ]
    violations = {column: int(as_bool(frame[column]).sum()) for column in boolean_false}
    violations["not_remote_homology_audited"] = int(
        (~as_bool(frame["remote_pair_homology_audited_v1"])).sum()
    )
    violations["not_strict_structure_homology_audited"] = int(
        (~as_bool(frame["dta_stage1_strict_homology_audited_v1"])).sum()
    )
    similarity = pd.to_numeric(frame["max_known_active_similarity_v5"], errors="coerce")
    violations["known_active_similarity_ge_0_40"] = int(similarity.ge(0.40).sum())
    failed = {key: value for key, value in violations.items() if value}
    if failed:
        raise ValueError(f"Remote discovery contract failed: {failed}")


def lexicographic_order(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    target_order = {
        "T1_dual_strong": 0,
        "T2_dual_pass": 1,
        "T3_single_strong": 2,
        "T4_single_pass": 3,
    }
    retrieval_order = {
        "strong_two_model_consensus": 0,
        "broad_two_model_consensus": 1,
        "drugclip_bidirectional_top50": 2,
        "conplex_bidirectional_top50": 3,
        "single_model_or_exploration": 4,
    }
    data["target_admission_order_final_v8"] = data["target_admission_tier_v8"].map(
        target_order
    )
    data["receptor_order_final_v8"] = ~data["docking_receptor_source"].eq(
        "experimental_holo"
    )
    data["retrieval_order_final_v8"] = data["stage2_evidence_lane_v1"].map(
        retrieval_order
    )
    return data.sort_values(
        [
            "pair_evidence_order_v8",
            "target_admission_order_final_v8",
            "receptor_order_final_v8",
            "retrieval_order_final_v8",
            "pair_primary_negative_percentile_v8",
            "pair_best_size_adjusted_negative_percentile_v8",
            "pair_primary_positive_percentile_v8",
            "pair_best_negative_percentile_v8",
            "physical_pair_id",
        ],
        ascending=[True, True, True, True, False, False, False, False, True],
        kind="mergesort",
    )


def diverse_select(
    ordered: pd.DataFrame, config: dict[str, Any], count: int
) -> pd.DataFrame:
    portfolio = config["portfolio"]
    family_caps = {str(key): int(value) for key, value in portfolio["family_caps"].items()}
    counters: dict[str, Counter[str]] = {
        "target": Counter(),
        "ligand": Counter(),
        "scaffold": Counter(),
        "scaffold_target": Counter(),
        "family": Counter(),
    }
    selected: list[int] = []
    eligible = ordered[ordered["target_calibrated_pair_supported_v8"]].copy()
    for index, row in eligible.iterrows():
        target = clean(row["sequence_key"])
        ligand = clean(row["model_ligand_smiles"])
        scaffold = clean(row["murcko_scaffold"]) or "ACYCLIC_OR_EMPTY"
        family = clean(row["target_assay_family_v2"]) or "other_assayable"
        scaffold_target = f"{target}|{scaffold}"
        if counters["target"][target] >= int(portfolio["target_pair_cap"]):
            continue
        if counters["ligand"][ligand] >= int(portfolio["ligand_pair_cap"]):
            continue
        if counters["scaffold"][scaffold] >= int(portfolio["scaffold_pair_cap"]):
            continue
        if counters["scaffold_target"][scaffold_target] >= int(
            portfolio["scaffold_per_target_cap"]
        ):
            continue
        if counters["family"][family] >= family_caps.get(family, count):
            continue
        selected.append(index)
        counters["target"][target] += 1
        counters["ligand"][ligand] += 1
        counters["scaffold"][scaffold] += 1
        counters["scaffold_target"][scaffold_target] += 1
        counters["family"][family] += 1
        if len(selected) == count:
            break
    result = ordered.loc[selected].copy()
    result.insert(0, "discovery_rank_v8", range(1, len(result) + 1))
    return result


def build_teacher_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "discovery_rank_v8": "候选顺序",
        "source_drug_names": "FDA药物",
        "source_drug_ids": "药物ChEMBL",
        "primary_gene": "新靶点",
        "target_assay_family_v2": "靶点实验类型",
        "source_indications": "原适应症",
        "source_therapeutic_areas": "原治疗领域",
        "pair_evidence_tier_v8": "GNINA靶点内证据级别",
        "target_admission_tier_v8": "靶点校准级别",
        "docking_receptor_source": "受体结构来源",
        "cnn_negative_percentile_v8": "CNN高于阴性对照比例",
        "vina_negative_percentile_v8": "Vina高于阴性对照比例",
        "cnn_size_adjusted_negative_percentile_v8": "CNN去大小偏差后高于阴性比例",
        "vina_size_adjusted_negative_percentile_v8": "Vina去大小偏差后高于阴性比例",
        "ligand_heavy_atoms_v8": "配体重原子数",
        "cnn_positive_percentile_v8": "CNN在阳性对照中的分位",
        "vina_positive_percentile_v8": "Vina在阳性对照中的分位",
        "stage2_evidence_lane_v1": "DTA召回证据",
        "rank_within_active_moiety": "ConPLEx药物内名次",
        "drugclip_rank_within_ligand_v1": "DrugCLIP药物内名次",
        "max_known_active_similarity_v5": "与该靶点已知活性配体最大相似度",
        "in_previous_final1000_v8": "是否复用旧1000",
        "boltz_support_tier_refined": "既有Boltz支持",
        "pose_stability_tier": "既有Boltz姿态稳定性",
        "physical_pair_id": "物理Pair ID",
    }
    available = [column for column in columns if column in frame.columns]
    return frame[available].rename(columns=columns)


def build_measurement_package(
    final1000: pd.DataFrame,
    controls_manifest: pd.DataFrame,
    repeat_count: int = 40,
    preferred_targets: int = 80,
    max_targets: int = 100,
    min_candidates_per_target: int = 6,
) -> pd.DataFrame:
    target_stats = final1000.groupby("sequence_key", as_index=False).agg(
        candidate_count=("physical_pair_id", "size"),
        best_rank=("discovery_rank_v8", "min"),
        best_evidence_order=("pair_evidence_order_v8", "min"),
        primary_gene=("primary_gene", "first"),
    )
    target_stats = target_stats.sort_values(
        ["best_evidence_order", "best_rank", "candidate_count"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    eligible_stats = target_stats[
        target_stats["candidate_count"].ge(min_candidates_per_target)
    ].reset_index(drop=True)
    selected_targets: list[str] = []
    discovery_slots = 0
    for target_count in range(
        min(preferred_targets, len(eligible_stats)),
        min(max_targets, len(eligible_stats)) + 1,
    ):
        proposed = eligible_stats.head(target_count)
        proposed_slots = 1000 - repeat_count - 2 * target_count
        proposed_capacity = int(proposed["candidate_count"].sum())
        if proposed_slots > 0 and proposed_capacity >= proposed_slots:
            selected_targets = proposed["sequence_key"].tolist()
            discovery_slots = proposed_slots
            break
    if not selected_targets:
        raise RuntimeError(
            "Could not satisfy per-target discovery coverage in the 1000-measurement package"
        )

    candidates = final1000[final1000["sequence_key"].isin(selected_targets)].copy()
    candidates = candidates.sort_values("discovery_rank_v8", kind="mergesort")
    base = candidates.groupby("sequence_key", sort=False).head(min_candidates_per_target)
    remaining_slots = discovery_slots - len(base)
    remaining = candidates.loc[~candidates.index.isin(base.index)].head(remaining_slots)
    candidates = pd.concat([base, remaining]).sort_values("discovery_rank_v8")
    if (
        len(candidates) != discovery_slots
        or candidates.groupby("sequence_key").size().min() < min_candidates_per_target
    ):
        raise RuntimeError("Controlled measurement package lost per-target candidate coverage")
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        rows.append(
            {
                "measurement_role_v8": "discovery_candidate",
                "physical_pair_id": row["physical_pair_id"],
                "drug_or_control_name": clean(row["source_drug_names"]),
                "smiles": clean(row["model_ligand_smiles"]),
                "sequence_key": row["sequence_key"],
                "primary_gene": row["primary_gene"],
                "source_discovery_rank_v8": int(row["discovery_rank_v8"]),
                "replicate_of_measurement_id": "",
            }
        )

    controls = controls_manifest[
        controls_manifest["sequence_key"].isin(selected_targets)
        & controls_manifest["control_class"].isin(["positive", "negative"])
    ].copy()
    controls = controls.sort_values(
        ["sequence_key", "control_class", "target_control_rank"], kind="mergesort"
    ).groupby(["sequence_key", "control_class"], as_index=False).head(1)
    if len(controls) != 2 * len(selected_targets):
        missing = 2 * len(selected_targets) - len(controls)
        raise RuntimeError(f"Measurement controls incomplete: missing {missing}")
    for _, row in controls.iterrows():
        rows.append(
            {
                "measurement_role_v8": f"{row['control_class']}_control",
                "physical_pair_id": clean(row["control_pair_id"]),
                "drug_or_control_name": clean(row.get("parent_molecule_chembl_id")),
                "smiles": clean(row["canonical_control_smiles"]),
                "sequence_key": row["sequence_key"],
                "primary_gene": row["primary_gene"],
                "source_discovery_rank_v8": "",
                "replicate_of_measurement_id": "",
            }
        )

    base = pd.DataFrame(rows)
    base.insert(0, "measurement_id_v8", [f"M{i:04d}" for i in range(1, len(base) + 1)])
    repeat_sources = pd.concat(
        [
            base[base["measurement_role_v8"].eq("discovery_candidate")].head(repeat_count // 2),
            base[base["measurement_role_v8"].eq("positive_control")].head(
                repeat_count - repeat_count // 2
            ),
        ],
        ignore_index=True,
    )
    repeat_rows = []
    for _, row in repeat_sources.iterrows():
        repeated = row.to_dict()
        repeated["measurement_role_v8"] = "technical_repeat"
        repeated["replicate_of_measurement_id"] = row["measurement_id_v8"]
        repeat_rows.append(repeated)
    repeats = pd.DataFrame(repeat_rows)
    repeats["measurement_id_v8"] = [
        f"M{i:04d}" for i in range(len(base) + 1, len(base) + len(repeats) + 1)
    ]
    result = pd.concat([base, repeats], ignore_index=True)
    if len(result) != 1000:
        raise RuntimeError(f"Measurement package row contract failed: {len(result)}")
    result.insert(1, "plate_id_v8", ((np.arange(len(result)) // 96) + 1).astype(int))
    result.insert(2, "well_index_v8", ((np.arange(len(result)) % 96) + 1).astype(int))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--discovery-scores", default="")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    inputs = {key: ROOT / value for key, value in config["inputs"].items()}
    output_dir = ROOT / config["outputs"]["directory"] / "final_package"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = ROOT / config["outputs"]["directory"]
    queue_path = base / "discovery_queue/GNINA_REMOTE_DISCOVERY_QUEUE_V8.csv.gz"
    scores_path = (
        Path(args.discovery_scores).resolve()
        if args.discovery_scores
        else base / "gnina_discovery/GNINA_REMOTE_DISCOVERY_PAIR_SCORES_V8.csv.gz"
    )

    queue = pd.read_csv(queue_path, low_memory=False)
    scores = pd.read_csv(scores_path, low_memory=False)
    if scores["physical_pair_id"].duplicated().any():
        raise ValueError("Discovery score table has duplicate physical pair IDs")
    if len(scores) != len(queue) and not args.allow_partial:
        raise ValueError(f"Incomplete discovery scores: {len(scores)} != {len(queue)}")

    latest_calibration = pd.read_csv(
        base / "target_calibration/GNINA_TARGET_CHANNEL_CALIBRATION_V8.csv",
        low_memory=False,
    )
    calibration_columns = [
        "sequence_key",
        "target_admission_tier_v8",
        "target_admitted_v8",
        "cnn_affinity_pass_v8",
        "cnn_affinity_strong_v8",
        "vina_affinity_pass_v8",
        "vina_affinity_strong_v8",
        "auroc_cnn_affinity_v8",
        "average_precision_cnn_affinity_v8",
        "auroc_vina_affinity_v8",
        "average_precision_vina_affinity_v8",
    ]
    stale_columns = [
        column for column in calibration_columns if column != "sequence_key" and column in queue
    ]
    queue = queue.drop(columns=stale_columns).merge(
        latest_calibration[calibration_columns],
        on="sequence_key",
        how="left",
        validate="many_to_one",
    )
    queue = queue[as_bool(queue["target_admitted_v8"])].copy()
    data = merge_context(queue, scores, inputs["physical_universe"], inputs["current1000"])
    data = data[data["best_cnn_affinity"].notna() & data["best_vina_affinity"].notna()].copy()
    validate_remote_contract(data)

    control_scores = pd.read_csv(inputs["gnina_control_scores"], low_memory=False)
    controls_manifest = pd.read_csv(inputs["gnina_controls"], low_memory=False)
    data, control_scores, size_adjustment = add_size_adjusted_scores(
        data, control_scores, controls_manifest
    )
    data = normalize_against_controls(data, control_scores)
    data = assign_pair_tiers(data, config)
    data["disease_evidence_role_v8"] = "post_binding_annotation_only"
    ordered = lexicographic_order(data)
    all_path = output_dir / "REMOTE_DISCOVERY_ALL_TARGET_NORMALIZED_V8.csv.gz"
    ordered.to_csv(all_path, index=False, compression={"method": "gzip", "compresslevel": 5})

    requested = int(config["portfolio"]["discovery_candidates"])
    final1000 = diverse_select(ordered, config, requested)
    if len(final1000) != requested:
        raise RuntimeError(
            f"Only {len(final1000)} target-calibrated remote candidates meet frozen evidence and diversity rules"
        )
    final_path = output_dir / "DISCOVERY1000_AFFINITY_FIRST_V8.csv"
    final1000.to_csv(final_path, index=False)
    teacher = build_teacher_table(final1000)
    teacher_path = output_dir / "DISCOVERY1000_TEACHER_READABLE_ZH_V8.csv"
    teacher.to_csv(teacher_path, index=False)

    measurements = build_measurement_package(final1000, controls_manifest)
    measurement_path = output_dir / "ASSAY_MEASUREMENT1000_WITH_CONTROLS_V8.csv"
    measurements.to_csv(measurement_path, index=False)

    workbook_path = output_dir / "AFFINITY_EXPERIMENT_PACKAGE_V8.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        teacher.to_excel(writer, sheet_name="1000个发现候选", index=False)
        measurements.to_excel(writer, sheet_name="1000次实验测量包", index=False)
        pd.read_csv(
            base / "target_calibration/GNINA_TARGET_CHANNEL_CALIBRATION_V8.csv"
        ).to_excel(writer, sheet_name="靶点GNINA校准", index=False)

    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scored_remote_pairs": int(len(data)),
        "target_calibrated_supported_pairs": int(
            data["target_calibrated_pair_supported_v8"].sum()
        ),
        "pair_evidence_tier_counts_all": data["pair_evidence_tier_v8"].value_counts().to_dict(),
        "discovery_candidates": int(len(final1000)),
        "discovery_unique_ligands": int(final1000["model_ligand_smiles"].nunique()),
        "discovery_unique_targets": int(final1000["sequence_key"].nunique()),
        "discovery_unique_scaffolds": int(final1000["murcko_scaffold"].nunique()),
        "discovery_family_counts": final1000["target_assay_family_v2"].value_counts().to_dict(),
        "discovery_evidence_tier_counts": final1000["pair_evidence_tier_v8"].value_counts().to_dict(),
        "previous1000_overlap": int(final1000["in_previous_final1000_v8"].sum()),
        "measurement_rows": int(len(measurements)),
        "measurement_role_counts": measurements["measurement_role_v8"].value_counts().to_dict(),
        "size_adjustment": size_adjustment,
        "policy": {
            "affinity_selection": "target-wise calibrated raw and size-adjusted GNINA evidence, lexicographic tiers, no weighted sum",
            "novelty": "exact known, same scaffold/high similarity and known-target homology excluded",
            "disease": "post-binding annotation only",
            "current1000": "reuse annotation only",
        },
        "outputs": {
            "all_scored": str(all_path.relative_to(ROOT)),
            "discovery1000": str(final_path.relative_to(ROOT)),
            "teacher_csv": str(teacher_path.relative_to(ROOT)),
            "measurement1000": str(measurement_path.relative_to(ROOT)),
            "workbook": str(workbook_path.relative_to(ROOT)),
        },
    }
    summary_path = output_dir / "AFFINITY_EXPERIMENT_PACKAGE_V8_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a strengthened, non-destructive v5 candidate package from v4 Top3000.

The package separates pair physics, experimental readiness and novelty.  It
uses the known-positive Boltz96 panel only as a positive-calibration reference;
the resulting tiers are not binding probabilities and do not estimate FDR.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOP3000 = ROOT / "outputs/current_production_package_v2/formal_full_universe_v4/refined_top3000_v4_complete.csv"
CURRENT1000 = ROOT / "outputs/current_production_package_v2/final_delivery_v4/FINAL1000_RESERVE_FULL_V4.csv"
REVIEW512 = ROOT / "outputs/current_production_package_v2/final_delivery_v4/REVIEW512_POST_REVIEW_DISPOSITION_FULL_V4.csv"
KNOWN96 = ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_calibration_v4.csv"
OUT = ROOT / "outputs/current_production_package_v2/final1000_strengthened_v5"
REPORT = ROOT / "docs/FINAL1000_STRENGTHENED_V5_AUDIT_ZH.md"


FAMILY_CAPS_PRIMARY = {
    "enzyme": 650,
    "kinase": 200,
    "transporter": 200,
    "nuclear_epigenetic": 220,
    "ion_channel": 90,
    "other_assayable": 50,
}


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "1.0", "yes"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_percentile(values: pd.Series, reference: pd.Series) -> np.ndarray:
    ref = np.sort(pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float))
    val = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if not len(ref):
        return np.full(len(val), 0.5, dtype=float)
    return np.searchsorted(ref, val, side="right") / len(ref)


def calibration_thresholds(known: pd.DataFrame) -> pd.DataFrame:
    affinity = pd.to_numeric(known["boltz_affinity_probability_refined"], errors="coerce")
    pooled = affinity.quantile([0.10, 0.25, 0.50]).to_dict()
    rows: list[dict[str, Any]] = []
    families = sorted(set(known["target_assay_family_v2"].map(clean)) - {""})
    for family in families:
        values = affinity[known["target_assay_family_v2"].map(clean).eq(family)].dropna()
        use_family = len(values) >= 15
        rows.append(
            {
                "target_assay_family_v2": family,
                "known_positive_rows": int(len(values)),
                "threshold_source": "family_known_positive" if use_family else "pooled_known_positive_n_lt_15",
                "known_positive_q10": float(values.quantile(0.10) if use_family else pooled[0.10]),
                "known_positive_q25": float(values.quantile(0.25) if use_family else pooled[0.25]),
                "known_positive_median": float(values.quantile(0.50) if use_family else pooled[0.50]),
            }
        )
    return pd.DataFrame(rows)


def add_axes(frame: pd.DataFrame, reference: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    affinity = pd.to_numeric(out["boltz_affinity_probability_refined"], errors="coerce").fillna(0.0)
    out["v5_conplex_axis"] = (
        pd.to_numeric(out["pair_conplex_component_v2"], errors="coerce").fillna(0.0) / 25.0
    ).clip(0, 1)
    out["v5_boltz_affinity_raw"] = affinity.clip(0, 1)
    out["v5_boltz_global_percentile"] = empirical_percentile(
        affinity, reference["boltz_affinity_probability_refined"]
    )

    family_percentiles: list[float] = []
    target_percentiles: list[float] = []
    target_sizes: list[int] = []
    reference_affinity = pd.to_numeric(reference["boltz_affinity_probability_refined"], errors="coerce")
    for family, target, value in zip(
        out["target_assay_family_v2"].map(clean), out["sequence_key"].map(clean), affinity
    ):
        family_ref = reference_affinity[reference["target_assay_family_v2"].map(clean).eq(family)].dropna()
        target_ref = reference_affinity[reference["sequence_key"].map(clean).eq(target)].dropna()
        family_pct = float(empirical_percentile(pd.Series([value]), family_ref)[0])
        target_pct = float(empirical_percentile(pd.Series([value]), target_ref)[0])
        target_n = int(len(target_ref))
        shrunk_target_pct = (target_n / (target_n + 10.0)) * target_pct + (
            10.0 / (target_n + 10.0)
        ) * family_pct
        family_percentiles.append(family_pct)
        target_percentiles.append(shrunk_target_pct)
        target_sizes.append(target_n)
    out["v5_boltz_family_percentile"] = family_percentiles
    out["v5_boltz_target_percentile_shrunk"] = target_percentiles
    out["v5_boltz_target_reference_n"] = target_sizes
    out["v5_boltz_calibrated_axis"] = (
        0.80 * out["v5_boltz_affinity_raw"]
        + 0.10 * out["v5_boltz_global_percentile"]
        + 0.07 * out["v5_boltz_family_percentile"]
        + 0.03 * out["v5_boltz_target_percentile_shrunk"]
    ).clip(0, 1)
    # Boltz receives the larger weight because the positive-control audit
    # separates known positives from the unlabeled Top3000 more clearly.  This
    # is a prioritisation weight, not a calibrated probability.
    out["v5_pair_physics_score"] = 100.0 * (
        0.70 * out["v5_boltz_calibrated_axis"] + 0.30 * out["v5_conplex_axis"]
    )
    out["v5_cross_model_consensus"] = 100.0 * np.minimum(
        out["v5_boltz_calibrated_axis"], out["v5_conplex_axis"]
    )
    out["v5_cross_model_disagreement"] = 100.0 * (
        out["v5_boltz_calibrated_axis"] - out["v5_conplex_axis"]
    ).abs()

    out = out.merge(thresholds, on="target_assay_family_v2", how="left", validate="many_to_one")
    pooled = thresholds[
        ["known_positive_q10", "known_positive_q25", "known_positive_median"]
    ].median()
    for column in ["known_positive_q10", "known_positive_q25", "known_positive_median"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(float(pooled[column]))
    out["v5_boltz_positive_calibration_tier"] = "K0_below_known_positive_q10"
    out.loc[
        affinity >= out["known_positive_q10"], "v5_boltz_positive_calibration_tier"
    ] = "K1_ge_known_positive_q10"
    out.loc[
        affinity >= out["known_positive_q25"], "v5_boltz_positive_calibration_tier"
    ] = "K2_ge_known_positive_q25"
    out.loc[
        affinity >= out["known_positive_median"], "v5_boltz_positive_calibration_tier"
    ] = "K3_ge_known_positive_median"
    out["v5_passes_known_positive_q10_floor"] = affinity >= out["known_positive_q10"]
    out["v5_strength_tier"] = out["v5_boltz_positive_calibration_tier"].map(
        {
            "K3_ge_known_positive_median": "A_at_or_above_known_positive_median",
            "K2_ge_known_positive_q25": "B_at_or_above_known_positive_q25",
            "K1_ge_known_positive_q10": "C_at_or_above_known_positive_q10",
            "K0_below_known_positive_q10": "R_below_positive_floor_rescue_only",
        }
    )

    pose = out["pose_stability_tier"].fillna("").astype(str)
    out["v5_pose_primary_ready"] = pose.str.startswith(("A_", "B_"), na=False)
    out["v5_model_agreement_lane"] = "moderate_or_discordant"
    out.loc[
        (out["v5_conplex_axis"] >= 0.95) & (out["v5_boltz_calibrated_axis"] >= 0.65),
        "v5_model_agreement_lane",
    ] = "concordant_high"
    out.loc[
        (out["v5_conplex_axis"] < 0.95) & (out["v5_boltz_calibrated_axis"] >= 0.65),
        "v5_model_agreement_lane",
    ] = "boltz_led"
    out.loc[
        (out["v5_conplex_axis"] >= 0.95) & (out["v5_boltz_calibrated_axis"] < 0.65),
        "v5_model_agreement_lane",
    ] = "conplex_led"

    out["v5_experimental_readiness_score"] = 100.0 * (
        0.40
        * (pd.to_numeric(out["drug_feasibility_component_v2"], errors="coerce").fillna(0) / 10.0)
        + 0.30
        * (pd.to_numeric(out["experimental_feasibility_component_v2"], errors="coerce").fillna(0) / 5.0)
        + 0.20
        * (pd.to_numeric(out["target_tractability_component_v2"], errors="coerce").fillna(0) / 10.0)
        + 0.10
        * (pd.to_numeric(out["target_pocket_prior_component_v2"], errors="coerce").fillna(0) / 15.0)
    ).clip(0, 100)
    out["v5_novelty_lane"] = np.where(
        as_bool(out["same_assay_family_only"]),
        "same_broad_assay_family_context",
        "cross_assay_family_or_no_known_family",
    )
    return out


def add_review_context(frame: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    review_columns = [
        "pair_id",
        "post_review_disposition_v4",
        "agent_feasibility_grade",
        "agent_literature_class",
        "agent_active_species_status",
        "agent_repurposing_status",
        "agent_primary_disease",
        "agent_confidence",
        "chembl_exact_activity_status",
        "chembl_exact_max_binding_pchembl",
        "chembl_activity_query_ok",
        "lit_ok",
    ]
    available = [column for column in review_columns if column in review.columns]
    out = frame.merge(
        review[available].drop_duplicates("pair_id"), on="pair_id", how="left", validate="one_to_one"
    )
    out["v5_review_status"] = out.get(
        "post_review_disposition_v4", pd.Series("", index=out.index)
    ).fillna("").replace("", "not_yet_in_review512")
    exact_chembl = out.get("chembl_exact_activity_status", pd.Series("", index=out.index)).fillna("").eq(
        "exact_binding_activity_pchembl_ge_5"
    )
    exact_literature = out.get("agent_literature_class", pd.Series("", index=out.index)).fillna("").eq(
        "exact_pair_validated"
    )
    out["v5_candidate_role"] = "discovery_hypothesis"
    out.loc[exact_chembl | exact_literature, "v5_candidate_role"] = "validated_control_or_rediscovery"
    return out


def diverse_select(frame: pd.DataFrame, n: int, family_caps: dict[str, int]) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["v5_pair_physics_score", "v5_boltz_affinity_raw", "v5_conplex_axis", "pair_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected: list[int] = []
    counts: dict[str, dict[str, int]] = {key: {} for key in ["drug", "target", "scaffold", "family"]}

    def count(kind: str, key: str) -> int:
        return counts[kind].get(key, 0)

    for idx, row in ordered.iterrows():
        drug = clean(row.get("model_ligand_smiles")) or clean(row.get("active_moiety_smiles")) or clean(
            row.get("drug_chembl_id")
        )
        target = clean(row.get("sequence_key"))
        scaffold = clean(row.get("murcko_scaffold")) or drug or "NO_SCAFFOLD"
        family = clean(row.get("target_assay_family_v2")) or "other_assayable"
        if (
            count("drug", drug) >= 6
            or count("target", target) >= 14
            or count("scaffold", scaffold) >= 20
            or count("family", family) >= family_caps.get(family, n)
        ):
            continue
        selected.append(idx)
        for kind, key in [("drug", drug), ("target", target), ("scaffold", scaffold), ("family", family)]:
            counts[kind][key] = count(kind, key) + 1
        if len(selected) >= n:
            break
    result = frame.loc[selected].copy()
    result = result.sort_values(
        ["v5_pair_physics_score", "v5_boltz_affinity_raw", "pair_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    result.insert(0, "v5_rank", range(1, len(result) + 1))
    return result


def compact_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "v5_rank": "V5顺序",
        "fda_generic_name": "药物通用名",
        "drug_chembl_id": "药物ChEMBL",
        "primary_gene": "候选靶点",
        "target_assay_family_v2": "实验类型",
        "v5_pair_physics_score": "V5物理优先分",
        "v5_strength_tier": "V5强度分层",
        "v5_boltz_positive_calibration_tier": "Boltz阳性校准档",
        "v5_model_agreement_lane": "双模型关系",
        "conplex_score": "ConPLEx原始分",
        "rank_within_drug": "ConPLEx药物内排名",
        "target_rank": "ConPLEx靶点内排名",
        "boltz_affinity_probability_refined": "Boltz affinity",
        "pose_stability_tier": "条件姿势稳定性",
        "v5_experimental_readiness_score": "实验准备度",
        "v5_candidate_role": "候选角色",
        "v5_novelty_lane": "新颖性语境",
        "v5_review_status": "既有审阅状态",
        "agent_feasibility_grade": "Agent可行性等级",
        "agent_literature_class": "文献类别",
        "agent_repurposing_status": "老药新用类型",
        "agent_primary_disease": "建议疾病",
        "fda_indication": "原适应症",
        "default_assay_strategy": "建议首轮实验",
        "pair_id": "Pair ID",
    }
    available = [column for column in columns if column in frame.columns]
    return frame[available].rename(columns=columns)


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    affinity = pd.to_numeric(frame["boltz_affinity_probability_refined"], errors="coerce")
    return {
        "rows": int(len(frame)),
        "unique_model_ligands": int(frame["model_ligand_smiles"].nunique()),
        "unique_drug_ids": int(frame["drug_chembl_id"].nunique()),
        "unique_targets": int(frame["sequence_key"].nunique()),
        "unique_scaffolds": int(frame["murcko_scaffold"].replace("", np.nan).nunique()),
        "affinity_q10": float(affinity.quantile(0.10)),
        "affinity_q25": float(affinity.quantile(0.25)),
        "affinity_median": float(affinity.median()),
        "conplex_median": float(pd.to_numeric(frame["conplex_score"], errors="coerce").median()),
        "pose_ab_rows": int(frame["pose_stability_tier"].fillna("").astype(str).str.startswith(("A_", "B_")).sum()),
        "below_known_positive_q10_rows": int((~as_bool(frame["v5_passes_known_positive_q10_floor"])).sum()),
        "family_counts": frame["target_assay_family_v2"].value_counts().to_dict(),
        "calibration_tier_counts": frame["v5_boltz_positive_calibration_tier"].value_counts().to_dict(),
        "strength_tier_counts": frame["v5_strength_tier"].value_counts().to_dict(),
        "agreement_lane_counts": frame["v5_model_agreement_lane"].value_counts().to_dict(),
        "review_status_counts": frame["v5_review_status"].value_counts().to_dict(),
        "candidate_role_counts": frame["v5_candidate_role"].value_counts().to_dict(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    top = pd.read_csv(TOP3000, low_memory=False)
    current = pd.read_csv(CURRENT1000, low_memory=False)
    review = pd.read_csv(REVIEW512, low_memory=False)
    known = pd.read_csv(KNOWN96, low_memory=False)
    thresholds = calibration_thresholds(known)
    enhanced = add_review_context(add_axes(top, top, thresholds), review)

    reviewed_fail_pairs = set(
        review.loc[
            review["post_review_disposition_v4"].fillna("").astype(str).str.startswith("excluded_"),
            "pair_id",
        ].astype(str)
    )
    exact_nonbinding_pairs = set(
        review.loc[
            review["chembl_exact_activity_status"].fillna("").eq("exact_nonbinding_activity_record"),
            "pair_id",
        ].astype(str)
    )
    metabolite_review = review[
        review["agent_active_species_status"].fillna("").eq("prodrug_active_metabolite_requires_rerun")
    ]
    held_drug_ids = set(metabolite_review["drug_chembl_id"].map(clean)) - {""}
    held_ligands = set(metabolite_review["model_ligand_smiles"].map(clean)) - {""}
    enhanced["v5_review_fail_closed"] = enhanced["pair_id"].astype(str).isin(
        reviewed_fail_pairs | exact_nonbinding_pairs
    )
    enhanced["v5_active_species_rerun_hold"] = enhanced["drug_chembl_id"].map(clean).isin(
        held_drug_ids
    ) | enhanced["model_ligand_smiles"].map(clean).isin(held_ligands)
    enhanced["v5_hard_eligible"] = (
        as_bool(enhanced["boltz_completed_refined"])
        & ~as_bool(enhanced["exact_known_target_v2"])
        & ~as_bool(enhanced["family_or_rediscovery_risk_v2"])
        & ~as_bool(enhanced["severe_compound_liability"])
        & ~as_bool(enhanced["structure_sequence_mismatch_v4"])
        & ~enhanced["v5_review_fail_closed"]
        & ~enhanced["v5_active_species_rerun_hold"]
    )
    primary_pool = enhanced[
        enhanced["v5_hard_eligible"]
        & enhanced["v5_pose_primary_ready"]
        & enhanced["v5_passes_known_positive_q10_floor"]
    ].copy()
    selected = diverse_select(primary_pool, 1000, FAMILY_CAPS_PRIMARY)
    if len(selected) < 1000:
        # Preserve the physical floor and pose contract. Relax only family
        # composition, never pair-level scientific gates.
        selected = diverse_select(
            primary_pool,
            1000,
            {family: 1000 for family in FAMILY_CAPS_PRIMARY},
        )
        selected["v5_family_cap_relaxed"] = True
    else:
        selected["v5_family_cap_relaxed"] = False
    if len(selected) != 1000:
        raise RuntimeError(f"Unable to build strengthened 1000 under v5 hard contracts: {len(selected)}")

    selected_keys = set(selected["pair_id"].astype(str))
    rescue_pool = enhanced[
        enhanced["v5_hard_eligible"]
        & ~enhanced["pair_id"].astype(str).isin(selected_keys)
        & (
            (
                (enhanced["v5_conplex_axis"] >= 0.97)
                & ~enhanced["v5_passes_known_positive_q10_floor"]
            )
            | (
                ~enhanced["v5_pose_primary_ready"]
                & (enhanced["v5_boltz_affinity_raw"] >= 0.75)
                & (enhanced["v5_conplex_axis"] >= 0.90)
            )
        )
    ].copy()
    rescue_pool["v5_rescue_reason"] = np.where(
        ~rescue_pool["v5_pose_primary_ready"],
        "high_boltz_but_pose_unstable",
        "high_conplex_below_boltz_positive_floor",
    )
    rescue = diverse_select(
        rescue_pool,
        min(200, len(rescue_pool)),
        {family: 200 for family in FAMILY_CAPS_PRIMARY},
    ).rename(columns={"v5_rank": "v5_rescue_rank"})

    current_keys = set(current["pair_id"].astype(str))
    selected["in_current_final1000_v4"] = selected["pair_id"].astype(str).isin(current_keys)
    enhanced["in_strengthened_final1000_v5"] = enhanced["pair_id"].astype(str).isin(selected_keys)
    replacement = enhanced[
        enhanced["pair_id"].astype(str).isin(current_keys ^ selected_keys)
    ].copy()
    replacement["v5_change"] = np.where(
        replacement["pair_id"].astype(str).isin(selected_keys), "added_in_v5", "removed_from_v4"
    )
    removal_reason = []
    for row in replacement.itertuples(index=False):
        if row.v5_change == "added_in_v5":
            removal_reason.append("higher_v5_physics_under_contracts")
        elif bool(row.v5_review_fail_closed):
            removal_reason.append("review_fail_closed")
        elif bool(row.v5_active_species_rerun_hold):
            removal_reason.append("active_species_requires_rerun")
        elif not bool(row.v5_passes_known_positive_q10_floor):
            removal_reason.append("below_known_positive_q10_floor")
        elif not bool(row.v5_pose_primary_ready):
            removal_reason.append("pose_not_A_or_B")
        else:
            removal_reason.append("rank_or_diversity_replacement")
    replacement["v5_change_reason"] = removal_reason

    selected.to_csv(OUT / "FINAL1000_STRENGTHENED_V5.csv", index=False)
    compact_table(selected).to_csv(OUT / "FINAL1000_STRENGTHENED_V5_TEACHER_READABLE_ZH.csv", index=False)
    selected[selected["v5_review_status"].eq("not_yet_in_review512")].to_csv(
        OUT / "V5_UNREVIEWED625_REVIEW_QUEUE.csv", index=False
    )
    selected[~selected["v5_review_status"].eq("not_yet_in_review512")].to_csv(
        OUT / "V5_EXISTING_REVIEW_COVERED375.csv", index=False
    )
    rescue.to_csv(OUT / "HIGH_RECALL_RESCUE_V5.csv", index=False)
    replacement.to_csv(OUT / "V4_TO_V5_REPLACEMENT_AUDIT.csv", index=False)
    thresholds.to_csv(OUT / "BOLTZ_KNOWN96_FAMILY_CALIBRATION_THRESHOLDS.csv", index=False)

    current_enhanced = add_review_context(add_axes(current, top, thresholds), review)
    current_enhanced["v5_active_species_rerun_hold"] = current_enhanced["drug_chembl_id"].map(clean).isin(
        held_drug_ids
    ) | current_enhanced["model_ligand_smiles"].map(clean).isin(held_ligands)
    current_enhanced["v5_review_fail_closed"] = current_enhanced["pair_id"].astype(str).isin(
        reviewed_fail_pairs | exact_nonbinding_pairs
    )
    summary = {
        "version": "v5_strengthened_non_destructive",
        "inputs": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path)}
            for path in [TOP3000, CURRENT1000, REVIEW512, KNOWN96]
        },
        "formula": {
            "boltz_calibrated_axis": "0.80 raw affinity + 0.10 global ECDF + 0.07 assay-family ECDF + 0.03 shrunk target ECDF",
            "pair_physics_score": "100 * (0.70 Boltz calibrated axis + 0.30 ConPLEx composite axis)",
            "positive_floor": "family known-positive q10 when n>=15; otherwise pooled known96 q10",
            "interpretation": "positive calibration only; no specificity, precision, FDR or binding probability",
        },
        "hard_contracts": {
            "known_positive_q10_floor": True,
            "pose_tier_A_or_B": True,
            "review_fail_closed": True,
            "active_metabolite_status_propagated_by_drug_and_model_ligand": True,
            "exact_known_family_risk_severe_liability_sequence_mismatch_excluded": True,
        },
        "current_v4": metrics(current_enhanced.assign(v5_passes_known_positive_q10_floor=current_enhanced["v5_passes_known_positive_q10_floor"])),
        "strengthened_v5": metrics(selected),
        "overlap": {
            "retained_from_v4": int(selected["in_current_final1000_v4"].sum()),
            "added_in_v5": int((~selected["in_current_final1000_v4"]).sum()),
            "removed_from_v4": int(len(current_keys - selected_keys)),
        },
        "review_and_active_species": {
            "review_fail_pairs_in_v4": int(current_enhanced["v5_review_fail_closed"].sum()),
            "active_species_hold_rows_in_v4": int(current_enhanced["v5_active_species_rerun_hold"].sum()),
            "held_drug_ids": len(held_drug_ids),
            "held_model_ligands": len(held_ligands),
        },
        "rescue_rows": int(len(rescue)),
    }
    (OUT / "STRENGTHENED_V5_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    old = summary["current_v4"]
    new = summary["strengthened_v5"]
    lines = [
        "# Final1000 v5 增强审计",
        "",
        "## 定位",
        "",
        "v5 不覆盖 v4 正式文件。它从已完成 Boltz 的 v4 Top3000 重新构建一个更严格的亲和优先1000包，并把模型分歧候选保留在独立 rescue 表。",
        "",
        "## 主要增强",
        "",
        "1. 结合分只由 ConPLEx 与 Boltz pair-level 证据组成；口袋、Open Targets、药物可行性和 assay 可行性改为独立实验准备度。",
        "2. 按 assay family 使用 known96 阳性集的 q10/q25/median 做正例参照；主1000至少达到q10。该门只说明不低于90%阳性参照区间，不估计假阳性率。",
        "3. 主1000要求双样本条件姿势 A/B。高分但姿势不稳或 Boltz 低而 ConPLEx 高的pair进入 rescue，不直接删除。",
        "4. review512 中明确矛盾、D级、数据库失败和 exact nonbinding 采用 fail-closed。活性代谢物重算问题传播到同药物/同模型配体的全部pair。",
        "",
        "## v4 与 v5",
        "",
        "| 指标 | v4 final1000 | v5 strengthened1000 |",
        "|---|---:|---:|",
        f"| 行数 | {old['rows']} | {new['rows']} |",
        f"| 唯一模型配体 | {old['unique_model_ligands']} | {new['unique_model_ligands']} |",
        f"| 唯一靶点 | {old['unique_targets']} | {new['unique_targets']} |",
        f"| Murcko骨架 | {old['unique_scaffolds']} | {new['unique_scaffolds']} |",
        f"| Boltz affinity中位数 | {old['affinity_median']:.3f} | {new['affinity_median']:.3f} |",
        f"| Boltz affinity第10百分位 | {old['affinity_q10']:.3f} | {new['affinity_q10']:.3f} |",
        f"| 姿势A/B | {old['pose_ab_rows']} | {new['pose_ab_rows']} |",
        f"| 低于known-positive q10 | {old['below_known_positive_q10_rows']} | {new['below_known_positive_q10_rows']} |",
        f"| v4保留/替换 | - | {summary['overlap']['retained_from_v4']} / {summary['overlap']['added_in_v5']} |",
        "",
        "## v5 分层与审阅覆盖",
        "",
        f"- A档（达到同类known-positive中位数）：{new['strength_tier_counts'].get('A_at_or_above_known_positive_median', 0)}条。",
        f"- B档（达到known-positive第25百分位）：{new['strength_tier_counts'].get('B_at_or_above_known_positive_q25', 0)}条。",
        f"- C档（达到known-positive第10百分位）：{new['strength_tier_counts'].get('C_at_or_above_known_positive_q10', 0)}条。",
        f"- 已有review512覆盖且未触发排除：{new['rows'] - new['review_status_counts'].get('not_yet_in_review512', 0)}条；仍待逐条审阅：{new['review_status_counts'].get('not_yet_in_review512', 0)}条。",
        "",
        "## 限制",
        "",
        "- known96 只有阳性、没有可靠阴性，因此q10门不能给出precision或FDR。",
        "- v5只能在已运行Boltz的Top3000内增强，不能找回Top3000外的334,749空间。",
        "- 未进入review512的候选仍需文献、活性物种、暴露和assay逐条审阅。",
        "- 疾病证据不进入物理排序，需在互作hit后研判。",
        "",
        "## 输出",
        "",
        "- `FINAL1000_STRENGTHENED_V5.csv`：增强主包。",
        "- `FINAL1000_STRENGTHENED_V5_TEACHER_READABLE_ZH.csv`：简明中文表。",
        "- `HIGH_RECALL_RESCUE_V5.csv`：模型分歧/姿势不稳救援池。",
        "- `V5_UNREVIEWED625_REVIEW_QUEUE.csv`：尚未逐条审阅的后续队列。",
        "- `V5_EXISTING_REVIEW_COVERED375.csv`：已有review512覆盖且未触发排除的子集。",
        "- `V4_TO_V5_REPLACEMENT_AUDIT.csv`：逐条替换原因。",
        "- `BOLTZ_KNOWN96_FAMILY_CALIBRATION_THRESHOLDS.csv`：阳性校准阈值。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

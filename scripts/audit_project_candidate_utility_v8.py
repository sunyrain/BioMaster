#!/usr/bin/env python3
"""Audit candidate utility independently of the legacy A/B/C/D translation grade."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_final1000_strengthened_v5 as v5  # noqa: E402


FINAL1000 = (
    ROOT
    / "outputs/current_production_package_v2/final500_full_deep_review_v7/"
    "FINAL1000_FULL_DEEP_REVIEWED_V7.csv"
)
TOP3000 = (
    ROOT
    / "outputs/current_production_package_v2/formal_full_universe_v4/"
    "refined_top3000_v4_complete.csv"
)
KNOWN96 = (
    ROOT
    / "outputs/current_production_package_v2/full_untruncated_universe_v4/"
    "known_control_boltz96_calibration_v4.csv"
)
UNIVERSE_SUMMARY = (
    ROOT
    / "outputs/current_production_package_v2/full_untruncated_universe_v4_active_collapsed_sensitivity/"
    "full_untruncated_universe_v4_active_collapsed_sensitivity_summary.json"
)
OUT_DIR = ROOT / "outputs/current_production_package_v2/project_utility_audit_v8"


RESOLVED_SPECIES = {"parent_drug_relevant", "salt_normalization_adequate"}
UNRESOLVED_SPECIES = {
    "active_species_uncertain",
    "prodrug_active_metabolite_requires_rerun",
}
STANDARD_FAMILIES = {"enzyme", "kinase", "nuclear_epigenetic"}
SPECIALIZED_FAMILIES = {"transporter", "ion_channel"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def json_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def strength_ab(frame: pd.DataFrame) -> pd.Series:
    return frame["v5_strength_tier"].astype(str).str.startswith(("A_", "B_"))


def target_percentile(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def active_species_drug_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for drug_id, group in frame.groupby("drug_chembl_id", sort=True):
        statuses = sorted(set(group["active_species_status_v6"].astype(str)))
        material = bool(set(statuses) & UNRESOLVED_SPECIES) and len(statuses) > 1
        if len(statuses) == 1 and statuses[0] in RESOLVED_SPECIES:
            propagation = "reviewed_resolved"
        elif set(statuses).issubset(RESOLVED_SPECIES):
            propagation = "resolved_naming_variation"
        elif len(statuses) > 1:
            propagation = "reviewed_status_conflict"
        else:
            propagation = "reviewed_requires_resolution"
        rows.append(
            {
                "drug_chembl_id": drug_id,
                "drug_names": ";".join(sorted(set(group["drug_names"].astype(str)))),
                "candidate_rows": int(len(group)),
                "active_species_status_count": int(len(statuses)),
                "active_species_statuses": ";".join(statuses),
                "active_species_status_conflict": len(statuses) > 1,
                "material_active_species_conflict": material,
                "active_species_propagation_state": propagation,
            }
        )
    return pd.DataFrame(rows)


def add_final1000_axes(
    frame: pd.DataFrame,
    drug_audit: pd.DataFrame,
    target_context: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    exact = out["agent_literature_class"].eq("exact_pair_validated")
    contradictory = out["agent_literature_class"].eq("contradictory")
    resolved = out["active_species_status_v6"].isin(RESOLVED_SPECIES)
    standard = out["target_assay_family"].isin(STANDARD_FAMILIES)
    physical_ab = strength_ab(out)
    concordant = out["v5_model_agreement_lane"].eq("concordant_high")

    out["interaction_discovery_queue_v8"] = np.select(
        [exact, contradictory, ~resolved, standard],
        [
            "C0_validated_control",
            "X0_direct_contradiction",
            "R1_active_species_resolution",
            "Q1_standard_direct_assay",
        ],
        default="Q2_specialized_membrane_assay",
    )
    out["interaction_physics_axis_v8"] = np.select(
        [physical_ab & concordant, physical_ab & out["v5_model_agreement_lane"].eq("boltz_led"), physical_ab & out["v5_model_agreement_lane"].eq("conplex_led"), physical_ab],
        [
            "I1_cross_model_concordant_AB",
            "I2_boltz_led_AB",
            "I2_conplex_led_AB",
            "I2_moderate_or_discordant_AB",
        ],
        default="I3_positive_floor_only",
    )
    out["translation_axis_v8"] = out["feasibility_grade_v6"].map(
        {
            "A": "T1_translation_ready",
            "B": "T2_one_major_uncertainty",
            "C": "T3_translation_bridge_missing",
            "D": "TX_translation_stop",
        }
    )
    out["novelty_axis_v8"] = out["agent_literature_class"].map(
        {
            "exact_pair_validated": "N0_validated_not_novel",
            "functional_only": "N1_functional_not_direct",
            "indirect_or_family_only": "N2_indirect_or_family",
            "no_exact_report_found": "N3_no_exact_public_report",
            "contradictory": "NX_direct_contradiction",
        }
    )
    out = out.merge(
        drug_audit[
            [
                "drug_chembl_id",
                "active_species_status_conflict",
                "material_active_species_conflict",
                "active_species_propagation_state",
            ]
        ],
        on="drug_chembl_id",
        how="left",
        validate="many_to_one",
    )
    out = out.merge(target_context, on=["sequence_key", "primary_gene"], how="left", validate="many_to_one")

    standard_core = (
        out["interaction_discovery_queue_v8"].eq("Q1_standard_direct_assay")
        & physical_ab
        & ~out["feasibility_grade_v6"].eq("D")
    )
    high_consensus = standard_core & concordant
    out["discovery_core_v8"] = standard_core
    out["high_consensus_discovery_core_v8"] = high_consensus
    out["recommended_current_use_v8"] = np.select(
        [
            exact,
            contradictory,
            ~resolved,
            out["feasibility_grade_v6"].eq("D"),
            high_consensus,
            standard_core,
            standard & ~physical_ab,
            out["target_assay_family"].isin(SPECIALIZED_FAMILIES),
        ],
        [
            "validated_control_only",
            "stop_or_negative_control",
            "repair_active_species_before_rescoring",
            "translation_stop_optional_in_vitro_only",
            "priority_standard_binding_discovery",
            "standard_binding_discovery",
            "positive_floor_reserve",
            "specialized_assay_queue",
        ],
        default="manual_review",
    )
    use_order = {
        "priority_standard_binding_discovery": 0,
        "standard_binding_discovery": 1,
        "positive_floor_reserve": 2,
        "specialized_assay_queue": 3,
        "repair_active_species_before_rescoring": 4,
        "validated_control_only": 5,
        "translation_stop_optional_in_vitro_only": 6,
        "stop_or_negative_control": 7,
        "manual_review": 8,
    }
    strength_order = {
        "A_at_or_above_known_positive_median": 0,
        "B_at_or_above_known_positive_q25": 1,
        "C_at_or_above_known_positive_q10": 2,
    }
    out["_use_order"] = out["recommended_current_use_v8"].map(use_order).fillna(9)
    out["_strength_order"] = out["v5_strength_tier"].map(strength_order).fillna(9)
    out = out.sort_values(
        ["_use_order", "_strength_order", "v5_pair_physics_score", "pair_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    out["utility_review_rank_v8"] = range(1, len(out) + 1)
    return out.drop(columns=["_use_order", "_strength_order"])


def build_target_context(top3000: pd.DataFrame, known96: pd.DataFrame, final1000: pd.DataFrame) -> pd.DataFrame:
    top = top3000.copy()
    top["boltz_affinity_numeric"] = pd.to_numeric(
        top["boltz_affinity_probability_refined"], errors="coerce"
    )
    top["boltz_within_target_percentile_top3000"] = top.groupby("sequence_key")[
        "boltz_affinity_numeric"
    ].transform(target_percentile)
    top_stats = top.groupby(["sequence_key", "primary_gene"], as_index=False).agg(
        top3000_target_rows=("pair_id", "size"),
        top3000_target_boltz_median=("boltz_affinity_numeric", "median"),
        top3000_target_boltz_mean=("boltz_affinity_numeric", "mean"),
    )
    controls = known96.copy()
    controls["known_control_affinity"] = pd.to_numeric(
        controls["boltz_affinity_probability_refined"], errors="coerce"
    )
    control_stats = controls.groupby("sequence_key", as_index=False).agg(
        boltz_target_positive_control_count=("pair_id", "size"),
        boltz_target_positive_control_median=("known_control_affinity", "median"),
        boltz_target_positive_control_min=("known_control_affinity", "min"),
        boltz_target_positive_control_max=("known_control_affinity", "max"),
    )
    target_meta = final1000.groupby(["sequence_key", "primary_gene"], as_index=False).agg(
        final1000_target_rows=("pair_id", "size"),
        target_assay_family_context=("target_assay_family", "first"),
        target_has_experimental_ligand_structure=("anchor_sm_structure_with_ligand", "max"),
        target_has_high_quality_ligand=("anchor_sm_high_quality_ligand", "max"),
    )
    context = target_meta.merge(top_stats, on=["sequence_key", "primary_gene"], how="left")
    context = context.merge(control_stats, on="sequence_key", how="left")
    context["boltz_target_positive_control_count"] = (
        context["boltz_target_positive_control_count"].fillna(0).astype(int)
    )
    context["target_level_boltz_calibration_status_v8"] = np.where(
        context["boltz_target_positive_control_count"].gt(0),
        "has_1_to_3_positive_controls_no_inactives",
        "no_target_level_boltz_control",
    )
    return context


def build_rescue_pool(
    top3000: pd.DataFrame,
    final1000: pd.DataFrame,
    known96: pd.DataFrame,
    drug_audit: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = v5.calibration_thresholds(known96)
    enhanced = v5.add_axes(top3000, top3000, thresholds)
    enhanced["boltz_affinity_numeric"] = pd.to_numeric(
        enhanced["boltz_affinity_probability_refined"], errors="coerce"
    )
    enhanced["boltz_within_target_percentile_top3000"] = enhanced.groupby("sequence_key")[
        "boltz_affinity_numeric"
    ].transform(target_percentile)
    base = (
        truthy(enhanced["boltz_completed_refined"])
        & ~truthy(enhanced["exact_known_target_v2"])
        & ~truthy(enhanced["family_or_rediscovery_risk_v2"])
        & ~truthy(enhanced["severe_compound_liability"])
        & ~truthy(enhanced["structure_sequence_mismatch_v4"])
        & enhanced["v5_pose_primary_ready"]
        & enhanced["v5_passes_known_positive_q10_floor"]
        & enhanced["target_assay_family_v2"].isin(STANDARD_FAMILIES)
        & strength_ab(enhanced)
        & pd.to_numeric(enhanced["rank_within_drug"], errors="coerce").le(100)
        & ~enhanced["pair_id"].isin(set(final1000["pair_id"]))
    )
    rescue = enhanced.loc[base].copy()
    rescue = rescue.merge(
        drug_audit[
            [
                "drug_chembl_id",
                "active_species_statuses",
                "active_species_status_conflict",
                "material_active_species_conflict",
                "active_species_propagation_state",
            ]
        ],
        on="drug_chembl_id",
        how="left",
        validate="many_to_one",
    )
    rescue["active_species_propagation_state"] = rescue[
        "active_species_propagation_state"
    ].fillna("unreviewed_drug")
    rescue["rescue_review_state_v8"] = rescue["active_species_propagation_state"].map(
        {
            "reviewed_resolved": "R0_reviewed_drug_resolved",
            "resolved_naming_variation": "R0_reviewed_drug_resolved",
            "reviewed_status_conflict": "R1_reconcile_active_species_status",
            "reviewed_requires_resolution": "R1_reconcile_active_species_status",
            "unreviewed_drug": "R2_full_pair_review_required",
        }
    )
    rescue["_agreement_order"] = rescue["v5_model_agreement_lane"].map(
        {
            "concordant_high": 0,
            "boltz_led": 1,
            "conplex_led": 2,
            "moderate_or_discordant": 3,
        }
    ).fillna(4)
    rescue = rescue.sort_values(
        [
            "rescue_review_state_v8",
            "_agreement_order",
            "v5_pair_physics_score",
            "pair_id",
        ],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    rescue = rescue.drop(columns="_agreement_order")
    rescue.insert(0, "rescue_review_rank_v8", range(1, len(rescue) + 1))
    return rescue


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(FINAL1000, low_memory=False).fillna("")
    top3000 = pd.read_csv(TOP3000, low_memory=False).fillna("")
    known96 = pd.read_csv(KNOWN96, low_memory=False).fillna("")
    universe_summary = json.loads(UNIVERSE_SUMMARY.read_text(encoding="utf-8"))

    if len(final) != 1000 or final["pair_id"].nunique() != 1000:
        raise ValueError("Expected 1000 unique reviewed pairs")
    if len(top3000) != 3000 or top3000["pair_id"].nunique() != 3000:
        raise ValueError("Expected 3000 unique refined pairs")
    if len(known96) != 96:
        raise ValueError("Expected 96 Boltz known-positive calibration rows")

    drug_audit = active_species_drug_audit(final)
    target_context = build_target_context(top3000, known96, final)
    audited = add_final1000_axes(final, drug_audit, target_context)
    rescue = build_rescue_pool(top3000, final, known96, drug_audit)

    core = audited.loc[audited["discovery_core_v8"]].copy()
    high = audited.loc[audited["high_consensus_discovery_core_v8"]].copy()
    target_utility = audited.groupby(["sequence_key", "primary_gene"], as_index=False).agg(
        discovery_core_rows=("discovery_core_v8", "sum"),
        high_consensus_core_rows=("high_consensus_discovery_core_v8", "sum"),
        translation_AB_rows=("feasibility_grade_v6", lambda values: values.isin(["A", "B"]).sum()),
        direct_contradiction_rows=(
            "interaction_discovery_queue_v8",
            lambda values: values.eq("X0_direct_contradiction").sum(),
        ),
        active_species_repair_rows=(
            "interaction_discovery_queue_v8",
            lambda values: values.eq("R1_active_species_resolution").sum(),
        ),
    )
    target_context = target_context.merge(
        target_utility,
        on=["sequence_key", "primary_gene"],
        how="left",
        validate="one_to_one",
    )
    target_context["target_boltz_bias_decile_v8"] = pd.qcut(
        pd.to_numeric(target_context["top3000_target_boltz_median"], errors="coerce"),
        10,
        labels=False,
        duplicates="drop",
    ).add(1)
    target_context["target_calibration_priority_v8"] = np.select(
        [
            target_context["high_consensus_core_rows"].gt(0)
            & target_context["boltz_target_positive_control_count"].eq(0),
            target_context["high_consensus_core_rows"].gt(0),
            target_context["discovery_core_rows"].gt(0),
        ],
        [
            "P1_add_same_target_actives_and_inactives",
            "P2_expand_existing_positive_only_calibration",
            "P3_calibrate_before_promotion",
        ],
        default="P4_reserve_target",
    )
    target_context = target_context.sort_values(
        [
            "target_calibration_priority_v8",
            "high_consensus_core_rows",
            "discovery_core_rows",
            "primary_gene",
        ],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    conflict_drugs = set(
        drug_audit.loc[drug_audit["active_species_status_conflict"], "drug_chembl_id"]
    )
    conflicts = audited.loc[audited["drug_chembl_id"].isin(conflict_drugs)].copy()

    known_target_keys = set(known96["sequence_key"])
    final_target_keys = set(final["sequence_key"])
    target_medians = pd.to_numeric(
        target_context["top3000_target_boltz_median"], errors="coerce"
    )
    top3000_affinity = pd.to_numeric(
        top3000["boltz_affinity_probability_refined"], errors="coerce"
    )
    global_mean = float(top3000_affinity.mean())
    total_ss = float(((top3000_affinity - global_mean) ** 2).sum())
    affinity_frame = top3000[["sequence_key", "target_assay_family_v2"]].copy()
    affinity_frame["_aff"] = top3000_affinity.to_numpy()
    by_target = affinity_frame.groupby("sequence_key")["_aff"]
    target_means = by_target.mean()
    target_sizes = by_target.size()
    target_between_ss = float((((target_means - global_mean) ** 2) * target_sizes).sum())
    family_stats = affinity_frame.groupby("target_assay_family_v2")["_aff"]
    family_means = family_stats.mean()
    family_sizes = family_stats.size()
    family_between_ss = float((((family_means - global_mean) ** 2) * family_sizes).sum())

    output_paths = {
        "final1000_audit": OUT_DIR / "FINAL1000_MULTIAXIS_UTILITY_AUDIT_V8.csv",
        "discovery_core": OUT_DIR / f"DISCOVERY_CORE_{len(core)}_V8.csv",
        "high_consensus_core": OUT_DIR / f"HIGH_CONSENSUS_DISCOVERY_CORE_{len(high)}_V8.csv",
        "top3000_rescue": OUT_DIR / f"TOP3000_RESCUE_POOL_{len(rescue)}_V8.csv",
        "active_species_conflicts": OUT_DIR / f"ACTIVE_SPECIES_CONFLICT_ROWS_{len(conflicts)}_V8.csv",
        "active_species_drug_audit": OUT_DIR / "ACTIVE_SPECIES_DRUG_LEVEL_AUDIT_V8.csv",
        "target_context": OUT_DIR / "TARGET_CONTEXT_CALIBRATION_GAPS_V8.csv",
    }
    audited.to_csv(output_paths["final1000_audit"], index=False)
    core.to_csv(output_paths["discovery_core"], index=False)
    high.to_csv(output_paths["high_consensus_core"], index=False)
    rescue.to_csv(output_paths["top3000_rescue"], index=False)
    conflicts.to_csv(output_paths["active_species_conflicts"], index=False)
    drug_audit.to_csv(output_paths["active_species_drug_audit"], index=False)
    target_context.to_csv(output_paths["target_context"], index=False)

    physical_ab = strength_ab(final)
    translation_ab = final["feasibility_grade_v6"].isin(["A", "B"])
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": {
            "project_physical_pairs": int(universe_summary["project_physical_pair_rows"]),
            "top3000_rows": int(len(top3000)),
            "final1000_rows": int(len(final)),
        },
        "legacy_grade_audit": {
            "translation_grade_counts": json_counts(final["feasibility_grade_v6"]),
            "translation_AB_rows": int(translation_ab.sum()),
            "physics_AB_rows": int(physical_ab.sum()),
            "translation_AB_and_physics_AB": int((translation_ab & physical_ab).sum()),
            "translation_AB_but_physics_C": int((translation_ab & ~physical_ab).sum()),
            "physics_AB_but_translation_C_or_D": int((physical_ab & ~translation_ab).sum()),
        },
        "multi_axis_queues": {
            "interaction_discovery_queue_counts": json_counts(
                audited["interaction_discovery_queue_v8"]
            ),
            "recommended_use_counts": json_counts(audited["recommended_current_use_v8"]),
            "discovery_core_rows": int(len(core)),
            "discovery_core_unique_drugs": int(core["drug_chembl_id"].nunique()),
            "discovery_core_unique_targets": int(core["primary_gene"].nunique()),
            "high_consensus_core_rows": int(len(high)),
            "high_consensus_core_unique_drugs": int(high["drug_chembl_id"].nunique()),
            "high_consensus_core_unique_targets": int(high["primary_gene"].nunique()),
            "high_consensus_grade_counts": json_counts(high["feasibility_grade_v6"]),
        },
        "active_species_audit": {
            "reviewed_drugs": int(len(drug_audit)),
            "conflicting_status_drugs": int(drug_audit["active_species_status_conflict"].sum()),
            "rows_on_conflicting_drugs": int(len(conflicts)),
            "material_conflict_drugs": int(drug_audit["material_active_species_conflict"].sum()),
            "rows_on_material_conflict_drugs": int(
                final["drug_chembl_id"].isin(
                    set(
                        drug_audit.loc[
                            drug_audit["material_active_species_conflict"], "drug_chembl_id"
                        ]
                    )
                ).sum()
            ),
        },
        "boltz_calibration_audit": {
            "known_positive_rows": int(len(known96)),
            "known_positive_unique_targets": int(known96["sequence_key"].nunique()),
            "final1000_unique_targets": int(len(final_target_keys)),
            "final1000_targets_with_target_level_positive_control": int(
                len(final_target_keys & known_target_keys)
            ),
            "final1000_targets_without_target_level_positive_control": int(
                len(final_target_keys - known_target_keys)
            ),
            "final1000_rows_without_target_level_positive_control": int(
                (~final["sequence_key"].isin(known_target_keys)).sum()
            ),
            "known_positive_target_n_median": float(
                known96.groupby("sequence_key").size().median()
            ),
            "known_positive_target_n_max": int(known96.groupby("sequence_key").size().max()),
            "top3000_target_median_affinity_min": float(target_medians.min()),
            "top3000_target_median_affinity_max": float(target_medians.max()),
            "target_identity_explained_variance_eta_squared": float(
                target_between_ss / total_ss
            ),
            "assay_family_explained_variance_eta_squared": float(
                family_between_ss / total_ss
            ),
        },
        "structure_audit": {
            "final1000_all_strict_pocket_A": bool(
                final["strict_structure_tier"].eq("A_strict_overlapping_pocket").all()
            ),
            "rows_with_experimental_ligand_structure_annotation": int(
                truthy(final["anchor_sm_structure_with_ligand"]).sum()
            ),
            "interpretation": "Pocket fields are target-level gates and do not discriminate pairs inside Final1000.",
        },
        "top3000_rescue_audit": {
            "rescue_rows": int(len(rescue)),
            "rescue_unique_drugs": int(rescue["drug_chembl_id"].nunique()),
            "rescue_unique_targets": int(rescue["primary_gene"].nunique()),
            "rescue_review_state_counts": json_counts(rescue["rescue_review_state_v8"]),
        },
        "input_sha256": {
            "final1000": sha256(FINAL1000),
            "top3000": sha256(TOP3000),
            "known96": sha256(KNOWN96),
            "universe_summary": sha256(UNIVERSE_SUMMARY),
        },
    }
    summary_path = OUT_DIR / "PROJECT_CANDIDATE_UTILITY_AUDIT_V8.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    queue_counts = summary["multi_axis_queues"]["interaction_discovery_queue_counts"]
    rescue_counts = summary["top3000_rescue_audit"]["rescue_review_state_counts"]
    report = f"""# FDA老药新靶点项目：候选效用与全流程审计 V8

生成时间：{summary['created_utc']}

## 1. 结论

现有 `A/B/C/D` 是**转化与整体实验可行性档**，不是直接互作发现质量档。A/B只有142条，不能解释为其余858条都没有结合发现价值。当前1000条本身已经通过Boltz阳性下限、双构象A/B、ConPLEx药内Top100、结构口袋、已知pair和家族泄露等硬门；后续A/B/C/D又加入人体游离暴露、给药组织、作用方向和疾病可用性，因此两套等级回答的是不同问题。

在保持现有硬门且排除D级转化硬停止后，1000条中可形成：

- **571条标准直接实验发现核心**：活性实体已解决、enzyme/kinase/nuclear assay、物理A/B、无精确反证、不是已验证对照。
- **152条高一致性发现核心**：上述571条中ConPLEx与Boltz均处于高信号区；其中B级9条、C级143条。143条C主要表示尚无实测效力可与人体暴露桥接，不能据此否定离体结合实验。
- **270条Top3000救援审阅池**：不在正式1000内，但仍满足标准assay、物理A/B、Boltz姿势A/B、阳性下限和ConPLEx药内Top100。它们必须先补活性实体、文献和暴露审阅，不能直接并入正式包。

## 2. 为什么142条不能作为发现质量总数

| 口径 | 数量 | 含义 |
|---|---:|---|
| 当前整体可行性A/B | {summary['legacy_grade_audit']['translation_AB_rows']} | 暴露、assay和风险综合较成熟 |
| 当前物理A/B | {summary['legacy_grade_audit']['physics_AB_rows']} | Boltz达到家族阳性q25以上，非结合概率 |
| 两者交集 | {summary['legacy_grade_audit']['translation_AB_and_physics_AB']} | 同时满足两套口径 |
| 整体A/B但物理仅C | {summary['legacy_grade_audit']['translation_AB_but_physics_C']} | 实验容易但计算物理信号较弱 |
| 物理A/B但整体C/D | {summary['legacy_grade_audit']['physics_AB_but_translation_C_or_D']} | 主要被暴露、方向或转化问题降级 |

当前T1执行层也不是独立验证：代码直接以“整体A/B + 活性实体已解决 + 标准靶点家族”生成T1。因此不能再用T1反向证明A/B分级正确。

## 3. 1000条重新分流

| 队列 | 数量 | 当前用途 |
|---|---:|---|
| Q1 标准直接assay | {queue_counts.get('Q1_standard_direct_assay', 0)} | 从中构建结合发现核心 |
| Q2 膜蛋白专用assay | {queue_counts.get('Q2_specialized_membrane_assay', 0)} | transporter/ion channel独立排期 |
| R1 活性实体待解决 | {queue_counts.get('R1_active_species_resolution', 0)} | 先重建母体/代谢物/异构体输入 |
| C0 已验证对照 | {queue_counts.get('C0_validated_control', 0)} | 阳性或再发现校准，不包装为创新 |
| X0 精确反证 | {queue_counts.get('X0_direct_contradiction', 0)} | 停止或作为阴性对照 |

## 4. 当前最重要的数据与模型缺口

1. **缺少阴性校准。** Boltz校准集只有96个阳性、60个靶点，每靶点中位数1条、最多3条；没有匹配inactive/decoy，因此不能估计precision、FDR或富集率。
2. **靶点偏差很强。** Top3000中，靶点身份解释Boltz affinity方差的η²约{summary['boltz_calibration_audit']['target_identity_explained_variance_eta_squared']:.3f}，assay family解释约{summary['boltz_calibration_audit']['assay_family_explained_variance_eta_squared']:.3f}。各靶点Boltz中位数从{summary['boltz_calibration_audit']['top3000_target_median_affinity_min']:.3f}到{summary['boltz_calibration_audit']['top3000_target_median_affinity_max']:.3f}，跨靶点比较原始affinity会系统性偏向部分靶点。
3. **大多数靶点没有同靶点Boltz对照。** Final1000的163个靶点中只有{summary['boltz_calibration_audit']['final1000_targets_with_target_level_positive_control']}个进入known96，{summary['boltz_calibration_audit']['final1000_targets_without_target_level_positive_control']}个没有同靶点阳性参照，影响{summary['boltz_calibration_audit']['final1000_rows_without_target_level_positive_control']}条候选。
4. **口袋分不再有排序信息。** Final1000的1000条全部是P2Rank/PUResNet严格A档；这是靶点级进入条件，不是某个药物会结合该口袋的独立证据。
5. **活性实体字段过载。** {summary['active_species_audit']['conflicting_status_drugs']}个药物在不同pair上出现冲突状态，涉及{summary['active_species_audit']['rows_on_conflicting_drugs']}行；其中{summary['active_species_audit']['material_conflict_drugs']}个药物、{summary['active_species_audit']['rows_on_material_conflict_drugs']}行涉及“已解决/需代谢物重跑”的实质冲突。应拆成药物级`modeled_entity`与pair级`parallel_species_required`。
6. **Top3000仍由首轮ConPLEx主导。** 全物理空间334,749条中仅3,000条运行Boltz，且既有审计显示2,979/3,000来自旧106k。全空间扩展没有真正改变昂贵结构计算看到的候选分布。

## 5. 如何得到更有用的候选

### 5.1 立即可做：不新增计算

以152条高一致性发现核心作为第一批直接结合审阅池，以其余419条标准发现核心作为扩展池。当前A/B继续保留为转化轴，但不再作为发现硬门。实验命中后再用实测IC50/Kd与人体游离暴露比较；在没有实测效力前，以“无法桥接暴露”为由将新候选统一降为C具有循环性。

### 5.2 必须补的计算校准

152条高一致性候选分布在64个靶点，其中48个靶点没有同靶点Boltz阳性对照，16个只有1-3条阳性且没有inactive。SOAT2、SOAT1、PORCN和PDE4D分别占14、8、8和7条，提示高分还包含明显的靶点偏移。

对进入实验预算的每个靶点建立独立benchmark：实验holo结构/辅因子/组装状态，3-10个已知活性配体，性质匹配的inactive或decoy，同一协议重跑Boltz与传统docking。只有能够在该靶点上分离active/inactive的模型，才允许其新候选进入高置信队列。排序应使用同靶点相对分、相对阳性对照差值和重复seed稳定性，不再主要使用跨靶点原始Boltz affinity。

### 5.3 补真实assay准备度

将当前自动生成的assay文本替换为可核查字段：蛋白构建体或细胞系、底物/辅因子、商业试剂或文献SOP、阳性对照、匹配阴性、读数干扰、化合物溶解度和允许的最高非聚集浓度。做到这一层后，才有独立的“实验可立即启动”等级。

### 5.4 重新进入334k的多通道预算

下一轮3000不再由一个综合分顺序截断。建议按独立通道分配计算预算：ConPLEx药内异常、ConPLEx靶点内异常、实验holo/已知口袋结构筛选、模型分歧救援、每靶点探索配额和随机校准配额。每条候选记录来源通道，最终用命中率比较通道价值。

## 6. Top3000救援池状态

| 状态 | 数量 |
|---|---:|
| 已审药物且活性实体一致 | {rescue_counts.get('R0_reviewed_drug_resolved', 0)} |
| 活性实体状态需统一 | {rescue_counts.get('R1_reconcile_active_species_status', 0)} |
| 药物尚未完整逐对审阅 | {rescue_counts.get('R2_full_pair_review_required', 0)} |

270条是**补审池**，不是新增正式推荐。其价值在于检查固定1000容量与多样性上限是否遗漏更好的标准assay候选。

## 7. 正确的项目主口径

当前项目应同时报告三条轴：

1. **互作发现轴**：模型与结构是否值得做直接实验。
2. **实验执行轴**：真实试剂、SOP和反筛是否可获得。
3. **转化轴**：实测效力能否被批准剂量下的组织游离暴露覆盖，并且作用方向是否支持疾病。

142条A/B只属于第三轴与部分第二轴的综合结果。现阶段最有价值的扩充不是放宽A/B，而是把571条发现核心做靶点特异校准，从中得到有实验区分力的优先队列。

## 外部方法学依据

- ConPLEx输出是序列与化学结构嵌入空间的相互作用分数，适合召回排序，不是Kd或实验命中概率：[ConPLEx原始论文](https://pubmed.ncbi.nlm.nih.gov/37289807/)。
- Boltz-2原论文报告结构和affinity预测能力，但项目内仍需目标特异校准：[Boltz-2预印本](https://doi.org/10.1101/2025.06.14.659707)。
- 2026年的时间外cofolding评估显示，现有方法对训练相似性敏感，不能把高置信姿势直接当作de novo结合事实：[Runs N' Poses评估](https://www.nature.com/articles/s41594-026-01797-5)。
- LIT-PCBA强调用真实实验active/inactive和性质匹配集合评估虚拟筛选，避免只用人工decoy造成性能高估：[LIT-PCBA](https://doi.org/10.1021/acs.jcim.0c00155)。
- P2Rank预测的是候选结合位点，是自动流程的一步，不证明指定配体结合：[P2Rank](https://doi.org/10.1186/s13321-018-0285-8)。
"""
    report_path = OUT_DIR / "PROJECT_CANDIDATE_UTILITY_AUDIT_V8_ZH.md"
    report_path.write_text(report, encoding="utf-8")

    for key, path in output_paths.items():
        summary.setdefault("output_sha256", {})[key] = sha256(path)
    summary["output_sha256"]["report"] = sha256(report_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

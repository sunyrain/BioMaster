#!/usr/bin/env python3
"""Build a target-admission and candidate-action package from all reusable V5 evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evidence_reuse_action_v6.yaml"
V5 = (
    ROOT
    / "outputs/current_production_package_v2/calibrated_portfolio_v5"
    / "FINAL1000_EVIDENCE_STRATIFIED_V5.csv"
)
COVERAGE = (
    ROOT
    / "outputs/current_production_package_v2/chembl37_target_calibration_v5"
    / "PROJECT463_CALIBRATION_COVERAGE_V5.csv"
)
CALIBRATION_PAIRS = (
    ROOT
    / "outputs/current_production_package_v2/chembl37_target_calibration_v5"
    / "PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz"
)
QSAR = (
    ROOT
    / "outputs/current_production_package_v2/target_qsar_calibration_v5"
    / "TARGET_QSAR_SCAFFOLD_HOLDOUT_METRICS_V5.csv"
)
CONPLEX = (
    ROOT
    / "outputs/current_production_package_v2/conplex_target_calibration_v5_official/evaluation"
    / "CONPLEX_TARGET_CALIBRATION_METRICS_V5.csv"
)
TARGETS = ROOT / "configs/project_targets_v4.csv"
ANCHORS = (
    ROOT
    / "outputs/chembl_moa_enhanced_information_package_v1"
    / "chembl_moa_anchor_gene_table_v2.csv"
)
SMOKE = (
    ROOT
    / "outputs/current_production_package_v2/boltz_target_calibration_smoke_v5/evaluation"
    / "BOLTZ_SMOKE_PAIRED_TARGET_COMPARISON_V5.csv"
)
SMOKE_INPUT = (
    ROOT
    / "outputs/current_production_package_v2/boltz_target_calibration_smoke_v5/input_package"
    / "boltz2_smoke_queue.csv"
)
OLD_REVIEW = (
    ROOT
    / "outputs/current_production_package_v2/final500_full_deep_review_v7"
    / "FINAL1000_FULL_DEEP_REVIEWED_V7.csv"
)
NOVELTY_AUDIT = (
    ROOT
    / "outputs/current_production_package_v2/evidence_reuse_action_v6/known_pair_novelty_audit"
    / "FINAL1000_KNOWN_PAIR_NOVELTY_AUDIT_V6.csv"
)
OUT = ROOT / "outputs/current_production_package_v2/evidence_reuse_action_v6"
REPORT = ROOT / "docs/FDA_OLD_DRUG_CURRENT_EVIDENCE_REUSE_AND_NEXT_ACTION_V6_ZH.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bools(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def first_nonempty(values: pd.Series) -> Any:
    valid = values.dropna()
    if valid.empty:
        return ""
    valid = valid[valid.astype(str).str.strip().ne("")]
    return valid.iloc[0] if not valid.empty else ""


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def build_target_matrix(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lane = data["portfolio_lane_v5"].fillna("")
    counts = data.assign(
        _p1=lane.str.startswith("P1").astype(int),
        _p2=lane.str.startswith("P2").astype(int),
        _p3=lane.str.startswith("P3").astype(int),
    ).groupby(["sequence_key", "primary_gene", "target_assay_family"], as_index=False).agg(
        final1000_pair_n=("drug_chembl_id", "size"),
        p1_pair_n=("_p1", "sum"),
        p2_pair_n=("_p2", "sum"),
        p3_pair_n=("_p3", "sum"),
    )

    targets = pd.read_csv(TARGETS, low_memory=False)
    target_columns = [
        "sequence_key",
        "representative_protein_id",
        "structure_bin",
        "pdb_path",
        "scope_source",
        "inclusion_reason",
    ]
    targets = targets[target_columns].drop_duplicates("sequence_key")
    anchors = pd.read_csv(ANCHORS, low_memory=False)
    anchor_columns = [
        "gene",
        "canonical_uniprot",
        "protein_name",
        "sm_structure_with_ligand",
        "sm_high_quality_ligand",
        "sm_high_quality_pocket",
        "sm_approved_drug",
        "sm_advanced_clinical",
        "project_standard_direct_sm",
        "opentargets_match_status",
    ]
    anchor_columns = [column for column in anchor_columns if column in anchors.columns]
    anchors = anchors[anchor_columns].drop_duplicates("gene")
    target = counts.merge(targets, on="sequence_key", how="left", validate="many_to_one")
    target = target.merge(anchors, left_on="primary_gene", right_on="gene", how="left", validate="many_to_one")

    coverage = pd.read_csv(COVERAGE, low_memory=False)
    coverage_columns = [
        "sequence_key",
        "strict_compound_pairs",
        "positive_compounds",
        "negative_compounds",
        "grey_compounds",
        "conflicting_compounds",
        "strict_document_count",
        "calibration_tier_v5",
    ]
    target = target.merge(coverage[coverage_columns], on="sequence_key", how="left", validate="one_to_one")
    pair_columns = [
        "sequence_key",
        "min_pchembl",
        "max_pchembl",
        "mean_pchembl",
        "any_explicit_inactive",
    ]
    pairs = pd.read_csv(CALIBRATION_PAIRS, usecols=pair_columns, low_memory=False)
    minimum = pd.to_numeric(pairs["min_pchembl"], errors="coerce")
    maximum = pd.to_numeric(pairs["max_pchembl"], errors="coerce")
    mean = pd.to_numeric(pairs["mean_pchembl"], errors="coerce")
    explicit = pd.to_numeric(pairs["any_explicit_inactive"], errors="coerce").fillna(0).gt(0)
    pairs["_boltz_strong_positive"] = mean.ge(7.0) & minimum.ge(6.0)
    pairs["_boltz_strict_negative"] = (mean.le(4.5) | explicit) & maximum.fillna(float("-inf")).lt(6.0)
    strict_counts = pairs.groupby("sequence_key", as_index=False).agg(
        boltz_strong_positive_n_v6=("_boltz_strong_positive", "sum"),
        boltz_strict_negative_n_v6=("_boltz_strict_negative", "sum"),
    )
    target = target.merge(strict_counts, on="sequence_key", how="left", validate="one_to_one")

    qsar = pd.read_csv(QSAR, low_memory=False)
    qsar_columns = [
        "sequence_key",
        "positive_n",
        "negative_n",
        "scaffold_n",
        "qsar_oof_pr_auc",
        "similarity_oof_pr_auc",
        "qsar_oof_roc_auc",
        "similarity_oof_roc_auc",
        "qsar_minus_similarity_ap_ci95_low",
        "temporal_validation_status_v5",
        "target_ligand_model_status_v5",
    ]
    target = target.merge(qsar[qsar_columns], on="sequence_key", how="left", validate="one_to_one")

    conplex = pd.read_csv(CONPLEX, low_memory=False)
    conplex_columns = [
        "sequence_key",
        "conplex_pr_auc",
        "similarity_pr_auc",
        "conplex_roc_auc",
        "similarity_roc_auc",
        "conplex_minus_similarity_ap_ci95_low",
        "conplex_target_use_status_v5",
    ]
    target = target.merge(conplex[conplex_columns], on="sequence_key", how="left", validate="one_to_one")

    smoke = pd.read_csv(SMOKE, low_memory=False)
    smoke_columns = [
        "target",
        "boltzAffinityProbabilityBinary_delta_positive_minus_negative",
        "boltzAffinityProbabilityBinary_positive_wins",
        "boltzCompositeScore_delta_positive_minus_negative",
        "boltzCompositeScore_positive_wins",
    ]
    smoke = smoke[smoke_columns].rename(columns={"target": "primary_gene"})
    target = target.merge(smoke, on="primary_gene", how="left", validate="one_to_one")

    target["qsar_pair_ranking_admitted_v6"] = target["target_ligand_model_status_v5"].eq(
        "T1_qsar_beats_similarity"
    )
    target["similarity_pair_ranking_admitted_v6"] = target["target_ligand_model_status_v5"].isin(
        ["T1_qsar_beats_similarity", "T2_similarity_supported"]
    )
    target["conplex_pair_ranking_admitted_v6"] = False
    target["boltz_binding_pair_ranking_admitted_v6"] = False
    target["boltz_current_role_v6"] = "conditional_pose_qc_only"
    target["local_receptor_source_v6"] = "alphafold_template_not_experimental_holo"
    target["experimental_holo_downloaded_and_validated_v6"] = False

    allowed_families = set(config["boltz_recalibration"]["allowed_assay_families"])
    allowed_tiers = set(config["boltz_recalibration"]["target_data_tiers"])
    structure_ligand = bools(target.get("sm_structure_with_ligand", pd.Series(False, index=target.index)))
    high_quality_pocket = bools(target.get("sm_high_quality_pocket", pd.Series(False, index=target.index)))
    target["boltz_recalibration_data_ready_v6"] = target["calibration_tier_v5"].isin(allowed_tiers)
    per_target = config["boltz_recalibration"]["per_target"]
    target["boltz_recalibration_strict_label_ready_v6"] = (
        target["boltz_strong_positive_n_v6"].ge(int(per_target["positives"]))
        & target["boltz_strict_negative_n_v6"].ge(int(per_target["negatives"]))
    )
    target["boltz_recalibration_family_ready_v6"] = target["target_assay_family"].isin(allowed_families)
    target["external_structure_evidence_available_v6"] = structure_ligand | high_quality_pocket
    target["boltz_recalibration_eligible_v6"] = (
        target["boltz_recalibration_data_ready_v6"]
        & target["boltz_recalibration_strict_label_ready_v6"]
        & target["boltz_recalibration_family_ready_v6"]
        & target["external_structure_evidence_available_v6"]
        & target["p3_pair_n"].gt(0)
    )

    wave1 = set(config["boltz_recalibration"]["wave1_targets"])
    wave2 = set(config["boltz_recalibration"]["wave2_targets"])
    target["boltz_recalibration_wave_v6"] = "not_selected"
    target.loc[target["primary_gene"].isin(wave2), "boltz_recalibration_wave_v6"] = "wave2"
    target.loc[target["primary_gene"].isin(wave1), "boltz_recalibration_wave_v6"] = "wave1"
    selected = target["boltz_recalibration_wave_v6"].ne("not_selected")
    if not target.loc[selected, "boltz_recalibration_eligible_v6"].all():
        bad = target.loc[selected & ~target["boltz_recalibration_eligible_v6"], "primary_gene"].tolist()
        raise ValueError(f"Configured Boltz recalibration targets are not eligible: {bad}")

    target["target_model_admission_summary_v6"] = "no_pair_model_admitted_remote_exploration_only"
    target.loc[target["similarity_pair_ranking_admitted_v6"], "target_model_admission_summary_v6"] = (
        "validated_ligand_similarity_in_domain"
    )
    target.loc[target["qsar_pair_ranking_admitted_v6"], "target_model_admission_summary_v6"] = (
        "target_qsar_and_similarity_in_domain"
    )
    target["next_target_action_v6"] = "retain_target_annotations_wait_for_pair_model_calibration"
    target.loc[target["target_assay_family"].isin(["transporter", "ion_channel"]), "next_target_action_v6"] = (
        "separate_membrane_state_and_assay_protocol"
    )
    target.loc[target["boltz_recalibration_wave_v6"].eq("wave2"), "next_target_action_v6"] = (
        "run_after_wave1_protocol_is_admitted"
    )
    target.loc[target["boltz_recalibration_wave_v6"].eq("wave1"), "next_target_action_v6"] = (
        "build_matched_holo_boltz_benchmark_first"
    )
    target["boltz_benchmark_operational_priority_v6"] = (
        target["p3_pair_n"] * 10
        + target["final1000_pair_n"]
        + target["external_structure_evidence_available_v6"].astype(int) * 5
    )
    return target.sort_values(
        ["boltz_recalibration_wave_v6", "boltz_benchmark_operational_priority_v6", "primary_gene"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def add_historical_review(data: pd.DataFrame) -> pd.DataFrame:
    review = pd.read_csv(OLD_REVIEW, low_memory=False)
    columns = [
        "drug_chembl_id",
        "sequence_key",
        "agent_feasibility_grade",
        "agent_verdict",
        "agent_literature_class",
        "agent_primary_disease",
        "agent_repurposing_status",
        "agent_exposure_feasibility",
        "agent_active_species_status",
        "agent_assay_plan",
        "agent_key_risks",
        "literature_evidence_tier_v6",
        "literature_judgment_v6",
        "exposure_feasibility_v6",
        "candidate_disease_v6",
        "candidate_disease_area_v6",
        "ot_primary_disease_v6",
        "ot_primary_disease_evidence_tier_v6",
        "manual_pair_literature_review_completed_v6",
    ]
    columns = [column for column in columns if column in review.columns]
    review = review[columns].drop_duplicates(["drug_chembl_id", "sequence_key"])
    out = data.merge(review, on=["drug_chembl_id", "sequence_key"], how="left", validate="one_to_one")
    marker = "agent_verdict" if "agent_verdict" in out else columns[2]
    out["historical_pair_deep_review_reused_v6"] = out[marker].notna()
    return out


def build_candidate_actions(data: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    target_columns = [
        "sequence_key",
        "calibration_tier_v5",
        "qsar_pair_ranking_admitted_v6",
        "similarity_pair_ranking_admitted_v6",
        "conplex_pair_ranking_admitted_v6",
        "boltz_binding_pair_ranking_admitted_v6",
        "boltz_current_role_v6",
        "boltz_recalibration_wave_v6",
        "target_model_admission_summary_v6",
        "next_target_action_v6",
        "experimental_holo_downloaded_and_validated_v6",
    ]
    out = data.merge(target[target_columns], on="sequence_key", how="left", validate="many_to_one")
    novelty = pd.read_csv(NOVELTY_AUDIT, low_memory=False)
    novelty_columns = [
        "drug_chembl_id",
        "sequence_key",
        "primary_gene",
        "official_active_moiety_mapping_status_v6",
        "official_active_moiety_chembl_ids_v6",
        "known_pair_class_v6",
        "known_chembl_moa_component_v6",
        "known_chembl_moa_action_types_v6",
        "known_chembl_moa_descriptions_v6",
        "exact_chembl_activity_labels_v6",
        "discovery_eligible_after_known_pair_audit_v6",
        "newly_caught_known_or_conflicting_pair_v6",
    ]
    out = out.merge(
        novelty[novelty_columns],
        on=["drug_chembl_id", "sequence_key", "primary_gene"],
        how="left",
        validate="one_to_one",
    )
    out = add_historical_review(out)
    lane = out["portfolio_lane_v5"].fillna("")
    p1 = lane.str.startswith("P1")
    p2 = lane.str.startswith("P2")
    p3 = lane.str.startswith("P3")
    wave1 = out["boltz_recalibration_wave_v6"].eq("wave1")
    wave2 = out["boltz_recalibration_wave_v6"].eq("wave2")
    membrane = out["target_assay_family"].isin(["transporter", "ion_channel"])

    out["current_binding_claim_v6"] = "uncalibrated_remote_binding_hypothesis"
    out.loc[p1, "current_binding_claim_v6"] = "calibrated_in_domain_qsar_support_not_binding_probability"
    out.loc[p2, "current_binding_claim_v6"] = "validated_ligand_similarity_support_not_remote_discovery"
    out["current_novelty_claim_v6"] = "remote_chemistry_pair_hypothesis"
    out.loc[p1, "current_novelty_claim_v6"] = "new_pair_candidate_within_known_target_chemistry"
    out.loc[p2, "current_novelty_claim_v6"] = "conservative_known_chemistry_extension"
    out.loc[out["known_pair_class_v6"].eq("C1_known_chembl_moa_component"), "current_novelty_claim_v6"] = (
        "known_chembl_moa_rediscovery_control"
    )
    out.loc[out["known_pair_class_v6"].eq("C2_known_quantitative_binding_positive"), "current_novelty_claim_v6"] = (
        "known_quantitative_interaction_not_novel"
    )
    out.loc[out["known_pair_class_v6"].eq("N_known_quantitative_negative"), "current_novelty_claim_v6"] = (
        "known_negative_or_inactive_contradiction"
    )
    out.loc[out["known_pair_class_v6"].eq("R_quantitative_grey_or_conflicting"), "current_novelty_claim_v6"] = (
        "known_quantitative_record_requires_resolution"
    )
    out.loc[out["known_pair_class_v6"].eq("R_active_moiety_mapping_ambiguous"), "current_novelty_claim_v6"] = (
        "active_moiety_identity_unresolved_no_novelty_claim"
    )

    out["candidate_next_action_v6"] = "wait_for_target_specific_physics_model_admission"
    out.loc[p1, "candidate_next_action_v6"] = (
        "exact_pair_novelty_exposure_active_species_and_assay_audit"
    )
    out.loc[p2, "candidate_next_action_v6"] = "conservative_side_target_or_rediscovery_audit"
    out.loc[p3 & wave2, "candidate_next_action_v6"] = "hold_for_boltz_recalibration_wave2"
    out.loc[p3 & wave1, "candidate_next_action_v6"] = "link_to_boltz_recalibration_wave1"
    out.loc[p3 & membrane, "candidate_next_action_v6"] = "separate_membrane_state_specific_protocol"
    out.loc[out["known_pair_class_v6"].eq("C1_known_chembl_moa_component"), "candidate_next_action_v6"] = (
        "move_to_positive_control_known_moa"
    )
    out.loc[out["known_pair_class_v6"].eq("C2_known_quantitative_binding_positive"), "candidate_next_action_v6"] = (
        "move_to_literature_validated_interaction_control"
    )
    out.loc[out["known_pair_class_v6"].eq("N_known_quantitative_negative"), "candidate_next_action_v6"] = (
        "move_to_negative_control_or_drop"
    )
    out.loc[out["known_pair_class_v6"].eq("R_quantitative_grey_or_conflicting"), "candidate_next_action_v6"] = (
        "manual_resolve_chembl_conflict_before_progress"
    )
    out.loc[out["known_pair_class_v6"].eq("R_active_moiety_mapping_ambiguous"), "candidate_next_action_v6"] = (
        "resolve_active_moiety_identity_before_progress"
    )

    out["existing_boltz_result_reuse_v6"] = "not_available_or_not_needed"
    out.loc[p3, "existing_boltz_result_reuse_v6"] = (
        "reuse_pose_files_only_after_same_target_protocol_calibration"
    )
    out["existing_conplex_result_reuse_v6"] = "retrieval_provenance_only"
    out["disease_evidence_role_v6"] = "post_binding_annotation_only"
    out["cross_lane_global_rank_valid_v6"] = False
    out["pair_id_v6"] = "V6_" + out["drug_chembl_id"].astype(str) + "_" + out["sequence_key"].astype(str)
    return out


def teacher_table(data: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "portfolio_rank_v5": "原组合顺序_不可跨层解释",
        "portfolio_lane_zh_v5": "证据层",
        "display_drug_name_v5": "FDA药物",
        "drug_chembl_id": "药物ChEMBL",
        "primary_gene": "候选靶点",
        "sequence_key": "项目序列",
        "target_assay_family": "靶点实验类型",
        "calibration_tier_v5": "ChEMBL正负校准层级",
        "target_model_admission_summary_v6": "靶点模型准入状态",
        "current_binding_claim_v6": "当前结合证据可声称范围",
        "current_novelty_claim_v6": "当前新颖性定位",
        "known_pair_class_v6": "ChEMBL37已知Pair审计分类",
        "official_active_moiety_mapping_status_v6": "活性成分映射状态",
        "known_chembl_moa_descriptions_v6": "ChEMBL已知MoA",
        "exact_chembl_activity_labels_v6": "ChEMBL精确活性记录",
        "discovery_eligible_after_known_pair_audit_v6": "通过ChEMBL已知Pair排除",
        "max_known_active_similarity_v5": "与已知活性配体最大相似度",
        "target_qsar_percentile_v5": "靶点内QSAR百分位",
        "conplex_score": "ConPLEx原始分_仅召回",
        "boltz_affinity_probability_refined": "Boltz二分类输出_未校准",
        "pose_stability_tier": "Boltz条件姿势稳定性",
        "boltz_recalibration_wave_v6": "Boltz靶点重校准波次",
        "historical_pair_deep_review_reused_v6": "历史Pair深审已复用",
        "literature_evidence_tier_v6": "历史文献证据层级",
        "literature_judgment_v6": "历史文献判断",
        "exposure_feasibility_v6": "历史暴露可行性",
        "candidate_disease_v6": "历史候选疾病",
        "candidate_next_action_v6": "下一步动作",
        "existing_boltz_result_reuse_v6": "现有Boltz结果用途",
        "original_indication_v5": "FDA原适应症_非推荐疾病",
        "pair_id_v6": "Pair_ID",
    }
    present = [column for column in columns if column in data.columns]
    return data[present].rename(columns=columns)


def build_report(
    data: pd.DataFrame,
    target: pd.DataFrame,
    shortlist: pd.DataFrame,
    smoke_input: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    lane_counts = data["portfolio_lane_v5"].value_counts().to_dict()
    smoke_targets = sorted(smoke_input["target"].dropna().astype(str).unique())
    empty_msa = int(smoke_input.get("boltzSequenceSource", pd.Series(dtype=str)).eq("empty_single_sequence").sum())
    af_rows = int(smoke_input["receptorPdbPath"].fillna("").str.contains("alphafold", case=False).sum())
    pocket_rows = int(bools(smoke_input["pocketConstraintUsed"]).sum())
    lines = [
        "# FDA 老药新靶点：现有证据复用与下一阶段行动方案 V6",
        "",
        "## 一、当前决策",
        "",
        "现有结果不推倒重来，也不再合成为一个跨方法总分。正式组合改为：ChEMBL 正负集决定模型在每个靶点上是否准入；P1/P2 承担已知化学空间内的精度通道；P3 承担远程新颖性通道；结构、疾病、文献与实验信息分别用于条件质控、后处理和可执行性。",
        "",
        f"当前 1000 条保持不变：P1 {lane_counts.get('P1_calibrated_target_qsar_in_domain', 0)}、P2 {lane_counts.get('P2_validated_ligand_similarity_in_domain', 0)}、P3 {lane_counts.get('P3_remote_uncalibrated_physics_exploration', 0)}。它们不是同一置信度，也不存在有效的跨层统一顺序。",
        "",
        "## 二、现有方法的正式用途",
        "",
        "| 现有结果 | 继续使用方式 | 禁止解释 |",
        "| --- | --- | --- |",
        "| ChEMBL 37 正负集 | 靶点级模型准入与基准构建 | FDA 候选本身的结合证明 |",
        "| 靶点专属 QSAR | 通过 scaffold/time 审计后的域内 P1 排序 | 远程新骨架发现、跨靶点概率 |",
        "| 配体相似性 | 经验证靶点上的 P2 保守扩展 | de novo 新靶点发现 |",
        "| ConPLEx | P3 快速召回和遗漏保护 | 独立亲和证据 |",
        "| 现有 Boltz 3000/1000 | pose 文件、重复性、输入完整性和条件结构假说 | 已校准 binder probability |",
        "| AlphaFold/P2Rank/PUResNet/OT tractability | 靶点路由、口袋和实验可做性 | 具体 drug-target 结合支持 |",
        "| 文献、Agent、暴露、active species、assay | Pair 深审与实验准备；仅复用完全相同 pair | 生成新的物理证据 |",
        "| Open Targets/疾病图谱 | 结合假说之后的疾病与机制收敛 | 反向证明亲和 |",
        "",
        "## 三、Boltz pilot 的重新判定",
        "",
        f"本地 pilot 使用了正确的 `boltzAffinityProbabilityBinary`。20 条全部使用预测口袋约束，其中 AlphaFold 受体 {af_rows}/20；输入靶点为 {', '.join(smoke_targets)}。当前输入没有形成成熟 holo enzyme/kinase 主导的公平 benchmark。",
        "",
        "正负样本按骨架多样性抽取而非同 assay、理化性质和化学系列匹配；每靶点仅 1+1。故 ROC-AUC 0.59 的正式解释是“当前 pilot 设计没有提供判别证据”，不是“Boltz-2 已被否定”。现有 Boltz affinity 输出继续冻结为未校准字段，pose 和文件可复用。",
        "",
        "## 四、下一阶段：先建立靶点级模型准入",
        "",
        "Wave 1 只使用成熟、非膜、ChEMBL 正负充足且有外部配体结构证据的 5 个靶点：MAPK14、MAP2K1、PDE4D、BACE1、ESR2。每靶点 20 个强阳性与 20 个强阴性，优先同 assay/文献匹配，再匹配 MW、cLogP、电荷、TPSA 和相似性。",
        "",
        "必须分别评估三个问题：`affinity_probability_binary` 的 binder classification；`affinity_pred_value` 在已知 binder 系列中的连续 affinity ranking；实验 holo 配体的 pose reproduction。三个结果不得再合成单一 composite。",
        "",
        "Wave 1 通过后才运行 Wave 2 的 10 个靶点。只有某个靶点在同协议正负基准上通过，既有 P3 Boltz 结果才可在该靶点内重新解释；若协议改变，则只重跑该靶点的 P3 FDA pair，不重跑全部 3000。",
        "",
        "## 五、当前 1000 条的使用方式",
        "",
        "- P1：先做 exact-pair ChEMBL/PubMed 排重、active species、暴露和 assay 审计；它是精度通道，不是远程创新通道。",
        "- P2：作为保守 side-target/rediscovery 队列和流程阳性参照，不能计入远程新骨架发现率。",
        "- P3：保留 474 条作为创新池；按靶点等待 Boltz/docking 的准入结果，不再因为 Boltz A/B 或高置信结构直接晋级。",
        "- Transporter/ion-channel：建立膜状态和功能 assay 专用协议，不与可溶性 enzyme/kinase 共用同一校准阈值。",
        "",
        "## 六、历史结果复用范围",
        "",
        f"当前 V5 1000 中有 {summary['historical_pair_review_reused_rows']} 条与旧 V7 深审完全同 pair，可直接复用其文献、暴露、active species、assay 和疾病判断；其余 {1000-summary['historical_pair_review_reused_rows']} 条不能按相同药物或相同靶点自动继承 pair 结论。",
        "",
        "## 七、已知 Pair 与新颖性纠错",
        "",
        f"通过 FDA 结构到 ChEMBL 37 上市活性成分映射，并联合 drug mechanism 与精确定量 binding pair，旧标签共重新分流 {summary['newly_caught_known_or_conflicting_rows']} 条：明确 MoA {summary['known_pair_class_counts'].get('C1_known_chembl_moa_component', 0)}、定量阳性 {summary['known_pair_class_counts'].get('C2_known_quantitative_binding_positive', 0)}、定量阴性 {summary['known_pair_class_counts'].get('N_known_quantitative_negative', 0)}、灰区/冲突 {summary['known_pair_class_counts'].get('R_quantitative_grey_or_conflicting', 0)}、活性成分映射待解决 {summary['known_pair_class_counts'].get('R_active_moiety_mapping_ambiguous', 0)}。通过本地 ChEMBL 37 已知 pair 排除后剩余 {summary['known_pair_class_counts'].get('D_unreported_in_local_chembl37', 0)} 条。",
        "",
        f"这些 {summary['known_pair_class_counts'].get('D_unreported_in_local_chembl37', 0)} 条只能称为“本地 ChEMBL 37 未报道”，仍需 exact-pair PubMed 和专利审计后才能称为文献未报道。已知阳性转入 positive-control/rediscovery，已知阴性转入 negative-control 或停止，灰区/冲突与活性成分映射问题进入人工判定。",
        "",
        "## 八、交付文件",
        "",
        "- `TARGET_MODEL_ADMISSION_MATRIX_V6.csv`：每个靶点的模型准入和下一步。",
        "- `BOLTZ_RECALIBRATION_TARGET_SHORTLIST_V6.csv`：Wave 1/2 靶点与现有 P3 覆盖。",
        "- `CURRENT1000_EVIDENCE_REUSE_ACTION_V6.csv`：完整 1000 条逐条行动表。",
        "- `CURRENT1000_EVIDENCE_REUSE_TEACHER_ZH_V6.csv`：简洁中文表。",
        "- `CURRENT882_LOCAL_CHEMBL_UNREPORTED_ACTION_V6.csv`：通过本地已知 Pair 排除的发现审阅池。",
        "- `CURRENT50_POSITIVE_CONTROL_V6.csv`：已知 MoA 或定量阳性对照。",
        "- `CURRENT12_NEGATIVE_CONTROL_V6.csv`：已知定量阴性对照。",
        "- `CURRENT56_IDENTITY_OR_ACTIVITY_HOLD_V6.csv`：活性成分映射、灰区或冲突待解决。",
        "- `EVIDENCE_REUSE_ACTION_AUDIT_V6.json`：一致性和哈希审计。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    inputs = [
        CONFIG,
        V5,
        COVERAGE,
        CALIBRATION_PAIRS,
        QSAR,
        CONPLEX,
        TARGETS,
        ANCHORS,
        SMOKE,
        SMOKE_INPUT,
        OLD_REVIEW,
        NOVELTY_AUDIT,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = load_config()
    data = pd.read_csv(V5, low_memory=False)
    if len(data) != 1000 or data[["drug_chembl_id", "sequence_key"]].duplicated().any():
        raise ValueError("V5 portfolio must contain 1000 unique pairs")
    target = build_target_matrix(data, config)
    actions = build_candidate_actions(data, target)
    teacher = teacher_table(actions)
    shortlist = target[target["boltz_recalibration_wave_v6"].ne("not_selected")].copy()
    shortlist = shortlist.sort_values(
        ["boltz_recalibration_wave_v6", "target_assay_family", "primary_gene"], kind="mergesort"
    )
    smoke_input = pd.read_csv(SMOKE_INPUT, low_memory=False)

    OUT.mkdir(parents=True, exist_ok=True)
    target_path = OUT / "TARGET_MODEL_ADMISSION_MATRIX_V6.csv"
    shortlist_path = OUT / "BOLTZ_RECALIBRATION_TARGET_SHORTLIST_V6.csv"
    actions_path = OUT / "CURRENT1000_EVIDENCE_REUSE_ACTION_V6.csv"
    teacher_path = OUT / "CURRENT1000_EVIDENCE_REUSE_TEACHER_ZH_V6.csv"
    discovery_path = OUT / "CURRENT882_LOCAL_CHEMBL_UNREPORTED_ACTION_V6.csv"
    positive_control_path = OUT / "CURRENT50_POSITIVE_CONTROL_V6.csv"
    negative_control_path = OUT / "CURRENT12_NEGATIVE_CONTROL_V6.csv"
    hold_path = OUT / "CURRENT56_IDENTITY_OR_ACTIVITY_HOLD_V6.csv"
    target.to_csv(target_path, index=False)
    shortlist.to_csv(shortlist_path, index=False)
    actions.to_csv(actions_path, index=False)
    teacher.to_csv(teacher_path, index=False)
    actions[actions["known_pair_class_v6"].eq("D_unreported_in_local_chembl37")].to_csv(
        discovery_path, index=False
    )
    actions[
        actions["known_pair_class_v6"].isin(
            ["C1_known_chembl_moa_component", "C2_known_quantitative_binding_positive"]
        )
    ].to_csv(positive_control_path, index=False)
    actions[actions["known_pair_class_v6"].eq("N_known_quantitative_negative")].to_csv(
        negative_control_path, index=False
    )
    actions[
        actions["known_pair_class_v6"].isin(
            ["R_active_moiety_mapping_ambiguous", "R_quantitative_grey_or_conflicting"]
        )
    ].to_csv(hold_path, index=False)

    lane_counts = actions["portfolio_lane_v5"].value_counts().to_dict()
    checks = {
        "candidate_rows_1000": len(actions) == 1000,
        "teacher_rows_1000": len(teacher) == 1000,
        "candidate_pairs_unique": not actions[["drug_chembl_id", "sequence_key"]].duplicated().any(),
        "target_rows_match_unique_targets": len(target) == data["sequence_key"].nunique(),
        "p1_qsar_admitted": actions.loc[actions["portfolio_lane_v5"].str.startswith("P1"), "qsar_pair_ranking_admitted_v6"].all(),
        "p2_similarity_admitted": actions.loc[actions["portfolio_lane_v5"].str.startswith("P2"), "similarity_pair_ranking_admitted_v6"].all(),
        "boltz_binding_not_globally_admitted": not actions["boltz_binding_pair_ranking_admitted_v6"].any(),
        "cross_lane_global_rank_disabled": not actions["cross_lane_global_rank_valid_v6"].any(),
        "wave1_exact_5": shortlist["boltz_recalibration_wave_v6"].eq("wave1").sum() == 5,
        "wave2_exact_10": shortlist["boltz_recalibration_wave_v6"].eq("wave2").sum() == 10,
        "selected_targets_eligible": shortlist["boltz_recalibration_eligible_v6"].all(),
        "known_pair_audit_complete": actions["known_pair_class_v6"].notna().all(),
        "known_pair_audit_partitions_1000": actions["known_pair_class_v6"].value_counts().sum() == 1000,
        "local_chembl_unreported_882": int(actions["discovery_eligible_after_known_pair_audit_v6"].sum()) == 882,
        "positive_controls_50": int(
            actions["known_pair_class_v6"].isin(
                ["C1_known_chembl_moa_component", "C2_known_quantitative_binding_positive"]
            ).sum()
        )
        == 50,
        "negative_controls_12": int(actions["known_pair_class_v6"].eq("N_known_quantitative_negative").sum())
        == 12,
        "identity_or_activity_hold_56": int(
            actions["known_pair_class_v6"].isin(
                ["R_active_moiety_mapping_ambiguous", "R_quantitative_grey_or_conflicting"]
            ).sum()
        )
        == 56,
        "lane_count_unchanged": lane_counts
        == {
            "P3_remote_uncalibrated_physics_exploration": 474,
            "P2_validated_ligand_similarity_in_domain": 300,
            "P1_calibrated_target_qsar_in_domain": 226,
        },
    }
    checks = {key: bool(value) for key, value in checks.items()}
    failures = sorted(key for key, value in checks.items() if not value)
    summary = {
        "status": "passed" if not failures else "failed",
        "created_utc": now(),
        "checks": checks,
        "failures": failures,
        "candidate_rows": int(len(actions)),
        "target_rows": int(len(target)),
        "lane_counts": {str(key): int(value) for key, value in lane_counts.items()},
        "historical_pair_review_reused_rows": int(actions["historical_pair_deep_review_reused_v6"].sum()),
        "known_pair_class_counts": {
            str(key): int(value) for key, value in actions["known_pair_class_v6"].value_counts().items()
        },
        "newly_caught_known_or_conflicting_rows": int(
            actions["newly_caught_known_or_conflicting_pair_v6"].sum()
        ),
        "boltz_wave1_targets": shortlist.loc[
            shortlist["boltz_recalibration_wave_v6"].eq("wave1"), "primary_gene"
        ].tolist(),
        "boltz_wave2_targets": shortlist.loc[
            shortlist["boltz_recalibration_wave_v6"].eq("wave2"), "primary_gene"
        ].tolist(),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    # Pandas scalar conversion is explicit to prevent bool serialization surprises.
    summary["p3_pairs_covered_by_wave1"] = int(
        (
            actions["portfolio_lane_v5"].str.startswith("P3")
            & actions["boltz_recalibration_wave_v6"].eq("wave1")
        ).sum()
    )
    summary["p3_pairs_covered_by_wave2"] = int(
        (
            actions["portfolio_lane_v5"].str.startswith("P3")
            & actions["boltz_recalibration_wave_v6"].eq("wave2")
        ).sum()
    )

    report_text = build_report(actions, target, shortlist, smoke_input, summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text, encoding="utf-8")
    (OUT / REPORT.name).write_text(report_text, encoding="utf-8")
    output_paths = [
        target_path,
        shortlist_path,
        actions_path,
        teacher_path,
        discovery_path,
        positive_control_path,
        negative_control_path,
        hold_path,
        REPORT,
    ]
    summary["outputs"] = {str(path.relative_to(ROOT)): sha256(path) for path in output_paths}
    audit_path = OUT / "EVIDENCE_REUSE_ACTION_AUDIT_V6.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

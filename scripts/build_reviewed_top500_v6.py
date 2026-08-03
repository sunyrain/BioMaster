#!/usr/bin/env python3
"""Build a uniformly audited final1000 and a physics-first discovery Top500."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


OT_SCORE_COLUMNS = [
    "genetic_association_score",
    "somatic_mutation_score",
    "clinical_score",
    "affected_pathway_score",
    "literature_score",
    "animal_model_score",
    "rna_expression_score",
    "genetic_literature_score",
]

BROAD_DISEASE_NAMES = {
    "cancer",
    "carcinoma",
    "disease",
    "genetic disease",
    "hereditary disease",
    "infectious disease",
    "immune system disease",
    "metabolic disease",
    "neoplasm",
    "nervous system disease",
    "respiratory system disease",
    "syndrome",
}

AREA_PATTERNS = [
    ("Oncology", r"cancer|carcinoma|neoplasm|tumou?r|lymphoma|leukemia|melanoma|myeloma|sarcoma|glioma|blastoma|癌|瘤|白血病|骨髓瘤"),
    ("Neurology/Psychiatry", r"alzheimer|parkinson|dementia|epilep|seizure|schizophren|depress|anxiety|bipolar|neuro|neuropath|migraine|autism|amyotrophic|ataxia|cognitive|brunner|神经|癫痫|帕金森|精神分裂|抑郁|焦虑|疼痛"),
    ("Cardiovascular", r"cardio|cardiac|heart|coronary|myocard|hypertension|atherosclero|arteriosclero|arrhythm|stroke|thrombo|vascular|hypotension|心衰|心血管|房颤|卒中|动脉|血栓"),
    ("Hematology", r"anemia|haem|hemat|blood|coagulat|platelet|pancytopen|neutropen|hemophil|hemorrhage|贫血|血小板|凝血|出血"),
    ("Immunology/Inflammation", r"immune|inflamm|arthritis|lupus|psoriasis|allerg|asthma|atopic|crohn|ulcerative colitis|autoimmune|spondylitis|免疫|炎症|关节炎|狼疮|银屑病|哮喘|结肠炎"),
    ("Endocrinology/Metabolic", r"diabetes|obesity|metabolic|thyroid|adrenal|lipid|cholesterol|hyperlip|phenylketonuria|hyperphenyl|glycogen|endocrin|osteoporosis|gout|hyperuric|gaucher|tyrosinemia|desmosterol|orotic aciduria|糖尿病|代谢|甲状腺|苯丙酮尿|高胆固醇|高甘油|高尿酸|痛风|戈谢|酪氨酸"),
    ("Urology/Nephrology", r"kidney|renal|nephro|urinary|urolog|bladder|glomerul|肾|泌尿"),
    ("Gastroenterology", r"gastro|intestinal|bowel|liver|hepatic|cirrhosis|pancreatitis|cholest|bile|gallbladder|steatosis|steatohepat|肝|胃|肠|胆"),
    ("Respiratory", r"pulmonary|respiratory|lung|bronch|airway|copd|肺|呼吸|气道"),
    ("Infectious Disease", r"infect|viral|virus|bacter|fung|hiv|viral hepatitis|hepatitis [abc]|tuberculo|covid|malaria|leishmani|pathogen|感染|病毒|细菌|真菌"),
    ("Dermatology", r"skin|dermat|eczema|acne|alopecia|ichthyosis|vitiligo|皮肤|湿疹|白癜风"),
    ("Musculoskeletal", r"muscle|muscular|bone|skeletal|osteoarthritis|dystrophy|arthrogryposis|肌|骨"),
    ("Ophthalmology", r"retina|retinit|glaucoma|macular|ocular|ophthalm|amaurosis|blindness|keratoconus|eye|视网膜|青光眼|眼"),
    ("Reproductive", r"ovarian|uterine|endometri|fertility|menstrual|pregnan|testicular|卵巢|子宫|妊娠|生殖"),
    ("Rare Disease", r"congenital|familial|deficiency|orphan|rare|inherited|syndrome|先天|家族性|缺陷|综合征"),
]

CHINESE_AREA = {
    "Oncology": "肿瘤",
    "Neurology/Psychiatry": "神经/精神",
    "Cardiovascular": "心血管",
    "Hematology": "血液",
    "Immunology/Inflammation": "免疫/炎症",
    "Infectious Disease": "感染",
    "Endocrinology/Metabolic": "内分泌/代谢",
    "Urology/Nephrology": "泌尿/肾脏",
    "Gastroenterology": "消化/肝胆",
    "Respiratory": "呼吸",
    "Dermatology": "皮肤",
    "Musculoskeletal": "肌肉骨骼",
    "Ophthalmology": "眼科",
    "Reproductive": "生殖",
    "Rare Disease": "罕见病",
    "Other": "其他/未分类",
}


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"1", "1.0", "true", "yes", "y"}


def infer_area(text: Any) -> str:
    lowered = clean(text).lower()
    for area, pattern in AREA_PATTERNS:
        if re.search(pattern, lowered):
            return area
    return "Other"


def is_specific_disease(name: Any) -> bool:
    lowered = clean(name).lower()
    if not lowered or lowered in BROAD_DISEASE_NAMES:
        return False
    if lowered.startswith("measurement of ") or lowered.startswith("response to "):
        return False
    if "abnormality of" in lowered or "biological process" in lowered:
        return False
    return True


def evidence_channels(row: pd.Series) -> str:
    labels = {
        "genetic_association_score": "genetic",
        "somatic_mutation_score": "somatic",
        "clinical_score": "clinical",
        "affected_pathway_score": "pathway",
        "literature_score": "literature",
        "animal_model_score": "animal_model",
        "rna_expression_score": "rna_expression",
        "genetic_literature_score": "genetic_literature",
    }
    ranked = sorted(
        ((labels[column], float(row[column])) for column in OT_SCORE_COLUMNS if float(row[column]) > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return "; ".join(f"{label}={score:.3f}" for label, score in ranked[:5])


def build_ot_target_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    frame = long_df.copy()
    for column in ["overall_score", *OT_SCORE_COLUMNS]:
        frame[column] = numeric(frame[column])
    frame["ot_evidence_breadth_v6"] = (frame[OT_SCORE_COLUMNS] >= 0.05).sum(axis=1)
    genetic = frame[["genetic_association_score", "somatic_mutation_score"]].max(axis=1)
    frame["ot_disease_priority_score_v6"] = (
        0.35 * frame["overall_score"]
        + 0.25 * genetic
        + 0.20 * frame["clinical_score"]
        + 0.08 * frame["affected_pathway_score"]
        + 0.06 * frame["literature_score"]
        + 0.03 * frame["animal_model_score"]
        + 0.03 * (frame["ot_evidence_breadth_v6"].clip(upper=8) / 8.0)
    )
    frame.loc[~frame["disease_name"].map(is_specific_disease), "ot_disease_priority_score_v6"] -= 0.25
    chosen = (
        frame.sort_values(
            ["target_gene", "query_ensembl_id", "ot_disease_priority_score_v6", "overall_score"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(["target_gene", "query_ensembl_id"])
        .copy()
    )
    chosen["ot_primary_disease_v6"] = chosen["disease_name"].map(clean)
    chosen["ot_primary_disease_id_v6"] = chosen["disease_id"].map(clean)
    chosen["ot_primary_disease_area_v6"] = chosen["ot_primary_disease_v6"].map(infer_area)
    chosen["ot_primary_disease_evidence_channels_v6"] = chosen.apply(evidence_channels, axis=1)
    score = chosen["ot_disease_priority_score_v6"]
    breadth = chosen["ot_evidence_breadth_v6"]
    chosen["ot_primary_disease_evidence_tier_v6"] = np.select(
        [(score >= 0.55) & (breadth >= 2), score >= 0.30],
        ["A_multi_channel_target_disease", "B_supported_target_disease"],
        default="C_context_only_target_disease",
    )
    return chosen.rename(
        columns={"target_gene": "primary_gene", "query_ensembl_id": "anchor_opentargets_id"}
    )[
        [
            "primary_gene",
            "anchor_opentargets_id",
            "ot_primary_disease_v6",
            "ot_primary_disease_id_v6",
            "ot_primary_disease_area_v6",
            "ot_primary_disease_evidence_tier_v6",
            "ot_primary_disease_evidence_channels_v6",
            "ot_disease_priority_score_v6",
            "ot_evidence_breadth_v6",
            "overall_score",
        ]
    ].rename(columns={"overall_score": "ot_primary_disease_overall_score_v6"})


def merge_external_evidence(
    chembl: pd.DataFrame,
    pubmed: pd.DataFrame,
    review: pd.DataFrame,
    ot_summary: pd.DataFrame,
) -> pd.DataFrame:
    pubmed = pubmed.copy()
    if pubmed.duplicated(["drug_chembl_id", "candidate_anchor_gene"]).any():
        raise ValueError("PubMed audit contains duplicate drug-target rows")
    pubmed_fields = [
        "drug_chembl_id",
        "candidate_anchor_gene",
        "lit_ok",
        "pair_pubmed_count_2000_2026",
        "pair_pubmed_pmids_2000_2026",
        "pair_pubmed_url_2000_2026",
        "post_approval_pair_pubmed_count",
        "post_approval_pair_pubmed_pmids",
        "post_approval_pair_pubmed_url",
        "pubmed_screen_tier_v6",
        "pubmed_direct_measurement_language",
        "pubmed_functional_language",
        "pubmed_computational_language",
        "pubmed_screen_titles_v6",
        "pubmed_screen_dois_v6",
        "pubmed_screen_note_v6",
    ]
    base = chembl.drop(columns=[column for column in pubmed_fields[2:] if column in chembl.columns])
    base = base.merge(
        pubmed[pubmed_fields],
        left_on=["drug_chembl_id", "primary_gene"],
        right_on=["drug_chembl_id", "candidate_anchor_gene"],
        how="left",
        validate="one_to_one",
    ).drop(columns="candidate_anchor_gene")

    agent_columns = [column for column in review.columns if column.startswith("agent_")]
    review_fields = ["pair_id", "selected_final384_v4", "post_review_disposition_v4", *agent_columns]
    review_subset = review[[column for column in review_fields if column in review.columns]].copy()
    review_subset = review_subset.drop_duplicates("pair_id")
    base = base.drop(columns=[column for column in review_subset.columns if column != "pair_id" and column in base.columns])
    base = base.merge(review_subset, on="pair_id", how="left", validate="one_to_one")
    base = base.merge(
        ot_summary,
        on=["primary_gene", "anchor_opentargets_id"],
        how="left",
        validate="many_to_one",
    )
    return base.fillna("")


def classify_literature(row: pd.Series) -> tuple[str, str]:
    prior = clean(row.get("agent_literature_class"))
    chembl = clean(row.get("chembl_exact_activity_status"))
    pubmed = clean(row.get("pubmed_screen_tier_v6"))
    if prior == "exact_pair_validated":
        return "L1_prior_manual_exact_pair_validated", "既有逐文审阅确认精确药物-靶点关系；按再发现/对照处理。"
    if prior == "contradictory":
        return "L7_prior_manual_contradictory", "逐文审阅发现同一药物活性实体与人源靶点的直接阴性或矛盾记录；不进入发现池。"
    if prior == "functional_only":
        return "L3_prior_manual_functional_record", "逐文审阅确认精确药物-靶点功能记录，但不能据此自动认定直接结合。"
    if prior == "indirect_or_family_only":
        return "L5_prior_manual_indirect_or_family_only", "逐文审阅仅确认间接、同家族或通路证据，不能认定精确直接互作。"
    if chembl == "exact_binding_activity_pchembl_ge_5":
        return "L1_strict_chembl_binding_validated", "ChEMBL存在通过assay质量过滤的精确binding记录（pChEMBL>=5）。"
    if chembl == "exact_binding_record_without_strong_standardized_potency":
        return "L2_exact_binding_record_quality_or_potency_limited", "ChEMBL有精确binding记录，但标准化效力或assay质量不足，需回看原始实验。"
    if chembl == "manual_exact_binding_review":
        return "L2_exact_binding_record_manual_review", "ChEMBL有候选强binding记录，但assay元数据未通过自动严格过滤。"
    if chembl == "exact_nonbinding_activity_record":
        return "L3_exact_functional_or_nonbinding_record", "ChEMBL仅见非binding类型精确记录，不能自动证明直接结合。"
    if pubmed.startswith("P1_"):
        return "L4_pubmed_direct_language_manual_validation_needed", "摘要含直接测量措辞，但尚未逐文确认该措辞是否对应本精确pair。"
    if pubmed.startswith("P2_"):
        return "L5_pubmed_functional_language_manual_validation_needed", "摘要含功能调控措辞，不能据此认定直接结合。"
    if pubmed.startswith("P3_"):
        return "L6_pubmed_cooccurrence_only", "仅发现药名与靶点同文共现，未见直接测量措辞。"
    return "L0_no_exact_public_record_found_by_current_screen", "当前ChEMBL与精确词项PubMed筛查未发现公开精确pair证据。"


def active_species_status(row: pd.Series) -> str:
    prior = clean(row.get("agent_active_species_status"))
    if prior:
        return prior
    if clean(row.get("active_moiety_smiles")) == clean(row.get("canonical_smiles_rdkit")):
        return "parent_drug_relevant"
    return "salt_normalization_adequate"


def generated_feasibility_grade(row: pd.Series) -> str:
    strength = clean(row.get("v5_strength_tier"))[:1]
    family = clean(row.get("target_assay_family"))
    points = {"A": 3, "B": 2, "C": 1}.get(strength, 0)
    points += 2 if family in {"enzyme", "kinase"} else 1 if family == "nuclear_epigenetic" else 0
    readiness = float(row.get("v5_experimental_readiness_score") or 0)
    points += 2 if readiness >= 98 else 1 if readiness >= 94 else 0
    points += int(clean(row.get("pose_stability_tier")).startswith("A_"))
    points += int(clean(row.get("strict_structure_tier")).startswith("A_"))
    points += int(active_species_status(row) in {"parent_drug_relevant", "salt_normalization_adequate"})
    if family in {"enzyme", "kinase"} and points >= 8:
        return "B"
    if family == "nuclear_epigenetic" and points >= 9:
        return "B"
    return "C"


def generated_assay_plan(row: pd.Series) -> str:
    primary = clean(row.get("default_assay_strategy")) or "orthogonal target-engagement assay"
    return (
        f"Primary: {primary}; positive control: 该靶点已知高质量小分子; "
        "counterscreen: FDA原靶点/近邻家族; gates: 浓度依赖、重复性、细胞活力与聚集/膜干扰。"
    )


def generated_risks(row: pd.Series) -> str:
    risks = ["计算亲和不等于真实结合", "作用方向未知", "尚未建立人体游离暴露桥接"]
    family = clean(row.get("target_assay_family"))
    if family == "transporter":
        risks.append("膜转运体构象与功能assay依赖性高")
    elif family == "ion_channel":
        risks.append("离子通道需膜片钳/电生理复核")
    elif family == "nuclear_epigenetic":
        risks.append("需区分直接结合、辅因子效应和转录继发效应")
    if clean(row.get("chembl_exact_activity_status")) != "no_exact_chembl_activity_record":
        risks.append("存在既往精确记录，新颖性需降级")
    if clean(row.get("pubmed_screen_tier_v6")) != "P0_no_pair_cooccurrence":
        risks.append("PubMed命中需逐文核实")
    return "；".join(risks)


def generated_verdict(row: pd.Series, grade: str) -> str:
    family = clean(row.get("target_assay_family"))
    strength = clean(row.get("v5_strength_tier"))[:1]
    if clean(row.get("agent_literature_class")) == "exact_pair_validated":
        return "精确pair已有人工文献验证，移入阳性/再发现对照，不作为全新靶点发现。"
    if family in {"transporter", "ion_channel"}:
        return f"{grade}级保留：{strength}级物理证据可继续，但膜蛋白功能验证难度较高，需专门assay。"
    if family == "nuclear_epigenetic":
        return f"{grade}级保留：{strength}级物理证据，先做直接结合与报告基因/酶学正交验证。"
    return f"{grade}级保留：{strength}级物理证据且有可执行的{family}实验；先验证结合/活性，再讨论疾病。"


def compare_repurposing(row: pd.Series, disease: str, area: str, deep_review: bool) -> str:
    prior = clean(row.get("agent_repurposing_status"))
    if deep_review and prior:
        return prior
    indication = clean(row.get("fda_indication")).lower()
    normalized_disease = disease.lower()
    if len(normalized_disease) >= 7 and normalized_disease in indication:
        return "original_indication_or_not_repurposing_hypothesis"
    original_areas = {clean(value) for value in re.split(r"[;,|]+", clean(row.get("fda_therapeutic_area"))) if clean(value)}
    if not disease or area == "Other":
        return "target_only_no_disease_claim"
    if area in original_areas:
        return "new_indication_same_area_hypothesis"
    return "new_disease_area_hypothesis"


def annotate_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    has_agent_review = result["agent_feasibility_grade"].map(clean).ne("")
    result["review_depth_v6"] = np.where(
        has_agent_review,
        "agent_deep_review_plus_uniform_database_audit",
        "systematic_database_and_rule_review",
    )
    result["manual_pair_literature_review_completed_v6"] = has_agent_review
    result["active_species_status_v6"] = result.apply(active_species_status, axis=1)
    classifications = result.apply(classify_literature, axis=1)
    result["literature_evidence_tier_v6"] = [value[0] for value in classifications]
    result["literature_judgment_v6"] = [value[1] for value in classifications]

    generated_grades = result.apply(generated_feasibility_grade, axis=1)
    result["feasibility_grade_v6"] = np.where(
        result["agent_feasibility_grade"].map(clean).ne(""),
        result["agent_feasibility_grade"].map(clean),
        generated_grades,
    )
    result["review_confidence_v6"] = np.where(
        result["agent_confidence"].map(clean).ne(""),
        result["agent_confidence"].map(clean),
        np.where(result["feasibility_grade_v6"].eq("B"), "medium", "low"),
    )
    result["review_verdict_v6"] = [
        clean(row.get("agent_verdict")) or generated_verdict(row, grade)
        for (_, row), grade in zip(result.iterrows(), result["feasibility_grade_v6"], strict=True)
    ]
    result["mechanism_rationale_v6"] = [
        clean(row.get("agent_mechanism_rationale"))
        or (
            f"ConPLEx与Boltz-2支持{clean(row.get('drug_names'))}-{clean(row.get('primary_gene'))}潜在结合；"
            "模型不能确定激动/抑制方向，需先做直接结合与功能正交实验。"
        )
        for _, row in result.iterrows()
    ]
    result["exposure_feasibility_v6"] = [
        clean(row.get("agent_exposure_feasibility"))
        or "未做定量人体游离暴露桥接；首轮采用浓度梯度，后续以临床可达游离暴露作为go/no-go门。"
        for _, row in result.iterrows()
    ]
    result["assay_plan_v6"] = [
        clean(row.get("agent_assay_plan")) or generated_assay_plan(row) for _, row in result.iterrows()
    ]
    result["key_risks_v6"] = [
        clean(row.get("agent_key_risks")) or generated_risks(row) for _, row in result.iterrows()
    ]

    disease_values = []
    disease_basis = []
    disease_areas = []
    repurposing = []
    for _, row in result.iterrows():
        deep = bool(clean(row.get("agent_feasibility_grade")))
        prior_disease = clean(row.get("agent_primary_disease"))
        placeholders = {"未指定", "target_only_no_disease_claim", "not specified", "none"}
        if deep and prior_disease and prior_disease.lower() not in placeholders:
            disease = prior_disease
            basis = "agent_disease_review"
            area = infer_area(disease)
        else:
            disease = clean(row.get("ot_primary_disease_v6"))
            basis = "OpenTargets_target_disease_hypothesis"
            area = clean(row.get("ot_primary_disease_area_v6")) or infer_area(disease)
        disease_values.append(disease or "未指定")
        disease_basis.append(basis)
        disease_areas.append(area or "Other")
        repurposing.append(compare_repurposing(row, disease, area, deep))
    result["candidate_disease_v6"] = disease_values
    result["candidate_disease_basis_v6"] = disease_basis
    result["candidate_disease_area_v6"] = disease_areas
    result["repurposing_interpretation_v6"] = repurposing
    result["disease_claim_limit_v6"] = "靶点-疾病机制语境；药物作用方向与疾病效应均待实验确认"

    result["candidate_role_v6"] = np.select(
        [
            result["literature_evidence_tier_v6"].str.startswith("L7_"),
            result["literature_evidence_tier_v6"].str.startswith("L1_"),
            result["literature_evidence_tier_v6"].str.startswith("L2_"),
            result["literature_evidence_tier_v6"].str.startswith("L3_"),
            result["literature_evidence_tier_v6"].str.startswith(("L4_", "L5_", "L6_")),
        ],
        [
            "contradictory_or_negative_excluded",
            "validated_control_or_rediscovery",
            "reported_pair_revalidation",
            "functional_record_revalidation",
            "literature_triage_hypothesis",
        ],
        default="novel_binding_hypothesis",
    )
    resolved_species = result["active_species_status_v6"].isin(
        ["parent_drug_relevant", "salt_normalization_adequate"]
    )
    standard_family = result["target_assay_family"].isin(["enzyme", "kinase", "nuclear_epigenetic"])
    result["experimental_execution_tier_v6"] = np.select(
        [
            result["feasibility_grade_v6"].isin(["A", "B"]) & resolved_species & standard_family,
            resolved_species & standard_family,
        ],
        [
            "T1_immediate_standard_assay",
            "T2_standard_assay_major_uncertainty",
        ],
        default="T3_specialized_assay_or_active_species_resolution",
    )
    grade_score = result["feasibility_grade_v6"].map({"A": 100, "B": 80, "C": 55, "D": 10}).fillna(40)
    novelty_score = result["literature_evidence_tier_v6"].map(
        lambda value: 20
        if value.startswith("L1_")
        else 55
        if value.startswith("L2_")
        else 65
        if value.startswith("L3_")
        else 75
        if value.startswith("L4_")
        else 85
        if value.startswith("L5_")
        else 90
        if value.startswith("L6_")
        else 100
    )
    result["review_feasibility_component_v6"] = grade_score
    result["novelty_component_v6"] = novelty_score
    result["top500_selection_score_v6"] = (
        0.75 * numeric(result["v5_pair_physics_score"])
        + 0.15 * numeric(result["v5_experimental_readiness_score"])
        + 0.07 * grade_score
        + 0.03 * novelty_score
    )
    result["v5_strength_order_v6"] = result["v5_strength_tier"].map(
        {
            "A_at_or_above_known_positive_median": 0,
            "B_at_or_above_known_positive_q25": 1,
            "C_at_or_above_known_positive_q10": 2,
        }
    ).fillna(9)
    result["top500_hard_eligible_v6"] = (
        result["v5_hard_eligible"].map(truthy)
        & ~result["literature_evidence_tier_v6"].str.startswith("L1_")
        & ~result["feasibility_grade_v6"].eq("D")
        & ~result["agent_literature_class"].map(clean).eq("contradictory")
        & result["active_species_status_v6"].isin(
            ["parent_drug_relevant", "salt_normalization_adequate"]
        )
        & ~result["agent_database_query_resolution"].map(clean).eq("unresolved")
        & result["chembl_activity_query_ok"].map(truthy)
        & result["lit_ok"].map(truthy)
    )
    return result


def load_hot_targets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["primary_gene", "hot_target_groups_v6", "hot_target_tiers_v6"])
    frame = pd.read_csv(path, low_memory=False).fillna("")
    return frame.rename(
        columns={
            "gene_norm": "primary_gene",
            "hot_target_groups": "hot_target_groups_v6",
            "hot_target_tiers": "hot_target_tiers_v6",
        }
    )[["primary_gene", "hot_target_groups_v6", "hot_target_tiers_v6"]].drop_duplicates("primary_gene")


def greedy_select(frame: pd.DataFrame, n: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = frame.loc[frame["top500_hard_eligible_v6"]].sort_values(
        ["v5_strength_order_v6", "top500_selection_score_v6", "v5_pair_physics_score", "v5_rank"],
        ascending=[True, False, False, True],
    )
    passes = [
        {
            "name": "primary_caps_AB_only",
            "drug": 4,
            "target": 8,
            "scaffold": 12,
            "family": {"enzyme": 330, "nuclear_epigenetic": 90, "kinase": 65, "transporter": 60, "ion_channel": 12},
            "max_strength_order": 1,
        },
        {
            "name": "limited_relaxation_AB_only",
            "drug": 5,
            "target": 10,
            "scaffold": 15,
            "family": {"enzyme": 345, "nuclear_epigenetic": 100, "kinase": 75, "transporter": 70, "ion_channel": 15},
            "max_strength_order": 1,
        },
        {
            "name": "C_tier_only_if_AB_insufficient",
            "drug": 6,
            "target": 14,
            "scaffold": 18,
            "family": {},
            "max_strength_order": 2,
        },
        {
            "name": "minimal_fill_relaxation",
            "drug": 8,
            "target": 20,
            "scaffold": 25,
            "family": {},
            "max_strength_order": 2,
        },
    ]
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    selected_pass: dict[int, str] = {}
    drug_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    scaffold_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    pass_stats: list[dict[str, Any]] = []
    for config in passes:
        before = len(selected_indices)
        for index, row in pool.iterrows():
            if index in selected_set or len(selected_indices) >= n:
                continue
            strength_order = pd.to_numeric(row.get("v5_strength_order_v6"), errors="coerce")
            if pd.isna(strength_order) or float(strength_order) > config["max_strength_order"]:
                continue
            drug = clean(row.get("drug_chembl_id"))
            target = clean(row.get("primary_gene"))
            scaffold = clean(row.get("murcko_scaffold")) or clean(row.get("model_ligand_smiles"))
            family = clean(row.get("target_assay_family"))
            if drug_counts[drug] >= config["drug"]:
                continue
            if target_counts[target] >= config["target"]:
                continue
            if scaffold_counts[scaffold] >= config["scaffold"]:
                continue
            family_cap = config["family"].get(family)
            if family_cap is not None and family_counts[family] >= family_cap:
                continue
            selected_indices.append(index)
            selected_set.add(index)
            selected_pass[index] = config["name"]
            drug_counts[drug] += 1
            target_counts[target] += 1
            scaffold_counts[scaffold] += 1
            family_counts[family] += 1
        pass_stats.append({"pass": config["name"], "added": len(selected_indices) - before, "cumulative": len(selected_indices)})
        if len(selected_indices) >= n:
            break
    if len(selected_indices) != n:
        raise RuntimeError(f"Could select only {len(selected_indices)} of requested {n}")
    selected = frame.loc[selected_indices].copy()
    selected["selection_cap_pass_v6"] = [selected_pass[index] for index in selected_indices]
    selected = selected.sort_values(
        ["v5_strength_order_v6", "top500_selection_score_v6", "v5_pair_physics_score", "v5_rank"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    selected.insert(0, "top500_rank_v6", np.arange(1, len(selected) + 1))
    audit = {
        "eligible_discovery_rows": int(len(pool)),
        "selection_passes": pass_stats,
        "final_caps_observed": {
            "max_pairs_per_drug": int(selected["drug_chembl_id"].value_counts().max()),
            "max_pairs_per_target": int(selected["primary_gene"].value_counts().max()),
            "max_pairs_per_scaffold": int(selected["murcko_scaffold"].value_counts().max()),
        },
    }
    return selected, audit


def readable_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "top500_rank_v6": "Top500顺序",
        "drug_names": "FDA药物",
        "fda_therapeutic_area": "FDA原治疗领域",
        "fda_indication": "FDA原适应症",
        "primary_gene": "候选新靶点",
        "protein_names": "靶点名称",
        "target_assay_family": "靶点实验类型",
        "candidate_disease_v6": "候选细分病种",
        "candidate_disease_area_v6": "候选病种大类",
        "repurposing_interpretation_v6": "老药新用判断",
        "candidate_disease_basis_v6": "病种来源",
        "ot_primary_disease_evidence_tier_v6": "靶点疾病证据层级",
        "ot_primary_disease_evidence_channels_v6": "靶点疾病证据通道",
        "rank_within_drug": "ConPLEx药内Top100名次",
        "rank_within_drug_full891": "ConPLEx全891靶点名次",
        "conplex_score": "ConPLEx原始分",
        "v5_pair_physics_score": "统一物理分",
        "v5_strength_tier": "物理强度档",
        "boltz_support_tier_refined": "Boltz支持档",
        "boltz_affinity_probability_refined": "Boltz亲和概率",
        "boltz_affinity_pred_value_refined": "Boltz亲和值",
        "pose_stability_tier": "双构象稳定性",
        "strict_structure_tier": "口袋结构层级",
        "top_pocket_probability": "口袋概率",
        "literature_evidence_tier_v6": "文献证据层级",
        "literature_judgment_v6": "文献研判",
        "chembl_exact_activity_status": "ChEMBL精确pair状态",
        "chembl_exact_max_binding_pchembl": "ChEMBL最高binding pChEMBL",
        "pair_pubmed_count_2000_2026": "PubMed精确词项命中数",
        "pair_pubmed_pmids_2000_2026": "PubMed PMID",
        "pubmed_screen_tier_v6": "PubMed摘要研判层",
        "review_depth_v6": "研判深度",
        "feasibility_grade_v6": "实验可行性档",
        "experimental_execution_tier_v6": "实验执行层级",
        "review_verdict_v6": "综合结论",
        "mechanism_rationale_v6": "机制解释",
        "exposure_feasibility_v6": "暴露判断",
        "active_species_status_v6": "活性物种状态",
        "assay_plan_v6": "建议实验",
        "key_risks_v6": "主要风险",
        "candidate_role_v6": "候选角色",
        "hot_target_groups_v6": "2026热门靶点标签",
        "top500_selection_score_v6": "Top500选择分",
        "selection_cap_pass_v6": "多样性选择轮次",
        "pair_id": "pair_id",
    }
    available = [column for column in columns if column in frame.columns]
    result = frame[available].rename(columns=columns).copy()
    if "候选病种大类" in result:
        result["候选病种大类中文"] = result["候选病种大类"].map(CHINESE_AREA).fillna("其他/未分类")
        position = result.columns.get_loc("候选病种大类") + 1
        column = result.pop("候选病种大类中文")
        result.insert(position, "候选病种大类中文", column)
    return result


def json_counts(series: pd.Series) -> dict[str, int]:
    return {clean(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def write_excel(path: Path, readable: pd.DataFrame, full: pd.DataFrame, all1000: pd.DataFrame, controls: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readable.to_excel(writer, index=False, sheet_name="Top500中文简表")
        full.to_excel(writer, index=False, sheet_name="Top500完整表")
        all1000.to_excel(writer, index=False, sheet_name="全1000统一研判")
        controls.to_excel(writer, index=False, sheet_name="精确验证对照")
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            for cell in sheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font
            for column_cells in sheet.iter_cols(1, min(sheet.max_column, 45)):
                letter = column_cells[0].column_letter
                header = clean(column_cells[0].value)
                sheet.column_dimensions[letter].width = min(42, max(10, len(header) * 1.6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chembl-audit", required=True)
    parser.add_argument("--pubmed-audit", required=True)
    parser.add_argument("--review512", required=True)
    parser.add_argument("--ot-long", required=True)
    parser.add_argument("--hot-targets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chembl = pd.read_csv(args.chembl_audit, low_memory=False).fillna("")
    pubmed = pd.read_csv(args.pubmed_audit, low_memory=False).fillna("")
    review = pd.read_csv(args.review512, low_memory=False).fillna("")
    ot_long = pd.read_csv(args.ot_long, low_memory=False).fillna("")
    if len(chembl) != 1000 or chembl["pair_id"].nunique() != 1000:
        raise ValueError("Expected exactly 1000 unique V5 pairs")
    ot_summary = build_ot_target_summary(ot_long)
    frame = merge_external_evidence(chembl, pubmed, review, ot_summary)
    frame = annotate_reviews(frame)
    frame = frame.merge(load_hot_targets(Path(args.hot_targets)), on="primary_gene", how="left").fillna("")
    frame["is_hot_target_2026_v6"] = frame["hot_target_groups_v6"].map(clean).ne("")

    selected, selection_audit = greedy_select(frame, args.top_n)
    controls = frame.loc[frame["literature_evidence_tier_v6"].str.startswith("L1_")].copy()
    controls = controls.sort_values(["v5_pair_physics_score", "v5_rank"], ascending=[False, True])
    selected_ids = set(selected["pair_id"])
    frame["in_final500_v6"] = frame["pair_id"].isin(selected_ids)
    rank_map = selected.set_index("pair_id")["top500_rank_v6"].to_dict()
    frame["top500_rank_v6"] = frame["pair_id"].map(rank_map).fillna("")
    frame = frame.sort_values(["in_final500_v6", "top500_rank_v6", "v5_rank"], ascending=[False, True, True])

    all_path = out_dir / "FINAL1000_UNIFORMLY_REVIEWED_V6.csv"
    top_path = out_dir / "FINAL500_PHYSICS_FIRST_REVIEWED_V6.csv"
    readable_path = out_dir / "FINAL500_TEACHER_READABLE_ZH_V6.csv"
    controls_path = out_dir / "VALIDATED_CONTROLS_AND_REDISCOVERY_V6.csv"
    frame.to_csv(all_path, index=False)
    selected.to_csv(top_path, index=False)
    readable = readable_table(selected)
    readable.to_csv(readable_path, index=False)
    controls.to_csv(controls_path, index=False)
    write_excel(out_dir / "FINAL500_REVIEWED_PACKAGE_V6.xlsx", readable, selected, frame, controls)

    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_rows": int(len(frame)),
        "input_unique_drugs": int(frame["drug_chembl_id"].nunique()),
        "input_unique_targets": int(frame["primary_gene"].nunique()),
        "review_depth_counts": json_counts(frame["review_depth_v6"]),
        "chembl_status_counts": json_counts(frame["chembl_exact_activity_status"]),
        "pubmed_screen_tier_counts": json_counts(frame["pubmed_screen_tier_v6"]),
        "literature_evidence_counts": json_counts(frame["literature_evidence_tier_v6"]),
        "validated_controls_outside_discovery_top500": int(len(controls)),
        "selection": selection_audit,
        "top500": {
            "rows": int(len(selected)),
            "unique_drugs": int(selected["drug_chembl_id"].nunique()),
            "unique_targets": int(selected["primary_gene"].nunique()),
            "unique_scaffolds": int(selected["murcko_scaffold"].nunique()),
            "assay_family_counts": json_counts(selected["target_assay_family"]),
            "strength_tier_counts": json_counts(selected["v5_strength_tier"]),
            "feasibility_grade_counts": json_counts(selected["feasibility_grade_v6"]),
            "experimental_execution_tier_counts": json_counts(selected["experimental_execution_tier_v6"]),
            "active_species_status_counts": json_counts(selected["active_species_status_v6"]),
            "candidate_role_counts": json_counts(selected["candidate_role_v6"]),
            "repurposing_interpretation_counts": json_counts(selected["repurposing_interpretation_v6"]),
            "candidate_disease_area_counts": json_counts(selected["candidate_disease_area_v6"]),
            "hot_target_rows": int(selected["is_hot_target_2026_v6"].sum()),
            "hot_target_unique_genes": int(selected.loc[selected["is_hot_target_2026_v6"], "primary_gene"].nunique()),
            "physics_score_min": float(numeric(selected["v5_pair_physics_score"]).min()),
            "physics_score_median": float(numeric(selected["v5_pair_physics_score"]).median()),
            "collapsed_conplex_rank_max": int(numeric(selected["rank_within_drug"]).max()),
            "full891_conplex_rank_gt100": int((numeric(selected["rank_within_drug_full891"]) > 100).sum()),
            "manual_deep_review_rows": int(
                selected["review_depth_v6"].eq(
                    "agent_deep_review_plus_uniform_database_audit"
                ).sum()
            ),
            "systematic_review_rows": int(selected["review_depth_v6"].eq("systematic_database_and_rule_review").sum()),
        },
        "score_formula": "0.75*v5_pair_physics_score + 0.15*v5_experimental_readiness_score + 0.07*review_grade + 0.03*novelty",
        "selection_order": "v5_strength_tier A before B before C; composite score orders candidates only within the same physics tier",
        "disease_policy": "Disease evidence does not enter the affinity score; it is annotation and a downstream mechanism hypothesis only.",
        "literature_policy": "PubMed co-occurrence and abstract keywords are triage signals; only prior manual review or strict ChEMBL binding is called validated.",
        "integrity_checks": {
            "top500_rows_exact": bool(len(selected) == args.top_n and selected["pair_id"].nunique() == args.top_n),
            "no_known_fda_target_pair": bool(not selected["is_known_fda_target_pair"].map(truthy).any()),
            "no_family_or_rediscovery_risk": bool(not selected["family_or_rediscovery_risk_v2"].map(truthy).any()),
            "no_severe_compound_liability": bool(not selected["severe_compound_liability"].map(truthy).any()),
            "no_structure_sequence_mismatch": bool(not selected["structure_sequence_mismatch_v4"].map(truthy).any()),
            "all_pose_stability_A_or_B": bool(selected["pose_stability_tier"].str.startswith(("A_", "B_")).all()),
            "all_above_known_positive_q10": bool(selected["v5_passes_known_positive_q10_floor"].map(truthy).all()),
            "all_collapsed_conplex_rank_le_100": bool((numeric(selected["rank_within_drug"]) <= 100).all()),
            "no_validated_L1_pair_in_discovery_top500": bool(not selected["literature_evidence_tier_v6"].str.startswith("L1_").any()),
            "all_active_species_resolved": bool(
                selected["active_species_status_v6"].isin(
                    ["parent_drug_relevant", "salt_normalization_adequate"]
                ).all()
            ),
            "no_unresolved_database_identity": bool(
                not selected["agent_database_query_resolution"].map(clean).eq("unresolved").any()
            ),
            "all_required_review_fields_nonempty": bool(
                selected[["candidate_disease_v6", "assay_plan_v6", "review_verdict_v6", "key_risks_v6"]]
                .astype(str)
                .ne("")
                .all()
                .all()
            ),
        },
    }
    (out_dir / "FINAL500_SELECTION_AUDIT_V6.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# FDA老药新靶点：全1000统一研判与Top500结果（V6）",
        "",
        f"- 全量研判：{summary['input_rows']} 条，{summary['input_unique_drugs']} 个FDA药物，{summary['input_unique_targets']} 个靶点。",
        f"- 逐条深审：{summary['review_depth_counts'].get('agent_deep_review_plus_uniform_database_audit', 0)} 条；仅统一数据库/规则研判：{summary['review_depth_counts'].get('systematic_database_and_rule_review', 0)} 条。",
        f"- Top500：{summary['top500']['unique_drugs']} 个药物、{summary['top500']['unique_targets']} 个靶点、{summary['top500']['unique_scaffolds']} 个骨架。",
        f"- 精确验证/再发现对照另列：{summary['validated_controls_outside_discovery_top500']} 条，不占发现Top500。",
        "",
        "## 选择口径",
        "",
        "Top500首先满足V5硬门槛：非已知FDA靶点、非同家族泄露、无严重化合物责任、Boltz完整、双构象A/B、已知阳性家族q10以上、ConPLEx药内折叠Top100。",
        "活性实体必须已确认是母体或可充分归一的盐型；前药/活性代谢物重跑、实体不确定和数据库身份未解决项不进入正式Top500。",
        "先按已知阳性家族校准物理档排序（A优先于B，B优先于C），再在同档内使用复合分。",
        f"同档排序公式：`{summary['score_formula']}`。疾病证据不进入亲和主分。",
        "采用药物、靶点、Murcko骨架和assay family上限，防止少数药物或易打分蛋白垄断。",
        "",
        "## 文献与数据库结果",
        "",
        f"- ChEMBL状态：{json.dumps(summary['chembl_status_counts'], ensure_ascii=False)}",
        f"- PubMed摘要筛查：{json.dumps(summary['pubmed_screen_tier_counts'], ensure_ascii=False)}",
        "- PubMed药名-基因共现和摘要关键词不等于直接结合；对应PMID已保留供原文复核。",
        "",
        "## 解释边界",
        "",
        "Top500是直接结合/target-engagement计算假说优先队列，不是已证实亲和力、作用方向或疾病疗效。",
        "候选细分病种来自既有深审或Open Targets靶点-疾病证据，仅提供后续机制分流。",
        "未完成定量人体游离暴露桥接的候选必须先做浓度梯度，再按临床可达暴露设置go/no-go。",
    ]
    (out_dir / "FINAL500_REVIEW_SUMMARY_ZH.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

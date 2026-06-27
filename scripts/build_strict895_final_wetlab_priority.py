#!/usr/bin/env python3
"""Merge all strict895 evidence and produce first-round wetlab priority order."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/broad_mechanism_layer_v2"
FIRST_PASS = BASE / "strict895_agent_review/strict_top_ready_895_first_pass_analysis.csv"
SOURCE = BASE / "strict895_agent_review/strict_top_ready_895_source_rows.csv"
AGENT_DIR = BASE / "strict895_agent_review/agent_reviews"
CHILD_OT = BASE / "strict895_opentargets_child_disease/strict895_opentargets_child_disease_summary.csv"
OUTDIR = BASE / "strict895_final_wetlab_priority"
STRICT895_LITERATURE_AUDIT = BASE / "strict895_pubmed_literature_audit/strict895_pair_pubmed_literature_audit.csv"


SOURCE_CONTEXT_COLS = [
    "review_id",
    "primary_label_direction",
    "therapeutic_area",
    "known_anchor_genes",
    "known_family_set_for_drug",
    "known_pharmacology_summary",
    "approval_year_min",
    "route",
]


DECISION_SCORE = {
    "keep_high": 40.0,
    "keep_medium": 25.0,
    "review_low": 8.0,
    "control_only": 12.0,
    "deprioritize": -25.0,
}

ASSAY_SCORE = {
    "kinase_biochemical_cellular": 12.0,
    "enzyme_or_epigenetic_biochemical": 11.0,
    "transporter_uptake_efflux": 9.0,
    "ion_channel_functional": 7.0,
    "nuclear_receptor_or_tf_reporter": 6.0,
}

RISK_PENALTY = {
    "low": 0.0,
    "medium": -3.0,
    "中": -3.0,
    "中高": -6.0,
    "high": -9.0,
    "高": -9.0,
    "control": -2.0,
    "对照": -2.0,
}

ORIGINAL_AREA_ZH = {
    "oncology": "肿瘤",
    "Oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "Infectious Disease": "感染性疾病",
    "neurology_psychiatry": "神经/精神",
    "Neurology/Psychiatry": "神经/精神",
    "endocrinology_metabolic": "内分泌/代谢",
    "Endocrinology/Metabolic": "内分泌/代谢",
    "gastroenterology": "胃肠/消化",
    "Gastroenterology": "胃肠/消化",
    "cardiovascular": "心血管",
    "Cardiovascular": "心血管",
    "immunology_inflammation": "免疫/炎症",
    "Immunology/Inflammation": "免疫/炎症",
    "respiratory": "呼吸",
    "Respiratory": "呼吸",
    "other": "其他",
    "Other": "其他",
}

TRACK_ZH = {
    "first_wave_priority": "第一轮优先",
    "first_wave_or_backup": "第一轮候补",
    "control_or_rediscovery": "阳性/再发现对照",
    "secondary_review": "二级复核",
    "low_priority_review": "低优先复核",
    "deprioritized": "暂缓",
}

ASSAY_LANE_ZH = {
    "kinase_biochemical_cellular": "激酶生化/细胞",
    "enzyme_or_epigenetic_biochemical": "酶/表观遗传生化",
    "transporter_uptake_efflux": "转运体摄取/外排",
    "ion_channel_functional": "离子通道功能",
    "nuclear_receptor_or_tf_reporter": "核受体/转录因子 reporter",
}

LITERATURE_STATUS_ZH = {
    "not_audited_in_local_literature_table": "未进入上一轮12,696对文献审计范围",
    "unreported_in_local_pubmed_audit": "本地 PubMed 共现未见明确报道",
    "unreported_in_pubmed_pair_audit": "PubMed pair审计未见明确报道",
    "reported_post_approval_old_drug_new_target": "上市后文献有新靶点线索",
    "known_pharmacology_or_control_not_new": "已知药理/对照，不算新发现",
    "reported_only_before_or_without_approval_window": "上市前或窗口外报道",
    "pubmed_query_failed_needs_retry": "PubMed查询失败，需重跑",
}

DISEASE_NAME_ZH = {
    "breast carcinoma": "乳腺癌",
    "breast cancer": "乳腺癌",
    "acute myeloid leukemia": "急性髓系白血病",
    "psoriasis": "银屑病",
    "melanoma": "黑色素瘤",
    "cardiac arrhythmia": "心律失常",
    "non-small cell lung carcinoma": "非小细胞肺癌",
    "hepatocellular carcinoma": "肝细胞癌",
    "squamous cell lung carcinoma": "肺鳞癌",
    "gastrointestinal stromal tumor": "胃肠道间质瘤",
    "chronic kidney disease": "慢性肾病",
    "plasma cell myeloma": "浆细胞骨髓瘤",
    "prostate cancer": "前列腺癌",
    "prostate carcinoma": "前列腺癌",
    "epilepsy": "癫痫",
    "schizophrenia": "精神分裂症",
    "oral cavity squamous cell carcinoma": "口腔鳞癌",
    "atherosclerosis": "动脉粥样硬化",
    "pulmonary arterial hypertension": "肺动脉高压",
    "myotonia fluctuans": "波动性肌强直",
    "rheumatoid arthritis": "类风湿关节炎",
    "bipolar disorder": "双相障碍",
    "diffuse large B-cell lymphoma": "弥漫大B细胞淋巴瘤",
    "dilated cardiomyopathy": "扩张型心肌病",
    "small cell lung carcinoma": "小细胞肺癌",
    "metabolic dysfunction-associated steatohepatitis": "代谢功能障碍相关脂肪性肝炎",
    "B-cell chronic lymphocytic leukemia": "B细胞慢性淋巴细胞白血病",
    "esophageal adenocarcinoma": "食管腺癌",
    "esophageal squamous cell carcinoma": "食管鳞癌",
    "head and neck squamous cell carcinoma": "头颈鳞癌",
    "myeloproliferative neoplasm": "骨髓增殖性肿瘤",
    "colorectal cancer": "结直肠癌",
    "myocardial infarction": "心肌梗死",
    "Parkinson disease": "帕金森病",
    "endometrial cancer": "子宫内膜癌",
    "hypertensive disorder": "高血压性疾病",
    "osteoporosis": "骨质疏松",
    "peripartum cardiomyopathy": "围产期心肌病",
    "gastric cancer": "胃癌",
    "spinal cord injury": "脊髓损伤",
    "systemic lupus erythematosus": "系统性红斑狼疮",
    "renal cell carcinoma": "肾细胞癌",
    "gout": "痛风",
    "dermatitis": "皮炎",
    "postpartum depression": "产后抑郁",
    "diabetic kidney disease": "糖尿病肾病",
    "type 1 diabetes mellitus": "1型糖尿病",
    "lung cancer": "肺癌",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def normalize_drug_id(value: Any) -> str:
    return re.sub(r"__.*$", "", clean(value))


def norm_gene(value: Any) -> str:
    text = clean(value).upper()
    for sep in [";", ",", "|"]:
        if sep in text:
            text = next((part.strip() for part in text.split(sep) if part.strip()), "")
    return text


def numeric(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def yes_no(value: Any) -> str:
    text = clean(value).lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return "是"
    if text in {"false", "0", "no", "n", "否", ""}:
        return "否"
    return "是" if bool(value) else "否"


def norm_decision(value: str) -> str:
    text = clean(value).lower()
    aliases = {
        "high": "keep_high",
        "medium": "keep_medium",
        "low": "review_low",
        "control": "control_only",
        "control_only": "control_only",
        "deprioritize": "deprioritize",
        "keep_high": "keep_high",
        "keep_medium": "keep_medium",
        "review_low": "review_low",
    }
    return aliases.get(text, text or "unreviewed")


def conplex_component(score: float) -> float:
    # 0.20 is the floor; >=0.50 is saturated strong support.
    if score < 0.20:
        return 0.0
    return max(0.0, min(14.0, (score - 0.20) / 0.30 * 14.0))


def child_ot_component(score: float) -> float:
    return max(0.0, min(18.0, score / 0.80 * 18.0))


def novelty_component(row: pd.Series) -> float:
    decision = clean(row.get("agent_decision_norm"))
    lit = clean(row.get("literature_status")).lower()
    same_family = bool(row.get("same_target_family_as_known"))
    same_label = bool(row.get("same_label_direction"))
    exact_known = bool(row.get("exact_known_target_by_profile_or_control"))
    if decision == "control_only" or exact_known or "known_pharmacology" in lit:
        return -8.0
    score = 8.0
    if same_family:
        score -= 4.0
    if same_label:
        score -= 3.0
    if "reported" in lit and "unreported" not in lit:
        score -= 4.0
    if "unreported" in lit or "no_pubmed" in lit:
        score += 2.0
    return score


def risk_extra_penalty(row: pd.Series) -> float:
    gene = clean(row.get("target_gene")).upper()
    note = clean(row.get("agent_notes_zh")) + "；" + clean(row.get("major_risks_zh"))
    penalty = 0.0
    if gene in {"TP53", "CCND1", "KRAS", "NFE2L2"}:
        penalty -= 5.0
    if gene in {"KCNH2", "SCN5A"} and "安全" in note:
        penalty -= 4.0
    if "不建议进入主筛选" in note:
        penalty -= 10.0
    if "复方" in note or "combo" in note.lower():
        penalty -= 3.0
    return penalty


def disease_area_from_text(text: str) -> str:
    text = clean(text).lower()
    if any(k in text for k in ["leukemia", "lymphoma", "myeloma", "hematolog", "白血", "淋巴", "骨髓瘤"]):
        return "血液肿瘤/血液病"
    if any(
        k in text
        for k in [
            "cancer",
            "carcinoma",
            "tumor",
            "melanoma",
            "sarcoma",
            "glioma",
            "neoplasm",
            "gastrointestinal stromal",
            "adenocarcinoma",
            "肿瘤",
            "癌",
            "黑色素瘤",
            "肉瘤",
        ]
    ):
        return "实体瘤"
    if any(k in text for k in ["arrhythmia", "cardiac", "heart", "pulmonary arterial hypertension", "atherosclerosis", "cardiovascular", "心", "血管", "高血压"]):
        return "心血管"
    if any(k in text for k in ["kidney", "renal", "neph", "肾"]):
        return "肾脏"
    if any(k in text for k in ["psoriasis", "dermat", "skin", "皮肤", "银屑"]):
        return "皮肤/免疫"
    if any(k in text for k in ["rheumatoid", "lupus", "crohn", "colitis", "inflammatory bowel", "immune", "immun", "炎症", "免疫", "类风湿", "红斑狼疮"]):
        return "免疫/炎症"
    if any(k in text for k in ["schizophrenia", "epilepsy", "parkinson", "alzheimer", "migraine", "nervous", "psychiatry", "神经", "精神", "癫痫"]):
        return "神经/精神"
    if any(k in text for k in ["liver", "hepatic", "肝"]):
        return "肝脏"
    if any(k in text for k in ["diabetes", "obesity", "metabolic", "代谢", "糖尿病"]):
        return "代谢"
    if any(k in text for k in ["respiratory", "asthma", "lung", "肺"]):
        return "呼吸"
    if any(k in text for k in ["musculoskeletal", "bone", "骨", "肌"]):
        return "骨/肌肉"
    return ""


def final_disease_area_zh(row: pd.Series) -> str:
    primary = clean(row.get("best_child_ot_disease_name"))
    primary_area = disease_area_from_text(primary)
    if primary_area:
        return primary_area
    fallback_text = " ".join(
        [
            clean(row.get("final_candidate_diseases_zh")),
            clean(row.get("candidate_diseases_zh")),
            clean(row.get("direction")),
        ]
    )
    fallback_area = disease_area_from_text(fallback_text)
    if fallback_area:
        return fallback_area
    return clean(row.get("direction_zh")) or clean(row.get("direction")) or "未分类"


def original_area_zh(row: pd.Series) -> str:
    for col in ["primary_label_direction", "therapeutic_area"]:
        value = clean(row.get(col))
        if value:
            return ORIGINAL_AREA_ZH.get(value, value)
    return "未知"


def disease_name_zh(value: Any) -> str:
    text = clean(value)
    if not text:
        return ""
    return DISEASE_NAME_ZH.get(text, text)


def repurposing_type_zh(row: pd.Series) -> str:
    decision = clean(row.get("agent_decision_norm"))
    lit = clean(row.get("literature_status")).lower()
    exact_known = bool(row.get("exact_known_target_by_profile_or_control"))
    same_label = bool(row.get("same_label_direction"))
    same_family = bool(row.get("same_target_family_as_known"))
    if exact_known or "known_pharmacology" in lit:
        return "已知药理/阳性对照"
    if decision == "control_only":
        if "reported_post_approval_old_drug_new_target" in lit:
            return "文献再发现/阳性对照"
        return "再发现/流程对照"
    if not same_label and not same_family:
        return "跨大领域老药新用"
    if not same_label and same_family:
        return "跨领域但同靶点家族扩展"
    if same_label and not same_family:
        return "同领域新靶点/新病种探索"
    if same_label and same_family:
        return "同领域同靶点家族扩展"
    return "证据不足待复核"


def refresh_literature_texts(row: pd.Series) -> pd.Series:
    status = clean(row.get("literature_status"))
    status_zh = clean(row.get("literature_status_zh")) or status
    replacement = f"{status_zh}，详见PubMed计数/PMID列"
    for col in [
        "novelty_interpretation_zh",
        "agent_notes_zh",
        "final_mechanism_assessment_zh",
        "major_risks_zh",
    ]:
        text = clean(row.get(col))
        if not text:
            continue
        text = text.replace("文献状态未充分审计，需人工排重", replacement)
        text = text.replace("文献未充分审计，需人工排重", replacement)
        text = text.replace("文献未充分审计", status_zh)
        text = text.replace("literature_status=not_audited_in_local_literature_table", f"literature_status={status}")
        row[col] = text
    return row


def is_ion_channel_gene(gene: Any) -> bool:
    return bool(re.match(r"^(KCN|SCN|CACN|HCN)", clean(gene).upper()))


def refresh_assay_annotations(row: pd.Series) -> pd.Series:
    if not is_ion_channel_gene(row.get("target_gene")):
        return row
    row["assay_lane"] = "ion_channel_functional"
    row["primary_assay_zh"] = "优先手动/自动膜片钳、离子通量或电流-电压曲线；必要时加入过表达细胞和空载体对照"
    row["assay_recommendation_zh"] = (
        f"{clean(row.get('target_gene'))} 手动/自动膜片钳剂量反应；必要时用电流-电压曲线、"
        "洗脱可逆性和离子通量 readout 确认直接通道调节。"
    )
    row["counterscreen_zh"] = "hERG/KCNH2、SCN5A、同亚型通道panel、膜扰动、细胞毒性和非特异电生理伪影反筛"
    row["counterscreen_recommendation_zh"] = "hERG/KCNH2、SCN5A、同亚型通道panel、膜扰动、细胞毒性和非特异电生理伪影反筛。"
    return row


def recommended_track(row: pd.Series) -> str:
    decision = clean(row.get("agent_decision_norm"))
    score = numeric(row.get("final_wetlab_priority_score"))
    if decision == "control_only":
        return "control_or_rediscovery"
    if decision == "deprioritize" or score < 20:
        return "deprioritized"
    if decision == "keep_high" and score >= 60:
        return "first_wave_priority"
    if decision in {"keep_high", "keep_medium"} and score >= 45:
        return "first_wave_or_backup"
    if decision in {"keep_medium", "review_low"}:
        return "secondary_review"
    return "low_priority_review"


def dedupe_key(row: pd.Series) -> str:
    drug = clean(row.get("drug_name_agent") or row.get("drug_name"))
    # Reduce obvious combo suffixes without being too aggressive.
    drug = re.sub(r"\s+and\s+.*$", "", drug, flags=re.IGNORECASE)
    return f"{drug.lower()}|{clean(row.get('target_gene')).upper()}|{clean(row.get('best_child_ot_disease_name')).lower()}"


def read_agent_reviews() -> pd.DataFrame:
    frames = []
    for path in sorted(AGENT_DIR.glob("strict895_chunk_*_agent_review.csv")):
        df = pd.read_csv(path, low_memory=False)
        df["agent_review_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    reviews = pd.concat(frames, ignore_index=True)
    reviews["agent_decision_norm"] = reviews["agent_decision"].map(norm_decision)
    return reviews.drop_duplicates("review_id", keep="last")


def read_source_context() -> pd.DataFrame:
    available = pd.read_csv(SOURCE, nrows=0).columns
    usecols = [c for c in SOURCE_CONTEXT_COLS if c in available]
    if not usecols:
        return pd.DataFrame({"review_id": []})
    context = pd.read_csv(SOURCE, usecols=usecols, low_memory=False)
    return context.drop_duplicates("review_id", keep="first")


def read_strict895_literature_audit() -> pd.DataFrame:
    if not STRICT895_LITERATURE_AUDIT.exists():
        return pd.DataFrame(columns=["drug_lit_key", "target_lit_key"])
    lit = pd.read_csv(STRICT895_LITERATURE_AUDIT, low_memory=False)
    lit["drug_lit_key"] = lit["drug_chembl_id"].map(normalize_drug_id)
    lit["target_lit_key"] = lit["candidate_anchor_gene"].map(norm_gene)
    keep = [
        "drug_lit_key",
        "target_lit_key",
        "strict895_literature_status",
        "lit_ok",
        "pair_pubmed_count_2000_2026",
        "pair_pubmed_pmids_2000_2026",
        "pair_pubmed_url_2000_2026",
        "post_approval_pair_pubmed_count",
        "post_approval_pair_pubmed_pmids",
        "post_approval_pair_pubmed_url",
        "literature_class",
        "in_previous_12696_literature_audit",
    ]
    lit = lit[[c for c in keep if c in lit.columns]].copy()
    rename = {
        "lit_ok": "strict895_pubmed_lit_ok",
        "pair_pubmed_count_2000_2026": "strict895_pair_pubmed_count_2000_2026",
        "pair_pubmed_pmids_2000_2026": "strict895_pair_pubmed_pmids_2000_2026",
        "pair_pubmed_url_2000_2026": "strict895_pair_pubmed_url_2000_2026",
        "post_approval_pair_pubmed_count": "strict895_post_approval_pair_pubmed_count",
        "post_approval_pair_pubmed_pmids": "strict895_post_approval_pair_pubmed_pmids",
        "post_approval_pair_pubmed_url": "strict895_post_approval_pair_pubmed_url",
        "literature_class": "strict895_literature_class",
    }
    lit = lit.rename(columns=rename)
    return lit.drop_duplicates(["drug_lit_key", "target_lit_key"], keep="first")


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "final_rank_unique_hypothesis",
        "review_id",
        "recommended_track_zh",
        "repurposing_type_zh",
        "drug_name",
        "drug_chembl_id",
        "original_area_zh",
        "therapeutic_area",
        "target_gene",
        "target_name",
        "best_child_ot_disease_name_zh",
        "best_child_ot_disease_name",
        "final_disease_area_zh",
        "final_wetlab_priority_score",
        "conplex_score",
        "rank_within_drug",
        "best_child_ot_score",
        "assay_lane_zh",
        "agent_decision_norm",
        "risk_level",
        "same_label_direction_zh",
        "same_target_family_as_known_zh",
        "exact_known_target_by_profile_or_control_zh",
        "literature_status_zh",
        "strict895_pair_pubmed_count_2000_2026",
        "strict895_post_approval_pair_pubmed_count",
        "strict895_pair_pubmed_pmids_2000_2026",
        "strict895_post_approval_pair_pubmed_pmids",
        "final_candidate_diseases_zh",
        "final_mechanism_assessment_zh",
        "final_feasibility_assessment_zh",
        "assay_recommendation_zh",
        "counterscreen_recommendation_zh",
        "agent_notes_zh",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols].copy()


def teacher_table_zh(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "final_rank_unique_hypothesis": "排名",
        "review_id": "审计ID",
        "recommended_track_zh": "推荐层级",
        "repurposing_type_zh": "老药新用类型",
        "drug_name": "药物",
        "drug_chembl_id": "药物ChEMBL_ID",
        "original_area_zh": "原适应症领域",
        "therapeutic_area": "原领域英文",
        "target_gene": "候选靶点",
        "target_name": "靶点名称",
        "best_child_ot_disease_name_zh": "拟新用细分病种",
        "best_child_ot_disease_name": "拟新用病种英文",
        "final_disease_area_zh": "拟新用病种大类",
        "final_wetlab_priority_score": "优先级分数",
        "conplex_score": "ConPLEx分数",
        "rank_within_drug": "药物内靶点排名",
        "best_child_ot_score": "OpenTargets病种分数",
        "assay_lane_zh": "实验类型",
        "agent_decision_norm": "agent决策",
        "risk_level": "风险等级",
        "same_label_direction_zh": "是否同原适应症领域",
        "same_target_family_as_known_zh": "是否同已知靶点家族",
        "exact_known_target_by_profile_or_control_zh": "是否已知靶点/对照",
        "literature_status_zh": "文献状态",
        "strict895_pair_pubmed_count_2000_2026": "PubMed总窗计数",
        "strict895_post_approval_pair_pubmed_count": "上市后PubMed计数",
        "strict895_pair_pubmed_pmids_2000_2026": "PubMed总窗PMID",
        "strict895_post_approval_pair_pubmed_pmids": "上市后PMID",
        "final_candidate_diseases_zh": "候选病种说明",
        "final_mechanism_assessment_zh": "机制审计",
        "final_feasibility_assessment_zh": "可行性审计",
        "assay_recommendation_zh": "首选实验",
        "counterscreen_recommendation_zh": "反筛建议",
        "agent_notes_zh": "审计备注",
    }
    clean = clean_table(df)
    return clean.rename(columns=rename)


def summary_table_zh(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "final_rank_unique_hypothesis",
        "recommended_track_zh",
        "repurposing_type_zh",
        "drug_name",
        "original_area_zh",
        "therapeutic_area",
        "target_gene",
        "best_child_ot_disease_name_zh",
        "best_child_ot_disease_name",
        "final_disease_area_zh",
        "final_wetlab_priority_score",
        "conplex_score",
        "rank_within_drug",
        "best_child_ot_score",
        "assay_lane_zh",
        "risk_level",
        "literature_status_zh",
        "strict895_pair_pubmed_count_2000_2026",
        "strict895_post_approval_pair_pubmed_count",
        "same_label_direction_zh",
        "same_target_family_as_known_zh",
    ]
    rename = {
        "final_rank_unique_hypothesis": "排名",
        "recommended_track_zh": "推荐层级",
        "repurposing_type_zh": "老药新用类型",
        "drug_name": "药物",
        "original_area_zh": "原适应症领域",
        "therapeutic_area": "原领域英文",
        "target_gene": "候选靶点",
        "best_child_ot_disease_name_zh": "拟新用细分病种",
        "best_child_ot_disease_name": "拟新用病种英文",
        "final_disease_area_zh": "拟新用病种大类",
        "final_wetlab_priority_score": "优先级分数",
        "conplex_score": "ConPLEx分数",
        "rank_within_drug": "药物内靶点排名",
        "best_child_ot_score": "OpenTargets病种分数",
        "assay_lane_zh": "实验类型",
        "risk_level": "风险等级",
        "literature_status_zh": "文献状态",
        "strict895_pair_pubmed_count_2000_2026": "PubMed总窗计数",
        "strict895_post_approval_pair_pubmed_count": "上市后PubMed计数",
        "same_label_direction_zh": "是否同原适应症领域",
        "same_target_family_as_known_zh": "是否同已知靶点家族",
    }
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols].rename(columns=rename).copy()


def all_rows_summary_table_zh(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "final_rank_all_rows",
        "review_id",
        "recommended_track_zh",
        "repurposing_type_zh",
        "drug_name",
        "original_area_zh",
        "therapeutic_area",
        "target_gene",
        "best_child_ot_disease_name_zh",
        "best_child_ot_disease_name",
        "final_disease_area_zh",
        "final_wetlab_priority_score",
        "conplex_score",
        "rank_within_drug",
        "best_child_ot_score",
        "assay_lane_zh",
        "risk_level",
        "literature_status_zh",
        "strict895_pair_pubmed_count_2000_2026",
        "strict895_post_approval_pair_pubmed_count",
        "strict895_pair_pubmed_pmids_2000_2026",
        "strict895_post_approval_pair_pubmed_pmids",
        "same_label_direction_zh",
        "same_target_family_as_known_zh",
    ]
    rename = {
        "final_rank_all_rows": "排名",
        "review_id": "审计ID",
        "recommended_track_zh": "推荐层级",
        "repurposing_type_zh": "老药新用类型",
        "drug_name": "药物",
        "original_area_zh": "原适应症领域",
        "therapeutic_area": "原领域英文",
        "target_gene": "候选靶点",
        "best_child_ot_disease_name_zh": "拟新用细分病种",
        "best_child_ot_disease_name": "拟新用病种英文",
        "final_disease_area_zh": "拟新用病种大类",
        "final_wetlab_priority_score": "优先级分数",
        "conplex_score": "ConPLEx分数",
        "rank_within_drug": "药物内靶点排名",
        "best_child_ot_score": "OpenTargets病种分数",
        "assay_lane_zh": "实验类型",
        "risk_level": "风险等级",
        "literature_status_zh": "文献状态",
        "strict895_pair_pubmed_count_2000_2026": "PubMed总窗计数",
        "strict895_post_approval_pair_pubmed_count": "上市后PubMed计数",
        "strict895_pair_pubmed_pmids_2000_2026": "PubMed总窗PMID",
        "strict895_post_approval_pair_pubmed_pmids": "上市后PMID",
        "same_label_direction_zh": "是否同原适应症领域",
        "same_target_family_as_known_zh": "是否同已知靶点家族",
    }
    cols = [c for c in cols if c in df.columns]
    return df.loc[:, cols].rename(columns=rename).copy()


def strict_novelty_subset(df: pd.DataFrame) -> pd.DataFrame:
    excluded_targets = {"SLC22A12", "SCN5A", "KRAS", "SLC29A1", "SLC34A2", "SLC5A2"}
    out = df[
        df["recommended_track"].eq("first_wave_priority")
        & ~df["same_label_direction"].fillna(False).astype(bool)
        & ~df["same_target_family_as_known"].fillna(False).astype(bool)
        & df["literature_status"].eq("unreported_in_pubmed_pair_audit")
        & pd.to_numeric(df["conplex_score"], errors="coerce").ge(0.30)
        & pd.to_numeric(df["rank_within_drug"], errors="coerce").le(50)
        & df["best_child_ot_disease_name"].map(clean).ne("")
        & ~df["target_gene"].map(lambda value: clean(value).upper()).isin(excluded_targets)
    ].copy()
    out = out.sort_values(["final_wetlab_priority_score", "best_child_ot_score", "conplex_score"], ascending=[False, False, False])
    out.insert(0, "strict_novelty_rank", range(1, len(out) + 1))
    out["strict_novelty_selection_note_zh"] = (
        "第一轮优先；跨原适应症领域；非同已知靶点家族；PubMed pair未见报道；"
        "ConPLEx>=0.30；药物内rank<=50；有具体病种；排除KRAS/SCN5A/SLC22A12及ADME重的SLC29A1/SLC34A2/SLC5A2。"
    )
    return out


def strict_novelty_table_zh(df: pd.DataFrame) -> pd.DataFrame:
    table = all_rows_summary_table_zh(df)
    if "strict_novelty_rank" in df.columns:
        table.insert(0, "严格novelty排名", df["strict_novelty_rank"].values)
    if "strict_novelty_selection_note_zh" in df.columns:
        table["严格novelty选择说明"] = df["strict_novelty_selection_note_zh"].values
    return table


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    first = pd.read_csv(FIRST_PASS, low_memory=False)
    source_context = read_source_context()
    child = pd.read_csv(CHILD_OT, low_memory=False)
    reviews = read_agent_reviews()
    literature_audit = read_strict895_literature_audit()

    merged = first.merge(child.drop(columns=["drug_name", "target_gene", "direction"], errors="ignore"), on="review_id", how="left")
    merged = merged.merge(source_context, on="review_id", how="left")
    merged = merged.merge(reviews, on="review_id", how="left", suffixes=("", "_agent"))
    if not literature_audit.empty:
        merged["drug_lit_key"] = merged["drug_chembl_id"].map(normalize_drug_id)
        merged["target_lit_key"] = merged["target_gene"].map(norm_gene)
        merged = merged.merge(literature_audit, on=["drug_lit_key", "target_lit_key"], how="left")
        merged["literature_status_before_strict895_pubmed"] = merged["literature_status"].replace(
            {"not_audited_in_local_literature_table": "superseded_not_in_previous_12696_audit"}
        )
        merged["literature_status"] = merged["strict895_literature_status"].where(
            merged["strict895_literature_status"].map(clean).ne(""),
            merged["literature_status"],
        )
        merged["in_previous_12696_literature_audit_zh"] = merged["in_previous_12696_literature_audit"].map(yes_no)
    merged["has_agent_review"] = merged["agent_decision_norm"].notna() & merged["agent_decision_norm"].astype(str).str.len().gt(0)
    merged["agent_decision_norm"] = merged["agent_decision_norm"].fillna("unreviewed")
    merged = merged.apply(refresh_assay_annotations, axis=1)

    merged["decision_component"] = merged["agent_decision_norm"].map(DECISION_SCORE).fillna(0.0)
    merged["conplex_component"] = merged["conplex_score"].map(lambda x: conplex_component(numeric(x)))
    merged["child_ot_component"] = merged["best_child_ot_score"].map(lambda x: child_ot_component(numeric(x)))
    merged["assay_component"] = merged["assay_lane"].map(ASSAY_SCORE).fillna(4.0)
    merged["novelty_component"] = merged.apply(novelty_component, axis=1)
    merged["risk_component"] = merged["risk_level"].map(lambda x: RISK_PENALTY.get(clean(x), -5.0)).fillna(-5.0)
    merged["risk_extra_penalty"] = merged.apply(risk_extra_penalty, axis=1)
    merged["final_disease_area_zh"] = merged.apply(final_disease_area_zh, axis=1)
    merged["repurposing_type_zh"] = merged.apply(repurposing_type_zh, axis=1)
    merged["original_area_zh"] = merged.apply(original_area_zh, axis=1)
    merged["assay_lane_zh"] = merged["assay_lane"].map(ASSAY_LANE_ZH).fillna(merged["assay_lane"])
    merged["literature_status_zh"] = merged["literature_status"].map(LITERATURE_STATUS_ZH).fillna(merged["literature_status"])
    merged = merged.apply(refresh_literature_texts, axis=1)
    merged["best_child_ot_disease_name_zh"] = merged["best_child_ot_disease_name"].map(disease_name_zh)
    merged["same_label_direction_zh"] = merged["same_label_direction"].map(yes_no)
    merged["same_target_family_as_known_zh"] = merged["same_target_family_as_known"].map(yes_no)
    merged["exact_known_target_by_profile_or_control_zh"] = merged["exact_known_target_by_profile_or_control"].map(yes_no)

    merged["final_wetlab_priority_score"] = (
        merged["decision_component"]
        + merged["conplex_component"]
        + merged["child_ot_component"]
        + merged["assay_component"]
        + merged["novelty_component"]
        + merged["risk_component"]
        + merged["risk_extra_penalty"]
    ).clip(lower=0, upper=100)
    merged["recommended_track"] = merged.apply(recommended_track, axis=1)
    merged["recommended_track_zh"] = merged["recommended_track"].map(TRACK_ZH).fillna(merged["recommended_track"])
    merged["dedupe_key"] = merged.apply(dedupe_key, axis=1)
    merged = merged.sort_values(
        ["recommended_track", "final_wetlab_priority_score", "best_child_ot_score", "conplex_score"],
        ascending=[True, False, False, False],
    )
    # Put priority tracks in human order.
    order = {
        "first_wave_priority": 0,
        "first_wave_or_backup": 1,
        "control_or_rediscovery": 2,
        "secondary_review": 3,
        "low_priority_review": 4,
        "deprioritized": 5,
    }
    merged["_track_order"] = merged["recommended_track"].map(order).fillna(9)
    merged = merged.sort_values(["_track_order", "final_wetlab_priority_score"], ascending=[True, False]).drop(columns=["_track_order"])
    merged.insert(0, "final_rank_all_rows", range(1, len(merged) + 1))

    deduped = merged.sort_values("final_wetlab_priority_score", ascending=False).drop_duplicates("dedupe_key", keep="first")
    deduped = deduped.sort_values(["_track_order" if "_track_order" in deduped.columns else "final_wetlab_priority_score"], ascending=[True] if "_track_order" in deduped.columns else [False])
    # Re-sort explicitly after drop because _track_order was removed above.
    deduped["_track_order"] = deduped["recommended_track"].map(order).fillna(9)
    deduped = deduped.sort_values(["_track_order", "final_wetlab_priority_score"], ascending=[True, False]).drop(columns=["_track_order"])
    deduped.insert(0, "final_rank_unique_hypothesis", range(1, len(deduped) + 1))

    key_cols = [
        "final_rank_unique_hypothesis",
        "review_id",
        "drug_name",
        "target_gene",
        "best_child_ot_disease_name_zh",
        "best_child_ot_disease_name",
        "final_disease_area_zh",
        "best_child_ot_score",
        "direction",
        "agent_decision_norm",
        "recommended_track",
        "recommended_track_zh",
        "repurposing_type_zh",
        "final_wetlab_priority_score",
        "conplex_score",
        "rank_within_drug",
        "assay_lane_zh",
        "final_candidate_diseases_zh",
        "final_mechanism_assessment_zh",
        "assay_recommendation_zh",
        "counterscreen_recommendation_zh",
        "risk_level",
        "agent_notes_zh",
        "literature_status_zh",
        "strict895_pair_pubmed_count_2000_2026",
        "strict895_post_approval_pair_pubmed_count",
        "strict895_pair_pubmed_pmids_2000_2026",
        "strict895_post_approval_pair_pubmed_pmids",
    ]
    key_cols = [c for c in key_cols if c in deduped.columns]

    merged.to_csv(OUTDIR / "strict895_final_wetlab_priority_all_rows.csv", index=False)
    all_rows_summary = all_rows_summary_table_zh(merged)
    all_rows_summary.to_csv(OUTDIR / "strict895_all_895_summary_table_zh.csv", index=False)
    all_rows_summary.to_excel(OUTDIR / "strict895_all_895_summary_table_zh.xlsx", index=False)
    strict_novelty = strict_novelty_subset(merged)
    strict_novelty_table = strict_novelty_table_zh(strict_novelty)
    strict_novelty_table.to_csv(OUTDIR / "strict895_first_wave_strict_novelty_summary_table_zh.csv", index=False)
    strict_novelty_table.to_excel(OUTDIR / "strict895_first_wave_strict_novelty_summary_table_zh.xlsx", index=False)
    deduped.to_csv(OUTDIR / "strict895_final_wetlab_priority_unique_hypotheses.csv", index=False)
    deduped[key_cols].head(96).to_csv(OUTDIR / "strict895_first_wave_top96.csv", index=False)
    deduped[key_cols].head(192).to_csv(OUTDIR / "strict895_first_wave_top192.csv", index=False)
    deduped[key_cols].head(384).to_csv(OUTDIR / "strict895_first_wave_top384.csv", index=False)
    clean_deduped = clean_table(deduped)
    clean_deduped.to_csv(OUTDIR / "strict895_final_wetlab_priority_clean.csv", index=False)
    teacher_deduped = teacher_table_zh(deduped)
    teacher_deduped.to_csv(OUTDIR / "strict895_final_wetlab_priority_teacher_readable_zh.csv", index=False)
    summary_table_zh(deduped).to_csv(OUTDIR / "strict895_final_wetlab_priority_summary_table_zh.csv", index=False)
    for n in [96, 192, 384]:
        head = deduped.head(n)
        clean_table(head).to_csv(OUTDIR / f"strict895_first_wave_top{n}_clean.csv", index=False)
        teacher_table_zh(head).to_csv(OUTDIR / f"strict895_first_wave_top{n}_teacher_readable_zh.csv", index=False)
        summary_table_zh(head).to_csv(OUTDIR / f"strict895_first_wave_top{n}_summary_table_zh.csv", index=False)

    summary = {
        "all_rows": int(len(merged)),
        "unique_hypotheses": int(len(deduped)),
        "agent_reviewed_rows": int(merged["has_agent_review"].sum()),
        "missing_agent_review_rows": int((~merged["has_agent_review"]).sum()),
        "agent_decision_counts": merged["agent_decision_norm"].value_counts().to_dict(),
        "recommended_track_counts_all": merged["recommended_track"].value_counts().to_dict(),
        "recommended_track_counts_unique": deduped["recommended_track"].value_counts().to_dict(),
        "child_ot_match_rows": int(merged["child_ot_status"].eq("child_ot_match").sum()),
        "child_ot_score_ge_0_5_rows": int(merged["best_child_ot_score"].ge(0.5).sum()),
        "top96_track_counts": deduped.head(96)["recommended_track"].value_counts().to_dict(),
        "top96_disease_area_counts": deduped.head(96)["final_disease_area_zh"].value_counts().to_dict(),
        "repurposing_type_counts_unique": deduped["repurposing_type_zh"].value_counts().to_dict(),
        "top96_repurposing_type_counts": deduped.head(96)["repurposing_type_zh"].value_counts().to_dict(),
        "top96_direction_counts": deduped.head(96)["direction"].value_counts().to_dict(),
        "top96_assay_lane_counts": deduped.head(96)["assay_lane"].value_counts().to_dict(),
        "literature_status_counts_unique": deduped["literature_status"].value_counts(dropna=False).to_dict(),
        "top96_literature_status_counts": deduped.head(96)["literature_status"].value_counts(dropna=False).to_dict(),
        "strict895_pubmed_audit_rows_matched_all": int(merged["strict895_literature_status"].map(clean).ne("").sum()) if "strict895_literature_status" in merged.columns else 0,
        "first_wave_strict_novelty_rows": int(len(strict_novelty)),
    }
    (OUTDIR / "strict895_final_wetlab_priority_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# strict895 第一轮湿实验推荐顺序",
        "",
        "## 核心结果",
        "",
        f"- 全部 strict895 行：{summary['all_rows']}",
        f"- 去重后 drug-target-disease hypothesis：{summary['unique_hypotheses']}",
        f"- agent 已审计行：{summary['agent_reviewed_rows']}",
        f"- agent 缺失行：{summary['missing_agent_review_rows']}",
        f"- child Open Targets match：{summary['child_ot_match_rows']}",
        f"- child OT score >= 0.5：{summary['child_ot_score_ge_0_5_rows']}",
        "",
        "## 老药新用类型分布（unique hypotheses）",
        "",
    ]
    for k, v in summary["repurposing_type_counts_unique"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## 推荐 track 分布（unique hypotheses）",
        "",
    ]
    for k, v in summary["recommended_track_counts_unique"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Top96 track 分布",
        "",
    ]
    for k, v in summary["top96_track_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Top96 方向分布",
        "",
    ]
    for k, v in summary["top96_disease_area_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## 输出文件",
        "",
        f"- `{OUTDIR / 'strict895_final_wetlab_priority_unique_hypotheses.csv'}`",
        f"- `{OUTDIR / 'strict895_first_wave_top96.csv'}`",
        f"- `{OUTDIR / 'strict895_first_wave_top192.csv'}`",
        f"- `{OUTDIR / 'strict895_first_wave_top384.csv'}`",
    ]
    (OUTDIR / "STRICT895_FINAL_WETLAB_PRIORITY_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

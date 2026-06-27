#!/usr/bin/env python3
"""Build the strict_top_ready ConPLEx>=0.20 row-level review package.

The output is intentionally evidence-grounded and conservative. It prepares a
complete 895-row first-pass analysis for sub-agent chunk review.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "outputs/broad_mechanism_layer_v2/queue_v1_conplex_floor/queue_v1_all_discovery_scored.csv"
OUTDIR = ROOT / "outputs/broad_mechanism_layer_v2/strict895_agent_review"


DIRECTION_DISEASES_ZH = {
    "oncology": "实体瘤/血液肿瘤；优先按靶点落到乳腺癌、肺癌、前列腺癌、白血病/淋巴瘤等具体模型",
    "nervous_system": "神经退行性疾病、癫痫/兴奋性异常、神经炎症或睡眠/认知相关模型",
    "cardiovascular": "高血压、血栓/凝血、心律失常、心衰、动脉粥样硬化或血管重塑",
    "digestive_gi": "IBD/结肠炎、胃肠动力、肠上皮屏障、肝胆胃肠炎症模型",
    "immune": "自身免疫、炎症因子、IBD、银屑病、RA 或免疫细胞功能模型",
    "metabolic": "糖尿病、肥胖、脂代谢异常、胰岛素抵抗或线粒体代谢模型",
    "dermatology": "银屑病、特应性皮炎、痤疮/角化异常、皮肤炎症或屏障模型",
    "hematology": "血液肿瘤、贫血、血小板/凝血、造血分化或炎症性血液病",
    "endocrine": "激素受体相关疾病、甲状腺/肾上腺/性激素轴、代谢内分泌异常",
    "musculoskeletal": "骨质疏松、关节炎、肌病、骨/软骨重塑或疼痛相关肌骨模型",
    "renal": "慢性肾病、肾小球炎症、肾纤维化、离子/转运相关肾功能模型",
    "reproductive": "前列腺/乳腺/子宫内膜相关激素疾病、生殖轴调控或避孕相关模型",
    "urinary": "膀胱疾病、尿路功能、肾-泌尿上皮炎症或转运模型",
    "infectious": "host-directed anti-infection；优先做宿主因子、炎症、病毒复制辅助因子或细胞保护 readout",
    "respiratory": "哮喘、COPD、肺炎症、肺纤维化、气道上皮屏障或肺血管模型",
    "hepatic": "NASH/脂肪肝、胆汁淤积、肝炎症、肝纤维化或肝代谢毒性模型",
    "psychiatry": "抑郁、焦虑、精神分裂症、奖赏/应激通路或神经递质 readout",
    "pain": "神经病理性疼痛、炎症性疼痛、离子通道兴奋性或外周感觉神经模型",
    "ophthalmology": "视网膜退行性疾病、青光眼、眼部炎症、血管新生或上皮屏障模型",
}


TARGET_DISEASE_HINTS = {
    "AR": "前列腺癌、乳腺癌、雄激素相关皮肤/生殖疾病",
    "ESR1": "ER+ 乳腺癌、子宫内膜疾病、骨代谢/绝经相关疾病",
    "ERBB2": "HER2+ 乳腺癌、胃癌、肺癌",
    "EGFR": "NSCLC、头颈鳞癌、结直肠癌、皮肤/上皮增殖疾病",
    "RET": "RET 融合 NSCLC、甲状腺癌、神经内分泌肿瘤",
    "ALK": "ALK 融合 NSCLC、淋巴瘤、神经母细胞瘤",
    "CDK4": "HR+ 乳腺癌、脂肪肉瘤、细胞周期依赖肿瘤",
    "CDK6": "HR+ 乳腺癌、白血病/淋巴瘤、细胞周期依赖肿瘤",
    "BCL2": "CLL、AML、淋巴瘤、抗凋亡依赖肿瘤",
    "PARP1": "BRCA/HRD 卵巢癌、乳腺癌、前列腺癌、DNA repair 缺陷肿瘤",
    "PTGS2": "炎症、疼痛、结直肠癌炎症微环境",
    "JAK1": "RA、IBD、银屑病、JAK/STAT 炎症疾病",
    "JAK2": "骨髓增殖性肿瘤、RA/IBD、JAK/STAT 炎症疾病",
    "BTK": "B 细胞淋巴瘤、CLL、自身免疫 B 细胞疾病",
    "KIT": "GIST、肥大细胞疾病、AML、色素/造血相关疾病",
    "SCN5A": "心律失常、传导异常、钠通道阻滞安全性反筛",
    "SCN4A": "周期性麻痹、肌强直、肌肉兴奋性异常",
    "SLC6A4": "抑郁/焦虑、肠脑轴、血小板 5-HT 转运",
    "SLC29A1": "核苷转运、肿瘤代谢、抗病毒/化疗敏感性",
    "ABCB1": "多药耐药、血脑屏障转运、药物外排",
    "NFE2L2": "氧化应激、NASH、神经保护、肿瘤耐药",
    "KEAP1": "氧化应激/Nrf2、肺癌耐药、炎症和代谢应激",
    "NR3C1": "炎症、自身免疫、代谢/骨骼副作用相关模型",
    "NR3C2": "高血压、心衰、盐皮质激素/肾脏钠水潴留",
    "RXRA": "代谢、皮肤分化、肿瘤分化治疗",
    "PPARG": "糖尿病、脂代谢、NASH、炎症代谢疾病",
}


ASSAY_PLAN = {
    "kinase_biochemical_cellular": (
        "优先 kinase biochemical IC50/Kd 或 ADP-Glo/迁移率法；细胞层面用 phospho-substrate/target engagement 验证",
        "同家族 kinase panel、ATP-competitive counterscreen、细胞毒性/增殖非特异反筛",
        "高"
    ),
    "enzyme_or_epigenetic_biochemical": (
        "优先纯化蛋白酶活/热转移/DSF/CETSA；若是表观遗传酶可加底物修饰 readout",
        "同酶家族选择性、PAINS/聚集体、还原/螯合/荧光干扰反筛",
        "高"
    ),
    "ion_channel_functional": (
        "优先膜片钳或自动膜片钳；次选电位/钙流/通量 readout，并做剂量反应",
        "hERG/SCN5A 安全性、细胞毒性、膜扰动、非特异离子通道 panel",
        "中"
    ),
    "transporter_uptake_efflux": (
        "优先底物摄取/外排、竞争抑制或转运电流实验；结合过表达细胞和空载体对照",
        "同家族转运体、细胞活性、底物荧光/放射性干扰、P-gp/BCRP 外排反筛",
        "中高"
    ),
    "nuclear_receptor_or_tf_reporter": (
        "优先 reporter assay、cofactor recruitment、CETSA/target engagement；必要时加 qPCR 下游靶基因",
        "广谱转录抑制/激活、细胞毒性、荧光素酶干扰、同家族核受体选择性",
        "中"
    ),
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


def num(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def first_name(row: pd.Series) -> str:
    for col in ["generic_name", "drug_names"]:
        text = clean(row.get(col))
        if text:
            return text.split(";")[0].strip()
    return clean(row.get("drug_chembl_id"))


def clip_list(text: str, limit: int = 3) -> str:
    parts = [p.strip() for p in re.split(r";|\|\|", clean(text)) if p.strip()]
    return "；".join(parts[:limit])


def disease_hypothesis(row: pd.Series) -> str:
    direction = clean(row.get("direction"))
    gene = clean(row.get("target_gene_norm")).upper()
    creeds = clip_list(clean(row.get("creeds_matched_disease_names")), 3)
    target_hint = TARGET_DISEASE_HINTS.get(gene, "")
    broad = DIRECTION_DISEASES_ZH.get(direction, "需从 broad direction 继续落到具体疾病模型")
    pieces = []
    if target_hint:
        pieces.append(f"靶点提示：{target_hint}")
    if creeds:
        pieces.append(f"CREEDS 相关签名：{creeds}")
    pieces.append(f"方向候选：{broad}")
    return "；".join(pieces)


def mechanism_hypothesis(row: pd.Series) -> str:
    drug = first_name(row)
    gene = clean(row.get("target_gene_norm")).upper()
    protein = clean(row.get("protein_names") or row.get("representative_protein_name"))
    direction_zh = clean(row.get("direction_label_zh") or row.get("direction"))
    keywords = clip_list(clean(row.get("reactome_mechanism_keywords")), 4)
    pathways = clip_list(clean(row.get("reactome_top_pathways")), 3)
    evidence = (
        f"ConPLEx={num(row.get('conplex_score')):.3f}/rank{int(num(row.get('rank_within_drug')))}, "
        f"OT={num(row.get('open_targets_score')):.2f}, "
        f"TxGNN={num(row.get('txgnn_indication_score')):.2f}, "
        f"STRING={num(row.get('string_context_score')):.2f}, "
        f"tissue={num(row.get('tissue_context_score')):.2f}"
    )
    mech = keywords or pathways or "Reactome/GO 机制注释较泛"
    return f"{drug} 可能作用于 {gene}（{protein}），在{direction_zh}方向的假说为调节 {mech}；证据组合：{evidence}。"


def disease_fit_level(row: pd.Series) -> str:
    ot = num(row.get("open_targets_score"))
    tx = num(row.get("txgnn_indication_score"))
    st = num(row.get("string_context_score"))
    tissue = num(row.get("tissue_context_score"))
    creeds = num(row.get("creeds_total_signature_count"))
    score = 0
    score += ot >= 0.70
    score += tx >= 0.50
    score += st >= 0.25
    score += tissue >= 0.50
    score += creeds > 0
    if score >= 4:
        return "高"
    if score >= 3:
        return "中高"
    if score >= 2:
        return "中"
    return "低-需人工复核"


def feasibility_level(row: pd.Series) -> str:
    assay = clean(row.get("assay_lane"))
    conplex = num(row.get("conplex_score"))
    base = ASSAY_PLAN.get(assay, ("需定制 assay", "需定制反筛", "中"))[2]
    if conplex >= 0.30 and base in {"高", "中高"}:
        return "高"
    if conplex >= 0.30:
        return "中高"
    if conplex >= 0.20 and base == "高":
        return "中高"
    if conplex >= 0.20:
        return "中"
    return "低"


def novelty_status(row: pd.Series) -> str:
    if bool(row.get("is_known_fda_target_pair")) or bool(row.get("exact_known_target_by_profile_or_control")):
        return "已知/阳性对照，不作为新发现"
    if bool(row.get("same_target_family_as_known")):
        return "同靶点家族机制扩展，novelty 中等"
    if bool(row.get("same_label_direction")):
        return "同原适应症方向扩展，novelty 中等偏低"
    lit = clean(row.get("literature_status") or row.get("literature_class"))
    if "reported" in lit and "unreported" not in lit:
        return "已有 PubMed/本地文献线索，适合作为 rediscovery 或排重对象"
    if "unreported" in lit or "no_pubmed" in lit:
        return "本地/在线 PubMed 共现层面较干净，适合 novelty review"
    return "文献状态未充分审计，需人工排重"


def risks(row: pd.Series) -> str:
    gene = clean(row.get("target_gene_norm")).upper()
    assay = clean(row.get("assay_lane"))
    risks_out: list[str] = []
    if gene in {"TP53", "CCND1", "NFE2L2"}:
        risks_out.append("靶点更偏通路/转录调控，直接小分子结合验证难度较高")
    if gene in {"AR", "MET"}:
        risks_out.append("gene symbol/缩写可能产生文献或检索歧义")
    if assay == "nuclear_receptor_or_tf_reporter":
        risks_out.append("reporter 易受广谱转录、细胞毒性和荧光素酶干扰影响")
    if assay == "ion_channel_functional":
        risks_out.append("离子通道实验成本较高，需安全性通道反筛")
    if assay == "transporter_uptake_efflux":
        risks_out.append("转运体 readout 易受底物、外排和细胞状态影响")
    if bool(row.get("same_target_family_as_known")):
        risks_out.append("同靶点家族可能存在机制泄漏/选择性不足")
    if bool(row.get("same_label_direction")):
        risks_out.append("疾病方向接近原适应症，老药新用 novelty 较弱")
    if not risks_out:
        risks_out.append("主要风险是 ConPLEx 预测需正交实验验证，疾病方向仍需具体化")
    return "；".join(risks_out)


def priority_bucket(row: pd.Series) -> str:
    disease_fit = disease_fit_level(row)
    feasibility = feasibility_level(row)
    conplex = num(row.get("conplex_score"))
    novelty = novelty_status(row)
    if "已知/阳性" in novelty:
        return "P0_positive_control"
    if "rediscovery" in novelty:
        return "P1_rediscovery_control"
    if disease_fit in {"高", "中高"} and feasibility in {"高", "中高"} and conplex >= 0.30:
        return "P1_priority_manual_review"
    if disease_fit in {"高", "中高", "中"} and feasibility in {"高", "中高", "中"}:
        return "P2_review_candidate"
    return "P3_low_specificity_or_hard_assay"


def assay_plan(row: pd.Series) -> tuple[str, str, str]:
    return ASSAY_PLAN.get(clean(row.get("assay_lane")), ("需根据靶点定制 target-engagement assay", "同家族/细胞毒性/非特异 readout 反筛", "中"))


def go_no_go(row: pd.Series) -> str:
    primary, counterscreen, _ = assay_plan(row)
    return (
        "Go：剂量依赖 target engagement 或功能 readout 可重复，且细胞活性窗口内反筛不能解释主效应；"
        f"No-go：{counterscreen} 阳性或主 readout 无剂量依赖。"
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INFILE, low_memory=False)
    strict = df[df["strict_top_ready_exact"].eq(True) & pd.to_numeric(df["conplex_score"], errors="coerce").ge(0.20)].copy()
    strict = strict.sort_values(
        ["review_priority_score", "broad_mechanism_score", "conplex_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    strict.insert(0, "review_id", [f"strict895_{i:04d}" for i in range(1, len(strict) + 1)])

    rows = []
    for _, row in strict.iterrows():
        primary, counterscreen, _ = assay_plan(row)
        rows.append(
            {
                "review_id": row["review_id"],
                "drug_chembl_id": clean(row.get("drug_chembl_id")),
                "drug_name": first_name(row),
                "target_gene": clean(row.get("target_gene_norm")).upper(),
                "target_name": clean(row.get("protein_names") or row.get("representative_protein_name")),
                "direction": clean(row.get("direction")),
                "direction_zh": clean(row.get("direction_label_zh")),
                "candidate_diseases_zh": disease_hypothesis(row),
                "mechanism_hypothesis_zh": mechanism_hypothesis(row),
                "disease_fit_level": disease_fit_level(row),
                "target_engagement_feasibility": feasibility_level(row),
                "primary_assay_zh": primary,
                "counterscreen_zh": counterscreen,
                "go_no_go_zh": go_no_go(row),
                "novelty_interpretation_zh": novelty_status(row),
                "major_risks_zh": risks(row),
                "priority_bucket_auto": priority_bucket(row),
                "assay_lane": clean(row.get("assay_lane")),
                "conplex_score": num(row.get("conplex_score")),
                "rank_within_drug": int(num(row.get("rank_within_drug"))),
                "review_priority_score": num(row.get("review_priority_score")),
                "broad_mechanism_score": num(row.get("broad_mechanism_score")),
                "open_targets_score": num(row.get("open_targets_score")),
                "txgnn_indication_score": num(row.get("txgnn_indication_score")),
                "string_context_score": num(row.get("string_context_score")),
                "tissue_context_score": num(row.get("tissue_context_score")),
                "creeds_total_signature_count": int(num(row.get("creeds_total_signature_count"))),
                "reactome_mechanism_keywords": clean(row.get("reactome_mechanism_keywords")),
                "creeds_matched_disease_names": clean(row.get("creeds_matched_disease_names")),
                "literature_status": clean(row.get("literature_status") or row.get("literature_class")),
                "same_target_family_as_known": bool(row.get("same_target_family_as_known")),
                "same_label_direction": bool(row.get("same_label_direction")),
                "exact_known_target_by_profile_or_control": bool(row.get("exact_known_target_by_profile_or_control")),
                "agent_review_status": "first_pass_ready_for_chunk_agent",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "strict_top_ready_895_first_pass_analysis.csv", index=False)
    strict.to_csv(OUTDIR / "strict_top_ready_895_source_rows.csv", index=False)

    chunk_dir = OUTDIR / "agent_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 90
    for i in range(0, len(out), chunk_size):
        chunk_no = i // chunk_size + 1
        out.iloc[i : i + chunk_size].to_csv(chunk_dir / f"strict895_chunk_{chunk_no:02d}.csv", index=False)

    summary = {
        "rows": int(len(out)),
        "unique_drugs": int(out["drug_chembl_id"].nunique()),
        "unique_targets": int(out["target_gene"].nunique()),
        "directions": int(out["direction"].nunique()),
        "chunks": int(math.ceil(len(out) / chunk_size)),
        "definition": "strict_top_ready_exact == True and conplex_score >= 0.20",
        "outputs": {
            "first_pass": str(OUTDIR / "strict_top_ready_895_first_pass_analysis.csv"),
            "source_rows": str(OUTDIR / "strict_top_ready_895_source_rows.csv"),
            "chunks": str(chunk_dir),
        },
    }
    pd.Series(summary).to_json(OUTDIR / "strict895_agent_review_summary.json", force_ascii=False, indent=2)
    print(summary)


if __name__ == "__main__":
    main()

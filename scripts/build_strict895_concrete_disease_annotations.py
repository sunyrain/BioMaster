#!/usr/bin/env python3
"""Add concrete disease hypotheses to the strict895 review set.

This is a local evidence completion pass. It does not claim that each concrete
disease is proven; it converts broad directions into auditable disease-model
candidates using CREEDS disease names, target-specific disease priors, tissue
context, and assay feasibility.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "outputs/broad_mechanism_layer_v2/strict895_agent_review/strict_top_ready_895_first_pass_analysis.csv"
OUTDIR = ROOT / "outputs/broad_mechanism_layer_v2/strict895_concrete_disease_completion"


TARGET_DISEASE_PRIORS: dict[str, list[str]] = {
    "AR": ["前列腺癌", "AR阳性乳腺癌", "雄激素相关皮肤/生殖疾病"],
    "ESR1": ["ER阳性乳腺癌", "子宫内膜癌", "骨质疏松/绝经相关疾病"],
    "PGR": ["子宫内膜癌", "乳腺癌", "子宫肌瘤/内膜异位症"],
    "EGFR": ["EGFR突变NSCLC", "头颈鳞癌", "结直肠癌"],
    "ERBB2": ["HER2阳性乳腺癌", "HER2阳性胃癌", "HER2突变肺癌"],
    "RET": ["RET融合NSCLC", "甲状腺癌", "神经内分泌肿瘤"],
    "ALK": ["ALK融合NSCLC", "间变性大细胞淋巴瘤", "神经母细胞瘤"],
    "BRAF": ["BRAF突变黑色素瘤", "甲状腺癌", "结直肠癌"],
    "MAP2K1": ["MAPK通路依赖肿瘤", "黑色素瘤", "HCC/髓系白血病模型"],
    "KRAS": ["KRAS突变NSCLC", "胰腺癌", "结直肠癌"],
    "PTPN11": ["SHP2依赖RTK/RAS通路肿瘤", "AML/JMML", "NSCLC"],
    "CDK4": ["HR阳性乳腺癌", "脂肪肉瘤", "套细胞淋巴瘤"],
    "CDK6": ["HR阳性乳腺癌", "AML/ALL", "淋巴瘤"],
    "CCND1": ["CCND1扩增乳腺癌", "套细胞淋巴瘤", "结直肠癌"],
    "BCL2": ["CLL", "AML", "BCL2依赖淋巴瘤"],
    "BTK": ["CLL", "套细胞淋巴瘤", "B细胞自身免疫疾病"],
    "LCK": ["T细胞白血病", "T细胞淋巴瘤", "T细胞自身免疫炎症模型"],
    "ITK": ["T细胞白血病", "T细胞淋巴瘤", "TCR信号相关炎症"],
    "JAK1": ["RA/IBD/银屑病", "JAK-STAT炎症疾病", "AML/T细胞白血病"],
    "JAK2": ["骨髓增殖性肿瘤", "JAK-STAT炎症疾病", "贫血/造血异常"],
    "TYK2": ["银屑病", "特应性皮炎", "IBD/自身免疫炎症"],
    "KIT": ["GIST", "系统性肥大细胞增多症", "AML"],
    "PDGFRA": ["PDGFRA突变GIST", "肾纤维化/CKD", "胶质瘤/肉瘤"],
    "PDGFRB": ["肾纤维化/CKD", "血管重塑/心血管纤维化", "PDGFRB依赖白血病/肉瘤"],
    "FLT1": ["肾小球内皮损伤/蛋白尿性CKD", "肿瘤血管生成", "子痫前期/血管病变"],
    "KDR": ["肿瘤血管生成", "HCC/肺癌", "血管重塑疾病"],
    "AXL": ["EMT/转移性肿瘤", "EGFR耐药NSCLC", "纤维化/炎症"],
    "EPHA2": ["鳞癌", "尿路上皮癌", "乳腺癌侵袭转移模型"],
    "IGF1R": ["HCC", "IGF1R-PI3K/MAPK依赖实体瘤", "肉瘤"],
    "TGFBR1": ["胰腺癌/乳腺癌EMT模型", "纤维化", "免疫抑制肿瘤微环境"],
    "PARP1": ["BRCA/HRD卵巢癌", "乳腺癌", "前列腺癌"],
    "DNMT3A": ["AML/克隆性造血", "结直肠癌表观遗传模型", "髓系肿瘤"],
    "EZH2": ["DLBCL", "乳腺癌", "EZH2依赖表观遗传肿瘤"],
    "MEN1": ["menin依赖白血病", "内分泌肿瘤", "乳腺/前列腺内分泌相关肿瘤"],
    "TP53": ["TP53通路异常肿瘤", "DNA损伤/凋亡模型", "需避免直接结合过度解释"],
    "NFE2L2": ["氧化应激/Nrf2相关疾病", "NASH/肝损伤", "肺癌耐药"],
    "KEAP1": ["KEAP1/Nrf2异常肺癌", "氧化应激炎症模型", "代谢应激疾病"],
    "NOX4": ["动脉粥样硬化", "心肌纤维化", "肾纤维化"],
    "SCN5A": ["心律失常", "心脏传导异常", "hERG/心脏安全性反筛"],
    "KCNH2": ["hERG心脏安全性", "长QT风险模型", "不优先作为疗效新适应症"],
    "KCND3": ["心律失常", "心衰复极异常", "神经兴奋性异常"],
    "KCNK3": ["肺动脉高压", "右心室肥厚/心衰", "神经兴奋性模型"],
    "KCNJ2": ["心律失常", "扩张型心肌病", "Kir2.1复极异常"],
    "SCN4A": ["周期性麻痹", "肌强直", "骨骼肌兴奋性异常"],
    "SCN2A": ["癫痫", "神经发育障碍", "神经元兴奋性异常"],
    "GABRA1": ["癫痫", "焦虑/睡眠相关GABA模型", "神经兴奋性异常"],
    "GABRB3": ["癫痫", "神经发育障碍", "GABA通路异常"],
    "GRIA2": ["癫痫/兴奋性毒性", "阿尔茨海默病", "神经损伤模型"],
    "SV2A": ["癫痫", "突触囊泡功能异常", "神经兴奋性疾病"],
    "SLC6A1": ["GABA转运异常癫痫", "神经兴奋性异常", "发育性癫痫性脑病"],
    "SLC6A4": ["抑郁/焦虑", "肠脑轴/IBD相关5-HT", "血小板5-HT转运"],
    "SLC18A2": ["帕金森病/单胺储存", "神经精神疾病", "药物安全性模型"],
    "SLC22A12": ["高尿酸血症", "痛风/肾结石", "尿酸性肾病"],
    "SLC34A2": ["肺鳞癌/肺腺癌", "磷酸盐转运相关肺病", "肿瘤转运模型"],
    "SLC29A1": ["核苷转运/抗病毒敏感性", "肿瘤化疗敏感性", "ENT1转运模型"],
    "ABCB1": ["多药耐药", "血脑屏障转运", "药物外排反筛"],
    "CTSS": ["自身免疫炎症", "银屑病/IBD", "肿瘤免疫微环境"],
    "CTSK": ["骨质疏松", "骨重塑疾病", "肿瘤骨转移"],
    "MMP7": ["结直肠癌", "肺癌/上皮癌侵袭", "炎症性肠病屏障损伤"],
    "ODC1": ["AML/髓系肿瘤", "结直肠癌", "多胺代谢相关疾病"],
}


DIRECTION_FALLBACK: dict[str, list[str]] = {
    "oncology": ["具体肿瘤亚型待定", "优先按靶点选择乳腺癌/肺癌/白血病/结直肠癌等模型"],
    "cardiovascular": ["心律失常/心衰/血管重塑待定", "优先按靶点选择电生理或血管细胞模型"],
    "nervous_system": ["癫痫/神经退行/神经炎症待定", "优先按靶点选择神经元兴奋性或突触功能模型"],
    "psychiatry": ["抑郁/焦虑/精神分裂症待定", "优先按神经递质或突触 readout 具体化"],
    "immune": ["自身免疫/炎症疾病待定", "优先按细胞类型选择T/B/髓系或上皮炎症模型"],
    "infectious": ["host-directed infection model待定", "优先验证宿主因子或炎症/复制辅助 readout"],
    "respiratory": ["哮喘/COPD/肺纤维化/肺炎症待定", "优先选择气道上皮或肺血管模型"],
    "digestive_gi": ["IBD/结肠炎/肠上皮屏障待定", "优先选择肠上皮或免疫共培养模型"],
    "endocrine": ["激素轴/甲状腺/肾上腺/胰岛相关疾病待定"],
    "metabolic": ["糖尿病/NASH/脂代谢异常待定"],
    "renal": ["CKD/肾纤维化/肾小球损伤待定"],
    "hepatic": ["NASH/肝炎症/肝纤维化待定"],
    "dermatology": ["银屑病/特应性皮炎/角化异常待定"],
    "hematology": ["AML/淋巴瘤/血小板或造血异常待定"],
    "musculoskeletal": ["骨质疏松/肌病/关节炎待定"],
    "ophthalmology": ["视网膜退行/青光眼/眼炎症待定"],
    "reproductive": ["前列腺/乳腺/子宫内膜/生殖轴疾病待定"],
    "urinary": ["膀胱/尿路上皮/尿酸或肾泌尿转运模型待定"],
    "pain": ["神经病理性疼痛/炎症性疼痛待定"],
}


DIRECTION_KEYWORDS: dict[str, list[str]] = {
    "oncology": ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "tumor", "neoplasm", "glioma", "sarcoma", "myeloma", "adenocarcinoma", "gist", "seminoma", "blastoma"],
    "cardiovascular": ["cardio", "heart", "myocard", "ventric", "arrhythm", "tachy", "atherosclerosis", "hypertension", "vascular", "thromb", "stroke"],
    "nervous_system": ["alzheimer", "parkinson", "huntington", "seizure", "epile", "dementia", "sclerosis", "spinal", "neuro", "amyotrophic", "schizophrenia", "bipolar"],
    "psychiatry": ["schizophrenia", "bipolar", "depression", "anxiety", "addiction", "cocaine", "nicotine", "psychi"],
    "immune": ["psoriasis", "asthma", "dermatitis", "lupus", "crohn", "colitis", "arthritis", "autoimmune", "inflammatory", "ibd", "thrombocytopenic"],
    "infectious": ["infection", "virus", "viral", "bacterial", "rhinovirus", "influenza", "hiv", "hepatitis", "sepsis"],
    "respiratory": ["lung", "asthma", "copd", "pneumonia", "pulmonary", "rhinovirus", "interstitial"],
    "digestive_gi": ["crohn", "colitis", "colon", "colorectal", "esophagus", "gastric", "duodenum", "barrett", "bowel", "pancreatic"],
    "endocrine": ["diabetes", "thyroid", "adrenal", "pituitary", "endocrine"],
    "metabolic": ["diabetes", "obesity", "metabolic", "fatty", "nash", "lipid"],
    "renal": ["kidney", "renal", "nephro", "glomer", "ckd"],
    "hepatic": ["liver", "hepato", "hepatitis", "nash", "cirrhosis"],
    "dermatology": ["psoriasis", "dermatitis", "skin", "melanoma", "keratosis", "acne"],
    "hematology": ["leukemia", "lymphoma", "myeloma", "anemia", "platelet", "hemat", "thrombocytopenic", "myelodysplastic"],
    "musculoskeletal": ["muscular", "muscle", "dystrophy", "osteo", "arthritis", "bone", "fracture", "myopathy"],
    "ophthalmology": ["retina", "eye", "ocular", "glaucoma", "macular"],
    "reproductive": ["prostate", "ovarian", "uterine", "endometrial", "testis", "breast", "fibroid", "endometriosis"],
    "urinary": ["bladder", "urinary", "urothelial", "kidney", "renal"],
    "pain": ["pain", "neuropathic", "nocicept", "migraine"],
}


TISSUE_MODEL_BY_DIRECTION = {
    "oncology": "按候选病种选择肿瘤细胞系；优先加入正常细胞/非靶点依赖细胞反筛",
    "cardiovascular": "心肌细胞、血管平滑肌/内皮细胞；离子通道候选加自动膜片钳",
    "nervous_system": "神经元/神经胶质细胞、iPSC-neuron 或过表达电生理模型",
    "psychiatry": "神经元或神经递质转运/受体功能细胞模型",
    "immune": "PBMC、T/B细胞、巨噬细胞或炎症刺激上皮共培养",
    "infectious": "感染相关宿主细胞模型；先做 host target engagement，再做复制/炎症 readout",
    "respiratory": "气道上皮、肺成纤维细胞、肺血管内皮/平滑肌",
    "digestive_gi": "肠上皮、类器官、巨噬细胞/上皮炎症共培养",
    "endocrine": "对应激素轴细胞、报告基因或内分泌相关细胞模型",
    "metabolic": "肝细胞、脂肪细胞、肌管、胰岛β细胞模型",
    "renal": "肾小管上皮、足细胞、肾小球内皮/系膜细胞、成纤维细胞",
    "hepatic": "肝细胞、Kupffer/星状细胞或肝纤维化模型",
    "dermatology": "角质形成细胞、成纤维细胞、免疫刺激皮肤模型",
    "hematology": "血液肿瘤细胞系、原代免疫/造血细胞",
    "musculoskeletal": "成骨/破骨细胞、肌管、软骨细胞或炎症滑膜细胞",
    "ophthalmology": "RPE、视网膜/小梁网细胞模型",
    "reproductive": "前列腺/乳腺/子宫内膜相关细胞模型",
    "urinary": "膀胱上皮、肾转运体过表达细胞或尿酸转运模型",
    "pain": "感觉神经元、DRG/iPSC神经元或离子通道过表达系统",
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


def split_diseases(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r";|\|\|", clean(text)) if p.strip()]
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def direction_match_score(name: str, direction: str) -> int:
    lower = name.lower()
    score = 0
    for kw in DIRECTION_KEYWORDS.get(direction, []):
        if kw in lower:
            score += 2
    # Penalize terms that are often unrelated broad/noisy CREEDS signatures.
    noisy = ["senescence", "dehydration", "nicotine addiction"]
    if any(n in lower for n in noisy):
        score -= 2
    return score


def choose_creeds(row: pd.Series, limit: int = 5) -> list[str]:
    direction = clean(row.get("direction"))
    names = split_diseases(clean(row.get("creeds_matched_disease_names")))
    ranked = sorted(names, key=lambda n: (direction_match_score(n, direction), -len(n)), reverse=True)
    ranked = [n for n in ranked if direction_match_score(n, direction) > 0]
    return ranked[:limit]


def disease_source_level(row: pd.Series, creeds: list[str], priors: list[str]) -> str:
    if creeds and priors:
        return "A_CREEDS_and_target_prior"
    if creeds:
        return "B_CREEDS_specific"
    if priors:
        return "B_target_prior_specific"
    return "C_broad_direction_only"


def confidence(row: pd.Series, level: str) -> str:
    ot = float(row.get("open_targets_score") or 0)
    tx = float(row.get("txgnn_indication_score") or 0)
    conplex = float(row.get("conplex_score") or 0)
    if level.startswith("A") and ot >= 0.5 and conplex >= 0.3:
        return "高"
    if level.startswith("B") and (ot >= 0.5 or tx >= 0.5) and conplex >= 0.2:
        return "中高"
    if level.startswith("C"):
        return "低"
    return "中"


def missing_items(row: pd.Series, level: str) -> str:
    items = []
    if level == "C_broad_direction_only":
        items.append("缺少具体病种级 CREEDS/OpenTargets 证据")
    if float(row.get("txgnn_indication_score") or 0) == 0:
        items.append("缺少具体疾病 TxGNN drug-disease 分数")
    items.append("缺少具体疾病 OpenTargets child disease association")
    items.append("缺少疾病模型/细胞系依赖性证据")
    items.append("缺少该 drug-target 的直接 target-engagement 实验证据")
    return "；".join(dict.fromkeys(items))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INFILE, low_memory=False)
    rows = []
    for _, row in df.iterrows():
        gene = clean(row.get("target_gene")).upper()
        direction = clean(row.get("direction"))
        creeds = choose_creeds(row)
        priors = TARGET_DISEASE_PRIORS.get(gene, [])
        fallback = DIRECTION_FALLBACK.get(direction, ["具体疾病待人工确定"])

        candidates: list[str] = []
        evidence_tags: list[str] = []
        for d in creeds:
            candidates.append(d)
            evidence_tags.append("CREEDS_signature")
        for d in priors:
            if d not in candidates:
                candidates.append(d)
                evidence_tags.append("target_prior")
        for d in fallback:
            if len(candidates) >= 5:
                break
            if d not in candidates:
                candidates.append(d)
                evidence_tags.append("direction_fallback")
        candidates = candidates[:5]
        level = disease_source_level(row, creeds, priors)
        rows.append(
            {
                "review_id": clean(row.get("review_id")),
                "drug_name": clean(row.get("drug_name")),
                "target_gene": gene,
                "direction": direction,
                "direction_zh": clean(row.get("direction_zh")),
                "concrete_disease_1": candidates[0] if len(candidates) > 0 else "",
                "concrete_disease_2": candidates[1] if len(candidates) > 1 else "",
                "concrete_disease_3": candidates[2] if len(candidates) > 2 else "",
                "concrete_disease_4": candidates[3] if len(candidates) > 3 else "",
                "concrete_disease_5": candidates[4] if len(candidates) > 4 else "",
                "concrete_disease_evidence_level": level,
                "concrete_disease_confidence": confidence(row, level),
                "creeds_specific_diseases_used": "; ".join(creeds),
                "target_prior_diseases_used": "; ".join(priors),
                "recommended_disease_model_zh": TISSUE_MODEL_BY_DIRECTION.get(direction, "按具体病种选择细胞/组织模型"),
                "missing_for_final_disease_call_zh": missing_items(row, level),
                "assay_lane": clean(row.get("assay_lane")),
                "primary_assay_zh": clean(row.get("primary_assay_zh")),
                "counterscreen_zh": clean(row.get("counterscreen_zh")),
                "conplex_score": row.get("conplex_score"),
                "open_targets_parent_score": row.get("open_targets_score"),
                "txgnn_broad_direction_score": row.get("txgnn_indication_score"),
                "creeds_total_signature_count": row.get("creeds_total_signature_count"),
                "literature_status": clean(row.get("literature_status")),
                "novelty_interpretation_zh": clean(row.get("novelty_interpretation_zh")),
                "major_risks_zh": clean(row.get("major_risks_zh")),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "strict895_concrete_disease_annotations.csv", index=False)

    summary = {
        "rows": int(len(out)),
        "unique_drugs": int(out["drug_name"].nunique()),
        "unique_targets": int(out["target_gene"].nunique()),
        "directions": int(out["direction"].nunique()),
        "evidence_level_counts": out["concrete_disease_evidence_level"].value_counts().to_dict(),
        "confidence_counts": out["concrete_disease_confidence"].value_counts().to_dict(),
        "direction_counts": out["direction"].value_counts().to_dict(),
        "output": str(OUTDIR / "strict895_concrete_disease_annotations.csv"),
    }
    (OUTDIR / "strict895_concrete_disease_completion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# strict895 具体疾病补全 v0 报告",
        "",
        "## 结论",
        "",
        "本轮用本地证据对 895 条 strict_top_ready 候选补全了具体疾病候选。该结果是疾病模型候选，不是最终适应症结论。",
        "",
        "## 证据等级",
        "",
        "- `A_CREEDS_and_target_prior`：CREEDS 具体疾病名和靶点疾病先验同时支持。",
        "- `B_CREEDS_specific`：有 CREEDS 具体疾病签名支持。",
        "- `B_target_prior_specific`：有靶点疾病先验，但本地 CREEDS 未命中具体疾病。",
        "- `C_broad_direction_only`：只能落到 broad direction，需要进一步补 OpenTargets child disease / 文献 / 细胞模型证据。",
        "",
        "## 规模",
        "",
        f"- rows：{summary['rows']}",
        f"- unique drugs：{summary['unique_drugs']}",
        f"- unique targets：{summary['unique_targets']}",
        f"- directions：{summary['directions']}",
        "",
        "## 证据等级分布",
        "",
    ]
    for k, v in summary["evidence_level_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 置信度分布", ""]
    for k, v in summary["confidence_counts"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## 当前仍缺什么",
        "",
        "- 具体疾病级 Open Targets child disease association，而不是父级 cancer/cardiovascular disease 等。",
        "- 具体疾病级 TxGNN drug-disease 分数；当前 TxGNN 是 19 个 broad direction。",
        "- 具体疾病细胞系/组织模型依赖性证据，例如 DepMap、HPA single-cell、疾病组织表达或模型可用性。",
        "- 每条 drug-target 的直接 target-engagement 实验证据。",
        "- 文献/专利/临床试验层面的具体疾病排重。",
        "",
        "## 输出文件",
        "",
        f"- `{OUTDIR / 'strict895_concrete_disease_annotations.csv'}`",
        f"- `{OUTDIR / 'strict895_concrete_disease_completion_summary.json'}`",
    ]
    (OUTDIR / "STRICT895_CONCRETE_DISEASE_COMPLETION_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

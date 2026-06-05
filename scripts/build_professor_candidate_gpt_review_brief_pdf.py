#!/usr/bin/env python3
"""Build a compact GPT high-intensity professor feasibility brief."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("outputs/sota_validation/professor_candidate_gpt_review_brief")
CANDIDATES = Path("outputs/disease_directions/disease_direction_integrated_candidates.csv")
LITERATURE = Path(
    "outputs/sota_validation/professor_candidate_detailed_review/"
    "professor_candidate_detailed_review_literature_audit.csv"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_professor_candidate_detailed_review_pdf import (  # noqa: E402
    build_literature_audit,
    is_truthy_number,
    load_candidate_pool,
    num,
    select_direction_top10,
    target_plan,
    text,
    val,
)


PAIR_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    ("Afatinib Dimaleate", "EGFR"): {
        "meaning": "这是 EGFR 肺癌药和 EGFR 靶点本身的召回，说明流程能找回强已知机制。",
        "judgment": "不作为新发现推进；保留一个 EGFR 阳性对照即可。",
    },
    ("Pazopanib Hydrochloride", "KIT"): {
        "category": "机制延展/先排重",
        "meaning": "这是多靶点 TKI 与 KIT 激酶的机制延展，肿瘤背景合理但已有较多公开线索。",
        "judgment": "可用于 kinase panel 校准或机制延展讨论，不宜包装为新靶点发现。",
    },
    ("Gefitinib", "EGFR"): {
        "meaning": "这是经典 EGFR-TKI 正控召回，最适合证明筛选和实验读数能工作。",
        "judgment": "作为阳性对照保留，不作为候选发现。",
    },
    ("Dacomitinib", "EGFR"): {
        "meaning": "这是已知 EGFR 抑制剂与 EGFR 的直接召回，和 Gefitinib 类似。",
        "judgment": "若预算有限，可在 EGFR 正控中只保留一个代表。",
    },
    ("Neratinib Maleate", "EGFR"): {
        "meaning": "这是 ERBB/EGFR 轴上的已知药理召回，验证价值高但新颖性低。",
        "judgment": "作为体系校准或同类对照，不作为新发现。",
    },
    ("Quizartinib Dihydrochloride", "KIT"): {
        "category": "机制延展/先排重",
        "meaning": "这是 FLT3 类激酶药与 KIT 的交叉激酶假设，机制上可理解，但公开线索已经较多。",
        "judgment": "如果做，重点是 KIT 直接 engagement 和 kinase selectivity；创新性放二线。",
    },
    ("Degarelix Acetate", "GNRHR"): {
        "meaning": "这是前列腺癌激素轴的已知生物学，更多是疾病背景正控。",
        "judgment": "可作已知机制对照，不适合当新用途候选。",
    },
    ("Alvimopan", "GCGR"): {
        "meaning": "这是较新颖的组合：外周阿片拮抗剂指向胰高血糖素受体，直接文献少，但疾病机制较弱。",
        "judgment": "可作为探索性候选，但先做低成本 GCGR 功能实验；不要直接进入肿瘤表型实验。",
    },
    ("Palonosetron Hydrochloride", "HTR3A"): {
        "meaning": "这是 5-HT3A 已知药理召回，适合做受体功能正控。",
        "judgment": "保留作 receptor assay 校准，不作为新发现。",
    },
    ("Dasatinib", "FRK"): {
        "category": "机制延展/先排重",
        "meaning": "Dasatinib 是广谱激酶药，FRK 属相邻激酶空间，解释性强但新颖性弱。",
        "judgment": "适合 kinase panel 或机制延展，不是首选新用途。",
    },
    ("Maraviroc", "NPY5R"): {
        "meaning": "HIV/CCR5 药物指向 NPY5R，感染病方向解释不直接，更像模型外推或 off-target 信号。",
        "judgment": "暂缓，不建议首轮采购。",
    },
    ("Maraviroc", "CXCR4"): {
        "meaning": "CXCR4 和 HIV 入侵相关，但 Maraviroc 的经典靶点是 CCR5，不是 CXCR4。",
        "judgment": "可作为专家讨论的机制对照，但不能直接说找到了 Maraviroc 的新靶点。",
    },
    ("Maraviroc", "CHRM3"): {
        "meaning": "感染方向和胆碱能受体联系不直接，且同一药物连续命中多个受体是广义 off-target 风险信号。",
        "judgment": "暂缓，除非专家有明确感染免疫机制假设。",
    },
    ("Maraviroc", "EDNRB"): {
        "meaning": "内皮素受体和感染病之间需要额外机制桥接，当前证据不足。",
        "judgment": "暂缓，不适合作为首轮验证。",
    },
    ("Maraviroc", "OXTR"): {
        "meaning": "OXTR 与感染病方向距离较远，当前更像药物-受体泛化信号。",
        "judgment": "暂缓，优先级低。",
    },
    ("Maraviroc", "AVPR1B"): {
        "meaning": "AVPR1B 属神经内分泌受体，感染病解释链较弱。",
        "judgment": "暂缓，除非有明确应激轴/免疫轴假设。",
    },
    ("Maraviroc", "SLC5A2"): {
        "meaning": "SGLT2 转运体和感染方向的直接机制不清楚，且药物本身不是转运体药。",
        "judgment": "暂缓，不作为首轮。",
    },
    ("Maraviroc", "SLC6A4"): {
        "meaning": "5-HT 转运体与感染病方向关联较间接，实验容易变成 CNS/off-target 讨论。",
        "judgment": "暂缓。",
    },
    ("Maraviroc", "AVPR1A"): {
        "meaning": "血管加压素受体与感染病有应激/血流背景，但当前证据不足以采购验证。",
        "judgment": "二线或暂缓。",
    },
    ("Maraviroc", "CHRM2"): {
        "meaning": "胆碱能受体与感染方向需要很强的免疫调节假设，当前不足。",
        "judgment": "暂缓。",
    },
    ("Etonogestrel", "AR"): {
        "category": "暂缓/不建议首轮",
        "meaning": "激素类药物指向 AR，和感染病之间缺少直接机制桥接，更像疾病标签或宿主背景信号。",
        "judgment": "暂缓，不作为感染方向首轮候选；除非专家指定病毒-激素轴亚型。",
    },
    ("Cobicistat", "FZD2"): {
        "category": "二线探索",
        "meaning": "HIV 药物增强剂指向 FZD2/Wnt 受体，是宿主通路调节假设；有新颖性，但感染病解释链需要补强。",
        "judgment": "建议列为二线探索；先做 FZD2/Wnt reporter 和 Cobicistat assay-interference 排查，再决定是否进入感染模型。",
    },
    ("Darunavir And Cobicistat", "PTGER2"): {
        "category": "二线探索",
        "meaning": "抗 HIV 组合药指向 PTGER2，可能触及前列腺素-炎症轴，但组合制剂和宿主受体效应会强烈混杂。",
        "judgment": "二线保留；优先拆分 Darunavir 与 Cobicistat 单药验证 PTGER2 功能。",
    },
    ("Cyclosporine", "PPIA"): {
        "category": "阳性对照/校准",
        "meaning": "Cyclosporine-PPIA 是 cyclophilin 轴的经典机制，和病毒复制/宿主因子讨论有清楚入口。",
        "judgment": "适合作为感染方向机制正控或流程校准，不包装为新发现。",
    },
    ("Degarelix Acetate", "PTAFR"): {
        "category": "二线探索",
        "meaning": "PTAFR 与炎症和病原体-宿主相互作用有一定讨论空间，但 Degarelix 到 PTAFR 的药物-靶点证据较跳跃。",
        "judgment": "建议列为二线验证；先做 PTAFR 功能实验，再根据靶点证据决定是否做感染表型。",
    },
    ("Doxazosin Mesylate", "ADRA1A"): {
        "meaning": "这是 doxazosin 的经典 alpha-1 受体药理，心血管方向正控。",
        "judgment": "作阳性对照即可。",
    },
    ("Octreotide Acetate", "SSTR2"): {
        "category": "阳性对照/校准",
        "meaning": "SSTR2 是 Octreotide 的经典受体，但心血管方向不是最自然的验证入口。",
        "judgment": "作为已知受体对照可保留，新发现价值低。",
    },
    ("Dorzolamide Hydrochloride", "CA2"): {
        "meaning": "这是碳酸酐酶抑制剂和 CA2 的已知机制召回。",
        "judgment": "适合作为 enzyme assay 阳性对照。",
    },
    ("Dorzolamide Hydrochloride", "CA9"): {
        "meaning": "CA9 是低氧和 pH 调节相关同工酶，和血管/缺氧背景有可讨论空间。",
        "judgment": "建议进入专家讨论，优先做 CA 同工酶选择性实验。",
    },
    ("Doxazosin Mesylate", "HTR4"): {
        "meaning": "HTR4 是心脏和神经相关 GPCR，和心血管有潜在联系，但 doxazosin 的 alpha-1 主药理会强烈干扰解释。",
        "judgment": "建议进入专家讨论，但必须证明不是 alpha-1 受体效应。",
    },
    ("Octreotide Acetate", "EDNRA"): {
        "meaning": "EDNRA 是血管收缩核心受体，但 Octreotide 指向它的解释链弱。",
        "judgment": "暂缓，除非有明确内皮素轴机制。",
    },
    ("Octreotide Acetate", "PTGER4"): {
        "meaning": "PTGER4 与炎症和血管有关，但 Octreotide 到该受体的证据不足。",
        "judgment": "暂缓。",
    },
    ("Octreotide Acetate", "OPRM1"): {
        "meaning": "OPRM1 与 Octreotide 的经典机制距离较远，心血管解释弱。",
        "judgment": "暂缓。",
    },
    ("Doxazosin Mesylate", "OPRM1"): {
        "meaning": "这是 alpha-1 阻断药指向阿片受体的新靶点假设，直接文献少但疾病解释偏弱。",
        "judgment": "可做低成本 OPRM1 功能实验，但不建议优先进入疾病模型。",
    },
    ("Doxazosin Mesylate", "HTR2A"): {
        "meaning": "HTR2A 参与血管张力，心血管解释比 OPRM1 更自然，但 alpha-1 主药理仍是主要混杂。",
        "judgment": "建议进入专家讨论；必须同步做 ADRA1A counterscreen。",
    },
    ("Naloxone Hydrochloride", "OPRK1"): {
        "category": "阳性对照/校准",
        "meaning": "这是阿片受体药理空间的已知召回，适合神经/精神方向受体 assay 校准。",
        "judgment": "阳性/机制对照。",
    },
    ("Buprenorphine", "OPRK1"): {
        "meaning": "Buprenorphine 与 OPRK1 属已知阿片受体药理，实验解释清楚但不是新发现。",
        "judgment": "作对照即可。",
    },
    ("Naltrexone", "OPRK1"): {
        "category": "阳性对照/校准",
        "meaning": "Naltrexone 与阿片受体系统是已知机制召回。",
        "judgment": "作对照即可。",
    },
    ("Atropine", "CHRM2"): {
        "meaning": "这是 atropine 的经典抗胆碱能机制召回，适合作为 muscarinic assay 正控。",
        "judgment": "阳性对照。",
    },
    ("Atropine", "HTR1A"): {
        "meaning": "Atropine 指向 5-HT1A 是机制延展，已有公开线索，且胆碱能主药理会混杂。",
        "judgment": "若做，只能作为 off-target 机制延展，不作为高新颖性候选。",
    },
    ("Atropine", "DRD3"): {
        "meaning": "Atropine 指向 DRD3 直接文献少，神经方向可讨论，但胆碱能混杂很强。",
        "judgment": "可作为探索性 GPCR 候选，先做 DRD3 功能和 dopamine receptor counterscreen。",
    },
    ("Naloxone Hydrochloride", "F2R"): {
        "meaning": "阿片拮抗剂指向 PAR1/F2R，神经方向解释不直观。",
        "judgment": "暂缓。",
    },
    ("Naltrexone", "DRD3"): {
        "meaning": "Naltrexone 指向 DRD3 有神经相关性，但 opioid 主药理足以解释很多表型。",
        "judgment": "暂缓或二线。",
    },
    ("Naloxone Hydrochloride", "OXTR"): {
        "meaning": "OXTR 与行为/应激有关，但 Naloxone 到 OXTR 的直接可解释性弱。",
        "judgment": "暂缓。",
    },
    ("Naltrexone", "F2R"): {
        "meaning": "F2R/PAR1 与阿片药理距离较远，当前像 off-target 信号。",
        "judgment": "暂缓。",
    },
    ("Azelastine Hydrochloride", "RXFP1"): {
        "meaning": "Azelastine 是抗组胺药，RXFP1 属 relaxin 受体，炎症方向有组织重塑/免疫调节可讨论空间。",
        "judgment": "免疫方向里较值得专家讨论，先做 RXFP1 功能实验和 H1 receptor counterscreen。",
    },
    ("Azelastine Hydrochloride", "FSHR"): {
        "meaning": "FSHR 是内分泌受体，和免疫/炎症方向距离较远。",
        "judgment": "可做二线受体实验，但不是首选。",
    },
    ("Olopatadine Hydrochloride", "DRD1"): {
        "meaning": "抗组胺药指向 DRD1，实验可做但免疫方向解释间接。",
        "judgment": "二线探索，先做 receptor assay，不做复杂免疫模型。",
    },
    ("Calcipotriene", "VDR"): {
        "meaning": "这是维生素 D 类药物与 VDR 的已知炎症/皮肤免疫机制，解释最清楚。",
        "judgment": "很适合作为免疫/炎症方向阳性对照。",
    },
    ("Azelastine Hydrochloride", "TSHR"): {
        "meaning": "TSHR 属甲状腺轴，和炎症方向有自身免疫背景但药物-靶点假设较跳跃。",
        "judgment": "二线，需专家先确认疾病亚型。",
    },
    ("Azelastine Hydrochloride", "LHCGR"): {
        "meaning": "LHCGR 属生殖内分泌受体，免疫方向解释较弱。",
        "judgment": "暂缓或二线。",
    },
    ("Olopatadine Hydrochloride", "HTR1B"): {
        "meaning": "HTR1B 是 GPCR off-target 假设，和炎症关联间接。",
        "judgment": "二线探索。",
    },
    ("Oxycodone Hydrochloride", "EDNRB"): {
        "meaning": "阿片药物指向 EDNRB，安全和主药理混杂都很强。",
        "judgment": "不建议首轮。",
    },
    ("Olopatadine Hydrochloride", "HTR4"): {
        "category": "二线探索",
        "meaning": "HTR4 与免疫/肠道炎症有一定可讨论空间，但仍是抗组胺药 off-target 假设。",
        "judgment": "二线，可做 GPCR 功能实验。",
    },
    ("Olopatadine Hydrochloride", "DRD2"): {
        "meaning": "DRD2 与免疫方向联系间接，且容易偏向神经药理解释。",
        "judgment": "二线或暂缓。",
    },
}


DIRECTION_PAIR_OVERRIDES: dict[tuple[str, str, str], dict[str, str]] = {
    ("infectious_disease", "Gefitinib", "EGFR"): {
        "category": "机制延展/先排重",
        "meaning": "EGFR 是上皮细胞信号和部分感染模型中可讨论的宿主通路，但 Gefitinib-EGFR 本身是已知强机制。",
        "judgment": "先做文献排重；若专家关心宿主 EGFR 依赖感染模型，可作机制延展对照。",
    },
    ("infectious_disease", "Naloxone Hydrochloride", "OPRK1"): {
        "category": "暂缓/不建议首轮",
        "meaning": "这是阿片受体药理空间的已知信号，但感染病方向解释不自然。",
        "judgment": "不作为感染方向首轮候选；如做只能作为受体 assay 对照。",
    },
    ("infectious_disease", "Doxazosin Mesylate", "ADRA1A"): {
        "category": "暂缓/不建议首轮",
        "meaning": "这是 Doxazosin 的经典 alpha-1 受体药理，心血管解释强，但感染病解释弱。",
        "judgment": "不放入感染方向首轮；更适合作为心血管方向阳性对照。",
    },
    ("infectious_disease", "Octreotide Acetate", "SSTR2"): {
        "category": "暂缓/不建议首轮",
        "meaning": "Octreotide-SSTR2 是清楚的已知受体药理，但感染方向不是自然验证入口。",
        "judgment": "不作为感染方向首轮候选；可在内分泌/受体 assay 中作对照。",
    },
}


def esc(value: Any) -> str:
    return html.escape(text(value, ""), quote=True)


def category(row: pd.Series, lit: pd.Series) -> str:
    status = text(lit.get("literatureStatusZh"), "")
    novelty = text(row.get("noveltyClass"), "")
    group = text(row.get("expertNoveltyGroup"), "")
    if "已知机制" in status or is_truthy_number(row.get("knownDrugTargetPair")):
        return "阳性对照/校准"
    if group == "positive_control_or_known_mechanism":
        return "阳性对照/校准"
    if "风险" in status or group == "risk_review" or "negative" in novelty:
        return "暂缓/不建议首轮"
    if "公开组合线索较多" in status:
        return "机制延展/先排重"
    if "未见明显直接组合文献" in status or "有少量文献线索" in status:
        if text(row.get("target")) in {"GCGR", "CA9", "HTR4", "HTR2A", "DRD3", "RXFP1"}:
            return "优先专家讨论"
        return "二线探索"
    return "二线探索"


def action_verb(cat: str) -> str:
    return {
        "优先专家讨论": "建议进入专家讨论，但先做低成本靶点实验。",
        "二线探索": "保留为二线验证；先做低成本靶点或通路实验，结果清楚后再进入疾病模型。",
        "机制延展/先排重": "先人工排重，再决定是否做机制延展。",
        "阳性对照/校准": "只作阳性对照或流程校准。",
        "暂缓/不建议首轮": "暂缓或忽略，不进入首轮。",
    }.get(cat, "二线审阅。")


def main_confounder(row: pd.Series, cat: str) -> str:
    drug = text(row.get("drug"))
    target = text(row.get("target"))
    if cat == "阳性对照/校准":
        return "主要问题不是可行性，而是新颖性低；用途是证明实验体系和筛选流程能找回已知药理。"
    if drug == "Maraviroc":
        if target == "CXCR4":
            return "Maraviroc 的经典机制是 CCR5；CXCR4 虽与 HIV 相关，但需要先排除只是感染病背景相邻信号。"
        return "同一 Maraviroc 反复命中多个弱相关 GPCR/转运体，提示模型可能被药物-疾病背景和广义 off-target 性质放大。"
    if "Cobicistat" in drug:
        return "Cobicistat 是药代增强剂，CYP/转运体和 assay interference 风险高；组合制剂必须拆成单药验证。"
    if drug == "Cyclosporine":
        return "Cyclosporine-PPIA 机制清楚，但免疫抑制和抗感染表型容易混杂；用途更适合作为机制正控。"
    if drug == "Etonogestrel":
        return "激素轴和感染病之间需要明确亚型假设，否则很容易把内分泌背景误读成感染机制。"
    if drug.startswith("Doxazosin"):
        return "Doxazosin 的 alpha-1 主药理很强，任何心血管或 GPCR 表型都必须用 ADRA1A counterscreen 排除混杂。"
    if drug.startswith("Azelastine") or drug.startswith("Olopatadine"):
        return "抗组胺主药理和 GPCR off-target 容易混在一起，需要 H1 receptor 与同家族受体 counterscreen。"
    if drug in {"Naloxone Hydrochloride", "Naltrexone", "Buprenorphine", "Oxycodone Hydrochloride", "Alvimopan"}:
        return "阿片药理和安全上下文会强烈影响解释，必须先证明候选靶点依赖，而不是泛 opioid 表型。"
    if drug == "Atropine":
        return "Atropine 的抗胆碱能主药理很强，DRD/HTR 等新假设必须先和 muscarinic receptor 效应区分。"
    if drug.startswith("Octreotide"):
        return "Octreotide 的 SSTR 主药理和肽类药物暴露特征明显，非 SSTR 靶点需要非常直接的 target engagement 支持。"
    if target in {"EGFR", "KIT", "FRK"}:
        return "激酶药常有多靶点性质，需要 kinase panel 和通路 readout 区分真实靶点与相邻激酶抑制。"
    if target in {"CA2", "CA9"}:
        return "碳酸酐酶同工酶之间容易交叉，需要同工酶选择性和 pH/毒性干扰排查。"
    return "主要风险是靶点依赖性不足；必须先做直接功能或结合实验，再解释疾病表型。"


def first_round_bucket(row: pd.Series) -> str:
    pair = (text(row.get("drug")), text(row.get("target")))
    direction = text(row.get("direction"))
    if direction == "infectious_disease" and pair == ("Gefitinib", "EGFR"):
        return ""
    exploratory = {
        ("Alvimopan", "GCGR"),
        ("Dorzolamide Hydrochloride", "CA9"),
        ("Doxazosin Mesylate", "HTR4"),
        ("Doxazosin Mesylate", "HTR2A"),
        ("Atropine", "DRD3"),
        ("Azelastine Hydrochloride", "RXFP1"),
    }
    controls = {
        ("Gefitinib", "EGFR"),
        ("Dorzolamide Hydrochloride", "CA2"),
        ("Atropine", "CHRM2"),
        ("Buprenorphine", "OPRK1"),
        ("Calcipotriene", "VDR"),
        ("Cyclosporine", "PPIA"),
    }
    if pair in exploratory:
        return "首轮探索"
    if pair in controls:
        return "正控/校准"
    return ""


def candidate_source(root: Path) -> Path:
    return root / CANDIDATES


def gpt_review(row: pd.Series, lit: pd.Series) -> dict[str, str]:
    pair = (text(row.get("drug")), text(row.get("target")))
    direction_pair = (text(row.get("direction")), text(row.get("drug")), text(row.get("target")))
    override = DIRECTION_PAIR_OVERRIDES.get(direction_pair, PAIR_OVERRIDES.get(pair, {}))
    plan = target_plan(row)
    cat = override.get("category", category(row, lit))
    meaning = override.get(
        "meaning",
        f"{row.get('drug')} 指向 {row.get('target')} 的假设，需要先证明这个靶点确实被药物调节，再解释疾病表型。",
    )
    judgment = override.get("judgment", action_verb(cat))
    why_do = {
        "优先专家讨论": "新颖性较高，且靶点功能实验相对可执行；适合先用低成本实验判断是否有真实 target engagement。",
        "二线探索": "实验通常能做，但疾病方向或药理解释不够直接，需要专家确认机制链后再投入。",
        "机制延展/先排重": "机制上可解释，适合用来补充已知药理边界，但公开线索较多，发现价值有限。",
        "阳性对照/校准": "有清楚已知机制，能证明平台和实验 readout 是否可靠。",
        "暂缓/不建议首轮": "疾病解释链较弱、主药理混杂或安全/负向上下文较强，投入后很容易得到难解释结果。",
    }[cat]
    why_not = {
        "优先专家讨论": "不要直接做疾病细胞表型；如果靶点 readout 不成立，后续疾病实验没有解释力。",
        "二线探索": "当前最大问题不是能不能测，而是测出来以后是否能讲清楚机制。",
        "机制延展/先排重": "如果人工确认已有直接机制验证，应降级为背景材料或忽略。",
        "阳性对照/校准": "不是新用途/新靶点，不应占用太多探索预算。",
        "暂缓/不建议首轮": "同一药物反复命中多个弱相关靶点时，通常提示广义 polypharmacology 或模型泛化，不适合先做。",
    }[cat]
    first_test = plan["primary"]
    question = (
        f"在非毒性、临床可达或体外合理浓度下，{row.get('drug')} 是否能产生 "
        f"{row.get('target')} 依赖的剂量反应，并被正交实验和 counterscreen 支持？"
    )
    return {
        "gptCategoryZh": cat,
        "gptRecommendationZh": judgment,
        "gptMeaningZh": meaning,
        "gptWhyDoZh": why_do,
        "gptWhyNotZh": why_not,
        "gptFirstExperimentZh": first_test,
        "gptMainConfounderZh": main_confounder(row, cat),
        "gptGoNoGoZh": plan["go"],
        "gptExpertQuestionZh": question,
        "gptFirstRoundBucketZh": first_round_bucket(row),
    }


def load_panel(root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = load_candidate_pool(root)
    panel = select_direction_top10(candidates)
    lit_path = root / LITERATURE
    if lit_path.exists():
        literature = pd.read_csv(lit_path)
    else:
        literature = build_literature_audit(panel, out_dir, refresh=False)
    panel_keys = {(r["direction"], r["drug"], r["target"]) for _, r in panel.iterrows()}
    literature_keys = {(r["direction"], r["drug"], r["target"]) for _, r in literature.iterrows()}
    if panel_keys - literature_keys:
        missing_records = []
        for _, row in panel.iterrows():
            key = (row["direction"], row["drug"], row["target"])
            if key in literature_keys:
                continue
            missing_records.append(
                {
                    "direction": row["direction"],
                    "directionLabelZh": row["directionLabelZh"],
                    "drug": row["drug"],
                    "target": row["target"],
                    "pubmedDirectPairQuery": "",
                    "pubmedDirectPairCount": pd.NA,
                    "pubmedDirectPairPmids": "",
                    "pubmedDirectPairUrl": "",
                    "pubmedDiseasePairQuery": "",
                    "pubmedDiseasePairCount": pd.NA,
                    "pubmedDiseasePairPmids": "",
                    "pubmedDiseasePairUrl": "",
                    "clinicalTrialsDrugDiseaseQuery": "",
                    "clinicalTrialsDrugDiseaseCount": pd.NA,
                    "clinicalTrialsDrugDiseaseNcts": "",
                    "clinicalTrialsDrugDiseaseUrl": "",
                    "queriedUtc": "",
                    "literatureStatusZh": "待人工排重/缓存缺失",
                    "literatureActionZh": "本版先按计算证据和 GPT 可行性复核解释；正式立项前再补 PubMed/ClinicalTrials 排重。",
                    "reviewActionZh": "进入专家讨论前补做人工文献核查。",
                    "noveltyPriorityZh": "待排重。",
                }
            )
        literature = pd.concat([literature, pd.DataFrame(missing_records)], ignore_index=True)
    rows = []
    lookup = {(r["direction"], r["drug"], r["target"]): r for _, r in literature.iterrows()}
    for _, row in panel.iterrows():
        lit = lookup[(row["direction"], row["drug"], row["target"])]
        rows.append({**row.to_dict(), **gpt_review(row, lit)})
    reviewed = pd.DataFrame(rows)
    return reviewed, literature


def summary_table(reviewed: pd.DataFrame) -> str:
    rows = []
    for _, row in reviewed.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(row['reportRank'])}</td>"
            f"<td>{esc(row['directionLabelZh'])}<br><span>Top {int(row['reviewRankInDirection'])}</span></td>"
            f"<td><b>{esc(row['drug'])}</b><br><span>{esc(row['target'])}</span></td>"
            f"<td>{esc(row['gptCategoryZh'])}</td>"
            f"<td>{esc(row['gptMeaningZh'])}</td>"
            f"<td>{esc(row['gptRecommendationZh'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def first_round_table(reviewed: pd.DataFrame) -> str:
    block = reviewed[reviewed["gptFirstRoundBucketZh"] != ""].copy()
    bucket_order = {"首轮探索": 0, "正控/校准": 1}
    block["_bucket_order"] = block["gptFirstRoundBucketZh"].map(bucket_order)
    block = block.sort_values(["_bucket_order", "direction", "reviewRankInDirection"])
    rows = []
    for _, row in block.iterrows():
        rows.append(
            "<tr>"
            f"<td>{esc(row['gptFirstRoundBucketZh'])}</td>"
            f"<td>{esc(row['directionLabelZh'])}</td>"
            f"<td><b>{esc(row['drug'])}</b><br><span>{esc(row['target'])}</span></td>"
            f"<td>{esc(row['gptRecommendationZh'])}</td>"
            f"<td>{esc(row['gptFirstExperimentZh'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def candidate_cards(reviewed: pd.DataFrame, literature: pd.DataFrame) -> str:
    lit_lookup = {(r["direction"], r["drug"], r["target"]): r for _, r in literature.iterrows()}
    cards = []
    for _, row in reviewed.iterrows():
        lit = lit_lookup[(row["direction"], row["drug"], row["target"])]
        cards.append(
            f"""
            <article class="candidate {css_class(row['gptCategoryZh'])}">
              <div class="candidate-top">
                <div>
                  <p class="kicker">{esc(row['directionLabelZh'])} Top {int(row['reviewRankInDirection'])}</p>
                  <h3>{esc(row['drug'])} - {esc(row['target'])}</h3>
                  <p class="protein">{esc(row['proteinName'])}</p>
                </div>
                <div class="tag">{esc(row['gptCategoryZh'])}</div>
              </div>
              <p><b>专家先看这一句：</b>{esc(row['gptMeaningZh'])}</p>
              <p><b>GPT 可行性判断：</b>{esc(row['gptRecommendationZh'])}</p>
              <div class="two">
                <p><b>为什么可以考虑：</b>{esc(row['gptWhyDoZh'])}</p>
                <p><b>为什么可能不值得做：</b>{esc(row['gptWhyNotZh'])}</p>
              </div>
              <p><b>主要混杂因素：</b>{esc(row['gptMainConfounderZh'])}</p>
              <p><b>第一步实验：</b>{esc(row['gptFirstExperimentZh'])}</p>
              <p><b>Go/No-Go 门槛：</b>{esc(row['gptGoNoGoZh'])}</p>
              <p><b>排重状态：</b>{esc(lit.get('literatureStatusZh'))}。PubMed drug-target {int_or_dash(lit.get('pubmedDirectPairCount'))}，drug-target-disease {int_or_dash(lit.get('pubmedDiseasePairCount'))}，ClinicalTrials 药物-疾病方向 {int_or_dash(lit.get('clinicalTrialsDrugDiseaseCount'))}。</p>
              <p><b>讨论问题：</b>{esc(row['gptExpertQuestionZh'])}</p>
              <p class="small">保留的关键数值：direction {num(row.get('directionScore'))}，affinity {num(row.get('affinityScore'))}，DiffDock {num(row.get('diffdock'), 2)}。这些数值只作排序/结构审阅辅助，不等同于体内疗效。</p>
            </article>
            """
        )
    return "\n".join(cards)


def css_class(category_zh: str) -> str:
    if "优先" in category_zh:
        return "go"
    if "二线" in category_zh:
        return "review"
    if "校准" in category_zh:
        return "control"
    if "暂缓" in category_zh:
        return "hold"
    return "extension"


def int_or_dash(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{int(float(value)):,}"
    except Exception:
        return "-"


def build_html(reviewed: pd.DataFrame, literature: pd.DataFrame) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    counts = reviewed["gptCategoryZh"].value_counts().to_dict()
    direction_counts = reviewed["directionLabelZh"].value_counts().to_dict()
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>BioMaster GPT 精简专家可行性复核</title>
  <style>
    @font-face {{
      font-family: "Noto Sans CJK";
      src: url("file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc");
    }}
    @font-face {{
      font-family: "Noto Sans CJK";
      src: url("file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc");
      font-weight: 700;
    }}
    @page {{ size: A4; margin: 12mm 12mm 13mm 12mm; @bottom-right {{ content: counter(page); color: #667085; font-size: 8pt; }} }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Noto Sans CJK", sans-serif; color: #162033; font-size: 8.6pt; line-height: 1.42; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ font-size: 25pt; line-height: 1.1; color: #153b5c; margin-bottom: 5mm; }}
    h2 {{ font-size: 14pt; color: #153b5c; margin-bottom: 2mm; }}
    h3 {{ font-size: 11pt; color: #172b45; margin-bottom: 1mm; }}
    p {{ margin-bottom: 1.8mm; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 7.1pt; line-height: 1.25; }}
    th {{ background: #eaf2f8; color: #153b5c; text-align: left; padding: 1.4mm; border-bottom: 1px solid #bac9d8; }}
    td {{ vertical-align: top; padding: 1.25mm; border-bottom: 1px solid #e3e8ef; }}
    td span, .small, .protein {{ color: #667085; font-size: 7.2pt; }}
    .cover {{ min-height: 252mm; page-break-after: always; display: flex; flex-direction: column; justify-content: center; }}
    .lead {{ font-size: 11pt; color: #344054; max-width: 170mm; }}
    .kicker {{ text-transform: uppercase; letter-spacing: .06em; color: #0f6b7a; font-weight: 700; font-size: 7.2pt; margin-bottom: 1mm; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.5mm; margin: 6mm 0; }}
    .metric {{ border: 1px solid #d7e0ea; background: #f7fafc; border-radius: 2mm; padding: 2.7mm; }}
    .metric b {{ display: block; font-size: 15pt; color: #0f5d93; }}
    .note {{ border-left: 4px solid #2f7ca8; background: #f5faff; padding: 3mm; margin: 4mm 0; }}
    .section {{ page-break-after: always; }}
    .guide {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3mm; }}
    .guide div {{ border: 1px solid #d8e0ea; border-radius: 2mm; padding: 2.5mm; break-inside: avoid; }}
    .candidate {{ break-inside: avoid; page-break-inside: avoid; border: 1px solid #d8e0ea; border-left-width: 4px; border-radius: 2mm; padding: 2.6mm; margin-bottom: 3mm; }}
    .candidate-top {{ display: grid; grid-template-columns: 1fr 25mm; gap: 2mm; }}
    .tag {{ text-align: center; border-radius: 999px; padding: 1mm; font-size: 7pt; font-weight: 700; align-self: start; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2.5mm; }}
    .go {{ border-left-color: #1f8f5f; }}
    .go .tag {{ background: #e8f6ef; color: #12613c; }}
    .review {{ border-left-color: #3b7db8; }}
    .review .tag {{ background: #eaf3ff; color: #164d83; }}
    .extension {{ border-left-color: #b9811d; }}
    .extension .tag {{ background: #fff5e6; color: #8a4b08; }}
    .control {{ border-left-color: #7a8798; }}
    .control .tag {{ background: #eef2f6; color: #344054; }}
    .hold {{ border-left-color: #c14242; }}
    .hold .tag {{ background: #fff0f0; color: #9f2a2a; }}
  </style>
</head>
<body>
  <section class="cover">
    <p class="kicker">GPT high-intensity feasibility pass</p>
    <h1>BioMaster 候选精简专家可行性复核</h1>
    <p class="lead">这版按“GPT 高强度人工复核”的方式重读五个疾病方向各 Top10 候选：不再把分数当结论，而是逐项判断这个组合是什么意思、是否像新发现、第一步怎么验证、为什么可能不值得做。结论仍然是计算优先级和专家审阅建议，不是生物学验证结果。</p>
    <div class="metrics">
      <div class="metric"><b>{len(reviewed)}</b><span>候选</span></div>
      <div class="metric"><b>{len(direction_counts)}</b><span>疾病方向</span></div>
      <div class="metric"><b>{counts.get('优先专家讨论', 0)}</b><span>优先讨论</span></div>
      <div class="metric"><b>{counts.get('暂缓/不建议首轮', 0)}</b><span>暂缓/不首轮</span></div>
    </div>
    <div class="note">
      <p><b>核心口径。</b>先看这个候选能否被一个直接、便宜、可解释的靶点实验验证；再看疾病背景和已有文献。如果已经是老机制，就只作阳性对照；如果疾病解释链弱或同一药物反复命中许多弱相关靶点，就先暂缓。</p>
      <p class="small">生成时间：{generated}。长版数据 PDF 和 literature audit CSV 仍保留为证据备查。</p>
    </div>
  </section>

  <section class="section">
    <p class="kicker">How to read this report</p>
    <h2>专家阅读说明</h2>
    <div class="guide">
      <div><b>亲和分</b><p>说明模型认为药物和蛋白可能相互作用。它是筛选信号，不代表真实 IC50，也不代表体内有效。</p></div>
      <div><b>疾病证据</b><p>Open Targets、TxGNN 和 KG 说明靶点或药物是否和疾病方向有知识图谱关系。它能帮助解释，但不能替代实验。</p></div>
      <div><b>CMap/CREEDS</b><p>看药物表达签名是否可能反转疾病签名。它是方向性线索，受细胞系、剂量和时间影响很大。</p></div>
      <div><b>GTEx/HPA/DepMap</b><p>帮助判断靶点在哪些组织或疾病模型中表达，以及肿瘤中是否有依赖性。主要用于选模型。</p></div>
      <div><b>DiffDock</b><p>看结构姿态是否可讨论。负数 confidence 不是坏事，也不是结合自由能；只能作为结构审阅辅助。</p></div>
      <div><b>GPT 审阅</b><p>把上述证据合并成实际判断：先做哪个、哪些只作对照、哪些容易浪费实验预算。</p></div>
    </div>
  </section>

  <section class="section">
    <p class="kicker">Suggested first wave</p>
    <h2>建议首轮实验组合</h2>
    <p>50 个候选不建议一起做。首轮应由少量“正控/校准”证明实验体系有效，再选择能用低成本靶点实验判定真假的探索候选。下面清单是本次 GPT 复核后建议优先给专家讨论的组合。</p>
    <table>
      <thead><tr><th>用途</th><th>方向</th><th>候选</th><th>判断</th><th>第一步实验</th></tr></thead>
      <tbody>{first_round_table(reviewed)}</tbody>
    </table>
    <div class="note">
      <p><b>首轮解释。</b>正控不是发现结果，而是证明 assay 和读数能工作；探索候选也不应直接进入复杂疾病模型，应先用 target engagement、功能 readout 和 counterscreen 判断是否值得继续。</p>
    </div>
  </section>

  <section class="section">
    <p class="kicker">Triage summary</p>
    <h2>候选总览</h2>
    <p>疾病方向分布：{esc(json.dumps(direction_counts, ensure_ascii=False))}。GPT 审阅分类：{esc(json.dumps(counts, ensure_ascii=False))}。</p>
    <table>
      <thead><tr><th>#</th><th>方向</th><th>候选</th><th>GPT 分类</th><th>专家先看这一句</th><th>建议</th></tr></thead>
      <tbody>{summary_table(reviewed)}</tbody>
    </table>
  </section>

  <section>
    <p class="kicker">Candidate-level pass</p>
    <h2>逐项 GPT 可行性复核</h2>
    {candidate_cards(reviewed, literature)}
  </section>
</body>
</html>"""


def build_markdown(reviewed: pd.DataFrame, literature: pd.DataFrame) -> str:
    lit_lookup = {(r["direction"], r["drug"], r["target"]): r for _, r in literature.iterrows()}
    lines = [
        "# BioMaster 候选精简专家可行性复核",
        "",
        "这版把候选翻译成专家讨论口径：是什么意思、是否值得做、第一步做什么、为什么可能不值得做。",
        "",
        "## 候选总览",
        "",
        "| # | 方向 | 候选 | GPT 分类 | 专家先看这一句 | 建议 |",
        "|---:|---|---|---|---|---|",
    ]
    for _, row in reviewed.iterrows():
        lines.append(
            f"| {int(row['reportRank'])} | {row['directionLabelZh']} Top {int(row['reviewRankInDirection'])} | "
            f"{row['drug']} - {row['target']} | {row['gptCategoryZh']} | {row['gptMeaningZh']} | "
            f"{row['gptRecommendationZh']} |"
        )
    lines.append("\n## 逐项复核\n")
    for _, row in reviewed.iterrows():
        lit = lit_lookup[(row["direction"], row["drug"], row["target"])]
        lines.extend(
            [
                f"### {int(row['reportRank'])}. {row['drug']} - {row['target']} ({row['directionLabelZh']} Top {int(row['reviewRankInDirection'])})",
                "",
                f"- GPT 分类：{row['gptCategoryZh']}",
                f"- 专家先看这一句：{row['gptMeaningZh']}",
                f"- GPT 可行性判断：{row['gptRecommendationZh']}",
                f"- 为什么可以考虑：{row['gptWhyDoZh']}",
                f"- 为什么可能不值得做：{row['gptWhyNotZh']}",
                f"- 主要混杂因素：{row['gptMainConfounderZh']}",
                f"- 第一步实验：{row['gptFirstExperimentZh']}",
                f"- Go/No-Go 门槛：{row['gptGoNoGoZh']}",
                f"- 排重状态：{lit['literatureStatusZh']}；PubMed drug-target {int_or_dash(lit.get('pubmedDirectPairCount'))}；drug-target-disease {int_or_dash(lit.get('pubmedDiseasePairCount'))}；ClinicalTrials 药物-疾病方向 {int_or_dash(lit.get('clinicalTrialsDrugDiseaseCount'))}。",
                f"- 讨论问题：{row['gptExpertQuestionZh']}",
                "",
            ]
        )
    return "\n".join(lines)


def publish(root: Path, pdf_path: Path) -> Path:
    assets = root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    published = assets / "professor-candidate-gpt-review-brief.pdf"
    shutil.copy2(pdf_path, published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reviewed, literature = load_panel(root, out_dir)
    reviewed_csv = out_dir / "professor_candidate_gpt_review_brief.csv"
    keep_cols = [
        "reportRank",
        "direction",
        "directionLabelZh",
        "reviewRankInDirection",
        "drug",
        "target",
        "protein",
        "proteinName",
        "gptCategoryZh",
        "gptRecommendationZh",
        "gptMeaningZh",
        "gptWhyDoZh",
        "gptWhyNotZh",
        "gptFirstExperimentZh",
        "gptMainConfounderZh",
        "gptGoNoGoZh",
        "gptExpertQuestionZh",
        "gptFirstRoundBucketZh",
        "directionScore",
        "affinityScore",
        "diffdock",
        "noveltyClass",
        "expertNoveltyGroup",
    ]
    reviewed[[c for c in keep_cols if c in reviewed.columns]].to_csv(reviewed_csv, index=False)

    html_text = build_html(reviewed, literature)
    md_text = build_markdown(reviewed, literature)
    html_path = out_dir / "PROFESSOR_CANDIDATE_GPT_REVIEW_BRIEF.html"
    md_path = out_dir / "PROFESSOR_CANDIDATE_GPT_REVIEW_BRIEF.md"
    pdf_path = out_dir / "PROFESSOR_CANDIDATE_GPT_REVIEW_BRIEF.pdf"
    html_path.write_text(html_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    published = publish(root, pdf_path)

    summary = {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidateRows": int(len(reviewed)),
        "gptCategoryCounts": {
            str(k): int(v) for k, v in reviewed["gptCategoryZh"].value_counts().to_dict().items()
        },
        "outputs": {
            "html": str(html_path.relative_to(root)),
            "markdown": str(md_path.relative_to(root)),
            "pdf": str(pdf_path.relative_to(root)),
            "reviewCsv": str(reviewed_csv.relative_to(root)),
            "publishedPdf": str(published.relative_to(root)),
        },
    }
    summary_path = out_dir / "professor_candidate_gpt_review_brief_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a detailed wet-lab candidate review report.

The report is intentionally conservative: model evidence is described as
computational prioritization, while literature checks are used as automated
de-duplication signals that still require expert/manual confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = Path("outputs/sota_validation/wetlab_candidate_detailed_review")
TOP12 = Path("outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate_top12.csv")
FULL_GATE = Path("outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate.csv")
EXECUTION_PROTOCOL = Path(
    "outputs/sota_validation/wetlab_validation_package/wetlab_experiment_execution_protocol_12.csv"
)


TARGET_SYNONYMS: dict[str, list[str]] = {
    "KIT": ["KIT", "c-KIT", "CD117", "mast/stem cell growth factor receptor"],
    "EGFR": ["EGFR", "epidermal growth factor receptor"],
    "ADORA1": ["ADORA1", "adenosine receptor A1", "adenosine A1 receptor"],
    "F2R": [
        "F2R",
        "PAR1",
        "proteinase-activated receptor 1",
        "protease-activated receptor 1",
    ],
    "OXTR": ["OXTR", "oxytocin receptor"],
    "CRHR1": ["CRHR1", "corticotropin-releasing hormone receptor 1", "CRF receptor 1"],
    "EDNRA": ["EDNRA", "endothelin receptor type A", "endothelin-1 receptor"],
    "GLP1R": ["GLP1R", "glucagon-like peptide-1 receptor", "GLP-1 receptor"],
    "ADRA1A": ["ADRA1A", "alpha-1A adrenergic receptor", "alpha 1A adrenergic receptor"],
    "CHRM3": ["CHRM3", "muscarinic acetylcholine receptor M3", "M3 muscarinic receptor"],
}


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def text(value: Any, default: str = "-") -> str:
    if is_blank(value):
        return default
    s = " ".join(str(value).split())
    return s if s else default


def num(value: Any, digits: int = 3) -> str:
    if is_blank(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return text(value)


def int_text(value: Any) -> str:
    if is_blank(value):
        return "-"
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return text(value)


def tex_escape(value: Any) -> str:
    s = text(value, "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def base_drug_name(drug: str) -> str:
    base = re.sub(
        r"\b(hydrochloride|hydrochrloride|phosphate|bromide|succinate|mesylate|maleate|"
        r"dimaleate|dihydrochloride|acetate|sodium|potassium|calcium)\b",
        "",
        drug,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", base).strip()


def disease_query(direction: str) -> str:
    queries = {
        "oncology": "cancer OR tumor OR tumour OR neoplasm OR oncology",
        "infectious_disease": (
            "infection OR infectious OR viral OR virus OR HIV OR hepatitis OR influenza "
            "OR tuberculosis OR pneumonia OR bacterial OR fungal OR malaria OR pathogen "
            "OR antimicrobial OR antiviral"
        ),
        "cardiovascular": "cardiovascular OR vascular OR heart OR hypertension OR cardiology",
        "neurology_psychiatry": (
            "neurology OR psychiatric OR psychiatry OR depression OR anxiety OR Alzheimer OR nervous"
        ),
        "immunology_inflammation": (
            "immunology OR immune OR inflammation OR inflammatory OR autoimmune OR arthritis "
            "OR cytokine OR allergy OR asthma"
        ),
    }
    return queries.get(direction, "disease OR clinical OR therapy")


def pubmed_term(row: pd.Series, include_disease: bool) -> str:
    drug_names = list(dict.fromkeys([text(row.get("drug"), ""), base_drug_name(text(row.get("drug"), ""))]))
    drug_names = [name for name in drug_names if name]
    target_names = TARGET_SYNONYMS.get(text(row.get("target"), ""), [text(row.get("target"), "")])
    drug_part = " OR ".join(f'"{name}"[All Fields]' for name in drug_names)
    target_part = " OR ".join(f'"{name}"[All Fields]' for name in target_names)
    term = f"({drug_part}) AND ({target_part})"
    if include_disease:
        term += f" AND ({disease_query(text(row.get('direction'), ''))})"
    return term


def clinical_trials_term(row: pd.Series) -> str:
    drug = base_drug_name(text(row.get("drug"), ""))
    return f'"{drug}" ({disease_query(text(row.get("direction"), ""))})'


def pubmed_query(session: requests.Session, term: str) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 5,
        "sort": "relevance",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    api_url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        resp = session.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json().get("esearchresult", {})
        return {
            "count": int(data.get("count", 0)),
            "ids": data.get("idlist", [])[:5],
            "url": api_url,
            "ok": True,
        }
    except Exception as exc:
        return {"count": None, "ids": [], "url": api_url, "ok": False, "error": str(exc)}


def clinical_trials_query(session: requests.Session, term: str) -> dict[str, Any]:
    params = {
        "query.term": term,
        "pageSize": 5,
        "countTotal": "true",
        "format": "json",
    }
    url = "https://clinicaltrials.gov/api/v2/studies"
    api_url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        resp = session.get(url, params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        ncts: list[str] = []
        for study in data.get("studies", [])[:5]:
            ident = study.get("protocolSection", {}).get("identificationModule", {})
            nct = ident.get("nctId")
            if nct:
                ncts.append(nct)
        return {
            "count": int(data.get("totalCount", 0)),
            "ids": ncts,
            "url": api_url,
            "ok": True,
        }
    except Exception as exc:
        return {"count": None, "ids": [], "url": api_url, "ok": False, "error": str(exc)}


def literature_status(row: pd.Series, pair_count: int | None, pair_disease_count: int | None) -> dict[str, str]:
    drug = text(row.get("drug"), "")
    target = text(row.get("target"), "")
    known = int(float(row.get("knownDrugTargetPair", 0) or 0)) == 1
    role = text(row.get("validationRole"), "")

    if known or role == "positive_control":
        return {
            "status": "已知机制/阳性对照",
            "action": "不作为新机制发现宣传；建议保留为实验体系校准、阳性对照和召回能力证明。",
            "priority": "发现价值低，但实验价值高；若经费紧张，可以只保留一个同类阳性对照。",
        }

    if drug == "Gefitinib" and target == "KIT":
        return {
            "status": "已有较多公开共现线索",
            "action": "不建议作为纯新发现表述；应标注为机制延展/二线验证。只有在实验能排除 EGFR carryover 并证明 KIT engagement 时才值得推进。",
            "priority": "可做，但排序应低于未见直接组合文献的新靶点候选；也可替换为 backup 候选以提高新颖性。",
        }

    pair_count_value = pair_count if pair_count is not None else -1
    pair_disease_value = pair_disease_count if pair_disease_count is not None else -1
    if pair_count_value >= 20:
        return {
            "status": "直接组合文献线索较多",
            "action": "进入专家排重；若人工确认已有直接 drug-target-disease 机制验证，则降级为复核或忽略。",
            "priority": "保留为机制延展候选，暂不作为最高新颖性项目。",
        }
    if pair_count_value >= 1 or pair_disease_value >= 1:
        return {
            "status": "有少量文献线索",
            "action": "保留候选，但购买前需逐篇确认是否已完成直接靶点验证或只是同领域背景共现。",
            "priority": "中等新颖性；适合与更干净的新组合并行验证。",
        }
    if pair_count_value == 0:
        return {
            "status": "未见明显直接组合文献",
            "action": "可作为新用途/新靶点候选推进；购买前仍需 PubMed、ClinicalTrials 和专利/中文文献人工复核。",
            "priority": "新颖性较高；优先要求正交靶点 engagement 和 counterscreen 支撑。",
        }
    return {
        "status": "在线检索未完成",
        "action": "暂按本地 novelty 标签处理；购买前必须人工检索复核。",
        "priority": "待补充。",
    }


def build_literature_audit(root: Path, top12: pd.DataFrame, out_dir: Path, refresh: bool) -> pd.DataFrame:
    cache_path = out_dir / "wetlab_candidate_detailed_review_literature_cache.json"
    out_csv = out_dir / "wetlab_candidate_detailed_review_literature_audit.csv"
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cached = {}

    session = requests.Session()
    records: list[dict[str, Any]] = []
    changed = False
    for _, row in top12.iterrows():
        key = f"{text(row.get('drug'))}__{text(row.get('target'))}"
        if key in cached and not refresh:
            record = cached[key]
        else:
            direct_term = pubmed_term(row, include_disease=False)
            disease_term = pubmed_term(row, include_disease=True)
            ct_term = clinical_trials_term(row)
            direct = pubmed_query(session, direct_term)
            time.sleep(0.34)
            direct_disease = pubmed_query(session, disease_term)
            time.sleep(0.34)
            clinical = clinical_trials_query(session, ct_term)
            time.sleep(0.2)
            record = {
                "drug": text(row.get("drug")),
                "target": text(row.get("target")),
                "pubmedDirectPairQuery": direct_term,
                "pubmedDirectPairCount": direct.get("count"),
                "pubmedDirectPairPmids": ";".join(direct.get("ids", [])),
                "pubmedDirectPairUrl": direct.get("url"),
                "pubmedDiseasePairQuery": disease_term,
                "pubmedDiseasePairCount": direct_disease.get("count"),
                "pubmedDiseasePairPmids": ";".join(direct_disease.get("ids", [])),
                "pubmedDiseasePairUrl": direct_disease.get("url"),
                "clinicalTrialsDrugDiseaseQuery": ct_term,
                "clinicalTrialsDrugDiseaseCount": clinical.get("count"),
                "clinicalTrialsDrugDiseaseNcts": ";".join(clinical.get("ids", [])),
                "clinicalTrialsDrugDiseaseUrl": clinical.get("url"),
                "queriedUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            cached[key] = record
            changed = True

        status = literature_status(
            row,
            record.get("pubmedDirectPairCount"),
            record.get("pubmedDiseasePairCount"),
        )
        record = dict(record)
        record.update(
            {
                "literatureStatusZh": status["status"],
                "literatureActionZh": status["action"],
                "literaturePriorityZh": status["priority"],
            }
        )
        records.append(record)

    if changed or refresh or not cache_path.exists():
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = pd.DataFrame(records)
    audit.to_csv(out_csv, index=False)
    return audit


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top12 = pd.read_csv(root / TOP12).sort_values("finalGateRank")
    full_gate = pd.read_csv(root / FULL_GATE)
    protocol = pd.read_csv(root / EXECUTION_PROTOCOL)
    return top12, full_gate, protocol


def protocol_for(protocol: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
    mask = (protocol["drug"] == row.get("drug")) & (protocol["target"] == row.get("target"))
    if protocol[mask].empty:
        return {}
    return protocol[mask].iloc[0].to_dict()


def evidence_line(row: pd.Series) -> str:
    return (
        f"亲和分 {num(row.get('affinityScore'))}；Open Targets {num(row.get('openTargetsScore'))}；"
        f"TxGNN {num(row.get('txgnnScore'))}；KG 证据分 {num(row.get('kgEvidenceScore'), 1)}；"
        f"CMap/CREEDS {text(row.get('cmapReversalTier'))}，raw {num(row.get('cmapBestRawReversal'))}；"
        f"通路 {text(row.get('pathwayDiseaseContextTier'))}；组织/模型 {text(row.get('gtexContextTier'))}/"
        f"{text(row.get('tissueContextTier'))}；DepMap {text(row.get('depmapDependencyTier'))}；"
        f"结构 {text(row.get('structureConfidenceTier'))}，pose {text(row.get('poseAuditStatus'))}，DiffDock {num(row.get('diffdock'), 2)}。"
    )


def feasibility_line(row: pd.Series, proto: dict[str, Any]) -> str:
    primary = text(proto.get("primaryReadout") or row.get("primaryAssay"))
    orthogonal = text(proto.get("orthogonalValidation") or row.get("orthogonalAssay"))
    counter = text(proto.get("counterScreenDetail") or row.get("counterScreen"))
    model = text(proto.get("modelExamples") or row.get("diseaseModelRecommendation"))
    return (
        f"首选实验为 {text(row.get('assayModality'))}。Primary readout：{primary} "
        f"正交验证：{orthogonal} Counter-screen：{counter} 推荐模型：{model}"
    )


def recommendation_line(row: pd.Series, lit: pd.Series) -> str:
    role = text(row.get("validationRoleZh"))
    tier = text(row.get("finalGateTier"))
    decision = text(row.get("finalGateRecommendationZh"))
    lit_priority = text(lit.get("literaturePriorityZh"))
    if "阳性对照" in role:
        return f"{decision} 该候选属于 {role}，在报告中应明确为体系校准而非新发现。{lit_priority}"
    if "Gefitinib" == text(row.get("drug")) and text(row.get("target")) == "KIT":
        return f"{decision} 但需标注为已有较多公开线索的机制延展候选；如果专家更看重全新性，可后移或由 backup 替换。"
    return f"{decision} {lit_priority}"


def table_row(cells: list[Any]) -> str:
    return " & ".join(tex_escape(cell) for cell in cells) + r" \\"


def build_backup_table(full_gate: pd.DataFrame) -> str:
    backup = full_gate[full_gate["finalGateTier"].astype(str).str.contains("backup", na=False)].copy()
    if backup.empty:
        return "未形成 backup 候选。"
    backup = backup.sort_values("finalGateScore", ascending=False).head(11)
    rows = [
        table_row(
            [
                i,
                f"{row.get('drug')} - {row.get('target')}",
                row.get("directionLabelZh"),
                row.get("validationRoleZh"),
                num(row.get("finalGateScore"), 1),
                row.get("automaticGateSummaryZh"),
            ]
        )
        for i, (_, row) in enumerate(backup.iterrows(), start=1)
    ]
    return (
        r"{\small\setlength{\tabcolsep}{3pt}"
        "\n"
        r"\begin{longtable}{p{0.04\linewidth}p{0.20\linewidth}p{0.08\linewidth}p{0.12\linewidth}p{0.06\linewidth}p{0.36\linewidth}}"
        "\n"
        r"\toprule 序号 & 候选 & 方向 & 角色 & 分数 & 作为替补的理由 \\"
        "\n"
        r"\midrule"
        "\n"
        + "\n".join(rows)
        + "\n"
        r"\bottomrule"
        "\n"
        r"\end{longtable}"
        "\n"
        r"}"
    )


def build_tex(
    root: Path,
    top12: pd.DataFrame,
    full_gate: pd.DataFrame,
    protocol: pd.DataFrame,
    literature: pd.DataFrame,
    out_dir: Path,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lit_lookup = {
        (row["drug"], row["target"]): row
        for _, row in literature.iterrows()
    }

    role_counts = top12["validationRoleZh"].value_counts().to_dict()
    direction_counts = top12["directionLabelZh"].value_counts().to_dict()

    summary_rows = []
    for _, row in top12.iterrows():
        lit = lit_lookup[(row["drug"], row["target"])]
        summary_rows.append(
            table_row(
                [
                    int(row.get("finalGateRank")),
                    f"{row.get('drug')} - {row.get('target')}",
                    row.get("directionLabelZh"),
                    row.get("validationRoleZh"),
                    num(row.get("finalGateScore"), 1),
                    text(lit.get("literatureStatusZh")),
                    text(lit.get("literatureActionZh")),
                ]
            )
        )

    sections: list[str] = []
    for _, row in top12.iterrows():
        lit = lit_lookup[(row["drug"], row["target"])]
        proto = protocol_for(protocol, row)
        known_label = "已知" if int(float(row.get("knownDrugTargetPair", 0) or 0)) == 1 else "未标记为已知"
        section = rf"""
\subsection{{Rank {int(row.get("finalGateRank"))}: {tex_escape(row.get("drug"))} -- {tex_escape(row.get("target"))}}}
\begin{{tabularx}}{{\linewidth}}{{>{{\bfseries}}p{{0.19\linewidth}}X}}
候选身份 & {tex_escape(row.get("drug"))} / {tex_escape(row.get("target"))} / {tex_escape(row.get("protein"))}，{tex_escape(row.get("proteinName"))} \\
疾病方向与角色 & {tex_escape(row.get("directionLabelZh"))}；{tex_escape(row.get("validationRoleZh"))}；final gate: {tex_escape(row.get("finalGateTier"))}；综合分 {tex_escape(num(row.get("finalGateScore"), 1))} \\
已知性标签 & 本地 knownDrugTargetPair = {tex_escape(known_label)}；noveltyClass = {tex_escape(row.get("noveltyClass"))}；reviewTrack = {tex_escape(row.get("reviewTrack"))} \\
在线排重信号 & PubMed drug-target 共现 {tex_escape(int_text(lit.get("pubmedDirectPairCount")))} 篇；drug-target-disease 共现 {tex_escape(int_text(lit.get("pubmedDiseasePairCount")))} 篇；ClinicalTrials 药物-疾病方向 {tex_escape(int_text(lit.get("clinicalTrialsDrugDiseaseCount")))} 项 \\
排重结论 & {tex_escape(lit.get("literatureStatusZh"))}；{tex_escape(lit.get("literatureActionZh"))} \\
\end{{tabularx}}

\noindent\textbf{{推荐处理。}} {tex_escape(recommendation_line(row, lit))}

\noindent\textbf{{入选理由。}} {tex_escape(row.get("rationaleZh"))}

\noindent\textbf{{证据拆解。}} {tex_escape(evidence_line(row))}

\noindent\textbf{{机制可解释性。}} {tex_escape(text(proto.get("targetBiology") or ""))} {tex_escape(text(row.get("mechanismDirectionCheck")))}

\noindent\textbf{{实验可行性。}} {tex_escape(feasibility_line(row, proto))}

\noindent\textbf{{剂量与推进标准。}} {tex_escape(text(proto.get("doseResponsePlan") or row.get("concentrationPlan")))} {tex_escape(text(proto.get("goNoGoCriteria") or row.get("advanceCriteria")))}

\noindent\textbf{{停止/降级条件。}} {tex_escape(text(proto.get("stopCriteriaShort") or row.get("stopCriteria")))}

\noindent\textbf{{主要风险。}} {tex_escape(text(row.get("mainAssayRisk")))} ADMET tier {tex_escape(row.get("admetTier"))}；模型风险标记：{tex_escape(row.get("mlAdmetRiskFlags"))}。{tex_escape(text(proto.get("mainConfounder") or ""))}

\noindent\textbf{{购买前必须确认。}} {tex_escape(row.get("manualGateSummaryZh"))}

\noindent\textbf{{专家讨论问题。}} {tex_escape(text(proto.get("expertQuestion") or "该候选能否在非毒性浓度下产生可复现、靶点特异且方向一致的 readout，并能被正交验证排除 assay interference？"))}
"""
        sections.append(section)

    backup_table = build_backup_table(full_gate)
    md_path = out_dir / "WETLAB_CANDIDATE_DETAILED_REVIEW.md"
    csv_path = out_dir / "wetlab_candidate_detailed_review_literature_audit.csv"

    tex = rf"""
\documentclass[UTF8,11pt,a4paper]{{ctexart}}
\usepackage{{geometry}}
\geometry{{left=18mm,right=18mm,top=17mm,bottom=18mm}}
\usepackage{{fontspec}}
\usepackage{{tabularx,longtable,booktabs,array}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{microtype}}

\setmainfont{{Noto Serif CJK SC}}
\setsansfont{{Noto Sans CJK SC}}
\setmonofont{{Noto Sans Mono}}
\setCJKmainfont{{Noto Serif CJK SC}}
\setCJKsansfont{{Noto Sans CJK SC}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}}
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{BioMaster 首轮湿实验候选详细评审}}
\rhead{{{tex_escape(generated)}}}
\cfoot{{\thepage}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.25}}
\sloppy
\emergencystretch=3em

\begin{{document}}

\begin{{center}}
{{\LARGE\bfseries BioMaster 首轮湿实验候选详细评审报告}}\\[4pt]
{{\large 逐项理由、可行性、风险与已有研究排重}}\\[6pt]
生成时间：{tex_escape(generated)}
\end{{center}}

\section{{报告范围与结论}}
本报告面向进入首轮湿实验或最终人工复核的 12 个优先候选。候选来自当前 BioMaster 多证据排序与 final pre-experiment gate：亲和筛选、疾病证据、TxGNN/KG、Reactome/GO/CREEDS、CMap/LINCS 反转、GTEx/HPA/DepMap、结构对接、ADMET 与实验可执行性共同参与筛选。

需要强调：这里的结论是计算优先级与实验设计建议，不是生物学验证结论。DiffDock 数值用于姿态和口袋审计，不等同于结合自由能；PubMed/ClinicalTrials 自动检索用于提示排重，不能替代专家逐篇核查、专利检索和具体适应症审阅。

\begin{{itemize}}
\item 进入详细评审的候选：12 个；其中 {tex_escape(json.dumps(role_counts, ensure_ascii=False))}。
\item 疾病方向覆盖：{tex_escape(json.dumps(direction_counts, ensure_ascii=False))}。
\item 明确已知机制/阳性对照：3 个，主要用于实验体系校准和召回能力展示，不作为新发现候选宣传。
\item 自动检索提示已有较多公开线索：Gefitinib--KIT，应标注为机制延展或二线验证；若专家更重视新颖性，可由 backup 候选替换。
\item 新颖性较高且值得优先讨论的候选包括 Fludarabine--ADORA1、Palonosetron--F2R、Methylnaltrexone--OXTR/CRHR1、Donepezil--EDNRA、Amisulpride--ADRA1A 等，但这些候选必须通过正交靶点 engagement、counterscreen 与非毒性剂量窗口验证。
\end{{itemize}}

\section{{筛选与排重口径}}
筛选逻辑不是单一亲和排序，而是先在 druggable proteome 内完成药物-蛋白组合评分，再按疾病方向证据、机制路径、签名反转、组织/模型背景、结构可解释性、ADMET 风险和实验可执行性收敛。最终 gate 不直接决定购买，购买前仍需要确认化合物形态、纯度、供应商、溶解度、临床可达游离暴露、目标蛋白在实验模型中的表达、调控方向、正交实验和 counterscreen 可用性。

已有研究排重采用两层口径：第一层为本地 FDA/ChEMBL 已知 drug-target 标签；第二层为在线 PubMed 和 ClinicalTrials 自动查询。PubMed 的 drug-target 共现用于提示是否已有直接组合线索，ClinicalTrials 的药物-疾病方向计数仅说明该药在该领域是否常见，不等同于该靶点机制已经被验证。

\section{{候选总览}}
{{\small\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{p{{0.04\linewidth}}p{{0.19\linewidth}}p{{0.07\linewidth}}p{{0.11\linewidth}}p{{0.06\linewidth}}p{{0.13\linewidth}}p{{0.30\linewidth}}}}
\toprule
Rank & 候选 & 方向 & 角色 & 分数 & 已有研究信号 & 处理建议 \\
\midrule
{chr(10).join(summary_rows)}
\bottomrule
\end{{longtable}}
}}

\section{{逐项候选评审}}
{chr(10).join(sections)}

\section{{替补候选索引}}
以下候选未写成逐项实验计划，但可在阳性对照过多、已有研究过强、采购不可行或模型表达不合适时替换进入专家审阅。

{backup_table}

\section{{统一的 Go/No-Go 原则}}
\begin{{itemize}}
\item 只有在非毒性浓度下出现可复现剂量反应，且 primary target readout 与正交 target engagement/pathway readout 方向一致时，才进入下一轮。
\item 对 receptor 类候选，必须区分 agonist、antagonist 或间接调节；对 kinase/enzyme 类候选，必须加入同家族/相关通路 counterscreen。
\item 对已知机制或已有大量文献线索的候选，只能作为阳性对照、体系校准或机制延展，不应包装为新发现。
\item 对 CMap/CREEDS 支持但靶点 engagement 不明确的候选，不应仅凭疾病表型推进，必须先证明靶点相关性。
\item 对 ADMET 风险标记较高或临床暴露明显不可达的候选，应先做低成本体外验证，不建议直接进入复杂疾病模型。
\end{{itemize}}

\section{{数据源}}
\begin{{itemize}}
\item 候选与模型证据：final gate Top12 CSV 与完整 gate CSV。
\item 同目录产物：文献排重 CSV、Markdown 源和本 PDF。
\item 在线检索源：\href{{https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi}}{{NCBI PubMed ESearch API}}；\href{{https://clinicaltrials.gov/api/v2/studies}}{{ClinicalTrials.gov API v2}}。查询日期为 2026-06-05。
\end{{itemize}}

\end{{document}}
"""
    return tex


def build_markdown(
    top12: pd.DataFrame,
    full_gate: pd.DataFrame,
    protocol: pd.DataFrame,
    literature: pd.DataFrame,
    out_dir: Path,
) -> str:
    lit_lookup = {(row["drug"], row["target"]): row for _, row in literature.iterrows()}
    lines = [
        "# BioMaster 首轮湿实验候选详细评审报告",
        "",
        "本 Markdown 与 PDF 内容一致，用于后续人工编辑。",
        "",
        "## 候选总览",
        "",
        "| Rank | 候选 | 方向 | 角色 | 分数 | 已有研究信号 | 处理建议 |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for _, row in top12.iterrows():
        lit = lit_lookup[(row["drug"], row["target"])]
        lines.append(
            f"| {int(row['finalGateRank'])} | {row['drug']} - {row['target']} | {row['directionLabelZh']} | "
            f"{row['validationRoleZh']} | {num(row['finalGateScore'], 1)} | {lit['literatureStatusZh']} | "
            f"{lit['literatureActionZh']} |"
        )

    lines.append("\n## 逐项候选评审\n")
    for _, row in top12.iterrows():
        lit = lit_lookup[(row["drug"], row["target"])]
        proto = protocol_for(protocol, row)
        lines.extend(
            [
                f"### Rank {int(row['finalGateRank'])}: {row['drug']} - {row['target']}",
                "",
                f"- 疾病方向与角色：{row['directionLabelZh']}；{row['validationRoleZh']}；综合分 {num(row['finalGateScore'], 1)}。",
                f"- 在线排重：PubMed drug-target {int_text(lit.get('pubmedDirectPairCount'))}；"
                f"drug-target-disease {int_text(lit.get('pubmedDiseasePairCount'))}；"
                f"ClinicalTrials 药物-疾病方向 {int_text(lit.get('clinicalTrialsDrugDiseaseCount'))}。",
                f"- 排重结论：{lit['literatureStatusZh']}；{lit['literatureActionZh']}",
                f"- 推荐处理：{recommendation_line(row, lit)}",
                f"- 入选理由：{text(row.get('rationaleZh'))}",
                f"- 证据拆解：{evidence_line(row)}",
                f"- 机制可解释性：{text(proto.get('targetBiology') or '')} {text(row.get('mechanismDirectionCheck'))}",
                f"- 实验可行性：{feasibility_line(row, proto)}",
                f"- 主要风险：{text(row.get('mainAssayRisk'))} ADMET tier {text(row.get('admetTier'))}；"
                f"模型风险标记：{text(row.get('mlAdmetRiskFlags'))}。",
                f"- 购买前必须确认：{text(row.get('manualGateSummaryZh'))}",
                "",
            ]
        )

    backup = full_gate[full_gate["finalGateTier"].astype(str).str.contains("backup", na=False)].copy()
    backup = backup.sort_values("finalGateScore", ascending=False).head(11)
    lines.extend(["## 替补候选索引", "", "| 序号 | 候选 | 方向 | 角色 | 分数 |", "|---:|---|---|---|---:|"])
    for i, (_, row) in enumerate(backup.iterrows(), start=1):
        lines.append(
            f"| {i} | {row['drug']} - {row['target']} | {row['directionLabelZh']} | "
            f"{row['validationRoleZh']} | {num(row['finalGateScore'], 1)} |"
        )
    return "\n".join(lines) + "\n"


def compile_pdf(tex_path: Path) -> Path:
    cmd = [
        "lualatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    for _ in range(2):
        subprocess.run(cmd, cwd=tex_path.parent, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return tex_path.with_suffix(".pdf")


def publish_docs_assets(root: Path, pdf_path: Path | None) -> dict[str, str]:
    published: dict[str, str] = {}
    if pdf_path is None or not pdf_path.exists():
        return published
    docs_assets = root / "docs/assets"
    docs_assets.mkdir(parents=True, exist_ok=True)
    published_pdf = docs_assets / "wetlab-candidate-detailed-review.pdf"
    shutil.copy2(pdf_path, published_pdf)
    published["publishedPdf"] = str(published_pdf.relative_to(root))
    return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--refresh-literature", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    top12, full_gate, protocol = load_inputs(root)
    literature = build_literature_audit(root, top12, out_dir, refresh=args.refresh_literature)

    tex_text = build_tex(root, top12, full_gate, protocol, literature, out_dir)
    md_text = build_markdown(top12, full_gate, protocol, literature, out_dir)

    tex_path = out_dir / "WETLAB_CANDIDATE_DETAILED_REVIEW.tex"
    md_path = out_dir / "WETLAB_CANDIDATE_DETAILED_REVIEW.md"
    tex_path.write_text(tex_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")

    pdf_path = None
    if not args.no_pdf:
        pdf_path = compile_pdf(tex_path)
    published_outputs = publish_docs_assets(root, pdf_path)

    summary = {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidateRows": int(len(top12)),
        "literatureAuditRows": int(len(literature)),
        "outputs": {
            "tex": str(tex_path.relative_to(root)),
            "markdown": str(md_path.relative_to(root)),
            "literatureAuditCsv": str((out_dir / "wetlab_candidate_detailed_review_literature_audit.csv").relative_to(root)),
            "pdf": str(pdf_path.relative_to(root)) if pdf_path else None,
            **published_outputs,
        },
    }
    summary_path = out_dir / "wetlab_candidate_detailed_review_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

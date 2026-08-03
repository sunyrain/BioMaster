#!/usr/bin/env python3
"""Build the advisor-facing explanation of Final1000 feasibility grading."""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fitz
import numpy as np
import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "outputs/current_production_package_v2/final500_full_deep_review_v7"
INPUT_CSV = PACKAGE_DIR / "FINAL1000_FULL_DEEP_REVIEWED_V7.csv"
REVIEW_CONTRACT = (
    ROOT
    / "outputs/current_production_package_v2/final500_reviewed_v6/deep_review_625_v6/"
    "stage1_top500/AGENT_REVIEW_INSTRUCTIONS_ZH.md"
)
DRUG_METADATA = ROOT / "data/processed/drug_library_pubchem_chembl_mapped.metadata.json"
ACTIVE_MOIETY_METADATA = ROOT / "data/processed/drug_library_active_moiety_v4.summary.json"
CONPLEX_METADATA = ROOT / "outputs/full_conplex_active_moiety_v4/druggable_proteome_conplex_prep.metadata.json"
ALPHAFOLD_METADATA = ROOT / "data/processed/alphafold_receptor_manifest.metadata.json"
P2RANK_SUMMARY = ROOT / "outputs/p2rank26_pocket_audit_afcomplete_v1/p2rank_pocketability_summary.json"
PURESNET_SUMMARY = ROOT / "outputs/puresnet_gpu_pocket_audit_v1/puresnet_gpu_consensus_summary.json"
BOLTZ_RUN_PLAN = (
    ROOT
    / "outputs/current_production_package_v2/full_untruncated_universe_v4/"
    "boltz_full_run_v4_seeded/run_plan.json"
)
FORMAL_SUMMARY = (
    ROOT
    / "outputs/current_production_package_v2/formal_full_universe_v4/"
    "formal_completion_summary_v4_complete.json"
)
OUT_DIR = PACKAGE_DIR / "feasibility_grading_explainer"
OUTPUT_CSV = OUT_DIR / "FINAL1000_EXPERIMENTAL_FEASIBILITY_GRADE_AUDIT.csv"
OUTPUT_JSON = OUT_DIR / "FINAL1000_EXPERIMENTAL_FEASIBILITY_GRADE_SUMMARY_V3.json"
OUTPUT_PDF = OUT_DIR / "FINAL1000_EXPERIMENTAL_FEASIBILITY_GRADING_METHOD_AND_DATA_SOURCES_ZH_V3.pdf"


GRADE_LABELS = {
    "A": "完整可执行",
    "B": "可测但有一项重要不确定性",
    "C": "存在主要障碍，需先解锁",
    "D": "硬停止：矛盾、不可实现或非特异风险主导",
}

EXECUTION_LABELS = {
    "T1_immediate_standard_assay": "T1 标准平台可立即启动",
    "T2_standard_assay_major_uncertainty": "T2 标准平台存在主要不确定性",
    "T3_specialized_assay_or_active_species_resolution": "T3 专门平台或活性实体待解决",
}

OPERATIONAL_LABELS = {
    "X0_hard_stop": "X0 硬停止",
    "C0_validated_control": "C0 已验证对照",
    "E1_standard_assay_ready": "E1 标准实验优先",
    "E2_standard_assay_major_blocker": "E2 标准实验但先解锁主要障碍",
    "E3_specialized_or_active_species_hold": "E3 专门实验或活性实体处理",
}


def esc(value: Any) -> str:
    return html.escape(str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: int, total: int) -> str:
    return f"{100 * value / total:.1f}%"


def table(headers: list[str], rows: list[list[Any]], css_class: str = "") -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def metric(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        "</div>"
    )


def operational_status(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                frame["feasibility_grade_v6"].eq("D")
                | frame["agent_literature_class"].eq("contradictory"),
                frame["agent_literature_class"].eq("exact_pair_validated"),
                frame["experimental_execution_tier_v6"].eq("T1_immediate_standard_assay"),
                frame["experimental_execution_tier_v6"].eq(
                    "T2_standard_assay_major_uncertainty"
                ),
            ],
            [
                "X0_hard_stop",
                "C0_validated_control",
                "E1_standard_assay_ready",
                "E2_standard_assay_major_blocker",
            ],
            default="E3_specialized_or_active_species_hold",
        ),
        index=frame.index,
    )


def source_urls(value: Any) -> list[str]:
    return [url.rstrip(".,)") for url in re.findall(r"https?://[^;\s]+", str(value))]


def source_domains(value: Any) -> list[str]:
    domains = {
        urlparse(url).netloc.lower().removeprefix("www.")
        for url in source_urls(value)
        if urlparse(url).netloc
    }
    return sorted(domains)


def example_row(frame: pd.DataFrame, drug: str, gene: str) -> pd.Series:
    rows = frame.loc[
        frame["drug_names"].astype(str).str.contains(drug, case=False, regex=False)
        & frame["primary_gene"].eq(gene)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one example for {drug}-{gene}, observed {len(rows)}")
    return rows.iloc[0]


def example_card(row: pd.Series, title: str, interpretation: str) -> str:
    return f"""
    <div class="example-card">
      <div class="example-title">{esc(title)} · {esc(row['drug_names'])} → {esc(row['primary_gene'])}</div>
      <div class="tag-row">
        <span>{esc(GRADE_LABELS[row['feasibility_grade_v6']])}</span>
        <span>{esc(EXECUTION_LABELS[row['experimental_execution_tier_v6']])}</span>
        <span>{esc(row['target_assay_family'])}</span>
      </div>
      <p><b>暴露：</b>{esc(row['exposure_feasibility_v6'])}</p>
      <p><b>实验：</b>{esc(row['assay_plan_v6'])}</p>
      <p><b>主要风险：</b>{esc(row['key_risks_v6'])}</p>
      <div class="interpretation"><b>为何这样分：</b>{esc(interpretation)}</div>
    </div>
    """


def build_outputs() -> dict[str, Any]:
    frame = pd.read_csv(INPUT_CSV, low_memory=False).fillna("").copy()
    drug_metadata = load_json(DRUG_METADATA)
    active_metadata = load_json(ACTIVE_MOIETY_METADATA)
    conplex_metadata = load_json(CONPLEX_METADATA)
    alphafold_metadata = load_json(ALPHAFOLD_METADATA)
    p2rank_summary = load_json(P2RANK_SUMMARY)
    puresnet_summary = load_json(PURESNET_SUMMARY)
    boltz_run_plan = load_json(BOLTZ_RUN_PLAN)
    formal_summary = load_json(FORMAL_SUMMARY)
    if len(frame) != 1000 or frame["pair_id"].nunique() != 1000:
        raise ValueError("Final1000 must contain exactly 1000 unique pairs")
    if not frame["feasibility_grade_v6"].eq(frame["agent_feasibility_grade"]).all():
        raise ValueError("Final feasibility grade differs from the unified pair-review grade")
    if frame["feasibility_grade_v6"].astype(str).str.strip().eq("").any():
        raise ValueError("Blank feasibility grade found")

    frame["operational_experiment_status"] = operational_status(frame)
    frame["实验可行性档中文"] = frame["feasibility_grade_v6"].map(GRADE_LABELS)
    frame["实验执行层中文"] = frame["experimental_execution_tier_v6"].map(EXECUTION_LABELS)
    frame["汇报用实验状态中文"] = frame["operational_experiment_status"].map(OPERATIONAL_LABELS)
    frame["source_url_count"] = frame["agent_sources"].map(lambda value: len(source_urls(value)))
    frame["source_domains"] = frame["agent_sources"].map(
        lambda value: ";".join(source_domains(value))
    )

    audit_columns = [
        "v5_rank",
        "pair_id",
        "drug_names",
        "primary_gene",
        "target_assay_family",
        "feasibility_grade_v6",
        "实验可行性档中文",
        "experimental_execution_tier_v6",
        "实验执行层中文",
        "operational_experiment_status",
        "汇报用实验状态中文",
        "v5_strength_tier",
        "rank_within_drug",
        "boltz_support_tier_refined",
        "pose_stability_tier",
        "strict_structure_tier",
        "active_species_status_v6",
        "agent_literature_class",
        "review_confidence_v6",
        "exposure_feasibility_v6",
        "assay_plan_v6",
        "key_risks_v6",
        "review_verdict_v6",
        "source_url_count",
        "source_domains",
        "agent_sources",
    ]
    frame[audit_columns].to_csv(OUTPUT_CSV, index=False)

    grade_counts = frame["feasibility_grade_v6"].value_counts().to_dict()
    execution_counts = frame["experimental_execution_tier_v6"].value_counts().to_dict()
    operational_counts = frame["operational_experiment_status"].value_counts().to_dict()
    family_grade = pd.crosstab(frame["feasibility_grade_v6"], frame["target_assay_family"])
    grade_execution = pd.crosstab(
        frame["feasibility_grade_v6"], frame["experimental_execution_tier_v6"]
    )
    active_grade = pd.crosstab(frame["feasibility_grade_v6"], frame["active_species_status_v6"])
    important_domains = [
        "dailymed.nlm.nih.gov",
        "accessdata.fda.gov",
        "ebi.ac.uk",
        "pubmed.ncbi.nlm.nih.gov",
        "eutils.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "uniprot.org",
        "pubchem.ncbi.nlm.nih.gov",
        "guidetopharmacology.org",
    ]
    domain_row_coverage = {
        domain: int(frame["source_domains"].str.split(";").map(lambda values: domain in values).sum())
        for domain in important_domains
    }

    summary = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_rows": int(len(frame)),
        "input_unique_pairs": int(frame["pair_id"].nunique()),
        "final_grade_equals_unified_review_grade": bool(
            frame["feasibility_grade_v6"].eq(frame["agent_feasibility_grade"]).all()
        ),
        "grade_counts": {key: int(value) for key, value in grade_counts.items()},
        "execution_tier_counts_raw": {
            key: int(value) for key, value in execution_counts.items()
        },
        "operational_reporting_counts": {
            key: int(value) for key, value in operational_counts.items()
        },
        "resolved_active_species": int(
            frame["active_species_status_v6"]
            .isin(["parent_drug_relevant", "salt_normalization_adequate"])
            .sum()
        ),
        "standard_assay_family_rows": int(
            frame["target_assay_family"]
            .isin(["enzyme", "kinase", "nuclear_epigenetic"])
            .sum()
        ),
        "contradictory_rows": int(frame["agent_literature_class"].eq("contradictory").sum()),
        "exact_pair_validated_rows": int(
            frame["agent_literature_class"].eq("exact_pair_validated").sum()
        ),
        "source_url_total": int(frame["source_url_count"].sum()),
        "rows_with_resolvable_url": int(frame["source_url_count"].gt(0).sum()),
        "source_domain_row_coverage": domain_row_coverage,
        "source_lineage_metrics": {
            "drug_rows": int(drug_metadata["rows"]),
            "drug_pubchem_cid_rows": int(drug_metadata["nonempty_pubchem_cid"]),
            "drug_chembl_structure_rows": int(drug_metadata["nonempty_sdf_path"]),
            "active_moiety_unique_model_ligands": int(active_metadata["unique_model_ligands"]),
            "active_moiety_rows_changed": int(active_metadata["rows_changed_from_canonical"]),
            "chembl_workbook_protein_rows": int(conplex_metadata["protein_rows_input_valid"]),
            "chembl_workbook_genes": int(conplex_metadata["unique_gene_symbols"]),
            "chembl_workbook_unique_sequences": int(conplex_metadata["unique_sequences"]),
            "conplex_full_pairs": int(conplex_metadata["conplex_unique_sequence_pairs"]),
            "alphafold_structures_ready": int(alphafold_metadata["proteins_with_pdb_path"]),
            "p2rank_targets_completed": int(p2rank_summary["targets_completed_by_p2rank"]),
            "puresnet_targets_completed": int(puresnet_summary["puresnet_targets_run"]),
            "boltz_version": boltz_run_plan["modelEnvironment"]["boltz"],
            "boltz_top3000_completed": int(formal_summary["refined_completed_rows"]),
        },
        "input_sha256": sha256(INPUT_CSV),
        "review_contract_sha256": sha256(REVIEW_CONTRACT),
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    examples = {
        "A": example_row(frame, "Oxaprozin", "SLC22A8"),
        "B": example_row(frame, "Tovorafenib", "ENPP2"),
        "C": example_row(frame, "Rilpivirine", "PTK6"),
        "D": example_row(frame, "Loteprednol", "PDE4D"),
    }
    lineage = summary["source_lineage_metrics"]
    source_coverage = summary["source_domain_row_coverage"]

    grade_rows = [
        [grade, GRADE_LABELS[grade], grade_counts.get(grade, 0), pct(grade_counts.get(grade, 0), 1000)]
        for grade in ["A", "B", "C", "D"]
    ]
    execution_rows = [
        [
            label,
            operational_counts.get(code, 0),
            pct(operational_counts.get(code, 0), 1000),
        ]
        for code, label in OPERATIONAL_LABELS.items()
    ]
    cross_rows = []
    for grade in ["A", "B", "C", "D"]:
        cross_rows.append(
            [
                grade,
                int(grade_execution.loc[grade].get("T1_immediate_standard_assay", 0)),
                int(grade_execution.loc[grade].get("T2_standard_assay_major_uncertainty", 0)),
                int(
                    grade_execution.loc[grade].get(
                        "T3_specialized_assay_or_active_species_resolution", 0
                    )
                ),
            ]
        )
    family_rows = [
        [
            family,
            *[int(family_grade.get(family, pd.Series(dtype=int)).get(grade, 0)) for grade in ["A", "B", "C", "D"]],
            int((frame["target_assay_family"] == family).sum()),
        ]
        for family in ["enzyme", "kinase", "nuclear_epigenetic", "transporter", "ion_channel"]
    ]

    css = """
    @font-face { font-family: NotoSC; src: url('file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'); font-weight: 400; }
    @font-face { font-family: NotoSC; src: url('file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'); font-weight: 700; }
    @font-face { font-family: NotoSerifSC; src: url('file:///usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'); font-weight: 700; }
    @page { size: A4; margin: 15mm 15mm 16mm; @top-left { content: 'FINAL1000 实验可行性分级'; font: 8pt NotoSC; color: #66726f; } @top-right { content: string(section); font: 8pt NotoSC; color: #66726f; } @bottom-left { content: '统一口径逐对综合审阅 · 2026-07-15'; font: 7.5pt NotoSC; color: #7c8583; } @bottom-right { content: counter(page) ' / ' counter(pages); font: 8pt NotoSC; color: #56615f; } }
    @page cover { margin: 0; @top-left { content: none; } @top-right { content: none; } @bottom-left { content: none; } @bottom-right { content: none; } }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: NotoSC, sans-serif; font-size: 9.4pt; line-height: 1.55; color: #172421; }
    .cover { page: cover; height: 297mm; padding: 25mm 20mm; background: #f7faf9; position: relative; }
    .cover::before { content: ''; position: absolute; left: 0; top: 0; width: 8mm; height: 297mm; background: #16766f; }
    .kicker { color: #16766f; font-weight: 700; font-size: 11pt; margin-bottom: 15mm; }
    h1 { font-family: NotoSerifSC, serif; font-size: 29pt; line-height: 1.3; margin: 0; }
    .cover .lead { margin-top: 10mm; max-width: 155mm; font-size: 12pt; line-height: 1.8; color: #42504d; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5mm; margin-top: 18mm; }
    .metric { border-top: 1.2mm solid #16766f; padding-top: 4mm; }
    .metric-label { font-size: 8.5pt; color: #65716e; }
    .metric-value { font-size: 21pt; font-weight: 700; line-height: 1.2; }
    .metric-note { font-size: 7.7pt; color: #707b78; margin-top: 1.5mm; }
    .cover-note { position: absolute; left: 20mm; right: 20mm; bottom: 25mm; border-left: 1.4mm solid #d19b35; padding: 4mm 5mm; background: #fbf8f0; color: #494a42; }
    .section { page-break-before: always; }
    h2 { string-set: section content(text); font-family: NotoSerifSC, serif; font-size: 20pt; line-height: 1.25; margin: 0 0 6mm; padding-bottom: 3mm; border-bottom: .7mm solid #16766f; }
    h3 { font-size: 12.5pt; color: #1f5b55; margin: 6mm 0 3mm; }
    p { margin: 0 0 3mm; }
    .lead { font-size: 11pt; line-height: 1.75; color: #34423f; }
    .callout { border-left: 1.4mm solid #d19b35; background: #fbf8f0; padding: 4mm 5mm; margin: 5mm 0; }
    .callout.teal { border-left-color: #16766f; background: #f0f7f6; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; }
    .axis { border: .3mm solid #d9dfdd; padding: 4mm; break-inside: avoid; }
    .axis b { color: #1f5b55; }
    table { width: 100%; border-collapse: collapse; margin: 4mm 0; font-size: 8.5pt; }
    th { background: #eaf3f1; color: #184c47; text-align: left; padding: 2.5mm; border-bottom: .5mm solid #84a9a4; }
    td { padding: 2.3mm 2.5mm; border-bottom: .25mm solid #dce2e0; vertical-align: top; overflow-wrap: anywhere; }
    .compact td, .compact th { padding: 1.8mm 2mm; }
    .rule { display: grid; grid-template-columns: 14mm 1fr; gap: 4mm; padding: 4mm 0; border-bottom: .3mm solid #dce2e0; break-inside: avoid; }
    .grade { width: 11mm; height: 11mm; line-height: 11mm; text-align: center; background: #16766f; color: white; font-size: 14pt; font-weight: 700; }
    .example-card { border: .35mm solid #d5ddda; padding: 4.5mm; margin-bottom: 5mm; break-inside: avoid; }
    .example-title { font-size: 12pt; color: #1c5751; font-weight: 700; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 2mm; margin: 2mm 0 3mm; }
    .tag-row span { border: .25mm solid #91aca8; color: #315f5a; padding: .8mm 2mm; font-size: 7.8pt; }
    .interpretation { margin-top: 3mm; padding: 3mm; background: #f1f7f6; }
    .small { font-size: 8pt; color: #697471; }
    .source-note { font-size: 8.3pt; color: #4e5d5a; background: #f6f8f7; padding: 3mm 4mm; margin-top: 3mm; overflow-wrap: anywhere; }
    ul { margin: 2mm 0; padding-left: 6mm; }
    li { margin-bottom: 2mm; }
    """

    html_text = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{css}</style></head><body>
    <section class="cover">
      <div class="kicker">FDA OLD DRUG · NEW TARGET DISCOVERY</div>
      <h1>FINAL1000<br>实验可行性分级方法</h1>
      <div class="lead">说明每一条药物–新靶点假说如何按同一口径判断物理假说、活性实体、人体/体外暴露、实验可执行性和反证风险，并区分“候选整体可行性”与“实验平台复杂度”。</div>
      <div class="metrics">
        {metric('统一审阅', '1,000', '每行一药物–靶点假说')}
        {metric('A / B', str(grade_counts.get('A', 0) + grade_counts.get('B', 0)), '可直接或带一项重要不确定性')}
        {metric('C', str(grade_counts.get('C', 0)), '需先解决主要障碍')}
        {metric('D', str(grade_counts.get('D', 0)), '硬停止，不进入发现实验')}
      </div>
      <div class="cover-note"><b>核心口径：</b>A/B/C/D 不是结合概率，也不是疾病疗效分；它回答“在现有证据和可用实验条件下，这个精确 pair 是否值得、能否被干净地验证”。最终分级不是机械加权分，而是门槛式、最严重问题主导的综合判断。</div>
    </section>

    <section class="section">
      <h2>1. 分级对象与边界</h2>
      <p class="lead">分级单位是一个精确的 <b>FDA 活性实体 × 人源蛋白靶点</b> 假说。1000 条全部使用同一套字段、同一证据标准和同一 A/B/C/D 定义。</p>
      <div class="callout teal"><b>分级不回答三件事：</b>不证明真实结合；不确定激动、抑制或调节方向；不证明某一疾病中的疗效。它只决定下一步验证的成熟度和阻断因素。</div>
      <h3>进入分级前已具备的共同物理底座</h3>
      <ul>
        <li>ConPLEx 药物内折叠排名全部位于 Top100，本表实际最大名次为 {int(pd.to_numeric(frame['rank_within_drug']).max())}。</li>
        <li>1000 条均有 Boltz-2 精修结果和双构象位姿稳定性 A/B；均位于严格重叠口袋结构层。</li>
        <li>物理强度相对已知阳性校准：A {int((frame['v5_strength_tier'].str.startswith('A_')).sum())}、B {int((frame['v5_strength_tier'].str.startswith('B_')).sum())}、C {int((frame['v5_strength_tier'].str.startswith('C_')).sum())}。</li>
      </ul>
      <p>因此，实验可行性分级不是再次重复 ConPLEx/Boltz 排名，而是在“已有计算假说”的基础上追问：结构是否对应正确活性实体、体内是否可能覆盖、是否有清晰 assay、结果能否排除已知药理和非特异效应。</p>
      <h3>输出字段</h3>
      {table(['字段', '用途'], [
          ['feasibility_grade_v6', '最终 A/B/C/D 整体可行性档'],
          ['exposure_feasibility_v6', '人体暴露、游离浓度和体外可实现浓度判断'],
          ['active_species_status_v6', '母体、盐型、活性代谢物是否与计算结构一致'],
          ['assay_plan_v6', 'primary assay、阳性对照、已知靶点反筛和干扰门'],
          ['key_risks_v6', '选择性、毒性、膜干扰、聚集、组织暴露等主要风险'],
          ['review_verdict_v6', '保留、后置或剔除的逐对结论'],
      ])}
    </section>

    <section class="section">
      <h2>2. 五个统一判断维度</h2>
      <div class="grid-2">
        <div class="axis"><b>① 物理假说</b><p>检查 ConPLEx、Boltz-2、双构象稳定性、口袋和已知阳性校准是否相互支持。计算强只表示值得验证，不能单独给 A。</p></div>
        <div class="axis"><b>② 活性实体正确性</b><p>确认母体、盐型和前药/代谢物。若真正体内活性物种不是计算结构，必须先重跑，不能沿用原 pose。</p></div>
        <div class="axis"><b>③ 暴露可达性</b><p>优先比较人体游离 Cmax、组织暴露与待测效价；只有总 Cmax 时必须注明蛋白结合。无候选靶点效价时，只能写“覆盖未知”。</p></div>
        <div class="axis"><b>④ assay 可执行性</b><p>要求明确 primary readout、阳性对照、原靶点/近邻家族反筛、细胞活力及聚集/膜/光学干扰门。</p></div>
        <div class="axis"><b>⑤ 反证与可解释性</b><p>核查精确阴性、错误组织暴露、已知强药理混杂、非特异反应、选择性和安全窗。硬反证优先于计算高分。</p></div>
        <div class="axis"><b>判定方式</b><p>不是五项相加取总分，而是门槛式判断：A 要求全部通过；B 允许一项重要不确定性；C 有主要障碍；D 存在硬停止条件。</p></div>
      </div>
      <div class="callout"><b>为何没有大量 A：</b>在完全新 pair 中，通常缺少真实 Kd/IC50，因而无法把人体游离暴露与效价定量桥接。严格执行这一原则后，A 稀少是正常结果，不代表计算筛选失败。</div>
    </section>

    <section class="section">
      <h2>2.1 物理假说：数据和计算链</h2>
      <p class="lead">这一维度回答：<b>该药物结构与该蛋白是否形成了值得实验验证的直接结合假说</b>。它由原始分子/序列、靶点结构、口袋预测和 pair-level 结合模型共同构成。</p>
      {table(['数据层', '本项目来源', '进入的字段或结果'], [
          ['药物二维结构', f"FDA结构清单 → ChEMBL molecule API / PubChem PUG REST；{lineage['drug_rows']}条结构记录", 'canonical_smiles、model_ligand_smiles'],
          ['蛋白序列', f"druggable_proteome_chembl(1).xlsx + ChEMBL 37 target components；{lineage['chembl_workbook_genes']}基因、{lineage['chembl_workbook_unique_sequences']}唯一序列", 'sequence_key、representative sequence、UniProt'],
          ['蛋白单体结构', f"AlphaFold DB 人源reference proteome v6归档；本地{lineage['alphafold_structures_ready']}条结构可用", 'pdb_path；这是AlphaFold monomer v2结构，不是AF3复合物'],
          ['靶点口袋', f"P2Rank完成{lineage['p2rank_targets_completed']}个项目靶点；PUResNetV2.0完成{lineage['puresnet_targets_completed']}个", '口袋概率、口袋残基、双模型空间重叠、strict_structure_tier'],
          ['序列型DTI', f"本地ConPLEx BindingDB_ExperimentalValidModel.pt；全量{lineage['conplex_full_pairs']:,}个药物结构×唯一序列pair", 'conplex_score、药物内rank、靶点内rank'],
          ['复合物与亲和', f"Boltz-{lineage['boltz_version']}；Top3000完成{lineage['boltz_top3000_completed']}条精修", 'affinity probability/value、confidence、model 0/1复合物CIF'],
      ], 'compact')}
      <h3>如何形成判断</h3>
      <ul>
        <li>ConPLEx输入只有 <b>model_ligand_smiles + 蛋白序列</b>，输出是学习到的相互作用分数；它不是Kd，也没有显式三维pose。</li>
        <li>Boltz-2输入同一活性结构、蛋白序列、AlphaFold模板及非强制P2Rank口袋接触，输出复合物构象和亲和相关量。正式物理分为 <b>70%校准Boltz轴 + 30% ConPLEx轴</b>，这是排序分而非结合概率。</li>
        <li>双构象审计先按口袋Cα对齐，再计算配体对称性校正RMSD、质心漂移和界面残基Jaccard；A/B表示给定口袋条件下两次采样较一致。</li>
        <li>Boltz亲和强度再与96条已知阳性校准集比较，得到达到阳性q10、q25或中位数的相对档位。</li>
      </ul>
      <div class="callout"><b>边界：</b>P2Rank/PUResNet是靶点级先验，同一靶点的所有药物共享，不能证明某个具体药物占据该口袋；Boltz使用口袋接触提示，因此位姿稳定性是“条件重复性”，不是盲对接成功率；ConPLEx与已知数据库可能存在训练知识重叠。</div>
      <div class="source-note">官方方法来源：ConPLEx论文 PMID 37289807；AlphaFold DB https://alphafold.ebi.ac.uk/；P2Rank PMCID PMC6091426；PUResNetV2.0 PMCID PMC11157904；Boltz官方仓库 https://github.com/jwohlwend/boltz 。</div>
    </section>

    <section class="section">
      <h2>2.2 活性实体：结构到底代表什么</h2>
      <p class="lead">这一维度回答：<b>模型中计算的结构，是否就是人体和湿实验中真正应当测试的化学实体</b>。盐型归一与前药/代谢物判断是两件不同的事。</p>
      {table(['步骤', '数据来源', '处理与字段'], [
          ['原始药物身份', 'FDA小分子清单中的通用名、商品名、给药途径、原适应症和ChEMBL ID', 'drug_names、fda_route、fda_indication、drug_chembl_id'],
          ['结构补全', f"ChEMBL molecule molfile覆盖{lineage['drug_chembl_structure_rows']}条；PubChem CID覆盖{lineage['drug_pubchem_cid_rows']}条", 'canonical_smiles、InChIKey、分子式、分子量'],
          ['计算结构标准化', f"RDKit canonicalization + 最大片段 + Uncharger；{lineage['active_moiety_rows_changed']}条表示发生变化，得到{lineage['active_moiety_unique_model_ligands']}个唯一model ligand", 'active_moiety_smiles、model_ligand_smiles'],
          ['盐/父子层级核查', 'ChEMBL molecule hierarchy API查询parent及盐型/child IDs，1000条查询均成功', 'chembl_molecule_ids_queried、chembl_hierarchy_query_ok'],
          ['体内活性物种核查', 'FDA/DailyMed标签的代谢与PK段、PubMed药代研究、逐对来源', 'active_species_status_v6、review_verdict_v6'],
      ], 'compact')}
      <h3>四类状态如何解释</h3>
      {table(['状态', '数量', '含义'], [
          ['parent_drug_relevant', int((frame['active_species_status_v6']=='parent_drug_relevant').sum()), '上市母体就是主要测试对象'],
          ['salt_normalization_adequate', int((frame['active_species_status_v6']=='salt_normalization_adequate').sum()), '去掉盐/对离子后仍代表同一药理母体'],
          ['active_species_uncertain', int((frame['active_species_status_v6']=='active_species_uncertain').sum()), '母体与代谢物贡献不清，需要并行测定或补查'],
          ['prodrug_active_metabolite_requires_rerun', int((frame['active_species_status_v6']=='prodrug_active_metabolite_requires_rerun').sum()), '真正活性物种不同，现有ConPLEx/Boltz结果不能直接沿用'],
      ])}
      <div class="callout"><b>关键区别：</b>RDKit“最大有机片段+中和”只解决盐、溶剂化物和对离子造成的输入错位，不等于识别体内活性代谢物。例如前药的活性代谢物具有不同共价结构时，必须以代谢物重新计算并重新设计暴露比较。</div>
      <div class="source-note">本地原始文件：FDA_approved_small_molecules_2005_2026_with_structures.xlsx；结构主表：data/processed/drug_library_pubchem_chembl_mapped.csv；统一模型结构：data/processed/drug_library_active_moiety_v4.csv；ChEMBL API文档：https://www.ebi.ac.uk/chembl/api/data/docs 。</div>
    </section>

    <section class="section">
      <h2>2.3 暴露可达性：标签数据如何变成判断</h2>
      <p class="lead">这一维度回答：<b>即使体外能够结合，临床给药后正确活性实体是否有机会在目标组织达到足够游离浓度</b>。这是“计算命中”到“可转化互作”之间最容易断裂的一步。</p>
      {table(['来源', '覆盖行数', '主要提供的信息'], [
          ['DailyMed药品标签', source_coverage.get('dailymed.nlm.nih.gov', 0), 'Cmax/AUC、蛋白结合、代谢、给药途径、警告和药物相互作用'],
          ['FDA AccessData标签', source_coverage.get('accessdata.fda.gov', 0), '批准标签PDF和正式临床药理信息'],
          ['PubMed/PMC/NCBI文献', source_coverage.get('pubmed.ncbi.nlm.nih.gov', 0) + source_coverage.get('pmc.ncbi.nlm.nih.gov', 0), '组织/CSF暴露、体外效价、转运、代谢物和专门PK研究'],
          ['ChEMBL/EBI', source_coverage.get('ebi.ac.uk', 0), '候选靶点实验效价或精确阴性记录'],
          ['PubChem/化学记录', source_coverage.get('pubchem.ncbi.nlm.nih.gov', 0), '分子量与结构身份，用于浓度换算和实体核对'],
      ])}
      <h3>逐对判断顺序</h3>
      <ol>
        <li>把标签中的质量浓度按正确分子量换算为摩尔浓度，明确是母体还是盐、总药还是游离药。</li>
        <li>若有血浆蛋白结合率，近似使用 <b>Cmax,free = Cmax,total × fu</b>；高蛋白结合药不能用总Cmax直接与IC50比较。</li>
        <li>若有该精确靶点Kd/Ki/IC50，比较游离暴露与效价；若没有，只能定义待测浓度梯度，不能声称“临床可覆盖”。</li>
        <li>中枢、眼部、肠腔、肾小管或细胞内靶点需单独看组织/局部暴露，血浆总浓度不能替代。</li>
        <li>同时检查给药途径、毒性和原药理是否允许达到候选靶点所需浓度。</li>
      </ol>
      <div class="callout"><b>当前数据形态：</b>`exposure_feasibility_v6` 是带来源的逐对叙述性判断，不是统一PK数据库。1000条均有可解析来源，合计{summary['source_url_total']}个URL；但来源是行级引用，尚未把每一个Cmax、fu和组织浓度拆成独立数值字段。因此它适合A/B/C/D分层，不适合直接训练定量模型。</div>
      <div class="source-note">正式暴露字段：exposure_feasibility_v6；逐对来源：agent_sources。后续最有价值的结构化增强是新增 total_Cmax_uM、fu、free_Cmax_uM、tissue_exposure、pair_potency_uM 和 free_exposure_margin 六列。</div>
    </section>

    <section class="section">
      <h2>2.4 Assay可执行性：实验方案从哪里来</h2>
      <p class="lead">这一维度回答：<b>能否用一个对该靶点合适、带阳性对照和反筛的实验，干净地区分真实互作、原药理和非特异干扰</b>。</p>
      {table(['信息', '来源', '用途'], [
          ['靶点实验类型', 'ChEMBL-MoA靶点类别 + ChEMBL 37 protein class + Open Targets 26.06 tractability', '将靶点分为enzyme、kinase、nuclear/epigenetic、transporter、ion channel'],
          ['靶点功能与底物', 'UniProt、NCBI Gene、ChEMBL assay、IUPHAR及原始文献', '确定蛋白构建体、底物/配体、辅因子和primary readout'],
          ['阳性对照', 'ChEMBL高质量活性、已批准配体或公认工具化合物', '确认assay动态范围和靶点系统工作正常'],
          ['已知靶点反筛', 'FDA原MoA/target/action type和ChEMBL已知药理', '判断信号是否只是原上市机制'],
          ['家族与非特异反筛', '靶点近邻、RDKit PAINS/NIH/Brenk、药物物化性质', '检查同工型选择性、聚集、膜扰动、光学干扰和细胞毒性'],
      ], 'compact')}
      <h3>1000条按实验类型分布</h3>
      {table(['类型', '数量', '首轮primary readout', '必要反筛'], [
          ['enzyme', int((frame['target_assay_family']=='enzyme').sum()), '重组人蛋白酶活/底物转化；再做正交结合或占靶', 'FDA原靶点、同工酶、聚集/读数干扰'],
          ['kinase', int((frame['target_assay_family']=='kinase').sum()), '生化激酶活 + NanoBRET/CETSA等细胞占靶', 'ATP依赖、kinome/近邻激酶、原靶点'],
          ['nuclear_epigenetic', int((frame['target_assay_family']=='nuclear_epigenetic').sum()), '配体/辅因子结合、催化或报告基因 + 正交结合', '同家族受体、转录继发效应、细胞毒性'],
          ['transporter', int((frame['target_assay_family']=='transporter').sum()), '人源转运/摄取/外排功能或竞争结合', '空载细胞、近邻转运体、膜完整性'],
          ['ion_channel', int((frame['target_assay_family']=='ion_channel').sum()), '自动/手工膜片钳或通量实验', '状态依赖、近邻通道、SCN5A/hERG及膜扰动'],
      ], 'compact')}
      <div class="callout"><b>“有实验方案”不等于“实验已就绪”：</b>`assay_plan_v6` 给出方法学入口、阳性对照和反筛逻辑，但未统一核实供应商、蛋白构建体、底物批次、仪器容量、成本和实验室现有SOP。T1/T2/T3正是用来表达这部分执行复杂度。</div>
      <div class="source-note">Open Targets tractability用于判断靶点是否有小分子先例、配体或结构口袋，不用作药物-疾病疗效证据。官方说明：https://platform-docs.opentargets.org/target/tractability 。</div>
    </section>

    <section class="section">
      <h2>2.5 反证与可解释性：什么会推翻高分</h2>
      <p class="lead">这一维度回答：<b>当前信号是否已经被精确实验否定，或者即使出现信号也无法排除已知药理和非特异效应</b>。它是最差项优先的安全门。</p>
      {table(['审计通道', '本项目具体做法', '对应字段'], [
          ['ChEMBL精确pair', '按活性母体层级查询molecule parent/盐/child × 精确target ChEMBL ID，并读取activity和assay metadata', 'chembl_exact_activity_status、pChEMBL、document IDs'],
          ['严格binding判据', 'assay type=B、pChEMBL≥5、confidence≥8、relation="="、standard flag成立、无variant、无validity comment', 'exact_binding_activity_pchembl_ge_5；其余强记录进入manual review'],
          ['PubMed精确检索', '药物名/别名 × 基因/蛋白名检索2000–2026，并区分直接测量、功能、计算和仅共现', 'pubmed_screen_tier_v6、PMID、标题、DOI'],
          ['统一逐对文献判断', '回看精确活性实体、物种、靶点、assay类型和效价，不把同家族/通路/疾病共现当直接结合', 'agent_literature_class、literature_judgment_v6'],
          ['化合物干扰', 'RDKit PAINS、Brenk、NIH、形式电荷、金属、分子量、重原子数；并结合标签安全性', 'assay_interference_review、severe_compound_liability、key_risks_v6'],
          ['已知药理混杂', 'FDA原靶点、action type、同家族和近邻靶点作为counterscreen', 'family_or_rediscovery_risk、assay_plan_v6'],
      ], 'compact')}
      <h3>当前精确关系研判</h3>
      {table(['类别', '数量', '如何使用'], [
          ['no_exact_report_found', int((frame['agent_literature_class']=='no_exact_report_found').sum()), '只表示当前检索未找到，不等于已证明全新'],
          ['indirect_or_family_only', int((frame['agent_literature_class']=='indirect_or_family_only').sum()), '可作机制背景，不能作精确结合证据'],
          ['functional_only', int((frame['agent_literature_class']=='functional_only').sum()), '有精确功能信号，但需直接结合/占靶正交验证'],
          ['exact_pair_validated', int((frame['agent_literature_class']=='exact_pair_validated').sum()), '移入阳性对照或再发现，不包装为新靶点'],
          ['contradictory', int((frame['agent_literature_class']=='contradictory').sum()), '精确阴性或可靠矛盾，优先于模型高分'],
      ])}
      <div class="callout"><b>为什么反证能推翻高物理分：</b>模型回答“像不像可能结合”，精确实验回答“在指定活性实体、物种、构建体和浓度下是否发生”。例如已有30 µM精确阴性时，计算高分不能把它重新定义为高优先级；最多只能在确认原实验不适用后重新设计。</div>
      <div class="source-note">1000条ChEMBL activity、molecule hierarchy、assay metadata和PubMed自动查询均成功；1000条均有逐对来源。ChEMBL官方API：https://www.ebi.ac.uk/chembl/api/data/docs 。自动共现只用于找文献，最终类别来自精确pair语境核查。</div>
    </section>

    <section class="section">
      <h2>3. A/B/C/D 的门槛逻辑</h2>
      <div class="rule"><div class="grade">A</div><div><b>完整可执行。</b>物理假说充分；活性实体正确；已有定量暴露与功能/效价依据支持可覆盖；primary assay 和反筛明确；无明显反证。A 仍不是“已证实结合”。</div></div>
      <div class="rule"><div class="grade">B</div><div><b>总体可测，但有一项重要不确定性。</b>例如无候选靶点效价导致暴露桥接不完整、选择性未知、需要专门膜蛋白平台，或已有功能关系但缺少正交直接结合。</div></div>
      <div class="rule"><div class="grade">C</div><div><b>存在主要障碍，但仍可保留为探索性假说。</b>常见原因是游离暴露很低/未知、作用方向或选择性严重不明、活性实体待确认、实验需要较多开发，或已知主药理会强烈混杂 readout。</div></div>
      <div class="rule"><div class="grade">D</div><div><b>硬停止。</b>精确阴性或矛盾记录、目标组织暴露明显不可实现、前药/代谢物错配且当前结构无效、严重非特异/安全风险主导，或数据库身份无法可靠解析。</div></div>
      <h3>实际结果</h3>
      {table(['档位', '含义', '数量', '比例'], grade_rows)}
      <p class="small">A+B 共 {grade_counts.get('A', 0) + grade_counts.get('B', 0)} 条（{pct(grade_counts.get('A', 0) + grade_counts.get('B', 0), 1000)}）；非 D 共 {1000 - grade_counts.get('D', 0)} 条。D 级即使存在现成酶学平台，也不应进入发现实验。</p>
    </section>

    <section class="section">
      <h2>4. 可行性档与执行层不能混用</h2>
      <p class="lead"><b>A/B/C/D</b> 评价“这个 pair 是否值得且能被可靠验证”；<b>T1/T2/T3</b> 评价“需要什么实验平台和准备工作”。两者是正交的。</p>
      {table(['可行性档', '原始 T1', '原始 T2', '原始 T3'], cross_rows, 'compact')}
      <div class="callout"><b>关键例子：</b>唯一 A 级 Oxaprozin–SLC22A8 仍属于 T3，因为转运体需要专门转运实验；反之，91 条 D 级候选在原始映射中落入 T2，只表示靶点有标准实验平台，不表示这些 pair 值得做。</div>
      <h3>导师汇报采用的操作顺序</h3>
      <ol>
        <li>先按 A/B/C/D 和反证执行硬门：D 或明确 contradictory → X0 停止。</li>
        <li>精确 pair 已验证者单列 C0 对照，不包装为新发现。</li>
        <li>其余候选再按平台分为 E1 标准实验优先、E2 先解决主要障碍、E3 专门平台/活性实体处理。</li>
      </ol>
      {table(['汇报用状态', '数量', '比例'], execution_rows)}
    </section>

    <section class="section">
      <h2>5. 分布与一致性检查</h2>
      <h3>不同靶点实验类型中的可行性档</h3>
      {table(['实验类型', 'A', 'B', 'C', 'D', '合计'], family_rows, 'compact')}
      <div class="grid-2">
        <div class="axis"><b>活性实体</b><p>已解决 {summary['resolved_active_species']} 条；其中母体正确 {int((frame['active_species_status_v6']=='parent_drug_relevant').sum())}，盐型归一充分 {int((frame['active_species_status_v6']=='salt_normalization_adequate').sum())}。另有活性实体不确定 {int((frame['active_species_status_v6']=='active_species_uncertain').sum())}、需以活性代谢物重跑 {int((frame['active_species_status_v6']=='prodrug_active_metabolite_requires_rerun').sum())}。</p></div>
        <div class="axis"><b>文献关系</b><p>未发现精确报告 {int((frame['agent_literature_class']=='no_exact_report_found').sum())}；间接/同家族 {int((frame['agent_literature_class']=='indirect_or_family_only').sum())}；功能关系 {int((frame['agent_literature_class']=='functional_only').sum())}；精确已验证 {summary['exact_pair_validated_rows']}；矛盾/反证 {summary['contradictory_rows']}。</p></div>
      </div>
      <h3>质量控制</h3>
      <ul>
        <li>1000/1000 条 `feasibility_grade_v6` 与统一逐对审阅字段完全一致，无空值、无自动兜底。</li>
        <li>每条均要求可解析来源、暴露判断、活性实体状态、实验方案、主要风险和一句话结论。</li>
        <li>分级与物理强度并非同义：物理 A 档中仍有 C/D，说明精确反证和暴露问题可以推翻高计算优先级。</li>
      </ul>
    </section>

    <section class="section">
      <h2>6. 逐对示例：A 与 B</h2>
      {example_card(examples['A'], 'A级', '已有直接人 OAT3 功能效价，并以游离 Cmax/IC50 完成定量覆盖桥接；活性母体正确、实验和反筛清楚、无直接反证。因此整体实验可行性为 A。但它是转运体，执行层仍为 T3，且更像 DDI 机制而非已成立的新疾病疗法。')}
      {example_card(examples['B'], 'B级', '物理证据和标准酶学平台均可用，总暴露可覆盖初筛；但高蛋白结合使游离窗口不确定，且 RAF 主药理会混杂细胞表型。整体可以测，但存在一项关键暴露/解释不确定性，因此为 B 和 E1。')}
    </section>

    <section class="section">
      <h2>7. 逐对示例：C 与 D</h2>
      {example_card(examples['C'], 'C级', 'PTK6 生化 assay 可执行，盐型归一也正确；但药物蛋白结合超过 99%，临床游离暴露很低，且没有 PTK6 效价可完成覆盖桥接，激酶多靶点和聚集风险明显。可作为探索性体外假说，但需先解决主要障碍。')}
      {example_card(examples['D'], 'D级', '虽然 PDE4D 有标准酶学平台且计算物理分高，但已有精确人 PDE4D 实验显示至 30 µM 无显著活性，同时系统暴露未知且存在代谢与高浓度聚集问题。精确反证优先于计算分，故硬停止。')}
      <p class="small">数据源：FINAL1000_FULL_DEEP_REVIEWED_V7.csv；输入 SHA256：{summary['input_sha256']}。逐对审阅合同 SHA256：{summary['review_contract_sha256']}。</p>
    </section>
    </body></html>"""

    HTML(string=html_text, base_url=str(ROOT)).write_pdf(OUTPUT_PDF)
    document = fitz.open(OUTPUT_PDF)
    page_text = [page.get_text("text") for page in document]
    blank_pages = [index + 1 for index, text in enumerate(page_text) if not text.strip()]
    required = [
        "1000 条全部使用同一套字段",
        "五个统一判断维度",
        "物理假说：数据和计算链",
        "活性实体：结构到底代表什么",
        "暴露可达性：标签数据如何变成判断",
        "Assay可执行性：实验方案从哪里来",
        "反证与可解释性：什么会推翻高分",
        "A/B/C/D 的门槛逻辑",
        "可行性档与执行层不能混用",
        "Oxaprozin",
        "Loteprednol",
    ]
    phrase_checks = {phrase: any(phrase in text for text in page_text) for phrase in required}
    forbidden = ["应如何向导师解释", "建议采用的正式表述"]
    forbidden_checks = {phrase: any(phrase in text for text in page_text) for phrase in forbidden}
    document.close()
    if blank_pages or not all(phrase_checks.values()) or any(forbidden_checks.values()):
        raise RuntimeError(
            "PDF audit failed: "
            f"blank_pages={blank_pages}, phrase_checks={phrase_checks}, "
            f"forbidden_checks={forbidden_checks}"
        )
    summary["output_pdf"] = str(OUTPUT_PDF.relative_to(ROOT))
    summary["output_pdf_sha256"] = sha256(OUTPUT_PDF)
    summary["output_csv"] = str(OUTPUT_CSV.relative_to(ROOT))
    summary["output_csv_sha256"] = sha256(OUTPUT_CSV)
    summary["pdf_page_count"] = len(page_text)
    summary["pdf_blank_pages"] = blank_pages
    summary["pdf_required_phrase_checks"] = phrase_checks
    summary["pdf_forbidden_phrase_checks"] = forbidden_checks
    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_outputs()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

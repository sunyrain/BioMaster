#!/usr/bin/env python3
"""Build target evidence routing and the next-stage compute task plan."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/evidence_routing_compute_plan_20260808_v1"

CURRENT_MASTER = ROOT / "outputs/current_results_classified_20260806_v1/CURRENT_RESULTS_MASTER_14488_CLASSIFIED_V1.csv.gz"
CURRENT_SUMMARY = ROOT / "outputs/current_results_classified_20260806_v1/CURRENT_RESULTS_CLASSIFIED_SUMMARY_V1.json"
BOLTZ_TARGETS = ROOT / "outputs/boltz2_calibration_338_v1/evaluation/BOLTZ2_TARGET_QUALIFICATION_338_V1.csv"
GNINA_DISCOVERY = ROOT / "outputs/gnina_discovery_7511_v1/evaluation/GNINA_DISCOVERY_7511_TARGET_CALIBRATED_EVIDENCE_V1.csv.gz"
RECEPTOR_AUDIT = ROOT / "outputs/strict_receptor_protocol_338_v1/FINAL_RECEPTOR_PROTOCOL_AUDIT_338.csv"
POCKET_ATLAS = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas/KNOWN_POCKET_INSTANCES_ANNOTATED_FULL.csv.gz"
TECHNICAL_INCOMPLETE = ROOT / "outputs/current_results_classified_20260806_v1/TECHNICAL_INCOMPLETE_20_V1.csv"


TARGET_ROUTE_ORDER = [
    "E1_DUAL_REMOTE_QUALIFIED",
    "E2_SINGLE_MODEL_REMOTE_QUALIFIED",
    "E3_LOCAL_ONLY_QUALIFIED",
    "E4_RECEPTOR_PROTOCOL_REPAIR",
    "E5_NO_MODEL_QUALIFICATION",
    "E6_NO_ADEQUATE_HISTORICAL_CONTROL",
]

TARGET_ROUTE_ZH = {
    "E1_DUAL_REMOTE_QUALIFIED": "双模型远程资格",
    "E2_SINGLE_MODEL_REMOTE_QUALIFIED": "单模型远程资格",
    "E3_LOCAL_ONLY_QUALIFIED": "仅局部资格",
    "E4_RECEPTOR_PROTOCOL_REPAIR": "受体协议修复",
    "E5_NO_MODEL_QUALIFICATION": "当前模型未取得资格",
    "E6_NO_ADEQUATE_HISTORICAL_CONTROL": "历史对照不足",
}

TARGET_ROUTE_SCOPE = {
    "E1_DUAL_REMOTE_QUALIFIED": "允许N1/N2/N3；优先检验双模型一致与分歧",
    "E2_SINGLE_MODEL_REMOTE_QUALIFIED": "仅按取得远程资格的模型解释N3；另一模型不得否决",
    "E3_LOCAL_ONLY_QUALIFIED": "只允许N1/N2；N3只能保留为探索信息",
    "E4_RECEPTOR_PROTOCOL_REPAIR": "修复结构、链、状态和redocking前不得晋级pair",
    "E5_NO_MODEL_QUALIFICATION": "停止同协议扩筛；先诊断历史对照与结构状态",
    "E6_NO_ADEQUATE_HISTORICAL_CONTROL": "先补实测正负对照；仅允许限额探索",
}

TARGET_ROUTE_ACTION = {
    "E1_DUAL_REMOTE_QUALIFIED": "多seed Boltz；GNINA受体ensemble；稳定后每靶点最多2条进入MD候选",
    "E2_SINGLE_MODEL_REMOTE_QUALIFIED": "重复取得资格的模型；仅在另一模型也适用时分析分歧",
    "E3_LOCAL_ONLY_QUALIFIED": "复算现有N1/N2候选；不扩张远程化学空间",
    "E4_RECEPTOR_PROTOCOL_REPAIR": "选择替代实验holo结构并redocking；通过后重算当前102条A/B/C",
    "E5_NO_MODEL_QUALIFICATION": "检查标签、阴性、构象和口袋；不追加同协议pair筛选",
    "E6_NO_ADEQUATE_HISTORICAL_CONTROL": "从ChEMBL/BindingDB/PubChem补正负对照；未补齐前不作概率化排序",
}

PAIR_LANE_ORDER = [
    "P1_DUAL_STRUCTURE_SUPPORT",
    "P2_BOLTZ_SUPPORT_GNINA_DISAGREES",
    "P3_GNINA_SUPPORT_BOLTZ_DISAGREES",
    "P4_BOLTZ_ONLY_APPLICABLE",
    "P5_GNINA_ONLY_APPLICABLE",
]

PAIR_LANE_ZH = {
    "P1_DUAL_STRUCTURE_SUPPORT": "Boltz与GNINA共同支持",
    "P2_BOLTZ_SUPPORT_GNINA_DISAGREES": "Boltz支持、GNINA适用但不支持",
    "P3_GNINA_SUPPORT_BOLTZ_DISAGREES": "GNINA支持、Boltz适用但不支持",
    "P4_BOLTZ_ONLY_APPLICABLE": "仅Boltz具有适用资格并支持",
    "P5_GNINA_ONLY_APPLICABLE": "仅GNINA具有适用资格并支持",
}

PAIR_LANE_ACTION = {
    "P1_DUAL_STRUCTURE_SUPPORT": "最高物理优先；做多seed/受体ensemble和pose一致性复核",
    "P2_BOLTZ_SUPPORT_GNINA_DISAGREES": "模型分歧；用Boltz重复与GNINA ensemble判断结构敏感性",
    "P3_GNINA_SUPPORT_BOLTZ_DISAGREES": "模型分歧；用Boltz重复确认是否为单seed低估",
    "P4_BOLTZ_ONLY_APPLICABLE": "先验证Boltz重复稳定性；不得包装成多模型共识",
    "P5_GNINA_ONLY_APPLICABLE": "做GNINA受体ensemble；Boltz分数不具备否决资格",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def n(value: int | float) -> str:
    return f"{int(value):,}"


def esc(value: object) -> str:
    return html.escape(str(value))


def target_route(row: pd.Series) -> str:
    if str(row["final_audit_status"]) != "FROZEN_PASS":
        return "E4_RECEPTOR_PROTOCOL_REPAIR"
    boltz_remote = str(row["boltz_target_qualification"]) == "BOLTZ_REMOTE_QUALIFIED"
    gnina_remote = str(row["gnina_target_qualification"]) == "REMOTE_STRONG"
    boltz_local = str(row["boltz_target_qualification"]) == "BOLTZ_LOCAL_QUALIFIED"
    gnina_local = str(row["gnina_target_qualification"]) == "LOCAL_STRONG"
    if boltz_remote and gnina_remote:
        return "E1_DUAL_REMOTE_QUALIFIED"
    if boltz_remote or gnina_remote:
        return "E2_SINGLE_MODEL_REMOTE_QUALIFIED"
    if boltz_local or gnina_local:
        return "E3_LOCAL_ONLY_QUALIFIED"
    boltz_na = str(row["boltz_target_qualification"]) == "BOLTZ_NOT_EVALUABLE"
    gnina_na = str(row["gnina_target_qualification"]) == "NOT_EVALUABLE"
    if boltz_na and gnina_na:
        return "E6_NO_ADEQUATE_HISTORICAL_CONTROL"
    return "E5_NO_MODEL_QUALIFICATION"


def qualified_model(row: pd.Series) -> str:
    models: list[str] = []
    if str(row["boltz_target_qualification"]) in {"BOLTZ_REMOTE_QUALIFIED", "BOLTZ_LOCAL_QUALIFIED"}:
        models.append("Boltz-2")
    if str(row["gnina_target_qualification"]) in {"REMOTE_STRONG", "LOCAL_STRONG"}:
        models.append("GNINA")
    return "+".join(models) if models else "NONE"


def pair_lane(row: pd.Series) -> str:
    if bool(row["boltz_support"] and row["gnina_support"]):
        return "P1_DUAL_STRUCTURE_SUPPORT"
    if bool(row["boltz_support"] and row["gnina_applicable"]):
        return "P2_BOLTZ_SUPPORT_GNINA_DISAGREES"
    if bool(row["gnina_support"] and row["boltz_applicable"]):
        return "P3_GNINA_SUPPORT_BOLTZ_DISAGREES"
    if bool(row["boltz_support"]):
        return "P4_BOLTZ_ONLY_APPLICABLE"
    return "P5_GNINA_ONLY_APPLICABLE"


def table(headers: list[str], rows: list[list[object]], classes: str = "") -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(item)}</td>" for item in row) + "</tr>" for row in rows
    )
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def metric(label: str, value: str, note: str, tone: str = "teal") -> str:
    return (
        f'<div class="metric {tone}"><div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div><div class="metric-note">{esc(note)}</div></div>'
    )


def build_report(
    summary: dict[str, Any],
    target_routes: pd.DataFrame,
    pair_lanes: pd.DataFrame,
    task_summary: pd.DataFrame,
) -> tuple[str, str]:
    target_counts = summary["target_route_counts"]
    pair_counts = summary["pair_lane_counts"]
    created = summary["created_utc"][:10]

    target_rows = []
    for route in TARGET_ROUTE_ORDER:
        subset = target_routes[target_routes["target_route"].eq(route)]
        target_rows.append([
            TARGET_ROUTE_ZH[route], n(len(subset)), n(subset["strict_boltz_abc_pairs"].sum()),
            n(subset["strict_gnina_abc_pairs"].sum()), TARGET_ROUTE_SCOPE[route],
        ])
    pair_rows = []
    for lane in PAIR_LANE_ORDER:
        subset = pair_lanes[pair_lanes["pair_evidence_lane"].eq(lane)]
        pair_rows.append([
            PAIR_LANE_ZH[lane], n(len(subset)), n(subset["target_chembl_id"].nunique()),
            n(subset["project_entity_ids"].nunique()), PAIR_LANE_ACTION[lane],
        ])
    task_rows = [
        [r.priority, r.task_code, r.task_zh, r.unit, n(r.units), n(r.max_model_runs), r.start_condition]
        for r in task_summary.itertuples(index=False)
    ]
    route_class = pd.crosstab(target_routes["target_route_zh"], target_routes["target_class_zh"])
    class_headers = ["靶点通道"] + list(route_class.columns)
    class_rows = [[idx] + [n(v) for v in route_class.loc[idx].tolist()] for idx in route_class.index]

    css = """
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc); }
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc); font-weight:700; }
    @page { size:A4; margin:14mm 15mm 14mm 15mm;
      @top-left { content:"FDA老药新靶点 · 证据分流与计算安排"; color:#66716f; font:8pt NotoCJK; }
      @top-right { content:string(section); color:#66716f; font:8pt NotoCJK; }
      @bottom-left { content:"BioMaster · 下一阶段计算计划"; color:#7c8583; font:7.5pt NotoCJK; }
      @bottom-right { content:counter(page) " / " counter(pages); color:#7c8583; font:7.5pt NotoCJK; }
    }
    @page:first { @top-left{content:none;} @top-right{content:none;} @bottom-left{content:none;} }
    * { box-sizing:border-box; }
    body { margin:0; font-family:NotoCJK,sans-serif; color:#182422; font-size:9.2pt; line-height:1.5; }
    h1,h2,h3,p { margin-top:0; }
    h1 { font-size:27pt; line-height:1.28; letter-spacing:0; margin-bottom:7mm; }
    h2 { string-set:section content(); font-size:18pt; line-height:1.3; border-bottom:2px solid #147b73; padding-bottom:2.5mm; margin:0 0 5mm; }
    h3 { font-size:11.5pt; margin:4mm 0 2mm; color:#173f3a; }
    p { margin-bottom:3mm; }
    .page { min-height:267mm; page-break-after:always; }
    .page:last-child { page-break-after:auto; }
    .cover { padding-top:24mm; position:relative; }
    .cover:before { content:""; position:absolute; left:0; top:0; width:45mm; height:4mm; background:#c58b2a; }
    .eyebrow { color:#147b73; font-weight:700; font-size:10.5pt; margin-bottom:6mm; }
    .subtitle { font-size:13pt; color:#475653; line-height:1.65; max-width:165mm; }
    .rule { border-top:1px solid #cad4d1; margin:10mm 0 7mm; }
    .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; margin:4mm 0 6mm; }
    .metric { border:1px solid #d4ddda; border-top:3px solid #147b73; padding:3.5mm; min-height:27mm; background:#fbfcfb; }
    .metric.gold { border-top-color:#c58b2a; }
    .metric.blue { border-top-color:#4d7497; }
    .metric.red { border-top-color:#a75b62; }
    .metric-label { color:#66716f; font-size:8pt; }
    .metric-value { font-size:18pt; font-weight:700; color:#173f3a; line-height:1.25; margin:1mm 0; }
    .metric-note { color:#78827f; font-size:7.7pt; }
    .callout { padding:4mm 5mm; background:#f3f6f5; border-left:4px solid #147b73; margin:4mm 0; }
    .callout.gold { border-left-color:#c58b2a; background:#faf7f0; }
    .callout.red { border-left-color:#a75b62; background:#faf3f4; }
    .callout.blue { border-left-color:#4d7497; background:#f2f5f8; }
    table { width:100%; border-collapse:collapse; margin:3mm 0 5mm; font-size:8.1pt; }
    th { background:#213d39; color:#fff; padding:2.2mm; text-align:left; }
    td { border-bottom:1px solid #d9e1de; padding:2.1mm; vertical-align:top; }
    tr:nth-child(even) td { background:#f6f8f7; }
    .compact { font-size:7.6pt; }
    .compact td,.compact th { padding:1.7mm; }
    .two-col { display:grid; grid-template-columns:1fr 1fr; gap:5mm; }
    .card { border:1px solid #d4ddda; padding:4mm; page-break-inside:avoid; }
    .card h3 { margin-top:0; color:#147b73; }
    ul { margin:2mm 0 4mm; padding-left:5mm; }
    li { margin-bottom:1.3mm; }
    .stage { display:grid; grid-template-columns:30mm 1fr; gap:4mm; margin-bottom:3mm; page-break-inside:avoid; }
    .stage strong { background:#213d39; color:#fff; padding:3mm; text-align:center; }
    .stage div { border:1px solid #d5ddda; padding:3mm; }
    .small { color:#6c7774; font-size:7.8pt; }
    .footer-note { margin-top:8mm; color:#78817f; font-size:8pt; }
    """

    pages: list[str] = []
    pages.append(f"""
    <section class="page cover">
      <div class="eyebrow">TARGET ROUTING · EVIDENCE LANES · COMPUTE TASKS</div>
      <h1>FDA老药新靶点<br>证据分流与计算任务安排</h1>
      <p class="subtitle">基于338个冻结靶点、14,488条Boltz条件深算和7,511条GNINA结果，将靶点资格与pair证据分开组织，给出下一阶段可直接执行的计算队列。</p>
      <div class="rule"></div>
      <div class="metrics">
        {metric('靶点分流', '338', '六个互斥证据通道')}
        {metric('物理证据并集', n(summary['pair_union']['pairs']), f"覆盖{n(summary['pair_union']['targets'])}个靶点", 'blue')}
        {metric('立即Boltz复算', n(summary['compute_budget']['boltz_multiseed_runs']), '2个新增seed', 'gold')}
      </div>
      <div class="callout gold"><strong>核心原则：</strong>模型只有在对应靶点和新颖性范围内通过历史校准后才具有晋级资格；模型失败、结构失败和没有历史对照是三种不同状态。</div>
      <p class="footer-note">版本：EVIDENCE_ROUTING_COMPUTE_PLAN_20260808_V1 · 生成日期：{esc(created)}</p>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>1. 当前计算结果</h2>
      <div class="metrics">
        {metric('Boltz完成', '14,468 / 14,488', '99.86%；20条quinidine失败')}
        {metric('Boltz A/B/C', '587', '82个靶点；包含需清理和复核项', 'blue')}
        {metric('严格Boltz物理池', '466', '333 A/B核心 + 133 C储备', 'gold')}
      </div>
      <p>当前没有运行中的Boltz、GNINA、DrugCLIP或ConPLEx任务，4张RTX 4090均为空闲。全空间DTA和条件结构计算已经完成，后续重点应转为稳定性复算、模型分歧分析和结构协议修复。</p>
      <div class="two-col">
        <div class="card"><h3>已有可靠资产</h3><ul>
          <li>683个FDA药物实体、720个模型结构。</li><li>338个靶点的实验结构与口袋协议。</li>
          <li>243,360条ConPLEx/DrugCLIP全量结果。</li><li>GNINA与Boltz同靶点正负校准。</li>
          <li>333条严格A/B与133条C级储备。</li>
        </ul></div>
        <div class="card"><h3>当前不能直接做</h3><ul>
          <li>不能把所有338个靶点继续用同一模型扩筛。</li><li>不能把模型未通过与无历史对照混为一类。</li>
          <li>不能把适用域外分数用于晋级或淘汰。</li><li>不能直接从587条扩成1000条同等级候选。</li>
          <li>不能在稳定性复算前全面启动MD。</li>
        </ul></div>
      </div>
      <div class="callout blue"><strong>阶段判断：</strong>我们现在缺的不是另一轮全量排名，而是对不同证据状态使用不同计算问题。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>2. 338个靶点的六级分流</h2>
      {table(['靶点通道','靶点','Boltz严格Pair','GNINA严格Pair','允许范围与原因'], target_rows, 'compact')}
      <h3>靶点类型分布</h3>
      {table(class_headers, class_rows, 'compact')}
      <div class="callout gold"><strong>最关键的区分：</strong>E5表示模型在历史正负对照上没有取得资格；E6表示缺乏足够历史对照，尚不能评价模型。E6不能被当成“模型失败”。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>3. 每类靶点应该做什么</h2>
      <div class="stage"><strong>E1 · {n(target_counts['E1_DUAL_REMOTE_QUALIFIED'])}</strong><div><b>双模型远程资格</b><br>{esc(TARGET_ROUTE_ACTION['E1_DUAL_REMOTE_QUALIFIED'])}。这是最适合检验低相似发现的靶点层。</div></div>
      <div class="stage"><strong>E2 · {n(target_counts['E2_SINGLE_MODEL_REMOTE_QUALIFIED'])}</strong><div><b>单模型远程资格</b><br>{esc(TARGET_ROUTE_ACTION['E2_SINGLE_MODEL_REMOTE_QUALIFIED'])}。报告时必须注明是哪一个模型具有远程资格。</div></div>
      <div class="stage"><strong>E3 · {n(target_counts['E3_LOCAL_ONLY_QUALIFIED'])}</strong><div><b>仅局部资格</b><br>{esc(TARGET_ROUTE_ACTION['E3_LOCAL_ONLY_QUALIFIED'])}。不能用N3分数宣称远程发现。</div></div>
      <div class="stage"><strong>E4 · {n(target_counts['E4_RECEPTOR_PROTOCOL_REPAIR'])}</strong><div><b>受体协议修复</b><br>{esc(TARGET_ROUTE_ACTION['E4_RECEPTOR_PROTOCOL_REPAIR'])}。其中DGAT1、TRPV4、CACNA1H、MPO因已有102条A/B/C结果而优先。</div></div>
      <div class="stage"><strong>E5 · {n(target_counts['E5_NO_MODEL_QUALIFICATION'])}</strong><div><b>当前模型未取得资格</b><br>{esc(TARGET_ROUTE_ACTION['E5_NO_MODEL_QUALIFICATION'])}。</div></div>
      <div class="stage"><strong>E6 · {n(target_counts['E6_NO_ADEQUATE_HISTORICAL_CONTROL'])}</strong><div><b>历史对照不足</b><br>{esc(TARGET_ROUTE_ACTION['E6_NO_ADEQUATE_HISTORICAL_CONTROL'])}。</div></div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>4. Boltz与GNINA的639条物理证据并集</h2>
      <p>统一去除known关系、要求受体协议通过，并只保留N2/N3后，Boltz严格池466条与GNINA严格池188条合并为639条，覆盖83个靶点、305个FDA药物实体。</p>
      {table(['Pair证据层','Pair','靶点','药物','下一步'], pair_rows, 'compact')}
      <div class="callout"><strong>为什么不是简单取交集：</strong>只有15条得到两模型共同支持。只取交集会把项目重新压缩到少数模型熟悉空间；合理做法是保留共同支持、可解释分歧和单模型适用三条独立赛道。</div>
      <div class="callout red"><strong>模型分歧：</strong>P2和P3共223条，另一模型在该pair范围内具有适用资格但给出不同结论。这一层最适合做多seed和受体构象敏感性分析。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>5. 下一阶段计算任务</h2>
      {table(['优先级','任务代码','任务','单位','数量','最大模型运行数','启动条件'], task_rows, 'compact')}
      <div class="callout gold"><strong>立即队列：</strong>549条P1–P4 pair各增加2个Boltz seed，共1,098次预测；其目的是估计随机采样稳定性，而不是用重复次数人为增加置信度。</div>
      <div class="callout blue"><strong>条件队列：</strong>328条GNINA适用pair中，311条（27个靶点）具有替代实验结构，按每条最多2个结构可执行614次GNINA运行；另17条集中在1个无替代PDB的靶点，保持阻塞。</div>
      <p class="small">模型运行数不等于最终候选数。每个pair仍只是一条科学假说，重复与ensemble用于评估稳定性。</p>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>6. 晋级、降级与停止规则</h2>
      {table(['对象','晋级条件','降级/停止条件'],[
        ['P1共同支持','新增Boltz seed至少2/2保持原靶点内等级；GNINA替代结构方向一致','pose离开口袋、seed结果剧烈波动或替代结构反转'],
        ['P2/P3模型分歧','至少一个方法在多seed/多结构下稳定，且分歧可由适用域或构象解释','两种方法均不稳定，或支持只来自单一异常结构'],
        ['P4/P5单模型支持','取得资格的模型重复稳定，并通过家族、暴露和实验可行性审计','另一模型无资格时不得以其低分淘汰；但缺少正交证据应降低实验优先级'],
        ['E4受体修复','替代实验结构redocking RMSD≤2.5 Å并冻结新协议','没有结构能复现参考配体时，退出统一结构主队列'],
        ['E5 Gate失败','新结构/新状态在锁定历史集重新通过Gate-A','同一协议不再扩大未知pair计算'],
        ['E6对照不足','获得至少8阳性+8实测低活性/阴性并完成锁定评价','未补齐前只保留每靶点少量信息候选'],
      ], 'compact')}
      <div class="callout"><strong>MD准入：</strong>只在上述稳定性复核后，从每个靶点选择最多1–2条，整个项目上限100条。MD用于pose稳定性和相对比较，不用于证明Kd。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>7. 资源预算与执行顺序</h2>
      <div class="stage"><strong>阶段0</strong><div><b>输入与结构修复</b><br>20条quinidine模型输入；22个受体协议，其中4个高优先。CPU准备和人工结构审计是主要耗时。</div></div>
      <div class="stage"><strong>阶段1</strong><div><b>稳定性复算</b><br>1,098次Boltz新增seed。按本轮4×RTX 4090实测速率，纯推理约1小时量级，连同准备、失败重试按1–2小时预算。</div></div>
      <div class="stage"><strong>阶段2</strong><div><b>受体ensemble</b><br>当前可执行上限614次GNINA运行；计算很快，但必须先选择并冻结合格替代实验结构。</div></div>
      <div class="stage"><strong>阶段3</strong><div><b>重新分级</b><br>计算seed稳定率、pose口袋一致性、ensemble方向一致性，形成可进入pair级科学审计的候选。</div></div>
      <div class="stage"><strong>阶段4</strong><div><b>限额MD</b><br>仅对通过稳定性和可行性审计的最多100条执行；不对639条全量开展MD。</div></div>
      <h3>存储估计</h3>
      <p>当前Boltz条件队列约30 GB，折合约2 MB/任务。1,098次新增预测预计增加约2–3 GB；GNINA ensemble输出远小于Boltz。MD轨迹必须单独设置压缩和保留策略。</p>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <h2>8. 最终建议</h2>
      <div class="callout"><strong>现在可以立即启动：</strong>Boltz多seed 1,098次、quinidine 20条修复、4个高优先受体协议修复，以及100个Gate失败靶点的诊断。</div>
      <div class="callout gold"><strong>现在不应启动：</strong>对E5/E6继续全量扩筛、对7,206条适用域外结果强制排序、或对全部639条直接开展MD。</div>
      <div class="two-col">
        <div class="card"><h3>面向命中</h3><ul><li>优先P1共同支持。</li><li>其次选择多seed稳定的P2/P3分歧候选。</li><li>P4/P5按靶点配额和实验可行性补充。</li><li>每个靶点限制候选数量，避免模型热点垄断。</li></ul></div>
        <div class="card"><h3>面向学习</h3><ul><li>E1用于检验远程发现能力。</li><li>E3用于测量局部scaffold-hop能力。</li><li>E5用于识别模型和结构失败方式。</li><li>E6用于建立新的前瞻正负校准数据。</li></ul></div>
      </div>
      <div class="callout blue"><strong>项目主线：</strong>下一阶段不是继续追求一个更大的总分表，而是让每一种证据状态进入与其资格相匹配的计算和实验问题。</div>
    </section>
    """)

    html_doc = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(pages)}</body></html>"
    markdown = f"""# FDA老药新靶点：证据分流与计算任务安排

生成日期：{created}

## 靶点分流

| 通道 | 靶点数 | 允许用途 |
|---|---:|---|
"""
    for route in TARGET_ROUTE_ORDER:
        markdown += f"| {TARGET_ROUTE_ZH[route]} | {target_counts[route]} | {TARGET_ROUTE_SCOPE[route]} |\n"
    markdown += "\n## Pair物理证据并集\n\n| 证据层 | Pair数 | 下一步 |\n|---|---:|---|\n"
    for lane in PAIR_LANE_ORDER:
        markdown += f"| {PAIR_LANE_ZH[lane]} | {pair_counts[lane]} | {PAIR_LANE_ACTION[lane]} |\n"
    markdown += f"\n## 立即任务\n\n- Boltz新增seed：{summary['compute_budget']['boltz_multiseed_runs']}次。\n- GNINA替代结构：{summary['compute_budget']['gnina_ensemble_ready_pairs']}条可执行pair，共{summary['compute_budget']['gnina_ensemble_max_runs']}次；{summary['compute_budget']['gnina_ensemble_blocked_pairs']}条因无替代PDB阻塞。\n- 受体修复：22个靶点，其中4个高优先。\n- quinidine输入修复：20条。\n- MD：稳定性复核后最多100条。\n"
    return html_doc, markdown


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [CURRENT_MASTER, CURRENT_SUMMARY, BOLTZ_TARGETS, GNINA_DISCOVERY, RECEPTOR_AUDIT, POCKET_ATLAS, TECHNICAL_INCOMPLETE]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative inputs: {missing}")

    current_summary = read_json(CURRENT_SUMMARY)
    master = pd.read_csv(CURRENT_MASTER, low_memory=False)
    targets = pd.read_csv(BOLTZ_TARGETS, low_memory=False)
    receptors = pd.read_csv(RECEPTOR_AUDIT, low_memory=False)
    gnina = pd.read_csv(GNINA_DISCOVERY, low_memory=False)
    pockets = pd.read_csv(POCKET_ATLAS, low_memory=False)
    incomplete = pd.read_csv(TECHNICAL_INCOMPLETE, low_memory=False)

    receptor_cols = [
        "target_chembl_id", "final_audit_status", "redock_status_final",
        "redock_best_symmetry_rmsd_A_final", "pdb_id", "target_chain_id",
        "alternate_structure_used", "docking_context_included",
    ]
    targets = targets.drop(columns=[c for c in receptor_cols if c != "target_chembl_id" and c in targets.columns], errors="ignore")
    targets = targets.merge(receptors[receptor_cols], on="target_chembl_id", how="left", validate="one_to_one")
    if len(targets) != 338 or targets["target_chembl_id"].duplicated().any():
        raise RuntimeError("Target qualification table must contain 338 unique targets")

    usable_pockets = pockets[
        as_bool(pockets["canonical_residue_mapping_valid"])
        & (as_bool(pockets["usable_structure_quality"]) | as_bool(pockets["preferred_structure_quality"]))
    ]
    pocket_counts = usable_pockets.groupby("target_chembl_id").agg(
        usable_experimental_pocket_instances=("pocket_instance_id", "nunique"),
        usable_experimental_pdbs=("pdb_id", "nunique"),
    ).reset_index()
    targets = targets.merge(pocket_counts, on="target_chembl_id", how="left", validate="one_to_one")
    targets[["usable_experimental_pocket_instances", "usable_experimental_pdbs"]] = targets[
        ["usable_experimental_pocket_instances", "usable_experimental_pdbs"]
    ].fillna(0).astype(int)
    targets["alternate_experimental_pdb_count"] = (targets["usable_experimental_pdbs"] - 1).clip(lower=0)

    targets["target_route"] = targets.apply(target_route, axis=1)
    targets["target_route_zh"] = targets["target_route"].map(TARGET_ROUTE_ZH)
    targets["qualified_models"] = targets.apply(qualified_model, axis=1)
    targets["allowed_scope_zh"] = targets["target_route"].map(TARGET_ROUTE_SCOPE)
    targets["next_compute_action_zh"] = targets["target_route"].map(TARGET_ROUTE_ACTION)
    targets["target_route_priority"] = targets["target_route"].map({route: i + 1 for i, route in enumerate(TARGET_ROUTE_ORDER)})

    category_counts = master.pivot_table(
        index="target_chembl_id", columns="result_category", values="pairId", aggfunc="count", fill_value=0
    ).reset_index()
    targets = targets.merge(category_counts, on="target_chembl_id", how="left", validate="one_to_one")
    for category in current_summary["classification_counts"]:
        if category not in targets.columns:
            targets[category] = 0
        targets[category] = targets[category].fillna(0).astype(int)
    targets["strict_boltz_abc_pairs"] = (
        targets["01_STRICT_AB_DISCOVERY_CORE"] + targets["02_STRICT_C_DISCOVERY_RESERVE"]
    )
    targets["receptor_review_abc_pairs"] = (
        targets["04_RECEPTOR_PROTOCOL_REVIEW_AB"] + targets["05_RECEPTOR_PROTOCOL_REVIEW_C"]
    )

    gnina_known = as_bool(gnina["is_known_relationship_control"])
    gnina_strict_mask = (
        gnina["gnina_structure_evidence_tier"].isin(["STRUCTURE_A", "STRUCTURE_B", "STRUCTURE_C"])
        & ~gnina_known
        & gnina["receptor_protocol_status"].eq("FROZEN_PASS")
        & gnina["novelty_lane"].isin(["N2_SCAFFOLD_HOP", "N3_REMOTE"])
    )
    gnina_strict = gnina[gnina_strict_mask].copy()
    gnina_target_counts = gnina_strict.groupby("target_chembl_id").size().rename("strict_gnina_abc_pairs").reset_index()
    targets = targets.merge(gnina_target_counts, on="target_chembl_id", how="left", validate="one_to_one")
    targets["strict_gnina_abc_pairs"] = targets["strict_gnina_abc_pairs"].fillna(0).astype(int)

    gnina_pair_cols = [
        "ligand_inchikey", "target_chembl_id", "gnina_structure_evidence_tier",
        "gnina_target_qualification", "novelty_lane", "max_tanimoto_all_known_positive",
        "primary_cnn_affinity", "primary_pose_vina_affinity",
    ]
    merged = master.merge(
        gnina[gnina_pair_cols].drop_duplicates(["ligand_inchikey", "target_chembl_id"]),
        on=["ligand_inchikey", "target_chembl_id"], how="left", suffixes=("", "_gnina"), validate="one_to_one",
    )
    merged["unified_max_tanimoto_to_known_positive"] = merged[
        ["max_tanimoto_to_target_positive", "max_tanimoto_all_known_positive"]
    ].max(axis=1, skipna=True)
    merged["unified_novelty_class"] = np.where(
        merged["unified_max_tanimoto_to_known_positive"].gt(0.60), "N1_LOCAL_ANALOG",
        np.where(merged["unified_max_tanimoto_to_known_positive"].gt(0.40), "N2_SCAFFOLD_HOP", "N3_REMOTE"),
    )
    valid_pair = (
        ~as_bool(merged["effective_known_control"])
        & merged["receptor_protocol_status"].eq("FROZEN_PASS")
        & merged["unified_novelty_class"].isin(["N2_SCAFFOLD_HOP", "N3_REMOTE"])
    )
    merged["boltz_support"] = (
        valid_pair
        & merged["boltz_evidence_tier"].isin(["BOLTZ_STRUCTURE_A", "BOLTZ_STRUCTURE_B", "BOLTZ_STRUCTURE_C"])
    )
    merged["gnina_support"] = (
        valid_pair
        & merged["gnina_structure_evidence_tier"].isin(["STRUCTURE_A", "STRUCTURE_B", "STRUCTURE_C"])
    )
    merged["boltz_applicable"] = (
        merged["boltz_target_qualification"].eq("BOLTZ_REMOTE_QUALIFIED")
        | (merged["boltz_target_qualification"].eq("BOLTZ_LOCAL_QUALIFIED") & merged["unified_novelty_class"].eq("N2_SCAFFOLD_HOP"))
    )
    merged["gnina_applicable"] = (
        merged["gnina_target_qualification_gnina"].eq("REMOTE_STRONG")
        | (merged["gnina_target_qualification_gnina"].eq("LOCAL_STRONG") & merged["unified_novelty_class"].eq("N2_SCAFFOLD_HOP"))
    )
    pair_union = merged[merged["boltz_support"] | merged["gnina_support"]].copy()
    pair_union["pair_evidence_lane"] = pair_union.apply(pair_lane, axis=1)
    pair_union["pair_evidence_lane_zh"] = pair_union["pair_evidence_lane"].map(PAIR_LANE_ZH)
    pair_union["next_compute_action_zh"] = pair_union["pair_evidence_lane"].map(PAIR_LANE_ACTION)
    pair_union = pair_union.merge(
        targets[
            [
                "target_chembl_id",
                "target_route",
                "target_route_zh",
                "target_route_priority",
                "alternate_experimental_pdb_count",
            ]
        ],
        on="target_chembl_id", how="left", validate="many_to_one",
    )
    pair_union["pair_lane_priority"] = pair_union["pair_evidence_lane"].map({lane: i + 1 for i, lane in enumerate(PAIR_LANE_ORDER)})
    pair_union = pair_union.sort_values(
        ["pair_lane_priority", "target_route_priority", "dta_priority_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    pair_lane_counts = pair_union["pair_evidence_lane"].value_counts().to_dict()
    expected_pair_counts = {
        "P1_DUAL_STRUCTURE_SUPPORT": 15,
        "P2_BOLTZ_SUPPORT_GNINA_DISAGREES": 140,
        "P3_GNINA_SUPPORT_BOLTZ_DISAGREES": 83,
        "P4_BOLTZ_ONLY_APPLICABLE": 311,
        "P5_GNINA_ONLY_APPLICABLE": 90,
    }
    if pair_lane_counts != expected_pair_counts:
        raise RuntimeError(f"Pair evidence lanes changed: {pair_lane_counts}")
    pair_key = ["ligand_inchikey", "target_chembl_id"]
    if pair_union.duplicated(pair_key).any():
        raise RuntimeError("Physical evidence union contains duplicate ligand-target keys")
    if as_bool(pair_union["effective_known_control"]).any():
        raise RuntimeError("Known controls leaked into the 639-pair discovery union")
    if not pair_union["unified_novelty_class"].isin(["N2_SCAFFOLD_HOP", "N3_REMOTE"]).all():
        raise RuntimeError("N1/local analog rows leaked into the physical evidence union")

    pair_target_counts = pair_union.groupby(["target_chembl_id", "pair_evidence_lane"]).size().unstack(fill_value=0).reset_index()
    targets = targets.merge(pair_target_counts, on="target_chembl_id", how="left", validate="one_to_one")
    for lane in PAIR_LANE_ORDER:
        if lane not in targets.columns:
            targets[lane] = 0
        targets[lane] = targets[lane].fillna(0).astype(int)
    targets["physical_union_pairs"] = targets[PAIR_LANE_ORDER].sum(axis=1)
    targets["urgent_receptor_repair"] = (
        targets["target_route"].eq("E4_RECEPTOR_PROTOCOL_REPAIR") & targets["receptor_review_abc_pairs"].gt(0)
    )
    targets = targets.sort_values(
        ["target_route_priority", "urgent_receptor_repair", "physical_union_pairs", "gene_symbol"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    boltz_repeat_lanes = [
        "P1_DUAL_STRUCTURE_SUPPORT", "P2_BOLTZ_SUPPORT_GNINA_DISAGREES",
        "P3_GNINA_SUPPORT_BOLTZ_DISAGREES", "P4_BOLTZ_ONLY_APPLICABLE",
    ]
    boltz_repeat = pair_union[pair_union["pair_evidence_lane"].isin(boltz_repeat_lanes)].copy()
    boltz_repeat["task_code"] = "C1_BOLTZ_MULTI_SEED"
    boltz_repeat["additional_seed_runs"] = 2
    boltz_repeat["planned_seed_1"] = 20260881
    boltz_repeat["planned_seed_2"] = 20260882
    boltz_repeat["promotion_metric"] = "retain target-internal tier in both added seeds; inspect pose-pocket consistency"

    gnina_ensemble_lanes = [
        "P1_DUAL_STRUCTURE_SUPPORT", "P2_BOLTZ_SUPPORT_GNINA_DISAGREES",
        "P3_GNINA_SUPPORT_BOLTZ_DISAGREES", "P5_GNINA_ONLY_APPLICABLE",
    ]
    gnina_ensemble = pair_union[pair_union["pair_evidence_lane"].isin(gnina_ensemble_lanes)].copy()
    gnina_ensemble["task_code"] = "C2_GNINA_RECEPTOR_ENSEMBLE"
    gnina_ensemble["max_alternate_receptor_states"] = 2
    gnina_ensemble["task_status"] = np.where(
        gnina_ensemble["alternate_experimental_pdb_count"].gt(0), "READY_FOR_ALTERNATE_SELECTION", "BLOCKED_NO_ALTERNATE_PDB"
    )
    gnina_ensemble["planned_alternate_runs"] = gnina_ensemble["alternate_experimental_pdb_count"].clip(upper=2).astype(int)
    gnina_ensemble["promotion_metric"] = "directional support across qualified alternate experimental receptor states"

    gnina_ready = gnina_ensemble[gnina_ensemble["task_status"].eq("READY_FOR_ALTERNATE_SELECTION")].copy()
    gnina_blocked = gnina_ensemble[gnina_ensemble["task_status"].eq("BLOCKED_NO_ALTERNATE_PDB")].copy()
    gnina_planned_runs = int(gnina_ready["planned_alternate_runs"].sum())

    if len(boltz_repeat) != 549 or int(boltz_repeat["additional_seed_runs"].sum()) != 1098:
        raise RuntimeError("Boltz multi-seed task no longer matches the frozen 549-pair plan")
    if len(gnina_ensemble) != 328 or len(gnina_ready) != 311 or len(gnina_blocked) != 17 or gnina_planned_runs != 614:
        raise RuntimeError("GNINA receptor-ensemble task no longer matches available alternate structures")

    receptor_repair = targets[targets["target_route"].eq("E4_RECEPTOR_PROTOCOL_REPAIR")].copy()
    receptor_repair["task_code"] = "C0_RECEPTOR_PROTOCOL_REPAIR"
    receptor_repair["repair_priority"] = np.where(receptor_repair["urgent_receptor_repair"], "URGENT", "SECONDARY")
    receptor_repair["completion_rule"] = "alternate experimental holo protocol redocking RMSD <= 2.5 A or explicit downgrade"

    review_pairs = master[master["result_category"].isin(["04_RECEPTOR_PROTOCOL_REVIEW_AB", "05_RECEPTOR_PROTOCOL_REVIEW_C"])].copy()
    review_pairs["task_code"] = "C3_BOLTZ_AFTER_RECEPTOR_RESCUE"
    review_pairs["start_condition"] = "target receptor protocol repaired and frozen"
    review_pairs["planned_runs"] = 1

    quinidine = incomplete.copy()
    quinidine["task_code"] = "C0_QUINIDINE_INPUT_REPAIR"
    quinidine["start_condition"] = "enumerate correct stereochemistry/protonation and generate valid conformer"
    quinidine["planned_runs"] = 1

    gate_diagnostic = targets[targets["target_route"].eq("E5_NO_MODEL_QUALIFICATION")].copy()
    gate_diagnostic["task_code"] = "C0_GATE_FAILURE_DIAGNOSTIC"
    gate_diagnostic["diagnostic_scope"] = "control labels; property separability; receptor state; pocket; model direction"
    control_acquisition = targets[targets["target_route"].eq("E6_NO_ADEQUATE_HISTORICAL_CONTROL")].copy()
    control_acquisition["task_code"] = "C0_CONTROL_ACQUISITION"
    control_acquisition["minimum_goal"] = "8 measured positives + 8 measured low-activity/inactive controls"

    task_summary = pd.DataFrame([
        {"priority": 0, "task_code": "C0_QUINIDINE_INPUT_REPAIR", "task_zh": "quinidine输入修复", "unit": "pair", "units": len(quinidine), "max_model_runs": len(quinidine), "start_condition": "立即"},
        {"priority": 0, "task_code": "C0_RECEPTOR_PROTOCOL_REPAIR", "task_zh": "受体协议修复", "unit": "target", "units": len(receptor_repair), "max_model_runs": 0, "start_condition": "4个有A/B/C靶点优先"},
        {"priority": 0, "task_code": "C0_GATE_FAILURE_DIAGNOSTIC", "task_zh": "Gate失败诊断", "unit": "target", "units": len(gate_diagnostic), "max_model_runs": 0, "start_condition": "立即，先不扩筛pair"},
        {"priority": 0, "task_code": "C0_CONTROL_ACQUISITION", "task_zh": "历史正负对照补全", "unit": "target", "units": len(control_acquisition), "max_model_runs": 0, "start_condition": "立即，数据任务"},
        {"priority": 1, "task_code": "C1_BOLTZ_MULTI_SEED", "task_zh": "Boltz多seed稳定性", "unit": "pair", "units": len(boltz_repeat), "max_model_runs": len(boltz_repeat) * 2, "start_condition": "可立即运行"},
        {"priority": 2, "task_code": "C2_GNINA_RECEPTOR_ENSEMBLE", "task_zh": "GNINA替代受体ensemble", "unit": "pair", "units": len(gnina_ready), "max_model_runs": gnina_planned_runs, "start_condition": "27个靶点有合格替代实验结构；17条pair阻塞"},
        {"priority": 2, "task_code": "C3_BOLTZ_AFTER_RECEPTOR_RESCUE", "task_zh": "受体修复后重算", "unit": "pair", "units": len(review_pairs), "max_model_runs": len(review_pairs), "start_condition": "新受体协议redocking通过"},
        {"priority": 3, "task_code": "C4_MD_DEFERRED", "task_zh": "限额MD", "unit": "pair", "units": 100, "max_model_runs": 100, "start_condition": "多seed/ensemble稳定且pair级审计通过"},
    ])

    target_route_counts = targets["target_route"].value_counts().to_dict()
    expected_target_counts = {
        "E1_DUAL_REMOTE_QUALIFIED": 13,
        "E2_SINGLE_MODEL_REMOTE_QUALIFIED": 70,
        "E3_LOCAL_ONLY_QUALIFIED": 79,
        "E4_RECEPTOR_PROTOCOL_REPAIR": 22,
        "E5_NO_MODEL_QUALIFICATION": 100,
        "E6_NO_ADEQUATE_HISTORICAL_CONTROL": 54,
    }
    if target_route_counts != expected_target_counts:
        raise RuntimeError(f"Target routes changed: {target_route_counts}")
    if targets["target_route"].isna().any() or targets["target_chembl_id"].duplicated().any():
        raise RuntimeError("Target routing is not a complete mutually exclusive partition")
    if len(receptor_repair) != 22 or int(receptor_repair["urgent_receptor_repair"].sum()) != 4:
        raise RuntimeError("Receptor repair target counts changed")
    if len(review_pairs) != 102 or len(quinidine) != 20 or len(gate_diagnostic) != 100 or len(control_acquisition) != 54:
        raise RuntimeError("One or more downstream task manifests changed unexpectedly")

    summary = {
        "package_name": "EVIDENCE_ROUTING_COMPUTE_PLAN_20260808_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "current_result": {
            "boltz_input_pairs": int(current_summary["totals"]["boltz_input_pairs"]),
            "boltz_completed_pairs": int(current_summary["totals"]["boltz_completed_pairs"]),
            "strict_boltz_ab_core": int(current_summary["classification_counts"]["01_STRICT_AB_DISCOVERY_CORE"]),
            "strict_boltz_c_reserve": int(current_summary["classification_counts"]["02_STRICT_C_DISCOVERY_RESERVE"]),
            "technical_incomplete": int(current_summary["classification_counts"]["09_TECHNICAL_INCOMPLETE"]),
        },
        "target_route_counts": {route: int(target_route_counts[route]) for route in TARGET_ROUTE_ORDER},
        "pair_union": {
            "pairs": int(len(pair_union)),
            "targets": int(pair_union["target_chembl_id"].nunique()),
            "drug_entities": int(pair_union["project_entity_ids"].nunique()),
            "model_structures": int(pair_union["ligand_inchikey"].nunique()),
            "unified_n3": int(pair_union["unified_novelty_class"].eq("N3_REMOTE").sum()),
            "unified_n2": int(pair_union["unified_novelty_class"].eq("N2_SCAFFOLD_HOP").sum()),
        },
        "pair_lane_counts": {lane: int(pair_lane_counts[lane]) for lane in PAIR_LANE_ORDER},
        "compute_budget": {
            "boltz_multiseed_pairs": int(len(boltz_repeat)),
            "boltz_added_seeds_per_pair": 2,
            "boltz_multiseed_runs": int(len(boltz_repeat) * 2),
            "gnina_ensemble_pairs": int(len(gnina_ensemble)),
            "gnina_ensemble_ready_pairs": int(len(gnina_ready)),
            "gnina_ensemble_ready_targets": int(gnina_ready["target_chembl_id"].nunique()),
            "gnina_ensemble_blocked_pairs": int(len(gnina_blocked)),
            "gnina_ensemble_blocked_targets": int(gnina_blocked["target_chembl_id"].nunique()),
            "gnina_max_alternate_states": 2,
            "gnina_ensemble_max_runs": gnina_planned_runs,
            "receptor_repair_targets": int(len(receptor_repair)),
            "urgent_receptor_repair_targets": int(receptor_repair["urgent_receptor_repair"].sum()),
            "post_repair_boltz_pairs": int(len(review_pairs)),
            "quinidine_repair_pairs": int(len(quinidine)),
            "gate_failure_diagnostic_targets": int(len(gate_diagnostic)),
            "control_acquisition_targets": int(len(control_acquisition)),
            "md_pair_cap": 100,
        },
        "boundaries": [
            "Target routes are mutually exclusive; pair evidence lanes are mutually exclusive within the 639-pair union.",
            "Multi-seed and receptor ensembles estimate stability; they do not create independent experimental evidence.",
            "A model score cannot promote or reject a pair outside that model's target/novelty qualification scope.",
            "MD is deferred until multi-seed, receptor-state and pair-level audits are complete.",
        ],
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }

    target_path = OUT / "TARGET_EVIDENCE_ROUTING_338_V1.csv"
    pair_path = OUT / "PAIR_PHYSICAL_EVIDENCE_UNION_639_V1.csv.gz"
    targets.to_csv(target_path, index=False)
    pair_union.to_csv(pair_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    boltz_repeat.to_csv(OUT / "TASK_C1_BOLTZ_MULTI_SEED_549_PAIRS_V1.csv", index=False)
    gnina_ensemble.to_csv(OUT / "TASK_C2_GNINA_RECEPTOR_ENSEMBLE_328_PAIRS_V1.csv", index=False)
    receptor_repair.to_csv(OUT / "TASK_C0_RECEPTOR_PROTOCOL_REPAIR_22_TARGETS_V1.csv", index=False)
    review_pairs.to_csv(OUT / "TASK_C3_BOLTZ_AFTER_RECEPTOR_RESCUE_102_PAIRS_V1.csv", index=False)
    quinidine.to_csv(OUT / "TASK_C0_QUINIDINE_INPUT_REPAIR_20_PAIRS_V1.csv", index=False)
    gate_diagnostic.to_csv(OUT / "TASK_C0_GATE_FAILURE_DIAGNOSTIC_100_TARGETS_V1.csv", index=False)
    control_acquisition.to_csv(OUT / "TASK_C0_CONTROL_ACQUISITION_54_TARGETS_V1.csv", index=False)
    task_summary.to_csv(OUT / "COMPUTE_TASK_SUMMARY_V1.csv", index=False)

    summary_path = OUT / "EVIDENCE_ROUTING_COMPUTE_PLAN_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_doc, markdown = build_report(summary, targets, pair_union, task_summary)
    html_path = OUT / "FDA_OLD_DRUG_NEW_TARGET_EVIDENCE_ROUTING_COMPUTE_PLAN_ZH.html"
    pdf_path = OUT / "FDA_OLD_DRUG_NEW_TARGET_EVIDENCE_ROUTING_COMPUTE_PLAN_ZH.pdf"
    md_path = OUT / "FDA_OLD_DRUG_NEW_TARGET_EVIDENCE_ROUTING_COMPUTE_PLAN_ZH.md"
    html_path.write_text(html_doc, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    HTML(filename=str(html_path), base_url=str(ROOT)).write_pdf(str(pdf_path))

    doc = fitz.open(pdf_path)
    page_text_lengths = [len(page.get_text("text").strip()) for page in doc]
    pdf_audit = {
        "status": "PASS" if len(doc) == 9 and min(page_text_lengths) > 180 else "REVIEW",
        "pages": len(doc),
        "page_text_lengths": page_text_lengths,
        "pdf_bytes": pdf_path.stat().st_size,
        "pdf_sha256": sha256(pdf_path),
        "target_rows": len(targets),
        "target_route_total": int(sum(target_route_counts.values())),
        "pair_union_rows": len(pair_union),
        "pair_union_unique_keys": int(pair_union[pair_key].drop_duplicates().shape[0]),
        "pair_union_known_controls": int(as_bool(pair_union["effective_known_control"]).sum()),
        "pair_union_n2_n3_only": bool(pair_union["unified_novelty_class"].isin(["N2_SCAFFOLD_HOP", "N3_REMOTE"]).all()),
        "pair_lane_total": int(sum(pair_lane_counts.values())),
        "boltz_multiseed_pairs": len(boltz_repeat),
        "boltz_multiseed_runs": int(boltz_repeat["additional_seed_runs"].sum()),
        "gnina_ensemble_pairs": len(gnina_ensemble),
        "gnina_ensemble_ready_pairs": len(gnina_ready),
        "gnina_ensemble_blocked_pairs": len(gnina_blocked),
        "gnina_ensemble_planned_runs": gnina_planned_runs,
        "receptor_repair_targets": len(receptor_repair),
        "urgent_receptor_repair_targets": int(receptor_repair["urgent_receptor_repair"].sum()),
        "post_repair_boltz_pairs": len(review_pairs),
        "technical_repair_pairs": len(quinidine),
        "gate_failure_diagnostic_targets": len(gate_diagnostic),
        "control_acquisition_targets": len(control_acquisition),
    }
    doc.close()
    (OUT / "EVIDENCE_ROUTING_COMPUTE_PLAN_AUDIT_V1.json").write_text(
        json.dumps(pdf_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if pdf_audit["status"] != "PASS":
        raise RuntimeError(f"PDF audit requires review: {pdf_audit}")

    manifest = {
        "status": "FROZEN_COMPUTE_PLAN",
        "created_utc": summary["created_utc"],
        "authoritative_outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "EVIDENCE_ROUTING_COMPUTE_PLAN_MANIFEST_V1.json"
        },
        "execution_rule": "Repair tasks precede pair reruns. MD is not authorized by this manifest and requires a post-stability shortlist.",
    }
    (OUT / "EVIDENCE_ROUTING_COMPUTE_PLAN_MANIFEST_V1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(OUT),
        "target_route_counts": summary["target_route_counts"],
        "pair_lane_counts": summary["pair_lane_counts"],
        "compute_budget": summary["compute_budget"],
        "pdf": str(pdf_path),
        "pdf_pages": pdf_audit["pages"],
        "status": pdf_audit["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the polished Chinese V7 report for the fully reviewed physics-first Top500."""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/current_production_package_v2/final500_full_deep_review_v7"
TOP500 = OUT_DIR / "FINAL500_PHYSICS_FIRST_FULL_DEEP_REVIEWED_V7.csv"
ALL1000 = OUT_DIR / "FINAL1000_FULL_DEEP_REVIEWED_V7.csv"
V7_AUDIT = OUT_DIR / "FINAL500_FULL_DEEP_REVIEW_V7_AUDIT.json"
SELECTION_AUDIT = OUT_DIR / "FINAL500_SELECTION_AUDIT_V7.json"
SCOPE_AUDIT = ROOT / "outputs/current_production_package_v2/universe_scope_audit_v4/universe_scope_audit_v4.json"
UNIVERSE_SUMMARY = ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v4/full_untruncated_universe_v4_summary.json"
KNOWN_BOLTZ = ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_boltz96_calibration_v4.summary.json"
FORMAL_SUMMARY = ROOT / "outputs/current_production_package_v2/formal_full_universe_v4/formal_completion_summary_v4_complete.json"
OUTPUT_PDF = OUT_DIR / "FDA_OLD_DRUG_NEW_TARGET_TOP500_FULL_DEEP_REVIEW_REPORT_ZH_V7.pdf"
OUTPUT_AUDIT = OUT_DIR / "FINAL500_FULL_DEEP_REVIEW_PDF_V7_AUDIT.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value))


def truncate(value: Any, length: int = 78) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= length else text[: length - 1] + "…"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def metric(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        "</div>"
    )


def bar_chart(items: list[tuple[str, int]], total: int, accent: str = "#16766f") -> str:
    max_value = max((value for _, value in items), default=1)
    rows = []
    for label, value in items:
        width = 100 * value / max_value
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{esc(label)}</div>'
            '<div class="bar-track">'
            f'<div class="bar-fill" style="width:{width:.1f}%;background:{accent}"></div>'
            "</div>"
            f'<div class="bar-number">{value}<span>{100 * value / total:.1f}%</span></div>'
            "</div>"
        )
    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def funnel_step(number: str, title: str, count: str, detail: str) -> str:
    return (
        '<div class="funnel-step">'
        f'<div class="funnel-index">{esc(number)}</div>'
        '<div class="funnel-copy">'
        f'<div class="funnel-title">{esc(title)}</div>'
        f'<div class="funnel-detail">{esc(detail)}</div>'
        "</div>"
        f'<div class="funnel-count">{esc(count)}</div>'
        "</div>"
    )


def module_card(number: str, title: str, body: str, tag: str) -> str:
    return (
        '<div class="module-card">'
        '<div class="module-head">'
        f'<span class="module-number">{esc(number)}</span>'
        f'<span class="module-title">{esc(title)}</span>'
        "</div>"
        f'<div class="module-body">{esc(body)}</div>'
        f'<div class="module-tag">{esc(tag)}</div>'
        "</div>"
    )


def table(headers: list[str], rows: list[list[Any]], classes: str = "") -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f'<table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def role_zh(value: str) -> str:
    return {
        "novel_binding_hypothesis": "未见精确公开记录",
        "literature_triage_hypothesis": "文献线索待核",
        "functional_record_revalidation": "已知功能关系复验",
        "reported_pair_revalidation": "已报道pair复验",
    }.get(value, value)


def family_zh(value: str) -> str:
    return {
        "enzyme": "酶",
        "kinase": "激酶",
        "nuclear_epigenetic": "核受体/表观遗传",
        "transporter": "转运体",
        "ion_channel": "离子通道",
    }.get(value, value)


def primary_readout_zh(value: str) -> str:
    return {
        "enzyme": "重组人蛋白酶活 + 正交结合/占有",
        "kinase": "人源激酶活 + 结合/占有",
        "nuclear_epigenetic": "配体/共因子/报告 + 正交结合",
        "transporter": "转运/竞争结合 + 同家族反筛",
        "ion_channel": "膜片钳/通量 + 近邻通道反筛",
    }.get(value, "靶点功能 + 正交结合")


def build_html() -> tuple[str, dict[str, Any]]:
    scope = load_json(SCOPE_AUDIT)
    universe = load_json(UNIVERSE_SUMMARY)
    known_boltz = load_json(KNOWN_BOLTZ)
    formal = load_json(FORMAL_SUMMARY)
    v7 = load_json(V7_AUDIT)
    selection = load_json(SELECTION_AUDIT)
    top500 = pd.read_csv(TOP500, low_memory=False).fillna("")
    all1000 = pd.read_csv(ALL1000, low_memory=False).fillna("")

    project_cal = scope["conplex_rank_calibration"]["project463"]
    full_cal = scope["conplex_rank_calibration"]["full891"]
    deep = v7["deep_review"]
    top = v7["top500"]

    pose_counts = top500["pose_stability_tier"].value_counts().to_dict()
    repurpose_counts = top500["repurposing_interpretation_v6"].value_counts().to_dict()
    t1 = top500.loc[top500["experimental_execution_tier_v6"].eq("T1_immediate_standard_assay")].copy()
    representative = t1.head(9)

    css = """
    @font-face { font-family: 'NotoSC'; src: url('file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'); font-weight: 400; }
    @font-face { font-family: 'NotoSC'; src: url('file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'); font-weight: 700; }
    @font-face { font-family: 'NotoSerifSC'; src: url('file:///usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'); font-weight: 700; }
    @page {
      size: A4;
      margin: 15mm 15mm 17mm 15mm;
      @top-left { content: 'FDA老药新靶点发现 · V7'; font-family: NotoSC; font-size: 8pt; color: #64716f; }
      @top-right { content: string(section); font-family: NotoSC; font-size: 8pt; color: #64716f; }
      @bottom-left { content: '亲和优先 · 统一综合审阅 · 2026-07-14'; font-family: NotoSC; font-size: 7.5pt; color: #7b8583; }
      @bottom-right { content: counter(page) ' / ' counter(pages); font-family: NotoSC; font-size: 8pt; color: #4d5a58; }
    }
    @page cover { margin: 0; @top-left { content: none; } @top-right { content: none; } @bottom-left { content: none; } @bottom-right { content: none; } }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; color: #162321; font-family: NotoSC, sans-serif; font-size: 9.5pt; line-height: 1.55; }
    body { background: #fff; }
    .cover { page: cover; height: 297mm; padding: 24mm 20mm 20mm; background: #f8faf9; position: relative; overflow: hidden; }
    .cover-band { position: absolute; left: 0; top: 0; width: 8mm; height: 297mm; background: #16766f; }
    .cover-rule { width: 54mm; height: 2.2mm; background: #d09a35; margin: 0 0 18mm 0; }
    .cover-kicker { color: #16766f; font-size: 11pt; font-weight: 700; margin-bottom: 7mm; }
    .cover h1 { margin: 0; font-family: NotoSerifSC, serif; font-size: 30pt; line-height: 1.25; font-weight: 700; color: #12201e; letter-spacing: 0; }
    .cover h2 { margin: 7mm 0 0; font-size: 16pt; line-height: 1.5; color: #33423f; font-weight: 400; }
    .cover-summary { margin-top: 10mm; max-width: 155mm; font-size: 11pt; color: #42514e; line-height: 1.75; }
    .cover-metrics { position: absolute; left: 20mm; right: 18mm; bottom: 28mm; display: grid; grid-template-columns: repeat(5, 1fr); gap: 4mm; }
    .cover .metric { border-top: 1.5mm solid #16766f; padding-top: 4mm; }
    .cover .metric-value { font-size: 20pt; }
    .cover-meta { position: absolute; left: 20mm; bottom: 14mm; color: #63706e; font-size: 8.5pt; }
    .section { page-break-before: always; }
    .section-title { string-set: section content(text); margin: 0 0 6mm; padding-bottom: 3mm; border-bottom: 0.7mm solid #16766f; font-family: NotoSerifSC, serif; font-size: 20pt; line-height: 1.25; color: #142320; }
    .section-subtitle { margin: -3mm 0 7mm; color: #5f6d6a; font-size: 9.5pt; }
    h3 { margin: 6mm 0 3mm; font-size: 12.5pt; color: #1e5752; }
    p { margin: 0 0 3mm; }
    .lead { font-size: 11pt; color: #31413e; line-height: 1.75; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm; margin: 5mm 0 7mm; }
    .metric { min-height: 27mm; border-top: 1.2mm solid #16766f; padding: 3.5mm 1mm 1mm 0; }
    .metric-label { color: #61706d; font-size: 8.5pt; }
    .metric-value { color: #142320; font-size: 18pt; line-height: 1.25; font-weight: 700; }
    .metric-note { margin-top: 1.5mm; color: #6f7c79; font-size: 7.8pt; line-height: 1.35; }
    .card { border: 0.35mm solid #d8dfdd; padding: 5mm; background: #fff; break-inside: avoid; }
    .card-title { color: #1e5752; font-weight: 700; font-size: 11pt; margin-bottom: 2mm; }
    .callout { margin: 5mm 0; padding: 4mm 5mm; border-left: 1.4mm solid #d09a35; background: #fbf8f0; color: #4b493f; }
    .callout.teal { border-left-color: #16766f; background: #f1f7f6; color: #294440; }
    .small { font-size: 8pt; color: #687572; }
    .bullet-list { margin: 2mm 0 0; padding-left: 5mm; }
    .bullet-list li { margin: 0 0 2mm; padding-left: 1mm; }
    .check-list { list-style: none; padding: 0; margin: 2mm 0 0; }
    .check-list li { position: relative; padding-left: 6mm; margin-bottom: 2.4mm; }
    .check-list li::before { content: '✓'; position: absolute; left: 0; top: 0; color: #16766f; font-weight: 700; }
    .funnel { margin-top: 4mm; }
    .funnel-step { display: grid; grid-template-columns: 12mm 1fr 34mm; gap: 4mm; align-items: center; padding: 3.3mm 0; border-bottom: 0.3mm solid #dce2e0; break-inside: avoid; }
    .funnel-index { width: 9mm; height: 9mm; border-radius: 50%; background: #16766f; color: #fff; text-align: center; line-height: 9mm; font-weight: 700; }
    .funnel-title { font-weight: 700; color: #20302d; font-size: 10.5pt; }
    .funnel-detail { color: #66736f; font-size: 8pt; margin-top: 0.5mm; }
    .funnel-count { text-align: right; font-size: 15pt; font-weight: 700; color: #1d5b55; }
    .module-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }
    .module-card { min-height: 39mm; border: 0.35mm solid #d7dfdc; padding: 4mm; break-inside: avoid; }
    .module-head { display: flex; align-items: center; gap: 2mm; margin-bottom: 2mm; }
    .module-number { display: inline-block; min-width: 7mm; height: 7mm; line-height: 7mm; text-align: center; background: #16766f; color: #fff; font-weight: 700; }
    .module-title { font-weight: 700; color: #20322f; font-size: 10.5pt; }
    .module-body { color: #51605d; font-size: 8.5pt; min-height: 16mm; }
    .module-tag { display: inline-block; margin-top: 2mm; padding: 0.8mm 2mm; background: #f0f5f4; color: #1e665f; font-size: 7.5pt; font-weight: 700; }
    .bar-chart { margin: 2mm 0; }
    .bar-row { display: grid; grid-template-columns: 35mm 1fr 25mm; gap: 3mm; align-items: center; min-height: 8mm; }
    .bar-label { color: #34433f; font-size: 8.5pt; }
    .bar-track { height: 3.5mm; background: #e8edeb; overflow: hidden; }
    .bar-fill { height: 100%; }
    .bar-number { text-align: right; font-weight: 700; color: #263633; }
    .bar-number span { display: block; color: #7b8683; font-weight: 400; font-size: 7pt; }
    table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm; table-layout: fixed; }
    thead { display: table-header-group; }
    tr { break-inside: avoid; }
    th { background: #eaf2f0; color: #214843; font-weight: 700; text-align: left; padding: 2.2mm; border-bottom: 0.4mm solid #9eb5b0; font-size: 8pt; }
    td { padding: 2.1mm; border-bottom: 0.25mm solid #dfe5e3; vertical-align: top; color: #35433f; font-size: 7.8pt; overflow-wrap: anywhere; }
    .compact th, .compact td { padding: 1.7mm; font-size: 7.4pt; }
    .tiny th, .tiny td { padding: 1.35mm; font-size: 6.8pt; line-height: 1.35; }
    .appendix-table th:nth-child(1), .appendix-table td:nth-child(1) { width: 7%; }
    .appendix-table th:nth-child(2), .appendix-table td:nth-child(2) { width: 19%; }
    .appendix-table th:nth-child(3), .appendix-table td:nth-child(3) { width: 10%; }
    .appendix-table th:nth-child(4), .appendix-table td:nth-child(4) { width: 12%; }
    .appendix-table th:nth-child(5), .appendix-table td:nth-child(5) { width: 8%; }
    .appendix-table th:nth-child(6), .appendix-table td:nth-child(6) { width: 9%; }
    .appendix-table th:nth-child(7), .appendix-table td:nth-child(7) { width: 9%; }
    .appendix-table th:nth-child(8), .appendix-table td:nth-child(8) { width: 26%; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 2mm; margin: 3mm 0; }
    .tag { display: inline-block; padding: 1mm 2.5mm; border: 0.3mm solid #b9cbc7; color: #235f59; background: #f6f9f8; font-size: 8pt; }
    .grade { display: inline-block; min-width: 8mm; text-align: center; padding: 0.5mm 1mm; color: #fff; font-weight: 700; }
    .grade-a { background: #16766f; } .grade-b { background: #4b7c77; } .grade-c { background: #9a7a38; } .grade-d { background: #8c4c4c; }
    .source-box { margin-top: 5mm; border-top: 0.4mm solid #bcc9c6; padding-top: 3mm; font-size: 7pt; color: #6b7774; overflow-wrap: anywhere; }
    .no-break { break-inside: avoid; }
    """

    cover_metrics = "".join(
        [
            metric("原始设计空间", "815,265", "915个FDA条目 × 891条唯一蛋白序列"),
            metric("唯一模型pair", "334,749", "活性母体与项目靶点归一后"),
            metric("Boltz-2精修", "3,000", "全量完成双构象结构计算"),
            metric("统一综合审阅", "1,000", "相同字段、相同分级与证据标准"),
            metric("正式发现包", "500", "亲和优先、实体已解析、可实验分层"),
        ]
    )

    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><style>",
        css,
        "</style></head><body>",
        '<section class="cover">',
        '<div class="cover-band"></div><div class="cover-rule"></div>',
        '<div class="cover-kicker">FDA OLD DRUG · NEW TARGET DISCOVERY</div>',
        "<h1>FDA老药新靶点发现</h1>",
        "<h2>亲和优先 Top500 全流程与统一综合审阅报告</h2>",
        '<div class="cover-summary">从FDA小分子与ChEMBL-MoA人源靶点空间出发，经活性实体归一、可实验边界、ConPLEx排序、口袋与结构审计、Boltz-2精修、已知阳性校准及逐对综合审阅，形成500条直接结合计算假说。</div>',
        f'<div class="cover-metrics">{cover_metrics}</div>',
        '<div class="cover-meta">V7 · 2026-07-14 · BioMaster</div>',
        "</section>",
    ]

    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">01 · 结论摘要</h2>',
            '<div class="section-subtitle">最终输出是亲和发现包，不是疗效结论或500个同等成熟的转化项目。</div>',
            '<p class="lead">正式Top500全部通过统一物理与数据硬门：非已知FDA靶点对、非同家族/标签泄露、无明确直接反证、无结构序列错配，且活性实体均已解析。</p>',
            '<div class="metrics">',
            metric("FDA药物", "214", "按drug_chembl_id去重"),
            metric("候选靶点", "132", "人源单蛋白序列"),
            metric("Murcko骨架", "195", "限制化学系列垄断"),
            metric("未见精确公开记录", "438", "仍是待验证的结合假说"),
            "</div>",
            '<div class="grid-2">',
            '<div class="card"><div class="card-title">可直接确认的事实</div><ul class="check-list">',
            '<li>500/500处于已知阳性校准的物理A/B档。</li>',
            '<li>500/500具有严格重叠口袋，双构象稳定性均为A/B。</li>',
            '<li>500/500的母体或盐型可明确对应当前计算结构。</li>',
            '<li>每条均给出primary assay、阳性对照、已知靶点反筛和干扰门。</li>',
            "</ul></div>",
            '<div class="card"><div class="card-title">不能从本报告直接推出</div><ul class="bullet-list">',
            '<li>模型高分不等于实验亲和，更不等于临床疗效。</li>',
            '<li>结构模型不能自动判断激动、抑制或调节方向。</li>',
            '<li>已知阳性召回是校准，不是盲测泛化或精确率。</li>',
            '<li>疾病字段只是靶点机制语境，不能替代药物-疾病证据。</li>',
            "</ul></div></div>",
            '<div class="callout teal"><b>建议读取顺序：</b>先看68条T1标准实验队列，再看382条T2不确定性队列；50条T3需要膜蛋白、电生理或其他专门体系。完整500条保留用于广筛和多样性覆盖。</div>',
            "</section>",
        ]
    )

    drug_rows = [
        ["FDA结构条目", scope["drug_universe"]["all_fda_rows"], "原始药物表中的结构条目"],
        ["direct-action条目", scope["drug_universe"]["direct_action_rows"], "排除气体、诊断剂、物理吸附等非直接结合机制"],
        ["项目药物", scope["drug_universe"]["project_rows"], "活性母体/盐型/前药归一后进入计算范围"],
    ]
    target_rows = [
        ["ChEMBL-MoA基因", scope["target_universe"]["chembl_moa_genes"], "具明确人源单蛋白药物机制记录"],
        ["唯一蛋白序列", scope["target_universe"]["unique_sequences"], "合并重复基因/序列映射"],
        ["target-engagement序列", scope["target_universe"]["project_target_engagement_sequences"], "以酶、激酶、转运体、核蛋白和离子通道为主"],
    ]
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">02 · 药物与靶点宇宙</h2>',
            '<div class="section-subtitle">ChEMBL-MoA锚点用于定义已成药的人源单蛋白边界；Open Targets用于补充可做性与疾病语境。</div>',
            '<div class="grid-2">',
            '<div><h3>药物侧</h3>', table(["层级", "数量", "含义"], drug_rows, "compact"), "</div>",
            '<div><h3>靶点侧</h3>', table(["层级", "数量", "含义"], target_rows, "compact"), "</div></div>",
            '<div class="callout"><b>“成药锚点”不等于所有理论可成药蛋白。</b>本项目口径是ChEMBL中有明确药物机制/MoA记录的人源single-protein target；随后再按小分子target-engagement与实验边界收敛。</div>',
            '<div class="grid-3">',
            metric("GPCR排除", str(scope["target_universe"]["gpcr_excluded_genes"]), "本版优先非GPCR实验体系"),
            metric("分泌/表面结构排除", str(scope["target_universe"]["secreted_surface_structural_excluded_genes"]), "避免不适合当前小分子直接实验的靶点"),
            metric("项目笛卡尔空间", f"{scope['pair_spaces']['project_drugs_x_targets']:,}", "750 × 463"),
            "</div>",
            '<p class="small">Open Targets 26.06并不是“可成药蛋白列表”；其更大的靶点集合用于tractability、表达、遗传和疾病证据补全，不直接扩张本版主靶点宇宙。</p>',
            "</section>",
        ]
    )

    funnel = "".join(
        [
            funnel_step("1", "原始全空间", f"{scope['pair_spaces']['raw_915_x_891']:,}", "915个FDA结构条目 × 891条唯一蛋白序列"),
            funnel_step("2", "项目直接结合空间", f"{scope['pair_spaces']['unique_model_ligands_x_targets']:,}", "直接作用药物、活性母体归一与463条target-engagement序列"),
            funnel_step("3", "物理与数据硬门合格", f"{universe['eligible_rows']:,}", "去除已知pair、同家族泄露、不可实验类型、严重责任和结构问题"),
            funnel_step("4", "结构精修队列", f"{universe['selected_rows']:,}", "ConPLEx排序、口袋/结构可做性和多样性后进入Boltz-2"),
            funnel_step("5", "统一综合审阅池", "1,000", "已知阳性校准物理档、双构象稳定性和实验可行性综合"),
            funnel_step("6", "发现排序合格", f"{selection['selection']['eligible_discovery_rows']:,}", "统一排除D、直接反证、精确已知pair及活性实体未解决项"),
            funnel_step("7", "正式Top500", "500", "药物、靶点、骨架与assay family多样性约束"),
        ]
    )
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">03 · 全流程漏斗</h2>',
            '<div class="section-subtitle">旧106,561条Top300派生池不再作为主入口；正式流程从完整项目空间重新排序。</div>',
            f'<div class="funnel">{funnel}</div>',
            '<div class="callout teal">正式Top500的ConPLEx活性母体折叠药内名次全部≤100，实际最大名次为97；这是一道必要排序门，不将Top100解释为真实结合概率。</div>',
            "</section>",
        ]
    )

    modules = "".join(
        [
            module_card("1", "活性实体归一", "对母体、盐型、前药、活性代谢物和立体化学进行统一；计算实体必须能对应上市药物的实际活性物种。", "输入完整性"),
            module_card("2", "靶点与口袋", "人源唯一蛋白序列、实验类型、AlphaFold/结构来源、口袋重叠和结构序列一致性共同定义可做性。", "结构可验证性"),
            module_card("3", "ConPLEx", "输入SMILES与蛋白序列，输出无量纲相容性分；用于药内名次和候选召回，不作为Kd/Ki。", "快速排序"),
            module_card("4", "Boltz-2", "对3,000条进行蛋白-配体结构与affinity prediction，输出affinity probability、ligand iPTM、confidence与pose。", "结构精修"),
            module_card("5", "已知阳性校准", "96条序列匹配的已知直接作用pair定义物理q10/q25/median参照；仅校准信号尺度，不估计FDR。", "尺度参照"),
            module_card("6", "疾病与机制注释", "Open Targets与人工机制判断提供靶点-疾病语境；疾病证据不进入亲和主分，也不提前声称药物疗效。", "下游解释"),
        ]
    )
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">04 · 计算与证据模块</h2>',
            '<div class="section-subtitle">没有任何单一模型决定候选；输入实体、结构可做性、序列模型和复合物模型分别承担不同职责。</div>',
            f'<div class="module-grid">{modules}</div>',
            '<div class="callout"><b>统一物理分：</b>70% Boltz校准轴 + 30% ConPLEx复合轴。它用于优先级排序，不是实验亲和概率、选择性或临床成功率。</div>',
            "</section>",
        ]
    )

    cal_rows = []
    for label, source in [("项目463靶点", project_cal), ("完整891靶点", full_cal)]:
        cal_rows.extend(
            [
                [label, "Top10", pct(source["top10_recall"]), pct(source["top10_random_expectation"]), f"{source['top10_enrichment']:.2f}×"],
                [label, "Top50", pct(source["top50_recall"]), pct(source["top50_random_expectation"]), f"{source['top50_enrichment']:.2f}×"],
                [label, "Top100", pct(source["top100_recall"]), pct(source["top100_random_expectation"]), f"{source['top100_enrichment']:.2f}×"],
            ]
        )
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">05 · 已知关系校准</h2>',
            '<div class="section-subtitle">校准回答“已知信号在排序中能否富集”，不回答未知候选中有多少是真的。</div>',
            '<h3>ConPLEx药内排名校准</h3>',
            table(["靶点范围", "截断", "已知关系召回", "随机期望", "富集倍数"], cal_rows, "compact"),
            '<div class="grid-3">',
            metric("已知校准pair", str(scope["project_known_union_calibration"]["pairs"]), "label/ChEMBL已知关系并集"),
            metric("活性母体-靶点pair", str(scope["project_known_union_calibration"]["active_moiety_target_pairs"]), "盐型折叠后唯一关系"),
            metric("Top100完整891富集", f"{full_cal['top100_enrichment']:.2f}×", "召回41.8%，随机期望11.2%"),
            "</div>",
            '<h3>Boltz-2已知阳性参照</h3>',
            '<div class="grid-2">',
            '<div class="card"><div class="card-title">Affinity probability</div>',
            f'<p>96条已知阳性中位数：<b>{known_boltz["known_affinity_probability"]["median"]:.3f}</b></p>',
            f'<p>3,000条未标注候选中位数：<b>{known_boltz["discovery_affinity_probability"]["median"]:.3f}</b></p>',
            '<p class="small">该数值用于相对排序，不等价于真实结合概率。</p></div>',
            '<div class="card"><div class="card-title">双构象稳定性</div>',
            f'<p>已知阳性稳定比例：<b>{pct(known_boltz["known_stable_pose_fraction"])}</b></p>',
            f'<p>未标注候选稳定比例：<b>{pct(known_boltz["discovery_stable_pose_fraction"])}</b></p>',
            '<p class="small">稳定pose仅说明模型内部一致，不证明姿势真实。</p></div></div>',
            '<div class="callout"><b>防止误读：</b>已知药物-靶点知识可能与模型训练信息重叠，因此这里是production calibration，不是时间外盲测，也不能据此报告precision或AUROC。</div>',
            "</section>",
        ]
    )

    review_grade_items = [("A：直接可执行", deep["all1000_grade_counts"].get("A", 0)), ("B：可测，有一项重要不确定性", deep["all1000_grade_counts"].get("B", 0)), ("C：存在主要障碍", deep["all1000_grade_counts"].get("C", 0)), ("D：硬矛盾或不可实现", deep["all1000_grade_counts"].get("D", 0))]
    literature_items = [("未发现精确报告", deep["all1000_literature_counts"].get("no_exact_report_found", 0)), ("间接/同家族", deep["all1000_literature_counts"].get("indirect_or_family_only", 0)), ("直接反证", deep["all1000_literature_counts"].get("contradictory", 0)), ("功能关系", deep["all1000_literature_counts"].get("functional_only", 0)), ("精确已验证", deep["all1000_literature_counts"].get("exact_pair_validated", 0))]
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">06 · 1000条统一综合审阅</h2>',
            '<div class="section-subtitle">所有候选使用同一套字段、同一证据标准和同一A/B/C/D定义；不区分审核批次。</div>',
            '<div class="grid-2">',
            '<div><h3>实验可行性分级</h3>', bar_chart(review_grade_items, 1000, "#16766f"), "</div>",
            '<div><h3>精确pair文献结论</h3>', bar_chart(literature_items, 1000, "#9a7a38"), "</div></div>",
            '<h3>每条pair统一检查六个维度</h3>',
            '<div class="grid-3">',
            '<div class="card"><div class="card-title">实体</div><p>母体、盐型、前药、活性代谢物、立体化学与计算SMILES是否一致。</p></div>',
            '<div class="card"><div class="card-title">直接证据</div><p>必须是同一药物实体、同一人源靶点和可解释的binding/biochemical/functional assay。</p></div>',
            '<div class="card"><div class="card-title">人体暴露</div><p>优先游离暴露；不把总Cmax直接等同于体外可达浓度。</p></div>',
            '<div class="card"><div class="card-title">实验方案</div><p>明确primary readout、阳性对照、已知靶点反筛和干扰门。</p></div>',
            '<div class="card"><div class="card-title">作用方向</div><p>模型只提出结合假说，不预设激动、抑制或调节方向。</p></div>',
            '<div class="card"><div class="card-title">疾病边界</div><p>药物-疾病证据不足时只保留target-only，不扩写适应症。</p></div>',
            "</div>",
            '<div class="callout teal"><b>进入Top500的统一硬门：</b>排除D级、直接反证、精确已验证pair、前药/活性代谢物待重跑、活性实体不确定和数据库身份未解决项。</div>',
            "</section>",
        ]
    )

    family_items = [("酶", top["assay_family_counts"].get("enzyme", 0)), ("核受体/表观遗传", top["assay_family_counts"].get("nuclear_epigenetic", 0)), ("转运体", top["assay_family_counts"].get("transporter", 0)), ("激酶", top["assay_family_counts"].get("kinase", 0)), ("离子通道", top["assay_family_counts"].get("ion_channel", 0))]
    execution_items = [("T1 标准实验优先", top["execution_tier_counts"].get("T1_immediate_standard_assay", 0)), ("T2 标准实验+主要不确定性", top["execution_tier_counts"].get("T2_standard_assay_major_uncertainty", 0)), ("T3 专门实验体系", top["execution_tier_counts"].get("T3_specialized_assay_or_active_species_resolution", 0))]
    role_items = [("未见精确公开记录", top["candidate_role_counts"].get("novel_binding_hypothesis", 0)), ("文献线索待核", top["candidate_role_counts"].get("literature_triage_hypothesis", 0)), ("功能关系复验", top["candidate_role_counts"].get("functional_record_revalidation", 0)), ("已报道pair复验", top["candidate_role_counts"].get("reported_pair_revalidation", 0))]
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">07 · 正式Top500质量画像</h2>',
            '<div class="section-subtitle">物理强度、实验执行难度与文献新颖性是三个独立维度，不能合并为单一“可信概率”。</div>',
            '<div class="metrics">',
            metric("物理A档", str(top["physics_strength_counts"].get("A_at_or_above_known_positive_median", 0)), "≥已知阳性family median"),
            metric("物理B档", str(top["physics_strength_counts"].get("B_at_or_above_known_positive_q25", 0)), "≥已知阳性family q25"),
            metric("稳定pose A", str(pose_counts.get("A_stable_conditional_pose", 0)), "第二构象位置与界面一致"),
            metric("稳定pose B", str(pose_counts.get("B_moderate_conditional_pose", 0)), "中等一致，仍可验证"),
            "</div>",
            '<div class="grid-2">',
            '<div><h3>靶点实验类型</h3>', bar_chart(family_items, 500, "#16766f"), "</div>",
            '<div><h3>实验执行层级</h3>', bar_chart(execution_items, 500, "#9a7a38"), "</div></div>",
            '<h3>文献角色</h3>', bar_chart(role_items, 500, "#4f6f8b"),
            '<div class="callout"><b>C级候选仍可进入Top500的原因：</b>本项目目标是亲和优先的广泛发现。C表示暴露、选择性或assay存在主要不确定性，不表示物理计算无信号；因此按T1/T2/T3分层，而不是将500条包装为同等成熟。</div>',
            "</section>",
        ]
    )

    disease_rows = [
        ["仅靶点假说，不提出病种", repurpose_counts.get("target_only_no_disease_claim", 0), "药物作用方向或药物-疾病证据不足"],
        ["新治疗领域假说", repurpose_counts.get("new_disease_area", 0), "与FDA原治疗领域不同，仍需命中后复核"],
        ["同领域新适应症假说", repurpose_counts.get("new_indication_same_area", 0), "同大类疾病中的新病种"],
        ["原适应症/非新用", repurpose_counts.get("original_indication_or_not_repurposing", 0), "主要作为机制或实验参照"],
    ]
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">08 · 疾病信息的正确位置</h2>',
            '<div class="section-subtitle">本报告坚持“先验证结合，再判断疾病价值”；疾病图谱不参与亲和主排序。</div>',
            table(["疾病解释层", "Top500数量", "含义"], disease_rows, "compact"),
            '<div class="grid-2">',
            '<div class="card"><div class="card-title">Open Targets补充什么</div><ul class="bullet-list">',
            '<li>靶点-疾病遗传、临床、通路与文献证据。</li>',
            '<li>靶点tractability、组织表达和已知临床开发语境。</li>',
            '<li>为实验命中后的适应症收敛提供候选方向。</li>',
            "</ul></div>",
            '<div class="card"><div class="card-title">Open Targets不证明什么</div><ul class="bullet-list">',
            '<li>不证明当前FDA药物能够调控该靶点。</li>',
            '<li>不证明药物作用方向与疾病机制同向。</li>',
            '<li>不将靶点-疾病证据自动转成药物新适应症。</li>',
            "</ul></div></div>",
            '<h3>2026热门靶点覆盖</h3>',
            f'<div class="tag-row">{"".join(f"<span class=\"tag\">{esc(gene)}</span>" for gene in top["hot_target_genes"])}</div>',
            f'<p>Top500中共有<b>{top["hot_target_rows"]}</b>条候选命中<b>{top["hot_target_unique_genes"]}</b>个热门靶点。热门标签仅用于关注度说明，不对物理主分加分。</p>',
            "</section>",
        ]
    )

    assay_rows = [
        ["酶", "重组酶活/LC-MS或正交底物", "已知高质量抑制剂/底物", "近邻酶、已知FDA靶点、聚集/光学干扰"],
        ["激酶", "酶活 + binding/占有", "已知选择性抑制剂", "kinome近邻、原靶点、ATP竞争依赖"],
        ["核受体/表观遗传", "配体置换/共因子招募/酶活", "已知激动剂或拮抗剂", "GR/PR/AR等近邻、报告基因和细胞毒性"],
        ["转运体", "摄取/外排或竞争结合", "已知底物/抑制剂", "被动扩散、膜毒性、同家族转运体"],
        ["离子通道", "膜片钳/通量", "已知blocker/agonist", "膜完整性、近邻通道、非特异电生理效应"],
    ]
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">09 · 实验执行框架</h2>',
            '<div class="section-subtitle">第一阶段验证“是否存在可重复的直接或靶点功能互作”，不直接验证疾病疗效。</div>',
            '<div class="grid-3">',
            '<div class="card"><div class="card-title">T1 · 68条</div><p>标准酶学、结合或成熟报告体系可立即执行；A/B可行性为主。</p></div>',
            '<div class="card"><div class="card-title">T2 · 382条</div><p>标准assay可做，但暴露、选择性、效价或机制方向存在主要不确定性。</p></div>',
            '<div class="card"><div class="card-title">T3 · 50条</div><p>需要膜制剂、转运、电生理或其他专门体系，建议集中按靶点批量测试。</p></div>',
            "</div>",
            table(["靶点类型", "Primary readout", "阳性对照", "必要反筛与门控"], assay_rows, "compact"),
            '<div class="callout teal"><b>统一go/no-go：</b>剂量依赖、独立重复、正交readout一致；已知靶点反筛和活力/膜完整性不能解释主要效应。命中后再做作用方向、选择性、疾病机制和临床暴露的第二阶段研究。</div>',
            "</section>",
        ]
    )

    rep_rows = []
    for _, row in representative.iterrows():
        rep_rows.append(
            [
                int(row["top500_rank_v6"]),
                truncate(row["drug_names"], 27),
                row["primary_gene"],
                family_zh(row["target_assay_family"]),
                row["feasibility_grade_v6"],
                f"{float(row['v5_pair_physics_score']):.1f}",
                role_zh(row["candidate_role_v6"]),
                truncate(row["review_verdict_v6"], 66),
            ]
        )
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">10 · T1代表性候选</h2>',
            '<div class="section-subtitle">按正式Top500顺序展示前9条T1；均是实验优先级，不是已确认新靶点。</div>',
            table(["总排名", "FDA药物", "靶点", "类型", "可行性", "物理分", "文献角色", "综合判断"], rep_rows, "tiny"),
            "</section>",
        ]
    )

    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">11 · 结果解释与限制</h2>',
            '<div class="section-subtitle">本项目已经把设计空间压缩为可实验假说，但最终真值只能由实验产生。</div>',
            '<div class="grid-2">',
            '<div class="card"><div class="card-title">当前结果的价值</div><ul class="check-list">',
            '<li>从完整项目空间重建，不再依赖Top300派生池。</li>',
            '<li>所有最终pair均满足活性实体、口袋、物理档和结构一致性硬门。</li>',
            '<li>每条候选都有可执行readout与反筛，可直接形成实验队列。</li>',
            '<li>通过药物、靶点和骨架上限维持广筛发现能力。</li>',
            "</ul></div>",
            '<div class="card"><div class="card-title">仍然存在的未知</div><ul class="bullet-list">',
            '<li>无大规模真实阴性集，不能可靠估计precision/FDR。</li>',
            '<li>Boltz/ConPLEx无法独立证明选择性或细胞内target engagement。</li>',
            '<li>游离暴露数据不完整，尤其组织与细胞内浓度。</li>',
            '<li>疾病疗效、作用方向与安全窗必须在命中后另行判断。</li>',
            "</ul></div></div>",
            '<h3>正式输出口径</h3>',
            '<div class="callout teal">Top500 = “满足统一硬门、具有相对较强计算物理支持、并已定义实验验证路径的药物-新靶点假说”。不能表述为“500个预测真结合”或“500个老药新用适应症”。</div>',
            '<div class="source-box">配套文件：FINAL500_PHYSICS_FIRST_FULL_DEEP_REVIEWED_V7.csv；FINAL500_TEACHER_READABLE_ZH_V7.csv；FINAL500_FULL_DEEP_REVIEW_PACKAGE_V7.xlsx；FINAL1000_FULL_DEEP_REVIEWED_V7.csv。</div>',
            "</section>",
        ]
    )

    appendix_rows = []
    for _, row in t1.iterrows():
        readout = primary_readout_zh(row["target_assay_family"])
        appendix_rows.append(
            [
                int(row["top500_rank_v6"]),
                truncate(row["drug_names"], 30),
                row["primary_gene"],
                family_zh(row["target_assay_family"]),
                row["feasibility_grade_v6"],
                f"{float(row['v5_pair_physics_score']):.1f}",
                int(float(row["rank_within_drug"])),
                readout,
            ]
        )
    parts.extend(
        [
            '<section class="section">',
            '<h2 class="section-title">附录 · T1标准实验优先队列（68条）</h2>',
            '<div class="section-subtitle">完整Top500及全部证据字段见配套CSV/Excel；本附录只列最适合首先启动标准assay的候选。</div>',
            table(["排名", "FDA药物", "靶点", "类型", "档", "物理分", "药内名次", "Primary readout"], appendix_rows, "tiny appendix-table"),
            '<div class="source-box">数据版本：V7；Top500 SHA-256：' + esc(v7["sha256"]["top500"]) + '</div>',
            "</section>",
        ]
    )

    parts.extend(["</body></html>"])
    html_text = "".join(parts)
    report_meta = {
        "top500_rows": int(len(top500)),
        "all1000_rows": int(len(all1000)),
        "t1_rows": int(len(t1)),
        "raw_pair_space": int(scope["pair_spaces"]["raw_915_x_891"]),
        "model_pair_space": int(scope["pair_spaces"]["unique_model_ligands_x_targets"]),
        "eligible_rows": int(universe["eligible_rows"]),
        "top3000_rows": int(formal["top3000_rows"]),
    }
    return html_text, report_meta


def main() -> None:
    required = [TOP500, ALL1000, V7_AUDIT, SELECTION_AUDIT, SCOPE_AUDIT, UNIVERSE_SUMMARY, KNOWN_BOLTZ, FORMAL_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")
    html_text, meta = build_html()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML(string=html_text, base_url=str(ROOT)).write_pdf(OUTPUT_PDF)

    doc = fitz.open(OUTPUT_PDF)
    page_text = [page.get_text("text").strip() for page in doc]
    blank_pages = [index + 1 for index, text in enumerate(page_text) if len(text) < 40]
    required_phrases = [
        "1000条统一综合审阅",
        "正式Top500质量画像",
        "T1标准实验优先队列",
        "815,265",
        "334,749",
    ]
    full_text = "\n".join(page_text)
    phrase_checks = {phrase: phrase in full_text for phrase in required_phrases}
    if blank_pages or not all(phrase_checks.values()):
        raise RuntimeError(f"PDF validation failed: blank_pages={blank_pages}, phrase_checks={phrase_checks}")
    audit = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_pdf": str(OUTPUT_PDF.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT_PDF),
        "page_count": len(doc),
        "blank_pages": blank_pages,
        "required_phrase_checks": phrase_checks,
        "report_meta": meta,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in required},
    }
    OUTPUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

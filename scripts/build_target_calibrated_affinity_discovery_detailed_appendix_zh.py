#!/usr/bin/env python3
"""Build the detailed numerical and decision appendix for the affinity workflow."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/current_production_package_v2/project_progress_discussion_zh"
PDF_OUT = OUT_DIR / "FDA_OLD_DRUG_TARGET_CALIBRATED_AFFINITY_DISCOVERY_DETAILED_APPENDIX_ZH.pdf"
HTML_OUT = OUT_DIR / "FDA_OLD_DRUG_TARGET_CALIBRATED_AFFINITY_DISCOVERY_DETAILED_APPENDIX_ZH.html"
AUDIT_OUT = OUT_DIR / "FDA_OLD_DRUG_TARGET_CALIBRATED_AFFINITY_DISCOVERY_DETAILED_APPENDIX_AUDIT.json"
ROOT_PDF_OUT = ROOT / PDF_OUT.name
PROJECT_NAME = "FDA 老药新靶点：靶点内校准亲和发现流程"


SOURCES = {
    "scope_inventory": ROOT / "outputs/evaluation_candidate_inventory_20260713/candidate_scope_inventory.csv",
    "historical_criteria": ROOT / "outputs/evaluation_candidate_inventory_20260713/historical_screening_criteria_assessment.csv",
    "strict895": ROOT / "outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_final_wetlab_priority_summary.json",
    "physics_v1": ROOT / "outputs/final_1000_funnel_v1/funnel_summary.json",
    "enhanced_v1": ROOT / "outputs/chembl_moa_enhanced_information_package_v1/enhanced_funnel_summary.json",
    "reselection_v2": ROOT / "outputs/current_production_package_v2/stage1_106k_reselection/stage1_106k_reselection_summary_v2.json",
    "full_v4": ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v4/full_untruncated_universe_v4_summary.json",
    "formal_v4": ROOT / "outputs/current_production_package_v2/formal_full_universe_v4/formal_completion_summary_v4_complete.json",
    "remote_v1": ROOT / "outputs/affinity_first_remote_discovery_v1/AFFINITY_FIRST_REMOTE_DISCOVERY_V1_SUMMARY.json",
    "homology_v1": ROOT / "outputs/affinity_first_remote_discovery_v1/REMOTE_TARGET_HOMOLOGY_AUDIT_V1_SUMMARY.json",
    "remote_config": ROOT / "configs/affinity_first_remote_discovery_v1.yaml",
    "gate_config": ROOT / "configs/affinity_experiment_package_v8.yaml",
    "drugclip_input": ROOT / "outputs/affinity_first_remote_discovery_v1/drugclip_inputs_v1/DRUGCLIP_INPUT_PREPARATION_V1_SUMMARY.json",
    "drugclip_scores": ROOT / "outputs/affinity_first_remote_discovery_v1/drugclip_inference_v1/DRUGCLIP_PROJECT_INFERENCE_V1_SUMMARY.json",
    "stage2_30k": ROOT / "outputs/affinity_first_remote_discovery_v1/stage2_docking_queue_v1/STAGE2_DOCKING_QUEUE_V1_SUMMARY.json",
    "controls_v2": ROOT / "outputs/affinity_first_remote_discovery_v1/target_docking_calibration_v2/TARGET_DOCKING_CALIBRATION_V2_SUMMARY.json",
    "gnina_v2": ROOT / "outputs/affinity_first_remote_discovery_v1/gnina_target_calibration_v2/GNINA_TARGET_CALIBRATION_V2_SUMMARY.json",
    "target_gate": ROOT / "outputs/affinity_experiment_package_v8/target_calibration/GNINA_TARGET_CHANNEL_CALIBRATION_V8_SUMMARY.json",
    "discovery": ROOT / "outputs/affinity_experiment_package_v8/discovery_queue/GNINA_REMOTE_DISCOVERY_QUEUE_V8_SUMMARY.json",
    "expanded_controls": ROOT / "outputs/affinity_experiment_package_v8/expanded_controls_manifest/TARGET_DOCKING_CALIBRATION_V8_SUMMARY.json",
    "main_report_audit": ROOT / "outputs/current_production_package_v2/project_progress_discussion_zh/FDA_OLD_DRUG_NEW_TARGET_ROUTE_CHANGES_AND_CORE_CONTRADICTIONS_AUDIT.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def esc(value: object) -> str:
    return html.escape(str(value))


def pct(child: int, parent: int) -> str:
    return f"{100.0 * child / parent:.2f}%" if parent else "NA"


def table(headers: list[str], rows: list[list[object]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>"
        for row in rows
    )
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def metric(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        "</div>"
    )


def detail_card(title: str, text: str, tone: str = "") -> str:
    return f'<div class="detail-card {tone}"><h3>{esc(title)}</h3><p>{esc(text)}</p></div>'


def issue_page(index: int, item: dict[str, str]) -> str:
    return f"""
    <section class="page">
      <div class="section-kicker">问题详解 {index:02d} · {esc(item['category'])}</div>
      <h2>{esc(item['title'])}</h2>
      <div class="status-line"><span class="badge {esc(item['badge'])}">{esc(item['status'])}</span><span>{esc(item['one_line'])}</span></div>
      <div class="detail-grid">
        {detail_card('严格定义', item['definition'])}
        {detail_card('为什么重要', item['impact'], 'warn')}
        {detail_card('当前已有证据', item['evidence'], 'good')}
        {detail_card('边界或剩余缺口', item['gap'], 'warn')}
        {detail_card('当前决定或会上需冻结', item['decision'], 'blue')}
        {detail_card('验收标准与交付物', item['acceptance'], 'good')}
      </div>
      <div class="owner"><strong>责任边界：</strong>{esc(item['owner'])}</div>
    </section>
    """


def meeting_page(index: int, item: dict[str, str]) -> str:
    bullets = "".join(f"<li>{esc(x.strip())}</li>" for x in item["fields"].split("|") if x.strip())
    return f"""
    <section class="page">
      <div class="section-kicker">会议问题 {index:02d} · {esc(item['group'])}</div>
      <h2>{esc(item['title'])}</h2>
      <p class="lead">{esc(item['question'])}</p>
      <div class="detail-grid meeting-grid">
        {detail_card('为什么必须回答', item['why'], 'warn')}
        {detail_card('当前可以确定的回答', item['current'], 'good')}
        {detail_card('仍需实验组明确', item['needed'], 'blue')}
        {detail_card('若不明确的后果', item['risk'], 'warn')}
      </div>
      <div class="field-box"><h3>需要写入表格的字段</h3><ul>{bullets}</ul></div>
      <div class="decision-strip"><strong>会议应形成的决定：</strong>{esc(item['decision'])}</div>
    </section>
    """


def source_row(label: str, path: Path) -> list[str]:
    return [label, str(path.relative_to(ROOT)), sha256(path)[:16] + "…"]


def main() -> None:
    for label, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source {label}: {path}")

    strict895 = read_json(SOURCES["strict895"])
    physics_v1 = read_json(SOURCES["physics_v1"])
    enhanced_v1 = read_json(SOURCES["enhanced_v1"])
    reselection_v2 = read_json(SOURCES["reselection_v2"])
    full_v4 = read_json(SOURCES["full_v4"])
    formal_v4 = read_json(SOURCES["formal_v4"])
    remote = read_json(SOURCES["remote_v1"])
    homology = read_json(SOURCES["homology_v1"])
    drugclip_input = read_json(SOURCES["drugclip_input"])
    drugclip = read_json(SOURCES["drugclip_scores"])
    stage2 = read_json(SOURCES["stage2_30k"])
    controls = read_json(SOURCES["controls_v2"])
    gnina = read_json(SOURCES["gnina_v2"])
    target_gate = read_json(SOURCES["target_gate"])
    discovery = read_json(SOURCES["discovery"])
    expanded = read_json(SOURCES["expanded_controls"])
    main_audit = read_json(SOURCES["main_report_audit"])

    # Set-accounting assertions. They distinguish true partitions from parallel views.
    checks: dict[str, bool] = {}
    checks["id_cartesian"] = full_v4["project_cartesian_rows"] == 750 * 463
    checks["physical_cartesian"] = full_v4["project_physical_pair_rows"] == 723 * 463
    checks["physical_normalization_delta"] = (
        full_v4["project_cartesian_rows"] - full_v4["project_physical_pair_rows"] == 12501
    )
    queue_counts = remote["queue_counts"]
    checks["current_partition"] = sum(queue_counts.values()) == remote["physical_pair_rows"]
    checks["remote_partition"] = (
        queue_counts["remote_structure_strict"]
        + queue_counts["remote_sequence_dta_only"]
        + queue_counts["remote_structure_review"]
        == remote["remote_pair_eligible_rows"]
    )
    checks["remote_plus_controls"] = (
        remote["remote_pair_eligible_rows"]
        + queue_counts["family_extension_control"]
        + queue_counts["similarity_rediscovery_control"]
        + queue_counts["hold_structure_or_novelty"]
        + queue_counts["known_positive_calibration"]
        == remote["physical_pair_rows"]
    )
    checks["drugclip_lineage"] = (
        homology["strict_structure_remote_homology_audited_rows"]
        == drugclip["remote_strict_rows"]
    )
    checks["stage2_input"] = drugclip["remote_strict_scored_rows"] == stage2["input_remote_scored_rows"]
    checks["discovery_input"] = stage2["selected_rows"] == discovery["source_queue_rows"]
    checks["receptor_partition"] = sum(discovery["receptor_source_counts"].values()) == discovery["selected_rows"]
    checks["control_partition"] = controls["positive_control_rows"] + controls["negative_control_rows"] == controls["control_rows"]
    if not all(checks.values()):
        raise RuntimeError(f"Numerical lineage checks failed: {checks}")

    css = """
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc); }
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc); font-weight:700; }
    @page {
      size:A4; margin:14mm 15mm 14mm 15mm;
      @top-left { content:"FDA老药新靶点 · 详细附录"; color:#697270; font:8pt NotoCJK; }
      @top-right { content:string(section); color:#697270; font:8pt NotoCJK; }
      @bottom-left { content:"靶点内校准亲和发现流程"; color:#7c8583; font:7.5pt NotoCJK; }
      @bottom-right { content:counter(page) " / " counter(pages); color:#7c8583; font:7.5pt NotoCJK; }
    }
    @page:first { @top-left{content:none;} @top-right{content:none;} @bottom-left{content:none;} }
    * { box-sizing:border-box; }
    body { margin:0; font-family:NotoCJK,sans-serif; color:#182321; font-size:9.15pt; line-height:1.52; }
    h1,h2,h3,p { margin-top:0; }
    h1 { font-size:27pt; line-height:1.28; letter-spacing:0; margin:0 0 7mm; }
    h2 { string-set:section content(); font-size:19pt; line-height:1.3; border-bottom:2px solid #147b73; padding-bottom:3mm; margin:0 0 4.5mm; }
    h3 { font-size:11.5pt; line-height:1.4; margin-bottom:1.4mm; }
    p { margin-bottom:2.8mm; }
    ul { padding-left:5mm; margin:1.5mm 0 3mm; }
    li { margin-bottom:1mm; }
    .page { min-height:267mm; page-break-after:always; }
    .page:last-child { page-break-after:auto; }
    .cover { position:relative; padding-top:26mm; }
    .cover::before { content:""; position:absolute; top:0; left:0; width:44mm; height:4mm; background:#bd6d4f; }
    .eyebrow { color:#147b73; font-size:10.5pt; font-weight:700; margin-bottom:6mm; }
    .subtitle { font-size:13.7pt; color:#465451; line-height:1.65; max-width:158mm; }
    .cover-rule { border-top:1px solid #cbd4d1; margin:11mm 0 8mm; }
    .cover-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:5mm; }
    .cover-grid div { border-top:3px solid #147b73; padding-top:3mm; }
    .cover-grid strong { display:block; font-size:10.5pt; margin-bottom:1mm; }
    .cover-grid span { color:#687370; font-size:8.2pt; }
    .cover-callout { margin-top:12mm; background:#f4f1ef; border-left:4px solid #bd6d4f; padding:5mm 6mm; color:#4d4845; }
    .cover-foot { margin-top:13mm; color:#78817f; font-size:8.2pt; }
    .lead { font-size:11.7pt; line-height:1.68; color:#344744; }
    .section-kicker { color:#147b73; font-size:8.2pt; font-weight:700; margin-bottom:2mm; }
    .callout { border-left:4px solid #bd6d4f; background:#f4f1ef; padding:4mm 5mm; margin:4mm 0; }
    .callout.teal { border-left-color:#147b73; background:#eef4f2; }
    .callout.blue { border-left-color:#5277a2; background:#f1f4f8; }
    .callout strong { color:#823f2b; }
    .callout.teal strong { color:#126d67; }
    .callout.blue strong { color:#365c84; }
    .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; margin:4mm 0; }
    .metric { border:1px solid #d4dcda; border-top:3px solid #147b73; padding:3.5mm; min-height:27mm; background:#fbfcfb; }
    .metric-label { color:#65706d; font-size:8pt; }
    .metric-value { color:#173f3a; font-size:18pt; line-height:1.25; font-weight:700; margin:1mm 0; }
    .metric-note { color:#78827f; font-size:7.5pt; }
    table { width:100%; border-collapse:collapse; margin:3.5mm 0 5mm; font-size:8.05pt; }
    th { background:#203d39; color:#fff; padding:2.2mm; text-align:left; }
    td { border-bottom:1px solid #d9e0de; padding:2.2mm; vertical-align:top; }
    tr:nth-child(even) td { background:#f6f8f7; }
    .compact td, .compact th { padding:1.65mm 2mm; font-size:7.5pt; }
    .number td:nth-child(2), .number td:nth-child(3), .number td:nth-child(4) { text-align:right; white-space:nowrap; }
    .flow { display:grid; grid-template-columns:32mm 10mm 1fr; align-items:stretch; margin-bottom:2.5mm; page-break-inside:avoid; }
    .flow-num { background:#203d39; color:#fff; padding:3mm; font-weight:700; text-align:center; display:flex; align-items:center; justify-content:center; }
    .flow-arrow { display:flex; align-items:center; justify-content:center; color:#bd6d4f; font-size:15pt; font-weight:700; }
    .flow-body { border:1px solid #d4dcda; padding:2.8mm 3.5mm; }
    .flow-body strong { color:#126d67; }
    .branch-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:4mm; }
    .branch { border:1px solid #d4dcda; border-top:3px solid #147b73; padding:4mm; min-height:39mm; }
    .branch h3 { color:#126d67; }
    .branch p { margin-bottom:1mm; color:#52605d; }
    .equation { font-family:monospace; font-size:11pt; background:#f2f5f4; border:1px solid #d6dfdd; padding:4mm; margin:3mm 0; }
    .status-line { display:flex; align-items:center; gap:3mm; padding:3mm 4mm; background:#f3f6f5; margin-bottom:4mm; }
    .badge { display:inline-block; border-radius:2px; color:#fff; padding:1.2mm 2.8mm; font-weight:700; font-size:8pt; white-space:nowrap; }
    .badge.resolved { background:#147b73; }
    .badge.retained { background:#a45c3f; }
    .badge.new { background:#5277a2; }
    .detail-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:4mm; margin-top:3mm; }
    .detail-card { border:1px solid #d4dcda; border-top:3px solid #6d7d79; padding:4mm; min-height:49mm; page-break-inside:avoid; }
    .detail-card h3 { color:#344744; }
    .detail-card p { margin:0; color:#4f5e5b; }
    .detail-card.warn { border-top-color:#bd6d4f; background:#fbf7f5; }
    .detail-card.warn h3 { color:#8b4d39; }
    .detail-card.good { border-top-color:#147b73; background:#f5f9f8; }
    .detail-card.good h3 { color:#126d67; }
    .detail-card.blue { border-top-color:#5277a2; background:#f5f7fa; }
    .detail-card.blue h3 { color:#365c84; }
    .owner { margin-top:4mm; border-left:4px solid #203d39; padding:3mm 4mm; background:#f2f4f3; }
    .owner strong { color:#203d39; }
    .meeting-grid .detail-card { min-height:43mm; }
    .field-box { margin-top:4mm; border:1px solid #d4dcda; padding:4mm; }
    .field-box h3 { color:#126d67; }
    .field-box ul { columns:2; column-gap:8mm; }
    .decision-strip { margin-top:4mm; border-left:4px solid #5277a2; background:#f1f4f8; padding:4mm 5mm; }
    .decision-strip strong { color:#365c84; }
    .legend { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; margin:4mm 0; }
    .legend div { padding:4mm; border:1px solid #d4dcda; }
    .legend strong { display:block; color:#126d67; margin-bottom:1mm; }
    .source { font-size:7.1pt; word-break:break-all; }
    .small { color:#697370; font-size:7.7pt; }
    """

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    pages: list[str] = []
    pages.append(f"""
    <section class="page cover">
      <div class="eyebrow">FDA 老药新靶点 · 靶点内校准亲和发现流程</div>
      <h1>全局漏斗、历代口径<br>与关键问题详细附录</h1>
      <p class="subtitle">本附录将历史候选包与当前正式流程拆开核算，逐级说明输入、筛选规则、差额去向、科学含义和未完成条件，并对全部已解决、保留和新增问题给出可执行定义。</p>
      <div class="cover-rule"></div>
      <div class="cover-grid">
        <div><strong>数值主链</strong><span>所有可串联节点通过集合加减和来源文件校验</span></div>
        <div><strong>历史旁支</strong><span>不同母集、目标与计数单位不再强行串联</span></div>
        <div><strong>问题详解</strong><span>定义、证据、缺口、决定、验收和责任边界</span></div>
      </div>
      <div class="cover-callout"><strong>阅读原则</strong><br>箭头只用于真实子集关系；“转入控制队列”“技术未完成”“容量截断”与“质量删除”分别标记。当前流程到 16,995 条 Discovery 队列为止已有数值承接，后续数字必须等待 Gate-B、Pair gate 和实验能力回传。</div>
      <div class="cover-foot">独立详细附录 · 中文版 · {generated}</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">阅读说明</div><h2>一、这份附录解决什么</h2>
      <p class="lead">过去的全局漏斗把不同项目阶段的数字放在一条箭头上，造成“后一级是否真是前一级子集”“为什么会从 895 又回到 3,000”“1000 与 384 是否代表证据阈值”等问题。本附录重新建立集合账本。</p>
      <div class="legend">
        <div><strong>主链</strong>同一母集、同一 pair 定义，可直接计算保留率和差额。</div>
        <div><strong>旁支</strong>共享部分上游，但目标或筛选逻辑不同，只能比较交集。</div>
        <div><strong>侧车校准</strong>阳性/阴性和结构质控用于评估协议，不是 Discovery 候选。</div>
      </div>
      {table(['符号', '含义', '允许的解释'], [
          ['→', '严格子集或同一集合的状态推进', '可以计算保留率'],
          ['⇢', '单位归一化或实体折叠', '差额不是模型淘汰'],
          ['↘', '转入旁支、控制或待审队列', '不能称为被判定不结合'],
          ['TBD', '依赖尚未回传的数据', '禁止用固定规模补数'],
      ])}
      <div class="callout teal"><strong>当前可审计结论：</strong>正式物理空间是 723 × 463 = 334,749 对；当前 Discovery 队列是 16,995 对。二者之间的每一处差额已经能解释为实体归一、控制分流、结构状态、模型准备失败、计算配额或 Target gate，而不是笼统的“综合评分下降”。</div>
      <div class="callout"><strong>仍不能给出的数字：</strong>Gate-B 后剩多少、哪些 P1/P2/P3 可进入实验、171 个靶点中多少 assay-ready、以及最终是否恰好为 1000，当前都没有自然证据阈值。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">计数基础</div><h2>二、统一计数单位</h2>
      {table(['单位', '当前实例', '用途', '不能混用的原因'], [
          ['drug record', '915 条 FDA 结构记录', '最早输入与全序列 ConPLEx', '包含不同批准记录、盐型或归一化前重复'],
          ['project drug ID', '750 个项目药物 ID', 'ID 审计和适应症追踪', '同一物理结构可对应多个 ID'],
          ['physical model ligand', '723 个 active-moiety 结构', 'ConPLEx、DrugCLIP、docking 的真实输入', '这是物理 pair 的正确药物分母'],
          ['protein row', '5,307 个早期 ChEMBL/UniProt 行', '历史粗空间', '同蛋白、多 accession 和重复行导致膨胀'],
          ['unique sequence', '891 条唯一序列；项目主线 463 条', '避免相同蛋白重复计算', 'gene、accession、sequence 不能默认一一对应'],
          ['physical pair', 'model ligand × sequence', '当前主筛单位', '同一 pair 可映射多个药物 ID'],
          ['hypothesis row', 'drug-target-disease', 'strict895 疾病机制审阅', '同一物理 pair 可因疾病方向产生多行'],
          ['assay measurement', '尚未定义', '实验预算', 'pair、曲线、孔位和进样不是同一成本单位'],
      ], 'compact')}
      <div class="equation">750 × 463 = 347,250 个 ID-pair　⇢　723 × 463 = 334,749 个 physical pair</div>
      <p>两者相差 12,501 行，来自 27 个 ID 在 active-moiety/结构层面的折叠效应。这个差额是输入实体归一化，不是筛选器淘汰，也不能用于计算模型精度。</p>
      <div class="callout blue"><strong>附录统一约定：</strong>除非特别标记为 ID-pair 或 hypothesis，当前漏斗中的“pair”均指 physical model ligand × unique target sequence。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">数值承接</div><h2>三、哪些数字可以相加减</h2>
      {table(['关系', '核对式', '结果'], [
          ['ID 空间', '750 × 463', f"{full_v4['project_cartesian_rows']:,}"],
          ['物理空间', '723 × 463', f"{full_v4['project_physical_pair_rows']:,}"],
          ['物理空间完整分区', '310,544 + 17,361 + 4,750 + 1,368 + 726', f"{sum(queue_counts.values()):,}"],
          ['remote 空间分区', '202,725 + 54,764 + 53,055', f"{remote['remote_pair_eligible_rows']:,}"],
          ['历史控制分区', '4,601 + 4,598', f"{controls['control_rows']:,}"],
          ['Discovery 受体来源', '13,529 + 3,466', f"{discovery['selected_rows']:,}"],
      ], 'number')}
      <h3>不能直接相减的例子</h3>
      {table(['数字组合', '为什么不能串联'], [
          ['12,696 → 895', '12,696 是旧文献/A-B-C-D 推荐与审计池；895 来自 strict_top_ready + ConPLEx floor，只有 267 条覆盖交集。'],
          ['895 → 3,000', '3,000 是另一条 physics-first 计算配额，母集回到 106,561 或 334,749，不是 895 的扩增。'],
          ['1,000 → 384', '在部分历史版本中是容量子集；不同版本的 1000 和 384 由不同评分、审阅和多样性规则生成。'],
          ['234 → 171', '234 是全体已评分靶点上的 Target gate 并集；171 是它与 30,000 Discovery 计算队列实际靶点的交集。'],
      ])}
      <div class="callout"><strong>审核规则：</strong>今后每个漏斗节点必须同时记录母集 ID、计数单位、规则版本、输入 SHA256、输出 SHA256，以及移除行的处置标签。</div>
    </section>
    """)

    # Historical route overview.
    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史总览</div><h2>四、历代漏斗不是一条直线</h2>
      <div class="branch-grid">
        <div class="branch"><h3>分支 A：Top300 与疾病机制</h3><p>4,855,905 行级空间 → 815,265 唯一序列空间 → 695,400 非 GPCR → 274,500 Top300 → 106,561 discovery → strict895。</p><p><strong>目标：</strong>宽召回后用疾病证据收敛。</p></div>
        <div class="branch"><h3>分支 B：文献与固定包</h3><p>12,696 文献/A-B-C-D 审计池，以及 2,000/1,000/384 等容量化交付包。</p><p><strong>目标：</strong>文献排重、可读性和组合比例。</p></div>
        <div class="branch"><h3>分支 C：106k 物理优先</h3><p>106,561 → 66,924 → 3,000 → 1,000；后续加入 Open Targets tractability 与 Boltz。</p><p><strong>目标：</strong>直接 target-engagement 与结构可做性。</p></div>
        <div class="branch"><h3>分支 D：全空间物理重启</h3><p>334,749 → 264,352 → 3,000 → 1,000 → review512 → 384。</p><p><strong>目标：</strong>取消旧 Top300 截断并完成全量 Boltz/pose 审计。</p></div>
        <div class="branch"><h3>当前主链</h3><p>334,749 → remote/controls 严格分区 → 202,654 → 201,634 → 30,000 → 16,995。</p><p><strong>目标：</strong>低相似远程发现与靶点内历史校准。</p></div>
        <div class="branch"><h3>实验后半段</h3><p>Gate-B → Pair gate → assay-ready → pilot → broad screen → confirm binder。</p><p><strong>状态：</strong>规则和实验资源尚未冻结，暂不填数字。</p></div>
      </div>
      <div class="callout teal"><strong>当前唯一主线：</strong>“当前主链 + 实验后半段”。其余分支保留为方法演进、已知检索基线、文献排重或候选储备，不再并列为多个正式最终结果。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史分支 A</div><h2>五、Top300 与 strict895 漏斗</h2>
      {table(['级别', '数量', '保留率', '规则与含义'], [
          ['早期行级空间', '4,855,905', '100%', '915 × 5,307；含重复蛋白行，仅为粗空间'],
          ['唯一序列空间', '815,265', pct(815265, 4855905), '915 × 891；按蛋白序列去重'],
          ['非 GPCR 空间', '695,400', pct(695400, 815265), '915 × 760；GPCR 转入独立路线'],
          ['每药 Top300', '274,500', pct(274500, 695400), '每个药保留 ConPLEx 排名前 300；宽召回，不是质量层'],
          ['药物+靶点可做性', '120,256', pct(120256, 274500), '排除非治疗实体并限制 assay family'],
          ['direct-action 全池', '106,803', pct(106803, 120256), '要求可解释的 direct-action 标签，含 242 known controls'],
          ['discovery 106k', '106,561', pct(106561, 106803), '移出 242 个已知 pair'],
          ['strict_top_ready', '1,615', pct(1615, 106561), 'A_top5、非 ConPLEx 证据、rank 和 TxGNN 覆盖门'],
          ['strict895', '895', pct(895, 1615), '再要求 ConPLEx ≥ 0.20'],
      ], 'number compact')}
      <div class="callout"><strong>主要问题：</strong>Top300 在上游不可逆地删除了每个药物排名 300 以外的靶点；strict_top_ready 又把疾病图谱覆盖和 TxGNN 可用性纳入亲和候选资格。因此该漏斗适合解释历史 strict895 的来源，不再作为当前亲和发现主线。</div>
      <p class="small">历史已知 direct-action 审计：非 GPCR 可比较分母 408，Top300 找回 250，Recall=61.27%。该结果是历史校准，不是未知 pair 的 precision。</p>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史分支 A</div><h2>六、疾病机制层的并行档位</h2>
      {table(['输出', '数量', '构造方式', '当前定位'], [
          ['A_top5', '5,329', '19 个疾病方向综合分 Top 5%', '疾病解释高位层，不能证明直接结合'],
          ['A+B', '21,313', 'A_top5 + B_80_to_95', '宽机制候选集合'],
          ['basic_top_ready', '13,720', 'A/B + 非 ConPLEx 证据≥3 + 药物内 rank≤100 + 为固定规模校准的 broad score', '停用亲和主门'],
          ['strict_top_ready', '1,615', 'A_top5 + 证据≥3 + rank≤100 + TxGNN 可用', 'strict895 的直接父集'],
          ['very_strict_top_ready', '1,105', 'A_top5 + 证据≥4 + rank≤50 + TxGNN + OT>0.3', '并行更严审阅层，不是 895 的父集'],
          ['strict895', '895', 'strict_top_ready + ConPLEx≥0.20', '历史疾病机制审阅池'],
      ])}
      <div class="metrics">
        {metric('strict895 行', f"{strict895['all_rows']:,}", 'drug-target-disease 记录')}
        {metric('唯一 hypothesis', f"{strict895['unique_hypotheses']:,}", '去除重复疾病假说')}
        {metric('旧 12,696 覆盖', '267', '895 中与旧文献池的交集')}
      </div>
      <p>strict895 后形成 Top96/192/384，是审阅和实验容量截断，不是新的物理证据门。其文献状态中，626 个唯一 hypothesis 在当时 PubMed pair 审计中未报道，229 个被归为上市后新靶点报道；这些标签用于已知性，不等于实验结合概率。</p>
      <div class="callout blue"><strong>可复用部分：</strong>疾病方向、具体病种、作用方向、文献排重和 assay 备注保留到 confirmed hit 后；不再决定当前 334,749 物理 pair 的首轮亲和排序。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史分支 B</div><h2>七、12,696 与固定规模候选包</h2>
      <p class="lead">12,696 是旧 A/B/C/D 综合推荐与文献审计池。它的主表已归档，当前只能通过 strict895 中的覆盖字段确认交集，不能被放在 strict895 或 physics1000 的直接上游。</p>
      {table(['历史数字', '当时作用', '能否作为当前父集', '当前用途'], [
          ['12,696', '综合推荐、文献审计和报道比例统计', '否；母集和评分口径不同', '文献排重与 rediscovery 参考'],
          ['约 2,000', '非 GPCR A/B/C 与热门靶点补充等讨论性规模', '否；未形成统一 canonical lineage', '仅作为路线记录'],
          ['1,000', '多次作为教师可读包、物理储备包或 Boltz-priority 包', '需指定版本后才能比较', '旧包冻结，不代表统一高概率'],
          ['500', '深审或汇报容量讨论', '否；不是自然阈值', '人工审阅容量'],
          ['384', '四块 96 孔板的候选容量', '只有同版本文件内是 1000 的子集', '历史实验组织方案'],
      ])}
      <div class="equation">strict895 ∩ 旧12,696 = 267 条；strict895 其余 628 条并不等于“新增高质量发现”。</div>
      <div class="callout"><strong>数字治理结论：</strong>历史包名称必须带版本、母集和生成规则。仅用“1000”“384”指代文件会把不同候选表混为一谈。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史分支 C</div><h2>八、106k physics-first 漏斗</h2>
      {table(['级别', '数量', '相对上一级', '主要规则'], [
          ['discovery 106k', f"{physics_v1['discovery_input_rows']:,}", '100%', '旧 Top300 衍生的 direct-action discovery'],
          ['core physics eligible', f"{physics_v1['pool_core_physics_eligible_rows']:,}", pct(physics_v1['pool_core_physics_eligible_rows'], physics_v1['discovery_input_rows']), '结构 bin、口袋、ConPLEx、药物/实验可行性'],
          ['pre-Boltz 3,000', f"{physics_v1['pre_boltz_rows']:,}", pct(physics_v1['pre_boltz_rows'], physics_v1['pool_core_physics_eligible_rows']), '固定计算预算 + 多样性约束'],
          ['final 1,000', f"{physics_v1['final_rows']:,}", pct(physics_v1['final_rows'], physics_v1['pre_boltz_rows']), '综合物理分与固定 P1/P2/P3 配额'],
      ], 'number')}
      <div class="metrics">
        {metric('已知 control', f"{physics_v1['known_control_rows']:,}", '单独做漏斗保留审计')}
        {metric('core physics recall', '201 / 242', '83.06%，仅历史阳性')}
        {metric('ConPLEx/Top50 recall', '157 / 242', '64.88%，说明硬门会杀伤已知机制')}
      </div>
      <p>这条路线第一次明确把疾病证据后置，并引入口袋和结构可行性。但其母集已被旧 Top300 截断；Top3000 前 pair-specific 证据主要仍是 ConPLEx，口袋、tractability 和 assay family 多数是 target-level prior。</p>
      <div class="callout blue"><strong>增强版结果：</strong>同一 3,000 中完成 Boltz 2,988 条，A/B 支持 243 条；balanced final1000 中 A+B 193 条。该分档是模型启发式，不是校准后的 binder 概率。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史分支 C</div><h2>九、106k 重选与全空间过渡</h2>
      {table(['版本', '母集', 'eligible', 'Top3000', '核心变化'], [
          ['106k reselection', f"{reselection_v2['source_rows']:,}", f"{reselection_v2['eligible_rows']:,}", f"{reselection_v2['selected_rows']:,}", '重算 family/rediscovery 风险；Boltz 不参与 stage-1 选择'],
          ['v3 全空间过渡', '344,190', '未作为最终正式口径', '3,000', '取消每药 Top300；实体尚未完全 active-moiety 折叠'],
          ['v4 全空间', f"{full_v4['project_physical_pair_rows']:,}", f"{full_v4['eligible_rows']:,}", f"{full_v4['selected_rows']:,}", '723 active-moiety × 463；禁用旧 Boltz 复用'],
      ])}
      <div class="metrics">
        {metric('旧106k重选重叠', f"{reselection_v2['old_top3000_overlap']:,}", f"占新Top3000的 {reselection_v2['old_top3000_overlap_pct']}%")}
        {metric('v4 旧106k外候选', f"{full_v4['selected_outside_old_106k']:,}", 'Top3000 中仅21条')}
        {metric('旧Top300外空间', f"{full_v4['project_rows_previously_outside_top300_derived_106k']:,}", '说明旧106k覆盖不足')}
      </div>
      <p>虽然 v4 取消了 Top300 硬门，但新的综合预分数仍使 Top3000 中 2,979 条来自旧106k、仅21条来自旧106k外。这表明“取消硬门”并不自动消除排序偏置，必须另设远程化学与同家族风险控制。</p>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史分支 D</div><h2>十、full-universe v4 物理漏斗</h2>
      {table(['级别', '数量', '相对上一级', '含义'], [
          ['物理空间', f"{full_v4['project_physical_pair_rows']:,}", '100%', '723 active-moiety × 463 target'],
          ['eligible', f"{full_v4['eligible_rows']:,}", pct(full_v4['eligible_rows'], full_v4['project_physical_pair_rows']), '结构、direct-SM、风险和元数据硬门'],
          ['Top3000', f"{formal_v4['top3000_rows']:,}", pct(formal_v4['top3000_rows'], full_v4['eligible_rows']), '连续预分数 + drug/target/scaffold caps'],
          ['Boltz 完成', f"{formal_v4['refined_completed_rows']:,}", '100% of 3,000', '禁用历史复用，结果带输入签名'],
          ['final1000', f"{formal_v4['final1000_rows']:,}", pct(formal_v4['final1000_rows'], formal_v4['top3000_rows']), '结构证据、风险和多样性后的储备包'],
          ['review512', f"{formal_v4['agent_review_pool_rows']:,}", pct(formal_v4['agent_review_pool_rows'], formal_v4['final1000_rows']), '文献、活性物种、assay、机制与 agent 审阅'],
          ['reviewed384', f"{formal_v4['final384_rows']:,}", pct(formal_v4['final384_rows'], formal_v4['agent_review_pool_rows']), '容量化提名包'],
      ], 'number compact')}
      <div class="callout"><strong>为什么未成为最终可信实验包：</strong>Boltz A/B、pose stability 和综合分尚未在同靶点可靠正负数据上校准；1000 与384主要由计算预算和组合多样性产生，不能解释为自然真假边界。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">历史版本关系</div><h2>十一、历代候选包的当前处置</h2>
      {table(['候选包', '是否继续作为主结果', '保留价值', '禁止表述'], [
          ['Top300 274,500', '否', '历史宽召回与失败分析', '不能称为高质量候选'],
          ['strict895', '否', '疾病、文献、assay 注释和审阅经验', '不能称为亲和高置信'],
          ['旧12,696', '否', '文献排重和已报道 pair 参考', '不能与895串成父子漏斗'],
          ['106k physics final1000', '否', '旧物理规则消融和候选交叉比较', '不能称为1000个高概率binder'],
          ['v4 final1000', '历史储备', '全空间 Boltz/pose/agent 结果', '不能跨靶点比较原始 affinity'],
          ['v4 reviewed384', '历史容量子集', '实验可行性和深审样例', '不能当作当前正式实验板'],
          ['当前 16,995 Discovery', '是，当前计算发现队列', 'Target gate 后的靶点内候选来源', '不能称为16,995个真实binder'],
      ])}
      <div class="callout teal"><strong>使用方式：</strong>旧包不删除；在新的 pair decision card 中作为“历史是否入选、为何入选、当前为何保留或退出”的旁证字段，但不继承旧等级。</div>
    </section>
    """)

    # Current funnel.
    pages.append(f"""
    <section class="page">
      <div class="section-kicker">当前主链</div><h2>十二、当前正式漏斗总览</h2>
      <div class="flow"><div class="flow-num">347,250</div><div class="flow-arrow">⇢</div><div class="flow-body"><strong>ID 审计空间</strong><br>750 个药物 ID × 463 个靶点。active-moiety 折叠后转为物理输入。</div></div>
      <div class="flow"><div class="flow-num">334,749</div><div class="flow-arrow">→</div><div class="flow-body"><strong>正式物理空间</strong><br>723 个模型配体 × 463 个唯一序列靶点。完整分流到 remote、known、similarity、family 和 hold。</div></div>
      <div class="flow"><div class="flow-num">310,544</div><div class="flow-arrow">→</div><div class="flow-body"><strong>remote chemistry eligible</strong><br>排除 exact known、close chemistry、同家族传播和严重待审；仍按结构状态分三路。</div></div>
      <div class="flow"><div class="flow-num">202,654</div><div class="flow-arrow">→</div><div class="flow-body"><strong>homology-audited strict structure</strong><br>严格结构可做且通过扩展已知靶点同源风险审计。</div></div>
      <div class="flow"><div class="flow-num">201,634</div><div class="flow-arrow">→</div><div class="flow-body"><strong>ConPLEx + DrugCLIP 双通道可评分</strong><br>1 个 ligand 和 1 个 pocket 准备失败导致 1,020 条 remote pair 无 DrugCLIP 分数。</div></div>
      <div class="flow"><div class="flow-num">30,000</div><div class="flow-arrow">→</div><div class="flow-body"><strong>结构计算资源队列</strong><br>覆盖优先、证据 lane 和 target/ligand/scaffold caps；不是质量阈值。</div></div>
      <div class="flow"><div class="flow-num">16,995</div><div class="flow-arrow">→</div><div class="flow-body"><strong>Target gate 后 Discovery</strong><br>171 靶点、717 配体、580 骨架；只在同靶点历史校准合格的协议内解释。</div></div>
      <div class="callout blue"><strong>下一节点：</strong>Gate-B 与 Pair gate 仍未冻结；因此当前不能合理写成“16,995 → 1,000”。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">当前第 1 级</div><h2>十三、药物与靶点实体冻结</h2>
      <div class="metrics">
        {metric('FDA 结构记录', '915', '早期输入，不等于当前模型配体')}
        {metric('项目药物 ID', '750', '保留批准与适应症追踪')}
        {metric('模型配体', '723', 'active-moiety/结构归一后')}
      </div>
      {table(['硬门', '处理', '理由', '是否提供 pair 证据'], [
          ['气体、诊断/显像、非治疗分子', '退出主线', '不符合常规治疗性小分子直接结合任务', '否，仅定义项目边界'],
          ['polymer/resin/sequestrant/物理吸附', '退出主线', '机制不对应单蛋白口袋结合', '否'],
          ['螯合、还原、交联、广谱 cytotoxic', '退出或专线', '非特异化学反应会使结构模型任务错配', '否'],
          ['大分子、核酸、肽样', '退出 small-molecule 主线', '当前模型与实验协议不覆盖', '否'],
          ['GPCR/复杂膜蛋白', '独立功能赛道', '需要膜环境、构象状态和专门 readout', '否'],
          ['enzyme/kinase/channel/transporter/nuclear/epigenetic', '进入463靶点空间', '第一轮更容易配置 target-engagement assay', '否，仍只是可做性 prior'],
      ], 'compact')}
      <div class="callout"><strong>关键区分：</strong>这些规则提高任务一致性和实验可执行性，但不能证明某个具体 FDA 药物会结合某个具体靶点。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">当前第 2 级</div><h2>十四、334,749 的完整分流</h2>
      {table(['互斥队列', '行数', '占物理空间', '处置与含义'], [
          ['remote_structure_strict', '202,725', pct(202725, 334749), '远程化学 + 严格结构；进入同源审计和双 DTA'],
          ['remote_sequence_dta_only', '54,764', pct(54764, 334749), 'remote 但缺严格 pocket；保留序列 DTA 旁支'],
          ['remote_structure_review', '53,055', pct(53055, 334749), '结构需人工复核；不直接进入首轮'],
          ['family_extension_control', '17,361', pct(17361, 334749), '同靶点家族/同源扩展，作为 rediscovery/control'],
          ['similarity_rediscovery_control', '4,750', pct(4750, 334749), '与候选靶点已知配体过近，保留为 N1/阳性基线'],
          ['hold_structure_or_novelty', '1,368', pct(1368, 334749), '结构或新颖性无法可靠定性，暂缓'],
          ['known_positive_calibration', '726', pct(726, 334749), 'exact known 或 exact active species；只作校准'],
      ], 'number compact')}
      <div class="equation">202,725 + 54,764 + 53,055 + 17,361 + 4,750 + 1,368 + 726 = 334,749</div>
      <p>其中前三个 remote 队列合计 310,544。已知、相似和 family 行没有被“判为差”，而是为了避免把再发现包装成新发现而转入控制通道。</p>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">当前第 3 级</div><h2>十五、结构与同源风险审计</h2>
      {table(['步骤', '输入', '输出', '差额', '解释'], [
          ['remote 全部同源审计', '310,544', f"{homology['remote_homology_audited_rows']:,}", f"{310544-homology['remote_homology_audited_rows']:,}", '扩展 ChEMBL 已知靶点后发现同源/家族传播风险'],
          ['strict structure 同源审计', '202,725', f"{homology['strict_structure_remote_homology_audited_rows']:,}", f"{202725-homology['strict_structure_remote_homology_audited_rows']:,}", '同源风险行退出 strict remote'],
          ['review structure 同源审计', '255,780', f"{homology['review_structure_remote_homology_audited_rows']:,}", f"{255780-homology['review_structure_remote_homology_audited_rows']:,}", 'review 是 strict + manual-review 的嵌套视图，不与 strict 相加'],
      ], 'number')}
      <div class="callout blue"><strong>同源门定义：</strong>{esc(homology['homology_method'])}。</div>
      <p>结构来源包括 experimental holo 与 AlphaFold/P2Rank。experimental holo 提供已知配体定义的真实口袋和构象；AlphaFold/P2Rank 提供无共晶时的可计算假说。二者不能视为同等证据，也不能仅因存在口袋就推断候选 pair 真实。</p>
      <div class="callout"><strong>协议单位：</strong>Target gate 最终应具体到 target + construct + receptor structure + pocket + preparation protocol，而不是只按 gene 名称。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">当前第 4 级</div><h2>十六、DrugCLIP 准备差额与 30,000 队列</h2>
      <div class="metrics">
        {metric('strict remote', f"{drugclip['remote_strict_rows']:,}", '同源审计后结构严格层')}
        {metric('DrugCLIP 完成', f"{drugclip['remote_strict_scored_rows']:,}", f"损失 {drugclip['remote_strict_rows']-drugclip['remote_strict_scored_rows']:,} 条")}
        {metric('结构计算队列', f"{stage2['selected_rows']:,}", f"占可评分层 {pct(stage2['selected_rows'], stage2['input_remote_scored_rows'])}")}
      </div>
      <p>DrugCLIP 输入准备中，723 个 ligand 有 722 个成功、308 个 strict pocket 有 307 个成功。落在失败 ligand 或 pocket 上的 remote pair 共 1,020 条没有 DrugCLIP 分数。这是技术未完成，不是模型低分淘汰。</p>
      {table(['30,000 证据 lane', '数量', '含义'], [[k, f"{v:,}", {
          'strong_two_model_consensus':'ConPLEx 与 DrugCLIP 双向 Top50',
          'broad_two_model_consensus':'两模型双向 Top100',
          'drugclip_bidirectional_top50':'DrugCLIP 双向 Top50，ConPLEx未同级',
          'conplex_bidirectional_top50':'ConPLEx 双向 Top50，DrugCLIP未同级',
          'single_model_or_exploration':'单模型或覆盖探索配额',
      }.get(k, '')] for k, v in stage2['evidence_lane_counts'].items()], 'number')}
      <div class="callout teal"><strong>配额规则：</strong>先为每个靶点保留最强 10 条，再按 lane 与正交共识分数填充；target cap=130、ligand cap=70、scaffold cap=300。该 30,000 是昂贵结构计算的资源分配，不是“Top30,000 高亲和”。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">侧车校准</div><h2>十七、历史正负对照如何进入 Target gate</h2>
      <div class="metrics">
        {metric('历史控制', f"{controls['control_rows']:,}", f"{controls['positive_control_rows']:,} 阳性 / {controls['negative_control_rows']:,} 阴性")}
        {metric('完成评分靶点', f"{target_gate['targets_with_control_scores']:,}", '11个受体准备失败')}
        {metric('可评价靶点', f"{target_gate['targets_evaluable']:,}", '至少 8 阳性 + 8 阴性')}
      </div>
      {table(['层级', '靶点数', '定义'], [
          ['CNNaffinity pass', target_gate['cnn_affinity_pass_targets'], '同靶点 AUROC≥0.65 且 AP 超过 prevalence 0.10'],
          ['Vina pass', target_gate['vina_affinity_pass_targets'], '同样规则，方向按更负更好处理'],
          ['dual pass', target_gate['dual_pass_targets'], '两个评分通道均通过'],
          ['raw union', target_gate['admitted_union_targets'], '至少一个通道通过；当前 Target gate'],
          ['size-adjusted union', main_audit['key_numbers']['size_adjusted_union_targets'], '对分子量/重原子大小偏差做稳健性审计后的敏感性结果'],
      ], 'number')}
      <p>阴性由 3,738 条 explicit inactive 和 860 条定量低活性组成，没有把人工 decoy 当作真实阴性。但不同 assay、浓度、construct 和活性标签仍然异质，因此只能称历史低活性/inactive 对照。</p>
      <div class="callout"><strong>统计限制：</strong>两通道 OR 会产生通道选择膨胀；8+8 只够最低可评价；同一批历史样本同时选通道和设门会产生赢家偏差。当前 Target gate 证明局部判别信号，不证明远程发现资格。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">当前第 5 级</div><h2>十八、30,000 如何成为 16,995</h2>
      <div class="equation">30,000 pair / 301 targets　→　Target gate 与 strict remote 交集　→　16,995 pair / 171 targets</div>
      <div class="metrics">
        {metric('保留 pair', f"{discovery['selected_rows']:,}", f"保留率 {pct(discovery['selected_rows'], discovery['source_queue_rows'])}")}
        {metric('保留靶点', f"{discovery['unique_targets']:,}", '30k 中有候选且 gate 通过')}
        {metric('化学覆盖', f"{discovery['unique_ligands']:,} / {discovery['unique_scaffolds']:,}", 'ligands / Murcko scaffolds')}
      </div>
      {table(['Target admission tier', 'pair 数', '解释'], [
          ['T1 dual strong', discovery['target_admission_tier_counts']['T1_dual_strong'], '双通道通过且 bootstrap 下界较强'],
          ['T2 dual pass', discovery['target_admission_tier_counts']['T2_dual_pass'], '双通道满足基本门'],
          ['T3 single strong', discovery['target_admission_tier_counts']['T3_single_strong'], '单一通道强支持'],
          ['T4 single pass', discovery['target_admission_tier_counts']['T4_single_pass'], '单一通道基本通过'],
      ], 'number')}
      <p>受体来源为 13,529 条 experimental holo 与 3,466 条 AlphaFold/P2Rank，二者之和严格等于 16,995。这里的 T1–T4 是靶点协议资格，不是 pair 的 P1–P3，也不是具体候选的结合概率。</p>
      <div class="callout blue"><strong>为什么不是 234 个靶点：</strong>234 是所有已评分靶点上至少一个 GNINA 通道通过的数量；171 是其中同时出现在 30,000 strict remote 计算队列中的靶点。两者是全集与候选队列的交集关系。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">后半段</div><h2>十九、当前漏斗为什么停在 16,995</h2>
      {table(['待完成节点', '输入', '自然输出由什么决定', '当前不能填数的原因'], [
          ['Gate-B 远程适用域', '171 靶点的历史控制与候选', 'cold-scaffold、低相似、时间外推阳性的稳定性', '尚未构建冻结留出与阈值'],
          ['Pair gate', '16,995', '同靶点主通道分位、阴性尾部、阳性位置、置信区间', 'P1/P2/P3 数值规则尚未预注册'],
          ['Assay readiness', '通过 pair gate 的靶点', 'construct、平台、阳性/阴性、通量和成本', '实验组尚未回传171靶点矩阵'],
          ['资格先导', '4–8个可做靶点', 'assay QC、技术失败率、候选层级差异', '需与正式验证集分开'],
          ['广覆盖初筛', '60–80个 assay-ready 靶点', '每靶点3–5条和真实平台预算', '“1000”单位未定义'],
          ['自适应追加', '出现信号且QC合格的靶点', '预注册分配规则', '依赖第一阶段结果'],
          ['confirmed binder', 'primary hits', '复测、正交、竞争和反筛', '需预留预算并冻结证据链'],
      ], 'compact')}
      <div class="callout"><strong>原则：</strong>当前若直接从16,995固定取1000，只能得到新的计算容量包，无法解决历史流程中“固定数量冒充自然质量阈值”的问题。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">方法角色</div><h2>二十、新增方法各自解决什么</h2>
      {table(['方法', '输入', '输出', '当前正确角色', '不能证明'], [
          ['ConPLEx', 'SMILES + protein sequence', 'DTA score/rank', '低成本序列检索与覆盖通道', 'Kd、跨靶点概率'],
          ['DrugCLIP', '分子与结构口袋 embedding', 'cosine、z-score、双向 rank', '独立于序列的口袋检索通道', '直接亲和常数'],
          ['experimental holo', '共晶结构与配体位点', 'receptor/pocket protocol', '最优先结构起点', '新配体一定适合该构象'],
          ['AlphaFold + P2Rank', '预测结构', '候选口袋与 docking box', '无实验结构时的可计算假说', '口袋真实开放或可结合'],
          ['GNINA CNNaffinity', '固定受体/口袋 + ligand poses', 'CNN score', '同靶点历史校准后的内部排序通道', '真实 Kd 或跨靶点概率'],
          ['Vina', '同上', '经验势能打分', '与 CNN 的同结构双评分一致性', '独立的正交验证'],
          ['Boltz-2', 'protein-ligand cofolding 输入', 'binder score、affinity head、pose', '高成本结构假说与冲突审阅', '未经本项目校准的真实概率'],
          ['Gate-A', '同靶点历史阳性/inactive', '局部 protocol admission', '限制模型使用资格', '低相似外推'],
          ['Gate-B', '低相似/时间/骨架留出', '远程适用域资格', '未来 N2/N3 关键门', '前瞻命中率'],
      ], 'compact')}
      <div class="callout teal"><strong>组合逻辑：</strong>硬门定义任务边界；ConPLEx/DrugCLIP分配昂贵计算；结构协议提供可检验 pose；GNINA历史对照决定靶点内使用资格；Gate-B限制新颖度适用域；最终真实性由前瞻实验给出。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">术语字典 1/4</div><h2>二十一、Gate 到底是什么</h2>
      <p class="lead">Gate 是“准入条件”，不是分数，也不是对真实结合的证明。不同 Gate 回答不同层级的问题，不能互相替代。</p>
      {table(['术语', '操作定义', '当前状态', '绝对不能解释为'], [
          ['Hard gate', '实体、任务和技术边界：药物结构有效、靶点在项目范围、非已知pair、非严重责任等', '已实施', '候选更可能结合'],
          ['Target gate', '只有历史控制显示局部判别信号的靶点协议，才允许GNINA参与未知候选解释', '已实施，当前按靶点汇总', '该靶点所有候选都可信'],
          ['Gate-A', 'Target gate的历史局部资格测试：同靶点至少8阳性+8阴性；通道AUROC≥0.65且AP高于类别先验0.10', '349可评价，234至少一通道通过', '远程外推能力或真实binder概率'],
          ['Protocol gate', 'Gate-A的严格升级：target+isoform+construct+structure+pocket+cofactor+preparation+scoring protocol', '应补全，尚未逐靶点完成', 'gene名称本身通过'],
          ['Gate-B', '在cold-scaffold、低相似和时间外推阳性上检验远程适用域', '尚未实现', '前瞻命中率'],
          ['Pair gate', '在通过协议的同一靶点内，把未知pair放入阳性、inactive和背景分布中分层', '规则待冻结', '跨靶点可比Kd'],
      ], 'compact')}
      <div class="callout teal"><strong>一句话：</strong>Gate-A回答“这套局部协议在历史数据上有没有判别信号”；Gate-B回答“离开熟悉骨架后是否仍有信号”；Pair gate回答“某个未知候选在该靶点内部处于什么位置”。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">术语字典 2/4</div><h2>二十二、“远程”具体指什么</h2>
      <p class="lead">“远程”是相对候选靶点的既有配体化学和药物既有靶点知识而言，不是三维空间距离、疾病距离，也不等于已经证明全新。</p>
      <div class="equation">当前 remote eligible = 非已知pair ∩ 最大已知活性配体相似度&lt;0.40 ∩ 不同已知活性scaffold ∩ 无既有family/rediscovery风险 ∩ 无严重化学责任</div>
      {table(['检查项', '当前操作规则', '目的'], [
          ['exact known', 'FDA/ChEMBL已知pair或exact active species转入阳性校准', '不把已知药理包装成发现'],
          ['靶点已知配体相似性', 'Morgan最大相似度≥0.40即转入similarity rediscovery control', '避免普通ligand fishing'],
          ['scaffold', '与该靶点已知活性配体同Murcko scaffold即退出remote', '避免同骨架系列扩展'],
          ['药物已知靶点家族', '已有family/label风险转入family extension control', '避免同家族再发现'],
          ['扩展同源审计', 'MMseqs identity≥0.40且coverage≥0.60，或identity≥0.55且coverage≥0.30，记同源扩展风险', '补充原标签遗漏'],
          ['无靶点配体参考', '仍可进入remote_no_target_ligand_reference', '保留真正冷靶点，但不确定性更高'],
      ], 'compact')}
      <div class="callout"><strong>重要限制：</strong>当前remote集合把“低相似”和“没有参考配体”混在同一大类中；它只是新颖性硬门后的探索空间，不代表所有pair都属于同等程度的frontier，也不代表模型具备远程外推资格。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">术语字典 3/4</div><h2>二十三、N、T、Q、P、AD、S分别表示什么</h2>
      {table(['代号', '评价对象', '建议含义', '当前成熟度'], [
          ['N1/N2/N3', '新颖性赛道', 'N1：新pair但接近已知化学；N2：scaffold-hop/中低相似；N3：低相似、无参考或新口袋frontier', '语义已确定，精确阈值待冻结；当前remote不是N3同义词'],
          ['T1–T4', '靶点评分协议表现', '当前T1双通道强、T2双通道通过、T3单通道强、T4单通道通过', '已用于16,995队列；不是pair等级'],
          ['Q1–Q4', '历史控制数据质量', 'Q1数量与骨架充分且inactive可靠；Q2系列偏集中；Q3仅最低样本或标签弱；Q4不可评价', '建议新增，当前未完整生成'],
          ['P1/P2/P3/R', '同靶点pair证据', 'P1最强一致、P2单通道强或有限冲突、P3探索性支持、R不足/拒绝解释', '数值规则必须在实验前冻结'],
          ['AD-A–AD-D', '适用域', '从配体/靶点/口袋均在熟悉范围，到多维分布外仅探索使用', '建议新增'],
          ['S1–S4', '结构来源与协议风险', 'S1匹配人源holo；S2高同源实验结构；S3预测结构；S4动态/复合物/状态不确定', '建议在Protocol registry中新增'],
      ], 'compact')}
      <div class="callout blue"><strong>读法示例：</strong>P1 / AD-C / N3 不表示“高概率命中”，而表示“当前同靶点计算证据较强，但候选明显分布外且新颖性高，因此实验价值与失败风险同时较高”。</div>
      <div class="callout"><strong>禁止：</strong>把T、Q、P、AD、N重新加权成一个0–100“真实亲和可信度”。这些维度应并列展示并用于Pareto选择。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">术语字典 4/4</div><h2>二十四、实验与统计术语</h2>
      {table(['术语', '准确含义', '常见误解'], [
          ['direct binding', '纯蛋白或明确竞争体系中观测到直接结合；可报告Kd/Ki等适当量', '任何功能IC50都等于Kd'],
          ['target engagement', '在细胞、组织或复杂体系中观察药物占据/稳定目标，如NanoBRET、CETSA', '一定是纯蛋白直接结合'],
          ['functional activity', '酶活、通道、转运、reporter或细胞功能发生变化', '功能阳性自动证明直接靶点'],
          ['primary signal', '初筛中满足预注册最低信号标准', '已经是confirmed binder'],
          ['confirmed binder', '独立复测、适当反筛并由不同原理或竞争证据确认的直接结合', '一次漂亮曲线或pose即可确认'],
          ['technical failure', '不溶、蛋白失活、阳性失败、干扰或无法拟合', '可以计作实验阴性'],
          ['assay-ready', 'construct、蛋白、阳性、inactive、动态范围、通量和成本已满足启动条件', '计算上有结构就能直接做'],
          ['enrichment factor', '选中组命中率 ÷ 预先定义eligible随机基线命中率', '等同于绝对阳性率或临床成功率'],
          ['prospective calibration', '候选与规则冻结后，对未知pair做盲法/独立实验评价', '用历史样本回看即可替代'],
      ], 'compact')}
      <div class="callout teal"><strong>统计分母：</strong>命中率必须明确按全部提交pair、assay成功pair或完成确认pair中的哪一个计算；technical failure应单独报告，不应静默并入negative。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">AI能力上限 1/4</div><h2>二十五、实验前AI团队的六类核心能力</h2>
      <div class="detail-grid">
        {detail_card('1. 实体正确性', '统一administered drug、parent、active moiety、代谢物、盐型、立体化学、质子化/互变异构、共价与实验建议实体。', 'good')}
        {detail_card('2. 局部协议资格', '把资格具体到target、construct、结构、口袋、辅因子、准备和评分协议，而不是只看gene。', 'good')}
        {detail_card('3. Pair证据矩阵', '在同靶点历史分布内并列展示ConPLEx、DrugCLIP、GNINA、Boltz、pose、geometry与冲突，不制造新总分。', 'blue')}
        {detail_card('4. 适用域与不确定性', '量化配体、靶点、口袋和组合是否分布外，并区分数据、模型和输入三类不确定性。', 'blue')}
        {detail_card('5. 实验设计支持', '为每个target program配置候选层、阳性/inactive、随机eligible基线、浓度建议、反筛和stop/go。', 'warn')}
        {detail_card('6. 前瞻学习闭环', '预注册候选和指标；实验返回后区分输入失败、模型失败、实验失败和机制不一致，更新下一轮。', 'warn')}
      </div>
      <div class="callout teal"><strong>能力极限的正确定义：</strong>不是实验前证明候选一定结合，而是最大限度减少输入错误、模型误用、适用域外推和实验浪费，并为每个候选提供可审计、可证伪的证据包。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">AI能力上限 2/4</div><h2>二十六、第一轮实验前的核心计算交付</h2>
      {table(['交付物', '核心内容', '当前状态与下一步'], [
          ['VERSION_AND_PROVENANCE_MANIFEST', '数据、代码commit、hash、运行状态、冻结时间', '已有大量hash与summary；需形成单一冻结清单'],
          ['DRUG_ENTITY_REGISTRY', 'active moiety、代谢物、盐型、立体/质子化/互变异构、实验实体', 'active-moiety已完成；多状态和代谢物敏感性需补'],
          ['TARGET_PROTOCOL_REGISTRY', 'construct、结构、口袋、辅因子、protocol、S级、Gate结果', '结构路径已有；实验construct和Protocol gate待补'],
          ['CONTROL_QUALITY_TABLE', '阳性、inactive/低活性、assay类型、骨架、标签冲突、Q级', '9,199与8,119控制已有；质量分级和严格留出待补'],
          ['KNOWN_RECALL_WATERFALL', 'eligible分母、召回、富集、失败原因、训练可见性', '历史数据分散；需统一时间/骨架分层'],
          ['TARGET_GATE_REPORT', 'PR-AUC、EF、bootstrap、主通道、scaffold/低相似留出', 'Gate-A已有；Gate-B和Protocol gate待补'],
          ['PAIR_EVIDENCE_MATRIX', 'P、T、Q、AD、N、S、多模型、多结构、多seed和风险', '16,995已有部分字段；统一矩阵待生成'],
          ['ASSAY_CANDIDATE_PACKAGE', '候选层、控制、随机基线、实验建议、浓度、干扰与解释边界', '等待assay readiness与预算单位'],
      ], 'compact')}
      <div class="callout"><strong>不作为主线：</strong>对所有pair做长MD/FEP、继续扩库后无限重排、增加未经校准模型、用疾病网络重写亲和排序，或再生成一个统一综合分。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">AI能力上限 3/4</div><h2>二十七、压力测试、不确定性与拒绝预测</h2>
      {table(['层面', '可执行检查', '能发现什么'], [
          ['配体状态敏感性', '质子化、互变异构、立体、原药/代谢物、去盐、共价分流', '输入实体是否决定结论'],
          ['受体/口袋敏感性', '2–5构象、apo/holo、box大小、金属/水/辅因子、正构/变构', '结论是否只依赖单一结构假设'],
          ['配体反事实', '去关键基团、改电荷、翻手性、matched molecular pair', '模型是否只偏好大小或疏水性'],
          ['靶点反事实', '同家族错误靶点、关键残基突变、错误口袋、去辅因子', '模型是否具有pair和口袋特异性'],
          ['模型捷径测试', '打乱序列、同长度随机蛋白、ligand-only和target-frequency基线', '完整模型是否真正超过简单偏差'],
          ['重复稳定性', '多seed、多初始pose、多模型版本', '模型方差与模式切换'],
      ], 'compact')}
      <h3>允许拒绝预测</h3>
      <div class="branch-grid">
        <div class="branch"><h3>PASS</h3><p>在已声明的协议和适用域内可用于排序。</p></div>
        <div class="branch"><h3>LOW_CONFIDENCE / OUT_OF_DOMAIN</h3><p>仅作探索，不能使用常规命中承诺。</p></div>
        <div class="branch"><h3>ENTITY_UNRESOLVED / STRUCTURE_UNRESOLVED</h3><p>先修输入，不应继续堆模型。</p></div>
        <div class="branch"><h3>ASSAY_MISMATCH / DO_NOT_INTERPRET</h3><p>当前计算问题与实验问题不一致。</p></div>
      </div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">AI能力上限 4/4</div><h2>二十八、高成本物理方法的合理位置</h2>
      {table(['方法', '适合的问题', '不适合的用法', '建议规模'], [
          ['ensemble docking', '有多个实验构象或明确活化/失活状态', '把多个高分平均成绝对亲和', '少数assay-ready靶点'],
          ['局部MD', '检查氢键、金属、水网络、明显解离、侧链重排和替代pose', '短程稳定即证明结合', '前几十个高价值pair'],
          ['MM/GBSA', '同靶点、相近pose的辅助比较', '跨靶点或完全不同化学型绝对排序', '小型同靶点集合'],
          ['相对自由能', '已知binder附近、同骨架系列或inactive analog对比', '大量不同FDA药物的远程筛选', '局部系列'],
          ['绝对自由能', '体系标准、结构明确、实验价值很高的研究性分析', '替代实验或覆盖1000条', '极少数pair'],
          ['cryptic pocket', '静态结构失败但有apo/holo或动态证据且可实验验证', '为所有低分候选事后找解释', '1–2个高价值靶点'],
      ], 'compact')}
      <div class="callout blue"><strong>启动条件：</strong>高价值、assay-ready、结构和活性物种明确、存在可证伪机制假说。高成本计算的目标是减少具体输入/pose不确定性，不是制造确定性。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">科学边界 1/5</div><h2>二十九、无法同时最优化的四角约束</h2>
      <p class="lead">当前要求同时包含极少候选、极高阳性率、高度新颖和广泛靶点覆盖，同时缺少项目专属前瞻真值。这不是一个模型升级即可消除的工程缺口。</p>
      <div class="detail-grid">
        {detail_card('候选必须极少', '候选越少，对排序精度和校准要求越高，也减少用规模对冲极低先验概率的机会。', 'warn')}
        {detail_card('阳性率必须极高', '最有效的办法通常是依赖已知配体、成熟靶点和局部插值。', 'warn')}
        {detail_card('候选必须高度新颖', '低相似、跨家族或新口袋会离开训练与历史校准分布，方差必然增大。', 'warn')}
        {detail_card('还要覆盖很多靶点', '跨靶点协议差异增大，每靶点候选和控制变少，命中率估计更不稳定。', 'warn')}
      </div>
      <div class="root-box"><div class="label">当前统计条件</div><div class="quote">极低先验概率 + 可靠阴性不足 + 明显分布外 + 无前瞻校准，不支持用极少样本获得接近确定性的通用预测。</div></div>
      <div class="callout teal"><strong>边界说明：</strong>这是当前数据和实验闭环条件下的科学不确定性，不是对个人执行能力的评价。计算可以提高富集和决策质量，但不能充当oracle。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">科学边界 2/5</div><h2>三十、必须共同选择的四种模式</h2>
      {table(['模式', '优先优化', '候选构成', '必须接受的代价', '正确评价'], [
          ['高命中模式', '第一轮阳性率', 'N1、已知化学邻域、成熟T/Q靶点为主', '新颖性和靶点覆盖下降', '局部前瞻命中率与确认率'],
          ['平衡发现模式', '命中率与新颖性兼顾', 'N1/N2、少量N3，多靶点但非全覆盖', '需要中等候选数，不能承诺极高阳性率', '分赛道命中率与富集'],
          ['Frontier模式', '低相似/新口袋发现', 'N3、AD-C/D、模型分歧和信息增益候选', '较低命中率和更大方差', '远程命中、信息增益和新机制'],
          ['广谱target fishing', '覆盖更多靶点', '每靶点少量候选、广覆盖pilot', '单靶点统计不稳定、assay切换成本高', '有至少一个hit的靶点比例'],
      ], 'compact')}
      <div class="callout"><strong>不可同时使用的评价：</strong>选择高命中模式，就不能以“与已知配体太相似”否定全部结果；选择Frontier模式，就不能再按ligand-fishing的局部插值命中率要求评价。</div>
      <div class="callout blue"><strong>建议默认：</strong>平衡发现模式，并显式配置N1命中基线、N2主发现、N3探索、模型分歧、随机eligible和阳性/inactive控制六组。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">科学边界 3/5</div><h2>三十一、把“提高阳性率”改成可测量目标</h2>
      <p class="lead">“极大提高”没有分母、基线和置信区间，无法审计。合理承诺应是相对预先定义基线的富集，而不是保证绝对阳性率。</p>
      <div class="equation">Confirmed hit rate = 正交确认binder数 ÷ assay成功且完成确认的未知pair数</div>
      <div class="equation">Enrichment factor = 计算选择组confirmed hit rate ÷ 随机eligible基线confirmed hit rate</div>
      {table(['必须预先冻结', '示例问题'], [
          ['阳性定义', '一次primary signal、可重复primary hit，还是正交confirmed binder？'],
          ['分母', '全部提交pair、assay成功pair，还是完成确认pair？'],
          ['技术失败', '是否从命中率分母排除并单独报告？'],
          ['新颖度分层', 'N1、N2、N3是否分别统计？'],
          ['候选数量与靶点数', '总pair、每靶点数量和覆盖目标是什么？'],
          ['随机eligible基线', '从同一硬门和assay-ready空间如何随机抽样？'],
          ['统计不确定性', '报告Wilson/Clopper-Pearson区间、靶点聚类效应和bootstrap。'],
      ], 'compact')}
      <div class="callout teal"><strong>最低可验证目标：</strong>P1高于随机eligible；P1高于P3；N1通常高于N3；即使命中少，也能识别哪些家族、协议和适用域可预测。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">科学边界 4/5</div><h2>三十二、计算组能承诺什么、不能承诺什么</h2>
      <div class="detail-grid">
        <div class="detail-card good"><h3>能够承诺</h3><ul>
          <li>药物、蛋白、结构和口袋实体尽可能正确并可追溯。</li>
          <li>Known recall分母明确，不用Discovery包伪造召回。</li>
          <li>不跨靶点混用原始分数，只在通过局部资格的协议内解释。</li>
          <li>标注新颖性、适用域、模型冲突、不确定性和拒绝预测原因。</li>
          <li>冻结候选、分层、基线和统计方案；实验回来后更新模型。</li>
        </ul></div>
        <div class="detail-card warn"><h3>不能承诺</h3><ul>
          <li>未知远程pair已有准确真实结合概率。</li>
          <li>GNINA、Boltz或MD输出等于Kd或结合事实。</li>
          <li>极少候选必然出现hit。</li>
          <li>高新颖性、高阳性率和广覆盖可以同时保证。</li>
          <li>一个模型适用于所有靶点、构象和实验技术。</li>
          <li>任何实验阴性都一定是模型错误。</li>
        </ul></div>
      </div>
      <div class="callout blue"><strong>专业表述：</strong>AI提供的是经实体、协议、适用域和历史校准约束后的富集与优先级，不是在远程未知空间中的确定性答案。</div>
      <div class="callout"><strong>第一批实验失败的解释：</strong>可能来自输入实体、结构/口袋、蛋白状态、assay、化合物质量或模型。只有阳性、inactive、技术控制和正交确认完整时，才能归因。</div>
    </section>
    """)

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">科学边界 5/5</div><h2>三十三、责任分配与第一轮实验定位</h2>
      {table(['责任方', '必须负责', '不应单独承担'], [
          ['计算组', '候选冻结；实体/结构/适用域审计；P/T/Q/AD/N/S；Known waterfall；不确定性；预注册；结果后更新', '蛋白是否有效、实验技术失败、最终必然命中'],
          ['实验组', 'construct正确；蛋白有效；阳性正常；inactive合理；化合物质量/溶解度；失败编码；原始QC；正交确认', '计算适用域、历史召回分母和候选算法'],
          ['双方共同', '第一轮目标；候选数量；阳性定义；失败率；新颖性/命中/覆盖权衡；继续/停止标准', '把整个链条的结果归为单方责任'],
      ], 'compact')}
      <div class="root-box"><div class="label">第一轮正式名称</div><div class="quote">模型—实验联合资格先导与前瞻校准实验</div></div>
      {table(['可能结果', '科学含义'], [
          ['命中', '获得新pair，并为相应lane和protocol提供正向前瞻证据'],
          ['合格实验阴性', '获得项目专属真实负样本，校准模型和适用域'],
          ['技术失败', '暴露construct、蛋白、化合物或平台问题，不计模型阴性'],
          ['P层级无差异', '当前分层在该空间不适用，需要修改或停止'],
          ['某家族明显更好', '确定下一轮应重点扩展的target family与protocol'],
      ], 'compact')}
      <div class="callout teal"><strong>核心定位：</strong>实验不是计算完成后的奖励或最终考试，而是未知pair真实性和项目专属概率校准不可缺少的数据来源。</div>
    </section>
    """)


    issues = [
        {
            "category":"已经解决并冻结", "badge":"resolved", "status":"已解决", "title":"第一阶段任务：亲和发现优先",
            "one_line":"第一阶段寻找可重复直接 interaction；疾病和治疗价值在 hit 后研判。",
            "definition":"项目第一阶段的成功单位是新的 drug-target direct-binding 或明确 target-engagement 关系，不要求候选在同一轮同时证明疾病疗效。",
            "impact":"若同时要求结合、疾病方向、暴露和疗效，疾病知识会提前接管排序；若只谈疾病，不再是亲和发现；若只谈结合却称老药新用，又会夸大转化结论。",
            "evidence":"当前主链不再使用 Open Targets、TxGNN、STRING 或疾病方向改变 affinity queue；16,995 的生成依据是 remote chemistry、结构和同靶点历史校准。",
            "gap":"解决的是阶段边界，不是最终项目价值。confirmed binder 之后仍必须进入功能、作用方向、暴露、安全性和疾病机制研判。",
            "decision":"正式表述为“两阶段双终点”：先报告 biophysical discovery hit，再报告 translational repurposing hit。",
            "acceptance":"主筛代码不得读取疾病分数；实验报告分开统计 direct binding、target engagement、functional activity 和转化可行性。",
            "owner":"模型组维护亲和漏斗；机制组只在 confirmed hit 后启动疾病研判；实验组冻结第一阶段可接受终点。",
        },
        {
            "category":"已经解决并冻结", "badge":"resolved", "status":"已解决", "title":"疾病知识不进入亲和主排序",
            "one_line":"疾病图谱保留解释价值，但不再补偿低物理证据。",
            "definition":"Open Targets、STRING、TxGNN、Reactome、表达签名和组织表达只描述 target-disease、drug-disease 或生物语境，不是 drug-target direct-binding 证据。",
            "impact":"知识图谱研究密度和适应症标签会偏向热门靶点；把它与 affinity score 加权会让“疾病相关”被误读为“会结合”。",
            "evidence":"当前 334,749→16,995 主链的规则和 summary 均声明 disease_role=not used in affinity stage。",
            "gap":"疾病信息仍需在 hit 后判断作用方向和临床价值；后置不等于删除。Open Targets tractability 可作为 assay 注释，但不是疗效证据。",
            "decision":"亲和排序和疾病研判分成两个独立表；不得用疾病分数覆盖 pair gate 的失败。",
            "acceptance":"候选卡同时展示 affinity lane 与 disease appendix，但总分字段不得把二者相加；所有疾病结论标注为 hit 后假说。",
            "owner":"模型组保证字段隔离；机制组负责 confirmed hit 的病种、方向和组织语境。",
        },
        {
            "category":"已经解决并冻结", "badge":"resolved", "status":"已解决", "title":"Known recall 与 Discovery 分开",
            "one_line":"已知关系用于校准和流程质控，不进入新发现包。",
            "definition":"Known recall 的分母是预先冻结且符合当前实体/靶点/机制边界的已知 direct-action pair；Discovery 主动排除 exact known 和 close rediscovery。",
            "impact":"若在 Discovery 最终包上计算已知召回，主动去已知会被错误解释为模型失效；若把已知重新放回又会制造高召回和知识泄漏。",
            "evidence":"当前物理空间中 726 条 known-positive calibration 单独分流；family 17,361 和 similarity 4,750 也进入控制通道。",
            "gap":"Known recall 仍可能与模型训练知识重叠，只能证明历史检索/富集；需时间外推和 cold-scaffold 留出。",
            "decision":"并行交付 KNOWN_RECALL_WATERFALL 与 DISCOVERY_QUEUE，不报“最终候选包已知召回率”。",
            "acceptance":"每个已知 pair 标注实体资格、是否训练可见、失败阶段和 rank；Discovery exact-known 行数必须为0。",
            "owner":"模型与数据负责人冻结分母和标签版本；实验组用 known controls 检查 assay 是否工作。",
        },
        {
            "category":"已经解决并冻结", "badge":"resolved", "status":"已解决", "title":"停止跨靶点混合原始分数",
            "one_line":"GNINA、Vina、Boltz 和 ConPLEx 不再被解释为统一亲和尺度。",
            "definition":"不同靶点、口袋、受体构象和分数函数的原始值分布不同；只有在同靶点、同结构和同协议的校准分布内才进行相对排序。",
            "impact":"跨靶点统一 TopK 会偏向天然易对接、疏水大口袋或特定分数尺度的靶点，无法比较真实 Kd。",
            "evidence":"当前流程先 Target gate，再将通道用于同靶点内部；30,000 明确是 compute allocation，而非全局 affinity ranking。",
            "gap":"同靶点内部仍存在小样本、阴性异质和适用域问题；局部 rank 也不等于校准概率。",
            "decision":"pair card 保留 raw score、同靶点阴/阳性百分位和置信区间，不提供跨靶点“亲和可信度总分”。",
            "acceptance":"所有候选排序字段必须带 target/protocol key；跨靶点组合只通过配额和 Pareto 选择，不比较原始值。",
            "owner":"模型组维护 target-specific calibration；实验组确认 construct 和 protocol 是否与计算一致。",
        },
        {
            "category":"已经解决并冻结", "badge":"resolved", "status":"已解决", "title":"旧1000与384重新定位",
            "one_line":"旧候选包是历史计算储备和容量子集，不是统一高概率 binder。",
            "definition":"历史多个1000/384来自不同母集、评分、Boltz版本、疾病平衡和审阅规则；数字是预算或展示容量，不是自然证据边界。",
            "impact":"继续把旧等级继承到新流程，会把未经靶点校准的综合分误当作真实概率，也会混淆哪个文件是正式结果。",
            "evidence":"full-universe v4 的1000由334,749→264,352→3,000后产生；strict895 的384则来自疾病/agent审阅，二者不是同一分支。",
            "gap":"旧包仍包含已经完成的Boltz、pose、文献和agent信息，不能简单删除；需要作为候选特征和历史对照复用。",
            "decision":"所有旧包冻结只读；当前实验候选必须从16,995与后续门重新生成，同时记录与旧包交集。",
            "acceptance":"正式交付文件名含母集和规则版本；报告不再单独使用“1000”“384”指代候选质量。",
            "owner":"数据负责人维护候选包注册表、SHA256与版本关系。",
        },
        {
            "category":"已经解决并冻结", "badge":"resolved", "status":"已解决", "title":"GPCR与复杂膜蛋白单列",
            "one_line":"第一轮通用生化主线优先可执行靶点，复杂受体进入专门功能赛道。",
            "definition":"GPCR、复杂膜蛋白和依赖特殊构象/复合物的靶点不进入统一纯蛋白亲和流程，但保留在独立 receptor-panel 或功能实验路线。",
            "impact":"把所有蛋白塞入同一 docking/SPR 流程会产生构象、膜环境和作用方向错配；全部删除又会系统性损失真实药理空间。",
            "evidence":"当前463项目靶点以 enzyme、kinase、ion channel、transporter、nuclear/epigenetic 等 assay family 为主；复杂对象已分流。",
            "gap":"ion channel 和 transporter 仍可能需要膜环境或功能读出，不能因为在463中就默认容易完成 direct-binding assay。",
            "decision":"通用首轮与复杂膜蛋白专线使用不同实验终点、控制和统计，不再将是否纳入主线视为项目争议。",
            "acceptance":"Assay Readiness Matrix 必须按 family 指定平台；复杂膜蛋白不得被无条件映射到普通 docking Kd。",
            "owner":"实验组定义各家族可用平台；模型组保持专线候选，不作永久删除。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"Gate-A 不能证明远程发现能力",
            "one_line":"历史可分仅代表局部判别，不能自动外推到低相似 Discovery。",
            "definition":"Gate-A检验同靶点历史阳性与inactive是否可分；Gate-B应检验时间外推、cold-scaffold和接近Discovery新颖度的阳性是否仍可分。",
            "impact":"GNINA可能依赖训练集中相似配体或某一化学系列；高相似历史AUROC不能保证对经过相似性排除的远程分子有效。",
            "evidence":"349个靶点可评价，234至少一通道通过；但当前没有按最大已知配体Tanimoto、首次公开时间或骨架留出重新计算。",
            "gap":"未定义N1/N2/N3新颖度阈值；未冻结低相似阳性分母；未完成leave-one-scaffold-out稳定性。",
            "decision":"建立Gate-B，至少输出常规局部资格和远程资格两个层级；没有Gate-B的靶点只能支持N1/N2探索，不能包装为远程发现。",
            "acceptance":"逐靶点报告低相似/时间留出AUROC、AP lift、bootstrap CI和删除主骨架后的敏感性；规则在看Discovery实验前冻结。",
            "owner":"模型组负责构建留出与统计；实验组不负责证明计算外推。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"历史阴性并非统一 non-binder",
            "one_line":"explicit inactive与低活性有价值，但标签、浓度和实验条件异质。",
            "definition":"真实阴性应在明确construct、assay、浓度上限和检测能力下未结合；当前历史阴性来自ChEMBL不同实验中的inactive或pChEMBL低活性。",
            "impact":"弱活性、技术检测上限、错误活性物种和不同机制可能被混为阴性；阴性物化性质与阳性不同会制造虚假可分性。",
            "evidence":"9,199控制中4,598阴性，包括3,738 explicit inactive和860定量低活性；未使用人工decoy冒充真实不结合。",
            "gap":"尚未逐条统一最高测试浓度、assay relation、construct、化合物溶解度和物化匹配；部分阴性可能是未被充分测试的binder。",
            "decision":"标签拆成measured inactive、low activity、property-matched decoy和unknown background；主校准优先前两类。",
            "acceptance":"每靶点报告阴性来源比例、浓度/assay可比性和阳阴性物化可分性；技术失败不得作为阴性。",
            "owner":"数据组清洗历史标签；实验组产生统一条件下的前瞻阴性。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"小样本与双通道选择偏差",
            "one_line":"8+8是最低可评价，不足以稳定选择最优通道或估计概率。",
            "definition":"当前CNNaffinity和Vina分别检验，至少一通道通过即准入；这包含多重比较和看结果选通道的赢家偏差。",
            "impact":"两个无效通道也会提高偶然至少一个通过的概率；单一骨架或一个样本可显著改变8+8靶点的AUROC。",
            "evidence":"194个CNN pass、166个Vina pass、126双通道、234并集；大小校正敏感性将并集降至221、双通道降至94。",
            "gap":"未预先指定主通道；未做完整多重检验；同一数据用于通道选择和准入；缺独立留出。",
            "decision":"按T1–T4区分样本与骨架质量；冻结主通道或组合规则；第二通道仅增强、冲突复核，不允许事后择优。",
            "acceptance":"报告bootstrap CI、leave-one-out、leave-one-scaffold-out和效应量；独立留出上复核准入通道。",
            "owner":"统计与模型负责人冻结规则并生成可重复代码。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"结构、口袋与实验实体一致性",
            "one_line":"gene级通过不等于具体construct、结构和口袋协议有效。",
            "definition":"可用单位应是target + isoform/construct + receptor conformation + pocket + cofactors + preparation protocol，而非仅gene symbol。",
            "impact":"同一蛋白不同构象、金属、辅因子、复合物或膜环境可彻底改变对接结果；共晶redocking成功不代表cross-docking新配体成功。",
            "evidence":"当前Discovery中13,529对使用experimental holo，3,466对使用AlphaFold/P2Rank；结构和序列路径可追溯。",
            "gap":"171靶点尚未逐个与实验construct核对；正构/变构/共价配体可能混入同一口袋；结构水和辅因子处理未按实验统一。",
            "decision":"Target gate升级为protocol gate；没有construct与阳性redocking/cross-docking证据的靶点只能进入pilot。",
            "acceptance":"每靶点保存PDB/AF版本、序列范围、辅因子、box、redocking RMSD、cross-docking和实验construct差异。",
            "owner":"结构组维护protocol manifest；实验组提供真实蛋白实体。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"没有未知 pair 的前瞻命中率",
            "one_line":"历史回顾性校准无法给出precision、FDR或真实binder概率。",
            "definition":"前瞻命中率要求在看到实验结果前冻结候选、实验协议、hit定义和统计，然后对未知pair做独立验证。",
            "impact":"没有前瞻实验时，任何P1/P2/P3或0–1模型输出都只是优先级证据；无法回答1000中有多少真实结合。",
            "evidence":"已有ConPLEx、DrugCLIP、GNINA、Boltz、pose、文献和agent审阅，但均未产生统一条件的未知pair正负标签。",
            "gap":"没有locked validation set、primary/confirmed定义、技术失败编码、正交确认率和暴露可达性分层。",
            "decision":"先做资格pilot，再启动独立locked validation；无论命中或失败都完整回传原始数据。",
            "acceptance":"分别报告各lane、新颖度层级和target family的primary hit率、正交确认率、CI与每个hit成本。",
            "owner":"实验组执行预注册实验；模型组不得用validation结果回头改当轮阈值。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"靶点可做性不等于 pair 真实性",
            "one_line":"好口袋、好结构和成熟assay只是target prior。",
            "definition":"target tractability描述靶点是否有小分子先例、结构和实验体系；pair真实性描述当前FDA药物是否直接结合该靶点。",
            "impact":"同一靶点的所有候选共享口袋与assay加分，无法区分药物；target-level prior过重会让热门、易做靶点占据候选。",
            "evidence":"Open Targets、P2Rank、PUResNet、结构来源和assay family已经补全，但当前Target gate仍只限定协议资格。",
            "gap":"pair-specific物理信号主要来自DTA/docking，且未前瞻校准；暴露和非特异结合风险仍未进入硬门。",
            "decision":"target prior只决定能否启动program和实验成本；pair gate必须使用同靶点分布、活性物种和新颖度证据。",
            "acceptance":"候选卡分开显示target readiness与pair evidence；不得用target readiness补偿pair低于阴性背景。",
            "owner":"模型组拆分字段；实验组返回真实assay readiness。",
        },
        {
            "category":"仍然保留", "badge":"retained", "status":"未解决", "title":"新颖性与命中率仍然冲突",
            "one_line":"相似性越低越新颖，但也越远离模型适用域。",
            "definition":"新颖性至少包括pair novelty、与该靶点已知配体的相似度、原靶点到新靶点的家族/口袋距离，以及是否使用新口袋。",
            "impact":"全部删除N1会失去可预测基线并把候选推到极端分布外；全部保留高相似又会退化为ligand fishing。",
            "evidence":"当前已把17,361 family和4,750 similarity行转为控制；remote队列强调低相似与同源审计。",
            "gap":"N1/N2/N3阈值未冻结；Gate-B未说明不同新颖度的适用资格；实验配额尚未按lane分层。",
            "decision":"保留N1高命中基线、N2 scaffold-hop主发现、N3 frontier探索三条赛道，分别评价，不混成一个总命中率。",
            "acceptance":"每个候选报告最大Tanimoto、Murcko关系、target family/sequence identity和pocket类型；实验按赛道固定配额。",
            "owner":"化学信息与模型负责人定义新颖度；实验组按盲法执行分层候选。",
        },
        {
            "category":"本轮新增", "badge":"new", "status":"新增", "title":"实验终点必须分层",
            "one_line":"direct binding、target engagement和functional activity不能统一称为亲和命中。",
            "definition":"SPR/BLI/ITC等可提供直接结合；CETSA/NanoBRET等主要提供细胞或体系内engagement；酶活、膜片钳和reporter测功能效应。",
            "impact":"不同终点的阴性含义和数值尺度不同；功能IC50/EC50受底物、信号放大和细胞背景影响，不能自动解释为Kd。",
            "evidence":"当前项目目标已冻结为interaction-first，但171个靶点跨多个assay family，无法用一种实验统一覆盖。",
            "gap":"尚未形成技术到终点类型的正式映射，也未规定每类终点怎样升级为confirmed binder。",
            "decision":"会议逐项冻结各平台属于direct binding、target engagement或functional activity，并设置双重终点。",
            "acceptance":"实验报告分开统计三类hit；任何functional hit如需称binder，必须有直接或竞争结合的正交证据。",
            "owner":"实验组定义平台；模型组按终点类型维护标签。",
        },
        {
            "category":"本轮新增", "badge":"new", "status":"新增", "title":"171个计算靶点不等于171套实验",
            "one_line":"Target gate只说明计算协议局部可分，不说明蛋白和平台现成。",
            "definition":"assay-ready要求正确construct可得、活性QC通过、阳性对照可复现、动态范围足够、通量和成本可接受。",
            "impact":"实验成本主要按靶点program发生，而不是每个pair等成本；171个靶点可能意味着171套开发任务。",
            "evidence":"当前171靶点已扩展8,119条历史控制且每靶点至少12+12，但这些是计算校准材料，不是实验平台可用清单。",
            "gap":"缺蛋白来源、construct、平台状态、阳/阴性对照、throughput、开发周期与主要伪影。",
            "decision":"实验组返回Assay Readiness Matrix，仅成熟或可快速转移靶点进入第一批广筛。",
            "acceptance":"每靶点状态只能为validated、transferable、development、unavailable，并附证据和负责人。",
            "owner":"实验组拥有assay readiness最终签字；模型组据此重算可执行候选。",
        },
        {
            "category":"本轮新增", "badge":"new", "status":"新增", "title":"实验实体与计算实体需要逐靶点对齐",
            "one_line":"isoform、construct、辅因子和活性物种必须成为pair身份。",
            "definition":"一个实验pair不仅是drug ID + gene，还包括modelled species、compound batch、target construct、protein state和assay protocol。",
            "impact":"前药母体、错误立体异构体、不同isoform或缺失辅因子都会使真实阳性被计算或实验打低；结果无法回流。",
            "evidence":"当前已完成active-moiety折叠和结构序列完整性审计，并识别过前药/活性物种与模板错配问题。",
            "gap":"实验组尚未确认每个候选实际测原药还是代谢物、使用哪个construct与蛋白批次。",
            "decision":"pair ID扩展为compound-species × target-construct × assay-protocol，并与计算输入签名绑定。",
            "acceptance":"所有实验结果必须包含compound batch、active species、construct ID、sequence hash、protocol ID和原始数据路径。",
            "owner":"数据组制定ID规范；实验组填写；结构组核对输入一致性。",
        },
        {
            "category":"本轮新增", "badge":"new", "status":"新增", "title":"“1000”的预算单位尚未定义",
            "one_line":"1000个pair、曲线、孔位或进样对应完全不同实验规模。",
            "definition":"预算需要拆成未知pair初筛、浓度点、技术重复、板内控制、独立复测、正交确认和技术失败补测。",
            "impact":"若1000指孔位，无法完成80×10完整曲线；若指1000条曲线，则成本和时间远高于1000个单点；不定义会导致候选包不可执行。",
            "evidence":"当前不存在正式Assay1000；主报告已明确不能从16,995直接固定截取。",
            "gap":"缺各平台单pair资源、DMSO/蛋白/化合物用量、板容量、复测上限和hit率情景。",
            "decision":"建立Experiment Budget Ledger，按平台和阶段核算；至少给5%、10%、20% primary hit率情景。",
            "acceptance":"预算表能由pair数推导孔位/进样、时间和费用，并明确是否含控制、复测和正交。",
            "owner":"实验组提供单位成本和通量；项目负责人冻结总预算；模型组按预算选样。",
        },
        {
            "category":"本轮新增", "badge":"new", "status":"新增", "title":"先导开发集必须与正式验证集隔离",
            "one_line":"用于调整规则的数据不能再用于无偏命中率评价。",
            "definition":"assay-development pilot用于优化蛋白、缓冲、浓度、hit规则和候选层级；locked validation在规则冻结后独立测试。",
            "impact":"若根据pilot结果调整P1/P2/P3后仍把pilot算入性能，命中率会产生乐观偏差，无法复现。",
            "evidence":"当前尚无前瞻数据，因此现在仍可在看结果前完成预注册和集合分割。",
            "gap":"未确定4–8个pilot靶点、候选数、正式验证靶点、数据可见性和重用规则。",
            "decision":"先冻结pilot manifest与locked validation manifest；pilot只用于开发，正式指标只在locked set上计算。",
            "acceptance":"两个manifest具有不同pair ID、生成时间和SHA256；规则变更有版本记录；validation结果揭盲前不可修改。",
            "owner":"模型组负责分割和预注册；实验组按盲法执行。",
        },
        {
            "category":"本轮新增", "badge":"new", "status":"新增", "title":"需要自适应广筛与数据闭环",
            "one_line":"固定80×10偏向覆盖，不一定最大化确认hit或信息增益。",
            "definition":"阶段A在60–80个assay-ready靶点各测3–5条；阶段B按预注册规则向有信号、QC稳定或高信息价值靶点追加。",
            "impact":"平均分配会在无信号或assay失败靶点浪费预算；完全事后自由追加又会引入选择偏差。",
            "evidence":"当前目标是多靶点广筛并尽可能撞到hit，且不同靶点候选数量、预测强度和实验成本显著不同。",
            "gap":"未定义阶段A/B预算比例、追加触发条件、每靶点上限、技术失败补偿和数据回传周期。",
            "decision":"在会议上比较固定设计与自适应设计；若采用自适应，先冻结分配算法和停止规则。",
            "acceptance":"输出Experiment Data Dictionary与allocation log；每次追加可由预注册条件重现，technical failure不计入negative。",
            "owner":"实验组提供运行节奏；模型组执行候选再分配；统计负责人审计选择偏差。",
        },
    ]

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">问题分类</div><h2>三十四、19个关键问题的当前状态</h2>
      {table(['类别', '问题', '当前结论'], [[x['category'], x['title'], x['one_line']] for x in issues], 'compact')}
      <div class="callout teal"><strong>已解决：</strong>项目边界和信息角色已冻结，不再反复讨论。</div>
      <div class="callout"><strong>仍然保留：</strong>涉及模型适用域和未知pair真实性，只能通过更严格留出与前瞻实验解决。</div>
      <div class="callout blue"><strong>本轮新增：</strong>流程进入实验决策后出现的执行与数据治理问题，需要会议形成书面协议。</div>
    </section>
    """)
    pages.extend(issue_page(i, item) for i, item in enumerate(issues, start=1))

    meeting_questions = [
        {
            "group":"实验终点", "title":"第一阶段是否只评价直接结合或target engagement", "question":"第一轮是否可以把成功目标限定为可重复直接结合或明确target engagement，暂不要求同步完成疾病机制和细胞表型？",
            "why":"这是亲和优先路线能否执行的前提。若实验组仍要求每个候选先证明疾病价值，漏斗会再次回到疾病优先。",
            "current":"项目方法上已经采用interaction-first；疾病、暴露与安全性作为confirmed hit后的第二终点。",
            "needed":"哪些靶点必须同步做功能实验；哪些平台只能给target engagement；是否接受biophysical hit作为第一阶段成果。",
            "risk":"结果出来后双方用不同成功标准评价，同一信号被一方称hit、另一方称无价值。",
            "fields":"target|primary endpoint|secondary endpoint|disease assay required|acceptance owner",
            "decision":"形成逐靶点终点矩阵，并由实验组和项目负责人签字。",
        },
        {
            "group":"实验终点", "title":"哪些技术计入direct-binding evidence", "question":"SPR、BLI、MST、ITC、竞争结合、NanoBRET、CETSA、DSF、酶活和功能实验分别属于哪类证据？",
            "why":"不同技术测量的物理量不同，不能把热稳定、功能效力和Kd统一为亲和。",
            "current":"直接生物物理或竞争结合可支持direct binding；CETSA/NanoBRET偏engagement；酶活/膜片钳/reporters偏functional。",
            "needed":"本地或CRO的每项技术、可用靶点家族、检测范围、主要伪影和正交方法。",
            "risk":"将功能IC50误称Kd，或把高浓度热稳定变化误称直接binder。",
            "fields":"assay type|evidence class|measured quantity|detection range|artifact|orthogonal assay",
            "decision":"冻结技术到证据等级的映射表。",
        },
        {
            "group":"实验终点", "title":"是否按三类终点分别报告", "question":"是否同意分别统计direct-binding primary hit、target-engagement hit和functional activity hit？",
            "why":"只有分开报告，才能知道计算模型预测的是结合、体系内engagement还是功能。",
            "current":"方法上应分开；confirmed binder还需复测和正交，而functional hit不能自动升级为binder。",
            "needed":"每类终点的成功阈值、可接受证据链和最终汇报口径。",
            "risk":"总体“命中率”失去可解释性，模型无法根据结果校准。",
            "fields":"endpoint class|primary result|confirmation result|final label|conversion rule",
            "decision":"实验数据库强制使用三类终点和标准标签。",
        },
        {
            "group":"预算", "title":"“可以做1000个”的真实单位", "question":"1000是pair、单浓度点、完整曲线、孔位、仪器进样，还是包括控制和重复的总预算？",
            "why":"这是决定能覆盖多少靶点、每靶点测几条和是否能做浓度曲线的首要约束。",
            "current":"当前没有正式Assay1000；16,995不能在单位未知时直接截取1000。",
            "needed":"各平台的预算单位、是否包含阳性/阴性/空白、重复、复测与正交。",
            "risk":"候选数量看似满足要求，但实际孔位或仪器循环超出数倍。",
            "fields":"budget unit|total capacity|controls included|replicates included|retest included|orthogonal included",
            "decision":"用Experiment Budget Ledger替代一个笼统的1000。",
        },
        {
            "group":"预算", "title":"一个未知pair需要多少资源", "question":"每种primary assay需要多少浓度点、技术重复、独立重复、化合物量、蛋白量、时间和费用？",
            "why":"pair并非同成本；不同靶点的蛋白准备和平台切换成本可能远高于加测同靶点候选。",
            "current":"计算侧只能给pair数量，无法从现有文件推断实验单位成本。",
            "needed":"逐平台资源模板和最大并行通量，包含DMSO、参考表面、板内控制和失败补测。",
            "risk":"均匀80×10导致assay开发成为瓶颈，或没有预算做确认。",
            "fields":"concentration points|technical replicates|independent runs|compound amount|protein amount|time|cost",
            "decision":"按平台计算每个pair的真实成本并重算候选容量。",
        },
        {
            "group":"预算", "title":"是否预留复测和正交确认", "question":"当primary hit率为5%、10%或20%时，可以复测和正交确认多少个候选？",
            "why":"primary signal不等于binder；不预留确认预算会得到大量无法定性的初筛信号。",
            "current":"项目成功标准需要orthogonally confirmed binder，而非只看单次primary。",
            "needed":"独立复测上限、正交平台容量、是否用新化合物/蛋白批次、费用归属。",
            "risk":"初筛命中越多，后续反而无法完成，最终没有可信结果。",
            "fields":"assumed primary hit rate|retest capacity|orthogonal capacity|new batch required|reserved budget",
            "decision":"在提交候选前冻结至少三种hit-rate情景的确认预算。",
        },
        {
            "group":"Assay readiness", "title":"171个靶点中哪些现在可做", "question":"哪些靶点已有成熟assay、可直接转移、需少量优化、需重新表达或当前不可做？",
            "why":"计算Target gate不等于实验可运行；广谱发现能覆盖多少靶点由实验能力决定。",
            "current":"171靶点有计算准入与expanded controls，但尚无实验组readiness回传。",
            "needed":"逐靶点平台状态、蛋白可得性、开发周期、通量、成本和负责人。",
            "risk":"把计算候选当成可执行实验包，实际大部分时间耗在assay development。",
            "fields":"target|family|assay status|protein availability|throughput|cost|owner|ETA",
            "decision":"只有validated和transferable靶点进入第一批广筛。",
        },
        {
            "group":"Assay readiness", "title":"每个靶点的准确蛋白实体", "question":"实验使用哪个isoform、construct、序列范围、标签、表达系统、PTM、辅因子和复合状态？",
            "why":"这是判断结构计算是否与实验问题一致的必要条件。",
            "current":"项目已有canonical sequence、结构和口袋路径，但未获得实验construct。",
            "needed":"construct sequence或hash、蛋白批次、辅因子/金属/底物、活性QC和保存条件。",
            "risk":"计算和实验实际上测试不同蛋白状态，已知阳性也可能失败。",
            "fields":"isoform|construct residues|sequence hash|tag|expression system|PTM|cofactor|oligomer state",
            "decision":"实验construct与计算protocol逐靶点配对后才可提交候选。",
        },
        {
            "group":"Assay readiness", "title":"每个靶点是否有可靠阳性对照", "question":"阳性化合物的活性物种、亲和/效力范围、实验类型和本平台复现状态是什么？",
            "why":"未知候选无信号时，必须知道是药物不结合还是蛋白/assay失效。",
            "current":"ChEMBL历史阳性可用于计算校准，但不等于实验组平台已经复现。",
            "needed":"可采购阳性、批次、浓度范围、预期曲线和运行接受窗口。",
            "risk":"没有阳性控制的阴性结果无法用于模型评价。",
            "fields":"positive control|active species|reference value|assay matched|procurement|acceptance range",
            "decision":"无可复现阳性的靶点只进入assay-development pilot。",
        },
        {
            "group":"Assay readiness", "title":"是否有同实验inactive参照", "question":"每个靶点是否有实测inactive、inactive analog或至少高浓度低活性参照？",
            "why":"它用于识别非特异信号、平台背景和模型局部分离；人工decoy不能替代真实实验阴性。",
            "current":"计算侧有历史inactive/低活性，但实验条件异质。",
            "needed":"同construct、同assay、测试上限、溶解度和结构关系。",
            "risk":"模型和实验可能只区分物化性质或化学系列，而非结合。",
            "fields":"inactive type|same assay|highest tested concentration|result|solubility|chemical relation",
            "decision":"优先选择具备同实验阳性和inactive的靶点进入locked validation。",
        },
        {
            "group":"结果判定", "title":"什么条件定义primary hit", "question":"最低信号、浓度依赖、饱和趋势、重复一致性和curve-fit质量分别要求什么？",
            "why":"不预先定义会在看结果后灵活移动阈值，命中率不可复现。",
            "current":"当前没有统一实验数据，因此模型等级不能替代实验hit定义。",
            "needed":"每种技术的阈值、最高有效浓度、阳性窗口、信噪比和拟合规则。",
            "risk":"只有最高浓度异常或聚集信号被计为hit。",
            "fields":"signal threshold|dose dependence|saturation|required replicates|fit quality|max concentration",
            "decision":"在揭盲候选分层前冻结HIT_DEFINITION_AND_QC。",
        },
        {
            "group":"结果判定", "title":"什么是technical failure而不是阴性", "question":"不溶、蛋白失活、阳性失败、参考异常、聚集和无法拟合分别如何编码？",
            "why":"技术失败若记作阴性，会错误降低模型命中率并污染下一轮训练。",
            "current":"历史流程已认识到前药、构象和准备失败；当前数据字典尚未冻结。",
            "needed":"平台级failure taxonomy、补测规则、整批作废标准和最大可接受失败率。",
            "risk":"模型会学习实验故障，而不是不结合。",
            "fields":"failure code|failure evidence|retest required|batch invalid|final label|comments",
            "decision":"technical failure永不自动转为negative，必须单独统计。",
        },
        {
            "group":"结果判定", "title":"什么才能升级为confirmed binder", "question":"是否要求独立配液、完整曲线、可饱和、竞争、排除聚集和第二种原理实验？",
            "why":"单一技术primary signal容易受非特异吸附、检测干扰或功能间接效应影响。",
            "current":"项目需要将primary hit和confirmed binder作为两级指标。",
            "needed":"每个family的最小确认链、新批次要求、无关蛋白反筛和冲突处理。",
            "risk":"最终只能汇报“有信号”，无法证明新drug-target pair。",
            "fields":"independent retest|full curve|competition|aggregation screen|orthogonal result|final binder status",
            "decision":"预注册family-specific确认链并预留预算。",
        },
        {
            "group":"先导与验证", "title":"是否接受4–8个靶点资格先导", "question":"是否先用少量靶点验证蛋白、assay、通量、失败率和候选层级差异？",
            "why":"pilot不是缩小广谱目标，而是防止直接在数十个未成熟靶点上消耗全部预算。",
            "current":"当前缺前瞻数据，pilot可用于建立实验条件，但不能计入locked命中率。",
            "needed":"pilot靶点、候选构成、时间、成功/停止标准和结果可见范围。",
            "risk":"直接广筛时无法区分模型阴性与assay失败。",
            "fields":"pilot target|candidate lanes|controls|timeline|success criterion|rule changes allowed",
            "decision":"冻结pilot与validation两个不重叠manifest。",
        },
        {
            "group":"先导与验证", "title":"正式广筛是否采用两阶段自适应设计", "question":"固定80×10，还是60–80个靶点各测3–5条后向有信号靶点追加？",
            "why":"固定均匀设计最大化覆盖，自适应设计更可能提高确认hit和信息增益；两者优化目标不同。",
            "current":"当前目标同时要求多靶点广度和尽可能撞到hit，自适应方案更匹配，但需要预注册。",
            "needed":"平台换靶点成本、是否能分阶段提交、追加周期、每靶点上限和剩余预算比例。",
            "risk":"完全平均分配浪费预算；完全自由追加产生事后选择偏差。",
            "fields":"stage A targets|candidates per target|stage B trigger|allocation cap|stopping rule|turnaround",
            "decision":"根据平台能力在会上二选一，并冻结追加算法。",
        },
    ]

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">会议决策</div><h2>三十五、15个实验问题总览</h2>
      {table(['组别', '问题', '必须形成的输出'], [[q['group'], q['title'], q['decision']] for q in meeting_questions], 'compact')}
      <div class="callout blue"><strong>会议目标：</strong>不是要求实验组评价计算模型，而是让实验目标、平台能力、预算、判定和数据回传成为可写入流程的字段。</div>
    </section>
    """)
    pages.extend(meeting_page(i, item) for i, item in enumerate(meeting_questions, start=1))

    pages.append(f"""
    <section class="page">
      <div class="section-kicker">会后交付</div><h2>三十六、四张表与一个预注册包</h2>
      {table(['交付物', '最少字段', '决定什么'], [
          ['ASSAY_READINESS_MATRIX', 'target、construct、平台、状态、阳性、inactive、通量、成本、负责人', '哪些靶点现在真的能做'],
          ['EXPERIMENT_BUDGET_LEDGER', '预算单位、浓度、重复、控制、复测、正交、时间、费用', '实际能测试多少pair'],
          ['HIT_DEFINITION_AND_QC', 'primary阈值、failure code、确认链、批次作废规则', '什么结果进入模型'],
          ['EXPERIMENT_DATA_DICTIONARY', 'pair ID、raw data、metadata、fit、QC、final label', '数据是否可复现和回流'],
          ['PILOT_AND_LOCKED_VALIDATION_REGISTRY', '不重叠manifest、规则版本、SHA256、揭盲时间', '性能评价是否无偏'],
      ])}
      <h3>实验返回的标准标签</h3>
      <div class="branch-grid">
        <div class="branch"><h3>有效负面结果</h3><p>confirmed negative、primary negative。</p><p>必须有合格QC和足够测试范围。</p></div>
        <div class="branch"><h3>不可用于阴性</h3><p>technical failure、inconclusive。</p><p>单独统计并决定是否补测。</p></div>
        <div class="branch"><h3>初步信号</h3><p>primary signal、reproducible primary hit。</p><p>尚不能称confirmed binder。</p></div>
        <div class="branch"><h3>确认结果</h3><p>orthogonally confirmed binder、target-engagement hit、functional hit。</p><p>三类终点仍需分别报告。</p></div>
      </div>
      <div class="callout teal"><strong>下一版漏斗的数字来源：</strong>只有当上述五个交付物冻结后，才能把16,995继续写成可核算的Gate-B、Pair gate、assay-ready、pilot、broad screen与confirmed binder数字链。</div>
    </section>
    """)

    source_rows = [source_row(label, path) for label, path in SOURCES.items()]
    pages.append(f"""
    <section class="page">
      <div class="section-kicker">来源与审计 1/2</div><h2>三十七、数字来源与校验和</h2>
      <p class="lead">以下文件共同定义历史分支、当前主链、Target gate 和 Discovery 队列。路径用于项目内追溯，SHA256 用于确认报告读取的具体版本。</p>
      {table(['来源标签', '项目内路径', 'SHA256 前缀'], source_rows[:11], 'compact source')}
      <div class="callout blue"><strong>来源分组：</strong>本页覆盖候选口径盘点、strict895、106k物理分支、全空间v4以及当前remote主表。</div>
    </section>
    """)
    pages.append(f"""
    <section>
      <div class="section-kicker">来源与审计 2/2</div><h2>三十八、当前计算来源与审计结论</h2>
      {table(['来源标签', '项目内路径', 'SHA256 前缀'], source_rows[11:], 'compact source')}
      <div class="callout teal"><strong>自动核对：</strong>{sum(checks.values())}/{len(checks)} 项集合与分区等式通过。PDF生成后另做页数、空白页、文本缺失和页面边界审计。</div>
      <p class="small">内部目录仍保留历史版本号用于文件追溯；正式流程名称统一为“FDA 老药新靶点：靶点内校准亲和发现流程”。本附录中的计算分数、结构pose和历史校准均不是湿实验事实。</p>
    </section>
    """)

    html_text = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{PROJECT_NAME}：详细附录</title><style>{css}</style></head><body>{''.join(pages)}</body></html>"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html_text, encoding="utf-8")
    HTML(filename=str(HTML_OUT), base_url=str(ROOT)).write_pdf(str(PDF_OUT))

    doc = fitz.open(PDF_OUT)
    pdf_text = "\n".join(page.get_text() for page in doc)
    required = [
        "全局漏斗、历代口径", "统一计数单位", "哪些数字可以相加减", "历代漏斗不是一条直线",
        "当前正式漏斗总览", "334,749 的完整分流", "30,000 如何成为 16,995",
        "当前漏斗为什么停在 16,995", "19个关键问题", "15个实验问题总览",
        "ASSAY_READINESS_MATRIX", "EXPERIMENT_BUDGET_LEDGER", "Gate-A", "Gate-B",
        "Gate 到底是什么", "“远程”具体指什么", "实验前AI团队的六类核心能力",
        "无法同时最优化的四角约束", "必须共同选择的四种模式",
        "把“提高阳性率”改成可测量目标", "模型—实验联合资格先导与前瞻校准实验",
        "DRUG_ENTITY_REGISTRY", "PAIR_EVIDENCE_MATRIX", "OUT_OF_DOMAIN",
    ]
    missing = [term for term in required if term not in pdf_text]
    layout: list[str] = []
    blank_pages: list[int] = []
    for page_no, page in enumerate(doc, start=1):
        if not page.get_text().strip():
            blank_pages.append(page_no)
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                layout.append(f"page{page_no}:{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}")
    if missing or blank_pages or layout:
        raise RuntimeError(f"PDF audit failed: missing={missing}; blank={blank_pages}; layout={layout[:20]}")

    shutil.copy2(PDF_OUT, ROOT_PDF_OUT)
    audit = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": PROJECT_NAME,
        "pdf": str(PDF_OUT.relative_to(ROOT)),
        "root_pdf": str(ROOT_PDF_OUT.relative_to(ROOT)),
        "html": str(HTML_OUT.relative_to(ROOT)),
        "pdf_pages": len(doc),
        "pdf_bytes": PDF_OUT.stat().st_size,
        "pdf_sha256": sha256(PDF_OUT),
        "source_sha256": {label: sha256(path) for label, path in SOURCES.items()},
        "numerical_checks": checks,
        "required_sections_missing": missing,
        "blank_pages": blank_pages,
        "layout_boundary_problems": layout,
        "key_lineage": {
            "id_pairs": full_v4["project_cartesian_rows"],
            "physical_pairs": full_v4["project_physical_pair_rows"],
            "remote_eligible": remote["remote_pair_eligible_rows"],
            "strict_homology_audited": homology["strict_structure_remote_homology_audited_rows"],
            "drugclip_scored": drugclip["remote_strict_scored_rows"],
            "stage2_queue": stage2["selected_rows"],
            "target_gated_discovery": discovery["selected_rows"],
            "discovery_targets": discovery["unique_targets"],
        },
        "problem_pages": len(issues),
        "meeting_question_pages": len(meeting_questions),
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

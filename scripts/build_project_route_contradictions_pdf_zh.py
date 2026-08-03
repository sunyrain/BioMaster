#!/usr/bin/env python3
"""Build a teacher-facing PDF for the target-calibrated affinity discovery workflow."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/current_production_package_v2/project_progress_discussion_zh"
PDF_OUT = OUT_DIR / "FDA_OLD_DRUG_NEW_TARGET_ROUTE_CHANGES_AND_CORE_CONTRADICTIONS_ZH.pdf"
HTML_OUT = OUT_DIR / "FDA_OLD_DRUG_NEW_TARGET_ROUTE_CHANGES_AND_CORE_CONTRADICTIONS_ZH.html"
AUDIT_OUT = OUT_DIR / "FDA_OLD_DRUG_NEW_TARGET_ROUTE_CHANGES_AND_CORE_CONTRADICTIONS_AUDIT.json"
ROOT_PDF_OUT = ROOT / PDF_OUT.name
CANONICAL_PDF_OUT = (
    ROOT / "FDA_OLD_DRUG_TARGET_CALIBRATED_AFFINITY_DISCOVERY_KEY_QUESTIONS_ZH.pdf"
)
PROJECT_NAME = "FDA 老药新靶点：靶点内校准亲和发现流程"

UNIVERSE_PATH = (
    ROOT
    / "outputs/current_production_package_v2/full_untruncated_universe_v4"
    / "full_untruncated_universe_v4_summary.json"
)
FORMAL_PATH = (
    ROOT
    / "outputs/current_production_package_v2/formal_full_universe_v4"
    / "formal_completion_summary_v4_complete.json"
)
UTILITY_PATH = (
    ROOT
    / "outputs/current_production_package_v2/project_utility_audit_v8"
    / "PROJECT_CANDIDATE_UTILITY_AUDIT_V8.json"
)
V8_CALIBRATION_SUMMARY_PATH = (
    ROOT
    / "outputs/affinity_experiment_package_v8/target_calibration/"
    "GNINA_TARGET_CHANNEL_CALIBRATION_V8_SUMMARY.json"
)
V8_CALIBRATION_PATH = (
    ROOT
    / "outputs/affinity_experiment_package_v8/target_calibration/"
    "GNINA_TARGET_CHANNEL_CALIBRATION_V8.csv"
)
V8_CONTROL_MANIFEST_PATH = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/target_docking_calibration_v2/"
    "GNINA_TARGET_CALIBRATION_CONTROLS_V2.csv.gz"
)
V8_CONTROL_SCORES_PATH = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/gnina_target_calibration_v2/"
    "GNINA_TARGET_CALIBRATION_LIGAND_SCORES_V2.csv.gz"
)
V8_DISCOVERY_SUMMARY_PATH = (
    ROOT
    / "outputs/affinity_experiment_package_v8/discovery_queue/"
    "GNINA_REMOTE_DISCOVERY_QUEUE_V8_SUMMARY.json"
)
V8_EXPANDED_SUMMARY_PATH = (
    ROOT
    / "outputs/affinity_experiment_package_v8/expanded_controls_manifest/"
    "TARGET_DOCKING_CALIBRATION_V8_SUMMARY.json"
)
V8_RECEPTOR_PATH = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/experimental_holo_validation_v2/"
    "TARGET_DOCKING_RECEPTOR_SELECTION_463_V2.csv"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        "</div>"
    )


def route_stage(label: str, title: str, text: str, conclusion: str) -> str:
    return f"""
    <article class="stage">
      <div class="stage-label">{esc(label)}</div>
      <div class="stage-body">
        <h3>{esc(title)}</h3>
        <p>{esc(text)}</p>
        <div class="stage-conclusion"><strong>阶段认识：</strong>{esc(conclusion)}</div>
      </div>
    </article>
    """


def question_rows(rows: list[tuple[str, str, str]]) -> str:
    return "".join(
        f"<tr><td>{esc(name)}</td><td>{esc(status)}</td><td>{esc(action)}</td></tr>"
        for name, status, action in rows
    )


def compute_v8_control_audit() -> dict[str, Any]:
    """Quantify control provenance and target-gate sensitivity to ligand size."""

    manifest = pd.read_csv(V8_CONTROL_MANIFEST_PATH, low_memory=False)
    positive = manifest[manifest["control_class"].eq("positive")].copy()
    negative = manifest[manifest["control_class"].eq("negative")].copy()
    explicit_inactive = int(negative["any_explicit_inactive"].eq(1).sum())
    numeric_low_activity = int(
        (
            negative["any_explicit_inactive"].eq(0)
            & negative["max_pchembl"].notna()
        ).sum()
    )

    properties = [
        "control_heavy_atoms",
        "control_rotatable_bonds",
        "control_formal_charge",
    ]
    scores = pd.read_csv(V8_CONTROL_SCORES_PATH, low_memory=False)
    controls = scores.merge(
        manifest[["control_pair_id", *properties]].drop_duplicates("control_pair_id"),
        on="control_pair_id",
        how="left",
        validate="one_to_one",
    )
    centers = controls.groupby("sequence_key")[properties].transform("median")
    features = controls[properties].to_numpy(dtype=float) - centers.to_numpy(dtype=float)
    is_negative = controls["control_class"].eq("negative").to_numpy()

    adjusted_channels = {
        "best_cnn_affinity": "cnn_size_adjusted",
        "score_vina_directional": "vina_size_adjusted",
    }
    coefficients: dict[str, dict[str, float]] = {}
    for raw_column, adjusted_column in adjusted_channels.items():
        raw = pd.to_numeric(controls[raw_column], errors="coerce").to_numpy(dtype=float)
        target_center = (
            controls.groupby("sequence_key")[raw_column]
            .transform("median")
            .to_numpy(dtype=float)
        )
        train = is_negative & np.isfinite(raw) & np.isfinite(features).all(axis=1)
        model = HuberRegressor(epsilon=1.35, alpha=0.1, max_iter=1000)
        model.fit(features[train], (raw - target_center)[train])
        controls[adjusted_column] = raw - model.predict(features)
        coefficients[raw_column] = {
            property_name: float(value)
            for property_name, value in zip(properties, model.coef_)
        }

    target_rows: list[dict[str, Any]] = []
    for sequence_key, group in controls.groupby("sequence_key"):
        labels = group["control_class"].eq("positive").astype(int).to_numpy()
        positive_n = int(labels.sum())
        negative_n = int((1 - labels).sum())
        if positive_n < 8 or negative_n < 8:
            continue
        prevalence = positive_n / (positive_n + negative_n)
        row: dict[str, Any] = {"sequence_key": sequence_key}
        for column in [
            "best_cnn_affinity",
            "score_vina_directional",
            "cnn_size_adjusted",
            "vina_size_adjusted",
        ]:
            channel_scores = pd.to_numeric(
                group[column], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(channel_scores)
            auc = roc_auc_score(labels[valid], channel_scores[valid])
            ap = average_precision_score(labels[valid], channel_scores[valid])
            row[f"{column}_pass"] = bool(
                auc >= 0.65 and ap >= prevalence + 0.10
            )
        target_rows.append(row)

    target_metrics = pd.DataFrame(target_rows)
    raw_union = (
        target_metrics["best_cnn_affinity_pass"]
        | target_metrics["score_vina_directional_pass"]
    )
    adjusted_union = (
        target_metrics["cnn_size_adjusted_pass"]
        | target_metrics["vina_size_adjusted_pass"]
    )
    raw_dual = (
        target_metrics["best_cnn_affinity_pass"]
        & target_metrics["score_vina_directional_pass"]
    )
    adjusted_dual = (
        target_metrics["cnn_size_adjusted_pass"]
        & target_metrics["vina_size_adjusted_pass"]
    )
    return {
        "control_rows": int(len(manifest)),
        "positive_rows": int(len(positive)),
        "negative_rows": int(len(negative)),
        "explicit_inactive_rows": explicit_inactive,
        "numeric_low_activity_rows": numeric_low_activity,
        "positive_mw_median": float(positive["control_mw"].median()),
        "negative_mw_median": float(negative["control_mw"].median()),
        "positive_heavy_atoms_median": float(
            positive["control_heavy_atoms"].median()
        ),
        "negative_heavy_atoms_median": float(
            negative["control_heavy_atoms"].median()
        ),
        "raw_union_targets": int(raw_union.sum()),
        "adjusted_union_targets": int(adjusted_union.sum()),
        "raw_dual_targets": int(raw_dual.sum()),
        "adjusted_dual_targets": int(adjusted_dual.sum()),
        "raw_union_lost_after_adjustment": int((raw_union & ~adjusted_union).sum()),
        "new_after_adjustment": int((~raw_union & adjusted_union).sum()),
        "size_model_coefficients": coefficients,
    }


def main() -> None:
    universe = read_json(UNIVERSE_PATH)
    formal = read_json(FORMAL_PATH)
    utility = read_json(UTILITY_PATH)
    calibration = read_json(V8_CALIBRATION_SUMMARY_PATH)
    discovery = read_json(V8_DISCOVERY_SUMMARY_PATH)
    expanded = read_json(V8_EXPANDED_SUMMARY_PATH)
    control_audit = compute_v8_control_audit()
    receptors = pd.read_csv(V8_RECEPTOR_PATH, low_memory=False)

    full_pairs = int(universe["full_conplex_rows"])
    physical_pairs = int(utility["scope"]["project_physical_pairs"])
    model_ligands = 723
    project_targets = 463
    calibration_scored = int(calibration["targets_with_control_scores"])
    calibration_evaluable = int(calibration["targets_evaluable"])
    calibration_admitted = int(calibration["admitted_union_targets"])
    calibration_dual = int(calibration["dual_pass_targets"])
    discovery_pairs = int(discovery["selected_rows"])
    discovery_targets = int(discovery["unique_targets"])
    discovery_ligands = int(discovery["unique_ligands"])
    discovery_scaffolds = int(discovery["unique_scaffolds"])
    experimental_holo_pairs = int(
        discovery["receptor_source_counts"]["experimental_holo"]
    )
    alphafold_pairs = int(
        discovery["receptor_source_counts"]["alphafold_p2rank"]
    )
    expanded_controls = int(expanded["control_rows"])
    expanded_targets = int(expanded["project_targets"])
    experimental_holo_targets = int(
        receptors["docking_receptor_source"].eq("experimental_holo").sum()
    )
    alphafold_targets = int(
        receptors["docking_receptor_source"].eq("alphafold_p2rank").sum()
    )

    if control_audit["raw_union_targets"] != calibration_admitted:
        raise RuntimeError("Target-gate sensitivity audit does not reproduce current workflow")
    if experimental_holo_pairs + alphafold_pairs != discovery_pairs:
        raise RuntimeError("Discovery receptor-source counts do not sum")

    css = """
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc); }
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc); font-weight:700; }
    @page {
      size:A4; margin:15mm 16mm 15mm 16mm;
      @top-left { content:"FDA老药新靶点 · 路线与关键问题"; color:#697270; font:8.2pt NotoCJK; }
      @top-right { content:string(section); color:#697270; font:8.2pt NotoCJK; }
      @bottom-left { content:"BioMaster · 项目讨论材料"; color:#7c8583; font:7.8pt NotoCJK; }
      @bottom-right { content:counter(page) " / " counter(pages); color:#7c8583; font:7.8pt NotoCJK; }
    }
    @page:first { @top-left{content:none;} @top-right{content:none;} @bottom-left{content:none;} }
    * { box-sizing:border-box; }
    body { margin:0; font-family:NotoCJK,sans-serif; color:#182321; font-size:9.45pt; line-height:1.55; }
    h1,h2,h3,p { margin-top:0; }
    h1 { font-size:27pt; line-height:1.3; letter-spacing:0; margin:0 0 7mm; }
    h2 { string-set:section content(); font-size:20pt; line-height:1.3; border-bottom:2px solid #147b73; padding-bottom:3mm; margin:0 0 5mm; }
    h3 { font-size:12.2pt; line-height:1.4; margin-bottom:1.5mm; }
    p { margin-bottom:3mm; }
    ul { padding-left:5mm; margin:2mm 0 4mm; }
    li { margin-bottom:1.5mm; }
    a { color:#126d67; text-decoration:none; }
    .page { min-height:266mm; page-break-after:always; }
    .page:last-child { page-break-after:auto; }
    .cover { position:relative; padding-top:28mm; }
    .cover::before { content:""; position:absolute; top:0; left:0; width:42mm; height:4mm; background:#bd6d4f; }
    .eyebrow { color:#147b73; font-size:10.5pt; font-weight:700; margin-bottom:6mm; }
    .subtitle { font-size:14pt; color:#465451; line-height:1.65; max-width:158mm; }
    .cover-rule { border-top:1px solid #cbd4d1; margin:12mm 0 8mm; }
    .cover-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6mm; }
    .cover-grid div { border-top:3px solid #147b73; padding-top:3mm; }
    .cover-grid strong { display:block; font-size:11pt; margin-bottom:1mm; }
    .cover-grid span { color:#687370; font-size:8.5pt; }
    .cover-callout { margin-top:13mm; background:#f4f1ef; border-left:4px solid #bd6d4f; padding:5mm 6mm; color:#4d4845; }
    .cover-foot { margin-top:16mm; color:#78817f; font-size:8.4pt; }
    .lead { font-size:12.2pt; line-height:1.72; color:#344744; }
    .route-map { display:grid; grid-template-columns:repeat(3,1fr); gap:5mm; margin:6mm 0; }
    .route { border:1px solid #d3dcda; min-height:61mm; page-break-inside:avoid; }
    .route .route-no { background:#203d39; color:#fff; font-weight:700; padding:3mm 4mm; }
    .route .route-body { padding:4mm; }
    .route h3 { color:#126d67; margin-bottom:2mm; }
    .route p { color:#505e5b; }
    .route strong { color:#8b4d39; }
    .callout { border-left:4px solid #bd6d4f; background:#f4f1ef; padding:4mm 5mm; margin:5mm 0; }
    .callout.teal { border-left-color:#147b73; background:#eef4f2; }
    .callout.blue { border-left-color:#5277a2; background:#f1f4f8; }
    .callout strong { color:#823f2b; }
    .callout.teal strong { color:#126d67; }
    .callout.blue strong { color:#365c84; }
    .stage { display:grid; grid-template-columns:30mm 1fr; gap:4mm; margin-bottom:4mm; page-break-inside:avoid; }
    .stage-label { background:#203d39; color:#fff; font-weight:700; padding:4mm; display:flex; align-items:center; justify-content:center; text-align:center; }
    .stage-body { border:1px solid #d4dcda; padding:4mm; }
    .stage-body p { color:#475451; margin-bottom:2mm; }
    .stage-conclusion { background:#f4f6f5; padding:2.5mm 3mm; color:#52605d; }
    .stage-conclusion strong { color:#126d67; }
    .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; margin:5mm 0; }
    .metric { border:1px solid #d4dcda; border-top:3px solid #147b73; padding:4mm; min-height:28mm; background:#fbfcfb; }
    .metric-label { color:#65706d; font-size:8.2pt; }
    .metric-value { color:#173f3a; font-size:19pt; line-height:1.25; font-weight:700; margin:1mm 0; }
    .metric-note { color:#78827f; font-size:7.8pt; }
    table { width:100%; border-collapse:collapse; margin:4mm 0 6mm; font-size:8.55pt; }
    th { background:#203d39; color:#fff; padding:2.5mm; text-align:left; }
    td { border-bottom:1px solid #d9e0de; padding:2.5mm; vertical-align:top; }
    tr:nth-child(even) td { background:#f6f8f7; }
    .conflict td:first-child { width:33mm; color:#126d67; font-weight:700; }
    .conflict td:nth-child(2) { width:61mm; }
    .root-box { margin:8mm 0; padding:8mm; background:#f3eeee; border:2px solid #bd6d4f; }
    .root-box .label { color:#8b4d39; font-weight:700; margin-bottom:3mm; }
    .root-box .quote { font-size:17pt; line-height:1.65; font-weight:700; color:#263431; }
    .logic { display:grid; grid-template-columns:1fr 15mm 1fr; align-items:stretch; margin:6mm 0; }
    .logic-box { border:1px solid #d4dcda; padding:5mm; background:#fbfcfb; }
    .logic-box h3 { color:#126d67; }
    .logic-arrow { display:flex; align-items:center; justify-content:center; color:#bd6d4f; font-size:20pt; font-weight:700; }
    .evidence-row { display:grid; grid-template-columns:35mm 1fr 1fr; margin-bottom:3mm; page-break-inside:avoid; }
    .evidence-name { background:#203d39; color:#fff; padding:3mm; font-weight:700; display:flex; align-items:center; }
    .evidence-answer, .evidence-gap { border:1px solid #d7dddc; border-left:none; padding:3mm; }
    .evidence-answer strong { color:#126d67; }
    .evidence-gap strong { color:#8b4d39; }
    .audit-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:5mm; }
    .audit-card { border:1px solid #d4dcda; padding:4mm; page-break-inside:avoid; }
    .audit-card h3 { color:#126d67; margin-bottom:2mm; }
    .audit-card .big { font-size:22pt; color:#173f3a; font-weight:700; line-height:1.2; }
    .audit-card p { color:#52605d; margin:2mm 0 0; }
    .decision { border:1px solid #d4dcda; border-left:4px solid #147b73; padding:4mm 5mm; margin-bottom:4mm; page-break-inside:avoid; }
    .decision h3 { color:#126d67; }
    .decision p { margin:0; }
    .small { color:#697370; font-size:8pt; }
    """

    resolved_questions = [
        (
            "第一阶段任务",
            "亲和与直接 interaction 优先；疾病、暴露和治疗价值在 hit 后研判。",
            "不再讨论“亲和优先还是疾病优先”，只需冻结实验终点。",
        ),
        (
            "疾病知识的角色",
            "Open Targets、网络和通路不进入亲和主排序。",
            "疾病证据仅用于 confirmed hit 的机制解释与转化分层。",
        ),
        (
            "Known recall 与 Discovery",
            "已知关系用于独立校准；Discovery 排除 exact known pair。",
            "不再用 Discovery 候选包计算已知召回。",
        ),
        (
            "跨靶点原始分数",
            "停止把 GNINA、Boltz 或综合分跨靶点解释为统一亲和尺度。",
            "先做 Target gate，再仅在同靶点、同协议内排序。",
        ),
        (
            "旧 1000 的定位",
            "旧包冻结为历史计算储备库，不再称为高概率 binder。",
            "新实验包由当前门槛和实验容量产生，不继承旧等级。",
        ),
        (
            "第一轮靶点边界",
            "GPCR、复杂膜蛋白及需特殊状态的靶点单列，不进入通用首轮 assay。",
            "这是实验范围选择，不再列为待解决问题；后续可设专门赛道。",
        ),
    ]
    retained_questions = [
        (
            "远程发现适用域",
            "Gate-A 只证明历史局部判别，尚无 cold-scaffold/时间外推 Gate-B。",
            "完成低相似留出后，才能决定 N2/N3 候选的使用资格。",
        ),
        (
            "历史阴性的真实性",
            "explicit inactive 与低活性来自实验，但不是统一热力学 non-binder。",
            "按标签、assay、最高测试浓度和物化匹配分层报告。",
        ),
        (
            "统计稳定性与通道偏差",
            "8+8 是最低条件；234 为两通道取并集，仍有小样本和选择膨胀。",
            "增加 bootstrap、scaffold sensitivity，并预先冻结主通道。",
        ),
        (
            "结构与实验实体一致性",
            "受体、口袋和 pose 可追溯，但 construct、辅因子和构象状态未逐靶点确认。",
            "Target gate 应具体到 target+construct+structure+pocket+protocol。",
        ),
        (
            "高覆盖与前瞻精度",
            "历史校准不能给出未知 pair 的 precision、FDR 或真实 binder 概率。",
            "只能通过预注册、正交的前瞻实验建立。",
        ),
        (
            "靶点准入与 pair 真实性",
            "口袋和 assayable 只证明靶点可做，不能证明具体药物会结合。",
            "保留 pair-level 分层，但不把 P1/P2/P3 解释为命中概率。",
        ),
    ]
    new_questions = [
        (
            "实验终点分层",
            "direct binding、target engagement 和 functional activity 不能合并。",
            "会议冻结各技术属于哪类终点及分别的成功指标。",
        ),
        (
            "Assay readiness",
            "171 个计算准入靶点不等于已有 171 套可运行实验。",
            "实验组返回成熟/可转移/需开发/不可做清单。",
        ),
        (
            "实验实体与对照",
            "计算结构必须对应真实 isoform、construct、辅因子和蛋白状态。",
            "每个靶点确认 construct、阳性对照和同实验 inactive。",
        ),
        (
            "“1000”的预算单位",
            "pair、浓度曲线、孔位、进样次数及是否含对照的成本完全不同。",
            "形成按平台拆分的预算账本，并预留复测和正交确认。",
        ),
        (
            "先导与正式验证",
            "用于调规则的先导数据不能再作为无偏命中率验证集。",
            "预先分开 assay-development set 与 locked validation set。",
        ),
        (
            "实验闭环与自适应追加",
            "技术失败不能记作阴性，且固定 80×10 未必最大化 hit。",
            "冻结 hit/QC 标签、原始数据回传格式和广筛后预算再分配规则。",
        ),
    ]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    html_text = f"""<!doctype html>
    <html lang="zh-CN"><head><meta charset="utf-8"><title>{PROJECT_NAME}：路线与关键问题</title><style>{css}</style></head><body>
      <section class="page cover">
        <div class="eyebrow">FDA 老药新靶点 · 靶点内校准亲和发现流程</div>
        <h1>项目路线回顾、<br>关键问题与实验闭环</h1>
        <p class="subtitle">“靶点内校准亲和发现流程”以广谱直接结合发现为目标，使用历史同靶点正负对照限定模型适用范围，并将问题区分为已经解决、仍然保留和本轮新增三类。</p>
        <div class="cover-rule"></div>
        <div class="cover-grid">
          <div><strong>正式流程名称</strong><span>FDA 老药新靶点：靶点内校准亲和发现流程</span></div>
          <div><strong>当前关键进展</strong><span>建立历史同靶点正负对照与 Target gate</span></div>
          <div><strong>本轮会议重点</strong><span>把漏斗后半段转化为可执行实验协议</span></div>
        </div>
        <div class="cover-callout"><strong>材料目的</strong><br>明确哪些事项已经冻结、哪些科学问题仍需验证、哪些实验问题是本轮新增，并说明当前 16,995 条 Discovery 队列如何继续收敛为可信实验包。</div>
        <div class="cover-foot">项目讨论材料 · 中文版 · {generated}</div>
      </section>

      <section class="page">
        <h2>一、项目经历的三次路线切换</h2>
        <p class="lead">项目并非沿着单一目标连续收敛，而是在“广谱亲和发现”“疾病机制收敛”和“物理优先重构”三条路线之间依次切换。每次切换都解决了上一阶段的一部分问题，同时也改变了评价标准。</p>
        <div class="route-map">
          <div class="route"><div class="route-no">路线一</div><div class="route-body"><h3>全量亲和与高召回</h3><p>直接在 FDA 药物–人源靶点空间中寻找可能结合 pair，并通过已知机制评估召回。</p><strong>核心评价：</strong>覆盖与历史阳性召回。</div></div>
          <div class="route"><div class="route-no">路线二</div><div class="route-body"><h3>Disease-Mechanism 收敛</h3><p>加入疾病图谱、网络、通路、表达、文献和方向平衡，形成具有疾病解释的候选包。</p><strong>核心评价：</strong>疾病相关性与机制完整性。</div></div>
          <div class="route"><div class="route-no">路线三</div><div class="route-body"><h3>物理优先与靶点内校准</h3><p>回到 target-engagement，重建实体、靶点和结构空间，并用 ChEMBL 历史正负对照限定 GNINA 的靶点内使用资格。</p><strong>核心评价：</strong>局部判别资格与靶点内排序。</div></div>
        </div>
        <div class="callout"><strong>当前统一口径：</strong>亲和发现是第一阶段，疾病价值是 hit 后第二终点；旧版 A/B 不再作为统一置信度。靶点内历史校准已经建立，前瞻真实性仍需实验形成。</div>
      </section>

      <section class="page">
        <h2>二、路线一：全量亲和与高召回</h2>
        {route_stage("阶段 1", "全量亲和阶段", "最初问题是：FDA 药物与全部人源成药靶点组成数百万 pair，能否直接预测亲和并取前列实验，疾病放到命中以后再讨论。", "任务清楚地指向广谱 drug–target discovery，但搜索空间大、真值极稀疏，单一模型排名难以同时保证覆盖和精度。")}
        {route_stage("阶段 2", "高召回与 ligand fishing 阶段", "为回应“能召回多少已有老药机制”，加入已知靶点、同序列、同家族配体、配体相似性和 ConPLEx，获得了较高的历史 recall。", "该 recall 主要衡量已有知识检索和相似机制传播能力；即使移除直接标签，仍会使用真靶点的其他已知配体，因此不能代表未知新靶点的无泄漏发现能力。")}
        <div class="logic">
          <div class="logic-box"><h3>提高历史召回的做法</h3><p>已知标签、同序列、同家族、靶点已知配体、化学相似性。</p></div>
          <div class="logic-arrow">→</div>
          <div class="logic-box"><h3>同时引入的限制</h3><p>知识泄漏、靶点研究密度偏倚、同 scaffold 再发现，以及对未知 pair 精度缺乏说明。</p></div>
        </div>
        <div class="metrics">
          {metric("完整序列–配体空间", f"{full_pairs:,}", "用于全量 ConPLEx 计算")}
          {metric("正式物理候选空间", f"{physical_pairs:,}", "实体与实验边界统一后")}
          {metric("正式设计边界", f"{model_ligands} × {project_targets}", "723 个 active-moiety 结构 × 463 个唯一序列靶点")}
        </div>
      </section>

      <section class="page">
        <h2>三、路线二：Disease-Mechanism 收敛</h2>
        {route_stage("阶段 3", "Disease-Mechanism 阶段", "为降低大空间假阳性，引入 Open Targets、STRING、TxGNN、Reactome、表达签名和组织表达，先后建立五方向、十九方向、机制桶和疾病分级。", "这些方法提高了候选的疾病可解释性，但没有增加药物是否直接结合该靶点的 pair-level 物理证据。")}
        {route_stage("阶段 4", "文献与固定湿实验包阶段", "项目先后形成 12,696、895、约 2,000/1,000 和 384 等不同规模候选包，并使用文献分级、靶点关注度和疾病平衡组织结果。", "候选数量和文献构成逐渐承担了交付功能，但固定规模并不是从实验命中率或自然证据阈值中产生。")}
        <table>
          <thead><tr><th>加入的信息</th><th>实际增强的维度</th><th>没有直接解决的问题</th></tr></thead>
          <tbody>
            <tr><td>Open Targets / STRING / TxGNN</td><td>靶点–疾病、网络和药物–疾病关联</td><td>候选药物是否直接结合该靶点</td></tr>
            <tr><td>Reactome / GO / 表达签名</td><td>通路、表型和机制解释</td><td>结合强度和特异性</td></tr>
            <tr><td>文献支持与疾病平衡</td><td>已知性、可读性和候选组合结构</td><td>未知 pair 的假阳性率</td></tr>
          </tbody>
        </table>
        <div class="callout blue"><strong>当前定位：</strong>疾病知识保留为 hit 后机制解释和“初步转化可行性”二级终点，不进入亲和主评分，也不替代直接 interaction 的物理验证。</div>
      </section>

      <section class="page">
        <h2>四、路线三：靶点内校准亲和发现流程</h2>
        {route_stage("阶段 5", "物理优先重启", "清理 active moiety、非治疗性分子和不适合第一轮直接 engagement 的靶点，将正式空间冻结为 723 × 463，共 334,749 条。", "先用不可补偿硬门定义实验边界，疾病图谱不参与亲和主筛。")}
        {route_stage("阶段 6", "结构计算队列", "ConPLEx 与 DrugCLIP 只承担低成本计算资源分配；30,000 条进入受体、口袋、Boltz-2 与 GNINA 结构计算，不把低成本分数解释为 Kd。", "结构证据用于形成可检验假说，并保持实验 holo 与 AlphaFold/P2Rank 受体来源可追溯。")}
        {route_stage("阶段 7", "历史局部校准", "从 ChEMBL 37 构建同靶点阳性与实测 inactive/低活性对照，在相同受体、口袋和协议下评估 CNNaffinity 与 Vina，并仅在靶点内部排序 Discovery。", "已经建立局部判别资格；尚未由低相似留出或前瞻实验证明远程发现能力。")}
        <div class="metrics">
          {metric("历史校准对照", f"{control_audit['control_rows']:,}", f"{control_audit['positive_rows']:,} 阳性 / {control_audit['negative_rows']:,} 阴性")}
          {metric("可评价 Target gate", f"{calibration_evaluable}/{calibration_scored}", "仅代表历史局部判别")}
          {metric("Discovery 队列", f"{discovery_pairs:,}", f"{discovery_targets} 靶点 / {discovery_ligands} 药物")}
        </div>
      </section>

      <section class="page">
        <h2>五、已经解决并冻结的问题</h2>
        <p class="lead">这些事项已经转化为明确的项目边界，不再作为本轮会议中的争议项。</p>
        <table class="conflict">
          <thead><tr><th>已解决事项</th><th>当前决定</th><th>后续边界</th></tr></thead>
          <tbody>{question_rows(resolved_questions)}</tbody>
        </table>
        <div class="callout teal"><strong>明确结论：</strong>亲和优先、疾病后置，以及 GPCR/复杂膜蛋白单列，均属于已确定的阶段性选择；讨论重点应转向如何形成可执行、可评价的实验闭环。</div>
      </section>

      <section class="page">
        <h2>六、仍需解决的关键问题</h2>
        <p class="lead">这些问题涉及模型适用域与未知 pair 的真实性，不能由现有历史回顾性结果直接消除。</p>
        <table class="conflict">
          <thead><tr><th>保留问题</th><th>当前证据边界</th><th>下一步处理</th></tr></thead>
          <tbody>{question_rows(retained_questions)}</tbody>
        </table>
        <div class="callout"><strong>核心科学缺口：</strong>历史正负 benchmark 可以证明局部协议有判别信号，但不能直接给出低相似未知 pair 的 precision、FDR 或真实 binder 概率。</div>
      </section>

      <section class="page">
        <h2>七、本轮新增的关键问题</h2>
        <p class="lead">随着流程进入实验决策阶段，重点从“是否继续增加模型”转为“实验怎样执行、怎样判定、怎样回传”。</p>
        <table class="conflict">
          <thead><tr><th>新增问题</th><th>为什么现在必须处理</th><th>会上需要冻结的决定</th></tr></thead>
          <tbody>{question_rows(new_questions)}</tbody>
        </table>
        <div class="callout blue"><strong>本轮应形成四项交付：</strong>Assay Readiness Matrix、Experiment Budget Ledger、Hit Definition and QC、Experiment Data Dictionary。</div>
      </section>

      <section class="page">
        <h2>八、漏斗后半段与新增方法的意义</h2>
        <p>后半段不再增加跨靶点总分，而是让每一步排除一种明确的不确定性。</p>
        <div class="evidence-row"><div class="evidence-name">Gate-A</div><div class="evidence-answer"><strong>作用：</strong>用 ChEMBL 历史阳性/inactive 判断当前结构与评分协议是否具有局部判别资格。</div><div class="evidence-gap"><strong>不代表：</strong>远程发现能力或真实 binder 概率。</div></div>
        <div class="evidence-row"><div class="evidence-name">Gate-B</div><div class="evidence-answer"><strong>作用：</strong>用时间外推、cold-scaffold 和低相似阳性测试远程适用域。</div><div class="evidence-gap"><strong>当前状态：</strong>规则待实现，是 N2/N3 进入实验前的关键补充。</div></div>
        <div class="evidence-row"><div class="evidence-name">Pair gate</div><div class="evidence-answer"><strong>作用：</strong>在同靶点内使用主通道、大小校正和冻结 P1/P2/P3 规则进行相对分层。</div><div class="evidence-gap"><strong>不代表：</strong>不同靶点之间可比较的 Kd 或命中概率。</div></div>
        <div class="evidence-row"><div class="evidence-name">Assay readiness</div><div class="evidence-answer"><strong>作用：</strong>核对 construct、蛋白状态、平台、阳性/阴性对照和可用通量。</div><div class="evidence-gap"><strong>输出：</strong>哪些靶点现在能进入正式筛选，而非哪些靶点更可能结合。</div></div>
        <div class="evidence-row"><div class="evidence-name">资格先导</div><div class="evidence-answer"><strong>作用：</strong>用 4–8 个靶点确认蛋白、assay、失败率和不同候选层级是否可区分。</div><div class="evidence-gap"><strong>限制：</strong>用于调规则的数据必须与 locked validation set 隔离。</div></div>
        <div class="evidence-row"><div class="evidence-name">自适应广筛</div><div class="evidence-answer"><strong>作用：</strong>60–80 个 assay-ready 靶点先测 3–5 条，再向稳定且有信号的靶点追加。</div><div class="evidence-gap"><strong>前提：</strong>预算允许分阶段重新分配，并预留复测与正交确认。</div></div>
        <div class="callout blue"><strong>完整后半段：</strong>16,995 Discovery → Gate-B → Pair gate → assay-ready → 资格先导 → 多靶点广筛 → 自适应追加 → 正交 confirmed binder → hit 后疾病机制。</div>
      </section>

      <section class="page">
        <h2>九、当前漏斗推进到哪里</h2>
        <div class="audit-grid">
          <div class="audit-card"><h3>正式物理空间</h3><div class="big">{physical_pairs:,}</div><p>723 个 active-moiety 结构 × 463 个唯一序列靶点。</p></div>
          <div class="audit-card"><h3>结构计算队列</h3><div class="big">30,000</div><p>ConPLEx/DrugCLIP 只负责计算资源分配，不解释为 Kd。</p></div>
          <div class="audit-card"><h3>历史正负对照</h3><div class="big">{control_audit['control_rows']:,}</div><p>{control_audit['positive_rows']:,} 阳性 / {control_audit['negative_rows']:,} inactive 或低活性。</p></div>
          <div class="audit-card"><h3>Gate-A 可评价</h3><div class="big">{calibration_evaluable}/{calibration_scored}</div><p>{calibration_admitted} 个靶点至少一通道通过历史局部门槛。</p></div>
          <div class="audit-card"><h3>Discovery 队列</h3><div class="big">{discovery_pairs:,}</div><p>{discovery_targets} 靶点、{discovery_ligands} 药物、{discovery_scaffolds} 骨架。</p></div>
          <div class="audit-card"><h3>偏差敏感性</h3><div class="big">{control_audit['raw_union_targets']}→{control_audit['adjusted_union_targets']}</div><p>物化大小校正后 Gate-A 通过靶点变化，说明仍需稳健化。</p></div>
        </div>
        <div class="callout"><strong>尚未完成的后半段：</strong>Gate-B 尚未运行；P1/P2/P3 需在独立留出前冻结；171 个靶点尚未取得实验组 assay-readiness 回传；正式 Assay1000 因预算单位未定义而不能生成。</div>
        <p class="small">171 个 Discovery 靶点已扩展到 {expanded_controls:,} 条历史控制、每靶点至少 12+12。阴性包括 {control_audit['explicit_inactive_rows']:,} 条 explicit inactive 和 {control_audit['numeric_low_activity_rows']:,} 条定量低活性；它们仍不是统一实验条件下的前瞻 non-binder。</p>
      </section>

      <section>
        <h2>十、本轮会议需要形成的决定</h2>
        <div class="decision"><h3>1. 冻结实验终点矩阵</h3><p>逐项确认 SPR、BLI、MST、ITC、竞争结合、NanoBRET、CETSA、DSF、酶活及功能实验分别计入 direct binding、target engagement 或 functional activity。</p></div>
        <div class="decision"><h3>2. 返回 assay-ready 靶点清单</h3><p>对 171 个靶点标注成熟/可转移/需开发/不可做，并提供 isoform、construct、辅因子、阳性对照、同实验 inactive 和平台通量。</p></div>
        <div class="decision"><h3>3. 定义“1000”的真实预算单位</h3><p>明确它是 pair、曲线、孔位还是进样次数，是否包含浓度梯度、重复、板内控制、primary hit 复测与正交确认。</p></div>
        <div class="decision"><h3>4. 拆分资格先导与正式验证</h3><p>先用 4–8 个靶点做 assay-development pilot；正式广筛采用固定 80×10 或 60–80×3–5 后自适应追加，由实验平台成本决定。</p></div>
        <div class="decision"><h3>5. 冻结 hit、失败和数据回传标准</h3><p>区分 primary signal、technical failure、reproducible hit、orthogonally confirmed binder，并返回原始曲线、批次、construct、QC 和拟合元数据。</p></div>
        <div class="callout teal"><strong>会议目标：</strong>把“前瞻验证”具体化为哪些靶点、什么蛋白、哪种技术、多少浓度、哪些对照、什么算 hit、失败如何编码、数据怎样返回。</div>
        <p class="small">数据来源：full-universe、formal completion、ChEMBL controls、GNINA target calibration、Discovery queue、receptor audit 与 expanded-controls manifest；关键数字均由脚本读取和复核。</p>
      </section>
    </body></html>"""

    banned_tone = [
        "老师要求的“召回”究竟是什么",
        "质问老师",
        "老师的路线错误",
        "实验组的要求不合理",
        "V8",
        "核心矛盾",
        "科学矛盾",
    ]
    banned_found = [term for term in banned_tone if term in html_text]
    if banned_found:
        raise RuntimeError(f"Teacher-facing tone check failed: {banned_found}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html_text, encoding="utf-8")
    HTML(filename=str(HTML_OUT), base_url=str(ROOT)).write_pdf(str(PDF_OUT))

    doc = fitz.open(PDF_OUT)
    pdf_text = "\n".join(page.get_text() for page in doc)
    required = [
        "靶点内校准亲和发现流程",
        "项目经历的三次路线切换",
        "路线一：全量亲和与高召回",
        "路线二：Disease-Mechanism 收敛",
        "路线三：靶点内校准亲和发现流程",
        "已经解决并冻结的问题",
        "仍需解决的关键问题",
        "本轮新增的关键问题",
        "漏斗后半段与新增方法的意义",
        "当前漏斗推进到哪里",
        "本轮会议需要形成的决定",
        "Gate-A",
        "Gate-B",
        "Assay Readiness Matrix",
        "Experiment Budget Ledger",
        "technical failure",
    ]
    missing = [term for term in required if term not in pdf_text]
    banned_pdf = [term for term in banned_tone if term in pdf_text]
    layout = []
    if len(doc) != 11:
        layout.append(f"unexpected_page_count:{len(doc)}")
    for page_no, page in enumerate(doc, start=1):
        if not page.get_text().strip():
            layout.append(f"blank_page_{page_no}")
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                layout.append(f"outside_page_{page_no}:{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}")
    if missing or banned_pdf or layout:
        raise RuntimeError(f"PDF audit failed: missing={missing}, banned={banned_pdf}, layout={layout}")
    shutil.copy2(PDF_OUT, CANONICAL_PDF_OUT)

    shutil.copy2(PDF_OUT, ROOT_PDF_OUT)

    audit = {
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(PDF_OUT.relative_to(ROOT)),
        "canonical_pdf": str(CANONICAL_PDF_OUT.relative_to(ROOT)),
        "project_name": PROJECT_NAME,
        "root_pdf": str(ROOT_PDF_OUT.relative_to(ROOT)),
        "html": str(HTML_OUT.relative_to(ROOT)),
        "pdf_pages": len(doc),
        "pdf_bytes": PDF_OUT.stat().st_size,
        "pdf_sha256": sha256(PDF_OUT),
        "root_pdf_sha256": sha256(ROOT_PDF_OUT),
        "canonical_pdf_sha256": sha256(CANONICAL_PDF_OUT),
        "source_sha256": {
            "universe": sha256(UNIVERSE_PATH),
            "formal": sha256(FORMAL_PATH),
            "utility": sha256(UTILITY_PATH),
            "v8_calibration_summary": sha256(V8_CALIBRATION_SUMMARY_PATH),
            "v8_calibration": sha256(V8_CALIBRATION_PATH),
            "v8_control_manifest": sha256(V8_CONTROL_MANIFEST_PATH),
            "v8_control_scores": sha256(V8_CONTROL_SCORES_PATH),
            "v8_discovery_summary": sha256(V8_DISCOVERY_SUMMARY_PATH),
            "v8_expanded_summary": sha256(V8_EXPANDED_SUMMARY_PATH),
            "v8_receptors": sha256(V8_RECEPTOR_PATH),
        },
        "required_sections_missing": missing,
        "banned_tone_phrases_found": banned_pdf,
        "layout_boundary_problems": layout,
        "key_numbers": {
            "full_pairs": full_pairs,
            "physical_pairs": physical_pairs,
            "control_rows": control_audit["control_rows"],
            "positive_control_rows": control_audit["positive_rows"],
            "negative_control_rows": control_audit["negative_rows"],
            "explicit_inactive_rows": control_audit["explicit_inactive_rows"],
            "numeric_low_activity_rows": control_audit["numeric_low_activity_rows"],
            "calibration_scored_targets": calibration_scored,
            "calibration_evaluable_targets": calibration_evaluable,
            "raw_union_targets": control_audit["raw_union_targets"],
            "size_adjusted_union_targets": control_audit["adjusted_union_targets"],
            "raw_dual_targets": control_audit["raw_dual_targets"],
            "size_adjusted_dual_targets": control_audit["adjusted_dual_targets"],
            "discovery_pairs": discovery_pairs,
            "discovery_targets": discovery_targets,
            "discovery_ligands": discovery_ligands,
            "discovery_scaffolds": discovery_scaffolds,
            "experimental_holo_pairs": experimental_holo_pairs,
            "alphafold_p2rank_pairs": alphafold_pairs,
            "experimental_holo_targets": experimental_holo_targets,
            "alphafold_p2rank_targets": alphafold_targets,
            "expanded_control_rows": expanded_controls,
            "expanded_control_targets": expanded_targets,
        },
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

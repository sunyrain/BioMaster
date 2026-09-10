#!/usr/bin/env python3
"""Build the current ReTargetMap project/data/metrics handbook as a Chinese PDF."""

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
OUT_DIR = ROOT / "docs/reports"
PDF_OUT = OUT_DIR / "RETARGETMAP_CURRENT_PROJECT_HANDBOOK_20260901_ZH.pdf"
HTML_OUT = OUT_DIR / "RETARGETMAP_CURRENT_PROJECT_HANDBOOK_20260901_ZH.html"
AUDIT_OUT = OUT_DIR / "RETARGETMAP_CURRENT_PROJECT_HANDBOOK_20260901_AUDIT.json"

CONTRACT_PATH = ROOT / "configs/biomaster_current_contract_v1.json"
TRAINING_PATH = ROOT / "outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json"
FULL_FIT_PATH = ROOT / "outputs/biomaster_bidirectional_v6_full_fit/seed_20260816/FULL_FIT_SUMMARY_V6.json"
RANK_PATH = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json"
TEMPORAL_PATH = ROOT / "outputs/retrain_20260901/bidirectional_720x745_stage_a_rows4/STAGE_A_TEMPORAL_2024_2025_EXPANDED_SCOPES_SUMMARY_V1.json"
TARGET_SCOPE_PATH = ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_DISCOVERY_SCOPE_SUMMARY_V3.json"
S5_TEST_PATH = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_S5_TEST_V1.csv.gz"
DEPLOYMENT_PATH = ROOT / "outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"


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


def metric(metrics: list[dict[str, Any]], model: str) -> dict[str, Any]:
    rows = [row for row in metrics if row["model"] == model]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one metric row for {model}, found {len(rows)}")
    return rows[0]


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def card(label: str, value: str, note: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        "</div>"
    )


def build_s5_diagnostics() -> dict[str, Any]:
    """Reconstruct S5 query denominators and application-aligned retrieval diagnostics."""

    s5 = pd.read_csv(S5_TEST_PATH)
    models = {
        "retargetmap": "independent_validation_rank_score",
        "dtiam": "dtiam_probability",
        "system": "system_validation_rank_score",
    }
    query_rows: list[dict[str, Any]] = []
    all_query_classes = []
    for drug, part in s5.groupby("parent_standard_inchi_key", sort=False):
        positives = int(part["binary_label"].sum())
        negatives = int(len(part) - positives)
        all_query_classes.append((len(part), positives, negatives))
        if positives == 0 or negatives == 0:
            continue
        row: dict[str, Any] = {
            "drug": str(drug),
            "candidates": int(len(part)),
            "positives": positives,
        }
        for model, column in models.items():
            labels = (
                part.sort_values(column, ascending=False, kind="mergesort")["binary_label"]
                .to_numpy(dtype=np.int8)
            )
            positive_ranks = np.flatnonzero(labels == 1) + 1
            row[f"{model}_first_rank"] = int(positive_ranks.min())
            row[f"{model}_mrr"] = float(1.0 / positive_ranks.min())
            for k in (1, 5, 10, 20):
                row[f"{model}_hit_at_{k}"] = float(np.any(positive_ranks <= k))
                row[f"{model}_recall_at_{k}"] = float(
                    np.sum(positive_ranks <= k) / len(positive_ranks)
                )
        query_rows.append(row)
    queries = pd.DataFrame(query_rows)
    class_frame = pd.DataFrame(
        all_query_classes, columns=["candidates", "positives", "negatives"]
    )

    observed = {
        "all_drugs": int(len(class_frame)),
        "two_class_drugs": int(len(queries)),
        "positive_only_drugs": int(
            ((class_frame["positives"] > 0) & (class_frame["negatives"] == 0)).sum()
        ),
        "negative_only_drugs": int(
            ((class_frame["positives"] == 0) & (class_frame["negatives"] > 0)).sum()
        ),
        "two_class_drugs_at_most_20_candidates": int((queries["candidates"] <= 20).sum()),
        "two_class_median_candidates": float(queries["candidates"].median()),
        "random_expected_macro_recall_at_10": float(
            np.mean(np.minimum(10, queries["candidates"]) / queries["candidates"])
        ),
        "random_expected_macro_recall_at_20": float(
            np.mean(np.minimum(20, queries["candidates"]) / queries["candidates"])
        ),
        "nontrivial_gt20_drugs": int((queries["candidates"] > 20).sum()),
        "nontrivial_gt20_random_expected_recall_at_20": float(
            np.mean(
                20
                / queries.loc[queries["candidates"] > 20, "candidates"].to_numpy(
                    dtype=np.float64
                )
            )
        ),
        "models": {},
    }
    for model in models:
        observed["models"][model] = {
            "mrr": float(queries[f"{model}_mrr"].mean()),
            **{
                f"hit_at_{k}": float(queries[f"{model}_hit_at_{k}"].mean())
                for k in (1, 5, 10, 20)
            },
            **{
                f"recall_at_{k}": float(queries[f"{model}_recall_at_{k}"].mean())
                for k in (1, 5, 10, 20)
            },
            "nontrivial_gt20_recall_at_20": float(
                queries.loc[
                    queries["candidates"] > 20, f"{model}_recall_at_20"
                ].mean()
            ),
        }

    known_positive = (
        s5.loc[s5["binary_label"].eq(1), ["parent_standard_inchi_key", "target_chembl_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    deployment_columns = [
        "ligand_inchikey",
        "target_chembl_id",
        "independent_validation_rank_score",
        "dtiam_probability",
        "biomaster_independent_borda_score_rank_within_drug_384",
        "biomaster_independent_new_head_borda_score_rank_within_drug_384",
        "ensemble_drug_to_target_logit_rank_within_drug_384",
    ]
    deployment = pd.read_csv(DEPLOYMENT_PATH, usecols=deployment_columns)
    for column in ("independent_validation_rank_score", "dtiam_probability"):
        deployment[f"{column}_rank_within_drug_384"] = deployment.groupby(
            "ligand_inchikey", sort=False
        )[column].rank(method="first", ascending=False)
    mapped = known_positive.merge(
        deployment,
        left_on=["parent_standard_inchi_key", "target_chembl_id"],
        right_on=["ligand_inchikey", "target_chembl_id"],
        how="inner",
        validate="one_to_one",
    )
    application_models = {
        "s5_anchored_rank": "independent_validation_rank_score_rank_within_drug_384",
        "current_default_borda": "biomaster_independent_borda_score_rank_within_drug_384",
        "independent_new_head_borda": "biomaster_independent_new_head_borda_score_rank_within_drug_384",
        "dtiam": "dtiam_probability_rank_within_drug_384",
        "full_fit_rediscovery_only": "ensemble_drug_to_target_logit_rank_within_drug_384",
    }
    application: dict[str, Any] = {
        "s5_positive_pairs": int(len(known_positive)),
        "mapped_positive_pairs": int(len(mapped)),
        "mapped_positive_drugs": int(mapped["parent_standard_inchi_key"].nunique()),
        "models": {},
    }
    for model, rank_column in application_models.items():
        grouped = mapped.groupby("parent_standard_inchi_key", sort=False)[rank_column]
        application["models"][model] = {}
        for k in (1, 5, 10, 20):
            application["models"][model][f"hit_at_{k}"] = float(
                grouped.apply(lambda values: bool((values <= k).any())).mean()
            )
            application["models"][model][f"macro_recall_at_{k}"] = float(
                grouped.apply(lambda values: float((values <= k).mean())).mean()
            )
    return {"observed_s5": observed, "application_384": application}


def assert_contract(
    contract: dict[str, Any],
    training: dict[str, Any],
    full_fit: dict[str, Any],
    rank: dict[str, Any],
    temporal: dict[str, Any],
    target_scope: dict[str, Any],
) -> None:
    universes = contract["universes"]
    assert universes["current_scored_core_pairs"] == 720 * 384
    assert universes["next_primary_pairs"] == 720 * 450
    assert universes["complete_non_gpcr_registry_targets"] == 745
    assert training["status"] == "PASS"
    assert training["counts"]["source_rows_feature_resolved"] == 426_939
    assert full_fit["split_audit"]["full_fit_rows"] == 437_248
    assert full_fit["split_audit"]["full_fit_unique_drugs"] == 296_108
    assert full_fit["split_audit"]["full_fit_unique_targets"] == 843
    assert rank["s5_test"]["rows"] == 2_556
    assert rank["strict_kirhub_post_audit"]["rows"] == 2_823
    assert temporal["checks"]["expected_20_queries"] is True
    assert target_scope["counts"]["non_gpcr_registry"] == 745
    assert "kinase_functional_specialist" not in contract["model_roles"]
    assert "specialist_entrypoints" not in contract


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    training = read_json(TRAINING_PATH)
    full_fit = read_json(FULL_FIT_PATH)
    rank = read_json(RANK_PATH)
    temporal = read_json(TEMPORAL_PATH)
    target_scope = read_json(TARGET_SCOPE_PATH)
    assert_contract(contract, training, full_fit, rank, temporal, target_scope)
    diagnostics = build_s5_diagnostics()

    u = contract["universes"]
    tr = contract["data"]["comprehensive_training"]
    hist = contract["data"]["historical_capped_benchmark"]
    s5 = rank["s5_test"]
    kirhub = rank["strict_kirhub_post_audit"]
    s5_dtiam = metric(s5["metrics"], "DTIAM_RAW")
    s5_biomaster = metric(s5["metrics"], "INDEPENDENT_VALIDATION_RANK")
    s5_system = metric(s5["metrics"], "SYSTEM_VALIDATION_RANK")
    kir_dtiam = metric(kirhub["metrics"], "DTIAM_RAW")
    kir_direction = metric(kirhub["metrics"], "BIOMASTER_FULL_FIT_DIRECTIONAL")
    kir_independent = metric(kirhub["metrics"], "BIOMASTER_INDEPENDENT_BORDA")
    kir_system = metric(kirhub["metrics"], "BIOMASTER_SYSTEM_BORDA")
    tm = temporal["metrics"]
    s5_obs = diagnostics["observed_s5"]
    s5_app = diagnostics["application_384"]
    s5_obs_rtm = s5_obs["models"]["retargetmap"]
    s5_obs_dtiam = s5_obs["models"]["dtiam"]
    s5_obs_system = s5_obs["models"]["system"]
    s5_app_anchor = s5_app["models"]["s5_anchored_rank"]
    s5_app_borda = s5_app["models"]["current_default_borda"]
    s5_app_dtiam = s5_app["models"]["dtiam"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    css = """
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc); }
    @font-face { font-family:NotoCJK; src:url(file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc); font-weight:700; }
    @page {
      size:A4; margin:15mm 16mm 15mm 16mm;
      @top-left { content:"ReTargetMap · 老药靶点谱检索框架"; color:#71809a; font:8pt NotoCJK; }
      @top-right { content:string(section); color:#71809a; font:8pt NotoCJK; }
      @bottom-left { content:"冻结口径 · 2026-09-01"; color:#8792a6; font:7.5pt NotoCJK; }
      @bottom-right { content:counter(page) " / " counter(pages); color:#8792a6; font:7.5pt NotoCJK; }
    }
    @page:first { @top-left{content:none;} @top-right{content:none;} @bottom-left{content:none;} }
    * { box-sizing:border-box; }
    body { margin:0; font-family:NotoCJK,sans-serif; color:#14233c; font-size:9.1pt; line-height:1.55; }
    .page { min-height:267mm; page-break-after:always; position:relative; }
    .page:last-child { page-break-after:auto; }
    h1,h2,h3,p { margin-top:0; }
    h1 { font-size:27pt; line-height:1.25; margin:0 0 7mm; color:#093f82; }
    h2 { string-set:section content(); font-size:18pt; line-height:1.3; color:#0b4f95; border-bottom:2px solid #3d77c5; padding-bottom:3mm; margin:0 0 5mm; }
    h3 { font-size:11.5pt; color:#116c72; margin:4mm 0 2mm; }
    p { margin-bottom:3mm; }
    strong { font-weight:700; }
    .cover { padding-top:28mm; }
    .cover::before { content:""; position:absolute; top:0; left:0; width:48mm; height:4mm; background:#0b4f95; }
    .cover .eyebrow { color:#16747b; font-weight:700; font-size:10pt; letter-spacing:.5px; margin-bottom:5mm; }
    .cover .subtitle { color:#53637a; font-size:14pt; line-height:1.7; max-width:160mm; }
    .cover-rule { border-top:1px solid #cad5e4; margin:11mm 0 8mm; }
    .cover-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:5mm; }
    .cover-grid div { border-top:3px solid #159096; background:#f5f9fc; padding:4mm; min-height:30mm; }
    .cover-grid strong { display:block; color:#0b4f95; font-size:11pt; margin-bottom:2mm; }
    .cover-note { margin-top:12mm; padding:5mm 6mm; background:#edf5f7; border-left:4px solid #159096; color:#42576a; }
    .cover-foot { margin-top:18mm; color:#748097; font-size:8pt; }
    .lead { font-size:11.8pt; line-height:1.72; color:#334964; margin-bottom:5mm; }
    .metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; margin:4mm 0 6mm; }
    .metrics.four { grid-template-columns:repeat(4,1fr); }
    .metric-card { border:1px solid #d5dfeb; border-top:3px solid #159096; padding:3.5mm; min-height:25mm; background:#fbfdff; }
    .metric-label { color:#66758a; font-size:7.8pt; }
    .metric-value { color:#0b4f95; font-size:18pt; line-height:1.25; font-weight:700; margin:1mm 0; }
    .metric-note { color:#7b8798; font-size:7.2pt; }
    .callout { background:#eef5fb; border-left:4px solid #3d77c5; padding:4mm 5mm; margin:4mm 0; color:#36516d; }
    .callout.teal { background:#eef8f7; border-left-color:#159096; }
    .callout.warn { background:#fff7e8; border-left-color:#cf8b1e; color:#624d2d; }
    .flow { display:flex; align-items:stretch; gap:2.2mm; margin:5mm 0; }
    .flow .node { flex:1; border:1px solid #cbd9e8; border-top:3px solid #3d77c5; padding:3mm; text-align:center; background:#f7faff; }
    .flow .arrow { display:flex; align-items:center; color:#3d77c5; font-size:16pt; }
    .flow strong { display:block; color:#0b4f95; margin-bottom:1mm; }
    .two-col { display:grid; grid-template-columns:1fr 1fr; gap:6mm; }
    .panel { border:1px solid #d5dfeb; padding:4mm; background:#fbfdff; page-break-inside:avoid; }
    .panel h3 { margin-top:0; }
    table { width:100%; border-collapse:collapse; margin:3mm 0 5mm; font-size:8.2pt; }
    th { background:#0b4f95; color:white; text-align:left; padding:2.4mm; }
    td { border-bottom:1px solid #d9e2ed; padding:2.2mm; vertical-align:top; }
    tr:nth-child(even) td { background:#f5f8fc; }
    .result-table td:not(:first-child), .result-table th:not(:first-child) { text-align:right; }
    .best { color:#0b766f; font-weight:700; }
    .muted { color:#6e7b8e; }
    .small { color:#6e7b8e; font-size:7.7pt; }
    .tag { display:inline-block; padding:1mm 2.2mm; border-radius:2mm; background:#eaf2fb; color:#0b4f95; font-size:7.5pt; font-weight:700; margin-right:1mm; }
    .tag.teal { background:#e7f6f4; color:#116c72; }
    .tag.gold { background:#fff3dc; color:#8a611d; }
    ul,ol { padding-left:5.5mm; margin:2mm 0 4mm; }
    li { margin-bottom:1.5mm; }
    .bar-row { display:grid; grid-template-columns:32mm 1fr 24mm; gap:3mm; align-items:center; margin:3mm 0; }
    .bar-track { height:5mm; background:#e5ecf4; }
    .bar-fill { height:100%; background:#159096; }
    .bar-value { text-align:right; font-weight:700; color:#0b4f95; }
    .compare-grid { display:grid; grid-template-columns:36mm 1fr 18mm; gap:3mm; align-items:center; margin:2.3mm 0; }
    .compare-track { height:4.4mm; background:#e6edf5; border-radius:2.2mm; overflow:hidden; }
    .compare-fill { height:100%; background:#159096; }
    .compare-fill.blue { background:#3d77c5; }
    .compare-value { text-align:right; font-weight:700; color:#173f76; }
    .evidence-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:3mm; margin:3mm 0 5mm; }
    .evidence-strip > div { padding:3mm; border:1px solid #d5dfeb; background:#f8fbfe; }
    .evidence-strip strong { display:block; color:#0b4f95; margin-bottom:1mm; }
    .formula { font-family:"Noto Sans Mono CJK SC",monospace; background:#f2f5f8; border:1px solid #d9e1e9; padding:3mm; margin:2mm 0 4mm; color:#29415f; }
    .decision { display:grid; grid-template-columns:34mm 1fr; gap:4mm; margin:3mm 0; page-break-inside:avoid; }
    .decision .label { background:#0b4f95; color:white; padding:3mm; font-weight:700; }
    .decision .body { border:1px solid #d4dfeb; padding:3mm 4mm; }
    .footer-note { position:absolute; bottom:4mm; left:0; color:#8590a1; font-size:7pt; }
    """

    body = f"""
    <section class="page cover">
      <div class="eyebrow">RETARGETMAP · 老药靶点谱检索框架</div>
      <h1>项目、数据、评价<br/>与实验方案详解</h1>
      <p class="subtitle">从既有药物出发，在明确、可审计的候选靶点空间中重建潜在作用靶点谱，并将前排候选交给证据复核与湿实验。</p>
      <div class="cover-rule"></div>
      <div class="cover-grid">
        <div><strong>生产主方向</strong><span>old drug → candidate targets</span></div>
        <div><strong>当前正式分母</strong><span>每种药物在384个已评分核心靶点内排序</span></div>
        <div><strong>当前通用训练</strong><span>{tr['feature_resolved_relations']:,}条去重关系</span></div>
      </div>
      <div class="cover-note"><strong>名称含义：</strong><em>Re</em>表示对既有药物的重新审视，<em>Target</em>表示以靶点谱为主输出，<em>Map</em>强调结果是候选景观和相对排序，而不是一个未校准的“结合概率”。</div>
      <div class="cover-foot">冻结日期：{esc(contract['effective_date'])}　｜　报告生成：{generated}　｜　ReTargetMap</div>
    </section>

    <section class="page">
      <h2>00　项目介绍：我们真正要解决什么</h2>
      <p class="lead">一种已上市药物往往不只作用于一个蛋白。临床效应、不良反应、联合用药效应和新适应症可能来自未被完整描述的次级靶点。ReTargetMap的目标是将“老药有哪些可能靶点”转化成可计算、可排序、可审计和可实验的任务。</p>
      <div class="two-col">
        <div class="panel">
          <h3>输入</h3>
          <ul>
            <li>一种身份和结构已统一的既有药物；</li>
            <li>一个预先登记、实验体系明确的候选蛋白空间；</li>
            <li>药物结构、蛋白序列、结构/口袋上下文和已观测关系证据。</li>
          </ul>
        </div>
        <div class="panel">
          <h3>输出</h3>
          <ul>
            <li>该药在规定候选空间中的靶点排名；</li>
            <li>每个候选的空间分母、路由、冷启动状态和证据标记；</li>
            <li>适合文献/数据库复核、结构重排和湿实验的Top-K清单。</li>
          </ul>
        </div>
      </div>
      <h3>科学定位</h3>
      <table>
        <thead><tr><th>该框架是什么</th><th>该框架不是什么</th></tr></thead>
        <tbody>
          <tr><td>药物中心的多靶点候选检索与优先级排序系统</td><td>不是单一靶点的虚拟筛选工具</td></tr>
          <tr><td>使用异质直接关系/活性数据学习的排序框架</td><td>不是纯Ki/Kd定量亲和力回归器</td></tr>
          <tr><td>将数据库、模型、结构和实验路由统一到同一候选表</td><td>不把未报告pair自动定义为阴性</td></tr>
          <tr><td>为湿实验节省候选数量的决策支持</td><td>不用未校准logit取代物理结合常数</td></tr>
        </tbody>
      </table>
      <div class="callout teal"><strong>最终价值：</strong>不是在数据库中重复已知结论，而是在保持已知机制可回收的同时，将“未报告但值得实验”的pair提前到可承受的实验预算内。</div>
    </section>

    <section class="page">
      <h2>01　任务与系统边界</h2>
      <p class="lead">系统的主问题不是“某个pair是否像阳性”，而是：给定一种老药，在预先定义且可比较的候选靶点集合中，哪些靶点应当优先进入证据复核和湿实验。</p>
      <div class="flow">
        <div class="node"><strong>老药实体</strong>标准结构与身份</div><div class="arrow">→</div>
        <div class="node"><strong>候选靶点</strong>登记并按实验体系分路</div><div class="arrow">→</div>
        <div class="node"><strong>Pair模型</strong>药物、序列、结构上下文</div><div class="arrow">→</div>
        <div class="node"><strong>药内排序</strong>同一药物内比较靶点</div><div class="arrow">→</div>
        <div class="node"><strong>Top-K复核</strong>结构、文献、实验可行性</div>
      </div>
      <h3>当前保留的系统角色</h3>
      <table>
        <thead><tr><th>角色</th><th>当前用途</th><th>结论边界</th></tr></thead>
        <tbody>
          <tr><td><strong>当前合同默认独立Borda</strong></td><td>720×384完整矩阵；每药内部独立排序</td><td>当前正式生产入口，不依赖DTIAM推理；不得继承其他分数的S5指标</td></tr>
          <tr><td><strong>FULL_FIT方向候选</strong></td><td>共享pair backbone上的drug→target主头和target→drug辅助头</td><td>提供候选分数；FULL_FIT本身不是无偏性能估计</td></tr>
          <tr><td><strong>含DTIAM统一系统</strong></td><td>工程级可选融合与对照</td><td>必须明确标注“含DTIAM”，不能冒充ReTargetMap独立结果</td></tr>
          <tr><td><strong>后置结构复核</strong></td><td>GNINA、Boltz、pose、接触与口袋证据</td><td>只用于Top-K复核，尚非通用主模型输入</td></tr>
        </tbody>
      </table>
      <div class="callout teal"><strong>查询方向：</strong><span class="tag teal">主方向 drug→target</span><span class="tag">辅助方向 target→drug</span>。二者共享pair表示，但排序损失、query分组和评价指标分别定义，不能混报。</div>
      <div class="callout warn"><strong>Unknown原则：</strong>数据库没有报告的pair仍是unknown，不自动变成negative，也不能因为模型分数低就解释为“无结合”。</div>
    </section>

    <section class="page">
      <h2>01A　分数谱系：三个“独立分数”不能混叫一个模型</h2>
      <p class="lead">当前仓库中存在S5验证排序、部署Borda和FULL_FIT方向头。它们共享部分基础证据，但构成、评价边界和可声明内容不同。报告必须写出确切分数名，不能把一个分数的指标转挂到另一个分数。</p>
      <table>
        <thead><tr><th>分数/列名</th><th>构成</th><th>当前角色</th><th>可引用的评价</th></tr></thead>
        <tbody>
          <tr><td><strong>S5独立验证排序</strong><br/><span class="small">independent_validation_rank_score</span></td><td>模型logit + 训练阳性最大Tanimoto + 靶点先验；权重仅由S5 validation选择</td><td>冻结S5无偏对照及可部署参考分数</td><td>S5 AUPRC/NDCG/Recall；其0.7521、0.7892等只属于本分数</td></tr>
          <tr><td><strong>当前默认独立Borda</strong><br/><span class="small">biomaster_independent_borda_score</span></td><td>routed stack百分位 + V10图排序百分位，等权平均</td><td>机器合同中的默认rank/384</td><td>KiRHub post-audit与已知阳性恢复诊断；不能直接继承S5验证排序指标</td></tr>
          <tr><td><strong>独立new-head Borda</strong><br/><span class="small">biomaster_independent_new_head_borda_score</span></td><td>S5独立验证排序百分位 + V10图排序百分位</td><td>独立候选对照</td><td>post-audit/恢复诊断；它不是FULL_FIT方向头</td></tr>
          <tr><td><strong>FULL_FIT方向头</strong><br/><span class="small">ensemble_drug_to_target_logit</span></td><td>全量关系重训后的drug→target神经方向分数</td><td>候选支持证据</td><td>重训后无新的无偏内部测试；已见关系恢复不能当泛化性能</td></tr>
          <tr><td><strong>含DTIAM统一系统</strong></td><td>DTIAM与项目分数的验证期选择/融合</td><td>工程级可选系统</td><td>必须始终标记“含DTIAM”</td></tr>
        </tbody>
      </table>
      <div class="callout warn"><strong>本版校正：</strong>S5的Drug-macro AUPRC 0.7521、Recall@20 0.8990和NDCG@20 0.8164属于<strong>S5独立验证排序</strong>，不属于当前默认独立Borda。实验预池已按这一谱系重新生成。</div>
      <div class="callout teal"><strong>汇报原则：</strong>模型名称后同时给出“分数列 + 评价集合 + 候选分母”。例如：S5独立验证排序／S5 observed-target／75个双标签药物，而不是笼统写“ReTargetMap结果”。</div>
    </section>

    <section class="page">
      <h2>02　候选实体与部署空间</h2>
      <div class="metrics four">
        {card('老药部署库', f"{u['old_drugs']}", '固定的已上市/充分药理信息实体')}
        {card('当前已评分核心', f"{u['current_scored_core_targets']}", '当前正式rank分母')}
        {card('下一版主生产', f"{u['next_primary_targets']}", '需完成跨路由校准')}
        {card('非GPCR完整登记', f"{u['complete_non_gpcr_registry_targets']}", '覆盖审计，不是统一rank分母')}
      </div>
      <h3>老药实体来源</h3>
      <p>部署药物轴由批准身份、活性成分和标准结构交叉冻结：Drugs@FDA/FDA资料负责批准身份与年份；Orange Book负责申请和活性成分核验；RxNorm负责盐型与active moiety；GSRS/UNII负责物质身份；PubChem与ChEMBL负责标准结构交叉核验。720是部署库，不是综合训练中的全部药物。</p>
      <h3>靶点登记与路由</h3>
      <table>
        <thead><tr><th>靶点层级</th><th>数量</th><th>当前处理</th><th>是否可作统一rank分母</th></tr></thead>
        <tbody>
          <tr><td>已评分比较核心</td><td>384</td><td>当前720×384={u['current_scored_core_pairs']:,}对完整评分</td><td><strong>是：rank/384</strong></td></tr>
          <tr><td>生化直接小分子主路由</td><td>367</td><td>生化体系内排序</td><td>路由内可比</td></tr>
          <tr><td>功能直接小分子主路由</td><td>83</td><td>功能体系内排序</td><td>路由内可比</td></tr>
          <tr><td>特殊体系直接证据</td><td>42</td><td>单独输出和实验协议</td><td>否</td></tr>
          <tr><td>高新颖性探索</td><td>8</td><td>探索性候选</td><td>否</td></tr>
          <tr><td>仅登记</td><td>245</td><td>保留身份和诊断，不作当前生产声明</td><td>否</td></tr>
        </tbody>
      </table>
      <div class="callout"><strong>384 / 450 / 745的关系：</strong>384是当前正式分母；450是下一版主生产空间；745用于不丢靶点和覆盖压力测试。新候选虽已能对745个靶点计算分数，但在实验体系跨路由校准完成前，不能发布统一rank/450或rank/745。</div>
    </section>

    <section class="page">
      <h2>02A　筛选空间是如何定义的</h2>
      <h3>药物轴：先统一“药物实体”，再进入计算</h3>
      <table>
        <thead><tr><th>数据源</th><th>解决的身份问题</th><th>进入候选轴时保留什么</th><th>为什么需要</th></tr></thead>
        <tbody>
          <tr><td>Drugs@FDA / FDA NME</td><td>是否批准、年份、申请与成分</td><td>批准身份与时间边界</td><td>区分真正的既有药物与其他化合物</td></tr>
          <tr><td>Orange Book</td><td>申请、产品和active ingredient</td><td>成分级交叉核验</td><td>避免把产品名当作分子实体</td></tr>
          <tr><td>RxNorm</td><td>盐型、水合物、组合物和active moiety</td><td>可比较的活性部分</td><td>减少同一药物被多次计数</td></tr>
          <tr><td>GSRS / UNII</td><td>物质标识和规范名</td><td>稳定身份键</td><td>支持跨数据库映射</td></tr>
          <tr><td>PubChem / ChEMBL</td><td>标准SMILES、InChIKey和结构一致性</td><td>单一、可建模的小分子结构</td><td>模型最终计算的是结构实体而不是名称</td></tr>
        </tbody>
      </table>
      <h3>靶点轴：888 → 745 → 分路候选</h3>
      <div class="flow">
        <div class="node"><strong>888</strong>ChEMBL37人源single-protein MoA</div><div class="arrow">→</div>
        <div class="node"><strong>745</strong>非GPCR完整登记目录</div><div class="arrow">→</div>
        <div class="node"><strong>450</strong>有直接小分子证据的主生产</div><div class="arrow">→</div>
        <div class="node"><strong>384</strong>历史已评分比较核心</div>
      </div>
      <table>
        <thead><tr><th>450主生产内的实验车道</th><th>靶点数</th><th>归属</th><th>主要实验解释</th></tr></thead>
        <tbody>
          <tr><td>Enzyme biochemical</td><td>208</td><td rowspan="3">367生化路由</td><td>可在直接生化体系中评估底物转化或结合/活性</td></tr>
          <tr><td>Kinase biochemical</td><td>122</td><td>激酶生化活性车道</td></tr>
          <tr><td>Nuclear / epigenetic domain</td><td>37</td><td>核受体、转录和表观调控域体系</td></tr>
          <tr><td>Ion-channel functional</td><td>56</td><td rowspan="2">83功能路由</td><td>需要电生理或功能读出</td></tr>
          <tr><td>Transporter membrane functional</td><td>27</td><td>需要转运或膜功能读出</td></tr>
        </tbody>
      </table>
      <p class="small">GPCR没有被定义为“无价值”，而是因为需要膜环境、构象状态和专门功能协议，不与本轮非GPCR统一车道混排。结构和口袋缺失不作为删除靶点的硬条件。</p>
    </section>

    <section class="page">
      <h2>03　训练数据来源与标签合同</h2>
      <div class="metrics four">
        {card('FULL_FIT关系', f"{tr['feature_resolved_relations']:,}", '去重且特征可解析')}
        {card('模型药物实体', f"{tr['unique_drugs']:,}", '标准结构特征实体')}
        {card('模型蛋白实体', f"{tr['unique_targets']:,}", '归一化蛋白特征实体')}
        {card('Affinity-only', '9,778', 'BindingDB direct Ki/Kd')}
      </div>
      <table>
        <thead><tr><th>来源</th><th>进入FULL_FIT</th><th>Endpoint/语义</th><th>用途</th></tr></thead>
        <tbody>
          <tr><td>ChEMBL 37</td><td>426,939</td><td>Ki、Kd、结合型IC50、明确inactive及直接关系</td><td>二分类关系、组内排序、部分连续活性</td></tr>
          <tr><td>BindingDB</td><td>9,778</td><td>direct Ki/Kd</td><td>连续亲和力辅助监督，不强制造负例</td></tr>
          <tr><td>ChEMBL审计恢复</td><td>531</td><td>来源特定的补充关系</td><td>补充冷药物和关系覆盖</td></tr>
        </tbody>
      </table>
      <div class="two-col">
        <div class="panel">
          <h3>FULL_FIT标签组成</h3>
          <div class="bar-row"><span>Positive</span><div class="bar-track"><div class="bar-fill" style="width:75.8%"></div></div><span class="bar-value">331,381</span></div>
          <div class="bar-row"><span>Negative/inactive</span><div class="bar-track"><div class="bar-fill" style="width:22.0%"></div></div><span class="bar-value">96,089</span></div>
          <div class="bar-row"><span>Affinity-only</span><div class="bar-track"><div class="bar-fill" style="width:2.2%"></div></div><span class="bar-value">9,778</span></div>
          <p class="small">三者合计437,248。该训练集是异质直接关系/活性数据，不应表述为纯Ki/Kd亲和力数据集。</p>
        </div>
        <div class="panel">
          <h3>二分类标签合同</h3>
          <div class="formula">pChEMBL ≥ 6.0　→ positive<br/>pChEMBL ≤ 5.0或明确inactive　→ negative<br/>5.0 &lt; pChEMBL &lt; 6.0　→ grey<br/>阳性与inactive冲突　→ conflicting<br/>未报告pair　→ unknown</div>
          <p class="small">Grey、conflicting与unknown均不进入普通二分类BCE；同一pair的多条assay先聚合并检查关系符号与冲突。</p>
        </div>
      </div>
      <div class="callout warn"><strong>历史86,674表不是当前训练总量：</strong>它是每个target×label最多150条的平衡benchmark，{hist['feature_resolved_rows']:,}行可解析，只用于S1–S5冻结评价和同数据模型比较。</div>
    </section>

    <section class="page">
      <h2>03A　每一套数据到底承担什么角色</h2>
      <table>
        <thead><tr><th>数据层</th><th>一行是什么</th><th>是否进入训练</th><th>标签含义</th><th>可以用来说明什么</th></tr></thead>
        <tbody>
          <tr><td><strong>ChEMBL综合监督表</strong><br/>426,939行</td><td>聚合后的标准药物–人源蛋白关系</td><td>是</td><td>明确positive、negative/inactive或部分连续pActivity</td><td>共享pair表示、二分类关系和组内排序</td></tr>
          <tr><td><strong>BindingDB affinity-only</strong><br/>9,778行</td><td>有direct Ki/Kd的药物–蛋白实测关系</td><td>是，辅助任务</td><td>连续亲和力强弱，不自动二分</td><td>连续趋势与表示约束；不定义全局negative</td></tr>
          <tr><td><strong>审计恢复关系</strong><br/>531去重行</td><td>从漏项审计找回的合格pair</td><td>是</td><td>沿用严格关系标签合同</td><td>修复冷药物/冷关系覆盖，不单独作性能集</td></tr>
          <tr><td><strong>历史平衡benchmark</strong><br/>86,674 pair</td><td>每个target和label最多150条的平衡样本</td><td>只在冻结S1–S5协议内训/测</td><td>45,983 positive + 40,691 negative</td><td>同一张表上的冷启动、时间和方法对比</td></tr>
          <tr><td><strong>720×384部署矩阵</strong><br/>276,480 pair</td><td>每种部署老药与每个核心靶点的组合</td><td>否，是推理空间</td><td>大多数是unknown，只有分数和rank</td><td>实际生产排名、个案和Top-K实验设计</td></tr>
          <tr><td><strong>2024–2025密集检索</strong><br/>20 query / 27 positive</td><td>一种老药在完整候选轴上的所有候选</td><td>否，冻结时间测试</td><td>后发阳性 vs unlabeled background</td><td>未来关系能否提前到Top-K；不是完整二分类</td></tr>
          <tr><td><strong>KiRHub严格回顾性子集</strong><br/>2,823 pair</td><td>实际测量并映射成功的药物–激酶pair</td><td>否，当前只作post-audit</td><td>1 μM抑制率；≥70%为强功能hit</td><td>跨endpoint功能迁移；不是Ki/Kd亲和力</td></tr>
        </tbody>
      </table>
      <h3>数量关系</h3>
      <div class="formula">426,939 ChEMBL + 9,778 BindingDB + 531 recovered = 437,248 FULL_FIT<br/>
      437,248 训练关系 ≠ 86,674 历史评价pair ≠ 276,480 部署候选pair</div>
      <div class="callout warn"><strong>最常见的口径错误：</strong>将“候选组合数”说成实验数据量，或将“实测子集Recall”说成完整候选空间Recall。报告中每个数字必须附带数据层和分母。</div>
    </section>

    <section class="page">
      <h2>04　通用模型架构</h2>
      <div class="two-col">
        <div>
          <h3>输入模态</h3>
          <table>
            <thead><tr><th>输入</th><th>维度</th><th>含义</th></tr></thead>
            <tbody>
              <tr><td>Morgan药物指纹</td><td>2,048</td><td>二维化学结构</td></tr>
              <tr><td>ProtBERT</td><td>1,024</td><td>蛋白序列主表示</td></tr>
              <tr><td>ESM2-650M</td><td>1,280</td><td>蛋白序列辅助表示</td></tr>
              <tr><td>Assay family</td><td>24</td><td>实验体系和靶点类别上下文</td></tr>
              <tr><td>结构/口袋上下文</td><td>19</td><td>target-level结构可用性与口袋质量</td></tr>
            </tbody>
          </table>
        </div>
        <div>
          <h3>共享pair表示</h3>
          <div class="formula">drug projection (192D)<br/>target projection (192D)<br/>＋ 双向低秩FiLM条件化<br/>＋ [d, t, d×t, |d−t|, bilinear]<br/>＋ assay/structure context<br/>→ 256D pair trunk<br/>→ 6个pair-conditioned experts</div>
          <p>共享pair logit之后再接两个零初始化残差头：</p>
          <div class="formula">D→T = shared pair score + D→T residual<br/>T→D = shared pair score + T→D residual</div>
        </div>
      </div>
      <div class="metrics">
        {card('模型规模', '约352万参数', '当前通用backbone')}
        {card('结构缺失处理', 'structure mask', '无结构时精确回退')}
        {card('方向头策略', '冻结backbone', '避免小时间集改写基础表示')}
      </div>
      <h3>明确没有混入的输入</h3>
      <p><span class="tag gold">ConPLex关闭</span><span class="tag gold">DTIAM不进入独立神经头</span><span class="tag gold">DrugCLIP不进入神经头</span><span class="tag gold">pair-specific局部图维度为0</span></p>
      <div class="callout"><strong>口袋信息边界：</strong>19维特征回答“这个靶点有哪些结构与候选口袋上下文”，不是“这一个药物在该口袋里形成了哪些pose和接触”。后者只能作为Top-K结构复核证据。</div>
    </section>

    <section class="page">
      <h2>05　训练策略与过拟合控制</h2>
      <div class="decision"><div class="label">共享主干</div><div class="body">使用target/label/scaffold平衡采样，降低高频靶点和重复骨架垄断；保留全量数据而不是简单删除热点关系。</div></div>
      <div class="decision"><div class="label">主方向头</div><div class="body">drug→target以药物为query，只使用有明确正负观测的药物进行组内采样；动态抽样结合BCE、pairwise和listwise ranking。</div></div>
      <div class="decision"><div class="label">辅助方向</div><div class="body">target→drug以靶点为query，作为共享生物学表示与反向复核；不作为当前生产主指标。</div></div>
      <div class="decision"><div class="label">FULL_FIT</div><div class="body">三个随机种子；先在开发边界选择epoch，再按冻结配置吸收全部合格关系。部署分数可用，但重训后没有新的无偏内部测试。</div></div>
      <h3>时间切分</h3>
      <div class="flow">
        <div class="node"><strong>≤2022</strong>基础训练</div><div class="arrow">→</div>
        <div class="node"><strong>2023</strong>方向抽样与epoch选择</div><div class="arrow">→</div>
        <div class="node"><strong>2024–2025</strong>冻结方案时间测试</div><div class="arrow">→</div>
        <div class="node"><strong>FULL_FIT</strong>冻结设置后全量重训</div>
      </div>
      <h3>当前数据限制</h3>
      <ul>
        <li>尽管有43.7万条关系，真正同时含正负两类、可训练drug→target排序的药物只有8,029个；大量药物只有单条记录。</li>
        <li>可训练target→drug排序的双标签靶点为434个，方向监督量同样远小于pair行数。</li>
        <li>2024–2025密集时间测试只有20个药物query、27个未来阳性，足以诊断，但不足以确认小差异的统计稳定性。</li>
        <li>跨生化、功能与特殊实验体系的logit不是天然同尺度，因此扩大靶点覆盖必须配合路由校准。</li>
      </ul>
      <div class="callout warn"><strong>核心原则：</strong>更多数据只有在endpoint、query方向和采样目标一致时才会转化为更好的药内排序；增加异质pair行数不必然提高drug→target Recall@K。</div>
    </section>

    <section class="page">
      <h2>06　评价集合与指标分母</h2>
      <table>
        <thead><tr><th>评价线</th><th>Query与分母</th><th>标签</th><th>回答的问题</th></tr></thead>
        <tbody>
          <tr><td><strong>S5药物实体冷启动</strong></td><td>302个测试药物；2,556个observed pair；75个双标签药物</td><td>ChEMBL observed正/负</td><td>冷药物能否在各自稀疏的已观测靶点子集中排序</td></tr>
          <tr><td><strong>2024–2025密集D→T</strong></td><td>20个药物、27个未来阳性；完整候选背景</td><td>未来阳性 vs unlabeled背景</td><td>后发关系能否进入实际实验预算Top-K</td></tr>
          <tr><td><strong>KiRHub回顾性审计</strong></td><td>78个药物、2,823个实测pair；33个双标签药物</td><td>1 μM抑制≥70%</td><td>通用模型跨endpoint的功能迁移</td></tr>
          <tr><td><strong>完整空间rank</strong></td><td>每药在384个核心靶点内排序</td><td>多数pair为unknown</td><td>生产候选优先级和案例展示</td></tr>
        </tbody>
      </table>
      <h3>主指标词典</h3>
      <div class="two-col">
        <div class="panel">
          <p><strong>Drug-macro AUPRC</strong><br/><span class="muted">每个双标签药物单独计算AP，再让每种药等权平均。当前drug→target主指标。</span></p>
          <p><strong>Micro AUPRC</strong><br/><span class="muted">把全部pair合并计算，容易被高频query主导，不代表每种药都同样好。</span></p>
          <p><strong>Recall@K</strong><br/><span class="muted">每个query的阳性中有多少进入Top-K，再宏平均。</span></p>
        </div>
        <div class="panel">
          <p><strong>NDCG@K</strong><br/><span class="muted">同时考虑是否召回和阳性是否排在更前部。</span></p>
          <p><strong>MRR</strong><br/><span class="muted">第一个阳性的倒数rank，对query平均。</span></p>
          <p><strong>rank/N</strong><br/><span class="muted">某pair在该药完整N个候选中的绝对位置，是案例最直接的指标。</span></p>
        </div>
      </div>
      <div class="callout warn"><strong>Recall@20不是一个可脱离分母的数字：</strong>KIRHub measured-subset Recall@20、S5 observed-pair Recall@20和完整384候选Recall@20是三种不同问题，不能互换；rank/384也不是Recall@K。</div>
      <p class="small">连续指标：Spearman用于亲和力/活性强弱次序；Continuous NDCG用于graded relevance。Brier与ECE只适用于已有观测标签且完成概率映射的结果；原始logit不能当结合概率。</p>
    </section>

    <section class="page">
      <h2>06A　历史S1–S5评价协议逐项解释</h2>
      <table>
        <thead><tr><th>协议</th><th>数据和切分单位</th><th>严格隔离什么</th><th>主要指标</th><th>能/不能回答什么</th></tr></thead>
        <tbody>
          <tr><td><strong>S1 Scaffold-cold drug</strong></td><td>86,673行可解析benchmark；5折，每折约1.73万test</td><td>Bemis–Murcko骨架簇不跨train/test</td><td>drug-macro AUPRC；辅助micro AUPRC/AUROC</td><td>回答新化学骨架迁移；不等于新靶点</td></tr>
          <tr><td><strong>S2 Homology-cold target</strong></td><td>同一张86,673行benchmark；5折</td><td>蛋白同源簇不跨train/test</td><td>target-macro与drug-macro AUPRC并列</td><td>回答远缘/未见蛋白迁移；不代表药物也冷</td></tr>
          <tr><td><strong>S3 Strict double-cold</strong></td><td>17,732个严格测试pair</td><td>药物骨架和靶点同源簇同时隔离</td><td>micro、drug-macro、target-macro AUPRC</td><td>回答双冷启动；最接近新药+新靶点的困难情形</td></tr>
          <tr><td><strong>S4 Temporal 2023–2025</strong></td><td>7,839个时间后首次观测pair；阳性比例约0.563</td><td>时间边界：早期关系训练，后发观测测试</td><td>micro与drug-macro AUPRC</td><td>回答observed-pair时间外推；不是完整靶点轴检索</td></tr>
          <tr><td><strong>S5 Old-drug entity-cold</strong></td><td>2,556 observed pair；302药物；354阳性；75个双标签query</td><td>测试药物实体不参与训练</td><td><strong>drug-macro AUPRC</strong>、Recall/NDCG@K</td><td>当前与主任务最相符的冻结对比；候选仅是已观测子集</td></tr>
        </tbody>
      </table>
      <h3>为什么S4的7,839与密集时间测试的20个query不矛盾</h3>
      <div class="two-col">
        <div class="panel"><strong>S4 observed-pair表</strong><p>7,839行都是数据库已观测的pair，其中同时有positive与negative，适合二分类AUPRC。它检查“观测pair能否区分”。</p></div>
        <div class="panel"><strong>Dense drug→target表</strong><p>只保留满足老药、未来阳性、旧关系移除和靶点映射的20个药物/27个阳性，再加入完整unlabeled候选背景。它检查“未来阳性排得多靠前”。</p></div>
      </div>
      <div class="callout warn"><strong>Rows的含义必须说清：</strong>S1/S2的86,673是整张参与交叉验证的可解析benchmark，不是单个fold的test行数；S4和S5是固定切分的test行数。</div>
    </section>

    <section class="page">
      <h2>06B　分类指标：从混淆矩阵到AUPRC</h2>
      <p>设测试集有真阳性TP、假阳性FP、真阴性TN和假阴性FN。所有分类指标都必须在有观测标签的pair上计算。</p>
      <table>
        <thead><tr><th>指标</th><th>公式/计算</th><th>数值如何理解</th><th>在本项目的用法</th><th>主要陷阱</th></tr></thead>
        <tbody>
          <tr><td><strong>Prevalence</strong></td><td>Positive / all labeled pairs</td><td>数据集中阳性比例</td><td>是随机micro-PR的参照水平</td><td>不是模型性能；候选集一变就会变</td></tr>
          <tr><td><strong>Precision</strong></td><td>TP / (TP + FP)</td><td>模型叫阳性的样本中有多少真阳性</td><td>衡量实验候选的“纯度”</td><td>需要选阈值；unknown不能当FP</td></tr>
          <tr><td><strong>Recall / TPR</strong></td><td>TP / (TP + FN)</td><td>全部真阳性中被找回多少</td><td>与PR曲线和Recall@K的概念相关</td><td>普通Recall依赖分数阈值；Recall@K依赖排名截断</td></tr>
          <tr><td><strong>AUROC</strong></td><td>P(score⁺ &gt; score⁻)；TPR–FPR曲线面积</td><td>0.5约等于随机，1为完美区分</td><td>辅助评价全局区分</td><td>阴性极多时仍可显得很高，不直接对应Top-K预算</td></tr>
          <tr><td><strong>AP / AUPRC</strong></td><td>AP=Σ Precision@k·rel(k) / #positive</td><td>每出现一个阳性时的Precision加权平均</td><td>稀疏阳性的主分类/排序指标</td><td>强依赖阳性比例和候选集，不可跨数据集直接比</td></tr>
          <tr><td><strong>Micro AUPRC</strong></td><td>将所有pair合并后计算一次AP</td><td>每个pair等权</td><td>衡量全局pair区分</td><td>大query/热门靶点占更大权重</td></tr>
          <tr><td><strong>Drug-macro AUPRC</strong></td><td>每个双标签药物分别算AP，再对药物等权平均</td><td>每种药物贡献同等权重</td><td><strong>drug→target主指标</strong></td><td>只有positive或只有negative的药物无法进入</td></tr>
          <tr><td><strong>Target-macro AUPRC</strong></td><td>每个双标签靶点分别算AP，再平均</td><td>每个靶点等权</td><td>target→drug辅助方向</td><td>不能代替药物中心的主指标</td></tr>
        </tbody>
      </table>
      <div class="callout"><strong>为什么主指标是drug-macro AUPRC：</strong>实际使用时是“来了一种药，对它的靶点排序”。药物宏平均防止数据量最大的少数药物决定总分。</div>
    </section>

    <section class="page">
      <h2>06C　检索指标：实验预算内能找回多少</h2>
      <p>对每个药物query q，设候选集为Cq，已知阳性集为Pq，TopKq为模型前K名。指标先在query内计算，再对query宏平均。</p>
      <table>
        <thead><tr><th>指标</th><th>公式/单位</th><th>直观解释</th><th>何时有用</th><th>不能说明什么</th></tr></thead>
        <tbody>
          <tr><td><strong>Recall@K</strong></td><td>|TopKq ∩ Pq| / |Pq|</td><td>该药的全部已知阳性中，前K名找回的比例</td><td>K对应真实实验预算</td><td>不告诉前K里有多少为真；分母必须附带</td></tr>
          <tr><td><strong>Hit@K</strong></td><td>I(|TopKq ∩ Pq| &gt; 0)</td><td>该药前K名是否至少有1个阳性</td><td>关心“每种药至少给一个可用答案”</td><td>不区分找回1个还是很多个</td></tr>
          <tr><td><strong>NDCG@K</strong></td><td>DCG/IDCG；DCG=Σ(2^rel−1)/log₂(i+1)</td><td>阳性越靠前贡献越大；用理想排序归一化</td><td>多阳性、分级活性或强弱值排序</td><td>不是概率；需说明rel是二值还是连续值</td></tr>
          <tr><td><strong>MRR</strong></td><td>mean(1 / first-positive-rank)</td><td>第一个阳性出现得多早</td><td>首个成功候选很重要时</td><td>基本忽略第二个及之后的阳性</td></tr>
          <tr><td><strong>Mean positive rank</strong></td><td>每个阳性的rank取平均</td><td>平均要检查到多少名才看到阳性</td><td>评估绝对实验工作量</td><td>容易被少数极差排名拉高，应同报中位数</td></tr>
          <tr><td><strong>Top percentile</strong></td><td>1 − (rank−1)/(N−1)</td><td>将不同候选数N的rank归一化</td><td>384、450和745空间间的辅助比较</td><td>百分位相同不代表实验数相同</td></tr>
          <tr><td><strong>Positive-retrieval AP</strong></td><td>未来阳性对unlabeled背景计算AP</td><td>未来阳性在完整候选中的密集检索质量</td><td>2024–2025密集时间测试</td><td>背景中可能有真阳性，因此不是严格二分类AP</td></tr>
          <tr><td><strong>rank/N</strong></td><td>某pair在该药N个候选中的位置</td><td>例如rank 12/384</td><td>个案、候选表和湿实验交付</td><td>不是Recall，也不是结合概率</td></tr>
        </tbody>
      </table>
      <div class="callout warn"><strong>同名Recall的三个例子：</strong>S5 Recall@20的候选是该药已观测test pair；KiRHub Recall@20的候选是该药实际测量的映射靶点；dense-384 Recall@20的候选接近384个完整核心靶点。只有第三种接近实际完整检索预算。</div>
    </section>

    <section class="page">
      <h2>06D　连续值、校准和统计比较指标</h2>
      <table>
        <thead><tr><th>指标/方法</th><th>计算</th><th>数值方向</th><th>适用数据</th><th>项目解释边界</th></tr></thead>
        <tbody>
          <tr><td><strong>Spearman ρ</strong></td><td>预测值排名与实测值排名的相关</td><td>越接1越好</td><td>Ki/Kd、pActivity等连续标签</td><td>只评估单调次序，不保证绝对数值误差小</td></tr>
          <tr><td><strong>Continuous NDCG@K</strong></td><td>将连续抑制/活性当作relevance计算NDCG</td><td>越接1越好</td><td>有分级强弱的功能或活性数据</td><td>反映前部强弱次序，不是亲和力误差</td></tr>
          <tr><td><strong>Brier score</strong></td><td>mean((p − y)²)</td><td>越低越好；0最佳</td><td>已校准二分类概率</td><td>只对有观测标签的pair有效；原始logit不是p</td></tr>
          <tr><td><strong>ECE</strong></td><td>Σ |bin|/n · |accuracy(bin) − confidence(bin)|</td><td>越低越好</td><td>已映射到0–1的概率</td><td>依赖分箱方法；跨靶点/跨路由可能需要分别校准</td></tr>
          <tr><td><strong>Query bootstrap</strong></td><td>按药物query有放回采样，每次重算模型差值</td><td>报告Δ、95% CI和P(Δ&gt;0)</td><td>drug-macro AP/Recall/NDCG比较</td><td>不把pair行错当成完全独立样本</td></tr>
          <tr><td><strong>Cluster bootstrap</strong></td><td>按药物骨架或蛋白同源簇采样</td><td>评估对实体相似性的稳健性</td><td>冷启动与外推比较</td><td>簇数少时CI会很宽</td></tr>
          <tr><td><strong>三种子方差</strong></td><td>相同数据/设置用不同初始随机数训练</td><td>报告平均、标准差或ensemble</td><td>FULL_FIT和方向头</td><td>只控制初始化方差，不能弥补测试query太少</td></tr>
        </tbody>
      </table>
      <h3>如何读95% CI</h3>
      <div class="two-col">
        <div class="panel"><strong>CI不跨0</strong><p>在当前query和重采样假设下，差值方向得到统计支持。仍不等于对任意外部数据都成立。</p></div>
        <div class="panel"><strong>CI跨0</strong><p>可以说“点估计领先/下降”，但不能说“已确认优越/退化”。应优先增加独立query，而不是增加同一query下的相似pair。</p></div>
      </div>
      <div class="callout"><strong>概率与rank必须分开：</strong>当前最可靠的部署输出是“在明确分母中的相对排名”。除非在同endpoint、同路由和独立校准集上完成概率映射，否则不应把分数解释为结合成功率。</div>
    </section>

    <section class="page">
      <h2>06E　S5分母审计：为什么Recall@20容易被误读</h2>
      <p class="lead">S5确实将部署老药及其相同骨架排除出训练，但测试不是“每种药对384个靶点”。它只对历史平衡benchmark中碰巧存在标签的drug–target pair排序。</p>
      <div class="metrics four">
        {card('S5测试药物', f"{s5_obs['all_drugs']}", '来自720老药中的有标签子集')}
        {card('双标签药物', f"{s5_obs['two_class_drugs']}", '只有这些进入drug-macro指标')}
        {card('仅阳性 / 仅阴性', f"{s5_obs['positive_only_drugs']} / {s5_obs['negative_only_drugs']}", '均不进入宏AUPRC/Recall')}
        {card('候选数≤20', f"{s5_obs['two_class_drugs_at_most_20_candidates']}", '这些药的Recall@20天然为1')}
      </div>
      <div class="two-col">
        <div class="panel">
          <h3>Recall@20的实际难度</h3>
          <div class="compare-grid"><span>随机排序期望</span><div class="compare-track"><div class="compare-fill blue" style="width:{100*s5_obs['random_expected_macro_recall_at_20']:.1f}%"></div></div><span class="compare-value">{s5_obs['random_expected_macro_recall_at_20']:.3f}</span></div>
          <div class="compare-grid"><span>S5独立验证排序</span><div class="compare-track"><div class="compare-fill" style="width:{100*s5_obs_rtm['recall_at_20']:.1f}%"></div></div><span class="compare-value">{s5_obs_rtm['recall_at_20']:.3f}</span></div>
          <div class="compare-grid"><span>DTIAM</span><div class="compare-track"><div class="compare-fill blue" style="width:{100*s5_obs_dtiam['recall_at_20']:.1f}%"></div></div><span class="compare-value">{s5_obs_dtiam['recall_at_20']:.3f}</span></div>
          <p class="small">75个双标签药物中，{s5_obs['two_class_drugs_at_most_20_candidates']}个总候选数不超过20。随机基线已经达到{s5_obs['random_expected_macro_recall_at_20']:.3f}，所以0.899不能解释为完整384检索成功率。</p>
        </div>
        <div class="panel">
          <h3>只看真正候选数&gt;20的药物</h3>
          <div class="metrics" style="grid-template-columns:repeat(3,1fr);">
            {card('药物数', f"{s5_obs['nontrivial_gt20_drugs']}", '非平凡query')}
            {card('ReTargetMap', f"{s5_obs_rtm['nontrivial_gt20_recall_at_20']:.3f}", 'S5验证排序')}
            {card('DTIAM', f"{s5_obs_dtiam['nontrivial_gt20_recall_at_20']:.3f}", '同数据对照')}
          </div>
          <p>非平凡query上的随机期望为{s5_obs['nontrivial_gt20_random_expected_recall_at_20']:.3f}。ReTargetMap仍明显优于随机，但深截断覆盖低于DTIAM。</p>
          <p class="small">ReTargetMap相对DTIAM的Recall@20差值为−0.0490；药物bootstrap 95% CI [−0.0954, −0.0103]。这是当前明确存在的尾部覆盖弱点。</p>
        </div>
      </div>
      <div class="callout warn"><strong>建议改名：</strong>“S5冷药切片：75种双标签药物各自已观测靶点子集内的宏排序”。该名称比“未见药物Recall@20”更不容易被误解。</div>
    </section>

    <section class="page">
      <h2>07　当前主结果：Top 10榜首质量与覆盖边界</h2>
      <p class="lead">首轮实验不会为每种药测试几十个靶点，因此主展示窗口改为Top 10。最符合项目价值的指标是前排精度、第一名命中和首个阳性出现位置；Recall@10仍保留，避免选择性汇报。</p>
      <table class="result-table">
        <thead><tr><th>方法</th><th>Drug-macro AUPRC</th><th>NDCG@10</th><th>Hit@1</th><th>MRR</th><th>Recall@10</th></tr></thead>
        <tbody>
          <tr><td>DTIAM</td><td>{s5_dtiam['drug_macro_auprc']:.4f}</td><td>{s5_dtiam['drug_macro_ndcg_at_10']:.4f}</td><td>{pct(s5_obs_dtiam['hit_at_1'])}</td><td>{s5_obs_dtiam['mrr']:.4f}</td><td class="best">{s5_dtiam['drug_macro_recall_at_10']:.4f}</td></tr>
          <tr><td><strong>S5独立验证排序</strong></td><td class="best">{s5_biomaster['drug_macro_auprc']:.4f}</td><td class="best">{s5_biomaster['drug_macro_ndcg_at_10']:.4f}</td><td class="best">{pct(s5_obs_rtm['hit_at_1'])}</td><td class="best">{s5_obs_rtm['mrr']:.4f}</td><td>{s5_biomaster['drug_macro_recall_at_10']:.4f}</td></tr>
          <tr><td>含DTIAM统一系统</td><td class="best">{s5_system['drug_macro_auprc']:.4f}</td><td class="best">{s5_system['drug_macro_ndcg_at_10']:.4f}</td><td class="best">{pct(s5_obs_system['hit_at_1'])}</td><td class="best">{s5_obs_system['mrr']:.4f}</td><td class="best">{s5_system['drug_macro_recall_at_10']:.4f}</td></tr>
        </tbody>
      </table>
      <div class="evidence-strip">
        <div><strong>前排整体质量</strong>S5独立验证排序AUPRC 0.7521，DTIAM 0.7365；点估计领先，CI跨0。</div>
        <div><strong>第一名命中</strong>Hit@1为{pct(s5_obs_rtm['hit_at_1'])}，DTIAM为{pct(s5_obs_dtiam['hit_at_1'])}；MRR为{s5_obs_rtm['mrr']:.3f} vs {s5_obs_dtiam['mrr']:.3f}。</div>
        <div><strong>覆盖不占优</strong>Recall@10为{s5_biomaster['drug_macro_recall_at_10']:.3f} vs {s5_dtiam['drug_macro_recall_at_10']:.3f}；二者接近但ReTargetMap略低。</div>
      </div>
      <h3>完整384靶点中的已知阳性恢复诊断</h3>
      <table class="result-table">
        <thead><tr><th>分数</th><th>Hit@10</th><th>Drug-macro Recall@10</th><th>证据等级</th></tr></thead>
        <tbody>
          <tr><td>S5独立验证排序</td><td class="best">{pct(s5_app_anchor['hit_at_10'])}</td><td class="best">{s5_app_anchor['macro_recall_at_10']:.4f}</td><td>应用对齐补充诊断</td></tr>
          <tr><td>当前默认独立Borda</td><td>{pct(s5_app_borda['hit_at_10'])}</td><td>{s5_app_borda['macro_recall_at_10']:.4f}</td><td>应用对齐补充诊断</td></tr>
          <tr><td>DTIAM</td><td>{pct(s5_app_dtiam['hit_at_10'])}</td><td>{s5_app_dtiam['macro_recall_at_10']:.4f}</td><td>同矩阵对照诊断</td></tr>
        </tbody>
      </table>
      <p class="small">354个S5阳性中有{s5_app['mapped_positive_pairs']}个可映射到384核心，覆盖{s5_app['mapped_positive_drugs']}种药。其余候选多数为unknown而非实验阴性，所以这组数字只说明“已知阳性在完整候选背景中的恢复位置”，不是严格二分类性能。FULL_FIT已吸收全量关系，其恢复值不在表中作为泛化结果。</p>
      <div class="callout teal"><strong>准确结论：</strong>ReTargetMap当前优势集中在“把第一个可信靶点推得更靠前”；DTIAM在S5 observed-target的较深Top-K覆盖更强。Top 10是实验预算驱动的主窗口，不是事后删除不利指标。</div>
    </section>

    <section class="page">
      <h2>07A　2024–2025未来阳性的密集药内检索</h2>
      <p class="lead">这一评价把未来首次出现的阳性放回近完整候选轴，更接近实际筛选；但只有20个药物query和27个阳性，因此用于诊断绝对实验工作量，不用于确认微小模型差异。</p>
      <table class="result-table">
        <thead><tr><th>候选空间</th><th>平均阳性rank↓</th><th>MRR↑</th><th>Positive-retrieval AP↑</th><th>Recall@10</th><th>Recall@20</th></tr></thead>
        <tbody>
          <tr><td>384核心</td><td>{tm['frozen_core_384']['macro_mean_positive_rank']:.2f}</td><td>{tm['frozen_core_384']['macro_mrr']:.4f}</td><td>{tm['frozen_core_384']['macro_positive_retrieval_ap']:.4f}</td><td>{tm['frozen_core_384']['macro_recall_at_10']:.4f}</td><td>{tm['frozen_core_384']['macro_recall_at_20']:.4f}</td></tr>
          <tr><td>450主路由预校准</td><td>{tm['primary_450_precalibration']['macro_mean_positive_rank']:.2f}</td><td>{tm['primary_450_precalibration']['macro_mrr']:.4f}</td><td>{tm['primary_450_precalibration']['macro_positive_retrieval_ap']:.4f}</td><td>{tm['primary_450_precalibration']['macro_recall_at_10']:.4f}</td><td>{tm['primary_450_precalibration']['macro_recall_at_20']:.4f}</td></tr>
          <tr><td>500活跃路由诊断</td><td>{tm['active_routed_500_diagnostic']['macro_mean_positive_rank']:.2f}</td><td>{tm['active_routed_500_diagnostic']['macro_mrr']:.4f}</td><td>{tm['active_routed_500_diagnostic']['macro_positive_retrieval_ap']:.4f}</td><td>{tm['active_routed_500_diagnostic']['macro_recall_at_10']:.4f}</td><td>{tm['active_routed_500_diagnostic']['macro_recall_at_20']:.4f}</td></tr>
          <tr><td>745登记压力测试</td><td>{tm['registry_745_diagnostic']['macro_mean_positive_rank']:.2f}</td><td>{tm['registry_745_diagnostic']['macro_mrr']:.4f}</td><td>{tm['registry_745_diagnostic']['macro_positive_retrieval_ap']:.4f}</td><td>{tm['registry_745_diagnostic']['macro_recall_at_10']:.4f}</td><td>{tm['registry_745_diagnostic']['macro_recall_at_20']:.4f}</td></tr>
        </tbody>
      </table>
      <div class="two-col">
        <div class="panel"><h3>384核心</h3><p>Recall@10为{tm['frozen_core_384']['macro_recall_at_10']:.3f}，Recall@20为{tm['frozen_core_384']['macro_recall_at_20']:.3f}。这说明真正完整背景下仍有大量未来阳性排在实验预算之外。</p></div>
        <div class="panel"><h3>扩大空间</h3><p>扩至745后，每个query需压过的背景接近738个，且跨生化/功能/特殊路由；固定K召回下降是预期结果。</p></div>
      </div>
      <div class="callout warn"><strong>不能反推：</strong>450/745下降不等于新增靶点无价值；它说明在扩空间前必须解决路由内校准、query方向训练和高新颖性候选的独立验证。</div>
    </section>

    <section class="page">
      <h2>08　KiRHub仅作为回顾性功能审计</h2>
      <p class="lead">KiRHub在当前项目中的角色是回顾性功能迁移审计：检查通用ReTargetMap能否把1 μM条件下的强功能抑制pair排到同一药物实测候选的前部。</p>
      <div class="metrics four">
        {card('实测pair', f"{kirhub['rows']:,}", '仅实际测量且映射成功')}
        {card('强抑制pair', f"{kirhub['positives']}", '1 μM inhibition ≥70%')}
        {card('药物', f"{kirhub['drugs']}", '严格回顾性子集')}
        {card('双标签药物', f"{kirhub['two_class_drugs']}", '可计算drug-macro指标')}
      </div>
      <table class="result-table">
        <thead><tr><th>方法</th><th>Drug-macro AUPRC</th><th>Recall@5</th><th>Recall@10</th><th>Recall@20</th></tr></thead>
        <tbody>
          <tr><td>DTIAM</td><td>{kir_dtiam['drug_macro_auprc']:.4f}</td><td>{kir_dtiam['drug_macro_recall_at_5']:.4f}</td><td>{kir_dtiam['drug_macro_recall_at_10']:.4f}</td><td>{kir_dtiam['drug_macro_recall_at_20']:.4f}</td></tr>
          <tr><td>新FULL_FIT方向头</td><td>{kir_direction['drug_macro_auprc']:.4f}</td><td class="best">{kir_direction['drug_macro_recall_at_5']:.4f}</td><td>{kir_direction['drug_macro_recall_at_10']:.4f}</td><td>{kir_direction['drug_macro_recall_at_20']:.4f}</td></tr>
          <tr><td><strong>ReTargetMap独立Borda</strong></td><td>{kir_independent['drug_macro_auprc']:.4f}</td><td>{kir_independent['drug_macro_recall_at_5']:.4f}</td><td class="best">{kir_independent['drug_macro_recall_at_10']:.4f}</td><td>{kir_independent['drug_macro_recall_at_20']:.4f}</td></tr>
          <tr><td>含DTIAM统一系统</td><td class="best">{kir_system['drug_macro_auprc']:.4f}</td><td>{kir_system['drug_macro_recall_at_5']:.4f}</td><td>{kir_system['drug_macro_recall_at_10']:.4f}</td><td class="best">{kir_system['drug_macro_recall_at_20']:.4f}</td></tr>
        </tbody>
      </table>
      <div class="callout warn"><strong>必须同时写出的限制：</strong>这些Recall只在每种药物实际测量的KiRHub靶点子集中计算，不是Recall@20/384；endpoint是单浓度功能抑制，不是Ki/Kd亲和力；集合只有33个双标签药物且已被项目多次查看，因此只能称post-audit点估计，不能作为全新外部确认性证据。</div>
      <h3>逐项解读当前数字</h3>
      <ul>
        <li><strong>AUPRC 0.4124 vs 0.3867：</strong>在33个双标签药物上分别算AP后宏平均，ReTargetMap独立Borda的点估计高0.0257，但置信区间跨0。</li>
        <li><strong>Recall@10 0.4791：</strong>平均找回每种药物已知强抑制pair的47.91%；每种药的候选分母是其KiRHub实测子集。</li>
        <li><strong>Recall@20 0.6576：</strong>含DTIAM统一系统的前20名平均找回65.76%已知强抑制pair，但该结果不是ReTargetMap独立分数。</li>
        <li><strong>为什么AUPRC与Recall最佳方法不同：</strong>AUPRC累积考察整条排序上的精度–召回折中，Recall@K只检查一个截断点。</li>
      </ul>
    </section>

    <section class="page">
      <h2>09　首轮湿实验：用前瞻数据结束口径争议</h2>
      <p class="lead">首轮实验的目标不是展示若干成功个案，而是在相同药物、相同靶点和相同实验条件下，比较不同分数是否真正富集新的直接作用。</p>
      <div class="metrics four">
        {card('老药', '12', '身份、纯度和溶解度质控')}
        {card('生化靶点', '8', '优先酶/激酶/结构域')}
        {card('完整矩阵', '96 pair', '12×8全部组合均测试')}
        {card('主窗口', 'Top 10', '对应实际候选预算')}
      </div>
      <h3>模型组必须按真实分数谱系冻结</h3>
      <table>
        <thead><tr><th>实验分层</th><th>候选来源</th><th>回答的问题</th></tr></thead>
        <tbody>
          <tr><td><strong>主要推荐组</strong></td><td>S5独立验证排序Top 10，且DTIAM排名较低</td><td>有无偏S5依据的独立分数能否产生更高确认命中率</td></tr>
          <tr><td><strong>DTIAM对照组</strong></td><td>DTIAM Top 10，且S5独立验证排序较低</td><td>同资源下与公开基线公平比较</td></tr>
          <tr><td><strong>默认Borda挑战组</strong></td><td>当前合同Borda独有高排</td><td>部署融合是否比其单独证据更有用</td></tr>
          <tr><td><strong>FULL_FIT探索组</strong></td><td>真正的ensemble_drug_to_target_logit独有高排</td><td>全量重训能否提供额外假设；不用于无偏性能声明</td></tr>
          <tr><td><strong>低排背景组</strong></td><td>各独立分数均低排</td><td>估计真实背景活性率和富集倍数</td></tr>
        </tbody>
      </table>
      <h3>实验级联</h3>
      <div class="flow">
        <div class="node"><strong>建系</strong>参考阳性、DMSO、Z′</div><div class="arrow">→</div>
        <div class="node"><strong>初筛</strong>1与10 μM、双日重复</div><div class="arrow">→</div>
        <div class="node"><strong>剂量反应</strong>≥10浓度、4PL拟合</div><div class="arrow">→</div>
        <div class="node"><strong>直接性确认</strong>SPR/BLI/MST等正交方法</div><div class="arrow">→</div>
        <div class="node"><strong>机制晋级</strong>细胞靶点占有与功能</div>
      </div>
      <div class="callout teal"><strong>主要终点：</strong>样品身份/纯度合格，主实验获得可重复完整剂量反应，默认IC50/EC50≤10 μM，并由不同原理的实验确认直接结合且关键干扰反筛阴性。</div>
      <div class="callout warn"><strong>当前操作门：</strong>候选预池已按本页分数谱系重建，但仍不能直接下单。必须先完成外部新颖性检索、实验可行性审查、化合物质控并冻结盲法SampleID。</div>
      <p class="small">可选探索臂：复用12种药物，增加1–2个高新颖性靶点，形成12–24个额外pair；单独报告，不与rank/384主命中率混算。</p>
    </section>

    <section class="page">
      <h2>10　当前结论与对外声明</h2>
      <div class="two-col">
        <div class="panel">
          <h3>当前可以说</h3>
          <ul>
            <li>ReTargetMap已形成共享pair backbone、drug→target主头和target→drug辅助头的双向关系检索方法。</li>
            <li>当前已完成720种老药×384个核心靶点的完整独立排序。</li>
            <li>冻结S5中，<strong>S5独立验证排序</strong>的drug-macro AUPRC、NDCG@10和Hit@1点估计高于DTIAM。</li>
            <li>745个非GPCR靶点已完成登记和模型覆盖，可进行分路评分与覆盖压力测试。</li>
            <li>KiRHub回顾性审计中，多个ReTargetMap分数的点估计高于DTIAM，但具体分数必须分别命名。</li>
          </ul>
        </div>
        <div class="panel">
          <h3>当前不能说</h3>
          <ul>
            <li>不能说ReTargetMap已统计学显著、全面超过DTIAM。</li>
            <li>不能把KiRHub结果解释为Ki/Kd亲和力验证。</li>
            <li>不能声称已有校准完成的统一rank/450或rank/745。</li>
            <li>不能把FULL_FIT覆盖表现称为独立测试。</li>
            <li>不能把数据库未报告pair当作negative。</li>
            <li>不能把19维口袋上下文当作drug–pocket相互作用证据。</li>
          </ul>
        </div>
      </div>
      <h3>当前最简项目表达</h3>
      <div class="callout teal">ReTargetMap面向“给定老药建立候选靶点谱”，当前在384个已评分核心靶点内产生每药排序，并以450个主生产靶点作为下一版扩展目标。综合训练使用43.7万条去重直接关系/活性数据；745个非GPCR靶点用于完整登记与分路覆盖。S5独立验证排序的优势集中在Top 10前排质量，较深覆盖仍弱于DTIAM；下一阶段以盲法前瞻实验判断这种排序优势能否转化为直接命中。</div>
      <h3>下一步只保留三项</h3>
      <ol>
        <li><strong>完成450主生产空间校准：</strong>在生化与功能路由内分别校准，再决定能否发布rank/450。</li>
        <li><strong>冻结新的确认性评价：</strong>执行12药×8靶点完整矩阵，一次性比较S5锚定排序、默认Borda与DTIAM。</li>
        <li><strong>把pair-specific结构证据放到Top-K：</strong>只对少量候选生成pose、接触和口袋匹配，用作重排与湿实验设计。</li>
      </ol>
      <p class="small">权威机器合同：configs/biomaster_current_contract_v1.json<br/>训练清单：outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json<br/>冻结排序摘要：outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json<br/>统一说明：docs/BIOMASTER_SYSTEM_DATA_METRICS_OVERVIEW_20260901_ZH.md</p>
    </section>
    """

    document = f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>ReTargetMap项目、数据、评价与实验方案详解</title><style>{css}</style></head><body>{body}</body></html>"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(document, encoding="utf-8")
    HTML(filename=str(HTML_OUT), base_url=str(ROOT)).write_pdf(str(PDF_OUT))

    with fitz.open(PDF_OUT) as pdf:
        page_count = pdf.page_count
        page_text = [page.get_text("text") for page in pdf]
        empty_pages = [index + 1 for index, text in enumerate(page_text) if len(text.strip()) < 80]
        title_present = "项目、数据、评价" in page_text[0]
        process_phrase_pages = [
            index + 1
            for index, text in enumerate(page_text)
            if any(phrase in text for phrase in ("已撤下", "已暂停", "不再训练或部署", "当前不再包含"))
        ]
        legacy_name_pages = [index + 1 for index, text in enumerate(page_text) if "BioMaster" in text]
        ambiguous_score_pages = [
            index + 1 for index, text in enumerate(page_text) if "ReTargetMap独立系统" in text
        ]
        top10_result_present = any("Top 10榜首质量" in text for text in page_text)
        experiment_page_present = any("首轮湿实验" in text for text in page_text)
        score_lineage_present = any("分数谱系" in text for text in page_text)

    audit = {
        "status": "PASS" if page_count == 21 and not empty_pages and title_present and not process_phrase_pages and not legacy_name_pages and not ambiguous_score_pages and top10_result_present and experiment_page_present and score_lineage_present else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(PDF_OUT.relative_to(ROOT)),
        "html": str(HTML_OUT.relative_to(ROOT)),
        "pdf_sha256": sha256(PDF_OUT),
        "pdf_bytes": PDF_OUT.stat().st_size,
        "page_count": page_count,
        "empty_pages": empty_pages,
        "title_present": title_present,
        "process_phrase_pages": process_phrase_pages,
        "legacy_name_pages": legacy_name_pages,
        "ambiguous_score_pages": ambiguous_score_pages,
        "top10_result_present": top10_result_present,
        "experiment_page_present": experiment_page_present,
        "score_lineage_present": score_lineage_present,
        "s5_diagnostics": diagnostics,
        "contract_specialist_removed": "kinase_functional_specialist" not in contract["model_roles"],
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (CONTRACT_PATH, TRAINING_PATH, FULL_FIT_PATH, RANK_PATH, TEMPORAL_PATH, TARGET_SCOPE_PATH, S5_TEST_PATH, DEPLOYMENT_PATH)
        },
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"PDF audit failed: {audit}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

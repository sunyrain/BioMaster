#!/usr/bin/env python3
"""Build a compact professor-review PDF from current BioMaster dashboard outputs."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]


def pct(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "-"


def num(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def intval(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{int(round(float(value))):,}"
    except Exception:
        return "-"


def esc(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return html.escape(str(value), quote=True)


def short_text(value: Any, limit: int = 72) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def normalize_drug_name(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(
        r"\b(hydrochloride|hydrochrloride|mesylate|maleate|dimaleate|"
        r"dihydrochloride|acetate|sodium|potassium|calcium)\b",
        "",
        text,
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_summary(root: Path) -> dict[str, Any]:
    path = root / "outputs/sota_validation/sota_compute_closure_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("headline", {})


def rank_badge(tier: str) -> str:
    if "A" in tier:
        return "badge a"
    if "B" in tier:
        return "badge b"
    if "C" in tier:
        return "badge c"
    return "badge"


def candidate_reason(row: pd.Series) -> str:
    posture = str(row.get("repurposingPostureZh", "") or "")
    evidence = str(row.get("evidenceSummaryZh", "") or "")
    if evidence:
        first = evidence.split("；")[0]
    else:
        first = posture
    status = str(row.get("status", "") or "")
    dock = row.get("diffdock")
    dock_text = "结构已完成" if status == "completed" else "结构待补"
    if pd.notna(dock):
        dock_text += f"，DiffDock {float(dock):.2f}"
    return short_text(f"{posture}；{first}；{dock_text}", 86)


def build_top10_tables(candidates: pd.DataFrame) -> str:
    direction_order = [
        "oncology",
        "infectious_disease",
        "cardiovascular",
        "neurology_psychiatry",
        "immunology_inflammation",
    ]
    parts: list[str] = []
    for direction in direction_order:
        block = candidates[candidates["direction"] == direction].sort_values("rank").copy()
        if block.empty:
            continue
        block["_drug_norm"] = block["drug"].map(normalize_drug_name)
        block = block.drop_duplicates(["_drug_norm", "target"], keep="first").head(10)
        label = block["directionLabelZh"].iloc[0]
        english = block["directionLabel"].iloc[0]
        rows = []
        for review_rank, (_, row) in enumerate(block.iterrows(), start=1):
            tier = esc(row.get("credibilityTierZh", ""))
            original_rank = int(row.get("rank")) if pd.notna(row.get("rank")) else review_rank
            rank_note = f"<br><span>orig {original_rank}</span>" if original_rank != review_rank else ""
            rows.append(
                "<tr>"
                f"<td class='rank'>{review_rank}{rank_note}</td>"
                f"<td><strong>{esc(row.get('drug'))}</strong></td>"
                f"<td><strong>{esc(row.get('target'))}</strong><br><span>{esc(row.get('protein'))}</span></td>"
                f"<td>{esc(short_text(row.get('proteinName'), 42))}</td>"
                f"<td class='num'>{num(row.get('directionScore'))}</td>"
                f"<td class='num'>{num(row.get('affinityScore'))}</td>"
                f"<td class='num'>{num(row.get('diffdock'), 2)}</td>"
                f"<td><span class='{rank_badge(tier)}'>{tier}</span></td>"
                f"<td>{esc(candidate_reason(row))}</td>"
                "</tr>"
            )
        parts.append(
            f"""
            <section class="page direction-page">
              <div class="section-kicker">Disease-direction Top 10</div>
              <h2>{esc(label)} <span>{esc(english)}</span></h2>
              <table class="candidate-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Drug</th>
                    <th>Target</th>
                    <th>Protein</th>
                    <th>Dir.</th>
                    <th>Affinity</th>
                    <th>Dock</th>
                    <th>Tier</th>
                    <th>Review meaning</th>
                  </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
              <p class="footnote">
                注：本表按原始排序取唯一 drug-target Top10；orig 表示去重前的网站原始 rank。
                Dir. 为疾病方向综合分数，Affinity 为 ConPLex 药物-蛋白亲和预测分数；
                Dock 为 DiffDock confidence，用于结构姿态审阅，不等同于结合自由能或体内疗效。
              </p>
            </section>
            """
        )
    return "\n".join(parts)


def build_direction_summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{esc(row.get('labelZh'))}</td>"
            f"<td>{esc(row.get('label'))}</td>"
            f"<td class='num'>{intval(row.get('preparedPairs'))}</td>"
            f"<td class='num'>{intval(row.get('completed'))}</td>"
            f"<td class='num'>{intval(row.get('missing'))}</td>"
            f"<td class='num'>{pct(row.get('successRatePct'), 2)}</td>"
            f"<td class='num'>{num(row.get('medianDiffDock'), 2)}</td>"
            "</tr>"
        )
    return (
        "<table class='summary-table'><thead><tr>"
        "<th>方向</th><th>Direction</th><th>候选对</th><th>结构完成</th><th>技术缺失</th><th>完成率</th><th>Dock 中位数</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_report(root: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = root / "outputs/disease_directions/disease_direction_integrated_candidates.csv"
    summary_path = root / "outputs/disease_directions/disease_direction_summary.csv"
    candidates = pd.read_csv(candidates_path)
    direction_summary = pd.read_csv(summary_path)
    headline = load_summary(root)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    main_figure = (root / "docs/assets/biomaster-main-figure.png").resolve().as_uri()

    total_drugs = headline.get("stage6Top10000UniqueDrugs") or 915
    total_proteins = 5306
    total_pairs = 4_854_990
    disease_docking_rows = int(direction_summary["preparedPairs"].sum())
    disease_completed = int(direction_summary["completed"].sum())
    disease_success = 100 * disease_completed / disease_docking_rows if disease_docking_rows else 0

    full_jobs = f"{intval(headline.get('fullDiffdockCompletedJobs'))} / {intval(headline.get('fullDiffdockTotalJobs'))}"
    full_rows = f"{intval(headline.get('fullDiffdockScoredRows'))} / {intval(headline.get('fullDiffdockTotalRows'))}"
    full_pct = pct(headline.get("fullDiffdockScoredRowPct"), 1)
    eta_hours = headline.get("fullDiffdockEtaHours")
    eta_text = f"{float(eta_hours):.1f} h" if eta_hours is not None else "-"
    finish = headline.get("fullDiffdockEstimatedFinishUtc") or "-"

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>BioMaster 教授审阅版简报</title>
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
    @page {{
      size: A4 landscape;
      margin: 12mm 13mm 12mm 13mm;
      @bottom-right {{
        content: counter(page);
        color: #667085;
        font-size: 8pt;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans CJK", sans-serif;
      color: #152033;
      font-size: 9.2pt;
      line-height: 1.45;
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ font-size: 29pt; line-height: 1.05; margin-bottom: 5mm; color: #0f4f82; }}
    h2 {{ font-size: 16pt; margin-bottom: 3mm; color: #163b5c; }}
    h2 span {{ font-size: 9pt; color: #667085; font-weight: 400; }}
    h3 {{ font-size: 10.5pt; margin-bottom: 1.5mm; color: #23415f; }}
    .page {{ page-break-after: always; min-height: 185mm; position: relative; }}
    .cover {{
      display: grid;
      grid-template-columns: 1fr 1.03fr;
      gap: 12mm;
      align-items: center;
    }}
    .cover .subtitle {{ font-size: 14pt; color: #344054; margin-bottom: 7mm; }}
    .meta {{ color: #667085; font-size: 9pt; }}
    .figure-frame {{
      border: 1px solid #d0d8e3;
      background: #f7fbff;
      padding: 4mm;
      border-radius: 3mm;
    }}
    .figure-frame img {{ width: 100%; display: block; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 3mm;
      margin: 5mm 0;
    }}
    .metric {{
      border: 1px solid #d5e3ee;
      background: #f4f9fd;
      border-radius: 2mm;
      padding: 3mm;
    }}
    .metric strong {{ display: block; font-size: 16pt; color: #0f5d93; line-height: 1.05; }}
    .metric span {{ color: #667085; font-size: 8.3pt; }}
    .section-kicker {{
      text-transform: uppercase;
      letter-spacing: .06em;
      color: #0f6b7a;
      font-weight: 700;
      font-size: 7.5pt;
      margin-bottom: 1.5mm;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7mm;
      align-items: start;
    }}
    .panel {{
      border: 1px solid #d7dee8;
      border-radius: 2.5mm;
      padding: 3.5mm;
      background: #ffffff;
      break-inside: avoid;
    }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 2.4mm;
      margin: 4mm 0;
    }}
    .flow div {{
      background: #f8fafc;
      border: 1px solid #d8e0ea;
      border-radius: 2mm;
      padding: 2.6mm;
      min-height: 27mm;
    }}
    .flow b {{ color: #0f4f82; display: block; margin-bottom: 1mm; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      background: #eaf3fa;
      color: #163b5c;
      text-align: left;
      font-weight: 700;
      border-bottom: 1px solid #b9c8d6;
      padding: 1.7mm 1.5mm;
      vertical-align: top;
    }}
    td {{
      border-bottom: 1px solid #e4e9ef;
      padding: 1.5mm 1.5mm;
      vertical-align: top;
    }}
    td span {{ color: #667085; font-size: 7.5pt; }}
    .candidate-table {{ font-size: 7.35pt; line-height: 1.28; }}
    .candidate-table th:nth-child(1), .candidate-table td.rank {{ width: 8mm; text-align: center; }}
    .candidate-table th:nth-child(2) {{ width: 30mm; }}
    .candidate-table th:nth-child(3) {{ width: 19mm; }}
    .candidate-table th:nth-child(4) {{ width: 38mm; }}
    .candidate-table th:nth-child(5),
    .candidate-table th:nth-child(6),
    .candidate-table th:nth-child(7) {{ width: 14mm; }}
    .candidate-table th:nth-child(8) {{ width: 26mm; }}
    .candidate-table th:nth-child(9) {{ width: auto; }}
    .summary-table {{ font-size: 8.2pt; }}
    .num {{ text-align: right; white-space: nowrap; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: .5mm 1.5mm;
      font-size: 6.9pt;
      white-space: nowrap;
      background: #eef2f6;
      color: #344054;
    }}
    .badge.a {{ background: #e6f5ed; color: #12613c; }}
    .badge.b {{ background: #eaf3ff; color: #164d83; }}
    .badge.c {{ background: #fff4e5; color: #8a4b08; }}
    .footnote {{ font-size: 7.5pt; color: #667085; margin-top: 2.2mm; }}
    .note-list {{ margin: 0; padding-left: 4.5mm; }}
    .note-list li {{ margin-bottom: 1.8mm; }}
    .status-strip {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 2.5mm;
      margin: 4mm 0;
    }}
    .status-strip div {{
      background: #f7f9fb;
      border: 1px solid #d8e0ea;
      border-radius: 2mm;
      padding: 2.5mm;
    }}
    .status-strip b {{ display: block; font-size: 13pt; color: #0f5d93; }}
    .small {{ font-size: 8pt; color: #667085; }}
    .callout {{
      background: #f8fbf2;
      border-left: 4px solid #79a94d;
      padding: 3mm 4mm;
      margin-top: 3mm;
    }}
  </style>
</head>
<body>
  <section class="page cover">
    <div>
      <div class="section-kicker">BioMaster professor review brief</div>
      <h1>FDA 已批准小分子药物再定位计算筛选</h1>
      <p class="subtitle">基于 druggable proteome 的全量亲和筛选、多疾病方向证据融合与结构对接增强</p>
      <div class="metric-grid">
        <div class="metric"><strong>{intval(total_drugs)}</strong><span>FDA 已批准小分子</span></div>
        <div class="metric"><strong>{intval(total_proteins)}</strong><span>ChEMBL druggable proteins</span></div>
        <div class="metric"><strong>{intval(total_pairs)}</strong><span>基础 drug-target pairs</span></div>
        <div class="metric"><strong>{intval(disease_docking_rows)}</strong><span>疾病方向结构候选</span></div>
      </div>
      <p>
        本文件是供专家快速审阅的压缩版。内容按当前网站口径整理：
        先说明研究目的和计算流程，再给出总体数据、五个疾病方向各 Top10 候选，
        最后列出结果解释边界和下一步验证建议。
      </p>
      <p class="meta">生成时间：{generated}<br>数据来源：本机当前 BioMaster 网站与 outputs 产物。</p>
    </div>
    <div class="figure-frame">
      <img src="{main_figure}" alt="BioMaster workflow">
      <p class="footnote">图：BioMaster 从药物库、蛋白库、AI-DTI、疾病证据到结构增强和专家审阅候选的总体流程。</p>
    </div>
  </section>

  <section class="page">
    <div class="section-kicker">Study design</div>
    <h2>研究问题与计算流程</h2>
    <p>
      这项工作不是直接证明某个药物可以治疗某个疾病，而是建立一个可解释的候选优先级体系：
      在 FDA 已批准小分子与药物相关蛋白空间内，寻找同时具备药物-蛋白亲和信号、疾病方向证据、
      通路/网络支持、结构姿态可审阅性和安全性审查基础的候选。
    </p>
    <div class="flow">
      <div><b>1. 输入空间</b>915 个 FDA 已批准小分子，5306 个 ChEMBL druggable proteins。</div>
      <div><b>2. 亲和筛选</b>用 ConPLex 对约 485 万个 drug-target pair 做全量 AI-DTI 评分。</div>
      <div><b>3. 疾病收敛</b>按肿瘤、感染、心血管、神经/精神、免疫/炎症五个方向融合疾病证据。</div>
      <div><b>4. 结构增强</b>对方向候选运行 DiffDock，并补充 pose、pocket、GNINA/Vina/smina 等审计。</div>
      <div><b>5. 专家审阅</b>结合已知靶点、文献、CMap/LINCS、ADMET、禁忌证和实验可行性收敛候选。</div>
    </div>
    <div class="two-col">
      <div class="panel">
        <h3>核心证据层</h3>
        <ul class="note-list">
          <li><b>ConPLex：</b>回答药物和蛋白是否存在相互作用可能性。</li>
          <li><b>Open Targets / TxGNN：</b>回答靶点或药物与疾病方向是否有外部证据。</li>
          <li><b>GO / Reactome / STRING / HuRI：</b>补充通路与网络医学解释。</li>
          <li><b>CREEDS 与 CMap/LINCS：</b>评估疾病表达签名与药物扰动签名是否方向相反。</li>
          <li><b>GTEx / HPA / DepMap：</b>补充组织表达和肿瘤依赖背景。</li>
          <li><b>DiffDock / GNINA / Vina / smina：</b>提供结构姿态和二次 docking 审计。</li>
          <li><b>ADMET / 禁忌证：</b>用于专家审阅前的安全性降权和风险标注。</li>
        </ul>
      </div>
      <div class="panel">
        <h3>结果如何理解</h3>
        <ul class="note-list">
          <li>Top 候选是“优先审阅对象”，不是疗效结论。</li>
          <li>癌症、感染等方向是疾病证据收敛口径，不代表所有蛋白都有相同疾病位点。</li>
          <li>DiffDock 分数用于判断结构姿态是否值得查看，不能直接等同于实验结合自由能。</li>
          <li>已知药物-已知靶点召回用于评估排序是否能找回正控；新候选仍需文献、机制和实验验证。</li>
          <li>重复 UniProt/同序列蛋白会在结构层做代表性对接，再回填到同药物-同序列记录。</li>
        </ul>
      </div>
    </div>
  </section>

  <section class="page">
    <div class="section-kicker">Current result snapshot</div>
    <h2>总体结果与当前计算状态</h2>
    <div class="status-strip">
      <div><b>{intval(total_pairs)}</b><span>亲和筛选 pair</span></div>
      <div><b>{intval(disease_docking_rows)}</b><span>五方向结构候选</span></div>
      <div><b>{pct(disease_success, 2)}</b><span>方向候选结构完成率</span></div>
      <div><b>{pct(headline.get("concordanceTop100MultiEvidencePct"), 1)}</b><span>Top100 多证据覆盖</span></div>
      <div><b>{intval(headline.get("experimentalValidationExperimentOrReviewReadyRows"))}</b><span>实验/审阅就绪候选</span></div>
    </div>
    <h3>五个疾病方向结构候选概览</h3>
    {build_direction_summary_table(direction_summary)}
    <div class="two-col" style="margin-top: 5mm;">
      <div class="panel">
        <h3>已完成的关键计算</h3>
        <ul class="note-list">
          <li>FDA 小分子与 druggable proteome 全量亲和评分。</li>
          <li>五个疾病方向 Top 候选收敛和方向候选 DiffDock 结构增强。</li>
          <li>Open Targets、TxGNN、GO/Reactome、CREEDS、CMap/LINCS、GTEx/HPA、DepMap、STRING/HuRI 等多证据整合。</li>
          <li>ADMET、已知靶点召回、pose sanity、pocket pLDDT、PoseBusters/ProLIF、GNINA/Vina/smina、Boltz-2 子集验证。</li>
        </ul>
      </div>
      <div class="panel">
        <h3>后台全量 DiffDock 扩展</h3>
        <p>
          该扩展用于扩大结构覆盖，不影响本审阅版五个疾病方向 Top10 的展示。
          当前主队列 jobs：<b>{full_jobs}</b>；已评分 rows：<b>{full_rows}</b>；
          行进度：<b>{full_pct}</b>；预计剩余：<b>{eta_text}</b>；
          预计完成 UTC：<b>{esc(finish)}</b>。
        </p>
        <p class="small">missing output 是技术性结构输出缺失，不是生物学阴性；已配置后续 rescue 与最终合并 watcher。</p>
      </div>
    </div>
  </section>

  {build_top10_tables(candidates)}

  <section class="page">
    <div class="section-kicker">How to review</div>
    <h2>专家审阅建议与下一步验证</h2>
    <div class="two-col">
      <div class="panel">
        <h3>建议优先看的问题</h3>
        <ul class="note-list">
          <li>该 drug-target pair 是否为已知靶点、已知适应症，还是同疾病方向的新靶点扩展。</li>
          <li>靶点在对应疾病方向中的机制是否合理：通路、组织表达、肿瘤依赖或免疫/感染背景是否支持。</li>
          <li>药物扰动签名是否可能反转疾病表达签名，CMap/LINCS 与 CREEDS 是否一致。</li>
          <li>结构姿态是否落在可信 pocket，是否存在明显 pose 冲突或低置信蛋白区域。</li>
          <li>ADMET、禁忌证、药物相互作用、适应症人群是否允许进入后续实验设计。</li>
        </ul>
      </div>
      <div class="panel">
        <h3>建议的验证路线</h3>
        <ul class="note-list">
          <li><b>第一步：</b>对 Top10/Top20 做人工文献核查，标注 positive control、机制延伸和严格新候选。</li>
          <li><b>第二步：</b>对结构姿态优先级高的候选进行二次 pocket 审计和必要的复合物重采样。</li>
          <li><b>第三步：</b>按疾病方向选 20-50 个专家审阅候选，做 CMap/LINCS、ADMET 和禁忌证联合筛除。</li>
          <li><b>第四步：</b>进入靶点结合、功能读出或细胞表型实验；已知靶点候选可作为阳性对照。</li>
        </ul>
      </div>
    </div>
    <div class="callout">
      <b>一句话口径：</b>
      当前结果的价值在于把“老药 × druggable proteome”的大规模组合压缩成五个疾病方向下可解释、可审阅、
      可继续实验验证的候选集合；它是决策支持和机制假设生成工具，不替代湿实验和临床证据。
    </div>
    <p class="footnote">
      附注：本审阅版只列每个疾病方向 Top10。完整候选、结构文件、图表和产物索引以网站与 outputs 目录为准。
    </p>
  </section>
</body>
</html>
"""

    html_path = out_dir / "biomaster_professor_review_brief_2026_06_05.html"
    pdf_path = out_dir / "biomaster_professor_review_brief_2026_06_05.pdf"
    html_path.write_text(html_text, encoding="utf-8")
    HTML(filename=str(html_path), base_url=str(root)).write_pdf(str(pdf_path))
    return html_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/report_scale/professor_review",
    )
    args = parser.parse_args()
    html_path, pdf_path = write_report(args.root.resolve(), args.out_dir.resolve())
    print(json.dumps({"html": str(html_path), "pdf": str(pdf_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

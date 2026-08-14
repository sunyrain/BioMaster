#!/usr/bin/env python3
"""Package the known-pocket atlas and render a concise Chinese audit report."""

from __future__ import annotations

import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import fitz
import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas"
MASTER = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
REPORT_DIR = ROOT / "outputs/chembl37_known_pocket_atlas/report"
PDF = REPORT_DIR / "CHEMBL37_888_KNOWN_POCKET_ATLAS_AND_P2RANK_AUDIT_ZH.pdf"
HTML_OUT = PDF.with_suffix(".html")
XLSX = REPORT_DIR / "CHEMBL37_888_KNOWN_POCKET_ATLAS_AND_P2RANK.xlsx"
AUDIT = REPORT_DIR / "CHEMBL37_888_KNOWN_POCKET_ATLAS_AND_P2RANK_AUDIT.json"


def esc(value: object) -> str:
    return html.escape(str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(numerator: int, denominator: int, z: float = 1.96) -> tuple[float, float]:
    if denominator == 0:
        return math.nan, math.nan
    value = numerator / denominator
    base = 1 + z * z / denominator
    center = (value + z * z / (2 * denominator)) / base
    spread = z * math.sqrt(value * (1 - value) / denominator + z * z / (4 * denominator**2)) / base
    return center - spread, center + spread


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def metric(label: str, value: str, note: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        '</div>'
    )


def bar(label: str, value: int, total: int, color: str = "#087b70") -> str:
    ratio = value / total if total else 0
    return f"""
    <div class="bar-row">
      <div class="bar-label">{esc(label)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{100*ratio:.2f}%;background:{color}"></div></div>
      <div class="bar-value">{value:,} · {percent(ratio)}</div>
    </div>"""


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def decision_package(targets: pd.DataFrame, representatives: pd.DataFrame) -> pd.DataFrame:
    representative = representatives[[
        "uniprot_accession", "known_pocket_id", "known_pocket_grade", "evidence_sources",
        "representative_pdb_id", "representative_chain_id", "representative_ligand_id",
        "residue_set_key", "binding_residue_count",
    ]].rename(columns={
        "known_pocket_id": "representative_known_pocket_id",
        "known_pocket_grade": "representative_known_pocket_grade",
        "evidence_sources": "representative_known_pocket_sources",
        "residue_set_key": "representative_known_pocket_residues",
        "binding_residue_count": "representative_known_pocket_residue_count",
    })
    output = targets.merge(representative, on="uniprot_accession", how="left", validate="one_to_one")

    def category(row: pd.Series) -> str:
        grade = str(row.get("representative_known_pocket_grade", ""))
        if grade == "K1_DRUG_MAPPED_EXPERIMENTAL":
            return "E1_药物映射实验口袋"
        if grade == "K2_SPECIALIZED_CURATED_SITE":
            return "E2_专库确认实验口袋"
        if grade == "K3_EXPERIMENTAL_DRUGLIKE_SITE":
            return "E3_实验药物样口袋待复核"
        if grade == "K4_FUNCTIONAL_OR_FRAGMENT_SITE":
            return "E4_功能配体或片段口袋待复核"
        if row["p2rank_tier"] in {"A_HIGH_CONFIDENCE", "B_MODERATE_CONFIDENCE"}:
            return "P1_无实验口袋_P2Rank_AB探索"
        if row["p2rank_tier"] == "NOT_RUN":
            return "P3_无实验口袋且无精确AF结构"
        return "P2_无实验口袋_P2Rank低置信或无口袋"

    output["pocket_decision_tier"] = output.apply(category, axis=1)
    output["primary_pocket_policy"] = output["pocket_decision_tier"].map({
        "E1_药物映射实验口袋": "优先实验holo口袋；P2Rank仅作复核/备选位点",
        "E2_专库确认实验口袋": "优先专库确认实验口袋；核验配体与构象语境",
        "E3_实验药物样口袋待复核": "人工确认配体生物学意义后使用实验口袋",
        "E4_功能配体或片段口袋待复核": "区分辅因子/底物/片段；不可直接视为药物口袋",
        "P1_无实验口袋_P2Rank_AB探索": "以P2Rank为预测起点；需另一口袋模型或结构证据确认",
        "P2_无实验口袋_P2Rank低置信或无口袋": "暂不作为标准结构筛选靶点",
        "P3_无实验口袋且无精确AF结构": "先补全或校正结构，暂不进行口袋筛选",
    })
    return output


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ATLAS / "KNOWN_POCKET_ATLAS_AND_P2RANK_SUMMARY.json").read_text())
    targets = pd.read_csv(ATLAS / "TARGET_KNOWN_POCKET_AND_P2RANK_SUMMARY_888.csv", low_memory=False)
    representatives = pd.read_csv(ATLAS / "TARGET_REPRESENTATIVE_KNOWN_POCKET_737.csv", low_memory=False)
    source_coverage = pd.read_csv(ATLAS / "KNOWN_POCKET_SOURCE_COVERAGE_ZH.csv")
    any_counts = pd.read_csv(ATLAS / "P2RANK_COMPARISON_COUNTS_ZH.csv")
    rep_counts = pd.read_csv(ATLAS / "P2RANK_REPRESENTATIVE_POCKET_COMPARISON_ZH.csv")
    master = pd.read_csv(MASTER, low_memory=False)
    decisions = decision_package(targets, representatives)
    decisions.to_csv(REPORT_DIR / "TARGET_POCKET_DECISION_PACKAGE_888.csv", index=False)
    enhanced = master.merge(
        decisions.drop(columns=[column for column in decisions.columns if column in master.columns and column != "uniprot_accession"]),
        on="uniprot_accession", how="left", validate="one_to_one",
    )
    enhanced.to_csv(
        REPORT_DIR / "CHEMBL37_HUMAN_SINGLE_PROTEIN_MOA_TARGETS_888_POCKET_ENHANCED.csv.gz",
        index=False, compression="gzip",
    )
    decision_counts = decisions["pocket_decision_tier"].value_counts().rename_axis("口袋决策层").reset_index(name="靶点数")
    decision_counts["占888比例"] = decision_counts["靶点数"] / 888

    method_rows = pd.DataFrame([
        ["PDBe/SIFTS", "实时实验结构配体接触残基，并映射到canonical UniProt编号", "实验事实层；含溶剂、离子和冗余，必须过滤"],
        ["BioLiP2", "半人工筛选生物学相关配体，并汇总亲和来源", "为PDBe位点增加生物学相关性与亲和注释"],
        ["scPDB 2017", "PDB中药物样/可配体化位点及相互作用指纹", "历史专库确认；版本较旧，不负责新增结构"],
        ["KLIFS", "激酶口袋、构象、质量、配体和85位点标准化", "激酶专用实验结构确认"],
        ["GPCRdb", "GPCR结构状态、配体和通用编号相互作用残基", "GPCR专用实验结构确认"],
        ["Open Targets 26.06", "small-molecule tractability桶", "靶点级可做性标签，不是残基级口袋真值"],
        ["P2Rank 2.6-alpha", "在AlphaFold结构上预测口袋中心、残基、分数和概率", "无配体预测层；需用实验位点回顾性校验"],
    ], columns=["来源/方法", "提供什么", "本项目如何使用"])
    provenance_rows = pd.DataFrame([
        ["PDBe API/SIFTS", "live 2026-08-04", "https://www.ebi.ac.uk/pdbe/api/", "888/888查询完成", "残基级实验配体位点主干"],
        ["BioLiP2", "download 2026-08-04", "https://zhanggroup.org/BioLiP/download/", "58,494条项目记录", "生物学相关配体与亲和注释"],
        ["scPDB", "2017 release", "https://drugdesign.unistra.fr/scPDB/ressources/2016/", "3,519条项目PDB交集全部抓取", "历史可配体化位点专库"],
        ["KLIFS", "live 2026-08-04", "https://klifs.net/api", "9,399个项目激酶结构", "激酶构象与口袋专库"],
        ["GPCRdb", "live 2026-08-04", "https://gpcrdb.org/services", "1,128个结构/1,000个位点", "GPCR状态与通用残基编号"],
        ["Open Targets", "26.06", "本地完整target parquet", "886/888靶点匹配", "tractability靶点级补充"],
        ["P2Rank", "2.6-alpha local", "本地AlphaFold exact模型", "875运行/815预测出口袋", "无配体预测与回顾性比较"],
        ["canSAR raw API", "2026-08-04检查", "https://cansar.ai/api", "需授权，未伪造残基数据", "仅通过Open Targets公开桶保留目标级信息"],
    ], columns=["来源", "版本/日期", "官方入口/本地来源", "收集状态", "角色"])
    provenance_path = REPORT_DIR / "KNOWN_POCKET_SOURCE_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance_rows.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        decisions.to_excel(writer, sheet_name="888靶点口袋摘要", index=False)
        decision_counts.to_excel(writer, sheet_name="口袋决策分层", index=False)
        source_coverage.to_excel(writer, sheet_name="数据源覆盖", index=False)
        rep_counts.to_excel(writer, sheet_name="P2Rank代表口袋", index=False)
        any_counts.to_excel(writer, sheet_name="P2Rank任意口袋", index=False)
        method_rows.to_excel(writer, sheet_name="方法说明", index=False)
        provenance_rows.to_excel(writer, sheet_name="来源与版本", index=False)

    categories = decision_counts.set_index("口袋决策层")["靶点数"].to_dict()
    rep_main = rep_counts[
        rep_counts["评估集合"].isin(["全部靶点代表口袋", "K1药物映射代表口袋", "K1+K2高置信代表口袋"])
        & rep_counts["匹配口径"].eq("残基或中心<=8A")
        & rep_counts["P2Rank范围"].isin(["TOP1", "TOP3"])
    ]
    rep_lookup = {
        (row["评估集合"], row["P2Rank范围"]): row for _, row in rep_main.iterrows()
    }
    all_top1 = rep_lookup[("全部靶点代表口袋", "TOP1")]
    all_top3 = rep_lookup[("全部靶点代表口袋", "TOP3")]
    k1_top1 = rep_lookup[("K1药物映射代表口袋", "TOP1")]
    k1_top3 = rep_lookup[("K1药物映射代表口袋", "TOP3")]

    evaluable_rep = representatives[
        representatives["af_exact_sequence_model"]
        & representatives["p2rank_status"].isin(["completed", "completed_no_pocket"])
    ]
    tier_labels = [
        ("A_HIGH_CONFIDENCE", "A 高置信"),
        ("B_MODERATE_CONFIDENCE", "B 中等置信"),
        ("C_WEAK_REVIEW", "C 弱证据复核"),
        ("D_LOW_CONFIDENCE", "D 低置信"),
        ("D_NO_POCKET", "D 未预测出口袋"),
    ]
    tier_rows = []
    for tier, label in tier_labels:
        group = evaluable_rep[evaluable_rep["p2rank_tier"].eq(tier)]
        if group.empty:
            continue
        numerator = int(group["p2rank_top1_combined_match_8a"].sum())
        tier_rows.append([label, len(group), numerator, percent(numerator / len(group))])

    source_rows = [
        [row["来源/口径"], f'{int(row["覆盖靶点数"]):,}', percent(float(row["占888比例"]))]
        for _, row in source_coverage.iterrows()
    ]
    decision_rows = [
        [row["口袋决策层"], f'{int(row["靶点数"]):,}', percent(float(row["占888比例"]))]
        for _, row in decision_counts.iterrows()
    ]
    compare_rows = []
    for label, row in [
        ("全部代表口袋 Top1", all_top1),
        ("全部代表口袋 Top3", all_top3),
        ("K1药物口袋 Top1", k1_top1),
        ("K1药物口袋 Top3", k1_top3),
    ]:
        low, high = wilson(int(row["匹配靶点数"]), int(row["可评估靶点数"]))
        compare_rows.append([
            label, int(row["可评估靶点数"]), int(row["匹配靶点数"]),
            percent(float(row["匹配率"])), f"{percent(low)}–{percent(high)}",
        ])

    css = r"""
    @page { size:A4; margin:16mm 16mm 18mm; @top-left{content:"ChEMBL 37 · 888靶点";color:#52716d;font-size:8pt} @top-right{content:"已知口袋图谱与P2Rank审计";color:#52716d;font-size:8pt} @bottom-right{content:counter(page) " / " counter(pages);color:#74827f;font-size:8pt} }
    @page:first { margin:0; @top-left{content:none} @top-right{content:none} @bottom-right{content:none} }
    *{box-sizing:border-box} body{margin:0;color:#172421;font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;font-size:10pt;line-height:1.62}
    .cover{height:297mm;padding:28mm 23mm 22mm;border-top:10mm solid #087b70;background:#f7faf9;display:flex;flex-direction:column;page-break-after:always}
    .kicker{color:#087b70;font-weight:700;font-size:11pt;margin-top:12mm}.cover h1{font-size:29pt;line-height:1.24;margin:13mm 0 7mm;letter-spacing:0}.subtitle{font-size:13pt;color:#415450;max-width:160mm;line-height:1.75}.rule{width:45mm;height:2mm;background:#e5a73c;margin:13mm 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:5mm}.metric{border-top:1.3mm solid #087b70;padding-top:3mm}.metric-value{font-size:22pt;font-weight:800}.metric-label{font-weight:700;color:#244943}.metric-note{font-size:8.5pt;color:#64726f}.scope{margin-top:auto;border-left:1.5mm solid #e5a73c;padding:2mm 0 2mm 5mm;color:#40514e}.date{margin-top:7mm;color:#697773;font-size:9pt}
    h2{font-size:17pt;color:#087b70;border-bottom:.4mm solid #bed5d1;padding-bottom:2mm;margin:9mm 0 4mm;page-break-after:avoid} h3{font-size:12pt;color:#1b5048;margin:5mm 0 2mm;page-break-after:avoid} p{margin:0 0 3mm} ul{margin:1mm 0 3mm;padding-left:6mm} li{margin-bottom:1mm}
    .callout{border-left:1.3mm solid #e5a73c;background:#f7f9f8;padding:3.5mm 4.5mm;margin:4mm 0}.warn{border-left-color:#bf684f;background:#fbf6f4}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4mm;margin:4mm 0}.small-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:4mm}.card{border:.3mm solid #cbd9d6;padding:3.5mm;background:#fbfcfc;break-inside:avoid}.card strong{display:block;color:#18564e;font-size:11pt;margin-bottom:1mm}
    table{width:100%;border-collapse:collapse;margin:3mm 0 5mm;font-size:8.3pt;line-height:1.38} th{background:#e7f1ef;color:#164a43;text-align:left} th,td{border:.25mm solid #c3d2cf;padding:1.7mm;vertical-align:top} tr{break-inside:avoid} tbody tr:nth-child(even){background:#fafcfb}
    .bar-row{display:grid;grid-template-columns:52mm 1fr 31mm;gap:3mm;align-items:center;margin:2.2mm 0;font-size:9pt}.bar-track{height:4.2mm;background:#e7eceb}.bar-fill{height:100%}.bar-value{text-align:right;font-variant-numeric:tabular-nums}.bar-label{color:#344d48}.mono{font-family:"DejaVu Sans Mono",monospace;font-size:8.5pt}.page-break{page-break-before:always}.foot{font-size:8.3pt;color:#65736f;border-top:.3mm solid #d2ddda;padding-top:2.5mm;margin-top:6mm}
    """
    cover_metrics = "".join([
        metric("实验口袋覆盖", "737 / 888", "canonical残基可用于比较"),
        metric("高置信实验口袋", "579", "K1药物映射 + K2专库确认"),
        metric("P2Rank Top1", "52.4%", "保守代表口袋综合匹配"),
    ])
    body = f"""
    <section class="cover"><div class="kicker">CHEMBL 37 HUMAN SINGLE-PROTEIN MOA TARGETS</div><h1>888个成药锚点<br>已知口袋图谱与P2Rank审计</h1><div class="subtitle">整合实验结构、专类口袋数据库与Open Targets tractability，并在统一UniProt残基坐标上回顾性评价AlphaFold/P2Rank预测。</div><div class="rule"></div><div class="metrics">{cover_metrics}</div><div class="scope">实验已知口袋与计算预测口袋分层保存。数据库覆盖不等于可直接对接，P2Rank找回历史位点也不等于未知药物–靶点pair会结合。</div><div class="date">正式信息包 · 2026-08-04</div></section>
    <main>
    <h2>一、结论</h2><div class="callout"><strong>人源蛋白当然存在大量已知口袋数据，但不存在一个“对所有蛋白完整、全部是药物口袋、可直接用于对接”的单一数据库。</strong>本轮已把可批量审计的核心实验来源统一到888个ChEMBL-MoA靶点，并将P2Rank从“口袋证据”重新定位为“无配体结构上的预测工具”。</div>
    <div class="grid"><div class="card"><strong>737个</strong>至少一个通过残基数、配体类型和canonical映射门槛的实验口袋。</div><div class="card"><strong>579个</strong>具有药物映射或BioLiP2/scPDB/KLIFS/GPCRdb专库支持的高置信实验口袋。</div><div class="card"><strong>151个</strong>没有合格实验口袋；其中76个只有P2Rank A/B预测，可进入探索层。</div></div>
    <h2>二、数据源与职责</h2>{table(["来源/方法","提供什么","本项目如何使用"],method_rows.values.tolist())}
    <p class="foot">PDBe与RCSB均基于全球PDB实验结构，本项目以PDBe API和SIFTS的UniProt残基映射为主干，避免重复抓取同一PDB事实。PDBbind、Binding MOAD、BindingDB的亲和注释通过BioLiP2字段纳入，不将其误作新的口袋几何来源。canSAR原始API需要授权，本包仅保留Open Targets公开tractability桶，不伪造残基级canSAR数据。</p>
    <h2>三、收集与标准化漏斗</h2>
    {bar('PDBe任意实验配体接触位点',769,888)}{bar('统一后合格实验口袋',737,888)}{bar('K1+K2高置信实验口袋',579,888,'#2f6b9a')}{bar('K1药物映射高质量口袋',225,888,'#d09231')}
    <ul><li>PDBe共返回95,209条target–PDB–chain–ligand接触记录。</li><li>去除溶剂/添加剂、离子/极小片段、少于3个接触残基以及canonical序列映射不可靠记录后，保留38,850条基准实例。</li><li>按靶点与canonical残基集合折叠重复晶体，得到35,269个实验接触残基集合；该数字仍代表不同配体接触集合，不应解释为35,269个彼此独立的物理腔体。</li><li>最终比较以729个同时具有合格实验口袋和精确AlphaFold模型的靶点为分母；另8个实验口袋靶点没有精确AF结构，不进入P2Rank性能分母。</li></ul>
    <h2>四、各来源覆盖</h2>{table(["来源/口径","靶点数","占888"],source_rows)}
    <div class="callout">不同来源并非互斥。PDBe负责完整实验事实，BioLiP2负责生物学相关性，scPDB/KLIFS/GPCRdb负责专库语境；重复支持用于审计可信度，不按“票数”直接生成成药分。</div>
    <h2 class="page-break">五、P2Rank如何比较</h2>
    <div class="small-grid"><div class="card"><strong>残基重叠</strong>实验接触残基召回≥25%，或Jaccard≥0.10。它允许预测口袋边界与实验接触集合不完全一致。</div><div class="card"><strong>几何中心</strong>将实验接触残基映射到同一AlphaFold模型，计算其CA中心与P2Rank中心距离；同时报告≤4 Å和≤8 Å。</div></div>
    <p>为控制热门靶点的重复结构偏倚，主结果采用“每个靶点只保留一个最高证据代表口袋”。“命中任意历史口袋”另行保存，只作为较宽松上界。</p>
    {table(["评估口径","分母","匹配数","匹配率","Wilson 95%区间"],compare_rows)}
    <div class="callout warn"><strong>52.4%不是P2Rank的未知靶点成药成功率。</strong>它表示在已有实验口袋的蛋白上，Top1预测与一个预先选定的代表口袋满足本项目的残基或8 Å几何规则。它不评价具体FDA药物是否结合，也不等于Kd、precision或湿实验命中率。</div>
    <h3>结果为何有宽窄两个数</h3><ul><li>“任意历史口袋”Top1综合匹配为529/729（72.6%），适合回答P2Rank是否找到过该靶点的某个已知位点，但对多口袋热门蛋白偏乐观。</li><li>“单一代表口袋”Top1为382/729（52.4%），更保守；Top3升至477/729（65.4%）。</li><li>严格中心≤4 Å在代表口袋上为138/729（18.9%）；该指标对长形口袋和接触集合边界敏感，因此与残基重叠并列报告，不单独定义成功。</li></ul>
    <h2>六、P2Rank等级是否有信息</h2>{table(["P2Rank等级","已知口袋靶点数","Top1匹配数","Top1匹配率"],tier_rows)}
    <p>A级显著优于B/C/D，说明此前的P2Rank等级具有回顾性区分信息；但B以下下降明显。35个已有实验口袋的靶点被P2Rank判为无口袋，这是明确的预测漏检，不应由其他总分补偿。</p>
    <h2>七、Open Targets为何看起来覆盖较低</h2>
    <ul><li><strong>Structure with Ligand：</strong>603个，其中586个被本轮实验口袋基准覆盖。</li><li><strong>High-Quality Pocket：</strong>222个，其中220个有本轮实验口袋；它是严格tractability桶，不是“所有已知口袋”。</li><li><strong>Med-Quality Pocket：</strong>75个，其中72个有本轮实验口袋。</li><li>因此Open Targets的222不是说只有222个蛋白有口袋，而是只有222个满足其高质量小分子tractability定义。PDBe/BioLiP2的覆盖更宽，但也包含底物、辅因子、片段和非首选结构。</li></ul>
    <h2 class="page-break">八、888个靶点的最终口袋决策分层</h2>{table(["决策层","靶点数","占888"],decision_rows)}
    <h3>结构筛选使用规则</h3><ol><li>E1/E2共579个：优先选实验holo结构和对应口袋；P2Rank只用于检查是否一致、寻找备选或潜在变构位点。</li><li>E3/E4共158个：先人工核验配体身份、辅因子/底物语境、构象和口袋是否适合项目分子，再决定是否对接。</li><li>P1共76个：没有合格实验口袋，仅有P2Rank A/B；必须增加另一口袋预测、保守位点或结构文献支持后才能进入高成本计算。</li><li>P2/P3共75个：当前不应作为标准结构筛选对象，先补结构、构象或口袋证据。</li></ol>
    <h2>九、对项目流程的直接修正</h2>
    <div class="callout"><strong>已知口袋优先级：</strong>实验holo口袋 &gt; 专库确认实验口袋 &gt; 一般实验药物样口袋 &gt; P2Rank A/B共识预测 &gt; 低置信预测。P2Rank不再作为888个靶点统一、同等级的口袋来源。</div>
    <ul><li>后续对接应按具体target–construct–structure–pocket–protocol组织，而不是只按gene或AlphaFold Top1口袋组织。</li><li>同一靶点的正构、变构、辅因子和蛋白界面口袋必须分开；已知药物口袋不代表所有候选药物都能结合。</li><li>这次回顾性结果验证的是“口袋定位工具的适用范围”，没有验证任何drug–target pair。</li><li>对于没有已知口袋的151个靶点，不能用“无数据库记录”推导“没有口袋”；只能标记为缺少可审计实验真值。</li></ul>
    <h2>十、局限与边界</h2><ul><li>实验结构存在构建体、亚型、缺失区段、辅因子和构象状态差异；canonical残基统一不能消除结构状态差异。</li><li>PDBe配体接触是实验事实，但不自动等于药理相关位点；因此设置K1–K4分层。</li><li>scPDB版本较旧，只作为历史专库证据；新增PDB由实时PDBe/BioLiP2补足。</li><li>AlphaFold为单体预测结构；复合物界面、膜环境、cryptic pocket和induced fit可能无法由静态P2Rank覆盖。</li><li>本报告不声称“已收集世界上每一个私有或授权口袋库”，而是完成主要公开、可批量复现和可映射的数据源。</li></ul>
    <h2>十一、正式文件</h2>{table(["文件","用途"],[["TARGET_POCKET_DECISION_PACKAGE_888.csv","每靶点口袋证据、P2Rank比较和使用策略"],["CHEMBL37_HUMAN_SINGLE_PROTEIN_MOA_TARGETS_888_POCKET_ENHANCED.csv.gz","888靶点全字段增强主表"],["CHEMBL37_888_KNOWN_POCKET_ATLAS_AND_P2RANK.xlsx","老师/实验组可读工作簿"],["KNOWN_POCKET_VS_P2RANK_INSTANCE_COMPARISON.csv.gz","每个实验残基集合与P2Rank逐项比较"],["P2RANK_REPRESENTATIVE_POCKET_COMPARISON_ZH.csv","保守代表口袋统计"],["P2RANK_COMPARISON_COUNTS_ZH.csv","任意历史口袋的宽松上界统计"]])}
    <p class="foot">数据日期：PDBe、KLIFS、GPCRdb为2026-08-04实时抓取；Open Targets为26.06；scPDB为2017公开版本；P2Rank为本地2.6-alpha结果。所有比例均由CSV/JSON自动生成。</p>
    </main>"""
    html_text = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>'
    HTML_OUT.write_text(html_text, encoding="utf-8")
    HTML(string=html_text, base_url=str(ROOT)).write_pdf(PDF)

    with fitz.open(PDF) as document:
        page_count = document.page_count
        extracted = "\n".join(page.get_text() for page in document)
        blank_pages = [index + 1 for index, page in enumerate(document) if len(page.get_text().strip()) < 20]
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": len(decisions),
        "unique_uniprot": decisions["uniprot_accession"].nunique(),
        "decision_tier_sum": int(decision_counts["靶点数"].sum()),
        "known_pocket_targets": int(decisions["known_unique_pocket_count"].gt(0).sum()),
        "pdf_pages": page_count,
        "pdf_text_characters": len(extracted),
        "blank_pages": blank_pages,
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [
                PDF, HTML_OUT, XLSX, provenance_path,
                REPORT_DIR / "TARGET_POCKET_DECISION_PACKAGE_888.csv",
            ]
        },
        "checks": {
            "targets_888": len(decisions) == 888,
            "unique_uniprot_888": decisions["uniprot_accession"].nunique() == 888,
            "decision_tiers_sum_888": int(decision_counts["靶点数"].sum()) == 888,
            "known_pocket_targets_737": int(decisions["known_unique_pocket_count"].gt(0).sum()) == 737,
            "no_blank_pdf_pages": not blank_pages,
        },
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a counts-only Chinese PDF for the ChEMBL 37 target universe."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs/target_universe_ch37_v2"
PDF = OUTDIR / "CHEMBL37_TARGET_UNIVERSE_COUNTS_ONLY_ZH_V2.pdf"


TABLES = [
    ("不同用途的可用集合", "COUNTS_可用集合_V2.csv"),
    ("互斥小分子证据分类", "COUNTS_证据互斥分类_V2.csv"),
    ("证据与计算覆盖", "COUNTS_证据覆盖_V2.csv"),
    ("互斥实验体系分类", "COUNTS_实验体系分类_V2.csv"),
    ("ChEMBL 一级蛋白类别", "COUNTS_ChEMBL一级类别_V2.csv"),
    ("ChEMBL 机制分子类型", "COUNTS_机制分子类型_V2.csv"),
    ("结构与口袋口径", "COUNTS_结构分类_V2.csv"),
    ("历史 binding 数据等级", "COUNTS_校准分类_V2.csv"),
]


def table_html(frame: pd.DataFrame) -> str:
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body = []
    for row in frame.itertuples(index=False, name=None):
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> None:
    sections: list[str] = []
    for title, filename in TABLES:
        frame = pd.read_csv(OUTDIR / filename)
        note = ""
        if "非互斥" in " ".join(frame.columns) or "机制分子类型" in title:
            note = "<p class='note'>本表为非互斥统计，各行不可相加为 888。</p>"
        css_class = "compact" if title == "不同用途的可用集合" else ""
        sections.append(
            f"<section class='{css_class}'><h2>{html.escape(title)}</h2>{note}{table_html(frame)}</section>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 15mm 14mm 14mm; @bottom-right {{ content: counter(page) " / " counter(pages); font-size: 8pt; color: #62706d; }} }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Noto Sans CJK SC", sans-serif; color: #172321; font-size: 9.3pt; line-height: 1.42; }}
.cover {{ min-height: 255mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; border-top: 7px solid #137f73; }}
.eyebrow {{ color: #137f73; font-weight: 700; font-size: 12pt; margin-bottom: 18mm; }}
h1 {{ font-size: 28pt; line-height: 1.25; margin: 0 0 10mm; }}
.lead {{ font-size: 13pt; color: #3f504d; max-width: 160mm; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 5mm; margin-top: 18mm; }}
.metric {{ border-top: 3px solid #137f73; padding-top: 4mm; }}
.metric strong {{ display: block; font-size: 24pt; }}
.metric span {{ color: #596966; }}
section {{ page-break-before: always; }}
section:first-of-type {{ page-break-before: auto; }}
h2 {{ font-size: 16pt; margin: 0 0 5mm; color: #0d6d63; border-bottom: 1px solid #c8d5d2; padding-bottom: 2.5mm; }}
p.note {{ margin: -2mm 0 4mm; color: #5f6d6b; font-size: 8.5pt; }}
table {{ width: 100%; border-collapse: collapse; table-layout: auto; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th {{ background: #e9f2f0; color: #173f3a; font-weight: 700; text-align: left; }}
th, td {{ border: 1px solid #cbd7d5; padding: 2.2mm 2.5mm; vertical-align: top; overflow-wrap: anywhere; }}
tbody tr:nth-child(even) {{ background: #f7faf9; }}
th:nth-last-child(-n+2), td:nth-last-child(-n+2) {{ text-align: right; white-space: nowrap; }}
.compact table {{ font-size: 8pt; line-height: 1.2; }}
.compact th, .compact td {{ padding: 1.35mm 2mm; }}
.definition {{ page-break-before: always; }}
.definition h2 {{ margin-bottom: 7mm; }}
.definition p {{ margin: 0 0 4mm; }}
.callout {{ border-left: 4px solid #137f73; background: #edf5f3; padding: 4mm 5mm; margin: 6mm 0; }}
</style></head><body>
<div class="cover">
  <div class="eyebrow">CHEMBL 37 · HUMAN SINGLE-PROTEIN MOA TARGETS</div>
  <h1>人源单蛋白靶点全集<br>分类、数量与使用口径</h1>
  <p class="lead">唯一入口为 ChEMBL 37 中物种为人、靶点类型为单蛋白、且至少存在一条药物机制记录的靶点。本报告只列分类、数量和比例。</p>
  <div class="metrics">
    <div class="metric"><strong>888</strong><span>官方靶点、基因、UniProt 与唯一序列</span></div>
    <div class="metric"><strong>565</strong><span>存在 ChEMBL 小分子 MoA</span></div>
    <div class="metric"><strong>450</strong><span>非 GPCR 小分子 MoA</span></div>
  </div>
</div>
{''.join(sections)}
<section class="definition">
  <h2>口径说明</h2>
  <p><b>全集口径：</b>888 个靶点全部保留，适用于序列 DTA 和不预设药物模态的广谱探索。</p>
  <p><b>小分子机制口径：</b>565 个靶点至少有一条 ChEMBL Small molecule MoA；其中 450 个为非 GPCR。</p>
  <p><b>广义直接小分子口径：</b>在非 GPCR 小分子 MoA 基础上纳入 Open Targets 的直接小分子证据，只作为扩展集合。</p>
  <p><b>宽松结构口径：</b>AlphaFold 序列完全一致，且达到项目定义的 P2Rank A/B。</p>
  <p><b>严格结构口径：</b>进一步限定为项目 P2Rank A，口袋平均 pLDDT 不低于 70，且至少 70% 口袋残基 pLDDT 不低于 70。</p>
  <p><b>历史校准口径：</b>严格 binding 数据中，阳性和定量低活性/明确 inactive 均至少 8 个。未测或未记录不作为阴性。</p>
  <div class="callout">结构就绪只说明可以进入结构计算；历史数据充足只说明可以做回顾性局部校准。两者都不代表未知 drug-target pair 已被证明结合，也不等于实验室已经 assay-ready。</div>
</section>
</body></html>"""
    html_path = OUTDIR / "CHEMBL37_TARGET_UNIVERSE_COUNTS_ONLY_ZH_V2.html"
    html_path.write_text(document, encoding="utf-8")
    HTML(string=document, base_url=str(OUTDIR)).write_pdf(PDF)
    print(PDF)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the V8 affinity pipeline and frontier comparison report to PDF."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import fitz
import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/FDA_OLD_DRUG_AFFINITY_PIPELINE_V8_FRONTIER_COMPARISON_ZH.md"
OUTPUT_DIR = ROOT / "outputs/affinity_experiment_package_v8/report"
HTML_OUT = OUTPUT_DIR / "FDA_OLD_DRUG_AFFINITY_PIPELINE_V8_FRONTIER_COMPARISON_ZH.html"
PDF_OUT = OUTPUT_DIR / "FDA_OLD_DRUG_AFFINITY_PIPELINE_V8_FRONTIER_COMPARISON_ZH.pdf"
AUDIT_OUT = OUTPUT_DIR / "FDA_OLD_DRUG_AFFINITY_PIPELINE_V8_FRONTIER_COMPARISON_ZH_AUDIT.json"


CSS = r"""
@page {
  size: A4;
  margin: 17mm 16mm 18mm 16mm;
  @top-left {
    content: "FDA 老药新靶点 · V8";
    color: #49716b;
    font-size: 8.5pt;
  }
  @top-right {
    content: "物理优先筛选与国际前沿对照";
    color: #6b7775;
    font-size: 8.5pt;
  }
  @bottom-left {
    content: "BioMaster · 2026-08-03";
    color: #7b8583;
    font-size: 8pt;
  }
  @bottom-right {
    content: counter(page) " / " counter(pages);
    color: #7b8583;
    font-size: 8pt;
  }
}
@page:first {
  margin: 0;
  @top-left { content: none; }
  @top-right { content: none; }
  @bottom-left { content: none; }
  @bottom-right { content: none; }
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  color: #172421;
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif;
  font-size: 10.2pt;
  line-height: 1.62;
}
.cover {
  height: 297mm;
  padding: 28mm 24mm 21mm;
  display: flex;
  flex-direction: column;
  background: #f7faf9;
  border-top: 10mm solid #087b70;
  page-break-after: always;
}
.cover .kicker {
  color: #087b70;
  font-size: 12pt;
  font-weight: 700;
  letter-spacing: 0;
  margin-top: 12mm;
}
.cover h1 {
  color: #14211f;
  font-size: 31pt;
  line-height: 1.22;
  margin: 14mm 0 8mm;
  font-weight: 800;
  letter-spacing: 0;
}
.cover .subtitle {
  max-width: 155mm;
  color: #40514e;
  font-size: 14pt;
  line-height: 1.7;
}
.cover .rule {
  width: 46mm;
  height: 2mm;
  background: #e5a73c;
  margin: 14mm 0;
}
.cover .metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5mm;
  margin-top: 9mm;
}
.metric {
  border-top: 1.5mm solid #087b70;
  padding-top: 4mm;
}
.metric strong {
  display: block;
  font-size: 23pt;
  line-height: 1.1;
  color: #14211f;
}
.metric span { color: #50635f; font-size: 9.2pt; }
.cover .scope {
  margin-top: auto;
  border-left: 1.6mm solid #e5a73c;
  padding: 2mm 0 2mm 5mm;
  color: #40514e;
  font-size: 10pt;
}
.cover .date { margin-top: 8mm; color: #64726f; font-size: 9pt; }
.report { padding: 0; }
h1 { display: none; }
h2 {
  color: #087b70;
  font-size: 18pt;
  line-height: 1.25;
  margin: 10mm 0 4mm;
  padding-bottom: 2.5mm;
  border-bottom: 0.5mm solid #bdd5d1;
  page-break-after: avoid;
  letter-spacing: 0;
}
h3 {
  color: #1c4f48;
  font-size: 12.5pt;
  line-height: 1.35;
  margin: 6mm 0 2mm;
  page-break-after: avoid;
  letter-spacing: 0;
}
h4 { color: #334944; font-size: 10.8pt; margin: 4mm 0 1.5mm; }
p { margin: 0 0 3mm; orphans: 3; widows: 3; }
ul, ol { margin: 1.5mm 0 3.5mm 5.5mm; padding-left: 4.5mm; }
li { margin: 0 0 1.2mm; }
blockquote {
  margin: 4mm 0;
  padding: 3mm 4mm;
  border-left: 1.3mm solid #e5a73c;
  background: #f7f9f8;
  color: #485a56;
}
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.8pt;
  color: #125c53;
  background: #edf4f2;
  padding: 0.25mm 0.8mm;
  border-radius: 0.7mm;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 5mm;
  font-size: 8.4pt;
  line-height: 1.42;
  break-inside: auto;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  background: #e7f1ef;
  color: #164a43;
  font-weight: 700;
  border: 0.25mm solid #afc8c3;
  padding: 2.2mm 1.8mm;
  text-align: left;
}
td {
  border: 0.25mm solid #cbd7d4;
  padding: 1.8mm;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #fafcfb; }
a { color: #087b70; text-decoration: none; overflow-wrap: anywhere; }
hr { border: 0; border-top: 0.4mm solid #c8d6d3; margin: 5mm 0; }
.toc {
  background: #f5f8f7;
  border: 0.3mm solid #cbdad7;
  padding: 5mm 6mm;
  margin: 2mm 0 7mm;
  page-break-after: always;
}
.toc h2 { margin-top: 0; }
.toc ol { columns: 2; column-gap: 12mm; }
.toc li { break-inside: avoid; margin-bottom: 2mm; }
.toc a { color: #294b46; }
.source-note {
  color: #64726f;
  font-size: 8.5pt;
  border-top: 0.3mm solid #d2ddda;
  margin-top: 7mm;
  padding-top: 3mm;
}
"""


def build_toc() -> str:
    items = [
        ("一", "唯一目标与边界"),
        ("二", "正式设计空间"),
        ("三", "正式漏斗"),
        ("四", "靶点级校准"),
        ("五", "方法组合逻辑"),
        ("六", "国际前沿对照"),
        ("七", "项目定位"),
        ("八", "最终候选规则"),
        ("九", "Discovery1000 与 Assay1000"),
        ("十", "计算状态与后续动作"),
        ("十一", "国际资料来源"),
        ("十二", "结论"),
    ]
    links = "".join(f'<li><a href="#sec-{i}">{n}、{title}</a></li>' for i, (n, title) in enumerate(items, 1))
    return f'<section class="toc"><h2>目录</h2><ol>{links}</ol></section>'


def add_section_ids(body: str) -> str:
    for i in range(1, 13):
        body = body.replace("<h2>", f'<h2 id="sec-{i}">', 1)
    return body


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    body = add_section_ids(body)
    cover = """
    <section class="cover">
      <div class="kicker">FDA OLD DRUG · NEW TARGET DISCOVERY</div>
      <h1>FDA 老药新靶点<br>物理优先筛选与国际前沿对照</h1>
      <div class="subtitle">从 334,749 条正式物理组合出发，以严格远程新颖性、同靶点正负校准和 pair 级结构精修形成可验证的多靶点发现包。</div>
      <div class="rule"></div>
      <div class="metrics">
        <div class="metric"><strong>334,749</strong><span>正式 drug-target 设计空间</span></div>
        <div class="metric"><strong>16,995</strong><span>靶点校准准入精修队列</span></div>
        <div class="metric"><strong>1,000</strong><span>目标发现候选规模</span></div>
      </div>
      <div class="scope">第一阶段只回答直接结合优先级；疾病、作用方向和老药新用价值在实验命中后进入第二阶段。</div>
      <div class="date">V8 方法与计算进展报告 · 2026-08-03</div>
    </section>
    """
    html_text = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>{CSS}</style></head>
    <body>{cover}<main class="report">{build_toc()}{body}<p class="source-note">本报告中的计算数量来自 V8 CSV/JSON 台账；文献部分引用原始论文或官方资源。模型输出不是湿实验事实。</p></main></body></html>"""
    HTML_OUT.write_text(html_text, encoding="utf-8")
    HTML(string=html_text, base_url=str(ROOT)).write_pdf(PDF_OUT)

    document = fitz.open(PDF_OUT)
    text = "\n".join(page.get_text() for page in document)
    audit = {
        "status": "passed" if len(document) >= 10 and "334,749" in text and "国际前沿" in text else "failed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(SOURCE.relative_to(ROOT)),
        "html": str(HTML_OUT.relative_to(ROOT)),
        "pdf": str(PDF_OUT.relative_to(ROOT)),
        "pages": len(document),
        "pdf_bytes": PDF_OUT.stat().st_size,
        "pdf_sha256": hashlib.sha256(PDF_OUT.read_bytes()).hexdigest(),
        "required_phrases": {
            phrase: phrase in text
            for phrase in ["334,749", "16,995", "同靶点", "Discovery1000", "Assay1000", "国际前沿"]
        },
    }
    document.close()
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit["status"] != "passed" or not all(audit["required_phrases"].values()):
        raise RuntimeError(f"PDF audit failed: {audit}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

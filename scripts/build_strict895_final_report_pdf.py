#!/usr/bin/env python3
"""Build a concise Chinese report for the strict895 first-wave package."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/broad_mechanism_layer_v2"
OUTDIR = BASE / "strict895_final_wetlab_priority"
SUMMARY_PATH = OUTDIR / "strict895_final_wetlab_priority_summary.json"
OT_SUMMARY_PATH = BASE / "strict895_opentargets_child_disease/strict895_opentargets_child_disease_run_summary.json"
CONCRETE_SUMMARY_PATH = BASE / "strict895_concrete_disease_completion/strict895_concrete_disease_completion_summary.json"
UNIQUE_PATH = OUTDIR / "strict895_final_wetlab_priority_unique_hypotheses.csv"
TOP96_PATH = OUTDIR / "strict895_first_wave_top96.csv"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(counts: dict, labels: dict | None = None) -> str:
    labels = labels or {}
    lines = []
    for key, value in counts.items():
        name = labels.get(key, key)
        lines.append(f"- {name}: {value}")
    return "\n".join(lines)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    rows = []
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows.extend([header, sep])
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row.get(col, "")
            if pd.isna(value):
                value = ""
            text = str(value).replace("\n", " ").replace("|", "/")
            values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def make_markdown() -> str:
    summary = read_json(SUMMARY_PATH)
    ot = read_json(OT_SUMMARY_PATH)
    concrete = read_json(CONCRETE_SUMMARY_PATH)
    unique = pd.read_csv(UNIQUE_PATH, low_memory=False)
    top96 = pd.read_csv(TOP96_PATH, low_memory=False)

    labels = {
        "first_wave_priority": "第一轮优先",
        "first_wave_or_backup": "第一轮候补",
        "secondary_review": "二级复核",
        "control_or_rediscovery": "阳性/再发现对照",
        "deprioritized": "暂缓",
    }
    assay_zh = {
        "kinase_biochemical_cellular": "激酶生化/细胞",
        "enzyme_or_epigenetic_biochemical": "酶/表观遗传生化",
        "transporter_uptake_efflux": "转运体摄取/外排",
        "ion_channel_functional": "离子通道功能",
        "nuclear_receptor_or_tf_reporter": "核受体/转录因子 reporter",
    }

    top_table = top96.head(20).copy()
    top_table["rank"] = top_table["final_rank_unique_hypothesis"].astype(int)
    top_table["候选"] = top_table["drug_name"].astype(str) + " - " + top_table["target_gene"].astype(str)
    if "best_child_ot_disease_name_zh" in top_table.columns:
        top_table["拟新用细分病种"] = top_table["best_child_ot_disease_name_zh"].fillna(top_table["best_child_ot_disease_name"])
    else:
        top_table["拟新用细分病种"] = top_table["best_child_ot_disease_name"].fillna("")
    top_table["拟新用病种大类"] = top_table["final_disease_area_zh"].fillna(top_table["direction"])
    top_table["新用类型"] = top_table["repurposing_type_zh"].fillna("")
    top_table["分数"] = top_table["final_wetlab_priority_score"].map(lambda x: f"{float(x):.1f}")
    top_table["ConPLEx"] = top_table["conplex_score"].map(lambda x: f"{float(x):.3f}")
    top_table["OT"] = top_table["best_child_ot_score"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    if "assay_lane_zh" in top_table.columns:
        top_table["实验入口"] = top_table["assay_lane_zh"].fillna("")
    else:
        top_table["实验入口"] = top_table["assay_lane"].map(assay_zh).fillna(top_table["assay_lane"])

    text = f"""# strict895 第一轮湿实验推荐包汇报

生成日期：2026-06-27
项目位置：`/root/autodl-tmp/BioMaster`

## 一、这一轮做了什么

本轮从 `strict_top_ready` 的 895 条 drug-target-direction 候选出发，完成了三件事：

1. 补全 Open Targets 细分病种：不再只停留在 oncology、cardiovascular 这类宽方向，而是把候选靶点对应到更具体的疾病名称和 Open Targets 分数。
2. 对 895 条候选逐条做机制/可行性审计：判断药物-靶点-疾病轴是否有可解释机制、是否容易做 target engagement、需要哪些反筛。
3. 生成第一轮湿实验推荐顺序：输出去重后的 drug-target-disease hypothesis，并拆成 Top96、Top192、Top384 三个实验规模。

## 二、Open Targets 细分病种补全结果

- strict895 总行数：{ot["rows"]}
- 唯一靶点：{ot["unique_targets"]}
- 有 Ensembl ID 的靶点：{ot["targets_with_ensembl"]}
- 候选疾病名：{ot["candidate_disease_names"]}
- 可映射到 Open Targets 的疾病名：{ot["mapped_candidate_disease_names"]}
- 有 child Open Targets match 的行：{ot["rows_with_child_ot_match"]}
- child OT score >= 0.2 的行：{ot["rows_with_child_ot_score_ge_0_2"]}
- child OT score >= 0.5 的行：{ot["rows_with_child_ot_score_ge_0_5"]}

方法说明：对每个候选靶点调用 Open Targets `target.associatedDiseases`，每个靶点最多取前 4000 个疾病关联；再把本地候选疾病名映射到 Open Targets disease ID，判断该靶点是否支持相应细分病种。Open Targets 在这里不是证明药物能结合靶点，而是证明“靶点-疾病”这半条机制链是否有外部疾病证据。

## 三、strict895 审计结果

- 原始 strict895 行数：{summary["all_rows"]}
- 去重后 drug-target-disease hypothesis：{summary["unique_hypotheses"]}
- agent 已审计行：{summary["agent_reviewed_rows"]}
- agent 缺失行：{summary["missing_agent_review_rows"]}

agent 决策分布：

{fmt_counts(summary["agent_decision_counts"])}

老药新用类型分布：

{fmt_counts(summary["repurposing_type_counts_unique"])}

去重后推荐层级：

{fmt_counts(summary["recommended_track_counts_unique"], labels)}

## 四、第一轮湿实验优先包

Top96 是最建议第一轮先做的规模；它包含全部第一轮优先候选和少量第一轮候补。

- Top96 第一轮优先：{summary["top96_track_counts"].get("first_wave_priority", 0)}
- Top96 第一轮候补：{summary["top96_track_counts"].get("first_wave_or_backup", 0)}
- Top96 唯一药物：{top96["drug_name"].nunique()}
- Top96 唯一靶点：{top96["target_gene"].nunique()}
- Top96 细分病种数：{top96["best_child_ot_disease_name"].nunique()}

Top96 拟新用病种大类分布：

{fmt_counts(summary["top96_disease_area_counts"])}

Top96 实验类型分布：

{fmt_counts({assay_zh.get(k, k): v for k, v in summary["top96_assay_lane_counts"].items()})}

Top96 老药新用类型分布：

{fmt_counts(summary["top96_repurposing_type_counts"])}

## 五、Top20 示例

{md_table(top_table, ["rank", "候选", "拟新用细分病种", "拟新用病种大类", "新用类型", "分数", "ConPLEx", "OT", "实验入口"])}

## 六、这些分数应该怎么解释

最终优先级不是单一亲和分数，也不是“已证实结合”。它由五类信息合成：

- agent 审计：候选是否有机制解释、是否适合首轮 target engagement。
- ConPLEx：药物-靶点相互作用预测，只作为计算证据。
- child Open Targets：靶点-细分疾病是否有遗传、临床、文献或药物证据。
- assay lane：该靶点是否容易做激酶、酶活、转运体、离子通道或 reporter 类实验。
- novelty/risk：是否只是已知靶点/同家族再发现，是否存在明显安全性、毒性或 assay 干扰风险。

所以，本推荐包的正确说法是：这些候选是“适合第一轮湿实验验证 target engagement 与机制 readout 的优先假说”，不是已经证明疗效，也不是已经证明新靶点结合。

## 七、推荐实验策略

第一轮不建议直接做疾病疗效结论。建议每个候选先过三道门：

1. 非毒性浓度窗口：先做 viability / cytotoxicity gate。
2. target engagement 或功能 readout：例如 kinase IC50、CETSA/NanoBRET、转运体摄取、离子通道电生理/膜电位、核受体 reporter。
3. 反筛：同家族 panel、原批准靶点 counterscreen、PAINS/聚集/荧光干扰、细胞毒性解释。

Go 标准：剂量依赖、可重复、反筛不能解释主效应，并且在疾病相关细胞模型中能看到方向一致的机制 readout。

## 八、当前边界和下一步

当前已经把宽方向收敛到可落地的细分病种，但仍有两个边界：

- 文献层面：本轮 agent 审计主要基于本地证据和 Open Targets child disease，并没有对 895 条逐条做联网全文级人工文献排重。建议下一步只对 Top96 或 Top192 做逐条 PubMed/Google Scholar 证据核验。
- 结合层面：除已知/对照类候选外，大多数 discovery 候选仍需要湿实验确认 drug-target engagement，不能在汇报中称为已验证新靶点。

## 九、输出文件

- 总表：`{OUTDIR / "strict895_final_wetlab_priority_unique_hypotheses.csv"}`
- 最简扫读总表：`{OUTDIR / "strict895_final_wetlab_priority_summary_table_zh.csv"}`
- 证据卡总表：`{OUTDIR / "strict895_final_wetlab_priority_teacher_readable_zh.csv"}`
- Top96：`{OUTDIR / "strict895_first_wave_top96.csv"}`
- Top96 最简扫读表：`{OUTDIR / "strict895_first_wave_top96_summary_table_zh.csv"}`
- Top96 证据卡表：`{OUTDIR / "strict895_first_wave_top96_teacher_readable_zh.csv"}`
- Top192：`{OUTDIR / "strict895_first_wave_top192.csv"}`
- Top192 最简扫读表：`{OUTDIR / "strict895_first_wave_top192_summary_table_zh.csv"}`
- Top192 证据卡表：`{OUTDIR / "strict895_first_wave_top192_teacher_readable_zh.csv"}`
- Top384：`{OUTDIR / "strict895_first_wave_top384.csv"}`
- Top384 最简扫读表：`{OUTDIR / "strict895_first_wave_top384_summary_table_zh.csv"}`
- Top384 证据卡表：`{OUTDIR / "strict895_first_wave_top384_teacher_readable_zh.csv"}`
- 本报告 Markdown：`{OUTDIR / "STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.md"}`
- 本报告 PDF：`{OUTDIR / "STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.pdf"}`
"""
    return text


def markdown_to_html(markdown_text: str) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables"])
    css = """
    @page { size: A4; margin: 18mm 15mm; }
    body {
      font-family: "Noto Sans CJK SC", "Noto Sans CJK", sans-serif;
      font-size: 10.5pt;
      line-height: 1.55;
      color: #1f2933;
    }
    h1 { font-size: 20pt; margin: 0 0 14pt; }
    h2 { font-size: 14pt; margin: 18pt 0 8pt; border-bottom: 1px solid #d7dde5; padding-bottom: 3pt; }
    p { margin: 5pt 0; }
    ul, ol { margin: 5pt 0 7pt 18pt; padding: 0; }
    li { margin: 2pt 0; }
    code { font-family: "Noto Sans Mono", monospace; font-size: 9pt; }
    table { width: 100%; border-collapse: collapse; font-size: 8pt; margin: 8pt 0; }
    th, td { border: 1px solid #d7dde5; padding: 4pt 5pt; vertical-align: top; }
    th { background: #eef2f7; font-weight: 700; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def main() -> None:
    md = make_markdown()
    md_path = OUTDIR / "STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.md"
    html_path = OUTDIR / "STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.html"
    pdf_path = OUTDIR / "STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.pdf"
    md_path.write_text(md, encoding="utf-8")
    html_text = markdown_to_html(md)
    html_path.write_text(html_text, encoding="utf-8")
    HTML(string=html_text, base_url=str(OUTDIR)).write_pdf(pdf_path)
    print(json.dumps({
        "markdown": str(md_path),
        "html": str(html_path),
        "pdf": str(pdf_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

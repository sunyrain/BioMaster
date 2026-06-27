# strict895 当前交付包说明

更新日期：2026-06-27

## 定位

本包是 FDA 老药新用项目当前主线的 strict895 审阅结果。它的定位是：

- `895` 行全量审阅池：用于向导师展示完整筛选和分层结果。
- `34` 行严格 novelty 子表：用于从第一轮优先候选中挑选更干净的新颖验证对象。
- PubMed 文献审计：用于区分“上市后已有新靶点线索”和“PubMed pair 未见明确报道”。

它不是最终 wet-lab 命中表。第一轮实验建议从严格 novelty 子表、机制扩展候选和阳性/再发现对照中组合选择。

## 关键输出

核心目录：

`outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority`

建议优先查看：

- `strict895_all_895_summary_table_zh.xlsx`：895 行全量中文简表。
- `strict895_first_wave_strict_novelty_summary_table_zh.xlsx`：34 行严格 novelty 候选。
- `strict895_final_wetlab_priority_teacher_readable_zh.csv`：873 个去重 hypothesis 的证据卡式总表。
- `STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.pdf`：中文汇报 PDF。

文献审计：

- `outputs/broad_mechanism_layer_v2/strict895_pubmed_literature_audit/strict895_pair_pubmed_literature_audit.csv`
- `outputs/broad_mechanism_layer_v2/strict895_pubmed_literature_audit/strict895_pubmed_literature_audit_summary.json`

## 当前主要数字

- strict895 原始行：895
- 去重 drug-target-disease hypothesis：873
- 第一轮优先：87
- 第一轮候补：260
- 二级复核：307
- 阳性/再发现对照：86
- 暂缓：155
- 严格 novelty 子表：34

895 行文献状态：

- PubMed pair 审计未见明确报道：646
- 上市后文献有新靶点线索：231
- 上市前或窗口外报道：9
- 已知药理/对照：9

## 复现脚本

核心脚本：

- `scripts/audit_strict895_pubmed_literature.py`
- `scripts/build_strict895_final_wetlab_priority.py`
- `scripts/build_strict895_final_report_pdf.py`

上游辅助脚本：

- `scripts/build_strict895_agent_review_package.py`
- `scripts/build_strict895_concrete_disease_annotations.py`
- `scripts/fetch_strict895_opentargets_child_diseases.py`
- `scripts/merge_strict895_agent_reviews.py`
- `scripts/build_comprehensive_repurposing_literature_report.py`

典型重建顺序：

```bash
python scripts/audit_strict895_pubmed_literature.py --workers 1 --delay-s 0.34
python scripts/build_strict895_final_wetlab_priority.py
python scripts/build_strict895_final_report_pdf.py
```

## 注意事项

- 原始大数据、模型缓存、第三方代码和旧版本报告不纳入 Git。
- `outputs/` 默认忽略，只放行本包需要的 strict895 精选结果。
- PubMed pair 审计是自动共现/窗口审计，不等价于直接结合实验证据。
- 严格 novelty 子表排除了 KRAS、SCN5A、SLC22A12、SLC29A1、SLC34A2、SLC5A2 等高风险或 ADME 混杂明显对象。

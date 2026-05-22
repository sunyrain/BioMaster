# BioMaster 当前结果状态

更新时间：2026-05-22

## 摘要

BioMaster 已完成 915 个 FDA 已批准小分子与 1000 个人类蛋白靶点的全组合 AI-DTI 主筛选，并完成 cancer 方向疾病证据融合排序。高优先级 1000 个 drug-target pair 已形成结构增强共识结果，可用于候选审阅、机制分析和实验设计。

更大规模结构扩展正在作为补充证据继续推进。该扩展用于增加结构层面的解释信息，不改变主筛选和高优先级候选集已经形成的结论。

## 已形成产物

| 模块 | 状态 | 主要产物 |
| --- | --- | --- |
| 药物库 | 已完成 | `data/processed/drug_library_pubchem_chembl_mapped.csv` |
| 蛋白靶点库 | 已完成 | `data/processed/protein_library_1000_alphafold_paths.csv` |
| AI-DTI 主筛选 | 已完成 | `outputs/report_scale/conplex_affinity_scores_915k.csv` |
| 结构就绪清单 | 已完成 | `outputs/report_scale/manifest_915k_diffdock_ready.csv` |
| 疾病证据排序 | 已完成 | `outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv` |
| 高优先级结构增强 | 已完成 | `outputs/report_scale/stage6_top1000_consensus_candidates.csv` |
| 前端展示页面 | 已完成 | `docs/index.html` |
| 正式英文 PDF | 已完成 | `docs/assets/biomaster-external-report.pdf` |

## 核心规模

| 指标 | 数值 |
| --- | ---: |
| FDA 已批准小分子 | 915 |
| 人类蛋白靶点 | 1000 |
| ConPLex 预测 pair | 915000 |
| 疾病证据排序 pair | 915000 |
| DiffDock-ready pair | 913170 |
| 高优先级结构增强候选 | 1000 |
| 结构增强候选覆盖唯一药物 | 337 |
| 结构增强候选覆盖唯一蛋白 | 64 |
| 结构增强 Top100 覆盖唯一药物 | 67 |
| 结构增强 Top100 覆盖唯一蛋白 | 20 |

## 最新结构扩展审计

截至 2026-05-22 本地审计：

| 指标 | 数值 |
| --- | ---: |
| 结构扩展总 chunk | 3653 |
| 已生成 score 文件 | 995 |
| chunk 覆盖率 | 27.24% |
| DiffDock-ready pair 总数 | 913170 |
| 已评分 pair | 248750 |
| pair 覆盖率 | 27.24% |
| 已获得 DiffDock 输出 | 195497 |
| 缺失或失败输出 | 53253 |
| 已评分 pair 中输出率 | 78.59% |
| 0 completed chunk | 77 |

本次新增完成的结构扩展 chunk 为 993 和 994。993 属于 `CHEMBL6068045` 片段；994 横跨 `CHEMBL6068045` 和 `CHEMBL3184128`：

| Chunk | Rows | Completed | Missing output | Return code |
| ---: | ---: | ---: | ---: | ---: |
| 993 | 250 | 203 | 47 | 0 |
| 994 | 250 | 203 | 47 | 0 |

986-989 日志中出现大量 `mask rotate exception`，`CHEMBL4594262` 片段输出率显著偏低；该药物应进入后续失败输入审阅和参数补跑策略设计。990-994 进入 `CHEMBL6068045` 后输出率恢复到可用区间，说明异常主要集中在 `CHEMBL4594262` 片段。

审计时未检测到正在运行的结构扩展队列；两张 GPU 空闲，工作目录剩余磁盘空间约 31.8 GB。

## 证据模型解释

BioMaster 的排序对象是 drug-target pair，而不是唯一药物或唯一蛋白。不同证据源分别回答不同问题：

| 证据源 | 解释 |
| --- | --- |
| ConPLex | 药物是否可能与某个蛋白发生相互作用。 |
| Open Targets | 某个基因/蛋白是否与 cancer 疾病概念存在数据库关联。 |
| STRING | 靶点是否位于疾病相关蛋白互作网络中。 |
| TxGNN | 药物是否具有 cancer 方向的药物-疾病图谱信号。 |
| DiffDock | 高优先级候选是否具备可解释的结构姿态线索。 |

Open Targets 证据是靶点-疾病关联证据。它不表示所有蛋白都有 cancer-specific site，也不表示药物已经被证明对癌症有效。当前 disease scope 是 pan-cancer；如果后续聚焦具体癌种、突变或分子亚型，应切换到更具体的 disease ID 或加入突变/亚型数据重新排序。

## 候选收敛方法

当前结果可支持两类收敛：

1. 以 drug-target pair 为单位，直接使用疾病证据优先级或结构增强共识分数排序。
2. 以唯一药物为单位，从高到低扫描 pair 排名，每个药物只保留最高分代表靶点，再形成唯一药物候选清单。

第二种方式更适合用于讨论“优先验证哪些药物”；第一种方式更适合用于讨论具体靶点机制和实验读出。

## 结果使用方式

当前结果可用于候选筛选、文献审阅、机制假设生成和实验设计。进入实验前仍需结合具体癌种背景、药物可获得性、安全性、已知适应症、创新性和实验可操作性进行二次收敛。

## 相关产物

- 前端展示页面：`docs/index.html`
- Markdown 结果摘要：`docs/BIOMASTER_RESULTS_REPORT_2026_05_11_ZH.md`
- 正式英文 PDF：`docs/assets/biomaster-external-report.pdf`
- 主流程图：`docs/assets/biomaster-main-figure.png`
- 最新审计 JSON：`outputs/report_scale/biomaster_status_audit_2026_05_22.json`

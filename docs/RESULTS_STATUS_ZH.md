# BioMaster 当前结果状态

更新时间：2026-05-25

## 总体状态

当前 druggable-proteome 路线已完成本轮全量计算和结果合并：

| 阶段 | 状态 |
| --- | --- |
| FDA 已批准小分子整理 | 完成 |
| ChEMBL druggable protein 表解析 | 完成 |
| ConPLex 唯一序列亲和预测 | 完成 |
| 全量 drug-target 矩阵扩展 | 完成 |
| Top10000 候选选择 | 完成 |
| AlphaFold receptor 准备 | 完成 |
| DiffDock 代表结构对接 | 完成 |
| Stage 6 结果合并 | 完成 |
| 展示页面数据刷新 | 完成 |
| 正式英文 PDF | 完成 |

## 关键数字

| 指标 | 数值 |
| --- | ---: |
| FDA 已批准小分子 | 915 |
| 有效 ChEMBL druggable protein 记录 | 5306 |
| 唯一蛋白序列 | 891 |
| 唯一 drug-sequence 预测 | 815265 |
| 全量 drug-target pair | 4854990 |
| Top affinity candidates | 10000 |
| 结构代表候选 | 1872 |
| DiffDock chunks | 8 / 8 |
| 完成 rank-1 pose 的代表 | 1370 |
| 缺失或失败输出的代表 | 502 |
| 代表结构输出率 | 73.18% |
| Top10000 中映射到完成结构的行 | 7216 |

## 计算解释

本轮流程先用 ConPLex 对药物和蛋白序列进行亲和预测，再把高分候选按 drug + protein sequence 去重，避免相同序列重复进入结构对接。DiffDock 只作用于 Top10000 中的 1872 个结构代表，输出再映射回完整 Top10000 候选表。

UniProt accession 是蛋白记录编号，不是结合位点。不同药物面对同一蛋白仍分别对接；只有同一药物面对完全相同蛋白序列时才合并重复 accession。

缺失输出已被保留为审计状态。它们不被当作负样本，也不被静默删除；后续可根据候选价值决定是否补跑。

## 当前主要产物

| 产物 | 路径 |
| --- | --- |
| 展示首页 | `docs/index.html` |
| 正式结果页 | `docs/results-report.html` |
| 状态审计页 | `docs/status-report.html` |
| PDF 阅读页 | `docs/formal-report.html` |
| 正式 PDF | `docs/assets/biomaster-external-report.pdf` |
| Dashboard 数据 | `docs/assets/dashboard-data.js` |
| Stage 6 代表结果 | `outputs/druggable_proteome/stage6_druggable_top_unique_diffdock_consensus.csv` |
| Stage 6 Top10000 结果 | `outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.csv` |

## 后续安排

1. 从结构增强 Top100 中筛选 20-50 个候选 pair。
2. 对高价值 missing-output 候选评估是否需要补跑。
3. 若进入具体疾病方向，引入对应 Open Targets disease ID、癌种、突变或分子亚型上下文重新排序。
4. 选择 5-20 个候选进入文献审阅、正交对接或实验验证设计。

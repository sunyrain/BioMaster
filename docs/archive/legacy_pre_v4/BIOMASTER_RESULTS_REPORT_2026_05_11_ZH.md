# BioMaster Druggable Proteome 计算筛选结果汇报

更新时间：2026-05-25

## 项目概述

BioMaster 当前阶段建立了一条面向药物相关蛋白空间的计算筛选流程。输入不再是泛化的人类蛋白全集，而是 `druggable_proteome_chembl(1).xlsx` 中的 ChEMBL druggable target 表。这样可以把筛选范围收敛到具有药物相关性或历史靶点注释的蛋白集合，同时保留全量药物-蛋白亲和预测能力。

本轮结果是计算候选优先级，用于专家审阅、文献分析和实验设计，不构成药效或机制的实验证明。

## 已完成工作

| 模块 | 结果 |
| --- | --- |
| 药物库标准化 | 915 个 FDA 已批准小分子可用于筛选 |
| 蛋白库标准化 | 5306 个有效 ChEMBL druggable protein 记录 |
| 序列去重 | 5306 个蛋白记录压缩为 891 条唯一蛋白序列 |
| ConPLex 主筛选 | 完成 815265 个唯一 drug-sequence 预测 |
| 全量矩阵扩展 | 回填为 4854990 个 drug-target pair |
| Top 候选选择 | 选出 Top10000 亲和候选 |
| 结构代表选择 | Top10000 压缩为 1872 个 drug + protein-sequence 代表 |
| AlphaFold 受体准备 | 225 个代表蛋白均获得可用 receptor 结构 |
| DiffDock 对接 | 8 个分块全部完成，1370 个代表获得 rank-1 pose |
| Stage 6 合并 | DiffDock 结果已映射回代表表和 Top10000 候选表 |

## 流程逻辑

1. 整理 FDA 已批准小分子，统一 ChEMBL/PubChem 标识、SMILES、SDF 和药物名称。
2. 读取 ChEMBL druggable protein 表，保留 UniProt、gene symbol、protein name、sequence 和 receptor 路径。
3. 按蛋白序列去重，避免同一序列重复执行 ConPLex 推理。
4. 对 915 个药物和 891 条唯一序列执行 ConPLex 亲和预测。
5. 将唯一序列预测回填到 5306 个蛋白记录，形成 4854990 个 drug-target pair。
6. 按亲和分数选出 Top10000 候选。
7. 将 Top10000 按 drug + protein sequence 压缩为 1872 个结构代表。
8. 使用 AlphaFold receptor 和 DiffDock 对结构代表进行 rank-1 pose 预测。
9. 以 85% ConPLex 亲和分数和 15% DiffDock confidence 形成结构增强共识排序。

## 模型解释

| 模型或数据层 | 作用 |
| --- | --- |
| ConPLex | 从药物结构和蛋白序列预测潜在 drug-target affinity，是主筛选模型。 |
| ChEMBL druggable target 表 | 限定筛选范围，让候选蛋白具有药物相关性或靶点背景。 |
| AlphaFold | 为 Top 候选代表提供 receptor 结构，使结构对接可执行。 |
| DiffDock | 为高优先级候选生成 ligand pose 和 confidence，作为结构层面的补充证据。 |
| Open Targets | 当前不是主流程前置过滤。只有在聚焦具体疾病、癌种、突变或分子亚型时，才应作为 target-disease evidence 重新引入。 |

## 当前结果

| 指标 | 数值 |
| --- | ---: |
| 全量亲和矩阵 | 4854990 pairs |
| Top affinity candidates | 10000 |
| 结构代表候选 | 1872 |
| DiffDock score chunks | 8 / 8 |
| 完成 rank-1 pose 的代表 | 1370 |
| 缺失或失败输出的代表 | 502 |
| 代表结构输出率 | 73.18% |
| Top10000 中映射到完成结构的行 | 7216 |
| Top10000 中映射到缺失结构的行 | 2784 |

代表性已完成结构候选包括 Naltrexone--OPRK1、Dorzolamide--CA2、Naloxone--OPRK1、Cyclosporine--PPIA、Gefitinib--EGFR、Octreotide--SSTR2 等。这些结果包含已知药理关系和可解释的候选关系，可作为流程阳性对照、一致性证据和后续候选筛选入口。

这些可解释结果可以分成三类：第一类是已知药理关系回收，例如 opioid receptor、carbonic anhydrase、EGFR、cyclophilin A 和 somatostatin receptor；第二类是同家族扩展候选，例如 GPCR 或 kinase family 内的高分结果；第三类是高亲和但 missing-output 的补跑对象，例如 Doxazosin--ADRA1A、Afatinib--EGFR 和 Buprenorphine--OPRK1。

## 结果解释

当前排序对象是 drug-target pair。ConPLex 分数表示序列和药物层面的亲和优先级，DiffDock confidence 表示结构姿态层面的补充证据。两者合并后可以帮助从 Top 候选中优先挑选更值得审阅的对象。

UniProt accession 是蛋白数据库记录编号，不是结合位点。当前只合并同一药物面对完全相同蛋白序列的重复 accession；不同药物仍分别对接，同一药物面对不同蛋白序列也分别对接。若要研究突变位点或特定 pocket，需要引入对应突变结构或口袋约束后单独建模。

DiffDock 缺失输出并不等同于候选无效。它通常意味着该复合物在图构建、采样、结构输出或 ligand/receptor 准备中失败。此类候选保留在审计表中，可按价值决定是否补跑。

## 产物位置

| 产物 | 路径 |
| --- | --- |
| 交互展示首页 | `docs/index.html` |
| 正式 PDF | `docs/assets/biomaster-external-report.pdf` |
| Markdown 工作说明 | `docs/BIOMASTER_WORK_OVERVIEW_2026_05_23.md` |
| 全量亲和矩阵 | `outputs/druggable_proteome/conplex_affinity_scores_druggable.csv` |
| Top10000 候选 | `outputs/druggable_proteome/stage4_affinity_candidates_druggable_top10000.csv` |
| Stage 6 代表结果 | `outputs/druggable_proteome/stage6_druggable_top_unique_diffdock_consensus.csv` |
| Stage 6 Top10000 回填结果 | `outputs/druggable_proteome/stage6_druggable_top10000_with_diffdock.csv` |

## 下一步

建议从结构增强 Top100 中选择 20-50 个候选进入专家审阅，优先考虑亲和分数高、结构输出完成、靶点生物学可解释、药物可获得且实验可操作的 pair。之后可进一步收敛到 5-20 个候选，用于文献复核、正交对接或实验验证设计。

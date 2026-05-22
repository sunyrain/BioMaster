# BioMaster 药物再定位计算筛选结果摘要

更新时间：2026-05-22  
适用范围：FDA 已批准小分子与人类蛋白靶点的 cancer 方向计算筛选结果展示

## 项目目标

BioMaster 旨在建立一个可复现的药物再定位计算筛选流程，从已获批小分子药物中发现值得进一步机制分析和实验验证的药物-靶点组合。本阶段以 pan-cancer 为疾病方向，将药物-蛋白互作预测、靶点疾病证据、药物-疾病图谱信号和结构对接证据整合为候选优先级。

该流程输出的是计算候选优先级，不是药效或机制的实验证明。其价值在于把 915000 个 drug-target pair 收敛为更适合文献审阅、机制推断和实验设计的候选集合。

## 已完成工作

| 模块 | 阶段性结果 |
| --- | --- |
| 药物库构建 | 整理 915 个 FDA 已批准小分子，统一 ChEMBL/PubChem 标识、SMILES、InChIKey、分子式和结构文件。 |
| 蛋白靶点库构建 | 构建 1000 个人类蛋白靶点库，保留 UniProt accession、gene symbol、蛋白名称、序列和可用 AlphaFold 结构信息。 |
| AI-DTI 主筛选 | 对 915 个药物和 1000 个蛋白进行全组合筛选，完成 915000 个 drug-target pair 的 ConPLex 预测。 |
| 疾病证据融合 | 使用 Open Targets 和 STRING 引入靶点-疾病与蛋白网络证据，形成 cancer 方向靶点相关性优先级。 |
| 药物-疾病证据补充 | 使用 TxGNN 补充药物-疾病图谱信号；未映射药物保留靶点侧证据，不作为负证据。 |
| 结构增强 | 对高优先级候选进行 DiffDock 结构对接增强，形成结构增强共识排序。 |

## 流程设计思路

本阶段采用“广覆盖初筛、疾病证据融合、结构增强收敛”的分层策略。

1. 先进行广覆盖 AI-DTI 筛选，避免过早依赖人工选择靶点。
2. 再加入 cancer 方向疾病证据，将单纯互作可能性转化为疾病相关候选优先级。
3. 对高优先级候选加入结构对接信息，使结果具备进一步机制解释和实验设计的基础。

不同证据源回答的问题不同：

| 证据源 | 回答的问题 |
| --- | --- |
| ConPLex | 药物是否可能与某个蛋白发生相互作用。 |
| Open Targets | 某个基因/蛋白是否与 cancer 疾病概念存在数据库关联。 |
| STRING | 靶点是否位于疾病相关蛋白互作网络中。 |
| TxGNN | 药物是否具有 cancer 方向的图谱推断信号。 |
| DiffDock | 高优先级候选是否具备可解释的结构姿态线索。 |

## 排序方法

疾病证据优先级的排序对象是 drug-target pair。Open Targets 与 STRING 融合分数为：

```text
OpenTargets_STRING_priority
= 0.55 * normalized_combined_ai_score
+ 0.30 * OpenTargets_direct_score
+ 0.15 * STRING_network_score
```

TxGNN 映射成功时，最终疾病证据优先级分数为：

```text
final_priority_score
= 0.80 * OpenTargets_STRING_priority
+ 0.20 * TxGNN_indication_score
```

结构增强共识排序在疾病证据优先级基础上加入标准化后的 DiffDock confidence：

```text
Structure_enhanced_consensus
= 0.85 * disease_priority_score
+ 0.15 * normalized_DiffDock_confidence
```

这种设计以 AI-DTI 和疾病证据为主体，以结构对接作为高优先级候选的补充解释证据。

## 结果概览

| 指标 | 数值 |
| --- | ---: |
| FDA 已批准小分子 | 915 |
| 人类蛋白靶点 | 1000 |
| ConPLex 预测 pair | 915000 |
| 疾病证据排序 pair | 915000 |
| 结构增强候选 pair | 1000 |
| 结构增强候选覆盖唯一药物 | 337 |
| 结构增强候选覆盖唯一蛋白 | 64 |
| 结构增强 Top100 覆盖唯一药物 | 67 |
| 结构增强 Top100 覆盖唯一蛋白 | 20 |

疾病证据覆盖如下：

| 证据状态 | pair 数 |
| --- | ---: |
| Open Targets direct + STRING network | 666120 |
| Open Targets direct only | 234240 |
| STRING network only | 915 |
| none | 13725 |

结构增强 Top1000 候选均具有 direct + network disease evidence，其中 660 个 pair 对应 TxGNN 映射药物，340 个 pair 对应未映射药物。

## 代表性候选

| Consensus rank | 药物 | 靶点 | 疾病证据优先级分数 | DiffDock confidence | 结构增强共识分数 |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Afatinib Dimaleate | EGFR | 0.926857684 | -0.46 | 0.917978844 |
| 2 | Dacomitinib | EGFR | 0.889965753 | -1.12 | 0.874261152 |
| 3 | Cabozantinib S-Malate | KIT | 0.864187850 | NA | 0.864187850 |
| 4 | Momelotinib Dihydrochloride | EGFR | 0.861371239 | -0.75 | 0.856884654 |
| 5 | Repotrectinib | JAK1 | 0.860247310 | -1.02 | 0.850873135 |
| 6 | Pazopanib Hydrochloride | KIT | 0.901545965 | -3.00 | 0.848898340 |
| 7 | Osimertinib Mesylate | EGFR | 0.867670073 | -1.98 | 0.839204955 |
| 8 | Trilaciclib Dihydrochloride | JAK1 | 0.797277260 | 0.60 | 0.827685671 |
| 9 | Neratinib Maleate | EGFR | 0.841781555 | -2.44 | 0.808585483 |
| 10 | Tucatinib | EGFR | 0.845577840 | -2.70 | 0.806943411 |

这些候选体现了多源证据融合后的计算优先级。前列结果集中于 EGFR、KIT、JAK1 等具有较强 cancer 相关证据的靶点，符合本阶段的 disease-oriented screening 设计。

## Open Targets 结果解释

Open Targets 对所有输入蛋白都可以尝试查询疾病关联，但并不是所有蛋白都会获得同等强度的 cancer 证据。这里的 Open Targets 分数表示基因/蛋白与 cancer 疾病概念之间的数据库关联强度，是靶点层面的疾病证据。

这与“癌症突变位点”不是同一层概念。某些 cancer 位点只存在于特定蛋白、特定癌种或特定突变背景中；而本阶段的 Open Targets 证据用于回答的是“这个靶点与 cancer 是否相关”。因此，BioMaster 当前流程先在 pan-cancer 层面建立广覆盖候选池，后续若聚焦具体癌种、突变或分子亚型，应切换到更具体的 disease ID 或引入突变/亚型数据重新排序。

## 结果使用方式

当前结果可用于：

1. 从大规模 drug-target pair 中筛选高优先级候选；
2. 选择值得进行文献审阅和机制分析的药物-靶点组合；
3. 结合结构姿态设计靶点结合、通路响应、细胞活性和剂量反应实验；
4. 为后续特定癌种或分子亚型分析提供候选基础。

进入实验前仍需结合具体癌种背景、药物可获得性、安全性、已知适应症、创新性和实验可操作性进行二次收敛。

## 后续工作

1. 将 disease scope 从 pan-cancer 进一步收敛到具体癌种、分子亚型或突变背景；
2. 从结构增强 Top100 中筛选 20-50 个候选 pair 进行人工审阅；
3. 结合文献、结构姿态、靶点功能和药物安全性选择 5-20 个验证对象；
4. 设计靶点结合、通路响应、细胞表型和剂量反应实验；
5. 将实验反馈纳入下一轮候选优先级模型更新。

## 相关产物

- 前端展示页面：`docs/index.html`
- 正式英文 PDF：`docs/assets/biomaster-external-report.pdf`
- GPT-image-2 主图：`docs/assets/biomaster-main-figure.png`
- 结构增强共识候选：`outputs/report_scale/stage6_top1000_consensus_candidates.csv`

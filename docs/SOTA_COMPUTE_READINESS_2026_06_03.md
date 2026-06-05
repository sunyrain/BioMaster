# BioMaster SOTA 对标计算审计

生成时间：2026-06-05T04:11:08Z

## 当前基础

- FDA 小分子：915
- Druggable protein records：5306
- 全量 ConPLex drug-target pairs：4854990
- 五个疾病方向 Top 候选：50000
- DiffDock-ready 结构代表：23744
- 已完成结构输出：23707，剩余缺失：37，完成率：99.84%
- Top1000 KG explanation path 覆盖：3810 / 3921，97.17%
- ADMET/再利用审计：915 个药物，3921 条候选记录
- Pose sanity：23051 / 23060 个已完成姿态可读，99.96%

## 已知关系召回

- 已知 target records in scope：589
- Top10000 target-record recall：40 hits，6.79%
- Top100000 target-record recall：115 hits，19.52%
- 随机期望 Top10000 百分比约：0.21%
- Pair-level benchmark：4854990 pairs，positive pairs 892，AUROC 0.669，AUPRC 0.00197
- Pair Recall@10000：49 hits，5.49% recall，enrichment 26.67x
- Pair Recall@100000：147 hits，16.48% recall，enrichment 8.00x
- Record-level stratified Recall@100000：19.52%；Recall@1000000：52.63%

## 新增计算闭环

- KG explainability：生成 28774 条候选证据路径，直接 drug-target 边 288 条，target-disease 直接关联 968 条。
- ADMET/safety：药物 ADMET tier 分布 {'A': 610, 'B': 82, 'D': 165, 'C': 58}；候选短表已排除明显 PAINS/诊断类/结构缺失记录。
- Pose sanity：基础几何通过 15983 条，warning 7068 条，fail 9 条。

## SOTA readiness matrix

| 维度 | 当前状态 | 下一优先级 | 说明 |
|---|---:|---:|---|
| AI-DTI affinity screen | 已完成 | done | ConPLex 全量 drug-target pair 亲和矩阵已完成。 |
| Disease-aware ranking | 已完成/待校准 | P0 | 五个疾病方向排序已完成，并补入 Open Targets 与多方向 TxGNN 外部证据。 |
| Structural docking | 已完成/待共识重评分 | P2 | DiffDock 代表任务完成率已达 99.84%；pose sanity 可读率 99.96%，仍可补 GNINA/Vina 共识重评分。 |
| KG explainability | 已完成 | P1 | 候选级 TxGNN KG 浅层路径已完成，Top1000 层路径覆盖 97.17%。 |
| Known-positive validation | 已完成/待外推增强 | P0 | 已有 FDA known-target AUROC/AUPRC、Recall@K、随机富集和批准年份/治疗领域分层验证。 |
| Expression reversal | 未完成 | P1 | 缺少 LINCS/CMap disease signature reversal。 |
| ADMET/safety | 已完成/待深度模型增强 | P1 | 已完成 RDKit/PAINS/Brenk/route/text-flag 透明审计，覆盖 915 个药物。 |
| Tissue/disease context | 未完成 | P2 | 缺少 HPA/GTEx/DepMap 等上下文过滤。 |
| Second-model structure check | 未完成 | P2 | 缺少 AF3/Boltz/Chai-style 小样本结构对照。 |

## 疾病方向证据覆盖

| 方向 | 候选数 | 结构完成率 | Open Targets 覆盖 | TxGNN 覆盖 | 多源证据覆盖 |
|---|---:|---:|---:|---:|---:|
| 肿瘤 | 5562 | 99.73% | 41.93% | 93.04% | 99.96% |
| 感染性疾病 | 4995 | 99.90% | 14.73% | 95.40% | 94.91% |
| 心血管 | 4937 | 99.72% | 40.79% | 92.61% | 88.58% |
| 神经/精神 | 3341 | 99.91% | 43.25% | 94.19% | 68.60% |
| 免疫/炎症 | 4909 | 100.00% | 39.19% | 95.11% | 93.62% |

## 下一步计算队列

| 优先级 | 模块 | 计算资源 | 规模 | 产物 |
|---|---|---|---|---|
| P0 | 疾病方向证据矩阵扩展审计（已完成） | CPU / network-data | top1000 rows still missing at least one expanded evidence layer: 2421 | per-direction evidence gap list, rescued target-disease support, final evidence coverage table |
| P0 | 已知关系召回、负例基准与分层验证（已完成） | CPU | current known target records in scope: 589 | Recall@K, enrichment over random, AUROC/AUPRC, approval-year/therapeutic-area stratified recall |
| P1 | TxGNN / KG evidence path expansion（已完成） | CPU/GPU optional | top1000 candidate rows with KG path: 3810 / 3921 | drug-disease and target-disease explanation paths, graph-neighborhood support, missing mapping audit |
| P1 | ADMET / safety / repurposability filter（已完成） | CPU | 915 drugs; 3921 candidate rows | RDKit descriptor tiers, PAINS/Brenk alerts, route feasibility, label-text safety flags, translational shortlist |
| P1 | LINCS/CMap disease signature reversal | CPU | start with 5 directions x top 100 drugs | per-drug connectivity score, direction-specific expression reversal support |
| P2 | 结构共识重评分与 pose sanity checks（轻量几何审计已完成） | GPU/CPU mixed | readable poses: 23051 / 23060 | GNINA/Vina rescoring, PoseBusters-like geometry audit, pocket consistency, known-ligand pocket alignment |
| P2 | AF3/Boltz/Chai-style complex prediction spot-check | GPU | 50-100 complexes | second-model protein-ligand complex plausibility, confidence comparison, disagreement flags |
| P2 | 组织表达、DepMap 与疾病上下文过滤 | CPU | unique target genes from candidate shortlist | tissue expression support, cancer dependency support, disease-context mismatch flags |
| P3 | 剩余 missing-output 定向补跑 | GPU | 37 representatives after final rescue | rank-1 pose recovery or definitive failure tags |

## 建议执行顺序

1. Open Targets、TxGNN 多方向证据、KG explanation path、ADMET 透明审计、known-target 分层验证和 pose sanity 已完成。
2. 继续补 LINCS/CMap disease signature reversal 和 HPA/GTEx/DepMap 上下文过滤，需要外部数据文件接入。
3. 结构层下一步是安装 GNINA/Vina/obabel 或接入容器，再做共识重评分；Top10-20 可进入 AF3/Boltz/Chai-style 小样本验证。

## 参考 SOTA 口径

- TxGNN：知识图谱和 zero-shot drug repurposing 强调 drug-disease/target-disease explanation path。
- DiffDock：扩散式 ligand pose 生成，适合作为结构姿态线索，但需要几何与二次评分审计。
- AlphaFold 3 / Boltz / Chai-style models：代表更强的复合物结构预测方向，适合小样本高优先级候选验证。
- Open Targets：target-disease evidence 和 association score 是疾病方向靶点优先级的标准证据层。
- TDC/ADMET 与 CMap/LINCS：分别补安全性可转化与表达反转证据。

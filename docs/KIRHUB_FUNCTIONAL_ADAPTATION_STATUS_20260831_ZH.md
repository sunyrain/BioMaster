# KIRHub 功能抑制适配：当前状态与下一步

> **状态：暂停/非当前系统（2026-09-01）。** 本文仅保留为历史研究和复现记录。KIRHub功能适配头已从当前生产合同和对外汇报范围撤下；KIRHub 1 μM数据仅保留为通用模型的回顾性功能迁移审计。

## 1. 先回答“嵌入基座是不是已经预训练”

是。当前这条实验线没有重新预训练药物或蛋白基座，也没有把 KIRHub 标签灌入 BerMol 或 ESM2：

- 药物输入：冻结的 BerMol 768 维预训练表示；
- 蛋白输入：冻结的 ESM2-650M 1280 维预训练表示；
- 新训练部分：小型双投影 pair interaction adapter；
- 新任务：1 μM HotSpot 残余激酶活性/功能抑制排序；
- 原通用 direct-binding 头：保持冻结，不受本轮训练影响。

因此，“用 KIRHub 继续训练”准确地说是：用新的密集功能实验标签训练任务适配头，而不是重新训练嵌入基座。

## 2. 数据从局部交集扩展到严格主集

KIRHub WT 表共有 92 个药物 × 409 个实验构建体，即 37,628 个矩阵位置；其中 36,593 个位置有实际数值，1,035 个缺失位置继续保留为 unknown，未当作阴性。

第一轮仅使用现成 720×384 部署交集：

- 79 个药物、103 个项目靶点、8,058 个测量对；
- 排除一个基因映射到多个实验构建体的行后，严格主集为 7,505 对、79 个药、96 个靶点、987 个 ≥70% 抑制对。

随后完成了全缓存补齐与缺失嵌入生成：

- 药物：79 个直接复用 720 药物缓存，8 个复用完整 ChEMBL37 缓存，剩余 5 个按 PubChem 结构使用同一冻结 BerMol 推理，最终覆盖 92/92；
- 靶点：375 个构建体带补充表明确 UniProt accession；修正 3 个旧 accession/笔误后，排除同一蛋白在多个复合物构建体中重复出现的冲突，得到 345 个无歧义靶点；
- 蛋白特征：92 个复用项目缓存，10 个复用完整 ChEMBL37 缓存，243 个用同一冻结 ESM2 checkpoint 推理；
- 最终严格训练表：30,973 个实测对、92 个药、345 个靶点、2,306 个 ≥70% 抑制对。

变体/融合蛋白未并入 WT 输入。没有真实突变或构建体序列时，把它们并到 WT 序列会造成“相同输入、冲突标签”。

## 3. 模型与训练方式

```text
BerMol768（冻结） ─→ 药物投影 ─┐
                              ├─ [d, t, d×t, |d−t|] ─→ 功能抑制分数
ESM2-1280（冻结） ─→ 靶点投影 ─┘
```

联合损失包括：

1. 连续抑制比例 SmoothL1，保留 1 μM 实验的强弱梯度；
2. ≥70% 抑制二分类 BCE，仅作辅助；
3. 同一药物内高抑制靶点相对低抑制靶点的 pairwise ranking loss。

防过拟合措施：

- 5 折 generic Bemis–Murcko scaffold cold，不允许同骨架跨训练/测试；
- 每个测试折使用另一个完整药物折早停；
- dropout 0.18、AdamW weight decay、梯度裁剪；
- 训练上限 600 epoch，但 5 折均由 14 次验证无提升早停结束；
- 5 折最佳/终止 epoch 分别为 75/145、270/340、135/205、190/260、240/310。

## 4. 当前结果

### 4.1 扩展 KIRHub 内部 OOF

在 30,973 对、92 个药、345 个靶点的药物骨架冷启动 OOF 上，独立功能适配头得到：

| 指标 | 数值 |
|---|---:|
| Micro AUROC | 0.7202 |
| Micro AUPRC | 0.1904 |
| 药物宏 AUPRC | 0.3240 |
| 药物宏 AUROC | 0.8054 |
| 连续 NDCG@20 | 0.4504 |
| Recall@20 | 0.2908 |

这里每个药面对约 345 个实测靶点，不能与下表约 93 个候选的 Recall@K 直接横向比较。

### 4.2 与冻结 BioMaster、DTIAM 的同场比较

共同可评估范围为 7,269 对、79 个药、93 个靶点、985 个强抑制对。融合权重只在验证折选择，测试药物折不可见。

| 指标 | KIRHub 功能融合 | 冻结 BioMaster drug→target | 冻结 DTIAM |
|---|---:|---:|---:|
| Micro AUROC | 0.7603 | **0.7614** | 0.6554 |
| Micro AUPRC | 0.4130 | **0.4490** | 0.2823 |
| 药物宏 AUPRC | **0.5619** | 0.5475 | 0.4698 |
| 药物宏 AUROC | **0.8170** | 0.7852 | 0.7288 |
| 连续 NDCG@20 | **0.6369** | 0.6189 | 0.5670 |
| Recall@5 | **0.3347** | 0.3341 | 0.2633 |
| Recall@10 | **0.4699** | 0.4442 | 0.3569 |
| Recall@20 | **0.6196** | 0.6086 | 0.5306 |

结论分两层：

- 对 DTIAM：药物宏 AUPRC、NDCG@20、Recall@20 的药物级 bootstrap 95% 区间均大于 0，当前 drug→target 排序在该内部 KIRHub 开发范围内明确更好；
- 对旧 BioMaster：宏 AUPRC、Recall@5/10/20 和 NDCG 均有点估计提升，但只有 NDCG@20 的 95% 区间大于 0；Micro AUPRC 仍低于旧头，所以不能称为全面替换。

## 5. 当前决策

1. 保留冻结的通用 BioMaster drug→target 头作为生产基线；
2. 将 KIRHub 适配结果保留为“激酶 1 μM 功能抑制候选融合头”；
3. 非激酶及不在功能路由范围的查询继续走原模型，原 S1–S5 结果不因独立路由而变化；
4. KIRHub 已成为训练/开发数据，不能再把它描述成适配后模型的 untouched external test；
5. 在新的未接触密集功能面板确认前，不将该融合头升级为通用生产头。

已额外生成 kinase-only 的全量候选应用包：使用 5 折最佳 epoch 中位数 190，训练 3 个随机种子并取均值，对 720 个老药 × 108 个项目激酶靶点共 77,760 对完成评分。固定融合权重取 5 个验证折所选权重的中位数：BioMaster 0.5、功能头 0.2、DTIAM 0.3。该包用于候选筛选，不作为性能证据；性能证据仍以上述 OOF 为准。

## 6. 从 DTIAM 借鉴了什么

- 真正启用 BerMol + ESM2 预训练表示，而不是只在文档中列出；
- 保留模型异质性：DTIAM 分数作为冻结互补证据，而不是覆盖 BioMaster；
- 让任务头与应用方向一致：训练和验证均以 drug→target 的同药物内排序为主；
- 模型选择指标从单纯 Micro AUROC 转为药物宏 AUPRC、连续 NDCG 与 Recall@K；
- 用验证折选择融合权重，避免在 KIRHub 测试折上事后调权。

## 7. 下一步最重要的实验

优先获取一个新的、未参与本轮设计和训练的密集激酶功能面板，保持 drug scaffold cold 评估。升级门槛至少应包括：

- 药物宏 AUPRC 与 Recall@20 相对冻结 BioMaster 的 95% CI 均大于 0；
- Recall@5 不退化；
- Micro AUPRC 非劣；
- 新面板不用于重新选融合权重。

若新面板确认，再用全部 KIRHub 训练一个全量功能头并固定单一生产融合配方；否则 KIRHub 仅作为辅助训练数据和激酶路由证据。

## 8. 主要产物

- 数据构建：`scripts/build_kirhub_expanded_pretrained_features_v1.py`
- 分组训练：`scripts/train_kirhub_expanded_functional_adapter_v1.py`
- 全量候选重训与 720×108 评分：`scripts/refit_and_score_kirhub_expanded_functional_adapter_v1.py`
- 数据摘要：`outputs/kirhub_functional_adaptation_v1/expanded_pretrained_features_v1/KIRHUB_EXPANDED_PRETRAINED_FEATURE_SUMMARY_V1.json`
- 训练摘要：`outputs/kirhub_functional_adaptation_v1/expanded_functional_adapter_v1/KIRHUB_EXPANDED_FUNCTIONAL_ADAPTER_SUMMARY_V1.json`
- 同场指标：`outputs/kirhub_functional_adaptation_v1/expanded_functional_adapter_v1/KIRHUB_EXPANDED_COMMON_SCOPE_METRICS_V1.csv`
- 药物级 bootstrap：`outputs/kirhub_functional_adaptation_v1/expanded_functional_adapter_v1/KIRHUB_EXPANDED_COMMON_SCOPE_BOOTSTRAP_V1.csv`
- 测试：`tests/test_kirhub_functional_adapter.py`
- kinase-only 候选排序：`outputs/kirhub_functional_adaptation_v1/expanded_full_fit_kinase_candidate_v1/KIRHUB_FULL_FIT_720X108_KINASE_CANDIDATE_V1.csv.gz`

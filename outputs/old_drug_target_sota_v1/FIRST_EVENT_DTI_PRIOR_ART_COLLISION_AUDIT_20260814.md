# 老药中心首次事件 DTI：prior-art 碰撞审计

生成日期：2026-08-14

## 裁决

“时间模型”“MNAR”“药物中心排序”“竞争风险”“双向检索”任一单项都不是新颖贡献。当前只允许把下列整体作为待证伪候选：

> 在严格的历史风险集中，将药物–靶点 pair 从尚未记录到首次测得活性或首次测得弱/无活性的过程建模为竞争事件；显式分离实验选择/记录 hazard 与条件活性结果，并以药物内未来靶点检索为主目标，同时让结构证据只产生资格受控增量、在无实验口袋时精确退回非结构函数。

这个联合对象尚未通过实验，也尚不能声称文献首次。

## 直接先例与差异

| 先例 | 已覆盖内容 | 对候选新颖性的限制 | 当前候选剩余差异 |
|---|---|---|---|
| FNML, IEEE TNB 2019, https://doi.org/10.1109/TNB.2019.2909293 | 联合建模 DTI 标签与非随机缺失/观测过程 | 不能声称首次解决 MNAR 或首次分离 label/response | FNML 是静态矩阵分解；候选使用首次记录时间、活性/弱活性竞争事件和分子/序列冷启动检索 |
| Bayesian ranking DTI, 2018, https://doi.org/10.1016/j.cmpb.2017.10.016 | 明确以每个药物的靶点排序服务重定位 | 不能声称首次药物中心 ranking | 候选排序的是未来事件累计发生概率，并维护逐时间风险集 |
| NeuRank, 2021 | 神经 pairwise/listwise DTI ranking | pairwise/listwise loss 不是创新 | 排序损失仅是事件似然的辅助任务 |
| TAPB, Nature Communications 2025, https://doi.org/10.1038/s41467-025-66915-1 | 靶点先验偏差、氨基酸随机化、backdoor adjustment | 一般因果去偏和 target-prior intervention 不是空白 | 候选针对实验选择/记录过程与时间风险集，不把 target prior 当作唯一混杂量 |
| KGOT, ICLR 2026 submission, https://openreview.net/forum?id=UoYdZQIZWj | mutual molecule–protein retrieval、optimal transport、严格双侧过滤 | 双向检索和 OT 不能声称新颖 | 候选不依赖 OT；主要估计 cause-specific first-event risk |
| TAMI, NeurIPS 2025 | 时间图链路预测、时间间隔编码和历史聚合 | 一般 temporal link prediction 不是创新 | 候选区分首次实验记录及其活性类型，并禁止将未记录 pair 当负例 |
| Future Link Prediction, NeurIPS 2025 | 未来边排序、target-aware matching | 未来 link ranking 不是创新 | 候选包含测量选择与条件结果的可审计分解以及 DTI 竞争事件 |
| Chemical-similarity network time split, 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC9455815/ | 用较早 ChEMBL 构图并以较新 ChEMBL 外部时间切分验证 | 时间外评估不是创新 | 候选把时间写入训练似然而非只用于 split |
| Open Targets temporal novelty, 2026, https://doi.org/10.1038/s41467-025-67180-y | 全量证据时间戳和靶点新颖度回顾分析 | “timestamped novelty” 不是空白 | 该工作不是小分子–蛋白条件结合事件模型 |
| PIGLET, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12934570/ | 蛋白口袋图、DrugBank 网络、严格药物相似性 split | 结构图和严格 drug split 不是空白 | 候选重点是历史观测机制与首次活性事件，不是新增 GNN 层 |

## 必须加入的控制

1. 静态 BCE：同编码器、同参数量、未记录 pair 不作负例。
2. 静态 FNML-style 双头：有观测头和活性头，但去掉时间与风险集。
3. 只做离散时间 survival、去掉活性/弱活性竞争事件。
4. 只做药物中心 listwise、去掉事件似然。
5. 打乱训练窗内时间戳；若仍有相同收益，则时间机制不成立。
6. 去掉实验选择 hazard；若无退化，则 observation-process 机制不成立。
7. 目标先验/药物先验控制，以及 TAPB-inspired 同容量去偏控制。
8. 在无结构资格 pair 上验证结构分支与基础函数逐位完全一致。

## 开发门槛

- 只用 2014 年及以前训练，预测 2015–2018 首次事件；再只用 2018 年及以前训练，预测 2019–2022 首次事件。
- 两个开发窗均需在老药/药物查询 macro AUPRC 或 Recall@20 上超过静态 BCE 与静态 FNML-style，药物聚类 bootstrap 95% CI 下界大于 0。
- target-macro AUPRC 相对最强控制非劣界为 `-0.005`。
- 完整模型必须同时胜过时间戳打乱和去 observation-hazard 消融。
- 2023–2025 结果只能称重复使用的时间外压力测试，不能称独立确认，因为 S4 标签和基线结果此前已经检查。
- 任何机制门槛失败，结论为 `PHASE_A_MECHANISM_NOT_SUPPORTED`；不得用最终时间窗或 720×384 部署排序反向调参。

## 数据可行性

冻结 428 个训练靶点的全量未截断 ChEMBL37 严格 binding pair：509,168 行、336,316 个化合物；排除冲突、重复和无年份后有 419,642 个候选首次事件。

- 2015–2018：130,152 个事件，99,164 活性、30,988 弱/无活性；
- 2019–2022：121,180 个事件，94,659 活性、26,521 弱/无活性；
- 2023–2025：35,152 个事件，27,733 活性、7,419 弱/无活性，仅作非独立压力测试。

机器可读证据：`outputs/old_drug_target_sota_v1/first_event_dti_feasibility_v1/FIRST_EVENT_DTI_FEASIBILITY_SUMMARY_V1.json`。

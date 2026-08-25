# BioMaster 研究主线与展示案例收束（V5）

快照日期：2026-08-22

## 一、收束后的核心结论

BioMaster 当前最有依据、也最适合继续推进的任务，不是“任意新蛋白上的双冷启动预测”，而是：

> 在固定的 720 种已上市/老药中，针对已有训练关系的目标蛋白，检索“药物和靶点分别见过、但该药物–靶点关系没有作为正例进入训练”的新关系，并用结构计算和实验逐级验证。

这是一条 **double-warm、relation-held-out 的老药新靶点关系发现线**。当前生产输出仍对 720 × 384 全矩阵打分，但对外主张必须再应用 warm-entity mask；720 种药物中目前有 498 种在综合训练关系中至少出现过，另外 222 种仍属于 drug-cold，不能混入 double-warm 成功率。

截至本快照：

- 已有一个可作为主展示的外部关系留出命中：**Lorlatinib–PLK4，4/720**。
- 有一个只能作为次级支持的案例：**Pitavastatin–IDO1，146/720**。
- 有一组很强的内部关系留出检索对照：21 个正例中位排名 6/720，Top 36 命中率 71.4%，16 个负例无一进入 Top 36。
- Tepotinib–IRAK1/4、SEPHS2、VPS37C 和 SIRT3 案例没有被主排序正确召回，必须作为边界或失败案例报告。
- 实验 holo pocket 局部图出现了很小的正信号，但置信区间跨 0，double-warm 子集也未达晋级门槛，因此尚未进入生产 checkpoint。

一句话概括当前状态：**我们已经证明模型在熟悉的药物和靶点之间能够检索一部分未见关系，但尚未证明它能够可靠解决新靶点冷启动；结构分支目前更适合作为候选后的正交复核，而不是直接替换主排序。**

## 二、这一线的研究主线

### 2.1 科学问题

当前主问题是：

> 给定一个目标蛋白，从 720 种老药中找出值得进一步验证的潜在新作用药物。

其中最现实、证据最充分的部署范围是：

- 药物实体在训练关系中见过；
- 靶点实体在训练关系中见过；
- 待预测的 exact drug–target pair 没有作为正例训练；
- 排名在每个靶点内部独立完成，不能把不同靶点间的分数解释为可比的结合概率。

“药物是否上市”与“药物是否在 DTI 训练集中出现”不是同一件事。720 是药物筛选库的业务范围，498/720 才是当前关系数据库中的 warm drugs。冷启动是相对于某个训练 checkpoint 定义的，而不是相对于年份或上市状态定义的。

### 2.2 研究路径

当前主线可以收束为以下闭环：

```text
全面、去重、带正负标签的 DTI 数据
  → 药物与蛋白表示学习
  → binary + affinity/ranking 多任务交互模型
  → 多种子、多模型族共识排序
  → 时间留出与关系留出检验
  → double-warm 候选筛选
  → pair-specific 结构复核
  → 生化/细胞实验
```

这里最重要的策略变化有三点：

1. **从每靶点截取少量样本，转为综合数据训练。** 当前不再用“每个靶点各取 150 条”构造生产训练集；保留所有可用的标准化关系，再用 target-balanced sampler 控制高频靶点支配训练。
2. **从单一分数，转为关系检索。** 评价单位是“已知正关系在同靶点 720 种药中的排名”，而不只是随机 pair 分类 AUPRC。
3. **从盲目加入结构，转为分阶段结构证据。** target-level 结构上下文只能提供受体先验；真正能区分具体药物的结构信息必须来自 candidate ligand 与 pocket 的共同构象、距离和接触。

## 三、当前生产架构

当前正式结果来自 V5 六 checkpoint 共识，而不是早期 ConPLex/DiffDock 全蛋白筛选线。

| 模块 | 当前实现 |
|---|---|
| 药物输入 | Morgan fingerprint，2048 维 |
| 蛋白输入 | ProtBERT pooled 1024 维 + ESM2 pooled 1280 维 |
| 交互 | low-rank FiLM，rank 48，FiLM scale 0.10 |
| 专家路由 | 6-expert family MoE，family embedding 24 维 |
| 主隐空间 | embedding 192，hidden 256，dropout 0.12 |
| 训练目标 | binary classification + target-side ranking + affinity + affinity ranking + contrastive auxiliary losses |
| 结构输入 | 19 维 target-level 结构/口袋上下文，masked residual |
| 参数量 | 每个 checkpoint 3,517,940 |
| 集成 | conservative 与 recall 两个模型族，各 3 个 seeds，等权共识 |
| 最终融合 | binary 先取 Top 20%，短名单内 binary 0.55 + affinity 0.45 重排 |

需要明确：

- **当前生产模型没有启用 ConPLex。** `conplex_enabled=false`。
- **当前生产模型没有 ligand–pocket 局部图、candidate pose 或 token branch。**
- 19 维结构特征在同一 target 的 720 种药之间基本是 target-constant，只能调整靶点上下文，不能单独识别某个药物是否适配口袋。
- `structure_mask=0` 时严格回退到非结构 logit。当前 276,480 个部署 pair 中，182,880 个有 19D 结构输入，对应 254 个 target；93,600 个没有结构输入，对应 130 个 target。

六个 FULL_FIT checkpoint 的选定 epoch 分别为：

- conservative：9、8、11；
- recall：11、8、6。

epoch 由 development early stopping 决定，不是简单把训练无限延长到训练损失最低。当前外部案例标签没有参与 checkpoint、fusion 或权重选择。

## 四、训练数据与评估设计

### 4.1 FULL_FIT 训练数据

| 项目 | 数量 |
|---|---:|
| ChEMBL 37 原始候选关系 | 509,172 |
| 标签规则后、特征过滤前 | 427,526 |
| 成功解析特征的 ChEMBL 关系 | 426,939 |
| 其中正例 | 331,135 |
| 其中负例 | 95,804 |
| BindingDB direct Ki/Kd rows | 9,778 |
| recovered rows | 914 |
| 最终去重 FULL_FIT rows | 437,248 |
| FULL_FIT unique drugs | 296,108 |
| FULL_FIT unique targets | 843 |
| 720 药库中有任意训练关系的药物 | 498 |
| 720 药库中有正训练关系的药物 | 346 |

这里的 BindingDB 主要提供连续 affinity 与强弱排序约束；ChEMBL 的正负关系仍是 binary 主任务的主体。recovered 数据用于补充原先缺少关系的部署实体，同时保持重复关系审计。

### 4.2 为什么既有 FULL_FIT，又有“截至 2022”的模型

这是两个不同用途：

- **FULL_FIT checkpoint** 使用当前可用的综合数据，负责最后的 720 × 384 排序。
- **evaluation checkpoint** 只使用截至 2022 年的数据，用于测试 2023 development 和 2024–2025 frozen test。

因此，同一个关系在“截至 2022 的评价模型”中可能是 single-cold 或 double-cold，但在吸收了后续数据的 FULL_FIT 中可能已经变成 double-warm。冷启动标签必须绑定具体 checkpoint，不能只说“模型截至 2026，所以不应该冷”。

### 4.3 正式评估结果

| 评估 | 结果 | 正确解释 |
|---|---|---|
| 2023 temporal development | 18 个正查询；中位 rank 41；Top 36 命中率 44.4% | 用于选择 fusion，不能当独立测试 |
| 2024–2025 strict temporal test | 27 个正查询；中位 rank 120；Top 36 命中率 22.2%；Top 72 为 37.0% | 独立时间测试，表现中等偏弱，不能包装成强成功 |
| recovered double-warm test | 37 relations，21 正/16 负；AUPRC 0.933，AUROC 0.881 | 当前最有力的内部关系留出证据 |
| recovered 正例检索 | 中位 rank 6；15/21 进入 Top 36 | 支持 double-warm relation retrieval |
| recovered 负例检索 | 中位 rank 416；0/16 进入 Top 36 | 支持高位候选并非简单由药物/靶点热度造成 |
| 全关系 coverage audit | 5,992 rows；AUPRC 0.895，AUROC 0.969 | 包含 in-fit/capped/newly-added 数据，只作覆盖审计，不作独立泛化证据 |

这些结果共同说明：模型对部分 double-warm 新关系有可用排序能力，但时间外推仍不稳定。全面补数据后，已知关系覆盖和内部留出改善，并没有自动转化成所有近年新关系都能排到前列；药理分布偏移、负标签冲突和缺少 pair-specific 几何仍是主要限制。

## 五、目前可用的展示例子

### 5.1 A 级：主展示案例

#### Lorlatinib–PLK4

| 项目 | 结果 |
|---|---|
| V5 排名 | **4/720，Top 0.56%** |
| 问题类型 | 双方实体均见过，exact relation 未进入 FULL_FIT 关系表 |
| conservative / recall binary rank | 4 / 6 |
| 外部证据 | 2024 年化学蛋白组研究列出 PLK4 为 lorlatinib 候选靶点，报告 Ki 0.1074 μM；2025 年神经母细胞瘤研究观察到 PLK4 的 MIB binding 随 lorlatinib 处理下降 |
| 结构诊断 | 旧 3COK receptor 的 GNINA control calibration 未通过，不能当结构确认；应使用最终 strict PLK4 template 重新计算 |
| 展示等级 | **当前唯一合适的外部关系留出主案例** |

原始文献：[Cell Chemical Biology 2024](https://doi.org/10.1016/j.chembiol.2023.09.011)，[Molecular Cancer Therapeutics 2025](https://doi.org/10.1158/1535-7163.MCT-24-0684)。

建议表述：

> 在固定的 720 种老药中，BioMaster 将 Lorlatinib–PLK4 排在第 4。药物和靶点分别存在于训练数据中，但该 exact pair 没有作为训练关系；近年的独立化学蛋白组和功能 kinome 研究提供了外部支持。

不能表述为“模型在论文发表前做出了前瞻预测”，因为这是论文发表后进行的数据库关系留出回溯；它证明的是 **external relation-held-out rediscovery/generalization**。

### 5.2 B 级：次级支持案例

#### Pitavastatin–IDO1

| 项目 | 结果 |
|---|---|
| V5 排名 | 146/720，约 Top 20.3% |
| 问题类型 | 双方实体均见过，exact relation 未进入 FULL_FIT；药物缺少 positive-warm 关系 |
| conservative / recall binary rank | 181 / 107 |
| 外部证据 | 2024 年研究报告 pitavastatin 对 hIDO1 的体外 IC50 为 351 ± 16 nM，并有 docking、MD 和细胞实验 |
| GNINA 诊断 | target-control consensus 0.4375；优于 8/12 negatives，但仅优于 1/12 positives，属于有限支持 |
| 展示等级 | **可作次页补充，不应与 PLK4 并列为高排名命中** |

原始文献：[Scientific Reports / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11039737/)。

这个案例的价值是“关系确实有独立实验依据，而且模型没有彻底沉底”，但 146/720 不是有显示度的强命中。如果展示，必须把排名如实写出。

### 5.3 C 级：内部关系留出能力对照

以下关系来自 recovered double-warm test。它们的 exact relation 在 evaluation fit 中被留出；这里给出六个评价 checkpoint 的 rank 范围，用来展示跨模型族和 seeds 的稳定检索。它们是已知关系控制，不是新发现。

| 药物–靶点 | 六 checkpoint rank 范围 | 中位表现 | 用途 |
|---|---:|---:|---|
| Leniolisib–PIK3CB | 2–6 / 720 | 约 3 | 稳定 Top 1% 正对照 |
| Talazoparib–PARP2 | 1–8 / 720 | 约 3 | 稳定 Top 1% 正对照 |
| Osilodrostat–CYP11B1 | 1–9 / 720 | 1 | 稳定靶点内检索正对照 |

对外演示时，这一页的标题应是“关系留出检索控制”，而不是“新药靶发现”。它的作用是证明 Lorlatinib–PLK4 不是孤立的偶然分数，同时给出模型在受控 double-warm 场景下的基础检索能力。

## 六、应该展示、但必须作为反例的案例

### 6.1 Tepotinib–IRAK1/4：主排序与结构信号分裂

[Nature Biotechnology 2026](https://www.nature.com/articles/s41587-026-03090-8) 在 758 个 kinase 的功能筛选中报告 tepotinib 对 IRAK1 和 IRAK4 均为 72% inhibition，并进一步用 NanoBRET、CETSA、细胞和 PDX 证据验证了 repurposing 方向。

我们的 V5 排名却是：

| 关系 | V5 rank | 训练关系状态 | GNINA 诊断 | 结论 |
|---|---:|---|---|---|
| Tepotinib–IRAK1 | 658/720 | exact pair 存在，但没有 positive label | consensus 0.833；优于全部 negatives 和 10/12 positives | DTA 主排序明显漏检；pair-specific 结构信号很强 |
| Tepotinib–IRAK4 | 573/720 | exact pair 存在，但没有 positive label | consensus 0.417，证据一般 | DTA 与结构均未给出强支持 |

这两个案例不能算 BioMaster 成功，但非常适合作为“为什么不能只依赖 pooled sequence/target-level pocket、为什么负标签冲突会伤害召回”的机制页。尤其 IRAK1 显示：一个候选可以在序列 DTA 排序中失败，却在 target-calibrated 物理结构诊断中出现强信号。

### 6.2 真正新靶点冷启动的失败边界

| 外部关系 | rank | 类型 | 结论 |
|---|---:|---|---|
| Dasatinib–SEPHS2 | 395/720 | target-cold | 未进入预设 Top 5% |
| Hydroxychloroquine–VPS37C | 219/720 | target-cold | 未进入预设 Top 5% |
| Binimetinib–SIRT3 | 211/720 | target 已见、关系未见 | 未进入预设 Top 5% |
| Olaparib–SIRT3 | 290/720 | target 已见、关系未见 | 未进入预设 Top 5% |

SIRT3 的 2026 年 SPR 报告可见 [PubMed](https://pubmed.ncbi.nlm.nih.gov/42447746/)。这些结果说明当前模型不能宣称“新蛋白也能普遍做好”。它们应保留在报告中作为诚实边界，而不是删除后只展示成功样本。

## 七、实验 pocket 局部图试验的最终定位

本轮已经把实验 holo template 直接转换为 pocket residue graph：

- 338 个 strict experimental assets；
- 其中 326 个与旧 428-target benchmark 轴 exact sequence overlap；
- 16,117 个 pocket residues，target 中位 pocket size 50；
- residue ESM2、Cα 坐标、氨基酸/B-factor/序列位置，以及 24 个参考配体接触描述符；
- 不使用候选药物标签，也不使用候选药物 pose。

正式 paired ablation 结果：

| 指标 | baseline | experimental graph | Δ |
|---|---:|---:|---:|
| S3 two-seed ensemble micro-AUPRC | 0.640760 | 0.641773 | +0.001013 |
| S3 target-macro AUPRC | 0.661032 | 0.663862 | +0.002830 |
| S3 Brier | 0.242051 | 0.241583 | −0.000467 |
| S4 exact-entity double-warm AUPRC | 0.747870 | 0.747999 | +0.000129 |

但 target-cluster bootstrap 的 95% CI 为 −0.001826 到 +0.003711，跨越 0；S4 double-warm 子集的 Brier 还恶化了 0.000386。因此正式决定是：

> `KEEP_RESEARCH_CANDIDATE_DO_NOT_PROMOTE`

原因不是实验口袋“没有信息”，而是当前图使用的是 **reference ligand–pocket 模板信息**，仍没有 candidate drug 在同一坐标系中的 pose 和 cross-distance。它能更好地表示 pocket，却没有充分回答“这个具体药物是否适配这个 pocket”。

下一步结构研究应集中在已筛出的少量候选上生成 candidate pose，再提取：

- ligand atom–pocket residue 最小距离与 RBF；
- 氢键、盐桥、疏水、芳香接触；
- pocket occupancy、clash 和 buried surface；
- 多 pose 共识与 target-specific positive/negative control percentile。

这些 pair-specific 特征先作为独立重排器验证，不直接并入 FULL_FIT 主模型，直到在预注册的 double-warm external/heldout panel 上获得稳定、bootstrap CI 排除 0 的增益。

## 八、当前可以和不可以对外宣称什么

### 可以宣称

- 构建了面向 720 种老药和 384 个目标蛋白的综合 DTI 排序系统。
- 生产模型使用 437,248 条去重关系、双蛋白语言模型表示和六 checkpoint 共识。
- 在 recovered double-warm relation-held-out test 上，正例中位排名 6/720，Top 36 命中率 71.4%，负例 Top 36 为 0/16。
- Lorlatinib–PLK4 是当前最强的外部关系留出案例，排名 4/720，且 exact pair 不在 FULL_FIT 关系表。
- 模型对某些已报告关系会明显漏检；Tepotinib–IRAK1 表明 pair-specific 结构计算可能提供与 DTA 互补的信号。

### 不可以宣称

- 不能把 FULL_FIT score 解释为物理结合概率或跨靶点可比 affinity。
- 不能说模型已经可靠解决 target-cold 或 double-cold。
- 不能把 Lorlatinib–PLK4 描述为论文发表前的前瞻发现。
- 不能把 Pitavastatin–IDO1 的 146/720 描述为 Top hit。
- 不能把 Tepotinib–IRAK1/4 描述为模型成功；主模型分别排在 658 和 573。
- 不能说实验 pocket graph 已经优于或替代当前 champion。
- 不能说当前生产模型使用了 ConPLex、candidate pose 或 ligand–pocket cross-attention。

## 九、建议的展示组合

如果只做一个短展示，建议使用四页：

1. **任务与架构**：720 × 384、double-warm relation retrieval、六 checkpoint 共识。
2. **主命中**：Lorlatinib–PLK4，4/720，外部关系留出和近年文献证据。
3. **受控能力**：recovered heldout test 的整体结果，加 Leniolisib–PIK3CB、Talazoparib–PARP2、Osilodrostat–CYP11B1 三个内部正对照。
4. **边界与下一步**：Tepotinib–IRAK1 的 DTA 失败/结构强信号，以及 pair-specific pose 特征路线。

Pitavastatin–IDO1 可以放在附页；SEPHS2、VPS37C、SIRT3 和 IRAK4 放在完整技术报告的负结果表中。

之前生成的两案例简版 PDF 仍可用于内部讨论，但应按照本文件的证据等级理解：其中 PLK4 是主案例，IDO1 是次级案例，而不是两个等强成功案例。

## 十、下一阶段的最小研究议程

1. 冻结 V5 FULL_FIT checkpoint 和当前 720 × 384 排名，避免继续用外部案例反复调权。
2. 从全矩阵中严格筛出 entity-double-warm、exact-positive-pair-absent 的候选池；所有展示和湿实验候选只从该池产生。
3. 建立一个预注册的外部 double-warm panel，至少包含多靶点正例和 matched negatives；外部标签只用于一次性测试。
4. 对 Top 候选使用最终 strict receptor/template 运行 candidate-specific pose 与 target-calibrated controls。
5. 只有 pair-specific 结构重排在多 seeds、target/scaffold cluster bootstrap 和外部 panel 上稳定提高 Top-k retrieval 时，才进入下一版生产模型。
6. target-cold 与 double-cold 保留为独立研究支线，不再用其结果评价当前 double-warm 主线是否成功。

## 十一、关键产物

- 训练数据清单：[COMPREHENSIVE_TRAINING_MANIFEST_V1.json](../outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json)
- V5 共识摘要：[CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json](../outputs/biomaster_comprehensive_consensus_720x384_v5/CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json)
- FULL_FIT checkpoint 清单：[FULL_FIT_CHECKPOINT_MANIFEST_V5.json](../outputs/biomaster_comprehensive_consensus_720x384_v5/FULL_FIT_CHECKPOINT_MANIFEST_V5.json)
- V5 720 × 384 分数：[CONSENSUS_FULL_FIT_720X384_SCORES_V5.csv.gz](../outputs/biomaster_comprehensive_consensus_720x384_v5/CONSENSUS_FULL_FIT_720X384_SCORES_V5.csv.gz)
- 当前外部案例排名：[CURRENT_DOUBLE_WARM_CASE_RANKS_V5.csv](../outputs/biomaster_comprehensive_consensus_720x384_v5/CURRENT_DOUBLE_WARM_CASE_RANKS_V5.csv)
- 时间/关系留出评估：[FUSION_CALIBRATION_SUMMARY_V2.json](../outputs/biomaster_comprehensive_consensus_fusion_calibration_v5/FUSION_CALIBRATION_SUMMARY_V2.json)
- pair-specific GNINA 诊断：[CURRENT_CASE_GNINA_SUMMARY_V1.json](../outputs/current_case_pair_specific_gnina_v1/CURRENT_CASE_GNINA_SUMMARY_V1.json)
- 实验 pocket graph feature manifest：[LOCAL_GRAPH_FEATURE_MANIFEST_V1.json](../outputs/biomaster_odti_experimental_template_graph_features_v1/LOCAL_GRAPH_FEATURE_MANIFEST_V1.json)
- 实验 pocket graph 正式消融：[EXPERIMENTAL_TEMPLATE_GRAPH_PILOT_AUDIT_V1.json](../outputs/biomaster_odti_experimental_template_graph_pilot_v1/pilot_audit_v1/EXPERIMENTAL_TEMPLATE_GRAPH_PILOT_AUDIT_V1.json)
- 两案例内部简版 PDF：[BIOMASTER_SUCCESS_CASES_1_2_20260821_ZH.pdf](BIOMASTER_SUCCESS_CASES_1_2_20260821_ZH.pdf)

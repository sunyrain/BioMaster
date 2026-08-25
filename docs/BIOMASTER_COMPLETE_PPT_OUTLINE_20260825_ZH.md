# BioMaster 完整汇报 PPT 大纲

版本日期：2026-08-25
建议标题：**BioMaster：给定老药的潜在靶点发现与证据分层**
建议副标题：**面向 720 种老药 × 384 个候选靶点的数据、模型、药物中心检索与外部验证**
建议时长：30–35 分钟正文 + 10 分钟讨论
建议页数：25 页正文 + 10 页附录

---

## 0. 本版汇报必须统一的口径

本次完整汇报不再把所有数字笼统称作“当前模型结果”，而是明确区分三条证据线：

| 证据线 | 模型/数据 | 回答的问题 | 主要结果 |
|---|---|---|---|
| A. 冻结泛化基准 | E0 / pooled-ESM2 ODTI V2，冻结 ChEMBL37 切分 | 模型在 scaffold-cold、target-cold、double-cold、temporal、drug-entity-cold 下是否有内部泛化能力 | S1–S5；其中 S4 为 7,839 个 2023–2025 pair，AUPRC 0.8325 |
| B. 任务对齐的生产检索 | 冻结 V10 drug-centric evidence ranker | 给定一种老药，能否在 384 个靶点中把真实作用靶点排到前面 | 720 个药物查询 × 384 个靶点；KiRHub strict-unreported drug-macro AUPRC 0.398、Recall@10 0.505 |
| C. 最新双向 DTI | V5 共享骨干 + V6 D→T/T→D residual heads；三种子 head-only FULL_FIT | 是否能在不破坏 pair backbone 的前提下直接优化老药→靶点排序 | Stage A 三种子开发集均提升；2024–2025 集成点估计提升但 bootstrap CI 跨 0，故定位为探索性候选 |

必须明确：

- 实际主任务是 **old drug → candidate targets**：固定一种老药，在 384 个候选靶点内排序。
- 模型虽然是 drug–target 双输入，但“排序方向”由 sampler、ranking loss、校准和评测分组决定，不会因双输入而自动双向等价。
- `7,839` 是 S4 observed-pair temporal benchmark；主任务应优先看其中的 **drug-macro AUPRC 0.8251**。`0.8325` 是全局 micro-AUPRC，不是 384 靶点检索成功率。
- 原 V5 的 `18/27` 与 `x/720` 是 target → drugs 反向评测；V6 已将同一批 18/27 个未来关系重构为 **11/20 个老药 query → 384 targets** 的药物中心评测。
- 当前 V5 的 `drug_rank_weight=0`、sampler 以 target 为中心、fusion 也按每靶点 720 药物选择，因此它还不是主任务完全对齐的最终生产头。
- 双向 V6 已实现为 **共享 V5 pair backbone + 两个零初始化 residual heads**；drug→targets 是 primary，target→drugs 是 auxiliary。当前冻结 backbone 的三种子证据支持继续 head-only 使用，但不支持解冻主干或宣称统计学确认优于基线。
- 目前最有依据的应用范围仍是 **warm drug / warm target、exact relation held out**；target-cold 与 double-cold 是独立研究支线。
- 当前生产模型没有启用 ConPLex，也没有把 candidate pose 或 ligand–pocket 局部图并入主模型。
- 案例也必须改用 `target rank within drug`。当前可优先展示 **Lazertinib 的靶点谱**（ERBB4 6/384、AXL 7/384）和 **Repotrectinib 的靶点谱**（AXL 9/384）；原先的 Repotrectinib–PLK4 `3/720`、Deucravacitinib–FGFR4 `14/720` 不能继续作为主任务排名。

---

## 1. 正文页序总览

| 页码 | 标题 | 该页承担的任务 |
|---:|---|---|
| 1 | 封面 | 定义项目和汇报范围 |
| 2 | 一页结论 | 先交付当前最重要的结果与边界 |
| 3 | 科学问题与部署任务 | 把“pair分类”转换为“给定老药找靶点” |
| 4 | 双向矩阵与主次方向 | 明确 drug→targets 是主任务，target→drugs 是辅助任务 |
| 5 | 数据构建全景 | 解释 509,172 如何变成 437,248 |
| 6 | 标签与防泄漏合同 | 说明正负、affinity-only、时间和未知关系的处理 |
| 7 | 720 × 384 部署空间 | 解释为何上市药仍可能 drug-cold |
| 8 | V5 共享骨干与 V6 双向头 | 区分 pair evidence 与两个查询方向 |
| 9 | 结构信息到底是什么 | 澄清 19D pocket context 与 pair-specific 结构的区别 |
| 10 | 训练策略与方向错位 | 展示当前 target-side 设计及真正双向训练需求 |
| 11 | 收敛与过拟合控制 | 回答“是否训练到收敛” |
| 12 | 评测地图 | 所有主结果改为按药物分组的 target retrieval |
| 13 | E0 五类泛化基准 | 汇总 S1–S5 的 drug-macro 结果 |
| 14 | S4 Temporal 2023–2025 | 详细解释 7,839、micro 与 drug-macro 指标 |
| 15 | V10 药物中心外部结果 | 展示 strict-unreported KiRHub 的直接任务结果 |
| 16 | 从单头到双向 V6 | 解释为何旧 dual-query 不等于真正双向模型 |
| 17 | 全面补数据后的覆盖审计 | 解释新增数据和总体指标下降 |
| 18 | 2026 KiRHub 药物中心外部审计 | 展示按每种药物跨靶点检索的总体表现 |
| 19 | 案例一：Lazertinib 靶点谱 | 一个老药检索 ERBB4、AXL 等候选靶点 |
| 20 | 案例二：Repotrectinib 靶点谱 | 一个老药检索 AXL、PLK4 等候选靶点 |
| 21 | 方向纠正后的案例重分级 | 说明 Deucravacitinib、Lorlatinib、Tepotinib 的新定位 |
| 22 | Pair-specific pocket 尝试 | 汇报局部图、GNINA 和为何未晋级生产 |
| 23 | 当前研究主线的收束 | 明确已经证明和尚未证明的内容 |
| 24 | 双向 V6 训练路线与成功门槛 | 先完成方向头，再走向结构重排和湿实验 |
| 25 | 总结 | 用三句话结束汇报 |

若汇报时间只有 20 分钟，可将第 6、11、17、21、22 页移入附录。

---

# 第一部分：问题、数据与模型

## 第 1 页｜封面

### 页面目的

让听众立即知道这不是泛泛的“AI 制药”，而是一个有固定候选库、固定靶点轴和明确排序输出的项目。

### 页面内容

- 主标题：BioMaster：给定老药的潜在靶点发现与证据分层
- 副标题：面向 720 种老药 × 384 个候选靶点的数据、模型、药物中心检索与外部验证
- 汇报人、单位、日期
- 页脚小字：最新双向候选 `BIOMASTER_BIDIRECTIONAL_V6_HEAD_ONLY_FULL_FIT`；冻结任务对齐基线 `OLD_DRUG_LEAKAGE_SAFE_V10`

### 建议视觉

中央使用一条药物中心流程：

```text
one approved/old drug
        ↓
384 candidate targets
        ↓
BioMaster target ranking
        ↓
Top targets → evidence/structure review → experiment
```

### 口头讲述

“我们的目标不是给一个蛋白筛 720 种药，而是从一种已知老药出发，在 384 个候选靶点中建立它的潜在作用谱，并优先验证可能解释新机制或新适应症的靶点。”

---

## 第 2 页｜一页结论：我们现在做到哪一步

### 页面目的

先给出结果，不让听众在方法细节中等待结论。

### 核心数字

- FULL_FIT 训练数据：437,248 行
- 完整矩阵：720 × 384 = 276,480 个 pair scores；主输出应为每种药物的 384 靶点排序
- 最新双向候选：V6 三个 FULL_FIT checkpoint；单 checkpoint 3,551,222 参数，其中两个方向头各 16,641 参数
- E0 S4 temporal：7,839 对；micro-AUPRC 0.8325，**drug-macro AUPRC 0.8251**
- 冻结 V10 drug-centric 外部 strict-unreported：2,823 pairs、202 positives、prevalence 0.0716
- V10 strict-unreported：micro-AUPRC 0.1858、drug-macro AUPRC 0.3978、Recall@10 0.505
- V6 2024–2025 药物中心时间测试（三种子集成，20 个药物、27 个未来关系）：平均正样本 rank 91.7→88.7，MRR 0.177→0.208，Recall@5 0.108→0.158，NDCG@20 0.211→0.237
- V6 配对 drug-query bootstrap：综合增量 +0.0206，改善概率 95.2%，但 95% CI [-0.0026, 0.0495]；属于趋势性增益，不是显著性确认
- 药物中心案例：Lazertinib→ERBB4 6/384、AXL 7/384；Repotrectinib→AXL 9/384
- V6 FULL_FIT 候选案例：Lazertinib→ERBB4 3/384、Repotrectinib→AXL 2/384、PLK4 6/384、Pitavastatin→IDO1 10/384；Tepotinib→IRAK1 仍为 336/384 的明确失败

### 建议版式

左侧放“已做到”，右侧放“尚未做到”：

| 已做到 | 尚未做到 |
|---|---|
| 已有专门按药物在 384 靶点中排序的冻结 V10 分支 | V6 双向 residual heads 与三种子 head-only FULL_FIT 已完成 |
| 标准 temporal benchmark 的 drug-macro 指标较好 | target-cold / double-cold 靶点发现仍不可靠 |
| 2026 外部筛选中按药物分组出现富集和高排名靶点 | 模型分数不能当作物理 affinity 或临床概率 |

### 本页结论

> 当前系统已形成“共享 pair backbone + D→T 主头 + T→D 辅头”的双向候选；Stage A 支持该方向设计，但 20-query 时间测试的不确定性仍较大，因此 V10 继续作为冻结任务基线，V6 暂作为探索性生产评分候选。

---

## 第 3 页｜科学问题：我们实际要解决的是什么

### 页面目的

把任务从随机 pair 分类重新定义为目标驱动的检索。

### 页面内容

主问题：

> 给定一种已上市或已有充分药理信息的老药，从 384 个候选靶点中找出值得优先验证的潜在新作用靶点。

三个层次：

1. **药物查询表示**：模型是否能够建立该老药的已知与潜在 target profile？
2. **跨靶点关系外推**：对同一种药，能否把未见 exact pair 的真实靶点排在 384 个候选靶点前部？
3. **物理与机制验证**：该药物进入候选靶点 pocket 后是否存在结构、结合和功能证据？

### 建议视觉

画一个 720 行（药物查询）× 384 列（候选靶点）的稀疏矩阵：

- 已知正关系：红点
- 已知负关系：蓝点
- 未知关系：灰点
- 当前任务：固定一行老药，对这一行的 384 个靶点排序

### 必须强调

- 未记录关系不是负例。
- 主排名必须在同一种药物的 384 个靶点内部解释。
- 原始 pair logits 通常不具备天然跨靶点可比性，需要 drug-centric normalization/calibration 后才能用于 target profile 排序。
- target→720 drugs 可以作为双向鲁棒性诊断，但不是本项目的主交付。

---

## 第 4 页｜双向矩阵不等于双向任务等价

### 页面目的

这是全场最重要的任务定义页，应在后续任何结果之前出现。

### 建议视觉

先展示同一个双输入矩阵的两种查询方向：

```text
主任务：old drug query → rank 384 targets
辅助任务：target query → rank 720 old drugs
```

### 为什么双输入仍然有方向

- 如果数据是对 720×384 所有 pair 在同一 assay 下测得的稠密 Kd/Ki，一个物理 affinity score 理论上可以同时用于两个方向排序。
- 但现实训练数据是异质 assay、不完全观测和严重偏置的关系表；模型分数同时混有 target/drug popularity、assayability 与记录概率，因此两个 retrieval directions 需要独立的 nuisance correction 和校准。
- binary head 给 pair 打分，但 ranking loss 决定“和谁比较”。
- target-balanced batch 与 within-target ranking 学的是同一靶点下区分不同药物。
- within-drug ranking 才直接学习同一药物下区分不同靶点。
- 校准若在每个 target 内做 percentile，也不能直接当作跨 target affinity。
- 评测必须以 drug 为 query 分组，报告 drug-macro AUPRC、Recall@K、MRR 和 target rank。

### 当前两条实现线

#### 任务对齐线：V10 drug-centric

- 已有冻结的 720 drugs × 384 targets 排序
- 使用 `old_drug_leakage_safe_score_v10`
- 有 KiRHub drug-grouped 外部评测和 target rank within drug

#### 最新表示线：V5 six-checkpoint DTI

- 使用更全面的 437,248 行数据和更新的交互骨干
- 但 sampler、显式 ranking loss 与 final fusion 仍以 target query 为中心
- 当前 `x/720` 只作为辅助结果；需在共享 pair representation 上增加独立的 drug→targets 主头

### 建议的统一形态：一个骨干，三种输出

```text
shared drug–target representation
      ├─ pair evidence head：二分类 / affinity 锚点
      ├─ primary D→T head：同一药物内排 384 targets
      └─ auxiliary T→D head：同一靶点内排 720 drugs
```

两个 retrieval heads 可以从 pair score 的零初始化 residual 开始，使 V5 能力得以保留；两个方向各自校准、各自输出 rank，不将 retrieval score 冒充成统一结合概率。

### 本页结论

> 模型是双输入不代表两个排序方向自动等价；本项目最终只把“每种老药的候选靶点排序”作为主任务，所有反向结果必须降级为辅助诊断。

---

## 第 5 页｜数据构建：从 509,172 到 437,248

### 页面目的

说明训练规模、来源和去重过程，回应“是否全面补全”。

### 建议使用漏斗图

| 阶段 | 行数 |
|---|---:|
| ChEMBL 37 标签过滤前 | 509,172 |
| 满足标签条件 | 427,526 |
| 成功解析特征的 ChEMBL 关系 | 426,939 |
| BindingDB direct Ki/Kd affinity-only | 9,778 |
| recovered 数据去重后新增 | 531 |
| **最终 FULL_FIT** | **437,248** |

### 标签组成

| 类型 | 行数 |
|---|---:|
| 二分类正例 | 331,381 |
| 二分类负例 | 96,089 |
| affinity-only | 9,778 |

### 补充数字

- 模型表示层面唯一药物：约 296,108
- 唯一靶点：843
- ChEMBL 特征失败：587 行，未静默填零

### 口头讲述重点

“现在已经不是每个靶点各抽 150 条。所有合格、可解析的关系都保留；频率不平衡通过 sampler 处理，而不是删除长尾或高频数据。”

---

## 第 6 页｜标签合同与防泄漏设计

### 页面目的

说明模型学习的标签语义，以及为什么时间测试可以被信任。

### 页面内容

- active/positive：符合冻结 activity 规则的明确活性关系
- inactive/negative：明确测量且满足 inactive 规则
- affinity-only：只有 Ki/Kd 连续值，不强行转换成二分类负例
- unknown relation：不作为负例
- duplicate：按标准化 drug entity、target identity 和关系来源去重
- 时间切分：以 pair 的 first-seen era 定义，而非数据库下载时间
- 外部案例标签：不参与 checkpoint、epoch 或融合权重选择

### 建议视觉

画一个“允许进入哪一个损失”的标签路由图：

```text
binary positive/negative → BCE + ranking
Ki/Kd rows               → affinity + affinity ranking
unknown pairs            → no supervised label
external 2026 labels     → read-only evaluation
```

### 风险提示

FULL_FIT 吸收了当前全部合格数据，因此 FULL_FIT 上的已知关系表现只能叫 coverage audit，不能叫独立测试。

---

## 第 7 页｜720 × 384 部署空间：为什么上市药仍然会冷启动

### 页面目的

彻底解释“上市”与“出现在训练关系中”不是同一概念。

### 实体覆盖

| 轴 | warm | cold |
|---|---:|---:|
| 720 种药物 | 498 | 222 |
| 384 个靶点 | 359 | 25 |

### 276,480 对的四格分解

| pair 类型 | 数量 |
|---|---:|
| double-warm | 178,782 |
| drug-cold only | 79,698 |
| target-cold only | 12,450 |
| double-cold | 5,550 |

### 已知关系的稀疏度

- 与 FULL_FIT 精确重合的关系：5,992
- 其中正关系：890
- 负关系：5,102
- 其余约 270,488 对没有精确关系记录

### 本页结论

> 对每一种老药，我们要在 384 个靶点上形成 target profile；双温表示药物和靶点分别出现过，但 exact pair 未见，这仍是当前最可靠的新关系发现空间。

---

## 第 8 页｜V5 共享骨干与 V6 双向检索头

### 页面目的

用一张图区分共享 pair representation 与已经实现的两个查询方向。

### 架构图建议

```text
Morgan 2048 ── drug encoder ───────────────┐
                                           ├─ low-rank FiLM interaction
ProtBERT 1024 ─┐                           │
ESM2 1280 ─────┴─ target encoder ──────────┤
family embedding 24 ───────────────────────┤
19D structure context + mask ──────────────┘
                        ↓
                 6-expert family MoE
                        ↓
                  shared pair hidden
             ┌─────┼─────┐
             ↓           ↓           ↓
       pair evidence   D→T primary   T→D auxiliary
     binary/affinity   rank 384 targets  rank 720 drugs
```

### 关键配置

- 单 checkpoint 参数量：3,517,940
- embedding：192；hidden：256；dropout：0.12
- interaction rank：48；FiLM scale：0.10
- 6 个 expert
- binary、ranking、affinity、affinity ranking、contrastive 多任务

### 需要明确

- conservative 与 recall 两组使用同一主干，不应表述为完全不同的模型架构。
- 多样性主要来自独立训练权重、随机种子、选中 epoch 和少量训练配置。
- 当前生产模型 `conplex_enabled=false`。
- 双输入 encoder 和 pair head 本身可以对任意 drug–target pair 打分，但最终“药物找靶点”仍需要按 drug 分组训练、标准化、选模和评测。
- 一个在 target→drugs 上表现好的 pair head，不保证在 drug→targets 上同样好；target prior、family prior 和 target-level structure context 都可能造成跨靶点捷径。
- V6 已使用轻量 residual heads：`directional_score = pair_score + small_residual`；最后一层零初始化并加 residual norm，启用方向头的初始逐 pair 输出与 V5 完全一致。
- shared pair/affinity head 负责保留两个方向共享的生物相互作用核心；directional residual 只负责纠正候选集、观测偏置和 query-level 竞争差异。
- 主头和辅头不必等权；项目的生产目标决定 D→T 是 primary，T→D 主要作为表示约束、反向应用和非退化检查。

---

## 第 9 页｜19D pocket 信息是什么，又不是什么

### 页面目的

回应“既然有口袋，为什么不能直接判断具体药物是否结合”。

### 页面内容

19D 包括：

- 受体/实验 holo 可用性
- pocket score、概率、体积和残基数
- pocket 共识与实验结构覆盖
- receptor 长度、序列匹配和结构等级等

覆盖情况：

- 254/384 个靶点有结构输入
- 对应 182,880/276,480 个 pair
- 130 个靶点缺结构，对应 93,600 个 pair
- `structure_mask=0` 时严格回退到非结构 score

### 左右对比图

| 当前 19D target context | 尚未进入生产的 pair-specific 信息 |
|---|---|
| 对同一靶点的 720 种药基本共享；在同一药物的 384 靶点间会变化 | 每一个 drug–target pair 有自己的 pose |
| 说明 pocket 是否存在、大小和质量 | ligand–residue 距离、氢键、盐桥、疏水和 clash |
| 能调整 target context | 才能回答“这个具体药是否适配” |

### 本页结论

> 当前模型“使用了 pocket context”，但在 drug-centric 排序中必须防止它退化成 target quality prior；真正与具体老药相关的增量应来自 candidate drug–pocket 局部几何。

---

## 第 10 页｜训练策略与当前方向错位

### 页面目的

说明最新 V5 如何训练，同时诚实指出它尚未完全对齐“给定老药找靶点”。

### 训练设置

- batch size：1,024
- 每 epoch：128 steps，约 131,072 个采样行
- target-balanced sampler
- `rows_per_target=16`
- `target_frequency_power=0.5`
- inactive-only negative fraction：0.25
- FULL_FIT affinity chunk fraction：0.09375
- target-side ranking weight：0.12
- **drug-side ranking weight：0**
- listwise weight：0

### 六 checkpoint

| 分支 | seeds | 选定 epoch |
|---|---|---|
| conservative | 20260816 / 17 / 20 | 9 / 8 / 11 |
| recall | 20260816 / 17 / 20 | 11 / 8 / 6 |

### V5 当前最终推理

1. 两组各三个 seed 得到共识
2. conservative 与 recall 等权
3. 对每个 target，在 720 drugs 内由 binary 先取前 20%
4. target shortlist 内使用 binary 0.55 + affinity 0.45 精排

### 方向错位审计

- sampler、显式 ranking loss、fusion selection 和最终 rank 都是 target-centric。
- 因此 V5 的完整矩阵可以提供 drug-centric reciprocal diagnostic，但不能直接把 `ensemble_fusion_score` 当作跨靶点 affinity。
- 既有 `drug_rank_weight=0.08 + dual_query sampler` 短屏的 S5 micro-AUPRC 为 0.4637、drug-macro AUPRC 为 0.7185，低于对照汇总的 0.5396 / 0.7743；但它仍只有一个 `final_logit`，且不是干净的双向架构实验。
- 该短屏的每条训练行在每个 epoch 被随机分配给 target 或 drug 一侧；target-rank weight 0.12，drug-rank 0.08，listwise 仍只对 target groups，选模 composite 中 drug-macro 只占 20%。
- 另外，该 dual checkpoint 的 `structure_input_dim=0`，而 S5 正式基线的 `structure_input_dim=19`；因此下降同时混入了结构输入差异，不能用来否定真正双向 V6。
- V6 已改成 query-first batches、独立 D→T/T→D residual heads、两个方向各自的 pairwise/listwise objective，以及以 D→T 为主、pair/T→D 非退化为约束的 model selection。

### V6 实际 Stage A 设置与结果

- 评估 fit：截至 2022 的 384,542 行；只使用 `binary_observed=1` 的显式正负，不将 affinity-only/unreported 当负例。
- 可形成两类排序的训练 query：7,149 个 drug queries、407 个 target queries。
- D→T/T→D 每 epoch 分别 48/32 steps，batch size 1,024，`rows_per_query=16`；V5 backbone 全冻结。
- 2023 开发：11 个老药、18 个未来正关系；三个 seed 的选择值分别提升 +0.0449、+0.0498、+0.0197，选中 epoch 为 6/3/2。
- 2024–2025 只在选模后查看：20 个老药、27 个未来正关系；三种子集成多数指标改善，但 composite 的 95% bootstrap CI 仍跨 0。

### 本页结论

> 共享骨干、方向分头的 V6 已通过三种子开发门槛并完成 head-only FULL_FIT；由于冻结时间测试样本只有 20 个药物且 CI 跨 0，当前不解冻 backbone，任务对齐的强基线仍以 V10 为准。

---

## 第 11 页｜是否收敛，以及如何控制过拟合

### 页面目的

直接回答“有没有训练到收敛”和“是否注意过拟合”。

### 页面内容

- 选模训练继续跑到 11–16 epochs，用 patience 观察开发指标
- 最佳 checkpoint 出现在 6–11 epochs
- 后续训练损失继续下降，但开发选择值多数下降
- FULL_FIT 再按最佳 epoch 预算从头训练，而不是使用最后一轮

### 建议折线图

选两个代表 seed：

- conservative seed16：train loss 0.1945 → 0.1474；dev 0.7690 → 0.7663
- recall seed16：train loss 0.1729 → 0.1342；dev 0.7712 → 0.7521

### 准确表述

> V5 已达到其现有 target-centric 开发目标意义上的收敛；继续降低训练损失会增加过拟合。但这不等于 drug-centric 目标也已收敛，后者需要新的按药物分组验证集和早停指标。

不建议说“训练损失已经完全不再下降”。

---

# 第二部分：评测设计与结果

## 第 12 页｜评测地图：每个数字回答什么问题

### 页面目的

在展示指标前建立统一坐标系。

### 建议表格

| 评测 | 模型 | query 与候选空间 | 独立性 | 在本汇报中的角色 | 主指标 |
|---|---|---|---|---|---|
| S1–S5 drug-grouped | E0/ODTI V2 | drug query → benchmark targets | 内部冻结 | 表示泛化证据 | drug-macro AUPRC |
| S4 temporal | E0/ODTI V2 | 83 个有双类的 drug groups；总计 7,839 observed pairs | 2023–2025 test | 时间泛化证据 | drug-macro AUPRC、micro-AUPRC |
| V10 KiRHub all-active | V10 leakage-safe | 72 个有双类 drug queries；8,058 measured pairs | 外部只读 | 较宽外部任务 | drug-macro AUPRC、Recall@K |
| V10 KiRHub strict-unreported | V10 leakage-safe | 33 个有双类 drug queries；2,823 pairs | 外部只读 | **当前主外部证据** | drug-macro AUPRC、Recall@K、rank/384 |
| V5 reciprocal drug diagnostic | V5 FULL_FIT | 每种药 → 384 targets | 非预注册、score 为 target-wise fusion | 探索性，不作正式主结果 | reciprocal target rank |
| V5 target→drug 18/27 | V5 evaluation/fusion | target query → 720 drugs | dev/test | 反向辅助诊断 | rank/720、Hit@K |
| FULL_FIT relation audit | V5 FULL_FIT | 5,992 known relations | 非独立 | coverage only | AUPRC/AUROC |

### 本页结论

任何结果都必须同时报告：query 是 drug 还是 target、候选空间、模型版本、是否参与选模。正文主结果只接受 drug query。

---

## 第 13 页｜E0 五类泛化基准

### 页面目的

一页交代模型在不同泛化压力下的完整谱系，而不是只选最好的一项。

### 结果表

| 角色 | 含义 | rows | 正例率 | **drug-macro AUPRC** | micro-AUPRC | 与主任务的关系 |
|---|---|---:|---:|---:|---:|---|
| S1 | scaffold-cold drug | 86,673 | 0.531 | **0.8594** | 0.9316 | 药物骨架冷启动证据，但不是固定老药的最直接场景 |
| S2 | homology-cold target | 86,673 | 0.531 | **0.5540** | 0.7939 | 跨同源簇的药物中心排序明显变难 |
| S3 | strict double-cold | 17,732 | 0.533 | **0.6365** | 0.6344 | 严格双冷仍是研究支线，不是当前部署能力 |
| S4 | first-seen 2023–2025 | 7,839 | 0.563 | **0.8251** | 0.8325 | 时间外推中，同药物下的 observed targets 有较强排序信号 |
| S5 | old-drug entity-cold | 2,556 | 0.138 | **0.7759** | 0.5569 | 接近老药应用，但是 entity-cold，且各 drug group 候选数较少 |

### 图形建议

主图使用 drug-macro AUPRC 横向柱状图，micro-AUPRC 用浅色点标记。不要将全局 prevalence 直接画成 macro-AUPRC 的“随机基线”，因为各 drug group 的正例率不同；可在附图单独展示 group-specific prevalence 分布。

### 讲述边界

- S1/S2/S3 是 5 folds × 5 seeds，共 25 runs
- S4/S5 是 5 seeds
- 这些是冻结内部 benchmark，不是 2026 外部实验
- drug-macro AUPRC 是主方向指标；micro-AUPRC 只说明全部 pair 混合后的区分能力
- S3 不支持“已经解决 double-cold”

---

## 第 14 页｜S4 Temporal：7,839 与 0.8325 的完整含义

### 页面目的

正式纠正此前对 7,839 与 18/27 的混淆。

### 冻结切分

```text
截至 2022 的 train pool：76,965
├─ train：61,634
└─ validation：15,331

2023–2025 first-seen test：7,840
└─ quarantine 1 → 最终 7,839
```

### 测试组成

- 4,411 positives
- 3,428 negatives
- prevalence：0.5627
- 6,624 个化合物
- 349 个靶点

### 模型对比

| 模型 | **drug-macro AUPRC** | micro-AUPRC |
|---|---:|---:|
| pooled-ESM2 ODTI V2，五种子均值 | **0.8251** | **0.8325** |
| 旧 BioMaster routed stack，单种子 | 0.7751 | 0.7957 |
| ConPLex frozen external | 0.5855 | 0.5878 |
| 全局正例率（仅为 micro PR 参照） | 不适用 | 0.5627 |

补充：

- ODTI V2 在 83 个同时含正负例的 drug groups 上，drug-macro AUROC 为 0.7282。
- drug-macro Recall@5/10/20 分别为 0.9639 / 0.9880 / 1.0000，但这是 observed-pair 子集：每种药只有少量被记录候选，且可能有多个正例，不可类比为完整 384-target 检索的 Recall@K。
- ODTI V2 micro-AUROC 为 0.7791，ECE15 为 0.0748。

### 必须写在页脚

> 这是按 drug 分组的 observed-pair temporal benchmark，不是在完整 384 个候选靶点上的 dense retrieval。`0.8325` 是 micro-AUPRC；与主任务方向更一致的数字是 drug-macro AUPRC `0.8251`。

---

## 第 15 页｜V10 药物中心的外部结果

### 页面目的

汇报当前与“给定老药找靶点”最直接对齐的冻结结果。

### 外部数据与排序口径

- 外部标签：KiRHub 2026，1 µM inhibition ≥70% 定义 strong hit
- query：每一种老药
- 生产候选轴：384 个靶点；个案的 `rank/384` 来自全候选排名
- 外部 AUPRC/Recall@K：只在 KiRHub 实际测量且完成精确映射的 pair 上计算
- V10 分数在外部标签之前冻结，未使用 KiRHub 结果调整后处理组合

### 主结果

| 外部切片 | pairs / positives | prevalence | micro-AUPRC | **drug-macro AUPRC** | drug-macro Recall@5 / 10 / 20 |
|---|---:|---:|---:|---:|---:|
| all KiRHub active | 8,058 / 1,041 | 0.1292 | 0.4365 | **0.5399** | 0.341 / 0.442 / 0.591 |
| **strict frozen-unreported** | **2,823 / 202** | **0.0716** | **0.1858** | **0.3978** | **0.391 / 0.505 / 0.677** |

strict slice 中有 33 种同时具有正负外部标签的药物，drug-macro AUROC 为 0.6805。

### strict-unreported 对比

| 模型 | micro-AUPRC | drug-macro AUPRC |
|---|---:|---:|
| **V10 leakage-safe drug-centric** | **0.1858** | **0.3978** |
| V8 previous graph fusion | 0.1491 | 0.3722 |
| Frozen DTA consensus | 0.0851 | 0.2452 |
| ConPLex drug-centric | 0.0762 | 0.2968 |

### 本页结论

V10 相对 7.16% 的严格外部正例率存在富集，且比对照模型更适合 drug-centric 方向；但这仍是回顾性外部测试，不是前瞻实验或临床有效性证明。

---

## 第 16 页｜从单头 V5 到真正的双向 V6

### 页面目的

解释“双输入”、“旧 dual-query 实验”与“真正的双向检索模型”之间的差别。

### 三种实现的本质对比

| 组件 | V5 单头 | 旧 dual-query 短屏 | 已实现的双向 V6 |
|---|---|---|---|
| pair representation | 共享 | 共享 | **继承 V5 共享骨干** |
| retrieval score | 单一 `final_logit` | 仍是单一 `final_logit` | pair anchor + **D→T/T→D 两个 residual scores** |
| query sampling | target-balanced | 每行随机分配一个方向 | **query-first 独立 drug/target neighborhoods** |
| ranking objective | within-target | target + 较弱 within-drug pairwise | 两方向各自 pairwise + listwise |
| unknown pair | 不是负例 | 不是负例 | 仍不是负例；只用明确 inactive/可审计 PU 策略 |
| selection | target-side/composite | drug-macro 仅占 composite 20% | **D→T 主指标 + pair/T→D 非退化约束** |
| output | `rank/720` | 仍无独立方向校准 | `rank/384` 与 `rank/720` 分开输出 |

### 建议的损失与选模逻辑

```text
shared V5 pair score: frozen in Stage A / FULL_FIT head refit
L_D2T = 0.25 BCE + 1.00 pairwise + 0.25 listwise + 0.001 ||Δ_D2T||²
L_T2D = 0.25 BCE + 1.00 pairwise + 0.25 listwise + 0.001 ||Δ_T2D||²
updates per epoch: D2T 48 + T2D 32
```

- D→T 是主任务，因此 `λ_D2T` 与更新频率可高于 T→D；“双向”不意味着两边必须 1:1 等权。
- 选模不用一个可被辅助任务拉高的加权平均；先最大化冻结 D→T drug-macro AUPRC/Recall@K，再要求 pair AUPRC、calibration 和 T→D 不超过预注册退化幅度。
- 若共享骨干上两方向梯度冲突，先采用 alternating updates；必要时再测试 PCGrad/GradNorm，不在第一版同时引入太多变量。

### 实际训练顺序与决策

1. 已冻结截至 2022 的 V5 evaluation backbone，只训两个零初始化 residual heads。
2. 2023 密集 D→T 开发集三种子均提升；pair backbone 逐张量保持不变，T→D target-macro AUPRC 未退化。
3. 2024–2025 三种子集成点估计改善，但 paired-query bootstrap CI 略跨 0；因此 **没有进入解冻 pair trunk/MoE 的 Stage B**。
4. 已按 6/3/2 epoch 在各自 V5 FULL_FIT backbone 上完成 head-only FULL_FIT；主干逐张量核验完全一致，仅新增 12 个方向头参数张量。
5. KiRHub 保持只读，未用于决定双头权重或 checkpoint。

### 原 V5 结果的正确去向

- 2023 development：18 个 target queries，median 41/720，用于 fusion selection。
- 2024–2025 frozen test：27 个 target queries，median 120/720，Hit@10 7.4%。
- recovered relation-held-out：37 relations，AUPRC 0.933；21 个正例在 target→720 drugs 中 median 6/720。
- 它们仍能证明 pair signal 和反向检索能力，但应放入“辅助证据/附录”，不再充当主部署结果。

### V5 reciprocal diagnostic 的使用限制

对 V5 同一药物的 384 个 target-wise fusion scores 强行排序，可用于发现方向性问题；但这些分数是分别在各 target 内生成和筛选的，未经跨 target 校准，因此只能标为 exploratory reciprocal rank。

### 本页结论

> 双向模型已经按“一个共享 pair backbone + 两个方向头 + 两套查询级评测”实现；现阶段证据支持把它作为 head-only 探索性评分候选，但不支持解冻主干或宣称已显著超过 V10。

---

## 第 17 页｜全面补数据后，为什么总体结果会下降

### 页面目的

解释“全面补全”之后某些排名和汇总指标变差的原因。

### 数据变化

- 旧 capped positives：310
- 新增 positives：580
- 旧 capped negatives：1,925
- 新增 negatives：3,177

### V5 coverage audit（反向辅助证据）

- 5,992 个矩阵内已知关系
- AUPRC：0.8952
- AUROC：0.9689

### 新旧正例难度（target→720 drugs 口径）

| cohort | median rank | Hit@36 | Hit@72 |
|---|---:|---:|---:|
| 原 capped positives，310 | 11 | 70.6% | 90.0% |
| 新增 positives，580 | 22.5 | 61.9% | 75.2% |

### 原因解释

1. 新增关系更长尾、更弱、更异质，不是原高置信样本的随机扩充。
2. 全面关系中包含冲突标签和非典型药理，例如 Tepotinib–IRAK1 的 exact pair 有非正标签。
3. sampler 从拟合少量高频关系转向平衡更多靶点，旧案例排名可能下降。
4. 候选排序是相对竞争；某些新关系上升必然挤压原候选。
5. 更根本的是，这些旧排名只检查“给定靶点找药”；它们不能用来证明全面补数后“给定老药找靶点”也下降或上升。

### 页脚限制

这是 FULL_FIT coverage audit，不是独立泛化结果；表中 rank 是 `rank/720` 的反向诊断，不应出现在药物中心的总结页。

---

# 第三部分：外部验证与展示案例

## 第 18 页｜2026 KiRHub：药物中心的富集与失败景观

### 页面目的

在第 15 页的总体指标后，展示“一种药对多个靶点”的真实检索形态，同时不回避药物间差异。

### 外部数据

- 92 种临床 kinase inhibitors
- 758 个 kinase/variant targets
- 约 29 万次 1 µM 测量
- 本项目命中定义：抑制率 ≥70%
- 数据未用于 V10 分数建模，也未用于 V5 checkpoint、epoch 或 fusion 选择

### 冻结 strict-unreported 主集

- 2,823 个 measured pairs，202 strong hits，prevalence 7.16%
- 33 个可计算 drug-macro AUPRC 的双类 drug queries
- V10 drug-macro AUPRC 0.3978，Recall@10 0.5051
- 个案在全部 384 targets 内排名，而非在被 KiRHub 测量的小子集内排名

### 可用于展示“target profile”的外部强命中

| 老药 | strict-unreported 强命中 | V10 target rank within drug | 1 µM inhibition |
|---|---|---:|---:|
| Gilteritinib | KDR / FLT1 / PDGFRA / CSF1R | 1 / 2 / 3 / 4 of 384 | 76.37% / 81.83% / 78.09% / 90.75% |
| Ripretinib | FLT3 / CSF1R / FLT1 / BRAF | 2 / 3 / 5 / 8 of 384 | 75.22% / 97.15% / 93.66% / 100% |
| Lazertinib | ERBB4 / AXL | 6 / 7 of 384 | 98.98% / 81.07% |
| Repotrectinib | AXL | 9/384 | 77.77% |

可另列为简短小卡：Encorafenib→RAF1 1/384，Pacritinib→FLT1 2/384，Alpelisib→JAK2 3/384，Ponatinib→TEK 3/384。

### 结论与限制

- 这些主要是“本地 ChEMBL37 未报道、后被外部 panel 测到”的 database-gap rediscovery controls；未完成逐例文献新颖性审计前，不能称为首次发现。
- 平均富集并不表示每种药都准确；例如 Repotrectinib→PLK4 和 Tepotinib→IRAK1/4 在 V10 中都排名很后。
- 这是 1 µM 生化筛选，不等同于细胞作用、体内疗效或临床有效。
- KiRHub 标签未用于自动选择 target-centric 与 drug-centric 分数的后验组合。

参考：[Nature Biotechnology 2026](https://www.nature.com/articles/s41587-026-03090-8)

---

## 第 19 页｜案例一：Lazertinib 的候选靶点谱

### 页面目的

用一种老药找到两个外部强命中靶点，直接展示项目的真实查询方向。

### 核心结果卡

| 靶点 | 冻结 V10 rank/384 | V6 FULL_FIT 候选 rank/384 | KiRHub 1 µM inhibition | 冻结状态 |
|---|---:|---:|---:|---|
| **ERBB4** | **6** | **3** | **98.98%** | local ChEMBL37 unreported |
| **AXL** | **7** | 41 | **81.07%** | local ChEMBL37 unreported；V6 未保持 Top 20 |
| BTK | 29 | 待独立表中查看 | 77.07% | 外部强命中，但不在 V10 Top 20 |

### 推荐视觉

画 Lazertinib 为中心，右侧是 384-target 排名条，同时高亮 ERBB4 和 AXL。旁边标注三层证据：冻结 V10 高排名、local exact pair unreported、2026 外部生化强抑制。

### 推荐表述

> 给定 Lazertinib，冻结 V10 将 ERBB4 和 AXL 排在第 6 和第 7；V6 将 ERBB4进一步推到第 3，但 AXL 降到第 41。这个案例说明 V6 不是对所有已有强案例的单调升级，应保留 V10/V6 双轨审计。

### 不能表述

- 不能说 ERBB4/AXL 最早由我们发现。
- local-unreported 是针对冻结本地数据的状态，不等于全球文献中从未报道。
- 更准确的定位是“外部强命中的 database-gap rediscovery profile”。

---

## 第 20 页｜案例二：Repotrectinib 的稳定命中与版本分歧

### 页面目的

展示一个在正确任务方向下仍成功的 AXL 命中，并用 PLK4 说明反向高排名不能直接转成正向结论。

### 核心结果卡

| 关系 | 冻结 V10 | V6 FULL_FIT 候选 | KiRHub inhibition | 解释 |
|---|---:|---:|---:|---|
| **Repotrectinib→AXL** | **9/384** | **2/384** | **77.77%** | 两条 drug-centric 线均为高排名 |
| Repotrectinib→PLK4 | **373/384** | **6/384** | 83.49% | V6 与冻结 V10 强烈分歧；只能标为 post-refit 探索性恢复 |

辅助诊断：V5 reciprocal 为 11/384，PLK4→Repotrectinib 反向任务为 3/720；二者均不能取代冻结 V10 或 V6 的正向 `rank/384`。

### 推荐视觉

用一个 2×2 对照框：横轴是 AXL/PLK4，纵轴是冻结 V10 与 V6 FULL_FIT。绿色高亮 AXL 的 9→2，分歧色高亮 PLK4 的 373→6；旁边另列 V5 反向 3/720，仅作方向历史说明。

### 推荐表述

> Repotrectinib→AXL 在冻结 V10 和 V6 中分别为 9/384 与 2/384，是方向一致的成功案例。PLK4 则从冻结 V10 的 373/384 变为 V6 的 6/384；由于 V6 是 FULL_FIT 探索性候选，不能后验地用它抹去原冻结失败，必须把这种版本分歧作为独立复核对象。

### 价值与限制

- AXL 案例是可用证据，但仍需要文献时间线和正交实验审计。
- PLK4 是必须保留的方向敏感反例；它清楚证明了为什么本次不只做文字对调。

---

## 第 21 页｜方向纠正后的案例重分级

### 页面目的

把过去用 `x/720` 挑出的案例逐一放回“老药→靶点”口径，决定哪些保留、降级或退役。

### 重分级表

| 关系 | 原反向结果 | 冻结 V10 rank/384 | V5 reciprocal | V6 FULL_FIT rank/384 | 新定位 |
|---|---:|---:|---:|---:|---|
| Lazertinib→BTK | 26/720 | 29 | 14 | 待查完整表 | 外部强命中，但降为辅助案例 |
| Repotrectinib→PLK4 | 3/720 | **373** | 11 | **6** | 冻结失败、V6 恢复；版本分歧审计，不作无条件旗舰 |
| Deucravacitinib→FGFR4 | 14/720 | **179** | 78 | **25** | V6 明显改善，但仍需 entity/analog 与独立冻结复核 |
| Lorlatinib→PLK4 | 4/720 | **372** | 67 | **24** | V6 改善，但 2022 已有临床线索，不作新颖案例 |
| Pitavastatin→IDO1 | 反向候选 | 待冻结表核对 | 诊断性 | **10** | V6 高排名候选，可进入文献/实验复核 |
| Tepotinib→IRAK1 | 658/720 | **369** | 264 | **336** | 仍为明确漏检 |
| Tepotinib→IRAK4 | 573/720 | **338** | 164 | **155** | 有改善但仍不够，不能当成功案例 |

注：V5 reciprocal rank 只能用于诊断；V6 rank 是正确的药物中心方向，但来自 FULL_FIT 候选，没有独立 post-refit 性能估计，不能覆盖冻结 V10 的成功/失败判定。

### 仍需在附录保留的 target-cold 失败

- Dasatinib→SEPHS2、Hydroxychloroquine→VPS37C
- Binimetinib/Olaparib→SIRT3

这些原来的 `rank/720` 同样是 target→drug 结果；如果要讨论正向 target-cold 能力，必须在冻结 drug-centric 头上重新计算 rank/384。

### 讲述重点

“方向纠正不是更改排名名称，而是重新判定成功与失败。如果一个案例只在 target→drugs 上排名高，它不能证明 drug→targets 的发现能力。”

---

# 第四部分：结构尝试、研究收束与下一步

## 第 22 页｜Pair-specific pocket：已经试了什么，为什么还没进入生产

### 页面目的

解释结构分支不是没有做，而是按照可归因门槛没有晋级。

### 实验 pocket graph 试验

- 338 个 strict experimental assets
- 326 个与 benchmark target exact-sequence overlap
- 16,117 个 pocket residues
- 使用 residue ESM2、Cα 坐标、残基类型、B-factor、位置及 reference-ligand contact descriptors
- 未使用候选药物标签或候选 pose

### 正式消融

| 指标 | baseline | graph | Δ |
|---|---:|---:|---:|
| S3 two-seed micro-AUPRC | 0.640760 | 0.641773 | +0.001013 |
| S3 target-macro AUPRC | 0.661032 | 0.663862 | +0.002830 |
| S4 double-warm AUPRC | 0.747870 | 0.747999 | +0.000129 |

- bootstrap 95% CI：[-0.001826, +0.003711]，跨 0
- 决策：`KEEP_RESEARCH_CANDIDATE_DO_NOT_PROMOTE`

### GNINA 互补案例

- Tepotinib–IRAK1：DTA rank 658，但 target-calibrated GNINA consensus 0.833，优于全部 negatives 和 10/12 positives
- 说明 pair-specific 几何可能补充 DTA，但单例不能证明总体重排有效
- 更重要的限制：现有 GNINA 分数是 **target-calibrated**，适合同一 target 下比较不同药物；原始 docking score 也受 pocket 尺寸和构象影响，不可直接用来对同一药的不同 targets 做物理结合强度排名。

### 下一版真正需要的输入

```text
candidate ligand pose
+ pocket residue graph
+ ligand–residue cross-distance/orientation
+ H-bond / salt bridge / hydrophobic / aromatic / clash
+ multi-pose consensus
→ per-target calibration + cross-target evidence calibration
→ drug-centric reranker
```

### 与主任务对齐的结构验证单元

给定同一种老药，对其 Top-N 候选靶点和 matched negative targets 生成结构证据；先在每个 target 内使用已知 ligand controls 做校准，再融合为跨靶点可比的 evidence score。该分数不应被称为可直接比较的 absolute affinity。

---

## 第 23 页｜当前研究主线的最终收束

### 页面目的

明确项目故事，不再同时追逐互相冲突的主张。

### 主线

```text
全面 DTI 数据
→ 可审计的 drug/protein representation
→ 双输入 pair backbone
→ pair evidence anchor + D→T 主头 + T→D 辅头
→ 主部署：给定一种老药，对 384 targets 排序
→ warm-drug / warm-target exact-relation-absent candidates
→ 跨靶点校准的 pair-specific structure evidence
→ biochemical / cellular validation
```

### 已经证明

- 标准内部 scaffold-cold 和 temporal benchmark 有较强信号。
- 冻结 V10 已直接实现每种老药在 384 个靶点中的排序，并在 KiRHub strict-unreported 上超过多个对照。
- 2026 外部 kinase 筛选存在总体富集，且可见同一药物的多靶点高排名命中。
- V5 的全面数据、去重、早停和 pair backbone 已完成；V6 已补上独立 D→T/T→D heads，并完成三种子 Stage A 与 head-only FULL_FIT。
- 方向头在 2023 三种子开发集一致提升；2024–2025 集成的 MRR、Recall@5、Recall@20、NDCG@20 和 retrieval AP 点估计改善。

### 尚未证明

- target-cold 或 double-cold 的可靠实际应用。
- 未来所有新关系都能进入 Top K。
- 模型 score 等于 affinity 或临床疗效概率。
- target-level pocket 或当前局部图已带来稳定增益。
- 尚未统计学确认双向 V6 优于其 V5 pair ensemble，更未证明优于冻结 V10：20-query bootstrap 的 composite 95% CI 仍跨 0。

### 建议一句话

> BioMaster 当前的主线是：给定一种老药，在 384 个候选靶点中排序未见 exact relations。V10 是冻结任务基线；V6 已完成共享骨干、D→T 主头和 T→D 辅头，下一阶段不是继续解冻，而是扩大独立药物 query、做跨靶点结构重排和前瞻实验。

---

## 第 24 页｜双向 V6 训练路线与晋级门槛

### 页面目的

汇报已经完成的 Stage A/FULL_FIT，并把下一步写成可检验计划。

### 已完成里程碑

| 项目 | 结果 |
|---|---|
| zero-start 合同 | 方向头启用前与 V5 `final_logit` 逐元素相等 |
| 三种子 Stage A | 2023 D→T 选择值均提升；最佳 epoch 6/3/2 |
| pair 非退化 | backbone 冻结，FULL_FIT 后逐张量完全一致 |
| T→D 辅头 | seed16 开发 target-macro AUPRC 0.7998→0.8044；三种子均通过预设非退化门槛 |
| 时间测试集成 | composite 0.3891→0.4098；bootstrap 改善概率 95.2%，CI 仍跨 0 |
| FULL_FIT | 三个 head-only checkpoint，437,248 行，状态均 PASS |

### 工作包 1：数据与查询合同

- 保持当前 train/validation/test 和时间泄漏合同，但另外生成 drug-query 与 target-query manifests
- drug-query batch：一种 drug + 同一 drug 的明确 positives/inactives + 可审计 hard negatives
- target-query batch：一个 target + 同一 target 的明确 positives/inactives
- unknown/unreported pair 不直接当负例；若启用 PU/observation correction，必须作为独立消融
- 对 query 数、每 query 的正负数、target family 和 drug scaffold 做完整覆盖审计

### 工作包 2：从轻量双头到联合微调

- 保留 V5 drug/protein encoder、FiLM interaction、MoE 和 pair evidence heads
- 新增零 residual 初始化的 D→T 主头和 T→D 辅头，两头各自计算 pairwise/listwise loss
- Stage A 已完成；由于时间测试 CI 跨 0，Stage B 解冻被主动搁置，避免用 20 个药物 query 追逐噪声
- D→T 是主头；选模最大化 drug-macro AUPRC/Recall@K/MRR，并对 pair calibration 和 T→D 设非退化门槛
- 对 target frequency、family prior 和 19D structure availability 做捷径审计，不让 D→T 主头只学 target popularity/assayability
- 冻结 V10 为 D→T baseline，当前 V5 为 T→D baseline；两者都不因 KiRHub 结果后验调整

### 工作包 3：冻结外部评测

- 固化 KiRHub 匹配表、过滤规则、输入哈希和评测脚本
- 预注册 primary cohort：double-warm、exact-pair-absent
- secondary cohort：drug-entity-cold、target-warm
- 不用外部标签继续调 fusion
- 所有 resampling 和 confidence interval 应按 ligand/drug cluster 进行，而不是将 pair rows 视为完全独立

### 工作包 4：pair-specific 结构重排

- 只对每种老药的 Top 20–50 个候选 targets 生成 pose
- 每个 target 配置已知 ligand 正负 controls，先做 per-target calibration
- 提取 cross-distance/contact/clash/multi-pose 特征
- 先作为独立 reranker，不直接改 FULL_FIT backbone

### 工作包 5：前瞻湿实验

- 固定一种或少数老药，对同一药的 Top targets、中位 targets 和 matched negatives 预注册
- 优先 warm-drug / warm-target、exact-relation-absent
- 至少包含 binding/enzymatic + cellular target engagement 两级
- 解盲后 append-only 写入 observation store

### 晋级门槛

| 模块 | 最低门槛 |
|---|---|
| V6 D→T 主头 | 当前只达到探索性门槛；正式晋级仍要求相对冻结 V10/V5 的 drug-query bootstrap CI 排除 0，且 pair calibration 不退化 |
| V6 T→D 辅头 | 相对当前 V5 反向基线不超过预注册退化幅度；达不到时不阻止 D→T 研究结论，但不得宣称“双向无损” |
| 结构 reranker | 多 seed、ligand-cluster bootstrap 的 drug-centric Hit@K/AUPRC 增益 CI 排除 0 |
| 外部 panel | primary cohort 显著优于冻结 V10、DTA 和 ConPLex baselines |
| prospective | Top-ranked candidates 相对 matched negatives 有可重复富集 |
| target-cold 支线 | 独立靶点数量显著扩大，连续 affinity 与 Top-K 同时改善 |

---

## 第 25 页｜总结

### 页面只保留三句话

1. **数据与任务**：我们要解决的是“给定一种老药，对 384 个候选靶点排序”；437,248 条去重训练关系和 720×384 全矩阵是表示基础，不决定排序方向。
2. **当前证据**：S4 temporal 的 drug-macro AUPRC 为 0.8251；任务对齐的 V10 在 KiRHub strict-unreported 上 drug-macro AUPRC 0.3978、Recall@10 0.505，并有 Lazertinib→ERBB4/AXL 等 `rank/384` 外部强命中。
3. **边界与下一步**：V6 双向头与 head-only FULL_FIT 已完成，时间测试点估计改善但 CI 跨 0；下一步应扩大独立 drug queries、做跨靶点结构校准和前瞻湿实验，而不是解冻 backbone 追逐小样本。

### 结束语

> 方向纠正后，我们不仅划清了任务，也完成了第一版双向 V6；它在同一药物内跨靶点排序上出现可重复趋势，但下一步必须用更多独立药物 query 和前瞻实验把趋势变成可靠结论。

---

# 附录建议

## A1｜完整训练标签与损失权重

- BCE：V5 pair-level 主任务
- target ranking weight：0.12（同一靶点下排药）
- affinity weight：0.06
- affinity ranking weight：0.08
- contrastive weight：0.05
- drug ranking/listwise/observation：V5 当前权重 0
- 说明 affinity-only 行不参与 BCE
- 结论：上述权重解释了为什么 V5 输出是 target-centric；V6 不能将 target 与 drug 字样对调，需重新定义 query batches、负样本竞争集、两个方向头的 loss 和早停指标

### 旧 dual-query 屏幕的可比性审计

- 架构：单 `final_logit`，没有 D→T/T→D 独立 heads
- sampler：每个 epoch 将每行随机分配给 target 或 drug 一侧，不是两套 query-first batches
- loss：target-rank 0.12，drug-rank 0.08，listwise 0，不是对称的 pairwise + listwise
- selection：composite = 0.50 micro + 0.30 target-macro + 0.20 drug-macro
- 特征差异：dual checkpoint `structure_input_dim=0`，正式 S5 baseline `structure_input_dim=19`
- 结论：该结果证明“在当时设置下机械加 drug-rank 没有晋级”，不证明“双向模型不可行”

## A2｜六 checkpoint 的最佳轮数和开发值

| family | seed | selected epoch | completed epoch | best selection value |
|---|---:|---:|---:|---:|
| conservative | 20260816 | 9 | 14 | 0.7690 |
| conservative | 20260817 | 8 | 13 | 0.7795 |
| conservative | 20260820 | 11 | 16 | 0.7577 |
| recall | 20260816 | 11 | 16 | 0.7712 |
| recall | 20260817 | 8 | 13 | 0.7835 |
| recall | 20260820 | 6 | 11 | 0.7433 |

## A3｜冷热启动定义

- drug-warm：精确药物实体在对应 checkpoint 的训练关系中出现
- target-warm：精确 target identity 在训练关系中出现
- relation-held-out：两个实体均 warm，但 exact pair 未出现
- drug-entity-cold：精确实体未见；必须另外报告最近结构近邻
- scaffold-cold：骨架簇隔离，强于 entity-cold
- target-homology-cold：同源簇隔离，强于 exact-ID target-cold
- double-cold：药物和靶点两侧都按预设规则隔离

## A4｜S4 时间切分与泄漏审计

- train pool through 2022：76,965
- test first-seen 2023–2025：7,840
- mixed/missing era excluded：1,869
- quarantine：1
- 最终 test：7,839
- 83 个有双类标签的 drug groups
- drug-macro AUROC / AUPRC：0.7282 / 0.8251
- micro-AUROC / AUPRC：0.7791 / 0.8325
- 外部案例标签未用于模型选择

## A5｜S1/S3 bootstrap 与 baseline

- S1 对 target prior、ConPLex、Morgan similarity、DTIAM 的 paired target-cluster bootstrap
- S3 对 DTIAM 的差距与限制
- 强调 cluster bootstrap，而不是 row bootstrap

## A6｜BindingDB target-cold affinity 结果

- 325 rows，8 targets
- macro Spearman：0.040–0.155，均值 0.076
- NDCG：均值 0.852
- strong-vs-weak AUPRC：均值 0.739
- 结论：粗强弱富集尚可，连续 affinity 排序较弱

## A7｜完整案例表

包括：

- Lazertinib→ERBB4 / AXL / BTK
- Repotrectinib→AXL / PLK4
- Gilteritinib→KDR / FLT1 / PDGFRA / CSF1R
- Ripretinib→FLT3 / CSF1R / FLT1 / BRAF
- Deucravacitinib→FGFR4
- Lorlatinib→PLK4
- Tepotinib–IRAK1/4
- SIRT3、SEPHS2、VPS37C 等失败案例

每个案例必须同时报告：query 方向、rank、候选数、模型版本、冷热类型、exact-pair 状态、外部 assay、年份、是否参与训练/选模、局限。主案例必须使用 drug query 的 `rank/384`。

## A8｜结构研究的完整消融

- 19D target context
- 47D label-free structure package
- low-rank FiLM interaction
- experimental pocket graph
- GNINA target-calibrated diagnostics
- Boltz2 仅作为条件性复核，不作为生产分数

## A9｜对外主张白名单与禁用表述

### 可以说

- 构建了 720 种老药 × 384 靶点的完整候选矩阵；实际主任务是固定药物后对 384 个靶点排序。
- 最新 V5 pair backbone 使用 437,248 条去重训练行和六 checkpoint 共识，但其生产 fusion 仍是 target-centric。
- S4 冻结时间基准为 7,839 rows；drug-macro AUPRC 0.8251，micro-AUPRC 0.8325。
- 冻结 V10 drug-centric 在 KiRHub strict-unreported 上为 2,823 pairs、drug-macro AUPRC 0.3978、Recall@10 0.505。
- 外部 KiRHub 审计中，Lazertinib→ERBB4/AXL 分别为 6/384 和 7/384，Repotrectinib→AXL 为 9/384。

### 不可以说

- “模型已经解决新靶点或双冷启动。”
- “0.8325 表示从 720 种药中找对的成功率。”
- “3/720 表示给定 Repotrectinib 后 PLK4 排第 3。”
- “14/720 表示给定 Deucravacitinib 后 FGFR4 排第 14。”
- “V5 score 是结合概率或临床有效率。”
- “当前生产模型使用了 ConPLex、GNINA、Boltz2 或 candidate pose。”
- “Repotrectinib–PLK4 或 Lorlatinib–PLK4 是模型在论文发表前首次发现。”
- “Deucravacitinib 是完全无近邻的 scaffold-cold。”
- “Tepotinib–IRAK1/4 是成功案例。”

## A10｜预备问答

### Q1：为什么全面补数据后一些结果反而下降？

新增关系更长尾、更难，并包含冲突标签；同时排序是相对竞争，不能用旧 capped 高置信样本的成绩代表全面数据。

### Q2：为什么 S4 AUPRC 0.8325，但 27 个未来 query 表现较弱？

前者是 7,839 个 observed pairs 的时间分类基准，其中与主方向对齐的 drug-macro AUPRC 是 0.8251；后者是另一模型在 target→720 drugs 上的 27-query 反向 dense retrieval。两者测试集、候选集、模型和查询方向都不同。

### Q3：模型是双输入，为什么不能直接把排名方向反过来？

pair head 可以计算任意 drug–target 分数，但“谁和谁比”由 sampler、ranking loss、normalization、fusion 和评测分组共同决定。V5 在每个 target 内学习和选模，不保证不同 target 的分数已经可比。

### Q4：那么能否直接训一个双向模型？

可以，而且这是建议方案。但应使用一个共享 pair backbone、一个 pair evidence anchor 和两个轻量方向 residual heads；D→T 是 primary，T→D 是 auxiliary。两头使用各自的 query batches、listwise loss、校准和排名输出，而不是共用一个未校准 logit。

### Q5：模型是否使用口袋？

使用 19D target-level pocket context，但没有在生产主排序中使用 candidate-specific pose 和 ligand–residue cross geometry。

### Q6：为什么不继续训练？

V5 在原 target-centric 开发目标上，最佳 epoch 后训练损失继续下降而开发指标下降，继续同样的训练只会增加过拟合。现在需要的不是在原目标上多跑 epoch，而是加入方向分头、新的 query-level validation 和损失后重新训练。

### Q7：上市药为什么仍然是 cold？

上市状态属于药政属性；warm/cold 定义取决于精确实体是否在特定 checkpoint 的训练关系中出现。

### Q8：现在最适合做什么实验？

固定一种 warm old drug，优先其 warm-target、exact-relation-absent 的 Top candidates，并配置同一药下的 matched negative targets；先做生化 binding/enzymatic，再做细胞 target engagement。

---

# 制图与素材准备清单

## 必须制作的主图

1. 720 × 384 稀疏关系矩阵示意
2. 509,172 → 437,248 数据漏斗
3. 共享 pair backbone + pair evidence/D→T/T→D 三输出架构图
4. 三条证据线泳道图
5. S1–S5 drug-macro AUPRC 柱状图，micro-AUPRC 作辅助标记
6. V5 单头、旧 dual-query 与 V6 双向分头的任务对比图
7. 六 checkpoint 训练损失/开发值曲线
8. V10 KiRHub 的 per-drug Recall@K / AP 分布
9. KiRHub 外部 drug-macro PR 或 enrichment 图
10. Lazertinib→ERBB4/AXL 的 384-target 排名卡
11. Repotrectinib→AXL 成功与 PLK4 方向分歧对照
12. DTA 与 pair-specific structure 分层流程图

## 不建议制作的图

- 可以画 drug-centric target rank 矩阵，但不能把未校准的跨靶点 score 颜色解释成 affinity。
- 不把 276,480 个分数都画成不可读的大矩阵。
- 不用单个成功案例代替总体评测。
- 不把 AUPRC、Hit@K、percentile 放在同一纵轴。

## 视觉标记建议

- 蓝色：内部冻结 benchmark
- 绿色：独立或外部支持
- 橙色：development / exploratory
- 红色：失败或限制
- 灰色：FULL_FIT coverage、非独立证据

---

# 关键数字速查表

| 项目 | 正式数字 |
|---|---:|
| FULL_FIT rows | 437,248 |
| binary positives / negatives | 331,381 / 96,089 |
| affinity-only rows | 9,778 |
| unique drugs / targets | 296,108 / 843 |
| deployment scores | 276,480 |
| warm/cold drugs | 498 / 222 |
| warm/cold targets | 359 / 25 |
| double-warm / drug-cold / target-cold / double-cold pairs | 178,782 / 79,698 / 12,450 / 5,550 |
| exact known relations in deployment matrix | 5,992 |
| E0 S4 rows / prevalence | 7,839 / 0.5627 |
| E0 S4 drug-macro / micro-AUPRC | **0.8251 / 0.8325** |
| E0 S4 drug groups with both classes | 83 |
| V10 KiRHub all rows / positives / prevalence | 8,058 / 1,041 / 0.1292 |
| V10 KiRHub all drug-macro AUPRC / Recall@10 | 0.5399 / 0.4423 |
| V10 KiRHub strict rows / positives / prevalence | **2,823 / 202 / 0.0716** |
| V10 KiRHub strict micro / drug-macro AUPRC | **0.1858 / 0.3978** |
| V10 KiRHub strict drug-macro Recall@5 / 10 / 20 | 0.391 / 0.505 / 0.677 |
| Lazertinib→ERBB4 / AXL | **6/384 / 7/384**；98.98% / 81.07% inhibition |
| Repotrectinib→AXL | **9/384**；77.77% inhibition |
| Repotrectinib→PLK4 | V10 **373/384**；V5 反向 3/720；83.49% inhibition |
| Deucravacitinib→FGFR4 | V10 **179/384**；V5 反向 14/720；92.02% inhibition |
| Lorlatinib→PLK4 | V10 372/384；V5 反向 4/720 |
| Tepotinib→IRAK1 / IRAK4 | V10 369/384 / 338/384 |
| V5 2024–2025 target queries / Hit@36 | 27 / 22.2%（反向辅助） |
| V5 recovered relation test | 37 rows；AUPRC 0.933；AUROC 0.881（反向辅助） |

---

# 关键产物与数据来源

- 训练数据清单：[`COMPREHENSIVE_TRAINING_MANIFEST_V1.json`](../outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json)
- V5 生产摘要：[`CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json`](../outputs/biomaster_comprehensive_consensus_720x384_v5/CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json)
- 六 checkpoint 清单：[`FULL_FIT_CHECKPOINT_MANIFEST_V5.json`](../outputs/biomaster_comprehensive_consensus_720x384_v5/FULL_FIT_CHECKPOINT_MANIFEST_V5.json)
- V5 生产分数：[`CONSENSUS_FULL_FIT_720X384_SCORES_V5.csv.gz`](../outputs/biomaster_comprehensive_consensus_720x384_v5/CONSENSUS_FULL_FIT_720X384_SCORES_V5.csv.gz)
- V5 关系审计：[`COMPREHENSIVE_RELATION_AUDIT_SUMMARY_V5.json`](../outputs/biomaster_comprehensive_consensus_720x384_v5/COMPREHENSIVE_RELATION_AUDIT_SUMMARY_V5.json)
- V10 drug-centric 方向冻结摘要：[`BIOMASTER_DIRECTIONAL_APPLICATION_SUMMARY_V2.json`](../outputs/old_drug_target_sota_v1/biomaster_directional_application_v2/BIOMASTER_DIRECTIONAL_APPLICATION_SUMMARY_V2.json)
- V10 KiRHub 总体与 strict 指标：[`OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_METRICS_V10.csv`](../outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v10/OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_METRICS_V10.csv)
- V10 KiRHub 逐 pair 结果与 rank/384：[`OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_V10.csv`](../outputs/evidence_routing_compute_execution_20260808_v1/leakage_safe_ranker_v10/external_evaluation/OLD_DRUG_LEAKAGE_SAFE_EXTERNAL_KIRHUB_V10.csv)
- E0 S1–S5 汇总：[`BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json`](../outputs/biomaster_odti_unified_e0_summary_v1/BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json)
- S4 五种子结果：[`V2_MULTI_SEED_SUMMARY.json`](../outputs/old_drug_target_sota_v1/biomaster_odti_v2_s4_esm2_formal/V2_MULTI_SEED_SUMMARY.json)
- S4 冻结切分：[`FROZEN_SPLIT_ROLE_MANIFEST_V1.csv`](../outputs/old_drug_target_sota_v1/benchmark_splits_v1/FROZEN_SPLIT_ROLE_MANIFEST_V1.csv)
- S4 baseline：[`BASELINE_METRICS_V1.json`](../outputs/old_drug_target_sota_v1/baseline_results_v1/S4_FIRST_SEEN_TEMPORAL_2023_2025/fixed_split/BASELINE_METRICS_V1.json)
- 旧 dual-query S5 短屏：[`V2_MULTI_SEED_SUMMARY.json`](../outputs/old_drug_target_sota_v1/biomaster_odti_v2_s5_dualrank_2seed_screen_v2/V2_MULTI_SEED_SUMMARY.json)
- V6 三种子 Stage A 与 bootstrap：[`ENSEMBLE_STAGE_A_SUMMARY_V6.json`](../outputs/biomaster_bidirectional_v6_stage_a_dense/ENSEMBLE_STAGE_A_SUMMARY_V6.json)
- V6 seed16 Stage A 训练历史：[`STAGE_A_TRAINING_HISTORY_V6.csv`](../outputs/biomaster_bidirectional_v6_stage_a_dense/seed_20260816/STAGE_A_TRAINING_HISTORY_V6.csv)
- V6 2024–2025 三种子集成候选排序：[`ENSEMBLE_DENSE_D2T_TEST_2024_2025_ALL_CANDIDATES_V6.csv.gz`](../outputs/biomaster_bidirectional_v6_stage_a_dense/ENSEMBLE_DENSE_D2T_TEST_2024_2025_ALL_CANDIDATES_V6.csv.gz)
- V6 head-only FULL_FIT seed16：[`FULL_FIT_SUMMARY_V6.json`](../outputs/biomaster_bidirectional_v6_full_fit/seed_20260816/FULL_FIT_SUMMARY_V6.json)
- V6 FULL_FIT 720×384 主排序：[`BIDIRECTIONAL_V6_FULL_FIT_720X384_SCORES.csv.gz`](../outputs/biomaster_bidirectional_v6_720x384/BIDIRECTIONAL_V6_FULL_FIT_720X384_SCORES.csv.gz)
- V6 展示案例 rank/384：[`BIDIRECTIONAL_V6_SHOWCASE_CASE_RANKS.csv`](../outputs/biomaster_bidirectional_v6_720x384/BIDIRECTIONAL_V6_SHOWCASE_CASE_RANKS.csv)
- Fusion calibration：[`FUSION_CALIBRATION_SUMMARY_V2.json`](../outputs/biomaster_comprehensive_consensus_fusion_calibration_v5/FUSION_CALIBRATION_SUMMARY_V2.json)
- Pair-specific GNINA：[`CURRENT_CASE_GNINA_SUMMARY_V1.json`](../outputs/current_case_pair_specific_gnina_v1/CURRENT_CASE_GNINA_SUMMARY_V1.json)
- Experimental pocket graph：[`EXPERIMENTAL_TEMPLATE_GRAPH_PILOT_AUDIT_V1.json`](../outputs/biomaster_odti_experimental_template_graph_pilot_v1/pilot_audit_v1/EXPERIMENTAL_TEMPLATE_GRAPH_PILOT_AUDIT_V1.json)

---

# 制作 PPT 与下一轮验证前的固化任务

1. 从现有 V10 逐 pair artifact 导出一张主任务案例表，主键为 `drug entity + target identity + model version`，固化 `rank/384`、local-unreported 状态、KiRHub inhibition、匹配规则与哈希。优先包含 Lazertinib→ERBB4/AXL、Repotrectinib→AXL，并显式保留 Repotrectinib→PLK4 等失败。
2. 将 V5 所有 `rank/720` 结果在 artifact 与图表中加上 `query_direction=target_to_drugs`；将所有 V5 reciprocal ranks 加上 `exploratory_uncalibrated_cross_target=true`，防止再次混用。
3. 双向 V6 的 D→T 384-target 时间合同与 T→D 辅助非退化评测已经落地；下一轮需新增未被本轮查看过的 drug-query 外部集，并在更多独立药物上预注册 bootstrap 门槛。KiRHub 继续保持只读。

内部 PPT 现在可以使用已冻结的 V10 外部 artifact，但对外版在宣称“新发现”前仍需对每个展示关系做独立文献时间线审计。

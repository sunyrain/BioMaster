# BioMaster 训练数据与标签定义

> 当前口径：2026-09-01。项目使用437,248条综合关系训练通用模型；86,674是历史平衡校准/评估表，不再称为“当前训练数据量”。完整任务和模型合同见`CURRENT_PROJECT_CONTRACT_ZH.md`。

## 一页汇报版

| 数据层 | 规模 | 用途 |
|---|---:|---|
| ChEMBL37来源记录 | 509,172 | 标签聚合前的直接生化activity/pair记录 |
| ChEMBL37可解析关系 | 426,939 | 综合训练主体 |
| BindingDB Ki/Kd补充 | 9,778 | affinity-only辅助关系 |
| 经审计恢复的关系 | 914 | 补充覆盖；与主体重复的383条不重复训练 |
| 综合训练表 | **437,248** | 当前通用模型与FULL_FIT的去重、特征可解析训练关系 |
| 历史平衡评估表 | **86,674**（86,673行特征可解析） | 每靶点/每标签最多150条的冻结S1–S5 benchmark |

综合训练表覆盖296,108个药物实体和843个靶点。训练endpoint混合Ki、Kd、结合型IC50、明确inactive及可审计的直接关系语义，因此模型学习的是**直接生化药物–靶点关系/活性排序**，不是纯定量亲和力回归。

标签和声明边界固定为：

```text
observed positive / active       → 可作为正类或排序正证据
observed weak / inactive         → 在满足来源合同时可作为负类
Ki/Kd affinity-only              → 保留连续亲和力语义，不强制制造二分类负例
grey / conflicting / unresolved  → 不进入普通二分类项
unreported pair                  → unknown，绝不是negative
model score                      → 候选排序证据，不是物理亲和力或临床疗效概率
```

## 1. 综合训练表如何得到437,248

权威来源是`outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json`和FULL_FIT审计摘要。

| 步骤 | 行数 | 说明 |
|---|---:|---|
| ChEMBL37标签过滤前 | 509,172 | 原始来源记录 |
| 标签合格、特征过滤前 | 427,526 | 通过activity语义与冲突规则 |
| ChEMBL37特征可解析 | 426,939 | 587条分子特征失败被记录 |
| BindingDB direct Ki/Kd | 9,778 | 保留affinity-only语义 |
| recovered关系 | 914 | 其中383条与综合主体重复 |
| 合并审计行 | 437,631 | 仍含用于审计的重复行 |
| 最终FULL_FIT去重关系 | **437,248** | 当前实际训练合同 |

最终FULL_FIT还固定记录：296,108个唯一药物、843个唯一靶点。训练、评分和汇报若出现其他规模，必须标明它属于哪个历史切片，不能替换437,248。

## 2. 86,674到底是什么

86,674是早期ChEMBL37平衡校准表：覆盖428个靶点、62,488个compound、45,983个positive和40,691个negative/inactive；每个target × label最多150条，并进行scaffold-balanced选择。一个分子Morgan特征失败，所以用于模型的indexed rows是86,673。

它有两个用途：

1. 构造S1 scaffold-cold、S2 target-homology-cold、S3 double-cold、S4 temporal和S5 old-drug-entity-cold等冻结benchmark；
2. 复现早期模型与公开基线的同数据比较。

它不是当前综合训练表，也不是数据库全部观测。今后统一称“历史平衡评估表”或“86,674 capped benchmark”。

## 3. 二分类标签语义

历史平衡表将同一drug–target pair的多条实验记录聚合后判定。常用pChEMBL边界为：

```text
mean(pChEMBL) >= 6.0，且无正负冲突       → positive
mean(pChEMBL) <= 5.0或明确inactive，且无冲突 → negative_or_inactive
5.0 < mean(pChEMBL) < 6.0                  → grey，不进入主二分类
强阳性与inactive/弱活性证据冲突             → conflicting，不进入主二分类
没有合格观测                                → unknown
```

同一pair的多个assay可能存在差异，所以判定必须使用聚合值、关系符号、assay类型和冲突审计，不能只取一条最强记录。综合训练表还包含affinity-only和来源特定关系；它们按照各自mask进入多任务损失，不应被强行压成同一种binary label。

## 4. 为什么unknown不能当negative

数据库未报告可能意味着尚未测试、未公开、尚未收录、物种/构建体不一致或在其他来源中报告。只有经过合格实验观察的弱/无活性关系才能作为负证据。若把全空间未报告pair批量设为0，模型会主要学习文献密度和历史测试偏好，并夸大随机切分性能。

因此全项目保持：

```text
observed weak/inactive → eligible negative under its source contract
unreported             → unknown
unknown                → excluded from ordinary BCE negatives
```

## 5. 训练、评价和部署不是同一张表

| 层级 | 候选或关系 | 标签状态 | 用途 |
|---|---:|---|---|
| 综合训练 | 437,248条关系 | 按任务mask具有binary/affinity/关系证据 | 拟合通用pair backbone和方向头 |
| 冻结S1–S5 | 来自86,674 benchmark的隔离测试切片 | 有观测标签 | 衡量指定冷启动与时间泛化 |
| 当前部署核心 | 720 × 384 = 276,480 pair | 多数unknown | 每种老药内部产生`rank/384` |
| 下一版主空间 | 720 × 450 = 324,000 pair | 多数unknown | 补完66个靶点后产生`rank/450` |
| KIRHub严格审计 | 2,823个1 μM实测pair | 202个≥70%功能抑制pair | 回顾性功能迁移审计，不是亲和力测试 |

S5的Recall@K是在冻结observed-pair测试切片内计算；KIRHub Recall@K是在每个药物实际测量并映射成功的靶点子集内计算；个案`rank/384`才是完整384靶点候选轴。三者分母不同，不能互换。

## 6. 模型输入和输出边界

通用模型使用冻结/预训练的药物与蛋白表示、关系特征和可用性mask。target-level pocket context用于补充受体结构环境；它不是特定药物–口袋pose。pair-specific docking、接触图或局部几何目前属于Top-K候选后的结构复核，不是通用主排序已验证的输入。

模型原始logit、百分位和融合分数都只用于排序。未经独立校准和物理实验，不能解释为Kd/Ki、结合概率、抑制率或临床成功概率。

## 7. KIRHub必须单列endpoint

KIRHub提供HotSpot单浓度1 μM功能抑制读数。`inhibition >= 70%`是项目定义的strong functional hit，不是Ki、Kd或直接亲和力。

- 对冻结通用头：KIRHub是已反复查看的回顾性功能迁移审计，不是新的确认性外部集。
- 对KIRHub专项头：KIRHub已经参与训练和选择，只能报告药物骨架冷启动OOF开发结果。
- 对外汇报：必须写“1 μM功能抑制”，不能简写为“亲和力命中”。

## 8. 权威产物

- 当前机器合同：`configs/biomaster_current_contract_v1.json`
- 综合训练清单：`outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json`
- FULL_FIT训练行数审计：`outputs/biomaster_bidirectional_v6_full_fit/seed_20260816/FULL_FIT_SUMMARY_V6.json`
- 历史平衡表：`outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz`
- 药物中心评估摘要：`outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json`
- 靶点空间摘要：`outputs/target_discovery_scope_ch37_v3/TARGET_DISCOVERY_SCOPE_SUMMARY_V3.json`

任何新统计先写入机器可读审计摘要，再更新合同和文档；不再从PPT图片反推数字。

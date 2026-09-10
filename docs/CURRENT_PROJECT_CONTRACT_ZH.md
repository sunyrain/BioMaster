# ReTargetMap 当前项目唯一口径

> 生效日期：2026-09-01。若其他文档、PPT 或旧产物与本页冲突，以本页和 `configs/biomaster_current_contract_v1.json` 为准。内部版本号只用于复现，不进入项目汇报。

## 一句话目标

给定一种已上市或已有充分药理信息的老药，在可比较的候选靶点中建立潜在作用靶点谱，并将前排候选交给证据复核和湿实验验证。

主方向始终是 **老药 → 靶点**。靶点 → 老药只保留为反向辅助证据，不与主结果混报。

## 只保留三层靶点空间

| 层级 | 数量 | 当前处理方式 |
|---|---:|---|
| 已评分比较核心 | 384 | 当前所有完整矩阵、模型比较和个案排名使用 `rank/384` |
| 下一版主生产空间 | 450 | 包含上述384；其余66补齐特征、推理和跨通道校准后，才发布 `rank/450` |
| 完整非GPCR登记目录 | 745 | 用于不丢靶点和分路审计，不是统一排名分母 |

745的路由关系固定为：450个主生产靶点 + 42个特殊体系直接证据靶点 + 8个高新颖性探索靶点 + 245个仅登记靶点。42和8均单独输出，不能和主榜直接拼接。

当前已完成的矩阵是720种老药 × 384个靶点，共276,480对。下一版主矩阵是720 × 450，共324,000对，但在66个新增靶点完成统一评分前不能宣称已有全局 `rank/450`。

## 训练数据只保留两个名称

| 名称 | 规模 | 唯一用途 |
|---|---:|---|
| 综合训练表 | 437,248条去重、特征可解析关系；296,108个药物；843个靶点 | 当前通用模型训练与FULL_FIT |
| 历史平衡评估表 | 86,674个有标签pair；其中86,673行可解析特征 | 旧的每靶点/每标签最多150条校准表及S1–S5冻结评估，不代表当前全量训练规模 |

综合训练表来自ChEMBL37直接生化活性/关系证据、BindingDB Ki/Kd affinity-only补充以及经审计恢复的关系。它混合Ki、Kd、结合型IC50和明确inactive等语义，因此任务应表述为“直接生化药物–靶点关系/活性排序”，不是纯定量亲和力回归。

数据库未报告的pair始终是unknown，不自动作为negative。

## 模型只保留三种角色

| 角色 | 当前定义 | 是否对外作为独立模型 |
|---|---|---|
| 通用主系统 | ReTargetMap药物中心独立排序；在同一药物的384个靶点内产生最终排名，不依赖DTIAM推理 | 是，统一称“ReTargetMap老药靶点谱检索框架” |
| FULL_FIT方向头 | 三种子全量关系重训后的神经候选分数，为通用系统提供候选与辅助证据 | 否；没有重训后的无偏性能估计 |
| 反向辅助输出 | 共享pair backbone上的target→drug方向头 | 否；只用于候选复核，不与主方向结果混报 |

DTIAM、ConPLex和DrugCLIP统一归为对照或外部证据。允许DTIAM参与的融合结果必须明确写“统一系统/含DTIAM”，不能冒充ReTargetMap独立模型。

## 评价只保留三条证据线

1. **冻结S5药物实体冷启动**：评价未见药物的已观测pair排序。主指标是drug-macro AUPRC。独立头为0.7521，DTIAM为0.7365；点估计领先，但置信区间跨0，不能宣称已确认全面优越。
2. **KIRHub严格回顾性审计**：2,823个1 μM功能抑制实测pair、202个强抑制pair、78个药物；只有33个药物同时含正负样本，可计算drug-macro指标。这里的Recall@K在“每个药物实际测量并映射成功的候选子集”内计算，不是完整384靶点Recall@K，也不是亲和力评价。
3. **完整空间候选排名**：对720 × 384全矩阵给出每药`rank/384`，用于候选优先级和案例展示。FULL_FIT结果属于生产候选评分，不是独立测试。

KIRHub标签已被项目反复查看，因此对通用系统只称“回顾性功能迁移审计”，不作为新的untouched external test。

## 当前唯一产物入口

- 机器可检验合同：`configs/biomaster_current_contract_v1.json`
- 训练清单：`outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_MANIFEST_V1.json`
- FULL_FIT审计：`outputs/biomaster_bidirectional_v6_full_fit/seed_20260816/FULL_FIT_SUMMARY_V6.json`
- 720 × 384评分摘要：`outputs/biomaster_bidirectional_v6_720x384/BIDIRECTIONAL_V6_FULL_FIT_720X384_SUMMARY.json`
- 通用药物中心排序与评估摘要：`outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json`
- 靶点路由摘要：`outputs/target_discovery_scope_ch37_v3/TARGET_DISCOVERY_SCOPE_SUMMARY_V3.json`

## 命名和维护规则

- 外部汇报只用“ReTargetMap老药靶点谱检索框架”和“通用主系统”，不用V5/V6/V10等内部代号。
- 每个指标名必须同时带方向、候选分母或测量子集、切分和endpoint。例如：`drug→target / measured-subset / KIRHub-1μM / drug-macro Recall@10`。
- 新模型若没有在冻结主指标上通过预设门槛，只能进入探索区，不能替换当前入口。
- 新数据必须声明endpoint、unknown处理、实体去重规则、时间边界和是否参与过模型选择。
- 旧脚本、旧输出和旧PPT保留用于审计，但不得从默认README链接为“当前结果”。

## 当前可说与不可说

可以说：ReTargetMap已完成老药→靶点方向的完整384靶点排序；在冻结S5和KIRHub回顾性审计中，独立系统的药物宏AUPRC点估计高于DTIAM；450靶点主生产空间已定义，仍有66个待统一评分。

不能说：ReTargetMap已在全外部数据上统计学显著全面优于DTIAM；KIRHub验证了亲和力；当前已有rank/450或rank/745；FULL_FIT结果是独立测试；未报告pair是阴性。

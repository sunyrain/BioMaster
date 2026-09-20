# BioMaster 当前文档入口

更新：2026-09-20。项目分为已交付湿实验、现成模型比较、A受控重训三条线。历史报告中的“当前”“唯一”“下一步”和ETA只代表其日期，不覆盖本页指向的有效协议。

## 当前研究与执行

| 内容 | 入口 | 当前状态 |
|---|---|---|
| 架构、数据、权重 | [资源总表](BIOMASTER_DTI_ARCHITECTURE_DATA_WEIGHTS_20260920_ZH.md) | 保留A/B及官方资产；资产存在不等于完成独立验证 |
| 新增训练 | [A主线重训方案](BIOMASTER_DTI_A_ONLY_RETRAIN_PLAN_20260920_ZH.md) | A训练337,570、验证41,806、测试39,507对；主线45项、先30项；新增B训练0；尚未启动 |
| 官方权重比较 | [用户价值与公平比较](BIOMASTER_OFFICIAL_WEIGHTS_COMPARISON_20260920_ZH.md) | 五个已登记官方模型＋待确认官方任务权重的DTIAM；ReTargetMap及本地DTIAM A/B不参加；完整训练暴露审计待做 |
| 有效机器协议 | [ACTIVE_PROTOCOL.json](../configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json) | A协议及现成系统比较补充规则的路径与哈希 |
| 研究问题与证据 | [问题设计](BIOMASTER_DTI_RELIABILITY_RESEARCH_PLAN_20260918_ZH.md)、[结果解释](BIOMASTER_MODEL_RESULT_INTERPRETATION_20260920_ZH.md)、[架构审计](BIOMASTER_SEVEN_MODEL_ARCHITECTURE_AUDIT_20260919_ZH.md)、[分歧文献](BIOMASTER_MODEL_DISAGREEMENT_LITERATURE_20260918_ZH.md) | 既有标签已被查看；低相关不能直接说明错误或架构因果 |

现成模型按冻结的实际完成范围报告，明确全目录、共同覆盖和实测子集的不同分母；不把尚未完成的全矩阵当作已有结果。未测关系不作阴性。

## 湿实验交付与平台

- [SPR384最终实验表](../outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv)：384候选配对，参考对照另计；交付不等于实测结合。
- [对照身份修订](BIOMASTER_SPR384_FINAL_FDA_CONTROLS_20260910_ZH.md)、[亲和证据质量审查](BIOMASTER_SPR384_BINDINGDB_QUALITY_REVIEW_20260910_ZH.md)、[实验回填说明](BIOMASTER_SPR_RESULTS_UPLOAD_ZH.md)。
- [平台使用说明](BIOMASTER_EXPLORER_ZH.md)、[七模型登记](BIOMASTER_SEVEN_MODEL_CATALOG_20260916_ZH.md)、[DTIAM A网站替换记录](BIOMASTER_DTIAM_A_WEBSITE_UPGRADE_20260917_ZH.md)。本轮文档整理不调整实验名单或网站模型。

## 可复用的已完成结果

- [DTIAM A/B最终测试](BIOMASTER_DTIAM_FINAL_REVIEW_20260915_ZH.md)：已完成六套，9月13–15日进度稿归档。
- [A/B多任务结果](BIOMASTER_AB_MULTITASK_RESULTS_20260911_ZH.md)、[assay-aware结果](BIOMASTER_ASSAY_AWARE_RESULTS_20260911_ZH.md)：历史训练继续作为参照，不能假装与新共同输入协议完全一致。
- [9月6日模型交付](BIOMASTER_SELECTED_MODEL_20260906_ZH.md)、[原SPR筛选的规则复现](BIOMASTER_ORIGINAL_SPR384_POLICY_REPLAY_20260911_ZH.md)：保留历史模型和实验选择的溯源。
- [训练标签语义](BIOMASTER_TRAINING_DATA_LABEL_DEFINITION_ZH.md)、[全量清洗数据](BIOMASTER_FULL_TRAINING_DATASET_20260911_ZH.md)、[数据访问](DATA_ACCESS.md)：区分母库、任务标签、训练成员与已测阴性。

## 历史与后续研究

- [归档索引](archive/README.md)、[本轮文档整理记录](archive/superseded_20260920/README.md)。13份旧路线/阶段进度移入归档；6份历史脚本依赖报告原位标记。
- [v1准备报告](BIOMASTER_DTI_RESEARCH_PREPARATION_20260920_ZH.md)和[v1协议](protocols/DTI_RELIABILITY_PROTOCOL_20260920_ZH.md)保持冻结。原87项队列已由A主线替代，数据与权重清单仍复用。
- [9月1日生产合同](CURRENT_PROJECT_CONTRACT_ZH.md)、[阶段演进](BIOMASTER_PHASE_ROADMAP_AND_RESULTS_20260908_ZH.md)仅用于追溯；437,248关系、745登记等旧数字不代表新训练范围。
- [JEPA设计](BIOMASTER_PM_JEPA_DESIGN_20260914_ZH.md)、[教师模态与资源](BIOMASTER_PM_JEPA_MODALITIES_AND_RESOURCES_20260914_ZH.md)：后续方法线；不列为本轮已完成训练。

维护规则：优先查有效协议、资产清单和对应结果摘要；阶段进度结束后链接到完成报告。冻结材料不原地改写，用版本化补充和当前入口说明替代关系。

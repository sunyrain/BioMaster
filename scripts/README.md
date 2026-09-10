# 脚本入口与维护边界

## 当前实验与网站交付

- 网站：`run_explorer.py`、`manage_explorer_users.py`、`package_explorer_offline.py`；账号文件放在仓库外。
- 多方向疾病证据：`restore_disease_evidence_720x888_20260909.py`、`run_txgnn_all_diseases_720_20260909.py`、`fetch_ot888_disease_evidence_20260909.py`及对应组装、校验脚本。
- 384设计与导出：`screen_joint_720x384_20260909.py`、`build_joint384_comprehensive_20260909.py`、`export_spr384_final_experiment_table_20260910.py`、`finalize_spr384_fda_controls_20260910.py`；冻结候选与对照另计，历史设计不能覆盖当前交付。
- 亲和证据：`collect_affinity_refresh_20260910.py`、`build_affinity_evidence_refresh_20260910.py`、`evaluate_affinity_refresh_predictor_20260910.py`及对应报告脚本。
- 磁盘收束：`execute_outputs_retirement_20260910.py`只对应已完成的固定审批清单，禁止作为日常通用删除入口；原始审计脚本依赖清理前文件，当前不能原样重跑。

`scripts/` 保持扁平结构，因为部分训练入口通过脚本目录直接导入共享函数；贸然移动会改变现有checkpoint和运行命令的复现路径。脚本多不等于当前模型多：唯一项目合同是`configs/biomaster_current_contract_v1.json`，可用`validate_current_project_contract.py`检查。

## 通用模型复现链

KIRHub 与时间外推：`evaluate_v3_upgrade_kirhub_20260906.py`复测冻结 J/V3 和现有基线；时间链依次为`extract_v3_temporal_raw_20260906.py`、`prepare_v3_temporal_20260906.py`、`audit_v3_temporal_registry_age_20260906.py`、`train_v3_temporal_20260906.py`、`evaluate_v3_temporal_20260906.py`。先按原始文献年份截断再聚合标签，2021–2022 验证选轮数，最终从头训练至 2022 年；测试脚本要求模型与选择清单已冻结。获批年份审计在解封测试前固定594个早期获批药的保守子集。训练入口不读 2023–2025 标签，完成产物禁止原地重训。`summarize_v3_kirhub_temporal_20260906.py`核验并生成[报告](../docs/BIOMASTER_V3_KIRHUB_TEMPORAL_20260906_ZH.md)；无测试证据文件时会先运行对应自动测试。

旧V3增量升级与消融：`prepare_biomaster_v3_incremental_ablation.py`缓存固定底座的隐状态与分数；`train_biomaster_v3_incremental_ablation.py`执行首轮18组匹配消融；`train_biomaster_v3_old_relation_ablation.py`执行训练老药已知关系监督的15组消融；两阶段验证选择冻结后，由`evaluate_biomaster_v3_incremental_ablation.py`解封回顾性测试，`summarize_biomaster_v3_incremental_ablation.py`核验并生成[完整报告](../docs/BIOMASTER_V3_INCREMENTAL_ABLATION_20260906_ZH.md)和候选推理清单。全程保留旧底座，配置/来源变化会阻止复用训练产物。

真实项目老药双向测试：`prepare_old_drug_bidirectional_20260906.py`冻结720老药×384靶点、已知关系和训练接触范围；`evaluate_old_drug_bidirectional_20260906.py`评分并分别评估两个方向（`--evaluate-only`复用已保存分数）；`summarize_old_drug_bidirectional_20260906.py`核对来源哈希及支持集自身实体排除，导出可读排名和[结果报告](../docs/BIOMASTER_OLD_DRUG_BIDIRECTIONAL_TEST_20260906_ZH.md)。不重新拟合模型，DrugCLIP仅在同覆盖子集比较。

新一轮架构与数据组织入口：[`organize_biomaster_odti_v4_assets.py`](organize_biomaster_odti_v4_assets.py)只读取既有资产，生成独立V4分子/靶点索引、逐测量表、缓存补建队列和来源目录；不训练或修改生产模型。设计见[ODTI V4方案](../docs/BIOMASTER_ODTI_V4_ARCHITECTURE_AND_DATA_20260905_ZH.md)。

V4首轮开发训练链：`build_biomaster_odti_v4_features.py`补建公共预训练特征，`prepare_biomaster_odti_v4_experiment.py`固定历史开发协议与完整靶点面板，`train_biomaster_odti_v4.py`训练Morgan/BerMol/有符号证据及统一测试，`evaluate_biomaster_v3_on_v4_panel.py`复测旧V3对照，`summarize_biomaster_odti_v4.py`验证产物并生成[实测报告](../docs/BIOMASTER_ODTI_V4_R1_TRAIN_TEST_20260905_ZH.md)。本轮未包含DrugCLIP三维交互，未替换生产。

1. `build_biomaster_comprehensive_training_v1.py`：构建去重综合关系与特征索引。
2. `train_biomaster_comprehensive_balanced_v2.py`：target/label/scaffold 平衡训练与冻结评估。
3. `train_biomaster_bidirectional_v6.py`：训练药物→靶点主头和靶点→药物辅助头。
4. `refit_biomaster_bidirectional_v6_full_fit.py`：在冻结 epoch 合同下完成 FULL_FIT 方向头拟合。
5. `score_biomaster_bidirectional_v6_720x384.py`：生成 720 × 384 双向部署分数。
6. `summarize_biomaster_bidirectional_v6.py`：汇总开发集、时间集和 bootstrap 结果。

方向头 FULL_FIT 的采样必须与 Stage-A 保持一致。2026-09-01开发集消融选出的
新候选为药物→靶点每个药物 query 动态取4条已测记录，靶点→药物每个靶点
query取16条；冻结旧候选曾使用2/16。旧参数
`--rows-per-query` 仅作为同时覆盖两个方向的兼容入口；新实验应分别使用
`--d2t-rows-per-query` 与 `--t2d-rows-per-query`。采样消融必须使用
`--skip-test`，避免在模型选择时重复计算2024--2025测试集。

完整745靶点覆盖使用以下入口：

- `build_biomaster_target_registry_745_feature_store_v1.py`：复用384核心和综合训练特征，补齐全部745靶点的ProtBERT、ESM2及19维口袋上下文。
- `score_biomaster_bidirectional_720x745_v1.py`：生成720×745全覆盖分数及367/83/42/8/245路由内rank。
- `evaluate_biomaster_stage_a_720x745_v1.py`：在同一冻结2024--2025阳性上比较384/450/500/745候选空间；745结果只作压力测试。

KiRHub 专项训练脚本支持 `--out-dir`，专项 FULL_FIT 脚本支持
`--trained-dir` 与 `--out-dir`，因此审计性复训可写入独立目录而不覆盖当前产物。
专项 FULL_FIT 的轮数与融合权重从本次 OOF 产物读取，不再把历史的 190 轮和
0.50/0.20/0.30 权重写成通过条件。

上述内部版本号只用于复现。项目交付的默认药物中心排序由`build_biomaster_drug_centric_ranker_v1.py`统一汇总；`validate_current_project_contract.py`负责核对训练量、384/450/745靶点空间、S5和KIRHub口径。

完整重训若要保留旧训练包，设置`BIOMASTER_COMPREHENSIVE_PACKAGE`指向新的综合训练目录；平衡训练、方向头和FULL_FIT会沿同一环境变量读取relations、Morgan和manifest。最终ranker通过`--full-fit-deployment`与`--full-fit-summary`接收本次重训分数，避免覆盖旧720×384产物。

共享依赖包括 `train_biomaster_comprehensive_full_fit_v1.py`、`train_biomaster_deployment_augmented_v1.py`、`train_biomaster_bindingdb_affinity_augmented_v1.py`、`score_biomaster_deployment_augmented_720x384_v1.py` 和 `score_biomaster_full_fit_current_new_relations_v1.py`。这些文件均属于可复现依赖，不是可删除的临时脚本。

## 湿实验候选准备

- `build_retargetmap_experiment_candidate_pool_v1.py`：冻结首轮盲法比较的候选预池。主要推荐使用具有S5无偏依据的`independent_validation_rank_score`；DTIAM为外部对照，当前默认独立Borda单列为挑战组，真正FULL_FIT使用`ensemble_drug_to_target_logit`并只作探索支持。最终入选前仍需外部新颖性、实验可行性和化合物质控审查。

## 冻结对照与依赖

- `train_v10_leakage_safe_ranker.py`
- `evaluate_v10_leakage_safe_external.py`
- `audit_drug_centric_kirhub_v1.py`
- `audit_v10_nested_model_selection.py`
- `audit_v10_cold_start_generalization.py`
- `build_drug_centric_cross_target_v1.py`

这些入口产生对照分数或通用主系统的依赖证据，不单独称为当前生产模型。允许DTIAM参与的结果必须标注为“含DTIAM统一系统”。

## 激酶功能专项路由

- `build_kirhub_expanded_pretrained_features_v1.py`
- `train_kirhub_expanded_functional_adapter_v1.py`
- `refit_and_score_kirhub_expanded_functional_adapter_v1.py`

该路由学习的是KIRHub 1 μM功能抑制，不是亲和力；它独立于通用主系统，也不能用其内部OOF结果替换通用模型性能。

## 正式候选筛选与结构复核

正式 v4、ChEMBL37 target universe、已知 pocket atlas、GNINA/Boltz 和 evidence-routing 脚本继续保留，用于候选后的结构与证据复核。入口合同见 `docs/PRODUCTION_PIPELINE_V4_ZH.md` 与 `docs/V4_RUNBOOK_ZH.md`。

## 新脚本规则

工作区中的新 `.py` 默认被 `.gitignore` 视为探索脚本。只有满足以下条件才提升为正式源代码：

- 被当前文档、测试或正式入口引用；
- 输入/输出合同明确，未知关系不被静默当作负例；
- 通过语法检查和相关测试；
- 不把 checkpoint、全量模型输出或第三方代码一并提交。

已有 Git 跟踪文件不受该默认规则影响。废弃脚本优先依靠 Git 历史追溯，不在当前目录继续复制多个近似版本。

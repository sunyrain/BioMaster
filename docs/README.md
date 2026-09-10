# ReTargetMap 文档入口

**当前实验交付与维护（2026-09-10）**：最终为384候选、112另计对照，已转交湿实验。入口：[最终表](../outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv)、[对照修订](BIOMASTER_SPR384_FINAL_FDA_CONTROLS_20260910_ZH.md)、[近期亲和证据与384质量](BIOMASTER_SPR384_BINDINGDB_QUALITY_REVIEW_20260910_ZH.md)、[实验回填](BIOMASTER_SPR_RESULTS_UPLOAD_ZH.md)、[磁盘清理执行记录](BIOMASTER_DISK_CLEANUP_COMPLETED_20260910_ZH.md)。下方32/64靶点及512配对设计按日期保留为前期研究记录。

**TxGNN/BioPathNet候选覆盖核对（2026-09-09）**：[覆盖、共同划分与尚未完成的性能比较](BIOMASTER_TXGNN_BIOPATHNET_COMPARISON_20260909_ZH.md)。108/120种药可映射；发现ponesimod立体化学身份待复核。

**64靶点/512对计算设计（2026-09-09）**：[候选设计与证据缺口](BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md)。已生成内部及角色盲化清单；未实验放行，不替代原32靶点冻结包。

**各阶段路线与结果总览（2026-09-08）**：[阶段梳理](BIOMASTER_PHASE_ROADMAP_AND_RESULTS_20260908_ZH.md)。覆盖历史物理筛选、ODTI基准、V3/J、selected交付、结构迁移与SPR候选；最新ContextPocket已于第25步触发结构保持门停止，训练未完成。本文总览不改变冻结生产合同或实验名单身份。

**本轮模型交付入口**：[BIOMASTER_SELECTED_MODEL_20260906_ZH.md](BIOMASTER_SELECTED_MODEL_20260906_ZH.md)。包含78次开发训练的选型依据、实际简化架构、时间/KIRHub三种子回归、独立推理包及已知退步；完整登记目录与既有生产报告仍遵循下方合同。

本目录的**默认入口**只认当前口径和仍在执行的正式合同。部分旧材料为维持历史链接暂留顶层，但已在`archive/DEPRECATED_TOP_LEVEL_SNAPSHOTS_20260901_ZH.md`中逻辑归档，不再提供当前数字。

## 当前汇报口径

- `CURRENT_PROJECT_CONTRACT_ZH.md`：**唯一当前入口**。若其他文档、PPT或旧结果说明与它冲突，以该文件及`configs/biomaster_current_contract_v1.json`为准。
- `BIOMASTER_COMPLETE_PPT_OUTLINE_20260825_ZH.md`：当前最完整的大纲。主任务统一为“给定老药，在450个路由后可比较的主生产靶点中排序”；既有384靶点结果保留为已评分比较核心，不能改写分母。
- `target_discovery_scope_ch37_v3/`（位于 `outputs/`）：745靶点完整登记目录、450主生产清单、42特殊体系分支、8高新颖性探索清单及相应pair清单。745不是统一rank分母。
- `presentations/`：25 页完整学术汇报图片、4 页项目汇报插页和原始 PPTX。

## 当前方法与结果合同

- [BIOMASTER_UNIFIED_INTERACTION_FIRST_ROUND_20260906_ZH.md](BIOMASTER_UNIFIED_INTERACTION_FIRST_ROUND_20260906_ZH.md)：统一全局、原子—残基和映射口袋CA几何的新主干实现；五组受控对照、时间训练及回归状态，研究候选不改变生产合同。
- [BIOMASTER_CURRENT_ARCHITECTURE_TESTS_AND_STRENGTHENING_20260905_ZH.md](BIOMASTER_CURRENT_ARCHITECTURE_TESTS_AND_STRENGTHENING_20260905_ZH.md)：分版本描述实际架构，汇总S1–S5、功能、亲和力、完整面板及V3表现，并给出预训练、交互和学习算法升级重点；含历史结果的证据边界，不修改冻结生产合同。
- `BIOMASTER_TRAINING_DATA_LABEL_DEFINITION_ZH.md`：训练数据、正负标签、未知关系和 affinity-only 行的语义。
- `BIOMASTER_DRUG_TO_TARGET_VS_DTIAM_20260827_ZH.md`：当前药物中心冻结评估与DTIAM对照；KIRHub只按功能抑制回顾性审计解释。
- `BIOMASTER_RETRAIN_RETEST_20260901_ZH.md`：43.7万关系三种子重训、方向采样消融、745靶点分层压力测试及新生产候选结论；候选尚未替换冻结当前合同。
- `BIOMASTER_SYSTEM_DATA_METRICS_OVERVIEW_20260901_ZH.md`：系统角色、模型输入、全部主要数据来源、标签合同、S1–S5/KIRHub/密集检索评价分母和指标定义的统一总览。
- `reports/RETARGETMAP_CURRENT_PROJECT_HANDBOOK_20260901_ZH.pdf`：ReTargetMap项目、数据来源、筛选空间、指标分母、Top 10结果和首轮湿实验方案的21页中文详解PDF。
- `RETARGETMAP_FIRST_WETLAB_EXPERIMENT_PLAN_20260901_ZH.md`：首轮12药×8靶点盲法湿实验方案，包括推荐模型、对照、命中定义和晋级门。
- `PRODUCTION_PIPELINE_V4_ZH.md`：正式候选筛选流程与输出合同；它是结构/物理筛选线，不替代当前药物中心 ODTI 排序定义。
- `RESULTS_STATUS_ZH.md`：正式候选筛选的计算与交付状态。
- `V4_RUNBOOK_ZH.md`：上述正式流程的执行手册。
- `DATA_ACCESS.md`：数据库、模型、权重和本地大文件的获取边界。

## 技术研究记录

2026-09-06：[统一交互主干修订](BIOMASTER_UNIFIED_INTERACTION_REVISION_20260906_ZH.md)。保留架构评审与原设计；对应实现、五组对照及时间/KIRHub回归现已完成，见[首轮结果](BIOMASTER_UNIFIED_INTERACTION_FIRST_ROUND_20260906_ZH.md)。局部/几何净增益仍未确立。

2026-09-06：[升级版 KIRHub 与 2023–2025 时间测试](BIOMASTER_V3_KIRHUB_TEMPORAL_20260906_ZH.md)。冻结当前 J 候选测功能抑制；另从原始测量按年份生成标签，早期验证选轮数，再以截至 2022 年数据从头训练 V3/J，评估首次出现关系的双向排序。

2026-09-06：[旧V3增量升级与33组消融](BIOMASTER_V3_INCREMENTAL_ABLATION_20260906_ZH.md)。固定旧底座，验证集选择已知关系检索监督、正负证据和预训练交互组合；含历史未见老药双向测试、匹配消融区间及可加载候选清单，未替换生产默认。

2026-09-06：[720老药×384靶点双向排序实测](BIOMASTER_OLD_DRUG_BIDIRECTIONAL_TEST_20260906_ZH.md)。已完成807条已知关系的双向恢复、未见老药子集、独立实测正负排序及DrugCLIP同覆盖比较，附带药名/基因名的双向Top 20；不改变冻结生产合同。

2026-09-06：[V4 R1测试集审计与结论更正](BIOMASTER_ODTI_V4_TESTSET_AUDIT_20260906_ZH.md)。此前1,024×495面板不是项目老药集或密集实测选择性面板，其结果仅限一般化合物的稀疏已知阳性检索。

以下文件保留为模型演化与结构分支的技术依据，不作为当前数据量、生产模型或指标的默认来源：

- [BIOMASTER_ODTI_V4_R1_TRAIN_TEST_20260905_ZH.md](BIOMASTER_ODTI_V4_R1_TRAIN_TEST_20260905_ZH.md)：全量BerMol/ESM2特征、9次R1/证据分支训练、相同1,024×495面板的13组测试及失败机制；尚未训练DrugCLIP三维交互，未晋级生产。
- [BIOMASTER_ODTI_V4_ARCHITECTURE_AND_DATA_20260905_ZH.md](BIOMASTER_ODTI_V4_ARCHITECTURE_AND_DATA_20260905_ZH.md)：完整架构设计、模块接口、预训练接入方案、可用数据与补建队列；首轮R1实验已完成，完整V4仍待实现。
- [BIOMASTER_V3_RESULTS_AND_NEXT_DIRECTION_20260905_ZH.md](BIOMASTER_V3_RESULTS_AND_NEXT_DIRECTION_20260905_ZH.md)：六配置、两种子全部12次训练结果与失败机制；下一步以配体证据和相对活性学习为主，尚未晋级生产。
- [BIOMASTER_V3_BUILD_TRAIN_TEST_20260905_ZH.md](BIOMASTER_V3_BUILD_TRAIN_TEST_20260905_ZH.md)：V3实现、训练协议、评分与工程验证记录。
- `BIOMASTER_ODTI_V2_DESIGN_ZH.md`
- `BIOMASTER_PRETRAINED_REPRESENTATION_SCREEN_20260819_ZH.md`
- `BIOMASTER_STRUCTURE_INTERACTION_UPGRADE_20260819_ZH.md`
- `BIOMASTER_NEXT_MODEL_DATA_ROADMAP_20260817_ZH.md`
- `BIOMASTER_ACTIVE_EXECUTION_QUEUE_20260817_ZH.md`
- `BIOMASTER_STRENGTHENING_SPEAKER_NOTES_20260817_ZH.md`
- `KIRHUB_FUNCTIONAL_ADAPTATION_STATUS_20260831_ZH.md`：KIRHub早期方法学研究记录，不作为当前系统数字来源。

## 历史材料

- `archive/legacy_pre_v4/`：2026 年 5–6 月的早期报告、计算状态和旧候选包说明。
- `archive/reverse_query_20260822/`：以“给定靶点找 720 种药物”为主方向的旧版研究收束和两案例 PDF；查询方向已被当前大纲替代。
- 其他旧网页与报告仍按原路径保留，以维持历史链接；后续只做归档，不再从其中复制当前数字。
- 仍暂留顶层的日期化路线图、旧HTML/PDF和“当前状态”快照统一视为逻辑归档，清单见`archive/DEPRECATED_TOP_LEVEL_SNAPSHOTS_20260901_ZH.md`。

完整归档说明见 `archive/README.md`。

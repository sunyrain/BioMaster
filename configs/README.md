# 配置文件分层

本目录不再把所有配置视为并行的“当前方案”。默认入口只有一个：

- `biomaster_current_contract_v1.json`：项目任务、数据量、靶点空间、模型角色、评价口径和声明边界的唯一当前合同。

仍在使用但只负责局部范围的合同：

- `target_discovery_scope_ch37_v3.yaml`：745个非GPCR登记靶点向450主生产、42特殊、8探索和245仅登记分支的路由规则。
- `target_universe_ch37_v2.yaml`：ChEMBL37人源single-protein MoA全集构建。
- `known_pocket_atlas_ch37.yaml`：已知口袋与结构证据清单。
- `current_pipeline_v4.yaml`：旧物理筛选/湿实验候选交付线；不是当前药物→靶点模型总合同。

其余带日期、`freeze`、`amendment`、`phase`或内部实验代号的JSON/YAML均为冻结实验快照。它们保留用于复现，不得作为默认入口，也不应把其中的模型名或数字复制到新汇报。

KIRHub在当前合同中只是通用模型的回顾性功能迁移审计数据；其他历史配置、脚本和输出不属于当前系统入口。

新配置晋级为当前合同必须同时满足：有对应审计摘要、明确unknown与endpoint语义、通过`validate_current_project_contract.py`，并在本README中替换而不是并列增加默认入口。

当前研发设计：[`biomaster_odti_v4.yaml`](biomaster_odti_v4.yaml)固定完整架构模块、输入维度和实验矩阵，完整几何交互路线尚未实现。[`biomaster_odti_v4_r1_development.yaml`](biomaster_odti_v4_r1_development.yaml)记录已完成9次训练的R1/证据分支实际协议，是对应脚本的描述性记录；[统一测试](../docs/BIOMASTER_ODTI_V4_R1_TRAIN_TEST_20260905_ZH.md)未达到升级目标，未晋级为当前合同。

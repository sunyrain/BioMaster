# BioMaster 文档入口

## 当前正式文档

- `PRODUCTION_PIPELINE_V4_ZH.md`：v4 冻结研究问题、实体空间、评分、Boltz 运行与正式输出合同。
- `RESULTS_STATUS_ZH.md`：当前计算和交付状态，所有数字以本地 CSV/JSON 审计产物为准。
- `DATA_ACCESS.md`：大数据、模型和外部资源的获取边界。

## 当前正式路径

- 配置：`configs/current_pipeline_v4.yaml`
- 药物范围：`configs/project_drugs_v4.csv`
- 靶点范围：`configs/project_targets_v4.csv`
- 全集/Top3000：`outputs/current_production_package_v2/full_untruncated_universe_v4/`
- 签名 Boltz 输入：`boltz_full_input_package_v4_signed/`
- 固定种子正式运行：`boltz_full_run_v4_seeded/`
- 正式 final1000/review512/final384：仅在 `formal_full_universe_v4/` 出现；最终 384 必须来自 512 条逐条审阅后的替换选择。
- 正式交付：`outputs/current_production_package_v2/final_delivery_v4/`，包括完整/中文 CSV、Excel、4×96 候选位图、实验矩阵和两份中文 PDF。

## 历史文档

以下文档记录早期探索，不能作为当前候选或数字来源：

- `BIOMASTER_RESULTS_REPORT_2026_05_11_ZH.md`
- `BIOMASTER_WORK_OVERVIEW_2026_05_23.md`
- `EXPERIMENT_CLOSURE_TRACKER_2026_06_05.md`
- `SOTA_COMPUTE_READINESS_2026_06_03.md`
- `STRICT895_CURRENT_PACKAGE_ZH.md`
- `docs/assets/` 下旧 PDF、网页数据和结构示例

最终 v4 交付完成后，这些历史材料统一移至 `docs/archive/legacy_pre_v4/`，正式入口只保留本页列出的当前文档。

# 脚本入口与维护边界

`scripts/` 保持扁平结构，因为部分训练入口通过脚本目录直接导入共享函数；贸然移动到子包会改变现有 checkpoint 和运行命令的复现路径。当前通过本索引区分正式入口与历史工具。

## 当前双向 ODTI 主线

1. `build_biomaster_comprehensive_training_v1.py`：构建去重综合关系与特征索引。
2. `train_biomaster_comprehensive_balanced_v2.py`：target/label/scaffold 平衡训练与冻结评估。
3. `train_biomaster_bidirectional_v6.py`：训练药物→靶点主头和靶点→药物辅助头。
4. `refit_biomaster_bidirectional_v6_full_fit.py`：在冻结 epoch 合同下完成 FULL_FIT 方向头拟合。
5. `score_biomaster_bidirectional_v6_720x384.py`：生成 720 × 384 双向部署分数。
6. `summarize_biomaster_bidirectional_v6.py`：汇总开发集、时间集和 bootstrap 结果。

共享依赖包括 `train_biomaster_comprehensive_full_fit_v1.py`、`train_biomaster_deployment_augmented_v1.py`、`train_biomaster_bindingdb_affinity_augmented_v1.py`、`score_biomaster_deployment_augmented_720x384_v1.py` 和 `score_biomaster_full_fit_current_new_relations_v1.py`。这些文件均属于可复现依赖，不是可删除的临时脚本。

## 药物中心冻结基线

- `train_v10_leakage_safe_ranker.py`
- `evaluate_v10_leakage_safe_external.py`
- `audit_drug_centric_kirhub_v1.py`
- `audit_v10_nested_model_selection.py`
- `audit_v10_cold_start_generalization.py`
- `build_drug_centric_cross_target_v1.py`

## 正式候选筛选与结构复核

正式 v4、ChEMBL37 target universe、已知 pocket atlas、GNINA/Boltz 和 evidence-routing 脚本继续保留，用于候选后的结构与证据复核。入口合同见 `docs/PRODUCTION_PIPELINE_V4_ZH.md` 与 `docs/V4_RUNBOOK_ZH.md`。

## 新脚本规则

工作区中的新 `.py` 默认被 `.gitignore` 视为探索脚本。只有满足以下条件才提升为正式源代码：

- 被当前文档、测试或正式入口引用；
- 输入/输出合同明确，未知关系不被静默当作负例；
- 通过语法检查和相关测试；
- 不把 checkpoint、全量模型输出或第三方代码一并提交。

已有 Git 跟踪文件不受该默认规则影响。废弃脚本优先依靠 Git 历史追溯，不在当前目录继续复制多个近似版本。

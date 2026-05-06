# BioMaster 当前结果状态

更新时间：2026-05-06

## 总结

当前可以准确表述为：

> Step 1-5 全量主筛选已经完成，Top1000 已完成结构对接增强。全量 DiffDock 仍在后台运行，不应表述为已完成。

## 五步走结果

| 步骤 | 状态 | 主要产物 |
| --- | --- | --- |
| Step 1 药物库 | 完成 | `data/processed/drug_library_pubchem_chembl_mapped.csv` |
| Step 2 蛋白库和受体结构 | 完成 | `data/processed/protein_library_1000_alphafold_paths.csv` |
| Step 3 ConPLex 全量亲和力 | 完成 | `outputs/report_scale/conplex_affinity_scores_915k.csv` |
| Step 4 结构就绪和 AI 排序 | 完成 | `outputs/report_scale/stage4_affinity_candidates_915k.csv` |
| Step 5 疾病证据排序 | 完成 | `outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv` |

## 关键规模

- FDA 小分子：915
- 蛋白靶点：1000
- ConPLex pair：915,000
- Stage 5 排序 pair：915,000
- DiffDock-ready pair：913,170
- Stage 5 Top1000：337 个药物，64 个蛋白
- Stage 6 Top1000 DiffDock：940/1000 completed，60/1000 missing

## Stage 5 如何选择 Top100 分子

Stage 5 排序对象是 drug-target pair，不是唯一分子。

Pair 级别分数：

```text
OpenTargets_STRING_priority
= 0.55 * normalized_ConPLex_or_Stage4_AI_score
+ 0.30 * OpenTargets_direct_disease_score
+ 0.15 * STRING_network_disease_score
```

如果药物映射到 TxGNN：

```text
final_priority_score
= 0.80 * OpenTargets_STRING_priority
+ 0.20 * TxGNN_indication_score
```

如果药物没有 TxGNN 映射，则保留 Open Targets/STRING 分数，不作为负证据。

如果要选 100 个唯一分子，推荐口径是：

1. 对 915,000 个 pair 按 `final_priority_score` 降序排序。
2. 从高到低扫描。
3. 每个 `drug_id` 只保留最高分 pair 作为该药物代表靶点。
4. 取前 100 个唯一 `drug_id`。

这比直接取 Stage 5 前 100 行更适合与合作人员讨论“优先验证哪些分子”。

## Stage 6 和 full DiffDock 边界

Top1000 的 SMILES DiffDock 已完成并形成 Stage 6 共识排序：

```text
outputs/report_scale/stage6_top1000_consensus_candidates.csv
```

Stage 6 共识公式：

```text
Stage6_consensus
= 0.85 * Stage5_final_priority_score
+ 0.15 * normalized_DiffDock_confidence
```

全量 DiffDock 是后续结构证据增强，不是 Step 1-5 完成条件。当前 full 队列采用 score-only 策略，避免 91 万级 pose 文件占满磁盘。少数 SDF 药物出现系统性失败，后续应使用 SMILES 补跑队列修复。

## 论文稿

完整中文论文/汇报稿在本地生成于：

```text
outputs/report_scale/biomaster_full_paper_zh_cn.md
```

该文件未默认纳入 git，因为 `outputs/` 是本地结果目录。需要对外共享时，可单独导出或复制到协作平台。

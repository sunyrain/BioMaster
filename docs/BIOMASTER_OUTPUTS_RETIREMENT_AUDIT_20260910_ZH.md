# outputs 淘汰产物专项清理方案（2026-09-10）

**执行状态：已获用户确认并清理完成，实际释放约 101.6 GB。见[执行结果与验证](BIOMASTER_DISK_CLEANUP_COMPLETED_20260910_ZH.md)。以下为执行前的方案记录，原逐文件清单不可重复执行。**

**可以继续清理。此前把一些目录整体视为未来研究资产，保护范围过宽。细分后，新增找到 15.64 GiB 候选；现在建议保留全部 DTIAM 历史折模型，改为收束旧试跑、工程检查点和已停止结构训练的派生特征，仍能回收约 94.67 GiB，即 101.65 GB。**

本轮仍是清单审计，没有执行删除、替换链接或重启网站。下列空间是候选文件占用，缓存去重必须保持引用路径可用；历史计算产物收束需要接受重算成本。

## 目录很多，应该怎样收束

`outputs` 占用约 210 GiB。本轮盘点到 237 个含文件的一级目录，其中 99 个目录分别不足 10 MiB，合计仅约 0.09 GiB。把这些小报告、表格和状态记录清掉，对释放容量几乎没有帮助。

真正占空间的是模型权重、特征矩阵、结构预测的中间文件和打包展开副本。对已结束的实验，可以保留配置、数据划分、预测表、指标和结论，移除大文件；不必因为旧代码引用过某个运行目录，就将整个运行永远保留。旧模型在当前生产中未使用，也不等于当时试验失败，两者分开记录。

## 这次新增的 15.64 GiB

| 新增类别 | 可回收 GiB | 具体处理范围与保留内容 |
|---|---:|---|
| 历史试跑、调参和调试模型 | 4.169 | 从 52 个试跑目录中列出 133 个 `.pt/.pth/.ckpt`；保留预测表、指标、日志、配置和数据来源。旧模型重新推理需要重训 |
| 结构/context 工程检查点 | 1.735 | 5 个工程调试权重；正式训练的 BEST、LATEST 和结构预训练权重不在这一组 |
| 已停止结构训练的派生特征 | 8.864 | 仅 `training_2020/esm2/` 和 `training_2020/encoded/`；保留原始复合物、映射、序列、标签、划分与生成脚本 |
| 额外核实的重复特征与快照 | 0.871 | 4 个可去重副本；SHA256 相同，保留源文件并保持旧路径可读 |

旧试跑目录的例子：

- `outputs/biomaster_odti_pretrained_residual_screen_v1/`：模型约 0.990 GiB。
- `outputs/biomaster_odti_e1_s3_screen_20260817/`：模型约 0.409 GiB。
- `outputs/biomaster_odti_esmc_debug/`：模型约 0.374 GiB。
- `outputs/biomaster_bindingdb_affinity_augmented_screen_v1/`：模型约 0.311 GiB。
- `outputs/old_drug_target_sota_v1/` 下的部分 `smoke/screen/selftest` 子运行，以及其他旧版结构、图模型、cross-attention 试跑。

工程检查点范围：

- `outputs/biomaster_pocket_precision_20260906/engineering/` 内两个 `.pt`。
- `outputs/biomaster_context_full_20260908/engineering_resume/LATEST.pt`。
- `outputs/biomaster_context_full_20260908/engineering_continuous/LATEST.pt`。
- `outputs/biomaster_context_full_20260908/final_driver_check/LATEST.pt`。

结构训练分支在 `closure_20260907/RESULT.json` 明确记录为 `STOPPED_BY_USER`、`training_complete=false`、`promotion_eligible=false`，没有确认排序收益。`structural_data` 名字中虽有 data，但其中的 ESM2 表征和 encoded 张量是派生数据，由 `scripts/prepare_biomaster_structural_training.py` 生成，并非原始实验结构。上一轮整体保护该目录过于保守，本轮将这 8.864 GiB 单独列出。

代价是：清理后不能立即续跑这条结构训练路线，也会影响依赖它的 context 训练恢复，需要重建这些特征。当前生产包的 `metadata.json` 记录 `local_interactions=false`、`structure_fallback=global model has no receptor structure dependency`；收束这些派生特征不等于删除当前生产模型。

额外重复内容包括：retrain 的 Morgan 特征副本、两个 package smoke 内的 residue tokens 副本，以及正式 BEST 的诊断快照副本。另有一组原子特征虽然路径不同、内容相同，但已经是硬链接共享实体，**释放量为零，未计入可回收空间**。校验覆盖 360 个同尺寸大型二进制文件，发现 9 组完全相同内容；与旧清单和试跑模型重叠的部分不重复计数。

## 推荐的约 100GB 组合

这份组合替代上一轮“删除 DTIAM 非 S5 折”的建议，**DTIAM 所有折整套保留**。

| 组合 | GiB |
|---|---:|
| 上轮已核实的重复缓存、下载分片、旧离线包展开副本 | 8.578 |
| 历史 Boltz PAE/PDE，剔除当前湿实验关联记录 | 50.750 |
| 历史 Boltz processed，保护当前湿实验关联批次 | 2.580 |
| unified 轮旧原子特征 ATOM_TOKENS.npy | 8.021 |
| 停止的 context 轮 predicted/p2rank 派生输入 | 9.097 |
| 本轮新增四类 | 15.639 |
| **合计** | **94.666 GiB / 101.647 GB** |

上轮与本轮全部候选的并集是 113.71 GiB，其中包含 19.04 GiB DTIAM 旧折模型；推荐组合主动保留这些模型，使用其余 94.67 GiB。两份逐文件清单已检查无交集，不是将父目录与子目录重复相加。

约 100GB 是十进制；如果将“100G”理解为 GiB，这份方案是约 94.7 GiB。没有必要为了补足几 GiB 删除训练原始数据或实验资产。

## 哪些旧父目录不能整包删除

- `current_production_package_v2/`：名字旧，但网站的 `explorer_annotations.py` 仍读取其中 OT 表。可以继续拆分其中的历史 Boltz 运行，不能直接删父目录。本轮没有把它新增计入推荐释放量。
- `evidence_routing_compute_execution_20260808_v1/`：混有旧 Boltz/MD 大产物，也有网站读取的 GtoPdb 来源清单和增量审计。没有把全部 14.8 GiB 当成废弃数据。
- `old_drug_target_sota_v1/`：包含现用冻结排名、DTIAM 比较结果与可运行模型；仅指定试跑权重进入本轮新增清单。
- `biomaster_v3_kirhub_temporal_20260906/`、`biomaster_unified_interaction_20260906/`：现用生产包的来源记录仍引用这里的索引和特征清单，不能因为版本旧就删除整个目录。
- 最终384及对照、实验回填、亲和数据库与原始数据、TxGNN/OT/EC-KG、当前网站缓存、生产模型与其训练来源继续保留。

这意味着可以大幅收束 outputs，但应按文件和运行子目录处理。小体积审计记录可以日后合并归档，用来减少目录数量；它们不是这次容量压力的主要来源。

## 可执行前审阅的清单

- [全目录处置表](../outputs/disk_reclamation_audit_20260910/OUTPUTS_DIRECTORY_DISPOSITION.csv)：各目录占用、推荐释放量、是否存在网站字面路径引用；未选中不代表永久保留。
- [本轮新增逐文件清单](../outputs/disk_reclamation_audit_20260910/ADDITIONAL_CANDIDATE_FILES.csv.gz)。
- [推荐约100GB逐文件清单](../outputs/disk_reclamation_audit_20260910/RECOMMENDED_100GB_FILES.csv.gz)：本轮建议以这份为准，包含保留路径、时间、inode 与动作类型。
- [汇总与边界](../outputs/disk_reclamation_audit_20260910/OUTPUTS_RETIREMENT_SUMMARY.json)。
- [重复文件 SHA256 证据](../outputs/disk_reclamation_audit_20260910/OUTPUT_EXACT_BINARY_DUPLICATES.json)。
- [网站目录引用](../outputs/disk_reclamation_audit_20260910/WEBSITE_OUTPUT_DIRECTORY_REFERENCES.csv)与[新增范围的静态引用](../outputs/disk_reclamation_audit_20260910/ADDITIONAL_STATIC_REFERENCES.csv)。

静态检索在检查的网站模块、便携推理模块和新靶点比较入口中未发现新增试跑/工程目录引用，但不等于穷尽所有动态路径。推荐清单检查时没有打开的文件描述符；这是瞬时检查。后续真正清理前仍需核对文件未变化，去重后验证加载；当前没有执行删除。

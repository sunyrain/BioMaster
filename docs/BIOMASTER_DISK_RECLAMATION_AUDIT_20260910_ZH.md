# BioMaster 收束阶段磁盘清理审计（2026-09-10）

**执行状态：后续推荐方案已清理完成，实际释放约 101.6 GB。见[执行结果](BIOMASTER_DISK_CLEANUP_COMPLETED_20260910_ZH.md)。本文为首次盘点历史记录，不代表当前文件存在状态。**

后续已完成 [outputs 淘汰产物专项审计](BIOMASTER_OUTPUTS_RETIREMENT_AUDIT_20260910_ZH.md)：新增 15.64 GiB 候选，并给出保留全部 DTIAM 历史折、仍可回收约 101.65 GB 的替代组合。以下保留首次盘点记录；后续操作应参考新版逐文件清单。

本轮只盘点、核对依赖并生成清单，**没有删除、移动或替换项目文件，也没有重启网站**。384 候选已经交给湿实验，本轮不改变候选、对照或优先级。

**结论：找到约 98.07 GiB（105.30 GB）的清理候选，但不能把这些全部称为“现在未来完全用不上”。其中 8.58 GiB 是经过核实的重复内容或已有归档的展开副本；其余 89.49 GiB 是可以考虑收束的历史计算产物，仍有复用价值。**

## 磁盘现状与范围

审计结束时，所在磁盘容量 550 GiB，已用约 533.33 GiB，可用约 16.67 GiB。BioMaster 约占 347 GiB，其中 outputs 约 210 GiB、data 约 40 GiB、downloads 约 29 GiB、Git 历史约 19 GiB、项目 Conda 环境约 19 GiB。

同盘 `/root/autodl-tmp/thermoprot` 约占 157 GiB，属于另一个项目，本轮不纳入清理。共享缓存只用于核对重复内容，不能按“项目外文件”整体删除。

大小以文件实际分配的磁盘块计算，GiB = 2³⁰ 字节，GB = 10⁹ 字节。清单不包含目录元数据，因此不是实际释放空间的精确承诺。

## A 类：优先处理的重复内容，合计 8.58 GiB

| 项目 | 可回收 GiB | 已核对内容 | 处理边界 |
|---|---:|---|---|
| Boltz 权重和分子包的重复缓存 | 5.778 | `boltz2_aff.ckpt`、`boltz2_conf.ckpt`、`mols.tar`，项目副本与共享副本逐文件 SHA256 相同 | 只保留一份实体，同时保持两个引用路径有效；不能直接删断模型路径 |
| BindingDB 下载分片 | 0.552 | 8 个分片拼接后的 SHA256 与已组装 ZIP 相同 | 保留完整 ZIP，仅移除分片 |
| 旧版 Windows 离线包展开目录 | 2.248 | ZIP 内 6,985 个文件与展开目录大小、CRC 一致，ZIP 自检通过 | 保留 ZIP、校验文件和说明；展开目录另有一个未归档的可重建 Python 字节码文件 |

对应位置：

- `outputs/boltz2_structure_affinity_v1/boltz_cache/` 下上述三个文件；相同内容保存在 `/root/autodl-tmp/boltz_cache/`。
- `.tmp/bindingdb_202608_full/segments/`；保留相邻的 `BindingDB_All_202608_tsv.zip`。
- `outputs/offline_release_20260909/BioMaster_Offline_Windows_x64/`；保留相邻的 `BioMaster_Offline_Windows_x64_20260909.zip` 及校验文件。新版 `offline_release_20260909_r2` 不在本轮候选内。

这一类保留了模型或归档内容，是本轮最接近“无需继续占用额外空间”的部分。缓存去重仍需实施后验证原路径可以加载。

## B 类：有条件收束的历史产物，合计 89.49 GiB

| 项目 | 可回收 GiB | 保留内容 | 代价与判断 |
|---|---:|---|---|
| 两轮历史 Boltz 的 PAE/PDE 矩阵 | 50.750 | CIF 结构、affinity/confidence JSON、汇总 CSV、输入与运行记录；当前湿实验关联矩阵也保留 | 丢失其他历史配对的逐残基误差分析能力；仅靠汇总分数不能恢复矩阵，需重新推理 |
| 上述 Boltz 的部分 processed 文件 | 2.580 | 输入、源结构、环境、模型和结果；湿实验关联批次的 processed 全部保留 | 再运行需重新预处理 |
| DTIAM 非 S5 历史折的 predictor 目录 | 19.044 | S5 部署 predictor 完整保留；目录外的折级预测、汇总、训练来源与配置保留 | 删除旧折模型及其缓存训练特征，无法直接再次调用旧折预测器；恢复需重训，未必位级一致 |
| 未晋升 unified 轮的 ATOM_TOKENS.npy | 8.021 | LMDB、索引、其他特征、checkpoint 和日志 | 该轮未晋升为生产模型；复训或旧轮审计仍可能用到，恢复需重建原子特征 |
| 未晋升 context 轮的 predicted 与 p2rank 派生数据 | 9.097 | 原始结构、索引、全局特征、模型与训练记录 | 恢复需要重新做口袋识别和编码；未来训练可能复用，优先级低于重复文件清理 |

严格范围如下，**不是删除整个父目录**：

- Boltz：`outputs/boltz2_calibration_338_v1/formal_screen_run/` 和 `outputs/boltz2_discovery_conditional_v1/screen_run/` 内清单指定的 `pae_*.npz`、`pde_*.npz` 及部分 `processed` 文件。
- DTIAM：`outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_same_data_compatible_v1/` 下非 `S5_` 运行的 `predictor/`；**S5 整套不动**。当前新靶点比较和 720×384 部署脚本仍会使用 S5，不能把整个 DTIAM 目录当作旧垃圾。
- 原子特征：`outputs/biomaster_unified_interaction_20260906/features/ATOM_TOKENS.npy`。
- context 派生数据：`outputs/biomaster_context_full_20260908/data/predicted/` 和 `data/p2rank/`。该轮状态为 `STOPPED_STRUCTURAL_RETENTION_GATE`、`training_complete=false`、`promotion_eligible=false`；进程退出码为 0 不代表训练完整成功。

上表是可审阅的空间方案，并不表示已经授权或完成删除。尤其 PAE/PDE 和旧折模型，不能以“有脚本可以重跑”为由说没有信息或成本损失。

## 当前湿实验与未来主线的保护范围

以当前 384 候选、112 个现用对照以及替换前对照建立精确药物 InChIKey—靶点集合，去重后为 527 对。将历史 calibration 的 ChEMBL 化合物编号映射到完整 InChIKey 后再匹配，没有未解析的 calibration 化合物身份。

匹配到 calibration 90 条、discovery 36 条运行记录。已从候选清理清单中剔除它们的 PAE/PDE，并保守保留这些记录所在 99 个批次的全部 processed 文件，共额外保护约 1.20 GiB。126 是历史运行记录数，不能解释成 126 个当前候选都有结构结果；其中包含新旧对照。

以下内容继续保留，不为凑空间而删除：

- 最终 384 实验表、112 对照及 FDA 替换前后的版本、导出表、评价证据、冻结记录和实验回填数据库。
- 网站代码、现用缓存及生产模型；720×384 完整评分空间、888 靶点名册。
- BindingDB/ChEMBL 亲和原始数据、证据数据库、训练来源、数据划分、增量与性能审计。
- TxGNN 权重及图谱、多方向疾病推理、Open Targets 和 EC-KG 数据；后续仍需解释湿实验命中结果。
- LINCS 原始扰动数据约 33 GiB；虽然体积大，仍是未来细胞效应与疾病方向分析的数据资产。
- DrugCLIP 全部相关折权重、DTIAM S5、生产模型包以及原始结构数据。
- `.git`、环境与第三方代码。本轮没有证明 Git 可安全回收多少空间，不手工删除 objects/packs。

## 建议与审计产物

如果坚持“只清理明确重复、保留全部研究能力”，当前可支持的范围是 **8.58 GiB**。如果接受历史计算产物收束，则 A+B 共 **98.07 GiB**，满足约 100G 的目标，按当前占用估算可用空间可提高到约 115 GiB。

我的建议是先落实 A 类去重；对 B 类保留这个具体清单，按是否还计划重做历史模型比较、逐残基分析和 context 训练来决定。不要将整片 `outputs`、原始数据库或现用比较模型删除来凑数。若需要保留 B 类全部复用能力，只能将这些产物归档到本磁盘以外，单纯移动到同盘另一个目录不会释放容量。

本轮已生成：

- [分类与大小清单](../outputs/disk_reclamation_audit_20260910/CLEANUP_CANDIDATES.csv)
- [逐文件候选清单](../outputs/disk_reclamation_audit_20260910/CANDIDATE_FILES.csv.gz)：包括相对路径、实际占用、修改时间、设备与 inode；后续执行需重新核对，不能使用目录通配符扩大范围。
- [湿实验关联 Boltz 保护清单](../outputs/disk_reclamation_audit_20260910/PROTECTED_WETLAB_BOLTZ.csv)
- [审计汇总](../outputs/disk_reclamation_audit_20260910/SUMMARY.json) 及同目录三个重复/归档核验 JSON。
- [只读审计脚本](../scripts/audit_disk_reclamation_20260910.py)。该脚本只生成盘点文件，没有删除或链接替换能力。

扫描时未发现候选文件被进程打开，且候选文件没有硬链接；这是某一时刻的检查，不能据此证明未来没有依赖。所有原业务文件均未清理。

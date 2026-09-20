# DTIAM 进度记录（2026-09-13）

> 历史归档（2026-09-20）：保留当时设计/进度及结果，不作为当前执行队列。当前入口见[文档索引](../../README.md)。

最新状态见 [2026-09-14 进度与评价](BIOMASTER_DTIAM_PROGRESS_20260914_ZH.md)；以下保留 9 月 13 日恢复和系统盘清理记录。

核对时间：2026-09-13 15:53 UTC。**环境重启后训练已恢复；完整拟合及公共验证仍为 1/6 组，B 首组已完成 7/11 个基础学习器。系统盘已释放约 9.78 GiB，可用约 10.68 GiB。统一 TEST 尚未启动。**

| 组别 | 当前进度 |
|---|---|
| A / 20260921 | 11 个基础学习器和加权集成完成，公共验证完成 |
| B / 20260921 | 七个基础学习器完成，XGBoost 重跑中，已重新输出至至少 300 轮 |
| A / 20260922、20260923 | 待运行 |
| B / 20260922、20260923 | 待运行 |

B 已完成 LightGBMXT、LightGBM、RandomForestGini、RandomForestEntr、CatBoost、ExtraTreesGini、ExtraTreesEntr。首个模型与其余六个成功阶段记录累计约 162.5 分钟，包含少量保存与核验开销，不含失败或停机时间。剩余基础学习器为 XGBoost、NeuralNetTorch、LightGBMLarge、NeuralNetFastAI，随后还需集成和公共验证。

队列在 11:55 UTC 因 XGBoost 的实际 RSS 超过 78 GiB 保护线而停止，和之前仅由内存估算拒绝启动的情况不同。停止发生在首个报告的 boosting 迭代前；原生数值输入会转换成 CSR 再拼接，由源码推断这一转换是主要内存风险，但没有捕获证明峰值所在位置的 Python 调用栈。

14:08 UTC 已恢复队列，14:09 UTC 进入 XGBoost。新增的 `NumericDenseXGBoostModel` 只替换原生数值输入转换：float32 稠密表示中的零值转成 NaN，以保留原 CSR 将零值作为缺失的语义；训练和推理一致，非 float32 输入会拒绝。继续复用原生损失、树参数、轮数上限、早停、CPU 数量和保存加载逻辑。实际模型类型会在最终台账中注明，推理环境需要能导入 `scripts/dtiam_numeric_xgboost_20260913.py`。

已完成三项输入适配测试，覆盖零值/NaN、保存加载、原生模型名称和输入类型保护；此前两项断点恢复检查亦通过。真实训练样本的原生与适配预测差异为 0，保存加载差异也为 0；容差为 2e-7，不主张所有全量拟合逐位一致。完整 B 的 1,105,107 个基础训练成员、11,163 个内部留出成员和 2,048 维输入完成两轮资源探测，峰值为 **30.44 GiB**。这两轮是独立资源检查，不能当成正式模型训练完成或性能结果。

七个已完成 B 模型的文件哈希与恢复前一致，A 沿用原已完成权重。成员池仍为 1,116,270 对，未缩小数据或删除必要子模型；实际 RSS 保护线仍为 78 GiB。修订、失败现场和资源探测有独立审计记录。

15:39 UTC 左右环境重启，原队列和训练子进程已退出，状态文件仍停在 TRAINING。上次 XGBoost 日志停在 15:38:51 UTC、约 5,600 轮，运行约 89.05 分钟；未完成学习器没有保存中间权重，因此本次从第 0 轮重跑 XGBoost，不能称从 5,600 轮续训。已完成 A 和七个 B 学习器继续复用。15:43 UTC 恢复队列，15:44:44 UTC 启动当前 XGBoost；队列 PID 2015、子进程 PID 2593，核查时子进程 RSS 约 30.1 GiB，迭代日志持续增加。冻结数据、协议、代码及七个 B 模型文件哈希再次通过检查。

恢复现场记录见 [RECOVERY.json](../../../outputs/biomaster_dtiam_ab_20260912/all_inactive__seed_20260921/ENV_RESTART_RECOVERY_20260913/RECOVERY.json)。训练临时目录已指向数据盘 `/root/autodl-tmp/BioMaster/.runtime_tmp/dtiam`。

从本次 15:44 UTC 重启 XGBoost 起，按以下范围安排：

| 里程碑 | 预计剩余时间 | 中心估计 |
|---|---:|---:|
| 首组 B 完整拟合及公共验证 | 6—12 小时 | 约 8.8 小时 |
| 全部六组、统一 TEST 及报告 | 30—50 小时 | 约 38.2 小时 |

全轮中心完成时间约为 **9 月 15 日 06:00 UTC（北京时间 14:00）**。这是工作安排范围，不是置信区间或保证时限；前提是没有再次中断。依据是首组 A 实测 3.00 小时、B 已完成七阶段实测 2.71 小时、先前 XGBoost 5,600 轮耗时。后续 B 神经和大树模型按 A 分项耗时乘训练行数比 3.307 外推；神经模型实际早停轮数是主要不确定项。没有按完成模型个数线性估算。明细和公式见 [RESTORE_ETA_20260913.json](../../../outputs/biomaster_dtiam_ab_20260912/RESTORE_ETA_20260913.json)，首套 B 完成后需重新估计。没有新增 B 公共验证或 TEST 成绩。

系统盘清理已完成：

| 项目 | 结果 |
|---|---|
| 系统盘 `/`，容量 30 GiB | 可用从约 0.90 GiB 增至 10.68 GiB，占用从 97% 降至约 65% |
| 缓存及临时文件清理 | 实际释放约 3.98 GiB；清除五个非活动 VSCode 服务版本、安装下载缓存及临时测试环境 |
| BioEmu/ColabFold 环境迁移 | 从系统盘迁至 `/root/autodl-tmp/.system_migrated/bioemu_colabfold`，释放约 5.80 GiB；原路径 `/root/.bioemu_colabfold` 以符号链接保留 |
| 迁移校验 | rsync 全文件内容校验无差异，原路径 Python 可运行，JAX/JAXLIB/ColabFold 包版本一致；未执行完整 BioEmu 推理 |
| 数据盘可用空间 | 迁移后约 68.5 GiB；迁移释放系统盘空间，同时占用数据盘空间 |

当前 VSCode 服务、活动 Python/CUDA 环境、训练输入与权重、SPR 交付文件、Codex 会话历史和安全日志均保留。清理清单、实测磁盘变化和迁移记录见 [清理汇总](../../../outputs/system_disk_cleanup_20260913/SUMMARY.json)、[缓存清理明细](../../../outputs/system_disk_cleanup_20260913/CACHE_CLEANUP_RESULT.json)、[环境迁移校验](../../../outputs/system_disk_cleanup_20260913/ENV_RELOCATION.json)。

阶段性能见 [A 阶段评价](../../BIOMASTER_DTIAM_INTERIM_REVIEW_20260913_ZH.md)，实时状态见 [STATUS.json](../../../outputs/biomaster_dtiam_ab_20260912/STATUS.json)，本次快照见 [PROGRESS_20260913.json](../../../outputs/biomaster_dtiam_ab_20260912/PROGRESS_20260913.json)。

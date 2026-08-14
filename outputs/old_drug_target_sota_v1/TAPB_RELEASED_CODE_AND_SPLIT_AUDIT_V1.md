# TAPB 官方代码与释放切分审计（2026-08-14）

## 裁决

TAPB 是靶点先验干预的直接先例，应作为 BioMaster 的必要去偏参照；但发布 headline 不能直接进入冻结 S1–S5 的同数据 SOTA 比较。官方仓库的全部 20 个 CSV、核心代码和许可证已经在固定 commit 下审计。

| 数据/协议 | train/source | validation/target-train | test/target-test | 关键边界 |
|---|---:|---:|---:|---|
| BindingDB random | 34,439 | 4,920 | 9,840 | entity-warm；无完全 pair 跨 split 重叠 |
| BindingDB cluster | 14,928 | 7,114 | 1,779 | source 与 target 实体不重叠；发布执行却用 target-test 同时选模和测试 |
| BioSNAP random | 19,224 | 2,747 | 5,493 | train–val/test 有 1/3 个完全 pair 重叠 |
| BioSNAP cluster | 9,766 | 3,628 | 907 | source 与 target 实体不重叠；同样复用 target-test |
| Davis random | 2,086 | 3,006 | 6,011 | train–val/train–test/val–test 重叠 126/177/246 pair |
| Human cold | 3,453 | 155 | 311 | train 对 val/test 双实体零重叠；val–test 重叠 12 pair |

Davis 的三个跨集合比较还有 24/46/28 个标签集合冲突。cluster 数据本身的 source–target 隔离是有效的，但 `main.py` 将同一个 `target_test_with_id.csv` 同时赋给 validation 与 test，`Trainer` 再以 validation AUROC 选择最佳 epoch，因此该代码路径使用最终测试集合做模型选择。

官方方法使用 ESM-2 靶点残基表征、MolFormer tokenizer、训练集靶点 KMeans 混杂字典、先验加权 backdoor aggregation、靶点删除/突变和药物 MLM。仓库没有附带生成后的 ESM-2 特征、混杂字典或训练权重。因此现阶段是代码与数据审计完成，不是数值复现完成。

后续若移植 TAPB，必须只使用冻结 BioMaster TRAIN 建混杂字典、使用独立 VALID 选模、保持 TEST 锁定、去重并处理标签冲突，同时报告 micro、target-macro、drug-macro 和双查询置换门槛。

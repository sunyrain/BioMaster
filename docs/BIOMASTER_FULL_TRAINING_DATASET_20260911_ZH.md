# BioMaster 全靶点最终训练数据集

数据版本：`20260910_v1`。范围为人源单蛋白；不限制项目720药、384/888靶点或MoA目录。数据整理完成不表示已经训练新模型。

## 交付数量

- 可判定二分类标签：**1,457,138行**，对应**1,409,474个不同药物—蛋白配对**。
- 涉及**867,739个标准化分子身份**、**4,777条不同蛋白序列**。
- 主训练集：**1,158,103行**，其中阳性802,849、阴性355,254；涉及1,120,319个不同配对。
- 严格去除与验证/测试共享文献后的训练备选集：172,500行。

同一药物—蛋白可以有Kd/Ki、IC50、EC50等不同任务，故标签行数不等于独立配对数，更不等于独立实验数。所有灰区、冲突与来源重复均有独立审计文件。

| 划分 | 任务 | 标签行数 | 阳性 | 阴性 |
|---|---|---:|---:|---:|
| train | ACTIVITY_EC50 | 102,821 | 72,224 | 30,597 |
| train | ACTIVITY_IC50 | 721,383 | 541,497 | 179,886 |
| train | AFFINITY_KD_KI | 287,296 | 189,128 | 98,168 |
| train | ANNOTATED_INACTIVITY | 46,603 | 0 | 46,603 |
| validation | ACTIVITY_EC50 | 13,033 | 9,178 | 3,855 |
| validation | ACTIVITY_IC50 | 85,842 | 64,502 | 21,340 |
| validation | AFFINITY_KD_KI | 34,745 | 24,471 | 10,274 |
| validation | ANNOTATED_INACTIVITY | 6,725 | 0 | 6,725 |
| test | ACTIVITY_EC50 | 11,985 | 8,551 | 3,434 |
| test | ACTIVITY_IC50 | 85,763 | 65,227 | 20,536 |
| test | AFFINITY_KD_KI | 34,385 | 23,216 | 11,169 |
| test | ANNOTATED_INACTIVITY | 4,822 | 0 | 4,822 |
| reference_control_holdout | ACTIVITY_EC50 | 22 | 20 | 2 |
| reference_control_holdout | ACTIVITY_IC50 | 99 | 98 | 1 |
| reference_control_holdout | AFFINITY_KD_KI | 113 | 113 | 0 |
| previous_benchmark_scaffold_holdout | ACTIVITY_EC50 | 1,811 | 1,046 | 765 |
| previous_benchmark_scaffold_holdout | ACTIVITY_IC50 | 8,833 | 3,748 | 5,085 |
| previous_benchmark_scaffold_holdout | AFFINITY_KD_KI | 9,504 | 5,101 | 4,403 |
| previous_benchmark_scaffold_holdout | ANNOTATED_INACTIVITY | 1,353 | 0 | 1,353 |

## 直接使用哪些文件

主入口为本目录的 `TRAIN.parquet`、`VALIDATION.parquet`、`TEST.parquet`，并提供同名 `.csv.gz`。这三个表是多任务长表，**必须保留task列**。

若先训练直接亲和力分类器，使用 `TRAIN_AFFINITY_KD_KI.parquet` 及对应验证/测试文件。`ACTIVITY_IC50`和`ACTIVITY_EC50`为不同辅助任务，不能自动当成Kd或直接结合概率。`ANNOTATED_INACTIVITY`仅表示来源明确失活注释，不是未知关系生成的负例。

`MOLECULES.parquet`含标准化SMILES、完整InChIKey、连接身份、骨架、重原子数和形式电荷。`TARGETS.parquet`含实际氨基酸序列、序列SHA256、UniProt/ChEMBL别名和来源等级。**drug_feature_index / target_feature_index只是本次数据版本的内部稠密编号，不可套用旧模型的特征数组。** 全量特征编码和模型训练尚未执行。

```python
from pathlib import Path
import pandas as pd

root = Path('data/processed/biomaster_training_full_20260910_v1')
train = pd.read_parquet(root / 'TRAIN_AFFINITY_KD_KI.parquet')
drugs = pd.read_parquet(root / 'MOLECULES.parquet')
targets = pd.read_parquet(root / 'TARGETS.parquet')
train = train.merge(drugs[['drug_feature_index', 'smiles']], on='drug_feature_index', validate='many_to_one')
train = train.merge(targets[['target_feature_index', 'sequence']], on='target_feature_index', validate='many_to_one')
assert train.binary_label.isin([0, 1]).all()
```

## 标签与身份规则

数值统一为nM。Kd/Ki、IC50、EC50分别保存；≤1000nM为阳性，≥10000nM为弱活性/弱结合阴性；1–10µM为灰区。严格解析`< <= > >=`，不能把上界当成精确浓度，也不能把模糊的`>100nM`当成阴性。明确阳性和阴性同时出现时，整条配对/任务隔离，重复来源不能投票消除矛盾。

ChEMBL要求直接、高可信、无标注变体的靶点归属。Kd/Ki/IC50要求结合类assay，EC50可来自结合或功能assay并独立建任务；保留出处和年份。BindingDB按人源单链、确切序列或可解析accession映射，突变/融合、序列不一致、多链和不明身份单列排除。已知错误记录，包括nevirapine误配NVP-BHG712，按人工QC表隔离。

分子以标准完整InChIKey归并；验证源SMILES与登记key一致。保留立体化学、同位素、形式电荷，不额外执行中和或互变异构体规范化；标准InChI本身仍有自身归一化规则。多有机组分混合物隔离；单有机组分可去除无机对离子。不会将前药直接替换成活性代谢物。模型SMILES取固定来源扫描顺序的首个合格标准化表示。

蛋白以完整氨基酸序列SHA256归并；相同序列的不同数据库ID作为别名。可精确匹配的野生型片段映射到参考序列并保留构建范围；没有外部参考序列的BindingDB条目须有明确人源声明、accession和序列，来源等级单列。序列标准化不是逐篇确认实验构建或治疗机制。

本版逐一对比BindingDB全量包及Articles、PubChem、ChEMBL、Patents、PDSPKi分包。同一source record ID若出现不同有效身份/数值版本，或另一个分包明确拒绝其身份，整组隔离。ChEMBL与BindingDB之间相同数值证据记录保留出处，按配对/任务只生成一个标签，不按重复次数加权。独立实验也可能给出相同数值，所以重复数值组只是保守的审计单位。

## 验证集与实验保护

主划分按化学骨架固定80/10/10哈希；同一连接身份涉及不同骨架表示时合并分组，保证分子连接身份和骨架不跨train/validation/test。实际行数比例不要求恰好80/10/10。无环分子按连接身份分组，不把全部无环分子放成一组。

当前384实验候选和112个参考对照的历史证据分别放入独立holdout文件。此前增量实验的留出药物按整个骨架组保留，不能再次混入主训练。湿实验保留是配对层面的，不宣称对应药物/靶点从未出现于历史训练。

主骨架划分可以共享论文/专利；`shares_document_with_*`明确标记。要求来源分离时，使用 `TRAIN_DOCUMENT_PURGED.parquet`、`VALIDATION_DOCUMENT_PURGED.parquet`与`TEST.parquet`。这仍是回顾性数据，不是前瞻湿实验，也不是靶点同源簇冷启动验证。`seen_by_frozen_parent_connectivity_pair`用于识别旧模型已见关系，不能把其表现冒充独立评估。

回归数据在 `*_REGRESSION.parquet`：按Kd、Ki、IC50、EC50分别取去重精确数值的p尺度中位数；范围超过1 log、分类冲突或与删失界限相抵触的值不进入推荐回归表。删失观测完整保留在证据表，可供以后使用区间损失。

## 追溯与复现

`OBSERVATIONS_WITH_QC.parquet`、`SOURCE_OCCURRENCES.parquet`、`PAIR_CITATIONS.parquet`、`EVIDENCE.sqlite`提供来源记录和引用；`QUARANTINED_SOURCE_RECORDS.parquet`、`GREY_AND_CONFLICT_PAIR_TASKS.parquet`及各来源`*_EXCLUSIONS.csv.gz`保留排除原因。HiQBind附带结构标签以及其他构建/端点/物种不符的数据保留在辅助审查表，本版不硬塞入人源直接亲和力训练。

协议、源文件SHA256、分阶段计数、完整文件清单和验证结果位于 `outputs/training_dataset_full_20260910_v1/`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_final_training_dataset_20260910.py
```

已经完成的导入阶段会复用；源文件或协议改变时拒绝静默继续。大数据文件留在本地，代码、说明和紧凑审计快照进入Git。没有修改生产模型、网站排名或冻结实验表。

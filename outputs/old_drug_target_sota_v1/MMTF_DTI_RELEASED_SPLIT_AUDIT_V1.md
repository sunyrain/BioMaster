# MMTF-DTI 释放切分审计（2026-08-14）

## 裁决

四个公开数据集的 12 个 split 文件已逐行审计。它们不是本项目的 scaffold/homology/double cold 切分，论文 headline 不能与 S1–S5 直接比较。

| 数据集 | 总行数 | train/val/test | 全局药物/靶点 | 测试药物/靶点已见比例 | train–val/train–test 完全 pair 重叠 |
|---|---:|---:|---:|---:|---:|
| BindingDB | 49,199 | 34,439/4,920/9,840 | 14,643/2,623 | 0.904/0.864 | 0/0 |
| Celegans | 7,786 | 6,228/779/779 | 1,767/1,876 | 0.790/0.863 | 141/137 |
| DrugBank | 35,022 | 28,016/3,503/3,503 | 6,645/4,256 | 0.971/0.994 | 48/47 |
| Human | 5,997 | 4,197/600/1,200 | 2,726/2,001 | 0.517/0.748 | 0/0 |

C. elegans 与 DrugBank 的释放 split 存在完全相同 pair 跨集合重叠；DrugBank train–test 重叠中还有 2 个 pair 的标签集合不一致。BindingDB 与 Human 没有完全 pair 重叠，但测试药物与测试靶点仍大量出现在训练中。

此外，训练代码在释放 train 文件内部重新做五折 StratifiedKFold，读取的释放 validation 文件并未用于该模型选择过程。任何移植必须重新使用冻结的 BioMaster S1–S5 切分，并修复代码语义交换和 pair 重复/冲突后才能称为 corrected port。

# AD-LSF 官方代码、数据与切分审计（2026-08-14）

## 裁决

AD-LSF 是目前更近的多模态架构先例，但论文 headline 来自 warm random pair split，不能与 BioMaster 的实体冷启动 S1–S5 直接比较。官方仓库附原始数据和 cold-drug/cold-target 文件，却缺失论文 random split、预计算嵌入、模型权重、依赖清单和一个被代码强制导入的 `model/Fusion1.py`。

| 数据集 | 论文行数 | 释放行数 | 唯一完整行 | 重复正/负行 | 释放药物/靶点 | 与论文主要差异 |
|---|---:|---:|---:|---:|---:|---|
| bindingdb | 49,199 | 49,199 | 49,199 | 0/0 | 14,643/2,623 | 一致 |
| biosnap | 27,464 | 27,464 | 27,457 | 0/7 | 4,505/2,181 | drugs -5 |
| human | 5,997 | 6,728 | 5,997 | 731/0 | 2,726/2,001 | rows +731; positive +731 |
| celegans | 7,785 | 7,786 | 6,552 | 1,234/0 | 1,767/1,876 | rows +1; negative +1 |

Human 的 6,728 个释放行恰好由论文的 5,997 个唯一 interaction 加 731 个重复阳性组成；C. elegans 释放文件也有 1,234 个重复阳性。训练时这些不是权重，而是被当作独立样本读取，并在部分 released cold split 的 validation/test 间形成相同 pair 重叠。

## 代码语义

- 论文 random split 文件没有释放，生成脚本 `np.random.shuffle` 前不设 seed；seed 42 与 43 的代表性重放产生不同测试成员，因此无法识别论文五次实验的确切切分。
- 释放的 cold-drug/cold-target 文件覆盖四个原始文件的全部行，TRAIN 在命名冷轴上与 VALID/TEST 不相交；但生成脚本不支持这两个类型，且重复样本导致 VALID/TEST pair 重叠。
- `model/dti.py` 强制导入不存在的 `model/Fusion1.py`，仓库也没有依赖清单、嵌入、checkpoint 或日志，因而不能按发布状态运行。
- 运行时隔离验证显示 positional encoding 把位置向量加在 batch slot 而非 token 位置：同一样本所有 token 的位置差为 0，batch 置换等变误差为 `1.530295`。
- 论文写学习率 5e-4；释放配置实际为 BindingDB/BioSNAP 1e-4、Human/C. elegans 5e-5。论文称五次独立重复，但仓库只固定 seed 42，没有重复调度、预测或日志。

## 对 BioMaster 的约束

非对称门控、动态频率分解、潜在信号解耦和双向交互对齐均已有直接先例，不能成为 EQIR 的模块首创。AD-LSF 的高 random-split 数值不证明严格冷启动能力；若后续纳入数值对照，只能构建明确标注的 corrected AD-LSF-inspired port，在冻结 S1–S5 前去重并修复位置语义。

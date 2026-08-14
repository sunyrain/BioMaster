# MFDR-DTI 官方代码、数据与切分审计（2026-08-14）

## 裁决

MFDR-DTI 是多源特征融合的近期先例，但其论文 headline 不能作为 BioMaster 的同数据 SOTA 对照。官方仓库按发布状态不可执行，且实际切分不是覆盖全部样本的五折交叉验证，而是五个模型反复评估同一个 warm random 固定测试集。

| 数据集 | 释放行数 | 正/负 | 药物 ID/靶点 ID | SMILES/序列 | 固定测试行（实际评估） | 测试药物/靶点在 trainval 已见 | trainval-test ID pair 重叠/反标签碰撞 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DrugBank | 35,022 | 17,511/17,511 | 6,647/4,294 | 6,645/4,254 | 7,003 (6,992) | 0.974/0.993 | 87/3 |
| Enzyme | 5,840 | 2,920/2,920 | 444/660 | 444/660 | 1,167 (1,152) | 0.930/0.949 | 92/92 |
| GPCRs | 6,197 | 3,098/3,099 | 567/296 | 567/296 | 1,238 (1,232) | 0.940/0.946 | 270/258 |

## 代码语义

- `RunModel.py` 第 38 行存在发布阻断语法错误；代码还依赖三个 `/public/home/...` 绝对路径，仓库未附对应 BERT 权重。
- 所有数据先按 pair 随机打乱，前 80% 作为内层五折 train/valid；同一个后 20% 测试集被五个模型重复评价，最后一个打乱行还被 `[-1]` 静默排除。
- train、valid、test 三个 DataLoader 都使用 `drop_last=True`；checkpoint 依据 validation accuracy 选择，发布代码没有实际调用 early stopping。
- 164 维描述符在任何切分前由全数据 `StandardScaler.fit_transform` 处理，构成无标签测试分布泄漏。
- 每个特征向量先被扩成长度 1 的序列，注意力 softmax 永远只有一个元素。改变 query 后输出最大差为 `0.0`，WQ/WK 最大梯度均为 0；它不是实际的 query-dependent cross-attention。
- DWLoss 的三个权重都有非零梯度，但不属于 model optimizer；一次 `optimizer.step()` 后权重变化全部为 0。因此发布实现的动态权重固定在初始化值。

## 对 BioMaster 的约束

多源特征解耦、cross-property attention、同质特征保留和动态多分支损失均不能作为我们的单独创新。MFDR-DTI 的高指标来自不同数据、warm pair 切分和不可复现发布语义，不能直接与 S1–S5 比高低；EQIR 必须继续依赖同数据实体冷启动、双查询风险和严格消融建立贡献。

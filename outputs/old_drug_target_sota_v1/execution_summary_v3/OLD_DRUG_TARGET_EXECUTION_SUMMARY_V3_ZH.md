# 老药新靶点计算与验证总览 V3

状态：PASS；结论边界：NO_FULL_SOTA_CLAIM。

## 不可变靶点范围

- 888 个官方靶点；前序硬门槛淘汰 480，后续口袋质量淘汰 24。
- 活跃 384 = 338 个实验口袋主线 + 46 个仅因无实验口袋而恢复的靶点。
- 720 个老药 × 384 个靶点 = 276,480 对；480 个硬门槛淘汰靶点从未恢复。

## DTIAM 同数据 AUPRC

| 协议 | DTIAM | BioMaster | BioMaster 证据 | 完成折 |
|---|---:|---:|---|---|
| S1_SCAFFOLD_COLD_DRUG | 0.8829 | 0.9267 | BIOMASTER_STACK_SINGLE_SEED_20260813 | 0;1;2;3;4 |
| S2_HOMOLOGY_COLD_TARGET | 0.8145 | 0.7895 | BIOMASTER_STACK_FIVE_SEED_ENSEMBLE | 0;1;2;3;4 |
| S3_STRICT_DOUBLE_COLD | 0.6517 | 0.6624 | BIOMASTER_STACK_FIVE_SEED_ENSEMBLE | 0;1;2;3;4 |
| S4_FIRST_SEEN_TEMPORAL_2023_2025 | 0.7962 | 0.7957 | BIOMASTER_STACK_SINGLE_SEED_20260813 | -1 |
| S5_OLD_DRUG_ENTITY_COLD | 0.3988 | 0.5623 | BIOMASTER_STACK_FIVE_SEED_ENSEMBLE | -1 |

## 当前结论

- DTIAM 使用官方 BerMol/ESM2 表征和相同冻结监督数据；AutoGluon 为兼容重训，不是论文旧环境的逐位复现。
- 公开检查点路线与同数据重训路线分开报告；存在源重叠的公开检查点不能支撑无泄漏 SOTA 声明。
- KIRHub 只作训练后源独立审计，不用于融合系数拟合；现有证据仍不足以替换冻结 V10 老药中心头。
- 真实前瞻实验结果仍为 0，因此维持 NO_FULL_SOTA_CLAIM。

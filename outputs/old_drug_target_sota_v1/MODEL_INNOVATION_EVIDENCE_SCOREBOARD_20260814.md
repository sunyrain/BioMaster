# BioMaster 模型创新证据记分牌（2026-08-14）

## 结论

当前没有通过验证的核心算法创新，结论保持 `NO_FULL_SOTA_CLAIM`。已经成立的是严格范围治理、双方向老药重定位任务、冷启动/时间外评估和标签盲结构增量验证协议；这些属于系统与评价创新。

V1 与验证集融合保留为强传统基线；V2 已被五种子审计淘汰；V3 EQIR 只是冻结候选机制，必须先通过结构 Stage 0，才能训练完整模型。
数学审计还确认：EQIR 的概率残差误差与直接最终概率误差代数等价，残差重写本身不计为创新；只有资格、精确回退与双查询约束的联合机制仍可作为待证伪候选。

## S1–S5 统一结果

| 协议/范围 | 模型 | micro AUPRC | target-macro | drug-macro | 靶点查询信号 | 老药查询信号 | 定位 |
|---|---|---:|---:|---:|:---:|:---:|---|
| S1_SCAFFOLD_COLD_DRUG / ALL_FOLDS_0_4 | dtiam_probability | 0.88286 | 0.83860 | 0.84131 | ✓ | ✓ | PUBLIC_SAME_DATA_BASELINE |
| S1_SCAFFOLD_COLD_DRUG / ALL_FOLDS_0_4 | biomaster_stack_score | 0.92672 | 0.87396 | 0.84511 | ✓ | ✓ | CUSTOM_CONVENTIONAL_MODEL_SINGLE_SEED |
| S1_SCAFFOLD_COLD_DRUG / ALL_FOLDS_0_4 | DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION | 0.93646 | 0.88495 | 0.86200 | ✓ | ✓ | STRONG_NONNOVEL_BASELINE |
| S2_HOMOLOGY_COLD_TARGET / ALL_FOLDS_0_4 | dtiam_probability | 0.81447 | 0.77691 | 0.57869 | ✓ | ✗ | PUBLIC_SAME_DATA_BASELINE |
| S2_HOMOLOGY_COLD_TARGET / ALL_FOLDS_0_4 | biomaster_stack_score | 0.78951 | 0.75385 | 0.56358 | ✓ | ✗ | CUSTOM_CONVENTIONAL_MODEL_FIVE_SEED |
| S2_HOMOLOGY_COLD_TARGET / ALL_FOLDS_0_4 | DTIAM_BIOMASTER_SAME_DATA_FUSION | 0.81632 | 0.77980 | 0.55963 | — | — | NONNOVEL_BLEND |
| S3_STRICT_DOUBLE_COLD / ALL_FOLDS_0_4 | dtiam_probability | 0.65167 | 0.69029 | 0.65971 | ✓ | ✓ | PUBLIC_SAME_DATA_BASELINE |
| S3_STRICT_DOUBLE_COLD / ALL_FOLDS_0_4 | biomaster_stack_score | 0.66235 | 0.67475 | 0.63276 | ✓ | ✗ | CUSTOM_CONVENTIONAL_MODEL_FIVE_SEED |
| S3_STRICT_DOUBLE_COLD / DEVELOPMENT_FOLDS_0_2_CROSSFIT | QUERY_BALANCED_BLEND | 0.66747 | 0.69526 | 0.65706 | ✓ | ✓ | STRONG_NONNOVEL_BASELINE |
| S4_FIRST_SEEN_TEMPORAL_2023_2025 / FROZEN_TEMPORAL_TEST | DTIAM_RAW | 0.79620 | 0.76531 | 0.83333 | ✓ | ✓ | PUBLIC_SAME_DATA_BASELINE |
| S4_FIRST_SEEN_TEMPORAL_2023_2025 / FROZEN_TEMPORAL_TEST | BIOMASTER_STACK_SINGLE_SEED_20260813 | 0.79573 | 0.77602 | 0.77509 | ✓ | ✓ | CUSTOM_CONVENTIONAL_MODEL_SINGLE_SEED |
| S4_FIRST_SEEN_TEMPORAL_2023_2025 / FROZEN_TEMPORAL_TEST | DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION | 0.81952 | 0.80125 | 0.80862 | ✓ | ✓ | STRONG_NONNOVEL_BASELINE |
| S5_OLD_DRUG_ENTITY_COLD / FROZEN_OLD_DRUG_TEST | DTIAM_RAW | 0.39883 | 0.60605 | 0.73650 | ✓ | ✓ | PUBLIC_SAME_DATA_BASELINE |
| S5_OLD_DRUG_ENTITY_COLD / FROZEN_OLD_DRUG_TEST | BIOMASTER_STACK_FIVE_SEED | 0.56234 | 0.63634 | 0.75739 | ✓ | ✓ | CUSTOM_CONVENTIONAL_MODEL_FIVE_SEED |
| S5_OLD_DRUG_ENTITY_COLD / FROZEN_OLD_DRUG_TEST | DTIAM_BIOMASTER_SAME_DATA_ROUTED_FUSION | 0.57895 | 0.66737 | 0.77090 | ✓ | ✓ | STRONG_NONNOVEL_BASELINE |
| S5_OLD_DRUG_ENTITY_COLD / FROZEN_OLD_DRUG_TEST | SCOPE_PUBLIC_CHECKPOINT | 0.64277 | 0.73559 | 0.84833 | — | — | EXTERNAL_OVERLAP_LIMITED_REFERENCE |

## 机制裁决

- V1 routed stack：自定义但仍是传统组件组合；保留为基线。
- validation-only routed fusion：S1 很强，但属于集成工程，且 S2/S4 的老药方向不一致。
- BiRoute V2：仅 3/5 种子同向，双聚类轴置信区间跨 0；不晋级。
- EQIR V3：潜在创新点是“标签独立资格 + 交叉拟合结构增量 + 精确回退 + 双查询增量约束”的整体机制，而非残差重写、docking、注意力、MoE 或简单多模态融合本身。尚未完成实证。
- MFDR-DTI 审计进一步排除了多源解耦、cross-property attention 和动态多分支损失的单独新颖性；Stage 0 若通过，必须加入 direct multiview 与 auxiliary-branch 容量匹配控制。
- AD-LSF 审计进一步排除了非对称动态门控、潜在信号协调和双向交互对齐的单独新颖性；Stage 0 若通过，必须加入 corrected asymmetric-gated 与 bidirectional-alignment 容量匹配控制。

## 下一道硬门槛

完整标签盲 GNINA 队列必须先结束；Stage 0 任一总体/方向/置换/双聚类门槛失败，就停止结构路线，不训练完整 EQIR，也不宣称算法创新。

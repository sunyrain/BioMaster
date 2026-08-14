# BioMaster 最终收束报告（2026-08-14）

## 一句话结论

本轮已经完成靶点范围治理、720 个老药部署、19 篇近年方法审计、S1–S5 同数据冷启动比较、外部/来源保持验证和三条突变创新路线的机制检验；但证据结论是：**当前没有通过验证的核心算法创新，也没有成立的 full SOTA。**

当前最准确的定位是：

> BioMaster 是一个覆盖完整、证据路由严格、可追溯性较强的老药–靶点排序系统；其现有预测器主要是传统表示与模型融合，在若干协议上是强基线，但尚未证明普适算法优势。

统一声明：`NO_FULL_SOTA_CLAIM`。

## 1. 不可变靶点和部署范围

| 层级 | 数量 | 最终处理 |
|---|---:|---|
| 官方初始候选 | 888 | 仅为上游全集，不直接进入计算 |
| 前序硬门槛淘汰 | 480 | 永不恢复，未因口袋缺失而找回 |
| 硬门槛后候选 | 408 | 进入结构/口袋质量审计 |
| 后续口袋质量淘汰 | 24 | 永不恢复 |
| 活跃靶点 | **384** | 最终部署全集 |
| ├─ 严格实验口袋主线 | **338** | 可进入实验结构、docking 和结构证据分支 |
| └─ 仅因无实验口袋找回 | **46** | 保留序列/化学/数据库证据；不能伪装为实验结构靶点 |
| 老药 | **720** | 药物中心检索查询集 |
| 部署 pair | **276,480** | 720 × 384，已全部生成评分 |

项目从未恢复 480 个硬门槛淘汰项，也没有把 888 个靶点全部重新纳入。

## 2. 训练数据与评估框架

冻结的监督训练集为 ChEMBL37 人类单蛋白直接结合数据：

- 86,674 个 pair；
- 45,983 个活性、40,691 个数值弱活性/明确无活性；
- 62,488 个化合物、428 个训练靶点；
- 未记录 pair 从未当作真实负样本。

完成的协议：

- S1：药物骨架冷启动；
- S2：靶点同源簇冷启动；
- S3：药物骨架 + 靶点同源双冷启动；
- S4：2023–2025 首次记录时间外测试；
- S5：720 个老药实体及其骨架完全从拟合中排除。

所有 S1–S5 预定折均已运行；S2、S3、S5 有五种子 BioMaster 结果，S1、S4 的 BioMaster 主结果仍只有单种子，因此相关证据强度较低。

## 3. 近年论文与 SOTA 审计

已系统审计 19 个主要方法：1 个 2023、1 个 2024、7 个 2025、10 个 2026。包括 DTIAM、SCOPE-DTI、EviDTI、GS-DTI、SP-DTI、DrugLAMP、FusionDTI、DrugCLIP、DrugCMF、TAPB、KGOT、CrossLinker、ConfDTI、FMIN、MMTF-DTI、GADFDTI、GATv2-TransDTI、MFDR-DTI、AD-LSF。

审计维度包括训练来源、样本规模、负样本语义、冷启动切分、headline 指标、官方代码/权重、实现缺陷、与 BioMaster 候选创新的碰撞关系。

关键结论：

- random/warm split 高分不能与本项目冷启动 AUPRC 横向比较；
- 多个公开实现存在 validation/test 复用、重复 pair、标签冲突、模态语义交换或源码不完整；
- attention、GNN、MoE、动态融合、置信度融合、缺失模态路由、双向排序、因果去偏、MNAR、optimal transport 和孪生差分均已有明确先例；
- 不能通过更换编码器或增加网络层数获得可信“创新”声明。

## 4. 当前主任务数值结果

### 4.1 与 DTIAM 的同数据结果

| 协议 | DTIAM micro AUPRC | BioMaster V1 | 最强非新颖融合 | 裁决 |
|---|---:|---:|---:|---|
| S1 骨架冷 | 0.88286 | 0.92672 | **0.93646** | BioMaster 对 DTIAM 有显著优势，但 V1 单种子、融合非新颖 |
| S2 同源冷 | **0.81447** | 0.78951 | 0.81632 | V1 显著落后；融合仅微小提升且老药方向退化 |
| S3 双冷 | 0.65167 | 0.66235 | **0.66747**（开发折） | 点估计提高，但双聚类置信区间未全面通过 |
| S4 时间外 | 0.79620 | 0.79573 | **0.81952** | micro 改善，但老药方向不一致，且是已看过的测试 |
| S5 老药冷 | 0.39883 | 0.56234 | **0.57895** | 相对 DTIAM 明显提高，是现有最有价值应用结果 |

V1 相对 DTIAM 的严格 bootstrap 证据：

- S1：差值 `+0.04386`，骨架聚类 95% CI `[+0.03431, +0.05442]`；
- S2：差值 `-0.02496`，同源簇 95% CI `[-0.03734, -0.01435]`；
- S3：差值 `+0.01068`，两个聚类轴 CI 均跨 0；
- S5：差值 `+0.16350`，老药聚类 95% CI `[+0.08125, +0.24744]`。

这说明模型擅长新骨架和老药实体检索，但对未见同源靶点的表示泛化仍是核心短板。

### 4.2 外部和更强参考

- S5 上 SCOPE 公开 checkpoint micro AUPRC 为 `0.64277`，高于 BioMaster 融合的 `0.57895`；但 SCOPE 存在公开源重叠，属于 overlap-limited reference，不能当作完全独立 head-to-head。
- 来源保持 KIRHub 严格未报告子集有 2,823 pair、202 阳性：集成模型 micro AUPRC `0.18365`，V10 为 `0.18580`，差异置信区间跨 0；集成相对 SCOPE 有显著提升，但没有胜过 V10。
- 720 × 384 的 276,480 个部署分数和每药 Top-20 已生成，但它们是候选排序，不是前瞻实验验证。

## 5. 突变与药物条件创新路线

### 5.1 KiRHub 药物条件突变增量

数据：19,639 次测量、76 个药物、35 个基因、299 个变体构建体，完成药物冷、变体冷和基因冷各 5 折。

完整模型在药物冷 active AUPRC 从 `0.59437` 提高到 `0.60121`，但没有稳定胜过 WT-only、同基因打乱突变增量及去药物条件控制。冻结裁决：`PHASE_A_MECHANISM_NOT_SUPPORTED`。

### 5.2 PIC-DTA 配对删失潜在结果

DAVIS-complete 得到 3,960 个 WT–修饰体药物配对，其中 1,601 个精确效应。完整神经模型在变体冷 exact MSE 为 `1.226`、Pearson `0.208`；简单 Ridge 为 `0.603`、Pearson `0.322`。五个主要机制门槛全部失败。

冻结裁决：`PHASE_A_MECHANISM_NOT_SUPPORTED`。

### 5.3 TRACE-PL 原子接触共享势能差

在排除 PBCNet2 外部测试靶点及同源簇后，使用 Platinum 384 个样本、84 个同源簇、105 个配体，完成 50 个神经拟合。

同源冷结果：

| 模型 | RMSE | Pearson |
|---|---:|---:|
| TRACE-PL | 1.518 | 0.038 |
| 参数匹配直接配对网络 | 1.361 | 0.184 |
| 训练折均值 | 1.335 | 不具机制解释性 |

TRACE 相对直接配对 RMSE 差 `+0.157`，95% CI `[+0.034, +0.287]`，即显著更差。距离配体 ≤4 Å 的 212 个局部突变中 TRACE Pearson 仅 `0.005`，直接配对为 `0.201`。旋转异构体、原子接触、真实配体接触和共享势能差均未得到消融支持。

冻结裁决：`PHASE_A_MECHANISM_NOT_SUPPORTED; DO_NOT_OPEN_PBC65`。

### 5.4 PBCNet2.0 外部参考

官方模型与公开 65 个突变推理已精确复现，最大预测误差 `2.38×10⁻⁶`；8 个靶点宏平均 Pearson `0.54501`、Spearman `0.52968`。该数据保持为锁定外部测试，未用于 TRACE 训练、选模或调参。由于 TRACE 开发门槛失败，未消耗该外部测试。

## 6. 结构分支收束状态

标签盲 GNINA Stage 0 共 289 个 READY 靶点。截至收束：

- 完成 92/289 个靶点；
- 完成输出覆盖 4,830 个对照配体；
- 尚不足以运行预注册的总体、双方向、置换和双聚类 Stage 0 裁决；
- 队列已安全停止；完成状态保留，可断点续跑；当时正在运行的 8 个未完成任务未计入结果。

因此 EQIR 完整结构模型从未获得训练授权，不能作为算法创新或 SOTA 证据。

## 7. OFER-DTI 新候选的实际状态

为避免继续做静态二分类，已提出 OFER-DTI：把 pair 从“尚未记录”到“首次测得活性”或“首次测得弱/无活性”建模为竞争事件，并分离实验选择/记录 hazard 与条件活性结果；主要任务为老药内未来靶点检索。

已完成：

- prior-art 碰撞审计；
- 数据可行性审计；
- 训练前配置与失败门槛冻结，SHA256 `b11f437e...a4dc13`；
- 特征存储构建脚本语法验证。

可行数据：冻结 428 个训练靶点上有 509,168 个全量严格 binding pair，419,642 个合格首次事件；2015–2018 和 2019–2022 两个开发窗分别有 130,152 和 121,180 个事件；720 个老药中 281 个至少有历史事件。

未完成：特征存储未实际构建、模型未训练、机制门槛未验证。因此 OFER-DTI 只是未来候选，不计入本轮算法创新。

## 8. 最终可以与不可以声称的内容

### 可以声称

- 建立了严格区分硬门槛淘汰、口袋质量淘汰和“仅缺实验口袋找回”的 384 靶点系统；
- 建立了药物中心与靶点中心双方向、S1–S5 冷启动/时间外评价框架；
- BioMaster 在 S1 和 S5 相对 DTIAM 有显著同数据增益；
- 完成 19 个近年方法的训练数据、结果、实现和新颖性碰撞审计；
- 三个看似创新的突变机制均通过严格负对照被证伪，避免了虚假创新声明；
- 生成了 720 × 384 全量候选排序与可追溯证据路由。

### 不可以声称

- 不可以称 BioMaster 为 overall/full SOTA；
- 不可以称现有 Morgan/ProtBERT/MLP/树模型/融合为核心算法创新；
- 不可以称 docking、注意力、MoE、双向排序、删失损失或缺失模态路由为首次；
- 不可以将 276,480 个部署分数称为验证结合、作用机制或疗效；
- 不可以恢复 480 个硬门槛淘汰靶点或 24 个后续质量淘汰靶点；
- 不可以把 2023–2025 时间窗或已查看的外部结果重新包装成独立确认。

## 9. 本轮最终交付

主要结果文件：

- `outputs/old_drug_target_sota_v1/execution_summary_v3/OLD_DRUG_TARGET_EXECUTION_SUMMARY_V3.json`
- `outputs/old_drug_target_sota_v1/MODEL_INNOVATION_EVIDENCE_SCOREBOARD_20260814.md`
- `outputs/old_drug_target_sota_v1/SOTA_PRIMARY_PAPER_TRAINING_DATA_RESULTS_MATRIX_V12.csv`
- `outputs/old_drug_target_sota_v1/biomaster_scope_integrated_deployment_v1/BIOMASTER_SCOPE_INTEGRATED_720X384_V1.csv.gz`
- `outputs/old_drug_target_sota_v1/trace_pl_phase_a_v1/TRACE_PL_PHASE_A_SUMMARY_V1.json`
- `outputs/old_drug_target_sota_v1/trace_pl_failure_audit_v1/TRACE_PL_FAILURE_AUDIT_SUMMARY_V1.json`
- `outputs/old_drug_target_sota_v1/pbcnet2_mutation_benchmark_audit_v1/PBCNET2_MUTATION_BENCHMARK_AUDIT_SUMMARY_V1.json`
- `outputs/old_drug_target_sota_v1/first_event_dti_feasibility_v1/FIRST_EVENT_DTI_FEASIBILITY_SUMMARY_V1.json`
- `configs/biomaster_first_event_dti_phase_a_freeze_20260814.json`

## 最终决策

本轮研究在诚实证据意义上已经收束：保留 BioMaster V1/融合为强传统基线和部署排序器；三条失败创新路线归档；EQIR 因结构 Stage 0 未完成不晋级；OFER-DTI 仅保留为未来预注册候选。

**最终状态：系统与评估创新成立；核心算法创新未验证；full SOTA 未实现。**

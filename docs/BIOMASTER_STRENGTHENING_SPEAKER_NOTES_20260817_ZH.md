# BioMaster 强化算法阶段性汇报讲稿

日期：2026-08-17
建议时长：15–20 分钟（含 3–5 分钟讨论）
配套 HTML：`docs/biomaster-strengthening-briefing-20260817-zh.html`

## 一句话主线

本轮不是单纯增加模型复杂度，而是在冻结数据和五种冷启动角色下，筛选出当前最稳健的 pooled-ESM2 ODTI V2，并用内部 cluster-bootstrap、外部 BindingDB source-heldout 和明确的晋级门槛，把后续工作收束到 entity-cold 外部验证与 W1 湿实验闭环。

## 建议页序

### 1. 问题与目标（约 1 分钟）

目标是从老药–靶点配对空间中获得可验证的候选排序，同时回答三个问题：

1. 模型能否泛化到新骨架、新靶点同源簇、时间外和老药实体冷启动？
2. 新增的 ESM2、结构 residual、路由专家和 ranking 分支是否带来稳定增益？
3. 模型输出能否被外部来源和 prospective 湿实验检验，而不是只在同一数据集内自洽？

开场强调：当前目标是“可靠候选排序和验证闭环”，不是把模型分数直接解释为真实结合概率。

### 2. 五种 S 的实验角色（约 2 分钟）

| 角色 | 含义 | 主要输入/输出 | 当前状态 |
|---|---|---|---|
| S1 | scaffold-cold：新药物骨架 | 输入未见过的 scaffold 组合；输出 pair activity score | E0 unified：25 runs PASS；micro-AUPRC 0.931574；ECE 0.064389 |
| S2 | target-homology-cold：新靶点同源簇 | 输入同源簇外靶点；输出跨家族/同源边界的 pair score | pooled-ESM2：25 runs；micro-AUPRC 0.793853；增益主要体现在 scaffold 轴 |
| S3 | double-cold：新骨架 + 新同源簇 | 同时隔离 drug scaffold 和 target homology | E0 unified：25 runs PASS；micro-AUPRC 0.634394；target-macro AUPRC 0.671758 |
| S4 | temporal：时间外泛化 | 用较早时期训练、后续时期测试 | pooled-ESM2：5 seeds；micro-AUPRC 0.832526 |
| S5 | old-drug entity-cold：老药实体冷启动 | 药物实体不在训练配对中，测试对既有药物的跨靶点迁移 | pooled-ESM2：5 seeds；micro-AUPRC 0.556940，drug-macro AUPRC 0.775932；ECE 0.341023 |

解释重点：S1–S3 更偏结构/同源冷启动，S4 检验时间漂移，S5 更接近部署时的老药跨靶点筛选。五种 S 不是五个模型，而是五种泛化压力测试。

### 3. ODTI V2 架构（约 2 分钟）

基础输入：

- drug：Morgan radius-2 2048-bit，经 2048→512→192 编码；
- target：ProtBERT 1024，经 1024→512→192 编码；
- family：target assay-family embedding；
- optional target view：pooled ESM2-650M 1280→512→192 residual；
- optional structure：19 维 pair-aligned context，并带显式 `structure_mask`。

核心结构：

- pair features：drug、target、逐元素乘积、绝对差、learned bilinear interaction、family；
- 6 个 pair-conditioned routed experts；
- structure 只作为 gated residual，不替换 base score；
- 输出 activity logit、affinity 辅助头、observation propensity、log variance 和诊断门控量。

必须说清的安全不变量：

```text
structure_mask = 0  ⇒  final_logit == base_logit
```

这意味着结构缺失不会被模型错误地编码成负相互作用。

### 4. 这一轮哪些分支保留（约 1.5 分钟）

- 保留：pooled-ESM2 auxiliary。S2 scaffold-cold paired CI 为 `[+0.00201, +0.00602]`，支持小幅稳定增益。
- 暂不晋级：residue-token cross-attention。筛选相对 V2.1 的 micro-AUPRC 约 `-0.00710`，target-homology CI 为负。
- 暂不晋级：within-drug ranking + dual-query sampler。S5 2-seed micro-AUPRC 约 `0.4637`，低于同 seed baseline `0.5396`。
- 研究分支：OFER-DTI。已实现 competing-risk、observation hazard 和 listwise controls，但仍处 development screen。

转折句：因此当前 champion 的选择来自“跨角色稳定性和可审计性”，不是来自参数量或模块数量。

### 5. S1 正式结果与 bootstrap（约 2 分钟）

S1 V2.1 aggregate：

- 86,673 pair rows；5 folds × 5 seeds；25/25 runs；
- micro-AUPRC `0.930656`；target-macro `0.881221`；drug-macro `0.855139`；ECE `0.066674`。

统一 E0-S1 新 aggregate（现在已经属于完整 E0 汇总）为：

- 86,673 pair rows；5 folds × 5 seeds；25/25 runs，strict audit 25 PASS、0 FAIL；
- micro-AUPRC `0.931574`；target-macro `0.882337`；drug-macro `0.859378`；ECE `0.064389`；
- 相对 V2.1 S1，micro-AUPRC 约 `+0.000918`、drug-macro AUPRC 约 `+0.004239`、ECE 约 `−0.002285`；S1 paired cluster-bootstrap 已通过，但仍只属于内部证据。

2,000 次 scaffold cluster-bootstrap 的 AUPRC 差异：

| 对照 | Δ AUPRC | 95% CI |
|---|---:|---:|
| Target empirical-Bayes prior | +0.25860 | [+0.24757, +0.27109] |
| ConPLEx frozen external | +0.31846 | [+0.30548, +0.33072] |
| Train positive Morgan similarity | +0.03011 | [+0.02605, +0.03429] |

解释边界：这些 CI 支持 S1 scaffold-cold 的冻结内部增量；它们不是 entity-cold external validation，也不等价于 full SOTA。S3 paired bootstrap 对 DTIAM 的 AUPRC 差异约 `−0.0173`，95% CI 不跨 0，因此不能宣称当前模型在所有泛化协议上优于 DTIAM。

### 6. S2–S5 结果如何解读（约 2 分钟）

- S2：ESM2 带来 scaffold-cold 小增益，但 target-homology gain 的 CI 跨 0，说明远距离 target 泛化仍是瓶颈。
- S3：double-cold 是最严格的内部泛化角色，E0 unified micro-AUPRC `0.634394`、target-macro `0.671758`、ECE `0.069175`；相对 DTIAM 的 paired AUPRC 约低 `0.0173`，所以应诚实报告为“严格 double-cold 仍是短板”，而不是宣称全面领先。
- S4：时间外 micro-AUPRC `0.832526`，说明模型有一定 temporal transfer，但仍需来源外部验证。
- S5：drug-macro AUPRC `0.775932` 显示排序有用，但 ECE `0.341023` 表明概率校准不足；部署时应使用 ranking、ensemble spread 和 uncertainty triage，而非把 score 当作概率。

### 7. BindingDB 外部证据（约 1.5 分钟）

当前结果是严格 positive-only pair-heldout：

- strict support rows：628；
- 去除 ChEMBL37 exact overlap：65；
- 对齐 pair：235；aligned ligands/targets：44/101；
- Recall@5/10/20：`0.4925 / 0.5731 / 0.6514`；
- median best-positive rank：2。

正确表述：模型在不同来源的已知正例上有检索信号。限制是 aligned entities 仍在 feature universe 中，且没有可靠伪负样本，所以这不是 entity-cold external benchmark，也不能提供 calibrated binding probability。

### 8. 当前 champion 与闭环状态（约 1 分钟）

当前默认 champion：`pooled-ESM2 ODTI V2`。

已经完成：

- 模型实现、结构 fallback invariant、训练/评估入口；
- S1 25/25 formal 与 paired bootstrap；
- S2/S4/S5 的主要 pooled-ESM2 内部结果；
- BindingDB positive-only source-heldout retrieval；
- 64 个回归测试。

当前最终内部状态（2026-08-17）：E0 unified S1–S5 已全部完成，所有角色逐 run artifact audit PASS，S1/S3 paired cluster-bootstrap PASS，统一 E0 summary 为 PASS。

E0 的五个核心结果为：

- S1 scaffold-cold：micro-AUPRC `0.931574`，drug-macro AUPRC `0.859378`，ECE `0.064389`；
- S2 homology-cold：micro-AUPRC `0.793853`，target-macro AUPRC `0.754567`，ECE `0.147793`；
- S3 strict double-cold：micro-AUPRC `0.634394`，target-macro AUPRC `0.671758`，ECE `0.069175`；
- S4 temporal：micro-AUPRC `0.832526`，ECE `0.074813`；
- S5 old-drug entity-cold：micro-AUPRC `0.556940`，drug-macro AUPRC `0.775932`，ECE `0.341023`。

补充校准审计：在5个冻结 S5 checkpoint 上仅用 validation 拟合后处理校准器，test labels 没有参与拟合或方法选择。beta map 的 test ECE15 降到 `0.286668`，isotonic 降到 `0.277593`；但 isotonic 的 target-macro AUPRC 降到 `0.628751`，所以两者均只能作为 development candidate，不能直接替换当前部署口径。

E1 BERMOL768 已完成 S3/S5 两 seed paired screen，但冻结门槛未通过：S3 Δmicro-AUPRC `−0.002520`、Δdrug-macro AUPRC `−0.019310`；S5 Δmicro-AUPRC `−0.014566`、Δtarget-macro AUPRC `−0.016132`。因此 promotion decision 是 `KEEP_E0_CHAMPION`，不扩展五 seed、不替换 Morgan2048/ProtBERT/pooled ESM2。

仍未完成：

- 真正 entity-cold external validation；
- W1 的真实湿实验结果；
- OFER-DTI 的完整 Phase-A variant comparison 和 mechanism gates；
- E2 新蛋白 residual 与 E3 pair-specific 3D（均未启动）；E2 已完成本地权重盘点，当前状态是 `PROTOCOL_READY_BLOCKED_NO_DISTINCT_LOCAL_FEATURE_SOURCE`。ThermoProt 也已单独审计为“接收 ESM 输入的 IDP 结构/接触任务 checkpoint”，不是独立 protein-PLM，不能计入 E2 source。
- W1 V17 模板和 identity/provenance bridge 已 PASS，但真实 readout 尚未通过 activity/censor/replicate/assay 语义适配器，不能直接拿模板启动 V3。
- 当前已生成 semantic input template 和 fail-closed adapter；全 `PENDING/LOCKED` 模板被拒绝且不写出训练 CSV。只有真实解盲后的 assay semantic rows 通过 adapter，才允许进入 append-only observation store。

### 9. 下一阶段路线（约 1 分钟）

1. 冻结 pooled-ESM2 champion 和当前候选生成规则；
2. 完成去除同实体、同 active-moiety 和同源泄漏的 BindingDB/多来源 entity-cold 验证；
3. 在未揭盲条件下执行 W1，预先锁定候选、阴阳性对照和 assay readout；
4. 将真实 observation labels、active/inactive readout 和 assay quality 纳入下一轮再训练；先通过 W1→V3 training preflight，再启动 V3；
5. 只有当外部验证和 W1 结果支持排序，才宣称研究故事闭环。

Davis 外部评分现在已经完成：去除 ChEMBL37 exact overlap 后有 24,962 个 pair、8,120 个 both-unseen；mean affinity ≥6.0 的 derived-positive 检索 Recall@5/10/20 为 `0.1389/0.1878/0.2598`，both-unseen 为 `0.0154/0.0332/0.0812`。这只能称 positive-only numeric-affinity entity-cold retrieval，不能称 binary specificity 或 calibrated probability validation。

模型权重决策补充：当前不整体替换 Morgan2048、ProtBERT1024 或 pooled ESM2；BERMOL768 的受控 paired screen 已完成但未通过冻结非劣门槛，因此保留为负向 ablation，不进入 promotion。新蛋白权重和 pair-specific 3D 只在后续数据补齐后推进，且必须保留 structure fallback 和完整泄露审计。

外部验证新增一层：对原先因特征缺失而无法评分的9个 ligand 进行 exact InChIKey 结构复核；6个通过，3个因立体 InChIKey 不一致被隔离，4个对应 pair 没有被静默当作阴性。五 seed、6×428候选、12个 exact 正例的 ligand-cold positive-only retrieval 为 Recall@5/10/20 = 0.7167/0.7500/0.7500，21项身份、泄露、有限值和候选完整性审计全部通过。准确口径是 ligand-entity-cold，靶点仍已见，不是 both-entity-cold。

GtoPdb 的补充外部评估也已完成：3个 ligand-unseen、target-seen ligand，1,284个候选 pair，5个 exact positive；五 seed pooled-ESM2 检索 Recall@5/10/20 = 0/0.3333/0.4444，正例最佳排名中位数为16，19项独立审计全部通过。这里有2个 PubChem 立体结构缺口通过带来源哈希的本地结构救援解决；没有使用 GtoPdb 标签训练模型。该结果仍只支持 ligand-entity-cold positive-only retrieval，不能称为 target-cold、both-entity-cold 或 prospective validation。

## 汇报中建议避免的表述

- 不说“已经达到 full SOTA”；
- 不说“BindingDB 证明了真实结合概率”；
- 不说“结构模块普遍有效”；当前证据支持的是 gated residual 在特定内部角色上的增量；
- 不把 S1 的 V2.1、S2/S4/S5 的 pooled-ESM2 和 S3 的 V2.1 internal evidence 混称为完全统一的 champion suite；
- 不把 OFER 的 development screen 当成时间外生物学结论。

## 可能被问到的问题

**问：为什么不继续堆更复杂的结构或 residue-token 模块？**
答：因为当前筛选显示 residue-token 分支在有限预算下负迁移；结构模块只有在缺失回退和跨冷启动角色上同时稳定才值得晋级。当前最大不确定性已经转移到外部来源和 prospective 实验。

**问：S5 的 ECE 很高，模型还能用于候选选择吗？**
答：可以用于相对排序和 uncertainty triage，但不能直接把 score 当作结合概率。实验选择应使用 ranking、ensemble spread、assay readiness 和人工证据共同决策。

**问：什么时候故事才算闭环？**
答：当冻结模型在未参与训练的 entity-cold 外部来源和预注册 W1 湿实验中，排序与真实 readout 方向一致，并把新观测用于下一轮更新时，才算真正闭环。

## 现场打开顺序

1. HTML 汇报页：`docs/biomaster-strengthening-briefing-20260817-zh.html`；
2. S1 aggregate JSON；
3. S1 paired bootstrap JSON；
4. BindingDB summary JSON；
5. ODTI V2 design doc；
6. 8-hour audit Markdown。
7. 模型/数据 readiness 快照：`outputs/biomaster_odti_model_data_readiness_v1/BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.md`。
8. E1 BERMOL768 gate audit：`outputs/biomaster_odti_e1_gate_audit_20260817/BIOMASTER_ODTI_E1_GATE_AUDIT_V1.json`。

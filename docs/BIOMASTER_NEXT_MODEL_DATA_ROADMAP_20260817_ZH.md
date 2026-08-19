# BioMaster ODTI：模型与数据下一阶段推进路线

更新时间：2026-08-17
状态：E0 已完成；E1 BERMOL768 screen 不晋级；默认 champion 不变

## 结论

当前默认模型仍为 **pooled-ESM2 ODTI V2**。下一阶段不应以“继续增加模块数量”作为目标，而应按以下顺序推进：

1. 已用同一 frozen champion 配置完成 S1–S5 统一确认性评估；
2. 不晋级未通过门槛的 BERMOL768，保留当前 E0 champion；
3. 完成 source-heldout、entity-cold 外部验证和 W1 prospective 实验；
4. 只有在数据与外部证据成立后，才推进新的蛋白 residual 或 pair-specific 3D 模型。

## 当前证据边界

已经完成的工作包括：

- ODTI V2 routed interaction ranker、bilinear interaction、6-expert routing、structure residual、pooled ESM2 auxiliary 和多任务 loss；
- S1 V2.1 25/25 formal runs 及 paired cluster-bootstrap；
- S2 pooled-ESM2 5-fold × 5-seed；
- S3 V2.1 5-fold × 5-seed；
- S4/S5 pooled-ESM2 5-seed；
- BindingDB positive-only pair-heldout retrieval；
- structure fallback、feature alignment、temperature validation-only 和 ensemble summary 审计。

仍未完成的关键 gate：

- 同一 pooled-ESM2 + structure-context champion 的 S1–S5 全套统一 confirmatory suite（已 PASS）；
- S1/S3 paired cluster-bootstrap（已 PASS）；
- 真正 source-heldout、entity-cold 外部 benchmark；
- W1 湿实验 active/inactive/failed readout；
- OFER-DTI Phase-A 完整时间外推与机制 gate。

## 当前不建议继续投入的分支

### Residue-token cross-attention

短预算筛选相对 V2.1 的 micro-AUPRC 下降约 0.00710，不能因为训练时间不足就默认长训会恢复。若重新启动，应先改为 pocket-local token、结构 mask 和固定长度策略，并重新锁定 paired benchmark。

### Within-drug ranking / dual-query

面向老药检索的筛选低于同 seed baseline，因此保持 opt-in，默认权重为 0。继续调权重前必须证明它改善的是每药 Top-K，而不是牺牲 target-cold 泛化。

### ConPLEx 或其他外部 pair score 直接入主模型

只保留为独立 baseline 或审计分支。若进入训练，必须在每个训练 fold 内重建 controls、计算 target-local percentile，并记录 label dependency。

## 推荐的受控模型实验

### E0：统一 champion（已完成，PASS）

固定：

- Morgan2048；
- ProtBERT1024；
- pooled ESM2 1280 residual；
- 19-D structure-context residual；
- 6 experts；
- 当前 loss 权重、temperature calibration 和 frozen S1–S5 split。

要求：S1–S5 的版本、特征 hash、label contract 完全一致后再汇总总表。

S1/S2/S3/S4/S5 已完成逐 run artifact audit，所有角色 aggregate 存在且 PASS；S1/S3
paired cluster-bootstrap 也 PASS。统一机器可读汇总见
`outputs/biomaster_odti_unified_e0_summary_v1/BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json`。

### E1：药物侧语言模型 residual（screen 已完成，不晋级）

先加入冻结的 SMILES/分子语言模型 embedding，与 Morgan 并行：

```text
drug = Morgan2048 + gated(molecular_PL_embedding)
```

当前已经完成模型层、训练入口、train-only normalization、checkpoint provenance、
CLI 参数和正式 S3/S5 两 seed screen。结果显示：S3 Δmicro-AUPRC = −0.002520、
Δdrug-macro AUPRC = −0.019310；S5 Δmicro-AUPRC = −0.014566、
Δtarget-macro AUPRC = −0.016132。冻结非劣门槛失败，因此不扩展五 seed、不替换
E0 champion。完整 gate audit 见
`outputs/biomaster_odti_e1_gate_audit_20260817/BIOMASTER_ODTI_E1_GATE_AUDIT_V1.json`。

冻结的两阶段筛选协议见
`configs/biomaster_odti_e1_drug_aux_screen_20260817.json`。候选采用 gate bias `-4.0`
的近零残差冷启动；先做 S3/S5 两 seed，同预算通过后再扩展到 S2/S3/S5 五 seed。

### E2：蛋白侧新权重 residual

保留 ProtBERT 和 pooled ESM2 base，单独加入新蛋白语言模型 residual。不得同时替换 drug encoder、target encoder 和 loss，否则无法归因。

当前 readiness 为 `PROTOCOL_READY_BLOCKED_NO_DISTINCT_LOCAL_FEATURE_SOURCE`：本地没有新的、独立于当前 ProtBERT/ESM2 champion 的合格 protein PLM feature matrix。现有 ProtBERT 是 base，ESM2-650M 已经是 target auxiliary，ESM2-8M 是同一模型家族且不符合冻结的 1280-D feature contract，因此不启动伪增量训练。机器可读审计：

`outputs/biomaster_odti_e2_feature_readiness_v1/E2_FEATURE_SOURCE_READINESS_V1.json`

E2 外部权重获取与验收契约已经固化，预期覆盖冻结的 428 个 target entity；当前不下载、不启动 formal screen：

`outputs/biomaster_odti_e2_external_weight_contract_v1/E2_EXTERNAL_WEIGHT_ACQUISITION_CONTRACT_V1.json`

候选 feature package 到达后，先通过 fail-closed validator：

`scripts/validate_biomaster_odti_e2_feature_package_v1.py`

对本地 ThermoProt checkpoint 的独立适用性审计也已完成：它是接收 ESM 输入的 IDP 结构/接触任务模型，包含结构任务头，但没有 BioMaster E2 所需的独立 pooled target feature matrix，因此不计入 distinct protein-PLM source，不启动 E2。审计见：

`outputs/biomaster_odti_e2_thermoprot_audit_v1/THERMOPROT_E2_ELIGIBILITY_AUDIT_V1.json`

DrugCLIP / Drug-The-Whole-Genome 的六折权重也已完成独立适用性审计。它的输入是 Uni-Mol 分子和三维 pocket（token、距离矩阵、edge type），输出是 pocket–ligand cosine/z-score；当前项目已有 722 个 ligand × 307 个 pocket 的独立推理矩阵。它不是 target sequence PLM，不能拼接进 E2 residual，也不能把 z-score 当作 Kd/IC50 或校准概率。该分支只保留为后续 E3 的独立 pocket–ligand comparator/结构 triage；模型权重和生成输出受 CC BY-NC 4.0 非商业限制。审计见：

`outputs/biomaster_odti_drugclip_audit_v1/DRUGCLIP_E2_ELIGIBILITY_AUDIT_V1.json`

S5 另已完成 validation-only post-hoc calibration audit：beta map 将 test ECE15 从 `0.3410` 降至 `0.2867`，isotonic 降至 `0.2776`；但 isotonic 的 target-macro AUPRC 从 `0.6415` 降至 `0.6288`。两者均为 development candidate，必须经 untouched external/W1 复验后才能考虑部署：

`outputs/biomaster_odti_s5_calibration_audit_v1/ODTI_S5_CALIBRATION_AUDIT_V1.json`

### E3：pair-specific 3D interaction

将 19-D target-context 升级为 pocket residue graph + ligand atom graph + distance/orientation features。该实验必须使用 target、scaffold、exact pair 三重泄露审计，优先覆盖 338 strict experimental-pocket targets。

## 数据补充原则

### 不要先简单扩充 ChEMBL 行数

当前训练池为 86,674 个 observed calibration pairs、428 个靶点。真正缺口是 observation bias、显式 inactive、来源多样性和 target/scaffold coverage，而不是单纯的行数。

### 外部数据分层

1. **先冻结外部测试集**：BindingDB、GtoPdb 或 PubChem BioAssay 的未重叠子集先作为 untouched external evaluation；
2. **再分训练子集**：只有外部测试集冻结后，剩余 source-heldout 数据才可进入 V3 训练；
3. **结构数据独立处理**：结构/pose 特征必须记录来源、构建体、口袋 QC、feature hash 和标签依赖；
4. **湿实验优先**：新 active、inactive、borderline、replicate variance 和 assay failure 都要保留，不能只回流阳性结果。

W1 V17 的模板和 identity/provenance bridge 已通过审计，但模板不是 V3 训练表。真实 readout 回流后，必须先生成 `activity_class`、`readout_value/unit`、`censor_lower/upper`、`replicate_id/variance`、`assay_status` 和 `assay_metadata_json`，再重新通过训练前 preflight；当前不得直接启动 V3：

`outputs/biomaster_odti_w1_v3_training_preflight_v1/W1_V3_TRAINING_PREFLIGHT_AUDIT_V1.json`

另外已生成 16 行 provenance-preserving semantic input template，并实现 fail-closed adapter：当前全 `PENDING/LOCKED` 模板被拒绝，未写出训练 CSV；只有真实解盲、QC、readout、censor bounds、replicate variance 和 raw-data hash 完整后才允许通过：

`outputs/biomaster_odti_w1_v3_semantic_input_v1/W1_V3_SEMANTIC_RESULT_INPUT_TEMPLATE_V1.json`

`outputs/biomaster_odti_w1_v3_semantic_adapter_v1/W1_V3_SEMANTIC_ADAPTER_AUDIT_V1.json`

## 晋级规则

任何新权重或新分支只有同时满足以下条件，才可以替换 pooled-ESM2 champion：

- S2/S3/S5 在相同 split、相同 seed 和相同训练预算下有稳定提升；
- 至少一个 target-level 或 drug-level cluster bootstrap 区间支持增益；
- calibration 不恶化，或者排序提升与概率校准清晰分开报告；
- structure missing fallback 仍逐元素成立；
- source-heldout external 不下降；
- 不使用 W1 结果调参后再把 W1 当测试集。

## 当前执行队列

可直接交接的逐项执行版见：
`docs/BIOMASTER_ACTIVE_EXECUTION_QUEUE_20260817_ZH.md`

```text
PASS     E0 unified S1–S5 aggregate + strict audits + S1/S3 paired bootstrap
FAIL     E1 BERMOL768 S3/S5 two-seed non-inferiority screen; keep E0 champion
PASS     Davis positive-only numeric-affinity entity-cold scoring (binary gate remains closed)
NEXT     W1 wet-lab panel and observation update
HOLD     E2 protein PL residual ablation (no distinct local feature source yet)
LATER    E3 pair-specific 3D pocket/ligand graph
```

统一套件完成后先运行 `scripts/audit_biomaster_odti_v2_suite_v1.py`。该审计会逐个
检查 checkpoint、prediction、结构维度、ESM2 hash、drug auxiliary 是否关闭、
有限值边界和 aggregate 是否存在；部分完成只报告 `INCOMPLETE`，不能被写成正式总表。

当前 BindingDB 对齐前缺口已单独审计：251 个 exact-pair 去重后的候选中，235 个是双方实体都已在当前 feature store 中的 pair，另有 16 个为 ligand unseen / target seen；没有 both-unseen pair。对齐后的正例覆盖率因此为 ligand 1.0、target 1.0，不能称为 entity-cold。机器可读审计见：

`outputs/biomaster_bindingdb_entity_cold_gap_audit_v1/BINDINGDB_ENTITY_COLD_GAP_AUDIT_V1.json`

GtoPdb 2026.2 的独立 ligand-cold 补充评估已经完成：3 个 ligand-unseen/target-seen ligand，1,284 个候选 pair，5 个 exact positive；Recall@5/10/20 = `0 / 0.3333 / 0.4444`，正例最佳排名中位数为16，19项独立审计全部通过。两个 PubChem 立体结构缺口通过带来源哈希的本地结构 rescue 解决；GtoPdb 标签没有回流训练。该结果只支持 ligand-entity-cold positive-only retrieval，不支持 target-cold、both-entity-cold 或 prospective validation。结果与审计见：

`outputs/biomaster_odti_gtopdb_ligand_cold_v1/GTOPDB_LIGAND_COLD_SUMMARY.json`

`outputs/biomaster_odti_gtopdb_ligand_cold_v1/GTOPDB_LIGAND_COLD_AUDIT_V1.json`

外部 entity-cold landscape 已进一步核对 BindingDB、GtoPdb 和 KiRHub：这些来源没有合格的 scored both-unseen binary benchmark；KiRHub 项目映射中有 WT 测量的目标全部已在 ChEMBL37 feature store。因此 `true_entity_cold_external` gate 仍为 `false`，但不再把“没有评分”与“没有数据”混为一谈：Davis 已完成 label-blind 特征生成和 preliminary positive-only numeric-affinity scoring，仍需要显式 inactive/observation contract 才能打开 binary entity-cold gate：

`outputs/biomaster_odti_external_entity_cold_landscape_v1/EXTERNAL_ENTITY_COLD_LANDSCAPE_V1.json`

本地外部文件进一步审计后，Davis 是当前最值得推进的候选来源：30,056 行、379 个蛋白序列、68 个配体、25,772 个 exact unique pairs；去掉与冻结 ChEMBL37 的 810 个 exact pair overlap 后剩 24,962 个 non-overlap pairs，其中 8,120 个为 both-unseen。Davis 只有 numeric affinity，没有可靠的 explicit inactive，因此它只能支持 positive-only entity-cold retrieval，不能直接支持二分类 specificity 或概率校准。现在 379 条 target sequence 的 ProtBERT/ESM2 特征和 68 个 ligand Morgan 特征已经冻结，评分也已完成：主规则为 pair-level mean affinity ≥6.0，共 4,142 个 derived positives；整体 Recall@5/10/20 = `0.1389/0.1878/0.2598`，both-unseen 分层为 `0.0154/0.0332/0.0812`。该结果应写成“Davis positive-only numeric-affinity entity-cold retrieval”，不能写成 binary entity-cold validation：

`outputs/biomaster_odti_local_external_candidates_v1/LOCAL_EXTERNAL_ENTITY_COLD_CANDIDATES_V1.json`

`outputs/biomaster_odti_local_external_candidates_v1/DAVIS_ENTITY_COLD_PAIR_MANIFEST_V1.csv.gz`

评分摘要与特征 provenance：

`outputs/biomaster_odti_davis_entity_cold_v1/DAVIS_ENTITY_COLD_SCORING_SUMMARY_V1.json`

`outputs/biomaster_odti_davis_feature_store_v1/DAVIS_FEATURE_STORE_MANIFEST_V1.json`

当前 pooled-ESM2 五 seed 的 BindingDB positive-only source-heldout 评估已重新生成，包含 18,832 个候选 pair、44 个可评估 ligand，Recall@5/10/20 分别为 0.4925/0.5731/0.6514；该结果仍只支持 pair-heldout retrieval，不支持 negative、entity-cold 或 SOTA claim。机器可读结果见：

`outputs/biomaster_bindingdb_source_heldout_current_20260817/BINDINGDB_PAIR_HELDOUT_SUMMARY.json`

该外部证据的输入 hash、重叠排除、实体覆盖和 claim contract 已冻结并通过严格验证：

`configs/biomaster_odti_external_evaluation_freeze_20260817.json`

审计结果：

`outputs/biomaster_odti_external_evaluation_freeze_audit_20260817/EXTERNAL_EVALUATION_FREEZE_AUDIT_V1.json`

统一运行状态快照：

`outputs/biomaster_odti_live_status_v1/BIOMASTER_ODTI_LIVE_STATUS_V1.json`

ODTI V2 角色级证据矩阵见：
`outputs/biomaster_odti_v2_evidence_matrix_v1/BIOMASTER_ODTI_V2_EVIDENCE_MATRIX_V1.json`。

## 结论口径

当前最准确的说法是：

> ODTI V2 已完成统一 E0 内部验证；BERMOL768 受控 screen 未通过非劣门槛，因此当前 champion 不变。下一阶段的核心是外部 entity-cold 和 W1 实验闭环；新预训练权重只允许以受控 residual ablation 进入，不直接整体替换。

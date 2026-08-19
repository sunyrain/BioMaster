# BioMaster ODTI：当前执行队列与收口规则

更新时间：2026-08-17
用途：把当前“全面推进”拆成可检查、可交接、不会造成数据泄露的执行单元。

## 当前总状态

```text
E0 unified S1–S5                  PASS
S1/S3 paired cluster-bootstrap    PASS
E1 BERMOL768 screen              FAIL / DO NOT PROMOTE
true entity-cold external        POSITIVE-ONLY SCORED / BINARY GATE CLOSED
W1 infrastructure                PASS / AWAITING REAL ASSAY DATA
W1 prospective readout           NOT AVAILABLE
E2 protein residual              NOT STARTED
E3 pair-specific 3D             NOT STARTED
```

当前 champion 固定为：

```text
Morgan2048 + ProtBERT1024 + pooled ESM2-650M 1280 residual
+ 19-D structure-context residual + 6 routed experts + 5-seed ensemble
```

## 队列 A：冻结当前证据，不再重复无目的长训

### A1. 当前模型与声明边界

权威输入：

- `outputs/biomaster_odti_unified_e0_summary_v1/BIOMASTER_ODTI_UNIFIED_E0_SUMMARY_V1.json`
- `outputs/biomaster_odti_e1_gate_audit_20260817/BIOMASTER_ODTI_E1_GATE_AUDIT_V1.json`
- `outputs/biomaster_odti_model_data_readiness_v1/BIOMASTER_ODTI_MODEL_DATA_READINESS_V1.json`
- `outputs/biomaster_odti_external_entity_cold_landscape_v1/EXTERNAL_ENTITY_COLD_LANDSCAPE_V1.json`

执行规则：

1. 不扩展 BERMOL768 到五 seed；
2. 不整体替换 Morgan2048、ProtBERT 或 pooled ESM2；
3. 不把 S5 score 当真实 binding probability；
4. 不把 BindingDB/GtoPdb ligand-cold positive-only retrieval 称为 both-entity-cold；
5. 不把 W1 同一批结果同时用于调参和最终测试。

## 队列 B：外部 entity-cold 数据

当前 landscape audit 已确认：BindingDB、GtoPdb 和 KiRHub 的已完成产物中，没有合格的 both-unseen binary benchmark。进一步审计本地 Davis 后，已完成一个 positive-only numeric-affinity 外部评分：去重后 25,772 个 pair，去掉 810 个 ChEMBL37 exact overlap 后剩 24,962 个 non-overlap，其中 8,120 个为 both-unseen。Davis 的评分链路已经完成，但因为没有 explicit inactive/observation contract，true binary entity-cold gate 仍必须保持关闭。

Davis 的冻结候选审计与 pair manifest：

```text
outputs/biomaster_odti_local_external_candidates_v1/LOCAL_EXTERNAL_ENTITY_COLD_CANDIDATES_V1.json
outputs/biomaster_odti_local_external_candidates_v1/DAVIS_ENTITY_COLD_PAIR_MANIFEST_V1.csv.gz
```

下一份合格来源必须同时满足：

| 条件 | 要求 |
|---|---|
| entity identity | exact ligand key、exact target key、版本/来源 ID |
| cold split | 至少 unseen ligand + unseen target，最好 both-unseen |
| source holdout | 在训练前冻结测试子集；训练只用剩余来源 |
| labels | positive、inactive/censored 或明确 observation contract |
| negative policy | unknown pair 不得自动当作 negative |
| audit | overlap、实体覆盖、结构 hash、feature provenance 全部可重算 |

在具备 explicit inactive/observation contract 的合格来源到达前，只能把当前结果称为 positive-only entity-cold retrieval，不能宣称 binary external entity-cold 或 calibrated validation 已完成。

Davis 已完成 source-heldout positive-only entity-cold 评分：379 条 target sequence 的 ProtBERT/ESM2 特征、68 个 ligand Morgan 特征和 feature hash manifest 均已冻结。主分析使用 pair-level mean affinity ≥6.0，共 4,142 个 derived positives；整体 Recall@5/10/20 = `0.1389/0.1878/0.2598`，both-unseen 为 `0.0154/0.0332/0.0812`。Davis 没有 explicit inactive，不能用来建立 binary specificity 或 calibrated probability gate。

评分摘要：

```text
outputs/biomaster_odti_davis_entity_cold_v1/DAVIS_ENTITY_COLD_SCORING_SUMMARY_V1.json
```

特征 provenance：

```text
outputs/biomaster_odti_davis_feature_store_v1/DAVIS_FEATURE_STORE_MANIFEST_V1.json
```

S5 校准审计已经完成，但只属于 validation-only development artifact：beta ECE15 = `0.2867`，isotonic ECE15 = `0.2776`，均需 untouched external/W1 复验；当前 champion 仍使用原 temperature-calibrated ensemble。

## 队列 C：W1 湿实验闭环

W1 的计算准备已经完成，V17 ingestion infrastructure 独立审计为 `25/25 PASS`。模板位于：

```text
outputs/evidence_routing_compute_execution_20260808_v1/w1_result_ingestion_v17/input_templates_v17/
```

关键模板：

- `W1_PROCUREMENT_AND_RECEIVED_LOT_QC_INPUT_8_V17.csv`
- `W1_BLINDED_PRIMARY_RUN_RESULT_INPUT_16_V17.csv`
- `W1_CONTROLLED_UNBLINDING_KEY_16_V17.csv`
- `W1_ACTIVE_SPECIES_RESULT_INPUT_3_V17.csv`
- `W1_ORTHOGONAL_RESULT_INPUT_8_V17.csv`

实验数据回流顺序必须是：

```text
采购/批次 QC
→ 盲法 primary assay（每个候选两次独立 run）
→ 先锁定阳性/弱活性/载体/无靶 QC
→ 再受控解盲
→ active / inactive / borderline / failed 分类
→ active-species 与 orthogonal confirmation
→ 形成 prospective evaluation
→ 再训练 V3
→ 用 W2 或新外部集合复验
```

必须保留的字段包括：active、inactive、borderline、failed、replicate variance、assay metadata、raw-data hash、lot purity、readout interference 和 cytotoxicity。禁止只回流阳性结果。

## 队列 D：下一版模型 V3

V3 的第一版应优先解决标签和校准，而不是扩大网络：

当前冻结池的自动化数据契约见：
`outputs/biomaster_odti_v3_data_contract_v1/BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.json`
和
`outputs/biomaster_odti_v3_data_contract_v1/BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.md`

该契约确认：86,674 条 pair 中有 25,947 条显式 inactive，25,220 条缺少 numeric pChEMBL；当前没有 replicate variance、完整 censor bounds 或 observation indicator。

V17 W1 到 V3 的 identity/provenance bridge 也已建立并通过审计：
`outputs/biomaster_odti_w1_v3_bridge_v1/W1_V17_TO_ODTI_V3_BRIDGE_SUMMARY_V1.json`

该 bridge 包含 8 个候选、2 次 independent run（共 16 行），确认 operator-facing primary file 仍保持盲法；8 个候选均未在冻结 ChEMBL37 pair store 中出现 exact InChIKey–target pair，因此使用 namespaced `prospective_pair_id`，不伪造 `calibration_pair_id`。

1. 在冻结 W1 和外部测试集后，加入显式 inactive、censored affinity、assay metadata 和 replicate variance；
2. 保留 unknown pair 的 PU/未观测语义；
3. 用 group-aware calibration 分开报告排序和概率校准；
4. 重新运行 S1–S5、source-heldout、entity-cold 和 W1-independent summary；
5. 只有当 target-macro、drug-macro、cluster bootstrap 和 calibration 同时不恶化，才允许进入下一模型筛选。

## 队列 E：受控新权重实验

### E2：protein residual

只允许：

```text
E0 target encoder + frozen new protein PLM + gated residual
```

不允许同时替换 drug encoder、loss 和 split。优先在 S2/S3/S5 做两阶段 screen，随后才扩展五 seed。

### E3：pair-specific 3D

必须具备 pair-specific pocket/pose、构建体和质量分数，并进行 target、scaffold、exact-pair 三重泄露审计。缺结构时必须逐元素满足：

```text
structure_mask = 0  ⇒  final_logit == base_logit
```

E2/E3 都不是当前主线；它们必须等待外部数据和 W1 结果提供新的归因依据。

E2 的冻结筛选协议已经写入：

```text
configs/biomaster_odti_e2_protein_residual_screen_20260817.json
```

当前状态是 `PROTOCOL_READY_BLOCKED_NO_DISTINCT_LOCAL_FEATURE_SOURCE`：协议、门槛和禁止性变更已固定，但本地没有新的、独立于当前 ProtBERT/ESM2 champion 的合格 protein PLM feature matrix。现有 ProtBERT 是 base，ESM2-650M 已是 target auxiliary，ESM2-8M 是同一模型家族且不符合冻结的 1280-D contract，因此没有启动伪增量训练。readiness audit：

```text
outputs/biomaster_odti_e2_feature_readiness_v1/E2_FEATURE_SOURCE_READINESS_V1.json
```

外部权重获取与验收契约（冻结 target entity 预期 428 行）已固化：

```text
outputs/biomaster_odti_e2_external_weight_contract_v1/E2_EXTERNAL_WEIGHT_ACQUISITION_CONTRACT_V1.json
```

DrugCLIP / Drug-The-Whole-Genome 已单独审计为
`POCKET_LIGAND_3D_RETRIEVAL_MODEL_NOT_SEQUENCE_PROTEIN_PLM`：它只能作为后续 E3 的
pocket–ligand 独立比较器或结构 triage 层，不能作为 E2 protein residual；其六折模型
权重和生成输出受 CC BY-NC 4.0 非商业限制。审计：

```text
outputs/biomaster_odti_drugclip_audit_v1/DRUGCLIP_E2_ELIGIBILITY_AUDIT_V1.json
```

S5 同时完成了 validation-only post-hoc calibration audit：beta map 的 test ECE15 为 `0.2867`，isotonic 为 `0.2776`，但 isotonic target-macro AUPRC 下降到 `0.6288`；两者都必须经过 untouched external/W1 复验，暂不改变 champion：

```text
outputs/biomaster_odti_s5_calibration_audit_v1/ODTI_S5_CALIBRATION_AUDIT_V1.json
```

## 晋级闸门

任何候选模型只有同时满足以下条件才可替换 champion：

- 相同 split、seed、训练预算下 S2/S3/S5 稳定提升；
- 至少一个 target-level 或 drug-level cluster bootstrap 支持增益；
- ECE 或其他校准指标不恶化，或排序与校准明确拆分；
- structure fallback、有限值和 feature hash 审计通过；
- untouched source-heldout external 不下降；
- 不使用 W1 调参后再把 W1 作为测试集。

## 当前最短执行路径

```text
冻结 E0 预测与候选
→ 获取合格 unseen-target/both-unseen 来源
→ 执行已冻结的 W1 盲法实验
→ 回流完整 observation labels
→ 训练 V3 校准/负向监督版
→ 重新跑 S1–S5 + 外部 + W2
→ 通过后才评估 E2，再评估 E3
```

这条队列把当前已完成的计算证据、尚未获得的外部状态和允许启动的模型实验分开，避免把“基础设施 PASS”误写成“生物学验证 PASS”。

## 旧 DiffDock/Stage6 分支说明

总项目旧审计中仍能看到 2026-06-05 的 full-DiffDock tracker，但当前工作区没有对应的
`outputs/report_scale`、job index 或 score files。当前状态应写为：

```text
legacy full DiffDock: NOT_INITIALIZED_IN_CURRENT_WORKSPACE
legacy experiment closure: NOT_COMPLETE / separate branch
```

因此本轮不启动 913k 级别的旧 docking 队列，也不把旧分支的缺失输出当成 ODTI 模型失败。若未来重新纳入该分支，必须先重建并冻结 manifest、受体/配体身份、chunk job index、GPU 预算和技术失败 rescue policy，再单独运行和审计。

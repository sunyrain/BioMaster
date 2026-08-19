# BioMaster-ODTI V2 强化算法设计与实现

更新时间：2026-08-16
状态：V2.1 训练修正版已实现；V2.2 的 pooled ESM2 auxiliary、target-wise listwise loss 和 residue-token cross-attention 已实现并通过单测。药物侧 gated auxiliary residual 已完成训练/评估入口、train-only normalization、provenance 和 BERMOL768 smoke，但尚未正式晋级。ESM2 auxiliary 已完成 S2 五种子 × 五折正式套件；residue-token 分支已完成 2 seed × 5 fold 筛选但未晋级。BindingDB positive-only pair-heldout retrieval 已跑通，但更广泛 source/entity-cold external 和 prospective W1 仍未完成。

## 1. 为什么需要 V2

当前 V1 在 S1 scaffold-cold 和 S5 old-drug entity-cold 上有应用价值，但 S2 homology-cold target 泛化弱于 DTIAM。V1 的 pair interaction 主要由两个全局 embedding 的拼接、逐元素乘积和绝对差构成；它没有直接建模局部 pocket 或 pair-level structure evidence。

V2 不改变冻结的 ChEMBL37 数据和 S1–S5 切分。它把增强限制为三个可审计组件：

1. learned bilinear drug–target interaction；
2. optional structure/pocket residual；
3. multi-task activity/affinity/observation/contrastive objectives。

本轮训练修正还包括：pair-conditioned expert routing、target-aware minibatch、pairwise within-target ranking、可选 target-wise listwise ranking、异方差 interval-censored affinity NLL、train-only family vocabulary、结构输入 train-only 标准化，以及以 micro/target/drug retrieval 指标组成的 validation composite selection。

V2.2 还提供两类可选靶点增强：

1. pooled ESM2-650M auxiliary residual branch；
2. 由重叠窗口拼接的 residue-level ESM2 token feature store，以及 drug-conditioned target-token cross-attention。

两者默认关闭，只有在对应 feature hash、train-only normalization 和 paired split 实验完成后才能晋级为默认配置。

当前筛选结论：pooled ESM2 相对 V2.1 的 S2 scaffold-cluster micro-AUPRC 增益为
`+0.00393`（95% CI `[+0.00201, +0.00602]`），但 target-homology cluster CI 为
`[-0.00100, +0.00794]`，因此只能作为有希望的辅助分支；residue-token cross-attention
在 2-seed × 5-fold 短预算筛选中相对 V2.1 的 micro-AUPRC Δ 为 `-0.00710`，暂不设为默认。

pooled ESM2 的 S4 first-seen temporal 5-seed aggregate 为 micro-AUPRC `0.83253`、
micro-AUROC `0.77913`、target-macro AUPRC `0.78316`、drug-macro AUPRC `0.82508`。
这说明该分支在当前冻结 temporal role 上具有内部泛化信号，但仍不能替代 source-held-out
外部验证或 W1 prospective 实验。

S5 old-drug entity-cold 的 5-seed aggregate 为 micro-AUPRC `0.55694`、micro-AUROC
`0.78663`、target-macro AUPRC `0.64149`、drug-macro AUPRC `0.77593`。该角色的
ECE 仍较高（约 `0.34`），因此湿实验候选集合必须使用排序、ensemble spread 和
selective/uncertainty gate，不能直接把校准概率当作真实结合概率。

已实现 `scripts/evaluate_biomaster_odti_selective_uncertainty_v1.py`。在 S5 上，
保留最低 10% ensemble spread 的 exploitation 子集，micro-AUPRC 为 `0.90703`；
全量为 `0.55694`。这只能说明 seed spread 可用于风险分层，不能把它解释为已经
验证的 conformal guarantee 或真实结合概率。

V2 的结构分支是 residual，不是全量替换。缺结构时必须精确回退到 sequence/chemistry base。

## 2. 实现文件

- `biomaster/odti_v2.py`：模型、loss、censored affinity、contrastive alignment、ensemble summary。
- `scripts/train_biomaster_odti_v2.py`：沿用冻结 S1–S5 split contract 的训练入口。
- `scripts/evaluate_biomaster_odti_v2.py`：按 protocol/fold/seed 执行并聚合 V2，生成 pair-aligned seed mean 预测。
- `tests/test_odti_v2.py`：结构缺失回退、loss、结构 CSV 对齐、ensemble summary 测试。
- `scripts/build_biomaster_odti_target_token_features_v1.py`：构建无标签依赖的 residue-level ESM2 token store。
- `configs/biomaster_odti_v2_freeze_20260816.json`：V2 架构和晋级门槛。

## 3. V2 架构

```text
Morgan 2048 ── drug base encoder ────────────────┐
optional molecular PL 768 ── gated residual ─────┴─ d

ProtBERT 1024 ── target base encoder ────────────┐
optional pooled ESM2 ── gated residual ──────────┴─ t
optional ESM2 residue tokens + drug query ───────── target-token cross-attention

d + t + target family ───────────────────────────── pair features
                                                   (d, t, d*t, |d-t|, bilinear, family)
                                              │
                                              ├─ shared head
                                              ├─ 6 pair-conditioned gated expert heads
                                              ├─ cosine alignment
                                              └─ base_logit

optional pocket vector ── structure encoder ── gated residual ── final_logit

final_logit = base_logit + structure_mask × structure_gate × residual_logit
```

药物辅助分支采用实体对齐的 dense matrix，当前 smoke 使用
`DTIAM_BERMOL768_FLOAT32_V1.npy`。只用训练 fold 中出现的 drug index 计算逐特征
mean/std；融合形式为：

```text
drug = drug_base + drug_aux_mask × sigmoid(gate([drug_base, drug_aux])) × drug_aux
```

`drug_aux_mask` 由 BERMOL drug index 的可用性字段生成；当前 62,477 个训练药物中
62,476 个有可用 BERMOL 表示，唯一失败实体保持 quarantine，不被当作可用零向量参与
残差融合。

不传 `--drug-aux-features` 时，`drug_aux_input_dim=0`，模型和既有 checkpoint 语义
保持不变。该接口已通过 S3 128-row、1-epoch CPU smoke，但还没有正式多 seed
性能结论。

结构输入是 pair-aligned CSV，必须包含：

```text
calibration_pair_id,<numeric structure features...>,structure_mask(optional)
```

缺失 pair 会填充零向量并设 `structure_mask=0`。如果 `structure_mask=0`，代码检查要求 `final_logit` 与 `base_logit` 逐元素完全相等。

当前已生成一个安全的 target-context feature store：

```text
outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz
outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1_MANIFEST.json
```

该 store 只包含 receptor/pocket/experimental-structure 上下文，不直接写入由正负标签挑选出的 GNINA 校准分数。GNINA pair score 若要进入正式 fold，必须在每个训练 fold 内重新选择 controls、计算 target-local percentile，并由 manifest 记录其标签依赖。

## 4. 多任务目标

V2.1 的总损失为：

```text
target-balanced BCE
+ 0.12 × pairwise within-target ranking loss
+ drug_rank_weight × pairwise within-drug ranking loss (默认 0；面向 old-drug query 的可选分支)
+ expert_balance_weight × routed-expert load-balance loss (默认 0；仅用于防止专家坍缩的消融)
+ listwise_weight × target-wise listwise ranking loss (默认 0，可单独消融)
+ 0.06 × interval/censored affinity loss
+ 0.10 × observation propensity loss (有 observation labels 时启用)
+ 0.05 × bidirectional contrastive loss
```

当前 ChEMBL37 数据全部是已观测 pair，因此 observation head 在现有数据上默认不参与训练；它为后续 OFER-DTI 或带 observation indicator 的数据保留接口，避免把“未记录”误当作阴性。

若数据表包含 observation indicator，可通过 `--observation-column` 显式启用 observation propensity loss。

`censored_affinity_loss` 现在是 heteroscedastic Gaussian-style interval NLL，并使用模型输出的 `affinity_log_variance`。它支持：

- exact measurement：`lower == upper`；
- lower-bounded：只有 `lower`；
- upper-bounded：只有 `upper`；
- interval measurement：`lower < upper`。

V2 还加入了两个默认关闭的训练增强：

- `--drug-rank-weight`：对同一 drug 的 target 候选施加 within-drug pairwise ranking；
- `--expert-balance-weight`：对 routed expert 的 batch 平均使用率施加轻量 anti-collapse 正则。

若启用 `--drug-rank-weight`，建议同时使用 `--batch-sampler dual_query`。该 sampler
将每个 epoch 的行无重复地分配到 target-chunk 或 drug-chunk，再混合装箱，使两个
query 方向在 batch 中都有机会产生有效 ranking 对。它目前只是实验分支：S5 两种子、
`drug_rank_weight=0.08` 的 pooled-ESM2 screening micro-AUPRC 为 `0.46374`，低于
同两种子基线 `0.53964`，因此默认权重仍为 0。

## 5. 运行方式

最小 smoke run：

```bash
python scripts/train_biomaster_odti_v2.py \
  --protocol S3_STRICT_DOUBLE_COLD \
  --fold 0 \
  --epochs 1 \
  --max-rows 128 \
  --batch-size 64 \
  --cpu \
  --out-dir .tmp/biomaster_odti_v2_smoke
```

默认使用 target-aware minibatch 和 composite validation selection；如需复现旧版随机 batch，可加 `--random-batches`。

带结构特征时：

```bash
python scripts/train_biomaster_odti_v2.py \
  --protocol S2_HOMOLOGY_COLD_TARGET \
  --fold 0 \
  --structure-features outputs/structure_features.csv \
  --structure-dim 128 \
  --out-dir outputs/old_drug_target_sota_v1/biomaster_odti_v2
```

启用 pooled ESM2 auxiliary：

```bash
python scripts/train_biomaster_odti_v2.py \
  --protocol S2_HOMOLOGY_COLD_TARGET \
  --fold 0 \
  --target-aux-features outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy \
  --target-aux-dim 1280 \
  --out-dir outputs/old_drug_target_sota_v1/biomaster_odti_v2_esm2
```

构建 residue-level ESM2 token store：

```bash
python scripts/build_biomaster_odti_target_token_features_v1.py \
  --output-dir outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1
```

启用 token cross-attention 时，同时传入 `--target-token-features`、
`--target-token-index` 和 `--target-token-dim 1280`。token 模式会自动把训练
batch 限制为最多 64、推理 batch 限制为最多 128，以避免长蛋白 padding 导致不可审计的显存峰值。

先只查看正式任务计划，不启动训练：

```bash
python scripts/evaluate_biomaster_odti_v2.py \
  --protocol S3_STRICT_DOUBLE_COLD \
  --seeds 20260816,20260817 \
  --plan-only
```

V2 的正式多种子入口会为每个 fold/seed 产生 `RUN_SUMMARY_V2.json` 和
`TEST_PREDICTIONS_V2.csv.gz`，然后按 `calibration_pair_id` 聚合为
`<PROTOCOL>_V2_SEED_MEAN_PREDICTIONS.csv.gz`。聚合前不允许跨 fold 合并重复 pair；
S1–S3 仍按 frozen OOF 语义解释，S4/S5 仍按 fixed test 语义解释。

结构特征文件不能混入测试标签；结构来源、构建体、口袋 QC、feature columns 和 hash 必须另存到运行 manifest。

当前安全结构构建入口为：

```bash
python scripts/build_biomaster_odti_structure_features_v1.py \
  --out outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz \
  --manifest outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1_MANIFEST.json
```

## 6. 当前验证状态

已完成：

- `python -m py_compile biomaster/odti_v2.py scripts/train_biomaster_odti_v2.py scripts/evaluate_biomaster_odti_v2.py scripts/evaluate_biomaster_odti_bindingdb_source_heldout_v1.py`；
- 全量测试 64 个通过；
- S1 V2.1 scaffold-cold 已完成 5 folds × 5 seeds（25/25）正式运行，aggregate micro-AUPRC 0.930656；
- S1 已完成 2,000 次 paired cluster-bootstrap：相对 target empirical-Bayes prior、冻结 ConPLEx external、训练集 Morgan-positive similarity 的 scaffold 轴 AUPRC 95% CI 均高于 0；
- 真实 ChEMBL37 S3 fold-0、CPU、128-row、1-epoch smoke run 通过；
- 安全结构上下文 19 维、86,674 pair 对齐 feature store 构建通过；
- S3 五折、两种子、每角色 64-row smoke 通过；
- BindingDB source-heldout positive-only retrieval 已跑通：628 条严格支持、去除 65 条 ChEMBL37 exact overlap，235 个 pair 对齐；Recall@5/10/20 为 0.4925/0.5731/0.6514；
- within-drug ranking 与 linear dual-query sampler 已实现并完成 smoke，但 S5 2-seed screening 未晋级；
- `biomaster/ofer_dti.py` 已实现 exact first-event competing-risk likelihood、observation hazard、conditional activity、direct-active hazard、HT-weighted loss 与累积 first-event score；pre-event rows 不进入 activity loss；
- `scripts/evaluate_biomaster_ofer_dti_phase_a.py` 已接入 `TARGET_PRIOR`、`STATIC_OBSERVED_ONLY_BCE`、`STATIC_FNML_STYLE`、`DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD`、`TIMESTAMP_SHUFFLED_OFER` 和 `OFER_FULL` 六个冻结 variant；六 variant 小规模矩阵 smoke 全部 PASS；
- `scripts/train_biomaster_ofer_dti_phase_a.py` 已完成 1-epoch / 2,048-risk-row CPU smoke；内部验证只用 source-event 分组 loss，开发窗口标签仅最终评估读取，输出边界检查通过；
- OFER-DTI DEV_2015_2018 全窗口 1-epoch CUDA 吞吐已通过：1,205,098 risk rows，82,487 个 development pairs；初始 drug-macro AUPRC 0.05099、Recall@20 0.09035，仅作为训练链路检查，不是 Phase-A 结果；
- OFER-DTI DEV_2019_2022 全窗口 1-epoch CUDA 吞吐也已通过：2,455,264 risk rows，113,675 个 development pairs；drug-macro AUPRC 0.08681、Recall@20 0.17105，仅作为时间窗兼容性检查；
- OFER_FULL 的 competing-risk + same-year listwise 5-epoch DEV_2015–2018 screen：internal valid loss 0.05157、drug-macro AUPRC 0.04665、Recall@20 0.05169；仍只是 development screen，未通过完整 Phase-A gates；
- smoke run 检查了有限输出、概率范围、validation-only calibration 和 structure fallback invariant。

当前仍未完成或未晋级：

- residue-level ESM2 token 分支的正式升级；当前短预算筛选为负向，安全结构 store 仍是 target-context vector，不是完整 pair-level residue graph；
- 新蛋白预训练 residual（E2）和 pair-specific 3D interaction（E3），两者均未启动；
- source-held-out BindingDB 更广泛来源/真正 entity-cold 验证；当前 BindingDB/GtoPdb/KiRHub 结果仍不提供 scored unseen-target 或 both-unseen pair；
- W1 实验真实结果作为 untouched prospective test；W1 V17 结果模板、盲法键和独立审计已完成，但尚无真实 assay readout；
- OFER-DTI 的完整 Phase-A variant comparison 和机制 gates；现有 CUDA 运行与 5-epoch screen 仍是 development evidence；
- 任何 full SOTA、真实阴性识别、绝对结合概率或已确认生物学结论。

## 7. 晋级规则

本轮八小时迭代的机器可读审计见：
`outputs/old_drug_target_sota_v1/biomaster_strengthening_iteration_audit_v1/BIOMASTER_STRENGTHENING_ITERATION_AUDIT_V1.json`
及对应 Markdown 报告。该审计明确区分完整 formal、partial run、positive-only external 和
prospective 尚未执行的状态；当前 S1 已标记为 `COMPLETE_INTERNAL_S1`，不可再按 partial run 汇报。

V2 不能因为结构更复杂就自动替换 V1。正式晋级至少要求：

1. 保持当前 S1–S5 的数据、标签和切分不变；
2. S2 target-cold 与 S3 double-cold 完成五种子；
3. target-homology 和 scaffold 两个 cluster-bootstrap 区间独立报告；
4. 结构 residual 相对 base-only 和 V1 有预注册增益；
5. 结构缺失回退行为通过全量审计；
6. 外部 source-held-out 和 W1 prospective 结果单独报告。

在这些门槛完成前，V2 的正确表述是：

> 已实现的强化模型候选；不是已验证的算法创新，也不是 full SOTA。

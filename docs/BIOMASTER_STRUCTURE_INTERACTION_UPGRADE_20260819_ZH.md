# BioMaster：19D 结构上下文与分子–蛋白交互升级

更新时间：2026-08-19

## 已完成的第一阶段升级

本轮没有改变 ChEMBL37 标签、S1–S5 切分或缺结构回退语义，完成了两项可归因的架构升级。

### 结构上下文：19D → 47D

新特征文件：

```text
outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2.csv.gz
outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2_MANIFEST.json
```

| 分组 | 维度 | 内容 |
|---|---:|---|
| quality | 10 | 序列匹配、结构等级、口袋概率、局部 pLDDT、实验结构覆盖度 |
| chemistry | 14 | 疏水、芳香、极性、正/负电、氢键 donor/acceptor、硫、疏水性和残基熵 |
| geometry | 12 | log 体积、残基密度、sphericity、主轴比例、Cα 局部半径、接触密度、尺度不变量 |
| consensus | 11 | top-vs-second pocket gap、概率熵、残基 Jaccard、实验结构支持、多 pocket 共识 |

数据审计结果：

- 428 个 target 中 287 个有可解析结构口袋；
- 61,450 / 86,674 个 pair 有结构上下文；
- 301 个结构文件成功解析，0 个解析错误；
- 在有结构的 target 内，47D 没有发现 `|Pearson r| >= 0.95` 的列对；
- 绝对 `x/y/z` 坐标不再直接作为主模型输入，几何量改为局部和归一化不变量；
- 所有特征均为 label-free，不含 GNINA、正负控制选择或 affinity 标签。

构建脚本：

```text
scripts/build_biomaster_odti_structure_features_v2.py
```

### 分子–蛋白交互：全秩 bilinear → 低秩 FiLM + 结构条件交互

新增可选模式：

```text
interaction_mode = low_rank_film
interaction_rank = 48
film_scale = 0.10
```

其核心为：

```text
drug-conditioned target projection
target-conditioned drug projection
low-rank multiplicative interaction
drug–target–structure product / difference features
semantic structure-group attention
```

旧的 `legacy_full` 模式仍保留，旧 checkpoint 语义不变。新模式的模型参数约 3.79M，旧全秩 bilinear 模式约 10.42M，减少约 64%，同时保留显式乘性交互。

新结构分支保持严格不变量：

```text
structure_mask = 0  ⇒  final_logit == base_logit
```

## 配对短屏结果

这是同一 seed（20260819）、同一 split、3 epoch 的 exploratory screen，不是 promotion 证据。

| Protocol | Variant | Δ micro-AUPRC vs E0 | Δ target-macro | Δ drug-macro | Δ Brier |
|---|---|---:|---:|---:|---:|
| S3 | 47D structure only | −0.0151 | −0.0071 | +0.0003 | +0.0022 |
| S3 | low-rank interaction only | −0.0063 | −0.0037 | +0.0395 | +0.0031 |
| S3 | 47D + low-rank interaction | −0.0179 | −0.0201 | +0.0182 | +0.0052 |
| S5 | 47D structure only | +0.0007 | −0.0184 | −0.0075 | +0.0030 |
| S5 | low-rank interaction only | −0.0236 | +0.0020 | +0.0118 | −0.0267 |
| S5 | 47D + low-rank interaction | +0.0137 | +0.0046 | +0.0077 | −0.0364 |

解释：短预算下联合分支在 S5 有正向信号，但 S3 仍为负；因此当前不能替换 champion。S3 的负向结果也说明结构特征或交互分支尚需更长预算、结构可靠性分层和 target/scaffold cluster bootstrap，不能仅靠加大模型解决。

短屏机器可读结果：

```text
outputs/biomaster_odti_structure_interaction_screen_v2/STRUCTURE_INTERACTION_SCREEN_SUMMARY_V1.json
```

## 运行方式

单个新模型 smoke：

```bash
python scripts/train_biomaster_odti_v2.py \
  --protocol S3_STRICT_DOUBLE_COLD --fold 0 --epochs 40 \
  --structure-features outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2.csv.gz \
  --structure-dim 47 --structure-encoder grouped \
  --enhanced-structure-interaction --structure-gate-init-bias -4.0 \
  --interaction-mode low_rank_film --interaction-rank 48 --film-scale 0.10
```

四格归因 screen：

```bash
python scripts/run_biomaster_odti_structure_interaction_screen_v1.py
```

## 下一阶段：真正的局部 pair-specific 图交互

本轮已把 target-level 结构输入和 pooled-vector 交互升级到可审计的第一阶段，但还没有把 ligand atom graph 与 pocket residue graph 的局部 token 交互纳入正式 champion。下一阶段应在 338 个 strict experimental-pocket targets 上实现：

```text
ligand atom graph
+ pocket residue tokens/graph
+ jointly posed distance/orientation features
+ bidirectional cross-attention or geometric message passing
→ masked pair residual
```

在获得 ligand–pocket 同一坐标系的可靠 pose 前，不把分子自身 conformer 与蛋白口袋坐标直接拼成距离特征；否则会产生没有物理意义的伪 pair signal。

## 当前结论

第一阶段升级已实现并通过 80 个测试。默认 champion 暂不改变；新 47D 结构包和低秩交互模式进入候选队列，必须完成 S2/S3/S5 五 seed、target/scaffold cluster bootstrap、source-heldout 和结构缺失回退审计后，才可考虑 promotion。

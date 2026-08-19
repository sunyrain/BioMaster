# BioMaster-ODTI：训练数据与 Label 定义

> 用途：补充汇报初稿中的“训练数据与 Label 定义”页。本文档是独立补充材料，不覆盖原始 PPT。

## 一页 PPT 版（可直接粘贴）

### 训练数据与 Label 定义

**数据来源**

ChEMBL37 human single-protein direct-binding calibration store

| 项目 | 当前冻结规模/规则 |
|---|---:|
| 原始来源 pair 记录 | 509,172 条 activity/pair 记录 |
| 去重并完成标签判定后 | 86,674 个唯一 drug–target pairs |
| Positive | 45,983（binary label = 1） |
| Negative / inactive | 40,691（binary label = 0） |
| 唯一 compound entities | 62,488 |
| 唯一 target entities | 428 |
| target assay families | 6 |
| 每个 target、每个 label 上限 | 150 条；确定性 scaffold-balanced 采样 |

**Label 规则**

```text
先按 drug–target pair 聚合多个定量活性观测：
Ki / Kd / IC50 → pChEMBL；同时保留 assay、文献和 inactive 语义。

mean(pChEMBL) ≥ 6.0，且没有正负冲突       → Positive（1）
mean(pChEMBL) ≤ 5.0，或明确 inactive 语义，且没有冲突 → Negative / inactive（0）
5.0 < mean(pChEMBL) < 6.0                  → Grey zone，不进入主二分类训练
正负定量结果冲突，或 inactive 与强阳性冲突     → Conflicting，剔除
没有观察记录                               → Unknown，不是 Negative
```

**最重要的语义边界**

```text
Unknown ≠ Negative
数据库未记录 ≠ 实验无活性
训练 label = 在冻结 assay/data contract 下观察到的 activity class
模型输出 = 候选排序分数，不等于绝对 binding probability、疗效或临床结论
```

**训练与部署的关系**

```text
训练：86,674 个有标签 calibration pairs
部署：720 个老药 × 384 个靶点 = 276,480 个候选 pair
部署空间中绝大多数 pair 是未知关系；真正的闭环来自 W1 湿实验 readout。
```

---

## 1. 这批数据到底是什么

当前主线不是把所有数据库记录直接塞进模型，而是建立一个经过数据契约约束的校准集（calibration store）。它来自 **ChEMBL37**，只保留：

1. 人源（`Homo sapiens`）靶点；
2. single-protein target；
3. 直接小分子结合/作用语义；
4. 具有可解释的定量活性（主要为 `Ki`、`Kd`、`IC50`）或明确的 weak/inactive 语义；
5. 能够映射到项目使用的 target sequence、compound entity 和模型输入格式。

原始来源有 **509,172 条** activity/pair 记录。经过 target/compound 映射、同一 pair 多条实验记录聚合、冲突审计、结构标准化、去重和确定性抽样后，形成 **86,674 个唯一 drug–target pairs**。因此，509,172 是来源记录量，86,674 才是实际用于当前校准训练的数据量，不能把两者混称。

当前冻结集的精确分区为：

- Positive：45,983；
- Negative / inactive：40,691；
- 总计：86,674；
- 唯一 compound entities：62,488；
- 唯一 target entities：428；
- target assay families：6。

权威准备摘要见：

`outputs/current_production_package_v2/conplex_target_calibration_v5_official/CONPLEX_CALIBRATION_PREPARATION_V5.json`

标签文件见：

`outputs/current_production_package_v2/conplex_target_calibration_v5_official/CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz`

---

## 2. Label 不是“数据库出现/未出现”

每个 pair 的 label 不是由模型自己生成，也不是简单根据“ChEMBL 里有没有这条边”生成。标签来自该 pair 的可追溯实验观测，流程是：

```text
activity records
    ↓
标准化为 pChEMBL / explicit inactive 语义
    ↓
按 target sequence × parent compound 聚合
    ↓
检查均值、灰区和正负冲突
    ↓
得到 positive / negative_or_inactive / grey_or_unresolved / conflicting_exclude
    ↓
仅把合格的 positive 与 negative_or_inactive 送入主 binary 训练
```

`binary_label` 只是最终训练编码：

```text
positive             → 1
negative_or_inactive → 0
```

它不是“真实世界绝对活性”的完美真值，而是“在本项目冻结的数据契约和 assay 选择下，被观测到的 activity class”。

---

## 3. 具体的分类规则

### 3.1 定量活性转换

对于 `Ki`、`Kd`、`IC50` 等满足数据契约的定量记录，统一使用 pChEMBL 语义。直观上，pChEMBL 越大，代表以摩尔浓度计的活性越强；例如 pChEMBL = 6 大致对应微摩尔级边界，pChEMBL = 7 大致对应百纳摩尔级边界。

项目当前冻结阈值为：

```text
positive_pchembl_min = 6.0
negative_pchembl_max = 5.0
grey_zone = (5.0, 6.0)
```

### 3.2 Pair 级聚合

同一 drug–target pair 往往有多个 assay、文献或重复测量。项目先保存：

- `min_pchembl`：所有定量记录的最小值；
- `max_pchembl`：所有定量记录的最大值；
- `mean_pchembl`：所有定量记录的均值；
- `numeric_rows`：定量观测条数；
- `any_explicit_inactive`：是否出现明确的 inactive/no activity 语义；
- assay、文献、标准类型及数据关系字段。

主分类使用 **pair 级均值 + 冲突检查**，不是只看一条最强记录：

| 条件 | calibration label | binary label | 是否进入主训练 |
|---|---|---:|---|
| `mean_pchembl ≥ 6.0`，且无冲突 | `positive` | 1 | 是 |
| `mean_pchembl ≤ 5.0`，或明确 inactive，且无冲突 | `negative_or_inactive` | 0 | 是 |
| `5.0 < mean_pchembl < 6.0` | `grey_or_unresolved` | — | 否 |
| 定量记录同时跨越正负阈值 | `conflicting_exclude` | — | 否 |
| 明确 inactive 与强阳性定量同时出现 | `conflicting_exclude` | — | 否 |
| 没有可用观察记录 | `unknown` | — | 否 |

这里的“无冲突”很重要。例如，同一 pair 既有强阳性数据，又有明确无活性数据，不能强行投票成 1 或 0，而是进入冲突剔除集合。

### 3.3 为什么正类的 `min_pchembl` 可能低于 6

正类是按 pair 聚合后判定的。因此，一个 pair 可能有多次结果：一部分较弱、一部分较强。只要 pair 级均值达到正类阈值且没有冲突，它可以被标为 positive；所以在标签文件中可能看到 positive pair 的 `min_pchembl < 6`，但其 `mean_pchembl ≥ 6`。这不表示阈值失效，而是表示同一 pair 存在实验间差异。

---

## 4. 为什么不能把未知 pair 当负类

在 ChEMBL 或其他公共数据库中，没有出现某个 drug–target pair，可能有多种原因：

1. 尚未做过这个 assay；
2. 做过但结果未公开；
3. 结果在另一数据库或后续版本中；
4. assay 条件、target construct、物种或 isoform 不同；
5. 数据库尚未收录；
6. 确实无活性。

只有第 6 种情况是 negative，而前 5 种都不是。若把未观察 pair 批量标成 0，模型会学到“文献密度、热门靶点和历史测试偏好”，而不是药物–靶点生物学关系。这会造成：

- 训练集负类数量虚高；
- 随机切分指标异常乐观；
- 对冷靶点、新 scaffold 和老药候选空间的召回被系统性扭曲；
- 模型分数被错误解释成结合概率。

因此项目采用明确的 observation contract：

```text
已观察到的定量 weak/inactive → 可以是负类
未观察到的 pair              → unknown
unknown                       → 不进入普通 BCE 的负类项
```

---

## 5. 数据筛选和采样规则

### 5.1 去重粒度

主 pair 键是：

```text
target sequence key × parent compound entity
```

同一靶点的多个 ChEMBL target 记录会先按项目 sequence 和 parent compound 合并；标签冲突不会被静默覆盖，而会被显式标记并排除。

### 5.2 target-specific cap

为防止少数热门靶点支配训练，每个 target、每个 label 最多保留 **150 条**。在最终 cap 前先做最多 3 倍的确定性预筛选，再做 scaffold-balanced selection。

这意味着 86,674 不是数据库的全部观测，而是经过“可用性 + 语义一致性 + 结构多样性 + target 平衡”筛选后的冻结训练集。

### 5.3 scaffold-balanced selection

采样不是随机抽样。对每个 target × label 组：

1. 先按确定性 hash 排序，保证不同运行可复现；
2. 尽可能先保留不同 Bemis–Murcko scaffold 的代表分子；
3. 仍有空余名额时，再按同一确定性顺序补齐；
4. 任何被筛掉的记录都保留在来源审计中，不改变原始 label 语义。

---

## 6. Label 与模型输入/输出的关系

### 输入

对每一个 pair，模型接收：

- drug：标准化后的 active-moiety SMILES，经 Morgan-2048 表征；
- target：项目冻结的人源蛋白序列，经 ProtBERT-1024 表征；
- 可选 pair context：结构口袋/证据/突变等 19-D 对齐特征；
- 可用时的 pooled ESM2-650M 1280-D 辅助表征；
- evidence branch 的可用性 mask，告诉路由器某个模态是否存在。

Feature store 本身是 **label-free representation layer**，不包含由测试集反推的标签信息。当前审计规模为：

- Morgan：62,477 个可用 model SMILES × 2,048；
- ProtBERT：428 个精确 target sequence × 1,024；
- 结构 context：19-D optional pair-aligned feature。

### 主输出

模型输出的是 pair-level activity score/logit，随后可用于：

- 给定老药，对 384 个靶点排序；
- 给定靶点，对 720 个老药排序；
- 计算 Recall@K、NDCG、AUPRC 等检索指标；
- 在校准后提供风险/覆盖率控制。

需要明确：

```text
raw score/logit      ≠ binding probability
calibrated score     ≠ clinical efficacy probability
top-ranked candidate ≠ experimentally confirmed hit
```

---

## 7. 训练集、评估集和部署集不是一回事

| 层级 | 内容 | 是否有真实 binary label | 用途 |
|---|---|---:|---|
| 训练 calibration store | 86,674 个 ChEMBL37 合格 pair | 有 | 拟合模型参数/校准表示 |
| 严格评估协议 S1/S2/S3/S4/S5 | 从有标签 pair 中按 scaffold、target homology、时间或老药实体隔离出的测试部分 | 有 | 检查泛化和老药召回 |
| 完整部署空间 | 720 × 384 = 276,480 个候选 pair | 大多数没有 | 生成候选排序 |
| W1 湿实验集合 | 从部署排序中挑选的候选与 controls | 实验后才有 | 完成 observation → label 闭环 |

当前部署空间有 **807 个已知关系**作为 rediscovery control，但它们只占：

```text
807 / 276,480 ≈ 0.29%
```

因此，S5 或 rediscovery control 的阳性比例不能直接代表完整部署空间的真实阳性率；它们是有标签、可评估的控制子集，而不是整个虚拟空间的 prevalence。

---

## 8. W1 湿实验之后，label 如何更新

W1 不是把“模型预测”回写成训练标签。实验结果进入新的 observation layer 后，按预先定义的 readout contract 产生新标签：

```text
wet-lab measurement
    ↓
记录 assay 条件、target construct、浓度/曲线、重复和 QC
    ↓
判定 confirmed active / confirmed weak-inactive / technically inconclusive
    ↓
作为外部、时间切分或 source-held-out 的新证据
    ↓
重新训练/校准，并保留原始版本和时间戳
```

技术失败、重复间不一致或只得到单点弱信号时，不应直接写成 negative；应标记为 `inconclusive` 或进入 grey/uncertain 层，等待复测或正交 assay。

这样才形成真正的闭环：

```text
公共数据库 label → 模型排序 → W1 实验 observation → 新 label/证据 → 下一轮训练
```

---

## 9. 汇报讲稿（建议 60–90 秒）

> 我们的训练数据来自 ChEMBL37，但不是把数据库里所有记录直接作为真值。首先，我们限定在人源 single-protein direct-binding 场景，并把 Ki、Kd、IC50 等定量观测统一到 pChEMBL 语义；然后按 target sequence 和 parent compound 聚合同一个 drug–target pair 的多条实验记录。当前冻结集有 86,674 个唯一 pair，其中 45,983 个 positive、40,691 个 negative/inactive，覆盖 62,488 个 compound 和 428 个 target。正类要求 pair 级平均 pChEMBL 不低于 6，负类要求平均 pChEMBL 不高于 5，5 到 6 之间是灰区；如果同一个 pair 同时出现强阳性和弱/无活性证据，则作为冲突数据剔除。最关键的是，数据库没有记录的 pair 不会被当作 negative，因为未观察可能只是尚未测试或尚未公开。训练 label 表示的是在特定 assay 和数据契约下观测到的 activity class，不是绝对结合真值。部署时，我们把模型分数用于 720 个老药和 384 个靶点的排序，而最终的新标签要靠 W1 湿实验确认，实验结果再用于下一轮校准和训练。

---

## 10. 当前版本引用文件

- 冻结研究协议：`outputs/old_drug_target_sota_v1/OLD_DRUG_TARGET_SOTA_INNOVATION_PROTOCOL_FROZEN_V1.json`
- 标签准备摘要：`outputs/current_production_package_v2/conplex_target_calibration_v5_official/CONPLEX_CALIBRATION_PREPARATION_V5.json`
- 标签明细：`outputs/current_production_package_v2/conplex_target_calibration_v5_official/CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz`
- 标签生成脚本：`scripts/extract_chembl37_target_calibration_v5.py`
- ConPLex 校准准备脚本：`scripts/prepare_conplex_calibration_v5.py`
- 特征层审计：`outputs/old_drug_target_sota_v1/feature_store_v1/FEATURE_STORE_AUDIT_V1.json`

### 一句话结论

**我们训练的不是“数据库边预测器”，而是一个在明确观察契约下学习 activity class 的药物–靶点排序模型；正负标签来自可追溯实验，未知关系保持 unknown，W1 湿实验负责把部署空间中的未知 pair 转化为下一轮可用证据。**

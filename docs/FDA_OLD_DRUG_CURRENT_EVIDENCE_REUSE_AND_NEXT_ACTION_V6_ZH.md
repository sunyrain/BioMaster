# FDA 老药新靶点：现有证据复用与下一阶段行动方案 V6

## 一、当前决策

现有结果不推倒重来，也不再合成为一个跨方法总分。正式组合改为：ChEMBL 正负集决定模型在每个靶点上是否准入；P1/P2 承担已知化学空间内的精度通道；P3 承担远程新颖性通道；结构、疾病、文献与实验信息分别用于条件质控、后处理和可执行性。

当前 1000 条保持不变：P1 226、P2 300、P3 474。它们不是同一置信度，也不存在有效的跨层统一顺序。

## 二、现有方法的正式用途

| 现有结果 | 继续使用方式 | 禁止解释 |
| --- | --- | --- |
| ChEMBL 37 正负集 | 靶点级模型准入与基准构建 | FDA 候选本身的结合证明 |
| 靶点专属 QSAR | 通过 scaffold/time 审计后的域内 P1 排序 | 远程新骨架发现、跨靶点概率 |
| 配体相似性 | 经验证靶点上的 P2 保守扩展 | de novo 新靶点发现 |
| ConPLEx | P3 快速召回和遗漏保护 | 独立亲和证据 |
| 现有 Boltz 3000/1000 | pose 文件、重复性、输入完整性和条件结构假说 | 已校准 binder probability |
| AlphaFold/P2Rank/PUResNet/OT tractability | 靶点路由、口袋和实验可做性 | 具体 drug-target 结合支持 |
| 文献、Agent、暴露、active species、assay | Pair 深审与实验准备；仅复用完全相同 pair | 生成新的物理证据 |
| Open Targets/疾病图谱 | 结合假说之后的疾病与机制收敛 | 反向证明亲和 |

## 三、Boltz pilot 的重新判定

本地 pilot 使用了正确的 `boltzAffinityProbabilityBinary`。20 条全部使用预测口袋约束，其中 AlphaFold 受体 20/20；输入靶点为 ALOX5, EZH2, PDE4D, PTGS1, SLC22A12, SLC5A1, SLC6A2, SLC6A3, SLC6A4, SOAT1。当前输入没有形成成熟 holo enzyme/kinase 主导的公平 benchmark。

正负样本按骨架多样性抽取而非同 assay、理化性质和化学系列匹配；每靶点仅 1+1。故 ROC-AUC 0.59 的正式解释是“当前 pilot 设计没有提供判别证据”，不是“Boltz-2 已被否定”。现有 Boltz affinity 输出继续冻结为未校准字段，pose 和文件可复用。

## 四、下一阶段：先建立靶点级模型准入

Wave 1 只使用成熟、非膜、ChEMBL 正负充足且有外部配体结构证据的 5 个靶点：MAPK14、MAP2K1、PDE4D、BACE1、ESR2。每靶点 20 个强阳性与 20 个强阴性，优先同 assay/文献匹配，再匹配 MW、cLogP、电荷、TPSA 和相似性。

必须分别评估三个问题：`affinity_probability_binary` 的 binder classification；`affinity_pred_value` 在已知 binder 系列中的连续 affinity ranking；实验 holo 配体的 pose reproduction。三个结果不得再合成单一 composite。

Wave 1 通过后才运行 Wave 2 的 10 个靶点。只有某个靶点在同协议正负基准上通过，既有 P3 Boltz 结果才可在该靶点内重新解释；若协议改变，则只重跑该靶点的 P3 FDA pair，不重跑全部 3000。

## 五、当前 1000 条的使用方式

- P1：先做 exact-pair ChEMBL/PubMed 排重、active species、暴露和 assay 审计；它是精度通道，不是远程创新通道。
- P2：作为保守 side-target/rediscovery 队列和流程阳性参照，不能计入远程新骨架发现率。
- P3：保留 474 条作为创新池；按靶点等待 Boltz/docking 的准入结果，不再因为 Boltz A/B 或高置信结构直接晋级。
- Transporter/ion-channel：建立膜状态和功能 assay 专用协议，不与可溶性 enzyme/kinase 共用同一校准阈值。

## 六、历史结果复用范围

当前 V5 1000 中有 216 条与旧 V7 深审完全同 pair，可直接复用其文献、暴露、active species、assay 和疾病判断；其余 784 条不能按相同药物或相同靶点自动继承 pair 结论。

## 七、已知 Pair 与新颖性纠错

通过 FDA 结构到 ChEMBL 37 上市活性成分映射，并联合 drug mechanism 与精确定量 binding pair，旧标签共重新分流 118 条：明确 MoA 31、定量阳性 19、定量阴性 12、灰区/冲突 16、活性成分映射待解决 40。通过本地 ChEMBL 37 已知 pair 排除后剩余 882 条。

这些 882 条只能称为“本地 ChEMBL 37 未报道”，仍需 exact-pair PubMed 和专利审计后才能称为文献未报道。已知阳性转入 positive-control/rediscovery，已知阴性转入 negative-control 或停止，灰区/冲突与活性成分映射问题进入人工判定。

## 八、交付文件

- `TARGET_MODEL_ADMISSION_MATRIX_V6.csv`：每个靶点的模型准入和下一步。
- `BOLTZ_RECALIBRATION_TARGET_SHORTLIST_V6.csv`：Wave 1/2 靶点与现有 P3 覆盖。
- `CURRENT1000_EVIDENCE_REUSE_ACTION_V6.csv`：完整 1000 条逐条行动表。
- `CURRENT1000_EVIDENCE_REUSE_TEACHER_ZH_V6.csv`：简洁中文表。
- `CURRENT882_LOCAL_CHEMBL_UNREPORTED_ACTION_V6.csv`：通过本地已知 Pair 排除的发现审阅池。
- `CURRENT50_POSITIVE_CONTROL_V6.csv`：已知 MoA 或定量阳性对照。
- `CURRENT12_NEGATIVE_CONTROL_V6.csv`：已知定量阴性对照。
- `CURRENT56_IDENTITY_OR_ACTIVITY_HOLD_V6.csv`：活性成分映射、灰区或冲突待解决。
- `EVIDENCE_REUSE_ACTION_AUDIT_V6.json`：一致性和哈希审计。

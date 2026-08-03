# FDA 老药新靶点物理优先筛选流程 v4

## 1. 正式研究问题

本流程优先回答：在限定的 FDA 治疗性小分子和人源可做 target-engagement 的非 GPCR 靶点范围内，哪些“药物活性母体—新靶点”组合值得进入直接互作实验。

疾病、通路和知识图谱不参与主物理排序。它们只在互作候选形成后用于解释适应症、组织语境和转化价值。

## 2. 冻结空间

- FDA 结构条目：915。
- ChEMBL-MoA：892 个基因，对应 891 条唯一蛋白序列。
- 原始 ConPLEx 空间：915 × 891 = 815,265 对。
- v4 项目药物：750 个。745 个来自已审计直接作用药物集，另有 5 个边界小分子被显式恢复审计。
- v4 项目靶点：463 条唯一序列。原 462 条非 GPCR target-engagement 靶点，加上经 P2Rank/PUResNet 口袋共识恢复的 BCL2L10。
- ID 审计空间：750 个药物记录 × 463 个靶点 = 347,250 对。
- 物理排序空间：去盐结构折叠后 723 个模型配体 × 463 = 334,749 个唯一结构 pair。
- 精确实体边界冻结在 `configs/project_drugs_v4.csv` 与 `configs/project_targets_v4.csv`，配置同时固定两个文件的 SHA-256；构建器同时检查来源动态推导集合与冻结集合完全一致。

项目空间不是“所有可能生物作用”的全集。GPCR、分泌/表面结构蛋白、主要依靠 DNA 损伤、物理吸附、螯合、交联或非特异细胞毒起效的机制不属于第一版直接互作实验主线。

## 3. 统一分子表示

ConPLEx、Boltz、去重和结果复用均使用 `model_ligand_smiles`：RDKit 解析后保留最大结构片段，并用 Uncharger 统一电荷表示。

该处理解决盐型、溶剂化物和对离子造成的模型输入错位，但不等同于体内活性代谢物归一化。前药与代谢物仍需单独药代审计。

## 4. 已知阳性校准

v4 项目范围内有 491 个 ChEMBL/FDA 已知 ID—靶点对；盐型/等价结构按活性母体折叠后为 473 个唯一 active-moiety × target 对。它们只用于校准和流程审计，不进入 discovery 候选。

统一 active-moiety ConPLEx 在 463 靶点空间的校准结果：

- Recall@10：30.14%。
- Recall@50：49.29%。
- Recall@100：59.27%。
- Recall@300：83.10%。

按活性母体折叠后的对应结果为 30.02%、49.47%、59.20%、83.30%。所有 rank 使用并列平均名次；不再用最小名次把大块零分并列全部计作较高召回。

这些数字可能受 ConPLEx 训练知识和 ChEMBL 标签重叠影响，因此只能称为已知阳性校准，不能称为无泄露未来发现准确率。

v4 不使用 Top300 或 ConPLEx 绝对分数作为硬门槛。结构和直接小分子 tractability 层保留 427/491 个已知对；已知对随后从 discovery 中排除。

## 5. Top3000 物理预选

334,749 个唯一结构 pair 经以下硬条件形成预选池（347,250 条 ID 记录保留用于溯源）：

- 靶点有 AlphaFold 结构和 P2Rank/PUResNet 可审计口袋。
- Open Targets 支持直接小分子 tractability。
- 不是已知靶点或同 active-moiety 的已知靶点。
- 不是保守的同家族扩展、kinase-to-kinase、核受体扩展或碳酸酐酶 rediscovery。
- 药物和靶点元数据完整。

随后按连续预分数和硬多样性上限选出 3,000 对：每个活性母体最多 12 对、每个靶点最多 32 对、每个 Murcko 骨架最多 50 对。任何上限都不会静默放宽。

当前 Top3000 覆盖 436 个活性母体和 215 个靶点；active-moiety × target 重复为 0；20 条不在旧 106,561 表中。

## 6. 连续评分

总分 0–100，由不重复的证据源组成：

- ConPLEx 单一复合百分位：25 分。
- Boltz affinity 与双样本条件姿势稳定性：30 分。
- P2Rank/PUResNet 口袋共识等级：15 分。
- Open Targets 小分子 tractability 等级：10 分。
- 药物物化和实验干扰可行性：10 分。
- 靶点实验可行性：5 分。
- 新颖性：5 分。

ConPLEx 的全局百分位、药物内百分位和靶点内百分位先组合为一个分量，不再重复叠加多个阈值奖励。口袋分只使用共识等级，不再同时奖励其派生变量。

## 7. Boltz 精修合同

Top3000 全部从零运行，不复用旧 Boltz 结果。每个输入均满足：

- 统一 `model_ligand_smiles`。
- AlphaFold v6 receptor 模板。
- P2Rank 口袋接触约束，非强制。
- empty MSA、3 次 recycling、50 次结构采样步、2 个结构 diffusion sample。
- 50 次 affinity 采样步、2 个 affinity sample。
- 固定批次种子为 `20260710 + batch_index`，并记录 seed scheme。
- 每个输入同时记录 YAML、模型配体、蛋白序列、模板 PDB、口袋约束和来源行 SHA-256；批次签名绑定有序输入、运行参数和种子。
- 运行计划记录 Boltz/PyTorch/CUDA 版本及两份模型 checkpoint SHA-256，结果 provenance 必须与输入签名逐行一致。

Boltz A/B/C 只用于人工审阅，不作为最终硬门槛。正式 pair 分使用连续 affinity probability；结构/界面置信度不重复进入总分。

双样本姿势稳定性优先按输入 P2Rank 口袋残基的 Cα 做局部对齐，口袋残基不足时才回退全蛋白 Cα 对齐；随后计算配体重原子 RMSD、质心漂移和界面残基 Jaccard，同时保留全蛋白 RMSD 供审计。由于输入提供了口袋接触约束，该结果表示同一条件口袋内的姿势可重复性，不是盲口袋恢复证明。

## 8. 正式输出合同

- `final1000`：1,000 个完成 Boltz 的 pair 假说，允许保留姿势不稳定或低 affinity 的审阅储备，但按连续总分降序。
- `final384`：必须嵌套于 final1000，必须完成 Boltz 和姿势审计，只允许 A/B 条件姿势稳定性，排除 ion-channel 专项难度、已知对、同家族风险和严重化合物风险。
- 两个正式包均按 active-moiety × target 去重，并执行硬 drug/target/scaffold/family 上限。
- 任何未完成结果只能输出 `checkpoint_not_formal`，不能生成正式 384 或 `FORMAL_PRODUCTION_PACKAGE`。
- final1000 中先形成 512 条审阅池；完成 ChEMBL/PubMed 与逐条智能体审阅后，剔除 D/矛盾/未补查失败项并回填到 final384。随后输出 target-level 阳性对照、原靶点反筛、primary readout、浓度门和 go/no-go 的实验矩阵。
- 交付包含主报告 PDF、384 条详细证据卡 PDF、完整/中文 CSV 和多工作表 Excel；文件哈希写入最终 manifest。

384 行是 pair nomination queue，不是可直接上机的四块 96 孔板。不同靶点需要不同蛋白、阳性对照、反筛、浓度窗口和读数；物理板图必须在 assay 分组后另行生成。

## 9. 候选级审计状态与剩余缺口

正式 384 已逐条补充以下审阅信息：

- ChEMBL 非 MoA 活性记录和 PubMed 精确 drug-target 文献。
- 原适应症与候选靶点疾病方向是否真正跨领域。
- 可采购形式、溶解度、非特异聚集、PAINS/Brenk/NIH 警报。
- 具体 target-engagement assay、阳性对照、反筛和 go/no-go 标准。

暴露/Cmax、游离浓度、蛋白结合率和细胞可达性目前为逐条叙述性审阅，尚未形成统一、可计算的定量暴露边界。实验矩阵已经给出 assay 入口，但浓度、重复、蛋白构建体、辅因子和膜环境仍须按靶点形成正式 SOP。上述信息用于候选分层和实验设计，不反向伪装成亲和预测证据。

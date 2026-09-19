# 七模型最新计算结果与实际架构审计

结果快照：**2026-09-19 10:24:29 UTC / 北京时间18:24:29**。本报告对应目前网站使用的七个评分通道，DTIAM已经是九月加强版A。

**结论：架构与学到的表征确实会影响排名，本轮已复现ConPLex大量零分的具体计算机制。但七模型分歧还混合了输入信息、训练数据、监督目标、输出头和适用范围的差异，现有结果不能量化“多少分歧由架构造成”。** 同一Nesso主干的两类输出本身也明显不同，更说明需要把这些因素分开验证。

## 1. 最新计算结果

分母为720药物×384核心靶点＝276,480对。计数为已有数值评分；无分数不作阴性处理。

| 模型 | 已有评分 | 覆盖率 | 快照状态 |
|---|---:|---:|---|
| ReTargetMap | 276,480 | 100.00% | 完成 |
| DTIAM加强版A | 276,480 | 100.00% | 完成 |
| ConPLex | 276,480 | 100.00% | 完成；部分查询存在大量并列 |
| ProbeMatchDTI | 276,480 | 100.00% | 完成 |
| DrugCLIP | 275,040 | 99.48% | 382靶点完成；DIO1、RYR1缺少可用口袋输入 |
| DTBind occurrence | 150,818 | 54.55% | 209靶点各720药完成；继续推理 |
| Nesso binder | 77,255 | 27.94% | 106靶点各719药完成；继续推理 |

Nesso完成的106个靶点均缺quinidine，属于构象输入失败。DTBind按当前作者序列及图文件最多覆盖303靶点，即218,160对；其余71个无作者序列、7个序列不一致、3个缺图。尚未完成的计算与缺失输入是不同问题。

七模型都有分数的配对共27,118对，其中严格完成的共同矩阵为**37靶点×719药物＝26,603对**。早间报告为31靶点，本报告扩大到37靶点。以下排名比较采用这个共同矩阵，没有缩回SPR候选集合。

### 固定靶点，给719药物排序

下表为37个靶点等权平均；ReTargetMap使用该推荐方向对应的反向头。Top10为确定性并列处理后的交集，另保存并列随机打破的期望交集。

| 与ReTargetMap比较 | 平均Spearman | Top10平均重合数 |
|---|---:|---:|
| Nesso binder | 0.250 | 2.57 / 10 |
| DTBind occurrence | 0.197 | 1.14 / 10 |
| ProbeMatchDTI | 0.115 | 0.57 / 10 |
| DrugCLIP | 0.073 | 0.68 / 10 |
| ConPLex | 0.011 | 0.89 / 10 |
| DTIAM加强版A | −0.015 | 0.81 / 10 |

全部模型对中，该方向最高平均相关为Nesso—DTBind的0.293；ProbeMatch—DTBind为0.255。DrugCLIP—ConPLex虽然都采用双塔匹配，相关仍只有0.074。随机独立各选10药的期望交集为100/719＝0.139，不能把高于随机的模型重合解释成实验命中。

固定药物、给共同37靶点排序时，ReTargetMap—DTIAM A相关为0.221，ReTargetMap—Nesso为0.223，ProbeMatch—DTBind为0.375。两个方向不能混成一个数字；这里也尚非完整384靶点的七模型比较。ConPLex有7药在37靶点上恒为0，相关未定义，涉及它的该方向均值只使用712个有效查询。

## 2. 网站实际计算的七种架构

以下依据加载路径、权重配置和实际forward代码核对，区分论文能力与当前启用分支。

| 模型 | 实际输入 | 配对计算方式 | 网站当前评分 |
|---|---|---|---|
| **ReTargetMap** | DrugCLIP分子向量512维＋Morgan2048维＋分子图均值40维＋可用性标记；蛋白ESM2全局向量1280维 | 两侧投影后拼接`d、p、d×p、abs(d−p)`，残差MLP，192维配对表示；两个方向输出 | 方向性logit；不是实测亲和或校准命中率 |
| **DTIAM加强版A** | 冻结BerMol768维＋ESM2池化1280维，拼成2048维 | AutoGluon加权集成：树模型＋两种神经网络；没有逐原子—残基交互层 | Kd/Ki＋明确失活训练的分类分数 |
| **ConPLex** | Morgan2048维＋ProtBERT池化1024维 | 两侧各自单层Linear→ReLU到1024维，再计算余弦相似度 | 相似度；当前`SimpleCoembeddingNoSigmoid`不额外sigmoid |
| **DrugCLIP** | 分子三维构象＋指定蛋白口袋三维原子 | 两个独立UniMol式编码器，归一化投影后内积；当前六折模型分数平均 | 对比学习检索相似度 |
| **ProbeMatchDTI** | 蛋白/SMILES语言模型token、蛋白序列、分子二维原子键图 | 迭代探针模块、跨模态注意力与局部交互；三路logit按0.40/0.27/0.33融合后softmax | 二分类分数；当前输入不含蛋白三维坐标 |
| **DTBind occurrence** | ProtT5残基特征、蛋白结构图与表面特征、分子二维图 | 两侧GNN，双向交叉注意力、门控融合及池化，配对MLP | occurrence二分类分数；没有调用独立affinity分支 |
| **Nesso-1** | ESM2残基特征＋生成的分子三维构象 | 48层Pairformer主干及循环更新，预测距离关系、裁剪口袋，再进入两组affinity模块及分类/回归头 | 当前用binder分类输出；另有连续亲和回归输出 |

**ReTargetMap当前选中的确实是`variant=global`、`local_interactions=false`，约124万可训练参数。** 代码中虽然存在局部序列、位点及几何交互分支，但网站该权重没有启用它们。使用DrugCLIP的分子向量也不等于执行DrugCLIP的分子—口袋联合检索。以上描述的是当前网站通道；历史SPR384还经过图谱、已知关系排除、配额与人工/LLM审查，不能把它的选择过程等同于单个神经分数。

DTIAM当前选中的`WeightedEnsemble_L2`权重为：LightGBMXT 16.67%、RandomForestGini 4.17%、XGBoost 16.67%、LightGBMLarge 20.83%、NeuralNetTorch 16.67%、NeuralNetFastAI 25.00%。因此**树模型合计58.33%，神经网络合计41.67%**。它是验证集选出的单个seed 20260923集成，并非三个seed平均，也不是旧网站DTIAM权重。

双塔、全局融合与逐token交互对信息的保留及组合方式不同。全局池化可能丢失局部选择性，双塔末端匹配限制了配对计算形式，交互模型则可以依照另一侧输入更新表示。这些是需要验证的归纳偏好；全局模型仍可学到配对规律，注意力权重也不等于真实物理接触。

官方架构资料：[ConPLex](https://github.com/samsledje/ConPLex)、[DrugCLIP论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html)、[DTIAM论文](https://doi.org/10.1038/s41467-025-57828-0)、[ProbeMatchDTI代码](https://github.com/developer-hq/ProbeMatchDTI)、[DTBind代码](https://github.com/liqy09/DTBind)、[Nesso代码](https://github.com/recursionpharma/nesso)。公开资料用于核对定义，本地部署结论以上述实际权重与分支为准。

## 3. 本轮实际复现的原因与边界

### 3.1 ConPLex：34.5%共同配对为零，已追到计算机制

用官方`BindingDB_ExperimentalValidModel.pt`权重和现有Morgan/ProtBERT缓存，重新计算全部26,603对，与已存分数最大误差为9.31×10⁻⁷。结果：

- **9,180对（34.51%）精确为0。**
- 原始分子/蛋白特征及投影后的向量均没有全零对象。
- 这9,180对全部满足：两侧经过ReLU后的正值维度没有交集，因此点积及余弦相似度为0。
- pazopanib、quizartinib、sorafenib、fostamatinib、lazertinib、sunitinib、nintedanib在这37个靶点上均为0。它们在完整384靶点上并非全零。

这给出了**当前权重形成的稀疏表示，经过ReLU＋余弦计算后产生大量并列**的直接证据。该现象限制了这些对象之间的排序分辨率；不证明这些配对实际结合，也不说明全部跨模型分歧都由它造成。不能直接删除训练后模型的ReLU并把新分数当成修复结果，必须有匹配的重训对照和标签评估。

### 3.2 Nesso：同一主干，分类头与回归头也明显分歧

逐条读取原始`affinity.json`，确认网站binder分数一致。按[官方输出定义](https://github.com/recursionpharma/nesso/blob/main/docs/prediction.md)，连续值为`log10(IC50/µM)`，越小越强；本次转换为`pIC50=6−value`，使两路都按高分优先比较。

固定靶点、排列719药物：分类与回归输出的平均Spearman为**−0.076**，**28/37个靶点为负相关**。固定药物排列37靶点时，平均相关为0.138。

这不是仅靠不同模型分数单位解释的现象，也不能简单归为“不同主干”。两路共享主干，但分类/回归头参数、监督目标和监督样本适用范围并不相同；这里没有把这些因素控制住。尤其不能用无标签目录上的头间相关，判定哪个头正确或自动换成回归头。官方还标记`entropy_crop_pl=0`为不可靠情形，本矩阵共有23对；排除后平均相关为−0.077，仍有28个靶点负相关，基本结论不变。

作为对照，ReTargetMap自身正反两头的平均相关为：固定靶点0.900，固定药物0.838。在正确使用方向头后，它与其他模型的弱一致性仍然存在。

### 3.3 ProbeMatch：抽查没有发现推理随机性导致排名改变

实际概率注意力代码在eval时仍调用随机采样，生产适配器用配对ID固定seed。本轮从共同药物中固定随机抽32药，在AR、MIF上比较4组推理seed，总计64对×4次：

- 与生产环境相同的CPU线程数与原seed复算，最大分数差为0。
- 更换seed后，六次靶点内比较的Spearman均为1，诊断子集Top10交集均为10。
- 最大分数变化0.00320。

这项小规模结果不支持把当前显著分歧归因于推理随机性；不等于证明完整目录或训练seed都稳定。这里Top10只用于32药诊断，不是网站全目录Top10。最初CPU两线程复算与生产三线程相差至多7.28×10⁻⁶，严格精度检查未通过；对齐线程数后完全复现，另保留最初记录。

### 3.4 截断、输入及训练差异要单列

全384靶点中，74个超过DTIAM当前1022残基窗口，47个超过ProbeMatch当前1200残基窗口。**共同37靶点最长920残基，因此这两项蛋白截断不能解释本次共同矩阵的低一致性**，但在后续长蛋白中必须分层检验。

ProbeMatch分子LM另有100-token限制，完整分子图保留。11种药物的SMILES超过100字符，但字符数不是LM token数，不能据此声称11药结构被截断；当前fixed-word分支在forward中只用于取得batch大小。

DrugCLIP依赖指定口袋，DTBind依赖作者蛋白图，Nesso预测并裁剪口袋；三者并未在完全相同的蛋白局部结构上计算。“都有结构信息”不代表实际看到同一个结合位点。

此外，当前ReTargetMap部署训练记录为383,638行（301,637阳性，截止2025），DTIAM A为337,570训练对（Kd/Ki及明确失活）。这些集合及标签定义并不相同，不能把两者相关差直接解释成MLP与树模型的差。其他作者权重也没有与我们的数据做匹配重训。

## 4. 对研究和下一步工作的判断

目前可以陈述的结果是：**目录排名低一致性扩大到37靶点仍存在；至少一个具体的输出分辨率问题已被机制性复现；任务头的差异在同主干中也能产生显著排名分歧。** 还没有证据证明某类架构一定更接近真实结合，或者投票后必然提高SPR命中率。

最有区分力的后续对照应依次控制：

1. **固定数据、标签、划分、输入表征及验证集选择规则，比较简单余弦/双线性交互、全局MLP、树模型。** 这组能更直接回答配对计算形式的影响。
2. **固定主干和划分，比较分类、排序、亲和回归与组合目标。** 只对有相应实测标签的配对计算相应损失，明确失活不随意赋一个精确Kd；现有A/B多任务结果可复用。
3. **单独研究局部交互和结构输入的增量。** 逐残基/原子模型与池化模型的信息量并不相同，需报告输入增量，并设置容量及计算预算对照，不能把所有提升归为架构。
4. **至少多seed，比较双向排名、TopK命中、明确失活误报和相同标签上的共同错误。** 采用训练关系排除及适当的冷靶点/冷分子/时间划分；统计不确定性按查询或合理簇估计，不能把共享实体的26,603对当成独立样本。

冻结各模型原始输出、报告并列/缺失/输入可靠性，比为了提高一致性而换头、改分数或筛模型更适合作为研究起点。SPR384实验结果可用于评估，但是否构成独立盲测需取决于实际冻结、盲态和训练重叠记录。本轮没有新增训练，没有修改网站评分或湿实验清单。

附加描述性分析将完整719×37分数矩阵拆成药物均值、靶点均值和配对残差。DTIAM原始分数的两项主效应合计约77.8%；这仅描述输出结构，不证明“捷径学习”，残差也不是经过验证的结合机制。保存原始分数与总体百分位两种版本，不作为因果结论。

## 5. 复核材料

- [覆盖与运行快照](../outputs/model_architecture_audit_20260919/latest_results/SUMMARY.json)、[双向21组模型比较](../outputs/model_architecture_audit_20260919/latest_results/AGREEMENT_SUMMARY.csv)、[共同37靶点](../outputs/model_architecture_audit_20260919/latest_results/SHARED_TARGETS.csv)。
- [ConPLex逐层复算检查](../outputs/model_architecture_audit_20260919/CONPLEX_REPLAY_CHECK.json)、[7药零分机制](../outputs/model_architecture_audit_20260919/CONPLEX_ZERO_MECHANISM.csv)。
- [同主干双输出汇总](../outputs/model_architecture_audit_20260919/SAME_BACKBONE_HEAD_SUMMARY.csv)、[Nesso原始双输出及裁剪熵](../outputs/model_architecture_audit_20260919/NESSO_SAME_BACKBONE_HEADS.csv.gz)、[排除零裁剪熵复核](../outputs/model_architecture_audit_20260919/NESSO_HEADS_NONZERO_CROP_ENTROPY.csv)。
- [ProbeMatch重复推理检查](../outputs/model_architecture_audit_20260919/PROBEMATCH_INFERENCE_SEED_CHECK.json)、[全部诊断分数](../outputs/model_architecture_audit_20260919/PROBEMATCH_INFERENCE_SEED_SCORES.csv)。
- [架构诊断元数据](../outputs/model_architecture_audit_20260919/ARCHITECTURE_DIAGNOSTICS.json)、[描述性分数分解](../outputs/model_architecture_audit_20260919/DESCRIPTIVE_SCORE_DECOMPOSITION.csv)、[诊断脚本](../scripts/audit_model_architecture_20260919.py)。

诊断脚本默认读取本次冻结共同矩阵；`--conplex-replay`复算官方投影，`--probe-stability`执行64对重复推理。最新结果快照沿用`scripts/snapshot_catalog_results_20260919.py`逻辑，将模块`OUT`指定到本报告`latest_results`目录，以保留早间报告。重新取快照会改变正在增长的共同集合，论文分析应使用已冻结文件及哈希。

代码定位：ReTargetMap在`outputs/biomaster_best_model_20260906/retargetmap_selected_v1/retargetmap/{molecular_controls,unified_interaction}.py`；ConPLex在`third_party/ConPLex/conplex_dti/model/architectures.py`；DrugCLIP在`third_party/sota_dti_2026/Drug-The-Whole-Genome/unimol/models/drugclip.py`；DTIAM权重见`outputs/biomaster_dtiam_ab_20260912/kdki_inactive__seed_20260923/AUTOGLUON_INFO.json`；ProbeMatch在`.external/ProbeMatchDTI/networks/model.py`；DTBind在`.external/DTBind/script/occurrence/dti_model.py`；Nesso在`.external/nesso/nesso/model/models/nesso1.py`。不同目录不应混用为同一部署模型。

# FDA 老药新靶点：方法组合、校准结果与分层候选包 V5

## 一、结论

本项目不再把所有方法换算成一个 100 分总分。方法按其能回答的科学问题分工：

1. 实体、化学风险和序列一致性用于硬门控。
2. 口袋、结构和 Open Targets tractability 用于靶点可做性与实验路由。
3. 只有经过同靶点阳性–阴性校准、scaffold holdout 和适用域检查的方法，才允许给具体 drug–target pair 排序。
4. ConPLEx 降为远程探索召回；Boltz 降为条件结构/pose 证据；二者均不再解释为结合概率。
5. 疾病图谱、通路、组织表达和文献用于结合假说之后的适应症与机制解释，不参与亲和主排序。

最终形成 1000 条分层组合：P1 226 条、P2 300 条、P3 474 条。三层不是同一置信度，不能混称为 1000 条高置信结合候选。

## 二、为什么旧的综合加权口径不能继续

旧流程共使用 58 类方法，但其中很多输出回答的是同一个问题，或只是靶点级先验。V4 最终分与 Boltz 分量 Spearman 相关系数为 0.917，与 Boltz affinity 原始输出为 0.859；这说明旧总分在实际排序中主要由 Boltz 驱动。

旧 Boltz 校准集只有 96 条阳性、0 条阴性，因此无法估计 specificity、precision 或 FDR。口袋共识、tractability、药物物性和 assay readiness 即使全部通过，也不能证明某个具体药物结合该靶点。

## 三、ChEMBL 37 正负校准基准

- 项目靶点：463 个；canonical UniProt 修正后映射 463/463。
- 严格定量或明确 inactive 的靶点–化合物记录：509,172 条。
- 强活性：331,404；弱活性/不活跃：96,122；灰区：77,596；冲突排除：4,050。
- 靶点数据层级：T1 50+50 为 257；T2 20+20 为 65；正类为主 T3 为 38；稀疏 T4 为 103。

标签口径为人源 SINGLE PROTEIN、binding assay、confidence 9、Ki/Kd/IC50、等号关系；pChEMBL >= 6 为强活性，<= 5 或明确 inactive 为负类，5–6 为灰区。它是项目校准定义，不等价于所有实验语境下的绝对真值。

## 四、模型实测与取舍

| 方法 | 校准结果 | 正式角色 |
| --- | --- | --- |
| ConPLEx | 322 个靶点具备 20+20；无靶点显著优于配体相似性。T2 中位 PR-AUC 0.605，相似性 0.915 | 仅作远程探索召回，不作独立物理证据 |
| 靶点专属 Morgan-QSAR | 402 个靶点评估；时间审计后 79 个 T1、243 个 T2 | T1 且处于适用域时可作 P1 排序 |
| 配体相似性 | 在多数有数据靶点上强于 ConPLEx；但依赖已知配体化学空间 | P2 exploitation，不包装成远程新发现 |
| Boltz-2 | 20 条、10 个同靶点正负对；affinity ROC-AUC 0.59、阳性胜 5/10；综合分 ROC-AUC 0.51、阳性胜 4/10 | 停止扩展至 1000；仅作 pose 可生成性/条件质量 |
| AlphaFold2、P2Rank、PUResNet、holo pocket | 靶点级结构与口袋信息 | 路由、模板和硬门控，不给 pair 加结合分 |
| docking、pose consistency、PoseBusters、ProLIF | 可检查几何和接触合理性 | 质量否决或解释；未做同靶点正负校准前不排名 |
| MD | 可检查给定 pose 的短程保持 | 只用于少量入围 pose 压力测试，不用于全量召回 |
| Open Targets、TxGNN、STRING、通路、表达和组织 | 疾病/机制语境 | 结合之后作 disease-mechanism 收敛 |

P1 靶点在 scaffold holdout 下的中位 PR-AUC 为 0.949，相似性基线为 0.870；中位 ROC-AUC 为 0.935。时间切分中，69 个靶点支持或不劣于相似性，54 个出现矛盾并被降级，279 个因新时期正负样本不足无法评估。

## 五、新的组合流程

### 1. 实体与范围硬门控

FDA active moiety、盐型/前药归一化、RDKit 解析、非治疗性实体排除、项目 463 个直接 target-engagement 靶点、序列与结构 provenance。通过只表示可以计算和实验，不构成结合支持。

### 2. 靶点路由

根据 assay family、实验结构/AlphaFold、口袋共识和 Open Targets tractability 决定该靶点是否进入 enzyme、kinase、transporter、ion-channel 或 nuclear/epigenetic 实验通道。

### 3. 已知化学空间 exploitation

每个靶点单独训练 Morgan-QSAR，以 Murcko scaffold 五折留出评估，并与同折配体相似性比较；再做 2022 年及以前训练、2023–2025 年首次记录测试。只有显著优于相似性、无时间矛盾且 FDA 药物位于已知配体适用域的 pair 进入 P1。相似性有效但 QSAR 无显著增益的进入 P2。

### 4. 远程探索

与靶点已知活性配体最大 Tanimoto < 0.40 或无可用已知配体映射的候选进入 P3。ConPLEx 只负责召回；Boltz 完成、序列匹配、pose A/B 只作为结构可生成门控。P3 没有经校准的结合概率。

### 5. 机制与疾病后处理

候选通过结合证据分层后，再用 Open Targets target–disease、通路网络、组织表达、转录扰动和文献判断作用方向、疾病细分和实验 readout。疾病证据不能反向补强一个物理证据不足的 pair。

## 六、最终 1000 条的构成

- P1：226 条。靶点专属 QSAR 经 scaffold 校准、无时间矛盾、处于适用域。
- P2：300 条。已知配体相似性有效，但新颖性较低。
- P3：474 条。远程骨架、结构可生成、适合探索，但亲和未校准。
- 覆盖 381 个药物、234 个靶点、283 个 Murcko scaffold；与旧 Final1000 重叠 248 条。
- 靶点类型：enzyme 550、kinase 180、nuclear/epigenetic 155、transporter 81、ion channel 34。
- 原 FDA 适应症覆盖 970/1000；该字段只描述药物原用途，不是推荐新病种。
- 已知 FDA pair、同家族扩展风险和 exact known-active structure 均不进入 discovery 1000；另有 family-extension review 和 positive-control/rediscovery 队列单列。

## 七、可以与不可以声称的内容

可以声称：项目建立了同靶点正负校准；识别了哪些模型在什么适用域内有增益；形成了 1000 条证据分层、可审计、非已知 FDA pair 的组合。

不可以声称：P1 是真实结合概率；P2 是全新骨架发现；P3 已被 Boltz 证明结合；1000 条具有相同质量；疾病图谱能证明直接亲和。

## 八、下一步

1. 计算上不继续全量扩展 Boltz 1000 校准，除非更换 pocket protocol 后先通过新的正负配对门槛。
2. P1/P2 优先做 exact-pair 文献与 ChEMBL 排重、active species、暴露和 assay readiness 深审。
3. P3 若要推进，应作为探索/主动学习队列，实验同时配置同靶点阳性和阴性；其首轮价值是产生项目自己的校准数据。
4. 只在 pair 通过上述层级后再选择疾病、作用方向和细胞 readout。

## 九、主要交付文件

- `outputs/current_production_package_v2/calibrated_portfolio_v5/FINAL1000_EVIDENCE_STRATIFIED_V5.csv`
- `outputs/current_production_package_v2/calibrated_portfolio_v5/FINAL1000_EVIDENCE_STRATIFIED_TEACHER_ZH_V5.csv`
- `outputs/current_production_package_v2/calibrated_portfolio_v5/P1_CALIBRATED_TARGET_QSAR_IN_DOMAIN_V5.csv`
- `outputs/current_production_package_v2/calibrated_portfolio_v5/P2_VALIDATED_LIGAND_SIMILARITY_IN_DOMAIN_V5.csv`
- `outputs/current_production_package_v2/calibrated_portfolio_v5/P3_REMOTE_UNCALIBRATED_PHYSICS_EXPLORATION_V5.csv`
- `outputs/current_production_package_v2/calibrated_portfolio_v5/CALIBRATED_PORTFOLIO_INVARIANT_AUDIT_V5.json`

## 附录：58 类方法的正式角色

| 方法 | 证据家族 | V5 决策 | 满足校准条件后可进入 pair 排序 |
| --- | --- | --- | --- |
| FDA 小分子结构库整合 | entity_integrity | retain_hard_gate | 否 |
| ChEMBL-MoA 人源成药锚点构建 | entity_integrity | retain_hard_gate | 否 |
| UniProt 映射与唯一蛋白序列折叠 | entity_integrity | retain_hard_gate | 否 |
| Active moiety、盐型、前药与代谢物归一化 | entity_integrity | retain_hard_gate | 否 |
| RDKit 结构标准化与有效性检查 | entity_integrity | retain_hard_gate | 否 |
| 非 target-engagement 药物实体排除 | entity_integrity | retain_hard_gate | 否 |
| 靶点实验模态与 target-engagement 可做性分层 | target_eligibility | retain_route_or_stratum | 否 |
| 结构–序列–配体输入一致性与 provenance 审计 | engineering_qc | retain_engineering_qc | 否 |
| FDA 已知标签靶点与序列等价映射 | calibration_controls | retain_control_mapping | 否 |
| Ligand-similarity target fishing | target_specific_ligand_baseline | retain_calibrated_in_domain_exploitation | 是 |
| ConPLEx 序列–配体 DTI 预测 | fast_pair_retrieval | retain_retrieval_only_until_calibrated | 否 |
| 药物内、靶点内与全局相对秩校准 | fast_pair_retrieval | retain_retrieval_only_until_calibrated | 否 |
| 本地监督式独立 DTI（ExtraTrees） | retrieval_baseline | baseline_or_validation_only | 否 |
| EviDTI 不确定性 DTI 重排 | experimental_pair_models | defer_not_production | 否 |
| DrugCLIP / Drug-The-Whole-Genome 口袋–配体对比学习 | experimental_pair_models | defer_not_production | 否 |
| 其他现代 DTI 模型工程评估 | model_engineering | retain_engineering_assessment | 否 |
| 已知阳性召回与富集审计 | validation | retain_validation_only | 否 |
| AlphaFold2 / AlphaFold DB 受体结构 | target_structure_prior | retain_target_prior_no_pair_score | 否 |
| 实验 holo 结构与已知配体结构映射 | target_structure_prior | retain_target_prior_no_pair_score | 否 |
| fpocket 几何口袋识别 | target_structure_prior | retain_target_prior_no_pair_score | 否 |
| P2Rank 机器学习口袋预测 | target_structure_prior | retain_target_prior_no_pair_score | 否 |
| PUResNet 三维深度学习口袋分割 | target_structure_prior | retain_target_prior_no_pair_score | 否 |
| 多口袋模型空间共识 | target_structure_prior | retain_target_prior_no_pair_score | 否 |
| DiffDock 扩散式盲对接 | pair_structure_model | conditional_pair_evidence_after_target_calibration | 是 |
| AutoDock Vina 经典 docking 与重打分 | pair_structure_model | conditional_pair_evidence_after_target_calibration | 是 |
| smina 可定制结构重打分 | pair_structure_model | conditional_pair_evidence_after_target_calibration | 是 |
| GNINA 三维卷积神经网络重打分 | pair_structure_model | conditional_pair_evidence_after_target_calibration | 是 |
| PoseBusters 与本地几何 pose 质控 | pose_quality | retain_quality_veto_or_interpretation | 否 |
| ProLIF 蛋白–配体相互作用指纹 | pose_quality | retain_quality_veto_or_interpretation | 否 |
| Boltz-2 蛋白–配体共折叠与 affinity prediction | pair_structure_model | conditional_pair_evidence_after_target_calibration | 是 |
| 重复采样与条件姿势稳定性 | pose_stress_test | retain_conditional_quality_axis | 否 |
| 短程分子动力学（MD）pose-retention 审计 | pose_stress_test | retain_conditional_quality_axis | 否 |
| 基础药物样物性、QED 与规则审计 | chemistry_risk | retain_veto_or_risk_only | 否 |
| PAINS、Brenk 与 NIH 化学干扰警报 | chemistry_risk | retain_veto_or_risk_only | 否 |
| TDC 端点的本地机器学习 ADMET/QSAR | chemistry_risk | retain_veto_or_risk_only | 否 |
| 药物暴露、给药途径与实验浓度可行性 | experimental_readiness | retain_gate_or_readiness | 否 |
| Assay family 与 readout 映射 | experimental_readiness | retain_gate_or_readiness | 否 |
| 阳性对照、阴性对照与 counterscreen 设计 | experimental_readiness | retain_gate_or_readiness | 否 |
| Murcko scaffold、Morgan 指纹与 Butina 聚类 | portfolio_diversity | retain_diversity_constraint | 否 |
| 序列同源、靶点家族与 rediscovery 风险审计 | novelty_leakage | retain_queue_label_or_exclusion | 否 |
| Open Targets 靶点–疾病证据 | disease_context | retain_post_binding_annotation | 否 |
| Open Targets 小分子 tractability 注释 | target_tractability | retain_target_prior_no_pair_score | 否 |
| TxGNN 药物–疾病知识图谱推断 | disease_context | retain_post_binding_annotation | 否 |
| STRING/HuRI 网络近邻与网络医学 | disease_context | retain_post_binding_annotation | 否 |
| Reactome、GO 与 KEGG 通路/过程注释 | disease_context | retain_post_binding_annotation | 否 |
| LINCS/CMap 与 CREEDS 表达签名反转 | disease_context | retain_post_binding_annotation | 否 |
| GTEx 与 Human Protein Atlas 组织表达语境 | disease_context | retain_post_binding_annotation | 否 |
| DepMap 肿瘤依赖性 | disease_context | retain_post_binding_annotation | 否 |
| 知识图谱可解释路径、机制桶与作用方向 | disease_context | retain_post_binding_annotation | 否 |
| ChEMBL exact-pair 定量活性审计 | known_novelty_audit | retain_control_or_queue_label | 否 |
| PubMed/E-utilities 药物–靶点文献检索 | known_novelty_audit | retain_control_or_queue_label | 否 |
| FDA 标签机制与 action type 审计 | known_novelty_audit | retain_control_or_queue_label | 否 |
| 时间分层与上市后证据审计 | known_novelty_audit | retain_control_or_queue_label | 否 |
| 新颖性与已知机制泄漏审计 | known_novelty_audit | retain_control_or_queue_label | 否 |
| 结构化 AI agent 机制与可行性审阅 | structured_review | retain_review_no_new_evidence | 否 |
| 消融、分层随机基线与排名稳定性审计 | validation | retain_validation_only | 否 |
| 计算产物 manifest、哈希与可复现性 inventory | engineering_qc | retain_engineering_qc | 否 |
| 靶点专属 Morgan-QSAR（scaffold/time holdout） | target_specific_ligand_model | retain_calibrated_in_domain_exploitation | 是 |

完整性审计：`passed`；失败项 0；Final1000 SHA256 `414a8ec4ffc3756a743ac6d3ec6bd520bfdbe5e9d8f62c8cae33f2949e6cde00`。

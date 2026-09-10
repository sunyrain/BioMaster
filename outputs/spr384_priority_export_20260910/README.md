# 冻结SPR384优先级导出

主表SPR384_PRIORITY_LIST.csv包含所需四项名称、原编号、优先级、参考化合物和关键条件；详细表附机制、疾病、精确身份、原始端点与来源编号。UTF-8 BOM编码，可用Excel打开。

排序：P1前10且仍可探索100条；P2第11–20且仍可探索123条；P3先解决149条；P4降低投入12条。同级按结合排名升序、可用阳性化学支持优先、跨领域优先、原编号稳定排序。这是透明实验安排顺序，不是校准命中概率。P1也须蛋白体系/对照确认，不表示上机放行。

旧靶点取720分子机制注册表精确InChIKey记录，可能包含复合物、非人蛋白或作用物种限制，空缺明确标注。新靶点名称取ChEMBL37精确ID。新靶点已知药物通过直接机制记录筛选max_phase=4且未标记撤市；这不是全球完整药物清单或当前上市状态证明。精确靶点无结果时不从同家族、复合物或间接通路推补。已知药物列表可能同时含激动剂、拮抗剂及次要靶点机制，不表示作用方向相同或可互换，详细表另列作用类型。

另列冻结设计参考化合物，覆盖112靶点，可能只是研究化合物。它们不等同于新靶点已知获批药物。历史Kd/Ki/IC50端点不保证SPR适用性，也未把混合端点均值换算成Kd。

小分子显示名尽量按完整InChIKey对应GSRS实体，保留冻结原名和变更说明；fenofibric acid按已核查ChEMBL981显示。nicotine树脂制剂问题仍保留P3。未修改冻结清单、配对及对照，也未把新增384证据池混入。

来源：outputs/joint384_comprehensive_20260909/{RECOMMENDED_CANDIDATES_384.csv,REFERENCE_CONTROLS_EXTRA.csv}、outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv、outputs/spr_expanded_review_20260910/BASELINE_REASON_{1,2}.json、downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db（只读）。查询明细见NEW_TARGET_KNOWN_DRUG_EVIDENCE.csv。

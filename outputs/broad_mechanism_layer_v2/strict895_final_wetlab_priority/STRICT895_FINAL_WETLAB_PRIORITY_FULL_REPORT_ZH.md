# strict895 第一轮湿实验推荐包汇报

生成日期：2026-06-27
项目位置：`/root/autodl-tmp/BioMaster`

## 一、这一轮做了什么

本轮从 `strict_top_ready` 的 895 条 drug-target-direction 候选出发，完成了三件事：

1. 补全 Open Targets 细分病种：不再只停留在 oncology、cardiovascular 这类宽方向，而是把候选靶点对应到更具体的疾病名称和 Open Targets 分数。
2. 对 895 条候选逐条做机制/可行性审计：判断药物-靶点-疾病轴是否有可解释机制、是否容易做 target engagement、需要哪些反筛。
3. 生成第一轮湿实验推荐顺序：输出去重后的 drug-target-disease hypothesis，并拆成 Top96、Top192、Top384 三个实验规模。

## 二、Open Targets 细分病种补全结果

- strict895 总行数：895
- 唯一靶点：190
- 有 Ensembl ID 的靶点：190
- 候选疾病名：261
- 可映射到 Open Targets 的疾病名：207
- 有 child Open Targets match 的行：869
- child OT score >= 0.2 的行：670
- child OT score >= 0.5 的行：362

方法说明：对每个候选靶点调用 Open Targets `target.associatedDiseases`，每个靶点最多取前 4000 个疾病关联；再把本地候选疾病名映射到 Open Targets disease ID，判断该靶点是否支持相应细分病种。Open Targets 在这里不是证明药物能结合靶点，而是证明“靶点-疾病”这半条机制链是否有外部疾病证据。

## 三、strict895 审计结果

- 原始 strict895 行数：895
- 去重后 drug-target-disease hypothesis：873
- agent 已审计行：895
- agent 缺失行：0

agent 决策分布：

- review_low: 326
- keep_medium: 267
- deprioritize: 129
- keep_high: 87
- control_only: 86

老药新用类型分布：

- 跨大领域老药新用: 369
- 同领域同靶点家族扩展: 185
- 同领域新靶点/新病种探索: 125
- 跨领域但同靶点家族扩展: 109
- 文献再发现/阳性对照: 50
- 再发现/流程对照: 26
- 已知药理/阳性对照: 9

去重后推荐层级：

- 二级复核: 296
- 第一轮候补: 254
- 暂缓: 153
- 第一轮优先: 85
- 阳性/再发现对照: 85

## 四、第一轮湿实验优先包

Top96 是最建议第一轮先做的规模；它包含全部第一轮优先候选和少量第一轮候补。

- Top96 第一轮优先：85
- Top96 第一轮候补：11
- Top96 唯一药物：39
- Top96 唯一靶点：57
- Top96 细分病种数：25

Top96 拟新用病种大类分布：

- 实体瘤: 47
- 血液肿瘤/血液病: 22
- 心血管: 12
- 皮肤/免疫: 5
- 肾脏: 5
- 免疫/炎症: 4
- 神经/精神: 1

Top96 实验类型分布：

- 酶/表观遗传生化: 40
- 激酶生化/细胞: 34
- 离子通道功能: 11
- 转运体摄取/外排: 8
- 核受体/转录因子 reporter: 3

Top96 老药新用类型分布：

- 跨大领域老药新用: 54
- 同领域同靶点家族扩展: 26
- 跨领域但同靶点家族扩展: 11
- 同领域新靶点/新病种探索: 5

## 五、Top20 示例

| rank | 候选 | 拟新用细分病种 | 拟新用病种大类 | 新用类型 | 分数 | ConPLEx | OT | 实验入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Lubiprostone - TYK2 | 类风湿关节炎 | 免疫/炎症 | 跨大领域老药新用 | 91.1 | 0.459 | 0.754 | 激酶生化/细胞 |
| 2 | Emtricitabine - CDK4 | 乳腺癌 | 实体瘤 | 跨大领域老药新用 | 90.2 | 0.546 | 0.631 | 激酶生化/细胞 |
| 3 | Lubiprostone - DNMT3A | 急性髓系白血病 | 血液肿瘤/血液病 | 跨大领域老药新用 | 90.0 | 0.511 | 0.858 | 酶/表观遗传生化 |
| 4 | Emtricitabine - CDK6 | 乳腺癌 | 实体瘤 | 跨大领域老药新用 | 89.7 | 0.546 | 0.610 | 激酶生化/细胞 |
| 5 | Lubiprostone - JAK1 | 类风湿关节炎 | 免疫/炎症 | 跨大领域老药新用 | 89.4 | 0.445 | 0.710 | 激酶生化/细胞 |
| 6 | Emtricitabine - DNMT3A | 急性髓系白血病 | 血液肿瘤/血液病 | 跨大领域老药新用 | 89.1 | 0.480 | 0.858 | 酶/表观遗传生化 |
| 7 | Doravirine - ESR1 | 乳腺癌 | 实体瘤 | 跨大领域老药新用 | 86.6 | 0.533 | 0.736 | 核受体/转录因子 reporter |
| 8 | Emtricitabine - PTPN11 | 急性髓系白血病 | 血液肿瘤/血液病 | 跨大领域老药新用 | 86.5 | 0.502 | 0.643 | 酶/表观遗传生化 |
| 9 | Infigratinib Phosphate - PDGFRA | 胃肠道间质瘤 | 实体瘤 | 跨领域但同靶点家族扩展 | 86.0 | 0.648 | 0.825 | 酶/表观遗传生化 |
| 10 | Lubiprostone - PRKCD | 系统性红斑狼疮 | 免疫/炎症 | 跨大领域老药新用 | 84.7 | 0.512 | 0.387 | 激酶生化/细胞 |
| 11 | Palbociclib - TYK2 | 银屑病 | 皮肤/免疫 | 跨领域但同靶点家族扩展 | 84.4 | 0.481 | 0.722 | 激酶生化/细胞 |
| 12 | Afatinib Dimaleate - FLT3 | 急性髓系白血病 | 血液肿瘤/血液病 | 同领域同靶点家族扩展 | 84.0 | 0.631 | 0.835 | 激酶生化/细胞 |
| 13 | Rilpivirine Hydrochloride - LYN | 乳腺癌 | 实体瘤 | 跨大领域老药新用 | 83.8 | 0.457 | 0.437 | 激酶生化/细胞 |
| 14 | Tucatinib - FLT3 | 急性髓系白血病 | 血液肿瘤/血液病 | 同领域同靶点家族扩展 | 82.6 | 0.469 | 0.835 | 激酶生化/细胞 |
| 15 | Rilpivirine Hydrochloride - PTK6 | 食管鳞癌 | 实体瘤 | 跨大领域老药新用 | 82.3 | 0.493 | 0.294 | 激酶生化/细胞 |
| 16 | Selpercatinib - CHEK2 | 乳腺癌 | 实体瘤 | 同领域同靶点家族扩展 | 82.0 | 0.546 | 0.709 | 激酶生化/细胞 |
| 17 | Emtricitabine - BMPR1A | 乳腺癌 | 实体瘤 | 跨大领域老药新用 | 81.6 | 0.546 | 0.292 | 酶/表观遗传生化 |
| 18 | Maraviroc - SLC34A2 | 肺鳞癌 | 实体瘤 | 跨大领域老药新用 | 81.3 | 0.510 | 0.370 | 转运体摄取/外排 |
| 19 | Avibactam Sodium - SLC5A2 | 慢性肾病 | 肾脏 | 跨大领域老药新用 | 81.2 | 0.437 | 0.626 | 转运体摄取/外排 |
| 20 | Cabozantinib S-Malate - FLT3 | 急性髓系白血病 | 血液肿瘤/血液病 | 同领域同靶点家族扩展 | 81.0 | 0.679 | 0.835 | 激酶生化/细胞 |

## 六、这些分数应该怎么解释

最终优先级不是单一亲和分数，也不是“已证实结合”。它由五类信息合成：

- agent 审计：候选是否有机制解释、是否适合首轮 target engagement。
- ConPLEx：药物-靶点相互作用预测，只作为计算证据。
- child Open Targets：靶点-细分疾病是否有遗传、临床、文献或药物证据。
- assay lane：该靶点是否容易做激酶、酶活、转运体、离子通道或 reporter 类实验。
- novelty/risk：是否只是已知靶点/同家族再发现，是否存在明显安全性、毒性或 assay 干扰风险。

所以，本推荐包的正确说法是：这些候选是“适合第一轮湿实验验证 target engagement 与机制 readout 的优先假说”，不是已经证明疗效，也不是已经证明新靶点结合。

## 七、推荐实验策略

第一轮不建议直接做疾病疗效结论。建议每个候选先过三道门：

1. 非毒性浓度窗口：先做 viability / cytotoxicity gate。
2. target engagement 或功能 readout：例如 kinase IC50、CETSA/NanoBRET、转运体摄取、离子通道电生理/膜电位、核受体 reporter。
3. 反筛：同家族 panel、原批准靶点 counterscreen、PAINS/聚集/荧光干扰、细胞毒性解释。

Go 标准：剂量依赖、可重复、反筛不能解释主效应，并且在疾病相关细胞模型中能看到方向一致的机制 readout。

## 八、当前边界和下一步

当前已经把宽方向收敛到可落地的细分病种，但仍有两个边界：

- 文献层面：本轮 agent 审计主要基于本地证据和 Open Targets child disease，并没有对 895 条逐条做联网全文级人工文献排重。建议下一步只对 Top96 或 Top192 做逐条 PubMed/Google Scholar 证据核验。
- 结合层面：除已知/对照类候选外，大多数 discovery 候选仍需要湿实验确认 drug-target engagement，不能在汇报中称为已验证新靶点。

## 九、输出文件

- 总表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_final_wetlab_priority_unique_hypotheses.csv`
- 最简扫读总表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_final_wetlab_priority_summary_table_zh.csv`
- 证据卡总表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_final_wetlab_priority_teacher_readable_zh.csv`
- Top96：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top96.csv`
- Top96 最简扫读表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top96_summary_table_zh.csv`
- Top96 证据卡表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top96_teacher_readable_zh.csv`
- Top192：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top192.csv`
- Top192 最简扫读表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top192_summary_table_zh.csv`
- Top192 证据卡表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top192_teacher_readable_zh.csv`
- Top384：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top384.csv`
- Top384 最简扫读表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top384_summary_table_zh.csv`
- Top384 证据卡表：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/strict895_first_wave_top384_teacher_readable_zh.csv`
- 本报告 Markdown：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.md`
- 本报告 PDF：`/root/autodl-tmp/BioMaster/outputs/broad_mechanism_layer_v2/strict895_final_wetlab_priority/STRICT895_FINAL_WETLAB_PRIORITY_FULL_REPORT_ZH.pdf`

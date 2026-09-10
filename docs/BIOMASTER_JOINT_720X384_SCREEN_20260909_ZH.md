# 720×384全空间：身份／既有关系排除及TxGNN×Open Targets联合筛选

日期：2026-09-09。状态：已完成计算和结果复核；没有训练、没有补算神经网络、没有改动原512清单。原720×384结合评分与已完成的全疾病TxGNN推理均直接复用。

## 1. 结论与当前可用结果

从276,480对重新筛选，得到**1,046对主筛选线索（364药、187靶点）**，其中1,009对不在原448候选中，37对与原候选重叠。这说明扩大搜索范围能补充候选供应，尚不能证明新候选的实验成功率更高。

1,046对中74对有已知阳性化合物Tanimoto≥0.4；进一步要求同一个共同疾病有精确单节点映射、遗传或体细胞证据≥0.1，并排除更近的高相似阴性参考后，得到**14对／13药／11靶点优先人工核验**。阈值是探索性审查规则，未校准成命中概率。

其中574对归入生化／核受体等路线，472对属于离子通道或膜转运体路线；后者不能直接并入普通SPR板。14对中12对属于生化路线、2对属于通道／转运体路线。12对生化路线中8对已有严格结构就绪标记，但**全部14对的实验构建均未获实验室或供应商确认**。结构就绪标记也不是试剂或活性合格证明。

**建议先核验14对，保留574对作为生化路线候选池，不直接替换或补满512实验孔。** 完成既有文献、分子形态、作用方向及构建核验后，再按可用靶点和药物多样性重新安排实验，保留已知阳性对照及可比较的低／中排名对照。

## 2. 实际使用的范围和规则

- 结合评分：`BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz`，720个结构实体、384靶点。使用`independent_validation_rank_score`在每药完整384靶点中重新排序，过滤前固定分母和顺序；“Top20”是相对排名，不是测得亲和力或结合概率。
- TxGNN：既有608×17,080原始logit矩阵；身份解释规则允许605药。按每药排除后疾病集合排序，排除旧图已知适应证、off-label、禁忌及EC保守治疗／试验／禁忌掩码。EC排除不等于完整的当前获批适应证登记。通过本轮身份、化学、关系规则后，有500药进入可联合解释空间；缺分数保留缺失，不填零。
- Open Targets：复用26.06全888靶点快照，本轮384靶点全部有证据；采用全部关联，不限旧64靶点、Top3或癌症。疾病映射保留精确单节点与合并节点两种状态。
- 主条件：**结合Top20/384 AND TxGNN药内疾病Top50 AND 同一疾病OT≥0.3**。不把不同疾病的三个高分相加，也不将logit和OT当可直接相加的同尺度概率。
- 身份：模型SMILES的完整InChIKey在720药中全部一致；42个结构未能按完整键接入冻结FDA表，暂扣；64个需要给药／活性物种双重审查的实体暂扣，并保留原身份歧义规则及两项旧设计例外。本轮613药通过身份／范围检查。双物种暂扣是保守实验政策，不表示这些药都无效或不能测。TxGNN名称／归一化映射仍不等于独立立体结构核验。
- 化学：MW120–700、单片段、绝对电荷≤2；MW>550、cLogP>5、TPSA>140、可旋转键>12累计风险点≤1。属性为计算值，不能代替实测溶解性、聚集或非特异结合检查。
- 既有关系：重扫ChEMBL37严格校准对（包括阴性）、综合训练关系、KirHub、冻结FDA/ChEMBL已知机制（含家族／复合物到人源蛋白组分映射）、BindingDB两份2026-07快照及GtoPdb2026.2。按完整药物键及靶点／UniProt匹配；BindingDB多链记录命中任一相关链也保守排除，不据此宣称已经证实该单链直接结合。家族／复合物命中保守排除新颖性，不宣称每个组分均已证明直接结合；外部扫描沿用本地TSV解析范围，不是全网、专利或所有化学等价物的穷尽查新。
- EC：非文本、无映射歧义的物理关系进入已知关系排除；其余来源断言（表达、调控、关联等）以及部分映射歧义另设待审区，不把它们当已证实结合。文本／预测边保留为审计信息，不伪装成新增实验依据。

最终复核补上家族／复合物组分排除，剔除首轮遗漏的7个钠通道家族配对；14对优先项不变。首轮输出保留在`joint_screen_720x384_20260909_first_pass`供差异追溯，以下均为修正后结果。

## 3. 筛选漏斗

| stage | pairs | drugs | targets |
| --- | --- | --- | --- |
| full_space | 276480 | 720 | 384 |
| identity_and_scope | 235392 | 613 | 384 |
| chemistry_policy | 220032 | 573 | 384 |
| known_relation_exclusion | 210567 | 573 | 384 |
| prior_relation_hold_exclusion | 208561 | 573 | 384 |
| TxGNN_interpretable | 182258 | 500 | 384 |
| binding_rank_top20 | 8612 | 493 | 320 |
| TxGNN_top50_OT03_same_disease | 1046 | 364 | 187 |

表中按顺序累计过滤，因此9,465对“既有关系”排除仅指通过前面身份和化学规则后的增量。各来源原始命中有重叠，不能相加：

| reason | pairs |
| --- | --- |
| known_chembl37 | 6643 |
| known_training | 5994 |
| known_kirhub | 7505 |
| known_moa | 412 |
| known_moa_component | 797 |
| known_bindingdb_articles | 178 |
| known_bindingdb_pubchem | 226 |
| known_gtopdb | 586 |
| known_ec_physical | 1778 |
| ec_other_assertion_hold | 3708 |
| ec_ambiguous_relation_hold | 150 |

全276,480对均保留在Parquet，含逐来源标记、身份状态、排除原因、联合条件与原方案重叠信息。未知不自动算新关系，未命中本地快照也不等于首次发现。

## 4. 实验体系与14对人工核验清单

| assay_lane | pairs | targets | chemical_support_ge04 | priority |
| --- | --- | --- | --- | --- |
| ENZYME_BIOCHEMICAL | 335 | 76 | 27 | 3 |
| ION_CHANNEL_FUNCTIONAL | 343 | 25 | 6 | 1 |
| KINASE_BIOCHEMICAL | 128 | 53 | 7 | 5 |
| NUCLEAR_EPIGENETIC_DOMAIN | 111 | 21 | 25 | 4 |
| TRANSPORTER_MEMBRANE_FUNCTIONAL | 129 | 12 | 9 | 1 |

| 药物 | 靶点 | 结合排名/384 | 共同疾病线索 | TxGNN排名 | OT | 阳性近邻Tanimoto |
| --- | --- | --- | --- | --- | --- | --- |
| tecovirimat | AR | 1 | HER2 positive breast carcinoma | 41 | 0.37 | 0.463 |
| pitolisant | SLC6A4 | 1 | insomnia | 1 | 0.467 | 0.429 |
| etonogestrel | AR | 2 | androgen insensitivity syndrome | 19 | 0.845 | 0.449 |
| vismodegib | NTRK1 | 2 | prostate carcinoma | 8 | 0.378 | 0.408 |
| dacomitinib | ERBB3 | 4 | rheumatoid arthritis | 2 | 0.351 | 0.694 |
| olaparib | PIK3CA | 4 | female breast carcinoma | 1 | 0.37 | 0.603 |
| mebendazole | ESR1 | 6 | squamous cell lung carcinoma | 24 | 0.338 | 0.411 |
| raloxifene | KDR | 8 | endometriosis | 7 | 0.562 | 0.471 |
| norethindrone | ESR1 | 8 | ovarian endometriosis | 5 | 0.322 | 0.459 |
| mebendazole | MET | 8 | squamous cell lung carcinoma | 24 | 0.317 | 0.443 |
| bazedoxifene | KDR | 8 | endometriosis | 4 | 0.562 | 0.438 |
| esmolol | KCNH2 | 10 | ventricular tachycardia | 1 | 0.62 | 0.459 |
| lisinopril | F2 | 12 | stroke disorder | 28 | 0.548 | 0.474 |
| nilotinib | IDH1 | 20 | non-small cell lung carcinoma | 20 | 0.304 | 0.505 |

表内共同疾病是复核线索，**没有证明该药通过该靶点治疗该疾病**。遗传关联可能反映致病或保护作用，TxGNN没有建立分子→该靶点→疾病的因果路径，OT总分也不提供药物需要激动还是抑制的结论。即使结合真实，方向、选择性与可达到暴露仍可能不合适。

优先逐对查原始药理／专利及具体作用方向，特别是综合模型或图谱可能已经见过同家族机制的配对。pitolisant–SLC6A4、esmolol–KCNH2进入专门膜／通道体系审查；不是普通可溶性蛋白SPR直接候选。vismodegib–NTRK1、mebendazole–MET、lisinopril–F2、nilotinib–IDH1当前缺严格结构就绪标记，需要另核构建；其余有标记也未确认试剂。lisinopril–F2的最近阳性和阴性相似度相同，不能只展示阳性支持。

14对中只有tecovirimat–AR、lisinopril–F2出现在原448候选中；其余12对是本次新增的优先审查项。化学近邻基于完整ChEMBL37参考库，包含历史训练相关信息，不是时间留出验证，也不是独立实验复现。

## 5. 扩大范围后，证据是否真的更好？

所有行统一经过本轮身份、化学、关系与TxGNN可解释性规则，再按原384排名分组：

| binding_rank_range | eligible_pairs | joint_pairs | joint_fraction |
| --- | --- | --- | --- |
| 1-20 | 8612 | 1046 | 12.15% |
| 21-50 | 13798 | 1598 | 11.58% |
| 51-79 | 13533 | 1598 | 11.81% |
| 80-180 | 47658 | 5808 | 12.19% |
| 181-250 | 33477 | 4241 | 12.67% |
| 251-384 | 65180 | 7421 | 11.39% |

**Top20联合比例12.15%，80–180组12.19%，251–384组11.39%。** 这组描述性结果没有显示高结合排名与共同疾病图谱支持之间存在明显梯度。配对共享药物、靶点及训练信息，不能当作独立样本计算简单显著性；也没有SPR标签可评估真实结合性能。

联合筛选主动挑选满足条件的行，筛选后共同疾病比例更高是规则本身造成的，不能作为预测性能提升的证据。当前变化主要是扩大候选供应、让每对假设可追溯；尚未验证实验命中率提高。

将结合排名放宽到Top50，会得到2,644对、242靶点，但本轮没有必要先牺牲结合排名来凑数。保持Top20，仅把OT门槛提高到0.5仍有494对，但它们并非都适合SPR或满足化学支持。完整敏感性表见`THRESHOLD_SENSITIVITY.csv`。

## 6. 与原512的公平对比

原512=448候选+64对照。实际有406个候选配对落在720×384中；其中364对原本就是384排序路线，42对之前采用扩展367路线，重排后口径变化需要单列。另42候选对所在6靶点不在384中：**AOC3, CA9, FAP, FOLH1, NT5E, RORC**。不能把“12个扩展路线靶点”误写成“12个都不在历史384中”。

本轮1,046对中37对与原448重叠；原在范围内候选有52对进入身份／范围暂扣、4对新增既有关系排除、7对进入来源关系待审。具体逐对原因在`ORIGINAL_CANDIDATES_NEW_HOLDS.csv`。不同筛选范围与身份政策下，不能直接将37与上轮全64靶点的44作性能退步比较。

原64参考对照不按新候选规则筛选，目录外对照也不能因为不在720中就标成无效。逐行对比表显式标注其角色及范围。本次仅产生独立审查输出，原512配对、角色、排序和哈希不变。

## 7. 交付、验证和下一步

- [决策工作簿](../outputs/joint_screen_720x384_20260909/JOINT_SCREEN_DECISION_REVIEW.xlsx)：14对、574生化路线候选、实验体系、全空间排名对照、原512复审、720身份审计。
- [完整筛选工作簿](../outputs/joint_screen_720x384_20260909/JOINT_720X384_REVIEW.xlsx)：1,046主线索、2,644放宽线索及阈值敏感性。
- `FULL_276480_PAIR_AUDIT.parquet`：全空间逐对审计；`JOINT_DISEASE_EVIDENCE.csv.gz`：放宽候选的全部合格共同疾病、原始logit、排除后排名、OT分来源和映射类型。
- `KNOWN_RELATION_SOURCE_LEDGER.csv.gz`、BindingDB/GtoPdb原始匹配记录及`ECKG_ALL_MATCHING_RELATION_ROWS.parquet`：排除依据可追溯。
- 运行脚本：`scripts/screen_joint_720x384_20260909.py`；解释报告脚本：`scripts/report_joint_720x384_20260909.py`。
- 主运行10项检查全部通过；另对14对直接回读原始384评分、TxGNN矩阵和疾病掩码核验，原输出哈希及比较口径检查均通过。见`VALIDATION.json`和`DECISION_VALIDATION.json`。

下一步顺序：先对14对做证据与作用方向核验；按可用构建挑选少量药物／靶点进行试剂与检测体系预实验；再从574对中在固定总预算下做多样性和对照配置。应保留部分原规则候选形成可解释对照，才能在获得真实实验标签后判断联合筛选有没有收益。本次未创建新的512冻结清单、未安排实验，也未修改系统线上候选。

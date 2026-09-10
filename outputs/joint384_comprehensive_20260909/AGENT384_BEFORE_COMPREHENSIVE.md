# 384候选：三组agent逐行快审与独立设计

> 最新深入复核：原384保留；当前建议首选tecovirimat–AR单靶点侦察，pregabalin–MME仅条件候补。mebendazole–ESR1因原始负面端点撤回优先建议。详见[综合结论](BIOMASTER_BEST_CANDIDATE_SYNTHESIS_20260909_ZH.md)。

> 当前口径：按用户要求保留现有384，不筛减，仅统计跨领域属性。99对有跨领域线索（其中3对原用途已核、96对待核）；详见文末第9节。此前跨领域优先级审查不等于已经删除原候选。

日期：2026-09-09。用户口径：**384个候选，参考对照另计**。本轮没有训练、没有新增结合评分、没有更改原512冻结清单。结果仍是待实验确认的审查方案。

## 1. 已完成的结果

三组agent按酶335对、激酶128对、核受体／表观遗传111对分工，覆盖全部574个生化路线候选，每行保留判断、理由、来源范围和实验条件。其余472个通道／膜转运体配对仍在专项体系池，不是“绝对不结合”。

- 明确移出本轮候选：**3对**；包括新颖性不成立或已有具体负面功能结果，不统一解释为绝对不能结合。
- 暂缓：**44对**；包括构建／膜体系、旧文种属／亚型、近阴性及既有药理待核问题，不为凑数回填。
- 通过可进入选择池的规则：**527对**。按证据审查与药物／靶点分布，选出**384对、223药、96靶点**；另143对作为备选。
- 参考对照另列**96对**，合计建议配对数**480**。这是分子－靶点关系数，不包括浓度梯度、重复、空白或溶剂对照的实际孔数。
- 384中有**175对LOW_PRIORITY探索项**；并非384对都经过高置信验证。原14个多证据项经本轮审查，当前入选且仍保留优先标记的是**5对**。

**建议先做幸存优先项的原始证据／构建复核与检测体系预实验，再决定扩展到384。** 达到用户指定候选数量是预算设计完成，不是实验放行或命中率保证。

## 2. “逐一快审”的实际范围

| review_agent | EXCLUDE | HOLD | KEEP_REVIEW | LOW_PRIORITY |
| --- | --- | --- | --- | --- |
| enzyme | 1 | 30 | 101 | 203 |
| kinase | 1 | 5 | 55 | 67 |
| nuclear | 1 | 9 | 80 | 21 |

所有574行都完成了本地评分、化学近邻、图谱解释与实验路线审查；其中49行带在线来源。来源中包含靶点构建层面资料，不能把这个数字称作49对均有直接结合文献。各agent对高优先项、疑似既有药理及特殊构建做了定向原始来源核验；其余明确标记`local_descriptor_review_no_pair_literature`。**没有对574对逐一完成穷尽文献／专利综述。**

`KEEP_REVIEW`表示快审没有发现需暂停的问题，`LOW_PRIORITY`表示支持较薄、机制解释间接或证据相矛盾，可作为明确标识的探索位；这两个标签都不是阳性标签。三个agent的判断也不是三个独立实验重复或投票置信度。

## 3. 有依据的排除及暂缓

- **etonogestrel–AR**：FDA原始NuvaRing药理审查Table6：MCF7细胞质受体结合实验中etonogestrel对AR相对结合4.47±0.67（DHT=100）；已知交叉结合，不应占新关系发现名额；可另列已知关系对照。旧文3-ketodesogestrel为etonogestrel别名。 [来源1](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2001/21-187_NuvaRing_pharmr.pdf) [来源2](https://pubmed.ncbi.nlm.nih.gov/2175153/)
- **mebendazole–MET**：mebendazole–MET: 结合排名8/384；阳性近邻Tanimoto=0.443、阴性近邻=0.294；共同疾病=squamous cell lung carcinoma，OT=0.317、TxGNN排名=24。 RepurposeVS原始研究Table1已报告MET配对初筛数值32、Kd N/D；按既有配对测试排除新颖性候选。N/D是未测定，不是无结合或阴性。 [来源1](https://pmc.ncbi.nlm.nih.gov/articles/PMC5848469/)
- **rivaroxaban–F2**：结合排序3/384；同病TxGNN排序50，OT=0.456；阳性近邻Tanimoto=0.589、阴性近邻=0.703；已有近阴性参考，先复核原始端点/立体化学及浓度后再占实验位；近邻阴性不是该精确实体的阴性证明；原始研究Table 1：人thrombin IC50>20,000 nM；抑制thrombin生成是FXa路径效应。排出本轮高亲和功能性新靶点候选；不是任何浓度/位点绝不结合的证明 [来源1](https://onlinelibrary.wiley.com/doi/10.1111/j.1538-7836.2005.01166.x)

重要区别：rivaroxaban–F2的反证是所报告实验条件下thrombin功能抑制IC50>20 µM，不能推出任何浓度、任何位点均无结合。已发表配对测试即使没有测得Kd，也会使其不再适合占用“未经测试的新关系”名额。

暂缓项覆盖：膜体系SRD5A2/PORCN/DGAT1/VKORC1/NOX4；ERBB3等域／构建与催化解释；人／动物或受体亚型未澄清的既有核受体药理；显著更近的阴性参考；IDH1野生型／突变背景等。具体每对判断以574逐行表为准。成熟可溶胞外催化域可以设计的靶点没有仅因全长含膜就被一概排除。

olaparib–PIK3CA查得的联用论文不能作为本药直绑PIK3CA证据；lisinopril–F2的高浓度凝血读数也不能提升为高亲和结合依据。本轮降低它们的优先级，仍可能列入明确标记的探索位。没有因为“没找到文献”就断言没有结合。

## 4. 原14个优先项现在怎么样

| drug_names | gene_symbol | decision | selected_in384 | priority_after_agent_review |
| --- | --- | --- | --- | --- |
| tecovirimat | AR | KEEP_REVIEW | True | True |
| pitolisant | SLC6A4 | OUTSIDE_CONVENTIONAL_SPR_SCOPE | False | 不适用 |
| etonogestrel | AR | EXCLUDE | False | False |
| vismodegib | NTRK1 | KEEP_REVIEW | True | True |
| dacomitinib | ERBB3 | HOLD | False | False |
| olaparib | PIK3CA | LOW_PRIORITY | True | False |
| mebendazole | ESR1 | KEEP_REVIEW | True | True |
| raloxifene | KDR | KEEP_REVIEW | True | True |
| norethindrone | ESR1 | HOLD | False | False |
| mebendazole | MET | EXCLUDE | False | False |
| bazedoxifene | KDR | KEEP_REVIEW | True | True |
| esmolol | KCNH2 | OUTSIDE_CONVENTIONAL_SPR_SCOPE | False | 不适用 |
| lisinopril | F2 | LOW_PRIORITY | True | False |
| nilotinib | IDH1 | HOLD | False | False |

“仍保留优先”只表示原多证据条件存在且本轮agent未降级／暂缓；不提供疾病治疗方向、靶点占有率或有效暴露保证。tecovirimat–AR等没有直接结合验证的假设，不能因反证较少而被称为已证实。

## 5. 384选择规则

所有入选候选都来自原574池，继续满足：完整384靶点中的结合Top20、排除后TxGNN疾病Top50、同一疾病OT≥0.3、身份与既有关系过滤。不引入新模型、不放宽这些主条件、不从HOLD／EXCLUDE回填。

在固定384候选条件下，优化优先保留未被降级的多证据项，随后参考agent意见、阳性化学近邻、精确疾病／遗传支持及原结合排名；这些是公开的探索性选择权重，**不是校准过的总概率**。低优先级项不能被原先的多证据标记自动抬回高优先级。

约束包括64–96靶点、至少128个药物结构实体，并对每药及同连接骨架的重复配对数、每靶点配对数设上限。若第一档不满足384，只尝试事先列明的分布上限调整，不放松证据暂停规则。实际每药最多4对、每靶点最多8对；所用档位及求解状态见`SOLVER.json`。这是组合设计偏好，不是生物学可行性的保证。

| review_tier | pairs | drugs | targets |
| --- | --- | --- | --- |
| LOW_PRIORITY_EXPLORATORY | 175 | 130 | 66 |
| MULTI_EVIDENCE_SURVIVED_QUICK_REVIEW | 5 | 5 | 4 |
| REVIEW_CANDIDATE | 204 | 136 | 80 |

仅49个入选配对有阳性参考Tanimoto≥0.4；相似度不能独立证明结合，低相似度也没有被当作绝对阴性。选后图谱一致性是选择规则产生的，不能拿来宣布命中率提高。

## 6. 对照与实验就绪边界

每个入选靶点附一个本地ChEMBL37阳性参考提案，排除阳／阴性冲突并核对完整结构键；优先Kd记录，其次Ki，再其他功能端点，兼顾原对照、名称与重复来源。对照按单独预算列出：

| reference_endpoint_category | targets |
| --- | --- |
| FUNCTIONAL_OR_OTHER_ONLY | 2 |
| Kd_RECORD_PRESENT | 78 |
| Ki_NO_Kd_RECORD | 16 |

“有Kd记录”仅表示汇总行包含该端点，混合pChEMBL不能当作纯SPR Kd；Ki／IC50也不等于直接结合对照验证。所有参考仍需回看原始assay、核实采购和当前构建适用性。不同靶点即使选同一化合物也分别计一个对照配对。全部靶点的具体边界、异构体、活化态、辅因子与真实活性均待实验室确认；没有自动把这些状态改成就绪。

该384是发现候选集，未额外占用其中名额加入低排名生物学比较组；若要衡量联合筛选是否提高命中率，应另设计匹配比较组，不能仅凭这384的结果反推筛选相对收益。

## 7. 文件与验证

- [384候选与逐行审查工作簿](../outputs/joint384_design_20260909/JOINT384_AGENT_REVIEW.xlsx)。
- [384候选CSV](../outputs/joint384_design_20260909/CANDIDATES_384.csv)。
- [另计参考对照](../outputs/joint384_design_20260909/REFERENCE_CONTROLS_EXTRA.csv)。
- [574逐行总审查](../outputs/joint384_design_20260909/ALL_574_AGENT_REVIEW.csv)、[排除／待审／备选](../outputs/joint384_design_20260909/NOT_SELECTED_FROM_574.csv)、[靶点配额](../outputs/joint384_design_20260909/TARGET_ROSTER.csv)。
- Agent来源与说明：[酶](../outputs/joint384_agent_review_20260909/ENZYME_REVIEW_NOTES.md)、[核受体](../outputs/joint384_agent_review_20260909/nuclear_review.md)，激酶原始意见见同目录CSV与说明。

唯一配对、574全部去向、恰好384候选、无HOLD／EXCLUDE混入、每靶点对照、候选／对照无重复及原512哈希等检查全部通过。结果见`VALIDATION.json`；输入与产物哈希见`SOURCE_MANIFEST.json`。原512及上轮1046筛选结果均保留，当前384是新的独立审查设计。

## 8. 最新目标：跨疾病大类

已完成[跨适应症复审](BIOMASTER_CROSS_INDICATION_REVIEW_20260909_ZH.md)：癌症→另一癌种不符合本轮优先目标。vismodegib–NTRK1退出；raloxifene/bazedoxifene–KDR因邻近用途和既有临床探索降级，raloxifene内异症方向另有不利RCT。tecovirimat–AR与mebendazole–ESR1保留跨大类假设核验。574对中236个非低优先项有65条跨类线索待核原用途；仅5药完成本轮官方原用途审查，不能把65当确认通过。原384清单不变，本轮不再强求数量。

## 9. 保留现有384，仅统计跨领域数量

按用户最新要求，不删除、不替换原384候选。本节统计单位是药物—靶点配对，只要至少一个达标的精确疾病节点跨原用途大类即计一次；不是疾病条数，也不要求同时满足“非低优先级”条件。

| 当前384分类 | 配对数 |
| --- | ---: |
| 有跨领域线索 | 99（25.8%） |
| 现有分类仅同领域／邻近领域 | 158 |
| 原用途或目标疾病分类不足 | 116 |
| 疑似跨领域，但TxGNN疾病为合并节点 | 11 |
| 合计 | 384 |

99对中57对来自KEEP_REVIEW、42对来自LOW_PRIORITY。先前“65条待核队列”统计的是全部574池中的非低优先级项，并非当前384的全部跨领域项；当前计数纳入42个低优先级探索项。

99对中，只有3对的药物原用途在本轮已核官方来源：tecovirimat–AR、mebendazole–ESR1、mebendazole–FGFR2。第三对属于低优先级，因此不在之前两个优先复核项中。其余96对仍依赖旧图既往用途代理，可能因原用途记录不全而误判，需要继续核验；不能将99对全部写成确认跨获批适应症。跨领域本身也不证明直接结合或疗效。

[逐对分类工作簿](../outputs/joint384_cross_area_counts_20260909/CURRENT384_CROSS_AREA_COUNTS.xlsx)。原384文件哈希保持不变；本轮没有修改候选、角色或优先级字段，仅生成独立计数附表。

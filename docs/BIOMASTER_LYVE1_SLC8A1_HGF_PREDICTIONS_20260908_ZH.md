# LYVE1、SLC8A1、HGF：现有数据与720种老药预测

日期：2026-09-08。对象为人源标准序列。已实际完成3×720个组合的评分。

更新：DTIAM 与官方六折 DrugCLIP 已完成相同候选库的推理，见[三模型候选、交叉排名与口袋敏感性报告](BIOMASTER_LYVE1_SLC8A1_HGF_COMPARATORS_20260908_ZH.md)。

**以下是已交付全局模型对未纳入其下游训练目录的靶点所做的探索性外推。不是正在准备训练的 ContextPocket 新模型结果，不是已确认亲和力，也没有这三个靶点的独立准确率。**

## 1. 我们有没有这些靶点

| 靶点 | 人源 UniProt | 当前模型384目录 | 项目其他数据 | 本次新增 |
|---|---|---|---|---|
| LYVE1 | Q9Y5Y7；322 aa | 不在 | 当前888目录及本地ChEMBL37按该accession查询均未命中 | 官方序列、全链ESM2残基和均值表示、720药物排序 |
| SLC8A1 / NCX1 | P32418；973 aa | 不在 | 原始ChEMBL37有CHEMBL4076：427条活性记录，388种分子、9个assay；未纳入888目录 | 官方序列、全链ESM2表示、720药物排序、活性证据审计 |
| HGF | P14210；728 aa | 不在 | 在888及745登记目录；已有精确序列AlphaFold v6结构和P2Rank口袋；CHEMBL5479有10条活性记录 | 与当前模型协议一致的全链ESM2表示、720药物排序、证据审计 |

“不在模型目录”不等于蛋白不存在或所有数据库均无数据。SLC8A1的数据在原始ChEMBL库中；HGF已有较多序列/结构资产，但并未进入当前384靶点的模型训练和常规评分。

序列使用[LYVE1](https://rest.uniprot.org/uniprotkb/Q9Y5Y7.json)、[SLC8A1](https://rest.uniprot.org/uniprotkb/P32418.json)、[HGF](https://rest.uniprot.org/uniprotkb/P14210.json)的官方人源标准序列；没有将HGF替换为其受体MET。

## 2. 实际模型排序

使用 `retargetmap_selected_v1` 中截至2025年全量拟合的已交付全局模型：DrugCLIP + Morgan + ESM2，CPU FP32推断，固定批量1，按“靶点→老药”评分头排序。候选集合保持原有720个药物实体。模型权重及正式目录未修改。

| 靶点 | 排名 | 药物 | 原始排序logit |
|---|---:|---|---:|
| LYVE1 | 1 | fulvestrant / 氟维司群 | 4.369640 |
| LYVE1 | 2 | dexamethasone / 地塞米松 | 3.877033 |
| LYVE1 | 3 | betamethasone / 倍他米松 | 3.304294 |
| LYVE1 | 4 | berotralstat | 1.996696 |
| LYVE1 | 5 | triamcinolone / 曲安西龙 | 1.087026 |
| SLC8A1 | 1 | lacosamide / 拉考沙胺 | 2.718487 |
| SLC8A1 | 2 | aripiprazole / 阿立哌唑，目录合并其lauroxil前药名称 | -0.406063 |
| SLC8A1 | 3 | tetracaine / 丁卡因 | -0.441311 |
| SLC8A1 | 4 | tazemetostat / 他泽司他 | -1.927792 |
| SLC8A1 | 5 | ranolazine / 雷诺嗪 | -1.964462 |
| HGF | 1 | berotralstat | 5.437458 |
| HGF | 2 | brigatinib / 布格替尼 | -0.102615 |
| HGF | 3 | lacosamide / 拉考沙胺 | -0.123845 |
| HGF | 4 | sunitinib / 舒尼替尼 | -0.638464 |
| HGF | 5 | pyridostigmine / 吡啶斯的明 | -0.719766 |

**logit只能用于本次排序，不能转换成已校准的结合概率，不能换算Kd，也不能因数值较高就称为高亲和力。** 本次证据核验没有确认上述前列药物直接结合对应靶点。

全部结果：[LYVE1 720药物](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/LYVE1_RANKED_720.csv)、[SLC8A1 720药物](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/SLC8A1_RANKED_720.csv)、[HGF 720药物](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/HGF_RANKED_720.csv)。

## 3. 哪些有实验依据，哪些只是模型猜测

### LYVE1

原始研究证明其与透明质酸/透明质酸聚糖（hyaluronan）的结合；这不能转化为本次小分子候选的直接结合证据，见[LYVE1发现论文](https://pubmed.ncbi.nlm.nih.gov/10037799/)。氟维司群、地塞米松、倍他米松暂保留为模型假设，不列为已知LYVE1配体。

### SLC8A1 / NCX1

原始ChEMBL记录主要来自运输/活性抑制读数，没有Kd或Ki记录。即使assay_type标作B，也不能把功能IC50直接写成物理结合亲和力；部分实验体系还来自非人动物，需保留具体描述。

现有720药物中，**amiodarone / 胺碘酮在模型中排第87**。独立研究报告其对豚鼠心肌细胞NCX电流的急性抑制，见[原始电生理研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC1572287/)。这是功能药理依据，不能当成人SLC8A1的Kd。另有研究发现长期给药不改变该动物模型的NCX电流密度，不能混淆急性与慢性作用，见[长期给药研究](https://pubmed.ncbi.nlm.nih.gov/12396024/)。

Bepridil也有NCX电流抑制的原始研究，但不在本次720目录中，因此没有它的本次排名，见[原始研究](https://pubmed.ncbi.nlm.nih.gov/11388640/)。这类有机制依据的参照物可帮助校准SLC8A1实验；拉考沙胺排名第一仍需要直接的SLC8A1功能或结合实验支持。

### HGF

这里发现一个会误导候选判断的数据语义问题：原项目HGF校准表列出4个阳性化合物，对应8条IC50记录。实际测量的是HGF刺激后的MET磷酸化抑制，而原论文主题就是c-Met抑制剂。因此这些记录**不能标作直接HGF结合阳性**，见[原论文](https://pubmed.ncbi.nlm.nih.gov/18763753/)。本次输出已将其单独标记，未改写正在运行任务的数据源。

另有一条真正的直接SPR记录：研究用蒽醌衍生物compound 2a（原始分子CHEMBL2337901）结合固定化HGF，报告Kd=**1.95 μM**；该研究化合物不在720老药目录中，不能给berotralstat的预测背书。数据库assay描述对蛋白物种来源标注不明确，正式复现前还需核对构建体。见[原始SPR研究](https://pubmed.ncbi.nlm.nih.gov/24900685/)。

检索还遇到dihexa高亲和力结合HGF的报道，但对应2014年论文在PubMed中已标注2025年撤稿。本次不把它当作可靠的阳性标准，见[带撤稿标记的论文记录](https://pubmed.ncbi.nlm.nih.gov/25187433/)。

## 4. 结果可信到什么程度

- 特征技术协议已对照：重新生成核心靶点BCL2A1的ESM2全链均值，与交付目录对应特征最大绝对差为 **0**。
- 三个目标都没有进入所用模型的下游训练靶点目录。公共编码器是否见过相关蛋白不作“未见过”保证。原384靶点上的验证AP不能套用到这三个新靶点。
- 未补实验标签，没有针对这些目标微调，也没有根据搜索结果重排模型榜单。
- 437条原始ChEMBL记录经活性母体标准化匹配后，没有对应到720目录药物；这仅是本次本地数据库与匹配协议的结果，不代表文献中没有相关作用。
- 三组排序两两Spearman相关约0.557–0.607，前20重叠2–4种。模型并非对所有新靶点给出完全一样的榜单，但这不能验证其特异性。
- 全链表示最近邻中，HGF靠近PLG/PLAT/F11，LYVE1靠近FLT3/NR3C1/KDR，SLC8A1靠近HMGCR/KCNK10/TRPV4。这是表示空间的相似性，不是序列同一性或结合口袋等价性的证据。全局表示外推产生错误药物优先级的可能性仍需实验排除。

当前建议：把表中候选当作待验证列表；SLC8A1先建立有机制依据的功能对照，再检验模型前列药物；LYVE1、HGF应优先验证直接结合，避免只看表达变化或下游磷酸化变化就认定靶点成立。

复现脚本：[predict_biomaster_lyve1_slc8a1_hgf_20260908.py](../scripts/predict_biomaster_lyve1_slc8a1_hgf_20260908.py)。
证据文件：[活性语义审计](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/ACTIVITY_EVIDENCE_AUDIT.csv)、[推断来源与校验](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/PREDICTION_MANIFEST.json)。

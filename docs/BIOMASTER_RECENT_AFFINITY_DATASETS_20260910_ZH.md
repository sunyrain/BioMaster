# 近期实测亲和论文与数据集核查（2026-09-10）

重点范围2025至2026-09-10，另列直接相关的较早CACHE结果。仅使用原始论文、作者数据仓库与官方数据库页面。新论文不自动代表新实验，新结构不自动代表新亲和标签。

## BindingDB 202609

2026-08-30更新；202609版本；Kd;Ki;IC50等混合；3,241,782 measurements；1,440,011 compounds；11,509 targets（下载页总量）；持续汇集，需相对本地202607逐记录比较；优先核对本地720药/888靶点及384配对的新增与修订；同时去重ChEMBL37；官方TSV及assay描述可下载；本轮未下载全库。

[官方来源](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)

## OpenBind EV-A71/CVA16 2A

2026-05-05首发；2026-08-28 v2；GCI KD；ka/kd；原始sensorgram；699化合物；925晶体结合事件；601化合物附亲和测量；新实验；单病毒蛋白体系；独立结构/亲和基准及传感图拟合参考；不直接增加112个人体靶点覆盖；Zenodo/Fragalysis/GitHub；数据CC0；本轮核对官方数据说明，未下载全包。

[官方来源](https://zenodo.org/records/22142262)

## CACHE Challenge #3

2026-01-21在线发表；SPR KD；HTRF、DLS、溶解性及反筛需分列；按S2/S3/S5/S6原始表口径，未统计全部化合物数；前瞻性新实验；SARS-CoV-2 Nsp3 macrodomain；适合学习筛选到SPR复核流程、审查假阳性和低效力结合；非人体多靶点矩阵；ACS machine-readable XLSX，S3/S6为SPR；本轮核对数据声明，未逐行解析。

[官方来源](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02441)

## HiQBind

2025-02-22数据v3；2025论文；Kd;Ki;IC50;EC50（已读元数据）；18,160 PDB entries；32,275复合物结构；小分子表31,572行；主要整理既有实验标签并修复结构；不是32,275次新实验；结构、化学近邻与已知关系核验；必须对照来源去重，复合物/突变和结构变体不能当独立亲和测量；已下载8.31MB小分子元数据和README；整体数据7.53GB无需本轮下载。

[官方来源](https://figshare.com/articles/dataset/BioLiP2-Opt_Dataset/27430305)

## Inhibitors supercharge kinase turnover through native proteolytic circuits

2025-11-26在线；2026卷期；补充Data6 apparent Kd、EC50、R2、浓度归一化响应；补充Data6已核对：SI-3和TAK285在5CL/2CL混合裂解液的fit/norm表；新实验子集；主文1570×98为丰度扰动而非Kd矩阵；人体激酶及脱靶机制证据；表观Kd与纯蛋白SPR Kd分开保留；已下载并读取Supplementary Data6（约2MB）。

[官方来源](https://www.nature.com/articles/s41586-025-09763-9)

## GatorAffinity-DB

2025预印本及数据发布；BindingDB实测Kd/Ki标签＋预测结构；456,526复合物：69,201 Kd＋387,325 Ki；预测结构增量；亲和标签来自BindingDB，不能再次算独立实验；结构模型训练资源候选；对当前实验已知关系审查优先级低于原始BindingDB；Hugging Face公开；数据卡总量4.03TB；本轮未下载。

[官方来源](https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-DB)

## CACHE Challenge #1

较早挑战结果；2025有后续论文；SPR KD，ITC/19F NMR/DSF和DLS等复核；单一LRRK2 WDR结构域，非激酶域；较早新实验；不冒称2026新矩阵；直接参考SPR质控、非特异结合与弱命中复核；不可转移给LRRK2激酶域；官方all-data XLSX可下载；本轮读取官方结果页。

[官方来源](https://cache-challenge.org/results-cache-challenge-1)

## MEK interactions tune RAF kinase sensitivity to conformation-selective inhibition

2026-05-12；kinobead竞争、细胞target engagement、pERK/pMEK IC50和复合物DC50等；RAF抑制剂机制实验，未确认大规模KD矩阵；新机制数据；多数端点非亲和常数；构建、构象、细胞背景解释；不把细胞IC50或二聚DC50并入KD；公开正文/补充材料，本轮读正文方法与结果。

[官方来源](https://www.nature.com/articles/s41589-026-02212-2)

## 本轮实际检查与下一步

已读HiQBind小分子元数据：31,572条结构记录中Kd 9,653、Ki 9,891、IC50 11,592、EC50 436。来源MOAD 22,891、BindingDB 8,645、BioLiP 36。没有将同一实验对应的多个结构去重，故不可称31,572个独立配对。

已读Nature kinase turnover补充Data6，存在apparent Kd、EC50 Standard Error、R2、potTarget等字段；只包含SI-3/TAK285的所列体系，不把主文1,570抑制剂×98激酶丰度筛查宣传成对应亲和矩阵。非结合曲线或不可拟合记录不应强制补成精确Kd。

本项目scripts/build_retargetmap_spr512_ours_frozen_v2.py第42–43行及scripts/screen_joint_720x384_20260909.py引用BindingDB Articles/PubChem 202607版；官方已提供202609版。需要比较源记录ID、实验端点、条件和修订，而不是把9月全库总量作为增量。新增已知关系先标复核，不自动把现有候选删除。

建议顺序：先对BindingDB7月→9月做增量和ChEMBL37去重；核对人体靶点论文的亲和补充表；用HiQBind提供结构/标签追溯；用OpenBind及CACHE作独立方法和实验质控参照。当前未下载BindingDB全量、未执行模型训练、未改变冻结384。

## 端点与身份要求

Kd优先记录实测方法、构建、物种和浓度范围；Ki保留竞争结合或酶抑制背景；IC50/EC50与Kd分层，绝不直接混成同一亲和标签。保留>、<等测量界限、不可拟合和未检测标记，未测不作为阴性。Kinobead Kd_app不等于纯化蛋白SPR Kd；结构记录中的同源、复合物和突变体须单独核验。OpenBind实验实际使用CVA16 2A替代体系，不能因为标题EV-A71就改写靶标。

最近DTA模型论文常继续使用Davis/KIBA/旧BindingDB；例如2026 AdaMBind公开说明其训练采用这些既有集合，模型发表日期不能用作标签新增日期：[原论文](https://www.nature.com/articles/s41467-026-70554-5)。ProteinTalks的丰度扰动数据也不应混入直接亲和训练集。

[资源目录CSV](../outputs/recent_affinity_sources_20260910/RECENT_AFFINITY_SOURCE_CATALOG.csv)

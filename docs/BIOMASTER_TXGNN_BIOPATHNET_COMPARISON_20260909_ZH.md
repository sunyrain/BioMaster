# TxGNN与BioPathNet：120候选药覆盖及公平比较准备（2026-09-09）

**实际状态：覆盖核对、共同疾病集合、图谱已知关系排除及共同数据划分已生成；两模型在共同划分上的训练和性能比较尚未执行。因此目前没有本项目的模型胜负结论。** 原64靶点512对清单、原TxGNN权重和数据均未覆盖。

本次结果不是复述论文的23.2%提升，也不是对新配对的临床效果或SPR命中率评估。

## 1. 实际覆盖结果

输入：SPR64设计的448个候选配对中，去重后的120种候选药；不将参考对照计入120。

| 项目 | 本次核对结果 |
|---|---:|
| 可映射图谱节点的候选药 | 108/120，90.0% |
| 直接规范化名称匹配 | 105 |
| 完整InChIKey查询PubChem别名后补回 | 3 |
| 尚未映射 | 12 |
| 已映射药物对应的候选配对 | 402/448 |
| 已映射药物对应的高排配对 | 287/320 |
| 共同疾病节点数 | 17,080 |

补回的3种药：norethindrone→Norethisterone（DB00717），torsemide→Torasemide（DB00214），mirdametinib→PD0325901（DB07101）。具体节点、身份键和查询响应均保存。

12种未映射药：gepotidacin、sulopenem、ponesimod、ritlecitinib、olutasidenib、nirmatrelvir、nirogacestat、estetrol、benzgalantamine、acoramidis、avutometinib、fezolinetant。未映射只表示本次规则未能建立可靠映射，不证明该药在所有图谱或全部别名中不存在。

**两模型使用相同图谱输入时，名义覆盖相同。** BioPathNet的官方药物再利用流程本身使用TxGNN/PrimeKG数据，换算法不会自动产生缺失药物的节点。本次108是共享输入图谱覆盖，尚不是BioPathNet特定权重可推理覆盖。[官方数据准备代码](https://github.com/emyyue/BioPathNet/blob/main/reproduce/primekg/01_get_datasplit.py)

图谱节点不含用于本次核验的完整分子结构。105种直接名称匹配仍是名称/节点身份映射，不能宣称105种已完成立体化学一致性验证。没有沿用旧映射器删除盐名及单字母立体标识后任取第一个命中的规则；多个节点命中应停留在歧义状态。

## 2. ponesimod分子身份待核验

SPR清单InChIKey：`LPAUOXUZGSBGDU-STDDISTJSA-N`。

本次PubChem按ponesimod名称返回CID 11363176、InChIKey：`LPAUOXUZGSBGDU-ULCCENQXSA-N`。按清单完整InChIKey查询别名返回404。

两者连接部分相同、后续立体化学标识不同。不能自动认定两者是相同测试物种，也不能仅凭404宣判本地结构错误；需核对GSRS、原始结构及异构状态。该药涉及6个候选对，其中4个高排，已另列`MOLECULAR_IDENTITY_HOLDS.csv`，未改冻结清单。

原始响应见`raw/PONESIMOD_NAME_STRUCTURE_CHECK.json`及按完整InChIKey命名的缓存。其他未映射药的别名查询响应也保留；未用结构连接部分强行匹配异构体。

## 3. 共同疾病与已知关系排除

选用本地TxGNN处理图中的全部17,080个疾病节点，两模型用完全相同的疾病ID集合。部分图谱节点是合并疾病ID，不可直接视作一个Open Targets/EFO编号；本轮尚未做OT疾病交叉映射。

108×17,080=1,844,640个药物—疾病查询组合。对这108种药，本地图谱记录：

- indication：510条；
- off-label use：140条；
- contraindication：1,952条。

探索新适应证时排除已知indication、off-label和contraindication的配对并集，剩余1,842,051个图谱未标注组合。关系种类存在交叉，因此三个计数不可简单相加当唯一排除数。禁忌关系单列，不包装成可再利用候选。

**这是图谱快照内排除，不是截至2026-09-09的全部FDA已批准适应证排除。** 当前官方适应证、论文和专利查新未完成；未标注关系也不是“已证明新颖”或“已证明不适用”。完整掩码见`COMMON_QUERY_AND_EXCLUSION_MASK.csv.gz`。

## 4. 为什么旧权重不能直接打擂台

本地TxGNN有Explorer预训练权重，但其训练排除范围未与本次新测试集对齐。把已知适应证现在从输入表隐藏，不能擦掉权重可能学过的关系。

旧`full_graph_42`目录中，直接药物—疾病关系的valid/test交集为2,131条，train/valid和train/test交集均为0。该目录用于历史全图推理；**不能把它的valid和test当作两份独立评估集**。这不意味着TxGNN论文的独立实验无效，也不直接证明其原权重发生性能退化。

BioPathNet此次核对的仓库提交为`b051a9003a1363b5655cdb7dfdad60609c0e4fe3`。仓库展示的`model_epoch_12.pth`是Git LFS指针；实际训练日志明确指向`config/genefunction/gfun.yaml`、`PC_KEGG_0602`、14种关系，属于基因功能预测任务。不能将它作为PrimeKG药物—疾病权重加载比较。此次查询的GitHub release v0.1没有发布附件；本次检索未取得经过身份审计的药物—疾病checkpoint，不等于断言其他渠道一定没有。

日志和仓库/发布清单已保存在`raw/`。因此本轮没有用错误任务权重、随机初始化权重或全图记忆结果填入性能表。[BioPathNet官方仓库](https://github.com/emyyue/BioPathNet)

## 5. 已生成的共同划分

固定seed=42，用药物ID与疾病ID的组合做SHA256分桶，约80%训练、10%验证、10%测试。同一药物—疾病配对的indication、off-label、contraindication必须留在同一划分；不能仅按关系类型各自随机拆分。先保存无`rev_`的规范边，模型适配器再从训练边生成反向边。

| 关系 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| indication | 7,493 | 974 | 921 |
| off-label use | 2,093 | 237 | 238 |
| contraindication | 24,476 | 3,043 | 3,156 |

另有4,007,433条非药物—疾病背景边供两模型共用。共同背景和训练监督可用于消息传播；验证/测试监督及其反向边不得用于消息传播或预训练。是否使用其他预训练必须单独审计，不能直接加载旧Explorer。

本轮是**配对分组的随机回顾性/传导式划分**，不是疾病零样本，也不是时间外推。它回答已知图谱条件下药物→疾病的关系恢复能力；不复现BioPathNet论文五个疾病领域的零样本结论。

在本次108种已映射药内，测试集有54条indication，涉及34种药、51个疾病。其余已映射药不能凭没有留出阳性就记为模型失败或成功；必须与覆盖率分开。

## 6. 指标合同与运行条件

`evaluate_txgnn_biopathnet_comparison_20260909.py`接收两个真实模型在完整共同查询集合上的分数，检查ID、去重、有限值和训练来源声明，再在有测试阳性的药物上比较：AP、AUROC、阳性平均倒数排名、Recall@20/50，按药物宏平均。AP/AUROC的背景是图谱未标注关系，不是生物学已知阴性。

测试时保留该药的留出indication作为阳性，并过滤其余图谱已知关系；探索新适应证时则排除全部已知关系。两者不能混用掩码，否则会把全部测试阳性删掉。

另提供54×101=5,454行的共用抽样评估表，每个测试阳性配100个图谱未标注背景。它适合排查评分流程；与全疾病排序指标的分母不同，不可横向混报。

评估入口要求checkpoint文件哈希、共享输入文件哈希及无测试边预训练声明。声明仍需人工/训练日志审计，程序不是证明权重无泄漏的神谕。单次随机划分也不足以稳定宣称全面优于另一模型。

当前默认Python缺少BioPathNet的TorchDrug/OGB依赖；现有TxGNN环境是CPU版PyTorch。完成实际性能比较需要在隔离环境建立兼容依赖，再让两模型从共同训练数据训练、按验证集选checkpoint、最后一次性评估测试集。**本轮未启动训练，未生成两模型成绩。**

## 7. 交付与复现

目录：`outputs/txgnn_biopathnet_comparison_20260909/`。

- `COVERAGE_AND_PROTOCOL_REVIEW.xlsx`：覆盖、未映射、查询计数、已知关系、测试阳性、身份待核验。
- `DRUG_COVERAGE_120.csv`：逐药映射规则、来源、状态及SPR配对覆盖。
- `COMMON_DISEASE_UNIVERSE.csv`、`COMMON_QUERY_AND_EXCLUSION_MASK.csv.gz`：统一查询和过滤集合。
- `ALL_DRUG_DISEASE_LABELS_WITH_SPLIT.csv.gz`：原始标签及共同分组归属。
- `shared_split/train1.txt`、`train2.txt`、`valid.txt`、`test.txt`、`entity_types.txt`：规范背景/训练/验证/测试输入及类型。
- `PANEL_TEST_POSITIVES.csv`、`PANEL_VALID_POSITIVES.csv`、`SHARED_TEST_SAMPLED_LABELS.csv`：候选药留出集。
- `SUMMARY.json`、`VALIDATION.json`、`INDEPENDENT_VALIDATION.json`、`MANIFEST.json`：真实状态、核验与来源哈希。

准备：先运行`python scripts/fetch_spr120_graph_mapping_synonyms_20260909.py`取得未直接匹配药的PubChem响应，再运行`python scripts/prepare_txgnn_biopathnet_comparison_20260909.py`。独立核验：`python scripts/verify_txgnn_biopathnet_comparison_20260909.py`。评估入口的`--help`定义实际分数与来源文件要求。

决策：先解决ponesimod身份与缺失节点，保留TxGNN作基线；BioPathNet作为待训练验证的挑战模型。目前不能凭文献优势直接替换现有疾病评分，更不能据此放行SPR候选。

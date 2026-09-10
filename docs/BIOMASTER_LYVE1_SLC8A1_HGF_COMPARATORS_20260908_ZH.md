# LYVE1、SLC8A1、HGF：BioMaster、DTIAM、DrugCLIP 实际预测对比

2026-09-08；已经完成推理。本次是同一候选库上的外部靶点排序，不是带完整标签的准确率测试。

**结论：DTIAM 的跨靶点名单高度相似；DrugCLIP 对 HGF 口袋来源敏感。目前不能把任何一张名单当作已经确认的高亲和力药物，也不能据此宣布某模型胜出。**

解读修正：三模型 Top20 无交集本身不是充分证据。独立均匀随机抽取三份 20/720 名单时，约 98.47% 的情况下也无共同条目。应结合全榜相关性和实验参照判断；见[分歧原因补充诊断](BIOMASTER_QUERY_DISAGREEMENT_ANALYSIS_20260908_ZH.md)。

## 模型与输入

| 模型 | 本次实际版本 | 运行范围 |
|---|---|---|
| BioMaster | 已交付 selected_global_fullfit_2025；沿用上一轮真实推理结果 | 3 × 720；新 ContextPocket 未用于本次 |
| DTIAM | 官方 BerMol768 + ESM2-650M1280；项目 S5 兼容重训的 WeightedEnsemble_L2 | 3 × 720；65,276 训练关系；AutoGluon 1.4.0 |
| DrugCLIP | 官方 Drug-The-Whole-Genome 六折权重，全部实际编码推理 | 3 × 720 主结果；另加 HGF 720 条官方口袋敏感性结果 |

DTIAM 是此前项目部署使用的兼容重训版本，不是论文原始下游预测器。三靶点在其原始 86,674 行数据及靶点序列目录中都没有精确匹配；公共预训练编码器是否见过相关序列/结构未确定。BioMaster 与 DTIAM 的训练数据和目标并不相同，候选库相同不等于公平的性能基准。官方定义见 [DTIAM 代码](https://github.com/CSUBioGroup/DTIAM) 与 [DrugCLIP 代码和权重](https://github.com/THU-ATOM/Drug-The-Whole-Genome)。

DTIAM 使用官方的 FP32 ESM2 末层均值：去除 BOS，保留 EOS；三个查询均短于 1022 aa，无截断。现有特征控制向量最大误差为 0，32 个历史预测控制的最大误差为 2.76×10⁻⁸。

DrugCLIP 使用官方六折余弦均值，再按每个口袋在固定 720 条目上的 median/MAD 标准化，最后取口袋最大值。分数依赖候选库，不是 Kd、IC50 或结合概率。新编码为 FP32、TF32 关闭；完整九个 P2Rank 口袋均小于 256 原子，没有裁剪。

719 个配体复用已有独立 ETKDG 构象。奎尼丁原构象缺失且重试失败，改用 [PubChem 3D](https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/LOUPRKONTZGTKE-LHHVKLHASA-N/SDF?record_type=3d)，完整 InChIKey 和重原子数验证通过。这里的 720 是冻结库中 720 个唯一结构 ID，部分通用名对应不同结构条目；未按名称合并。

| 靶点 | 人源序列 | DrugCLIP 主输入 | 输入限制 |
|---|---|---|---|
| LYVE1 | Q9Y5Y7，322 aa | 官方 DTWG AFv4 同一位点的原始/精修两个表示 | 新 AFv6 的 P2Rank 最高概率仅 0.013，未达预设 0.2；没有强行降低阈值 |
| SLC8A1 | P32418，973 aa | AFv6 + P2Rank 六个口袋 | 官方 DTWG 表中无此 accession；全长单体 AF 输入，未显式建模膜环境或运输构象循环 |
| HGF | P14210，728 aa | AFv6 + P2Rank 三个口袋 | 另有官方 DTWG 六个位点各原始/精修，共十二个表示；HGF 本身不是 MET |

P2Rank 概率 ≥ 0.2 是事先设定的输入阈值。保留全部达标口袋，不根据药物分数挑选口袋。LYVE1 的官方缓存无法仅凭 embedding 文件检查残基范围或 pLDDT；原始/精修表示也不是两份独立实验。AF 结构的序列已与本次 UniProt 查询逐位匹配。

## 各模型的候选与交叉排名

下表的排名均在同一 720 个结构条目中，越小越靠前；不同模型的原始分数不作数值大小比较。所有条目仍是未验证假设。

### LYVE1

DTIAM Top10：

| 药物（库中原名） | BioMaster 排名 | DTIAM 排名 | DrugCLIP 排名 |
| --- | --- | --- | --- |
| tazemetostat | 13 | 1 | 495 |
| pralsetinib | 91 | 2 | 699 |
| sotorasib | 356 | 3 | 156 |
| vanzacaftor | 30 | 4 | 681 |
| entrectinib | 143 | 5 | 712 |
| brigatinib | 24 | 6 | 568 |
| ceritinib | 470 | 7 | 622 |
| palbociclib | 551 | 8 | 102 |
| avapritinib | 484 | 9 | 650 |
| ponatinib | 8 | 10 | 704 |

DrugCLIP Top10：

| 药物（库中原名） | BioMaster 排名 | DTIAM 排名 | DrugCLIP 排名 |
| --- | --- | --- | --- |
| diroximel fumarate | 572 | 690 | 1 |
| tebipenem pivoxil | 705 | 581 | 2 |
| tebipenem pivoxil | 591 | 649 | 3 |
| doripenem | 645 | 471 | 4 |
| indapamide | 36 | 217 | 5 |
| meropenem | 557 | 608 | 6 |
| amoxicillin | 225 | 454 | 7 |
| vorinostat | 304 | 263 | 8 |
| imipenem | 332 | 422 | 9 |
| linezolid | 82 | 288 | 10 |

[完整 LYVE1 三模型对照](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/LYVE1_THREE_MODEL_COMPARISON_720.csv)

### SLC8A1

DTIAM Top10：

| 药物（库中原名） | BioMaster 排名 | DTIAM 排名 | DrugCLIP 排名 |
| --- | --- | --- | --- |
| tazemetostat | 4 | 1 | 551 |
| elexacaftor | 84 | 2 | 517 |
| revumenib | 493 | 3 | 512 |
| brigatinib | 37 | 4 | 626 |
| pralsetinib | 513 | 5 | 634 |
| vanzacaftor | 103 | 6 | 580 |
| entrectinib | 259 | 7 | 699 |
| avatrombopag | 160 | 8 | 255 |
| ceritinib | 453 | 9 | 670 |
| avapritinib | 373 | 10 | 567 |

DrugCLIP Top10：

| 药物（库中原名） | BioMaster 排名 | DTIAM 排名 | DrugCLIP 排名 |
| --- | --- | --- | --- |
| dalfampridine | 367 | 719 | 1 |
| hydroxyurea | 226 | 710 | 2 |
| amifampridine | 342 | 708 | 3 |
| naproxen | 393 | 704 | 4 |
| metformin | 244 | 688 | 5 |
| ibuprofen | 445 | 703 | 6 |
| allopurinol | 157 | 682 | 7 |
| carbidopa | 489 | 534 | 8 |
| vorinostat | 28 | 486 | 9 |
| lisdexamfetamine | 245 | 652 | 10 |

[完整 SLC8A1 三模型对照](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/SLC8A1_THREE_MODEL_COMPARISON_720.csv)

### HGF

DTIAM Top10：

| 药物（库中原名） | BioMaster 排名 | DTIAM 排名 | DrugCLIP 排名 |
| --- | --- | --- | --- |
| brigatinib | 2 | 1 | 593 |
| revumenib | 624 | 2 | 631 |
| tazemetostat | 12 | 3 | 319 |
| sotorasib | 358 | 4 | 668 |
| vanzacaftor | 255 | 5 | 541 |
| ceritinib | 266 | 6 | 701 |
| pralsetinib | 212 | 7 | 640 |
| elexacaftor | 444 | 8 | 634 |
| avapritinib | 479 | 9 | 673 |
| crizotinib | 25 | 10 | 670 |

DrugCLIP Top10：

| 药物（库中原名） | BioMaster 排名 | DTIAM 排名 | DrugCLIP 排名 |
| --- | --- | --- | --- |
| famotidine | 31 | 309 | 1 |
| esmolol | 184 | 674 | 2 |
| metoprolol | 338 | 676 | 3 |
| cilastatin | 19 | 356 | 4 |
| pramipexole | 433 | 664 | 5 |
| imipenem | 76 | 393 | 6 |
| tapinarof | 218 | 582 | 7 |
| sotalol | 426 | 653 | 8 |
| ibuprofen | 594 | 704 | 9 |
| doripenem | 436 | 327 | 10 |

[完整 HGF 三模型对照](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/HGF_THREE_MODEL_COMPARISON_720.csv)

## 一致性与不稳定性

| gene_symbol | model_a | model_b | spearman | top20_overlap |
| --- | --- | --- | --- | --- |
| LYVE1 | ours | dtiam | 0.0522 | 3 |
| LYVE1 | ours | drugclip | -0.1303 | 0 |
| LYVE1 | dtiam | drugclip | -0.2858 | 0 |
| SLC8A1 | ours | dtiam | 0.0341 | 1 |
| SLC8A1 | ours | drugclip | 0.0864 | 0 |
| SLC8A1 | dtiam | drugclip | -0.4614 | 0 |
| HGF | ours | dtiam | -0.0196 | 4 |
| HGF | ours | drugclip | 0.1019 | 1 |
| HGF | dtiam | drugclip | -0.3232 | 0 |

三个靶点的三模型 Top20 交集均为 0；DTIAM 与 DrugCLIP 的 Top20 两两交集也均为 0。模型不一致提示候选选择依赖模型与输入，但不能直接证明其中任何一方“错误”。BioMaster 和 DrugCLIP 还共享 DrugCLIP 预训练成分，因此即使一致，也不应当当作完全独立证据。

DTIAM 跨靶点名单相似性：

| model | target_a | target_b | spearman | top20_overlap |
| --- | --- | --- | --- | --- |
| dtiam | LYVE1 | SLC8A1 | 0.9088 | 12 |
| dtiam | LYVE1 | HGF | 0.8941 | 11 |
| dtiam | SLC8A1 | HGF | 0.8734 | 15 |

0.87–0.91 的跨靶点相关性提示该兼容模型可能较多依赖药物自身特征，对这些外部靶点的区分有限；这是针对本次查询的诊断线索，不是论文模型整体质量结论。其约 0.9 的分类输出在这些查询上没有概率校准证据。

HGF 的 P2Rank 与官方 DTWG DrugCLIP 排名相关性为 **0.3367**，Top20 仅重合 **1** 个条目。
P2Rank 路线前三名为 famotidine、esmolol、metoprolol；官方 DTWG 路线前三名为 valproate、ibuprofen、belinostat。
我们之前排首位的 berotralstat：BioMaster 第 1、DTIAM 第 20、DrugCLIP P2Rank 第 477、DrugCLIP DTWG 第 31。现阶段不能把它称为稳定的多模型高亲和力候选。

SLC8A1 的 amiodarone：BioMaster 第 87、DTIAM 第 172、DrugCLIP 第 429。已有的是 [豚鼠 NCX 电流急性抑制证据](https://pmc.ncbi.nlm.nih.gov/articles/PMC1572287/)，不是人源 SLC8A1 的直接结合 Kd；因此只能作为功能文献参照，不能据单个排名计算模型准确率。

六折 Top20 次数已保存到 CSV，用于观察各折名单稳定性；六折来自同一模型家族，不能把次数解释为实验命中概率。

## 如何使用这些结果

当前没有这 2,160 对关系的完整直接结合标签，因此不报告 AP、Recall、命中率或 nM 亲和力。上一轮本地 ChEMBL 核验也没有与该 720 库精确匹配的直接证据；这不代表文献中完全没有关系。

LYVE1 应先确认构建体、糖基化状态和可测试结合位点；SLC8A1 应把膜蛋白功能实验与直接结合实验分开；HGF 应以 HGF 本体结合证据筛选，不能混入 MET 激酶抑制结果。三者都需要先解决具体实验机制和输入结构匹配，再决定实验优先级。

本次名单适合形成待核验的候选池。没有依据直接将三模型分数相加，或因模型给出高分就扩大实验采购。HGF 的口袋来源敏感性尤其需要先处理。

## 可复现产物

- [全部 2,160 行三模型比较](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/ALL_2160_THREE_MODEL_COMPARISONS.csv)
- [各模型 Top20 与交叉排名](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/ALL_MODELS_TOP20_WITH_CROSS_RANKS.csv)
- [HGF 口袋来源敏感性](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/HGF_POCKET_SOURCE_SENSITIVITY.csv)
- [验证清单](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/VERIFICATION.json)
- [机器可读摘要](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/COMPARISON_SUMMARY.json)
- [推理脚本](../scripts/predict_biomaster_target_comparators_20260908.py)；各执行阶段保留源码快照和哈希。

验证结果：35 项全部通过，包含 720 条目身份和顺序、三组完整排名、DTIAM 历史控制、DrugCLIP 六折权重哈希、独立打分公式重算、已交付模型全部文件哈希。训练、冻结实验候选及已交付模型均未修改。

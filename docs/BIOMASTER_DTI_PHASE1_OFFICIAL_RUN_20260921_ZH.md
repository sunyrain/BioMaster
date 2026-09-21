# 第一阶段官方模型全矩阵运行记录

2026-09-21。当前执行：清理可再生成的冗余文件，恢复官方模型的原生输入，在固定 **720分子×745靶点＝536,400配对** 上计算，并输出双向排序。范围来自原ChEMBL37人类888个机制注释单蛋白靶点，按既有GPCR规则排除143个，保留745个。不是全人类蛋白组，也不由SPR384候选定义。

有效协议：[范围v4](../configs/dti_ranking_scope_20260921/PROTOCOL_v4.json)、[本轮官方模型执行补充](../configs/dti_ranking_scope_20260921/OFFICIAL_EXECUTION_v1.json)。本轮新增训练0；不使用Davis评测，不接入湿实验标签，不修改网站生产分数或SPR交付。

## 第一阶段具体交付

1. **可追溯的评分矩阵**：每个模型固定官方任务权重、输出头、方向、输入构造和运行精度，记录文件哈希及原生推理复核。缺失标出原因。
2. **双向筛选表**：每个靶点在720药中排序；每个药在745靶点中排序。保留平均并列名次、Top20边界所有同分者，以及已评分/请求候选数。
3. **一致性结果**：在相同候选身份上计算逐查询Spearman、Kendall tau-b、Top5/10/20重合和随机参照，分别报告共同覆盖与两两覆盖。查询至少20个共同候选。
4. **差异的分层描述**：先给靶点家族分层；全量完成后补充家族内/跨家族、截断与口袋来源、候选池和批次敏感性。需要显著性结论时，再做同源/化学相关性下的不确定性分析。

这阶段回答“模型推荐是否一致、差异在哪里”。训练数据不同是现成系统差异的一部分；这些结果不能独立证明谁更准确或差异由架构造成。正确性和受控A训练留到第二阶段。

## 模型范围与结果状态

“全部”指收集到可用官方任务权重的12个模型家族；不把每一个epoch或数据集版本都当作独立模型。GraphBAN、CheMLT-F、ADME-DTI按输出前确定的发布包运行，未在本矩阵上挑最好的检查点。DTIAM官方下游任务预测器仍未确认；已有本地DTIAM A保留为另列参照，不混入这次官方权重集合。ReTargetMap不参加。

**状态会持续变化，请以[实时覆盖表](../outputs/dti_official_720x745_20260921/MODEL_COVERAGE.csv)和[实时状态](../outputs/dti_official_720x745_20260921/LIVE_STATUS.json)为准。** 每个模型目录包含`STATUS.json`、`CONTRACT.json`、数值复核文件和实际生成的`SCORES.parquet`；输入准备中的模型可能尚未生成这些文件。

| 模型 | 本轮使用的输出 | 已完成情况或执行方式 |
|---|---|---|
| ConPLex | 官方BindingDB实验权重的余弦分数 | 已完成536,400对；745全覆盖；复用精确ProtBert输入，重新统一计算 |
| DrugCLIP | 官方六折口袋–分子余弦均值 | 已完成486,720对，676靶点；69靶点缺可用原生口袋输入 |
| BALM | 官方PEFT模型的亲和分数 | 已完成446,400对，620靶点；125条序列不符合作者1024 token过滤规则 |
| CheMLT-F | 官方Scaffold多任务检查点的KIBA头 | 已完成536,400对；按作者双512 token蛋白分段，不称Kd |
| SCOPE | 官方Total五检查点概率均值 | 已完成536,400对；将原生序列编码扩展到固定745目录，作者GUI词表覆盖另列 |
| ProbeMatchDTI | 官方All_Model的output6正类概率 | 正在补算；先复核旧分数，再补齐原生ProtBert-BFD特征 |
| DTBind | 官方occurrence概率 | 正在补算；官方图输入取得425个，须继续满足精确序列匹配；不是复合物亲和回归头 |
| MAMMAL | 官方BindingDB Kd模型的pKd | 正在全量推理；严格加载任务权重和作者tokenizer/任务类 |
| Nesso-1 | 官方binder概率；原始回归头另存 | 排在MAMMAL等前序GPU任务及显存重试之后；续算历史覆盖，原始结果压缩校验后保留 |
| GraphBAN | BindingDB发布包zinc21的epoch31–50均值 | 已通过原生前向与分离执行的数值复核，正在计算20检查点集成 |
| ADME-DTI | BindingDB分类的16子模型＋Combined头 | 已完成530,440对：712药×745靶点；8药在原生SEC描述符计算中报错，保留身份及失败原因 |
| EviDTI | DrugBank证据输出的正类比例 | 恢复并复核MG-BERT、ProtT5和3D图后已开始推理；744靶点输入就绪，1个超长蛋白显存失败待独占空间后重试 |

DrugCLIP保留历史实验口袋，并为新增靶点使用精确序列匹配的AlphaFold/P2Rank口袋。两种输入来源已记录；它们不构成统一结构条件下的架构对照。Nesso和DTBind的历史分数只在身份、序列和模型一致的条件下复用；不把旧384范围改写成745完整覆盖。

新增ADME-DTI后已完成6个家族，最新六模型快照见[LATEST_RANK_ANALYSIS.json](../outputs/dti_official_720x745_20260921/LATEST_RANK_ANALYSIS.json)。下面保留首次五模型结果及其原始分母。

## 已得到的首版结果

五个已完成模型：ConPLex、DrugCLIP、BALM、SCOPE、CheMLT-F。共同集合为 **720药×562靶点**。这不是最终12模型交集，也没有把未完成的Nesso计算当作“不支持”。

在这一个共同集合上：

| 模型对 | 靶点选药：平均Spearman | 药物选靶点：平均Spearman |
|---|---:|---:|
| ConPLex–DrugCLIP | 0.0479 | 0.0457 |
| BALM–SCOPE | 0.2907 | 0.0179 |
| BALM–CheMLT-F | 0.1611 | 0.0150 |
| SCOPE–CheMLT-F | 0.1433 | 0.0289 |

这是逐查询的描述性平均值，没有用原始分数尺度算相关性。两个方向呈现不同关系；“有相关”也不等于Top10高度重叠。例如BALM–SCOPE的靶点选药平均Top10交集约0.185，随机参照为100/720≈0.139。尚未进行依赖结构下的显著性检验，不能据此宣称优劣。

首版固定结果：[汇总](../outputs/dti_official_720x745_20260921/rank_snapshots/2baeb17a9d936481/RANK_AGREEMENT_SUMMARY.csv)、[逐查询结果](../outputs/dti_official_720x745_20260921/rank_snapshots/2baeb17a9d936481/QUERY_AGREEMENT.parquet)、[靶点选药Top20](../outputs/dti_official_720x745_20260921/rank_snapshots/2baeb17a9d936481/TARGET_TO_DRUG_TOP20.csv.gz)、[药物选靶点Top20](../outputs/dti_official_720x745_20260921/rank_snapshots/2baeb17a9d936481/DRUG_TO_TARGET_TOP20.csv.gz)。后续完成的模型生成新快照，不覆盖这个五模型结果；最新入口见[LATEST_RANK_ANALYSIS.json](../outputs/dti_official_720x745_20260921/LATEST_RANK_ANALYSIS.json)。

## 清理与算力安排

已清理约 **15.61 GiB逻辑文件体积**，均有审计记录：

- 失败DTIAM历史尝试的可重建`X.pkl`/`X_val.pkl`缓存约8.53 GiB；成功模型和正式特征未删。
- 旧BindingDB临时下载约0.55 GiB；规范化数据与来源记录保留。
- Boltz重复`mols.tar`约1.73 GiB；45,227文件逐项SHA256确认与解包副本相同后移除。
- DTBind下载分块缓存约4.81 GiB；425个提取图文件逐项校验后移除分块。

审计：[第一批](../outputs/dti_official_720x745_20260921/CLEANUP_AUDIT.json)、[DTBind分块](../outputs/dti_official_720x745_20260921/CLEANUP_DTBIND_RANGES.json)。逻辑删除体积不等于净可用空间增加：本轮补齐原生蛋白编码器、隔离环境和新特征会重新占用空间。没有删除官方权重包、训练母库、历史评分或湿实验交付。

GPU特征编码按锁排队，减少大蛋白模型同时占显存；Nesso安排在MAMMAL、GraphBAN、EviDTI完成以及超长蛋白显存重试之后。前者是较快的序列任务推理，后者原生逐对计算明显更贵。按历史运行量级，Nesso补齐可需十余天，不能承诺当天12模型都完成。ProbeMatchDTI和DTBind在CPU持续补算；ETA必须以实际速度和可用输入计算，不把缺原生图的配对算成可推理工作量。

## 复现与运行入口

总目录：[本轮输出](../outputs/dti_official_720x745_20260921/)。`SCORE_MATRIX.parquet`是当前汇总，不同模型仍可能在续算；正式引用采用带哈希的固定快照。

- 实时整理：`python scripts/report_dti_official_matrix_20260921.py --analyze --watch`。
- 各模型：`scripts/run_dti_official_*_20260921.py`；其对应`*_JOB.json`保留解释器、进程和命令。不要在同一模型上同时启动多个写入进程。
- 特征：`scripts/prepare_dti_*_20260921.py`；不使用任务标签，不把其他模型的嵌入代替原生输入。
- 下载：`scripts/fetch_dti_native_sources_20260921.py`、`fetch_dti_native_encoders_20260921.py`、`fetch_dti_fair_encoders_20260921.py`和`fetch_dti_evidti_encoder_20260921.py`。

ADME-DTI的描述符保留作者对当前药物池做全局min/max归一化的流程；本轮池固定，描述符计算失败的分子单列。以后改变候选池时须区分“沿用冻结特征”与“重做这种归一化”，后者可能改变原始配对分数，不能直接归因于模型架构。

官方FAIR下载若没有发布上游SHA，报告文件长度、来源URL与本地SHA，不能称完成上游密码学身份校验。Hugging Face提供LFS哈希的权重则与其核对。特征或环境恢复失败会保留日志，不生成替代分数。

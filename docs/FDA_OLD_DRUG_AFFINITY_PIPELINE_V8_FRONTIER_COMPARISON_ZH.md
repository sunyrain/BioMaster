# FDA 老药新靶点：物理优先筛选与国际前沿对照

> 状态：V8 方法与计算进展报告（2026-08-03）

本文档说明项目的唯一正式目标、数据来源、筛选漏斗、各模型职责、国际前沿对照、当前计算进展和最终实验包设计。所有计算分数均为优先级证据，不等同于实验亲和力或命中概率。

## 一、唯一目标与边界

第一阶段目标是：在 FDA 已上市小分子与人源直接小分子作用靶点之间，尽可能找到真实直接结合，同时主动降低以下低价值再发现：

- 数据库已知 drug-target pair；
- 同一 active moiety、盐型或前药造成的重复；
- 与该靶点已知配体高度相似或共享已知骨架；
- 已知靶点的近同源、同亚家族机械扩展；
- 依靠疾病图谱、热门靶点或文献数量而非物理证据晋级的 pair。

第二阶段才处理老药新用：对实验确认的结合 hit 判断作用类型、人体暴露、组织语境、疾病机制和新适应症。第一阶段不预设疾病，也不把疾病分数加入亲和排序。

## 二、正式设计空间从哪里来

### 2.1 药物侧

原始 FDA 表包含约 915 个结构条目。经过盐型、前药、active moiety、重复结构和非目标实体审计后，得到 723 个可计算的唯一配体结构。

排除或分流包括：气体、诊断/显像剂、非治疗性分子、polymer、resin、sequestrant、超大分子、核酸/抗体样实体，以及主要依靠螯合、还原、交联、物理吸附或广谱非特异细胞毒起效的实体。

### 2.2 靶点侧

以 ChEMBL 37 的人源药物机制/MoA single-protein target 为锚点来源，经 canonical UniProt、唯一蛋白序列、直接小分子 target-engagement 和实验可做性审计后，当前正式主空间为 463 个靶点。

Open Targets 的约 2,397 个目标属于更宽的潜在 tractability/疾病证据宇宙，不等同于已有明确小分子 MoA 的锚点，因此用于注释和未来扩展，不直接混入第一版亲和主空间。

### 2.3 Pair 空间

`723 × 463 = 334,749` 条正式物理 pair。这里的每一条记录是一个可计算假说，不是预测阳性。

## 三、正式漏斗

| 层级 | 数量 | 核心规则 | 这一层回答什么 |
|---|---:|---|---|
| 原始物理空间 | 334,749 | 723 个 active moiety × 463 个靶点 | 哪些组合在项目边界内 |
| 新颖性与责任审计 | 310,423 | 去已知 pair、精确活性、同骨架、高相似和扩展同源风险 | 是否仍可称为远程发现 |
| 严格结构远程空间 | 202,654 | 受体、口袋、序列与结构边界完整 | 是否值得进入结构检索 |
| 低成本结构召回 | 30,000 | DrugCLIP、ConPLEx 仅分配计算预算 | 哪些 pair 先做昂贵计算 |
| 靶点校准准入 | 16,995 | 仅保留 GNINA 至少一个主通道能区分同靶点阳性/阴性的靶点 | 这个模型在该靶点是否有判别资格 |
| Pair 级 GNINA 精修 | 运行中 | 同受体、同 box、同参数；每个 pair 5 个 pose | 候选在该靶点内处于什么位置 |
| 靶点内分位数判定 | 待精修完成 | 分别相对阳性和阴性对照计算分位数 | pair 是否超过同靶点阴性背景并进入阳性分布 |
| 正交结构复核 | 后续 | Boltz-2、几何检查、相互作用指纹、必要时短程 MD | pose 是否自洽，是否存在结构性否决 |
| 发现候选 1000 | 后续 | 硬门控 + 校准证据 + 多样性约束 | 哪些新 pair 最值得进入下一步 |
| 1000 次实验测量 | 后续 | discovery + 每靶点正负对照 + 技术重复 | 既找 hit，也能判断实验和模型是否失效 |

固定 1000 不是自然证据阈值。若达到冻结证据条件的 pair 少于 1000，正式流程不得用弱尾部凑数，而应扩大经过校准的计算空间。

## 四、当前靶点级校准结果

ChEMBL 同靶点控制集覆盖 417 个完成 GNINA 的靶点，其中 349 个满足至少 8 个阳性和 8 个阴性。
控制母集来自 ChEMBL 37 人源 `SINGLE PROTEIN`、binding assay、confidence ≥ 9 的 `Ki/Kd/IC50` 严格记录，要求标准关系为 `=`。509,172 条 pair 中，pChEMBL ≥ 6 定义强活性（约 ≤1 µM），pChEMBL ≤ 5 或明确 inactive 定义阴性/不活跃；5–6 为灰区，阳性/阴性冲突记录剔除。

每个靶点优先选择 12 个阳性和 12 个阴性，先保证 Murcko scaffold 多样性，再经过与 FDA 配体相匹配的 MW、重原子数、柔性、电荷、单片段和元素域过滤。最终请求 9,199 个控制，构象与 docking 成功 9,084 个。

这些是历史 assay 标签，不是统一条件下的热力学真阴性；强阳性与强阴性的极端抽样也会使 AUROC 高于实际近边界区分。因此该校准只授予模型“在该靶点可参与排序”的资格，不产生 binder probability。

冻结的两个主通道是：

1. `CNNaffinity`：GNINA 的亲和相关回归分数；
2. `Vina affinity`：基于物理经验项的 docking score，方向统一后使用。

靶点准入条件为：AUROC ≥ 0.65，且 AP 至少高于该靶点阳性比例 0.10。Bootstrap AUROC 95% 下界 ≥ 0.50 时标记为强支持。两个通道按 OR 准入，不求平均、不训练跨靶点总分。

| 校准结果 | 靶点数 |
|---|---:|
| 有控制分数 | 417 |
| 可评估（≥8+8） | 349 |
| CNNaffinity 通过 | 194 |
| Vina 通过 | 166 |
| 两通道同时通过 | 126 |
| 至少一个通道通过 | 234 |
| 双通道强支持 T1 | 77 |
| 双通道通过 T2 | 49 |
| 单通道强支持 T3 | 75 |
| 单通道通过 T4 | 33 |

进入当前 30,000 条队列且靶点已准入的为 16,995 条、171 个靶点、717 个配体、580 个 Murcko scaffold。其中 13,529 条使用核验实验 holo 受体，3,466 条使用 AlphaFold/P2Rank 回退。

## 五、各方法如何组合，而不是盲目采信

### 5.1 ChEMBL 与同靶点正负对照

作用：定义历史阳性、阴性/不活跃和灰区，并检验一个评分通道在具体靶点上是否有判别能力。

不承担：新 pair 的直接证明。模型通过靶点校准，只说明它在该靶点有排序资格。

### 5.2 配体相似性与靶点专属 QSAR

作用：识别知识近邻、适用域和再发现风险；也可建立 positive-control/exploitation 队列。

不承担：远程新骨架主发现。它们天然更擅长给已知化学分布中的分子高分。

正式使用：高相似、同 scaffold、已知活性和近同源扩展作为否决或分流，不作为 discovery 加分。

### 5.3 ConPLEx

输入为药物结构表征和蛋白序列语言表征，输出是药物-蛋白相容性排序信号。

本地去泄漏审计中，ConPLEx 没有形成可跨靶点解释的统一亲和尺度，也没有稳定超过同靶点配体相似性基线。因此只用于大空间低成本召回和计算预算分配，不直接进入最终亲和结论。

### 5.4 SCOPE-DTI

SCOPE-DTI 是与本项目最相似的公开 drug-to-target fishing 路线之一：输入一个药物，在固定人源蛋白词表中输出 target rank。论文采用药物半归纳切分，靶点均可在训练中出现，并报告从两个天然产物各自 Top10 靶点经肿瘤相关性和 HepG2 表达过滤后，对 5 个未表征 pair 做 CETSA/BLI，确认 4 个微摩尔结合。

官方轻量数据审计显示，SCOPE-DTI 词表包含 4,893 个靶点，可映射本项目 463 个靶点中的 452 个（97.62%）；11 个未匹配靶点已单列。该结果说明它在“靶点词表覆盖”上足够全面，但不等同于在本项目远程新骨架域上已经校准。

它证明“先全靶点预测、再物理实验确认”可以工作，但也明确显示性能随靶点已有交互量上升。由于我们的 discovery 域主动排除同靶点近邻配体和同家族扩展，SCOPE-DTI 更适合作为已知空间对照，不能直接替代远程物理门控。

### 5.5 DrugCLIP

输入为三维蛋白口袋和小分子，输出为共同表征空间中的 cosine retrieval score。

它比纯序列模型更接近口袋，但分数仍不是 Kd、IC50 或 kcal/mol。正式用途是从 20 万级远程空间压到可做结构计算的队列；双向 rank 只表示检索互惠，不代表物理结合成立。

### 5.6 AlphaFold、实验 holo、P2Rank/PUResNet

实验 holo 结构用于选择受体构象、链和已知结合位置。AlphaFold 在没有合适实验结构时提供蛋白骨架；P2Rank/PUResNet用于寻找候选口袋。

它们是 target-level 可计算性证据：说明哪里可以放置小分子，不区分同一靶点上的不同药物。当前实验 holo 靶点的 GNINA 校准表现明显优于 AlphaFold/口袋回退，因此受体来源进入证据层级，但不被解释为 pair 阳性。

### 5.7 GNINA

GNINA 同时给出传统 docking 分数和三维卷积网络分数。正式流程先用同靶点 ChEMBL 阳性/阴性判断具体通道是否有效，再用完全相同的受体、box 和参数计算 discovery pair。

只在靶点内部比较；禁止拿不同蛋白的原始 Vina kcal/mol 或 CNNaffinity 直接横排。Pair 的核心判断是：是否超过该靶点阴性分布高分位，并进入该靶点阳性分布的合理区间。
本地控制集审计显示，CNNaffinity 与重原子数的逐靶点 Spearman 中位相关为 0.657，Vina 为 0.389；原始高分可能部分来自分子更大、接触项更多。

正式 pair 判定因此同时保留原始总打分，并用全部靶点的阴性控制拟合 target-centered Huber nuisance model，扣除重原子数、可旋转键和形式电荷趋势。驱动候选晋级的通道还必须超过同靶点 size-adjusted 阴性分布中位数。该修正用于阻止“仅靠体积换分”，不把 ligand efficiency 当成新的亲和事实。

### 5.8 Boltz-2

Boltz-2联合生成复合物结构和亲和相关输出，能够补充柔性结构和条件复合物假说。但本地旧 3,000/1,000 计算缺少完善的同靶点阴性校准，且国际独立研究提示其分类有时对关键突变或 target swap 不够敏感。

正式使用：GNINA 后的正交结构复核、pose 一致性和压力测试；不把 A/B 或 affinity probability 单独定义为 binder。后续应增加 target-swap、关键残基突变和多 seed 稳定性审计。

### 5.9 CORDIAL 前沿模型复核

CORDIAL 是 2025 年提出的结构亲和分类模型，采用 leave-superfamily-out 训练思路，目标是降低传统随机切分的同源泄漏。我们直接使用官方代码、权重和八个 pChEMBL 阈值输出，在与 GNINA 完全相同的 docked control pose 上做了 20 靶点、474 个同靶点阳性/阴性控制基准。

分层样本包含 8 个实验 holo 与 12 个 AlphaFold/P2Rank 受体。CORDIAL 中位 AUROC 为 0.528，GNINA CNNaffinity 为 0.764，Vina 为 0.771；CORDIAL 有 5/20 个靶点达到 AUROC 0.65，但没有任何靶点同时超过两条 GNINA 通道。与项目标签边界一致的 P(pChEMBL≥6) 敏感性分析中位 AUROC 也只有 0.524。因此 CORDIAL 不进入全局候选排序，只保留为方法审计和少数靶点的后续研究对象。这一结果说明“更前沿”不能替代本地部署域校准。

### 5.10 Nesso-1

Nesso-1 是 2026 年发布的开源快速粗粒度共折叠 affinity 模型，输入蛋白序列和配体，输出 binder probability、log10(IC50/μM) 与界面熵。作者报告其速度较 Boltz-2 快一个数量级，同时明确指出公共 affinity benchmark 与训练数据高度相似，OpenBind 上分子量甚至是第二强基线。

官方代码和权重已在本项目部署，并使用与 CORDIAL 完全相同的 20 靶点、474 个真实阳性/不活跃控制进行独立基准。截至本报告编译时已完成 121/474 条推理，完整结论将在全部完成并补做分子大小基线后冻结。最终权限只由这一本地测试决定；当前版本尚不支持指定 pocket 或结构模板，因此即使通过，也只能作为正交通道，不能取代项目的受体/box 约束。

### 5.11 PoseBusters、ProLIF 与 MD

PoseBusters检查键长、碰撞、几何和化学合理性；ProLIF提取关键残基相互作用；短程 MD观察 pose 在显式环境中的快速崩塌或明显不稳定。

这些方法主要用于否决错误 pose。短程稳定不等于有利结合自由能，MD 不适合在 10 万级 pair 上承担第一轮召回。

### 5.12 Open Targets 与疾病证据

Open Targets补全 tractability、靶点类型、组织和 target-disease 证据。STRING、Reactome、GTEx/HPA、表达签名和 TxGNN补充命中后的机制解释。

它们的正面意义是：实验 hit 出现后可以更快选疾病、细胞模型和 readout，也可提前识别完全缺乏转化语境的靶点；但它们不参与第一阶段物理亲和主排序。

## 六、与国际前沿逐项对照

| 国际路线 | 代表工作 | 国际上如何使用 | 我们如何采用 | 关键差异 |
|---|---|---|---|---|
| Ligand-centric target fishing | SEA/QSAR 类 | 依赖靶点已知配体集合扩展新配体或新靶点 | 只作已知空间识别与对照 | 主动牺牲部分历史 recall，换取远程新颖性 |
| 序列 DTI | ConPLEx | 在大规模蛋白-配体空间低成本检索 | 仅召回，不解释为亲和 | 增加本地去泄漏和靶点级校准限制 |
| 半归纳全靶点预测 | SCOPE-DTI | 新药物扫描训练中已知的蛋白词表；CETSA/BLI 验证优先 pair | 作为最接近的公开路线和已知空间对照 | 论文确认 4/5 个未表征 pair，但靶点并非 cold-start，且先经过疾病与表达过滤 |
| 不确定性感知 DTI | EviDTI、DTIAM | 组合分子图/三维特征与蛋白序列，输出 interaction、affinity、MoA 或 epistemic uncertainty | 借鉴不确定性门控；不直接加入物理主分 | EviDTI 的 DrugBank 随机负样本和 DTIAM 的冷启动交叉验证与本项目远程、稀疏真值部署域仍不同 |
| 口袋 dense retrieval | DrugCLIP | 全库快速 pocket-ligand matching | 分配 3D 计算预算 | 不把 cosine 当成亲和值 |
| 经典/深度 docking | Vina、GNINA | 常见于单靶点超大库虚筛；前瞻实验验证 | 多靶点运行，但先逐靶点判断评分是否有效 | 禁止跨靶点原始分数统一横排 |
| 复合物共折叠 | AlphaFold 3、Chai-1、Protenix、FlowDock | 生成蛋白-配体复合物和置信度 | 用于受体回退或少量争议 pair 复核 | 不用“能生成结构”替代结合证据 |
| 结构+亲和基础模型 | Boltz-2 | 同时输出结构与 affinity 相关量 | 作为 GNINA 后正交证据和 stress test | 采用 target-swap/突变审计，避免模型先验假阳性 |
| 快速粗粒度 affinity 模型 | Nesso-1 | 约秒级输出 binder 与 affinity，扩大共折叠筛选规模 | 官方模型做同一 20 靶点/474 控制基准后再授权 | 当前不能指定 pocket/template；作者也警告公共基准和分子量捷径 |
| 去同源结构亲和模型 | CORDIAL | leave-superfamily-out 训练并输出多阈值活性概率 | 官方权重完成 20 靶点本地控制基准；因无稳定增量不纳入排序 | 以本地部署域表现决定模型权限，而不是按论文新旧采信 |
| 物理自由能 | FEP/ABFE/QM/MM | 对已知靶点、可靠 pose 和同系列配体精修 | 仅适合最终几十条，不适合 33 万 target fishing | 计算贵且依赖体系准备，不是第一轮通用检索器 |
| 无偏基准 | Bento、LIT-PCBA、PLINDER、PoseBench、CleanSplit、NTAB | 用确认阴性、时间切分、蛋白/口袋去同源和 ligand novelty tier 测试泛化 | 用 ChEMBL 同靶点正负控制、相似度 <0.40、远程同源门控和大小偏差修正 | Bento 报告未见口袋上 AI 方法退化、GNINA 最稳健；支持保留经本地校准的经典基线 |
| 前瞻挑战 | CACHE | 计算提交后做正交结合实验、聚集/干扰反筛 | 设计 discovery+正负对照+重复的测量包 | 我们覆盖更多靶点，但每靶点统计功效更弱 |
| 无标记蛋白组 target engagement | TPP/CETSA/PISA、LiP-MS | 少量药物直接在细胞/裂解液中扫描数千蛋白 | 作为重点药物子集的计算外正交发现路线 | 更接近直接实验，但受表达、覆盖、热位移/肽段响应和成本限制 |
| 逐 pair 生物物理确认 | SPR、BLI、MST、ITC | 对候选做浓度梯度、动力学和正交复核 | 定义最终前瞻 hit，并与酶活/通道功能区分 | 这是计算分数转为真实 Kd/engagement 的必要终点 |

### 国际前沿给我们的直接约束

1. 公开 benchmark 高分不能直接转化为项目命中率；结构、蛋白和 ligand leakage 会显著抬高结果。
2. 2026 年 Novelty-Tiered Affinity Benchmark 进一步要求按时间和最大 Tanimoto 相似度分层；低于 0.35–0.40 的远程化学空间应单独报告，不能与同系列插值混合。
3. SCOPE-DTI、DTIAM 和 EviDTI 的前瞻案例说明 DTI 模型能提高命中机会，但这些案例通常是单靶点或已知靶点词表、再加疾病/表达过滤，不能外推为数百靶点统一命中率。
4. 单靶点前瞻 docking 可以产生命中，但不同靶点命中率差异很大，因此多靶点流程必须先做 target-specific calibration。
5. 共折叠模型改善 pose，不代表自动解决 affinity classification；需要物理检查、分子量基线和反事实压力测试。
6. FEP/ABFE 更适合可靠 pose、明确靶点和小规模精修，不能替代第一轮数十万 pair 的召回。
7. 蛋白组 target-deconvolution 是计算全靶点扫描的实验互补，而非彼此替代；前者覆盖真实细胞环境，后者可覆盖不表达或难质谱检测的靶点。
8. 真正有说服力的终点是前瞻、正交、可复现的 binding/target-engagement 实验，而不是模型间共识数量。

## 七、我们的工作相对国际路线的定位

### 已经具备的项目差异化

- **Drug-first、multi-target 的远程发现场景**：与常见 target-first 单靶点超大库筛选互为镜像，目标是系统重绘上市药物可直接作用的蛋白空间。
- **双重新颖性约束**：同时约束 ligand/scaffold 知识近邻与 target/homology 近邻，避免把同家族或相似配体扩展包装为新发现。
- **模型使用权按靶点发放**：不是所有靶点都使用同一模型；只有能区分该靶点阳性/阴性的评分通道才参与该靶点候选判断。
- **跨靶点原始分数禁用**：候选相对各自靶点的控制分布归一化，再通过受限组合形成多靶点包。
- **显式控制 docking 大小偏差**：原始总打分与阴性控制拟合的 size-adjusted 残差同时过门，防止大分子仅凭重原子数获得系统性优势。
- **发现与校准一体化实验包**：同一次首轮实验既尽量获得 hit，也产生以后判断模型精度、失效靶点和富集程度所需的前瞻数据。

这些是方法学和工程流程上的贡献，目前不能表述为已验证的命中率创新；只有首轮实验后才能证明其实际价值。

### 尚未解决的限制

- ChEMBL inactive 不是统一条件下的真正热力学阴性，且不同 assay 的标签噪声不可消除。
- 171 个靶点上的每靶点候选数量有限，广度与统计功效存在张力。
- AlphaFold/预测口袋不能完整表达诱导契合、膜环境、辅因子和多聚体状态。
- FDA 药物往往电荷复杂、柔性高或存在活性代谢物，标准 docking 对部分实体不适用。
- 任何计算 A/B/P1 等级在前瞻实验前都不是结合概率。

## 八、冻结的最终候选规则

### 8.1 Hard gate

必须同时满足：非已知 FDA pair、非 exact ChEMBL active、非同 known scaffold、与已知活性最大相似度低于 0.40、非扩展同源/同家族再发现、无严重化学责任、受体与口袋可执行。

### 8.2 Target gate

靶点至少有一个 GNINA 主通道通过同靶点正负校准。未通过不等于候选不结合，只表示 GNINA 对该靶点不可判定，不进入本轮 GNINA 主筛。

### 8.3 Pair gate

对每个有效通道同时计算原始分数和 size-adjusted 分数相对于同靶点阳性/阴性的经验分位数。Pair 分级遵循冻结逻辑：

- P1：两个原始通道均超过阴性第 90 百分位、达到阳性分布第 25 百分位，且两个 size-adjusted 通道均不低于阴性中位数；
- P2：至少一个有效原始通道超过阴性第 95 百分位，另一通道不矛盾，驱动通道 size-adjusted 分位不低于 0.50，且候选进入阳性分布；
- P3：有效原始通道超过阴性第 90 百分位，其他原始通道至少不低于阴性中位数，驱动通道 size-adjusted 分位不低于 0.50；
- R：未达到校准门槛，不用于凑满正式候选。

### 8.4 Portfolio gate

1000 条 discovery 候选按 pair tier、target tier、实验 holo、检索通道和靶点内分位数进行字典序排序，不使用任意加权总分。组合约束为：每靶点最多 14 条、每药最多 3 条、每 scaffold 最多 12 条、同一靶点同一 scaffold 最多 1 条，并控制 kinase、ion channel 等类别占比。

## 九、两种“1000”必须分开

### 9.1 Discovery1000

1000 条全部为新的远程 drug-target 假说，适合作为计算推荐储备表。它最大化候选覆盖，但单独拿去湿实验无法判断某个靶点没有信号是候选失败还是 assay 失败。

### 9.2 Assay1000

建议的 1000 次首轮实验测量由以下组成：

- 约 760–800 条 discovery pair；
- 优先覆盖约 80 个靶点，必要时扩展到 100 个；每个靶点至少 6 条 discovery pair；
- 每个入选靶点各 1 个已知阳性和 1 个阴性/不活跃对照，共约 160–200 条；
- 约 40 条技术重复、空白或非特异/聚集反筛。

候选先按靶点分配最低覆盖，再按全局证据顺序填充剩余孔，避免出现“有对照但没有足够 discovery 药物”的空靶点。这样仍是多靶点广筛，同时能够验证蛋白和 assay 是否工作、估计前瞻富集、判断模型在哪些靶点失效。若实际平台只能做 1000 个唯一候选，应另行预留控制孔，不能把控制完全取消。

## 十、当前计算状态与后续动作

截至 2026-08-03：

- 717/717 个 discovery 药物三维构象准备成功；
- 16,995 条 pair 已分配到 171 个准入靶点；
- discovery GNINA 精修已完成 8/171 个靶点；扩展控制精修已完成 5/171 个靶点，均仍在运行；
- GNINA 采用 2 张 RTX 4090、每任务 6 CPU、exhaustiveness 8、每 pair 5 个 pose；经资源审计后后续调度提高到每 GPU 4 个并发；
- 结果将保留原始 pose、每 pair 最佳分数、失败记录和靶点内归一化值；
- CORDIAL 官方模型已完成 20 靶点、474 个控制的独立基准，未显示相对 GNINA 的稳定增量，因此不进入主排序；
- Nesso-1 官方模型已完成 121/474 条控制推理，当前仅记录进度，不提前形成性能结论；
- SCOPE-DTI 官方词表覆盖本项目 452/463 个靶点，覆盖率 97.62%；
- 磁盘剩余约 19 GB，任务结束后压缩原始 SDF，需持续监控空间。

精修完成后的唯一顺序是：

1. 生成靶点内阳性/阴性分位数和 P1/P2/P3/R；
2. 执行 hard gate 与组合多样性审计；
3. 形成 Discovery1000 和 Assay1000；
4. 对优先的 1,000–2,000 条做 Boltz-2/反事实结构复核，而不是让 Boltz 原始概率接管排序；
5. 对最终少量 pose 做几何、相互作用和必要的短程 MD；
6. 湿实验命中后，再接入 Open Targets 和疾病机制确定新用途。

## 十一、国际资料来源

- Boltz-2: https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
- AlphaFold 3: https://www.nature.com/articles/s41586-024-07487-w
- GNINA: https://pmc.ncbi.nlm.nih.gov/articles/PMC8191141/
- Nesso-1 official model and report: https://github.com/recursionpharma/nesso and https://www.valencelabs.com/wp-content/uploads/2026/07/nesso1.pdf
- Bento docking benchmark: https://openreview.net/forum?id=kIxAQxUZHq
- SCOPE-DTI: https://www.nature.com/articles/s41467-025-66311-9
- Large-scale chemoproteomics: https://pubmed.ncbi.nlm.nih.gov/38662832/
- DrugCLIP: https://proceedings.nips.cc/paper_files/paper/2023/file/8bd31288ad8e9a31d519fdeede7ee47d-Paper-Conference.pdf
- ConPLEx: https://pubmed.ncbi.nlm.nih.gov/37289807/
- CACHE Challenge #1: https://pubmed.ncbi.nlm.nih.gov/39654129/
- LIT-PCBA: https://doi.org/10.1021/acs.jcim.0c00155
- PLINDER: https://www.biorxiv.org/content/10.1101/2024.07.17.603955v1
- PoseBusters: https://arxiv.org/abs/2308.05777
- PoseBench: https://arxiv.org/abs/2405.14108
- PDBbind CleanSplit/GEMS and data leakage: https://pubmed.ncbi.nlm.nih.gov/41143208/
- Novelty-Tiered Affinity Benchmark (2026): https://www.biorxiv.org/content/10.64898/2026.06.29.735309v1
- EviDTI: https://www.nature.com/articles/s41467-025-62235-6
- DTIAM: https://www.nature.com/articles/s41467-025-57828-0
- GEMS official implementation: https://github.com/camlab-ethz/GEMS
- CORDIAL: https://pubmed.ncbi.nlm.nih.gov/41100673/ and https://github.com/bpBrownLab/CORDIAL
- Independent Boltz-2 assessment: https://pubmed.ncbi.nlm.nih.gov/41592323/
- FEP benchmark and limits: https://www.nature.com/articles/s42004-023-01019-9
- Prospective ultra-large docking: https://www.nature.com/articles/s41586-019-0917-9

## 十二、结论

项目不再追求“一个总分从 33 万条中直接证明 1000 个真结合”。正式路线是：以严格远程空间保证新颖性，以低成本模型分配计算，以同靶点正负对照授予模型使用资格，以 GNINA 靶点内分位数判断 pair，以 Boltz/pose/MD 做正交复核和否决，最后用包含正负对照的前瞻实验产生真实命中率。

这一路线与国际前沿一致的部分是重视结构、无偏基准和前瞻实验；相对差异在于我们面对的是多靶点 drug-first target fishing，因此必须额外解决跨靶点不可比、知识近邻再发现和实验失效不可判定三个问题。计算1000包与蛋白组级 target-deconvolution 是互补路线：前者压缩明确 pair 的实验预算，后者可对少数重点药物在真实蛋白组中发现计算靶点宇宙之外的作用。

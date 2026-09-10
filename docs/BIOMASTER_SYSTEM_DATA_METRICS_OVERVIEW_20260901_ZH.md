# ReTargetMap系统、数据来源与评价指标统一说明

> 更新日期：2026-09-01  
> 主任务：给定一种老药，在规定候选靶点空间内建立潜在作用靶点谱。  
> 本文同时区分“冻结当前系统”和“新重训候选”；内部模型代号只用于复现，不用于项目汇报。
> 配套PDF：`docs/reports/RETARGETMAP_CURRENT_PROJECT_HANDBOOK_20260901_ZH.pdf`。

## 0. 项目名称与科学定位

对外工作名统一为 **ReTargetMap（老药靶点谱检索框架）**：

- `Re`：对已上市或药理信息充分的既有药物重新审视；
- `Target`：主任务是从药物出发找潜在作用靶点；
- `Map`：输出是一个候选靶点谱和相对排序，不是单一答案或未校准的结合概率。

项目要解决的问题是：当给定一种老药时，如何在明确、可计算且实验体系可解释的蛋白空间中，将它的已知与潜在次级靶点建立为可审计排名，再将Top-K交给数据库/文献复核、结构证据和湿实验。

| 框架是什么 | 框架不是什么 |
|---|---|
| 药物中心的多靶点检索和优先级排序 | 不是纯粹的单靶点虚拟筛选 |
| 学习异质直接关系/活性证据 | 不是纯Ki/Kd定量亲和力回归 |
| 为文献、结构和湿实验减少候选数 | 不用模型logit取代物理实验 |
| 将unknown保持为待发现空间 | 不把未报告pair伪造成negative |

## 1. 先把整个项目压缩成一条主线

```text
冻结老药实体
    ↓
候选靶点注册与实验体系分路
    ↓
药物/蛋白/结构特征
    ↓
共享 drug–target pair backbone
    ├─ pair关系证据与亲和力辅助输出
    ├─ drug→target主方向残差头
    └─ target→drug反向辅助头
    ↓
每种药物内部、同一路由候选靶点排序
    ↓
Top-K证据复核：数据库、文献、结构/口袋、相似性、选择性与可实验性
    ↓
湿实验验证
```

系统并不是“输入一个pair，输出一个结合概率”这么简单。双输入网络首先学习pair证据；真正的老药找靶点任务，还必须规定以药物为query、哪些靶点互相竞争、使用什么排序损失、在哪个候选空间归一化以及如何评价。

主方向始终是：

```text
drug query → rank candidate targets within the same drug
```

反方向`target→drugs`只保留为共享生物学表征和候选复核的辅助证据，不与主结果混报。

## 2. 目前实际保留的三个系统角色

| 系统角色 | 组成 | 当前用途 | 结论边界 |
|---|---|---|---|
| 冻结通用主系统 | ReTargetMap监督分数、训练集派生的药物相似性/靶点先验、泄漏安全图排序，经每药百分位/Borda汇总 | 当前384靶点核心的默认独立排序 | 不依赖DTIAM推理；有冻结S5证据 |
| 新重训方向候选 | 共享pair backbone + drug→target主头 + target→drug辅助头，三种子FULL_FIT | 720×384、720×745候选评分和新案例支持 | 时间测试样本小，状态为候选，尚未替换冻结主系统 |
| 含DTIAM统一系统 | ReTargetMap独立证据、DTIAM及图排序的系统级融合 | 可选的最强工程排序 | 必须明确标“含DTIAM”，不能作为ReTargetMap独立结果 |

此外还有一个后置的结构复核层：GNINA、Boltz、口袋接触和pose等pair-specific证据只用于Top-K重排与人工复核，目前不是通用主模型已经验证的输入。

### 2.1 冻结当前与新候选的关系

- `FROZEN_CURRENT`：正式结果仍以720×384和冻结S5独立排序为准。
- `RETRAIN_CANDIDATE`：新方向头已经为745靶点补齐输入并完成720×745打分，但跨生化与功能体系的分数尚未统一校准。
- “已经算出分数”不等于“可以发布统一rank/745”；当前最多发布路由内rank，并将完整745作为覆盖压力测试。

## 3. 通用神经模型的输入、架构和输出

### 3.1 输入特征

| 模态 | 当前输入 | 维度 | 作用 |
|---|---|---:|---|
| 药物化学 | Morgan指纹 | 2,048 | 稳定、可审计的二维化学表示 |
| 蛋白主表示 | ProtBERT pooled embedding | 1,024 | 蛋白序列语义 |
| 蛋白辅助表示 | ESM2-650M pooled embedding | 1,280 | 补充进化和序列上下文 |
| 实验体系 | target assay family embedding | 24 | 区分激酶、酶、离子通道、转运体等通道 |
| 结构上下文 | target-level pocket/structure context | 19 | 描述靶点结构可用性、口袋、holo证据与质量 |

19维口袋特征是“这个靶点有哪些结构和口袋上下文”，同一靶点面对不同药物时基本共享；它不是某个药物在某个口袋中的pose、接触图或局部相互作用指纹。缺失结构通过`structure_mask`精确回退，不把“没有结构”当作负作用证据。

当前通用模型中：

- ConPLex输入关闭，`conplex_enabled=false`；ConPLex只作为外部对照，不是通用主干的隐藏输入。
- DTIAM和DrugCLIP也不进入ReTargetMap独立神经头；只有明确标注的统一系统允许使用DTIAM。
- pair-specific局部图分支在代码中保留，但当前FULL_FIT配置维度为0，尚未进入生产主模型。

### 3.2 共享pair backbone

药物和蛋白分别映射到192维隐空间。低秩FiLM交互让药物表示受靶点条件化、靶点表示受药物条件化；随后拼接：

```text
drug
target
drug × target
|drug − target|
low-rank bilinear interaction
assay-family embedding
```

256维pair trunk后接6个pair-conditioned专家及门控，产生共享pair logit。模型另有连续亲和力、亲和力不确定性和结构残差输出。当前通用backbone约352万参数。

### 3.3 两个查询方向

共享pair hidden之后增加两个轻量、零初始化的残差头：

```text
D→T score = shared pair score + drug→target residual
T→D score = shared pair score + target→drug residual
```

零初始化保证方向头启用前与原pair模型逐元素一致。当前方向头训练时冻结共享backbone，避免20-query时间开发/测试集反向改写基础表示。

### 3.4 最终输出

| 输出 | 含义 | 能否解释为概率/亲和力 |
|---|---|---|
| pair logit | 共享关系/活性证据 | 否 |
| affinity output | 有连续标签时的辅助pActivity趋势 | 未经独立校准不能直接当Ki/Kd |
| drug→target logit | 同一药物内跨靶点排序分数 | 否 |
| target→drug logit | 同一靶点内跨药物辅助排序分数 | 否 |
| rank/384或路由rank | 规定候选空间内的相对优先级 | 是当前最可解释的部署输出 |

## 4. 数据来源

### 4.1 老药实体来源

当前部署轴固定为720种已上市或已有充分药理信息、具有标准单一结构的老药。上游冻结药物目录综合使用：

- Drugs@FDA与FDA NME/Novel资料：批准身份与时间；
- Orange Book：申请和活性成分交叉核验；
- RxNorm：盐型和active moiety关系；
- GSRS/UNII：物质身份；
- PubChem与ChEMBL：标准结构和实体交叉核验。

进入模型后以标准化结构和药物特征索引为计算实体。720是当前部署库，不是综合训练数据中的全部药物。

### 4.2 通用监督数据

| 来源 | 进入模型的数据 | endpoint | 用途 |
|---|---:|---|---|
| ChEMBL 37 | 426,939条可解析关系 | Ki、Kd、结合型IC50、明确inactive及直接关系语义 | 二分类关系、pair排序和部分连续活性 |
| BindingDB | 9,778条direct Ki/Kd | 连续亲和力 | affinity-only辅助监督，不强制造二分类负例 |
| ChEMBL审计恢复 | 914条审计记录，其中531条去重后进入FULL_FIT | 来源特定关系 | 补充冷药物/关系覆盖 |

完整漏斗为：

| 数据层 | 行数 |
|---|---:|
| ChEMBL标签过滤前 | 509,172 |
| 标签语义合格、特征过滤前 | 427,526 |
| ChEMBL特征可解析 | 426,939 |
| 合并BindingDB与恢复关系后的审计表 | 437,631 |
| FULL_FIT去重关系 | **437,248** |

FULL_FIT中包括：

- 二分类观测427,470条：331,381个positive、96,089个negative/inactive；
- affinity-only 9,778条；
- 296,885个标准parent InChIKey，映射为296,108个模型药物特征实体；
- 1,022个来源target ID，经序列/特征归一化为843个蛋白实体；
- 8,029个药物具有组内正负两类，可训练drug→target排序；
- 434个靶点具有组内正负两类，可训练target→drug排序。

这解释了为什么43.7万条关系并不等于43.7万条高质量药物内排序监督：约79.7%的二分类药物只有一条记录。

### 4.3 历史平衡评估表

86,674是早期ChEMBL平衡benchmark，不是当前训练总量：

- 62,488个化合物、428个靶点；
- 45,983个positive、40,691个negative；
- 每个target×label最多150条；
- 一个分子特征失败，因此模型可解析行为86,673。

它仍然有价值，因为S1–S5冻结切分、DTIAM/ConPLex等同数据比较和历史趋势均建立在这张表上。

### 4.4 靶点注册、序列与结构来源

靶点端从ChEMBL 37人源`SINGLE PROTEIN`、至少有一条小分子MoA的888个官方靶点开始；GPCR作为独立实验体系排除后，得到745个非GPCR登记靶点。

| 路由 | 数量 | 数据角色 |
|---|---:|---|
| 主路由：生化直接小分子 | 367 | 生化体系内排序 |
| 主路由：功能直接小分子 | 83 | 功能体系内排序 |
| 特殊体系且有直接小分子证据 | 42 | 独立实验路线，不并入主排名 |
| 高新颖性、可实验、无直接小分子证据 | 8 | 探索性候选 |
| 特殊且无直接小分子证据 | 245 | 登记和诊断，不作当前生产声明 |

蛋白序列和身份以ChEMBL/UniProt映射为基础；Open Targets等外部证据用于靶点扩展和路由，不作为普通pair监督标签。结构/口袋证据来自历史实验holo上下文及可审计的AlphaFold/P2Rank补充。

450个主生产靶点进一步按五种可实验车道分路：

| 实验车道 | 数量 | 归属 | 为什么分开 |
|---|---:|---|---|
| Enzyme biochemical | 208 | 生化路由 | 直接酶活性/底物转化体系 |
| Kinase biochemical | 122 | 生化路由 | 激酶生化活性体系 |
| Nuclear/epigenetic domain | 37 | 生化路由 | 核受体、转录和表观调控域体系 |
| Ion-channel functional | 56 | 功能路由 | 需要电生理或功能读出 |
| Transporter membrane functional | 27 | 功能路由 | 需要转运/膜功能读出 |

GPCR并非“无价值”，而是需要膜环境、构象状态和专门功能协议，因此不与本轮非GPCR统一筛选空间混排。结构/口袋缺失也不是删除靶点的硬条件。

新745特征库中：

- 745/745具有ProtBERT与ESM2输入；
- 427个target-warm，318个target-cold；
- 387个具有可用19维结构/口袋上下文，358个通过mask使用无结构回退。

### 4.5 KIRHub功能实验数据

KIRHub是1 μM HotSpot功能抑制矩阵，不是Ki/Kd亲和力数据。

| 数据层 | 规模 | 用途 |
|---|---:|---|
| 原始WT矩阵位置 | 92药物×409构建体=37,628 | 数据登记 |
| 有实际数值 | 36,593 | 未测1,035个位置保持unknown |
| 严格通用模型回顾性子集 | 2,823对、78药物、202强抑制pair | 冻结/候选通用头post-audit |

项目将`1 μM inhibition ≥70%`定义为strong functional hit。该阈值只服务于功能筛选，不能改写成“高亲和力”。

### 4.6 数据层之间的关系

| 数据层 | 规模 | 一行的含义 | 是否训练 | 主要用途 |
|---|---:|---|---|---|
| ChEMBL综合监督 | 426,939 | 聚合后的标准药物–蛋白直接关系 | 是 | pair表示、二分类、组内排序 |
| BindingDB affinity-only | 9,778 | 有direct Ki/Kd的pair | 是，辅助 | 连续亲和力趋势 |
| 审计恢复 | 531 | 从漏项审计找回的去重关系 | 是 | 修复冷药物/关系覆盖 |
| FULL_FIT | 437,248 | 上述三类数据去重后的训练总表 | 是 | 最终候选评分模型 |
| 历史平衡benchmark | 86,674 | 每target×label最多150条的有标签pair | 只在S1–S5协议内 | 冷启动/时间对比 |
| 720×384部署矩阵 | 276,480 | 每种老药与每个核心靶点的候选组合 | 否 | `rank/384`与Top-K交付 |
| 2024–2025密集D→T | 20 query / 27 future positive | 未来阳性加完整unlabeled候选背景 | 否 | 时间后密集检索 |
| KiRHub严格子集 | 2,823 measured pair | 实际测量并映射成功的1 μM功能pair | 否，post-audit | 跨endpoint功能迁移审计 |

因此，`437,248训练关系`、`86,674评价pair`、`276,480部署候选pair`和`2,823 KiRHub实测pair`是四个不同概念，不能在汇报中都简写为“数据量”。

## 5. 标签合同

### 5.1 通用关系标签

```text
mean pChEMBL ≥ 6.0，且无冲突       → positive
mean pChEMBL ≤ 5.0或明确inactive    → negative/inactive
5.0 < mean pChEMBL < 6.0            → grey，普通二分类不使用
强阳性与inactive证据冲突             → conflicting，普通二分类不使用
BindingDB direct Ki/Kd                → continuous affinity-only
数据库未报告pair                      → unknown，不是negative
```

同一pair多条assay先聚合并检查关系符号、assay类型和冲突，不能只取最强的一条结果。

### 5.2 部署矩阵标签

720×384、720×450和720×745主要是推理候选矩阵，不是稠密实验标签矩阵。未报告pair只获得模型分数和相对rank，不获得伪造的0标签。

## 6. 训练方式与防过拟合

### 6.1 共享backbone

- target/label/scaffold平衡采样，而不是删除高频数据；
- 每个target query抽取16条，正负各半；
- target抽样频率使用0.5次幂，降低热点靶点垄断；
- FULL_FIT中约9.375%的batch chunk用于affinity-only关系；
- 三个随机种子，开发集早停后再按选定epoch重训FULL_FIT。

### 6.2 方向头

- drug→target主头：每个可训练药物动态抽4条观测，2正2负；
- target→drug辅助头：每个靶点抽16条观测；
- 只使用明确观测的正负关系，unknown不进入普通BCE；
- 损失由BCE、组内pairwise ranking、listwise ranking和残差L2约束组成；
- Stage-A冻结共享backbone，只训练方向头。

### 6.3 时间边界

```text
≤2022：训练
2023：方向采样和epoch选择
2024–2025：选定方案后的最终时间测试
FULL_FIT：结构、采样和epoch冻结后吸收全部合格关系，用于部署
```

2/4/8/16条/query消融只计算2023开发指标；选定4条后才运行三个种子的2024–2025测试。FULL_FIT本身没有重训后的无偏内部性能估计。

## 7. 评价集合：每条证据回答什么问题

### 7.1 历史S1–S5冻结benchmark

| 协议 | 隔离内容 | rows | 回答的问题 | 主方向指标 |
|---|---|---:|---|---|
| S1 scaffold-cold | 测试药物骨架与训练隔离 | 86,673 | 新化学骨架能否迁移 | drug-macro AUPRC |
| S2 target-homology-cold | 测试蛋白同源簇与训练隔离 | 86,673 | 新/远缘靶点能否迁移 | target-与drug-macro AUPRC |
| S3 strict double-cold | 药物骨架和蛋白同源簇同时隔离 | 17,732 | 双冷启动能力 | 三种AUPRC并列报告 |
| S4 temporal | 首次见于2023–2025的observed pair | 7,839 | 时间外推的pair区分能力 | drug-macro与micro AUPRC |
| S5 old-drug entity-cold | 测试药物实体不参加训练 | 2,556 | 未见药物的observed-target排序 | drug-macro AUPRC |

这些切片只包含数据库中有观测标签的pair，不是对完整384靶点的密集筛选。

其中S1/S2的`86,673 rows`是整张参与5折交叉验证的可解析benchmark，每折test约1.73万行，不是“单一test集有86,673行”。S4的7,839和S5的2,556则是固定切分中的测试行数。

| 协议 | 切分单位 | 严格隔离 | 最关键的不能误读之处 |
|---|---|---|---|
| S1 | Bemis–Murcko药物骨架簇 | 同骨架不跨train/test | 药物化学冷，靶点不一定冷 |
| S2 | 蛋白同源簇 | 同源蛋白不跨train/test | 靶点冷，药物不一定冷 |
| S3 | 药物骨架簇 + 蛋白同源簇 | 两侧同时隔离 | 最接近新骨架+新靶点，但仍只在有标签pair上评价 |
| S4 | 关系首次出现时间 | 早期训练、2023–2025观测测试 | 是observed-pair分类，不是完整候选轴检索 |
| S5 | 药物实体 | 测试药物整体不参与训练 | 与主方向最对齐，但每药只在observed test pair中排序 |

### 7.2 当前最重要的四条评价线

| 评价线 | query和分母 | 标签 | 主要用途 |
|---|---|---|---|
| S5冻结系统比较 | 302个测试药物；每药只在其observed测试pair中比较，75个双标签药物可算宏指标 | ChEMBL observed positive/negative | 当前独立主系统与DTIAM比较 |
| 2024–2025密集D→T | 20个药物、27个未来阳性；384核心平均剩余377.9候选/药 | 未来positive；其他候选为unlabeled background | 新方向头是否把后发关系排到前部 |
| KiRHub严格回顾性 | 78个药物、2,823个实测pair；33个双标签药物 | 1 μM功能抑制 | 跨endpoint功能迁移审计 |
| 完整空间rank | 每种药在384或路由候选集内完整排序 | 多数pair为unknown | 实际候选优先级与案例展示 |

必须区分：S4的7,839是observed-pair时间分类表；密集D→T的20 query/27 positive是在满足老药、warm靶点、未来阳性并移除旧已知关系后，对完整候选轴排序。二者不是同一测试。

## 8. 评价指标词典

### 8.1 分类/区分指标

| 指标 | 定义与方向 | 当前用途 | 主要局限 |
|---|---|---|---|
| AUROC↑ | 随机正例分数高于随机负例的概率 | 辅助衡量总体区分 | 类别极不平衡时可能显得过于乐观 |
| AUPRC/AP↑ | Precision–Recall曲线面积 | 稀疏阳性的主指标 | 强依赖正例率和候选集合，跨数据集不可直接比 |
| Micro AUPRC↑ | 把所有pair合并后计算 | 全局pair区分 | 被大query和高频靶点主导，不等于每种药都好 |
| Drug-macro AUPRC↑ | 每个具有正负两类的药物分别算AP，再等权平均 | **drug→target主指标** | 单类药物无法进入宏平均 |
| Target-macro AUPRC↑ | 每个具有正负两类的靶点分别算AP，再平均 | target→drug和蛋白侧指标 | 不是主生产方向 |

“阳性比例”只是随机排序的micro-PR参照；drug-macro随机基线应由各药物自身正例率决定，不能直接拿全局prevalence画一条宏指标基线。

### 8.2 检索/排序指标

| 指标 | 定义 | 最适合回答的问题 |
|---|---|---|
| Recall@K↑ | 每个query的阳性中有多少进入Top-K，再做宏平均 | 已知阳性能否在实验预算K内被找回 |
| Hit@K↑ | query的Top-K是否至少包含一个阳性 | 每个query是否至少获得一个可用候选 |
| NDCG@K↑ | 越靠前的相关项权重越高，并按理想排序归一化 | 多阳性或连续抑制强度的前部排序质量 |
| MRR↑ | 第一个阳性的倒数rank，对query取平均 | 首个成功候选出现得多早 |
| Mean/median positive rank↓ | 阳性在候选空间中的绝对位置 | 实验需要检查多少候选 |
| Mean top percentile↑ | 将rank按候选数归一化到顶部百分位 | 候选数不同的空间间做辅助比较 |
| Positive-retrieval AP↑ | 把冻结未来阳性对unlabeled背景做检索AP | 时间密集检索诊断；不是二分类性能声明 |
| rank/N↓ | 某pair在该query完整N个候选中的位置 | 个案和生产部署最直接的结果 |

Recall@K必须同时写清分母。例如：

```text
KIRHub measured-subset Recall@20
S5 observed-pair Recall@20
dense 384-target Recall@20
```

这三个数字不能互换。`rank/384`也不是Recall@K。

### 8.3 连续值与校准指标

| 指标 | 用途 | 边界 |
|---|---|---|
| Spearman↑ | 同一靶点内预测与Ki/Kd/pActivity强弱次序的一致性 | 只衡量单调关系，不代表绝对误差 |
| Continuous NDCG@K↑ | 用实际抑制率作为graded relevance | 适合KiRHub功能强弱排序 |
| Brier↓ | 二分类概率平方误差 | 仅对有观测标签且完成概率映射的结果有效 |
| ECE↓ | 预测置信度与实际频率的分箱偏差 | 当前原始logit跨靶点未必校准；不能当结合概率 |

### 8.4 统计比较

- 模型比较按drug query或药物簇bootstrap，不把pair行当作完全独立样本；
- 报告差值、95% CI和`P(Δ>0)`，不能只报点估计；
- CI跨0：只能说“点估计领先/下降”，不能说统计学确认；
- 多个种子控制模型初始化方差，不会自动解决测试query太少的问题。

### 8.5 公式、计算单位与数值解读

设药物query为`q`，候选集为`C_q`，已知阳性为`P_q`，排序第`i`位的相关性为`rel_i`。

| 指标 | 公式/计算单位 | 对当前数字的正确读法 |
|---|---|---|
| Precision | `TP/(TP+FP)` | 模型判为阳性的有标签pair中，真阳性的比例 |
| Recall/TPR | `TP/(TP+FN)` | 有标签阳性中被阈值模型找回的比例 |
| AUROC | `P(score_positive > score_negative)` | 0.5约为随机；极度不平衡时可能过于乐观 |
| AP/AUPRC | `Σ Precision@i·rel_i / |P|` | 阳性出现位置上Precision的平均；随机参照与阳性率有关 |
| Micro AUPRC | 合并全部pair后计算AP | 每个pair等权，数据量大的query影响更大 |
| Drug-macro AUPRC | 对每个双标签药物分别计算AP，再等权平均 | 每种药等权，与`drug→target`生产任务对齐 |
| Target-macro AUPRC | 对每个双标签靶点分别计算AP，再平均 | 是`target→drug`侧指标，不能代替主方向 |
| Recall@K | `|TopK_q ∩ P_q| / |P_q|`，再对query平均 | 实验预算K内找回的已知阳性比例；必须标候选分母 |
| Hit@K | `I(|TopK_q ∩ P_q|>0)` | 每个query前K中是否至少有1个阳性 |
| NDCG@K | `DCG/IDCG`，`DCG=Σ(2^rel_i-1)/log2(i+1)` | 阳性/强活性越靠前得分越高；不是概率 |
| MRR | `mean(1/first_positive_rank_q)` | 只强调第一个阳性出现得多早 |
| Mean positive rank | 阳性rank的平均 | 平均要查到多少名才覆盖阳性；越低越好 |
| Top percentile | `1-(rank-1)/(N-1)` | 将不同候选数N的rank归一化；不代表绝对实验量相同 |
| Positive-retrieval AP | 未来阳性对unlabeled候选背景的AP | 是密集检索诊断，背景不是已证实negative |
| Spearman | 预测排名与实测连续值排名的相关 | 只保证单调次序，不保证Ki/Kd数值准确 |
| Brier | `mean((p-y)^2)` | 越低越好；只对已校准概率和有标签pair有效 |
| ECE | `Σ |bin|/n·|accuracy-confidence|` | 越低越好；依赖分箱和路由校准 |

对模型差值的bootstrap以药物query或药物簇为重采样单位。95% CI跨0`0`时，只能说“点估计领先/下降”，不能说“已确认优越/退化”。

## 9. 当前主要结果应如何归位

### 9.1 冻结S5：当前独立主系统

| 方法 | 是否含DTIAM | Drug-macro AUPRC | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|
| DTIAM | 是 | 0.7365 | **0.9481** | 0.8107 |
| ReTargetMap独立验证排序 | 否 | **0.7521** | 0.8990 | **0.8164** |
| 含DTIAM统一系统 | 是 | **0.7933** | 0.9311 | **0.8499** |

独立ReTargetMap的AUPRC点估计高于DTIAM，但Recall@20更低，AUPRC差值置信区间跨0。统一系统最强，但属于含DTIAM的系统级结果。

### 9.2 新方向头：2024–2025密集检索

| 空间 | 平均阳性rank↓ | MRR↑ | Positive-retrieval AP↑ | Recall@20 |
|---|---:|---:|---:|---:|
| 384核心 | 79.28 | 0.1945 | 0.1521 | 0.4333 |
| 450主路由预校准 | 91.86 | 0.1847 | 0.1446 | 0.4333 |
| 500活跃路由诊断 | 105.95 | 0.1814 | 0.1393 | 0.3833 |
| 745完整登记压力测试 | 182.97 | 0.1634 | 0.1198 | 0.2333 |

450、500和745结果证明模型能够覆盖这些靶点，不证明不同实验路由已经校准。生产输出仍应分别报告生化rank/367、功能rank/83、特殊rank/42和探索候选。

### 9.3 KiRHub严格回顾性审计

| 方法 | Drug-macro AUPRC | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|
| DTIAM | 0.3867 | 0.3066 | 0.3917 | 0.6283 |
| 新FULL_FIT方向头 | **0.4027** | **0.3564** | **0.4732** | **0.6543** |
| ReTargetMap独立Borda | 0.4124 | 0.3001 | 0.4791 | 0.6196 |
| 含DTIAM统一系统 | **0.4377** | 0.3346 | 0.4610 | 0.6576 |

新方向头相对DTIAM各项点估计领先，但药物级CI跨0；KiRHub也已被历史迭代查看，只能称post-audit。

## 10. 实际应用时的决策规则

| 应用问题 | 应使用的系统 | 应报告的输出 |
|---|---|---|
| 给定老药建立通用靶点谱 | 冻结通用独立主系统；新方向头作候选辅助 | 当前rank/384；扩展空间按路由rank |
| 给定靶点找老药 | 反向辅助头 | rank/720，明确标辅助方向 |
| 验证某个Top-K pair是否物理合理 | pair-specific结构复核 | pose、接触、结构置信度，不改写为主模型概率 |
| 设计湿实验 | 综合模型、数据库和结构证据 | Top-K、阳性/阴性控制、路由匹配的实验endpoint |

正式候选表每一行至少应带：

```text
drug identity
target identity
query direction
candidate denominator / route
rank and ensemble spread
drug/target warmth
exact-pair known or unreported
training/selection involvement
external evidence and year
structure/pocket availability
recommended assay endpoint
```

## 11. 对外汇报的推荐指标层级

主文只保留三类数字：

1. 冻结S5：`drug→target / observed-pair / drug-macro AUPRC`，说明与DTIAM的同数据比较；
2. 密集时间检索或完整候选：`drug→target / rank或Recall@K / 候选分母`，说明实际检索能力；
3. KIRHub：`drug→target / measured-subset / 1 μM functional inhibition`，说明跨endpoint回顾性结果。

Micro AUPRC、target-macro、反向rank、亲和力Spearman、校准指标和历史S1–S4放入技术附录。任何指标名称都必须同时带方向、候选分母/实测子集、切分和endpoint。

## 12. 当前声明边界

可以说：

- ReTargetMap已经形成“共享pair backbone + drug→target主头 + target→drug辅助头”的双向系统；
- 冻结S5中，ReTargetMap独立系统的drug-macro AUPRC点估计高于DTIAM；
- 新方向头在KiRHub严格回顾性集合上的AUPRC和Recall点估计高于DTIAM；
- 745个靶点已具备模型输入并完成全覆盖评分，但生产结果按实验路由分开解释。

不能说：

- 已经统计学显著、全面超过DTIAM；
- KIRHub验证了Ki/Kd亲和力；
- 当前已有校准完成的统一rank/450或rank/745；
- FULL_FIT覆盖表现是独立测试；
- 未报告pair是阴性；
- 19维口袋上下文等于drug–pocket pair-specific结构证据。

## 13. 权威入口

- 冻结机器合同：`configs/biomaster_current_contract_v1.json`
- 冻结项目口径：`docs/CURRENT_PROJECT_CONTRACT_ZH.md`
- 训练与标签：`docs/BIOMASTER_TRAINING_DATA_LABEL_DEFINITION_ZH.md`
- 新重训审计：`docs/BIOMASTER_RETRAIN_RETEST_20260901_ZH.md`
- 药物中心与DTIAM：`docs/BIOMASTER_DRUG_TO_TARGET_VS_DTIAM_20260827_ZH.md`
- 新方向三种子测试：`outputs/retrain_20260901/bidirectional_stage_a_rows4/ENSEMBLE_STAGE_A_SUMMARY_V6.json`
- 745分层时间测试：`outputs/retrain_20260901/bidirectional_720x745_stage_a_rows4/STAGE_A_TEMPORAL_2024_2025_EXPANDED_SCOPES_SUMMARY_V1.json`
- 新药物中心汇总：`outputs/retrain_20260901/drug_centric_ranker_rows4_v1/BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json`

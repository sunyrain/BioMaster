# BioMaster-EQIR V3 候选机制预注册

状态：**只冻结 Stage 0 结构增量可用性筛查；尚不是经验证的算法创新。**

## 要解决的不是“换一个分类器”

当前真正未解决的问题是：结构证据只覆盖部分靶点，而且覆盖与研究热度、口袋质量和历史活性证据有关；简单拼接 docking 分数既会泄漏，也会让证据丰富靶点支配模型。V3 候选机制因此定义为 Evidence-Qualified Incremental Residual（EQIR）：

\[
s(d,t)=b(d,t)+a(t)\,r(d,t)
\]

- \(b(d,t)\)：在全部可用靶点上训练、交叉拟合得到的序列–化学基础 logit。
- \(a(t)\)：只由与候选 pair 标签无关的受体/口袋质控决定的资格变量。
- \(r(d,t)\)：结构证据对交叉拟合基础残差的增量解释，不允许重新学习完整标签。
- 当结构缺失或资格不通过时，\(a(t)=0\)，最终函数**精确等于**基础模型，而不是用零填充或隐式插补得到近似回退。

这个整体机制是候选创新假设；listwise loss、双向检索、置信度融合、pocket encoder、MNAR 或 LUPI 各自都已有先例，不能单独作为创新点。

在进入 EQIR 前，现有 V1 已完成查询捷径审计：S2/S3 的老药查询方向未超过保留药物先验的组内置换零分布。随后在 S3 开发折 0–2 上交叉拟合靶点均值中和，drug-macro AUPRC 从 0.62209 变为 0.61829，95% CI [−0.04890, 0.04092]，仍未通过老药方向置换门槛。因此 EQIR 不能以简单中心化或重新校准代替新增的 pair-level 结构残差信号。

同配对 DTIAM 对照进一步证明这不是协议不可评价：在五折相同 17,732 对上，DTIAM drug-macro AUPRC 0.65971 超过其组内置换上界 0.64159，双查询方向均通过；BioMaster stack 为 0.63276，未超过 0.63819。完整 EQIR 因此不仅要超过自身置换零分布，还要在开发折超过 DTIAM 的 drug-macro，并以按药物配对 bootstrap 下界大于 0 作为进入确认折的条件。

为避免把简单集成误称为新算法，另冻结一个更强的非创新下限：在 S3 开发折 0–2 上，每个 held-out 折只用另外两折选择 DTIAM 凸组合权重，得到 micro 0.66747、target-macro 0.69526、drug-macro 0.65706，两个查询方向均通过置换门槛。该交叉折融合的三个点估计均高于 DTIAM 和 BioMaster 单模型，但部分成对 bootstrap 区间仍跨 0；因此只作为性能基线。完整 EQIR 必须在预注册统计门槛下超过该融合，而不能以“超过 BioMaster V1”作为充分条件。

S2 同源冷靶点的完整五折结果又给出跨协议约束：DTIAM micro-AUPRC 0.81447 显著高于 BioMaster stack 的 0.78951（BioMaster−DTIAM 的同源簇 bootstrap 95% CI [−0.03734, −0.01435]）。validation-only 同数据融合虽把 micro/target-macro 提到 0.81632/0.77980，却把 drug-macro 从 DTIAM 的 0.57869 降到 0.55963；而其 micro 增益的同源簇 CI [−0.00443, 0.00809] 仍跨 0。因此完整 EQIR 的选择目标和晋级门槛必须同时保留 micro、target-macro 与 drug-macro；任何以老药查询方向退化换取 pooled 指标上升的版本均失败。

更严格的 S2 组内置换显示，DTIAM 的 drug-macro 0.57869 未超过药物内置换上界 0.63785，BioMaster 的 0.56358 也未超过 0.63989；两者只在靶点查询方向有超越组先验的 pair 信号。EQIR 因而不能只把 DTIAM 当作 pooled 分数参考，还必须把“老药内候选靶点排序超过置换上界”设成独立必要条件。

开发折 0–2 的交叉折查询平衡凸组合进一步排除了简单模型选择：其 DTIAM 权重已达 0.875–0.95，micro 相对 DTIAM 的靶点簇 CI 为 [0.00222, 0.00542]，但 drug-macro 显著下降 −0.00908，药物配对 CI [−0.01718, −0.00140]，且老药查询置换门槛仍失败。故 EQIR 必须提供 pair-level 新增信息或训练期方向约束，不能只对已有两个分数重标定。

S4/S5 的补充审计把这一要求推广到固定时间外和老药实体冷启动测试。S4 routed fusion 虽把 micro 从 DTIAM 的 0.79620 提到 0.81952，drug-macro 却由 0.83333 降到 0.80862；药物配对差的 CI [−0.08416, 0.03606]。S5 routed fusion 相对五种子 BioMaster 的 target-macro 增益显著，drug-macro 增益 +0.01350 的药物配对 CI [−0.00012, 0.03232] 仍触及 0。两协议中各模型均通过查询内置换信号门槛，说明“有 pair-level 信号”和“新机制在该方向显著增量”是两个不同判据；完整 EQIR 必须满足后者，不能仅因两个方向各自超过随机零分布而晋级。

## Stage 0：先证伪结构是否有增量

原始标签盲队列含 18,230 个 pair、290 个通过实验受体 redocking 的严格主线靶点。为保证 docking 对象与最终 720 个老药的化学适用域一致，又使用完全标签盲的老药包络（重原子数 ≤50、分子量 ≤700、可旋转键 ≤15）筛选；最终可计算队列为 17,241 个有效 pair、289 个靶点，另有 7 个确定性构象生成失败。每个靶点按 `SHA256(namespace|target|molecule)` 取最多 64 对，少于 16 对的靶点不进入队列。选样和适用域筛选均不读取二分类标签、pChEMBL、Gate A 或 Gate B；标签只在独立 evaluation ledger 中于选样完成后附加。

GNINA 输出后，原始分数只在同一靶点的标签盲候选集合内转成 ECDF rank/稳健分位数，不跨靶点比较。Stage 0 仅在 S3 折 0–2 上进行：在 validation role 调一个非负标量增量权重，在对应 test role 上一次性评估。折 3–4 保持确认锁定。基于独立 S2 结果、且在任何 GNINA 分数与标签合并前冻结的方向门槛修正，将权重选择目标改为 validation micro、target-macro、drug-macro AUPRC 的等权平均；不再允许只按 pooled AUPRC 选择结构增量。

全量任务运行期间已执行一次独立、无标签的提取烟测：截至烟测时完成的 27 个靶点产生 1,057 个唯一 pair 和 5,285 个 pose；三个靶点内 ECDF 字段均在 [0,1]，输出不含 binary label、pChEMBL、Gate A/B 或 control class，score 文件 SHA256 为 `fe09e0334867fc7eb9dd945c187f7e44d84f66dea3c2316b886297f689263c4a`。该烟测只验证解析与归一化管线，不接 evaluation ledger、不拟合 alpha、不生成 Stage 0 决策；正式评估仍锁定到 17,241 对全队列完成。

Stage 0 现在要求全部门槛同时通过：至少 4/5 种子 pooled 增益为正；五种子集成 pooled 增益为正；靶点同源簇、药物骨架两个 grouped bootstrap 下界均大于 0；target-macro 点增益非负且 95% CI 下界不低于预先冻结的 −0.002 非劣界；drug-macro 增益的药物分组 CI 下界大于 0；最终 drug-macro 还必须超过药物内分数置换零分布的 95% 上界。任一失败即停止结构路线，不训练完整 EQIR。Stage 0 本身无论通过与否都不是算法创新证明。

## Stage 1：只有 Stage 0 通过才训练

为避免看到 Stage 0 结果后再选择有利架构，条件式 Stage 1 已在任何正式结构分数–标签合并前冻结到 `configs/biomaster_eqir_v3_stage1_method_freeze_20260814.json`（SHA256 `ebb3255d0ea445d2c1220b61af1f3fabfce7af38ff4aa4b8740f669d2403cdd1`）。它不授权提前训练：只有 Stage 0 的全部修正门槛通过才激活。冻结的基础 offset 是 S3 查询平衡 DTIAM–BioMaster 非结构凸组合，而不是较弱的 BioMaster-only 分数。资格 `a(t)` 是由受体、redocking、文件一致性和完整提取状态决定的确定性二值量，不从 pair 标签学习；当 `a(t)=0` 时最终 logit 必须与基础 logit 逐位相同。完整目标同时包含增量残差、靶点查询与老药查询的组内排序项，搜索网格、选择约束、五种子、八个消融和停止门槛均已事前冻结。

在仍无正式 score–label join、无 Stage 0 裁决时，又因量纲审计和先例查重冻结了目标函数修正 `configs/biomaster_eqir_v3_stage1_objective_amendment_20260814.json`（SHA256 `b2c82e995d00bf09991633824197889656b50d0a4eaa297de556b4237388bdc4`）。原始 `u = y − p₀` 是概率残差，不能直接与 raw logit 增量比较；修正后令 `Δθ = σ(b + a·rθ) − σ(b)`，以 `Huber(u, Δθ)` 训练，目标与预测都在概率增量尺度。Orthogonal Sequential Fusion 与 2017 年残差多模态融合又排除了“orthogonal fusion”或“residual fusion”单独作为新颖性；有效候选表述只保留完整的 evidence-qualified incremental-value 机制，不作 Neyman-orthogonal 或残差融合首创声明。

随后数学审计证明 `(y−p₀)−(pθ−p₀)=y−pθ`，所以概率残差 Huber 与直接最终概率误差逐样本等价。为避免把代数改写包装成算法，在正式结构 score–label join 仍为 0 时冻结了 `configs/biomaster_eqir_v3_safe_dual_query_objective_amendment_20260814.json`（SHA256 `0f0b293c0584e9c2dc6b91716420b9d43b3811584d9ab8308c48cd0bd6a20f04`）。有效 full objective 改为：在每个靶点查询组和老药查询组内，计算最终模型相对冻结基础模型的 pairwise logistic 后悔值，并分别约束其最差 20% 查询组的 `CVaR_0.8 ≤ 0.01`。普通 pairwise ranking、CVaR、safe transfer 和 baseline regret 本身都不是新颖性；候选仍只允许以标签独立结构资格、冻结跨折基础、双查询基础相对上尾风险约束和无资格精确回退的联合机制接受证伪。完整数学边界见 `EQIR_V3_SAFE_DUAL_QUERY_NOVELTY_AUDIT_20260814.md`。

候选机制与先例的可证伪差异已冻结到 `EQIR_V3_PRIOR_ART_DIFFERENTIAL_CLAIMS_V5_20260814.{csv,json,md}`，覆盖 18 类直接相邻概念；V5 在 safe transfer、group no-harm、baseline regret、CVaR 和 MFDR-DTI 多源重组边界之外，又加入 AD-LSF 对非对称动态门控、各向异性/动态分解、潜在信号协调和双向交互对齐的直接碰撞。19 篇主要论文的训练数据、切分方式、结果和同数据可比性最初冻结到 V3，随后在不改写历史版本的前提下，按代码、释放 split、训练源重叠和实现语义审计依次生成 V4–V12。GADFDTI/GATv2-TransDTI 已覆盖细粒度原子–残基交互和门控残差融合；DrugCMF 直接覆盖置信度融合；TAPB 直接覆盖靶点先验干预；MFDR-DTI 直接覆盖普通多视图重组与辅助分支优化；AD-LSF 直接覆盖更复杂的非对称门控和潜在空间交互。因此这些模块均不计入 EQIR 新颖性。EQIR 只允许以“标签独立的证据资格 + 冻结跨折基础 + 双查询基础相对上尾风险约束 + 无资格时精确函数回退”这一不可拆分整体接受消融检验。公共 checkpoint 还必须按实际训练源给出与 86,674 冻结 pair 的规范化重叠及 source-disjoint 子集，不能替代同数据 retrain。

完整 EQIR 冻结基础模型的交叉拟合输出，只学习结构残差，并同时评价：

1. 靶点查询下的老药排序增量；
2. 老药查询下的新靶点排序增量；
3. 无结构/未合格靶点上的精确回退一致性；
4. 骨架冷启动和同源冷启动两个轴上的稳定性。

必要消融包括 base only、冻结的查询平衡凸组合、无资格门控、无残差化、无精确回退约束、去掉两个检索方向损失以及 full EQIR。每个版本至少五个种子；至少 4/5 种子同向，且两个聚类轴 bootstrap 下界均大于 0，才允许触碰 S3 折 3–4。

安全双查询目标另要求运行普通双查询目标、只约束靶点查询上尾、只约束老药查询上尾、无资格、无精确回退和 full safe-dual-query 六个目标消融。若任何 held fold 在冻结内层 validation 上没有同时满足两个 CVaR 约束的候选，该 fold 返回冻结基础模型并把 full objective 记录为不可行；不得看到结果后放宽 0.01 容差。

为防止只与弱基线比较，最近方法控制已独立冻结到 `configs/biomaster_eqir_v3_nearest_method_comparator_freeze_20260814.json`。Stage 0 若通过，必须在相同开发折与五种子下加入两个 DrugCMF-TCP 启发控制：一个只在 `a(t)=1` 的同一结构合格子集上比较，另一个在全部靶点上使用结构缺失 token，但允许非结构分支重新学习、没有 bitwise 回退。它们分别检验“直接置信度融合”和“通用缺失模态融合”能否解释 EQIR 的增益；均须标注为适配控制而非官方 DrugCMF 复现。完整 EQIR 未同时超过这两个最近控制时，不得以资格/回退联合机制主张算法贡献。

官方代码语义核验后又在仍无正式结构 score–label join 时冻结实现附录 `configs/biomaster_eqir_v3_nearest_method_implementation_amendment_20260814.json`。由于发布 TCP confidence 是无边界线性标量而非 sigmoid，合格子集和全靶点控制都必须同时跑 raw-confidence 代码语义版与 sigmoid-bounded 稳定版；主公平比较统一在 optimizer step 前做梯度裁剪，发布的 step 后裁剪只作诊断。这样既不通过偷偷修复竞争方法获得优势，也不让发布实现中的无效裁剪混淆架构比较。

TAPB 的官方代码和全部 20 个释放 CSV 也已固定审计。其 cluster 路径复用 `target_test` 做 validation/test，Davis 存在大量跨 split 重复和标签冲突；六个释放训练文件并集与冻结 BioMaster 重合 1,713 个规范化 pair，S5 受影响 136/2,556。因此 TAPB headline 不是同数据对照。若 Stage 0 通过，按 `configs/biomaster_eqir_v3_tapb_control_freeze_20260814.json`（SHA256 `de03acd7126d238c2c5013cadf62db969c465bac4d52e327e2f644fa97a72523`）同时运行直接冻结嵌入适配器和仅由当前 TRAIN 构建 target prototype/prior 的干预控制；二者均是 TAPB-inspired、容量匹配的 shortcut controls，不是官方复现或因果效应估计。完整 EQIR 未超过二者时，不得把靶点先验抗性写成新机制贡献。

MFDR-DTI 的官方 commit、三套释放数据和运行时语义也已固定审计。其仓库按发布状态有语法错误、绝对本地资源依赖和缺失 KIBA；实际代码复用 warm random 固定测试集，存在跨 split 反标签 pair，且全数据描述符在切分前统一标准化。长度 1 的 shared attention 经最小运行验证对 query 不敏感、WQ/WK 梯度为 0，DWLoss 参数又没有进入 optimizer。实现缺陷不改变论文概念已经构成先例，但排除了直接复用 headline 或官方代码作公平数值基线。若 Stage 0 通过，按 `configs/biomaster_eqir_v3_mfdr_control_freeze_20260814.json`（SHA256 `864d53ed1dbe5e11bdc5ba1ea9dd22888bcb9f19c57b7298f2695e93610f30dc`）运行容量匹配的 direct multiview fusion 和 multibranch auxiliary supervision；完整 EQIR 未同时超过二者时，任何增益只能解释成普通多源融合，不能写成资格/回退/双查询风险机制贡献。

AD-LSF 的官方 commit、四套原始数据和 24 个 released cold-split role 也已固定审计。论文 headline 来自 warm random pair split，但 exact random split、五次重复 seeds、嵌入和权重未释放，模型又导入缺失的 `Fusion1.py`。Human/C. elegans 释放文件分别包含 731/1,234 个重复阳性，部分 cold valid/test 出现重复 pair；隔离运行还证明 positional encoding 按 batch slot 而非 token 位置工作。该实现不能直接作官方数值对照，但其非对称门控、潜在信号协调和双向对齐概念已经构成先例。若 Stage 0 通过，按 `configs/biomaster_eqir_v3_adlsf_control_freeze_20260814.json`（SHA256 `5d7ec56a78e506732081abadf2e51913536160a9a5a1f4c652b86030079b72ab`）运行 corrected、容量匹配的 asymmetric gated fusion 和 bidirectional alignment fusion；完整 EQIR 未同时超过二者时，不得把效果归因于资格/精确回退/双查询风险机制。

老药查询方向另设捷径门槛：完整 EQIR 的 drug-macro AUPRC 必须超过在每个药物内打乱候选靶点顺序、但保留药物分数分布的 95% 置换上界；否则即使 micro-AUPRC 上升也不得晋级。

## 严格禁令

- 不使用全局 GNINA 正负控制 membership 训练；它与 86,674 对基准精确重合 6,251 对，并污染全部 17 个 S1–S5 测试折。
- 不使用由当前评估折活性标签推导的 Gate A/B 决定测试靶点是否启用结构分支。
- 不把 raw GNINA score 当跨靶点可比特征。
- 不把缺结构、Gate fail 或 46 个预测口袋找回靶点编码为阴性。
- 不恢复 480 个硬门槛淘汰靶点，也不恢复后续口袋质量淘汰的 24 个靶点。

## 当前可声称与不可声称

当前可以声称：已建立严格的老药中心/靶点中心双方向任务、冷启动评估、靶点证据路由和标签盲结构增量验证协议。

当前不可声称：已经有新的 SOTA 模型算法。只有 EQIR 通过 Stage 0、五种子消融、双轴 bootstrap、同数据 DTIAM/BioMaster V1 对照和锁定确认折后，才可能升级为算法贡献；否则保持 `NO_FULL_SOTA_CLAIM`。

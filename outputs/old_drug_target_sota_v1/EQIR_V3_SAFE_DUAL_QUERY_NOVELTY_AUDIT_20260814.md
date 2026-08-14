# EQIR V3 安全双查询目标数学与新颖性审计（2026-08-14）

## 裁决

概率残差 Huber 不构成新的统计目标。令基础概率为 \(p_0\)，最终概率为 \(p_\theta\)，则

\[
(y-p_0)-(p_\theta-p_0)=y-p_\theta.
\]

因此，原概率增量误差和直接拟合最终概率误差逐样本等价。它可以保留为诊断，但不能作为算法创新。

为解决这一问题，条件式 Stage 1 的有效候选目标改为**按查询聚合、相对冻结基础模型定义的排序后悔值及其上尾风险约束**。该目标仍没有被结果验证；它只是在正式结构分数与标签合并文件为 0、Stage 0 尚无裁决时冻结的候选机制。

## 有效预测函数

冻结基础 logit 为 \(b(d,t)\)，标签独立的证据资格为 \(a(t)\in\{0,1\}\)，结构适配器为 \(r_\theta(d,t)\)：

\[
s_\theta(d,t)=b(d,t)+a(t)\operatorname{clip}(r_\theta(d,t),-4,4).
\]

当 \(a(t)=0\) 时，\(s_\theta\) 必须与 \(b\) 在 sigmoid 前逐位一致。这是函数不变量，不是学习到的近似行为。

## 双查询基础相对后悔值

对同一靶点内的阳性药物–阴性药物 pair，分别定义基础和最终排序 margin：

\[
m^0_T=b(d^+,t)-b(d^-,t),\qquad
m^\theta_T=s_\theta(d^+,t)-s_\theta(d^-,t).
\]

令 \(\ell(m)=\operatorname{softplus}(-m)\)。对一个靶点查询组：

\[
R_T(t)=\mathbb E_{(d^+,d^-)\mid t}
[\ell(m^\theta_T)-\ell(m^0_T)].
\]

同理，对同一老药内的阳性靶点–阴性靶点 pair：

\[
R_D(d)=\mathbb E_{(t^+,t^-)\mid d}
[\ell(m^\theta_D)-\ell(m^0_D)].
\]

负值表示结构适配器相对冻结基础模型改善组内排序；正值表示伤害。

## 为什么它不再只是“减去一个常数”

若只优化所有 pair 的平均 \(\ell(m^\theta)-\ell(m^0)\)，基础损失项对参数而言是常数，梯度与普通 pairwise logistic loss 相同。这种写法仍不能算新的目标。

有效候选先在每个查询组内聚合相对后悔值，再约束查询组之间的上尾风险：

\[
\operatorname{CVaR}_{0.8}(R)
=\min_\eta\left[\eta+5\,\mathbb E(R-\eta)_+\right].
\]

分别要求最差 20% 靶点查询组和最差 20% 老药查询组的相对后悔值不超过冻结容差 0.01。因为每个查询组的基础损失不同，基础项会改变哪些组进入相对后悔值的上尾；这不等价于对最终绝对损失直接做 CVaR，也不等价于普通 pairwise loss 加常数。

## 有效约束问题

在合格结构 pair 上优化：

\[
\min_\theta L_{BCE}
+\lambda_T\overline{\ell_T^\theta}
+\lambda_D\overline{\ell_D^\theta}
+10^{-4}\lVert\theta\rVert_2^2,
\]

满足：

\[
\operatorname{CVaR}_{0.8}\{R_T(t)\}\le 0.01,
\qquad
\operatorname{CVaR}_{0.8}\{R_D(d)\}\le 0.01.
\]

采用投影 primal–dual 优化。若冻结的内层 validation 中没有候选同时可行，该 held fold 必须返回冻结基础模型，并把完整候选记录为不可行；不得事后放宽容差。

## 不能单独声称的新颖性

- pairwise/listwise 排序已有 BRDTI、NeuRank 等直接先例。
- CVaR、GroupDRO 和约束优化都是通用既有方法。
- safe transfer、baseline regret 和 group no-harm 都有跨领域先例。
- 结构质量门控、缺失模态处理、置信度融合和残差适配均有直接先例。

因此不能声称“首次使用 CVaR”“首次安全迁移”“首次双向排序”或“首次残差融合”。

## 仍可接受证伪的联合候选

候选贡献只允许表述为以下整体：

> 在标签独立、实验结构资格限定下，对冻结的交叉拟合 DTI 基础函数学习结构适配器；适配器在无资格区域精确退回基础函数，并以靶点查询和老药查询的基础相对组后悔值上尾约束共同限制负迁移。

即使这一组合在文字上与现有 DTI 方法存在差异，也仍不等于实证算法创新。它必须同时超过：

1. 冻结查询平衡基础模型；
2. 普通双查询目标、无资格和无精确回退消融；
3. DrugCMF/TCP-inspired 直接置信度与缺失模态融合控制；
4. TAPB-inspired 直接适配器和训练集限定的靶点先验干预控制；
5. 同数据 DTIAM 与 BioMaster V1。

并且必须通过五种子一致性、两个冷启动聚类轴 bootstrap、target-macro 非劣、drug-macro 显著增益和老药内置换门槛，之后才可打开锁定的 S3 折 3–4。

## 证据边界

该约束只能提供冻结验证协议下的经验性无伤害控制，不能被写成任意未来分布上的理论保证。Stage 0 或任何后续门槛失败，都应停止该路线并保持 `NO_FULL_SOTA_CLAIM`。

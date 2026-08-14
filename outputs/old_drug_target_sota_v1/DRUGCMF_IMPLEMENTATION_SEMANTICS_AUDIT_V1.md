# DrugCMF 实现语义审计（2026-08-14）

## 裁决

官方实现可作为先例和代码忠实控制，但需要把论文概念、发布代码语义与公平的数值比较分开。

- TCP confidence 头最后是线性标量，没有 sigmoid、clamp 或 softmax；该原始标量直接乘到模态特征上，同时用 MSE 拟合正确类别概率。
- 发布代码先执行 `optimizer.step()`，再执行梯度裁剪，因此裁剪不会改变刚刚已经应用的参数更新。
- `list >= numpy.float64` 通过 NumPy 反向比较可以执行，不是错误；此前的静态怀疑已被最小运行验证否定。
- 发布代码的 `sensitivity` 实际计算负类召回/TNR，`specificity` 实际计算正类召回/TPR，变量名相反；这不影响 AUROC/AUPRC，但会影响阈值指标的解释。

EQIR 的最近方法控制因此同时保留 raw-confidence 代码语义版和 sigmoid-bounded 稳定版。主公平比较统一在 `optimizer.step()` 前裁剪梯度；发布顺序只作诊断。两者仍只能称为 DrugCMF-inspired 控制，不能称为官方四模态复现。

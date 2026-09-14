# 新 A＋训练内近邻／靶点先验

已完成固定 A 神经模型的四组评分对照。中文结果与使用边界见 [实跑报告](../../docs/BIOMASTER_PALINOVA_A_TRAIN_ONLY_BLEND_20260914_ZH.md)。

训练参照来自 A TRAIN 的 337,570 对；权重和阈值仅由 VALIDATION 选择。三项组合在共同 Kd/Ki TEST 的药物内 AP 为 0.7145，明确失活 FPR 为 4.71%；无训练阳性参照层的阳性召回却为 0%。请连同分层结果阅读，不能据总指标淘汰探索候选。

| 文件 | 内容 |
|---|---|
| `PROTOCOL.json` | 分量、选择与评估协议，输入／权重／代码哈希 |
| `TRAIN_REFERENCE.json`、`TRAIN_TARGET_PRIORS.csv` | 训练参照库规模、每个精确序列的实测计数与平滑先验 |
| `SELECTION.json`、`VALIDATION_WEIGHT_SEARCH.csv` | 仅验证集选择的标准化、四组权重、校准与阈值；42 组权重搜索 |
| `METRICS.csv`、`QUERY_METRICS.csv.gz` | 总体及药物／靶点查询指标 |
| `PAIRED_INTERVALS.csv` | 固定预测下的配对 bootstrap 差值区间 |
| `PREVIOUS_PANEL_BLEND_PREDICTIONS.csv.gz` | 原 S5、合并证据面板逐对评分及 A/B 共同未见标记 |
| `FROZEN384_BLEND_RANK_DIAGNOSTIC.csv` | 原 384 在目录药物内排名的位置，非重选清单 |
| `SUPPORT_STRATIFIED_DIAGNOSTIC.csv` | 固定阈值下的训练支持分层，含召回、误报与原始计数 |
| `FROZEN384_SUPPORT_DIAGNOSTIC.csv`、`FROZEN384_SUPPORT_SUMMARY.csv` | 原 384 的训练支持与排名，逐对及分组表 |
| `SUPPORT_DIAGNOSTIC.json` | 事后分层口径与来源哈希；0.3 不是部署分界 |
| `SUMMARY.json`、`STATUS.json` | 运行结论、验证不变量与完成状态 |
| `VALIDATION_NEURAL_SCORES.parquet`、`VALIDATION_BLEND_SCORES.parquet` | 本地验证分数 |
| `COMMON_TEST_BLEND_PREDICTIONS.parquet` | 本地共同测试预测与训练参照信息 |
| `CATALOG_BLEND_COMPONENTS_AND_RANKS.parquet` | 本地 720×384 完整组合分数及排名 |

较大 Parquet 留在本地，轻量审计和结果随代码保存。复现入口：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/evaluate_palinova_a_train_only_blend_20260914.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/audit_palinova_blend_support_20260914.py
```

复现需要原模型、训练特征和前次同任务审计的本地输入；各文件身份见 `PROTOCOL.json`。运行入口会重新生成本目录输出，复现前应另存要比较的版本。原实验表、生产模型注册表、网站均未修改。

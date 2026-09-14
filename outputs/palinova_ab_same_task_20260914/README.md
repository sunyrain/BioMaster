# Palinova A/B：原SPR评价任务复跑

冻结推荐权重在原S5、488亲和配对及其中479个BindingDB覆盖配对上实际推理。完整旧面板成绩含新A/B训练重叠；必须与共同未见子集一起报告。没有重训、重新选384或修改网站。

- `SAME_PANEL_METRICS.csv`：旧组合、A二分类、B二分类、A排序＋回归参照的同样本指标，含全部/共同未见配对/共同未见划分组三层。
- `TRAIN_VALIDATION_OVERLAP.csv`：逐面板精确及连接骨架配对重叠、标签冲突与剩余样本量。
- `PAIRED_DRUG_BOOTSTRAP.csv`：新模型减原组合的AP、AUROC差及1,000次配对药物聚类区间。
- `RANK_BAND_MEASURED_OUTCOMES.csv`：每药完整384靶点轴上的前10、11–20、21–384的已测分布，含训练未见切片；不是实验命中率。
- `CATALOG_INPUT_SENSITIVITY.csv`：同权重使用旧目录输入的敏感性，主分析优先配套训练缓存。
- `DRUG_INPUT_AUDIT.csv`、`FEATURE_AUDIT.json`：精确身份、缓存来源和新旧特征差异。
- `PROTOCOL.json`、`SUMMARY.json`：冻结权重、输入哈希、评价口径、完整结果和推理核验。
- `ALL_PANEL_PREDICTIONS_AND_OVERLAP.csv`：本地3,044个面板行的标签、分数与重叠；其gzip副本随小型结果归档。
- `CORE720x384_RECOMMENDED_SCORES.parquet`、`INPUT_*.npy`：本地矩阵评分及本次实际输入；未部署，不更换原实验分数。

三份权重均通过4,096行已发布神经A/B测试预测复现，最大差0。后台DTIAM训练及其测试门控不变。

[中文结果与解释](../../docs/BIOMASTER_PALINOVA_AB_SPR_SAME_TASK_20260914_ZH.md)。复现命令：`python scripts/evaluate_palinova_ab_same_task_20260914.py`。

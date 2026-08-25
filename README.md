# BioMaster

BioMaster 是一个面向老药新靶点发现的可复现研究仓库。当前主任务是：给定一种已上市/老药，在固定的 384 个候选靶点中排序潜在作用靶点；反向的“给定靶点排序 720 种药物”保留为辅助评估，而不是主要交付。

## 当前研究边界

- 部署空间：720 种老药 × 384 个候选靶点，共 276,480 个 pair。
- 综合训练表：437,248 条去重关系，未知关系不会被自动当作负例。
- 模型形态：共享 drug–target pair backbone，加药物→靶点主排序头和靶点→药物辅助排序头。
- 当前最可靠范围：药物和靶点实体均在训练数据中出现、但 exact relation 被留出的 double-warm 检索。
- target-cold 与 double-cold 仍是研究边界，不能作为已经解决的问题对外宣称。
- target-level pocket 上下文与候选药物无关；pair-specific pose/contact 特征只用于候选后的结构复核，尚未晋级主模型。

完整口径和汇报结构见 [完整汇报大纲](docs/BIOMASTER_COMPLETE_PPT_OUTLINE_20260825_ZH.md)。

## 仓库结构

```text
biomaster/       可复用 Python 包：基础流程、生产筛选工具和 ODTI 模型组件
scripts/         数据构建、训练、评估、部署评分与结构复核入口
tests/           核心包、正式脚本合同和 ODTI 回归测试
configs/         冻结范围、标签和评估协议
docs/            当前文档、汇报材料和历史文档索引
examples/        不依赖大数据的小型演示输入
outputs/         仅跟踪少量审计摘要；全量结果、checkpoint 和分数表默认忽略
md/              分子动力学准备与分析工具
```

大型数据库、模型权重、第三方仓库、虚拟环境和生成的全量结果不进入 Git。具体边界见 [数据访问说明](docs/DATA_ACCESS.md)。

## 安装

基础演示：

```bash
python -m pip install -e .
```

当前 ODTI 训练与测试：

```bash
python -m pip install -e '.[odti,dev]'
```

其他按需安装组：

- `workflow`：Excel 与 YAML 工作流。
- `structure`：结构生物信息与 RDKit 工具。
- `reports`：PDF/HTML 报告生成。
- `production`：上述生产与训练依赖的完整集合。

## 当前可复现入口

主线按以下顺序组织：

```text
build_biomaster_comprehensive_training_v1.py
  → train_biomaster_comprehensive_balanced_v2.py
  → train_biomaster_bidirectional_v6.py
  → refit_biomaster_bidirectional_v6_full_fit.py
  → score_biomaster_bidirectional_v6_720x384.py
  → summarize_biomaster_bidirectional_v6.py
```

冻结的药物中心基线与外部审计：

```text
train_v10_leakage_safe_ranker.py
  → evaluate_v10_leakage_safe_external.py
  → audit_drug_centric_kirhub_v1.py
```

这些文件仍保留历史版本号，以便结果溯源；面向项目的汇报统一描述为“BioMaster 双向关系检索方法”。脚本职责和入口索引见 [scripts/README.md](scripts/README.md)。

## 小型演示

```bash
python -m biomaster.cli run-demo --out outputs/demo --offline
```

演示使用 `examples/` 内的小型 CSV，不下载模型或大型数据库。

## 验证

```bash
pytest -q
python -m compileall -q biomaster scripts tests
```

测试分为三类：纯单元测试、正式脚本合同测试、依赖本地紧凑审计产物的回归测试。仓库整理时不以“测试数量少”为目标；只有对应实现被删除且已有等价覆盖时才删除测试。

## 文档与汇报

- [文档入口](docs/README.md)
- [完整汇报大纲](docs/BIOMASTER_COMPLETE_PPT_OUTLINE_20260825_ZH.md)
- [正式候选筛选流程](docs/PRODUCTION_PIPELINE_V4_ZH.md)
- [当前结果状态](docs/RESULTS_STATUS_ZH.md)
- [汇报图片与源文件](docs/presentations/README.md)
- [历史文档索引](docs/archive/README.md)

## 发布边界

本仓库用于协作开发、结果审计和复现规划，不是数据库、模型权重或候选结果的公开发布包。任何对外结果应以冻结评估摘要和当前文档口径为准，不能把 FULL_FIT coverage、开发集选择结果或结构诊断分数表述为独立前瞻验证。

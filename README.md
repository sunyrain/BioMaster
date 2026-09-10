# BioMaster

**2026-09-10 实验交付**：当前湿实验基线为 **384 个候选配对＋112 个另计对照**。[最终实验表](outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv)包含药物、旧靶点、新靶点及指定参考化合物；[FDA 对照修订记录](docs/BIOMASTER_SPR384_FINAL_FDA_CONTROLS_20260910_ZH.md)与[实验回填说明](docs/BIOMASTER_SPR_RESULTS_UPLOAD_ZH.md)用于保持身份和版本一致。候选交付不代表已测得结合。

仓库发布代码、文档和明确列出的实验/审计快照；模型权重、完整评分缓存、原始数据库和账号文件留在本地。新检出仓库需要按[数据访问说明](docs/DATA_ACCESS.md)准备运行资产。已清理历史大文件约101.6GB，[执行记录](docs/BIOMASTER_DISK_CLEANUP_COMPLETED_20260910_ZH.md)标明哪些研究分支需重建特征。

**可视化平台**：[BioMaster Discovery Atlas 使用说明](docs/BIOMASTER_EXPLORER_ZH.md)。从药物／靶点检索真实双向排名、DrugCLIP／DTIAM／ConPLex、疾病与通路、三维口袋及 SPR 实验设计。构建 `web/` 后运行 `python scripts/run_explorer.py`，访问 `http://127.0.0.1:18765`。远程 IDE 在 Ports（端口）面板中将远程 18765 转发至本机 18765；如果 IDE 分配了其他本地端口，请使用该本地端口访问。

**2026-09-06独立模型交付**：[本轮选定模型、实际架构及完整回归结论](docs/BIOMASTER_SELECTED_MODEL_20260906_ZH.md)。完成54次全局比较与24次局部消融，交付一个DrugCLIP＋Morgan＋ESM2共享网络；[模型包与推理用法](outputs/biomaster_best_model_20260906/retargetmap_selected_v1/README.md)。范围为720药×384靶点，明确区分≤2022回归权重和≤2025部署拟合；以下冻结生产合同保留原用途。

BioMaster 是一个面向老药新靶点发现的可复现研究仓库。当前主任务是：给定一种已上市/老药，在候选靶点中排序潜在作用靶点；反向的“给定靶点排序老药”仅作为辅助证据。项目唯一口径见 [当前项目合同](docs/CURRENT_PROJECT_CONTRACT_ZH.md)，机器可检验版本见 [biomaster_current_contract_v1.json](configs/biomaster_current_contract_v1.json)。

## 当前研究边界

- 当前已评分比较核心：720 种老药 × 384 个靶点，共 276,480 个 pair；下一版主生产空间为450个靶点，其中66个尚待统一评分。
- 完整745靶点只是非GPCR登记与分路空间，不是全局排序分母。
- 综合训练表：437,248 条去重关系，未知关系不会被自动当作负例。
- 模型形态：通用主系统输出不依赖DTIAM的药物中心最终排序；FULL_FIT药物→靶点方向头提供神经候选证据；靶点→药物头只作辅助。KIRHub功能头是独立激酶专项路由。
- 当前最可靠范围：药物和靶点实体均在训练数据中出现、但 exact relation 被留出的 double-warm 检索。
- target-cold 与 double-cold 仍是研究边界，不能作为已经解决的问题对外宣称。
- target-level pocket 上下文与候选药物无关；pair-specific pose/contact 特征只用于候选后的结构复核，尚未晋级主模型。

完整口径以 [当前项目合同](docs/CURRENT_PROJECT_CONTRACT_ZH.md) 为准；汇报结构另见 [完整汇报大纲](docs/BIOMASTER_COMPLETE_PPT_OUTLINE_20260825_ZH.md)。

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
- `kinase`：KIRHub激酶功能专项头的数据构建与训练依赖。
- `structure`：结构生物信息与 RDKit 工具。
- `reports`：PDF/HTML 报告生成。
- `production`：上述生产与训练依赖的完整集合。

## 当前可复现入口

通用训练与方向头复现链按以下顺序组织：

```text
build_biomaster_comprehensive_training_v1.py
  → train_biomaster_comprehensive_balanced_v2.py
  → train_biomaster_bidirectional_v6.py
  → refit_biomaster_bidirectional_v6_full_fit.py
  → score_biomaster_bidirectional_v6_720x384.py
  → summarize_biomaster_bidirectional_v6.py
```

药物中心最终排序与冻结审计：

```text
build_biomaster_drug_centric_ranker_v1.py
  → validate_current_project_contract.py
```

底层依赖仍保留历史版本号，以便结果溯源；面向项目的汇报统一描述为“BioMaster 双向关系检索方法”。脚本职责和入口索引见 [scripts/README.md](scripts/README.md)。

## 小型演示

```bash
python -m biomaster.cli run-demo --out outputs/demo --offline
```

演示使用 `examples/` 内的小型 CSV，不下载模型或大型数据库。

## 验证

```bash
pytest -q
python -m compileall -q biomaster scripts tests
python scripts/validate_current_project_contract.py
```

测试分为三类：纯单元测试、正式脚本合同测试、依赖本地紧凑审计产物的回归测试。仓库整理时不以“测试数量少”为目标；只有对应实现被删除且已有等价覆盖时才删除测试。

## 文档与汇报

- [文档入口](docs/README.md)
- [当前项目唯一口径](docs/CURRENT_PROJECT_CONTRACT_ZH.md)
- [完整汇报大纲](docs/BIOMASTER_COMPLETE_PPT_OUTLINE_20260825_ZH.md)
- [正式候选筛选流程](docs/PRODUCTION_PIPELINE_V4_ZH.md)
- [当前结果状态](docs/RESULTS_STATUS_ZH.md)
- [汇报图片与源文件](docs/presentations/README.md)
- [历史文档索引](docs/archive/README.md)

## 发布边界

本仓库用于协作开发、结果审计和复现规划，不是数据库、模型权重或候选结果的公开发布包。任何对外结果应以冻结评估摘要和当前文档口径为准，不能把 FULL_FIT coverage、开发集选择结果或结构诊断分数表述为独立前瞻验证。

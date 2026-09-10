**BioMaster V3 构建、训练与测试记录**

日期：2026-09-05。本轮已将[架构与算法方案](BIOMASTER_ARCHITECTURE_ALGORITHM_UPGRADE_20260905_ZH.md)落实为可训练代码、训练折支持缓存、独立评分入口和实验队列。正式训练从随机初始化开始，不加载可能见过本轮留出药物的历史检查点。

2026-09-05 09:32 UTC最终状态：**12/12个正式任务全部完成**，训练队列已结束。完整结果与后续方向见[最终分析报告](BIOMASTER_V3_RESULTS_AND_NEXT_DIRECTION_20260905_ZH.md)：支持集有增益，但所有神经配置的完整面板检索仍落后于阳性近邻，当前局部主干明显退化。状态依据为[队列记录](../outputs/biomaster_v3_20260905/experiments/EXPERIMENT_STATUS_V3.json)；[实验manifest](../outputs/biomaster_v3_20260905/experiments/EXPERIMENT_MANIFEST_V3.json)固定全部命令和源码hash。

**已经实现的模型与算法**

| 部件 | 实现 |
|---|---|
| V3A 支持集模型 | 正负各最多16配体，共享可训练分子投影、候选药物/靶点条件注意力、证据门控、空支持回退 |
| 全局主干训练 | Morgan + ProtBERT + pooled ESM2，低秩 FiLM 与 pair 交互共同训练；D→T 梯度进入主干 |
| V3B 局部主干 | 显式R/S、E/Z分子图，跨层保留原子×局部蛋白配对状态，局部表征直接产生主分数 |
| 支持集算法 | 只使用训练折，逐查询排除自身实体；支持集合/支持项/同骨架遮蔽 |
| 采样与目标 | 每轮无放回关系覆盖 + 药物查询流；不重复填充短查询；药物宏pairwise排序，可选hard聚合 |
| 缺失局部训练 | 20%查询一致的局部遮蔽，使全局回退路径也能获得监督 |
| 验证与恢复 | 验证集药物宏AP选择epoch；测试不参与选epoch；LAST自带最佳权重、历史、优化器和RNG |
| 独立评分 | 固定训练支持快照、固定靶点家族索引、输入标签不参与评分、保留输入行序 |

首轮训练使用已有明确二元活性标签。亲和力only记录没有强转成阴性；本轮尚未训练独立的定量亲和力/功能专项头。局部模型的序列区域及预测多口袋两种输入均已通过真实训练。正式E/F配置显式传入`local_pocket_store`：438个靶点使用最多3个去冗余P2Rank预测口袋加1个完整序列区域，375个使用完整序列区域回退，30个缺少完整ESM而走全局路径。预测口袋经过概率阈值及逐残基编号/氨基酸核验；初版没有使用实验配体接触集合。

实现位置：[支持模型](../biomaster/odti_support_v3.py)、[支持库](../biomaster/odti_support_data_v3.py)、[局部交互](../biomaster/odti_local_v3.py)、[局部特征](../biomaster/odti_local_features_v3.py)、[预测口袋](../biomaster/odti_pockets_v3.py)、[采样与实测指标](../biomaster/selectivity_training_v3.py)、[面板检索指标](../biomaster/odti_v3_panel_metrics.py)。

**正式训练数据**

协议为`COMPREHENSIVE_DRUG_ENTITY_HOLDOUT_V3_20260905`，没有时间前瞻声明。以来源InChI与标准化特征索引的连通分量作为实体，再固定哈希切分80%/10%/10%。该处理合并了52个“同一标准化特征对应不同来源InChI”的情况；源别名不会分散到不同折。

| 划分 | 实测二元关系 | 药物实体 | 同时有正负靶点的药物 |
|---|---:|---:|---:|
| 训练 | 342,031 | 233,730 | 6,450 |
| 验证 | 43,331 | 29,437 | 807 |
| 测试 | 42,032 | 28,874 | 772 |
| 合计 | 427,394 | 292,041 | 8,029 |

原始综合表437,631行。这里的差额包含无二元标签记录和实体/关系去重，不代表把所有原始记录都作为二元监督使用。

全量支持缓存共13,550,547个有效项。逐项审计结果：训练外支持0、自实体支持0、靶点/正负归属错误0、重复支持0。见[数据manifest](../outputs/biomaster_v3_20260905/data/DATA_MANIFEST_V3.json)和[全支持缓存审计](../outputs/biomaster_v3_20260905/data/SUPPORT_CACHE_AUDIT_V3.json)。完整残基缓存可用于843个索引靶点中的813个，其余走缺失回退。

**固定训练矩阵**

每个配置使用20260905、20260906两个种子，6个完整epoch，首个epoch做observed warm-up。逻辑coverage batch为512，每次交替一个真实药物query批次。局部模型使用较小microbatch；梯度按逻辑批次中的行数/药物数累积，训练曝光可按日志SHA逐轮核对。

| 配置 | 支持集 | D→T排序 | 局部主干 |
|---|---|---|---|
| A_global_pair | 无 | 无 | 无 |
| B_support_pair | 正负 | 无 | 无 |
| C_global_rank | 无 | 有 | 无 |
| D_support_rank | 正负 | 有 | 无 |
| E_local_rank | 无 | 有 | 有 |
| F_local_support_rank | 正负 | 有 | 有 |

D配置优先启动。四格全局对照用于拆分支持集和D→T目标的增量；E/F进一步检验局部主干及其与支持集的互补。所有运行均保留阳性最大Tanimoto、相同支持候选上的正负相似度logistic基线。

主开发指标是**实测正负子集的drug-macro AP**。选定检查点后另对固定64个测试药物的完整指定靶点面板报告known-positive AP、MRR、Recall@5/20、NDCG@20。面板中的未标注组合只是检索背景，不是实测阴性；这两类AP不能混称。

本轮新实体协议不同于旧S5协议，不能直接把新数字与历史0.7521相减作为升级收益。新版本也不会自动取代现行生产排序器。

**首个正式结果存档：D_support_rank，seed 20260905**

已完成6个完整epoch，每轮关系覆盖均为342,031条。验证集选中第1轮，即observed warm-up；后续加入当前权重的D→T排序没有进一步改善本轮主验证指标，因此不能把选中检查点的收益归因于排序loss。

| 同协议指标 | V3A神经模型 | 阳性最大Tanimoto | 正负相似度Logistic |
|---|---:|---:|---:|
| 实测772个双类药物的drug-macro AP | 0.9533 | 0.9209 | 0.9574 |
| 完整495靶点×64药物的known-positive AP | 0.6746 | 0.9663 | 0.8797 |
| 同一完整面板Recall@20 | 0.8844 | 0.9844 | 0.9578 |

实测AP相对阳性近邻增量为+0.03233，药物级配对bootstrap 95%区间[+0.01954,+0.04456]；相对正负Logistic为−0.00418，区间[−0.01183,+0.00335]。这些是本轮开发比较，且完整面板仍明显落后于近邻基线，不能宣布新模型已改善核心全靶点检索能力。该首个结果已被纳入全部12次运行的[最终分析报告](BIOMASTER_V3_RESULTS_AND_NEXT_DIRECTION_20260905_ZH.md)，不应单独用于决定后续模型。

证据：[首个完整训练摘要](../outputs/biomaster_v3_20260905/experiments/D_support_rank/seed_20260905/RUN_SUMMARY_V3.json)、[同支持候选的基线比较](../outputs/biomaster_v3_20260905/experiments/D_support_rank/seed_20260905/DENSE_BASELINE_COMPARISON_V3.json)。

**已完成的端到端验证**

全项目测试`python -m pytest -q`：**207 passed**；当前项目合同校验：**PASS**。最后的队列参数调整另通过对应测试。现行生产入口与旧模型检查点保持原状。

真实数据烟雾集固定选择5,000个药物实体、6,974条关系。V3A、序列区域V3B和多口袋V3B各完成两轮短训练、验证选模、测试评分和8药物完整候选面板评分。它们只验证工程链路，不能据此宣称模型能力提升。

独立CPU评分器在7条真实测试记录上与原GPU预测核对，最大logit绝对差约`7.0e-7`、门控差为0、行序一致，见[评分核验](../outputs/biomaster_v3_20260905/smoke/support/SCORE_CLI_SMOKE_COMPARISON_V3.json)。

断点恢复测试模拟：提交LAST后中断、损坏BEST及history，再仅从LAST恢复。继续训练后的全部模型张量与未中断运行逐元素一致。测试还覆盖查询不等长时的microbatch梯度等价、输入资产变化拒绝、手性、padding/排列不变、缺失输入及梯度传递。

**运行与查看**

正式全局D首个任务输出于[训练目录](../outputs/biomaster_v3_20260905/experiments/D_support_rank/seed_20260905/)。全部运行均已生成`RUN_SUMMARY_V3.json`、最佳/最近检查点、实测与完整面板预测；整个矩阵见[结果CSV](../outputs/biomaster_v3_20260905/experiments/EXPERIMENT_RESULTS_V3.csv)。

完整队列入口：

```bash
python scripts/run_biomaster_v3_experiments.py
```

此命令用于创建固定矩阵；已有队列应使用`--resume`。队列启动后会固定源码hash，发现后续源码变动将在下一任务前停止，防止不同实现悄悄混入同一对照矩阵。运行中的同一目录不要另启重复任务。

独立训练示例：

```bash
python scripts/train_biomaster_selectivity_v3.py \
  --out outputs/biomaster_v3_20260905/custom_support_run \
  --support both --rank-weight 1 --epochs 6
```

独立评分示例，输入CSV使用本轮prepared表中的canonical `drug_feature_index`和`target_feature_index`：

```bash
python scripts/score_biomaster_selectivity_v3.py \
  --checkpoint outputs/biomaster_v3_20260905/experiments/D_support_rank/seed_20260905/BEST_MODEL_V3.pt \
  --pairs /path/to/indexed_pairs.csv \
  --out outputs/biomaster_v3_20260905/custom_scores.csv
```

评分器固定复用该检查点对应的训练支持快照。它当前处理已索引特征实体，不在评分过程中隐式新增分子、重新切分数据或重新拟合模型。

# 当前推荐模型与结果入口

当前SPR结合筛选主模型为 **A二分类／种子20260921**。权重文件已经存在，推荐清单引用原文件及SHA256，没有复制为另一套权重或替换生产配置。

| 用途 | 权重 | 评分 |
|---|---|---|
| 结合筛选主模型 | [A binary](../biomaster_endpoint_multitask_20260911/kdki_inactive__binary__seed_20260921/model.pt) | 两路分类logit均值 |
| 双向排序参照 | [A joint](../biomaster_endpoint_multitask_20260911/kdki_inactive__joint__seed_20260923/model.pt) | 两路分类logit均值，回归仍为辅助 |
| IC50/EC50活性 | [B binary](../biomaster_endpoint_multitask_20260911/all_inactive__binary__seed_20260922/model.pt) | 两路分类logit均值 |

[RECOMMENDED_MODELS.json](RECOMMENDED_MODELS.json)记录每个用途的具体检查点、校准文件、验证指标、具体权重的测试成绩和三种子方案均值。`representative_test`对应共同Kd/Ki面板，其他端点见`representative_endpoint_test`；方案均值不是分数集成。

原始模型为`AuxiliaryInteraction`，不能将其架构标记直接交给只接受旧基础网络的`build_model`。在项目根目录加载如下：

```python
import json
from pathlib import Path
import torch
from biomaster.endpoint_multitask import AuxiliaryInteraction

root = Path.cwd()
registry = json.loads((root / "outputs/biomaster_model_consolidation_20260911/RECOMMENDED_MODELS.json").read_text())
selected = registry["roles"]["binding"]  # 也可选 ranking 或 activity
state = torch.load(root / selected["checkpoint"], map_location="cpu", weights_only=True)
model = AuxiliaryInteraction(state["config"], state["endpoints"])
model.load_state_dict(state["model"], strict=True)
model.eval()
# batch须先按下面的数据契约准备。
# with torch.inference_mode():
#     logits, auxiliary = model(batch)
#     score = logits.float().mean(dim=1)
```

输入batch包含`drug_global`（DrugCLIP 512维＋Morgan 2048维）、`drug_graph_mean`（40维）、`pretrained_available`、`target_global`（ESM2 1280维），沿用训练的特征生成与实体映射规则。这不是直接接收任意SMILES/基因名称的独立编码器包。

本轮完整特征缓存为`outputs/biomaster_endpoint_ablation_20260911/features/`；其索引必须来自配套训练数据及实体映射。不能将其他旧目录的`drug_feature_index`或`target_feature_index`直接套进这里。现有推理实现可参照`scripts/run_endpoint_multitask_20260911.py`中的`predict`及`scripts/run_endpoint_ablation_20260911.py`中的`Bank`。本次核验只读取一个评估批次所需的缓存行，无需重新编码全库。

表内成绩沿用GPU BF16口径；CPU或FP32推理可运行，但微小数值/并列排序变化需要分别核验。三个代表模型均重新加载并复现4,096对冻结BF16预测，最大差为0。

分数越高表示模型更支持所训练的正类，不是Kd或保证的SPR命中概率。A binary和B binary的辅助回归头未经回归监督，不可使用其数值报亲和力。配套Platt校准均在Kd/Ki验证集拟合，不能将B的该校准直接宣称为IC50/EC50专用概率校准。

完整报告：[实验总整理与模型推荐](../../docs/BIOMASTER_EXPERIMENT_CONSOLIDATION_AND_BEST_MODELS_20260911_ZH.md)。复算入口：`python scripts/consolidate_biomaster_models_20260911.py`。本次推荐未部署到网站，尚未用这些代表权重重新评分全矩阵或重新选择384。

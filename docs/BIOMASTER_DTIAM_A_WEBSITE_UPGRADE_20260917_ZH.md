# 网站DTIAM升级为九月加强版A

2026-09-17：按用户要求，只替换当前网站DTIAM评分通道；其他六模型的权重、评分定义和已保存分数保持既有版本，新模型尚未结束的目录补算继续。

## 上线模型与选择依据

- 使用已完成的九月DTIAM兼容重训 **A：Kd/Ki＋明确失活**，训练池337,570对，其中105,368对明确失活。不重新训练编码器或下游模型。
- 选择规则在本次目录推理前写入：比较三个A种子的既有公共验证双向宏AP，采用最高者；不按TEST、SPR或新378对面板成绩选择。
- 选中 **seed 20260923 / WeightedEnsemble_L2**，验证选择分0.724859；这是该种子已经冻结的query/native共同选择，不是新建三个种子的预测平均。
- 原始分类输出用于排名；既有验证Platt概率另列在导出文件，不替代排名值。两者均不是实测Kd或已验证的湿实验命中概率。
- 固定版本号：`dtiam_a_kdki_inactive_20260917`。前端简称“DTIAM A”，完整名称“DTIAM A（九月加强版）”。

## 覆盖与同步

**720药×384核心靶点=276,480对已全部推理完成。** 每药排名分母384，每靶点排名分母720。完整目录矩阵、七模型分歧、模型切换、配对详情和排名CSV统一读取新版本。

SPR冻结候选复核中的384对及BindingDB479面板也按精确配对切换到同一DTIAM A结果，重新计算分歧、排名和指标。复核CSV与排名CSV附加`dtiam_model_version`，避免导出后混淆版本。原SPR实验交付表及实验顺序保持原记录。

旧评分、原训练文件及旧版回顾性报告保留。网站缺少或损坏新A评分时，不会把旧DTIAM分数悄悄显示成A：缺失保留为空，版本、校验和或配对错误会明确报错。

## 验证

- 复用原官方BerMol768、ESM2-t33-650M1280特征，核对720分子SMILES和384完整蛋白序列一致、全部特征有限且非零、文件哈希一致。
- 与九月训练特征库的精确身份重叠检查：472分子最大差2.09e−6，373蛋白最大差0。蛋白仍沿用官方前1022残基及含EOS池化规则，74靶点发生截断；未借本次升级更改模型输入规则。
- 重新加载选中检查点，重放32个已保存验证样本，最大概率误差1.19e−7。
- 276,480对唯一、齐全、概率有限且在[0,1]；网站分数逐对等于新发布表。目录与SPR复核、CSV版本一致。
- 与升级前固定快照逐对比较：ReTargetMap正反向FP32值完全一致，DrugCLIP和ConPLex分数完全一致；三个新模型此前已完成分数仅有CSV往返的约1e−16浮点误差。未更改其模型或输出。
- 19项相关测试通过：覆盖缺失、方向、排序分母、身份合并、导出和新A文件校验；生产前端构建通过。
- 使用生产代码、数据与构建产物的浏览器核验通过：AR全720目录、DTIAM A Top10、七列矩阵、七轴分歧、药物页面、CSV和手机布局，页面脚本错误0。

## 同集回顾性结果

沿用升级前固定的七模型共同378对（103阳性、275阴性），不因新版成绩重新选择成员：

| DTIAM版本 | AP | AUROC |
|---|---:|---:|
| 历史部署版 | 0.4152 | 0.6372 |
| 本次九月A | **0.7837** | **0.9211** |

这些是同集回顾性指标，不是SPR384实测命中率，也不是新的独立外部泛化证明；面板有与既有训练关系重叠的风险。升级前[Nesso等七模型比较报告](BIOMASTER_SEVEN_MODEL_AGREEMENT_20260917_ZH.md)保留历史DTIAM口径，不能把该报告的旧版分歧数值贴到新A上。

## 文件

- [完整新A评分](../outputs/dtiam_a_catalog_20260917/DTIAM_A_720X384_SCORES.csv.gz)
- [各靶点Top10](../outputs/dtiam_a_catalog_20260917/DTIAM_A_TOP10_PER_TARGET.csv)、[各药物Top10](../outputs/dtiam_a_catalog_20260917/DTIAM_A_TOP10_PER_DRUG.csv)
- [发布清单与权重哈希](../outputs/dtiam_a_catalog_20260917/MANIFEST.json)、[验证选择记录](../outputs/dtiam_a_catalog_20260917/SELECTION.json)
- [特征检查](../outputs/dtiam_a_catalog_20260917/FEATURE_CHECK.json)、[检查点重放](../outputs/dtiam_a_catalog_20260917/REPLAY_CHECK.json)
- [升级一致性检查](../outputs/dtiam_a_catalog_20260917/UPGRADE_CHECK.json)、[浏览器检查](../outputs/dtiam_a_catalog_20260917/website_check/WEBSITE_CHECK.json)
- [同378对成绩](../outputs/dtiam_a_catalog_20260917/SAME_378_LABEL_COMPARISON.csv)

后台统一评分入口为`scripts/deploy_dtiam_a_catalog_20260917.py`，需原AutoGluon兼容环境`.venv_dtiam_compat/bin/python`。网站只加载已经发布的CSV与清单，不在用户请求中启动模型推理。

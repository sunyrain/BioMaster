# DTIAM实际用法、近期模型与最终比较矩阵

核对日期：2026-09-20。近期检索窗口为2024-09-20至2026-09-20，保留少数更早的参照模型。这是围绕可用代码、任务权重和本项目适用性的定向调查，不声称穷尽全部DTI论文。

**执行方案：五个现有官方权重通道＋DTIAM本地A版，优先接入MAMMAL pKd、BALM、EviDTI、GraphBAN，形成十模型比较计划。ReTargetMap不参加。当前已有六个通道的结果，四个新增模型尚未完成本地推理适配，不能写成“十模型已经算完”。**

完整[23项模型资源/处置矩阵](../outputs/dti_final_model_matrix_20260920/FINAL_MODEL_MATRIX.csv)包括10个主比较计划、3个条件扩展、5个研究储备、1个A训练对照、2个历史参照，以及官方DTIAM来源条目和排除的ReTargetMap。来源条目不是额外参赛模型。机器配置见[MODEL_MATRIX_v3.json](../configs/dti_official_weights_20260920/MODEL_MATRIX_v3.json)，有效比较协议见[v3](../configs/dti_official_weights_20260920/PROTOCOL_v3.json)。

## DTIAM究竟如何使用

本轮直接检查本地官方源码及其GitHub版本，不能将脚本名中的`test`理解为“加载现成预测器”：

1. `code/data_prepare.sh`划分数据、提取分子和蛋白特征。
2. `code/test.sh`调用`training_validation.py dti yamanishi_08 warm_start`等实验命令。
3. `training_validation.py`在每折先调用`TabularPredictor(...).fit(train_data=...)`，随后`predict_proba`或`predict`。DTI、DTA、MoA是不同训练任务，不是同一个通用任务权重的三个开关。
4. 公开下载的BerMol是分子编码器。所查官方README、完整文件树、Release、论文关联Zenodo和Figshare包尚未确认可直接加载的下游AutoGluon预测器。该结论限于核查资源，不断言作者没有其他未公开文件。[官方源码](https://github.com/CSUBioGroup/DTIAM/blob/main/code/training_validation.py)、[官方入口](https://github.com/CSUBioGroup/DTIAM/blob/main/code/test.sh)、[来源审计](../outputs/dti_official_weights_comparison_20260920/DTIAM_OFFICIAL_SOURCE_AUDIT.json)。

我们已有可用的本地A任务权重，因此按用户最新授权直接复用，不需要为了“再运行DTIAM”重复训练：

| 项目 | 本次采用版本 |
|---|---|
| 名称 | **DTIAM-A（本地A数据训练）**，不是“官方预训练DTIAM” |
| 分子输入 | 官方BerMol CLS，768维 |
| 蛋白输入 | ESM2-t33-650M，1280维；按原代码截取前1022残基，均值含EOS |
| 配对输入 | 先分子、后蛋白，共2048列；保留既有特征名和顺序 |
| 任务 | Kd/Ki与明确失活监督的二分类；输出原生正类概率，不是Kd数值 |
| 训练规模 | 337,570对，含105,368对明确失活；验证41,806对，测试39,507对 |
| 已冻结选模 | 三个A种子中按验证双向macro AP选择seed20260923的`WeightedEnsemble_L2`；验证指标0.724859，不能称测试成绩 |
| 运行环境 | `.venv_dtiam_compat`，AutoGluon1.4.0；论文仓库说明0.5.2，不能称完全复现论文 |
| 原有目录结果 | 已有720×384＝276,480对 |
| 本轮实际验证 | 加载完整预测器，在16对固定验证样本上复算；最大绝对差5.96×10⁻⁸，通过 |

预测器位置：`outputs/biomaster_dtiam_ab_20260912/kdki_inactive__seed_20260923/predictor`。使用`TabularPredictor.load(..., require_version_match=True)`加载完整目录，调用`predict_proba(features, model="WeightedEnsemble_L2")[1]`；不能只复制顶层ensemble文件而遗漏依赖模型。可复现入口：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv_dtiam_compat/bin/python scripts/audit_dtiam_usage_20260920.py
```

[复算记录](../outputs/dti_final_model_matrix_20260920/DTIAM_A_REPLAY.csv)与[输入、版本及源码哈希](../outputs/dti_final_model_matrix_20260920/DTIAM_USAGE_AUDIT.json)已保存。16对复算证明加载和现有输入流程一致，不是新的泛化性能实验。原目录有74个蛋白超过1022残基，属于截断适用性问题，不能只凭“都有数值”称输入完整。

DTIAM A已使用本地监督及验证选模；论文应单列其来源，并额外报告去掉它的官方权重子集。其既有训练划分对应原骨架任务，不能直接充当新的冷靶点/来源隔离训练模型。

## 十模型主比较计划

| 模型 | 配对计算/主要输入 | 比较输出 | 权重及本地状态 |
|---|---|---|---|
| ConPLex | 分子/蛋白双塔与余弦 | 相似度排序 | 官方任务权重；目录全覆盖 |
| DrugCLIP | 分子3D与口袋3D双塔 | 检索排序 | 官方六折；缺2个靶点 |
| DTIAM-A | BerMol＋ESM2，AutoGluon集成 | 二分类概率排序 | **本地A训练**；目录全覆盖，复算通过 |
| Nesso-1 | 几何配对交互、Pairformer | binder主列；IC50头另列 | 官方权重；后台补算中 |
| ProbeMatchDTI | 多路探针、token/图融合 | DTI分类 | 官方All_Model；目录全覆盖 |
| DTBind occurrence | 蛋白结构图/表面与分子图交互 | 结合发生分类 | 官方权重；后台补算中 |
| **MAMMAL pKd** | SMILES/序列的多模态Transformer | **pKd回归** | 官方任务权重远端确认；未本地加载 |
| **BALM** | 蛋白/分子LM双塔，亲和监督的余弦对齐 | **亲和回归/排序** | 官方BindingDB任务文件确认；未本地加载 |
| **EviDTI** | 序列＋分子2D/3D融合，证据式分类 | 分类概率＋不确定性 | 3套官方任务权重已在本地；待目录适配 |
| **GraphBAN** | 图教师知识蒸馏与双线性注意力学生 | CPI分类 | 官方案例权重包确认；待加载及依赖核验 |

十个模型覆盖不同的学习目标和信息来源，**并不都是数值亲和回归器**。仅比较rank解决了量纲问题，没有消除二分类、亲和回归、口袋检索之间的任务差异。排序相关性、实测正确性和覆盖率必须分别报告。

新增模型的依据及接入顺序：

1. **MAMMAL pKd**：2024-10发布预印本及任务权重，2026-05正式发表。作者提供SMILES＋序列输入的BindingDB pKd任务模型、推理示例及反标准化参数。本次确认`model.safetensors`约1.832 GB；优先检验原生输出。它的20亿预训练样本是多模态样本量，不是20亿条亲和标签。[论文](https://www.nature.com/articles/s44386-026-00047-4)、[任务模型卡](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd)。
2. **BALM**：2025年亲和学习方法，用蛋白和分子语言模型的共同表示预测亲和。明确指定公开的清洗BindingDB random版，模型文件约0.612 GB；先核对PEFT依赖、分数映射及序列长度限制。不能查看本地测试后改选作者其他划分权重。[论文](https://doi.org/10.1021/acs.jcim.5c02063)、[作者代码](https://github.com/meyresearch/BALM)、[指定权重](https://huggingface.co/BALM/bdb-cleaned-r-esm-lokr-chemberta-loha-cosinemse)。
3. **EviDTI**：2025年模型，补充证据式不确定性这一维度。现有权重应先按原生DrugBank任务验证，Davis/KIBA版本分开登记，不能因数据名含亲和测量就认定输出是连续亲和。[论文](https://www.nature.com/articles/s41467-025-62235-6)、[作者资源](https://zenodo.org/records/14056305)。
4. **GraphBAN**：2025年模型，将二部关系图的教师知识与配对学生模型结合。作者案例包有BindingDB、BioSNAP、KIBA三个版本，各约9.5 GB。本计划优先核对BindingDB案例版；先查依赖和原生推理，再决定下载所需文件，避免把消融模型或`result_metrics.pt`当作主模型。[作者代码](https://github.com/HamidHadipour/GraphBAN)、[官方任务权重包](https://zenodo.org/records/14813233)。

以上排序依据资源可用性和研究互补性，不是本地性能排名。Hugging Face文件大小、版本与LFS哈希及Zenodo文件信息见[远端权重元数据](../outputs/dti_final_model_matrix_20260920/TASK_WEIGHT_REMOTE_METADATA.json)。本轮只取元数据，没有下载上述大包；远端哈希不等于本地文件已经验过。

## 其余近期模型如何处置

| 模型 | 年份/来源 | 本轮结论 |
|---|---|---|
| SCOPE | [2025论文](https://www.nature.com/articles/s41467-025-66311-9) | 本地已有25个轻量版权重，但入口使用固定靶点库；先逐序列映射384靶点，作为条件扩展 |
| CheMLT-F | [2026论文](https://doi.org/10.1186/s13321-026-01199-1) | 多任务权重已确认，每版约0.385 GB；Davis/KIBA任务头、标准化和适用性待核验 |
| ADME-DTI | [2026论文](https://doi.org/10.1002/minf.70033) | 作者仓库有分类/回归任务文件；需验证基模型与元集成完整性，列条件扩展 |
| 3DICE | [2026论文](https://doi.org/10.1093/bioinformatics/btag488) | UniMol＋ESM-IF1的3D交互；有代码，但所查文件主要为特征或分析缓存，未确认下游任务权重 |
| DrugCMF | [AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39972) | 四模态及置信加权适合可靠性问题；公开编码器不等于公开任务预测器，先列研究储备 |
| GRAM-DTI | [作者代码](https://github.com/uta-smile/GRAM-DTI) | 仓库标注ICLR 2026；会议录本轮未独立核验。未确认任务权重；额外功能文本需单独审计已知关系泄漏 |
| DrugLAMP | [2024-11论文](https://doi.org/10.1093/bioinformatics/btae693) | PLM与局部融合路线；有代码但未确认任务权重，暂不增加训练队列 |
| TAPB | [2025论文](https://doi.org/10.1038/s41467-025-66915-1) | 靶点先验干预与研究问题相关；沿用A协议的有/无干预条件实验，不自动开训 |

完整表还列出更早的DrugBAN、DeepDTA和日期边界尚未核实的DTI-LM，未将它们全部包装为近两年新模型。ProbeMatchDTI、DTBind在本轮已核查代码和本地资产，但没有据仓库存在就编造期刊出版日期。新仓库的固定revision及文件清单见[源码审计](../outputs/dti_final_model_matrix_20260920/NEW_MODEL_SOURCE_AUDIT.json)。

## 已导出的720×384实际分数矩阵

为避免“模型名单完整”和“计算完成”混淆，同时导出了[276,480行、十个计划模型列的分数快照](../outputs/dti_final_model_matrix_20260920/CATALOG_720X384_SCORE_SNAPSHOT.csv.gz)。ReTargetMap列已排除；尚未接入的四列为空；缺失不填0，不当阴性。当前矩阵范围是已有的720×384目录，**不是720×888**，也没有把自定义额外靶点悄悄并入分母。

快照时间：2026-09-20 10:16 UTC。后台任务继续运行，以下数值仅对应此快照：

| 模型 | 已有分数的配对 | 目录覆盖 |
|---|---:|---:|
| ConPLex | 276,480 | 100% |
| DTIAM-A | 276,480 | 100% |
| ProbeMatchDTI | 276,480 | 100% |
| DrugCLIP | 275,040 | 99.48% |
| DTBind occurrence | 204,830 | 74.08% |
| Nesso-1 | 98,528 | 35.64% |
| MAMMAL / BALM / EviDTI / GraphBAN | 尚未计算 | 待适配，不是失败率 |

六个当前通道共同有分数的配对为75,625。这是共同覆盖数，不是已有实验真值的测试量。本轮未重算相关性或性能榜。按查询比较TopK时还须固定相同候选集合，不能拿某模型已算完的一部分与另一个模型全目录直接比。逐靶点运行状态、覆盖及文件哈希见[覆盖表](../outputs/dti_final_model_matrix_20260920/MODEL_COVERAGE.csv)和[快照清单](../outputs/dti_final_model_matrix_20260920/MANIFEST.json)。

可重新生成快照：

```bash
OPENBLAS_NUM_THREADS=1 .venvs/frontier_dti/bin/python \
  scripts/build_dti_final_matrix_20260920.py
```

## 与A受控训练的衔接

现成权重比较回答“用户现在拿来用，推荐有多可靠”；A受控重训回答“在固定数据与输入下，什么因素造成排序差异”。十模型计划不会自动要求把十个原生模型全部重训，也不意味着它们适用同一输入或相同端点。

A主线仍为45次拟合、先30次、条件15次，新增B训练0；[具体训练矩阵](../outputs/dti_reliability_A_plan_20260920/EXPERIMENT_MATRIX_A.csv)及冻结的数据成员不变。共用输入的七种对照和原生DrugBAN仍按已有协议执行。新增模型的加载验证、样例复现、长度/结构覆盖及训练暴露审计通过后，再逐个进入正式共同排名分析；不能以下载成功代替这些检查。

本轮完成的是DTIAM真实用法复核、现有预测器复算、模型资源整理、范围修订和只读分数快照。没有新开训练，没有改动后台Nesso/DTBind计算，也没有更新网站模型。

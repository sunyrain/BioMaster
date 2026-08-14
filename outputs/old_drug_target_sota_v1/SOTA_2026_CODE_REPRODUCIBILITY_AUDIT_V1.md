# 2026 SOTA 官方代码复现审计（2026-08-14）

## 裁决

DrugCMF 与 MMTF-DTI 的官方仓库均已按固定 commit 获取；这证明代码可见，不等于已经完成同数据、同切分复现。两者当前都不能直接作为 BioMaster 的数值 SOTA 证据。

## DrugCMF

- 固定 commit：`85100c59b3da6f4e5d3da624dea842f81321c74b`。
- 仓库只有数据说明，没有实际 CSV、PT 嵌入或 checkpoint；完整运行需要离线生成 MolFormer、ProtT5、Uni-Mol、SaProt 四路嵌入。
- 数据加载与模型前向默认四种嵌入全部存在，没有结构缺失时的显式函数回退；因此不能直接覆盖本项目 46 个无实验口袋找回靶点。
- 其置信度融合是 EQIR 必须面对的直接先例。全量结构完成后应分别运行“结构合格子集上的官方式融合”和“明确标注为适配版的全靶点缺失模态控制”。

## MMTF-DTI

- 固定 commit：`0e8665238f1a739aa4fdf6a9180300a727972928`。
- 仓库包含四个数据集共 12 个原始 train/validation/test CSV，但不含预处理后的图/嵌入 pickle、依赖锁或许可证文件。
- `datapre.py` 保存顺序为蛋白图、化合物图、蛋白特征、化学特征；`main.py` 按化合物图、蛋白图、化学特征、蛋白特征读取。模型又令所谓化合物图输入维数为 128、蛋白图为 72，与该语义交换相互吻合。这意味着公开训练代码实际消费的是交换后的模态语义。
- 训练端读取外部 validation 文件，但五折模型选择只在 train 文件内部做 StratifiedKFold。未经上游澄清或明确修复，不能称为官方同数据复现。

## 对 SOTA 声明的影响

论文随机切分 headline、代码仓库存在和同数据冷启动复现是三种不同证据。当前只确认了代码仓库存在与静态行为；数值比较必须等到使用冻结 S1-S5 pair、相同标签、相同开发/测试边界的移植完成后才能进入 SOTA 表。

机器可读审计：`outputs/old_drug_target_sota_v1/SOTA_2026_CODE_REPRODUCIBILITY_AUDIT_V1.json`

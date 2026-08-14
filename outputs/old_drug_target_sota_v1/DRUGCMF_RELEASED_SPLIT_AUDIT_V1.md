# DrugCMF 释放数据与切分审计（2026-08-14）

## 裁决

官方外部数据包的 19 个 CSV 已完整下载并逐行审计。DrugCMF 是置信度融合的直接先例，但其原始 headline 不能直接进入 BioMaster 的同数据 SOTA 表。

| 数据/协议 | 规模 | 关键边界 |
|---|---:|---|
| BindingDB full | 49,189 | random 为 entity-warm；cluster source/target 实体不重叠 |
| BioSNAP full | 27,380 | random train–val/train–test 有 1/3 个完全 pair 重叠；cluster 实体不重叠 |
| Human full | 6,718 | 含 731 个重复 pair |
| Human cold | 3,450/154/309 | train 与 val/test 在药物和靶点轴均零重叠，但 val–test 有 12 个完全 pair 重叠 |

## 代码级评估边界

BindingDB/BioSNAP cluster 数据本身的 source 与 target 在药物和靶点实体上都不重叠；但官方 `dataloader.py` 将同一个 `processed_target_test.csv` 同时设为 validation 和 test。训练脚本又按 validation AUROC 选 checkpoint，因此发布代码路径会用测试集合做模型选择。

## 对 BioMaster 的要求

后续 TCP 置信度融合只能作为 DrugCMF-inspired 适配控制：必须使用冻结 S1–S5、不同的 validation/test、去重 pair 和显式结构缺失处理。未生成四路官方嵌入且未按官方数据重跑前，不得称为官方 DrugCMF 复现；即便官方重跑成功，其随机/cluster headline 也不能与 S3 严格双冷结果直接比较。

# 官方DTI权重收集与校验

工作日期：2026-09-21。当前状态：**DOWNLOADING**。

本轮覆盖12个有明确官方任务包的模型系列。新增文件237/570个已完成校验；复用并重新哈希核验已有44个模型/配置/编码器文件。文件数量包含作者附属文件，不是模型数量。

DTIAM仍使用已授权的本地A版，它不计入官方任务权重系列。未找到公开下游任务权重的研究储备项保留来源缺口，不用编码器冒充任务模型。

| 模型 | 本地状态 | 文件 | 校验体积GiB |
|---|---|---:|---:|
| ADME-DTI | IN_PROGRESS | 202/532 | 1.115 |
| BALM | DOWNLOADED_HASH_VERIFIED | 4/4 | 0.579 |
| CheMLT-F | DOWNLOADED_HASH_VERIFIED | 23/23 | 1.059 |
| GraphBAN | IN_PROGRESS | 0/3 | 0.000 |
| MAMMAL | DOWNLOADED_HASH_VERIFIED | 8/8 | 1.709 |
| ConPLex | EXISTING_HASH_REVERIFIED | 2/2 | 1.580 |
| DTBind | EXISTING_HASH_REVERIFIED | 3/3 | 0.017 |
| DrugCLIP | EXISTING_HASH_REVERIFIED | 6/6 | 6.703 |
| EviDTI | EXISTING_HASH_REVERIFIED | 3/3 | 0.838 |
| ProbeMatchDTI | EXISTING_HASH_REVERIFIED | 3/3 | 1.954 |
| ScopeDTI_light | EXISTING_HASH_REVERIFIED | 25/25 | 0.210 |
| nesso | EXISTING_HASH_REVERIFIED | 2/2 | 0.154 |

新增主任务包包括MAMMAL pKd、指定BindingDB版BALM和GraphBAN；已有EviDTI三套权重复用。条件扩展收齐CheMLT-F两个任务版本及其编码器、ADME-DTI作者saved_models，已有SCOPE25套权重复用。

GraphBAN按作者发布的BindingDB、BioSNAP、KIBA三个完整案例ZIP收集。包内含逐epoch训练状态，不能把每个文件算成独立模型，也不根据本地测试选择epoch；正式使用版本仍须按作者流程及预先冻结规则核对。归档保留压缩状态，避免重复占用数十GB解压空间。

所有新增文件存放在数据盘`data/research/dti_official_weights_20260921/`；已有文件通过相对链接复用。下载采用断点续传；大文件分段下载后整文件校验。HF LFS文件核对作者SHA256，GitHub普通文件核对Git blob SHA1并计算本地SHA256，Zenodo包核对作者MD5并计算本地SHA256。

下载完成只说明指定来源文件完整到位。原生forward复现、特征/任务头选择、目录覆盖和全矩阵计算仍是下一步；本轮不新开训练。

后台收尾程序在下载进程结束后检查完整清单；若存在失败项，最多自动续传三轮，再更新本报告。实时状态见[收尾状态](../outputs/dti_official_weights_download_20260921/FINALIZATION_STATUS.json)。作者模型包中原有Davis训练版本仅作资产溯源，不表示本轮使用Davis作评估数据集。

资源来源：[MAMMAL任务模型](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd)、[BALM任务模型](https://huggingface.co/BALM/bdb-cleaned-r-esm-lokr-chemberta-loha-cosinemse)、[GraphBAN案例包](https://zenodo.org/records/14813233)、[CheMLT-F](https://huggingface.co/BoulderyBoulder/CheMLT-F)、[ADME-DTI](https://github.com/tariqshaban/adme-dti)。

登记文件：[当前完整模型资源矩阵](../outputs/dti_official_weights_download_20260921/CURRENT_MODEL_RESOURCE_MATRIX.csv)、[逐模型状态](../outputs/dti_official_weights_download_20260921/MODEL_COLLECTION_STATUS.csv)、[固定下载清单](../outputs/dti_official_weights_download_20260921/DOWNLOAD_MANIFEST.json)、[逐文件校验](../outputs/dti_official_weights_download_20260921/VERIFIED_FILES.json)、[已有资产复核](../outputs/dti_official_weights_download_20260921/EXISTING_OFFICIAL_ASSETS.csv)、[收集汇总](../outputs/dti_official_weights_download_20260921/COLLECTION_SUMMARY.json)。

当前研究先做[双向排序一致性与推荐分歧](BIOMASTER_DTI_RANKING_SCOPE_20260921_ZH.md)，不使用Davis，不判断谁对谁错。

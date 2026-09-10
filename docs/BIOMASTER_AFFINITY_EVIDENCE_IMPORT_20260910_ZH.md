# 近期亲和证据入库与项目覆盖审计（2026-09-10）

本轮完成原始数据归档、来源与端点分层、本地 SQLite 入库以及精确配对覆盖审计。未训练模型，未改变冻结384、112个另计对照或既有优先级。

**结论：当前384中，精确分子＋单一目标蛋白、通过本轮数值与注释筛选的配对覆盖为 0/384；本轮新检索来源为 0/384。** 没有找到记录不等于没有结合，更不能证明这批候选的命中率或实验质量。它说明本轮资料不能为这些新配对提供现成亲和验证。

## 数据落地

- 数据库：`data/processed/biomaster_affinity_evidence_20260910.sqlite`。这是独立的版本化实验依据库，ChEMBL37原库保持只读。
- 原始归档：`data/external/affinity_refresh_20260910/`；HiQBind及Supercharge Data6沿用此前下载的 `outputs/recent_affinity_sources_20260910/` 文件。
- BindingDB完整源文件已归档并扫描；SQLite索引保留720药的连接骨架邻域以及额外对照分子，供精确匹配和身份复核。不是把全部数百万条记录复制到SQLite。其他小型实验表保留全部测量/拟合行。
- 原始XLSX中的HTRF、DLS、磷酸化和其他辅助读数保存在 `auxiliary_observations` 或原始工作簿；不能算作Kd。
- 来源记录数不是独立实验次数：重复结构、重复进样、来源间转录、失效拟合均可能存在。

## 1. 覆盖结果

|证据范围|当前384|720×384（276,480对）|720×888（639,360对）|
|---|---:|---:|---:|
|本轮来源：任一合格数值端点|0|2320|3270|
|合并既有ChEMBL37：任一合格数值端点|0|6661|8458|
|合并后：Kd或Ki（含上下界）|0|5489|6975|
|合并后：Kd等值记录（仍需构建/方法复核）|0|997|1166|

相对本轮按相同身份和数值规则读取的既有ChEMBL37，新增覆盖 490 对（720×384），以及 892 对（720×888）。这是**证据索引的覆盖增量**，不是这些配对都刚刚发表，也不表示均为阳性。

本轮来源中，有 253 对位于完整评分空间、此前 `known_relation_excluded=False`，已导出进一步复核。该比较针对先前筛选旗标；不等于它们从未进入训练集，也不表示全部应直接排除。冻结384仍无本轮精确数值新增。

## 2. 各来源收集状态

|入库来源|测量/拟合/响应行|解释|
|---|---:|---|
|BindingDB_202609|70,585|全包＋分来源包合并去重后，项目药物邻域的Kd/Ki/IC50/EC50|
|CACHE1|5,966|单浓度响应和剂量拟合；限定LRRK2 WDR/反筛构建|
|CACHE3|761|两轮SPR；作者伪影/低结合标记保留|
|ChEMBL37_existing|59,798|既有库中720药/888靶点的数值记录，含单列待审注释|
|HiQBind_v3|31,572|结构附带既有标签；不是新实验|
|OpenBind_20260828|2,733|GCI Kd；包括作者未用于分析的记录，限定病毒构建|
|RAF_MEK_2026|1,124|kinobead log2FC；不是亲和常数|
|Supercharge_Data6|4,013|表观Kd；包括非潜在靶点拟合行，不并入纯蛋白Kd|

**GatorAffinity：** 已保存公开文件目录；元数据下载返回HTTP401，未导入。其标签来自BindingDB，也不能重复算独立新实验。未下载TB级预测结构。

## 3. BindingDB实际增量与发布文件质量

官网下载页标称3,241,782条测量。实际全包逐物理行核验为3,237,052行，其中5个空行、1个空字节异常行，正常数值首字段行3,237,046。全包MD5与官方一致（见 `BINDINGDB_MD5_AUDIT.json`），因此不能把下载页标称数当作实际导入行数。补取Articles、PubChem、ChEMBL、Patents、PDSPKi分包进行交叉合并。

合并贡献（按源记录ID＋分子/靶点/数值核心字段去重，原始文件仍完整保留）：

```json
{
  "unique_record_cores": 70516,
  "rows_contributed_by_archive": {
    "All": 70510,
    "PubChem": 4,
    "ChEMBL": 2
  },
  "same_id_different_core": {}
}
```

202607→202609仅对本地已有的Articles和PubChem两个可比快照做全子集ID核对；没有声称对全BindingDB做了两个月的实验增量比较。

|子集|旧记录ID|新记录ID|新增ID|移除ID|共同ID核心字段修订|
|---|---:|---:|---:|---:|---:|
|BindingDB_Articles|93522|93522|0|0|5444|
|PubChem|106142|104678|0|1464|4249|

修订可能来自结构标准化、身份、目标或端点更正；新增覆盖与新增实验必须区分。`BINDINGDB_CHANGED_RECORDS.csv`保留逐ID变化线索。

## 4. 匹配和证据口径

1. 分子使用完整InChIKey匹配；仅前14位相同的盐形/质子化/立体异构差异进入复核表，不进入精确覆盖。不同游离态/盐形是否可合并需另外确认。
2. 靶点使用项目登记的UniProt accession匹配。复合物成员、标注突变、限定不同结构域、仅蛋白群与裂解液信号不进入主表覆盖。未标突变不等于序列已验证；主表仍标记构建未经验证。
3. Kd、Ki、IC50、EC50、Kd_app与响应分别保存；只有合法正数和已知浓度单位才换算nM。`>`、`>=`、`<`、`<=`、约值均原样保留。Ki不是Kd，IC50/EC50不自动代表直接结合。
4. ChEMBL主表限制直接、单蛋白、高置信注释并排除data_validity_comment；其他源仍需方法复核。相同UniProt不是实验可直接移用的保证。
5. 来源间疑似同实验按精确分子/蛋白/端点/值/文献分组，仅用于提示重复；不同assay和实验条件尚未全部逐文献合并，因此不报告“独立新增实验总数”。
6. SPR、GCI及kinobead的信号和拟合适用范围不同。作者标为未采用、伪影、参考表面结合或低质量的拟合被保留用于学习质控，不能当成可靠阳性。

## 5. 对实验的实际意义

当前384仍是需要前瞻验证的新配对，不能因为这次入库就上调成功率。新增库的直接价值是补充已知关系排除、找实测参照与弱/无结合例子、识别模型曾见过的关系，并提供SPR伪影与拟合质控材料。它没有新增可用于确认384配对的直接亲和结果。

后续筛选应先核对 `NEW_RELATIONS_TO_REVIEW_IN_SCORED_SPACE.csv` 中新补充的关系，逐条区分明确亲和、功能读数、弱结合/上下界及身份更正，再更新排除旗标。现有384和优先级保留；实验进程仍以参考化合物测通和可重复结合为依据。

## 6. 文件与调用

- `SPR384_PRIORITY_WITH_AFFINITY_AUDIT.csv`：原优先级列表附覆盖审计，384行。
- `SPR384_AFFINITY_COVERAGE.csv`：逐候选计数；`SPR384_MATCHED_EVIDENCE.csv`：精确匹配原始数值明细（若0匹配则仅表头）。
- `COVERAGE_SUMMARY.csv`：来源×范围×端点分层汇总。
- `PROJECT_EXACT_NUMERIC_EVIDENCE.csv.gz`：项目全部精确数值证据；`ENDPOINT_BOUND_DISTRIBUTION.csv`：上下界分布。
- `IDENTITY_CONSTRUCT_CONTEXT_REVIEW.csv`：仅连接骨架相同、突变、构建和非亲和上下文等待审映射。
- `PAIR_INCREMENT_AUDIT.csv`及`NEW_RELATIONS_TO_REVIEW_IN_SCORED_SPACE.csv`：相对既有库及原排除旗标的增量。
- `SOURCE_IMPORT_STATUS.csv`、`INPUT_CHECKSUMS.csv`、`SUMMARY.json`及原始归档中的`DOWNLOAD_MANIFEST.json`记录来源与校验。

数据库读取接口：

```python
from biomaster.affinity_evidence import get_pair_evidence, evidence_inventory
rows = get_pair_evidence("完整InChIKey", "CHEMBL靶点ID")
# 每行保留endpoint、relation、value_nM、target_scope、quality、metadata及BindingDB assay描述。
```

重建顺序：`collect_affinity_refresh_20260910.py` → `build_affinity_evidence_refresh_20260910.py` → `audit_affinity_evidence_refresh_20260910.py` → `report_affinity_evidence_refresh_20260910.py`。脚本均在`scripts/`。数据库完整性检查通过；单位/上下界、突变/复合物边界和端点规范化共8项测试通过。

## 官方来源

- [BindingDB下载与assay映射](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp)
- [HiQBind数据](https://figshare.com/articles/dataset/BioLiP2-Opt_Dataset/27430305)
- [OpenBind原始数据](https://zenodo.org/records/22142262)
- [CACHE 1](https://cache-challenge.org/results-cache-challenge-1)；[CACHE 3](https://cache-challenge.org/results-cache-challenge-3)
- [Supercharge论文](https://www.nature.com/articles/s41586-025-09763-9)
- [RAF–MEK论文](https://www.nature.com/articles/s41589-026-02212-2)
- [GatorAffinity目录](https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-DB)

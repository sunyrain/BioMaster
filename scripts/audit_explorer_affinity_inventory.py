"""Rebuild the read-only resource inventory and atlas scope report."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.explorer_affinity import AffinityAtlas, DB
out=ROOT/'outputs/affinity_atlas_inventory_20260910';out.mkdir(parents=True,exist_ok=True)
inventory=AffinityAtlas(ROOT).inventory()
archive_rows=[]
for path in sorted((ROOT/'outputs/affinity_evidence_refresh_20260910').glob('*.stats.json')):
    d=json.loads(path.read_text());archive_rows.append(dict(archive=path.name.removesuffix('.project.jsonl.stats.json'),**d,path=str(path.relative_to(ROOT))))
inventory['bindingdb_archive_scans']=archive_rows
with (ROOT/DB).open('rb') as handle:
    inventory['database_sha256']=hashlib.file_digest(handle,'sha256').hexdigest()
(out/'INVENTORY.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
for name,rows in [('SOURCE_ENDPOINT_COUNTS',inventory['endpoint_inventory']),('RESOURCE_LEDGER',inventory['legacy']),('BINDINGDB_ARCHIVE_SCANS',archive_rows)]:
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
endpoints=Counter()
for r in inventory['endpoint_inventory']:endpoints[r['endpoint']]+=r['rows']
text=f'''# 全项目亲和资源规模与矩阵口径（2026-09-10）

本报告按本地实际归档、数据库和历史审计核对。不能给出可信的“所有来源独立实验总数”：ChEMBL、BindingDB、HiQBind、历史版本及训练子集存在重叠，尚未全部按原始文献、assay 与条件完成跨库去重。

## 当前统一证据库

- {inventory['total_rows']:,} 条测量 / 拟合 / 响应行，另有 {inventory['auxiliary_rows']:,} 条辅助观察。
- 最新质控后，在 720×888 = 639,360 对空间内，{inventory['exact_rows']:,} 条精确数值证据覆盖 **{inventory['exact_pairs']:,} 对（{100*inventory['exact_pairs']/639360:.3f}%）**。
- 其中 **{inventory['kd_ki_pairs']:,} 对**具有 Kd 或 Ki（含上下界），**{inventory['kd_equal_pairs']:,} 对**具有 Kd 等值记录，仍需实验构建与方法复核。
- 当前基线 SPR384 的旧覆盖审计为 0/384；此处质控仅进一步剔除，因此没有新增能直接验证基线候选的外来精确证据。
- 原审计 39,580 行 / 8,458 对 / 6,975 Kd或Ki对 / 1,166 Kd等值对，因后续两条来源质控决定，现变为 39,578 / 8,456 / 6,973 / 1,165。历史文件保留，不覆写。

|当前库来源|行数|
|---|---:|
'''
for source,n in inventory['source_rows'].items():text+=f'|{source}|{n:,}|\n'
text+='\n## 端点分层（当前库全部行，不是项目覆盖行）\n\n|端点|行数|\n|---|---:|\n'
for endpoint,n in sorted(endpoints.items()):text+=f'|{endpoint}|{n:,}|\n'
text+='''
Kd 与 Ki 分开；IC50 / EC50 不自动当作直接结合；Kd_app、kinobead 和单浓度响应不并入纯蛋白 Kd。包含失效拟合、重复结构与来源转录的原始行，不把这些行全部称为合格亲和实验。

## 其他已归档资源与历史子集

下表与统一证据库重叠或计数单位不同，**不可直接相加**。Davis、GtoPdb、Platinum、MdrDB 尚未按当前 720 完整 InChIKey + 888 靶点规则完成统一复核，不强行点亮主矩阵。

|资源|本地规模|计数与边界|
|---|---:|---|
'''
for r in inventory['legacy']:text+=f"|{r['name']}|{r['count']:,}|{r['note']}|\n"
text+='''
BindingDB 202607 / 202609 的 Articles、PubChem、ChEMBL、Patents、PDSPKi 与全包扫描规模逐档案见 `BINDINGDB_ARCHIVE_SCANS.csv`。物理扫描行、唯一源ID、端点拆分行不是同一计数；分包是交叉补充，不是额外独立实验。

GatorAffinity：只有文件目录归档，元数据返回 HTTP401，未入库；其结构预测及 BindingDB 转引标签不计新增实验。PBCNet / Boltz / DrugCLIP / DTIAM / ConPlex / ReTargetMap 为模型或模型结果；TxGNN 与 Open Targets 为疾病关联；TDC 安全性 / ADMET、PPI、通路与结构口袋均不当作亲和实测总数。

## 点亮矩阵

入口：`#/affinity`，左侧“亲和证据矩阵”。

- 全局使用 30×37 个汇总块覆盖 720×888；每块最多 24×24 个精确配对。概览各色带按该状态配对数 / 块内全部配对数，以平方根色阶增强：颜色强度 = sqrt(min(真实占比 / 10%, 1))，≥10% 截断为最深色；不是完成率达到100%。所有块及各状态共用该尺度，不按块内最大值归一化。悬停显示实际数量与比例。
- 点击块后显示最多 576 个精确格点，点击格点查看原数值、端点、上下界、来源与质控。支持块行列选择和药物 / 靶点名称定位。
- 覆盖视图：绿色为外部精确数值记录，黄色为基线设计待测，蓝色为已提交 SPR（待审核），灰色为无所选证据。存在来源记录与有实验计划可能重叠，不报告这些指标之和为总配对数。
- 亲和视图：单独选择 Kd 或 Ki，按 100 / 1,000 / 10,000 nM 阈值探索。满足 ≤ 阈值为绿色，高于阈值为红色；这表示相对于阈值的强弱，不是绝对结合 / 不结合。上下界不能判定为黄色；同配对有两侧证据为紫色冲突。
- 已提交 SPR 的检测响应不会自动变成已审核结合标签。未检出、无数据、弱结合与失效拟合保留区别。矩阵完全不使用预测分数点亮实测状态。
- 单一 UniProt accession 仍不代表构建序列已经验证；所有来源记录保留这一边界。被后续质控排除的记录在详情可见，但不点亮有效证据。

## 模板退役

旧 CSV 模板已停用。当前 CSV 必须使用 `SPR_RESULTS_V2`，保留全部中文表头、模板版本列并填写中文响应 / 质控选项。在线录入使用结构化表单，不受文件格式影响。已保存的历史实验结果不删除。

## 可复现产物

- `outputs/affinity_atlas_inventory_20260910/INVENTORY.json`：当前库计数、来源、后续质控、历史资源路径、数据库 SHA-256。
- `SOURCE_ENDPOINT_COUNTS.csv`：来源×端点行数；`RESOURCE_LEDGER.csv`：历史资源规模；`BINDINGDB_ARCHIVE_SCANS.csv`：逐归档扫描记录。
- 运行 `python scripts/audit_explorer_affinity_inventory.py` 可重新生成；读取实验依据库，不改变亲和标签、模型、冻结 SPR 或优先级。
'''
(ROOT/'docs/BIOMASTER_AFFINITY_ATLAS_INVENTORY_20260910_ZH.md').write_text(text)
print(json.dumps({k:inventory[k] for k in ['total_rows','auxiliary_rows','exact_rows','exact_pairs','kd_ki_pairs','kd_equal_pairs']},ensure_ascii=False))

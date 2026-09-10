import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/spr512_integrated_disease_20260909';BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 s=json.loads((OUT/'SUMMARY.json').read_text());v=json.loads((OUT/'VALIDATION.json').read_text());smoke=json.loads((OUT/'SYSTEM_SMOKE_CHECK.json').read_text());assert v['all_pass'] and smoke['counts']['disease_targets']==886 and smoke['http_ranking_status']==200
 master=ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv';parent=json.loads((BASE/'SOURCE_AND_ARTIFACT_MANIFEST.json').read_text());assert sha(master)==parent['input_sha256'][str(master.relative_to(ROOT))]
 report=ROOT/'docs/BIOMASTER_SPR512_INTEGRATED_DISEASE_RUN_20260909_ZH.md'
 report.write_text(f'''# 当前512对的TxGNN、Open Targets联合运行与系统整合

## 结论

系统能够运行、提供分子—靶点排名、生成/读取实验候选设计，并展示逐对疾病证据。**当前512对是可审查的候选设计，不是自动放行实验清单。** 本轮没有训练模型、替换原模型权重或更改原512对及角色。

512对实际包含448个待发现配对和64个阳性对照，涉及181个不同分子实体、64个靶点。此前720药全方向任务未覆盖部分目录外对照；本轮已逐一核对181实体，按完整InChIKey查询新增对照的PubChem同义名，再匹配旧图节点。

## 实际运行结果

| 项目 | 结果 |
|---|---:|
| 对照分子补跑 | 24个 × 17,080疾病 = 409,920个新分数 |
| 全部配对有可用TxGNN | 447/512 |
| 待发现配对有可用TxGNN | 402/448 |
| 阳性对照配对有可用TxGNN | 45/64 |
| 可映射分子实体 | 150/181 |
| 未映射或身份待审 | 65配对、31实体 |
| Open Targets | 64/64靶点完整缓存，71,755条关联 |
| 明细疾病假设 | {s['hypothesis_rows']}条，分候选/对照展示 |

已有的候选药全疾病分数直接复用，不重复训练或无必要地重算。新增对照用原TxGNN权重、原full_graph和原prototype decoder实际补跑；原生接口与批量decoder核对通过。Open Targets复用已核验的26.06完整分页缓存，未将其描述为本轮重新在线下载。

未映射/歧义项保留为空，不填零、不通过相似药名补分。候选继续使用原保守已知关系及身份排除遮罩；对照允许查看已有适应证，并明确标注，不能把它们计入新用途发现。

## 怎么整合

分子—靶点排名保留原值；TxGNN、Open Targets、EC-KG分别提供可追溯证据，不把三个异质分数简单相加。

- 联合疾病明细按OT关联分数≥0.1筛选，再按药物内TxGNN疾病排名查看前三项；0.1只是减轻极弱关联展示的审查阈值，未经前瞻效果验证。
- Top50交集计数仍按全部OT关联计算，和上述展示筛选口径分开。明细显示实际药内排名，不能把排在数千位的交集叫作高预测阳性。
- 每对同时保留靶点独立OT前3项；即使没有TxGNN，也能查看靶点疾病背景。合并MONDO节点有显式标记，不能当子病种特异预测。
- EC-KG既有关系审查逐对加入。12个候选的来源关系仍待核查；表达调控不等于直接结合，DrugBank物理相互作用标注也需检查原始证据。

## 系统是否已经能运行

用真实项目数据启动新进程完成系统检查：720药、888靶点，当前标准排名表为720×384（276,480对）；药物排名接口返回384靶点分母，实体接口返回对应8行SPR设计附加证据。`infer=True`评分路径可用，未启动训练。

原512设计是历史冻结排序与既有FULL_FIT路由的组合，不应归因于尚未通过验收的PocketPrecision/ContextPocket结构新模型。能返回模型排名、能生成设计与能保证实验成功是三个不同结论。

全局疾病页面也已升级：605药的全疾病Top30共18,150条线索，886靶点共907,849条OT关联，替换旧癌症/不完整疾病缓存。原其他靶点属性、通路和结合排名保持对应来源。界面“SPR设计”页新增逐对联合审查，包含目录外对照，目录外分子可在其靶点页查看。

检查记录：`SYSTEM_SMOKE_CHECK.json`（实际实体及排名HTTP接口200）、`VALIDATION.json`（512精确连接、完整靶点覆盖、补评分数有限、缺失不生造假设、原生decoder一致）。相关后端测试16项通过，前端生产构建通过。

已有常驻服务若已加载旧快照，需要重启后加载新数据。启动方式：`python scripts/run_explorer.py`。本轮新进程已验证；未停止其他常驻进程。

## 实验前仍需完成

原设计的构建、对照体系、活性物种和化学支持缺口没有被疾病关联消除。尤其需要优先核对ulipristal–AR、fenofibrate–PPARD等已有来源关系、Ponesimod结构问题及对照冲突。当前不把447个可评分配对说成447个合格实验候选，也不以疾病交集保证命中率。

## 交付与重现

目录：`outputs/spr512_integrated_disease_20260909/`。

- `SPR512_TXGNN_OPENTARGETS_REVIEW.xlsx`：512配对联合审查、疾病假设明细、64靶点疾病前20、181分子覆盖。
- `INTEGRATED_MASTER_512.csv`：保留原主表所有字段并附加联合结果。
- `DISEASE_HYPOTHESES_LONG.csv`：可筛选的逐疾病明细，含原始logit、药内排名、OT数据类型和节点映射范围。
- `OPENTARGETS_64_FULL_ASSOCIATIONS.parquet`：全部71,755条关联，不是Top20截断表。
- `DRUG_COVERAGE.csv`：区分旧结果复用、新补跑、图外和身份待审。
- `MANIFEST.json`：原配对、权重、基础数据、追加前文档和本次产物哈希。

```bash
python scripts/integrate_spr512_disease_20260909.py
/root/miniconda3/envs/bm-dti/bin/python scripts/run_spr512_supplementary_txgnn_20260909.py
python scripts/integrate_spr512_disease_20260909.py --build
python scripts/check_spr512_integrated_system_20260909.py
python scripts/document_spr512_integration_20260909.py
```

数据整合成功后，后端优先加载本目录通过验证的审查包；若验证失败，不静默展示错误数据。
''')
 design=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md';before=OUT/'DESIGN_BEFORE_INTEGRATION.md'
 if not before.exists():before.write_bytes(design.read_bytes())
 marker='## 8. 当前512对联合运行及界面接入'
 text=design.read_text().split(marker)[0].rstrip();design.write_text(text+f'''\n\n{marker}

已完成[512对联合运行与系统整合](BIOMASTER_SPR512_INTEGRATED_DISEASE_RUN_20260909_ZH.md)，可直接打开[联合审查工作簿](../outputs/spr512_integrated_disease_20260909/SPR512_TXGNN_OPENTARGETS_REVIEW.xlsx)。

补跑24个目录外对照分子的全疾病TxGNN，共409,920个分数。当前447/512对有可用TxGNN（候选402/448、对照45/64）；64靶点的Open Targets完整覆盖，共71,755条关联。65对仍缺模型分数或身份待审，未填零。

系统实际数据加载、排名和实体接口通过；页面“SPR设计”已接入逐对TxGNN×OT与EC来源审查，全局疾病页面也已换用全方向快照。当前可以生成和审查实验候选，但所有512对仍需原有构建、对照及证据核验后再决定实验放行。没有修改原配对、角色或结合排序，没有训练模型。

本次文档追加前快照及哈希保存在新目录，前一轮报告哈希对应本次追加前版本；本次版本关联见 `outputs/spr512_integrated_disease_20260909/MANIFEST.json`。
''')
 inputs=[master,ROOT/'data/raw/txgnn/TxGNNExplorer/model.pt',BASE/'TXGNN_INDICATION_LOGITS.npy',BASE/'TXGNN_EXCLUSION_MASK.npy',BASE/'OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet',BASE/'SPR512_ECKG_RELATION_REVIEW.csv']
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and p.suffix!='.log']
 source=[ROOT/'scripts'/n for n in ['integrate_spr512_disease_20260909.py','run_spr512_supplementary_txgnn_20260909.py','check_spr512_integrated_system_20260909.py','document_spr512_integration_20260909.py']]+[ROOT/'biomaster'/n for n in ['explorer_data.py','explorer_annotations.py','explorer_disease_reviews.py']]+[ROOT/'web/src/App.tsx',ROOT/'web/src/SupplementalEvidence.tsx']
 manifest={'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source},'doc_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [design,report]},'parent_design_sha256':sha(before),'parent_manifest':str((BASE/'SOURCE_AND_ARTIFACT_MANIFEST.json').relative_to(ROOT)),'training':False,'original_pairs_unchanged':True}
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(report)
if __name__=='__main__':main()

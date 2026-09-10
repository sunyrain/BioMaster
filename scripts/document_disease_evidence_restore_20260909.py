"""Write the evidence audit, append the SPR design, and record snapshot hashes."""
import json,hashlib
from pathlib import Path
import pandas as pd
from finalize_disease_evidence_review_20260909 import sha
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def main():
 s=json.loads((OUT/'FINAL_SUMMARY.json').read_text());inc=json.loads((OUT/'ECKG_INCREMENT_SUMMARY.json').read_text());scan=json.loads((OUT/'ECKG_SCAN_SUMMARY.json').read_text());manifest=json.loads((OUT/'eckg/SOURCE_MANIFEST.json').read_text());valid=json.loads((OUT/'VALIDATION.json').read_text());assert valid['all_pass']
 new=pd.read_csv(OUT/'ECKG_NEW_COMPARABLE_ASSERTION_ENDPOINT_PAIRS.csv.gz').relation_class.value_counts().to_dict();c=pd.read_csv(OUT/'DRUG_COVERAGE_FINAL_720.csv');t=pd.read_csv(OUT/'TARGET_COVERAGE_888.csv');spr=pd.read_csv(OUT/'SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv');cand=spr[spr.selection_role.ne('POSITIVE_CONTROL')];high=cand[cand.selection_role.eq('OUR_FROZEN_MODEL_HIGH')];rel=pd.read_csv(OUT/'SPR512_ECKG_RELATION_REVIEW.csv');non=rel[rel.selection_role.ne('POSITIVE_CONTROL')&rel.non_text_unambiguous_assertion_rows.gt(0)]
 directions={'breast':'乳腺','cardiovascular':'心血管','chromosomal':'染色体','connective_tissue':'结缔组织','dermatologic':'皮肤','ear_nose_throat':'耳鼻喉','endocrine':'内分泌','eye_and_adnexa':'眼及附属器','gastrointestinal':'胃肠','hematologic':'血液','immune':'免疫','infection':'感染','metabolic':'代谢','musculoskeletal':'肌肉骨骼','neoplasm':'肿瘤','neurological':'神经','obstetric':'产科','poisoning_and_toxicity':'中毒与毒性','psychiatric':'精神','reproductive':'生殖','respiratory':'呼吸','renal_and_urinary':'肾及泌尿','syndromic':'综合征'}
 dirtext='\n'.join('| '+directions[k.removeprefix('speciality_')]+' | '+str(v)+' |' for k,v in s['direction_disease_node_counts'].items())
 report=f'''# BioMaster 全方向疾病推理恢复与新版图谱增量审计（2026-09-09）

本轮遵循“不训练”，已经执行推理和证据下载。主输出目录：`outputs/biomaster_disease_evidence_720x888_20260909/`。原SPR32/64配对、角色、盲表和模型权重保持原样；新增内容是附加疾病审查，不能当成SPR已验证。

## 1. 完成到哪里

| 层级 | 实际结果 | 边界 |
|---|---|---|
| 项目分母 | 720个模型分子实体、888个官方靶点，639,360个配对覆盖记录 | 不是850条FDA实体记录；不是仅SPR120药 |
| TxGNN原权重推理 | 608药 × 17,080疾病 = 10,384,640个原始logit | 仅indication查询；没有新训练或新图节点编码 |
| 可进入疾病线索审查 | 605药 | 另3药标识冲突暂扣，原始分数保留追溯 |
| 其余实体 | 97药仅来源关系线索；18药本轮无可用疾病推理/关系线索 | 全部保留在720分母，不能填0冒充阴性 |
| Open Targets | 886/888靶点，907,849条靶点—疾病/表型/性状关联 | 885单ID完整分页，IL3RA双ID按同UniProt合并；2靶点待身份处理 |
| EC-KG标识覆盖 | 717/720药，885/888靶点 | 药698单节点、19多节点；靶点877单节点、8多节点，不等于结构或作用已证实 |
| 全方向 | 全17,080疾病节点均计算；按来源本体整理23个多标签方向 | 2,041节点无对应方向标签，仍保留分数 |
| 原SPR设计补全 | 512行附加审查、64靶点汇总 | 448候选中402对/108药有TxGNN，46对/12药无该模型分数 |

**现阶段合适的组合是：原结合排序提供待测分子—靶点；TxGNN提供药—疾病假设；Open Targets提供靶点—疾病证据；新版图谱用于补充图外实体线索及检查已有关系。疾病一致性不证明结合，不证明应抑制还是激动，更不能推算实验命中率。**

## 2. 复用了什么，怎样证明没有换路线

复用 `data/raw/txgnn/TxGNNExplorer/` 下的原权重和配置、既有 `full_graph_42` 图及划分。旧图有129,375节点、4,050,249条canonical关系、7,957药节点、17,080疾病节点；它们是旧图分母，不是本轮药物数量。

原19方向脚本保留，但它依赖的代理疾病目录和完整结果未找到。这次保留原模型，直接遍历所有疾病，而不重新创造19个代理疾病。冻结图只编码一次，随后调用原版prototype decoder分批评分。未改模型源码、未加入新边、未调用优化器。

- 原生 `predict` 与批量decoder对1,824个查询的最大绝对误差为0。
- 与历史癌症结果共有525个药节点，logit最大差异约1.78e-15，相关系数1。这是历史路线复现检查，不是新效果验证。
- 固定随机种子0，沿用原代码的节点输入初始化。疾病原型聚合仍按原decoder执行，不能直接拿普通点积冒充TxGNN。
- 原始logit和sigmoid都没有经过本项目的临床/SPR概率校准。未训练BioPathNet，未声称完成同划分模型优劣比较。

验证详情见 `VALIDATION.json`、`TXGNN_DECODER_VERIFICATION.json`、`HISTORICAL_CANCER_REPRODUCTION.csv`。

## 3. 分子与靶点身份

720来自历史模型实体及其SMILES/完整InChIKey。与FDA主表结构连接时，有42个模型实体不能按完整键连回该主表，包含活性物种差异；不能只用原药名强行匹配。组合药名、多实体映射和这42项使用完整结构键查询PubChem同义名。

结果为608个唯一图节点映射、110未映射、2歧义（dextroamphetamine/lisdexamfetamine合并名称实体，以及levalbuterol）。名称映射本身仍不等于验证了旧图节点的立体结构。

再以EC-KG结构标识与DrugBank标识做交叉核对，发现ipratropium、naftifine、bepotastine分别落入不同规范节点，暂扣解释资格，剩605药进入线索表。这可能涉及盐型/标准化差异，不直接判定本地结构错误。另602个同节点吻合也只是规范化器一致性，不是独立立体化学实验证据。Ponesimod既有结构异议继续保留，没有用名称结果覆盖它。

888靶点在旧图的基因层面映射887个（采用来源标识/精确符号，非构建或异构体验证）。Open Targets剩余缺口：

- KIR2DL2/P43627：UniProt提供6个Ensembl候选，本轮API均返回无target；没有换成KIR2DL3。
- PDE4/Q86V67：UniProt为未审校的磷酸二酯酶片段，未给出唯一Ensembl；没有猜成PDE4A/B/C/D。
- IL3RA的两条Ensembl都回到P26951，分别完整取回后按疾病ID合并，保留所选记录的来源Ensembl。它不是查询失败，也不是把两套计数简单相加。

EC-KG未命中标识的3药为galantamine、bupivacaine、amoxicillin，三者仍有旧TxGNN结果；未命中标识不代表新图中绝对不存在该化合物。EC靶点缺口为PROS1、GAA、PDE4，其中前两者有Open Targets证据。

## 4. 全方向与疾病ID

按EC公开疾病清单的23个`speciality_*`字段做多标签归类，不用药物名称或字符串关键词猜方向。一个疾病可属于多个方向，计数不可相加当作独立疾病数。

| 方向 | 已归类旧图疾病节点 |
|---|---:|
{dirtext}

旧TxGNN有合并MONDO节点，例如一个节点含多个疾病ID。对齐Open Targets/EC时保留 `MEMBER_OF_MERGED_NODE`，**不把合并组分数声称为每个子病种的特异性预测**。无法精确用MONDO映射的OT关联仍保留在全量证据表，不强接到近似疾病名。方向归类缺失2,041节点仍在完整矩阵中。

已知关系排除：旧图indication/off-label/contraindication覆盖13,311查询格；EC治疗/试验/禁忌的保守来源标注覆盖13,000格；两者并集14,184格，其中EC相对旧图额外排除873格。再将3个身份冲突药整行暂扣。该EC遮罩按关系及初始agent/knowledge标记筛选，可能含混合来源，不能将它当成已核实适应证清单。EC试验记录是保守排除线索，不等于已获批适应证；此遮罩不是当前全球全部适应证/论文/专利查新完成证明。

## 5. 新版图谱审计：不能把总边数叫作有效增量

### EC-KG：已经下载并实际逐分片扫描

来源为[Every Cure公开数据](https://docs.dev.everycure.org/releases/public_data_releases/)。固定`kg-edges`修订 `{manifest['kg-edges']['revision']}`；各数据集独立修订和发布者SHA256见 `eckg/SOURCE_MANIFEST.json`。本轮扫描全部8个节点分片、49个关系分片：7,354,612节点、81,149,812关系。与项目实体相连的原始关系8,305,253条，完整限定条件及来源保存在Parquet。

筛出药—疾病、项目药—项目靶点、靶点—疾病，按选定关系/方向/来源字段去重后为855,294条审计记录。这不是855,294个新发现，也不是全部图边的唯一性声明。

再要求：旧图端点可比较、旧图没有该端点配对、存在非文本来源断言、项目节点映射唯一、无已发现身份冲突；得到 **{inc['new_non_text_comparable_pairs']:,}个待核验的新增端点关系**：

| 关系类别 | 符合以上筛选的新增端点对 |
|---|---:|
| 药—疾病 | {new.get('DRUG_DISEASE',0):,} |
| 项目药—项目靶点 | {new.get('DRUG_TARGET',0):,} |
| 靶点—疾病 | {new.get('TARGET_DISEASE',0):,} |

这里审计的是**相对旧快照的证据覆盖增量**，不等于真实生物学新关系：不同谓词、LINCS表达调控、CTD关联、DrugBank物理关系不能混作KD结合阳性。文本挖掘/模型预测、仅转载PrimeKG、来源或推断性质不清、映射多义，均独立标识；含文本挖掘来源的混合来源记录也保守归入文本审查。原始限定条件保留供后续细查。

旧TxGNN未出分的112药都能在EC匹配到唯一规范节点；94药有药—疾病关系，18药本轮仍只有身份、没有相应疾病关系。**节点补齐没有让旧模型自动具备这些新药的推理能力。**

### OptimusKG：官方后继已确认，本轮数据访问受阻

此前只谈PrimeKG重建和第三方更新不够完整。[PrimeKG官方README](https://github.com/mims-harvard/PrimeKG)现在明确推荐后继[OptimusKG](https://github.com/mims-harvard/OptimusKG)。官网宣称192,813节点、21,834,669关系、65来源；这些是作者总图统计，未计入本轮项目覆盖或增量数字。

其官方Dataverse DOI `10.7910/DVN/IYNGEV` 的多个元数据端点本轮返回403，网页要求人机验证。已存访问审计及官方修订信息，**没有取得数据，不报720/888覆盖，不宣称完成其有效增量实测**。这不影响已完成的TxGNN、OT和EC-KG工作。后续优先获得该官方快照并套用相同审计，而非直接重训。

[CellAwareGNN/PrimeKG-U代码](https://github.com/OHPENLab/CellAwareGNN)展示更新图的训练脚本，README要求自行准备数据，本轮未取得可核验的更新图快照。因此没有将论文增量当成本项目覆盖增量，也没有为此开始训练。

## 6. 对原512对意味着什么

64靶点的Open Targets证据现已补齐。448候选中402对有模型结果，其中350对在药物Top50疾病线索中有对应靶点OT证据；320个高排中287对可映射、255对有交集。中排为50/64、低排为45/64，**各组都容易出现疾病交集，不能据此证明高排组会结合、或KG融合提高了命中率**。

另外{len(non)}个候选对在EC出现非文本、无节点多义的既有来源关系，详见 `SPR512_ECKG_RELATION_REVIEW.csv`：

- ulipristal–AR、fenofibrate–PPARD有DrugBank来源的物理相互作用标注，优先查原始来源、真实测试物种及实验端点，再决定是否应归入已有关系。
- tamsulosin–MMP12、belinostat–NT5E等LINCS调控记录提示表达变化，不能当直接结合。
- 其他affects/调控关系与文本关系分别保留；未检出也不能证明全新。

疾病信息可以帮助说明“为什么值得研究”，不能补掉原报告中274/320高排缺乏近邻阳性化学支持、构建待确认、对照冲突或候选活性物种未核验的问题。本轮没有放行任何实验或偷偷重排盲表。

当前建议：先处理3药身份冲突、2靶点身份缺口，以及上述候选既有关系核查；以原结合排序为主，逐对确认方向、测试物种、构建和对照，再决定首批实验。暂不为更换图谱投入训练。若以后需要识别新图外药物的模型排序，再单独评估训练或其他已发布权重。

## 7. 文件、读取与复现

先打开 `DISEASE_EVIDENCE_REVIEW_720_888.xlsx`，有720药、888靶点、23方向、512对附加疾病信息和既有关系审查。大表不塞入Excel，避免行数截断。

| 文件 | 内容 |
|---|---|
| `DRUG_COVERAGE_FINAL_720.csv` / `TARGET_COVERAGE_888.csv` | 全分母覆盖、身份缺口、图外状态 |
| `TXGNN_INDICATION_LOGITS.npy` | 608×17,080原始分数，含暂扣行，仅供可追溯读取 |
| `TXGNN_SCORE_DRUG_ORDER.csv` / `TXGNN_SCORE_DISEASE_ORDER.csv` | 矩阵行列顺序，不能用药名排序后直接错位读取 |
| `TXGNN_EXCLUSION_MASK.npy` | bit1旧图已知关系，bit2新增图保守排除，bit4身份暂扣；取值0才进入线索表 |
| `DRUG_ALL_DISEASE_TOP30.csv.gz` / `DRUG_EVERY_DIRECTION_TOP3.csv.gz` | 605药的审查线索，不是临床治疗推荐 |
| `OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet` | 907,849来源关联，含数据类型、方向领域、MONDO对齐状态 |
| `PAIR_COVERAGE_720X888.parquet` | 639,360格覆盖及Top50疾病交集；不是720×888新结合预测 |
| `ECKG_RELEVANT_RELATION_AUDIT.parquet` / `ECKG_INCREMENT_COUNTS.csv` | 新图来源性质、旧图端点覆盖及映射歧义审计 |
| `SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv` / `SPR512_ECKG_RELATION_REVIEW.csv` | 原设计配对的新增审查层 |
| `VALIDATION.json` / `SOURCE_AND_ARTIFACT_MANIFEST.json` | 完整性核验与源文件哈希；不含前瞻性能保证 |

按顺序运行：

```bash
python scripts/restore_disease_evidence_720x888_20260909.py
python scripts/fetch_ot888_disease_evidence_20260909.py
/root/miniconda3/envs/bm-dti/bin/python scripts/run_txgnn_all_diseases_720_20260909.py
python scripts/download_eckg_audit_snapshot_20260909.py
python scripts/audit_eckg_project_increment_20260909.py
python scripts/assemble_disease_evidence_720x888_20260909.py
python scripts/finalize_disease_evidence_review_20260909.py
python scripts/enrich_disease_evidence_review_20260909.py
python scripts/document_disease_evidence_restore_20260909.py
```

已有原始缓存会复用；跨版本重跑应另建输出快照，不应混合版本。验证通过包括全配对分母、分数有限与维度、原生/历史复现、OT逐靶点唯一疾病数与总数、数据版本一致、EC全部分片SHA256、原图/权重/SPR主表未变，以及排除遮罩确实应用。图内推理覆盖、证据覆盖与前瞻预测性能严格分开。
'''
 reportpath=ROOT/'docs/BIOMASTER_ALL_DIRECTION_720_888_RESTORE_AUDIT_20260909_ZH.md';reportpath.write_text(report)
 design=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md';before=OUT/'SPR64_DESIGN_BEFORE_DISEASE_APPENDIX.md'
 if not before.exists():before.write_bytes(design.read_bytes())
 marker='## 7. 全方向疾病推理与新版图谱补全（2026-09-09追加）'
 base=design.read_text().split(marker)[0].rstrip()
 appendix=f'''\n\n{marker}

已按“不训练”恢复全疾病推理，并审计整个720药、888靶点分母。详见[全方向恢复与增量审计](BIOMASTER_ALL_DIRECTION_720_888_RESTORE_AUDIT_20260909_ZH.md)，审查表为[720药/888靶点工作簿](../outputs/biomaster_disease_evidence_720x888_20260909/DISEASE_EVIDENCE_REVIEW_720_888.xlsx)。

- **TxGNN实际完成608×17,080 = 10,384,640个分数**；3药身份冲突暂扣后605药进入线索表。另97药仅来源关系，18药本轮无可用疾病线索。恢复覆盖23个本体方向，同时保留未分类疾病，未局限癌症。
- **Open Targets补齐886/888靶点、907,849条关联**；KIR2DL2和PDE4片段仍有身份缺口。原设计64靶点均取得证据。IL3RA双Ensembl来源分别核验后合并。
- **EC-KG实扫81,149,812关系**，项目标识匹配717药/885靶点；严格分开旧图转载、文本预测、来源断言和映射歧义。在可比较端点上筛出{inc['new_non_text_comparable_pairs']:,}个新增来源断言端点对待核验，不代表新的真实结合或性能提升。
- 原448候选中402对/108药有TxGNN；350对在药物Top50疾病中与靶点OT有交集，高排255/320、中排50/64、低排45/64。**交集广泛存在，不能当结合模型验证。**
- 新增{len(non)}个候选对既有来源关系审查标记；ulipristal–AR、fenofibrate–PPARD等需优先查原始证据。附表不改变原选对、排名或角色，也未放行实验。

另纠正图谱后继信息：PrimeKG官方已推荐OptimusKG；本轮其Dataverse数据访问返回403/人机验证，尚未实测项目覆盖。不要把官网总图数字填进本项目增量，也无需因此开始训练。

本节是原设计的追加审查。原报告追加前快照保存在新输出目录，旧 `ARTIFACT_MANIFEST.json` 的 `design_report_sha256` 对应追加前版本；新报告哈希与旧哈希关联在 `SOURCE_AND_ARTIFACT_MANIFEST.json`，原设计清单及旧清单哈希不改写。
'''
 design.write_text(base+appendix)
 inputs=[ROOT/'data/raw/txgnn/TxGNNExplorer/model.pt',ROOT/'data/raw/txgnn/TxGNNExplorer/config.pkl',ROOT/'data/raw/txgnn/kg_directed.csv',ROOT/'data/raw/txgnn/node.csv',ROOT/'data/raw/txgnn/kg.csv',ROOT/'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv',ROOT/'outputs/final_design_space_2005_2026_v1/FINAL_FROZEN_FDA_DRUG_ENTITIES_2005_2026.csv',ROOT/'outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1/OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz',ROOT/'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz',ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv']
 inputs+=list((ROOT/'data/raw/txgnn/full_graph_42').glob('*.csv'))
 scripts=['restore_disease_evidence_720x888_20260909.py','fetch_ot888_disease_evidence_20260909.py','run_txgnn_all_diseases_720_20260909.py','download_eckg_audit_snapshot_20260909.py','audit_eckg_project_increment_20260909.py','assemble_disease_evidence_720x888_20260909.py','finalize_disease_evidence_review_20260909.py','enrich_disease_evidence_review_20260909.py','document_disease_evidence_restore_20260909.py']
 oldman=json.loads((ROOT/'outputs/retargetmap_spr64_design_20260909/ARTIFACT_MANIFEST.json').read_text());assert sha(before)==oldman['design_report_sha256']
 artifacts=[x for x in OUT.iterdir() if x.is_file() and x.suffix!='.log' and x.name!='SOURCE_AND_ARTIFACT_MANIFEST.json']
 provenance={'no_training':True,'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'raw_cache_sha256':{str(p.relative_to(ROOT)):sha(p) for folder in ['raw','ot_cache'] for p in (OUT/folder).iterdir() if p.is_file()},'vendored_txgnn_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'third_party/TxGNN/txgnn').rglob('*.py')},'script_sha256':{x:sha(ROOT/'scripts'/x) for x in scripts},'artifact_sha256':{str(p.relative_to(ROOT)):sha(p) for p in artifacts},'doc_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [reportpath,design]},'original_design_report_sha256':sha(before),'old_design_manifest_preserved':True,'ec_source_manifest':manifest,'limitations':['OptimusKG dataset inaccessible: HTTP403','2 target identity gaps','18 drugs without usable disease evidence','3 scored drug identity holds','no prospective efficacy or SPR performance claim']}
 (OUT/'SOURCE_AND_ARTIFACT_MANIFEST.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n');print(reportpath);print('DOCUMENTS AND MANIFEST COMPLETE')
if __name__=='__main__':main()

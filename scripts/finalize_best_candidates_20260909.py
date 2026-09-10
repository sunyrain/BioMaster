"""Evidence synthesis: preserve 384; propose one scout and one conditional backup."""
import json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/best_candidate_synthesis_20260909';PREV=ROOT/'outputs/joint384_design_20260909';DOC=ROOT/'docs/BIOMASTER_BEST_CANDIDATE_SYNTHESIS_20260909_ZH.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=PREV/'CANDIDATES_384.csv';before=sha(source);p=pd.read_csv(source);controls=pd.read_csv(PREV/'REFERENCE_CONTROLS_EXTRA.csv');parts=[]
 for name in ['enzyme_all_10_review.csv','kinase_final_review.csv','nuclear_final_review.csv']:
  d=pd.read_csv(OUT/name);d['review_source_file']=name;parts.append(d)
 deep=pd.concat(parts,ignore_index=True);assert len(deep)==24 and deep.pair_id.is_unique
 raw=pd.read_csv(OUT/'ALL384_RAW_ACTIVITY_NOVELTY_REAUDIT.csv');r=p.merge(raw[['pair_id','raw_activity_rows','raw_high_confidence_direct_target_annotation','novelty_reaudit_status']],on='pair_id',validate='one_to_one')
 r=r.rename(columns={'reason':'previous_quick_review_reason','sources':'previous_quick_review_sources'})
 r=r.merge(deep[['pair_id','recommendation','reason','sources','critical_gap']],on='pair_id',how='left',validate='one_to_one')
 r['deep_review_status']=r.recommendation.fillna('NOT_TARGETED_DEEP_REVIEWED_THIS_ROUND')
 r['new_scout_novelty_gate']=r.raw_activity_rows.eq(0)
 r['current_first_batch_status']='RETAIN_IN384_NOT_RECOMMENDED_FOR_FIRST_SCOUT'
 plans=[('tecovirimat','AR','A_FIRST_SINGLE_TARGET_SCOUT','原抗病毒用途；模型Top1/384、同病精确映射及遗传/体细胞条件、阳性近邻约0.463；24项重点核验中唯一保留单靶点侦察建议，原始384端点审计未查到该精确配对。不是已有亲和验证。','AR配体结合域，确认参考配体活性；先直接结合/竞争，确认特异性后再区分激动与拮抗；不能由此直接推荐HER2乳腺癌治疗。'),('pregabalin','MME','B_CONDITIONAL_FUNCTIONAL_BACKUP','非癌跨领域候补：原神经系统相关用途→血压调控；Top5/384，同病遗传关联0.4409；近邻0.368不足以证明结合，但不机械因低于0.4排除。无精确配对原始记录，不表示穷尽查新。','仅在已有成熟MME酶活体系可用时加入；确认Zn2+催化域与参考抑制剂，先做浓度依赖酶活，再决定直接结合；外周水肿等不良反应不能作为降压或MME结合证据。')]
 rows=[]
 for drug,gene,tier,why,plan in plans:
  hit=r[r.drug_names.eq(drug)&r.gene_symbol.eq(gene)];assert len(hit)==1
  x=hit.iloc[0].to_dict();assert x['raw_activity_rows']==0
  x.update(final_priority=tier,synthesis_reason=why,first_validation_plan=plan,release_status='COMPUTATIONAL_SCOUT_RECOMMENDATION_NOT_EXPERIMENT_RELEASE')
  rows.append(x);r.loc[r.pair_id.eq(x['pair_id']),'current_first_batch_status']=tier
 final=pd.DataFrame(rows);final.to_csv(OUT/'FIRST_SCOUT_AND_CONDITIONAL_BACKUP.csv',index=False)
 r.to_csv(OUT/'ALL384_UPDATED_TRIAGE_APPEND_ONLY.csv',index=False);deep.to_csv(OUT/'TARGETED24_DEEP_REVIEW.csv',index=False)
 reserve=deep[deep.recommendation.eq('CONDITIONAL')&~deep.pair_id.isin(final.pair_id)];reserve.to_csv(OUT/'OTHER_CONDITIONAL_NOT_FIRST_BATCH.csv',index=False)
 scoutcontrols=controls[controls.target_chembl_id.isin(final.target_chembl_id)].copy();scoutcontrols['note']='Existing reference proposal only; verify construct, assay and compound suitability. Backup target not automatically ordered.';scoutcontrols.to_csv(OUT/'SCOUT_REFERENCE_PROPOSALS.csv',index=False)
 with pd.ExcelWriter(OUT/'BEST_CANDIDATE_SYNTHESIS.xlsx') as w:
  final[['drug_names','gene_symbol','final_priority','synthesis_reason','first_validation_plan','critical_gap','sources','binding_rank_384','review_disease','review_ot_score','review_txgnn_rank','positive_max_tanimoto','negative_max_tanimoto','ligand_inchikey','target_chembl_id']].to_excel(w,sheet_name='首选1与条件候补1',index=False)
  deep.to_excel(w,sheet_name='24项深入核验',index=False);raw.to_excel(w,sheet_name='384原始端点复查',index=False);pd.read_csv(OUT/'ALL384_RAW_CHEMBL_ACTIVITY_RECORDS.csv.gz').to_excel(w,sheet_name='88条原始活动',index=False);reserve.to_excel(w,sheet_name='其他条件候补',index=False);scoutcontrols.to_excel(w,sheet_name='参考配体提案',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 summary={'preserved_original_candidates':len(p),'targeted_deep_review_pairs':len(deep),'deep_review_decisions':deep.recommendation.value_counts().to_dict(),'all384_with_additional_raw_activity_records':int(raw.raw_activity_rows.gt(0).sum()),'first_scout_pairs':1,'conditional_backup_pairs':1,'first_scout':'tecovirimat–AR','conditional_backup':'pregabalin–MME','other_conditional_pairs':len(reserve),'clinical_or_binding_success_probability_established':False,'original384_sha256':before,'original384_unchanged':before==sha(source)}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 report='''# 当前最合适的首批候选：综合证据后的收敛建议

2026-09-09。原384完整保留，不删除、不重排原文件。本报告回答的是“实验宝贵时，先验证谁最合理”，不是宣布发现了有效药物，也不声称已找到全空间唯一最优解。

## 1. 我的建议：先1对，另1对有条件候补

| 顺序 | 配对 | 当前建议 |
| --- | --- | --- |
| 首选 | **tecovirimat–AR** | 小规模单靶点直接结合侦察；不直接投入疾病模型或大规模SPR扩展 |
| 条件候补 | **pregabalin–MME** | 若已有成熟MME酶活体系，先以功能实验判断是否值得继续；不单独为它追加昂贵体系 |

**现在没有任何一对达到“已知高命中概率、可直接进入疾病验证”的证据水平。** 这份顺序兼顾跨领域、新颖性复核、实验可验证性和已有反证；不是预测成功率排名。未查得既有记录只减少了一个反对理由，不能作为结合支持。

### 首选：tecovirimat–AR

有利因素：原用途属于病毒感染，当前共同疾病线索是HER2阳性乳腺癌，符合跨大类目标；结合模型在该药384靶点中排1，药内TxGNN疾病排41，OT约0.370，阳性近邻Tanimoto约0.463、阴性近邻约0.246；本次全端点ChEMBL精确结构／父结构关联审计未发现该配对活动记录。AR-LBD及参考配体提供了相对明确的单靶点验证路径。

限制：模型分和相似度不等于Kd；近邻属于Ki/IC50混合参考，仍需核原始实验。EMA资料中未开展secondary pharmacology panel，并不是已经证实AR无作用或有作用。AR在具体疾病背景下需要何种作用方向、此药能否在可实现暴露下影响AR，都没有建立。此时应把“抗病毒药新AR结合假设”作为问题，把肿瘤用途留到后续机制阶段。

第一步用实际采购物种和经过活性核验的AR-LBD做直接结合／配体竞争。先确认样品身份、工作液溶解性及溶剂匹配，设置阳性参考、空白／非特异性对照；只有可重复、浓度依赖、可解释的特异信号才进入激动／拮抗与细胞机制。若在合格可测范围内不能复现特异结合，不因TxGNN/OT高分继续追加疾病实验。现有参考提案为testosterone，仍需适配当前构建，不能将混合pChEMBL当SPR Kd。

### 条件候补：pregabalin–MME

这条线索能保留非癌方向：神经系统相关原用途→血压调控。模型Top5/384；hypertensive disorder共同疾病TxGNN排30、OT约0.335，遗传关联约0.441。阳性近邻0.368低于原0.4启发式，但阴性近邻更低约0.286；不能只凭一个相似度阈值把可能的跨骨架探索全部淘汰。

目前未核得母药直接MME亲和测量。本次本地全端点审计也未查到该配对，但不代表穷尽文献。MME成熟含锌胞外酶可先考虑成熟酶活检测，给出比泛泛疾病图谱更直接的第一道判断；只有功能／结合结果值得跟进时才进一步投入。pregabalin标签中的外周水肿、血管性水肿等是风险信息，不能倒过来证明MME机制或降压治疗方向。当前参考提案omapatrilat只有Ki/IC50汇总，需验证MME体系，不能当已验证SPR对照。

## 2. 为什么撤回此前一些看上去更强的候选

本轮三组agent对24对做重点原始来源、分子形态、同工酶／构建与暴露解释核验。同时，我对**全部384对**重查ChEMBL37原始activities，包含AC50、百分比抑制及自由文本注释，而不限以前严格亲和表。

共查到**48对、88条额外原始活动记录**。其中有同源映射、细胞／功能、负面及非亲和端点，不能把48都说成真实结合或阴性；但它们不应未经复核就作为“从未测试的新配对”。原始审计保留关系符号、注释、assay描述、confidence与relationship标记。

| 配对／类别 | 深入核验结果 | 首批取舍 |
| --- | --- | --- |
| **mebendazole–ESR1** | 精确结构DRUGMATRIX注释10µM下抑制<50%；另一NIBR无细胞结合记录AC50 **>10µM**，不是10µM阳性 | 撤回原优先建议；跨癌种之外的跨大类属性不能覆盖实测反证 |
| **articaine–ACHE** | 精确结构NIBR酶活AC50 **>30µM** | 不推荐为高亲和新关系首批 |
| **baricitinib–HDAC6** | 已有百分比抑制记录−22.99%、1.75%，筛选浓度待核 | 既有测试，不作未经测试的新颖首批 |
| **fentanyl–CA1/CA7** | 0.78近邻是新增磺酰胺的fentanyl衍生物，母药缺少该关键改变 | 不将衍生物CA活性转给母药；操作与用途成本也不优先 |
| **esmolol/metoprolol–CA1** | 已有人CA1弱作用研究；报告效力分别到mM／数百µM范围 | 不是文献未知强结合候选 |
| **pomalidomide–MIF** | PROTAC中pomalidomide负责CRBN端，独立MIF配体在另一端 | 不能将PROTAC整体作用当游离母药MIF结合 |
| **ozanimod–XDH** | 母药暴露很低，不能拿主要活性代谢物替代模型母药 | 即便图谱和非癌方向好，也不进入当前首批 |
| **ulipristal–NR3C2** | 模型为去乙酰母体，标签／旧药理主要针对acetate共价物种 | 先解决药物实体，不直接套用标签或购买乙酸酯代替 |
| **lacosamide–KLKB1** | 共同疾病angioedema也出现在药物不良反应中 | 图谱治疗方向不明确，不优先 |

mebendazole原始ChEMBL activity 7766250/7766251注释和25167162的关系符号`>`已逐条保存；articaine对应25142808。NIBR来源为[原始数据研究](https://doi.org/10.1038/s41467-023-40064-9)。这些记录的confidence8／relationship H映射限制也保留，不能写成任何条件下绝对不结合。baricitinib–HDAC6对应[原始提交数据](https://doi.org/10.6019/CHEMBL4808148)，不将缺筛选浓度的百分比直接折算Kd。

同病交集没有验证药物作用方向；近邻高相似度也可能来自“保留母药骨架、添加第二靶点药效团”的设计。药物—靶点、靶点—疾病与药物—疾病三个环节必须分别审查。

## 3. 其他有条件线索为何暂不与首选并行

baricitinib–HDAC1/HDAC11、desonide–NR3C2、piperacillin–CA1保留为条件备选。

- baricitinib近邻为JAK/HDAC双靶点设计，共享JAK骨架不等于具备HDAC结合药效团；HDAC11还需匹配长链去酰化底物。没有理由同时购买三个HDAC体系赌母药作用。
- desonide–MR虽然相似度高，仍可能只是核受体类药的邻近作用；MR激动／拮抗与肾病方向未建立，局部用途和系统暴露也要审查。
- piperacillin另一个同工酶CA7只有mM级报告，不能当CA1效力；CHF交集主要来自clinical证据，没有CA1特异治疗机制。单药β内酰胺稳定性也需确认，不用与tazobactam复方代替。

这些不从384删除，但不属于当前“最适合先投实验”的并行首批。

## 4. 核验范围与决策边界

全部384完成新增原始端点审计；24对进行了定向深入复核，包含化学支持较强的跨领域／原用途不明项，以及2个较低相似度但遗传或非癌方向值得检查的补充项。并非全部384都完成穷尽的原始文献、专利、暴露与构建审核。因此不能声称这是全空间唯一最优两对，也不能估计它们的成功概率。

我不建议现在下达384对同等投入的实验计划。保留大池是为了后续恢复与比较；当前执行建议是先验证最清楚的一个结合问题，有现成低成本体系时才加入一个条件候补。真实阴性、样品问题、非特异信号或方向不合适都应及时止损；真实阳性也需要正交与功能验证后才能谈疾病用途。

## 5. 交付

- [综合审查工作簿](../outputs/best_candidate_synthesis_20260909/BEST_CANDIDATE_SYNTHESIS.xlsx)：首选／候补、24项深入意见、384原始活动审查、88条记录及参考配体提案。
- [首选与条件候补CSV](../outputs/best_candidate_synthesis_20260909/FIRST_SCOUT_AND_CONDITIONAL_BACKUP.csv)。
- [原384追加审查表](../outputs/best_candidate_synthesis_20260909/ALL384_UPDATED_TRIAGE_APPEND_ONLY.csv)，原文件未改。
- [酶类深入意见](../outputs/best_candidate_synthesis_20260909/ENZYME_FINAL_REVIEW.md)、[补充非癌方向](../outputs/best_candidate_synthesis_20260909/ENZYME_SUPPLEMENTAL_REVIEW.md)、[激酶相关意见](../outputs/best_candidate_synthesis_20260909/KINASE_FINAL_REVIEW.md)、[核受体意见](../outputs/best_candidate_synthesis_20260909/nuclear_final_review.md)。逐对原始／官方来源URL在对应CSV。

原始端点SQL、数据库版本及只读范围见`RAW_ACTIVITY_AUDIT.json`。检查包括24条唯一深入意见、全部384去向、首批无新查原始活动记录、原384及512哈希不变。无训练、无新模型分、无实验放行。
'''
 DOC.write_text(report)
 # Preserve previous document states and append the new, narrower decision.
 active=ROOT/'docs/BIOMASTER_JOINT384_AGENT_REVIEW_20260909_ZH.md';design=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md';cross=ROOT/'docs/BIOMASTER_CROSS_INDICATION_REVIEW_20260909_ZH.md'
 for path in [active,design,cross]:
  snapshot=OUT/('BEFORE_BEST_'+path.name)
  if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
 note='> 最新深入复核：原384保留；当前建议首选tecovirimat–AR单靶点侦察，pregabalin–MME仅条件候补。mebendazole–ESR1因原始负面端点撤回优先建议。详见[综合结论](BIOMASTER_BEST_CANDIDATE_SYNTHESIS_20260909_ZH.md)。\n\n'
 for path in [active,cross]:
  text=path.read_text()
  if note not in text:text=text.replace('\n\n','\n\n'+note,1)
  path.write_text(text)
 heading='\n## 13. 综合证据后的首批建议'
 design.write_text(design.read_text().split(heading)[0]+heading+'\n\n已完成[综合证据审查](BIOMASTER_BEST_CANDIDATE_SYNTHESIS_20260909_ZH.md)：24对重点深入核验、384对原始全端点复查，48对有额外活动记录。mebendazole–ESR1已有AC50>10µM，撤回优先建议；当前先考虑tecovirimat–AR单靶点结合侦察，pregabalin–MME仅在现成酶活体系下条件候补。原384／512均不改写，不强求实验数量，不代表已验证药效。\n')
 checks={'deep24_unique':len(deep)==24 and deep.pair_id.is_unique,'all384_preserved':len(r)==384 and set(r.pair_id)==set(p.pair_id),'first_scout_no_local_raw_activity':final.raw_activity_rows.eq(0).all(),'mebendazole_not_first':not final.drug_names.eq('mebendazole').any(),'articaine_not_first':not final.drug_names.eq('articaine').any(),'original384_unchanged':before==sha(source),'original512_unchanged':sha(ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv')=='e6501353312b62717c766156fdea4b145bbf9b135eef3efff332d491d1f02487'}
 checks={k:bool(v) for k,v in checks.items()};(OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks},indent=2));assert all(checks.values())
 files=[source,PREV/'REFERENCE_CONTROLS_EXTRA.csv',Path(__file__),ROOT/'scripts/audit_best_candidate_raw_activities_20260909.py',DOC,active,design,cross,*[f for f in OUT.iterdir() if f.is_file() and f.name!='MANIFEST.json']]
 (OUT/'MANIFEST.json').write_text(json.dumps({'artifacts_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files},'original384_sha256':before,'no_deletion':True},indent=2));print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()

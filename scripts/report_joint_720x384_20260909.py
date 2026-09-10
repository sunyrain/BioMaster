"""Interpret the completed full-space screen without changing frozen designs."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/joint_screen_720x384_20260909'
BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
DOC=ROOT/'docs/BIOMASTER_JOINT_720X384_SCREEN_20260909_ZH.md'
DESIGN=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def table(d):
 def cell(x):return str(x).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(map(cell,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell,row))+' |' for row in d.itertuples(index=False,name=None))
def main():
 assert json.loads((OUT/'VALIDATION.json').read_text())['all_pass']
 p=pd.read_parquet(OUT/'FULL_276480_PAIR_AUDIT.parquet');primary=pd.read_csv(OUT/'PRIMARY_JOINT_REVIEW.csv');priority=pd.read_csv(OUT/'PRIORITY_MULTI_EVIDENCE_REVIEW.csv');summary=json.loads((OUT/'SUMMARY.json').read_text())
 lanes=['ENZYME_BIOCHEMICAL','KINASE_BIOCHEMICAL','NUCLEAR_EPIGENETIC_DOMAIN']
 assay=primary.groupby('assay_lane').agg(pairs=('pair_id','size'),targets=('target_chembl_id','nunique'),chemical_support_ge04=('positive_chemical_support_ge04','sum'),priority=('priority_multi_evidence','sum')).reset_index()
 assay.to_csv(OUT/'ASSAY_LANE_COUNTS.csv',index=False)
 for frame in [primary,priority]:
  frame['conventional_biochemical_lane']=frame.assay_lane.isin(lanes)
  frame['assay_review_route']=np.where(frame.conventional_biochemical_lane,'CONSTRUCT_AND_ACTIVITY_REVIEW_REQUIRED','MEMBRANE_OR_CHANNEL_SYSTEM_REVIEW_REQUIRED')
  frame['treatment_direction_established']=False
  frame['novel_binding_confirmed']=False
  frame['identity_review_note']='Exact modeled InChIKey checked; TxGNN name/normalized graph mapping is not stereochemical proof.'
  frame['review_limit']='Snapshot novelty only; literature/patent, perturbation direction, achievable exposure and assay validation pending.'
 priority.to_csv(OUT/'PRIORITY_14_ASSAY_AND_DIRECTION_REVIEW.csv',index=False)
 spr=primary[primary.conventional_biochemical_lane].copy();spr.to_csv(OUT/'BIOCHEMICAL_LANE_574_REVIEW.csv',index=False)
 drugcols=['ligand_inchikey','drug_names','ligand_smiles','exact_structure_in_fda_registry','smiles_full_key_matches','active_species_status','active_species_review_hold','identity_hold','identity_concordance','mapping_rule','interpretation_eligible','identity_scope_pass','chemistry_policy_pass']
 identity=p[drugcols].drop_duplicates('ligand_inchikey');identity.to_csv(OUT/'DRUG_IDENTITY_REAUDIT_720.csv',index=False)
 bins=[]
 for lo,hi in [(1,20),(21,50),(51,79),(80,180),(181,250),(251,384)]:
  x=p[p.base_eligible&p.binding_rank_384.between(lo,hi)]
  bins.append({'binding_rank_range':f'{lo}-{hi}','eligible_pairs':len(x),'joint_pairs':int(x.joint_r50_ot03.sum()),'joint_fraction':float(x.joint_r50_ot03.mean())})
 bins=pd.DataFrame(bins);bins.to_csv(OUT/'FULL_SPACE_RANK_BAND_COMPARISON.csv',index=False)
 sensitivity=pd.read_csv(OUT/'THRESHOLD_SENSITIVITY.csv')
 ot05=int(sensitivity.loc[sensitivity.binding_rank_max.eq(20)&sensitivity.criterion.eq('joint_r50_ot05'),'pairs'].iloc[0])
 first=OUT.with_name(OUT.name+'_first_pass')
 if first.exists():
  before=pd.read_csv(first/'PRIMARY_JOINT_REVIEW.csv')
  removed=before[~before.pair_id.isin(primary.pair_id)][['pair_id','drug_names','gene_symbol']]
  removed=removed.merge(pd.read_csv(OUT/'KNOWN_MOA_COMPONENT_RECORDS.csv.gz')[['pair_id','known_target_chembl_id','known_target_name','known_target_type']].drop_duplicates(),on='pair_id',how='left')
  removed.to_csv(OUT/'FIRST_PASS_TO_FINAL_REMOVED.csv',index=False)
 original=pd.read_csv(ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv')
 comp=original[['experiment_id','pair_id','drug_names','gene_symbol','ligand_inchikey','target_chembl_id','selection_role','experiment_arm']].merge(p[['pair_id','screen_status','primary_joint_selected','binding_rank_384','all_exclusion_reasons']],on='pair_id',how='left',validate='one_to_one')
 comp['target_in384']=comp.target_chembl_id.isin(p.target_chembl_id);comp['drug_in720']=comp.ligand_inchikey.isin(p.ligand_inchikey)
 comp['comparison_scope']=np.select([comp.selection_role.eq('POSITIVE_CONTROL'),~comp.target_in384,~comp.drug_in720,comp.experiment_arm.eq('EXTENDED_367_TARGET_DISCOVERY')],['REFERENCE_CONTROL_NOT_A_DISCOVERY_CANDIDATE','TARGET_OUTSIDE384','DRUG_OUTSIDE720','IN384_BUT_PREVIOUSLY_RANKED_IN367'],default='SAME_384_RANKING_CANDIDATE')
 comp.to_csv(OUT/'ORIGINAL_512_COMPARISON_WITH_SCOPE.csv',index=False)
 changes=comp[comp.selection_role.ne('POSITIVE_CONTROL')&comp.screen_status.isin(['IDENTITY_OR_SCOPE_HOLD','KNOWN_RELATION_OR_TRAINING_EXCLUDED','PRIOR_RELATION_REVIEW_HOLD'])]
 changes.to_csv(OUT/'ORIGINAL_CANDIDATES_NEW_HOLDS.csv',index=False)
 # Independent numerical/source checks: stored ranks are checked against full raw rows,
 # not against the screen's own booleans, including old comparison and exclusion masks.
 raw=pd.read_csv(ROOT/'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz')
 order=pd.read_csv(BASE/'TXGNN_SCORE_DRUG_ORDER.csv');ds=pd.read_csv(BASE/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str});a=np.load(BASE/'TXGNN_INDICATION_LOGITS.npy');mask=np.load(BASE/'TXGNN_EXCLUSION_MASK.npy');di={k:i for i,k in enumerate(order.ligand_inchikey)}
 detail=pd.read_csv(OUT/'JOINT_DISEASE_EVIDENCE.csv.gz',dtype={'txgnn_disease_id':str});ci={k:i for i,k in enumerate(ds.id)}
 sample=priority if len(priority) else primary.head(10)
 raw_ranks=True;raw_evidence=True
 for r in sample.itertuples():
  g=raw[raw.ligand_inchikey.eq(r.ligand_inchikey)].sort_values('independent_validation_rank_score',ascending=False,kind='stable')
  raw_ranks &= list(g.target_chembl_id).index(r.target_chembl_id)+1==r.binding_rank_384
  i=di[r.ligand_inchikey];valid=np.where(mask[i]==0)[0];ranking=list(valid[np.argsort(-a[i,valid],kind='stable')])
  traces=detail[detail.pair_id.eq(r.pair_id)]
  for x in traces.itertuples():
   j=ci[x.txgnn_disease_id]
   raw_evidence &= mask[i,j]==0 and ranking.index(j)+1==x.txgnn_rank and x.txgnn_rank<=50 and x.overall_score>=.3 and abs(float(a[i,j])-x.txgnn_logit)<1e-6
 source_manifest=json.loads((OUT/'SOURCE_MANIFEST.json').read_text())
 artifact_hashes={k:sha(OUT/k)==v for k,v in source_manifest['outputs_sha256'].items()}
 checks={'initial_run_all_pass':True,'independent_raw_binding_ranks_14':bool(raw_ranks),'independent_mask_logit_disease_rank_14':bool(raw_evidence),'original_run_artifact_hashes_unchanged':all(artifact_hashes.values()),'cartesian_720x384':len(p)==276480 and p.pair_id.is_unique,'all_384_have_OT':summary['ot_targets']==384,'biochemical_plus_membrane_total':len(spr)+int((~primary.conventional_biochemical_lane).sum())==len(primary),'old_comparable_candidates406':int(comp.comparison_scope.isin(['SAME_384_RANKING_CANDIDATE','IN384_BUT_PREVIOUSLY_RANKED_IN367']).sum())==406,'no_new_512_release':bool(priority.release_status.eq('REVIEW_ONLY_NOT_RELEASED_FOR_EXPERIMENT').all())}
 (OUT/'DECISION_VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks,'original_output_hash_checks':artifact_hashes},indent=2));assert all(checks.values()),checks
 with pd.ExcelWriter(OUT/'JOINT_SCREEN_DECISION_REVIEW.xlsx') as w:
  show=['drug_names','gene_symbol','binding_rank_384','review_disease','review_txgnn_rank','review_ot_score','positive_max_tanimoto','negative_max_tanimoto','assay_lane','conventional_biochemical_lane','structure_ready_strict','assay_confirmation_status','treatment_direction_established','in_original_448_candidates','ligand_inchikey','target_chembl_id','review_limit']
  priority[show].to_excel(w,sheet_name='14对多证据人工核验',index=False);spr[show].to_excel(w,sheet_name='574对生化路线候选',index=False);assay.to_excel(w,sheet_name='实验体系分布',index=False);bins.to_excel(w,sheet_name='全空间高中低排名对照',index=False);comp.to_excel(w,sheet_name='原512同口径复审',index=False);identity.to_excel(w,sheet_name='720药身份复审',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 pp=priority[['drug_names','gene_symbol','binding_rank_384','review_disease','review_txgnn_rank','review_ot_score','positive_max_tanimoto']].copy();pp.review_ot_score=pp.review_ot_score.round(3);pp.positive_max_tanimoto=pp.positive_max_tanimoto.round(3);pp.columns=['药物','靶点','结合排名/384','共同疾病线索','TxGNN排名','OT','阳性近邻Tanimoto']
 displaybins=bins.copy();displaybins.joint_fraction=displaybins.joint_fraction.map(lambda x:f'{x:.2%}')
 outside=', '.join(sorted(original.loc[~original.target_chembl_id.isin(p.target_chembl_id),'gene_symbol'].unique()))
 report=f'''# 720×384全空间：身份／既有关系排除及TxGNN×Open Targets联合筛选

日期：2026-09-09。状态：已完成计算和结果复核；没有训练、没有补算神经网络、没有改动原512清单。原720×384结合评分与已完成的全疾病TxGNN推理均直接复用。

## 1. 结论与当前可用结果

从276,480对重新筛选，得到**{summary['primary_pairs']:,}对主筛选线索（{summary['primary_drugs']}药、{summary['primary_targets']}靶点）**，其中{summary['primary_new_vs_old_candidates']:,}对不在原448候选中，37对与原候选重叠。这说明扩大搜索范围能补充候选供应，尚不能证明新候选的实验成功率更高。

{summary['primary_pairs']:,}对中74对有已知阳性化合物Tanimoto≥0.4；进一步要求同一个共同疾病有精确单节点映射、遗传或体细胞证据≥0.1，并排除更近的高相似阴性参考后，得到**14对／13药／11靶点优先人工核验**。阈值是探索性审查规则，未校准成命中概率。

其中574对归入生化／核受体等路线，{len(primary)-len(spr)}对属于离子通道或膜转运体路线；后者不能直接并入普通SPR板。14对中12对属于生化路线、2对属于通道／转运体路线。12对生化路线中8对已有严格结构就绪标记，但**全部14对的实验构建均未获实验室或供应商确认**。结构就绪标记也不是试剂或活性合格证明。

**建议先核验14对，保留574对作为生化路线候选池，不直接替换或补满512实验孔。** 完成既有文献、分子形态、作用方向及构建核验后，再按可用靶点和药物多样性重新安排实验，保留已知阳性对照及可比较的低／中排名对照。

## 2. 实际使用的范围和规则

- 结合评分：`BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz`，720个结构实体、384靶点。使用`independent_validation_rank_score`在每药完整384靶点中重新排序，过滤前固定分母和顺序；“Top20”是相对排名，不是测得亲和力或结合概率。
- TxGNN：既有608×17,080原始logit矩阵；身份解释规则允许605药。按每药排除后疾病集合排序，排除旧图已知适应证、off-label、禁忌及EC保守治疗／试验／禁忌掩码。EC排除不等于完整的当前获批适应证登记。通过本轮身份、化学、关系规则后，有500药进入可联合解释空间；缺分数保留缺失，不填零。
- Open Targets：复用26.06全888靶点快照，本轮384靶点全部有证据；采用全部关联，不限旧64靶点、Top3或癌症。疾病映射保留精确单节点与合并节点两种状态。
- 主条件：**结合Top20/384 AND TxGNN药内疾病Top50 AND 同一疾病OT≥0.3**。不把不同疾病的三个高分相加，也不将logit和OT当可直接相加的同尺度概率。
- 身份：模型SMILES的完整InChIKey在720药中全部一致；42个结构未能按完整键接入冻结FDA表，暂扣；64个需要给药／活性物种双重审查的实体暂扣，并保留原身份歧义规则及两项旧设计例外。本轮613药通过身份／范围检查。双物种暂扣是保守实验政策，不表示这些药都无效或不能测。TxGNN名称／归一化映射仍不等于独立立体结构核验。
- 化学：MW120–700、单片段、绝对电荷≤2；MW>550、cLogP>5、TPSA>140、可旋转键>12累计风险点≤1。属性为计算值，不能代替实测溶解性、聚集或非特异结合检查。
- 既有关系：重扫ChEMBL37严格校准对（包括阴性）、综合训练关系、KirHub、冻结FDA/ChEMBL已知机制（含家族／复合物到人源蛋白组分映射）、BindingDB两份2026-07快照及GtoPdb2026.2。按完整药物键及靶点／UniProt匹配；BindingDB多链记录命中任一相关链也保守排除，不据此宣称已经证实该单链直接结合。家族／复合物命中保守排除新颖性，不宣称每个组分均已证明直接结合；外部扫描沿用本地TSV解析范围，不是全网、专利或所有化学等价物的穷尽查新。
- EC：非文本、无映射歧义的物理关系进入已知关系排除；其余来源断言（表达、调控、关联等）以及部分映射歧义另设待审区，不把它们当已证实结合。文本／预测边保留为审计信息，不伪装成新增实验依据。

最终复核补上家族／复合物组分排除，剔除首轮遗漏的7个钠通道家族配对；14对优先项不变。首轮输出保留在`joint_screen_720x384_20260909_first_pass`供差异追溯，以下均为修正后结果。

## 3. 筛选漏斗

{table(pd.read_csv(OUT/'SEQUENTIAL_FUNNEL.csv'))}

表中按顺序累计过滤，因此{int(pd.read_csv(OUT/'SEQUENTIAL_FUNNEL.csv').set_index('stage').loc['chemistry_policy','pairs']-pd.read_csv(OUT/'SEQUENTIAL_FUNNEL.csv').set_index('stage').loc['known_relation_exclusion','pairs']):,}对“既有关系”排除仅指通过前面身份和化学规则后的增量。各来源原始命中有重叠，不能相加：

{table(pd.read_csv(OUT/'EXCLUSION_COUNTS_OVERLAPPING.csv'))}

全276,480对均保留在Parquet，含逐来源标记、身份状态、排除原因、联合条件与原方案重叠信息。未知不自动算新关系，未命中本地快照也不等于首次发现。

## 4. 实验体系与14对人工核验清单

{table(assay)}

{table(pp)}

表内共同疾病是复核线索，**没有证明该药通过该靶点治疗该疾病**。遗传关联可能反映致病或保护作用，TxGNN没有建立分子→该靶点→疾病的因果路径，OT总分也不提供药物需要激动还是抑制的结论。即使结合真实，方向、选择性与可达到暴露仍可能不合适。

优先逐对查原始药理／专利及具体作用方向，特别是综合模型或图谱可能已经见过同家族机制的配对。pitolisant–SLC6A4、esmolol–KCNH2进入专门膜／通道体系审查；不是普通可溶性蛋白SPR直接候选。vismodegib–NTRK1、mebendazole–MET、lisinopril–F2、nilotinib–IDH1当前缺严格结构就绪标记，需要另核构建；其余有标记也未确认试剂。lisinopril–F2的最近阳性和阴性相似度相同，不能只展示阳性支持。

14对中只有tecovirimat–AR、lisinopril–F2出现在原448候选中；其余12对是本次新增的优先审查项。化学近邻基于完整ChEMBL37参考库，包含历史训练相关信息，不是时间留出验证，也不是独立实验复现。

## 5. 扩大范围后，证据是否真的更好？

所有行统一经过本轮身份、化学、关系与TxGNN可解释性规则，再按原384排名分组：

{table(displaybins)}

**Top20联合比例{bins.iloc[0].joint_fraction:.2%}，80–180组{bins.iloc[3].joint_fraction:.2%}，251–384组{bins.iloc[5].joint_fraction:.2%}。** 这组描述性结果没有显示高结合排名与共同疾病图谱支持之间存在明显梯度。配对共享药物、靶点及训练信息，不能当作独立样本计算简单显著性；也没有SPR标签可评估真实结合性能。

联合筛选主动挑选满足条件的行，筛选后共同疾病比例更高是规则本身造成的，不能作为预测性能提升的证据。当前变化主要是扩大候选供应、让每对假设可追溯；尚未验证实验命中率提高。

将结合排名放宽到Top50，会得到{summary['rank50_pairs']:,}对、{summary['rank50_targets']}靶点，但本轮没有必要先牺牲结合排名来凑数。保持Top20，仅把OT门槛提高到0.5仍有{ot05}对，但它们并非都适合SPR或满足化学支持。完整敏感性表见`THRESHOLD_SENSITIVITY.csv`。

## 6. 与原512的公平对比

原512=448候选+64对照。实际有406个候选配对落在720×384中；其中364对原本就是384排序路线，42对之前采用扩展367路线，重排后口径变化需要单列。另42候选对所在6靶点不在384中：**{outside}**。不能把“12个扩展路线靶点”误写成“12个都不在历史384中”。

本轮{summary['primary_pairs']:,}对中37对与原448重叠；原在范围内候选有52对进入身份／范围暂扣、4对新增既有关系排除、7对进入来源关系待审。具体逐对原因在`ORIGINAL_CANDIDATES_NEW_HOLDS.csv`。不同筛选范围与身份政策下，不能直接将37与上轮全64靶点的44作性能退步比较。

原64参考对照不按新候选规则筛选，目录外对照也不能因为不在720中就标成无效。逐行对比表显式标注其角色及范围。本次仅产生独立审查输出，原512配对、角色、排序和哈希不变。

## 7. 交付、验证和下一步

- [决策工作簿](../outputs/joint_screen_720x384_20260909/JOINT_SCREEN_DECISION_REVIEW.xlsx)：14对、574生化路线候选、实验体系、全空间排名对照、原512复审、720身份审计。
- [完整筛选工作簿](../outputs/joint_screen_720x384_20260909/JOINT_720X384_REVIEW.xlsx)：{summary['primary_pairs']:,}主线索、{summary['rank50_pairs']:,}放宽线索及阈值敏感性。
- `FULL_276480_PAIR_AUDIT.parquet`：全空间逐对审计；`JOINT_DISEASE_EVIDENCE.csv.gz`：放宽候选的全部合格共同疾病、原始logit、排除后排名、OT分来源和映射类型。
- `KNOWN_RELATION_SOURCE_LEDGER.csv.gz`、BindingDB/GtoPdb原始匹配记录及`ECKG_ALL_MATCHING_RELATION_ROWS.parquet`：排除依据可追溯。
- 运行脚本：`scripts/screen_joint_720x384_20260909.py`；解释报告脚本：`scripts/report_joint_720x384_20260909.py`。
- 主运行10项检查全部通过；另对14对直接回读原始384评分、TxGNN矩阵和疾病掩码核验，原输出哈希及比较口径检查均通过。见`VALIDATION.json`和`DECISION_VALIDATION.json`。

下一步顺序：先对14对做证据与作用方向核验；按可用构建挑选少量药物／靶点进行试剂与检测体系预实验；再从574对中在固定总预算下做多样性和对照配置。应保留部分原规则候选形成可解释对照，才能在获得真实实验标签后判断联合筛选有没有收益。本次未创建新的512冻结清单、未安排实验，也未修改系统线上候选。
'''
 DOC.write_text(report)
 snapshot=OUT/'DESIGN_BEFORE_FULL_SPACE_SCREEN.md'
 if not snapshot.exists():snapshot.write_bytes(DESIGN.read_bytes())
 header='\n## 10. 720×384全空间联合筛选'
 oldtext=DESIGN.read_text();prefix=oldtext.split(header)[0]
 DESIGN.write_text(prefix+header+f'''

已完成[全空间身份／既有关系复审与TxGNN×OT联合筛选](BIOMASTER_JOINT_720X384_SCREEN_20260909_ZH.md)，直接查看[决策工作簿](../outputs/joint_screen_720x384_20260909/JOINT_SCREEN_DECISION_REVIEW.xlsx)。

276,480对重新筛选后，结合Top20/384、TxGNN疾病Top50与同病OT≥0.3共同得到{summary['primary_pairs']:,}对（{summary['primary_drugs']}药／{summary['primary_targets']}靶点）；其中574对归入生化／核受体路线，{len(primary)-len(spr)}对为通道／膜转运体路线。额外同病精确映射、遗传／体细胞与化学近邻门槛筛出14对人工核验，未放行实验。全空间Top20共同疾病比例为{bins.iloc[0].joint_fraction:.2%}、80–180组为{bins.iloc[3].joint_fraction:.2%}，尚未显示联合证据随结合排名明显增强，不能把选后证据比例当作命中率提高。

范围更正：原12个扩展路线靶点中只有6个不在历史384；原448候选有406对可在本轮空间复审，42对在范围外。双物种药物新增保守暂扣，规则变化单独记录。原512不变；本节追加前快照与哈希链见新目录`DECISION_MANIFEST.json`。
''')
 artifacts=[DOC,DESIGN,Path(__file__),snapshot,*[OUT/f for f in ['ASSAY_LANE_COUNTS.csv','PRIORITY_14_ASSAY_AND_DIRECTION_REVIEW.csv','BIOCHEMICAL_LANE_574_REVIEW.csv','DRUG_IDENTITY_REAUDIT_720.csv','FULL_SPACE_RANK_BAND_COMPARISON.csv','ORIGINAL_512_COMPARISON_WITH_SCOPE.csv','ORIGINAL_CANDIDATES_NEW_HOLDS.csv','DECISION_VALIDATION.json','JOINT_SCREEN_DECISION_REVIEW.xlsx']]]
 if (OUT/'FIRST_PASS_TO_FINAL_REMOVED.csv').exists():artifacts.append(OUT/'FIRST_PASS_TO_FINAL_REMOVED.csv')
 (OUT/'DECISION_MANIFEST.json').write_text(json.dumps({'parent_run_manifest_sha256':sha(OUT/'SOURCE_MANIFEST.json'),'design_before_sha256':sha(snapshot),'design_after_sha256':sha(DESIGN),'artifacts_sha256':{str(f.relative_to(ROOT)):sha(f) for f in artifacts}},indent=2))
 print(json.dumps({'all_pass':True,'primary':len(primary),'biochemical':len(spr),'priority':len(priority),'priority_biochemical':int(priority.conventional_biochemical_lane.sum()),'comparison_scope_counts':comp.comparison_scope.value_counts().to_dict(),'new_holds':len(changes)},indent=2))
if __name__=='__main__':main()

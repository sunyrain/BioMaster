"""Export frozen384 with explicit experimental priority and exact-target reference evidence."""
import json,hashlib,sqlite3,sys
from pathlib import Path
from collections import Counter
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_spr_expansion import baseline_reasons
OUT=ROOT/'outputs/spr384_priority_export_20260910';OUT.mkdir(exist_ok=True)
BASE=ROOT/'outputs/joint384_comprehensive_20260909'
source=BASE/'RECOMMENDED_CANDIDATES_384.csv';x=pd.read_csv(source).fillna('');reg=pd.read_csv(ROOT/'outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv').fillna('').set_index('ligand_inchikey')
controls=pd.read_csv(BASE/'REFERENCE_CONTROLS_EXTRA.csv').fillna('').set_index('target_chembl_id');actions=baseline_reasons(ROOT)
c=sqlite3.connect(f"file:{ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'}?mode=ro",uri=True);c.row_factory=sqlite3.Row
ids=x.target_chembl_id.unique().tolist();holders=','.join('?' for _ in ids)
targets={r['chembl_id']:dict(r) for r in c.execute(f'SELECT chembl_id,pref_name,target_type,organism FROM target_dictionary WHERE chembl_id IN ({holders})',ids)}
query=f'''SELECT td.chembl_id target_chembl_id,td.pref_name target_name,md.chembl_id drug_chembl_id,md.pref_name drug_name,md.max_phase,md.withdrawn_flag,md.molecule_type,dm.action_type,dm.mechanism_of_action,dm.direct_interaction,dm.mec_id
FROM target_dictionary td JOIN drug_mechanism dm ON td.tid=dm.tid JOIN molecule_dictionary md ON md.molregno=dm.molregno
WHERE td.chembl_id IN ({holders}) AND dm.direct_interaction=1 AND md.max_phase=4 AND COALESCE(md.withdrawn_flag,0)=0 ORDER BY td.chembl_id,md.pref_name,md.chembl_id'''
moa=[dict(r) for r in c.execute(query,ids)];pd.DataFrame(moa).to_csv(OUT/'NEW_TARGET_KNOWN_DRUG_EVIDENCE.csv',index=False,encoding='utf-8-sig')
known={t:[] for t in ids}
for a in moa:
 name=a['drug_name'] or a['drug_chembl_id']
 if name not in known[a['target_chembl_id']]:known[a['target_chembl_id']].append(name)
rows=[]
for _,a in x.iterrows():
 r=reg.loc[a.ligand_inchikey];b=actions[a.pair_id];ctrl=controls.loc[a.target_chembl_id];ac=b['action_class'];rank=int(a.binding_rank_384)
 tier=1 if ac=='EXPLORABLE' and rank<=10 else 2 if ac=='EXPLORABLE' else 3 if ac=='FIRST_RESOLVE' else 4
 labels={1:'P1 前10排名且仍可探索',2:'P2 第11–20排名且仍可探索',3:'P3 先解决具体问题',4:'P4 降低投入／暂缓首批'}
 name=a.modeled_entity_name;note='沿用冻结名单名称；采购时仍须核对结构、盐形、纯度。'
 if r.gsrs_inchikey==a.ligand_inchikey and r.gsrs_display_name:
  name=r.gsrs_display_name.lower()
  if name!=a.modeled_entity_name:note='导出名按GSRS完整InChIKey对应实体显示；原冻结名称另列，配对与审查暂停条件不变。'
 if a.ligand_inchikey=='MQOBSOSZFYZQOK-UHFFFAOYSA-N':
  name='fenofibric acid';note='完整InChIKey对应ChEMBL981酸实体；源标签为choline fenofibrate，采购形式必须核实。'
 if a.candidate_id=='C384-289':note='模型为NICOTINE小分子，源名称为nicotine polacrilex树脂制剂；实际待测实体未确认，仍属P3，不能因更正显示名视为问题已解除。'
 old=r.known_target_names or '当前机制库未收录，不能据此认定无已知靶点'
 names='; '.join(known[a.target_chembl_id]) or '本地ChEMBL37未检出符合口径的药物；见另列参考化合物'
 rows.append({'排序':0,'优先级':labels[tier],'原候选编号':a.candidate_id,'小分子药物名称':name,'旧靶点名称':old,'新靶点名称':a.gene_symbol+' — '+targets[a.target_chembl_id]['pref_name'],'新靶点的已知药物名称':names,'新靶点实验参考化合物名称':ctrl.drug_names or ctrl.control_chembl_id,'结合排名_每药384靶点':rank,'当前实验建议':b['action_label'],'需先确认的事项':b['critical_condition'],
 '评价理由':b['reason_detail'],'原冻结药物名称':a.modeled_entity_name,'名称与实物核验说明':note,'原适应症':a.origin_summary,'目标适应症假设':a.recommended_disease,'跨原适应症大类':bool(a.recommended_disease_is_cross_area),'药物完整InChIKey':a.ligand_inchikey,'旧靶点ChEMBL编号':r.known_target_chembl_ids,'旧作用机制':r.known_mechanism_of_action,'新靶点基因':a.gene_symbol,'新靶点ChEMBL编号':a.target_chembl_id,'参考化合物ChEMBL编号':ctrl.control_chembl_id,'参考化合物端点类型':ctrl.reference_standard_types,'参考化合物证据层级':ctrl.reference_endpoint_category,'参考化合物实验编号':ctrl.reference_assay_ids,'参考化合物文献编号':ctrl.reference_doc_ids,'已知药物作用类型':'; '.join(dict.fromkeys((m['drug_name'] or m['drug_chembl_id'])+': '+str(m['action_type'] or '未标明') for m in moa if m['target_chembl_id']==a.target_chembl_id)),'已知药物查询口径':'ChEMBL37：精确靶点ID＋直接机制记录＋max_phase=4＋未标记撤市；不保证当前上市状态，也不证明适合作SPR对照','参考化合物说明':'来自冻结另计对照；可能为研究化合物，非必然获批药；尚需确认SPR适用性','阳性近邻Tanimoto':a.positive_max_tanimoto,'可用阳性近邻_ge04':bool(a.usable_chemical_support_ge04),'TxGNN疾病排名':a.recommended_txgnn_rank,'OT疾病关联分':a.recommended_ot_score,'pair_id':a.pair_id,'_tier':tier})
rows.sort(key=lambda a:(a['_tier'],a['结合排名_每药384靶点'],-int(a['可用阳性近邻_ge04']),-int(a['跨原适应症大类']),a['原候选编号']))
for i,r in enumerate(rows,1):r['排序']=i;r.pop('_tier')
d=pd.DataFrame(rows);d.to_csv(OUT/'SPR384_PRIORITY_DETAILED.csv',index=False,encoding='utf-8-sig')
cols=['排序','优先级','原候选编号','小分子药物名称','旧靶点名称','新靶点名称','新靶点的已知药物名称','新靶点实验参考化合物名称','结合排名_每药384靶点','当前实验建议','需先确认的事项','名称与实物核验说明']
d[cols].to_csv(OUT/'SPR384_PRIORITY_LIST.csv',index=False,encoding='utf-8-sig')
checks={'384_unique_pairs':len(d)==d.pair_id.nunique()==384,'same_frozen_pairs':set(d.pair_id)==set(x.pair_id),'four_requested_fields_present':bool(d[cols[3:7]].notna().all().all()),'all112_reference_targets':len(controls)==112 and set(x.target_chembl_id)<=set(controls.index),'priority_counts':d['优先级'].value_counts().to_dict(),'baseline_sha_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()=='bf1ac323287c3b54c4ca92a5326e9da3e39f355fad60f2ac3edf672d0c4ef042'}
assert all(checks[k] for k in ['384_unique_pairs','same_frozen_pairs','four_requested_fields_present','all112_reference_targets','baseline_sha_unchanged'])
assert [sum(r['优先级'].startswith('P'+str(i)) for r in rows) for i in (1,2,3,4)]==[100,123,149,12]
checks['known_drug_targets']=sum(bool(v) for v in known.values());checks['candidate_rows_with_known_drug']=sum(bool(known[t]) for t in x.target_chembl_id)
(OUT/'VALIDATION.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
(OUT/'README.md').write_text('''# 冻结SPR384优先级导出

主表SPR384_PRIORITY_LIST.csv包含所需四项名称、原编号、优先级、参考化合物和关键条件；详细表附机制、疾病、精确身份、原始端点与来源编号。UTF-8 BOM编码，可用Excel打开。

排序：P1前10且仍可探索100条；P2第11–20且仍可探索123条；P3先解决149条；P4降低投入12条。同级按结合排名升序、可用阳性化学支持优先、跨领域优先、原编号稳定排序。这是透明实验安排顺序，不是校准命中概率。P1也须蛋白体系/对照确认，不表示上机放行。

旧靶点取720分子机制注册表精确InChIKey记录，可能包含复合物、非人蛋白或作用物种限制，空缺明确标注。新靶点名称取ChEMBL37精确ID。新靶点已知药物通过直接机制记录筛选max_phase=4且未标记撤市；这不是全球完整药物清单或当前上市状态证明。精确靶点无结果时不从同家族、复合物或间接通路推补。已知药物列表可能同时含激动剂、拮抗剂及次要靶点机制，不表示作用方向相同或可互换，详细表另列作用类型。

另列冻结设计参考化合物，覆盖112靶点，可能只是研究化合物。它们不等同于新靶点已知获批药物。历史Kd/Ki/IC50端点不保证SPR适用性，也未把混合端点均值换算成Kd。

小分子显示名尽量按完整InChIKey对应GSRS实体，保留冻结原名和变更说明；fenofibric acid按已核查ChEMBL981显示。nicotine树脂制剂问题仍保留P3。未修改冻结清单、配对及对照，也未把新增384证据池混入。

来源：outputs/joint384_comprehensive_20260909/{RECOMMENDED_CANDIDATES_384.csv,REFERENCE_CONTROLS_EXTRA.csv}、outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv、outputs/spr_expanded_review_20260910/BASELINE_REASON_{1,2}.json、downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db（只读）。查询明细见NEW_TARGET_KNOWN_DRUG_EVIDENCE.csv。
''')
(OUT/'MANIFEST.json').write_text(json.dumps({'baseline_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name!='MANIFEST.json'},'experimental_release':False},ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False));print(d[cols[:8]].head(5).to_json(orient='records',force_ascii=False))

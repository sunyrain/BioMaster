"""Final priority table: fourth required field is the nominated per-target control."""
from pathlib import Path
import hashlib,json,sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/spr384_final_experiment_table_20260910'
BASE=ROOT/'outputs/joint384_comprehensive_20260909'
PRIOR=ROOT/'outputs/spr384_priority_export_20260910/SPR384_PRIORITY_DETAILED.csv'
FIELDS=['小分子药物名称','旧靶点名称','新靶点名称','新靶点的已知药物名称']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 # Once the curated revision exists, regenerate that version instead of reverting controls.
 if (OUT/'MANIFEST.json').exists() and json.loads((OUT/'MANIFEST.json').read_text()).get('version')=='20260910_FDA_V2':
  from finalize_spr384_fda_controls_20260910 import main as revised_main
  return revised_main()
 OUT.mkdir(exist_ok=True)
 d=pd.read_csv(PRIOR).fillna('');controls=pd.read_csv(BASE/'REFERENCE_CONTROLS_EXTRA.csv').fillna('');baseline=pd.read_csv(BASE/'RECOMMENDED_CANDIDATES_384.csv')
 c=sqlite3.connect(f"file:{ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'}?mode=ro",uri=True);c.row_factory=sqlite3.Row
 cr=[];evidence=[]
 for index,r in controls.iterrows():
  md=c.execute('select md.pref_name,md.max_phase,md.molecule_type,cs.standard_inchi_key from molecule_dictionary md join compound_structures cs using(molregno) where md.chembl_id=?',(r.control_chembl_id,)).fetchone()
  assert md and md['standard_inchi_key']==r.ligand_inchikey
  synonyms=[x[0] for x in c.execute('select synonyms from molecule_synonyms join molecule_dictionary using(molregno) where chembl_id=? order by synonyms',(r.control_chembl_id,)) if not x[0].startswith('CHEMBL')]
  name=md['pref_name'] or (synonyms[0] if synonyms else '')
  unresolved=not bool(name);name=name or f'研究化合物 {r.control_chembl_id}（无通用名称）'
  query='''select ac.activity_id,ac.standard_type,ac.standard_relation,ac.standard_value,ac.standard_units,ac.data_validity_comment,a.assay_id,a.chembl_id assay_chembl_id,a.assay_type,a.confidence_score,a.relationship_type,a.description,doc.chembl_id document_chembl_id,doc.doi,doc.pubmed_id from activities ac join assays a using(assay_id) join molecule_dictionary md on md.molregno=ac.molregno join target_dictionary td on td.tid=a.tid left join docs doc on doc.doc_id=ac.doc_id where md.chembl_id=? and td.chembl_id=? and ac.standard_type in ('Kd','Ki','IC50','EC50')'''
  assays=set(str(r.reference_assay_ids).split(','));rows=[dict(x) for x in c.execute(query,(r.control_chembl_id,r.target_chembl_id))]
  selected=[x for x in rows if str(x['assay_id']) in assays]
  exact_evidence_found=bool(selected)
  for x in selected:evidence.append(dict(target_chembl_id=r.target_chembl_id,control_chembl_id=r.control_chembl_id,control_name=name,**x))
  category=r.reference_endpoint_category
  status=('有Kd记录，待确认构建与SPR实测活性' if category=='Kd_RECORD_PRESENT' else '仅Ki记录，需先确认SPR结合适用性' if category=='Ki_NO_Kd_RECORD' else '仅功能/其他活性记录，SPR对照尚未落实')
  if unresolved:status+='；须按结构落实实物'
  if not exact_evidence_found:status+='；冻结来源未匹配到此精确实体的原始活动，须先复核'
  cr.append({'对照编号':f'CTRL-{index+1:03d}','新靶点名称':d.loc[d['新靶点ChEMBL编号']==r.target_chembl_id,'新靶点名称'].iloc[0],'新靶点的已知药物名称':name,'精确对照原始证据已匹配':exact_evidence_found,'对照身份':'通用名称已收录' if not unresolved else '仅研究编号，实物待落实','对照证据类型':r.reference_standard_types,'对照确认状态':status,'对照ChEMBL编号':r.control_chembl_id,'对照完整InChIKey':r.ligand_inchikey,'对照SMILES':r.ligand_smiles,'新靶点ChEMBL编号':r.target_chembl_id,'新靶点基因':r.gene_symbol,'原对照名称':r.drug_names,'参考证据层级':category,'真实ChEMBL实验编号':'; '.join(sorted({x['assay_chembl_id'] for x in selected})),'真实ChEMBL文献编号':'; '.join(sorted({x['document_chembl_id'] for x in selected if x['document_chembl_id']})),'证据文献DOI':'; '.join(sorted({x['doi'] for x in selected if x['doi']})),'原pair_id':r.pair_id,'对照用途':'每个靶点另计参考；不计入384候选；不是384次重复采购'})
 ct=pd.DataFrame(cr);ct.to_csv(OUT/'SPR112_REFERENCE_CONTROLS.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(evidence).to_csv(OUT/'CONTROL_SOURCE_ACTIVITY_EVIDENCE.csv',index=False,encoding='utf-8-sig')
 lookup=ct.set_index('新靶点ChEMBL编号')
 d=d.rename(columns={'新靶点的已知药物名称':'靶点已知药物目录_非指定对照'})
 for col in ['新靶点的已知药物名称','对照编号','对照身份','对照证据类型','对照确认状态','对照ChEMBL编号','对照完整InChIKey','真实ChEMBL实验编号','真实ChEMBL文献编号','证据文献DOI']:
  d[col]=d['新靶点ChEMBL编号'].map(lookup[col])
 d['新靶点实验参考化合物名称']=d['新靶点的已知药物名称']
 d['第四列用途']='本行新靶点的拟用实验参考对照；参见对照确认状态'
 d=d.sort_values('排序');short=['排序','优先级','原候选编号',*FIELDS,'对照编号','对照证据类型','对照确认状态','当前实验建议','需先确认的事项','名称与实物核验说明','药物完整InChIKey','新靶点ChEMBL编号','对照ChEMBL编号','对照完整InChIKey']
 d[short].to_csv(OUT/'SPR384_FINAL_EXPERIMENT_TABLE.csv',index=False,encoding='utf-8-sig')
 d[FIELDS].to_csv(OUT/'SPR384_FINAL_FOUR_COLUMNS.csv',index=False,encoding='utf-8-sig')
 d.to_csv(OUT/'SPR384_FINAL_DETAILED.csv',index=False,encoding='utf-8-sig')
 with pd.ExcelWriter(OUT/'SPR384_FINAL_EXPERIMENT_TABLE.xlsx') as w:
  d[short].to_excel(w,sheet_name='384候选_按优先级',index=False);ct.to_excel(w,sheet_name='112对照_另计',index=False)
  for ws in w.sheets.values():ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 checks={'candidate_rows':len(d),'candidate_pairs':d.pair_id.nunique(),'targets':d['新靶点ChEMBL编号'].nunique(),'control_rows':len(ct),'same_frozen_candidate_set':set(d.pair_id)==set(baseline.pair_id),'same_frozen_control_set':set(ct['原pair_id'])==set(controls.pair_id),'four_fields_complete':bool(d[FIELDS].notna().all().all()),'priority_unchanged':d['原候选编号'].tolist()==pd.read_csv(PRIOR)['原候选编号'].tolist(),'unnamed_research_controls':int(ct['对照身份'].str.startswith('仅研究').sum()),'control_endpoint_categories':ct['参考证据层级'].value_counts().to_dict(),'unique_control_compounds':ct['对照完整InChIKey'].nunique(),'exact_control_activity_unresolved':int((~ct['精确对照原始证据已匹配']).sum())}
 assert checks['candidate_rows']==checks['candidate_pairs']==384 and checks['targets']==checks['control_rows']==112
 assert all(checks[k] for k in ['same_frozen_candidate_set','same_frozen_control_set','four_fields_complete','priority_unchanged'])
 (OUT/'README.md').write_text('''# SPR384最终实验安排表

第四项“新靶点的已知药物名称”现在是实际指定的拟用参考对照，不是所有已知上市药的目录。一个靶点对应一个对照，112条对照单独列出；同一对照在候选表重复显示不表示需要按候选数重复采购。候选384及112对照的精确身份沿用冻结设计，排序沿用优先级导出。

推荐使用SPR384_FINAL_EXPERIMENT_TABLE.csv或Excel（附112对照分表）；SPR384_FINAL_FOUR_COLUMNS.csv仅含用户要求的四列。只含四列的表不包含实验暂停条件，实际安排请同时使用主表/Excel。研究编号无通用名的如实标注；这些是待落实实物的参考，不能称作已知获批药。端点只有Ki或功能活性的条目不能自动视为SPR阳性；有Kd也需要实验室确认构建和对照活性。此表为最终设计安排，不是实测结果或自动实验放行。

对照原始activity与真正的ChEMBL assay/document编号见CONTROL_SOURCE_ACTIVITY_EVIDENCE.csv，避免把数据库内部数字ID直接拼成CHEMBL编号。原已知药物目录保存在详细表的“靶点已知药物目录_非指定对照”列。
''')
 manifest=dict(checks=checks,baseline_sha256=sha(BASE/'RECOMMENDED_CANDIDATES_384.csv'),control_baseline_sha256=sha(BASE/'REFERENCE_CONTROLS_EXTRA.csv'),priority_source_sha256=sha(PRIOR),output_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name!='MANIFEST.json'})
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()

"""Attach final candidates and versioned controls with exact identity and lineage."""
from pathlib import Path
import csv,hashlib,json,io
DIRECTORY='outputs/spr384_final_experiment_table_20260910'
def load_final(root):
 directory=Path(root)/DIRECTORY
 if not (directory/'MANIFEST.json').exists():return {},{}
 manifest=json.loads((directory/'MANIFEST.json').read_text())
 tables=[]
 for name,key in [('SPR384_FINAL_DETAILED.csv','pair_id'),('SPR112_REFERENCE_CONTROLS.csv','原pair_id')]:
  data=(directory/name).read_bytes()
  if hashlib.sha256(data).hexdigest()!=manifest['output_sha256'][name]:raise ValueError('Final SPR table checksum mismatch')
  rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
  tables.append({r[key]:r for r in rows})
 assert len(tables[0])==384 and len(tables[1])==112
 return tuple(tables)
def final_items(root,items,final_tables=None):
 candidates,controls=final_tables if final_tables is not None else load_final(root)
 result=[]
 for item in items:
  r=(controls if item.get('is_control') else candidates).get(item.get('baseline_pair_id',item.get('pair_id')))
  if not r:result.append(item);continue
  item={**item,'final_table_version':r.get('对照修订版本','20260910'),'control_drug_name':r['新靶点的已知药物名称'],'reference_control':r['新靶点的已知药物名称'],'control_id':r['对照编号'],'control_status':r['对照确认状态'],'control_endpoint_types':r['对照证据类型'],'control_compound_chembl_id':r['对照ChEMBL编号'],'control_compound_inchikey':r['对照完整InChIKey'],'proposed_target':r['新靶点名称'],
        'control_assay_urls':['https://www.ebi.ac.uk/chembl/explore/assay/'+x.strip() for x in r['真实ChEMBL实验编号'].split(';') if x.strip()],
        'control_fda_status':r.get('FDA身份状态',''),'control_fda_urls':[x.strip() for x in r.get('FDA审批来源','').split(';') if x.strip()],
        'control_construct_requirement':r.get('对照构建适用条件',''),
        'control_replaced':r.get('本次FDA替换','').lower()=='true',
        'control_selected_endpoint':r.get('选用参考端点',''),'control_selected_value_nM':r.get('选用参考值_nM',''),
        'control_selected_assay_description':r.get('选用参考实验描述','')}
  if item.get('is_control'):
   item['drug_name']=r['新靶点的已知药物名称'];item['name']=item['drug_name']+' → '+item['gene_symbol']
   item.update(baseline_pair_id=r['原pair_id'],pair_id=r.get('pair_id',r['原pair_id']),
       drug_id=r['对照完整InChIKey'],id=r['对照编号'],experiment_id=r['对照编号'],
       source_path=DIRECTORY+'/SPR112_REFERENCE_CONTROLS.csv',
       control_evidence={'control_chembl_id':r['对照ChEMBL编号'],'reference_endpoint_category':r['参考证据层级'],
          'reference_standard_types':r['对照证据类型'],'reference_assay_ids':r['真实ChEMBL实验编号'],
          'reference_doc_ids':r['真实ChEMBL文献编号'],'说明':'选用记录或冻结来源；不代表本批SPR已经测通。'})
   if item['control_replaced']:
    item.update(design_id='COMPREHENSIVE384_FDA_CONTROLS_20260910',design_date='2026-09-10',
        previous_experiment_id=r['原对照编号'],previous_drug_name=r['替换前对照名称'],
        original_indications='FDA获批参考药物；具体批准适应症见药物档案或FDA来源',
        original_targets=r.get('对照原靶点名称') or '当前机制库未收录，见本靶点选用实验记录',
        original_target_ids=r.get('对照原靶点编号',''),original_mechanism=r.get('对照原作用机制',''),
        original_targets_source='outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv',
        endpoint_plan='拟用参考对照；按选用端点与构建条件先完成SPR质控',
        pair_review_status=r.get('替换依据',''),review={'对照替换依据':r.get('替换依据','')})
  else:
   item.update(drug_name=r['小分子药物名称'],original_targets=r['旧靶点名称'],final_priority_order=int(r['排序']),final_priority_label=r['优先级'],identity_note=r['名称与实物核验说明'])
   item['name']=item['drug_name']+' → '+item['gene_symbol']
  result.append(item)
 return sorted(result,key=lambda x:(bool(x.get('is_control')),x.get('final_priority_order',9999),x.get('experiment_id','')))

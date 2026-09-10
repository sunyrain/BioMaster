"""Prepare exact SPR512 identities; integrate existing and supplementary frozen inference."""
import json,re,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd,numpy as np,requests
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909';OUT=ROOT/'outputs/spr512_integrated_disease_20260909'
def norm(x):return re.sub('[^a-z0-9]','',str(x).lower())
def prepare():
 OUT.mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
 master=pd.read_csv(ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv');drugs=master.drop_duplicates('ligand_inchikey');old=pd.read_csv(BASE/'DRUG_COVERAGE_FINAL_720.csv').fillna('').set_index('ligand_inchikey');nodes=pd.read_csv(ROOT/'data/raw/txgnn/node.csv',dtype=str);nodes=nodes[nodes.node_type.eq('drug')];nodes['norm']=nodes.node_name.map(norm)
 def lookup(r):
  if r.ligand_inchikey in old.index:
   x=old.loc[r.ligand_inchikey];return {'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'graph_drug_id':x.graph_drug_id,'status':'REUSE_VERIFIED_RUN' if x.interpretation_eligible else 'UNMAPPED_OR_IDENTITY_HOLD','mapping_rule':x.mapping_rule}
  p=OUT/'raw'/f'{r.ligand_inchikey}.json'
  if not p.exists():
   try:
    resp=requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{r.ligand_inchikey}/synonyms/JSON',timeout=40);j={'status':resp.status_code,'body':resp.json()}
   except Exception as e:j={'status':'ERROR','error':str(e)}
   p.write_text(json.dumps(j))
  j=json.loads(p.read_text());syn=[]
  for x in j.get('body',{}).get('InformationList',{}).get('Information',[]):syn+=x.get('Synonym',[])
  match=nodes[nodes.norm.isin(set(map(norm,syn)))|nodes.node_id.isin(syn)].drop_duplicates('node_id');ok=len(match)==1
  return {'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'graph_drug_id':match.iloc[0].node_id if ok else '', 'status':'SUPPLEMENTARY_INFERENCE' if ok else ('AMBIGUOUS_HOLD' if len(match)>1 else 'UNMAPPED'),'mapping_rule':'FULL_INCHIKEY_PUBCHEM_SYNONYM;GRAPH_STRUCTURE_NOT_VERIFIED'}
 with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(lookup,drugs.itertuples()))
 c=pd.DataFrame(rows);c.to_csv(OUT/'DRUG_COVERAGE.csv',index=False);print(c.status.value_counts().to_dict())


def build():
 master=pd.read_csv(ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv');cov=pd.read_csv(OUT/'DRUG_COVERAGE.csv').fillna('').set_index('ligand_inchikey');ds=pd.read_csv(BASE/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str});di={str(x):i for i,x in enumerate(ds.id)}
 oldorder=pd.read_csv(BASE/'TXGNN_SCORE_DRUG_ORDER.csv');oldscore=np.load(BASE/'TXGNN_INDICATION_LOGITS.npy');oldmask=np.load(BASE/'TXGNN_EXCLUSION_MASK.npy');order=pd.read_csv(OUT/'TXGNN_SCORE_DRUG_ORDER.csv');extra=np.load(OUT/'TXGNN_INDICATION_LOGITS.npy');assert list(pd.read_csv(OUT/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str}).id)==list(ds.id)
 scores={k:oldscore[i] for i,k in enumerate(oldorder.ligand_inchikey)};scores.update({k:extra[i] for i,k in enumerate(order.ligand_inchikey)});masks={k:oldmask[i] for i,k in enumerate(oldorder.ligand_inchikey)}
 ot=pd.read_parquet(BASE/'OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet');ot=ot[ot.target_chembl_id.isin(master.target_chembl_id)];ot.to_parquet(OUT/'OPENTARGETS_64_FULL_ASSOCIATIONS.parquet',index=False)
 relations=pd.read_csv(BASE/'SPR512_ECKG_RELATION_REVIEW.csv');relations.to_csv(OUT/'SPR512_ECKG_RELATION_REVIEW.csv',index=False)
 rows=[];long=[]
 for r in master.to_dict('records'):
  key=r['ligand_inchikey'];target=r['target_chembl_id'];control=r['selection_role']=='POSITIVE_CONTROL';status=cov.loc[key,'status'];available=status in ['REUSE_VERIFIED_RUN','SUPPLEMENTARY_INFERENCE'] and key in scores
  ods=ot[ot.target_chembl_id.eq(target)].sort_values('overall_score',ascending=False);assert len(ods)>0
  top=ods.head(3)[['disease_id','disease_name','overall_score']].to_dict('records');hyp=[];overlap=None
  if available:
   values=scores[key];excluded=np.zeros(len(ds),bool) if control else masks[key]>0
   indices=np.where(~excluded)[0];rank=np.empty(len(ds),int);rank.fill(-1);rank[indices[np.argsort(-values[indices],kind='stable')]]=np.arange(1,len(indices)+1)
   linked=ods[(ods.txgnn_score_column>=0)&(ods.overall_score>=.1)].copy();linked['query_rank']=[int(rank[int(x)]) for x in linked.txgnn_score_column];linked=linked[linked.query_rank>0].sort_values(['query_rank','overall_score'],ascending=[True,False]).drop_duplicates('txgnn_score_column')
   anylinked=ods[ods.txgnn_score_column>=0];overlap=len({int(x) for x in anylinked.txgnn_score_column if 0<rank[int(x)]<=50})
   for x in linked.head(3).to_dict('records'):
    ci=int(x['txgnn_score_column']);h={'txgnn_disease_id':str(ds.iloc[ci].id),'txgnn_disease_name':ds.iloc[ci].node_name,'txgnn_logit':float(values[ci]),'txgnn_rank_within_drug':int(rank[ci]),'ot_disease_id':x['disease_id'],'ot_disease_name':x['disease_name'],'ot_score':x['overall_score'],'mapping_scope':x['txgnn_mapping_scope'],'ot_datatype_scores_json':x['datatype_scores_json'],'known_relations_allowed_for_control':control};hyp.append(h);long.append({'experiment_id':r['experiment_id'],'ligand_inchikey':key,'drug_names':r['drug_names'],'target_chembl_id':target,'gene_symbol':r['gene_symbol'],'selection_role':r['selection_role'],**h})
  rows.append({k:r[k] for k in ['experiment_id','ligand_inchikey','drug_names','target_chembl_id','gene_symbol','selection_role']}|{'txgnn_available':available,'txgnn_mapping_status':status,'ot_status':'COMPLETE','ot_association_count':len(ods),'ot_top3_json':json.dumps(top,ensure_ascii=False),'treatment_direction_established':False,'top50_ot_overlap':overlap,'disease_hypotheses_json':json.dumps(hyp,ensure_ascii=False),'hypothesis_rule':'OT_SCORE_GE_0.1_REVIEW_HEURISTIC_NOT_VALIDATED_FUSION','known_relations_allowed_for_control':control,'experiment_release_status':'REVIEW_REQUIRED_NOT_RELEASED'})
 review=pd.DataFrame(rows);review.to_csv(OUT/'SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv',index=False);pd.DataFrame(long).to_csv(OUT/'DISEASE_HYPOTHESES_LONG.csv',index=False)
 keys=['ligand_inchikey','target_chembl_id','selection_role'];integrated=master.merge(review.drop(columns=['drug_names','gene_symbol','experiment_id']),on=keys,validate='one_to_one').merge(relations.drop(columns=['drug_names','gene_symbol']),on=keys,validate='one_to_one');integrated.to_csv(OUT/'INTEGRATED_MASTER_512.csv',index=False)
 with pd.ExcelWriter(OUT/'SPR512_TXGNN_OPENTARGETS_REVIEW.xlsx') as w:
  cols=['experiment_id','drug_names','gene_symbol','selection_role','txgnn_available','txgnn_mapping_status','ot_association_count','top50_ot_overlap','review_status','non_text_unambiguous_assertion_rows','support_note','control_status','release_status','ot_top3_json']
  integrated[cols].to_excel(w,sheet_name='512配对联合审查',index=False);pd.DataFrame(long).to_excel(w,sheet_name='疾病假设明细',index=False);ods=None
  ot.sort_values('overall_score',ascending=False).groupby('target_chembl_id').head(20).to_excel(w,sheet_name='64靶点疾病前20',index=False);cov.reset_index().to_excel(w,sheet_name='181分子覆盖',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 summary={'rows':len(review),'candidate_rows':448,'positive_control_rows':64,'unique_compounds':len(cov),'mapped_unique_compounds':int(review[review.txgnn_available].ligand_inchikey.nunique()),'txgnn_rows':int(review.txgnn_available.sum()),'txgnn_candidate_rows':int(review[~review.known_relations_allowed_for_control].txgnn_available.sum()),'txgnn_control_rows':int(review[review.known_relations_allowed_for_control].txgnn_available.sum()),'opentargets_targets':ot.target_chembl_id.nunique(),'opentargets_rows':len(ot),'supplementary_scored_compounds':len(order),'supplementary_scores':int(extra.size),'hypothesis_rows':len(long),'candidate_rows_with_review_hypothesis':int((~review.known_relations_allowed_for_control & review.disease_hypotheses_json.ne('[]')).sum()),'no_training':True,'design_selection_changed':False,'release_ready':False}
 checks={'512_unique_joins':len(integrated)==512 and not integrated.duplicated(keys).any(),'all64_ot_complete':summary['opentargets_targets']==64,'supplementary_scores_finite':bool(np.isfinite(extra).all()),'no_hypotheses_for_unmapped':review[~review.txgnn_available].disease_hypotheses_json.eq('[]').all().item(),'candidate_count_preserved':(~review.known_relations_allowed_for_control).sum()==448,'supplementary_native_agreement':json.loads((OUT/'TXGNN_DECODER_VERIFICATION.json').read_text())['max_abs_error']<1e-5}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':{k:bool(v) for k,v in checks.items()}},indent=2));assert all(checks.values());print(summary)

if __name__=='__main__':
 import sys
 build() if '--build' in sys.argv else prepare()

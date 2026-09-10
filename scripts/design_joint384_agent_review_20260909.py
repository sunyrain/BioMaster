"""Assemble three agent audits and select 384 discovery candidates; controls extra."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/joint_screen_720x384_20260909'
REV=ROOT/'outputs/joint384_agent_review_20260909'
OUT=ROOT/'outputs/joint384_design_20260909'
STRICT=ROOT/'outputs/target_universe_ch37_v2/chembl37_calibration_all888_v2/TARGET_ALL888_STRICT_BINDING_PAIRS_V2.csv.gz'
ORIGINAL=ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def optimize(p,drugcap,targetcap):
 p=p.reset_index(drop=True);ts=sorted(p.target_chembl_id.unique());ds=sorted(p.ligand_inchikey.unique());n=len(p);ti={t:n+i for i,t in enumerate(ts)};di={d:n+len(ts)+i for i,d in enumerate(ds)};size=n+len(ts)+len(ds)
 rr=[];cc=[];vv=[];lo=[];hi=[]
 def add(coeff,a,b):
  k=len(lo);lo.append(a);hi.append(b)
  for j,v in coeff.items():rr.append(k);cc.append(j);vv.append(v)
 add({i:1 for i in range(n)},384,384)
 add({j:1 for j in ti.values()},64,min(96,len(ts)))
 add({j:1 for j in di.values()},min(128,len(ds)),len(ds))
 for t,g in p.groupby('target_chembl_id'):
  c={int(i):1 for i in g.index};add({**c,ti[t]:-targetcap},-np.inf,0);add({**c,ti[t]:-1},0,np.inf)
 for d,g in p.groupby('ligand_inchikey'):
  c={int(i):1 for i in g.index};add({**c,di[d]:-drugcap},-np.inf,0);add({**c,di[d]:-1},0,np.inf)
 for _,g in p.groupby('connectivity_key'):add({int(i):1 for i in g.index},0,drugcap)
 utility=(50*p.priority_after_agent_review.astype(int)+5*p.decision.eq('KEEP_REVIEW').astype(int)+3*p.positive_chemical_support_ge04.astype(int)+3*p.joint_r50_ot03_exact_genetic.astype(int)+1*p.joint_r50_ot03_exact.astype(int)+.05*(21-p.binding_rank_384)+.1*p.review_ot_score)
 c=np.zeros(size);c[:n]=-utility.to_numpy();c[n:n+len(ts)]=2.;c[n+len(ts):]=-.2
 A=coo_matrix((vv,(rr,cc)),shape=(len(lo),size)).tocsc()
 result=milp(c,integrality=np.ones(size),bounds=Bounds(np.zeros(size),np.ones(size)),constraints=LinearConstraint(A,lo,hi),options={'time_limit':55,'mip_rel_gap':.001})
 gap=getattr(result,'mip_gap',None)
 meta={'drug_cap':drugcap,'target_cap':targetcap,'status':int(result.status),'message':result.message,'mip_gap':None if gap is None else float(gap),'objective':'Exploratory review utility and diversity only; not a probability or validated efficacy score.'}
 if result.x is None:return None,meta
 x=np.rint(result.x);v=A@x
 if not(np.all(v>=np.array(lo)-1e-6) and np.all(v<=np.array(hi)+1e-6)):return None,meta
 out=p.loc[np.flatnonzero(x[:n])].copy();out['selection_utility_unvalidated']=utility.loc[out.index]
 return out,meta

def control_pool(strict,targets):
 s=strict[strict.target_chembl_id.isin(targets)&strict.calibration_label.eq('positive')&~strict.numeric_positive_negative_conflict&~strict.explicit_inactive_positive_conflict&strict.parent_standard_inchi_key.notna()&strict.parent_canonical_smiles.notna()].copy()
 # Endpoint presence is a proposal tier, not a claim that pooled pChEMBL is Kd.
 s['endpoint_tier']=np.select([s.standard_types.str.contains(r'(?:^|,)Kd(?:,|$)',regex=True,na=False),s.standard_types.str.contains(r'(?:^|,)Ki(?:,|$)',regex=True,na=False)],[0,1],default=2)
 s['named_reference']=s.parent_molecule_name.notna();s['repeat_evidence']=s.document_count.clip(upper=5)
 old=pd.read_csv(ORIGINAL);old=old[old.selection_role.eq('POSITIVE_CONTROL')]
 oldpairs=set(zip(old.target_chembl_id,old.ligand_inchikey));s['old_control']=list(zip(s.target_chembl_id,s.parent_standard_inchi_key));s['old_control']=s.old_control.isin(oldpairs)
 s=s.sort_values(['target_chembl_id','endpoint_tier','old_control','named_reference','repeat_evidence','mean_pchembl'],ascending=[True,True,False,False,False,False])
 chosen=[]
 for t,g in s.groupby('target_chembl_id',sort=False):
  for r in g.itertuples():
   m=Chem.MolFromSmiles(r.parent_canonical_smiles)
   if m is None or Chem.MolToInchiKey(m)!=r.parent_standard_inchi_key:continue
   chosen.append({'target_chembl_id':t,'gene_symbol':r.gene_symbol,'ligand_inchikey':r.parent_standard_inchi_key,'ligand_smiles':r.parent_canonical_smiles,'drug_names':r.parent_molecule_name if pd.notna(r.parent_molecule_name) else r.parent_molecule_chembl_id,'control_chembl_id':r.parent_molecule_chembl_id,'pair_id':r.parent_standard_inchi_key+'__'+t,'selection_role':'REFERENCE_CONTROL_EXTRA_TO_384','reference_endpoint_tier':int(r.endpoint_tier),'reference_endpoint_category':['Kd_RECORD_PRESENT','Ki_NO_Kd_RECORD','FUNCTIONAL_OR_OTHER_ONLY'][r.endpoint_tier],'reference_standard_types':r.standard_types,'reference_assay_ids':r.assay_ids,'reference_doc_ids':r.doc_ids,'mean_pchembl_mixed_endpoints':r.mean_pchembl,'reused_original_control':bool(r.old_control),'release_status':'REFERENCE_PROPOSAL_SPR_APPLICABILITY_AND_PROCUREMENT_PENDING'})
   break
 return pd.DataFrame(chosen)

def main():
 OUT.mkdir(exist_ok=True)
 source=pd.read_csv(BASE/'BIOCHEMICAL_LANE_574_REVIEW.csv');parts=[]
 for name,lane in [('enzyme','ENZYME_BIOCHEMICAL'),('kinase','KINASE_BIOCHEMICAL'),('nuclear','NUCLEAR_EPIGENETIC_DOMAIN')]:
  f=REV/(name+'_review.csv');d=pd.read_csv(f);d['review_agent']=name
  assert d.pair_id.is_unique and set(d.pair_id)==set(source.loc[source.assay_lane.eq(lane),'pair_id']),name
  assert d.decision.isin(['KEEP_REVIEW','LOW_PRIORITY','HOLD','EXCLUDE']).all()
  parts.append(d[['pair_id','decision','reason','evidence_scope','sources','assay_notes','review_agent']])
 reviewed=source.merge(pd.concat(parts,ignore_index=True),on='pair_id',validate='one_to_one')
 reviewed['priority_after_agent_review']=reviewed.priority_multi_evidence&reviewed.decision.eq('KEEP_REVIEW')
 reviewed['quick_review_limit']='Every row reviewed with local data/rules; literature searched only where evidence_scope/sources explicitly state it. Not 574 independent literature validations.'
 reviewed.to_csv(OUT/'ALL_574_AGENT_REVIEW.csv',index=False)
 usable=reviewed[reviewed.decision.isin(['KEEP_REVIEW','LOW_PRIORITY'])&~reviewed.close_negative_review].copy()
 if len(usable)<384:raise RuntimeError(f'Only {len(usable)} usable: cannot fill 384 without relaxing evidence holds')
 attempts=[];selected=None
 for drugcap,targetcap in [(4,8),(4,10),(6,10),(6,12)]:
  selected,meta=optimize(usable,drugcap,targetcap);attempts.append(meta)
  if selected is not None:break
 (OUT/'SOLVER.json').write_text(json.dumps(attempts,indent=2))
 if selected is None:raise RuntimeError('384 infeasible under declared diversity policies; do not pad with holds')
 selected=selected.sort_values(['priority_after_agent_review','decision','positive_chemical_support_ge04','binding_rank_384','gene_symbol','drug_names'],ascending=[False,True,False,True,True,True]).reset_index(drop=True)
 selected['candidate_id']=[f'J384-{i:03d}' for i in range(1,385)];selected['selection_role']='JOINT_DISCOVERY_CANDIDATE';selected['release_status']='AGENT_REVIEW_DESIGN_NOT_RELEASED_FOR_EXPERIMENT'
 selected['review_tier']=np.select([selected.priority_after_agent_review,selected.decision.eq('KEEP_REVIEW')],['MULTI_EVIDENCE_SURVIVED_QUICK_REVIEW','REVIEW_CANDIDATE'],default='LOW_PRIORITY_EXPLORATORY')
 selected.to_csv(OUT/'CANDIDATES_384.csv',index=False)
 excluded=reviewed[~reviewed.pair_id.isin(selected.pair_id)].copy();excluded['selection_disposition']=np.select([excluded.decision.eq('EXCLUDE'),excluded.decision.eq('HOLD'),excluded.close_negative_review],['EXCLUDED_WITH_RECORDED_REASON','HELD_FOR_TARGETED_REVIEW','CLOSE_NEGATIVE_REFERENCE_HOLD'],default='RESERVE_NOT_SELECTED_UNDER_DIVERSITY_AND_BUDGET')
 excluded.to_csv(OUT/'NOT_SELECTED_FROM_574.csv',index=False)
 controls=control_pool(pd.read_csv(STRICT),set(selected.target_chembl_id));controls.to_csv(OUT/'REFERENCE_CONTROLS_EXTRA.csv',index=False)
 roster=selected.groupby(['target_chembl_id','gene_symbol','assay_lane']).agg(candidates=('pair_id','size'),drugs=('ligand_inchikey','nunique'),low_priority=('decision',lambda s:int(s.eq('LOW_PRIORITY').sum())),multi_evidence=('priority_after_agent_review','sum')).reset_index()
 roster=roster.merge(controls[['target_chembl_id','drug_names','reference_endpoint_category']].rename(columns={'drug_names':'proposed_reference'}),on='target_chembl_id',how='left',validate='one_to_one');roster['construct_status']='BOUNDARIES_ISOFORM_COFACTORS_AND_ACTIVITY_NOT_LAB_CONFIRMED';roster.to_csv(OUT/'TARGET_ROSTER.csv',index=False)
 pd.concat([selected,controls],ignore_index=True,sort=False).to_csv(OUT/'CANDIDATES_AND_EXTRA_CONTROLS.csv',index=False)
 with pd.ExcelWriter(OUT/'JOINT384_AGENT_REVIEW.xlsx') as w:
  cols=['candidate_id','drug_names','gene_symbol','review_tier','decision','reason','evidence_scope','sources','binding_rank_384','review_disease','review_txgnn_rank','review_ot_score','positive_max_tanimoto','negative_max_tanimoto','assay_notes','ligand_inchikey','target_chembl_id','release_status']
  selected[cols].to_excel(w,sheet_name='384候选',index=False);controls.to_excel(w,sheet_name='另计参考对照',index=False);roster.to_excel(w,sheet_name='靶点配额',index=False)
  excluded[['drug_names','gene_symbol','decision','selection_disposition','reason','evidence_scope','sources']].to_excel(w,sheet_name='排除待审与备选',index=False)
  reviewed[['drug_names','gene_symbol','decision','reason','evidence_scope','sources','assay_notes','pair_id']].to_excel(w,sheet_name='574逐行快审',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 summary={'candidates':len(selected),'drugs':selected.ligand_inchikey.nunique(),'targets':selected.target_chembl_id.nunique(),'extra_reference_controls':len(controls),'total_proposed_pairs_with_controls':len(selected)+len(controls),'review_decisions_574':reviewed.decision.value_counts().to_dict(),'selected_decisions':selected.decision.value_counts().to_dict(),'selected_review_tiers':selected.review_tier.value_counts().to_dict(),'usable_before_optimization':len(usable),'selected_chemical_ge04':int(selected.positive_chemical_support_ge04.sum()),'original_priority_label_usable':int(usable.priority_multi_evidence.sum()),'priority_after_agent_review_usable':int(usable.priority_after_agent_review.sum()),'priority_after_agent_review_selected':int(selected.priority_after_agent_review.sum()),'reference_endpoint_categories':controls.reference_endpoint_category.value_counts().to_dict(),'candidate_target_max':int(selected.groupby('target_chembl_id').size().max()),'candidate_drug_max':int(selected.groupby('ligand_inchikey').size().max())}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 checks={'574_unique_reviews':len(reviewed)==574 and reviewed.pair_id.is_unique,'384_unique_candidates':len(selected)==384 and selected.pair_id.is_unique,'no_HOLD_EXCLUDE_selected':selected.decision.isin(['KEEP_REVIEW','LOW_PRIORITY']).all(),'no_close_negative':not selected.close_negative_review.any(),'all_original_joint_gates':selected.base_eligible.all() and selected.primary_joint_selected.all(),'one_reference_per_selected_target':set(controls.target_chembl_id)==set(selected.target_chembl_id) and controls.target_chembl_id.is_unique,'no_reference_candidate_overlap':not bool(set(controls.pair_id)&set(selected.pair_id)),'original512_unchanged':sha(ORIGINAL)=='e6501353312b62717c766156fdea4b145bbf9b135eef3efff332d491d1f02487','all_574_accounted_for':len(selected)+len(excluded)==574}
 checks={k:bool(v) for k,v in checks.items()};(OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks,'training':False,'experimental_release':False},indent=2));assert all(checks.values()),checks
 sources=[BASE/'BIOCHEMICAL_LANE_574_REVIEW.csv',STRICT,ORIGINAL,Path(__file__),*[REV/(k+'_review.csv') for k in ['enzyme','kinase','nuclear']]]
 (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps({'sources_sha256':{str(f.relative_to(ROOT)):sha(f) for f in sources},'outputs_sha256':{str(f.relative_to(OUT)):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='SOURCE_MANIFEST.json'}},indent=2));print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

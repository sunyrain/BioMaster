"""Modestly relax ranking gates; retain identity/known-relation guards and audit raw endpoints."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from design_spr64_512_20260909 import annotate_support
from design_joint384_agent_review_20260909 import STRICT
import audit_comprehensive_raw_activities_20260909 as raw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/spr_expanded_review_20260910'; OUT.mkdir(exist_ok=True)
BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
OLD=ROOT/'outputs/joint384_comprehensive_20260909'
SOURCE=ROOT/'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=pd.read_parquet(SOURCE)
 old=pd.read_csv(OLD/'RECOMMENDED_CANDIDATES_384.csv'); audits=pd.read_csv(OLD/'FULL574_COMPREHENSIVE_AUDIT.csv')
 held_drugs=set(audits.loc[audits.new_identity_hold,'ligand_inchikey'])
 # Newly unresolved modeled product identity, retain for resolution outside discovery queue.
 reg=pd.read_csv(BASE/'DRUG_REGISTRY_720.csv',low_memory=False)
 held_drugs.update(reg.loc[reg.drug_names.str.contains('nicotine polacrilex',case=False,na=False),'ligand_inchikey'])
 hold_cols=['new_identity_hold','disease_investment_hold','raw_prior_activity_hold','deep_review_investment_hold','close_negative_review']
 held_pairs=set(audits.loc[audits[hold_cols].any(axis=1)|audits.decision.isin(['HOLD','EXCLUDE']),'pair_id'])
 lane=p.assay_lane.isin(['ENZYME_BIOCHEMICAL','KINASE_BIOCHEMICAL','NUCLEAR_EPIGENETIC_DOMAIN'])
 safe=p.identity_scope_pass&p.chemistry_policy_pass&p.novelty_pass&lane&~p.ligand_inchikey.isin(held_drugs)&~p.pair_id.isin(held_pairs)
 joint=safe&p.interpretation_eligible&p.binding_rank_384.le(30)&p.joint_r100_ot03.fillna(False)
 binding=safe&p.binding_rank_384.le(10)
 scope=p[joint|binding].copy();scope['relaxed_joint_route']=joint.loc[scope.index];scope['binding_route_rank_gate']=binding.loc[scope.index]
 scope.to_csv(OUT/'PRE_CHEMISTRY_SCOPE.csv',index=False)
 print('Scope',len(scope),flush=True)
 raw.OUT=OUT/'raw_audit';raw.OUT.mkdir(exist_ok=True);raw.SOURCE=OUT/'PRE_CHEMISTRY_SCOPE.csv'
 raw.main()
 ra=pd.read_csv(raw.OUT/'ALL574_RAW_ACTIVITY_NOVELTY_REAUDIT.csv')
 scope=scope.merge(ra[['pair_id','raw_activity_rows']],on='pair_id',validate='one_to_one')
 cache=OUT/'CHEMISTRY_CACHE.parquet'; cachemeta=OUT/'CHEMISTRY_CACHE_IDENTITY.json'
 cache_identity={'strict_sha256':sha(STRICT),'input_sha256':hashlib.sha256(scope[['pair_id','ligand_smiles']].to_csv(index=False).encode()).hexdigest()}
 if cache.exists() and cachemeta.exists() and json.loads(cachemeta.read_text())==cache_identity:
  cached=pd.read_parquet(cache)
  if cached.pair_id.tolist()==scope.pair_id.tolist():scope=scope.merge(cached,on='pair_id',validate='one_to_one')
  else:scope=annotate_support(scope,pd.read_csv(STRICT,low_memory=False))
 else:scope=annotate_support(scope,pd.read_csv(STRICT,low_memory=False))
 scope[['pair_id']+[c for c in scope if c.startswith(('positive_','negative_')) and c not in ['positive_compounds','negative_compounds']]].to_parquet(cache,index=False)
 cachemeta.write_text(json.dumps(cache_identity))
 bad_refs={('pomalidomide','MIF'),('fentanyl','CA1'),('fentanyl','CA7'),('baricitinib','HDAC1'),('baricitinib','HDAC6'),('baricitinib','HDAC11'),('mirdametinib','PIK3CD'),('vismodegib','TYK2')}
 scope['reference_transfer_hold']=[(d,g) in bad_refs for d,g in zip(scope.drug_names,scope.gene_symbol)]
 scope['binding_only_route']=scope.binding_route_rank_gate&scope.positive_max_tanimoto.ge(.35)&~scope.reference_transfer_hold&~(scope.negative_max_tanimoto.ge(.4)&scope.negative_max_tanimoto.gt(scope.positive_max_tanimoto))
 # Apply prior drug–disease policy to joint admission, not only the displayed disease.
 policy_drugs=scope[scope.drug_names.isin(['raloxifene','bazedoxifene'])]
 if len(policy_drugs):
  pot=pd.read_parquet(ROOT/'outputs/joint_screen_720x384_20260909/OPENTARGETS_384_FULL_ASSOCIATIONS.parquet')
  order0=pd.read_csv(BASE/'TXGNN_SCORE_DRUG_ORDER.csv');v0=np.load(BASE/'TXGNN_INDICATION_LOGITS.npy');m0=np.load(BASE/'TXGNN_EXCLUSION_MASK.npy');indices={k:i for i,k in enumerate(order0.ligand_inchikey)}
  for key, group in policy_drugs.groupby('ligand_inchikey'):
   i=indices[key];ids=np.where(m0[i]==0)[0];rank=np.full(v0.shape[1],-1,dtype=int);rank[ids[np.argsort(-v0[i,ids],kind='stable')]]=np.arange(1,len(ids)+1)
   g=pot[(pot.txgnn_score_column>=0)&(pot.overall_score>=.3)&~pot.disease_name.str.contains('endometriosis',case=False,na=False)].copy();g['rank']=rank[g.txgnn_score_column.astype(int)];valid_targets=set(g.loc[g['rank'].between(1,100),'target_chembl_id'])
   scope.loc[group.index,'relaxed_joint_route']=scope.loc[group.index,'relaxed_joint_route'] & group.target_chembl_id.isin(valid_targets)
 scope['expanded_eligible']=(scope.relaxed_joint_route|scope.binding_only_route)&scope.raw_activity_rows.eq(0)
 scope['in_baseline384']=scope.pair_id.isin(old.pair_id)
 scope.to_csv(OUT/'FULL_EXPANDED_SCOPE_AUDIT.csv',index=False)
 pool=scope[scope.expanded_eligible&~scope.in_baseline384].copy()
 pool['route']=np.where(pool.relaxed_joint_route&pool.binding_only_route,'BOTH',np.where(pool.relaxed_joint_route,'RELAXED_JOINT','BINDING_CHEMISTRY_WITHOUT_GRAPH_GATE'))
 # Fixed first expansion budget, transparent diversity caps; not a fitted success score.
 ranked=pool.sort_values(['positive_max_tanimoto','binding_rank_384','pair_id'],ascending=[False,True,True])
 selected=[];drugs={};targets={}
 def take(frame,limit):
  added=0
  for r in frame.itertuples():
   if r.pair_id in selected or drugs.get(r.ligand_inchikey,0)>=4 or targets.get(r.target_chembl_id,0)>=8:continue
   selected.append(r.pair_id);drugs[r.ligand_inchikey]=drugs.get(r.ligand_inchikey,0)+1;targets[r.target_chembl_id]=targets.get(r.target_chembl_id,0)+1;added+=1
   if added>=limit:break
 # At least reserve an equal opportunity for graph-independent strong-binding hypotheses.
 take(ranked[ranked.binding_only_route&~ranked.relaxed_joint_route],192)
 take(ranked[ranked.relaxed_joint_route],384-len(selected))
 if len(selected)<384:take(ranked,384-len(selected))
 new=pool.set_index('pair_id').loc[selected].reset_index();new['candidate_id']=[f'EXP-{i:03d}' for i in range(1,len(new)+1)]
 # Recompute disease evidence from masked TxGNN ranks; never invent hypotheses for no-graph rows.
 ot=pd.read_parquet(ROOT/'outputs/joint_screen_720x384_20260909/OPENTARGETS_384_FULL_ASSOCIATIONS.parquet')
 order=pd.read_csv(BASE/'TXGNN_SCORE_DRUG_ORDER.csv');values=np.load(BASE/'TXGNN_INDICATION_LOGITS.npy');mask=np.load(BASE/'TXGNN_EXCLUSION_MASK.npy');ix={k:i for i,k in enumerate(order.ligand_inchikey)};ranks={}
 for key in new.ligand_inchikey.unique():
  if key not in ix:continue
  i=ix[key];valid=np.where(mask[i]==0)[0];rank=np.full(values.shape[1],-1,dtype=int);rank[valid[np.argsort(-values[i,valid],kind='stable')]]=np.arange(1,len(valid)+1);ranks[key]=rank
 details=[]
 for row in new.itertuples():
  if row.ligand_inchikey not in ranks or not row.interpretation_eligible:continue
  g=ot[(ot.target_chembl_id==row.target_chembl_id)&(ot.txgnn_score_column>=0)&(ot.overall_score>=.3)].copy()
  g['txgnn_rank']=ranks[row.ligand_inchikey][g.txgnn_score_column.astype(int)];g=g[g.txgnn_rank.between(1,100)]
  if row.drug_names in ['raloxifene','bazedoxifene']:g=g[~g.disease_name.str.contains('endometriosis',case=False,na=False)]
  g['pair_id']=row.pair_id;details.extend(g.to_dict('records'))
 detail=pd.DataFrame(details);detail.to_csv(OUT/'EXPANDED_JOINT_DISEASE_EVIDENCE.csv.gz',index=False)
 if len(detail):
  best=detail.sort_values(['txgnn_rank','overall_score'],ascending=[True,False]).drop_duplicates('pair_id')
  new=new.merge(best[['pair_id','disease_name','disease_id','txgnn_rank','overall_score','txgnn_mapping_scope']].rename(columns={'disease_name':'recommended_disease','disease_id':'recommended_disease_id','txgnn_rank':'recommended_txgnn_rank','overall_score':'recommended_ot_score','txgnn_mapping_scope':'recommended_mapping_scope'}),on='pair_id',how='left',validate='one_to_one')
 origins=pd.read_csv(OLD/'ORIGIN_REVIEWED_266.csv');print('origin columns',origins.columns.tolist(),flush=True)
 # Known MOA is exact full-InChIKey mapped; use reviewed labels only where available.
 new=new.merge(reg[['ligand_inchikey','ingredient_name_normalized','known_target_names','known_mechanism_of_action','gsrs_display_name']],on='ligand_inchikey',validate='many_to_one')
 label=audits[['ligand_inchikey','origin_summary','source_urls','modeled_entity_name']].drop_duplicates('ligand_inchikey')
 new=new.merge(label,on='ligand_inchikey',how='left',validate='many_to_one');new['source_ingredient_name']=new.ingredient_name_normalized;new['modeled_entity_name']=new.gsrs_display_name.str.lower().fillna(new.modeled_entity_name)
 new.loc[new.ligand_inchikey.eq('MQOBSOSZFYZQOK-UHFFFAOYSA-N'),'modeled_entity_name']='fenofibric acid'  # ChEMBL37 CHEMBL981 exact structure
 new['origin_summary']=new.origin_summary.fillna('本轮尚未完成该药原用途标签核实，不据此判定跨领域')
 new['review_status']='LLM_REVIEW_PENDING'
 new.to_csv(OUT/'NEW_CANDIDATES_FOR_LLM.csv',index=False);pool.to_csv(OUT/'ALL_NEW_ELIGIBLE_POOL.csv',index=False)
 old_keys=set(old.pair_id);assert not set(new.pair_id)&old_keys;assert new.pair_id.is_unique and new.raw_activity_rows.eq(0).all() and new.novelty_pass.all()
 summary={'full_scored_pairs':len(p),'pre_chemistry_scope':len(scope),'raw_activity_pairs':int(scope.raw_activity_rows.gt(0).sum()),'new_eligible_pool':len(pool),'new_llm_queue':len(new),'remaining_outside_review_queue':len(pool)-len(new),'new_reviewed_at_preparation':0,'new_pending_at_preparation':len(new),'new_drugs':new.ligand_inchikey.nunique(),'new_targets':new.target_chembl_id.nunique(),'route_counts':new.route.value_counts().to_dict(),'baseline_candidates':384,'combined_review_target':384+len(new),'relaxation':'Binding rank 20→30; masked TxGNN rank50→100; OT>=0.3 unchanged. Separate binding Top10 + positive Tanimoto>=0.35 route without mandatory graph agreement. Known/identity/raw guards retained. First 384 new rows, max4/drug,max8/target; reserve up to192 graph-independent and remaining places relaxed-joint; chemical-support/rank ordering within each route. Not calibrated probabilities.'}
 (OUT/'SCOPE_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
 for i in range(3):
  batch=new.iloc[i*128:(i+1)*128].replace({np.nan:None}).to_dict('records');(OUT/f'expanded_batch_{i+1}.json').write_text(json.dumps(batch,ensure_ascii=False,indent=2))
 (OUT/'INPUT_MANIFEST.json').write_text(json.dumps({'baseline_sha256':sha(OLD/'RECOMMENDED_CANDIDATES_384.csv'),'full_space_sha256':sha(SOURCE),'new_queue_sha256':sha(OUT/'NEW_CANDIDATES_FOR_LLM.csv')},indent=2))
if __name__=='__main__':main()

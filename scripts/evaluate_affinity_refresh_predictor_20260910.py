#!/usr/bin/env python3
"""Frozen-score retrospective audit. Does not train, tune, rerank or modify SPR.

Prespecified label bounds: positive <=1uM, weak-binding negative >=10uM.
Unknown observations are NEVER converted to negative labels. Conflicts excluded.
"""
from pathlib import Path
import hashlib,json,math
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/affinity_predictor_audit_20260910'
EVIDENCE=ROOT/'outputs/affinity_evidence_refresh_20260910/PROJECT_EXACT_NUMERIC_EVIDENCE.csv.gz'
SCORES=ROOT/'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet'
TRAIN_PATHS=[
'outputs/retrain_20260901/comprehensive_training_v1/COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz',
'outputs/biomaster_comprehensive_training_v1/COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz',
'outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz',
'outputs/biomaster_bindingdb_affinity_feature_package_v1/BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz']
MODELS={
'SPR_current':'independent_validation_rank_score',
'contract_primary':'biomaster_independent_borda_score',
'neural_full_fit':'ensemble_drug_to_target_logit',
'graph_ranker':'old_drug_leakage_safe_score_v10',
'DTIAM_comparator':'dtiam_probability'}
ENDPOINTS={'Kd_only':['Kd'],'Kd_Ki':['Kd','Ki'],'IC50_only':['IC50'],'EC50_only':['EC50'],'mixed_activity_descriptive':['Kd','Ki','IC50','EC50']}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def classify(value,relation,pos=1000.,neg=10000.):
 if not np.isfinite(value) or value<=0:return 'UNRESOLVED'
 if relation=='=':
  return 'POSITIVE' if value<=pos else 'WEAK_NEGATIVE' if value>=neg else 'GREY'
 if relation in ['<','<='] and value<=pos:return 'POSITIVE'
 if relation in ['>','>='] and value>=neg:return 'WEAK_NEGATIVE'
 return 'UNRESOLVED'
def aggregate(d):
 rows=[]
 for pid,g in d.groupby('pair_id',sort=False):
  states=set(g.observation_label);p='POSITIVE' in states;n='WEAK_NEGATIVE' in states
  label='CONFLICT' if p and n else 'POSITIVE' if p else 'WEAK_NEGATIVE' if n else 'GREY' if 'GREY' in states else 'UNRESOLVED'
  # Direction is robust to duplicate source copies; no strongest-record selection.
  rows.append(dict(pair_id=pid,ligand_inchikey=g.ligand_key.iloc[0],target_chembl_id=g.target_chembl_id.iloc[0],gene_symbol=g.gene_symbol.iloc[0],label=label,binary_label=1 if label=='POSITIVE' else 0 if label=='WEAK_NEGATIVE' else np.nan,
    source_rows=len(g),sources=';'.join(sorted(set(g.source))),endpoints=';'.join(sorted(set(g.endpoint))),relations=';'.join(sorted(set(g.relation))),min_nM=g.value_nM.min(),max_nM=g.value_nM.max(),has_grey='GREY' in states,has_unresolved='UNRESOLVED' in states,
    positive_record_ids=';'.join(g.loc[g.observation_label.eq('POSITIVE'),'id'].astype(str)),negative_record_ids=';'.join(g.loc[g.observation_label.eq('WEAK_NEGATIVE'),'id'].astype(str))))
 return pd.DataFrame(rows)

def load_training():
 exact=set();connect=set();drugs=set();targets=set();manifest=[]
 for rel in TRAIN_PATHS:
  p=ROOT/rel
  d=pd.read_csv(p,usecols=['parent_standard_inchi_key','target_chembl_id'],dtype=str).dropna()
  pairs=set(d.parent_standard_inchi_key+'__'+d.target_chembl_id)
  exact.update(pairs);connect.update(d.parent_standard_inchi_key.str.split('-').str[0]+'__'+d.target_chembl_id);drugs.update(d.parent_standard_inchi_key);targets.update(d.target_chembl_id)
  manifest.append(dict(path=rel,rows=len(d),unique_pairs=len(pairs),sha256=sha(p)))
  print('TRAINING',rel,len(d),flush=True)
 return exact,connect,drugs,targets,manifest

def metrics(g,model):
 score=MODELS[model];z=g[g.binary_label.notna()&g[score].notna()].copy();y=z.binary_label.astype(int).to_numpy();v=z[score].to_numpy();p=int(y.sum());n=len(y)-p
 r=dict(model=model,score_column=score,n_pairs=len(z),positive=p,weak_negative=n,drugs=z.ligand_inchikey.nunique(),targets=z.target_chembl_id.nunique(),positive_prevalence=p/len(y) if len(y) else None,auroc=None,average_precision=None,ap_over_prevalence=None,within_drug_macro_auroc=None,within_drug_both_class_queries=0)
 if p and n:r.update(auroc=float(roc_auc_score(y,v)),average_precision=float(average_precision_score(y,v)));r['ap_over_prevalence']=r['average_precision']/r['positive_prevalence']
 macro=[]
 for _,q in z.groupby('ligand_inchikey'):
  if q.binary_label.nunique()==2:macro.append(roc_auc_score(q.binary_label,q[score]))
 r['within_drug_macro_auroc']=float(np.mean(macro)) if macro else None;r['within_drug_both_class_queries']=len(macro)
 for k in [10,20,50]:
  top=z[z['rank_'+model]<=k];pos=int(top.binary_label.sum());neg=len(top)-pos
  r.update({f'top{k}_observed_positive':pos,f'top{k}_observed_weak_negative':neg,f'top{k}_observed_precision':pos/len(top) if len(top) else None,f'top{k}_recall_of_observed_positives':pos/p if p else None})
 return r

def clustered_ci(g,model,seed=20260910,reps=1000):
 z=g[g.binary_label.notna()&g[MODELS[model]].notna()].copy();groups=[x[['binary_label',MODELS[model]]].to_numpy() for _,x in z.groupby('ligand_inchikey')]
 rng=np.random.default_rng(seed);auc=[];ap=[]
 if int(z.binary_label.sum())<10 or int((z.binary_label==0).sum())<10:
  return dict(valid_bootstrap_replicates=0,auroc_ci95=None,ap_ci95=None,reason='Too few positives or negatives (<10); bootstrap interval suppressed.')
 if len(groups)<2 or z.binary_label.nunique()<2:return dict(valid_bootstrap_replicates=0,auroc_ci95=None,ap_ci95=None)
 for _ in range(reps):
  a=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);y=a[:,0];v=a[:,1]
  if np.unique(y).size<2:continue
  auc.append(roc_auc_score(y,v));ap.append(average_precision_score(y,v))
 return dict(valid_bootstrap_replicates=len(auc),auroc_ci95=np.quantile(auc,[.025,.975]).tolist() if auc else None,ap_ci95=np.quantile(ap,[.025,.975]).tolist() if ap else None)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 baseline=ROOT/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv';base_sha=sha(baseline)
 d=pd.read_csv(EVIDENCE,low_memory=False).fillna('')
 qcpath=ROOT/'data/external/affinity_refresh_20260910/SOURCE_RECORD_QC_OVERRIDES_20260910.csv'
 qc=pd.read_csv(qcpath,dtype=str) if qcpath.exists() else pd.DataFrame(columns=['source','record_id'])
 bad=set(qc.source+'::'+qc.record_id)
 excluded=d[(d.source+'::'+d.record_id.astype(str)).isin(bad)]
 excluded.to_csv(OUT/'EXCLUDED_SOURCE_RECORDS.csv',index=False,encoding='utf-8-sig')
 d=d[~(d.source+'::'+d.record_id.astype(str)).isin(bad)].copy()
 d['value_nM']=pd.to_numeric(d.value_nM);d['observation_label']=[classify(v,r) for v,r in zip(d.value_nM,d.relation)]
 d[['id','source','pair_id','endpoint','relation','value_nM','observation_label','source_url','doi','pmid']].to_csv(OUT/'LABELED_OBSERVATIONS.csv.gz',index=False,compression='gzip')
 p=pd.read_parquet(SCORES);assert p.pair_id.is_unique and len(p)==276480
 exact,connect,drugseen,targetseen,trainmanifest=load_training()
 old_pairs=set(d.loc[d.source.eq('ChEMBL37_existing'),'pair_id']);refresh_pairs=set(d.loc[d.source.ne('ChEMBL37_existing'),'pair_id'])
 all_labels=[];count_rows=[];metric_rows=[];ci_rows=[]
 usecols=['pair_id','drug_names','known_relation_excluded','known_training','known_chembl37','novelty_pass','identity_scope_pass','chemistry_policy_pass',*MODELS.values()]
 q=p[usecols].copy()
 for model,col in MODELS.items():q['rank_'+model]=p.groupby('ligand_inchikey')[col].rank(method='first',ascending=False)
 # Each group is a prespecified audit, not a train/test split selected by performance.
 for ep,eps in ENDPOINTS.items():
  x=d[d.endpoint.isin(eps)];a=aggregate(x)
  a['endpoint_group']=ep;a['refresh_covered']=a.pair_id.isin(refresh_pairs);a['existing_ChEMBL37_numeric_pair']=a.pair_id.isin(old_pairs)
  a['local_training_exact_pair']=a.pair_id.isin(exact)
  a['local_training_connectivity_pair']=(a.ligand_inchikey.str.split('-').str[0]+'__'+a.target_chembl_id).isin(connect)
  a['local_training_drug_seen']=a.ligand_inchikey.isin(drugseen);a['local_training_target_seen']=a.target_chembl_id.isin(targetseen)
  a=a.merge(q,on='pair_id',how='left',validate='one_to_one');a['scored']=a[MODELS['SPR_current']].notna()
  # Require current endpoint evidence itself from refresh, not only another endpoint.
  ep_refresh=set(x.loc[x.source.ne('ChEMBL37_existing'),'pair_id'])
  a['refresh_endpoint_covered']=a.pair_id.isin(ep_refresh)
  masks={
   'ALL_MERGED':np.ones(len(a),bool),
   'REFRESH_COVERED':a.refresh_endpoint_covered,
   'REFRESH_ABSENT_CHEMBL37':a.refresh_endpoint_covered&~a.existing_ChEMBL37_numeric_pair,
   'REFRESH_NO_LOCAL_TRAIN_PAIR':a.refresh_endpoint_covered&~a.local_training_connectivity_pair,
   'REFRESH_NO_PRIOR_RELATION':a.refresh_endpoint_covered&~a.local_training_connectivity_pair&a.known_relation_excluded.eq(False)&~a.existing_ChEMBL37_numeric_pair,
   'REFRESH_DRUG_UNSEEN_IN_LOCAL_TRAIN':a.refresh_endpoint_covered&~a.local_training_drug_seen}
  all_labels.append(a)
  for name,mask in masks.items():
   for space in ['720x888','720x384']:
    # Original known flags only exist in scored space; do not claim unscored coverage for that slice.
    if name=='REFRESH_NO_PRIOR_RELATION' and space=='720x888':continue
    g=a[mask & (a.scored if space=='720x384' else True)]
    counts=g.label.value_counts().to_dict()
    count_rows.append(dict(endpoint_group=ep,slice=name,scope=space,total_pairs=len(g),**{k:counts.get(k,0) for k in ['POSITIVE','WEAK_NEGATIVE','GREY','CONFLICT','UNRESOLVED']},training_exact_pairs=int(g.local_training_exact_pair.sum()),training_connectivity_pairs=int(g.local_training_connectivity_pair.sum())))
    if space!='720x384':continue
    for model in MODELS:metric_rows.append(dict(endpoint_group=ep,slice=name,**metrics(g,model)))
    if ep in ['Kd_only','Kd_Ki'] and name in ['REFRESH_NO_LOCAL_TRAIN_PAIR','REFRESH_NO_PRIOR_RELATION']:
     ci_rows.append(dict(endpoint_group=ep,slice=name,model='SPR_current',**clustered_ci(g,'SPR_current')))
  print('EVALUATED',ep,len(a),flush=True)
 labels=pd.concat(all_labels,ignore_index=True);labels.to_csv(OUT/'PAIR_LABELS_AND_FROZEN_SCORES.csv.gz',index=False,compression='gzip')
 counts=pd.DataFrame(count_rows);counts.to_csv(OUT/'LABEL_COUNTS.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(metric_rows).to_csv(OUT/'PREDICTOR_METRICS.csv',index=False,encoding='utf-8-sig')
 (OUT/'CLUSTER_BOOTSTRAP_CI.json').write_text(json.dumps(ci_rows,indent=2))
 # Inspect mistakes without changing ranking or interpreting unknowns as negatives.
 novel=labels[(labels.endpoint_group=='Kd_Ki')&labels.scored&labels.refresh_endpoint_covered&~labels.local_training_connectivity_pair&labels.known_relation_excluded.eq(False)&~labels.existing_ChEMBL37_numeric_pair].copy()
 novel.sort_values('rank_SPR_current').to_csv(OUT/'NOVEL_KD_KI_PAIR_AUDIT.csv',index=False,encoding='utf-8-sig')
 ids=set(novel.pair_id);d[d.pair_id.isin(ids)].to_csv(OUT/'NOVEL_PAIR_SOURCE_EVIDENCE.csv.gz',index=False,compression='gzip')
 # Continuous affinity correlation uses equal-weight unique exact values within endpoint,
 # after removing positive/negative conflicts; censored values never become exact Kd.
 from scipy.stats import spearmanr
 corr=[]
 for ep in ['Kd','Ki']:
  z=d[(d.endpoint==ep)&d.relation.eq('=')].drop_duplicates(['pair_id','value_nM']).copy();z['p_affinity']=9-np.log10(z.value_nM)
  z=z.groupby('pair_id',as_index=False).p_affinity.median().merge(q,on='pair_id')
  safe=set(labels.loc[(labels.endpoint_group==('Kd_only' if ep=='Kd' else 'Kd_Ki'))&~labels.local_training_connectivity_pair&~labels.label.eq('CONFLICT'),'pair_id']);z=z[z.pair_id.isin(safe)]
  for model,col in MODELS.items():
   valid=z.dropna(subset=[col]);rho,pv=spearmanr(valid.p_affinity,valid[col]) if len(valid)>=3 else (np.nan,np.nan)
   corr.append(dict(endpoint=ep,slice='NO_LOCAL_TRAIN_PAIR_EXACT_VALUES',model=model,n_pairs=len(valid),spearman_rho=rho,p_value=pv))
 pd.DataFrame(corr).to_csv(OUT/'EXACT_AFFINITY_CORRELATION.csv',index=False)
 assert sha(baseline)==base_sha
 manifest=dict(status='COMPLETED_RETROSPECTIVE_FROZEN_SCORE_AUDIT',targeted_source_qc_excluded_rows=len(excluded),qc_overrides_sha256=sha(qcpath) if qcpath.exists() else None,training_run_performed=False,baseline_unchanged=True,baseline_sha256=base_sha,inputs={str(EVIDENCE.relative_to(ROOT)):sha(EVIDENCE),str(SCORES.relative_to(ROOT)):sha(SCORES)},training_membership_audited=trainmanifest,label_rule=dict(positive_nM_le=1000,weak_negative_nM_ge=10000,conflict_policy='any decisive positive and negative -> exclude',unknown_policy='never negative',approximate_values='excluded as unresolved',Kd_app_and_response='excluded'),models=MODELS,ci=ci_rows,limitations=['Retrospective audit; candidate/target universe already selected using historical evidence.','Local training-table absence is not proof of unseen data in pretraining or external model checkpoints.','Drug/target/structural-neighbor leakage is not equivalent to pair overlap; drug-seen flags provided.','Exact accession does not validate construct, assay method or SPR transfer.','Scores are rankings, not calibrated probabilities or predicted Kd.','720x888 includes targets lacking current scores; model evaluation restricted to 720x384.','TopK recall uses observed positives only; observed precision excludes unknowns and is not prospective hit rate.'])
 (OUT/'SUMMARY.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()

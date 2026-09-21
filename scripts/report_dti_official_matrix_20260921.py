#!/usr/bin/env python3
"""Atomic live coverage and completed-model-only bidirectional rank analysis."""
import argparse,io,json,hashlib,itertools,time,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import rankdata,kendalltau
from dti_official_runtime_20260921 import ROOT,OUT,inputs,dump,sha
from analyze_dti_rank_agreement_20260921 import top_weights
MODELS=['ConPLex','DrugCLIP','Nesso-1','ProbeMatchDTI','DTBind_occurrence','MAMMAL_pKd','BALM','EviDTI','GraphBAN','ScopeDTI_light','CheMLT-F','ADME-DTI']
KEYS=['drug_id','target_id']
def alive(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();return s.split(') ')[1][0]!='Z'
 except (FileNotFoundError,TypeError):return False

def atomic_csv(df,p):
 t=p.with_name(p.stem+'.tmp'+p.suffix);df.to_csv(t,index=False,encoding='utf-8-sig');t.replace(p)

def collect(analyze=False):
 drugs,targets=inputs();base=pd.MultiIndex.from_product([targets.target_id,drugs.drug_id],names=['target_id','drug_id']).to_frame(index=False);rows=[];complete=[];hashes={}
 for model in MODELS:
  folder=OUT/model;p=folder/'SCORES.parquet';sp=folder/'STATUS.json';s=json.loads(sp.read_text()) if sp.exists() else {};state=s.get('state','NOT_STARTED');live=alive(s.get('pid'))
  # Producer status may point to a finished helper; expose its PID separately.
  if p.exists():
   with p.open('rb') as stream:data=stream.read()
   df=pd.read_parquet(io.BytesIO(data));assert not df[KEYS].duplicated().any() and np.isfinite(df.score).all();hashes[model]=hashlib.sha256(data).hexdigest()
   assert set(df.drug_id)<=set(drugs.drug_id) and set(df.target_id)<=set(targets.target_id)
   base=base.merge(df[KEYS+['score']].rename(columns={'score':model}),on=KEYS,how='left',validate='one_to_one')
   count=len(df);nt=df.target_id.nunique();nd=df.drug_id.nunique()
  else:count=nt=nd=0;base[model]=np.nan
  done=state.startswith('COMPLETE')
  if done and count:complete.append(model)
  eta=s.get('eta_seconds');eta_basis='producer estimate' if eta is not None else ''
  if model=='GraphBAN' and s.get('epoch') and s.get('checkpoint_scored_pairs',0)>0 and not done:
   work=s['checkpoint_scored_pairs'];remaining=(50-s['epoch'])*536400+536400-work
   eta=remaining*s.get('elapsed_seconds',0)/work;eta_basis='current checkpoint speed, all remaining20-checkpoint work'
  if model=='DTBind_occurrence' and (folder/'NATIVE_INPUT_ELIGIBILITY.json').exists() and not done:
   ready=json.loads((folder/'NATIVE_INPUT_ELIGIBILITY.json').read_text())['max_pairs_before_drug_failures'];rate=s.get('inference_pairs_per_second',0)
   if rate:eta=max(0,ready-s.get('scored_pairs',count))/rate;eta_basis='410 matching native graph inputs, before any pair-specific failures'
  rows.append(dict(model=model,state=state,producer_alive=live,persisted_scored_pairs=count,reported_progress_pairs=s.get('scored_pairs',count),requested_pairs=536400,coverage_percent=100*count/536400,targets_with_scores=nt,drugs_with_scores=nd,ensemble_checkpoint=s.get('epoch',s.get('checkpoint')),checkpoint_scored_pairs=s.get('checkpoint_scored_pairs'),remaining_status='FINAL_INPUT_OR_RUNTIME_MISSING' if done else 'PENDING_NOT_A_NEGATIVE',unscored_pairs=536400-count,updated_utc=s.get('updated_utc'),eta_seconds=eta,eta_basis=eta_basis,eta_upper_seconds=s.get('eta_upper_seconds')))
 atomic_csv(pd.DataFrame(rows),OUT/'MODEL_COVERAGE.csv');tmp=OUT/'SCORE_MATRIX.tmp.parquet';base.to_parquet(tmp,index=False);tmp.replace(OUT/'SCORE_MATRIX.parquet')
 stamp=datetime.now(timezone.utc).isoformat();dump(OUT/'LIVE_STATUS.json',dict(updated_utc=stamp,matrix=[720,745],official_families=12,completed_families=complete,models=rows,score_sha256=hashes,new_training=False,interpretation='Completed families only enter interim rank comparisons; current partial scores are not full-request ranks.'))
 if analyze and len(complete)>1:
  signature=hashlib.sha256(json.dumps({m:hashes[m] for m in complete},sort_keys=True).encode()).hexdigest()[:16]
  dest=OUT/'rank_snapshots'/signature;dest.mkdir(parents=True,exist_ok=True)
  if not (dest/'MANIFEST.json').exists():rank_analysis(base,complete,drugs,targets,dest,hashes,stamp)
  dump(OUT/'LATEST_RANK_ANALYSIS.json',dict(path=str(dest.relative_to(ROOT)),models=complete,scope='INTERIM_COMPLETED_OFFICIAL_MODELS_ONLY',updated_utc=stamp))
 return rows

def rank_analysis(base,models,drugs,targets,out,hashes,stamp):
 mat={m:base.pivot(index='target_id',columns='drug_id',values=m).reindex(index=targets.target_id,columns=drugs.drug_id).to_numpy() for m in models}
 common=np.logical_and.reduce([np.isfinite(mat[m]) for m in models]);details=[];exports=[];candidates=[]
 # Single-model recommendations: native full available pool, both directions.
 for m in models:
  frame=base.loc[base[m].notna(),KEYS+[m]].rename(columns={m:'score'})
  for direction,query,requested in [('target_to_drug','target_id',720),('drug_to_target','drug_id',745)]:
   f=frame.copy();g=f.groupby(query,sort=False)['score'];f['rank_average']=g.rank(ascending=False,method='average');f['rank_min']=g.rank(ascending=False,method='min');f['observed_candidates']=g.transform('size');f['requested_candidates']=requested;f['model']=m;f['direction']=direction
   exports.append(f.loc[f.rank_min<=20]);candidates.append(f[['model','direction',query,'observed_candidates','requested_candidates']].drop_duplicates().rename(columns={query:'query_id'}))
 for view in ['pairwise_common','completed_models_common']:
  for a,b in itertools.combinations(models,2):
   mask=(np.isfinite(mat[a])&np.isfinite(mat[b])) if view=='pairwise_common' else common
   for direction in ['target_to_drug','drug_to_target']:
    aa,bb,mm=(mat[a],mat[b],mask) if direction=='target_to_drug' else (mat[a].T,mat[b].T,mask.T)
    ids=targets.target_id if direction=='target_to_drug' else drugs.drug_id
    for qid,x,y,valid in zip(ids,aa,bb,mm):
     x=x[valid];y=y[valid];n=len(x);row=dict(view=view,model_a=a,model_b=b,direction=direction,query_id=qid,n_common=n,eligible=n>=20)
     if n>=20:
      rx,ry=rankdata(x),rankdata(y);constant=len(np.unique(x))==1 or len(np.unique(y))==1
      row.update(constant_output=constant,spearman=float(np.corrcoef(rx,ry)[0,1]) if not constant else np.nan,kendall_tau_b=float(kendalltau(x,y,variant='b').statistic) if not constant else np.nan)
      for k in [5,10,20]:
       overlap=float(top_weights(x,k)@top_weights(y,k));null=k*k/n;row.update({f'top{k}_overlap':overlap,f'top{k}_random':null,f'top{k}_adjusted':(overlap-null)/(k-null) if k<n and not constant else np.nan})
     details.append(row)
 detail=pd.DataFrame(details);detail.to_parquet(out/'QUERY_AGREEMENT.parquet',index=False)
 fields=['spearman','kendall_tau_b','top5_overlap','top10_overlap','top20_overlap','top10_random','top10_adjusted'];summaries=[]
 for keys,f in detail.groupby(['view','direction','model_a','model_b'],sort=False):
  row=dict(zip(['view','direction','model_a','model_b'],keys));row.update(requested_queries=len(f),eligible_queries=int(f.eligible.sum()),constant_queries=int(f.get('constant_output',pd.Series(dtype=bool)).fillna(False).sum()),mean_common_candidates=float(f.loc[f.eligible,'n_common'].mean()))
  for col in fields:row['mean_'+col]=float(f[col].mean());row['median_'+col]=float(f[col].median())
  summaries.append(row)
 atomic_csv(pd.DataFrame(summaries),out/'RANK_AGREEMENT_SUMMARY.csv')
 selected=pd.concat(exports,ignore_index=True).merge(drugs[['drug_id','name']],on='drug_id',validate='many_to_one').merge(targets[['target_id','gene','target_name','assay_lane']],on='target_id',validate='many_to_one');selected.to_parquet(out/'BIDIRECTIONAL_TOP20_WITH_TIES.parquet',index=False)
 for direction in ['target_to_drug','drug_to_target']:
  selected.loc[selected.direction.eq(direction)].to_csv(out/f'{direction.upper()}_TOP20.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
 pd.concat(candidates,ignore_index=True).to_csv(out/'QUERY_CANDIDATE_COUNTS.csv',index=False)
 # Family is a descriptive grouping of target queries, not an architecture effect.
 family=detail.loc[detail.direction.eq('target_to_drug')].merge(targets[['target_id','assay_lane']],left_on='query_id',right_on='target_id',validate='many_to_one')
 fs=family.groupby(['view','model_a','model_b','assay_lane']).agg(queries=('query_id','size'),eligible_queries=('eligible','sum'),mean_spearman=('spearman','mean'),mean_top10_overlap=('top10_overlap','mean'),mean_top10_adjusted=('top10_adjusted','mean')).reset_index();atomic_csv(fs,out/'FAMILY_AGREEMENT.csv')
 dump(out/'MANIFEST.json',dict(created_utc=stamp,models=models,score_sha256={m:hashes[m] for m in models},coverage='Only finished model families, not the final12-family comparison',no_labels=True,no_correctness_claim=True,statistics='Descriptive per-query macro summaries; dependence-aware confidence intervals and nested-pool experiments deferred until complete frozen inference',tie_rule='average ranks, tau-b, fractional TopK; all cutoff ties retained in exports',source_matrix_shape=[720,745],files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--analyze',action='store_true');p.add_argument('--watch',action='store_true');args=p.parse_args()
 while True:
  try:rows=collect(args.analyze)
  except Exception as e:
   dump(OUT/'REPORT_ERROR.json',{'error':repr(e),'updated_utc':datetime.now(timezone.utc).isoformat()})
   if not args.watch:raise
  if not args.watch:break
  time.sleep(60)
if __name__=='__main__':main()

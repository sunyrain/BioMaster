#!/usr/bin/env python3
"""Audit model agreement on shared, near-complete per-target catalogs, not SPR ranks."""
from datetime import datetime,timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_data import ExplorerData
from biomaster.catalog_models import connect
OUT=ROOT/'outputs/model_agreement_20260917'
MODELS=['biomaster','drugclip','dtiam','conplex','nesso','probematch','dtbind']
BASE5=['biomaster','drugclip','dtiam','conplex','probematch']
PROBABILITY_MODELS={'dtiam','nesso','probematch','dtbind'}
warnings.filterwarnings('ignore',category=ConstantInputWarning)

def nesso_affinity_and_confidence(frame):
    rows=[]
    for r in frame.itertuples():
        name=f'{r.drug_id}__{r.target_id}'
        candidates=[ROOT/f'outputs/catalog_seven_models_20260916/nesso/targets/{r.target_id}/predictions/{name}/affinity.json',ROOT/f'outputs/frontier_dti_20260916/nesso/predictions/{name}/affinity.json']
        found=next((p for p in candidates if p.exists()),None)
        if found is not None:
            record=json.loads(found.read_text())
            rows.append(dict(drug_id=r.drug_id,target_id=r.target_id,gene=r.gene,nesso_pic50=6-float(record['affinity_pred_value']),entropy_crop_pl=record.get('entropy_crop_pl')))
    return pd.DataFrame(rows)

def topweights(s,k):
    cutoff=s.nlargest(k).iloc[-1];above=s.gt(cutoff);equal=s.eq(cutoff)
    w=above.astype(float);w.loc[equal]=(k-int(above.sum()))/int(equal.sum())
    return w,int(equal.sum()),int(above.sum())

def agreement(frame,models,kind,scope):
    key='target_id' if kind=='target' else 'drug_id';other='drug_id' if kind=='target' else 'target_id'
    rows=[]
    for identity,g in frame.groupby(key,sort=True):
        g=g.sort_values(other,kind='stable').set_index(other)
        if kind=='target':g=g.copy();g['biomaster']=g.biomaster_reverse
        for a,b in combinations(models,2):
            sub=g[[a,b]].dropna();n=len(sub)
            if n<50:continue
            entry=dict(scope=scope,kind=kind,entity_id=identity,model_a=a,model_b=b,pairs=n,spearman=float(spearmanr(sub[a],sub[b]).statistic))
            for k in [10,20,50]:
                first=set(sub[a].sort_values(ascending=False,kind='stable').head(k).index);second=set(sub[b].sort_values(ascending=False,kind='stable').head(k).index)
                wa,ta,_=topweights(sub[a],k);wb,tb,_=topweights(sub[b],k)
                entry.update({f'top{k}_intersection':len(first&second),f'top{k}_chance':k*k/n,f'top{k}_expected_under_tie_breaks':float((wa*wb).sum()),f'top{k}_boundary_ties_a':ta,f'top{k}_boundary_ties_b':tb})
            rows.append(entry)
    return pd.DataFrame(rows)

def measures(y,s,model):
    y=np.asarray(y,dtype=int);s=np.asarray(s);pos=y==1;neg=~pos
    result=dict(pairs=len(y),positive=int(pos.sum()),negative=int(neg.sum()),prevalence=float(y.mean()),ap=average_precision_score(y,s),auroc=roc_auc_score(y,s))
    # A 0.5 threshold has no shared interpretation for logits, cosine or pIC50.
    if model in PROBABILITY_MODELS:
        result.update(fpr_at_0_5=float((s[neg]>=.5).mean()),tpr_at_0_5=float((s[pos]>=.5).mean()))
    return result

def main():
    OUT.mkdir(exist_ok=True,parents=True);snapshot=OUT/'SCORE_SNAPSHOT.csv.gz'
    if snapshot.exists():frame=pd.read_csv(snapshot)
    else:
        data=ExplorerData(ROOT,infer=False);data.ensure_loaded()
        frame=data.pairs[['ligand_inchikey','target_chembl_id','drug_names','gene_symbol','biomaster','biomaster_reverse','drugclip','dtiam','conplex']].rename(columns={'ligand_inchikey':'drug_id','target_chembl_id':'target_id','drug_names':'drug_name','gene_symbol':'gene'})
        db=connect(ROOT);db.execute('BEGIN')
        current=pd.read_sql_query("SELECT model,drug_id,target_id,score FROM predictions WHERE status='completed'",db);db.commit();db.close()
        current=current.pivot(index=['drug_id','target_id'],columns='model',values='score').reset_index()
        frame=frame.merge(current,on=['drug_id','target_id'],how='left',validate='one_to_one')
        assert len(frame)==276480;frame.to_csv(snapshot,index=False)
        (OUT/'SNAPSHOT_META.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),scores_source='one SQLite read transaction + immutable website FP32 baseline',snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),successful_counts={m:int(frame[m].notna().sum()) for m in MODELS}),indent=2))
    counts=frame.groupby('target_id')[MODELS].count()
    complete_targets=counts.index[counts.min(axis=1)>=719]
    joint=frame[frame.target_id.isin(complete_targets)].dropna(subset=MODELS)
    common5=frame.dropna(subset=BASE5)
    comparisons=[agreement(joint,MODELS,'target','SEVEN_SAME_TARGETS'),agreement(common5,BASE5,'target','FIVE_ALL_TARGETS'),agreement(common5,BASE5,'drug','FIVE_ALL_DRUGS')]
    # Broader pairwise diagnostics: each pair uses its own near-complete targets.
    # Their different target sets must not be presented as a common-seven matrix.
    for a,b in combinations(MODELS,2):
        ready=counts.index[counts[[a,b]].min(axis=1)>=719]
        pool=frame[frame.target_id.isin(ready)].dropna(subset=[a,b])
        comparisons.append(agreement(pool,[a,b],'target','PAIRWISE_COMPLETE_TARGETS'))
    per=pd.concat(comparisons,ignore_index=True)
    per.to_csv(OUT/'PER_ENTITY_AGREEMENT.csv',index=False)
    metrics=['pairs','spearman']+[f'top{k}_{s}' for k in [10,20,50] for s in ['intersection','chance','expected_under_tie_breaks']]
    aggregate=per.groupby(['scope','kind','model_a','model_b'])[metrics].agg(['mean','median','count']);aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate=aggregate.reset_index();aggregate.to_csv(OUT/'AGREEMENT_SUMMARY.csv',index=False)
    joint[['target_id','gene']].drop_duplicates().to_csv(OUT/'SEVEN_SHARED_TARGETS.csv',index=False)
    joint.to_csv(OUT/'SEVEN_SHARED_SCORES.csv.gz',index=False)
    nesso_affinity_and_confidence(joint).to_csv(OUT/'NESSO_SHARED_AFFINITY_AND_CONFIDENCE.csv',index=False)
    # Score concentration over exactly the same shared target and drug universe.
    distributions=[]
    for scope,pool in [('seven_shared',joint),('AR',frame[frame.gene.eq('AR')].dropna(subset=MODELS))]:
        for m in MODELS:
            s=pool['biomaster_reverse' if m=='biomaster' else m]
            item=dict(scope=scope,model=m,pairs=len(s),unique_scores=s.nunique(),minimum=s.min(),median=s.median(),maximum=s.max())
            if m in PROBABILITY_MODELS:
                item.update(fraction_ge_0_5=s.ge(.5).mean(),fraction_ge_0_9=s.ge(.9).mean(),fraction_ge_0_99=s.ge(.99).mean(),fraction_ge_0_999=s.ge(.999).mean(),fraction_eq_1=s.eq(1).mean())
            distributions.append(item)
    pd.DataFrame(distributions).to_csv(OUT/'SCORE_DISTRIBUTIONS.csv',index=False)
    ar=frame[frame.gene.eq('AR')].dropna(subset=MODELS).sort_values('drug_id').copy();ar['biomaster_forward']=ar.biomaster;ar['biomaster']=ar.biomaster_reverse
    pi=[]
    for r in ar.itertuples():
        name=f'{r.drug_id}__{r.target_id}'
        candidates=[ROOT/f'outputs/catalog_seven_models_20260916/nesso/targets/{r.target_id}/predictions/{name}/affinity.json',ROOT/f'outputs/frontier_dti_20260916/nesso/predictions/{name}/affinity.json']
        found=next((p for p in candidates if p.exists()),None)
        pi.append(6-float(json.loads(found.read_text())['affinity_pred_value']) if found else np.nan)
    ar['nesso_pic50']=pi
    ar[MODELS+['nesso_pic50','biomaster_forward']].corr(method='spearman').to_csv(OUT/'AR_SPEARMAN.csv')
    top=[]
    for m in MODELS+['nesso_pic50']:
        ranked=ar.sort_values(m,ascending=False,kind='stable')
        for rank,r in enumerate(ranked.head(10).itertuples(),1):top.append(dict(model=m,rank=rank,drug_id=r.drug_id,drug_name=r.drug_name,score=getattr(r,m)))
        ar[m+'_rank']=ar[m].rank(ascending=False,method='average')
    pd.DataFrame(top).to_csv(OUT/'AR_TOP10.csv',index=False);ar.to_csv(OUT/'AR_719_SCORES_AND_RANKS.csv',index=False)
    # Real-label comparisons on a single common sample set; keep prespecified forward
    # ReTargetMap benchmark score, with reverse head shown separately, not substituted.
    bench_path=OUT/'BENCHMARK_INPUT_SNAPSHOT.csv'
    if not bench_path.exists():
        b=pd.read_csv(ROOT/'outputs/frontier_dti_20260916/ALL_PREDICTIONS.csv',low_memory=False)
        b.to_csv(bench_path,index=False)
    b=pd.read_csv(bench_path,low_memory=False);b=b[b.cohort.eq('BINDINGDB479')].dropna(subset=MODELS)
    b=b.merge(frame[['drug_id','target_id','biomaster_reverse']],on=['drug_id','target_id'],validate='one_to_one')
    bm=[]
    for scope,part in [('COMMON_LABELS',b),('AB_UNSEEN_COMMON',b[b.ab_unseen.eq(True)])]:
        for m in MODELS+['nesso_pic50','biomaster_reverse']:
            valid=part.dropna(subset=[m]);bm.append(dict(scope=scope,model=m,**measures(valid.label,valid[m],m)))
    pd.DataFrame(bm).to_csv(OUT/'COMMON_LABEL_METRICS.csv',index=False)
    thresholds=[]
    for m in sorted(PROBABILITY_MODELS):
        for cutoff in [.5,.9,.99]:
            chosen=b[m].ge(cutoff);positive=b.label.eq(1);negative=~positive
            thresholds.append(dict(model=m,threshold=cutoff,selected=int(chosen.sum()),true_positive=int((chosen&positive).sum()),false_positive=int((chosen&negative).sum()),precision=float(b.loc[chosen,'label'].mean()),fpr=float(chosen[negative].mean()),tpr=float(chosen[positive].mean())))
    pd.DataFrame(thresholds).to_csv(OUT/'BINARY_THRESHOLD_DIAGNOSTIC.csv',index=False)
    # Paired bootstrap by target: shared-target correlations remain within a cluster.
    rng=np.random.default_rng(20260917);target_ids=b.target_id.unique();groups={k:g for k,g in b.groupby('target_id')};deltas=[]
    for _ in range(1200):
        sample=pd.concat([groups[k] for k in rng.choice(target_ids,len(target_ids),replace=True)])
        if sample.label.nunique()<2:continue
        deltas.append(average_precision_score(sample.label,sample.nesso)-average_precision_score(sample.label,sample.biomaster))
    result=dict(snapshot_utc=json.loads((OUT/'SNAPSHOT_META.json').read_text())['created_utc'],shared_targets=len(complete_targets),shared_pairs=len(joint),per_target_pairs=sorted(joint.groupby('target_id').size().unique().tolist()),all_five_targets=common5.target_id.nunique(),all_five_pairs=len(common5),common_labels=len(b),nesso_minus_retargetmap_ap=float(average_precision_score(b.label,b.nesso)-average_precision_score(b.label,b.biomaster)),paired_target_bootstrap_95_ci=np.quantile(deltas,[.025,.975]).tolist(),bootstrap_replicates=len(deltas),biomaster_benchmark_direction='prespecified forward; reverse also exported separately',coverage_note='>=719 of 720 scored per target for all seven; then exact intersection, no SPR-only restriction; selected targets biased toward early short proteins',rank_note='same-query score direction; tied ranks averaged for Spearman; TopK ID tie break and randomized-tie expected overlap both exported')
    (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

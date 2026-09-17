#!/usr/bin/env python3
"""Render frozen comparison artifacts and validate their scope and arithmetic."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_model_agreement_20260917 import MODELS,nesso_affinity_and_confidence
OUT=ROOT/'outputs/model_agreement_20260917'
NAMES={'biomaster':'ReTargetMap','drugclip':'DrugCLIP','dtiam':'DTIAM (historical)',
       'conplex':'ConPLex','nesso':'Nesso-1','probematch':'ProbeMatchDTI','dtbind':'DTBind (occurrence)'}

def main():
    agg=pd.read_csv(OUT/'AGREEMENT_SUMMARY.csv')
    shared=pd.read_csv(OUT/'SEVEN_SHARED_SCORES.csv.gz')
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    assert not shared.duplicated(['drug_id','target_id']).any()
    assert len(shared)==5752 and shared.target_id.nunique()==8
    assert shared.groupby('target_id').size().eq(719).all()
    assert shared[MODELS].notna().all().all()
    a=agg[agg.scope.eq('SEVEN_SAME_TARGETS')]
    assert len(a)==21 and a.pairs_count.eq(8).all()
    assert np.allclose(a.top10_chance_mean,100/719)
    assert a.top10_intersection_mean.between(0,10).all()
    metrics=pd.read_csv(OUT/'COMMON_LABEL_METRICS.csv')
    main_metrics=metrics[metrics.scope.eq('COMMON_LABELS')]
    assert main_metrics.pairs.eq(378).all()
    assert main_metrics.positive.eq(103).all() and main_metrics.negative.eq(275).all()
    assert main_metrics[main_metrics.model.isin(['biomaster','drugclip','conplex','nesso_pic50','biomaster_reverse'])].fpr_at_0_5.isna().all()
    thresholds=pd.read_csv(OUT/'BINARY_THRESHOLD_DIAGNOSTIC.csv')
    assert (thresholds.selected==thresholds.true_positive+thresholds.false_positive).all()
    # Re-read raw outputs to make the supplemental confidence audit reproducible.
    confidence=nesso_affinity_and_confidence(shared)
    confidence.to_csv(OUT/'NESSO_SHARED_AFFINITY_AND_CONFIDENCE.csv',index=False)
    assert len(confidence)==len(shared)
    flagged=confidence[confidence.entropy_crop_pl.eq(0)]
    bad=shared.merge(flagged,on=['drug_id','target_id','gene'],validate='one_to_one')
    bad[['drug_id','target_id','drug_name','gene','nesso','nesso_pic50','entropy_crop_pl']].to_csv(OUT/'NESSO_CONFIDENCE_FLAGS.csv',index=False)

    # A different training objective within the same checkpoint can reorder drugs.
    enriched=shared.merge(confidence,on=['drug_id','target_id','gene'],validate='one_to_one')
    sensitivity=[]
    for target,g in enriched.groupby('target_id'):
        clean=g[g.entropy_crop_pl.ne(0)]
        sensitivity.append(dict(target_id=target,gene=g.gene.iloc[0],pairs=len(g),
            binder_vs_pic50_spearman=g.nesso.corr(g.nesso_pic50,method='spearman'),
            retargetmap_vs_nesso_spearman=g.biomaster_reverse.corr(g.nesso,method='spearman'),
            valid_confidence_pairs=len(clean),
            retargetmap_vs_nesso_valid_confidence_spearman=clean.biomaster_reverse.corr(clean.nesso,method='spearman')))
    pd.DataFrame(sensitivity).to_csv(OUT/'NESSO_HEAD_AND_CONFIDENCE_DIAGNOSTIC.csv',index=False)

    # Pooled AP can hide the distinction between ranking drugs and ranking targets.
    # Use exactly the same mixed-label queries for every model; never add unmeasured
    # catalog pairs as negatives. Small observed panels are not full-catalog tests.
    b=pd.read_csv(OUT/'BENCHMARK_INPUT_SNAPSHOT.csv',low_memory=False)
    score_models=MODELS+['nesso_pic50']
    b=b[b.cohort.eq('BINDINGDB479')].dropna(subset=score_models)
    direction=pd.read_csv(OUT/'SCORE_SNAPSHOT.csv.gz',usecols=['drug_id','target_id','biomaster_reverse'])
    b=b.merge(direction,on=['drug_id','target_id'],validate='one_to_one')
    query_metrics=[]
    for key in ['target_id','drug_id']:
        for identity,g in b.groupby(key):
            if g.label.nunique()<2:
                continue
            for model in score_models:
                scores=g['biomaster_reverse' if model=='biomaster' and key=='target_id' else model]
                query_metrics.append(dict(direction=key,entity_id=identity,model=model,n=len(g),
                    positive=int(g.label.sum()),prevalence=float(g.label.mean()),
                    ap=average_precision_score(g.label,scores),auc=roc_auc_score(g.label,scores)))
    queries=pd.DataFrame(query_metrics)
    queries.to_csv(OUT/'PER_QUERY_LABEL_METRICS.csv',index=False)
    query_summary=queries.groupby(['direction','model']).agg(queries=('ap','size'),pairs=('n','sum'),
        mean_prevalence=('prevalence','mean'),macro_ap=('ap','mean'),macro_auc=('auc','mean')).reset_index()
    query_summary.to_csv(OUT/'QUERY_LABEL_SUMMARY.csv',index=False)

    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    specs=[('spearman_mean','Mean within-target Spearman','RdBu_r',-.5,.5),
           ('top10_intersection_mean','Mean shared drugs in Top 10','Blues',0,3)]
    for ax,(field,title,cmap,vmin,vmax) in zip(axes,specs):
        matrix=np.full((7,7),np.nan)
        for row in a.itertuples():
            i,j=MODELS.index(row.model_a),MODELS.index(row.model_b)
            matrix[i,j]=matrix[j,i]=getattr(row,field)
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_xticks(range(7),[NAMES[m] for m in MODELS],rotation=48,ha='right')
        ax.set_yticks(range(7),[NAMES[m] for m in MODELS])
        for i in range(7):
            for j in range(7):
                val=matrix[i,j]
                label='—' if np.isnan(val) else f'{val:.2f}'
                dark=not np.isnan(val) and ((abs(val)>.3) if field=='spearman_mean' else val>1.6)
                ax.text(j,i,label,ha='center',va='center',color='white' if dark else '#13253a',fontsize=10)
        ax.set_title(title,pad=15)
        fig.colorbar(im,ax=ax,shrink=.62)
    fig.suptitle('Seven deployed scores: 8 shared targets × 719 drugs = 5,752 pairs\n'
                 'Full catalog intersection; target→drug ReTargetMap head; snapshot 2026-09-17 07:00 UTC',fontsize=14)
    fig.supxlabel('Random Top-10 overlap: 0.139 drugs. Early-completed target subset; not a full-catalog accuracy evaluation.',fontsize=10)
    fig.savefig(OUT/'SEVEN_MODEL_AGREEMENT.png',dpi=180)
    fig.savefig(OUT/'SEVEN_MODEL_AGREEMENT.svg')
    plt.close(fig)
    checks=dict(status='PASS',shared_pairs=len(shared),shared_targets=shared.target_id.nunique(),
                metrics_same_labeled_pairs=378,positive=103,negative=275,
                nesso_zero_interface_confidence=int(len(flagged)),snapshot_utc=summary['snapshot_utc'],
                checks=['exact pair uniqueness','same target/drug sets','correct Top10 random baseline',
                        'same labeled comparison set','no probability thresholds for nonprobability scores',
                        'threshold count consistency','all raw Nesso outputs found'])
    (OUT/'ANALYSIS_CHECK.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False))

if __name__=='__main__':main()

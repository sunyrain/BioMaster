#!/usr/bin/env python3
"""Summarize completed runs only; never turn a partial table into a selection."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from biomaster.ranking_audit import query_bootstrap
from prepare_biomaster_unified_interaction import write_json

OUT=ROOT/'outputs/biomaster_best_model_20260906'
ORIGINAL=ROOT/'outputs/biomaster_unified_interaction_20260906'


def records():
    for variant in ['global','capacity']:
        for seed in [20260921,20260922,20260923]:
            for cutoff in [2018,2020]:
                if cutoff==2018:base=OUT/'rolling_original/roll_2018/development'
                else:base=ORIGINAL/'development' if seed==20260921 else OUT/'reproduction/development'
                yield cutoff,variant,seed,base/variant/f'seed_{seed}'
    for folder in sorted((OUT/'optimization').glob('roll_*/*/seed_*')):
        yield int(folder.parent.parent.name.split('_')[1]),folder.parent.name,int(folder.name[5:]),folder


def main():
    rows=[];frames=[];runs=[];history={}
    for cutoff,variant,seed,folder in records():
        path=folder/'RESULT.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        if r.get('smoke_only') or r['train_max_year']!=cutoff:raise ValueError('invalid development result')
        if r.get('test_labels_used'):raise ValueError('test-used training result')
        m=r['validation']
        row=dict(cutoff=cutoff,variant=variant,seed=seed,selected_epoch=r['selected_epoch'],
            trained_epochs=r['trained_epochs'],selection=m['selection'],parameters=r['trainable_parameters'])
        for direction in ['d2t','t2d']:
            for key in ['ap','r20','measured_ap']:row[f'{direction}_{key}']=m[direction][key]
        rows.append(row);runs.append(file_identity(path))
        f=pd.read_csv(folder/'VALIDATION_QUERY_METRICS.csv',dtype={'query_id':str})
        f['cutoff']=cutoff;f['variant']=variant;f['seed']=seed;frames.append(f)
        history[(cutoff,variant,seed)]=json.loads((folder/'HISTORY.json').read_text())
    table=pd.DataFrame(rows).sort_values(['cutoff','variant','seed'])
    table.to_csv(OUT/'DEVELOPMENT_RUNS.csv',index=False)
    metrics=['selection','d2t_ap','t2d_ap','d2t_r20','t2d_r20','d2t_measured_ap','t2d_measured_ap']
    summary=table.groupby(['cutoff','variant'])[metrics].agg(['count','mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(OUT/'DEVELOPMENT_BY_ROLL.csv',index=False)
    complete=[]
    for variant,part in table.groupby('variant'):
        if len(part)==6 and set(zip(part.cutoff,part.seed))=={(c,s) for c in [2018,2020] for s in [20260921,20260922,20260923]}:
            row={'variant':variant,'parameters':int(part.parameters.iloc[0])}
            by_seed=part.groupby('seed')[metrics].mean()
            for key in metrics:
                row[key+'_mean']=float(by_seed[key].mean());row[key+'_seed_sd']=float(by_seed[key].std())
            complete.append(row)
    pd.DataFrame(complete).to_csv(OUT/'DEVELOPMENT_COMPLETE_CANDIDATES.csv',index=False)
    comparisons=[];exposures=[]
    all_queries=pd.concat(frames,ignore_index=True)
    for first,second in [('capacity','global'),('global_ema','global'),('global_query','global_ema'),('global_assay','global_query')]:
        for cutoff in [2018,2020]:
            a=all_queries[(all_queries.variant==first)&all_queries.cutoff.eq(cutoff)]
            b=all_queries[(all_queries.variant==second)&all_queries.cutoff.eq(cutoff)]
            if a.seed.nunique()!=3 or b.seed.nunique()!=3:continue
            # Average seeds WITHIN each biological query before query bootstrap;
            # three seeds must not be counted as three independent queries.
            for direction in ['d2t','t2d']:
                aq=a[a.direction.eq(direction)].groupby('query_id',as_index=False).ap.mean()
                bq=b[b.direction.eq(direction)].groupby('query_id',as_index=False).ap.mean()
                comparisons.append(dict(first=first,second=second,cutoff=cutoff,direction=direction,
                    seeds_averaged=3,interval_scope='paired queries within this roll; not seed or prospective uncertainty',
                    multiple_testing_corrected=False,**query_bootstrap(aq,bq)))
            for seed in [20260921,20260922,20260923]:
                ha,hb=history[(cutoff,first,seed)],history[(cutoff,second,seed)]
                equal=all(x['exposure_sha256']==y['exposure_sha256'] for x,y in zip(ha,hb))
                if not equal:raise ValueError(f'observed/retrieval exposure mismatch {first}/{second}/{cutoff}/{seed}')
                exposures.append(dict(first=first,second=second,cutoff=cutoff,seed=seed,
                    common_epochs=min(len(ha),len(hb)),observed_and_retrieval_exposure_identical=True))
    write_json(OUT/'DEVELOPMENT_PAIRED_QUERIES.json',dict(comparisons=comparisons))
    write_json(OUT/'DEVELOPMENT_EXPOSURE_AUDIT.json',dict(comparisons=exposures))
    expected={('global',c,s) for c in [2018,2020] for s in [20260921,20260922,20260923]}
    for v in ['capacity','global_ema','global_query','global_assay']:
        expected.update((v,c,s) for c in [2018,2020] for s in [20260921,20260922,20260923])
    present=set(zip(table.variant,table.cutoff,table.seed))
    state=dict(status='COMPLETE' if expected<=present else 'PARTIAL',utc=datetime.now(timezone.utc).isoformat(),
        completed_primary_runs=len(expected&present),expected_primary_runs=len(expected),
        missing=[dict(variant=v,cutoff=c,seed=s) for v,c,s in sorted(expected-present)],
        final_model_selected=False,producer=file_identity(Path(__file__)),runs=runs)
    write_json(OUT/'DEVELOPMENT_SUMMARY.json',state)
    print(summary[['cutoff','variant','selection_count','selection_mean','selection_std']].to_string(index=False))
    print(json.dumps({k:state[k] for k in ['status','completed_primary_runs','expected_primary_runs','final_model_selected']}))


if __name__=='__main__':main()

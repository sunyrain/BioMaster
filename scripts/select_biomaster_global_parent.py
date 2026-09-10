#!/usr/bin/env python3
"""Combine all global development runs and freeze a parent without test metrics."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from biomaster.ranking_audit import query_bootstrap
from prepare_biomaster_unified_interaction import write_json

OUT=ROOT/'outputs/biomaster_best_model_20260906'
ORIGINAL=ROOT/'outputs/biomaster_unified_interaction_20260906'
PROTOCOL=ROOT/'configs/biomaster_best_model_refinement_20260906.json'
SEEDS=[20260921,20260922,20260923]
NAMES=['global','capacity','global_ema','global_query','global_assay','bermol','bermol_morgan','morgan','drugclip_morgan']


def global_runs():
    for name in NAMES:
        for cutoff in [2018,2020]:
            for seed in SEEDS:
                if name in ['global','capacity']:
                    if cutoff==2018:base=OUT/'rolling_original/roll_2018/development'
                    else:base=ORIGINAL/'development' if seed==20260921 else OUT/'reproduction/development'
                    folder=base/name/f'seed_{seed}';recipe='baseline';representation='drugclip'
                elif name in ['bermol','bermol_morgan','morgan','drugclip_morgan']:
                    folder=OUT/'representations'/name/'optimization'/f'roll_{cutoff}'/'global_ema'/f'seed_{seed}'
                    recipe='global_ema';representation=name
                else:
                    folder=OUT/'optimization'/f'roll_{cutoff}'/name/f'seed_{seed}'
                    recipe=name;representation='drugclip'
                yield dict(name=name,cutoff=cutoff,seed=seed,folder=folder,recipe=recipe,representation=representation)


def summarize(freeze=False):
    records=[];frames=[];history={};identities=[];missing=[]
    for run in global_runs():
        path=run['folder']/'RESULT.json'
        if not path.exists():missing.append({k:v for k,v in run.items() if k!='folder'});continue
        r=json.loads(path.read_text())
        if r['seed']!=run['seed'] or r['train_max_year']!=run['cutoff'] or r.get('smoke_only') or r.get('test_labels_used'):
            raise ValueError(f'invalid global development result {path}')
        if file_identity(Path(r['checkpoint']['path']))!=r['checkpoint']:raise ValueError('checkpoint drift')
        m=r['validation'];row={k:v for k,v in run.items() if k!='folder'}
        row.update(folder=str(run['folder']),selected_epoch=r['selected_epoch'],parameters=r['trainable_parameters'],selection=m['selection'])
        for direction in ['d2t','t2d']:
            for metric in ['ap','r20','measured_ap']:row[f'{direction}_{metric}']=m[direction][metric]
        records.append(row);identities.append(file_identity(path))
        f=pd.read_csv(run['folder']/'VALIDATION_QUERY_METRICS.csv',dtype={'query_id':str})
        f['name']=run['name'];f['cutoff']=run['cutoff'];f['seed']=run['seed'];frames.append(f)
        history[(run['name'],run['cutoff'],run['seed'])]=json.loads((run['folder']/'HISTORY.json').read_text())
    table=pd.DataFrame(records)
    if table.empty:raise ValueError('no completed global development runs')
    table.to_csv(OUT/'ALL_GLOBAL_RUNS.csv',index=False)
    metrics=['selection','d2t_ap','t2d_ap','d2t_r20','t2d_r20','d2t_measured_ap','t2d_measured_ap']
    candidates=[]
    for name,part in table.groupby('name',sort=False):
        if len(part)!=6:continue
        if set(zip(part.cutoff,part.seed))!={(c,s) for c in [2018,2020] for s in SEEDS}:raise ValueError('incomplete seed/window grid')
        by_seed=part.groupby('seed')[metrics].mean()
        row=dict(name=name,parameters=int(part.parameters.iloc[0]),recipe=part.recipe.iloc[0],representation=part.representation.iloc[0])
        for metric in metrics:
            row[metric+'_mean']=float(by_seed[metric].mean());row[metric+'_seed_sd']=float(by_seed[metric].std())
        row['seed_composites']={int(s):float(v) for s,v in by_seed.selection.items()}
        row['window_composites']={int(c):float(v) for c,v in part.groupby('cutoff').selection.mean().items()}
        candidates.append(row)
    compact=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in candidates])
    compact.to_csv(OUT/'ALL_GLOBAL_CANDIDATES.csv',index=False)
    all_queries=pd.concat(frames,ignore_index=True);comparisons=[];exposure=[]
    for first,second in [('capacity','global'),('global_ema','global'),('global_query','global_ema'),
                         ('global_assay','global_query'),('bermol','global_ema'),('bermol_morgan','bermol'),('bermol_morgan','global_ema'),
                         ('bermol_morgan','morgan'),('drugclip_morgan','global_ema'),('drugclip_morgan','morgan'),('morgan','global_ema')]:
        for cutoff in [2018,2020]:
            a=all_queries[all_queries.name.eq(first)&all_queries.cutoff.eq(cutoff)]
            b=all_queries[all_queries.name.eq(second)&all_queries.cutoff.eq(cutoff)]
            if a.seed.nunique()!=3 or b.seed.nunique()!=3:continue
            for direction in ['d2t','t2d']:
                aq=a[a.direction.eq(direction)].groupby('query_id',as_index=False).ap.mean()
                bq=b[b.direction.eq(direction)].groupby('query_id',as_index=False).ap.mean()
                comparisons.append(dict(first=first,second=second,cutoff=cutoff,direction=direction,
                    seeds_averaged_within_query=3,multiple_testing_corrected=False,**query_bootstrap(aq,bq)))
            for seed in SEEDS:
                ha,hb=history[(first,cutoff,seed)],history[(second,cutoff,seed)]
                assert all(x['exposure_sha256']==y['exposure_sha256'] for x,y in zip(ha,hb))
                exposure.append(dict(first=first,second=second,cutoff=cutoff,seed=seed,
                    common_epochs=min(len(ha),len(hb)),observed_and_retrieval_exposure_identical=True))
    write_json(OUT/'ALL_GLOBAL_PAIRED_QUERIES.json',dict(comparisons=comparisons,scope='development queries; not prospective or seed uncertainty'))
    write_json(OUT/'ALL_GLOBAL_EXPOSURE.json',dict(comparisons=exposure))
    write_json(OUT/'ALL_GLOBAL_STATUS.json',dict(status='COMPLETE' if not missing else 'PARTIAL',completed=len(records),expected=54,
        missing=missing,candidates=candidates,producer=file_identity(Path(__file__)),results=identities,final_model_selected=False))
    if freeze:
        if missing:raise ValueError('all 54 global development runs required before parent selection')
        loss_count={'baseline':0,'global_ema':0,'global_query':1,'global_assay':2}
        selected=sorted(candidates,key=lambda r:(-r['selection_mean'],r['parameters'],loss_count[r['recipe']],r['name']))[0]
        parents=[]
        for run in global_runs():
            if run['name']!=selected['name']:continue
            r=json.loads((run['folder']/'RESULT.json').read_text());checkpoint=run['folder']/'BEST.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['identity']!=r['identity'] or state['cutoff']!=run['cutoff'] or state['seed']!=run['seed']:
                raise ValueError('parent checkpoint metadata mismatch')
            parents.append(dict(cutoff=run['cutoff'],seed=run['seed'],checkpoint=file_identity(checkpoint),
                                config=state['config'],selected_epoch=r['selected_epoch'],result=file_identity(run['folder']/'RESULT.json')))
        result=dict(status='FROZEN_GLOBAL_PARENT',selected=selected,candidates=candidates,parents=parents,
            fixed_refit_global_epochs=int(np.rint(np.median([p['selected_epoch'] for p in parents]))),
            test_labels_used_for_selection=False,selection_scope='two development windows, three seeds, nine global candidates',
            refinement_protocol=file_identity(PROTOCOL),producer=file_identity(Path(__file__)),results=identities)
        result=json.loads(json.dumps(result))
        path=OUT/'GLOBAL_PARENT_SELECTION.json'
        if path.exists():
            existing=json.loads(path.read_text())
            if {k:v for k,v in existing.items() if k!='utc'}!=result:raise ValueError('frozen parent selection drift')
        else:write_json(path,dict(utc=datetime.now(timezone.utc).isoformat(),**result))
        print(json.dumps(dict(selected=selected['name'],composite=selected['selection_mean'],refit_epochs=result['fixed_refit_global_epochs'])))
    print(compact[['name','selection_mean','selection_seed_sd','d2t_ap_mean','t2d_ap_mean']].to_string(index=False))
    return candidates


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze-global',action='store_true')
    summarize(parser.parse_args().freeze_global)

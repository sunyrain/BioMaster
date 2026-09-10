#!/usr/bin/env python3
"""Freeze one architecture after all matched local ablations, without test reads."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from biomaster.ranking_audit import query_bootstrap
from prepare_biomaster_unified_interaction import write_json

OUT=ROOT/'outputs/biomaster_best_model_20260906'
PROTOCOL=ROOT/'configs/biomaster_best_model_refinement_20260906.json'


def summarize(freeze=False):
    selected=json.loads((OUT/'GLOBAL_PARENT_SELECTION.json').read_text());protocol=json.loads(PROTOCOL.read_text())
    if selected['refinement_protocol']!=file_identity(PROTOCOL):raise ValueError('refinement rule changed after parent selection')
    runs=[]
    for parent in selected['parents']:
        runs.append((parent['cutoff'],parent['seed'],'parent',Path(parent['result']['path']).parent))
    for cutoff in protocol['cutoffs']:
        for seed in protocol['seeds']:
            for variant in protocol['variants']:
                runs.append((cutoff,seed,variant,OUT/'anchored'/f'cutoff_{cutoff}'/variant/f'seed_{seed}'))
    records=[];frames=[];history={};missing=[];identities=[]
    for cutoff,seed,variant,folder in runs:
        path=folder/'RESULT.json'
        if not path.exists():missing.append(dict(cutoff=cutoff,seed=seed,variant=variant));continue
        r=json.loads(path.read_text())
        if r['train_max_year']!=cutoff or r.get('smoke_only') or r.get('test_labels_used'):raise ValueError('invalid ablation result')
        if file_identity(Path(r['checkpoint']['path']))!=r['checkpoint']:raise ValueError('checkpoint drift')
        if variant!='parent':
            if r['selected_epoch']!=protocol['extra_epochs'] or r['trained_epochs']!=protocol['extra_epochs']:
                raise ValueError('unequal extra training exposure')
            parent=next(p for p in selected['parents'] if p['cutoff']==cutoff and p['seed']==seed)
            if r['identity']['parent']!=parent['checkpoint']:raise ValueError('wrong warm initialization')
        m=r['validation'];row=dict(cutoff=cutoff,seed=seed,variant=variant,selection=m['selection'],parameters=r['trainable_parameters'],folder=str(folder))
        for direction in ['d2t','t2d']:
            for metric in ['ap','r20','measured_ap']:row[f'{direction}_{metric}']=m[direction][metric]
        records.append(row);identities.append(file_identity(path))
        f=pd.read_csv(folder/'VALIDATION_QUERY_METRICS.csv',dtype={'query_id':str})
        f['cutoff']=cutoff;f['seed']=seed;f['variant']=variant;frames.append(f)
        history[(cutoff,seed,variant)]=json.loads((folder/'HISTORY.json').read_text())
    table=pd.DataFrame(records);table.to_csv(OUT/'ANCHORED_RUNS.csv',index=False)
    metrics=['selection','d2t_ap','t2d_ap','d2t_r20','t2d_r20','d2t_measured_ap','t2d_measured_ap']
    summary={}
    for variant,part in table.groupby('variant'):
        if len(part)!=6:continue
        by_seed=part.groupby('seed')[metrics].mean();by_window=part.groupby('cutoff')[metrics].mean()
        summary[variant]=dict(variant=variant,parameters=int(part.parameters.iloc[0]),
            mean={k:float(by_seed[k].mean()) for k in metrics},seed_sd={k:float(by_seed[k].std()) for k in metrics},
            seed_composites={int(s):float(v) for s,v in by_seed.selection.items()},
            window_composites={int(c):float(v) for c,v in by_window.selection.items()})
    pd.DataFrame([dict(variant=v,parameters=r['parameters'],**{k+'_mean':x for k,x in r['mean'].items()},
        **{k+'_seed_sd':x for k,x in r['seed_sd'].items()}) for v,r in summary.items()]).to_csv(OUT/'ANCHORED_CANDIDATES.csv',index=False)
    exposures=[];comparisons=[];qall=pd.concat(frames,ignore_index=True)
    for cutoff in protocol['cutoffs']:
        for seed in protocol['seeds']:
            key=(cutoff,seed,'global')
            if key not in history:continue
            ref=history[key]
            for variant in ['capacity','site','geometry']:
                key=(cutoff,seed,variant)
                if key not in history:continue
                h=history[key]
                for a,b in zip(h,ref):
                    for field in ['exposure_sha256','query_exposure_sha256','assay_exposure_sha256','coverage_rows']:
                        if a[field]!=b[field]:raise ValueError('matched-control exposure mismatch')
                exposures.append(dict(cutoff=cutoff,seed=seed,variant=variant,epochs=len(h),all_exposures_identical=True))
    for a,b in [('global','parent'),('capacity','global'),('site','global'),('site','capacity'),
                ('geometry','global'),('geometry','capacity'),('geometry','site')]:
        for cutoff in protocol['cutoffs']:
            fa=qall[qall.variant.eq(a)&qall.cutoff.eq(cutoff)];fb=qall[qall.variant.eq(b)&qall.cutoff.eq(cutoff)]
            if fa.seed.nunique()!=3 or fb.seed.nunique()!=3:continue
            for direction in ['d2t','t2d']:
                qa=fa[fa.direction.eq(direction)].groupby('query_id',as_index=False).ap.mean()
                qb=fb[fb.direction.eq(direction)].groupby('query_id',as_index=False).ap.mean()
                comparisons.append(dict(candidate=a,control=b,cutoff=cutoff,direction=direction,seeds_averaged_within_query=3,
                    multiple_testing_corrected=False,**query_bootstrap(qa,qb)))
    write_json(OUT/'ANCHORED_EXPOSURE_AUDIT.json',dict(comparisons=exposures))
    write_json(OUT/'ANCHORED_PAIRED_QUERIES.json',dict(comparisons=comparisons))
    write_json(OUT/'ANCHORED_SUMMARY.json',dict(status='PARTIAL' if missing else 'COMPLETE',completed=len(records),expected=30,
        missing=missing,candidates=summary,producer=file_identity(Path(__file__)),results=identities,final_model_selected=False))
    if freeze:
        if missing:raise ValueError('all 24 refinements and six parent results required')
        def difference(a,b):
            x,y=summary[a],summary[b]
            seed=[x['seed_composites'][s]-y['seed_composites'][s] for s in protocol['seeds']]
            windows=[x['window_composites'][c]-y['window_composites'][c] for c in protocol['cutoffs']]
            return dict(mean=x['mean']['selection']-y['mean']['selection'],seed_deltas=seed,window_deltas=windows,
                        repeatable=bool(all(v>0 for v in windows) and sum(v>0 for v in seed)>=2))
        eligibility={'parent':dict(eligible=True,reason='selected global parent'),
                     'global':dict(eligible=True,reason='same architecture, four additional epochs and EMA')}
        cap=difference('capacity','global')
        eligibility['capacity']=dict(eligible=cap['repeatable'],versus_global=cap)
        for variant in ['site','geometry']:
            vs_global=difference(variant,'global');vs_capacity=difference(variant,'capacity')
            eligible=vs_global['repeatable'] and vs_capacity['mean']>0
            row=dict(eligible=eligible,versus_global=vs_global,versus_capacity=vs_capacity)
            if variant=='geometry':
                vs_site=difference('geometry','site');row['versus_site']=vs_site
                row['eligible']=bool(eligible and vs_site['repeatable'])
            eligibility[variant]=row
        allowed=[v for v in summary if eligibility[v]['eligible']]
        winner=sorted(allowed,key=lambda v:(-summary[v]['mean']['selection'],summary[v]['parameters'],v!='parent',v))[0]
        seed_values=summary[winner]['seed_composites'];median=float(np.median(list(seed_values.values())))
        seed=min(protocol['seeds'],key=lambda s:(abs(seed_values[s]-median),s))
        result=dict(status='FROZEN_FINAL_ARCHITECTURE',selected_variant=winner,global_model=selected['selected']['name'],
            representation=selected['selected']['representation'],recipe=selected['selected']['recipe'],
            deployment_seed=seed,seed_rule=protocol['representative_deployment_seed'],
            fixed_refit_global_epochs=selected['fixed_refit_global_epochs'],
            refinement_epochs=0 if winner=='parent' else protocol['extra_epochs'],
            selected_development=summary[winner],eligibility=eligibility,candidates=summary,
            global_parent_selection=file_identity(OUT/'GLOBAL_PARENT_SELECTION.json'),refinement_protocol=file_identity(PROTOCOL),
            test_labels_used_for_selection=False,tests_are_previously_opened_regressions=True,
            results=identities,producer=file_identity(Path(__file__)))
        result=json.loads(json.dumps(result))
        path=OUT/'FINAL_ARCHITECTURE_SELECTION.json'
        if path.exists():
            old=json.loads(path.read_text())
            if {k:v for k,v in old.items() if k!='utc'}!=result:raise ValueError('frozen final selection drift')
        else:write_json(path,dict(utc=datetime.now(timezone.utc).isoformat(),**result))
        print(json.dumps(dict(selected=winner,global_model=result['global_model'],seed=seed,
            composite=summary[winner]['mean']['selection'],eligibility=eligibility)))
    print(pd.DataFrame([dict(variant=v,**r['mean']) for v,r in summary.items()]).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');summarize(p.parse_args().freeze)

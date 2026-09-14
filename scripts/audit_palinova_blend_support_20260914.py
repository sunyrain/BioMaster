#!/usr/bin/env python3
"""Post-hoc support-stratum audit of frozen predictions; never fit a router."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/palinova_a_train_only_blend_20260914'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection=json.loads((OUT/'SELECTION.json').read_text())
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    assert summary['status']=='COMPLETE_VALIDATION_SELECTED_TRAIN_ONLY_BLEND'
    before=sha(OUT/'SELECTION.json')
    frame=pd.read_parquet(OUT/'COMMON_TEST_BLEND_PREDICTIONS.parquet')
    rows=[]
    for panel,g in frame.groupby('panel'):
        strata=[('all',g),('no_train_positive_target',g[g.eligible_positive_neighbors.eq(0)]),
                ('has_train_positive_target',g[g.eligible_positive_neighbors.gt(0)]),
                ('neighbor_lt_0.3',g[g.positive_neighbor.lt(.3)]),
                ('neighbor_ge_0.3',g[g.positive_neighbor.ge(.3)])]
        for scope,sub in strata:
            if sub.empty:continue
            for name in ['A_neural','A_plus_neighbor','A_plus_prior','A_joint_blend']:
                cal=selection['calibrations'][name]
                pred=expit(cal['slope']*sub[name+'_score'].to_numpy()+cal['intercept'])>=cal['threshold']
                y=sub.binary_label.to_numpy(int);tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum())
                rows.append(dict(panel=panel,scope=scope,model=name,pairs=len(sub),positive=int(y.sum()),
                    negative=int((y==0).sum()),tp=tp,fp=fp,recall=tp/int(y.sum()) if y.sum() else None,
                    fpr=fp/int((y==0).sum()) if (y==0).sum() else None))
    pd.DataFrame(rows).to_csv(OUT/'SUPPORT_STRATIFIED_DIAGNOSTIC.csv',index=False)
    frozen=pd.read_csv(OUT/'FROZEN384_BLEND_RANK_DIAGNOSTIC.csv')
    core=pd.read_parquet(OUT/'CATALOG_BLEND_COMPONENTS_AND_RANKS.parquet',
                         columns=['pair_id','eligible_positive_neighbors','same_connectivity_references_excluded'])
    frozen=frozen.merge(core,on='pair_id',validate='one_to_one')
    frozen['support_group']=np.select([frozen.eligible_positive_neighbors.eq(0),frozen.positive_neighbor.lt(.3)],
        ['no_train_positive_target','positive_target_but_neighbor_lt_0.3'],default='neighbor_ge_0.3')
    frozen.to_csv(OUT/'FROZEN384_SUPPORT_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
    groups=[]
    for group,f in frozen.groupby('support_group'):
        groups.append(dict(group=group,pairs=len(f),
            A_top20=int(f.A_neural_rank384.le(20).sum()),
            A_neighbor_top20=int(f.A_plus_neighbor_rank384.le(20).sum()),
            A_blend_top20=int(f.A_joint_blend_rank384.le(20).sum())))
    pd.DataFrame(groups).to_csv(OUT/'FROZEN384_SUPPORT_SUMMARY.csv',index=False)
    assert sha(OUT/'SELECTION.json')==before
    note=dict(status='COMPLETE_POST_HOC_DIAGNOSTIC',selection_sha256=before,
        source_sha256={p.name:sha(p) for p in [OUT/'COMMON_TEST_BLEND_PREDICTIONS.parquet',
            OUT/'FROZEN384_BLEND_RANK_DIAGNOSTIC.csv',Path(__file__)]},
        stratum_rule='0.3 is a post-hoc descriptive Tanimoto threshold, not a selected routing threshold; zero eligible pool is an exact training-support absence.',
        unchanged=['neural weights','reference pool','blend coefficients','calibration thresholds','original384'],
        caveat='Support categories can differ in targets, molecules and assays; this is not causal attribution. No fallback gate was tuned on these results.',
        experimental_candidate_support=groups)
    (OUT/'SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(note,indent=2,ensure_ascii=False)+'\n')
    print(pd.DataFrame(groups).to_string(index=False))


if __name__=='__main__':main()

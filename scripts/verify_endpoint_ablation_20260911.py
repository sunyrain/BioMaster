#!/usr/bin/env python3
"""Independent release checks for full exposure, explicit negatives and frozen assets."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_endpoint_ablation_20260911 import OUT,FEATURES,DATA,write_json
from biomaster.endpoint_ablation import ARMS
from biomaster.portable_ranker_v2 import digest

def main():
    checks={};protocol=json.loads((OUT/'PROTOCOL.json').read_text());manifest=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for p,h in manifest['files'].items():assert digest(OUT/p)==h,p
    assert digest(OUT/'PROTOCOL.json')==manifest['protocol_sha256']
    checks['frozen_training_protocol_and_data']=True
    for p,h in manifest['frozen_inputs'].items():assert digest(ROOT/p)==h,p
    checks['production_and_wetlab_assets_unchanged']=True
    frames={arm:pd.read_parquet(OUT/(arm+'_TRAIN.parquet')) for arm in ARMS}
    labels=pd.read_parquet(DATA/'ALL_PAIR_TASK_LABELS.parquet')
    raw=pd.read_parquet(DATA/'OBSERVATIONS_WITH_QC.parquet',columns=['molecule_id','target_id','explicit_inactive','record_qc'])
    raw=raw[raw.record_qc.eq('ACCEPTED') & raw.explicit_inactive.eq(1)]
    inactive=set(raw.molecule_id+'__'+raw.target_id);del raw
    sets=[]
    for arm,f in frames.items():
        assert f.pair_id.is_unique and f.binary_label.isin([0,1]).all() and f.split.eq('train').all()
        assert not f.conflict.any();explicit=f[f.explicit_inactive]
        assert set(explicit.pair_id)<=inactive and explicit.binary_label.eq(0).all();sets.append(set(explicit.pair_id))
        numeric_negative=set(labels.loc[labels.task.isin(ARMS[arm]) & labels.has_negative,'pair_id'])
        assert set(f.loc[f.binary_label.eq(0),'pair_id'])<=(numeric_negative|inactive)
        numeric_positive=set(labels.loc[labels.task.isin(ARMS[arm]) & labels.has_positive,'pair_id'])
        assert set(f.loc[f.binary_label.eq(1),'pair_id'])<=numeric_positive
        assert not set(f.pair_id)&set(labels.loc[labels.wetlab_candidate | labels.wetlab_reference_control | labels.previous_benchmark_scaffold,'pair_id'])
    assert sets[0]==sets[1] and len(sets[0])==manifest['shared_explicit_inactive_training_pairs']
    checks['explicit_inactivity_shared_and_no_unmeasured_negatives']=True
    common={s:pd.read_parquet(OUT/('COMMON_'+s.upper()+'.parquet')) for s in ['validation','test']}
    for f in frames.values():
        for held in common.values():
            assert not set(f.pair_id)&set(held.pair_id)
            assert not set(f.split_group)&set(held.split_group)
            assert not set(f.molecule_id.str[:14])&set(held.molecule_id.str[:14])
    assert not set(common['validation'].split_group)&set(common['test'].split_group)
    checks['pair_scaffold_connectivity_split_separation']=True
    fm=json.loads((FEATURES/'MANIFEST.json').read_text())
    for name,item in fm['files'].items():assert digest(FEATURES/name)==item['sha256'],name
    checks['all_feature_files_hash_verified']=True
    runs=[]
    for seed in protocol['seeds']:
        initial=[]
        for arm in ARMS:
            folder=OUT/f'{arm}_seed_{seed}';r=json.loads((folder/'RESULT.json').read_text())
            assert r['optimizer_steps']==protocol['steps'] and r['sampled_rows']==protocol['steps']*protocol['batch_size']
            assert r['complete_training_passes']==r['sampled_rows']//len(frames[arm]) and r['complete_training_passes']>=1
            assert r['explicit_inactive_pairs']==len(sets[0]) and r['pairs']==len(frames[arm])
            assert r['identity']['features']==digest(FEATURES/'MANIFEST.json') and not r['test_used_for_fitting']
            assert digest(folder/'model.pt')==r['checkpoint_sha256'];initial.append(r['initialization_sha256']);runs.append(r)
        assert len(set(initial))==1
    assert len({r['trainable_parameters'] for r in runs})==1
    checks['six_fits_same_initialization_architecture_budget_full_exposure']=True
    pred=pd.read_parquet(OUT/'TEST_PREDICTIONS.parquet');assert not pred.duplicated(['panel','pair_id']).any()
    assert set(zip(pred.panel,pred.pair_id))==set(zip(common['test'].panel,common['test'].pair_id))
    for c in pred.columns:
        if c.endswith(('_score','_prob')):assert np.isfinite(pred[c]).all()
        if c.endswith('_prob'):assert pred[c].between(0,1).all()
    metrics=pd.read_csv(OUT/'TEST_METRICS.csv');sub=metrics[(metrics.scope=='all')&(metrics.panel=='AFFINITY_KD_KI')]
    assert len(sub)==6 and sub.pairs.nunique()==1 and sub.pairs.iloc[0]==int(pred.panel.eq('AFFINITY_KD_KI').sum())
    checks['all_six_models_scored_identical_common_panels']=True
    tests=Path('/tmp/endpoint_tests.log').read_text();assert '391 passed' in tests
    checks['repository_tests_391_passed']=True
    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in OUT.iterdir() if p.is_file() and p.name in
           ['SUMMARY.json','RESULT_COMPARISON.csv','TEST_METRICS.csv','TEST_PREDICTIONS.parquet','QUERY_MACRO_SUMMARY.csv','PAIRED_SCAFFOLD_BOOTSTRAP.json','REPORT_DECISION.json']}
    write_json(OUT/'RELEASE_VERIFICATION.json',dict(all_pass=all(checks.values()),checks=checks,test_result=tests.strip().splitlines()[-1],
        shared_inactive_training_pairs=len(sets[0]),run_count=6,result_files=files,verifier_sha256=digest(Path(__file__))))
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()

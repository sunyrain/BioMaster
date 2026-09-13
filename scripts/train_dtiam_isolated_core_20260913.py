#!/usr/bin/env python3
"""Fit each full-B native learner in a fresh process on the same cached inner split."""
import argparse
import copy
import json
import random
import resource
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

from dtiam_ab_common_20260912 import ROOT,OUT,SOURCE,SEEDS,now,digest,write_json,table,banks
# Interrupted native trainer pickles can reference __main__.Progress.
from train_dtiam_ab_20260912 import Progress,EXPECTED


def core_plan(defaults):
    d=copy.deepcopy(defaults)
    plan={
        'LightGBMXT':{'GBM':[d['GBM'][0]]},
        'LightGBM':{'GBM':[d['GBM'][1]]},
        'RandomForestGini':{'RF':[d['RF'][0]]},
        'RandomForestEntr':{'RF':[d['RF'][1]]},
        'CatBoost':{'CAT':d['CAT']},
        'ExtraTreesGini':{'XT':[d['XT'][0]]},
        'ExtraTreesEntr':{'XT':[d['XT'][1]]},
        'XGBoost':{'XGB':d['XGB']},
        'NeuralNetTorch':{'NN_TORCH':d['NN_TORCH']},
        'LightGBMLarge':{'GBM':[d['GBM'][2]]},
    }
    assert set(plan)==EXPECTED-{'NeuralNetFastAI','WeightedEnsemble_L2'}
    return plan


def member_roles(predictor,frame):
    _,yt=predictor.load_data_internal('train',return_X=False,return_y=True)
    _,yv=predictor.load_data_internal('val',return_X=False,return_y=True)
    ti=yt.index.to_numpy(int);vi=yv.index.to_numpy(int)
    assert len(set(ti))==len(ti) and len(set(vi))==len(vi)
    assert set(ti).isdisjoint(vi) and set(ti)|set(vi)==set(range(len(frame)))
    np.testing.assert_array_equal(yt.to_numpy(),frame.binary_label.to_numpy()[ti])
    np.testing.assert_array_equal(yv.to_numpy(),frame.binary_label.to_numpy()[vi])
    roles=frame[['pair_id','binary_label']].copy();roles['native_role']='base_train'
    roles.loc[vi,'native_role']='internal_holdout'
    return roles,len(ti),len(vi)


def inventory(predictor,run):
    assert predictor.model_failures().empty, 'Preserve and inspect native failed models before resuming'
    return {name:digest(run/'predictor/models'/name/'model.pkl') for name in predictor.model_names()}


def verify_preserved(predictor,run,before,added):
    after=inventory(predictor,run)
    assert set(after)==set(before)|set(added), 'Missing or unexpectedly renamed native learners'
    for name,sha in before.items():
        assert after[name]==sha, f'Previously completed model changed: {name}'
    return after


def work(seed,model=None,finalize=False):
    arm='all_inactive';run=OUT/f'{arm}__seed_{seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'SELECTION.json').exists() or (run/'FIT_RETURNED.json').exists():return
    cfg=json.loads((OUT/'PROTOCOL.json').read_text())['fit']
    defaults=get_hyperparameter_config('default')
    assert defaults==json.loads((OUT/'AUTOGLUON_HYPERPARAMETERS.json').read_text())
    plan=core_plan(defaults)
    assert list(plan)==cfg['isolated_B_core_model_order']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(20)
    frame=pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet',columns=['pair_id','binary_label'])
    record_dir=run/'ISOLATED_CORE';record_dir.mkdir(exist_ok=True)
    existing=(run/'predictor').exists()
    if existing:
        predictor=TabularPredictor.load(str(run/'predictor'))
        before=inventory(predictor,run)
        assert set(before)<=set(plan)
        roles,nt,nv=member_roles(predictor,frame)
        role_path=run/'INTERNAL_MEMBER_ROLES.parquet'
        if role_path.exists():assert roles.equals(pd.read_parquet(role_path))
        else:roles.to_parquet(role_path,index=False)
        if not (record_dir/'INITIAL_STATE.json').exists():
            info=predictor.info()
            write_json(record_dir/'INITIAL_STATE.json',dict(utc=now(),inherited_models=before,
                inherited_native_fit_seconds=sum(m['fit_time'] for m in info['model_info'].values()),
                preprocessing_seconds=info.get('time_fit_preprocessing',0),
                members_sha256=digest(role_path),
                cache_sha256={n:digest(run/'predictor/utils/data'/n) for n in ['X.pkl','y.pkl','X_val.pkl','y_val.pkl']}))
    elif finalize or model!='LightGBMXT':
        raise RuntimeError('First isolated B learner must initialize the original native inner split')
    if finalize:
        assert set(before)==set(plan), 'All ten core learners are required before finalization'
        state=json.loads((record_dir/'INITIAL_STATE.json').read_text())
        for name,sha in state['cache_sha256'].items():
            assert digest(run/'predictor/utils/data'/name)==sha, 'Cached native inner data changed'
        records=[json.loads(p.read_text()) for p in record_dir.glob('MODEL_*.json')]
        duration=state['inherited_native_fit_seconds']+(state.get('preprocessing_seconds') or 0)+sum(r['seconds'] for r in records)
        peak=max([r['peak_RSS_GiB'] for r in records]+[resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2])
        # Include the preprocessing peak of the preserved, interrupted bootstrap.
        history=run/'ISOLATED_CORE_RECOVERY_20260913/FIT_STATUS.json'
        if history.exists():peak=max(peak,json.loads(history.read_text()).get('memory_GiB',0))
        write_json(run/'AUTOGLUON_INFO.json',predictor.info())
        predictor.leaderboard(silent=True).to_csv(run/'AUTOGLUON_LEADERBOARD.csv',index=False)
        predictor.model_failures().to_csv(run/'MODEL_FAILURES.csv',index=False)
        write_json(run/'FIT_RETURNED.json',dict(fit_seconds=duration,training_member_pool=len(frame),
            base_train_rows=nt,internal_holdout_rows=nv,native_best=predictor.model_best,
            model_names=predictor.model_names(),peak_RSS_GiB=peak,finished_utc=now(),smoke_only=False,
            core_soft_memory_limit_GB=cfg['core_memory_limit_GB_by_arm'][arm],
            execution='One fresh process per native core learner; exact same cached inner split',
            duration_note='Inherited successful native fit plus preprocessing and isolated process fit durations; excludes failed checks and downtime'))
        return
    if existing and model in before:return
    started=time.monotonic()
    write_json(run/'FIT_STATUS.json',dict(stage='FITTING_ISOLATED_CORE',model=model,
        completed=sorted(before) if existing else [],updated_utc=now(),training_member_pool=len(frame)))
    if existing:
        # fit_extra has no public raise_on_model_failure kwarg in frozen AG1.4.
        # Retain strict failure behavior; this does not change learner parameters.
        predictor._trainer.raise_on_model_failure=True
        predictor.fit_extra(hyperparameters=plan[model],time_limit=None,num_cpus=20,num_gpus=0,
            memory_limit=cfg['core_memory_limit_GB_by_arm'][arm],fit_strategy='sequential',fit_weighted_ensemble=False)
        verify_preserved(predictor,run,before,[model])
    else:
        if shutil.disk_usage(OUT).free<35*2**30:raise RuntimeError('Insufficient disk headroom for full B')
        full=pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet')
        x=table(full,banks(),labels=True)
        predictor=TabularPredictor(label='y',problem_type='binary',eval_metric='roc_auc',
            path=str(run/'predictor'),verbosity=3,learner_kwargs={'random_state':seed})
        predictor.fit(x,presets=None,hyperparameters=plan[model],time_limit=None,num_cpus=20,num_gpus=0,
            memory_limit=cfg['core_memory_limit_GB_by_arm'][arm],fit_strategy='sequential',
            raise_on_model_failure=True,calibrate_decision_threshold=False,fit_weighted_ensemble=False)
        verify_preserved(predictor,run,{},[model])
        roles,nt,nv=member_roles(predictor,frame)
        roles.to_parquet(run/'INTERNAL_MEMBER_ROLES.parquet',index=False)
        # The bootstrap is counted by its MODEL record, not again as inherited work.
        write_json(record_dir/'INITIAL_STATE.json',dict(utc=now(),inherited_models={},
            inherited_native_fit_seconds=0,preprocessing_seconds=0,
            members_sha256=digest(run/'INTERNAL_MEMBER_ROLES.parquet'),
            cache_sha256={n:digest(run/'predictor/utils/data'/n) for n in ['X.pkl','y.pkl','X_val.pkl','y_val.pkl']}))
    write_json(record_dir/f'MODEL_{model}.json',dict(status='COMPLETE',utc=now(),model=model,
        seconds=time.monotonic()-started,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        model_sha256=digest(run/'predictor/models'/model/'model.pkl'),
        fitted_parameters=predictor.info()['model_info'][model]['hyperparameters_fit']))
    write_json(run/'FIT_STATUS.json',dict(stage='ISOLATED_CORE_MODEL_FINISHED',model=model,
        completed=predictor.model_names(),updated_utc=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--model');group.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    try:work(args.seed,args.model,args.finalize)
    except Exception as error:
        write_json(OUT/f'all_inactive__seed_{args.seed}'/'ERROR.json',dict(utc=now(),error=repr(error),traceback=traceback.format_exc()))
        raise

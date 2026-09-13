#!/usr/bin/env python3
"""Check native parity and full-B memory before enabling dense numeric transport."""
import argparse
import ctypes
import gc
import os
import resource
import time

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
from dtiam_numeric_xgboost_20260913 import NumericDenseXGBoostModel
from dtiam_ab_common_20260912 import ROOT,OUT,SOURCE,SEEDS,now,digest,write_json
from train_dtiam_isolated_core_20260913 import member_roles


def main(seed):
    os.environ['AG_MEMORY_LIMIT_IN_GB']='84'
    run=OUT/f'all_inactive__seed_{seed}'
    p=TabularPredictor.load(str(run/'predictor'))
    frame=pd.read_parquet(SOURCE/'all_inactive_TRAIN.parquet',columns=['pair_id','binary_label'])
    member_roles(p,frame)
    x,y,xv,yv=p._trainer.load_data()
    assert x.shape==(1105107,2048) and xv.shape==(11163,2048)
    assert all(d==np.dtype('float32') for d in x.dtypes)
    path=OUT/'XGBOOST_DENSE_PROBE_20260913';path.mkdir(exist_ok=True)
    predictions=[];reload_differences=[]
    for cls,name in [(XGBoostModel,'native_sparse'),(NumericDenseXGBoostModel,'dense_numeric')]:
        m=cls(path=str(path),name=name,problem_type='binary',eval_metric='roc_auc',hyperparameters={'n_estimators':6})
        m.fit(X=x.iloc[:2048],y=y.iloc[:2048],X_val=xv.iloc[:512],y_val=yv.iloc[:512],num_cpus=4,num_gpus=0,verbosity=0)
        pred=m.predict_proba(xv.iloc[:512]);predictions.append(pred)
        m.save();reloaded=cls.load(m.path)
        reloaded_pred=reloaded.predict_proba(xv.iloc[:512])
        reload_differences.append(float(abs(pred-reloaded_pred).max()))
        np.testing.assert_allclose(pred,reloaded_pred,rtol=0,atol=2e-7)
        del m,reloaded
    difference=float(abs(predictions[0]-predictions[1]).max())
    np.testing.assert_allclose(predictions[0],predictions[1],rtol=0,atol=2e-7)
    print('PASS real training-data sample parity and save/load; max difference',difference,flush=True)
    gc.collect();ctypes.CDLL(None).malloc_trim(0)
    started=time.monotonic()
    m=NumericDenseXGBoostModel(path=str(path),name='full_B_two_rounds',problem_type='binary',eval_metric='roc_auc',hyperparameters={'n_estimators':2})
    m.fit(X=x,y=y,X_val=xv,y_val=yv,num_cpus=20,num_gpus=0,verbosity=3)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    assert peak<78
    write_json(OUT/'XGBOOST_DENSE_RESOURCE_PROFILE_20260913.json',dict(status='PASS_FULL_B_TWO_ROUNDS_ONLY',utc=now(),seed=seed,
        train_rows=len(x),inner_validation_rows=len(xv),features=x.shape[1],peak_RSS_GiB=peak,
        seconds=time.monotonic()-started,probe_rounds=2,formal_learning_budget_unchanged=10000,
        real_sample_max_probability_difference=difference,native_save_load_parity=True,
        save_load_max_probability_difference=max(reload_differences),probability_tolerance=2e-7,
        adapter_sha256=digest(ROOT/'scripts/dtiam_numeric_xgboost_20260913.py'),
        formal_models_modified=False,common_validation_inference=False,test_inference=False))
    print('PASS full B two-round probe, peak GiB',peak,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);a=p.parse_args();main(a.seed)

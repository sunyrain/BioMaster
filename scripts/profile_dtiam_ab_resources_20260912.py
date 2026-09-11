#!/usr/bin/env python3
"""Measure full-B native FastAI preprocessing and one batch, without fitting weights."""
import gc
import resource
import time
import numpy as np
import pandas as pd
from autogluon.tabular.models.fastainn.tabular_nn_fastai import NNFastAiTabularModel
from dtiam_ab_common_20260912 import OUT,SOURCE,banks,table,write_json,now


def main():
    started=time.monotonic()
    frame=pd.read_parquet(SOURCE/'all_inactive_TRAIN.parquet')
    x=table(frame,banks());y=pd.Series(frame.binary_label.to_numpy(int))
    model=NNFastAiTabularModel(path=str(OUT/'RESOURCE_PROFILE'),name='FastAI_preprocess_only',
        problem_type='binary',eval_metric='roc_auc',hyperparameters={})
    model.initialize(X=x,y=y)
    # Reproduce fit_extra: only native train/holdout tables stay resident.
    # Keeping the unsplit input as well overestimates the real isolated process.
    n=max(2500,int(len(frame)*.01));valid=x.iloc[:n].copy();train=x.iloc[n:].copy()
    features=x.shape[1]
    del x;gc.collect()
    vlabel=y.iloc[:n];tlabel=y.iloc[n:]
    data=model._preprocess_train(train,tlabel,valid,vlabel)
    dls=data.dataloaders(bs=512,num_workers=0);batch=dls.one_batch()
    rows=len(data.items)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    observation=dict(rows=rows,expected_rows=len(frame),features=features,peak_RSS_GiB=peak,
        seconds=time.monotonic()-started,batch_shapes=[list(v.shape) for v in batch],utc=now())
    write_json(OUT/'RESOURCE_PROFILE_OBSERVATION.json',observation)
    if rows!=len(frame) or peak>72:
        raise RuntimeError(f'Full-B native preprocessing exceeded the profiled budget: {peak} GiB')
    write_json(OUT/'RESOURCE_PROFILE.json',dict(status='PASS_FULL_B_PREPROCESS_ONLY',utc=now(),
        rows=rows,features=features,peak_RSS_GiB=peak,seconds=time.monotonic()-started,
        batch_shapes=[list(v.shape) for v in batch],supervised_weight_updates=0,
        interpretation='Native full-B preprocessing and loader; no prediction performance claim. Isolate FASTAI, allow its conservative 10x precheck ratio=2, stop child if actual RSS exceeds 78 GiB.'))


if __name__=='__main__':main()

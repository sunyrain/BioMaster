#!/usr/bin/env python3
"""Train one native DTIAM suite, freeze validation choices, or perform gated TEST inference."""
import argparse
import gc
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
from autogluon.core.callbacks import AbstractCallback
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import expit

from dtiam_ab_common_20260912 import (ROOT,OUT,SOURCE,FEATURES,ARMS,SEEDS,now,digest,write_json,
    table,banks,probability_score,calibrate,query_summary,basic)

EXPECTED={'LightGBMXT','LightGBM','LightGBMLarge','CatBoost','XGBoost',
          'RandomForestGini','RandomForestEntr','ExtraTreesGini','ExtraTreesEntr',
          'NeuralNetFastAI','NeuralNetTorch','WeightedEnsemble_L2'}


class Progress(AbstractCallback):
    def __init__(self,path):
        super().__init__();self.path=path;self.started=time.monotonic();self.finished=[]

    def _before_model_fit(self,trainer,model,**kwargs):
        self.current=model.name;self.model_started=time.monotonic()
        write_json(self.path/'FIT_STATUS.json',dict(stage='FITTING',model=model.name,completed=self.finished,
            updated_utc=now(),elapsed_seconds=time.monotonic()-self.started,
            memory_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
        return False,False

    def _after_model_fit(self,trainer,model_names,**kwargs):
        record=dict(model=self.current,success_names=model_names,seconds=time.monotonic()-self.model_started)
        self.finished.append(record)
        write_json(self.path/'FIT_STATUS.json',dict(stage='MODEL_FINISHED',completed=self.finished,
            updated_utc=now(),elapsed_seconds=time.monotonic()-self.started))
        return False


def prediction(predictor,frame,bank,models):
    # Panels overlap; infer once per physical pair, then restore exact frozen row order.
    unique=frame.drop_duplicates('pair_id').reset_index(drop=True)
    positions=pd.Series(np.arange(len(unique)),index=unique.pair_id)
    restore=frame.pair_id.map(positions).to_numpy(int)
    result={name:np.empty(len(unique),np.float32) for name in models}
    for start in range(0,len(unique),16384):
        f=unique.iloc[start:start+16384]
        values=predictor.predict_proba_multi(table(f,bank),models=models,as_pandas=False,as_multiclass=False)
        for name,p in values.items():
            p=np.asarray(p).reshape(-1)
            assert len(p)==len(f) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            result[name][start:start+len(f)]=p
    return {name:p[restore] for name,p in result.items()}


def fit(arm,seed,smoke=False,core_only=False):
    run=OUT/('SMOKE_NATIVE_API' if smoke else f'{arm}__seed_{seed}')
    if (run/'SELECTION.json').exists():
        return
    run.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(20)
    bank=tuple(np.load(FEATURES/n,mmap_mode='r') for n in ['BERMOL.npy','ESM2.npy']) if smoke else banks()
    frame=pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet')
    params=get_hyperparameter_config('default')
    if smoke:
        dd=np.load(FEATURES/'BERMOL_DONE.npy');td=np.load(FEATURES/'ESM2_DONE.npy')
        use=dd[frame.drug_feature_index.to_numpy(int)] & td[frame.target_feature_index.to_numpy(int)]
        frame=frame[use].sample(n=600,random_state=seed).reset_index(drop=True)
        params['NN_TORCH']={'num_epochs':2}
        params['FASTAI']={'epochs':2}
        params['CAT']={'iterations':4}
        params['XGB']={'n_estimators':4}
        for p in params['GBM']:p.update(num_boost_round=4)
        for family in ['RF','XT']:
            for p in params[family]:p.update(n_estimators=4,max_depth=4)
    else:
        assert params==json.loads((OUT/'AUTOGLUON_HYPERPARAMETERS.json').read_text())
        params.pop('FASTAI')
    if not (run/'FIT_RETURNED.json').exists():
        if (run/'predictor').exists():
            raise RuntimeError(f'Incomplete native fit preserved at {run}; inspect before resuming')
        if shutil.disk_usage(OUT).free < (20 if arm=='kdki_inactive' else 35)*2**30:
            raise RuntimeError('Insufficient disk headroom for native full-data fit')
        x=table(frame,bank,labels=True)
        predictor=TabularPredictor(label='y',problem_type='binary',eval_metric='roc_auc',
            path=str(run/'predictor'),verbosity=3,learner_kwargs={'random_state':seed})
        started=time.monotonic()
        predictor.fit(x,presets=None,hyperparameters=params,excluded_model_types=[],time_limit=None,
            num_cpus=20,num_gpus=0,memory_limit=68,fit_strategy='sequential',
            calibrate_decision_threshold=False,callbacks=[Progress(run)],fit_weighted_ensemble=smoke)
        del x;gc.collect()
        duration=time.monotonic()-started
        info=predictor.info();write_json(run/'AUTOGLUON_INFO.json',info)
        predictor.leaderboard(silent=True).to_csv(run/'AUTOGLUON_LEADERBOARD.csv',index=False)
        predictor.model_failures().to_csv(run/'MODEL_FAILURES.csv',index=False)
        # Index maps retain the outer member identities used by the native inner split.
        _,ytrain=predictor.load_data_internal('train',return_X=False,return_y=True)
        _,yval=predictor.load_data_internal('val',return_X=False,return_y=True)
        assert set(ytrain.index).isdisjoint(yval.index)
        assert set(ytrain.index)|set(yval.index)==set(range(len(frame)))
        internal=frame[['pair_id','binary_label']].copy();internal['native_role']='base_train'
        internal.loc[yval.index,'native_role']='internal_holdout'
        internal.to_parquet(run/'INTERNAL_MEMBER_ROLES.parquet',index=False)
        write_json(run/'FIT_RETURNED.json',dict(fit_seconds=duration,training_member_pool=len(frame),
            base_train_rows=len(ytrain),internal_holdout_rows=len(yval),native_best=predictor.model_best,
            model_names=predictor.model_names(),peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            finished_utc=now(),smoke_only=smoke))
    else:
        predictor=TabularPredictor.load(str(run/'predictor'))
    if core_only:
        missing=(EXPECTED-{'NeuralNetFastAI','WeightedEnsemble_L2'})-set(predictor.model_names())
        assert not missing and predictor.model_failures().empty, 'Core learner suite incomplete'
        return
    missing=EXPECTED-set(predictor.model_names())
    if missing or not predictor.model_failures().empty:
        write_json(run/'INCOMPLETE.json',dict(missing_models=sorted(missing),failures=predictor.model_failures().to_dict('records')))
        raise RuntimeError(f'Native suite incomplete: {sorted(missing)}; see MODEL_FAILURES.csv')
    if smoke:
        pred=prediction(predictor,frame.iloc[:20],bank,sorted(EXPECTED))
        assert set(pred)==EXPECTED
        write_json(run/'SMOKE_RESULT.json',dict(status='PASS_API_ONLY',rows=len(frame),model_count=len(pred),
            note='Only training rows; deliberately tiny learner budgets. Not a performance result or formal fit.'))
        return
    validation=pd.read_parquet(SOURCE/'COMMON_VALIDATION.parquet')
    kd=validation[validation.panel.eq('AFFINITY_KD_KI')].reset_index(drop=True)
    probabilities=prediction(predictor,kd,bank,predictor.model_names());rows=[]
    saved=kd[['pair_id','panel','binary_label','target_id','molecule_id']].copy()
    for model,p in probabilities.items():
        q=query_summary(kd,p)
        rows.append(dict(model=model,selection_score=q['selection_score'],
            target_macro_ap=q['target']['macro_ap'],drug_macro_ap=q['drug']['macro_ap'],
            ap=float(average_precision_score(kd.binary_label,p)),auroc=float(roc_auc_score(kd.binary_label,p))))
        saved[model]=p
    saved.to_parquet(run/'VALIDATION_ALL_LEARNERS.parquet',index=False)
    scores=pd.DataFrame(rows).sort_values(['selection_score','ap','model'],ascending=[False,False,True])
    scores.to_csv(run/'VALIDATION_LEADERBOARD.csv',index=False)
    choices={'native':predictor.model_best,'query':scores.iloc[0].model}
    chosen_prediction=prediction(predictor,validation,bank,sorted(set(choices.values())))
    vout=validation[['pair_id','panel','binary_label','molecule_id','target_id','split_group']].copy()
    calibrations={}
    for view,model in choices.items():
        score=probability_score(chosen_prediction[model]);calibrations[view]=calibrate(validation,score)
        vout[view+'_native_probability']=chosen_prediction[model]
        vout[view+'_score']=chosen_prediction[model]
        vout[view+'_prob']=expit(calibrations[view]['slope']*score+calibrations[view]['intercept'])
    vout.to_parquet(run/'VALIDATION_PREDICTIONS.parquet',index=False)
    write_json(run/'CALIBRATION.json',calibrations)
    # This marker is written last and is the prerequisite to entering the TEST gate.
    write_json(run/'SELECTION.json',dict(status='COMPLETE_FIT_AND_VALIDATION_FROZEN',arm=arm,seed=seed,
        frozen_utc=now(),choices=choices,validation_scores=scores.to_dict('records'),
        calibration_sha256=digest(run/'CALIBRATION.json'),fit_sha256=digest(run/'FIT_RETURNED.json')))
    # Release only this run's regenerated feature-table cache. Preserve every trained model,
    # its inner split, parameters, validation predictions and all provenance.
    predictor.save_space(remove_data=True,remove_fit_stack=True)
    write_json(run/'CACHE_RELEASE.json',dict(utc=now(),action='AutoGluon save_space: new training feature-table cache only',
        all_trained_predictors_retained=True,inner_member_roles_retained=True,
        future_refit='Rebuild exact feature table from frozen sources; cached fit_extra input removed'))


def fastai(arm,seed):
    """Fresh process releases core-fit tables before full-data native FASTAI."""
    run=OUT/f'{arm}__seed_{seed}'
    if (run/'FASTAI_COMPLETE.json').exists():return
    profile=json.loads((OUT/'RESOURCE_PROFILE.json').read_text())
    assert profile['status']=='PASS_FULL_B_PREPROCESS_ONLY' and profile['peak_RSS_GiB']<=65
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(20)
    predictor=TabularPredictor.load(str(run/'predictor'))
    started=time.monotonic()
    write_json(run/'FIT_STATUS.json',dict(stage='FITTING_ISOLATED_FASTAI',model='NeuralNetFastAI',updated_utc=now()))
    # This modifies only a conservative allocation precheck; all learning parameters
    # and all training members retain their native definitions.
    predictor.fit_extra(hyperparameters={'FASTAI':{'ag.max_memory_usage_ratio':2}},
        time_limit=None,num_cpus=20,num_gpus=0,memory_limit=68,fit_strategy='sequential',
        fit_weighted_ensemble=False)
    assert 'NeuralNetFastAI' in predictor.model_names() and predictor.model_failures().empty
    predictor.fit_weighted_ensemble(base_models=predictor.model_names(),name_suffix='',
        time_limit=None,num_cpus=20,num_gpus=0)
    predictor.save()
    assert EXPECTED<=set(predictor.model_names()) and predictor.model_failures().empty
    info=predictor.info();write_json(run/'AUTOGLUON_INFO.json',info)
    predictor.leaderboard(silent=True).to_csv(run/'AUTOGLUON_LEADERBOARD.csv',index=False)
    predictor.model_failures().to_csv(run/'MODEL_FAILURES.csv',index=False)
    fit=json.loads((run/'FIT_RETURNED.json').read_text())
    duration=time.monotonic()-started
    fit.update(core_fit_seconds=fit['fit_seconds'],fastai_ensemble_seconds=duration,
        fit_seconds=fit['fit_seconds']+duration,model_names=predictor.model_names(),native_best=predictor.model_best,
        finished_utc=now(),peak_RSS_GiB=max(fit['peak_RSS_GiB'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
    write_json(run/'FIT_RETURNED.json',fit)
    write_json(run/'FASTAI_COMPLETE.json',dict(status='COMPLETE',utc=now(),seconds=duration,
        model_names=predictor.model_names(),resource_profile_sha256=digest(OUT/'RESOURCE_PROFILE.json')))


def infer(arm,seed):
    for a in ARMS:
        for s in SEEDS:
            check=json.loads((OUT/f'{a}__seed_{s}'/'SELECTION.json').read_text())
            assert check['status']=='COMPLETE_FIT_AND_VALIDATION_FROZEN'
    gate=json.loads((OUT/'TEST_GATE.json').read_text())
    for name,sha in gate['selection_hashes'].items():
        assert digest(ROOT/name)==sha
    run=OUT/f'{arm}__seed_{seed}'
    if (run/'TEST_COMPLETE.json').exists():return
    selection=json.loads((run/'SELECTION.json').read_text());cal=json.loads((run/'CALIBRATION.json').read_text())
    predictor=TabularPredictor.load(str(run/'predictor'))
    frame=pd.read_parquet(SOURCE/'COMMON_TEST.parquet')
    values=prediction(predictor,frame,banks(),sorted(set(selection['choices'].values())))
    result=frame[['pair_id','panel','binary_label','molecule_id','target_id','split_group','document_disjoint']].copy()
    for view,model in selection['choices'].items():
        score=probability_score(values[model]);result[view+'_native_probability']=values[model]
        result[view+'_score']=values[model];result[view+'_prob']=expit(cal[view]['slope']*score+cal[view]['intercept'])
    result.to_parquet(run/'TEST_PREDICTIONS.parquet',index=False)
    write_json(run/'TEST_COMPLETE.json',dict(status='COMPLETE',utc=now(),test_rows=len(result),
        selection_sha256=digest(run/'SELECTION.json'),predictions_sha256=digest(run/'TEST_PREDICTIONS.parquet')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,default=ARMS[0])
    p.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0]);p.add_argument('--smoke',action='store_true')
    p.add_argument('--core-only',action='store_true');p.add_argument('--fastai',action='store_true')
    p.add_argument('--test',action='store_true');a=p.parse_args()
    try:
        if a.test:infer(a.arm,a.seed)
        elif a.fastai:fastai(a.arm,a.seed)
        else:fit(a.arm,a.seed,a.smoke,a.core_only)
    except Exception as error:
        run=OUT/('SMOKE_NATIVE_API' if a.smoke else f'{a.arm}__seed_{a.seed}')
        write_json(run/'ERROR.json',dict(utc=now(),error=repr(error),traceback=traceback.format_exc()))
        raise

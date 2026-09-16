"""Comparable retrospective evaluation and live SPR384 model-review snapshot."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DIRECTORY='outputs/frontier_dti_20260916'
NEW_MODELS=('nesso','probematch','dtbind')
BASE_MODELS=('biomaster','drugclip','dtiam','conplex')
ALL_MODELS=BASE_MODELS+NEW_MODELS
NAMES={'biomaster':'ReTargetMap（网站版）','drugclip':'DrugCLIP','dtiam':'DTIAM（网站历史版）','conplex':'ConPLex','nesso':'Nesso-1','probematch':'ProbeMatchDTI','dtbind':'DTBind · 结合预测','nesso_pic50':'Nesso-1 · pIC50辅助'}

def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,np.generic):return finite_json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def metrics(frame,model,scope):
    valid=frame.loc[frame[model].notna() & frame.label.notna()]
    n=len(valid);positive=int(valid.label.sum());negative=n-positive
    two=positive>0 and negative>0
    return dict(scope=scope,model=model,name=NAMES[model],pairs=n,positive=positive,negative=negative,
        prevalence=positive/n if n else None,ap=float(average_precision_score(valid.label,valid[model])) if two else None,
        auroc=float(roc_auc_score(valid.label,valid[model])) if two else None)

def common_ranks(frame,models=ALL_MODELS):
    """Every comparison uses exactly the same rows; ties use average ranks."""
    valid=frame[list(models)].notna().all(axis=1)
    ranks=pd.DataFrame(np.nan,index=frame.index,columns=models)
    ranks.loc[valid,list(models)]=frame.loc[valid,list(models)].rank(ascending=False,method='average')
    n=int(valid.sum())
    percent=(ranks-1)/(n-1) if n>1 else ranks*float('nan')
    return ranks,percent,n

def load_snapshot(root,kind='',identifier=''):
    if kind not in ('','drug','target'):raise ValueError('kind must be drug or target')
    path=Path(root)/DIRECTORY/'WEBSITE_SNAPSHOT.json'
    if not path.exists():return dict(available=False,items=[],models=[],metrics=[],message='新模型复核结果尚未生成。')
    payload=json.loads(path.read_text())
    if kind and identifier:
        payload['items']=[x for x in payload['items'] if x[kind+'_id']==identifier]
    payload['visible_count']=len(payload['items'])
    return payload

#!/usr/bin/env python3
"""Check whether weak geometry sensitivity persists without bf16 autocast."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.anchored_interaction import AnchoredFeatureBank
from biomaster.model_registry import build_model,infer_family
from biomaster.best_model_training import old_labels
from biomaster.ranking_audit import risk_set_ranking
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json
BASE=ROOT/'outputs/biomaster_best_model_20260906'
OUT=ROOT/'outputs/biomaster_local_failure_audit_20260906/fp32'


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=dict(precision='fp32, TF32 disabled',cutoffs=[2018,2020],seeds=[20260921,20260922,20260923],
        conditions=['full','geometry_off','coordinates_shuffled'],variant='geometry',
        role='post-selection development-only precision sensitivity; no training/reselection or test labels',
        producer=file_identity(Path(__file__)),original_diagnostic=file_identity(OUT.parent/'PROTOCOL.json'))
    write_json(OUT/'PROTOCOL.json',protocol)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz');nd,nt=len(old),len(targets)
    bank=AnchoredFeatureBank(OUTPUT,BASE/'data/supplemental_features',SOURCE,'drugclip_morgan',local=True)
    ca=bank.ca.clone();mask=bank.geom_mask.clone();shuffle=ca.clone();rng=np.random.default_rng(20260906)
    for ti in range(nt):
        ids=torch.where(mask[ti])[0];perm=torch.tensor(rng.permutation(len(ids)),device=ids.device)
        shuffle[ti,ids]=ca[ti,ids[perm]]
    rows=[];effects=[]
    for cutoff in protocol['cutoffs']:
        val=pd.read_csv(BASE/'data'/f'roll_{cutoff}/VALIDATION.csv.gz');y=old_labels(val,old,nt)
        needed=np.zeros_like(y);needed[np.flatnonzero(y.any(1)),:]=True;needed[:,np.flatnonzero(y.any(0))]=True
        di,ti=np.where(needed);risk=np.load(BASE/'data'/f'roll_{cutoff}/RISK.npy')
        for seed in protocol['seeds']:
            path=BASE/'anchored'/f'cutoff_{cutoff}/geometry'/f'seed_{seed}/BEST.pt'
            state=torch.load(path,map_location='cpu',weights_only=False)
            model=build_model(infer_family(state),state['config']).cuda().eval();model.load_state_dict(state['model'])
            scores_by={}
            for condition in protocol['conditions']:
                bank.ca=shuffle if condition=='coordinates_shuffled' else ca
                bank.geom_mask=torch.zeros_like(mask) if condition=='geometry_off' else mask
                scores=np.empty((len(di),2),np.float32)
                with torch.inference_mode():
                    for lo in range(0,len(di),128):
                        sl=slice(lo,min(lo+128,len(di)))
                        scores[sl]=model(bank.batch(old.drug_feature_index.to_numpy()[di[sl]],ti[sl],'geometry')).cpu().numpy()
                dense=np.zeros((nd,nt,2),np.float32);dense[di,ti]=scores;scores_by[condition]=scores
                row=dict(cutoff=cutoff,seed=seed,condition=condition);metrics={}
                for head,direction in enumerate(['d2t','t2d']):
                    a,b,c=y,dense[:,:,head],risk;qids,cids=np.arange(nd),np.arange(nt)
                    if head:a,b,c,qids,cids=a.T,b.T,c.T,cids,qids
                    m,_,_=risk_set_ranking(a,b,c,qids,cids)
                    row[direction+'_ap']=m['macro_ap'];row[direction+'_r20']=m['macro_recall_20'];metrics[direction]=m
                row['composite']=.35*(row['d2t_ap']+row['t2d_ap'])+.15*(row['d2t_r20']+row['t2d_r20']);rows.append(row)
                write_json(OUT/f'{cutoff}_{seed}_{condition}.json',dict(**row,metrics=metrics,checkpoint=file_identity(path)))
                print(json.dumps(row),flush=True)
            for condition in protocol['conditions'][1:]:
                diff=np.abs(scores_by[condition]-scores_by['full']);effects.append(dict(cutoff=cutoff,seed=seed,condition=condition,
                    mean_absolute_logit_change=float(diff.mean()),max_absolute_logit_change=float(diff.max())))
            del model,state;torch.cuda.empty_cache()
    table=pd.DataFrame(rows);table.to_csv(OUT/'RUNS.csv',index=False)
    table.groupby('condition')[['composite','d2t_ap','t2d_ap','d2t_r20','t2d_r20']].mean().to_csv(OUT/'SUMMARY.csv')
    pd.DataFrame(effects).to_csv(OUT/'SCORE_SENSITIVITY.csv',index=False)
    write_json(OUT/'STATUS.json',dict(status='COMPLETE',runs=len(rows),protocol=file_identity(OUT/'PROTOCOL.json')))


if __name__=='__main__':main()

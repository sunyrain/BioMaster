#!/usr/bin/env python3
"""Train unified ablations; this entrypoint NEVER reads post-2022 labels."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from biomaster.unified_interaction import UnifiedInteraction,UnifiedConfig,capacity_matched_config
from biomaster.unified_features import UnifiedFeatureBank
from biomaster.ranking_audit import risk_set_ranking
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json,SOURCE,OUTPUT

OUT=OUTPUT.parent
CONFIG=ROOT/'configs/biomaster_unified_interaction_20260906.json'


def identity():
    names=['biomaster/unified_interaction.py','biomaster/unified_features.py',
           'scripts/train_biomaster_unified_interaction.py','scripts/prepare_biomaster_unified_interaction.py',
           'biomaster/ranking_audit.py','biomaster/odti_local_features_v3.py','biomaster/odti_pockets_v3.py']
    return dict(config=file_identity(CONFIG),sources=[file_identity(ROOT/name) for name in names],
                atoms=file_identity(OUTPUT/'ATOM_MANIFEST.json'),targets=file_identity(OUTPUT/'TARGET_MANIFEST.json'),
                temporal_data=file_identity(SOURCE/'DATA_MANIFEST.json'))


def log(event):
    event={'utc':datetime.now(timezone.utc).isoformat(),**event}
    write_json(OUT/'TRAIN_STATUS.json',event)
    print(json.dumps(event),flush=True)


def old_labels(frame,old,targets):
    known=np.zeros((len(old),targets),bool)
    mapping=old.set_index('drug_feature_index').old_drug_index
    part=frame.loc[frame.binary_label.eq(1)&frame.drug_feature_index.isin(mapping.index)]
    known[part.drug_feature_index.map(mapping).to_numpy(int),part.target_feature_index.to_numpy(int)]=True
    return known


class TemporalStage:
    def __init__(self,name):
        if name not in ['development','final']:raise ValueError('invalid temporal stage')
        self.name=name;self.cutoff=2020 if name=='development' else 2022
        self.train=pd.read_csv(SOURCE/('DEVELOPMENT_TRAIN.csv.gz' if name=='development' else 'FINAL_TRAIN.csv.gz'))
        if self.train.max_document_year.max()>self.cutoff:raise ValueError('future label in training')
        if not self.train.binary_label.isin([0,1]).all():raise ValueError('BCE needs observed binary labels')
        if self.train.duplicated(['drug_feature_index','target_feature_index']).any():raise ValueError('duplicate supervised pairs')
        self.d=self.train.drug_feature_index.to_numpy(int);self.t=self.train.target_feature_index.to_numpy(int)
        self.y=self.train.binary_label.to_numpy(np.float32)
        self.old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
        self.nt=len(pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz'))
        self.known=old_labels(self.train,self.old,self.nt)
        self.risk=np.load(SOURCE/'RISK_SETS.npz')[name]
        self.val=None
        if name=='development':
            self.val=pd.read_csv(SOURCE/'DEVELOPMENT_VALIDATION_NEW.csv.gz')
            assert self.val.min_document_year.min()>=2021 and self.val.max_document_year.max()<=2022
            tr=set(zip(self.d,self.t));va=set(zip(self.val.drug_feature_index,self.val.target_feature_index))
            if tr & va:raise ValueError('training/first-measurement validation relation overlap')
            self.val_known=old_labels(self.val,self.old,self.nt)
            if (self.val_known&~self.risk).any():raise ValueError('validation positive excluded by risk set')
        self.positive_queries=[np.flatnonzero(self.known.any(1)),np.flatnonzero(self.known.any(0))]

    def retrieval(self,rng,p):
        ds,ts,ys,groups=[],[],[],[]
        start=0
        for direction in [0,1]:
            matrix=self.known if direction==0 else self.known.T
            queries=rng.choice(self.positive_queries[direction],p['retrieval_queries_per_direction'],replace=False)
            for query in queries:
                positive=np.flatnonzero(matrix[query]);other=np.flatnonzero(~matrix[query])
                # Retain ALL known positives; unknowns are only denominator entries.
                count=min(len(other),max(1,p['retrieval_candidates']-len(positive)))
                candidates=np.r_[positive,rng.choice(other,count,replace=False)]
                rng.shuffle(candidates)
                labels=matrix[query,candidates]
                if direction==0:
                    d=np.full(len(candidates),self.old.drug_feature_index.iloc[query]);t=candidates
                else:
                    d=self.old.drug_feature_index.to_numpy()[candidates];t=np.full(len(candidates),query)
                ds.extend(d);ts.extend(t);ys.extend(labels);groups.append((start,start+len(candidates),direction))
                start+=len(candidates)
        return np.array(ds,int),np.array(ts,int),np.array(ys,bool),groups


@torch.inference_mode()
def predict(model,bank,d,t,batch_size=256,geometry_shuffle=False):
    model.eval();out=np.empty((len(d),2),np.float32)
    for start in range(0,len(d),batch_size):
        end=min(start+batch_size,len(d));batch=bank.batch(d[start:end],t[start:end],model.cfg.variant)
        if geometry_shuffle:
            # Permute coordinates among geometry-bearing residues only, retaining
            # quality/masks/sequence tokens. No change for targets with <2 CAs.
            ca=batch['ca'].clone()
            for row in range(len(ca)):
                valid=torch.where(batch['geometry_mask'][row])[0]
                ca[row,valid]=ca[row,valid.flip(0)]
            batch['ca']=ca
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model(batch)
        out[start:end]=value.float().cpu().numpy()
    if not np.isfinite(out).all():raise FloatingPointError('nonfinite predicted score')
    return out


def validation(model,bank,stage,p):
    result={};query_frames=[]
    for head,direction in enumerate(['d2t','t2d']):
        known=stage.val_known if head==0 else stage.val_known.T
        risk=stage.risk if head==0 else stage.risk.T
        ids=np.flatnonzero(known.any(1));n=known.shape[1]
        if head==0:
            d=np.repeat(stage.old.drug_feature_index.to_numpy()[ids],n);t=np.tile(np.arange(n),len(ids))
        else:
            d=np.tile(stage.old.drug_feature_index.to_numpy(),len(ids));t=np.repeat(ids,n)
        scores=predict(model,bank,d,t,p['evaluation_batch_size'])[:,head].reshape(len(ids),n)
        metrics,queries,_=risk_set_ranking(known[ids],scores,risk[ids],ids,np.arange(n))
        result[direction]={'ap':metrics['macro_ap'],'r20':metrics['macro_recall_20'],
                           'queries':metrics['positive_queries'],'positive_pairs':metrics['positive_pairs']}
        queries['direction']=direction;query_frames.append(queries)
    val=stage.val
    observed=predict(model,bank,val.drug_feature_index.to_numpy(),val.target_feature_index.to_numpy(),p['evaluation_batch_size'])
    for head,key in enumerate(['drug_feature_index','target_feature_index']):
        y=val.binary_label.to_numpy();values=[]
        for ids in val.groupby(key,sort=False).indices.values():
            if np.unique(y[ids]).size==2:values.append(average_precision_score(y[ids],observed[ids,head]))
        result[['d2t','t2d'][head]]['measured_ap']=float(np.mean(values))
    result['selection']=sum(.35*result[d]['ap']+.15*result[d]['r20'] for d in ['d2t','t2d'])
    return result,pd.concat(query_frames,ignore_index=True)


def train(stage_name,variant,seed,smoke_steps=0):
    p=json.loads(CONFIG.read_text())
    if variant not in p['variants'] or seed not in p['seeds']+p['confirmation_seeds']:
        raise ValueError('run not included in frozen protocol')
    ident=identity();stage=TemporalStage(stage_name)
    folder=OUT/('smoke' if smoke_steps else stage_name)/variant/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():
        result=json.loads((folder/'RESULT.json').read_text())
        if result['identity']!=ident:raise ValueError('completed run identity mismatch')
        return result
    fixed_epoch=None
    if stage_name=='final':
        development=json.loads((OUT/'development'/variant/f'seed_{seed}'/'RESULT.json').read_text())
        if development['identity']!=ident:raise ValueError('development identity mismatch')
        fixed_epoch=development['selected_epoch']
    cfg=UnifiedConfig(variant=variant,**{k:p[k] for k in ['width','pair_width','blocks','dropout']})
    if variant=='capacity':cfg=capacity_matched_config(UnifiedConfig(**{k:p[k] for k in ['width','pair_width','blocks','dropout']}))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(seed);np.random.seed(seed)
    model=UnifiedInteraction(cfg).cuda()
    bank=UnifiedFeatureBank(OUTPUT,local=model.is_local)
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    start_epoch=1;best=-math.inf;best_epoch=0;history=[]
    last=folder/'LAST.pt'
    if last.exists():
        state=torch.load(last,map_location='cuda',weights_only=False)
        if state['identity']!=ident:raise ValueError('partial run source/protocol changed; use a new output directory')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch'];history=state['history']
        del state
    base_event=dict(stage=stage_name,variant=variant,seed=seed)
    log({**base_event,'event':'training_started','train_rows':len(stage.train),'old_positive_pairs':int(stage.known.sum()),
         'trainable_parameters':sum(x.numel() for x in model.parameters()),'configuration':asdict(cfg)})
    started=time.monotonic();epochs=1 if smoke_steps else (fixed_epoch or p['max_epochs'])
    y_positive=float(stage.y.mean())
    for epoch in range(start_epoch,epochs+1):
        rng=np.random.default_rng(seed+epoch*104729);retrieval_rng=np.random.default_rng(seed+epoch*104729+11)
        torch.manual_seed(seed+epoch*104729)
        order=rng.permutation(len(stage.train));exposure=hashlib.sha256();losses=[];retrieval_losses=[]
        lr=p['learning_rate']*(.2+.8*(1+math.cos(math.pi*(epoch-1)/p['max_epochs']))/2)
        for group in optimizer.param_groups:group['lr']=lr
        epoch_start=time.monotonic();model.train()
        for step,start in enumerate(range(0,len(order),p['batch_size'])):
            rows=order[start:start+p['batch_size']];exposure.update(rows.astype('<i8').tobytes())
            optimizer.zero_grad(set_to_none=True)
            y=torch.tensor(stage.y[rows],device='cuda')
            weight=torch.where(y.bool(),.5/y_positive,.5/(1-y_positive))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=model(bank.batch(stage.d[rows],stage.t[rows],variant))
                bce=F.binary_cross_entropy_with_logits(scores.float(),y[:,None].expand(-1,2),reduction='none')
                loss=(bce*weight[:,None]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite observed BCE')
            loss.backward();losses.append(float(loss.detach()))
            if step%p['retrieval_every_steps']==0:
                d,t,known,groups=stage.retrieval(retrieval_rng,p)
                exposure.update(np.stack([d,t,known],1).astype('<i8').tobytes())
                terms=[]
                # Microbatch each whole query; avoid holding observed activations.
                for lo,hi,head in groups:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        s=model(bank.batch(d[lo:hi],t[lo:hi],variant))[:,head].float()
                        positive=torch.tensor(known[lo:hi],device='cuda')
                        term=-F.log_softmax(s/p['retrieval_temperature'],dim=0)[positive].mean()
                        weighted=term*p['retrieval_weight']/len(groups)
                    if not torch.isfinite(weighted):raise FloatingPointError('nonfinite retrieval loss')
                    weighted.backward();terms.append(float(term.detach()))
                retrieval_losses.append(float(np.mean(terms)))
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%200==0:
                log({**base_event,'event':'training_step','epoch':epoch,'step':step+1,
                     'steps':math.ceil(len(order)/p['batch_size']),'mean_bce':float(np.mean(losses)),
                     'seconds':round(time.monotonic()-epoch_start,1)})
            if smoke_steps and step+1>=smoke_steps:break
        metrics=None
        if not smoke_steps and fixed_epoch is None:
            metrics,queries=validation(model,bank,stage,p);value=metrics['selection']
        else:value=float(epoch)
        if value>best:
            best,best_epoch=value,epoch
            torch.save(dict(model=model.state_dict(),config=asdict(cfg),seed=seed,cutoff=stage.cutoff,
                            epoch=epoch,identity=ident,validation=metrics),folder/'BEST.pt')
            if metrics is not None:queries.to_csv(folder/'VALIDATION_QUERY_METRICS.csv',index=False)
        row=dict(epoch=epoch,mean_bce=float(np.mean(losses)),mean_retrieval=float(np.mean(retrieval_losses)),
                 validation=metrics,selected_epoch=best_epoch,coverage_rows=min(len(order),len(losses)*p['batch_size']),
                 exposure_sha256=exposure.hexdigest(),learning_rate=lr,gradient_norm=float(grad),
                 epoch_seconds=time.monotonic()-epoch_start)
        history.append(row);write_json(folder/'HISTORY.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),identity=ident,
                        epoch=epoch,best=best,best_epoch=best_epoch,history=history),folder/'LAST.pt')
        log({**base_event,'event':'epoch_completed',**row})
        if not smoke_steps and fixed_epoch is None and epoch>=p['min_epochs'] and epoch-best_epoch>=p['patience']:break
    result=dict(identity=ident,stage=stage_name,variant=variant,seed=seed,selected_epoch=best_epoch,
                trained_epochs=len(history),train_rows=len(stage.train),train_max_year=stage.cutoff,
                old_positive_pairs=int(stage.known.sum()),trainable_parameters=sum(x.numel() for x in model.parameters()),
                seconds=time.monotonic()-started,validation=history[best_epoch-1]['validation'],
                max_epoch_reached=len(history)==p['max_epochs'],test_labels_used=False,smoke_only=bool(smoke_steps),
                checkpoint=file_identity(folder/'BEST.pt'),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write_json(folder/'RESULT.json',result)
    log({**base_event,'event':'run_completed','selected_epoch':best_epoch,'validation':result['validation']})
    del model,bank,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['development','final'],default='development')
    parser.add_argument('--variant',choices=['global','capacity','sequence','site','geometry'],required=True)
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args();train(args.stage,args.variant,args.seed,args.smoke_steps)


if __name__=='__main__':main()

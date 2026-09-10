#!/usr/bin/env python3
"""Controlled optimization and warm refinement; development labels only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.best_model_training import SupplementedFeatureBank,RollingStage,ObservedQueries,measured_pair_loss,WeightAverage
from biomaster.refined_interaction import RefinedInteraction,RefinedConfig,matched_refinement_config
from biomaster.unified_interaction import UnifiedInteraction,UnifiedConfig
from biomaster.odti_pockets_v3 import file_identity
from train_biomaster_unified_interaction import validation,identity as original_identity
from prepare_biomaster_unified_interaction import write_json,SOURCE,OUTPUT

OUT=ROOT/'outputs/biomaster_best_model_20260906'
DATA=OUT/'data'
CONFIG=ROOT/'configs/biomaster_best_model_optimization_20260906.json'


def identity(cutoff,parent=None):
    return dict(original=original_identity(),protocol=file_identity(CONFIG),cutoff=cutoff,
        sources=[file_identity(ROOT/n) for n in ['biomaster/best_model_training.py','biomaster/refined_interaction.py',
                 'scripts/train_biomaster_best_model.py']],
        data=[file_identity(DATA/n) for n in ['ROLL_MANIFEST.json','ASSAY_MANIFEST.json',
              'supplemental_features/ATOM_MANIFEST.json','supplemental_features/SUPPLEMENT_PROVENANCE.json']],
        parent=file_identity(parent) if parent else None)


def train(cutoff,variant,seed,refinement=None,parent=None,smoke_steps=0):
    p=json.loads(CONFIG.read_text())
    if cutoff not in p['cutoffs'] or variant not in p['variants'] or seed not in p['seeds']:
        raise ValueError('run outside frozen development protocol')
    if bool(refinement) != bool(parent):raise ValueError('refinement requires a declared parent')
    if refinement and refinement not in p['refinement']['variants']:raise ValueError('undeclared refinement')
    ident=identity(cutoff,parent)
    name=variant if refinement is None else variant+'__refine_'+refinement
    folder=OUT/('smoke' if smoke_steps else 'optimization')/f'roll_{cutoff}'/name/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RESULT.json').exists():
        result=json.loads((folder/'RESULT.json').read_text())
        if result['identity']!=ident:raise ValueError('completed run source/configuration drift')
        return result
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(seed);np.random.seed(seed)
    stage=RollingStage(DATA,SOURCE,cutoff)
    if refinement:
        cfg=RefinedConfig(variant=refinement,width=p['width'],dropout=p['dropout'],
                          pair_width=p['refinement']['pair_width'],blocks=p['refinement']['blocks'])
        if refinement=='capacity':cfg=matched_refinement_config(RefinedConfig(**{**asdict(cfg),'variant':'geometry'}))
        model=RefinedInteraction(cfg)
        state=torch.load(parent,map_location='cpu',weights_only=False)
        if state['cutoff']!=cutoff or state['seed']!=seed or state['variant']!=variant:
            raise ValueError('warm parent must match cutoff, seed and objective')
        if state.get('refinement') is not None:raise ValueError('warm parent must be an unrefined global model')
        model.warm_start(state['model'])
    else:
        cfg=UnifiedConfig(variant='global',width=p['width'],dropout=p['dropout'])
        model=UnifiedInteraction(cfg)
    model.cuda();bank=SupplementedFeatureBank(OUTPUT,DATA/'supplemental_features',local=model.is_local)
    flags=p['variants'][variant]
    queries=[]
    if flags['query']:
        queries=[ObservedQueries(stage.train,key,head) for head,key in enumerate(['drug_feature_index','target_feature_index'])]
    assay=None
    if flags['assay']:
        frame=pd.read_csv(DATA/f'ASSAY_CONTRASTS_{cutoff}.csv.gz')
        if frame.max_document_year.max()>cutoff:raise ValueError('future assay supervision')
        assay=ObservedQueries(frame,'assay_group',1)
    ema=WeightAverage(model,p['ema_decay'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    best=-math.inf;best_epoch=0;history=[];start_epoch=1
    if (folder/'LAST.pt').exists():
        state=torch.load(folder/'LAST.pt',map_location='cuda',weights_only=False)
        if state['identity']!=ident:raise ValueError('resume source/configuration drift')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        ema.model.load_state_dict(state['ema']);ema.updates=state['ema_updates']
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];start_epoch=state['epoch']+1
        del state
    started=time.monotonic();positive=float(stage.y.mean())
    max_epochs=p['refinement']['extra_epochs'] if refinement else p['max_epochs']
    if smoke_steps:max_epochs=1
    event_base=dict(cutoff=cutoff,variant=variant,refinement=refinement,seed=seed)
    def log(event):
        value={**event_base,'utc':datetime.now(timezone.utc).isoformat(),**event}
        write_json(folder/'STATUS.json',value);print(json.dumps(value),flush=True)
    log(dict(event='started',train_rows=len(stage.y),parameters=sum(x.numel() for x in model.parameters()),
             observed_query_groups=[len(q.groups) for q in queries],assay_groups=len(assay.groups) if assay else 0))
    already_stopped=(not refinement and history and len(history)>=p['min_epochs'] and len(history)-best_epoch>=p['patience'])
    for epoch in ([] if already_stopped else range(start_epoch,max_epochs+1)):
        rng=np.random.default_rng(seed+epoch*104729)
        retrieval_rng=np.random.default_rng(seed+epoch*104729+11)
        query_rng=np.random.default_rng(seed+epoch*104729+23)
        assay_rng=np.random.default_rng(seed+epoch*104729+37)
        torch.manual_seed(seed+epoch*104729)
        order=rng.permutation(len(stage.y));exposure=hashlib.sha256();qexposure=hashlib.sha256();aexposure=hashlib.sha256()
        losses={k:[] for k in ['bce','retrieval','query','assay']}
        base_lr=p['refinement']['learning_rate'] if refinement else p['learning_rate']
        lr=base_lr*(.2+.8*(1+math.cos(math.pi*(epoch-1)/max_epochs))/2)
        for group in optimizer.param_groups:group['lr']=lr
        model.train();epoch_start=time.monotonic()
        for step,start in enumerate(range(0,len(order),p['batch_size'])):
            rows=order[start:start+p['batch_size']];exposure.update(rows.astype('<i8').tobytes())
            optimizer.zero_grad(set_to_none=True)
            y=torch.tensor(stage.y[rows],device='cuda');weight=torch.where(y.bool(),.5/positive,.5/(1-positive))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=model(bank.batch(stage.d[rows],stage.t[rows],cfg.variant)).float()
                loss=(F.binary_cross_entropy_with_logits(scores,y[:,None].expand(-1,2),reduction='none')*weight[:,None]).mean()
            loss.backward();losses['bce'].append(float(loss.detach()))
            if step%p['retrieval_every_steps']==0:
                d,t,known,groups=stage.retrieval(retrieval_rng,p)
                exposure.update(np.stack([d,t,known],1).astype('<i8').tobytes())
                terms=[]
                for lo,hi,head in groups:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        s=model(bank.batch(d[lo:hi],t[lo:hi],cfg.variant))[:,head].float()
                        positive_mask=torch.tensor(known[lo:hi],device='cuda')
                        term=-F.log_softmax(s/p['retrieval_temperature'],dim=0)[positive_mask].mean()
                    (term*p['retrieval_weight']/len(groups)).backward();terms.append(float(term.detach()))
                losses['retrieval'].append(float(np.mean(terms)))
            for kind,active,every,weight_value,generator in [
                ('query',bool(queries),p['observed_query_every_steps'],p['observed_query_weight'],query_rng),
                ('assay',assay is not None,p['assay_every_steps'],p['assay_weight'],assay_rng)]:
                if not active or step%every:continue
                streams=queries if kind=='query' else [assay]
                samples=[]
                for stream in streams:
                    samples.extend(stream.sample(generator,p['observed_queries_per_direction'] if kind=='query' else p['assay_queries'],p['observed_per_class']))
                terms=[]
                for d,t,yb,head in samples:
                    digest=qexposure if kind=='query' else aexposure
                    digest.update(np.stack([d,t,yb,np.full(len(d),head)],1).astype('<i8').tobytes())
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        scores=model(bank.batch(d,t,cfg.variant))[:,head].float()
                        term=measured_pair_loss(scores,torch.tensor(yb,device='cuda'),p['pair_temperature'])
                    (term*weight_value/len(samples)).backward();terms.append(float(term.detach()))
                losses[kind].append(float(np.mean(terms)))
            if any(values and not np.isfinite(values[-1]) for values in losses.values()):raise FloatingPointError('nonfinite loss')
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();ema.update(model)
            if (step+1)%200==0:log(dict(event='step',epoch=epoch,step=step+1,seconds=time.monotonic()-epoch_start))
            if smoke_steps and step+1>=smoke_steps:break
        metrics,query_metrics=(None,None) if smoke_steps else validation(ema.model,bank,stage,p)
        value=float(epoch) if refinement or smoke_steps else metrics['selection']
        if value>best:
            best=value;best_epoch=epoch
            torch.save(dict(model=ema.model.state_dict(),config=asdict(cfg),seed=seed,cutoff=cutoff,variant=variant,
                refinement=refinement,epoch=epoch,identity=ident,validation=metrics,ema_updates=ema.updates),folder/'BEST.pt')
            if query_metrics is not None:query_metrics.to_csv(folder/'VALIDATION_QUERY_METRICS.csv',index=False)
        row=dict(epoch=epoch,validation=metrics,selected_epoch=best_epoch,learning_rate=lr,
            mean_losses={k:float(np.mean(v)) if v else None for k,v in losses.items()},gradient_norm=float(grad),
            exposure_sha256=exposure.hexdigest(),query_exposure_sha256=qexposure.hexdigest(),assay_exposure_sha256=aexposure.hexdigest(),
            coverage_rows=min(len(order),len(losses['bce'])*p['batch_size']),epoch_seconds=time.monotonic()-epoch_start)
        history.append(row);write_json(folder/'HISTORY.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),ema=ema.model.state_dict(),ema_updates=ema.updates,
            identity=ident,epoch=epoch,best=best,best_epoch=best_epoch,history=history),folder/'LAST.pt')
        log(dict(event='epoch_completed',**row))
        if not refinement and epoch>=p['min_epochs'] and epoch-best_epoch>=p['patience']:break
    result=dict(**event_base,identity=ident,selected_epoch=best_epoch,trained_epochs=len(history),
        validation=history[best_epoch-1]['validation'],train_rows=len(stage.y),train_max_year=cutoff,
        trainable_parameters=sum(x.numel() for x in model.parameters()),seconds=time.monotonic()-started,
        test_labels_used=False,smoke_only=bool(smoke_steps),ema_checkpoint=True,checkpoint=file_identity(folder/'BEST.pt'))
    write_json(folder/'RESULT.json',result);log(dict(event='completed',validation=result['validation']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff',type=int,choices=[2018,2020],required=True)
    parser.add_argument('--variant',choices=['global_ema','global_query','global_assay'],required=True)
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--refinement',choices=['global','capacity','site','geometry'])
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--smoke-steps',type=int,default=0)
    a=parser.parse_args();train(a.cutoff,a.variant,a.seed,a.refinement,a.parent,a.smoke_steps)

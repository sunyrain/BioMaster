#!/usr/bin/env python3
"""Fixed-exposure anchored development and selection-gated final refitting."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
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
from biomaster.anchored_interaction import AnchoredInteraction,AnchoredFeatureBank,anchored_config,matched_anchored_config
from biomaster.chemical_representation import ChemicalConfig,ChemicalInteraction
from biomaster.molecular_controls import MolecularControlConfig,MolecularControlInteraction
from biomaster.unified_interaction import UnifiedConfig,UnifiedInteraction
from biomaster.best_model_training import RollingStage,old_labels,ObservedQueries,measured_pair_loss,WeightAverage
from biomaster.odti_pockets_v3 import file_identity
from train_biomaster_unified_interaction import validation
from prepare_biomaster_unified_interaction import write_json,SOURCE,OUTPUT

OUT=ROOT/'outputs/biomaster_best_model_20260906'
DATA=OUT/'data'
PROTOCOL=ROOT/'configs/biomaster_best_model_refinement_20260906.json'
OPTIMIZATION=ROOT/'configs/biomaster_best_model_optimization_20260906.json'


class RefitStage:
    retrieval=RollingStage.retrieval
    def __init__(self,cutoff):
        if cutoff not in [2022,2025]:raise ValueError('invalid refit cutoff')
        self.cutoff=cutoff;self.name=f'refit_{cutoff}'
        path=SOURCE/'FINAL_TRAIN.csv.gz' if cutoff==2022 else DATA/'fullfit_2025/TRAIN.csv.gz'
        self.train=pd.read_csv(path)
        if self.train.max_document_year.max()>cutoff:raise ValueError('future supervised relation')
        if not self.train.binary_label.isin([0,1]).all():raise ValueError('observed binary labels required')
        if self.train.duplicated(['drug_feature_index','target_feature_index']).any():raise ValueError('duplicate relation')
        self.d=self.train.drug_feature_index.to_numpy(int);self.t=self.train.target_feature_index.to_numpy(int)
        self.y=self.train.binary_label.to_numpy(np.float32)
        self.old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
        self.nt=len(pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz'))
        self.known=old_labels(self.train,self.old,self.nt)
        self.positive_queries=[np.flatnonzero(self.known.any(1)),np.flatnonzero(self.known.any(0))]
        self.val=None;self.train_file=path


def source_identity():
    paths=['biomaster/anchored_interaction.py','biomaster/chemical_representation.py','biomaster/molecular_controls.py','biomaster/refined_interaction.py',
           'biomaster/unified_interaction.py','biomaster/unified_features.py','biomaster/best_model_training.py',
           'biomaster/ranking_audit.py','scripts/train_biomaster_unified_interaction.py',
           'scripts/fit_biomaster_selected_model.py']
    return [file_identity(ROOT/p) for p in paths]


def fit(cutoff,seed,variant,mode='refine',smoke_steps=0,parent_override=None):
    p=json.loads(OPTIMIZATION.read_text());protocol=json.loads(PROTOCOL.read_text())
    selection=json.loads((OUT/'GLOBAL_PARENT_SELECTION.json').read_text())
    if selection['status']!='FROZEN_GLOBAL_PARENT' or selection['test_labels_used_for_selection']:
        raise ValueError('global parent selection is not valid')
    if selection['refinement_protocol']!=file_identity(PROTOCOL):raise ValueError('refinement protocol drift')
    if seed not in protocol['seeds'] or variant not in protocol['variants']:raise ValueError('undeclared run')
    if mode not in ['refine','global_refit']:raise ValueError('invalid fitting mode')
    development=cutoff in protocol['cutoffs']
    final_selection=None
    if development:
        if mode!='refine':raise ValueError('global development models already exist')
        stage=RollingStage(DATA,SOURCE,cutoff)
    else:
        final_selection=json.loads((OUT/'FINAL_ARCHITECTURE_SELECTION.json').read_text())
        if final_selection['status']!='FROZEN_FINAL_ARCHITECTURE' or final_selection['test_labels_used_for_selection']:
            raise ValueError('final architecture must be frozen before refitting')
        if final_selection['global_parent_selection']!=file_identity(OUT/'GLOBAL_PARENT_SELECTION.json'):
            raise ValueError('global selection changed after final architecture selection')
        if mode=='refine' and final_selection['selected_variant']!=variant:raise ValueError('refit must use selected refinement')
        if cutoff==2025 and seed!=final_selection['deployment_seed']:raise ValueError('full fit uses one preselected seed')
        stage=RefitStage(cutoff)
    recipe=selection['selected']['recipe']
    flags=p['variants'].get(recipe,dict(ema=False,query=False,assay=False))
    representation=selection['selected']['representation']
    name=variant if mode=='refine' else 'global_refit'
    folder=OUT/('anchored_smoke' if smoke_steps else ('anchored' if development else 'final_fit'))/f'cutoff_{cutoff}'/name/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    run_lock=(folder/'TRAIN.lock').open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=None
    if mode=='refine':
        if development:
            matches=[x for x in selection['parents'] if x['cutoff']==cutoff and x['seed']==seed]
            if len(matches)!=1:raise ValueError('one parent per seed/window required')
            parent=Path(matches[0]['checkpoint']['path'])
            if file_identity(parent)!=matches[0]['checkpoint']:raise ValueError('parent weights changed')
        else:parent=OUT/'final_fit'/f'cutoff_{cutoff}'/'global_refit'/f'seed_{seed}'/'BEST.pt'
        if parent_override is not None:
            if not smoke_steps:raise ValueError('parent override permitted only for a labelled smoke test')
            parent=Path(parent_override)
    train_file=DATA/f'roll_{cutoff}/TRAIN.csv.gz' if development else stage.train_file
    ident=dict(sources=source_identity(),protocol=file_identity(PROTOCOL),optimization=file_identity(OPTIMIZATION),
        global_selection=file_identity(OUT/'GLOBAL_PARENT_SELECTION.json'),parent=file_identity(parent) if parent else None,
        final_selection=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json') if final_selection else None,
        train=file_identity(train_file),atoms=file_identity(OUTPUT/'ATOM_MANIFEST.json'),targets=file_identity(OUTPUT/'TARGET_MANIFEST.json'),
        supplement=file_identity(DATA/'supplemental_features/ATOM_MANIFEST.json'),
        chemical=file_identity(OUT/'CHEMICAL_FEATURE_IDENTITY.json') if representation!='drugclip' else None,
        assay=file_identity(DATA/(f'ASSAY_CONTRASTS_{cutoff}.csv.gz' if cutoff<=2022 else 'fullfit_2025/ASSAY_CONTRASTS.csv.gz')) if flags['assay'] else None)
    if (folder/'RESULT.json').exists():
        r=json.loads((folder/'RESULT.json').read_text())
        if r['identity']!=ident:raise ValueError('completed fitting identity changed')
        run_lock.close();return r
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(seed);np.random.seed(seed);torch.cuda.reset_peak_memory_stats()
    if mode=='refine':
        state=torch.load(parent,map_location='cpu',weights_only=False)
        if state['cutoff']!=cutoff or state['seed']!=seed:raise ValueError('parent cutoff/seed mismatch')
        cfg=anchored_config(state['config'],variant)
        if variant=='capacity':cfg=matched_anchored_config(anchored_config(state['config'],'geometry'))
        model=AnchoredInteraction(cfg);model.warm_start(state['model']);family='anchored'
        parent_epoch=state['epoch'];del state
    else:
        configuration=selection['parents'][0]['config']
        if representation=='drugclip':cfg=UnifiedConfig(**configuration);model=UnifiedInteraction(cfg);family='unified'
        elif representation in ['bermol','bermol_morgan']:cfg=ChemicalConfig(**configuration);model=ChemicalInteraction(cfg);family='chemical'
        else:cfg=MolecularControlConfig(**configuration);model=MolecularControlInteraction(cfg);family='molecular_control'
        parent_epoch=None
    model.cuda();bank=AnchoredFeatureBank(OUTPUT,DATA/'supplemental_features',SOURCE,representation,local=model.is_local)
    if not bank.required[torch.tensor(stage.d,device='cuda')].all():raise ValueError('missing training molecule features')
    use_ema=mode=='refine' or flags['ema'];average=WeightAverage(model,protocol['ema_decay']) if use_ema else None
    queries=[ObservedQueries(stage.train,key,h) for h,key in enumerate(['drug_feature_index','target_feature_index'])] if flags['query'] else []
    assay=None
    if flags['assay']:
        path=DATA/f'ASSAY_CONTRASTS_{cutoff}.csv.gz' if cutoff<=2022 else DATA/'fullfit_2025/ASSAY_CONTRASTS.csv.gz'
        af=pd.read_csv(path)
        if af.max_document_year.max()>cutoff:raise ValueError('future assay supervision')
        assay=ObservedQueries(af,'assay_group',1)
    max_epochs=protocol['extra_epochs'] if mode=='refine' else selection['fixed_refit_global_epochs']
    if smoke_steps:max_epochs=1
    base_lr=protocol['learning_rate'] if mode=='refine' else p['learning_rate']
    optimizer=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=p['weight_decay'])
    history=[];start_epoch=1;started=time.monotonic()
    if (folder/'LAST.pt').exists():
        state=torch.load(folder/'LAST.pt',map_location='cuda',weights_only=False)
        if state['identity']!=ident:raise ValueError('resume identity drift')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        if use_ema:average.model.load_state_dict(state['ema']);average.updates=state['ema_updates']
        history=state['history'];start_epoch=state['epoch']+1;del state
    event=dict(cutoff=cutoff,seed=seed,variant=variant,mode=mode,recipe=recipe,representation=representation)
    def log(value):
        row=dict(utc=datetime.now(timezone.utc).isoformat(),**event,**value)
        write_json(folder/'STATUS.json',row);print(json.dumps(row),flush=True)
    log(dict(event='started',train_rows=len(stage.y),epochs=max_epochs,parent_epoch=parent_epoch,
        parameters=sum(x.numel() for x in model.parameters()),family=family,ema=use_ema))
    prevalence=float(stage.y.mean())
    for epoch in range(start_epoch,max_epochs+1):
        rng=np.random.default_rng(seed+epoch*104729);retrieval_rng=np.random.default_rng(seed+epoch*104729+11)
        query_rng=np.random.default_rng(seed+epoch*104729+23);assay_rng=np.random.default_rng(seed+epoch*104729+37)
        torch.manual_seed(seed+epoch*104729);order=rng.permutation(len(stage.y))
        exposure=hashlib.sha256();extra={k:hashlib.sha256() for k in ['query','assay']}
        losses={k:[] for k in ['bce','retrieval','query','assay']};epoch_start=time.monotonic()
        schedule_epochs=protocol['extra_epochs'] if mode=='refine' else p['max_epochs']
        lr=base_lr*(.2+.8*(1+math.cos(math.pi*(epoch-1)/schedule_epochs))/2)
        for group in optimizer.param_groups:group['lr']=lr
        model.train()
        for step,start in enumerate(range(0,len(order),p['batch_size'])):
            rows=order[start:start+p['batch_size']];exposure.update(rows.astype('<i8').tobytes());optimizer.zero_grad(set_to_none=True)
            y=torch.tensor(stage.y[rows],device='cuda');weight=torch.where(y.bool(),.5/prevalence,.5/(1-prevalence))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                scores=model(bank.batch(stage.d[rows],stage.t[rows],cfg.variant)).float()
                loss=(F.binary_cross_entropy_with_logits(scores,y[:,None].expand(-1,2),reduction='none')*weight[:,None]).mean()
            loss.backward();losses['bce'].append(float(loss.detach()))
            if step%p['retrieval_every_steps']==0:
                d,t,known,groups=stage.retrieval(retrieval_rng,p);exposure.update(np.stack([d,t,known],1).astype('<i8').tobytes());terms=[]
                for lo,hi,head in groups:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        s=model(bank.batch(d[lo:hi],t[lo:hi],cfg.variant))[:,head].float()
                        term=-F.log_softmax(s/p['retrieval_temperature'],dim=0)[torch.tensor(known[lo:hi],device='cuda')].mean()
                    (term*p['retrieval_weight']/len(groups)).backward();terms.append(float(term.detach()))
                losses['retrieval'].append(float(np.mean(terms)))
            for kind,streams,generator,every,nqueries,loss_weight in [
                ('query',queries,query_rng,p['observed_query_every_steps'],p['observed_queries_per_direction'],p['observed_query_weight']),
                ('assay',[assay] if assay else [],assay_rng,p['assay_every_steps'],p['assay_queries'],p['assay_weight'])]:
                if not streams or step%every:continue
                samples=[]
                for stream in streams:samples.extend(stream.sample(generator,nqueries,p['observed_per_class']))
                terms=[]
                for d,t,yb,head in samples:
                    extra[kind].update(np.stack([d,t,yb,np.full(len(d),head)],1).astype('<i8').tobytes())
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        s=model(bank.batch(d,t,cfg.variant))[:,head].float()
                        term=measured_pair_loss(s,torch.tensor(yb,device='cuda'),p['pair_temperature'])
                    (term*loss_weight/len(samples)).backward();terms.append(float(term.detach()))
                losses[kind].append(float(np.mean(terms)))
            if any(v and not np.isfinite(v[-1]) for v in losses.values()):raise FloatingPointError('nonfinite fitting loss')
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            if use_ema:average.update(model)
            if (step+1)%200==0:log(dict(event='step',epoch=epoch,step=step+1,seconds=time.monotonic()-epoch_start))
            if smoke_steps and step+1>=smoke_steps:break
        metrics=None;evaluation_model=average.model if use_ema else model
        if development and epoch==max_epochs and not smoke_steps:
            metrics,q=validation(evaluation_model,bank,stage,p);q.to_csv(folder/'VALIDATION_QUERY_METRICS.csv',index=False)
        row=dict(epoch=epoch,validation=metrics,mean_losses={k:float(np.mean(v)) if v else None for k,v in losses.items()},
            exposure_sha256=exposure.hexdigest(),query_exposure_sha256=extra['query'].hexdigest(),assay_exposure_sha256=extra['assay'].hexdigest(),
            coverage_rows=min(len(order),len(losses['bce'])*p['batch_size']),learning_rate=lr,gradient_norm=float(grad),
            epoch_seconds=time.monotonic()-epoch_start)
        history.append(row);write_json(folder/'HISTORY.json',history)
        # Commit exported weights before advancing the resume epoch. A crash
        # cannot leave LAST at epoch 4 while BEST still contains epoch 3.
        temporary=folder/'BEST.pt.tmp'
        torch.save(dict(model=evaluation_model.state_dict(),config=asdict(cfg),family=family,seed=seed,cutoff=cutoff,
            epoch=epoch,identity=ident,validation=metrics,variant=variant,refinement=variant if mode=='refine' else None,
            recipe=recipe,mode=mode,ema=use_ema),temporary)
        temporary.replace(folder/'BEST.pt')
        temporary=folder/'LAST.pt.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),ema=average.model.state_dict() if use_ema else None,
            ema_updates=average.updates if use_ema else 0,identity=ident,epoch=epoch,history=history),temporary)
        temporary.replace(folder/'LAST.pt')
        log(dict(event='epoch_completed',**row))
    if len(history)!=max_epochs:raise ValueError('fixed training exposure incomplete')
    result=dict(**event,identity=ident,family=family,selected_epoch=max_epochs,trained_epochs=len(history),
        validation=history[-1]['validation'],train_rows=len(stage.y),train_max_year=cutoff,
        trainable_parameters=sum(x.numel() for x in model.parameters()),parent_epoch=parent_epoch,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,ema_checkpoint=use_ema,
        smoke_only=bool(smoke_steps),test_labels_used=cutoff>2022,test_labels_used_for_selection=False,
        training_role='development' if development else ('temporal_refit' if cutoff==2022 else 'fullfit_deployment'),
        checkpoint=file_identity(folder/'BEST.pt'))
    write_json(folder/'RESULT.json',result);log(dict(event='completed',validation=result['validation']))
    run_lock.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff',type=int,choices=[2018,2020,2022,2025],required=True)
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--variant',choices=['global','capacity','site','geometry'],default='global')
    parser.add_argument('--mode',choices=['refine','global_refit'],default='refine')
    parser.add_argument('--smoke-steps',type=int,default=0);parser.add_argument('--parent-override',type=Path)
    a=parser.parse_args();fit(a.cutoff,a.seed,a.variant,a.mode,a.smoke_steps,a.parent_override)

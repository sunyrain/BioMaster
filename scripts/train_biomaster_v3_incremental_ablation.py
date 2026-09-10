#!/usr/bin/env python3
"""Train a preregistered V3 residual factorial using validation labels only.

Test metrics are deliberately implemented in a separate release entrypoint.
"""
import argparse
from dataclasses import asdict
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
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.odti_v3_incremental import IncrementalConfig, IncrementalV3, measured_query_loss
from biomaster.selectivity_training_v3 import QueryBatchStream
from biomaster.old_drug_ranking import known_ranking
from prepare_biomaster_v3_incremental_ablation import OUT, OLD, DATA, CFG
from build_biomaster_odti_v4_features import sha256, write_json


def source_identity():
    names=['scripts/train_biomaster_v3_incremental_ablation.py','biomaster/odti_v3_incremental.py',
           'biomaster/selectivity_training_v3.py','biomaster/old_drug_ranking.py']
    return {name:sha256(ROOT/name) for name in names}


def two_class_groups(labels, ids):
    frame=pd.DataFrame({'y':labels,'g':ids})
    return [np.asarray(pos) for pos in frame.groupby('g',sort=False).indices.values()
            if np.unique(labels[pos]).size==2]


class Runtime:
    def __init__(self, verify=True):
        self.protocol=json.loads((OUT/'PROTOCOL.json').read_text())
        assert self.protocol['config_sha256']==sha256(CFG)
        features=json.loads((OUT/'FEATURES.json').read_text())
        if verify:
            for p,h in {**features['sources'],**features['files']}.items():
                assert sha256(ROOT/p)==h,p
        self.identity={'protocol_sha256':sha256(OUT/'PROTOCOL.json'),'features_sha256':sha256(OUT/'FEATURES.json'),
                       'source_sha256':source_identity()}
        self.data=pd.read_csv(DATA/'RELATIONS_V3.csv.gz')
        self.drugs=self.data.drug_feature_index.to_numpy(np.int64)
        self.targets=self.data.target_feature_index.to_numpy(np.int64)
        self.labels=self.data.binary_label.to_numpy(np.float32)
        self.positions={s:np.flatnonzero(self.data.split.eq(s)) for s in ['train','validation','test']}
        self.old_drugs=pd.read_csv(OLD/'OLD_DRUGS_720.csv')
        self.old_targets=pd.read_csv(OLD/'TARGETS_384.csv.gz')
        self.old_val=np.flatnonzero(self.old_drugs.training_role.eq('validation'))
        self.old_train=np.flatnonzero(self.old_drugs.training_role.eq('train'))
        self.old_val_rows=(self.old_val[:,None]*384+np.arange(384)).reshape(-1)
        with np.load(OLD/'LABELS_AND_SCOPES.npz') as label_arrays:
            # Retain only validation known labels and train biochemical labels.
            self.known_val=label_arrays['known_relationship'][self.old_val]
            self.train_observed=label_arrays['observed_binary'][self.old_train]
        self.banks={}
        for scope in ['RELATION','OLD']:
            self.banks[scope]={name:torch.from_numpy(np.load(OUT/f'{scope}_{name}.npy')).float().cuda()
                               for name in ['BASE','HIDDEN','SUPPORT']}
        self.axes={k:v for k,v in np.load(OUT/'ROW_AXES.npz').items()}
        self.gpu_axes={k:torch.from_numpy(v).cuda() for k,v in self.axes.items()}
        self.y=torch.from_numpy(self.labels).cuda()
        self.pretrained_drug=None
        self.pretrained_target=None
        self.streams=[QueryBatchStream(ids,self.labels,self.positions['train'],
            self.protocol['queries_per_batch'],self.protocol['max_items_per_query']) for ids in [self.drugs,self.targets]]
        valid=self.positions['validation']
        self.val_groups=[two_class_groups(self.labels[valid],ids[valid]) for ids in [self.drugs,self.targets]]
        self.base_validation=self.validation(None)

    def load_pretrained(self):
        if self.pretrained_drug is None:
            folder=ROOT/'outputs/biomaster_odti_v4_20260905/features'
            manifest=json.loads((folder/'FEATURE_MANIFEST_V4.json').read_text())
            used=[folder/'BERMOL768_FLOAT32_V4.npy',folder/'ESM2_FULL_MEAN1280_FLOAT32_V4.npy']
            for path in used:
                assert sha256(path)==manifest['files'][str(path.relative_to(ROOT))]
            available=np.load(folder/'BERMOL_AVAILABLE_V4.npy')
            assert available[self.drugs].all()
            self.pretrained_drug=torch.from_numpy(np.concatenate([np.load(used[0]),np.load(OLD/'features/OLD720_BERMOL768.npy')])).cuda()
            self.pretrained_target=torch.from_numpy(np.concatenate([np.load(used[1]),np.load(OLD/'features/ADDED_FULL_ESM2_MEANS.npy')])).cuda()

    def forward(self, model, scope, rows):
        if not isinstance(rows,torch.Tensor):
            rows=torch.as_tensor(rows,device='cuda')
        bank=self.banks[scope]
        kwargs={}
        if model.config.pretrained:
            key='relation' if scope=='RELATION' else 'old'
            kwargs={'drug':self.pretrained_drug[self.gpu_axes[key+'_drugs'][rows]],
                    'target':self.pretrained_target[self.gpu_axes[key+'_targets'][rows]]}
        return model(bank['BASE'][rows],bank['HIDDEN'][rows],bank['SUPPORT'][rows],**kwargs)

    @torch.no_grad()
    def predict(self,model,scope,rows):
        if model is None:
            base=self.banks[scope]['BASE'][torch.as_tensor(rows,device='cuda')].cpu().numpy()
            return np.repeat(base[:,None],2,axis=1)
        model.eval()
        return np.concatenate([self.forward(model,scope,rows[start:start+4096]).cpu().numpy()
                               for start in range(0,len(rows),4096)])

    def validation(self,model):
        old=self.predict(model,'OLD',self.old_val_rows).reshape(len(self.old_val),384,2)
        dnames=self.old_drugs.ligand_inchikey.to_numpy()[self.old_val]
        tnames=self.old_targets.target_chembl_id.to_numpy()
        result={}
        for direction,y,s,q,c in [('d2t',self.known_val,old[:,:,0],dnames,tnames),
                                  ('t2d',self.known_val.T,old[:,:,1].T,tnames,dnames)]:
            metrics,_,_=known_ranking(y,s,q,c)
            result[direction]={'ap':metrics['macro_ap'],'r20':metrics['macro_recall_at_20'],
                               'queries':metrics['evaluated_queries'],'candidates':metrics['candidates_per_query']}
        rows=self.positions['validation']
        values=self.predict(model,'RELATION',rows)
        for i,direction in enumerate(['d2t','t2d']):
            result[direction]['measured_ap']=float(np.mean([average_precision_score(self.labels[rows][g],values[g,i]) for g in self.val_groups[i]]))
            result[direction]['measured_queries']=len(self.val_groups[i])
        result['selection']=sum(.35*result[d]['ap']+.15*result[d]['r20'] for d in ['d2t','t2d'])
        return result

    def eligible(self,metrics):
        p=self.protocol
        return all(metrics[d]['ap']>=self.base_validation[d]['ap']-p['old_validation_ap_tolerance'] and
                   metrics[d]['measured_ap']>=self.base_validation[d]['measured_ap']-p['measured_validation_ap_tolerance']
                   for d in ['d2t','t2d'])


def query_batch(rows, ids, width):
    groups=list(dict.fromkeys(ids[rows].tolist()))
    padded=np.zeros((len(groups),width),np.int64)
    mask=np.zeros_like(padded,dtype=bool)
    for i,g in enumerate(groups):
        part=rows[ids[rows]==g]
        assert 0<len(part)<=width
        padded[i,:len(part)]=part
        mask[i,:len(part)]=True
    return padded,mask


def teacher_pairs(rt,seed):
    path=OUT/f'TEACHER_{seed}.npz'
    p=rt.protocol
    if path.exists():
        return {k:v for k,v in np.load(path).items()}
    sim=rt.banks['OLD']['SUPPORT'][:,1].cpu().numpy().reshape(720,384)[rt.old_train]
    result={}
    for direction in [0,1]:
        rng=np.random.default_rng(seed+direction*1009)
        collected=[]
        count=0
        while count<p['teacher_pairs_per_direction']:
            n=65536
            if direction==0:
                d1=d2=rng.integers(len(rt.old_train),size=n)
                t1,t2=rng.integers(384,size=(2,n))
            else:
                d1,d2=rng.integers(len(rt.old_train),size=(2,n))
                t1=t2=rng.integers(384,size=n)
            first,second=sim[d1,t1],sim[d2,t2]
            y1,y2=rt.train_observed[d1,t1],rt.train_observed[d2,t2]
            conflict=np.isfinite(y1)&np.isfinite(y2)&((first-second)*(y1-y2)<0)
            valid=(np.maximum(first,second)>=p['teacher_min_similarity'])&(np.abs(first-second)>=p['teacher_min_gap'])&~conflict
            a=rt.old_train[d1[valid]]*384+t1[valid]
            b=rt.old_train[d2[valid]]*384+t2[valid]
            prob=1/(1+np.exp(-(first[valid]-second[valid])/p['teacher_temperature']))
            part=np.column_stack([a,b,prob,np.maximum(first[valid],second[valid])**2])
            collected.append(part);count+=len(part)
        result[f'direction_{direction}']=np.concatenate(collected)[:p['teacher_pairs_per_direction']]
    np.savez_compressed(path,**result)
    return result


def train_one(rt,variant,seed):
    p=rt.protocol
    flags=p['variants'][variant]
    folder=OUT/'runs'/variant/f'seed_{seed}'
    if (folder/'RESULT.json').exists():
        result=json.loads((folder/'RESULT.json').read_text())
        assert result['identity']==rt.identity
        return result
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'incomplete run requires audit before resuming: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    config=IncrementalConfig(width=p['hidden_width'],residual_bound=p['residual_bound'],
        evidence_bound=p['evidence_bound'],evidence=flags['evidence'],pretrained=flags['pretrained'])
    model=IncrementalV3(config).cuda()
    if flags['pretrained']:
        rt.load_pretrained()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    best=rt.base_validation
    best_epoch=0
    history=[{'epoch':0,'validation':best,'eligible':True,'note':'exact frozen V3, no update'}]
    started=time.monotonic()

    def snapshot(epoch):
        torch.save({'config':asdict(config),'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'epoch':epoch,'variant':variant,'seed':seed,'identity':rt.identity,
                    'validation':best},folder/'BEST.pt')

    snapshot(0)
    teacher=teacher_pairs(rt,seed) if flags['teacher'] else None
    for epoch in range(1,p['epochs']+1):
        torch.manual_seed(seed+epoch*104729)
        rng=np.random.default_rng(seed+epoch*104729)
        exposure=hashlib.sha256()
        model.train()
        losses=[]
        lr=p['learning_rate']*(.2+.8*(1+math.cos(math.pi*(epoch-1)/p['epochs']))/2)
        for group in optimizer.param_groups:
            group['lr']=lr

        def update(loss):
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite adapter loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            model.constrain()
            losses.append(float(loss.detach()))

        coverage=rng.permutation(rt.positions['train'])
        for start in range(0,len(coverage),p['batch_size']):
            ix=coverage[start:start+p['batch_size']]
            exposure.update(ix.astype('<i8').tobytes())
            scores=rt.forward(model,'RELATION',ix)
            yy=rt.y[torch.as_tensor(ix,device='cuda')][:,None].expand(-1,2)
            update(.25*F.binary_cross_entropy_with_logits(scores,yy))
        for direction,(stream,ids) in enumerate(zip(rt.streams,[rt.drugs,rt.targets])):
            batches=stream.batches(seed+epoch*104729+direction+101)
            for _ in range(math.ceil(len(stream.keys)/p['queries_per_batch'])):
                ix=next(batches)
                exposure.update(ix.astype('<i8').tobytes())
                padded,valid=query_batch(ix,ids,p['max_items_per_query'])
                rows=torch.as_tensor(padded.reshape(-1),device='cuda')
                scores=rt.forward(model,'RELATION',rows)[:,direction].reshape(padded.shape)
                yy=rt.y[rows].reshape(padded.shape)
                update(measured_query_loss(scores,yy,torch.as_tensor(valid,device='cuda'),
                    p['rank_weight'] if flags['ranking'] else 0.))
        if teacher is not None:
            for direction in [0,1]:
                pairs=teacher[f'direction_{direction}']
                for ix in np.array_split(rng.permutation(len(pairs)),4):
                    sample=pairs[ix]
                    scores=rt.forward(model,'OLD',sample[:,:2].astype(np.int64).reshape(-1))[:,direction].reshape(-1,2)
                    soft=torch.as_tensor(sample[:,2],dtype=torch.float32,device='cuda')
                    confidence=torch.as_tensor(sample[:,3],dtype=torch.float32,device='cuda')
                    loss=F.binary_cross_entropy_with_logits((scores[:,0]-scores[:,1])/2,soft,reduction='none')
                    update(p['teacher_weight']*(loss*confidence).mean())
        current=rt.validation(model)
        eligible=rt.eligible(current)
        if eligible and current['selection']>best['selection']+1e-12:
            best,best_epoch=current,epoch
            snapshot(epoch)
        row={'epoch':epoch,'validation':current,'eligible':eligible,'best_epoch':best_epoch,
             'mean_loss':float(np.mean(losses)),'lr':lr,'observed_exposure_sha256':exposure.hexdigest(),
             'coverage_rows':len(coverage),'d2t_queries':len(rt.streams[0].keys),'t2d_queries':len(rt.streams[1].keys),
             'elapsed_seconds':time.monotonic()-started}
        history.append(row)
        write_json(folder/'HISTORY.json',history)
        write_json(OUT/'STATUS.json',{'status':'TRAINING','variant':variant,'seed':seed,**row})
        print(json.dumps({'variant':variant,'seed':seed,'epoch':epoch,'best_epoch':best_epoch,
                          'eligible':eligible,'validation':current}),flush=True)
        model.train()
    result={'status':'COMPLETE','variant':variant,'seed':seed,'selected_epoch':best_epoch,
            'validation':best,'baseline_validation':rt.base_validation,'identity':rt.identity,
            'checkpoint_sha256':sha256(folder/'BEST.pt'),'seconds':time.monotonic()-started,
            'trainable_parameters':sum(x.numel() for x in model.parameters()),
            'test_metrics_computed':False,'observed_exposure_sha256':[r['observed_exposure_sha256'] for r in history[1:]],
            'teacher_pairs_sha256':sha256(OUT/f'TEACHER_{seed}.npz') if teacher is not None else None}
    write_json(folder/'RESULT.json',result)
    return result


def run():
    torch.set_num_threads(4)
    p=json.loads((OUT/'PROTOCOL.json').read_text())
    if (OUT/'SELECTION.json').exists():
        raise FileExistsError('Selection is frozen; use test release/evaluation entrypoint')
    rt=Runtime()
    manifest=OUT/'TRAINING_MANIFEST.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['identity']==rt.identity
    else:
        write_json(manifest,{'identity':rt.identity,'validation_queries':len(rt.old_val),
            'baseline_validation':rt.base_validation,'test_used_for_selection':False,
            'base_training_predictions':'in-sample frozen V3 features used for adapter fitting; validation entities excluded from backbone fitting'})
    results=[]
    for variant in p['variants']:
        for seed in p['seeds']:
            assert source_identity()==rt.identity['source_sha256'],'training source changed during matrix'
            results.append(train_one(rt,variant,seed))
    # Verify identical measured-relation and query exposure in every paired run.
    for seed in p['seeds']:
        matched=[r['observed_exposure_sha256'] for r in results if r['seed']==seed]
        assert all(v==matched[0] for v in matched)
    scores={name:float(np.mean([r['validation']['selection'] for r in results if r['variant']==name])) for name in p['variants']}
    winner=max(scores,key=scores.get)
    selection={'status':'FROZEN_BEFORE_TEST','winner':winner,'validation_scores':scores,'results':results,
        'protocol_sha256':sha256(OUT/'PROTOCOL.json'),'training_manifest_sha256':sha256(manifest),
        'selection_used_test':False,'same_observed_exposure_verified':True,
        'baseline_selection':rt.base_validation['selection'],'production_promotion':False}
    write_json(OUT/'SELECTION.json',selection)
    write_json(OUT/'STATUS.json',{'status':'TRAINING_COMPLETE_TEST_NOT_RELEASED','winner':winner,'runs':len(results)})
    print(json.dumps({'status':'TRAINING_COMPLETE','winner':winner,'validation_scores':scores}),flush=True)


if __name__=='__main__':
    run()

#!/usr/bin/env python3
"""Train known-relation retrieval adapters on training old drugs only."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.odti_v3_incremental import IncrementalConfig,IncrementalV3
from biomaster.old_relation_retrieval import known_relation_loss
from train_biomaster_v3_incremental_ablation import Runtime,teacher_pairs
from prepare_biomaster_v3_incremental_ablation import OUT,OLD
from build_biomaster_odti_v4_features import sha256,write_json

STAGE=OUT/'old_relation_stage'
CONFIG=ROOT/'configs/biomaster_v3_old_relation_ablation_20260906.json'


def source_identity():
    return {str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'biomaster/old_relation_retrieval.py']}


def train(rt,p,identity,known,variant,seed):
    flags=p['variants'][variant]
    folder=STAGE/'runs'/variant/f'seed_{seed}'
    if (folder/'RESULT.json').exists():
        r=json.loads((folder/'RESULT.json').read_text());assert r['identity']==identity
        return r
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'incomplete run: {folder}')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    config=IncrementalConfig(evidence=flags['evidence'],pretrained=flags['pretrained'])
    model=IncrementalV3(config).cuda()
    if flags['pretrained']:rt.load_pretrained()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    best,best_epoch=rt.base_validation,0
    history=[{'epoch':0,'validation':best,'eligible':True}]
    teacher=teacher_pairs(rt,seed) if flags['teacher'] else None
    started=time.monotonic()
    queries=[np.flatnonzero(known.any(1)),np.flatnonzero(known.any(0))]
    steps=[math.ceil(len(q)/p['retrieval_queries_per_batch']) for q in queries]

    def save(epoch):
        torch.save({'config':asdict(config),'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'epoch':epoch,'variant':variant,'seed':seed,'identity':identity,'validation':best},folder/'BEST.pt')

    save(0)
    for epoch in range(1,p['epochs']+1):
        rng=np.random.default_rng(seed+104729*epoch)
        torch.manual_seed(seed+104729*epoch)
        exposure=hashlib.sha256()
        model.train();losses=[]
        lr=p['learning_rate']*(.2+.8*(1+math.cos(math.pi*(epoch-1)/p['epochs']))/2)
        for group in optimizer.param_groups:group['lr']=lr

        def update(loss):
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();model.constrain();losses.append(float(loss.detach()))

        coverage=rng.permutation(rt.positions['train'])
        for start in range(0,len(coverage),p['observed_batch_size']):
            ix=coverage[start:start+p['observed_batch_size']]
            exposure.update(ix.astype('<i8').tobytes())
            scores=rt.forward(model,'RELATION',ix)
            yy=rt.y[torch.as_tensor(ix,device='cuda')][:,None].expand(-1,2)
            update(p['observed_bce_weight']*F.binary_cross_entropy_with_logits(scores,yy))
        for direction,q in enumerate(queries):
            order=rng.permutation(q)
            for start in range(0,len(order),p['retrieval_queries_per_batch']):
                query=order[start:start+p['retrieval_queries_per_batch']]
                if direction==0:
                    rows=rt.old_train[query,None]*384+np.arange(384)[None,:]
                    y=known[query]
                else:
                    rows=rt.old_train[None,:]*384+query[:,None]
                    y=known[:,query].T
                exposure.update(rows.astype('<i8').tobytes())
                scores=rt.forward(model,'OLD',rows.reshape(-1))[:,direction].reshape(rows.shape)
                loss=known_relation_loss(scores,torch.as_tensor(y,device='cuda'),p['retrieval_temperature'])
                # Match scheduled data/optimizer exposure in the loss-off control.
                update(p['retrieval_weight']*(sum(steps)/2/steps[direction])*loss*float(flags['known_retrieval']))
        if teacher is not None:
            for direction in [0,1]:
                pairs=teacher[f'direction_{direction}']
                for ix in np.array_split(rng.permutation(len(pairs)),4):
                    sample=pairs[ix]
                    scores=rt.forward(model,'OLD',sample[:,:2].astype(np.int64).reshape(-1))[:,direction].reshape(-1,2)
                    soft=torch.as_tensor(sample[:,2],dtype=torch.float32,device='cuda')
                    confidence=torch.as_tensor(sample[:,3],dtype=torch.float32,device='cuda')
                    loss=F.binary_cross_entropy_with_logits((scores[:,0]-scores[:,1])/2,soft,reduction='none')
                    update(rt.protocol['teacher_weight']*(loss*confidence).mean())
        val=rt.validation(model);eligible=rt.eligible(val)
        if eligible and val['selection']>best['selection']+1e-12:
            best,best_epoch=val,epoch;save(epoch)
        row={'epoch':epoch,'validation':val,'eligible':eligible,'best_epoch':best_epoch,
             'mean_loss':float(np.mean(losses)),'matched_exposure_sha256':exposure.hexdigest(),
             'seconds':time.monotonic()-started}
        history.append(row);write_json(folder/'HISTORY.json',history)
        write_json(STAGE/'STATUS.json',{'status':'TRAINING','variant':variant,'seed':seed,**row})
        print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
    result={'status':'COMPLETE','variant':variant,'seed':seed,'selected_epoch':best_epoch,'validation':best,
            'baseline_validation':rt.base_validation,'identity':identity,'checkpoint_sha256':sha256(folder/'BEST.pt'),
            'seconds':time.monotonic()-started,'trainable_parameters':sum(x.numel() for x in model.parameters()),
            'matched_exposure_sha256':[r['matched_exposure_sha256'] for r in history[1:]],'test_used_for_selection':False}
    write_json(folder/'RESULT.json',result)
    return result


def main():
    torch.set_num_threads(4)
    if (OUT/'TEST_RELEASE.json').exists():
        raise RuntimeError('Stage 2 must be frozen before current-round test release')
    STAGE.mkdir(exist_ok=True)
    rt=Runtime()
    p=json.loads(CONFIG.read_text())
    with np.load(OLD/'LABELS_AND_SCOPES.npz') as f:
        known=f['known_relationship'][rt.old_train]
    assert len(rt.old_train)==391 and known.sum()==471
    assert set(rt.old_drugs.entity_key.iloc[rt.old_train]).issubset(set(rt.data.loc[rt.data.split.eq('train'),'entity_key']))
    p={**p,'config_sha256':sha256(CONFIG),'stage1_selection_sha256':sha256(OUT/'SELECTION.json'),
       'known_training_pairs':int(known.sum()),'known_training_drug_queries':int(known.any(1).sum()),
       'known_training_target_queries':int(known.any(0).sum()),'status':'FROZEN'}
    protocol=STAGE/'PROTOCOL.json'
    if protocol.exists():assert json.loads(protocol.read_text())==p
    else:write_json(protocol,p)
    identity={'backbone_and_cache':rt.identity,'protocol_sha256':sha256(protocol),'source_sha256':source_identity()}
    results=[]
    for variant in p['variants']:
        for seed in p['seeds']:
            assert source_identity()==identity['source_sha256']
            results.append(train(rt,p,identity,known,variant,seed))
    for seed in p['seeds']:
        histories=[r['matched_exposure_sha256'] for r in results if r['seed']==seed]
        assert all(h==histories[0] for h in histories)
    scores={v:float(np.mean([r['validation']['selection'] for r in results if r['variant']==v])) for v in p['variants']}
    phase1=json.loads((OUT/'SELECTION.json').read_text())
    combined={**phase1['validation_scores'],**scores}
    winner=max(combined,key=combined.get)
    selection={'status':'FROZEN_BEFORE_TEST','winner':winner,'stage2_validation_scores':scores,
       'all_validation_scores':combined,'results':results,'identity':identity,
       'stage1_selection_sha256':sha256(OUT/'SELECTION.json'),'test_used_for_selection':False,
       'same_exposure_verified':True,'production_promotion':False}
    write_json(STAGE/'SELECTION.json',selection)
    write_json(STAGE/'STATUS.json',{'status':'TRAINING_COMPLETE_TEST_NOT_RELEASED','winner':winner,'runs':len(results)})
    print(json.dumps({'status':'COMPLETE','winner':winner,'validation_scores':scores}),flush=True)


if __name__=='__main__':
    main()

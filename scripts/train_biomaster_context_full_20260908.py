#!/usr/bin/env python3
"""Joint full-query training with real structural context and complete dev eval."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.context_pocket_full import FullContextPocketRanker, observed_structure_losses
from biomaster.context_full_training import (RealContextStructures, PermutationStream, query_cycle,
    validation_pairs, evaluate_matrices, capture_rng, restore_rng, file_sha)
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig
from biomaster.pocket_precision_features import PocketPrecisionBank
from biomaster.pocket_precision_training import cached_query_backward
from biomaster.molecular_controls import MolecularControlBank, MolecularControlInteraction, MolecularControlConfig
from biomaster.best_model_training import RollingStage
from scripts.prepare_biomaster_context_full_20260908 import write_json, OUT as DATA

PREP=ROOT/'outputs/biomaster_training_preparation_20260908/context_pocket'
CONFIG=ROOT/'configs/biomaster_context_full_20260908.json'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data',type=Path,default=DATA)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--engineering-updates',type=int,default=0,help='No formal metrics/selection; use only to validate execution')
    ap.add_argument('--stop-after-update',type=int,default=0,help='Orderly checkpoint boundary for resume verification')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config=json.loads(CONFIG.read_text());p=config['training'];v=config['validation'];cfg=PocketPrecisionConfig(**config['model'])
    manifest=json.loads((PREP/'MANIFEST.json').read_text());paths={k:Path(x) for k,x in manifest['paths'].items()}
    data=json.loads((args.data/'MANIFEST.json').read_text())
    if not args.engineering_updates and (data['status']!='FULL_DATA_READY' or data['counts']!={'train':9903,'validation':1223}):
        raise ValueError('full audited dataset required for formal training')
    sources=[ROOT/f for f in ['biomaster/context_pocket.py','biomaster/context_pocket_full.py',
        'biomaster/context_full_training.py','biomaster/pocket_precision.py','biomaster/structural_training.py',
        'biomaster/pocket_precision_features.py','biomaster/pocket_precision_training.py','biomaster/molecular_controls.py',
        'biomaster/unified_features.py','biomaster/unified_interaction.py','biomaster/best_model_training.py',
        'biomaster/ranking_audit.py','scripts/prepare_biomaster_context_full_20260908.py',
        'scripts/train_biomaster_context_full_20260908.py']]
    identities={str(path):file_sha(path) for path in sources+[CONFIG,PREP/'MANIFEST.json',args.data/'MANIFEST.json',
                 paths['parent_checkpoint'],paths['structural_checkpoint']]}
    for item in manifest['identities']+manifest['prepared_artifacts']:
        if file_sha(item['path'])!=item['sha256']:raise ValueError('prepared input changed: '+item['path'])
    if args.resume:
        if json.loads((out/'IDENTITY.json').read_text())!=identities:raise ValueError('resume source/data/config changed')
    else:
        if (out/'STATUS.json').exists():raise ValueError('used output requires --resume')
        write_json(out/'IDENTITY.json',identities)
        for path in sources:
            dest=out/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        shutil.copy2(CONFIG,out/'CONFIG.json');shutil.copy2(args.data/'MANIFEST.json',out/'DATA_MANIFEST.json')
    started=time.monotonic();halt=[];updates=0;elapsed_before=0.;history=[];selection=None;baseline=None;pending_validation=False
    torch.set_num_threads(4);torch.manual_seed(config['seed']);rng=np.random.default_rng(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGTERM,lambda *_:halt.append('SIGTERM'));signal.signal(signal.SIGINT,lambda *_:halt.append('SIGINT'))
    def status(state,**kw):
        write_json(out/'STATUS.json',dict(status=state,utc=datetime.now(timezone.utc).isoformat(),updates=updates,
            elapsed_seconds=elapsed_before+time.monotonic()-started,formal=not bool(args.engineering_updates),
            training_complete=state=='PLANNED_TRAINING_COMPLETE',promotion_eligible=False,**kw))
    status('LOADING')
    parent_payload=torch.load(paths['parent_checkpoint'],map_location='cpu',weights_only=False)
    structure_payload=torch.load(paths['structural_checkpoint'],map_location='cpu',weights_only=False)
    model=FullContextPocketRanker(parent_payload,structure_payload,cfg).cuda();model.set_phase('joint')
    optimizer=torch.optim.AdamW(model.parameter_groups(p['global_lr'],p['structural_lr'],p['new_lr']),weight_decay=p['weight_decay'])
    stage=RollingStage(paths['data'],paths['source'],config['cutoff']);cycle=query_cycle(stage.positive_queries,rng)
    cycle_size=len(cycle);total=config['cycles']*cycle_size
    target_updates=args.engineering_updates or total
    def lr_factor(step):
        if step<p['warmup_updates']:return (step+1)/p['warmup_updates']
        fraction=min(1.,(step-p['warmup_updates'])/max(1,total-p['warmup_updates']))
        return p['minimum_lr_fraction']+(1-p['minimum_lr_fraction'])*.5*(1+math.cos(math.pi*fraction))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    row_indices=np.load(PREP/'TRAINING_INDICES.npz')
    streams={name:PermutationStream(row_indices[name+'_rows'],rng) for name in ['old','general']}
    assays=pd.read_csv(PREP/'ASSAY_TRAIN.csv.gz');assay_groups={int(k):v for k,v in assays.groupby('assay_group')}
    assay_stream=PermutationStream(sorted(assay_groups),rng)
    train_records=[r for r in data['records'] if r['split']=='train']
    val_records=[r for r in data['records'] if r['split']=='validation']
    if args.engineering_updates and not val_records:val_records=train_records
    clusters={}
    for i,r in enumerate(train_records):clusters.setdefault(r['cluster'],[]).append(i)
    cluster_stream=PermutationStream(sorted(clusters),rng)
    structural_streams={k:PermutationStream(ids,rng) for k,ids in clusters.items()}
    structures={s:RealContextStructures(args.data,train_records,predicted=s=='experimental_p2rank')
                for s in ['experimental_native','experimental_p2rank']}
    val_structures={s:RealContextStructures(args.data,val_records,predicted=s=='experimental_p2rank') for s in structures}
    baseline_path=out/'BASELINE.json'
    if args.resume:
        saved=torch.load(out/'LATEST.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identities or saved['target_updates']!=target_updates:raise ValueError('resume protocol mismatch')
        if saved['status']=='STOPPED_STRUCTURAL_RETENTION_GATE':raise ValueError('retention failure requires a new reviewed run')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        updates=saved['updates'];cycle=saved['cycle'];history=saved['history'];selection=saved['selection']
        elapsed_before=saved['elapsed_seconds'];baseline=saved['baseline']
        pending_validation=saved.get('pending_validation',False)
        for k,x in streams.items():x.load_state_dict(saved['row_streams'][k])
        assay_stream.load_state_dict(saved['assay_stream']);cluster_stream.load_state_dict(saved['cluster_stream'])
        for k,x in structural_streams.items():x.load_state_dict(saved['structural_streams'][k])
        restore_rng(saved['rng'],rng)
    globals_bank=MolecularControlBank(paths['global_features'],paths['supplemental_features'],paths['source'],'drugclip_morgan')
    pockets=PocketPrecisionBank(paths['pocket_features'],paths['global_features'],paths['supplemental_features'])
    # Bank loading has no intended RNG effects; restore again to guarantee the
    # exact optimizer/sampling/dropout state at the next update.
    if args.resume:restore_rng(saved['rng'],rng)
    def save(state):
        payload=dict(model=model.state_dict(),config=cfg.to_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),identity=identities,updates=updates,target_updates=target_updates,
            cycle=cycle,history=history,selection=selection,baseline=baseline,rng=capture_rng(rng),
            row_streams={k:x.state_dict() for k,x in streams.items()},assay_stream=assay_stream.state_dict(),
            cluster_stream=cluster_stream.state_dict(),structural_streams={k:x.state_dict() for k,x in structural_streams.items()},
            elapsed_seconds=elapsed_before+time.monotonic()-started,status=state,pending_validation=pending_validation)
        tmp=out/'LATEST.tmp';torch.save(payload,tmp);tmp.replace(out/'LATEST.pt');status(state)
    def batches(d,t):
        b=pockets.batch(d,t);n,a,r=len(b['owner']),b['atom_tokens'].shape[1],b['residue_tokens'].shape[1]
        if len(d)>1 and n*cfg.heads*(a*r*r+r*a*a)>40_000_000:
            del b;mid=len(d)//2;yield from batches(d[:mid],t[:mid]);yield from batches(d[mid:],t[mid:])
        else:yield globals_bank.batch(d,t,'global'),b
    def forward(d,t,fp32=False):
        values=[]
        for g,b in batches(np.asarray(d),np.asarray(t)):
            with nullcontext() if fp32 else torch.autocast('cuda',dtype=torch.bfloat16):values.append(model(g,b))
        return torch.cat(values)
    @torch.no_grad()
    def rank_evaluate(tag,parent=False):
        model.eval();dq,tq,d,t=validation_pairs(stage);scores=np.full((*stage.known.shape,2),np.nan,np.float32)
        reference_model=None
        if parent:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                reference_model=MolecularControlInteraction(MolecularControlConfig(**parent_payload['config'])).cuda().eval()
            reference_model.load_state_dict(parent_payload['model']);reference_model.requires_grad_(False)
        for i in range(len(d)):
            gd=stage.old.drug_feature_index.to_numpy()[d[i:i+1]];gt=t[i:i+1]
            s=reference_model(globals_bank.batch(gd,gt,'global')) if parent else forward(gd,gt,True)
            scores[d[i],t[i]]=s.float().cpu().numpy()[0]
            if (i+1)%1000==0:status('FULL_FP32_VALIDATION',tag=tag,scored=i+1,pairs=len(d))
        reference=None if parent else {name:pd.read_csv(out/f'parent_{name}_queries.csv',dtype={'query_id':str}) for name in ['d2t','t2d']}
        result,frames,positive=evaluate_matrices(stage,scores,reference)
        for name in frames:
            frames[name].to_csv(out/f'{tag}_{name}_queries.csv',index=False)
            positive[name].to_csv(out/f'{tag}_{name}_positive_ranks.csv',index=False)
        np.savez_compressed(out/f'{tag}_scores.npz',scores=scores,d2t_queries=dq,t2d_queries=tq)
        write_json(out/f'{tag}_RESULT.json',result)
        del reference_model
        return result
    @torch.no_grad()
    def structural_evaluate(tag,indices=None,teacher=False):
        model.eval();rows=[];indices=range(len(val_records)) if indices is None else indices
        old_model=None
        if teacher:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                old_model=PocketPrecision(cfg).cuda().eval()
            old_model.load_state_dict(structure_payload['local_model']);old_model.requires_grad_(False)
        centers=(torch.arange(cfg.distance_bins,device='cuda')+.5)*cfg.distance_max/(cfg.distance_bins-1);centers[-1]=cfg.distance_max
        for source,dataset in val_structures.items():
            if teacher and source!='experimental_native':continue
            for i in indices:
                record=val_records[i];packed=dataset.real_batch([i])
                row=dict(system_id=record['system_id'],cluster=record['cluster'],source=source,
                    predicted_residue_coverage=record['covered_positive_residues']/max(1,record['positive_residues']),
                    contact_ap=None,distance_mae=None,site_hit_1=0,available=packed is not None)
                if packed is not None:
                    b,labels,g=packed
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        if teacher:_,aux=old_model(b,torch.zeros(1,cfg.parent_width,device='cuda'),True)
                        else:aux=model.structural_forward(b,g,source=source)
                    selected=torch.zeros(len(b['owner']),dtype=torch.bool,device='cuda')
                    if source=='experimental_native':selected[labels['native'][0]]=True
                    else:selected[:]=True
                    mask=aux['pair_mask'][selected];truth=labels['distance'][selected][mask].float()
                    score=aux['contact_logits'][selected][mask].float();positive=(truth<4.5).cpu().numpy()
                    pred=(aux['distance_logits'][selected][mask].float().softmax(-1)*centers).sum(-1)
                    row['contact_ap']=float(average_precision_score(positive,score.cpu().numpy())) if positive.any() else None
                    row['distance_mae']=float((pred-truth.clamp(max=cfg.distance_max)).abs().mean())
                    best=int(aux['pocket_gate'].argmax());row['site_hit_1']=int(((labels['distance'][best]<4.5)&aux['pair_mask'][best]).any())
                rows.append(row)
        frame=pd.DataFrame(rows);frame.to_csv(out/f'{tag}_structures.csv',index=False);result={}
        for source,part in frame.groupby('source'):
            result[source]=dict(complexes=len(part),available=int(part.available.sum()),
                contact_positive_complexes=int(part.contact_ap.notna().sum()),contact_ap=float(part.contact_ap.mean()),
                distance_mae=float(part.distance_mae.mean()),site_hit_1=float(part.site_hit_1.mean()),
                predicted_residue_coverage=float(part.predicted_residue_coverage.mean()))
        write_json(out/f'{tag}_structures.json',result);del old_model;return result
    def retention_ok(ref,now):
        a,b=ref['experimental_native'],now['experimental_native']
        return b['contact_ap']>=a['contact_ap']-v['maximum_native_ap_drop'] and b['distance_mae']<=a['distance_mae']+v['maximum_native_mae_increase']
    def validate_cycle():
        nonlocal selection,pending_validation
        save('EVALUATING_CYCLE')
        current=structural_evaluate(f'cycle_{updates//cycle_size}')
        if not retention_ok(baseline['initial'],current):return False
        ranking=rank_evaluate(f'cycle_{updates//cycle_size}')
        eligible=all(ranking[name]['macro_ap']>=baseline['parent'][name]['macro_ap'] and all(
            ranking[name][f'macro_recall_{k}']>=baseline['parent'][name][f'macro_recall_{k}']-v['maximum_recall_drop'] for k in [5,10])
            for name in ['d2t','t2d'])
        score=sum(ranking[name]['macro_ap'] for name in ['d2t','t2d'])/2
        pending_validation=False
        if eligible and (selection is None or score>selection['score']):
            selection=dict(update=updates,score=score,metrics=ranking,promotion_eligible=False)
            save('CHECKPOINT_SELECTED_ON_DEVELOPMENT');shutil.copy2(out/'LATEST.pt',out/'BEST.pt');write_json(out/'SELECTION.json',selection)
        save('JOINT_TRAINING')
        return True
    def observed(ids,weight):
        loss_value=0.
        for start in range(0,len(ids),p['candidate_microbatch']):
            sub=ids[start:start+p['candidate_microbatch']];s=forward(stage.d[sub],stage.t[sub])
            y=torch.tensor(stage.y[sub],device='cuda')[:,None].expand(-1,2)
            loss=F.binary_cross_entropy_with_logits(s.float(),y)*len(sub)/len(ids)
            (loss*weight).backward();loss_value+=float(loss.detach())
        return loss_value
    # One complex from each distinct validation cluster until the sentinel is
    # full, selected deterministically without using any metric or model score.
    seen=set();sentinel=[]
    for i,r in enumerate(val_records):
        if r['cluster'] not in seen:seen.add(r['cluster']);sentinel.append(i)
        if len(sentinel)==p['sentinel_complexes']:break
    outcome='FAILED';checkpoint_safe=True
    try:
        if baseline is not None:
            if not retention_ok(baseline['original'],baseline['initial']):
                raise RuntimeError('saved structural migration baseline failed retention gate')
            if not args.engineering_updates and 'initial_ranking' not in baseline:
                if updates:raise RuntimeError('training checkpoint lacks full initial validation')
                baseline=None
        if baseline is None:
            status('REAL_CONTEXT_STRUCTURAL_BASELINE')
            original=structural_evaluate('original_teacher',teacher=True)
            initial=structural_evaluate('initial_real_context')
            short=structural_evaluate('initial_sentinel',sentinel)
            baseline=dict(original=original,initial=initial,sentinel=short)
            if not retention_ok(original,initial):raise RuntimeError('structural migration retention gate failed')
            if not args.engineering_updates:
                baseline['parent']=rank_evaluate('parent',True)
                baseline['initial_ranking']=rank_evaluate('initial')
            write_json(baseline_path,baseline);save('JOINT_TRAINING')
        # A crash during evaluation must finish that evaluation before any new
        # training update, even if the last planned update was already reached.
        if pending_validation and not validate_cycle():
            raise RuntimeError('structural retention gate failed on resumed cycle validation')
        while updates<target_updates:
            if halt:outcome='STOPPED_BY_SIGNAL';break
            if args.stop_after_update and updates>=args.stop_after_update:outcome='STOPPED_AT_REQUESTED_BOUNDARY';break
            step_started=time.monotonic();checkpoint_safe=False;model.train();optimizer.zero_grad(set_to_none=True)
            head,q=map(int,cycle[updates%cycle_size]);matrix=stage.known if head==0 else stage.known.T;n=matrix.shape[1]
            d=np.full(n,stage.old.drug_feature_index.iloc[q]) if head==0 else stage.old.drug_feature_index.to_numpy()
            t=np.arange(n) if head==0 else np.full(n,q)
            status('JOINT_FULL_QUERY_BACKWARD',head=head,query=q,candidates=n,planned_updates=target_updates)
            rank=cached_query_backward(forward,d,t,torch.tensor(matrix[q]),head,model,batch_size=p['candidate_microbatch'],weight=p['query_weight'])
            old=streams['old'].take(p['old_observed_batch'],rng);general=streams['general'].take(p['general_observed_batch'],rng)
            lo=observed(old,p['old_bce_weight']);lg=observed(general,p['general_bce_weight'])
            group_id=int(assay_stream.take(1,rng)[0]);group=assay_groups[group_id];parts=[]
            for label in [1,0]:
                part=group[group.binary_label.eq(label)];parts.append(part.iloc[rng.choice(len(part),p['assay_rows_per_class'],replace=len(part)<p['assay_rows_per_class'])])
            pair=pd.concat(parts);s=forward(pair.drug_feature_index.to_numpy(),pair.target_feature_index.to_numpy())[:,1].float();k=p['assay_rows_per_class']
            assay=F.softplus(s[k:][None,:]-s[:k][:,None]).mean();(assay*p['assay_weight']).backward()
            structural_losses={source:[] for source in structures};replayed=[]
            for cluster in cluster_stream.take(p['structural_complexes_per_update'],rng):
                ri=int(structural_streams[str(cluster)].take(1,rng)[0]);replayed.append(train_records[ri]['system_id'])
                for source,dataset in structures.items():
                    packed=dataset.real_batch([ri])
                    if packed is None:continue
                    b,labels,g=packed
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        aux=model.structural_forward(b,g,source=source);losses=observed_structure_losses(aux,b,labels,cfg)
                        loss=(losses*torch.tensor([1.,1.,.2],device='cuda')).sum()
                    weight=p['native_structure_weight'] if source=='experimental_native' else p['predicted_structure_weight']
                    (loss*weight/p['structural_complexes_per_update']).backward();structural_losses[source].append(losses.detach().cpu().tolist())
            norms={}
            for group in optimizer.param_groups:
                grads=[x.grad for x in group['params'] if x.grad is not None]
                norms[group['group_name']]=float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();updates+=1
            if updates%cycle_size==0:
                cycle=query_cycle(stage.positive_queries,rng)
                pending_validation=not bool(args.engineering_updates)
            event=dict(update=updates,head=head,query=q,candidates=n,rank_loss=rank,old_bce=lo,general_bce=lg,
                assay_loss=float(assay.detach()),structure_losses=structural_losses,grad_norm=float(norm),gradient_norms=norms,
                old_rows=old.tolist(),general_rows=general.tolist(),assay_group=group_id,replayed=replayed,
                lr=[g['lr'] for g in optimizer.param_groups],seconds=time.monotonic()-step_started)
            history.append(event);checkpoint_safe=True
            with (out/'TRAINING_LOG.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
            if updates%p['checkpoint_every']==0 or updates==target_updates or args.stop_after_update==updates:save('JOINT_TRAINING')
            if updates%p['sentinel_every']==0:
                current=structural_evaluate(f'sentinel_{updates}',sentinel)
                if not retention_ok(baseline['sentinel'],current):outcome='STOPPED_STRUCTURAL_RETENTION_GATE';break
            if not args.engineering_updates and updates%cycle_size==0:
                if not validate_cycle():outcome='STOPPED_STRUCTURAL_RETENTION_GATE';break
        else:outcome='ENGINEERING_COMPLETE' if args.engineering_updates else 'PLANNED_TRAINING_COMPLETE'
        if args.engineering_updates:write_json(out/'ENGINEERING_AFTER.json',structural_evaluate('engineering_after'))
        write_json(out/'RESULT.json',dict(status=outcome,updates=updates,planned_updates=target_updates,selection=selection,
            training_complete=outcome=='PLANNED_TRAINING_COMPLETE',promotion_eligible=False,
            trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
            unique_old_rows=len({x for e in history for x in e['old_rows']}),
            unique_general_rows=len({x for e in history for x in e['general_rows']}),
            unique_structure_complexes=len({x for e in history for x in e['replayed']}),
            data_counts=data['counts'],step_seconds=[e['seconds'] for e in history]))
    except Exception as e:
        write_json(out/'ERROR.json',dict(type=type(e).__name__,message=str(e),updates=updates,checkpoint_safe=checkpoint_safe));raise
    finally:
        # Never overwrite a coherent checkpoint with parameters/optimizer/RNG
        # from an interrupted partial update. Resume replays from last commit.
        if checkpoint_safe:save(outcome)
        else:status('FAILED_PARTIAL_UPDATE_USE_LAST_CHECKPOINT')
        pockets.close()


if __name__=='__main__':main()

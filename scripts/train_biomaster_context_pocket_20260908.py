#!/usr/bin/env python3
"""Bounded real-data preparation run with query supervision and retention gates.

No automatic continuation, architecture selection, or future/external test use.
All optimizer steps are durable. A larger explicit budget can reuse the runner;
formal readiness remains separate from completing this engineering run.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.context_pocket import ContextPocketRanker, sourced_pockets, structural_consistency
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig
from biomaster.pocket_precision_features import PocketPrecisionBank
from biomaster.pocket_precision_training import cached_query_backward
from biomaster.molecular_controls import MolecularControlBank
from biomaster.structural_training import StructuralDataset, complex_losses
from biomaster.best_model_training import RollingStage
from biomaster.ranking_audit import risk_set_ranking
from scripts.audit_biomaster_spr_candidates_20260908 import identity
from scripts.prepare_biomaster_context_pocket_20260908 import OUT as PREP, CONFIG


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n'); tmp.replace(path)


class HaltRun(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-updates', type=int, required=True)
    parser.add_argument('--phase', choices=['adaptation', 'joint'], default='adaptation')
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--skip-ranking-probe', action='store_true')
    parser.add_argument('--full-structural-validation', action='store_true',
                        help='use all internal structural validation complexes instead of the fixed engineering subset')
    args = parser.parse_args()
    if args.max_updates <= 0 or args.max_seconds <= 0:
        parser.error('explicit positive update and time budgets required')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'STATUS.json').exists():
        raise ValueError('output already used; preserve previous run and choose a new directory')
    config = json.loads(CONFIG.read_text()); p = config['training']; guards = config['preflight']
    manifest = json.loads((PREP/'MANIFEST.json').read_text()); paths = {k: Path(v) for k,v in manifest['paths'].items()}
    for item in manifest['identities']+manifest['prepared_artifacts']:
        if identity(Path(item['path'])) != item:
            raise ValueError('prepared source changed: '+item['path'])
    torch.set_num_threads(4); torch.manual_seed(manifest['seed'])
    np.random.seed(manifest['seed']); rng = np.random.default_rng(manifest['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic(); interrupted = []
    def stop_signal(signum, frame):
        interrupted.append(signum)
    signal.signal(signal.SIGTERM, stop_signal); signal.signal(signal.SIGINT, stop_signal)
    def budget():
        if interrupted:
            raise HaltRun('STOPPED_BY_SIGNAL')
        if time.monotonic()-start > args.max_seconds:
            raise HaltRun('STOPPED_TIME_BUDGET')
    sources = [identity(ROOT/f) for f in ['biomaster/context_pocket.py', 'biomaster/pocket_precision.py',
               'biomaster/pocket_precision_features.py', 'biomaster/pocket_precision_training.py',
               'biomaster/structural_training.py', 'biomaster/ranking_audit.py',
               'scripts/train_biomaster_context_pocket_20260908.py']]
    run_identity = dict(manifest=identity(PREP/'MANIFEST.json'), config=identity(CONFIG), sources=sources)
    write_json(out/'IDENTITY.json', run_identity)
    for item in sources:
        source=Path(item['path']); dest=out/'source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
    shutil.copy2(CONFIG,out/'CONFIG_SNAPSHOT.json')
    shutil.copy2(PREP/'MANIFEST.json',out/'DATA_MANIFEST_SNAPSHOT.json')
    cfg = PocketPrecisionConfig(**config['model'])
    parent = torch.load(paths['parent_checkpoint'], map_location='cpu', weights_only=False)
    structural = torch.load(paths['structural_checkpoint'], map_location='cpu', weights_only=False)
    model = ContextPocketRanker(parent, structural, cfg).cuda()
    model.set_phase(args.phase)
    teacher = PocketPrecision(cfg).cuda().eval(); teacher.load_state_dict(structural['local_model'], strict=True)
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameter_groups(p['global_lr'], p['structural_lr'], p['new_lr']),
                                  weight_decay=p['weight_decay'])
    globals_bank = MolecularControlBank(paths['global_features'], paths['supplemental_features'], paths['source'], 'drugclip_morgan')
    pockets = PocketPrecisionBank(paths['pocket_features'], paths['global_features'], paths['supplemental_features'])
    stage = RollingStage(paths['data'], paths['source'], manifest['cutoff'])
    cycle = pd.read_csv(PREP/'QUERY_CYCLE.csv')
    row_indices = np.load(PREP/'TRAINING_INDICES.npz')
    assays = pd.read_csv(PREP/'ASSAY_TRAIN.csv.gz')
    assay_groups = {int(k): v for k,v in assays.groupby('assay_group')}
    structural_manifest = json.loads((paths['structural_data']/'MANIFEST.json').read_text())
    train_records = [r for r in structural_manifest['records'] if r['split']=='train']
    cluster_indices = {}
    for i,r in enumerate(train_records):
        cluster_indices.setdefault(r['cluster'], []).append(i)
    replay = StructuralDataset(paths['structural_data'], train_records)
    probe_records = json.loads((PREP/'STRUCTURAL_PROBE.json').read_text())
    if args.full_structural_validation:
        probe_records=[r for r in structural_manifest['records'] if r['split']=='validation']
    probe = StructuralDataset(paths['structural_data'], probe_records)
    updates, history, coverage = 0, [], [[], []]
    last_structural_check = start
    progress = {'phase': args.phase, 'model_parameters': sum(x.numel() for x in model.parameters()),
                'trainable_parameters': sum(x.numel() for x in model.parameters() if x.requires_grad)}
    def status(state, **more):
        write_json(out/'STATUS.json', dict(status=state, utc=datetime.now(timezone.utc).isoformat(),
                   updates=updates, elapsed_seconds=time.monotonic()-start, automatic_continuation=False,
                   training_complete=False, promotion_eligible=False, **progress, **more))
    def save(state):
        payload = dict(status=state, model=model.state_dict(), config=cfg.to_dict(), identity=run_identity,
                       optimizer=optimizer.state_dict(), updates=updates, phase=args.phase, history=history,
                       coverage=coverage, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state())
        tmp=out/'LATEST.tmp'; torch.save(payload, tmp); tmp.replace(out/'LATEST.pt')
        status(state)
    def batches(d, t):
        # Split oversized ragged batches without dropping candidates or pockets.
        b = pockets.batch(d, t)
        n,a,r = len(b['owner']), b['atom_tokens'].shape[1], b['residue_tokens'].shape[1]
        if len(d)>1 and n*cfg.heads*(a*r*r+r*a*a)>40_000_000:
            del b
            mid = len(d)//2
            yield from batches(d[:mid],t[:mid]); yield from batches(d[mid:],t[mid:])
        else:
            yield globals_bank.batch(d, t, 'global'), b
    def forward(d,t,fp32=False):
        results=[]
        for g,b in batches(np.asarray(d),np.asarray(t)):
            budget()
            with nullcontext() if fp32 else torch.autocast('cuda', dtype=torch.bfloat16):
                results.append(model(g,b))
        return torch.cat(results)
    def bce_rows(ids, weight):
        losses=[]
        for offset in range(0,len(ids),p['candidate_microbatch']):
            take=ids[offset:offset+p['candidate_microbatch']]
            s=forward(stage.d[take],stage.t[take])
            y=torch.tensor(stage.y[take],device='cuda')[:,None].expand(-1,2)
            loss=F.binary_cross_entropy_with_logits(s.float(),y)*len(take)/len(ids)
            (loss*weight).backward(); losses.append(float(loss.detach()))
        return sum(losses)
    @torch.no_grad()
    def structural_metrics(tag, original=False):
        model.eval(); rows=[]
        centers=(torch.arange(cfg.distance_bins,device='cuda')+.5)*cfg.distance_max/(cfg.distance_bins-1)
        centers[-1]=cfg.distance_max
        for i,record in enumerate(probe_records):
            budget(); b,labels=probe.batch([i])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if original:
                    _,aux=teacher(b,torch.zeros(1,cfg.parent_width,device='cuda'),True)
                else:
                    aux=model.structural_forward(b)
            native=int(labels['native'][0]); mask=aux['pair_mask'][native]
            truth=labels['distance'][native][mask].float()
            score=aux['contact_logits'][native][mask].float()
            pred=(aux['distance_logits'][native][mask].float().softmax(-1)*centers).sum(-1)
            rows.append(dict(system_id=record['system_id'],cluster=record['cluster'],
                        native_contact_ap=float(average_precision_score((truth<4.5).cpu().numpy(),score.cpu().numpy())),
                        native_distance_mae=float((pred-truth.clamp(max=cfg.distance_max)).abs().mean())))
        frame=pd.DataFrame(rows); frame.to_csv(out/(tag+'.csv'),index=False)
        return frame[['native_contact_ap','native_distance_mae']].mean().to_dict()
    @torch.no_grad()
    def ranking_probe(tag):
        # Two fixed development queries; never presented as full benchmark AP.
        model.eval(); result={}
        for head in [0,1]:
            matrix=stage.val_known if head==0 else stage.val_known.T
            risk=stage.risk if head==0 else stage.risk.T
            q=int(np.flatnonzero(matrix.any(1))[0]); n=matrix.shape[1]
            d=np.full(n,stage.old.drug_feature_index.iloc[q]) if head==0 else stage.old.drug_feature_index.to_numpy()
            t=np.arange(n) if head==0 else np.full(n,q)
            # FP32 and fixed batch=1 for comparable deterministic rankings.
            scores=torch.cat([forward(d[i:i+1],t[i:i+1],True)[:,head] for i in range(n)]).cpu().numpy()
            metrics,query_rows,positive_rows=risk_set_ranking(matrix[q:q+1],scores[None],risk[q:q+1],
                                                             query_ids=[q],candidate_ids=np.arange(n))
            positive_rows.to_csv(out/f'{tag}_head{head}_positive_ranks.csv',index=False)
            np.savez_compressed(out/f'{tag}_head{head}.npz',scores=scores,labels=matrix[q],risk=risk[q],query=q)
            result['d2t' if head==0 else 't2d']=dict(query=q,metrics=metrics)
        return result
    def retention_ok(reference, current):
        return (current['native_contact_ap'] >= reference['native_contact_ap']-guards['maximum_absolute_native_contact_ap_drop']
                and current['native_distance_mae'] <= reference['native_distance_mae']+guards['maximum_native_distance_mae_increase_A'])
    status('PREFLIGHT_STRUCTURAL_BASELINE')
    initial, original, ranking_before, final_metrics = None, None, None, None
    outcome='PREPARATION_RUN_NOT_FINISHED'
    try:
        original=structural_metrics('original_structural_teacher',True)
        initial=structural_metrics('migrated_structural_before')
        write_json(out/'BASELINE.json',dict(original=original,migrated=initial,probe=len(probe_records)))
        if not retention_ok(original,initial):
            raise HaltRun('STOPPED_STRUCTURAL_MIGRATION_GATE')
        if not args.skip_ranking_probe:
            status('PREFLIGHT_FP32_RANKING_PROBE')
            ranking_before=ranking_probe('ranking_before')
            write_json(out/'RANKING_BEFORE.json',ranking_before)
        save('PREFLIGHT_TRAINING')
        for step in range(args.max_updates):
            budget(); model.train(); optimizer.zero_grad(set_to_none=True)
            entry=cycle.iloc[step%len(cycle)]; head,q=int(entry['head']),int(entry['query'])
            matrix=stage.known if head==0 else stage.known.T; n=matrix.shape[1]
            d=np.full(n,stage.old.drug_feature_index.iloc[q]) if head==0 else stage.old.drug_feature_index.to_numpy()
            t=np.arange(n) if head==0 else np.full(n,q)
            status('PREFLIGHT_FULL_QUERY_BACKWARD',query_head=head,query=q,candidates=n)
            loss_rank=cached_query_backward(forward,d,t,torch.tensor(matrix[q]),head,model,
                                            batch_size=p['candidate_microbatch'],weight=p['query_loss_weight'])
            old=rng.choice(row_indices['old_rows'],p['old_observed_batch'],replace=False)
            general=rng.choice(row_indices['general_rows'],p['general_observed_batch'],replace=False)
            loss_old=bce_rows(old,p['old_observed_bce_weight'])
            loss_general=bce_rows(general,p['general_observed_bce_weight'])
            group_id=int(rng.choice(sorted(assay_groups))); group=assay_groups[group_id]
            chosen=[]
            for label in [1,0]:
                subset=group[group.binary_label.eq(label)]
                chosen.append(subset.iloc[rng.choice(len(subset),p['assay_rows_per_class'],replace=len(subset)<p['assay_rows_per_class'])])
            pairs=pd.concat(chosen); s=forward(pairs.drug_feature_index.to_numpy(),pairs.target_feature_index.to_numpy())[:,1].float()
            k=p['assay_rows_per_class']; assay_loss=F.softplus(s[k:][None,:]-s[:k][:,None]).mean()
            (assay_loss*p['same_assay_pair_weight']).backward()
            cluster=str(rng.choice(sorted(cluster_indices))); ri=int(rng.choice(cluster_indices[cluster]))
            sb,labels=replay.batch([ri])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                aux=model.structural_forward(sb)
                structural_losses=complex_losses(aux,sb,labels,cfg)
                replay_loss=(structural_losses*torch.tensor([1.,1.,.2],device='cuda')).sum()
            (replay_loss*p['structural_replay_weight']).backward()
            # Preserve structural predictions on actual predicted-pocket inputs,
            # while labeling this explicitly as teacher consistency, not truth.
            consistency=[]
            for g,b in batches(stage.d[old[:2]],stage.t[old[:2]]):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,student_aux=model(g,b,True)
                    with torch.no_grad():
                        _,teacher_aux=teacher(sourced_pockets(b,'predicted'),
                                              torch.zeros(len(g['drug_global']),cfg.parent_width,device='cuda'),True)
                    loss=structural_consistency(student_aux,teacher_aux)
                (loss*p['predicted_pocket_teacher_weight']/2*len(g['drug_global'])).backward()
                consistency.append(float(loss.detach())*len(g['drug_global'])/2)
            norms={}
            for group in optimizer.param_groups:
                grads=[param.grad for param in group['params'] if param.grad is not None]
                norms[group['group_name']]=float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); updates+=1; coverage[head].append(q)
            event=dict(update=updates,head=head,query=q,candidates=n,loss_rank=loss_rank,
                       loss_old_observed=loss_old,loss_general_observed=loss_general,
                       loss_same_assay=float(assay_loss.detach()),loss_structural_replay=float(replay_loss.detach()),
                       loss_distance=float(structural_losses[0].detach()),loss_contact=float(structural_losses[1].detach()),
                       loss_site=float(structural_losses[2].detach()),loss_teacher_consistency=sum(consistency),
                       gradient_norm=float(grad_norm),gradient_norm_by_group=norms,
                       old_observed_rows=old.tolist(),general_observed_rows=general.tolist(),assay_group=group_id,
                       replay_system=train_records[ri]['system_id'],replay_cluster=cluster,
                       elapsed_seconds=time.monotonic()-start)
            history.append(event)
            with (out/'TRAINING_LOG.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            save('PREFLIGHT_TRAINING')
            print(json.dumps(event),flush=True)
            if updates%p['maximum_updates_between_structural_checks']==0 or time.monotonic()-last_structural_check>60*p['maximum_minutes_between_structural_checks']:
                final_metrics=structural_metrics(f'retention_update_{updates}')
                last_structural_check=time.monotonic()
                if not retention_ok(initial,final_metrics):raise HaltRun('STOPPED_STRUCTURAL_RETENTION_GATE')
        final_metrics=structural_metrics('migrated_structural_after')
        if not retention_ok(initial,final_metrics):raise HaltRun('STOPPED_STRUCTURAL_RETENTION_GATE')
        ranking_after=None
        if not args.skip_ranking_probe:
            status('PREFLIGHT_FP32_RANKING_PROBE_AFTER')
            ranking_after=ranking_probe('ranking_after'); write_json(out/'RANKING_AFTER.json',ranking_after)
        outcome='BOUNDED_PREFLIGHT_COMPLETE'
        result=dict(status=outcome,updates=updates,phase=args.phase,elapsed_seconds=time.monotonic()-start,
                    original_structural=original,migrated_before=initial,migrated_after=final_metrics,
                    retention_gate_pass=True,ranking_probe_before=ranking_before,ranking_probe_after=ranking_after,
                    unique_queries_seen=[len(set(q)) for q in coverage],history=history,
                    model_parameters=progress['model_parameters'],trainable_parameters=progress['trainable_parameters'],
                    full_training_complete=False,full_ranking_benchmark_complete=False,
                    structural_validation_complexes=len(probe_records),
                    full_structural_retention_measured=args.full_structural_validation,
                    long_training_ready=False,promotion_eligible=False,
                    limitation='A bounded optimization check, not a full ranking benchmark; any ranking probe has one query per direction. Holo/predicted domain-transfer work remains.')
        write_json(out/'RESULT.json',result)
    except HaltRun as e:
        outcome=str(e)
        write_json(out/'RESULT.json',dict(status=outcome,updates=updates,original_structural=original,
                   migrated_before=initial,migrated_after=final_metrics,long_training_ready=False,promotion_eligible=False))
    except Exception as e:
        outcome='PREFLIGHT_FAILED'
        write_json(out/'ERROR.json',dict(type=type(e).__name__,message=str(e)))
        raise
    finally:
        save(outcome); pockets.close()
        print(json.dumps(dict(status=outcome,updates=updates,output=str(out))),flush=True)


if __name__=='__main__':
    main()

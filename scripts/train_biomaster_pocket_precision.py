#!/usr/bin/env python3
"""Development-only full-candidate training with a protected parent backbone.

Default training requires an audited structural-pretraining checkpoint.
The explicitly labelled relation-only control is an ablation, not a release.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.best_model_training import RollingStage
from biomaster.molecular_controls import MolecularControlBank
from biomaster.pocket_precision import PocketPrecisionConfig
from biomaster.pocket_precision_features import PocketPrecisionBank
from biomaster.pocket_precision_training import ProtectedPocketRanker, cached_query_backward
from biomaster.ranking_audit import risk_set_ranking
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json
from prepare_biomaster_pocket_precision import BASE, SUPPLEMENT, SOURCE, OUTPUT


@torch.no_grad()
def evaluate(forward, model, stage, microbatch):
    model.eval(); result = {}
    for head, direction in enumerate(['d2t', 't2d']):
        known = stage.val_known if head == 0 else stage.val_known.T
        risk = stage.risk if head == 0 else stage.risk.T
        ids = np.flatnonzero(known.any(1)); candidates = known.shape[1]
        if head == 0:
            d = np.repeat(stage.old.drug_feature_index.to_numpy()[ids], candidates)
            t = np.tile(np.arange(candidates), len(ids))
        else:
            d = np.tile(stage.old.drug_feature_index.to_numpy(), len(ids))
            t = np.repeat(ids, candidates)
        scores = np.concatenate([forward(d[s:s+microbatch], t[s:s+microbatch]).float().cpu().numpy()[:, head]
                                 for s in range(0, len(d), microbatch)]).reshape(len(ids), candidates)
        if not np.isfinite(scores).all():
            raise FloatingPointError('nonfinite validation scores')
        metrics, _, _ = risk_set_ranking(known[ids], scores, risk[ids], query_ids=ids, candidate_ids=np.arange(candidates))
        result[direction] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', type=int, choices=[2018, 2020], required=True)
    parser.add_argument('--seed', type=int, choices=[20260921, 20260922, 20260923], required=True)
    parser.add_argument('--structural-checkpoint', type=Path)
    parser.add_argument('--relation-only-control', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-updates', type=int, default=0, help='engineering interruption; never marks training complete')
    args = parser.parse_args()
    if bool(args.structural_checkpoint) == args.relation_only_control:
        parser.error('provide structural checkpoint OR explicitly request a relation-only control')
    config_path = ROOT / 'configs/biomaster_pocket_precision_20260906.json'
    config = json.loads(config_path.read_text()); p = config['training']
    cfg = PocketPrecisionConfig(**config['model'])
    data = ROOT / 'outputs/biomaster_best_model_20260906/data'
    stage = RollingStage(data, SOURCE, args.cutoff)
    selection = json.loads((data.parent / 'GLOBAL_PARENT_SELECTION.json').read_text())
    parent = next(r for r in selection['parents'] if r['cutoff'] == args.cutoff and r['seed'] == args.seed)
    if file_identity(parent['checkpoint']['path']) != parent['checkpoint']:
        raise ValueError('parent identity changed')
    structural = None
    if args.structural_checkpoint:
        structural = torch.load(args.structural_checkpoint, map_location='cpu', weights_only=False)
        required = {'status': 'STRUCTURAL_PRETRAINING_COMPLETE', 'project_overlap_audit_pass': True,
                    'experimental_complexes_only': True, 'official_train_split_only': True}
        if any(structural.get(k) != v for k, v in required.items()) or structural['max_release_year'] > args.cutoff:
            raise ValueError('structural checkpoint lacks required data/split/time audit')
        if structural['config'] != cfg.to_dict():
            raise ValueError('structural architecture mismatch')
    mode = 'relation_only_control' if args.relation_only_control else 'structurally_pretrained'
    out = args.output or OUTPUT.parent / 'training' / mode / f'cutoff_{args.cutoff}' / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(config=file_identity(config_path), parent=parent['checkpoint'],
                    features=file_identity(OUTPUT / 'MANIFEST.json'), train=file_identity(data / f'roll_{args.cutoff}/TRAIN.csv.gz'),
                    validation=file_identity(data / f'roll_{args.cutoff}/VALIDATION.csv.gz'),
                    risk=file_identity(data / f'roll_{args.cutoff}/RISK.npy'),
                    structural=file_identity(args.structural_checkpoint) if args.structural_checkpoint else None,
                    sources=[file_identity(ROOT / x) for x in ['biomaster/pocket_precision.py', 'biomaster/pocket_precision_features.py',
                              'biomaster/pocket_precision_training.py', 'scripts/train_biomaster_pocket_precision.py']])
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ProtectedPocketRanker(torch.load(parent['checkpoint']['path'], map_location='cpu', weights_only=False), cfg).cuda()
    if structural:
        model.local.load_state_dict(structural['local_model'], strict=True)
    globals = MolecularControlBank(BASE, SUPPLEMENT, SOURCE, 'drugclip_morgan')
    pockets = PocketPrecisionBank(OUTPUT, BASE, SUPPLEMENT)
    def forward(d, t):
        g, b = globals.batch(d, t, 'global'), pockets.batch(d, t)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(g, b)
    optimizer = torch.optim.AdamW(model.local.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    steps_per_epoch = math.ceil(len(stage.d) / p['effective_batch_size'])
    total_steps = steps_per_epoch * p['max_epochs']
    def schedule(update):
        if update < p['warmup_steps']:
            return (update + 1) / p['warmup_steps']
        fraction = (update - p['warmup_steps']) / max(1, total_steps - p['warmup_steps'])
        return 0.5 * (1 + math.cos(math.pi * min(fraction, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(args.seed)
    state = dict(epoch=0, position=0, updates=0, order=None, best=-float('inf'), stale=0, history=[])
    latest = out / 'LATEST.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('resume source/data/config identity changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        state = saved['progress']; rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        if saved['status'] == 'COMPLETE':
            print('already complete'); return
    def save(status):
        payload = dict(status=status, mode=mode, identity=identity, config=cfg.to_dict(), model=model.state_dict(),
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), progress=state,
                       numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        tmp = latest.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(latest)
        write_json(out / 'STATUS.json', dict(status=status, mode=mode, epoch=state['epoch'], updates=state['updates'],
                                           best=None if not math.isfinite(state['best']) else state['best'],
                                           training_rows=len(stage.d), source_identity=identity,
                                           promotion_eligible=False))
    started = time.monotonic()
    while state['epoch'] < p['max_epochs'] and state['stale'] < p['patience']:
        model.train()
        if state['order'] is None:
            state['order'] = rng.permutation(len(stage.d)); state['position'] = 0
        order = state['order']
        while state['position'] < len(order):
            ids = order[state['position']:state['position']+p['effective_batch_size']]
            optimizer.zero_grad(set_to_none=True); total_loss = 0.0
            for start in range(0, len(ids), p['microbatch_size']):
                take = ids[start:start+p['microbatch_size']]
                scores = forward(stage.d[take], stage.t[take])
                labels = torch.tensor(stage.y[take], device='cuda')[:, None].expand(-1, 2)
                loss = F.binary_cross_entropy_with_logits(scores.float(), labels) * len(take) / len(ids)
                loss.backward(); total_loss += float(loss.detach())
            if (state['updates'] + 1) % p['retrieval_every_updates'] == 0:
                for head in [0, 1]:
                    matrix = stage.known if head == 0 else stage.known.T
                    query = rng.choice(stage.positive_queries[head]); n = matrix.shape[1]
                    if head == 0:
                        d = np.full(n, stage.old.drug_feature_index.iloc[query]); t = np.arange(n)
                    else:
                        d = stage.old.drug_feature_index.to_numpy(); t = np.full(n, query)
                    total_loss += cached_query_backward(forward, d, t, torch.tensor(matrix[query]), head, model,
                                                        batch_size=p['microbatch_size'], weight=p['retrieval_weight'] / 2)
            torch.nn.utils.clip_grad_norm_(model.local.parameters(), p['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            state['position'] += len(ids); state['updates'] += 1
            if state['updates'] % 20 == 0:
                print(json.dumps(dict(epoch=state['epoch'], updates=state['updates'], loss=total_loss, seconds=time.monotonic()-started)), flush=True)
            if state['updates'] % 200 == 0:
                save('TRAINING')
            if args.max_updates and state['updates'] >= args.max_updates:
                save('INTERRUPTED_ENGINEERING_LIMIT'); return
        metrics = evaluate(forward, model, stage, p['microbatch_size'])
        score = sum(weight * metrics[d][key] for d in ['d2t', 't2d'] for weight, key in [(0.35, 'macro_ap'), (0.15, 'macro_recall_20')])
        state['history'].append(dict(epoch=state['epoch']+1, metrics=metrics, composite=score))
        if score > state['best']:
            state['best'] = score; state['stale'] = 0
            torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), identity=identity, metrics=metrics,
                            status='DEVELOPMENT_CHECKPOINT_NOT_SELECTED', mode=mode), out / 'BEST.pt')
        else:
            state['stale'] += 1
        state['epoch'] += 1; state['position'] = 0; state['order'] = None
        write_json(out / 'HISTORY.json', state['history']); save('TRAINING')
    save('COMPLETE'); pockets.close()


if __name__ == '__main__':
    main()

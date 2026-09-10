#!/usr/bin/env python3
"""Train full-size interaction states on experimental distances and contacts."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig
from biomaster.structural_training import StructuralDataset, complex_losses
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json

DATA = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_data/training_2020'
OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_pretraining/cutoff_2020/seed_20260921'


@torch.inference_mode()
def evaluate(model, dataset, microbatch):
    model.eval(); rows = []
    cfg = model.cfg
    centers = (torch.arange(cfg.distance_bins, device='cuda') + 0.5) * cfg.distance_max / (cfg.distance_bins - 1)
    centers[-1] = cfg.distance_max
    for start in range(0, len(dataset), microbatch):
        ids = list(range(start, min(start + microbatch, len(dataset))))
        batch, labels = dataset.batch(ids)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, aux = model(batch, torch.zeros(len(ids), cfg.parent_width, device='cuda'), return_aux=True)
        for owner, index in enumerate(ids):
            take = batch['owner'] == owner; mask = aux['pair_mask'][take]
            distance = labels['distance'][take][mask]
            truth = (distance < 4.5).cpu().numpy()
            score = aux['contact_logits'][take][mask].float().cpu().numpy()
            ap = float(average_precision_score(truth, score))
            logits = aux['distance_logits'][take][mask].float()
            predicted = (logits.softmax(-1) * centers).sum(-1)
            mae = float((predicted - distance.clamp_max(cfg.distance_max)).abs().mean())
            selected = torch.nonzero(take, as_tuple=True)[0]
            winner = selected[aux['pocket_gate'][take].argmax()]
            rows.append(dict(system_id=dataset.records[index]['system_id'], contact_ap=ap,
                             contact_prevalence=float(truth.mean()), distance_mae_capped32_A=mae,
                             pocket_candidates=int(take.sum()), correct_native_pocket=bool(winner == labels['native'][owner])))
    frame = pd.DataFrame(rows)
    eligible = frame[frame.pocket_candidates.gt(1)]
    result = dict(complexes=len(frame), macro_contact_ap=float(frame.contact_ap.mean()),
                  macro_contact_prevalence=float(frame.contact_prevalence.mean()),
                  macro_distance_mae_capped32_A=float(frame.distance_mae_capped32_A.mean()),
                  pocket_assignment_accuracy=float(eligible.correct_native_pocket.mean()) if len(eligible) else None,
                  multi_region_complexes=len(eligible), scope='internal cluster-held-out structural contacts, not DTI ranking')
    return result, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-updates', type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((DATA / 'MANIFEST.json').read_text())
    if manifest['status'] != 'STRUCTURAL_DATA_READY' or not manifest['project_overlap_audit_pass']:
        raise ValueError('audited structural dataset required')
    config_file = ROOT / 'configs/biomaster_structural_pretraining_20260906.json'
    protocol = json.loads(config_file.read_text())
    cfg = PocketPrecisionConfig(**json.loads((ROOT / protocol['model_config']).read_text())['model'])
    if manifest['max_release_year'] > protocol['cutoff']:
        raise ValueError('future structural label')
    train = [r for r in manifest['records'] if r['split'] == 'train']
    validation = [r for r in manifest['records'] if r['split'] == 'validation']
    if not train or not validation or set(r['cluster'] for r in train) & set(r['cluster'] for r in validation):
        raise ValueError('nonempty disjoint-cluster split required')
    identity = dict(protocol=file_identity(config_file), data=file_identity(DATA / 'MANIFEST.json'),
                    model_config=file_identity(ROOT / protocol['model_config']),
                    sources=[file_identity(ROOT / p) for p in ['biomaster/pocket_precision.py', 'biomaster/structural_training.py',
                              'scripts/pretrain_biomaster_structural_interaction.py']])
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.manual_seed(protocol['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    model = PocketPrecision(cfg).cuda()
    training_data = StructuralDataset(DATA, train); validation_data = StructuralDataset(DATA, validation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    steps = math.ceil(len(train) / protocol['effective_batch_size']) * protocol['max_epochs']
    def schedule(i):
        if i < protocol['warmup_updates']:
            return (i + 1) / protocol['warmup_updates']
        return 0.5 * (1 + math.cos(math.pi * min(1., (i - protocol['warmup_updates']) / max(1, steps - protocol['warmup_updates']))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(protocol['seed'])
    state = dict(epoch=0, position=0, order=None, updates=0, best=-1., stale=0, history=[])
    latest = OUT / 'LATEST.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('pretraining resume identity changed')
        model.load_state_dict(saved['local_model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); state = saved['progress']
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
        if saved['status'] == 'STRUCTURAL_PRETRAINING_COMPLETE':
            return
    weights = torch.tensor(protocol['loss_weights'], device='cuda')
    def save(status):
        values = dict(status=status, identity=identity, config=cfg.to_dict(), local_model=model.state_dict(),
                      optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), progress=state,
                      numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                      project_overlap_audit_pass=True, experimental_complexes_only=True, official_train_split_only=True,
                      max_release_year=manifest['max_release_year'])
        tmp = latest.with_suffix('.tmp'); torch.save(values, tmp); tmp.replace(latest)
        write_json(OUT / 'STATUS.json', dict(status=status, epoch=state['epoch'], updates=state['updates'],
                   train_complexes=len(train), validation_complexes=len(validation), best_macro_contact_ap=state['best'],
                   identity=identity, downstream_dti_ranking_measured=False))
    if not state['history']:
        metrics, frame = evaluate(model, validation_data, protocol['microbatch_size'])
        frame.to_csv(OUT / 'INITIAL_VALIDATION.csv', index=False)
        state['history'].append(dict(epoch=0, updates=0, metrics=metrics))
        write_json(OUT / 'HISTORY.json', state['history']); save('STRUCTURAL_PRETRAINING_RUNNING')
        print(json.dumps(dict(stage='initial_validation', **metrics)), flush=True)
    started = time.monotonic()
    while state['epoch'] < protocol['max_epochs'] and state['stale'] < protocol['patience']:
        model.train()
        if state['order'] is None:
            state['order'] = rng.permutation(len(train)); state['position'] = 0
        while state['position'] < len(train):
            ids = state['order'][state['position']:state['position'] + protocol['effective_batch_size']]
            optimizer.zero_grad(set_to_none=True); total = np.zeros(3)
            for begin in range(0, len(ids), protocol['microbatch_size']):
                subset = ids[begin:begin+protocol['microbatch_size']]
                batch, labels = training_data.batch(subset)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, aux = model(batch, torch.zeros(len(subset), cfg.parent_width, device='cuda'), return_aux=True)
                    losses = complex_losses(aux, batch, labels, cfg)
                    loss = (losses * weights).sum() * len(subset) / len(ids)
                loss.backward(); total += losses.detach().float().cpu().numpy() * len(subset) / len(ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); state['updates'] += 1; state['position'] += len(ids)
            if state['updates'] % 20 == 0:
                event = dict(stage='training', epoch=state['epoch'], updates=state['updates'],
                             loss_distance=total[0], loss_contact=total[1], loss_pocket=total[2], seconds=time.monotonic()-started)
                print(json.dumps(event), flush=True); save('STRUCTURAL_PRETRAINING_RUNNING')
            if state['updates'] % protocol['validation_every_updates'] == 0 or state['position'] >= len(train):
                metrics, frame = evaluate(model, validation_data, protocol['microbatch_size'])
                state['history'].append(dict(epoch=state['epoch'], updates=state['updates'], metrics=metrics))
                if metrics['macro_contact_ap'] > state['best']:
                    state['best'] = metrics['macro_contact_ap']; state['stale'] = 0
                    torch.save(dict(local_model=model.state_dict(), config=cfg.to_dict(), identity=identity,
                                    status='STRUCTURAL_PRETRAINING_BEST_INTERIM', metrics=metrics,
                                    project_overlap_audit_pass=True, experimental_complexes_only=True,
                                    official_train_split_only=True, max_release_year=manifest['max_release_year']), OUT / 'BEST.pt')
                    frame.to_csv(OUT / 'BEST_VALIDATION.csv', index=False)
                elif state['position'] >= len(train):
                    state['stale'] += 1
                write_json(OUT / 'HISTORY.json', state['history']); save('STRUCTURAL_PRETRAINING_RUNNING')
                print(json.dumps(dict(stage='validation', updates=state['updates'], **metrics)), flush=True)
                model.train()
            if args.max_updates and state['updates'] >= args.max_updates:
                save('INTERRUPTED_ENGINEERING_LIMIT'); return
        state['epoch'] += 1; state['order'] = None; state['position'] = 0
    best = torch.load(OUT / 'BEST.pt', map_location='cpu', weights_only=False)
    best['status'] = 'STRUCTURAL_PRETRAINING_COMPLETE'
    torch.save(best, OUT / 'STRUCTURAL_PRETRAINED.pt')
    save('STRUCTURAL_PRETRAINING_COMPLETE')


if __name__ == '__main__':
    main()

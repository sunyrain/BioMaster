#!/usr/bin/env python3
"""Train complete-coverage R1 controls, then evaluate frozen checkpoints.

Uses the historical V3 development split and legacy binary observations. This
is not a full V4/DrugCLIP or endpoint-aware training implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.odti_v4 import GlobalConfigV4, GlobalPairV4, SignedEvidenceV4
from biomaster.selectivity_training_v3 import QueryBatchStream, observed_metrics, support_summary_features
from biomaster.odti_support_v3 import drug_macro_pairwise_loss
from biomaster.odti_v3_panel_metrics import positive_retrieval_metrics
from build_biomaster_odti_v4_features import sha256, write_json

OUT = ROOT / 'outputs/biomaster_odti_v4_20260905'


def save_model(path, value):
    temp = path.with_suffix('.pending.pt')
    torch.save(value, temp)
    temp.replace(path)


class Runtime:
    def __init__(self, out=OUT):
        self.out = out
        self.protocol = json.loads((out / 'protocol/PROTOCOL_V4.json').read_text())
        assert self.protocol['status'] == 'FROZEN'
        self.feature_manifest = json.loads((out / 'features/FEATURE_MANIFEST_V4.json').read_text())
        assert self.feature_manifest['status'] == 'COMPLETE'
        for name, digest in self.feature_manifest['files'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'feature checksum mismatch: {name}')
        self.identity = {'protocol_sha256': sha256(out / 'protocol/PROTOCOL_V4.json'),
                         'features_sha256': sha256(out / 'features/FEATURE_MANIFEST_V4.json')}
        self.data = pd.read_csv(self.protocol['data']['prepared_relations_path'])
        assert sha256(self.protocol['data']['prepared_relations_path']) == self.protocol['data']['prepared_relations_sha256']
        self.drugs = self.data.drug_feature_index.to_numpy(np.int64)
        available = np.load(out / 'features/BERMOL_AVAILABLE_V4.npy')
        if not available[self.drugs].all():
            raise ValueError('requested relation scope includes unavailable BerMol rows; revise and refreeze all controls before training')
        self.targets = self.data.target_feature_index.to_numpy(np.int64)
        self.labels = self.data.binary_label.to_numpy(np.float32)
        self.positions = {s: np.flatnonzero(self.data.split.eq(s)) for s in ['train', 'validation', 'test']}
        self.molecules = {}
        for kind, path in [('bermol', out / 'features/BERMOL768_FLOAT32_V4.npy'),
                           ('morgan', Path(self.protocol['data']['identity']['features_path']))]:
            self.molecules[kind] = torch.from_numpy(np.load(path)).cuda()
        self.proteins = torch.from_numpy(np.load(out / 'features/ESM2_FULL_MEAN1280_FLOAT32_V4.npy')).cuda()
        self.support = {n: np.load(self.protocol['data'][f'support_{n}_path'], mmap_mode='r') for n in ['indices','similarities','mask']}
        assert available[self.support['indices'][self.support['mask']]].all()
        # Audit existing cache bytes against its own immutable manifest.
        cachepath = Path(self.protocol['data']['support_indices_path']).parent
        cachemanifest = json.loads((cachepath / 'manifest.json').read_text())
        assert cachemanifest['identity']['store_hash'] == self.protocol['data']['store_hash']
        for n in self.support:
            assert sha256(cachepath / f'{n}.npy') == cachemanifest['array_sha256'][n]
        self.panels = {}
        for split, info in self.protocol['panels'].items():
            path = out / 'protocol' / f'{split}_panel.npz'
            assert sha256(path) == info['panel_sha256']
            with np.load(path) as f:
                panel = {k:f[k] for k in f.files}
            panel['drugs'] = np.repeat(panel['drug_indices'], len(panel['target_indices']))
            panel['targets'] = np.tile(panel['target_indices'], len(panel['drug_indices']))
            panel['support'] = {n:np.load(p, mmap_mode='r') for n,p in info['support'].items()}
            manifest = json.loads((Path(info['support']['indices']).parent / 'manifest.json').read_text())
            assert manifest['identity']['store_hash'] == self.protocol['data']['store_hash']
            for n,p in info['support'].items():
                assert sha256(p) == manifest['array_sha256'][n]
            self.panels[split] = panel
        self.query_stream = QueryBatchStream(self.drugs, self.labels, self.positions['train'], 64, 32)

    def batch(self, model, kind, ds, ts, evidence=None, rows=None, rng=None):
        ds = torch.as_tensor(ds, device='cuda')
        ts = torch.as_tensor(ts, device='cuda')
        drug, target = self.molecules[kind][ds].float(), self.proteins[ts]
        if not isinstance(model, SignedEvidenceV4):
            return {'logit': model(drug, target)}
        mask = np.array(evidence['mask'][rows], copy=True)
        if rng is not None:
            # Whole-branch episodes include positive-only, negative-only, and
            # no-support inputs; no held-out labels enter this dropout.
            mask &= rng.random((len(mask), 2, 1)) >= 0.2
        indices = np.maximum(evidence['indices'][rows], 0)
        return model(drug, target, self.molecules[kind][torch.as_tensor(indices, device='cuda')].float(),
                     torch.as_tensor(np.array(evidence['similarities'][rows]), device='cuda'),
                     torch.as_tensor(mask, device='cuda'))

    @torch.no_grad()
    def eval_cache(self, model, kind):
        model.eval()
        base = model.base if isinstance(model, SignedEvidenceV4) else model
        with torch.autocast('cuda', dtype=torch.bfloat16):
            d = torch.cat([base.drug(self.molecules[kind][s:s+4096].float()) for s in range(0, len(self.molecules[kind]), 4096)])
            t = base.target(self.proteins)
            reference = None
            if isinstance(model, SignedEvidenceV4):
                reference = torch.zeros((len(d), len(t)), dtype=torch.float32, device='cuda')
                train = self.positions['train']
                for start in range(0, len(train), 4096):
                    ix = train[start:start+4096]
                    di, ti = torch.as_tensor(self.drugs[ix], device='cuda'), torch.as_tensor(self.targets[ix], device='cuda')
                    reference[di, ti] = base.potential(d[di], t[ti]).float()
        return d,t,reference

    @torch.no_grad()
    def predict(self, model, kind, ds, ts, evidence=None, component=False):
        d,t,reference = self.eval_cache(model, kind)
        result = {}
        step = 1024 if isinstance(model, SignedEvidenceV4) else 4096
        for start in range(0, len(ds), step):
            sl = slice(start, start+step)
            di, ti = torch.as_tensor(ds[sl], device='cuda'), torch.as_tensor(ts[sl], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if isinstance(model, SignedEvidenceV4):
                    ids = torch.as_tensor(np.maximum(evidence['indices'][sl], 0), device='cuda')
                    scores = model.forward_states(d[di], t[ti], d[ids],
                        torch.as_tensor(np.array(evidence['similarities'][sl]), device='cuda'),
                        torch.as_tensor(np.array(evidence['mask'][sl]), device='cuda'), reference[ids, ti[:,None,None]])
                else:
                    scores = {'logit': model.potential(d[di], t[ti])}
            for key in ['logit', 'base_logit', 'positive_bonus', 'negative_penalty'] if component else ['logit']:
                if key in scores:
                    result.setdefault(key, []).append(scores[key].float().cpu().numpy())
        return {k:np.concatenate(v) for k,v in result.items()}

    def panel_metrics(self, model, kind, split):
        panel = self.panels[split]
        scores = self.predict(model, kind, panel['drugs'], panel['targets'], panel['support'])['logit']
        matrix = scores.reshape(panel['known_positive'].shape)
        return positive_retrieval_metrics(panel['known_positive'], matrix, panel['target_indices'])


def train(runtime, variant, seed, max_epochs=30):
    out = runtime.out / 'runs' / f'{variant}_{seed}'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'run already exists: {out}')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    kind = 'morgan' if variant == 'morgan' else 'bermol'
    config = GlobalConfigV4(drug_dim=2048 if kind == 'morgan' else 768)
    base = GlobalPairV4(config).cuda()
    if variant == 'signed':
        previous = torch.load(runtime.out / 'runs' / f'bermol_{seed}' / 'BEST_MODEL_V4.pt', map_location='cuda', weights_only=False)
        assert previous['identity'] == runtime.identity
        base.load_state_dict(previous['model'])
        model = SignedEvidenceV4(base).cuda()
        model.ramp.zero_()
        parameters = [{'params': base.parameters(), 'lr': 0.00002},
                      {'params': list(model.project.parameters()) + list(model.branches.parameters()), 'lr': 0.0002}]
    else:
        model = base
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=0.0002, weight_decay=0.01)
    history, best, best_epoch = [], -1., 0
    started = time.monotonic()

    def snapshot(epoch):
        return {'format': 'BIOMASTER_ODTI_V4_R1', 'variant': variant, 'seed': seed, 'epoch': epoch,
                'config': asdict(config), 'model': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                'identity': runtime.identity, 'validation_panel_ap': best}

    if variant == 'signed':
        initial = runtime.panel_metrics(model, kind, 'validation')
        best = initial['macro_positive_retrieval_ap']
        save_model(out / 'BEST_MODEL_V4.pt', snapshot(0))
        history.append({'epoch': 0, 'validation_panel': initial, 'note': 'unchanged fitted R1 baseline, ramp=0'})
    train_positions = runtime.positions['train']
    # Each ranking epoch visits every eligible drug once, with distinct rows.
    query_steps = int(np.ceil(len(runtime.query_stream.keys) / 64))
    for epoch in range(1, max_epochs + 1):
        model.train()
        if variant == 'signed':
            model.ramp.fill_(min(1., epoch / 2.))
            for p in base.parameters():
                p.requires_grad_(epoch > 2)
        loss_values = []
        batches = [(rows, False) for rows in np.array_split(rng.permutation(train_positions), int(np.ceil(len(train_positions) / 2048)))]
        query_batches = runtime.query_stream.batches(seed + epoch)
        # Spread equal-query updates across the full relation stream.
        queries = {int(i * len(batches) / query_steps): next(query_batches) for i in range(query_steps)}
        covered = 0
        for b, (rows, _) in enumerate(batches):
            updates = [(rows, False)] + ([(queries[b], True)] if b in queries else [])
            covered += len(rows)
            for ix, is_query in updates:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = runtime.batch(model, kind, runtime.drugs[ix], runtime.targets[ix], runtime.support, ix, rng)
                    logits = result['logit'].float()
                    y = torch.as_tensor(runtime.labels[ix], device='cuda')
                    loss = F.binary_cross_entropy_with_logits(logits, y)
                    if is_query:
                        loss = loss + drug_macro_pairwise_loss(logits, y, torch.as_tensor(runtime.drugs[ix], device='cuda'), torch.ones_like(y, dtype=torch.bool))
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                loss_values.append(float(loss.detach()))
        assert covered == len(train_positions)
        metrics = runtime.panel_metrics(model, kind, 'validation')
        value = metrics['macro_positive_retrieval_ap']
        if value > best:
            best, best_epoch = value, epoch
            save_model(out / 'BEST_MODEL_V4.pt', snapshot(epoch))
        record = {'epoch': epoch, 'train_mean_update_loss': float(np.mean(loss_values)),
                  'relation_rows_seen': covered, 'query_updates': query_steps, 'validation_panel': metrics,
                  'best_epoch': best_epoch, 'seconds': round(time.monotonic()-started, 2)}
        history.append(record)
        write_json(out / 'TRAINING_HISTORY_V4.json', history)
        write_json(out / 'STATUS_V4.json', {'status': 'TRAINING', 'variant': variant, 'seed': seed, **record})
        print(json.dumps({'variant': variant, 'seed': seed, 'epoch': epoch, 'val_panel_ap': value,
                          'val_r20': metrics['macro_recall_at_20'], 'best_epoch': best_epoch, 'seconds': record['seconds']}), flush=True)
        last = snapshot(epoch)
        last.update(optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), history=history, best_epoch=best_epoch)
        save_model(out / 'LAST_MODEL_V4.pt', last)
        if epoch >= 8 and epoch - best_epoch >= 5:
            break
    write_json(out / 'STATUS_V4.json', {'status': 'TRAINED_TEST_NOT_EVALUATED', 'variant': variant, 'seed': seed,
               'best_epoch': best_epoch, 'epochs': epoch, 'best_validation_panel_ap': best,
               'parameters': sum(p.numel() for p in model.parameters()), 'seconds': round(time.monotonic()-started, 2),
               'identity': runtime.identity})


def subgroup_metrics(panel, scores):
    similarity = np.where(panel['support']['mask'][:,1], panel['support']['similarities'][:,1], 0).max(-1).reshape(panel['known_positive'].shape).max(-1)
    result = {}
    for name, select in [('low_lt_0.4', similarity < .4), ('middle_0.4_to_0.7', (similarity >= .4) & (similarity < .7)), ('near_ge_0.7', similarity >= .7)]:
        result[name] = positive_retrieval_metrics(panel['known_positive'][select], scores[select], panel['target_indices'])
    return result


def evaluate(runtime):
    out = runtime.out / 'evaluation'
    out.mkdir(exist_ok=True)
    test = runtime.positions['test']
    panel = runtime.panels['test']
    observed_evidence = {k:v[test] for k,v in runtime.support.items()}
    results = {}
    modelscores = {}
    for variant in ['morgan','bermol','signed']:
        for seed in runtime.protocol['seeds']:
            name = f'{variant}_{seed}'
            run = runtime.out / 'runs' / name
            status = json.loads((run / 'STATUS_V4.json').read_text())
            assert status['status'] in {'TRAINED_TEST_NOT_EVALUATED', 'COMPLETE'}
            checkpoint = torch.load(run / 'BEST_MODEL_V4.pt', map_location='cuda', weights_only=False)
            assert checkpoint['identity'] == runtime.identity
            base = GlobalPairV4(GlobalConfigV4(**checkpoint['config'])).cuda()
            model = SignedEvidenceV4(base).cuda() if variant == 'signed' else base
            model.load_state_dict(checkpoint['model'])
            kind = 'morgan' if variant == 'morgan' else 'bermol'
            measured = runtime.predict(model, kind, runtime.drugs[test], runtime.targets[test], observed_evidence, component=True)
            dense = runtime.predict(model, kind, panel['drugs'], panel['targets'], panel['support'], component=True)
            matrix = dense['logit'].reshape(panel['known_positive'].shape)
            results[name] = {'training': status, 'observed': observed_metrics(runtime.labels[test], measured['logit'], runtime.drugs[test], runtime.targets[test]),
                'panel': positive_retrieval_metrics(panel['known_positive'], matrix, panel['target_indices']),
                'similarity_subgroups': subgroup_metrics(panel, matrix)}
            if variant == 'signed':
                results[name]['score_audit'] = {'negative_penalty_min': float(dense['negative_penalty'].min()),
                    'negative_penalty_max': float(dense['negative_penalty'].max()), 'positive_bonus_max': float(dense['positive_bonus'].max()),
                    'negative_only_rows': int((panel['support']['mask'][:,0].any(-1) & ~panel['support']['mask'][:,1].any(-1)).sum())}
                negative_only = panel['support']['mask'][:,0].any(-1) & ~panel['support']['mask'][:,1].any(-1)
                if negative_only.any():
                    assert np.all(dense['logit'][negative_only] <= dense['base_logit'][negative_only] + 1e-5)
            np.savez_compressed(out / f'{name}_PREDICTIONS_V4.npz', observed_rows=test, observed_labels=runtime.labels[test],
                observed_scores=measured['logit'], panel_scores=matrix, known_positive=panel['known_positive'],
                drug_indices=panel['drug_indices'], target_indices=panel['target_indices'], **{k:v for k,v in dense.items() if k != 'logit'})
            modelscores[name] = matrix
            write_json(out / 'RESULTS_V4.json', results)
            print(json.dumps({'evaluated': name, 'observed_AP': results[name]['observed']['micro_ap'],
                              'panel_AP': results[name]['panel']['macro_positive_retrieval_ap']}), flush=True)
            del model,base
            torch.cuda.empty_cache()
    features = support_summary_features(runtime.support['similarities'], runtime.support['mask'])
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1., random_state=0))
    train = runtime.positions['train']
    estimator.fit(features[train], runtime.labels[train])
    panel_features = support_summary_features(panel['support']['similarities'], panel['support']['mask'])
    for name, observed, dense in [('positive_nearest', features[test,1], panel_features[:,1]),
                                  ('positive_negative_logistic', estimator.decision_function(features[test]), estimator.decision_function(panel_features))]:
        matrix = dense.reshape(panel['known_positive'].shape)
        results[name] = {'observed': observed_metrics(runtime.labels[test], observed, runtime.drugs[test], runtime.targets[test]),
                         'panel': positive_retrieval_metrics(panel['known_positive'], matrix, panel['target_indices']),
                         'similarity_subgroups': subgroup_metrics(panel, matrix)}
        modelscores[name] = matrix
        np.savez_compressed(out / f'{name}_PREDICTIONS_V4.npz', observed_rows=test, observed_scores=observed, panel_scores=matrix,
                            known_positive=panel['known_positive'], drug_indices=panel['drug_indices'], target_indices=panel['target_indices'])
    write_json(out / 'PN_BASELINE_V4.json', {'training_split': 'train only', 'scaler_mean': estimator[0].mean_.tolist(),
         'scaler_scale': estimator[0].scale_.tolist(), 'coefficient': estimator[1].coef_.tolist(), 'intercept': estimator[1].intercept_.tolist()})
    # Paired drug bootstrap of seed-mean per-query AP differences. This measures
    # query sampling uncertainty; seed variation is reported separately.
    def per_query(scores):
        values = []
        for y,s in zip(panel['known_positive'], scores):
            ranks = np.flatnonzero(y[np.lexsort((panel['target_indices'], -s))]) + 1
            values.append(float(np.mean(np.arange(1, len(ranks)+1) / ranks)))
        return np.array(values)
    vectors = {k:per_query(v) for k,v in modelscores.items()}
    for variant in ['morgan','bermol','signed']:
        vectors[variant] = np.mean([vectors[f'{variant}_{s}'] for s in runtime.protocol['seeds']], axis=0)
    rng = np.random.default_rng(20260905)
    draws = rng.integers(0, len(panel['drug_indices']), size=(5000, len(panel['drug_indices'])))
    comparisons = {}
    for a,b in [('bermol','morgan'), ('signed','bermol'), ('signed','positive_nearest'), ('signed','positive_negative_logistic')]:
        delta = vectors[a] - vectors[b]
        comparisons[f'{a}_minus_{b}'] = {'mean_ap_difference': float(delta.mean()),
            'paired_drug_bootstrap_95_CI': np.quantile(delta[draws].mean(-1), [.025,.975]).tolist(), 'queries': len(delta), 'resamples': 5000}
    write_json(out / 'RESULTS_V4.json', results)
    write_json(out / 'PAIRED_COMPARISONS_V4.json', comparisons)
    write_json(out / 'EVALUATION_STATUS_V4.json', {'status':'COMPLETE', 'models':len(results), 'identity':runtime.identity,
        'claim':'R1 historical development experiment; not full V4, not an untouched confirmatory benchmark'})
    for variant in ['morgan', 'bermol', 'signed']:
        for seed in runtime.protocol['seeds']:
            path = runtime.out / 'runs' / f'{variant}_{seed}' / 'STATUS_V4.json'
            status = json.loads(path.read_text())
            status.update(status='COMPLETE', evaluation_results=str(out / 'RESULTS_V4.json'))
            write_json(path, status)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['morgan','bermol','signed'])
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--evaluate-all', action='store_true')
    p.add_argument('--max-epochs', type=int, default=30)
    args = p.parse_args()
    torch.set_num_threads(4)
    runtime = Runtime()
    if args.evaluate_all:
        evaluate(runtime)
    elif args.variant:
        train(runtime, args.variant, args.seed, args.max_epochs)
    else:
        p.error('--variant or --evaluate-all required')


if __name__ == '__main__':
    main()

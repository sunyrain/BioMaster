#!/usr/bin/env python3
"""Resumable 24-fit objective experiment with validation plateau and final test gate."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from prepare_endpoint_multitask_20260911 import OUT, SOURCE, DATA, FEATURES, prepare, write_json
from run_endpoint_ablation_20260911 import Bank, calibrate, report_metrics
from biomaster.endpoint_ablation import ARMS, CyclingRows
from biomaster.endpoint_multitask import (VARIANTS, ENDPOINTS, AuxiliaryInteraction, MeasuredRankPool,
    query_rank_loss, endpoint_huber, ValidationPlateau, stream_state, restore_stream)
from biomaster.best_model_training import WeightAverage
from biomaster.portable_ranker_v2 import digest


def now():
    return datetime.now(timezone.utc).isoformat()


def status(**value):
    value = dict(updated_utc=now(), pid=os.getpid(), **value)
    write_json(OUT/'STATUS.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def save_torch(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp); tmp.replace(path)


def parameters_hash(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


class ArmData:
    def __init__(self, arm):
        self.arm = arm; self.endpoints = ENDPOINTS[arm]
        self.frame = pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet')
        self.d = self.frame.drug_feature_index.to_numpy(np.int64)
        self.t = self.frame.target_feature_index.to_numpy(np.int64)
        self.y = self.frame.binary_label.to_numpy(np.float32)
        self.prevalence = float(self.y.mean())
        taskframe = pd.read_parquet(OUT/f'{arm}_RANK_TASKS.parquet')
        self.rank = {}
        for direction, column in [('target', 'target_feature_index'), ('drug', 'drug_feature_index')]:
            self.rank[direction] = [MeasuredRankPool(f, column) for _, f in taskframe.groupby('task', sort=True)]
        reg = pd.read_parquet(OUT/f'{arm}_TRAIN_REGRESSION.parquet')
        self.reg = [reg[reg.endpoint.eq(e)].reset_index(drop=True) for e in self.endpoints]
        self.reg_arrays = [(f.drug_feature_index.to_numpy(np.int64), f.target_feature_index.to_numpy(np.int64),
                            f.median_p_activity_unique.to_numpy(np.float32)) for f in self.reg]
        self.means = np.array([f.median_p_activity_unique.mean() for f in self.reg], np.float32)
        self.scales = np.array([f.median_p_activity_unique.std(ddof=0) for f in self.reg], np.float32)
        assert np.isfinite(self.means).all() and (self.scales > 0).all()
        self.valreg = pd.read_parquet(OUT/f'{arm}_VALIDATION_REGRESSION.parquet')


class Streams:
    def __init__(self, data, seed):
        self.binary = CyclingRows(len(data.frame), seed)
        self.rank = np.random.default_rng(seed+100000)
        self.reg = [CyclingRows(len(f), seed+200000+i) for i, f in enumerate(data.reg)]
        self.trace = '00'*32

    def state(self):
        return dict(binary=stream_state(self.binary), rank=self.rank.bit_generator.state,
                    reg=[stream_state(s) for s in self.reg], trace=self.trace)

    def restore(self, state):
        restore_stream(self.binary, state['binary']); self.rank.bit_generator.state = state['rank']
        for stream, value in zip(self.reg, state['reg'], strict=True):
            restore_stream(stream, value)
        self.trace = state['trace']


def step_loss(model, bank, data, streams, variant, p, scales):
    use_rank, use_reg = VARIANTS[variant]
    indices = streams.binary.take(p['batch_size'])
    streams.trace = hashlib.sha256(bytes.fromhex(streams.trace)+indices.astype('<i8').tobytes()).hexdigest()
    ds, ts = [data.d[indices]], [data.t[indices]]
    offset = len(indices); rank_specs = []; regression_start = None
    for direction in ['target', 'drug'] if use_rank else []:
        for pool in data.rank[direction]:
            d, t = pool.sample(streams.rank, p['rank_queries_per_task_direction'], p['rank_candidates_per_class'])
            ds.append(d.reshape(-1)); ts.append(t.reshape(-1))
            rank_specs.append((direction, offset, offset+d.size, d.shape)); offset += d.size
    rank_rows = offset-len(indices)
    values, endpoint_ids = [], []
    if use_reg:
        regression_start = offset
        for e, ((d, t, v), stream) in enumerate(zip(data.reg_arrays, streams.reg, strict=True)):
            ix = stream.take(p['regression_rows_per_endpoint'])
            ds.append(d[ix]); ts.append(t[ix]); values.append(v[ix]); endpoint_ids.append(np.full(len(ix), e))
    y = torch.as_tensor(data.y[indices], device='cuda')
    weight = torch.where(y.bool(), .5/data.prevalence, .5/(1-data.prevalence))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, regress = model(bank.batch(np.concatenate(ds), np.concatenate(ts)))
    logits, regress = logits.float(), regress.float()
    bce = (torch.nn.functional.binary_cross_entropy_with_logits(logits[:len(indices)], y[:, None].expand(-1, 2),
                                                               reduction='none')*weight[:, None]).mean()
    total = bce; losses = {'binary': bce}
    for direction in ['target', 'drug'] if use_rank else []:
        terms = [query_rank_loss(logits[a:b].mean(1).reshape(shape), p['rank_temperature'])
                 for name, a, b, shape in rank_specs if name == direction]
        loss = torch.stack(terms).mean(); losses['rank_'+direction] = loss
        total = total+p[direction+'_rank_weight']*loss
    if use_reg:
        truth = torch.as_tensor(np.concatenate(values), device='cuda')
        ids = torch.as_tensor(np.concatenate(endpoint_ids), dtype=torch.long, device='cuda')
        loss = endpoint_huber(regress[regression_start:], truth, ids, *scales)
        losses['regression'] = loss; total = total+p['regression_weight']*loss
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite training objective')
    counts = dict(binary=len(indices), rank=rank_rows, regression=sum(map(len, values)))
    return total, losses, counts


@torch.inference_mode()
def predict(model, bank, frame, batch_size=4096):
    model.eval(); scores = []; regressions = []
    for start in range(0, len(frame), batch_size):
        f = frame.iloc[start:start+batch_size]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, reg = model(bank.batch(f.drug_feature_index.to_numpy(), f.target_feature_index.to_numpy()))
        scores.append(logits.float().mean(1).cpu().numpy()); regressions.append(reg.float().cpu().numpy())
    s, r = np.concatenate(scores), np.concatenate(regressions)
    assert np.isfinite(s).all() and np.isfinite(r).all()
    return s, r


class QueryMetrics:
    def __init__(self, frame, column):
        self.y = frame.binary_label.to_numpy(int)
        self.groups = [ids for ids in frame.groupby(column, sort=True).indices.values()
                       if len(ids) >= 10 and 0 < self.y[ids].sum() < len(ids)]
        if not self.groups:
            raise ValueError('No qualifying validation queries')

    def evaluate(self, scores):
        ap = []; precisions = {5: [], 10: [], 20: []}
        for ids in self.groups:
            y, s = self.y[ids], scores[ids]
            ap.append(average_precision_score(y, s))
            order = np.argsort(-s, kind='stable')
            for k in precisions:
                if len(ids) >= k:
                    precisions[k].append(float(y[order[:k]].mean()))
        result = dict(queries=len(ap), macro_ap=float(np.mean(ap)))
        for k, values in precisions.items():
            result['p'+str(k)] = float(np.mean(values)) if values else None
            result['p'+str(k)+'_queries'] = len(values)
        return result


def regression_metrics(frame, predictions, data):
    result = {}
    for e, endpoint in enumerate(data.endpoints):
        mask = frame.endpoint.eq(endpoint).to_numpy()
        if not mask.any():
            continue
        truth = frame.loc[mask, 'median_p_activity_unique'].to_numpy(float)
        pred = predictions[mask, e]*data.scales[e]+data.means[e]
        residual = pred-truth
        result[endpoint] = dict(rows=len(truth), mae=float(np.abs(residual).mean()),
                                rmse=float(np.sqrt((residual**2).mean())))
    return result


class Validation:
    def __init__(self):
        self.frame = pd.read_parquet(SOURCE/'COMMON_VALIDATION.parquet').reset_index(drop=True)
        self.kd = self.frame.panel.eq('AFFINITY_KD_KI').to_numpy()
        f = self.frame[self.kd].reset_index(drop=True)
        self.query = {name: QueryMetrics(f, col) for name, col in [('target', 'target_id'), ('drug', 'molecule_id')]}

    def evaluate(self, model, bank, data, variant):
        scores, _ = predict(model, bank, self.frame)
        y = self.frame.loc[self.kd, 'binary_label'].to_numpy(int); s = scores[self.kd]
        q = {k: v.evaluate(s) for k, v in self.query.items()}
        precision, recall, thresholds = precision_recall_curve(y, s)
        f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
        threshold = float(thresholds[int(np.argmax(f1))])
        inactive = self.frame.panel.eq('EXPLICIT_INACTIVE').to_numpy()
        panels = {}
        for panel, ids in self.frame.groupby('panel').indices.items():
            truth = self.frame.binary_label.to_numpy(int)[ids]
            if 0 < truth.sum() < len(truth):
                panels[panel] = dict(rows=len(ids), ap=float(average_precision_score(truth, scores[ids])),
                                     auroc=float(roc_auc_score(truth, scores[ids])))
        _, reg = predict(model, bank, data.valreg)
        return dict(selection_score=(q['target']['macro_ap']+q['drug']['macro_ap'])/2,
                    query=q, panels=panels, kdki_raw_threshold_max_f1=threshold,
                    explicit_inactive_rows=int(inactive.sum()),
                    explicit_inactive_fpr=float((scores[inactive] >= threshold).mean()),
                    regression_auxiliary_trained=VARIANTS[variant][1],
                    regression=regression_metrics(data.valreg, reg, data))


def create_model(data, seed, config):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AuxiliaryInteraction(config, data.endpoints).cuda()
    initial = parameters_hash(model.base)
    # Head count must not alter the initial dropout RNG between A and B.
    torch.manual_seed(seed+400000)
    return model, initial


def identity():
    files = [Path(__file__), ROOT/'scripts/prepare_endpoint_multitask_20260911.py',
             ROOT/'biomaster/endpoint_multitask.py', ROOT/'biomaster/endpoint_ablation.py',
             ROOT/'biomaster/unified_interaction.py', ROOT/'biomaster/molecular_controls.py',
             ROOT/'biomaster/model_registry.py', ROOT/'biomaster/best_model_training.py',
             ROOT/'scripts/run_endpoint_ablation_20260911.py', OUT/'PROTOCOL.json', OUT/'DATA_MANIFEST.json',
             OUT/'MODEL_CONFIG.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def benchmark(bank, datasets, validation, p, config):
    rows = []
    for variant in VARIANTS:
        for arm, data in datasets.items():
            model, _ = create_model(data, 20260921, config); model.train()
            ema = WeightAverage(model, p['ema_decay'])
            opt = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
            streams = Streams(data, 20260921)
            scales = (torch.tensor(data.means, device='cuda'), torch.tensor(data.scales, device='cuda'))
            torch.cuda.synchronize(); started = time.monotonic()
            for step in range(120):
                if step == 20:
                    torch.cuda.synchronize(); started = time.monotonic()
                opt.zero_grad(set_to_none=True)
                loss, _, counts = step_loss(model, bank, data, streams, variant, p, scales)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                opt.step(); ema.update(model)
            torch.cuda.synchronize(); per_step = (time.monotonic()-started)/100
            start = time.monotonic(); validation.evaluate(ema.model, bank, data, variant)
            val_seconds = time.monotonic()-start
            interval = max(1000, math.ceil(2*len(data.frame)/p['batch_size']))
            row = dict(arm=arm, variant=variant, seconds_per_step=per_step, validation_seconds=val_seconds,
                       validation_interval_steps=interval, rows_per_step=counts, benchmark_steps=120,
                       benchmark_weights_discarded=True)
            for epochs in [30, 60]:
                steps = math.ceil(epochs*len(data.frame)/p['batch_size'])
                row[f'estimated_seconds_{epochs}_passes'] = steps*per_step+math.ceil(steps/interval)*val_seconds
            rows.append(row); status(stage='throughput_benchmark', **row)
            del model, ema, opt; torch.cuda.empty_cache()
    result = dict(created_utc=now(), rows=rows, total_fits=24,
                  seconds_at_30_passes=sum(r['estimated_seconds_30_passes'] for r in rows)*len(p['seeds']),
                  seconds_at_60_passes=sum(r['estimated_seconds_60_passes'] for r in rows)*len(p['seeds']),
                  caveat='Measured-speed scenarios, not known convergence epochs; excludes final evaluation and per-fit setup/checkpoint I/O.')
    write_json(OUT/'THROUGHPUT_ETA.json', result)
    return result


def fit(data, variant, seed, bank, validation, p, config, ident, fit_index):
    run = OUT/f'{data.arm}__{variant}__seed_{seed}'; run.mkdir(exist_ok=True)
    if (run/'RESULT.json').exists():
        r = json.loads((run/'RESULT.json').read_text())
        assert r['identity'] == ident and r['checkpoint_sha256'] == digest(run/'model.pt')
        assert r['status'] == 'CONVERGED_VALIDATION_PLATEAU'
        return r
    model, initial_hash = create_model(data, seed, config)
    ema = WeightAverage(model, p['ema_decay'])
    opt = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    streams = Streams(data, seed)
    rule = p['convergence']
    controller = ValidationPlateau(p['learning_rate'], rule['minimum_lr'], rule['improvement_delta'],
                                   rule['lr_reduce_every_stale_checks'], rule['stale_checks_to_stop'], rule['minimum_complete_passes'])
    scales = (torch.tensor(data.means, device='cuda'), torch.tensor(data.scales, device='cuda'))
    interval = max(1000, math.ceil(2*len(data.frame)/p['batch_size']))
    history = []; startstep = 0; elapsed_previous = 0.; counts_total = dict(binary=0, rank=0, regression=0)
    if (run/'RESUME.pt').exists():
        state = torch.load(run/'RESUME.pt', map_location='cpu', weights_only=False)
        assert state['identity'] == ident
        model.load_state_dict(state['model']); ema.model.load_state_dict(state['ema']); ema.updates = state['ema_updates']
        opt.load_state_dict(state['optimizer']); streams.restore(state['streams']); controller.__dict__.update(state['controller'])
        torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        history = state['history']; startstep = state['step']; elapsed_previous = state['seconds']; counts_total = state['counts']
        del state
    write_json(run/'REGRESSION_SCALING.json', dict(endpoints=data.endpoints, means=data.means.tolist(),
               scales=data.scales.tolist(), fit_split='TRAIN_ONLY', auxiliary_trained=VARIANTS[variant][1]))
    started = time.monotonic(); model.train(); losses = {}; last_log = started; converged = bool(history and history[-1]['converged'])
    status(stage='training_started', arm=data.arm, variant=variant, seed=seed, fit_index=fit_index, total_fits=24,
           train_pairs=len(data.frame), validation_interval_steps=interval, resumed_step=startstep)
    finalstep = startstep
    for step in range(startstep, p['emergency_max_steps']) if not converged else []:
        rate = controller.lr*min(1., (step+1)/p['warmup_steps'])
        for g in opt.param_groups:
            g['lr'] = rate
        opt.zero_grad(set_to_none=True)
        loss, components, counts = step_loss(model, bank, data, streams, variant, p, scales)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        opt.step(); ema.update(model)
        for key, value in components.items():
            losses.setdefault(key, []).append(float(value.detach()))
        for key in counts_total:
            counts_total[key] += counts[key]
        finalstep = step+1
        if finalstep % 250 == 0:
            status(stage='training', arm=data.arm, variant=variant, seed=seed, fit_index=fit_index, total_fits=24,
                   step=finalstep, complete_passes=streams.binary.passes, learning_rate=rate,
                   seconds=elapsed_previous+time.monotonic()-started,
                   recent_steps_per_second=250/(time.monotonic()-last_log),
                   losses={k:float(np.mean(v)) for k, v in losses.items()})
            last_log = time.monotonic()
        if finalstep % interval != 0 and finalstep != p['emergency_max_steps']:
            continue
        val_started = time.monotonic()
        metrics = validation.evaluate(ema.model, bank, data, variant)
        event = controller.observe(metrics['selection_score'], streams.binary.passes)
        converged = event['converged']
        row = dict(step=finalstep, passes=counts_total['binary']/len(data.frame), learning_rate=rate,
                   validation_seconds=time.monotonic()-val_started,
                   seconds=elapsed_previous+time.monotonic()-started,
                   training_losses={k:float(np.mean(v)) for k, v in losses.items()},
                   stale_validation_checks=controller.stale, **event, **metrics)
        history.append(row); losses = {}; write_json(run/'HISTORY.json', history)
        if event['best']:
            save_torch(run/'BEST.pt', dict(model=ema.model.state_dict(), step=finalstep, validation=metrics, identity=ident))
        state = dict(identity=ident, model=model.state_dict(), ema=ema.model.state_dict(), ema_updates=ema.updates,
                     optimizer=opt.state_dict(), streams=streams.state(), controller=controller.__dict__.copy(),
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), history=history,
                     step=finalstep, seconds=elapsed_previous+time.monotonic()-started, counts=counts_total)
        save_torch(run/'RESUME.pt', state); del state
        status(stage='validation', arm=data.arm, variant=variant, seed=seed, fit_index=fit_index, total_fits=24, **row)
        if converged:
            break
        model.train(); last_log = time.monotonic()
    if not converged:
        write_json(run/'NOT_CONVERGED.json', dict(status='NOT_CONVERGED_NEEDS_EXTENSION', step=finalstep,
                    reason='Emergency guard reached without validation plateau; not a completed fit.'))
        raise RuntimeError(f'{run.name}: emergency limit, convergence not established')
    best = torch.load(run/'BEST.pt', map_location='cpu', weights_only=False)
    ema.model.load_state_dict(best['model'])
    scores, _ = predict(ema.model, bank, validation.frame)
    cal = calibrate(validation.frame, scores); write_json(run/'CALIBRATION.json', cal)
    save_torch(run/'model.pt', dict(architecture='endpoint_multitask_research', config=config, endpoints=data.endpoints,
               model={k:v.detach().cpu() for k, v in ema.model.state_dict().items()}, arm=data.arm, variant=variant,
               seed=seed, identity=ident, regression_means=data.means.tolist(), regression_scales=data.scales.tolist(),
               score='mean of original two binary logits; no regression-score mixing'))
    result = dict(status='CONVERGED_VALIDATION_PLATEAU', arm=data.arm, variant=variant, seed=seed, identity=ident,
                  initial_base_sha256=initial_hash, best_step=best['step'], optimizer_steps=finalstep,
                  best_validation=best['validation'], selection_metric='mean_KdKi_target_and_drug_macro_AP',
                  complete_passes=streams.binary.passes, sample_counts=counts_total, sampling_trace=streams.trace,
                  checkpoint_sha256=digest(run/'model.pt'), seconds=elapsed_previous+time.monotonic()-started,
                  parameters=sum(p.numel() for p in model.parameters()), test_used_for_training=False,
                  production_replaced=False, wetlab_unchanged=True)
    write_json(run/'RESULT.json', result)
    status(stage='fit_complete', fit_index=fit_index, total_fits=24, arm=data.arm, variant=variant, seed=seed,
           optimizer_steps=finalstep, best_step=best['step'], seconds=result['seconds'])
    del model, ema, opt; torch.cuda.empty_cache()
    return result


def evaluate_all(bank, datasets, p, config, results, ident):
    assert len(results) == 24 and all(r['status'] == 'CONVERGED_VALIDATION_PLATEAU' for r in results)
    for r in results:
        run = OUT/f"{r['arm']}__{r['variant']}__seed_{r['seed']}"
        assert r['identity'] == ident and digest(run/'model.pt') == r['checkpoint_sha256']
    for seed in p['seeds']:
        assert len({r['initial_base_sha256'] for r in results if r['seed'] == seed}) == 1
    write_json(OUT/'ALL_FITS_FROZEN.json', dict(created_utc=now(), runs=results, all_converged=True,
               no_new_test_scoring_before_freeze=True))
    test = pd.read_parquet(SOURCE/'COMMON_TEST.parquet').reset_index(drop=True)
    reg = pd.read_parquet(DATA/'TEST_REGRESSION.parquet')
    # Every test feature must belong to the already prepared common test universe.
    reg = reg[reg.pair_id.isin(test.pair_id)].reset_index(drop=True)
    rows, queries, regrows = [], [], []
    predictions = test[['panel', 'pair_id', 'molecule_id', 'target_id', 'split_group', 'binary_label', 'document_disjoint']].copy()
    for r in results:
        arm, variant, seed = r['arm'], r['variant'], r['seed']
        run = OUT/f'{arm}__{variant}__seed_{seed}'
        model = AuxiliaryInteraction(config, ENDPOINTS[arm]).cuda()
        state = torch.load(run/'model.pt', map_location='cpu', weights_only=True); model.load_state_dict(state['model'])
        scores, _ = predict(model, bank, test); cal = json.loads((run/'CALIBRATION.json').read_text())
        pairrows, queryrows, f = report_metrics(test, scores, cal, arm, seed)
        rows.extend(dict(variant=variant, **x) for x in pairrows)
        queries.extend(dict(variant=variant, **x) for x in queryrows)
        key = f'{arm}__{variant}__{seed}'; predictions[key+'_score'] = scores
        predictions[key+'_prob'] = expit(cal['slope']*scores+cal['intercept'])
        for panel in ['AFFINITY_KD_KI', 'ACTIVITY_IC50', 'ACTIVITY_EC50']:
            sub = test[test.panel.eq(panel)].reset_index(drop=True)
            s = scores[test.panel.eq(panel).to_numpy()]
            for direction, column in [('target', 'target_id'), ('drug', 'molecule_id')]:
                try:
                    metric = QueryMetrics(sub, column).evaluate(s)
                except ValueError:
                    continue
                queries.append(dict(arm=arm, variant=variant, seed=seed, panel=panel,
                                    direction='macro_'+direction, query='ALL_ELIGIBLE', **metric))
        subreg = reg[reg.endpoint.isin(ENDPOINTS[arm])].reset_index(drop=True)
        _, rp = predict(model, bank, subreg)
        for endpoint, metric in regression_metrics(subreg, rp, datasets[arm]).items():
            regrows.append(dict(arm=arm, variant=variant, seed=seed, endpoint=endpoint,
                               auxiliary_trained=VARIANTS[variant][1], **metric))
        status(stage='test_evaluation', arm=arm, variant=variant, seed=seed)
        del model, state; torch.cuda.empty_cache()
    metrics = pd.DataFrame(rows); metrics.to_csv(OUT/'TEST_METRICS.csv', index=False)
    pd.DataFrame(queries).to_csv(OUT/'TEST_QUERY_METRICS.csv.gz', index=False)
    pd.DataFrame(regrows).to_csv(OUT/'TEST_REGRESSION_METRICS.csv', index=False)
    predictions.to_parquet(OUT/'TEST_PREDICTIONS.parquet', index=False, compression='zstd')
    metrics.groupby(['arm', 'variant', 'panel', 'scope'])[['auroc', 'ap', 'false_positive_rate', 'brier']].agg(['mean', 'std']).to_csv(OUT/'TEST_SUMMARY.csv')
    pd.DataFrame([{k:r[k] for k in ['arm', 'variant', 'seed', 'optimizer_steps', 'best_step', 'complete_passes', 'seconds']}
                  for r in results]).to_csv(OUT/'CONVERGENCE_SUMMARY.csv', index=False)
    manifest = json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for path, h in manifest['frozen_inputs'].items():
        assert digest(ROOT/path) == h, path
    write_json(OUT/'SUMMARY.json', dict(status='COMPLETE_24_VALIDATION_CONVERGED_FITS', completed_utc=now(),
               runs=24, production_replaced=False, wetlab_unchanged=True,
               protocol_sha256=digest(OUT/'PROTOCOL.json'), limitations=p['limitations'],
               no_claim_of_optimal_auxiliary_weights=True))
    status(stage='complete', runs=24)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'RUN.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(OUT/'PROCESS.json', dict(pid=os.getpid(), started_utc=now(), command=sys.argv))
    try:
        status(stage='preparing_and_verifying_inputs'); prepare()
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = True
        p = json.loads((OUT/'PROTOCOL.json').read_text()); config = json.loads((OUT/'MODEL_CONFIG.json').read_text())
        ident = identity(); bank = Bank(); datasets = {}
        validation = Validation()
        for arm in ARMS:
            status(stage='loading_training_queries', arm=arm)
            datasets[arm] = ArmData(arm)
        if not (OUT/'THROUGHPUT_ETA.json').exists():
            benchmark(bank, datasets, validation, p, config)
        if args.benchmark_only:
            status(stage='benchmark_complete_not_training'); return
        results = []; fit_index = 0
        for seed in p['seeds']:
            for variant in VARIANTS:
                for arm in ARMS:
                    fit_index += 1
                    result = fit(datasets[arm], variant, seed, bank, validation, p, config, ident, fit_index)
                    results.append(result)
                    write_json(OUT/'QUEUE_PROGRESS.json', dict(updated_utc=now(), completed_fits=len(results), total_fits=24,
                               runs=[{k:r[k] for k in ['arm', 'variant', 'seed', 'status', 'seconds', 'optimizer_steps']} for r in results]))
        evaluate_all(bank, datasets, p, config, results, ident)
    except Exception:
        write_json(OUT/'ERROR.json', dict(time_utc=now(), pid=os.getpid(), traceback=traceback.format_exc()))
        status(stage='failed', error_file=str(OUT/'ERROR.json')); raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Freeze 2 x 4 x 3 objective experiments using existing A/B pair memberships."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from prepare_endpoint_ablation_20260911 import DATA, OUT as SOURCE, FEATURES, BUNDLE, write_json
from biomaster.endpoint_ablation import ARMS
from biomaster.endpoint_multitask import ENDPOINTS, VARIANTS
from biomaster.portable_ranker_v2 import digest

OUT = ROOT/'outputs/biomaster_endpoint_multitask_20260911'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'DATA_MANIFEST.json').exists():
        m = json.loads((OUT/'DATA_MANIFEST.json').read_text())
        for p, h in m['inputs'].items():
            assert digest(ROOT/p) == h, p
        for p, h in m['prepared'].items():
            assert digest(OUT/p) == h, p
        assert digest(OUT/'PROTOCOL.json') == m['protocol_sha256']
        return m
    source_manifest = json.loads((SOURCE/'DATA_MANIFEST.json').read_text())
    for p, h in source_manifest['files'].items():
        assert digest(SOURCE/p) == h, p
    for p, h in source_manifest['frozen_inputs'].items():
        assert digest(ROOT/p) == h, p
    protocol = dict(status='FROZEN_BEFORE_NEW_TRAINING', arms=ARMS, variants=VARIANTS,
        seeds=[20260921, 20260922, 20260923], total_fits=24,
        initialization='fresh identical base initialization per seed; no supervised parent weights',
        endpoints=ENDPOINTS, batch_size=1024, learning_rate=3e-4, weight_decay=1e-4, ema_decay=.995,
        warmup_steps=200, rank_queries_per_task_direction=2, rank_candidates_per_class=16,
        regression_rows_per_endpoint=128, rank_temperature=1.,
        target_rank_weight=.1, drug_rank_weight=.1, regression_weight=.1,
        weight_policy='Fixed preregistered starting weights; no test tuning or claim of optimal loss weights.',
        binary_objective='Original prevalence-balanced observed BCE on both heads; all original rows cycle before repeating.',
        rank_objective='Mean-logit pairwise softplus; equal queries, equal numeric tasks, separate drug and target directions.',
        inactivity='Original explicit negatives remain in BCE; no unknown-negative sampling or assigning generic inactivity to an endpoint.',
        regression_objective='Exact same-member endpoint values; train-only z-score and endpoint-balanced Huber; no censored or fabricated targets.',
        validation_interval='max(1000, ceil(2 * train_pairs / batch_size)) optimizer steps',
        checkpoint_selection='Mean of Kd/Ki validation target-macro AP and drug-macro AP, each query >= 10 and both labels; maximum across checkpoints.',
        convergence=dict(minimum_complete_passes=10, improvement_delta=.0002, lr_reduce_every_stale_checks=4,
                         lr_factor=.5, minimum_lr=.000005, stale_checks_to_stop=12, minimum_lr_reductions_since_improvement=2),
        minimum_lr_exception='A plateau already at minimum LR may stop without two further impossible LR reductions.',
        emergency_max_steps=200000,
        emergency_policy='Stop with NOT_CONVERGED_NEEDS_EXTENSION, never report a budget stop as convergence.',
        test_policy='No new test scores until all 24 validation-selected fits converge and checkpoint hashes are frozen.',
        comparisons='All seeds; same A/B members, frozen features, BCE definition, validation queries and score convention. Report differing sample exposure and auxiliary computation.',
        limitations=['Validation plateau is an operational early-stopping criterion, not proof of a global optimum.',
                     'Same-task ranking does not certify same assay conditions; Kd/Ki remain one binary task.',
                     'Existing test panels were inspected in prior research; this is not a new pristine prospective test.',
                     'No automatic replacement of the frozen SPR list or production scorer.'])
    if (OUT/'PROTOCOL.json').exists():
        assert json.loads((OUT/'PROTOCOL.json').read_text()) == json.loads(json.dumps(protocol)), 'protocol drift'
    else:
        write_json(OUT/'PROTOCOL.json', protocol)
    state = torch.load(BUNDLE/'model.pt', map_location='cpu', weights_only=True)
    write_json(OUT/'MODEL_CONFIG.json', state['config'])
    paths = [SOURCE/'DATA_MANIFEST.json', SOURCE/'PROTOCOL.json', FEATURES/'MANIFEST.json',
             DATA/'TRAIN.parquet', DATA/'TRAIN_REGRESSION.parquet', DATA/'VALIDATION_REGRESSION.parquet',
             DATA/'TEST_REGRESSION.parquet', SOURCE/'COMMON_VALIDATION.parquet', SOURCE/'COMMON_TEST.parquet']
    tasks = pd.read_parquet(DATA/'TRAIN.parquet')
    regs = {s: pd.read_parquet(DATA/(s.upper()+'_REGRESSION.parquet')) for s in ['train', 'validation']}
    validation = pd.read_parquet(SOURCE/'COMMON_VALIDATION.parquet')
    test = pd.read_parquet(SOURCE/'COMMON_TEST.parquet', columns=['pair_id', 'split_group'])
    stats, prepared, checks = [], [OUT/'MODEL_CONFIG.json'], {}
    done_d = np.load(FEATURES/'DRUG_DONE.npy') & np.load(FEATURES/'CHEM_DONE.npy')
    done_t = np.load(FEATURES/'TARGET_DONE.npy')
    feature_manifest = json.loads((FEATURES/'MANIFEST.json').read_text())
    for name, info in feature_manifest['files'].items():
        assert digest(FEATURES/name) == info['sha256'], name
    for arm, selected in ARMS.items():
        frames = {}
        for split in ['train', 'validation']:
            path = SOURCE/f'{arm}_{split.upper()}.parquet'; paths.append(path)
            f = pd.read_parquet(path); frames[split] = f
            assert f.pair_id.is_unique and f.split.eq(split).all()
            assert done_d[f.drug_feature_index.to_numpy()].all() and done_t[f.target_feature_index.to_numpy()].all()
            r = regs[split]
            r = r[r.pair_id.isin(f.pair_id) & r.endpoint.isin(ENDPOINTS[arm])].copy()
            assert r.regression_eligible.all() and r.split.eq(split).all()
            assert np.isfinite(r.median_p_activity_unique).all() and not r.duplicated(['pair_id', 'endpoint']).any()
            assert r.label_status.ne('CONFLICT').all() and not r.interval_review.any()
            assert (r.max_p_activity-r.min_p_activity).le(1.).all()
            joined = r.merge(f[['pair_id', 'drug_feature_index', 'target_feature_index', 'split_group']],
                             on='pair_id', validate='many_to_one', suffixes=('', '_arm'))
            for col in ['drug_feature_index', 'target_feature_index', 'split_group']:
                assert joined[col].eq(joined[col+'_arm']).all(), col
            path = OUT/f'{arm}_{split.upper()}_REGRESSION.parquet'
            r[['pair_id', 'drug_feature_index', 'target_feature_index', 'endpoint', 'median_p_activity_unique']].to_parquet(path, index=False)
            prepared.append(path)
            for endpoint, g in r.groupby('endpoint'):
                stats.append(dict(arm=arm, split=split, type='exact_regression', group=endpoint, rows=len(g)))
        f = frames['train']
        assert not set(f.pair_id) & (set(validation.pair_id) | set(test.pair_id))
        assert not set(f.split_group) & (set(validation.split_group) | set(test.split_group))
        pool = tasks[tasks.pair_id.isin(f.pair_id) & tasks.task.isin(selected)].copy()
        assert pool.binary_label.isin([0, 1]).all() and not pool.duplicated(['pair_id', 'task']).any()
        j = pool.merge(f[['pair_id', 'drug_feature_index', 'target_feature_index', 'binary_label']],
                       on='pair_id', validate='many_to_one', suffixes=('', '_arm'))
        for col in ['drug_feature_index', 'target_feature_index', 'binary_label']:
            assert j[col].eq(j[col+'_arm']).all(), col
        path = OUT/f'{arm}_RANK_TASKS.parquet'
        pool[['pair_id', 'drug_feature_index', 'target_feature_index', 'binary_label', 'task']].to_parquet(path, index=False)
        prepared.append(path)
        stats.append(dict(arm=arm, split='train', type='binary', group='ALL', rows=len(f)))
        for task, g in pool.groupby('task'):
            for col in ['drug_feature_index', 'target_feature_index']:
                q = g.groupby(col).binary_label.agg(['size', 'sum'])
                stats.append(dict(arm=arm, split='train', type='rank_queries', group=task+'__'+col,
                                  rows=int((q['sum'].gt(0) & q['sum'].lt(q['size'])).sum())))
        checks[arm+'_same_member_supervision_and_split_isolation'] = True
    stats_path = OUT/'SUPERVISION_COUNTS.csv'; pd.DataFrame(stats).to_csv(stats_path, index=False); prepared.append(stats_path)
    manifest = dict(status='COMPLETE', inputs={str(p.relative_to(ROOT)):digest(p) for p in paths},
        prepared={p.name:digest(p) for p in prepared}, frozen_inputs=source_manifest['frozen_inputs'],
        protocol_sha256=digest(OUT/'PROTOCOL.json'), producer_sha256=digest(Path(__file__)),
        checks=checks, feature_files_verified=True, no_new_training_pairs=True)
    write_json(OUT/'DATA_MANIFEST.json', manifest)
    print(json.dumps(dict(stage='data_prepared', checks=checks, output=str(OUT)), ensure_ascii=False), flush=True)
    return manifest


if __name__ == '__main__':
    prepare()

#!/usr/bin/env python3
"""Diagnose existing neural/tree comparison from saved training and validation artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DT = ROOT / 'outputs/biomaster_dtiam_ab_20260912'
MT = ROOT / 'outputs/biomaster_endpoint_multitask_20260911'
AB = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'
OUT = ROOT / 'outputs/palinova_dtiam_gap_audit_20260915'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def health(bank, ids):
    finite, nonzero = 0, 0
    for start in range(0, len(ids), 8192):
        x = bank[ids[start:start + 8192]]
        finite += int(np.isfinite(x).all(axis=1).sum())
        nonzero += int(np.any(x != 0, axis=1).sum())
    return dict(rows=len(ids), finite_rows=finite, nonzero_rows=nonzero)


def main():
    OUT.mkdir(exist_ok=True)
    dm = json.loads((MT / 'DATA_MANIFEST.json').read_text())
    protocol = json.loads((DT / 'PROTOCOL.json').read_text())
    datasets, training, learners, health_rows, sources = [], [], [], [], {}
    clip = np.load(AB / 'features/DRUG_CLIP.npy', mmap_mode='r')
    bert = np.load(DT / 'features/BERMOL.npy', mmap_mode='r')
    target = np.load(AB / 'features/TARGET.npy', mmap_mode='r')
    dt_target = np.load(DT / 'features/ESM2.npy', mmap_mode='r')
    available = np.load(AB / 'features/AVAILABLE.npy', mmap_mode='r')
    for spec in protocol['arms']:
        arm = spec['arm']; path = AB / f'{arm}_TRAIN.parquet'
        digest = sha(path)
        assert digest == spec['train_sha256'] == dm['inputs'][str(path.relative_to(ROOT))]
        f = pd.read_parquet(path)
        assert len(f) == spec['pairs'] and int(f.binary_label.sum()) == spec['positive']
        datasets.append(dict(arm=arm, pairs=len(f), positive=int(f.binary_label.sum()),
            negative=int((f.binary_label == 0).sum()), molecules=f.molecule_id.nunique(),
            target_sequences=f.target_id.nunique(), explicit_inactive=spec['explicit_inactive'],
            train_sha256=digest, training_pool_identical=True,
            neural_positive_weight=.5 / f.binary_label.mean(),
            neural_negative_weight=.5 / (1 - f.binary_label.mean())))
        ids = np.unique(f.drug_feature_index.to_numpy(int))
        tids = np.unique(f.target_feature_index.to_numpy(int))
        for name, bank, index in [('Palinova_DrugCLIP', clip, ids), ('DTIAM_BerMol', bert, ids),
                                 ('Palinova_ESM2', target, tids), ('DTIAM_ESM2', dt_target, tids)]:
            health_rows.append(dict(arm=arm, feature=name, **health(bank, index)))
        datasets[-1]['palinova_drugclip_available_molecules'] = int(available[ids].sum())
        missing_clip = ~available[f.drug_feature_index.to_numpy(int)]
        datasets[-1]['pairs_without_drugclip'] = int(missing_clip.sum())
        datasets[-1]['pair_fraction_without_drugclip'] = float(missing_clip.mean())
        for seed in [20260921, 20260922, 20260923]:
            nr = MT / f'{arm}__binary__seed_{seed}'
            dr = DT / f'{arm}__seed_{seed}'
            result = json.loads((nr / 'RESULT.json').read_text())
            history = json.loads((nr / 'HISTORY.json').read_text())
            fit = json.loads((dr / 'FIT_RETURNED.json').read_text())
            members = pd.read_parquet(dr / 'INTERNAL_MEMBER_ROLES.parquet')
            assert f[['pair_id', 'binary_label']].equals(members[['pair_id', 'binary_label']])
            assert fit['base_train_rows'] + fit['internal_holdout_rows'] == len(f)
            training.append(dict(arm=arm, seed=seed, neural_status=result['status'],
                neural_parameters=result['parameters'], neural_optimizer_steps=result['optimizer_steps'],
                neural_best_step=result['best_step'], neural_complete_passes=result['complete_passes'],
                neural_seconds=result['seconds'], neural_last_streaming_bce=history[-1]['training_losses']['binary'],
                neural_best_validation_ap=result['best_validation']['panels']['AFFINITY_KD_KI']['ap'],
                neural_best_validation_query_ap=result['best_validation']['selection_score'],
                dtiam_member_pool=fit['training_member_pool'], dtiam_base_train_rows=fit['base_train_rows'],
                dtiam_internal_holdout_rows=fit['internal_holdout_rows'], dtiam_fit_seconds=fit['fit_seconds'],
                dtiam_inner_union_exactly_matches_neural_training=True))
            l = pd.read_csv(dr / 'VALIDATION_LEADERBOARD.csv')
            learners.extend(dict(arm=arm, seed=seed, **r) for r in l.to_dict('records'))
            learners.append(dict(arm=arm, seed=seed, model='Palinova_binary',
                ap=training[-1]['neural_best_validation_ap'],
                selection_score=training[-1]['neural_best_validation_query_ap']))
            for p in [nr / 'RESULT.json', nr / 'HISTORY.json', dr / 'FIT_RETURNED.json',
                      dr / 'INTERNAL_MEMBER_ROLES.parquet', dr / 'VALIDATION_LEADERBOARD.csv']:
                sources[str(p.relative_to(ROOT))] = sha(p)
        sources[str(path.relative_to(ROOT))] = digest
    pd.DataFrame(datasets).to_csv(OUT / 'TRAINING_DATA_EQUALITY.csv', index=False)
    pd.DataFrame(training).to_csv(OUT / 'TRAINING_EXPOSURE_AND_OBJECTIVES.csv', index=False)
    pd.DataFrame(health_rows).to_csv(OUT / 'TRAINING_FEATURE_HEALTH.csv', index=False)
    lf = pd.DataFrame(learners)
    lf.to_csv(OUT / 'ALL_LEARNERS_VALIDATION_PER_SEED.csv', index=False)
    means = lf.groupby(['arm', 'model'])[['ap', 'selection_score']].agg(['mean', 'std', 'count'])
    means.columns = ['_'.join(c) for c in means.columns]
    means.reset_index().to_csv(OUT / 'ALL_LEARNERS_VALIDATION_SUMMARY.csv', index=False)
    # Previously viewed TEST: descriptive availability audit only, not model selection.
    test = pd.read_parquet(AB / 'COMMON_TEST.parquet')
    neural = pd.read_parquet(MT / 'TEST_PREDICTIONS.parquet')
    keys = ['panel', 'pair_id', 'binary_label']
    assert test[keys].equals(neural[keys])
    kd = test.panel.eq('AFFINITY_KD_KI').to_numpy()
    has_clip = available[test.drug_feature_index.to_numpy(int)]
    availability_rows = []
    for seed in [20260921, 20260922, 20260923]:
        path = DT / f'kdki_inactive__seed_{seed}/TEST_PREDICTIONS.parquet'
        pred = pd.read_parquet(path)
        assert test[keys].equals(pred[keys])
        for name, scores in [('Palinova_A', neural[f'kdki_inactive__binary__{seed}_score'].to_numpy()),
                             ('DTIAM_A_query', pred.query_score.to_numpy())]:
            for scope, mask in [('all', kd), ('DrugCLIP_available', kd & has_clip),
                                ('DrugCLIP_unavailable', kd & ~has_clip)]:
                y = test.binary_label.to_numpy()[mask]
                availability_rows.append(dict(model=name, seed=seed, scope=scope, pairs=len(y),
                    positive=int(y.sum()), ap=average_precision_score(y, scores[mask]),
                    auroc=roc_auc_score(y, scores[mask])))
        sources[str(path.relative_to(ROOT))] = sha(path)
    pd.DataFrame(availability_rows).to_csv(OUT / 'DRUGCLIP_AVAILABILITY_TEST_DIAGNOSTIC.csv', index=False)
    for p in [AB / 'COMMON_TEST.parquet', MT / 'TEST_PREDICTIONS.parquet', MT / 'MODEL_CONFIG.json',
              AB / 'features/DRUGCLIP_COMPLETE.json', AB / 'features/MANIFEST.json', DT / 'features/MANIFEST.json',
              MT / 'DATA_MANIFEST.json', DT / 'PROTOCOL.json']:
        sources[str(p.relative_to(ROOT))] = sha(p)
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result = dict(status='COMPLETE_READ_ONLY_GAP_AUDIT', sources=sources,
        same_supervised_members_labels_and_outer_splits=True, pretraining_data_equivalence_claimed=False,
        feature_health=health_rows, observed_configuration=json.loads((MT / 'MODEL_CONFIG.json').read_text()),
        verified_architecture='Global branch projects molecule 2601->192, protein1280->192, merges d/t/product/absolute difference, one global fusion residual and one additional residual; local token/pocket blocks inactive.',
        note='global variant ignores capacity_hidden and cfg.blocks for residual depth; increasing only those config values would not deepen the active branch.',
        limitations=['Learner comparison is validation, not new TEST inference for unselected base learners',
            'Availability strata use the previously inspected TEST for retrospective diagnosis, not model selection',
            'Changing feature families, pooling, heads, weights and model search together does not isolate causal contribution',
            'Streaming training BCE is not an independently rescored train/validation loss gap',
            'GPU neural and CPU AutoGluon elapsed times do not measure equal compute budgets'],
        training_started=False, production_changed=False)
    (OUT / 'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(datasets).to_string(index=False))
    print(means.loc['kdki_inactive'].to_string())
    print(pd.DataFrame(health_rows).to_string(index=False))


if __name__ == '__main__':
    main()

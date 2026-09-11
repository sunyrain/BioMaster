#!/usr/bin/env python3
"""Score frozen production weights on the existing endpoint test, without training."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_endpoint_ablation_20260911 import Bank, basic, calibrate
from prepare_endpoint_ablation_20260911 import BUNDLE, DATA, OUT, write_json
from biomaster.model_registry import build_model
from biomaster.portable_ranker_v2 import digest

DEST = ROOT / 'outputs/biomaster_old_production_comparison_20260911'


@torch.inference_mode()
def score_heads(model, bank, frame):
    chunks = []
    for start in range(0, len(frame), 2048):
        f = frame.iloc[start:start + 2048]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            s = model(bank.batch(f.drug_feature_index.to_numpy(), f.target_feature_index.to_numpy()))
        chunks.append(s.float().cpu().numpy())
    scores = np.concatenate(chunks)
    assert scores.shape == (len(frame), 2) and np.isfinite(scores).all()
    return scores


def main():
    DEST.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    protocol = json.loads((OUT / 'PROTOCOL.json').read_text())
    frozen_paths = [BUNDLE / 'model.pt', OUT / 'COMMON_TEST.parquet', OUT / 'COMMON_VALIDATION.parquet',
                    OUT / 'TEST_PREDICTIONS.parquet', OUT / 'features/MANIFEST.json']
    frozen_paths += list((ROOT / 'outputs/spr384_final_experiment_table_20260910').glob('*.csv'))
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    assert digest(BUNDLE / 'model.pt') == protocol['production_model_sha256']
    test = pd.read_parquet(OUT / 'COMMON_TEST.parquet').reset_index(drop=True)
    val = pd.read_parquet(OUT / 'COMMON_VALIDATION.parquet').reset_index(drop=True)
    predictions = pd.read_parquet(OUT / 'TEST_PREDICTIONS.parquet')
    pd.testing.assert_frame_equal(test[['panel', 'pair_id', 'binary_label']],
                                  predictions[['panel', 'pair_id', 'binary_label']], check_dtype=False)
    exposure = pd.read_parquet(DATA / 'ALL_PAIR_TASK_LABELS.parquet',
                              columns=['pair_id', 'seen_by_frozen_parent_connectivity_pair']).drop_duplicates()
    assert exposure.pair_id.is_unique
    for name, f in [('test', test), ('validation', val)]:
        joined = f[['pair_id']].merge(exposure, on='pair_id', how='left', validate='many_to_one')
        assert joined.seen_by_frozen_parent_connectivity_pair.notna().all()
        f['old_train_pair_seen'] = joined.seen_by_frozen_parent_connectivity_pair.astype(bool).to_numpy()
    predictions['old_train_pair_seen'] = test.old_train_pair_seen
    bank = Bank()
    state = torch.load(BUNDLE / 'model.pt', map_location='cpu', weights_only=True)
    model = build_model(state['architecture'], state['config']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    print('Scoring frozen production checkpoint', flush=True)
    vs = score_heads(model, bank, val)
    ts = score_heads(model, bank, test)
    cal = calibrate(val, vs.mean(axis=1))
    cal['old_training_overlap_in_validation_kdki'] = int(val.loc[val.panel.eq('AFFINITY_KD_KI'), 'old_train_pair_seen'].sum())
    cal['note'] = 'New validation-only calibration for common threshold metrics; production checkpoint unchanged. Validation may overlap old training.'
    write_json(DEST / 'CALIBRATION.json', cal)
    predictions['old_production_score'] = ts.mean(axis=1)
    predictions['old_production_head0'] = ts[:, 0]
    predictions['old_production_head1'] = ts[:, 1]
    predictions['old_production_prob'] = expit(cal['slope'] * predictions.old_production_score + cal['intercept'])
    predictions.to_parquet(DEST / 'TEST_PREDICTIONS.parquet', index=False, compression='zstd')
    rows, diagnostics = [], []
    specs = [('old_production', 0, 'old_production_score', 'old_production_prob', cal['threshold'])]
    for arm in ['kdki_inactive', 'all_inactive']:
        for seed in protocol['seeds']:
            c = json.loads((OUT / f'{arm}_seed_{seed}/CALIBRATION.json').read_text())
            specs.append((arm, seed, f'{arm}_{seed}_score', f'{arm}_{seed}_prob', c['threshold']))
    for panel, g in predictions.groupby('panel', sort=False):
        scopes = [('all', g), ('old_train_pair_seen', g[g.old_train_pair_seen]),
                  ('old_train_pair_unseen', g[~g.old_train_pair_seen]),
                  ('document_disjoint_new_training', g[g.document_disjoint])]
        for scope, sub in scopes:
            if sub.empty:
                continue
            for arm, seed, scorecol, probcol, threshold in specs:
                rows.append(dict(model=arm, seed=seed, panel=panel, scope=scope,
                                 **basic(sub.binary_label, sub[scorecol], sub[probcol], threshold)))
            for head in [0, 1]:
                s = sub[f'old_production_head{head}']
                b = basic(sub.binary_label, s, np.full(len(sub), .5), .5)
                diagnostics.append(dict(panel=panel, scope=scope, head=head,
                                        **{k: b[k] for k in ['pairs', 'positive', 'negative', 'auroc', 'ap', 'top1pct_precision', 'top5pct_precision']}))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(DEST / 'TEST_METRICS.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(DEST / 'OLD_HEAD_DIAGNOSTICS.csv', index=False)
    keys = ['pairs', 'positive', 'negative', 'auroc', 'ap', 'top1pct_precision', 'top5pct_precision', 'false_positive_rate', 'brier']
    comparison = metrics.groupby(['panel', 'scope', 'model'], sort=False)[keys].mean().reset_index()
    comparison.to_csv(DEST / 'COMPARISON.csv', index=False)
    assert all(digest(ROOT / p) == h for p, h in frozen.items())
    write_json(DEST / 'MANIFEST.json', dict(
        status='COMPLETE', frozen_inputs=frozen, producer_sha256=digest(Path(__file__)),
        checkpoint_loaded_strictly=True, identical_panel_order_and_labels=True,
        all_predictions_finite=True, score='mean of both output logits, bf16 autocast, same as A/B',
        new_model_summary='arithmetic mean of 3 seed metrics, not ensemble predictions',
        old_model_count=1, retrained=False, production_and_wetlab_unchanged=True,
        exposure_definition='old supervised molecular connectivity plus exact protein sequence pair',
        limitations=['Full test contains old training pairs; report seen/unseen separately.',
                     'Pair unseen does not certify molecule/scaffold/citation or encoder-pretraining unseen.',
                     'Threshold metrics use new validation-only calibration, not deployed probability.',
                     'Pooled measured-pair classification is not full-catalog directional retrieval or prospective SPR.'],
        artifacts={p.name: digest(p) for p in DEST.iterdir() if p.is_file() and p.name != 'MANIFEST.json'}))
    print(comparison[comparison.scope.isin(['all', 'old_train_pair_unseen'])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

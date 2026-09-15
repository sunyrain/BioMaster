#!/usr/bin/env python3
"""Audit completed DTIAM TEST against fixed Palinova baselines; no fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from dtiam_ab_common_20260912 import OUT, SOURCE, SEEDS, ARMS, basic, digest, query_summary, write_json

DEST = OUT / 'final_review_20260915'
PARENT = ROOT / 'outputs/biomaster_endpoint_multitask_20260911'
BLEND = ROOT / 'outputs/palinova_a_train_only_blend_20260914'
KEYS = ['pair_id', 'panel', 'binary_label']


def main():
    summary = json.loads((OUT / 'SUMMARY.json').read_text())
    assert summary['status'] == 'COMPLETE_6_NATIVE_DTIAM_SUITES'
    assert json.loads((OUT / 'STATUS.json').read_text())['stage'] == 'COMPLETE'
    gate = json.loads((OUT / 'TEST_GATE.json').read_text())
    for path, expected in gate['selection_hashes'].items():
        assert digest(ROOT / path) == expected
    DEST.mkdir(exist_ok=True)
    frame = pd.read_parquet(SOURCE / 'COMMON_TEST.parquet')
    prior = pd.read_parquet(PARENT / 'TEST_PREDICTIONS.parquet')
    assert frame[KEYS].equals(prior[KEYS])
    ledger = pd.read_csv(ROOT / 'outputs/biomaster_model_consolidation_20260911/MODEL_LEDGER.csv')
    registry = json.loads((ROOT / 'outputs/biomaster_model_consolidation_20260911/RECOMMENDED_MODELS.json').read_text())
    support = pd.read_parquet(BLEND / 'COMMON_TEST_BLEND_PREDICTIONS.parquet')
    supported = frame[KEYS].merge(support[['pair_id', 'panel', 'eligible_positive_neighbors', 'positive_neighbor']],
                                 on=['pair_id', 'panel'], how='left', validate='one_to_one')
    kd = frame.panel.eq('AFFINITY_KD_KI').to_numpy()
    inactive = frame.panel.eq('EXPLICIT_INACTIVE').to_numpy()
    rows, endpoint_rows, strata_rows, sources = [], [], [], {}

    def collect(family, arm, variant, seed, score, prob, threshold, expected_ap=None):
        score, prob = np.asarray(score, float), np.asarray(prob, float)
        metrics = basic(frame.loc[kd, 'binary_label'], score[kd], prob[kd], threshold)
        if expected_ap is not None:
            assert abs(metrics['ap'] - expected_ap) < 1e-10
        q = query_summary(frame[kd].reset_index(drop=True), score[kd])
        identity = dict(family=family, arm=arm, variant=variant, seed=seed)
        row = dict(**identity, ap=metrics['ap'], auroc=metrics['auroc'], recall=metrics['recall'],
                   kdki_fpr=metrics['false_positive_rate'],
                   inactive_fpr=float((prob[inactive] >= threshold).mean()),
                   target_macro_ap=q['target']['macro_ap'], drug_macro_ap=q['drug']['macro_ap'],
                   balanced_query_ap=q['selection_score'],
                   target_p5=q['target']['p5'], drug_p5=q['drug']['p5'])
        for panel, ix in frame.groupby('panel').indices.items():
            m = basic(frame.binary_label.to_numpy()[ix], score[ix], prob[ix], threshold)
            endpoint_rows.append(dict(**identity, panel=panel, **m))
            if panel in ['ACTIVITY_IC50', 'ACTIVITY_EC50']:
                row[panel.removeprefix('ACTIVITY_').lower() + '_ap'] = m['ap']
        rows.append(row)
        for panel_mask, panel in [(kd, 'AFFINITY_KD_KI'), (inactive, 'EXPLICIT_INACTIVE')]:
            assert supported.loc[panel_mask, 'positive_neighbor'].notna().all()
            for label, mask in [
                ('all', panel_mask),
                ('no_A_training_positive_reference', panel_mask & supported.eligible_positive_neighbors.eq(0).to_numpy()),
                ('A_positive_neighbor_lt_0.3', panel_mask & supported.positive_neighbor.lt(.3).to_numpy()),
                ('A_positive_neighbor_ge_0.3', panel_mask & supported.positive_neighbor.ge(.3).to_numpy())]:
                strata_rows.append(dict(**identity, panel=panel, stratum=label,
                    **basic(frame.loc[mask, 'binary_label'], score[mask], prob[mask], threshold)))

    auto = pd.read_csv(OUT / 'MODEL_LEDGER.csv')
    for arm in ARMS:
        for seed in SEEDS:
            run = OUT / f'{arm}__seed_{seed}'
            completed = json.loads((run / 'TEST_COMPLETE.json').read_text())
            selection = json.loads((run / 'SELECTION.json').read_text())
            assert digest(run / 'TEST_PREDICTIONS.parquet') == completed['predictions_sha256']
            assert digest(run / 'SELECTION.json') == completed['selection_sha256']
            assert digest(run / 'CALIBRATION.json') == selection['calibration_sha256']
            pred = pd.read_parquet(run / 'TEST_PREDICTIONS.parquet')
            assert frame[KEYS].equals(pred[KEYS])
            cal = json.loads((run / 'CALIBRATION.json').read_text())
            for view in ['native', 'query']:
                expected = auto[(auto.arm == arm) & (auto.seed == seed) & (auto.variant == view)].iloc[0]
                collect('DTIAM', arm, view, seed, pred[view + '_score'], pred[view + '_prob'],
                        cal[view]['threshold'], expected.ap)
            sources[str((run / 'TEST_COMPLETE.json').relative_to(ROOT))] = digest(run / 'TEST_COMPLETE.json')
    for arm in ARMS:
        for variant in ['binary', 'joint']:
            for seed in SEEDS:
                prefix = f'{arm}__{variant}__{seed}'
                run = PARENT / f'{arm}__{variant}__seed_{seed}'
                cal = json.loads((run / 'CALIBRATION.json').read_text())
                expected = ledger[(ledger.stage == 'multitask') & (ledger.arm == arm) &
                                  (ledger.variant == variant) & (ledger.seed == seed)].iloc[0]
                collect('Palinova', arm, variant, seed, prior[prefix + '_score'], prior[prefix + '_prob'],
                        cal['threshold'], expected.ap)

    # The blend only scored the two direct-binding/explicit-inactive panels.
    # Keep its one-checkpoint comparison separate from the three-seed family table.
    selected = json.loads((BLEND / 'SELECTION.json').read_text())
    blend_rows = []
    for name in ['A_neural', 'A_plus_neighbor', 'A_plus_prior', 'A_joint_blend']:
        c = selected['calibrations'][name]
        for panel, f in support.groupby('panel'):
            score = f[name + '_score'].to_numpy(float)
            prob = expit(c['slope'] * score + c['intercept'])
            m = basic(f.binary_label, score, prob, c['threshold'])
            q = query_summary(f.reset_index(drop=True), score) if panel == 'AFFINITY_KD_KI' else None
            blend_rows.append(dict(model=name, seed=20260921, panel=panel, **m,
                target_macro_ap=q['target']['macro_ap'] if q else None,
                drug_macro_ap=q['drug']['macro_ap'] if q else None,
                balanced_query_ap=q['selection_score'] if q else None))

    raw = pd.DataFrame(rows)
    raw.to_csv(DEST / 'COMMON_TEST_PER_SEED.csv', index=False)
    numeric = [c for c in raw.columns if c not in ['family', 'arm', 'variant', 'seed']]
    family = raw.groupby(['family', 'arm', 'variant'])[numeric].agg(['mean', 'std', 'count'])
    family.columns = ['_'.join(c) for c in family.columns]
    family = family.reset_index()
    family.to_csv(DEST / 'COMMON_TEST_FAMILY_COMPARISON.csv', index=False)
    pd.DataFrame(endpoint_rows).to_csv(DEST / 'ENDPOINT_COMPARISON_PER_SEED.csv', index=False)
    pd.DataFrame(strata_rows).to_csv(DEST / 'TRAINING_SUPPORT_DIAGNOSTIC_PER_SEED.csv', index=False)
    pd.DataFrame(blend_rows).to_csv(DEST / 'PREVIOUS_SINGLE_SEED_BLEND_REFERENCE.csv', index=False)
    for p in [Path(__file__), OUT / 'SUMMARY.json', OUT / 'TEST_GATE.json', SOURCE / 'COMMON_TEST.parquet',
              PARENT / 'TEST_PREDICTIONS.parquet', BLEND / 'SELECTION.json',
              BLEND / 'COMMON_TEST_BLEND_PREDICTIONS.parquet']:
        sources[str(p.relative_to(ROOT))] = digest(p)
    write_json(DEST / 'AUDIT.json', dict(status='COMPLETE_READ_ONLY_REVIEW', input_hashes=sources,
        validations=['All six selections match the pre-test gate', 'Every TEST identity/label matches common TEST',
                     'DTIAM and neural AP reproduce saved ledgers', 'Original neural and blend thresholds reused'],
        kdki_pairs=int(kd.sum()), kdki_positive=int(frame.loc[kd, 'binary_label'].sum()),
        kdki_negative=int((frame.loc[kd, 'binary_label'] == 0).sum()),
        kdki_random_ap=float(frame.loc[kd, 'binary_label'].mean()), inactive_pairs=int(inactive.sum()),
        neural_representatives=registry['roles'],
        caveats=['Seed means are not an ensemble of predictions',
                 'Support strata are retrospective; 0.3 is not a deployment boundary',
                 'Observed-panel metrics do not estimate SPR384 hit rate',
                 'DTIAM versus Palinova compares full methods, including embeddings/inner split/weights',
                 'DTIAM used the common TEST only after freezing all selections; this existing benchmark was seen in prior method development'],
        retraining=False,threshold_fitting=False,production_changed=False,wetlab_changed=False))
    print(family.to_string(index=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Audit endpoint specialization, loss-weight allocation and a hypothetical 150 cap."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/biomaster_ab_balance_audit_20260911'
SOURCE = ROOT/'outputs/biomaster_endpoint_ablation_20260911'
RESULT = ROOT/'outputs/biomaster_endpoint_multitask_20260911'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT/'data/processed/biomaster_training_full_20260910_v1/TRAIN.parquet',
             RESULT/'TEST_METRICS.csv', SOURCE/'kdki_inactive_TRAIN.parquet',
             SOURCE/'all_inactive_TRAIN.parquet', ROOT/'scripts/run_endpoint_multitask_20260911.py']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    tasks = pd.read_parquet(paths[0], columns=['pair_id', 'task'])
    groups = {k:set(g.pair_id) for k, g in tasks.groupby('task')}
    stats, mix = [], []
    for model, arm, path in [('A', 'kdki_inactive', paths[2]), ('B', 'all_inactive', paths[3])]:
        f = pd.read_parquet(path)
        assert f.pair_id.is_unique and f.binary_label.isin([0, 1]).all() and f.split.eq('train').all()
        n = f.groupby('target_id').size().sort_values(ascending=False)
        c = f.groupby(['target_id', 'binary_label']).size()
        prevalence = f.binary_label.mean()
        f['coefficient'] = np.where(f.binary_label.eq(1), .5/prevalence, .5/(1-prevalence))
        assert np.isclose(f.coefficient.sum(), len(f))
        assert np.isclose(f.loc[f.binary_label.eq(1), 'coefficient'].sum()/len(f), .5)
        has_kdki = f.pair_id.isin(groups['AFFINITY_KD_KI'])
        if model == 'A':
            f['supervision_group'] = np.where(has_kdki, 'numeric_KdKi_present', 'inactive_without_KdKi')
            assert f.loc[~has_kdki, 'explicit_inactive'].all()
        else:
            functional = f.pair_id.isin(groups['ACTIVITY_IC50'] | groups['ACTIVITY_EC50'])
            f['supervision_group'] = np.select([has_kdki, functional],
                ['numeric_KdKi_present', 'IC50_EC50_without_KdKi'], default='inactive_without_numeric_task')
        for key, g in f.groupby('supervision_group'):
            mix.append(dict(model=model, group=key, pairs=len(g), positive=int(g.binary_label.sum()),
                            pair_fraction=len(g)/len(f), expected_bce_coefficient_fraction=g.coefficient.sum()/len(f)))
        cutoff = int(np.ceil(len(n)*.1))
        pos_cap = int(c[c.index.get_level_values(1)==1].clip(upper=150).sum())
        neg_cap = int(c[c.index.get_level_values(1)==0].clip(upper=150).sum())
        stats.append(dict(model=model, arm=arm, pairs=len(f), targets=len(n), positive_fraction=prevalence,
            median_target_pairs=n.median(), maximum_target_pairs=n.max(),
            top10_target_row_fraction=n.head(10).sum()/len(f), top10pct_target_count=cutoff,
            top10pct_target_row_fraction=n.head(cutoff).sum()/len(f),
            explicit_inactive_pairs=int(f.explicit_inactive.sum()), explicit_inactive_row_fraction=f.explicit_inactive.mean(),
            explicit_inactive_bce_coefficient_fraction=f.loc[f.explicit_inactive, 'coefficient'].sum()/len(f),
            numeric_KdKi_pairs=int(has_kdki.sum()), numeric_KdKi_row_fraction=has_kdki.mean(),
            numeric_KdKi_bce_coefficient_fraction=f.loc[has_kdki, 'coefficient'].sum()/len(f),
            target_class_150_cap_positive=pos_cap, target_class_150_cap_negative=neg_cap,
            target_class_150_cap_pairs=pos_cap+neg_cap, cap_retained_fraction=(pos_cap+neg_cap)/len(f),
            target_only_sqrt_sampling_top10pct_mass=np.sqrt(n.head(cutoff)).sum()/np.sqrt(n).sum()))
    pd.DataFrame(stats).to_csv(OUT/'BALANCE_SUMMARY.csv', index=False)
    pd.DataFrame(mix).to_csv(OUT/'SUPERVISION_MIX.csv', index=False)
    f = pd.read_csv(paths[1]); f = f[f.scope.eq('all') & f.variant.eq('binary')]
    metric = f.groupby(['panel', 'arm'])[['ap', 'auroc', 'false_positive_rate']].agg(['mean', 'std'])
    metric.columns = ['_'.join(c) for c in metric.columns]
    metric.reset_index().to_csv(OUT/'BINARY_TASK_PERFORMANCE.csv', index=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    summary = dict(status='COMPLETE_DESCRIPTIVE_AUDIT_NO_NEW_TRAINING', input_sha256=hashes,
        inputs_unchanged=True, cap_applied=False, producer_sha256=sha(Path(__file__)),
        caveats=['Coefficient fractions are expected sampling/loss weights, not actual error or gradient contributions.',
                 'The 150 cap is count-only per current target and class; no actual rows or scaffolds were selected.',
                 'Square-root target-only sampling is an illustrative exposure calculation, not a tested performance gain or the proposed fully class-normalized weighting.',
                 'A retains non-KdKi rows only via accepted inactivity; other numerical endpoints on these pairs do not supervise A.',
                 'Observed imbalance and task performance do not identify the causal source of the A/B gap.'])
    (OUT/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(pd.DataFrame(stats)[['model','pairs','numeric_KdKi_row_fraction','top10pct_target_row_fraction','target_class_150_cap_pairs']].to_string(index=False))


if __name__ == '__main__':
    main()

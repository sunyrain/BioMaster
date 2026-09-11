#!/usr/bin/env python3
"""Query-wise ranking evaluation on frozen measured test candidates."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.portable_ranker_v2 import digest

OUT = ROOT / 'outputs/biomaster_ab_ranking_selection_20260911'
PREV = ROOT / 'outputs/biomaster_old_production_comparison_20260911'
SEEDS = [20260921, 20260922, 20260923]


def query_metrics(y, score):
    y = np.asarray(y, int)
    order = np.argsort(-score, kind='stable')
    ranked = y[order]
    result = dict(pairs=len(y), positive=int(y.sum()), ap=average_precision_score(y, score),
                  auroc=roc_auc_score(y, score), reciprocal_rank=1 / (np.flatnonzero(ranked)[0] + 1))
    for k in [1, 5, 10, 20]:
        if len(y) < k:
            result.update({f'{m}_at_{k}': np.nan for m in ['precision', 'recall', 'ndcg']})
            continue
        hits = int(ranked[:k].sum())
        discount = 1 / np.log2(np.arange(2, k + 2))
        result.update({f'precision_at_{k}': hits / k, f'recall_at_{k}': hits / y.sum(),
                       f'ndcg_at_{k}': float((ranked[:k] * discount).sum() / discount[:min(k, int(y.sum()))].sum())})
    return result


def main():
    OUT.mkdir(exist_ok=True)
    source = PREV / 'TEST_PREDICTIONS.parquet'
    before = digest(source)
    protocol = dict(test_predictions_sha256=before, minimum_measured_candidates=10, require_both_labels=True,
                    cutoffs=[1, 5, 10, 20], insufficient_candidates='K metric missing when query size < K',
                    old_score='head0 within drug; head1 within target', new_score='mean of two logits, each seed separately',
                    aggregation='unweighted query macro, then arithmetic mean of three seed metrics',
                    tie_break='pair_id ascending; AP and AUROC use tie-aware sklearn',
                    bootstrap='1000 paired queries, new per-query metrics averaged over seeds minus old',
                    interpretation='Only measured test candidates; unmeasured pairs are never negative labels.')
    (OUT / 'RANKING_PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
    frame = pd.read_parquet(source).sort_values(['panel', 'pair_id']).reset_index(drop=True)
    rows = []
    for panel, full in frame.groupby('panel'):
        if full.binary_label.nunique() < 2:
            continue
        for scope, f in [('all', full), ('old_train_pair_unseen', full[~full.old_train_pair_seen])]:
            for direction, key, head in [('drug_to_target', 'molecule_id', 0), ('target_to_drug', 'target_id', 1)]:
                specs = [('old_production', 0, f'old_production_head{head}')]
                specs += [(arm, seed, f'{arm}_{seed}_score') for arm in ['kdki_inactive', 'all_inactive'] for seed in SEEDS]
                n = 0
                for query, g in f.groupby(key, sort=True):
                    if len(g) < 10 or g.binary_label.nunique() < 2:
                        continue
                    n += 1
                    for arm, seed, col in specs:
                        rows.append(dict(panel=panel, scope=scope, direction=direction, query=query,
                                         model=arm, seed=seed, **query_metrics(g.binary_label.to_numpy(), g[col].to_numpy())))
                print(panel, scope, direction, n, 'queries', flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'QUERY_METRICS.csv.gz', index=False)
    values = ['ap', 'auroc', 'reciprocal_rank'] + [f'{m}_at_{k}' for k in [1, 5, 10, 20] for m in ['precision', 'recall', 'ndcg']]
    keys = ['panel', 'scope', 'direction', 'model', 'seed']
    macro = metrics.groupby(keys)[values].mean()
    macro['queries'] = metrics.groupby(keys).size()
    macro['queries_at_20'] = metrics.groupby(keys).precision_at_20.count()
    macro.reset_index().to_csv(OUT / 'QUERY_MACRO_PER_SEED.csv', index=False)
    summary = macro.reset_index().groupby(keys[:-1])[[*values, 'queries', 'queries_at_20']].mean().reset_index()
    summary.to_csv(OUT / 'QUERY_MACRO_COMPARISON.csv', index=False)
    ci = []
    for (panel, scope, direction), g in metrics.groupby(['panel', 'scope', 'direction']):
        old = g[g.model.eq('old_production')].set_index('query')
        for arm in ['kdki_inactive', 'all_inactive']:
            new = g[g.model.eq(arm)].groupby('query')[values].mean().reindex(old.index)
            for metric in ['ap', 'precision_at_5', 'recall_at_20', 'ndcg_at_20']:
                delta = (new[metric] - old[metric]).dropna().to_numpy()
                if not len(delta):
                    continue
                rng = np.random.default_rng(20260911)
                means = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(1000)]
                ci.append(dict(panel=panel, scope=scope, direction=direction, model=arm, metric=metric,
                               queries=len(delta), delta_new_minus_old=float(delta.mean()),
                               ci95_low=float(np.quantile(means, .025)), ci95_high=float(np.quantile(means, .975))))
    pd.DataFrame(ci).to_csv(OUT / 'PAIRED_QUERY_BOOTSTRAP.csv', index=False)
    assert digest(source) == before
    print(summary[summary.panel.eq('AFFINITY_KD_KI')][['scope', 'direction', 'model', 'queries', 'queries_at_20', 'ap', 'precision_at_5', 'recall_at_20']].to_string(index=False))


if __name__ == '__main__':
    main()

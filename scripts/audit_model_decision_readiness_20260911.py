#!/usr/bin/env python3
"""Post-hoc diagnostics on frozen predictions; no fitting or candidate replacement."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_model_decision_audit_20260911'
AB = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'
DATA = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
SEEDS = [20260921, 20260922, 20260923]
ARMS = {'A': 'kdki_inactive', 'B': 'all_inactive'}


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT / 'outputs/biomaster_old_production_comparison_20260911/TEST_PREDICTIONS.parquet',
             ROOT / 'outputs/biomaster_ab_ranking_selection_20260911/CORE276480_SELECTION_AUDIT.parquet',
             DATA / 'TRAIN.parquet', AB / 'COMMON_TEST.parquet',
             *[AB / f'{arm}_TRAIN.parquet' for arm in ARMS.values()],
             ROOT / 'outputs/biomaster_spr384_ab_rank_explanation_20260911/SPR384_AB_RANKING_EXPLANATION.csv',
             ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    f = pd.read_parquet(paths[0])
    f = f[f.panel.eq('AFFINITY_KD_KI')].sort_values('pair_id').reset_index(drop=True)
    assert len(f) == f.pair_id.nunique() == 34327
    core = pd.read_parquet(paths[1], columns=['pair_id', 'ligand_inchikey', 'drug_names', 'target_chembl_id',
                                            'gene_symbol', 'sequence_target_id', 'in_frozen384'])
    frozen = core[core.in_frozen384].copy()
    ranks = pd.read_csv(paths[-2])
    assert len(frozen) == 384 and set(frozen.pair_id) == set(pd.read_csv(paths[-1]).pair_id) == set(ranks.pair_id)
    ts112, ts384 = set(frozen.sequence_target_id), set(core.sequence_target_id)
    ds229, ds720 = set(frozen.ligand_inchikey), set(core.ligand_inchikey)
    support = core[['target_chembl_id', 'gene_symbol', 'sequence_target_id']].drop_duplicates().copy()
    support['in_frozen112'] = support.sequence_target_id.isin(ts112)
    support['wetlab_pairs'] = support.sequence_target_id.map(frozen.sequence_target_id.value_counts()).fillna(0).astype(int)
    train_stats = []
    for label, arm in ARMS.items():
        t = pd.read_parquet(AB / f'{arm}_TRAIN.parquet')
        g = t.groupby('target_id').binary_label.agg(['count', 'sum'])
        g['negative'] = g['count'] - g['sum']
        # Training-only target prevalence; no test/validation fitting, and no molecule information.
        prior = (g['sum'] + 1) / (g['count'] + 2)
        f[label + '_target_prior'] = f.target_id.map(prior).fillna((t.binary_label.sum() + 1) / (len(t) + 2))
        for dest, col in [('train_pairs', 'count'), ('train_positive', 'sum'), ('train_negative', 'negative')]:
            support[label + '_' + dest] = support.sequence_target_id.map(g[col]).fillna(0).astype(int)
        inactive = t[t.explicit_inactive].target_id.value_counts()
        support[label + '_explicit_inactive'] = support.sequence_target_id.map(inactive).fillna(0).astype(int)
        train_stats.append(dict(model=label, pairs=len(t), targets=t.target_id.nunique(), molecules=t.molecule_id.nunique(),
                                positive=int(t.binary_label.sum()), negative=int((1-t.binary_label).sum()),
                                equivalent_full_passes=6000*1024/len(t), old_parent_seen_pairs=int(t.seen_by_parent.sum())))
    pd.DataFrame(train_stats).to_csv(OUT / 'TRAINING_SCOPE.csv', index=False)
    support.to_csv(OUT / 'CORE384_TARGET_TRAINING_SUPPORT.csv', index=False)
    pair_support = frozen.merge(support, on=['target_chembl_id', 'gene_symbol', 'sequence_target_id'], validate='many_to_one')
    pair_support = pair_support.merge(ranks[['pair_id', '排序', 'A药内排名_384', 'B药内排名_384']], on='pair_id', validate='one_to_one').sort_values('排序')
    pair_support.to_csv(OUT / 'SPR384_TRAINING_SUPPORT.csv', index=False, encoding='utf-8-sig')

    old_unseen = ~f.old_train_pair_seen
    scopes = {
        'all_kdki': np.ones(len(f), bool),
        'old_pair_unseen': old_unseen,
        'document_disjoint_old_pair_unseen': f.document_disjoint & old_unseen,
        'catalog384_targets_old_pair_unseen': f.target_id.isin(ts384) & old_unseen,
        'wetlab112_targets_old_pair_unseen': f.target_id.isin(ts112) & old_unseen,
        'catalog720_drugs_old_pair_unseen': f.molecule_id.isin(ds720) & old_unseen,
        'catalog720x384_old_pair_unseen': f.molecule_id.isin(ds720) & f.target_id.isin(ts384) & old_unseen,
        'wetlab229x112_old_pair_unseen': f.molecule_id.isin(ds229) & f.target_id.isin(ts112) & old_unseen,
    }
    pair_rows, query_rows, coverage = [], [], []
    for scope, mask in scopes.items():
        sub = f[mask]
        coverage.append(dict(scope=scope, pairs=len(sub), positives=int(sub.binary_label.sum()),
                             negatives=int((1-sub.binary_label).sum()), molecules=sub.molecule_id.nunique(),
                             targets=sub.target_id.nunique(), wetlab_drugs=sub[sub.molecule_id.isin(ds229)].molecule_id.nunique(),
                             wetlab_targets=sub[sub.target_id.isin(ts112)].target_id.nunique()))
        specs = [('old_neural', 0, 'old_production_score')]
        specs += [(label, seed, f'{arm}_{seed}_score') for label, arm in ARMS.items() for seed in SEEDS]
        specs += [(label + '_target_prior', 0, label + '_target_prior') for label in ARMS]
        if sub.binary_label.nunique() == 2:
            for model, seed, col in specs:
                pair_rows.append(dict(scope=scope, model=model, seed=seed, pairs=len(sub),
                                      auroc=roc_auc_score(sub.binary_label, sub[col]),
                                      ap=average_precision_score(sub.binary_label, sub[col])))
        for direction, key, head in [('drug_to_target', 'molecule_id', 0), ('target_to_drug', 'target_id', 1)]:
            for query, g in sub.groupby(key, sort=True):
                if len(g) < 10 or g.binary_label.nunique() != 2:
                    continue
                for model, seed, col in specs:
                    if model == 'old_neural':
                        col = f'old_production_head{head}'
                    y = g.binary_label.to_numpy()
                    score = g[col].to_numpy()
                    order = np.argsort(-score, kind='stable')
                    # No arbitrary top-K result for constant/query-tied prior baselines.
                    top5 = np.nan if model.endswith('_prior') else float(y[order[:5]].mean())
                    query_rows.append(dict(scope=scope, direction=direction, query=query, model=model, seed=seed,
                                           pairs=len(g), prevalence=float(y.mean()),
                                           ap=average_precision_score(y, score), auroc=roc_auc_score(y, score), precision_at_5=top5))
        print(scope, len(sub), flush=True)
    pd.DataFrame(coverage).to_csv(OUT / 'TEST_SCOPE_COVERAGE.csv', index=False)
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT / 'PAIR_METRICS_PER_SEED.csv', index=False)
    pair.groupby(['scope', 'model'])[['pairs', 'auroc', 'ap']].mean().reset_index().to_csv(OUT / 'PAIR_COMPARISON.csv', index=False)
    q = pd.DataFrame(query_rows)
    q.to_csv(OUT / 'QUERY_METRICS.csv.gz', index=False)
    per = q.groupby(['scope', 'direction', 'model', 'seed']).agg(
        queries=('query', 'size'), ap=('ap', 'mean'), auroc=('auroc', 'mean'),
        precision_at_5=('precision_at_5', 'mean'), prevalence=('prevalence', 'mean')).reset_index()
    macro = per.groupby(['scope', 'direction', 'model'])[['queries', 'ap', 'auroc', 'precision_at_5', 'prevalence']].mean().reset_index()
    macro.to_csv(OUT / 'QUERY_COMPARISON.csv', index=False)
    ci = []
    for (scope, direction), g in q.groupby(['scope', 'direction']):
        for label in ['A', 'B']:
            for baseline in ['old_neural', label + '_target_prior']:
                for metric in (['ap', 'precision_at_5'] if baseline == 'old_neural' else ['ap']):
                    values = g.groupby(['model', 'query'])[metric].mean().unstack('model')
                    delta = (values[label] - values[baseline]).dropna().to_numpy()
                    rng = np.random.default_rng(20260911)
                    boots = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(1000)] if len(delta) >= 10 else []
                    ci.append(dict(scope=scope, direction=direction, model=label, baseline=baseline, metric=metric, queries=len(delta),
                                   delta=float(delta.mean()), ci95_low=float(np.quantile(boots, .025)) if boots else np.nan,
                                   ci95_high=float(np.quantile(boots, .975)) if boots else np.nan,
                                   ci_status='POSTHOC_QUERY_BOOTSTRAP' if boots else 'TOO_FEW_QUERIES_NO_INTERVAL'))
    pd.DataFrame(ci).to_csv(OUT / 'QUERY_PAIRED_BOOTSTRAP.csv', index=False)

    # Include all 112 targets; a missing matched test is visible rather than silently omitted.
    target_table = support[support.in_frozen112].copy()
    matched = f[f.target_id.isin(ts112) & old_unseen]
    counts = matched.groupby('target_id').binary_label.agg(['count', 'sum'])
    target_table['unseen_test_pairs'] = target_table.sequence_target_id.map(counts['count']).fillna(0).astype(int)
    target_table['unseen_test_positive'] = target_table.sequence_target_id.map(counts['sum']).fillna(0).astype(int)
    tq = q[q.scope.eq('wetlab112_targets_old_pair_unseen') & q.direction.eq('target_to_drug')]
    target_table['query_test_eligible'] = target_table.sequence_target_id.isin(tq['query'])
    for model in ['old_neural', 'A', 'B']:
        metrics = tq[tq.model.eq(model)].groupby('query')[['ap', 'precision_at_5']].mean()
        for metric in metrics:
            target_table[model + '_' + metric] = target_table.sequence_target_id.map(metrics[metric])
    target_table.to_csv(OUT / 'WETLAB112_TARGET_BENCHMARK.csv', index=False)

    # Release date does not imply newly performed experiments: retain evidence-year missingness.
    raw = pd.read_parquet(DATA / 'TRAIN.parquet', columns=['pair_id', 'task', 'binary_label', 'earliest_year', 'latest_year', 'missing_year_records', 'sources'])
    raw['earliest_evidence_year_band'] = np.select([raw.earliest_year.isna(), raw.earliest_year.le(2024), raw.earliest_year.eq(2025), raw.earliest_year.ge(2026)],
                                                 ['unknown', 'through_2024', '2025', '2026_or_later'], default='other')
    raw.groupby(['task', 'earliest_evidence_year_band']).agg(task_rows=('pair_id', 'size'), positive=('binary_label', 'sum'),
                                                           rows_with_missing_year_records=('missing_year_records', lambda x: int(x.gt(0).sum()))).reset_index().to_csv(OUT / 'TRAINING_EVIDENCE_YEARS.csv', index=False)
    raw.groupby(['task', 'sources']).size().rename('task_rows').reset_index().to_csv(OUT / 'TRAINING_SOURCE_COMBINATIONS.csv', index=False)
    f[['pair_id', 'target_id', 'molecule_id', 'document_disjoint', 'old_train_pair_seen', 'A_target_prior', 'B_target_prior']].to_csv(OUT / 'TEST_SCOPE_AND_PRIORS.csv.gz', index=False)
    stats = {}
    for label in ['A', 'B']:
        g = support[support.in_frozen112]
        stats[label] = dict(frozen_targets=len(g), no_training_targets=int(g[label+'_train_pairs'].eq(0).sum()),
                            no_positive_targets=int(g[label+'_train_positive'].eq(0).sum()),
                            no_negative_targets=int(g[label+'_train_negative'].eq(0).sum()),
                            both_classes_at_least10_targets=int((g[label+'_train_positive'].ge(10)&g[label+'_train_negative'].ge(10)).sum()),
                            both_classes_at_least10_pairs=int(g.loc[g[label+'_train_positive'].ge(10)&g[label+'_train_negative'].ge(10), 'wetlab_pairs'].sum()))
    gaps = pair_support[pair_support.A_train_positive.eq(0)]
    stats['A_no_positive_pairs'] = dict(pairs=len(gaps), A_median_rank=float(gaps['A药内排名_384'].median()),
                                       B_median_rank=float(gaps['B药内排名_384'].median()),
                                       A_top20=int(gaps['A药内排名_384'].le(20).sum()), B_top20=int(gaps['B药内排名_384'].le(20).sum()))
    assert all(digest(ROOT / p) == h for p, h in hashes.items())
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv', '.gz']]
    summary = dict(status='COMPLETE_POSTHOC_DIAGNOSTIC', input_sha256=hashes, inputs_unchanged=True,
                   frozen_target_support=stats, training_stats=train_stats,
                   protocol={'thresholds_fitted': False, 'model_training': False,
                             'target_prior': '(train positives+1)/(train pairs+2); absent target uses global training prevalence',
                             'metrics': 'A/B per-seed metrics averaged; old neural direction-specific head for query ranking',
                             'query_eligibility': 'at least 10 measured candidates and both labels within each scope',
                             'scope': 'Kd/Ki only; no unmeasured negatives',
                             'bootstrap': '1000 paired query resamples, seed-mean metric difference; at least 10 queries for intervals; post hoc, not multiplicity corrected',
                             'missing_cold_target_evidence': 'train counts do not certify nearest-neighbor or protein-homology generalization',
                             'legacy_combined_ranker': 'not evaluated on these new TEST pairs; old_neural is a different baseline'},
                   artifacts={p.name: digest(p) for p in artifacts}, source_sha256=digest(Path(__file__)))
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(stats, indent=2), flush=True)
    print(macro[macro.scope.isin(['old_pair_unseen', 'wetlab112_targets_old_pair_unseen'])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

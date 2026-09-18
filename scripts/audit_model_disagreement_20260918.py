#!/usr/bin/env python3
"""Reassess frozen agreement after the DTIAM A release; never rescore production."""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/model_agreement_20260917'
OUT = ROOT / 'outputs/model_disagreement_literature_20260918'
RELEASE = ROOT / 'outputs/dtiam_a_catalog_20260917/DTIAM_A_720X384_SCORES.csv.gz'
MODELS = ['biomaster', 'drugclip', 'dtiam', 'conplex', 'nesso', 'probematch', 'dtbind']
FIVE = ['biomaster', 'drugclip', 'dtiam', 'conplex', 'probematch']
KEYS = ['drug_id', 'target_id']


def top_weights(scores, k=10):
    cutoff = scores.nlargest(k).iloc[-1]
    above, tied = scores.gt(cutoff), scores.eq(cutoff)
    w = above.astype(float)
    w.loc[tied] = (k - int(above.sum())) / int(tied.sum())
    return w


def compare(frame, pairs, group_key, scope):
    other = 'drug_id' if group_key == 'target_id' else 'target_id'
    rows = []
    for entity, g in frame.groupby(group_key, sort=True):
        g = g.sort_values(other, kind='stable').set_index(other).copy()
        if group_key == 'target_id':
            g['biomaster'] = g.biomaster_reverse
        for a, b in pairs:
            s = g[[a, b]].dropna()
            assert len(s) >= 50
            first = set(s[a].sort_values(ascending=False, kind='stable').head(10).index)
            second = set(s[b].sort_values(ascending=False, kind='stable').head(10).index)
            rows.append(dict(scope=scope, query=group_key, entity=entity,
                             model_a=a, model_b=b, n=len(s),
                             spearman=spearmanr(s[a], s[b]).statistic,
                             top10_overlap=len(first & second),
                             top10_tie_expected=float((top_weights(s[a])*top_weights(s[b])).sum()),
                             top10_chance=100 / len(s)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [OLD/'SCORE_SNAPSHOT.csv.gz', OLD/'SEVEN_SHARED_SCORES.csv.gz',
              OLD/'BENCHMARK_INPUT_SNAPSHOT.csv', RELEASE]
    frame = pd.read_csv(inputs[0])
    shared = pd.read_csv(inputs[1])
    new = pd.read_csv(RELEASE, usecols=['ligand_inchikey', 'target_chembl_id', 'dtiam_probability'])
    new = new.rename(columns={'ligand_inchikey': 'drug_id', 'target_chembl_id': 'target_id',
                              'dtiam_probability': 'dtiam'})
    frame = frame.rename(columns={'dtiam': 'dtiam_historical'}).merge(new, on=KEYS, validate='one_to_one')
    assert len(frame) == 276480 and frame.dtiam.notna().all()
    joint = shared.rename(columns={'dtiam': 'dtiam_historical'}).merge(new, on=KEYS, validate='one_to_one')
    assert len(joint) == 5752 and joint.target_id.nunique() == 8
    assert joint.groupby('target_id').size().eq(719).all()
    assert joint[MODELS].notna().all().all()
    base = frame.dropna(subset=FIVE)
    assert len(base) == 275040 and base.target_id.nunique() == 382
    parts = [compare(joint, list(combinations(MODELS, 2)), 'target_id', 'SEVEN_FIXED_8_TARGETS'),
             compare(joint, [('dtiam_historical', 'dtiam')], 'target_id', 'OLD_NEW_FIXED_8_TARGETS')]
    for key, scope in [('target_id', 'FIVE_FIXED_382_TARGETS'), ('drug_id', 'FIVE_FIXED_720_DRUGS')]:
        parts.append(compare(base, [('dtiam', m) for m in FIVE if m != 'dtiam'], key, scope))
        parts.append(compare(base, [('dtiam_historical', 'dtiam')], key, 'OLD_NEW_'+scope))
    per = pd.concat(parts, ignore_index=True)
    per.to_csv(OUT/'PER_QUERY_AGREEMENT.csv', index=False)
    summary = per.groupby(['scope', 'query', 'model_a', 'model_b'], as_index=False).agg(
        queries=('entity', 'size'), pairs_per_query=('n', 'mean'), mean_spearman=('spearman', 'mean'),
        mean_top10_overlap=('top10_overlap', 'mean'), mean_top10_tie_expected=('top10_tie_expected', 'mean'),
        mean_top10_chance=('top10_chance', 'mean'))
    summary.to_csv(OUT/'AGREEMENT_SUMMARY.csv', index=False)
    joint.to_csv(OUT/'SEVEN_FIXED_SCORES_WITH_DTIAM_A.csv.gz', index=False)
    # Fix the 378 historical, jointly observed labels; no expansion as workers finish.
    bench = pd.read_csv(inputs[2], low_memory=False)
    bench = bench[bench.cohort.eq('BINDINGDB479')].dropna(subset=MODELS)
    bench = bench.rename(columns={'dtiam': 'dtiam_historical'}).merge(
        frame[KEYS+['dtiam', 'biomaster_reverse']], on=KEYS, validate='one_to_one')
    assert len(bench) == 378 and int(bench.label.sum()) == 103
    measures, queries = [], []
    for scope, part in [('COMMON_378', bench), ('AB_UNSEEN_COMMON', bench[bench.ab_unseen.eq(True)])]:
        for m in MODELS+['dtiam_historical', 'biomaster_reverse', 'nesso_pic50']:
            measures.append(dict(scope=scope, model=m, n=len(part), positive=int(part.label.sum()),
                                 ap=average_precision_score(part.label, part[m]),
                                 auroc=roc_auc_score(part.label, part[m])))
        for key in KEYS:
            for entity, g in part.groupby(key):
                if g.label.nunique() != 2:
                    continue
                for m in MODELS+['dtiam_historical', 'nesso_pic50']:
                    score = 'biomaster_reverse' if m == 'biomaster' and key == 'target_id' else m
                    queries.append(dict(scope=scope, query=key, entity=entity, model=m, n=len(g),
                                        prevalence=g.label.mean(), ap=average_precision_score(g.label, g[score])))
    pd.DataFrame(measures).to_csv(OUT/'SAME_LABEL_METRICS.csv', index=False)
    q = pd.DataFrame(queries)
    q.to_csv(OUT/'PER_QUERY_LABEL_METRICS.csv', index=False)
    qm = q.groupby(['scope', 'query', 'model'], as_index=False).agg(
        queries=('entity', 'size'), labeled_pairs=('n', 'sum'), macro_ap=('ap', 'mean'),
        macro_prevalence=('prevalence', 'mean'))
    qm.to_csv(OUT/'QUERY_LABEL_METRICS.csv', index=False)
    # These checks guarantee the other six channels were not changed by this audit.
    original = pd.read_csv(inputs[1]).set_index(KEYS).sort_index()
    updated = joint.set_index(KEYS).sort_index()
    for col in [m for m in MODELS if m != 'dtiam']+['biomaster_reverse']:
        np.testing.assert_array_equal(original[col].values, updated[col].values)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), checks='PASS',
                    dtiam_release='dtiam_a_kdki_inactive_20260917', shared_targets=8,
                    shared_pairs=5752, benchmark_pairs=378, benchmark_positive=103,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    scope='Only replace DTIAM in the frozen Sep17 snapshot. Other six unchanged. No production changes.',
                    limitations=['Eight early-completed targets are not representative of all 384.',
                                 'Known relations retained; model training overlap not removed.',
                                 'Agreement is not accuracy; the observed-label benchmark is not a prospective test.',
                                 'AB unseen excludes our A/B train/validation only; not all public checkpoints.'])
    (OUT/'AUDIT_CHECK.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mat = pd.DataFrame(np.eye(7), index=MODELS, columns=MODELS)
    for r in summary[summary.scope.eq('SEVEN_FIXED_8_TARGETS')].itertuples():
        mat.loc[r.model_a, r.model_b] = mat.loc[r.model_b, r.model_a] = r.mean_spearman
    names = ['ReTargetMap', 'DrugCLIP', 'DTIAM A', 'ConPLex', 'Nesso binder', 'ProbeMatchDTI', 'DTBind']
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(7), names, rotation=40, ha='right'); ax.set_yticks(range(7), names)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{mat.iloc[i,j]:.2f}', ha='center', va='center',
                    color='white' if abs(mat.iloc[i,j]) > .65 else '#172b4d')
    ax.set_title('Within-target rank agreement after DTIAM A upgrade\n8 fixed targets x 719 common drugs; mean Spearman', pad=18)
    fig.colorbar(im, ax=ax, shrink=.8, label='Rank correlation (not accuracy)')
    fig.tight_layout(); fig.savefig(OUT/'CURRENT_SEVEN_AGREEMENT.png', dpi=180); plt.close(fig)
    print(summary[summary.scope.eq('SEVEN_FIXED_8_TARGETS')].to_string(index=False))
    print(qm[qm.scope.eq('COMMON_378')].to_string(index=False))


if __name__ == '__main__':
    main()

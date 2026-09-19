#!/usr/bin/env python3
"""Read-only live score snapshot and query-wise agreement; never touches workers."""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import hashlib
import json
import re
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/catalog_results_20260919'
LIVE = ROOT / 'outputs/catalog_seven_models_20260916'
MODELS = ['biomaster', 'drugclip', 'dtiam', 'conplex', 'nesso', 'probematch', 'dtbind']
KEYS = ['drug_id', 'target_id']


def agreement(frame, query):
    other = 'drug_id' if query == 'target_id' else 'target_id'
    records = []
    for identifier, g in frame.groupby(query, sort=True):
        g = g.sort_values(other, kind='stable').copy()
        if query == 'target_id':
            g['biomaster'] = g['biomaster_reverse']
        scores = g[MODELS]
        assert scores.notna().all().all() and len(scores) >= 10
        correlations = scores.rank(method='average').corr().to_numpy()
        top, weights = [], []
        for model in MODELS:
            s = scores[model]
            top.append(set(np.argsort(-s.to_numpy(), kind='stable')[:10]))
            cutoff = s.nlargest(10).iloc[-1]
            above, tied = s.gt(cutoff), s.eq(cutoff)
            w = above.astype(float)
            w.loc[tied] = (10 - int(above.sum())) / int(tied.sum())
            weights.append(w.to_numpy())
        expected = np.asarray(weights) @ np.asarray(weights).T
        for a, b in combinations(range(len(MODELS)), 2):
            records.append(dict(query=query, entity=identifier, model_a=MODELS[a],
                                model_b=MODELS[b], n=len(g), spearman=correlations[a, b],
                                top10_overlap=len(top[a] & top[b]),
                                top10_tie_expected=float(expected[a, b]), top10_chance=100 / len(g)))
    return pd.DataFrame(records)


def validate_and_audit(frame, full, per):
    old_dir = ROOT / 'outputs/model_disagreement_literature_20260918'
    old = pd.read_csv(old_dir / 'SEVEN_FIXED_SCORES_WITH_DTIAM_A.csv.gz').set_index(KEYS).sort_index()
    current = frame.set_index(KEYS).loc[old.index].sort_index()
    delta = 0.0
    for model in MODELS + ['biomaster_reverse']:
        np.testing.assert_allclose(old[model], current[model], atol=1e-12, rtol=0)
        delta = max(delta, float(np.max(np.abs(old[model] - current[model]))))
    for row in per.groupby('query').head(4).itertuples():
        g = full[full[row.query].eq(row.entity)]
        a = 'biomaster_reverse' if row.model_a == 'biomaster' and row.query == 'target_id' else row.model_a
        np.testing.assert_allclose(spearmanr(g[a], g[row.model_b]).statistic, row.spearman, atol=1e-12, rtol=0)
    previous = pd.read_csv(old_dir / 'PER_QUERY_AGREEMENT.csv')
    previous = previous[previous.scope.eq('SEVEN_FIXED_8_TARGETS')]
    paired = previous.merge(per, on=['query', 'entity', 'model_a', 'model_b'], suffixes=('_old', '_new'))
    assert len(paired) == 168
    rows = []
    for identifier, g in full.groupby('drug_id'):
        if g.conplex.nunique() != 1:
            continue
        all_targets = frame[frame.drug_id.eq(identifier)]
        rows.append(dict(drug_id=identifier, drug_name=g.iloc[0].drug_name,
                         shared_score=float(g.iloc[0].conplex), shared_targets=g.target_id.nunique(),
                         full_catalog_unique_scores=all_targets.conplex.nunique(),
                         full_catalog_min=float(all_targets.conplex.min()), full_catalog_max=float(all_targets.conplex.max())))
    pd.DataFrame(rows).to_csv(OUT / 'CONPLEX_CONSTANT_QUERY_AUDIT.csv', index=False)
    result = dict(checks='PASS', scipy_rank_checks='PASS',
                  old_score_max_abs_delta=delta, old_score_comparison_atol=1e-12,
                  old_top10_max_abs_delta=float((paired.top10_overlap_old-paired.top10_overlap_new).abs().max()),
                  old_spearman_max_abs_delta=float((paired.spearman_old-paired.spearman_new).abs().max()),
                  precision_note='Live database scores retain original precision; prior CSV round trips can alter near-ties.',
                  conplex_constant_drug_queries_in_shared_subset=len(rows))
    (OUT / 'VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    db = sqlite3.connect(f'file:{LIVE / "scores.sqlite"}?mode=ro', uri=True, timeout=10)
    db.execute('BEGIN')
    pred = pd.read_sql_query('SELECT * FROM predictions', db)
    target_states = pd.read_sql_query('SELECT * FROM target_status', db)
    db.rollback()
    db.close()
    assert not pred.duplicated(['model'] + KEYS).any()
    completed = pred[pred.status.eq('completed')]
    assert np.isfinite(completed.score).all()
    base_path = ROOT / 'outputs/model_agreement_20260917/SCORE_SNAPSHOT.csv.gz'
    release_path = ROOT / 'outputs/dtiam_a_catalog_20260917/DTIAM_A_720X384_SCORES.csv.gz'
    base = pd.read_csv(base_path, usecols=KEYS + ['drug_name', 'gene', 'biomaster', 'biomaster_reverse', 'drugclip', 'conplex'])
    release = pd.read_csv(release_path, usecols=['ligand_inchikey', 'target_chembl_id', 'dtiam_probability']).rename(
        columns={'ligand_inchikey': 'drug_id', 'target_chembl_id': 'target_id', 'dtiam_probability': 'dtiam'})
    frame = base.merge(release, on=KEYS, validate='one_to_one')
    live = completed.pivot(index=KEYS, columns='model', values='score').reset_index()
    frame = frame.merge(live, on=KEYS, how='left', validate='one_to_one')
    assert len(frame) == 276480 and not frame.duplicated(KEYS).any()
    frame.to_csv(OUT / 'SCORE_SNAPSHOT.csv.gz', index=False)
    pred[pred.status.ne('completed')].to_csv(OUT / 'FAILED_PREDICTIONS.csv', index=False)
    target_states.to_csv(OUT / 'TARGET_STATUS_SNAPSHOT.csv', index=False)
    coverage = []
    for model in MODELS:
        part = frame[frame[model].notna()]
        counts = part.groupby('target_id').size()
        coverage.append(dict(model=model, completed=len(part), percent=100*len(part)/276480,
                             targets_with_any_score=len(counts), targets_720=int(counts.eq(720).sum()),
                             targets_at_least_719=int(counts.ge(719).sum())))
    pd.DataFrame(coverage).to_csv(OUT / 'COVERAGE.csv', index=False)
    common = frame.dropna(subset=MODELS)
    counts = common.groupby('target_id').size()
    full = common[common.target_id.isin(counts[counts.ge(719)].index)].copy()
    # No partly computed target enters the ranking comparison.
    assert full.groupby('target_id').size().eq(719).all()
    assert full.groupby('drug_id').size().eq(full.target_id.nunique()).all()
    assert full.drug_id.nunique() == 719
    full.to_csv(OUT / 'SEVEN_SHARED_COMPLETE_TARGETS.csv.gz', index=False)
    per = pd.concat([agreement(full, 'target_id'), agreement(full, 'drug_id')], ignore_index=True)
    per.to_csv(OUT / 'PER_QUERY_AGREEMENT.csv', index=False)
    summary = per.groupby(['query', 'model_a', 'model_b'], as_index=False).agg(
        queries=('entity', 'size'), valid_correlations=('spearman', 'count'),
        n_per_query=('n', 'mean'), mean_spearman=('spearman', 'mean'),
        median_spearman=('spearman', 'median'), mean_top10_overlap=('top10_overlap', 'mean'),
        mean_top10_tie_expected=('top10_tie_expected', 'mean'), mean_top10_chance=('top10_chance', 'mean'))
    summary.to_csv(OUT / 'AGREEMENT_SUMMARY.csv', index=False)
    targets = pd.read_csv(LIVE / 'TARGETS.csv')
    shared_targets = targets[targets.target_id.isin(full.target_id.unique())]
    shared_targets.to_csv(OUT / 'SHARED_TARGETS.csv', index=False)
    # Compare with published scores, allowing only CSV floating-point round trips.
    old_path = ROOT / 'outputs/model_disagreement_literature_20260918/SEVEN_FIXED_SCORES_WITH_DTIAM_A.csv.gz'
    validation = validate_and_audit(frame, full, per)
    # Audit availability directly; target_status can retain stale download-pending entries.
    seqs, key = {}, None
    for line in (ROOT / '.external/DTBind/Data/dti/biosnap_protein_seq.fasta').read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            seqs[key] = ''
        elif key:
            seqs[key] += line.strip()
    availability = []
    for t in targets.itertuples():
        state = ('no_author_sequence' if t.uniprot_id not in seqs else
                 'sequence_mismatch' if seqs[t.uniprot_id] != t.sequence else
                 'no_author_graph' if not (ROOT / f'outputs/frontier_dti_20260916/dtbind/protein_graph/{t.uniprot_id}.pt').exists() else 'available')
        availability.append(dict(target_id=t.target_id, gene=t.gene, state=state))
    availability = pd.DataFrame(availability)
    availability.to_csv(OUT / 'DTBIND_INPUT_AVAILABILITY.csv', index=False)
    timings = []
    for line in (LIVE / 'nesso/RUN.log').read_text().splitlines():
        match = re.match(r'(.+) completed (\d+) of (\d+) elapsed ([\d.]+)', line)
        if match:
            timings.append(dict(gene=match[1], completed=int(match[2]), seconds=float(match[4])))
    pd.DataFrame(timings).to_csv(OUT / 'NESSO_TARGET_TIMINGS.csv', index=False)
    statuses = {m: json.loads((LIVE / m / 'STATUS.json').read_text()) for m in ['nesso', 'dtbind', 'probematch']}
    metadata = dict(snapshot_utc=now, checks='PASS', dtiam_release='dtiam_a_kdki_inactive_20260917',
                    common_pairs_any_coverage=len(common), complete_shared_targets=full.target_id.nunique(),
                    complete_shared_drugs=full.drug_id.nunique(), complete_shared_pairs=len(full),
                    shared_target_length_min=int(shared_targets.protein_length.min()),
                    shared_target_length_max=int(shared_targets.protein_length.max()),
                    original_5752_scores_equal_within_1e_12=True, validation=validation,
                    coverage=coverage, worker_status=statuses,
                    dtbind_inputs=availability.state.value_counts().to_dict(),
                    failures=pred[pred.status.ne('completed')].groupby(['model', 'reason']).size().to_dict(),
                    nesso_recent20_mean_target_seconds=float(np.mean([r['seconds'] for r in timings[-20:]])),
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [base_path, release_path, old_path]},
                    limitations=['Unlabeled catalog agreement is not accuracy or SPR success.',
                                 'Common completed targets favor early/short/available proteins.',
                                 'Drug-wise ranks cover only the current shared target subset, not all 384.',
                                 'No new controlled training or independent experimental labels in this snapshot.'])
    metadata['failures'] = {':'.join(k): int(v) for k,v in metadata['failures'].items()}
    metadata['snapshot_sha256'] = hashlib.sha256((OUT / 'SCORE_SNAPSHOT.csv.gz').read_bytes()).hexdigest()
    (OUT / 'SUMMARY.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['sources','worker_status']}, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()

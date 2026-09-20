#!/usr/bin/env python3
"""Descriptive phase1 ranks only: no truth labels, Davis or performance claims."""
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_rank_agreement_20260921'
SCOPE = ROOT / 'outputs/dti_ranking_scope_20260921'
LIVE = ROOT / 'outputs/catalog_seven_models_20260916'
MODELS = ['ConPLex', 'DrugCLIP', 'DTIAM_A', 'Nesso-1', 'ProbeMatchDTI', 'DTBind_occurrence']
KEYS = ['drug_id', 'target_id']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def top_weights(x, k):
    cutoff = np.partition(x, len(x)-k)[len(x)-k]
    above, tied = x > cutoff, x == cutoff
    out = above.astype(float)
    out[tied] = (k-int(above.sum()))/int(tied.sum())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    base_path = ROOT / 'outputs/model_agreement_20260917/SCORE_SNAPSHOT.csv.gz'
    a_path = ROOT / 'outputs/dtiam_a_catalog_20260917/DTIAM_A_720X384_SCORES.csv.gz'
    base = pd.read_csv(base_path, usecols=KEYS+['drugclip', 'conplex'])
    a = pd.read_csv(a_path, usecols=['ligand_inchikey', 'target_chembl_id', 'dtiam_probability']).rename(
        columns={'ligand_inchikey':'drug_id','target_chembl_id':'target_id','dtiam_probability':'DTIAM_A'})
    with sqlite3.connect(f'file:{LIVE / "scores.sqlite"}?mode=ro', uri=True, timeout=20) as db:
        db.execute('BEGIN')
        scores = pd.read_sql_query("SELECT model,drug_id,target_id,score FROM predictions WHERE status='completed'", db)
        db.rollback()
    scores = scores.loc[scores.model.isin(['nesso','probematch','dtbind'])]
    assert not scores.duplicated(['model']+KEYS).any() and np.isfinite(scores.score).all()
    frame = base.merge(a, on=KEYS, validate='one_to_one').merge(
        scores.pivot(index=KEYS, columns='model', values='score').reset_index(), on=KEYS, how='left', validate='one_to_one')
    frame = frame.rename(columns={'drugclip':'DrugCLIP','conplex':'ConPLex','nesso':'Nesso-1',
        'probematch':'ProbeMatchDTI','dtbind':'DTBind_occurrence'})
    targets = pd.read_csv(SCOPE / 'APPLICATION_TARGET_SELECTION_384.csv')
    frame = frame.merge(targets[['target_id','sampling_family','selected_APP96']], on='target_id', validate='many_to_one')
    assert len(frame)==276480 and frame.target_id.nunique()==384 and frame.drug_id.nunique()==720
    frame.to_csv(OUT/'SIX_CURRENT_SCORE_SNAPSHOT.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    coverage, records = [], []
    for matrix, data in [('PRIMARY_720X96', frame.loc[frame.selected_APP96]), ('PARENT_720X384', frame)]:
        for model in MODELS:
            coverage.append(dict(matrix=matrix, model=model, requested_pairs=len(data),
                scored_pairs=int(data[model].notna().sum()), coverage_percent=100*data[model].notna().mean()))
        for view in ['ALL_SIX_COMMON','PAIRWISE_COMMON']:
            usable = data.dropna(subset=MODELS) if view=='ALL_SIX_COMMON' else data
            for query, direction in [('target_id','target_to_drug'),('drug_id','drug_to_target')]:
                for entity, group in usable.groupby(query, sort=True):
                    family = group.iloc[0].sampling_family if query=='target_id' else 'ALL_SELECTED_FAMILIES'
                    for ma, mb in combinations(MODELS,2):
                        part = group[[ma,mb]].dropna()
                        n = len(part)
                        if n<20:
                            continue
                        x,y=part[ma].to_numpy(),part[mb].to_numpy()
                        constant = np.ptp(x)==0 or np.ptp(y)==0
                        rho = np.nan if constant else float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
                        tau = np.nan if constant else float(kendalltau(x,y).statistic)
                        row=dict(matrix=matrix, view=view, direction=direction, query_id=entity, family=family,
                            model_a=ma, model_b=mb, candidates=n, spearman=rho, kendall_tau_b=tau,
                            constant_model_a=bool(np.ptp(x)==0), constant_model_b=bool(np.ptp(y)==0))
                        for k in [5,10,20]:
                            overlap=float(top_weights(x,k)@top_weights(y,k))
                            row[f'top{k}_tie_expected_overlap']=overlap
                            row[f'top{k}_random_expected_overlap']=k*k/n
                        records.append(row)
    detail=pd.DataFrame(records)
    detail.to_csv(OUT/'PER_QUERY_RANK_AGREEMENT.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    keys=['matrix','view','direction','model_a','model_b']
    def summarize(group_keys, name):
        summary=detail.groupby(group_keys,as_index=False).agg(queries=('query_id','size'),
            valid_correlations=('spearman','count'), minimum_candidates=('candidates','min'), maximum_candidates=('candidates','max'),
            median_spearman=('spearman','median'), mean_spearman=('spearman','mean'),
            mean_kendall_tau_b=('kendall_tau_b','mean'),
            mean_top10_overlap=('top10_tie_expected_overlap','mean'), random_top10_overlap=('top10_random_expected_overlap','mean'))
        summary['top10_overlap_ratio_to_random']=summary.mean_top10_overlap/summary.random_top10_overlap
        summary.to_csv(OUT/name,index=False,encoding='utf-8-sig')
        return summary
    summary=summarize(keys,'RANK_AGREEMENT_SUMMARY.csv')
    summarize(keys+['family'],'FAMILY_RANK_AGREEMENT_SUMMARY.csv')
    pd.DataFrame(coverage).to_csv(OUT/'COVERAGE.csv',index=False,encoding='utf-8-sig')
    manifest=dict(snapshot_utc=timestamp, current_models=MODELS,
        planned_additions_not_in_current_statistics=['MAMMAL_pKd','BALM','EviDTI','GraphBAN'],
        primary_requested_pairs=69120, parent_requested_pairs=276480,
        no_experimental_labels_read=True, davis_used=False, new_training=False,
        truth_or_accuracy_claim=False, common_coverage_not_complete_matrix=True,
        uncertainty='Descriptive point summaries only. No significance or population-wide stability claim from this snapshot.',
        query_gate='At least20 jointly scored candidates; disclose coverage and candidate ranges.',
        selection_manifest_sha256=sha(SCOPE/'SCOPE_SUMMARY.json'),
        native_score_sources={str(p.relative_to(ROOT)):sha(p) for p in [base_path,a_path]},
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['MANIFEST.json','VALIDATION.json']})
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    assert not detail.empty and detail.candidates.ge(20).all()
    assert detail.spearman.dropna().between(-1-1e-10,1+1e-10).all()
    assert detail.top10_tie_expected_overlap.between(-1e-10,10+1e-10).all()
    assert 'ReTargetMap' not in MODELS
    # Check analytical boundary behavior without assigning order to tied items.
    np.testing.assert_allclose(top_weights(np.ones(100),10)@top_weights(np.ones(100),10),1)
    np.testing.assert_allclose(top_weights(np.arange(100),10)@top_weights(np.arange(100),10),10)
    np.testing.assert_allclose(top_weights(np.arange(100),10)@top_weights(-np.arange(100),10),0)
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='PASS', finite_input_scores=True,
        unique_pair_keys=True,tie_boundary_checks=True,no_truth_labels=True,
        producer_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(pd.DataFrame(coverage).to_string(index=False))
    print(summary.loc[summary.matrix.eq('PRIMARY_720X96') & summary.view.eq('ALL_SIX_COMMON')].to_string(index=False))


if __name__=='__main__':
    main()

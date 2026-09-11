#!/usr/bin/env python3
"""Trace S5 row accounting and audit existing A/B data for auxiliary objectives."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_biomaster_odti_baselines_v1 import split_masks

OUT = ROOT/'outputs/biomaster_s5_origin_auxiliary_plan_20260911'
DATA = ROOT/'data/processed/biomaster_training_full_20260910_v1'
AB = ROOT/'outputs/biomaster_endpoint_ablation_20260911'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def queries(f, model, scope):
    rows = []
    for direction, col in [('drug_to_target', 'molecule_id'), ('target_to_drug', 'target_id')]:
        g = f.groupby(col).binary_label.agg(['size', 'sum'])
        both = g['sum'].gt(0) & g['sum'].lt(g['size'])
        rows.append(dict(model=model, scope=scope, direction=direction, pairs=len(f), all_queries=len(g),
                         both_label_queries=int(both.sum()), rows_in_both_label_queries=int(g.loc[both, 'size'].sum()),
                         both_label_queries_at_least10=int((both & g['size'].ge(10)).sum()),
                         median_both_label_query_size=float(g.loc[both, 'size'].median())))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT/'outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz',
             ROOT/'outputs/current_production_package_v2/conplex_target_calibration_v5_official/CONPLEX_CALIBRATION_PREPARATION_V5.json',
             ROOT/'outputs/current_production_package_v2/chembl37_target_calibration_v5/PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz',
             DATA/'TRAIN.parquet', DATA/'TRAIN_REGRESSION.parquet',
             AB/'kdki_inactive_TRAIN.parquet', AB/'all_inactive_TRAIN.parquet',
             ROOT/'scripts/prepare_conplex_calibration_v5.py', ROOT/'scripts/run_biomaster_odti_baselines_v1.py',
             ROOT/'scripts/train_biomaster_odti_routed_ranker_v1.py', ROOT/'biomaster/best_model_training.py',
             ROOT/'biomaster/unified_interaction.py']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    old = pd.read_csv(paths[0], low_memory=False)
    prep = json.loads(paths[1].read_text())
    source = pd.read_csv(paths[2], usecols=['calibration_label'])
    assert len(source) == prep['source_pair_rows'] == 509172 and digest(paths[2]) == prep['source_sha256']
    assert len(old) == old.calibration_pair_id.nunique() == prep['calibration_rows'] == 86674
    assert old.groupby(['sequence_key','binary_label']).size().max() == prep['max_per_label_target'] == 150
    masks = split_masks(old, 'S5_OLD_DRUG_ENTITY_COLD', -1)
    assert np.stack(list(masks.values())).sum(axis=0).max() == 1
    available = old.drug_feature_available.astype(bool)
    role = pd.Series('OLD_SCAFFOLD_NONCATALOG_HOLDOUT', index=old.index)
    for name, mask in masks.items():
        role.loc[mask] = name.upper()
    role.loc[~available] = 'FEATURE_QUARANTINE'
    counts = old.assign(role=role).groupby('role').agg(rows=('calibration_pair_id','size'),positive=('binary_label','sum')).reset_index()
    counts['negative'] = counts.rows - counts.positive
    counts.to_csv(OUT/'S5_ROW_ACCOUNTING.csv', index=False)
    assert counts.set_index('role').rows.to_dict() == {'FEATURE_QUARANTINE':1, 'OLD_SCAFFOLD_NONCATALOG_HOLDOUT':2173, 'TEST':2556, 'TRAIN':65276, 'VALID':16668}
    members = old[['calibration_pair_id','parent_standard_inchi_key','target_chembl_id','scaffold_group','binary_label','drug_feature_available']].copy()
    members['S5_role'] = role
    members.to_csv(OUT/'S5_ROLE_MEMBERSHIP.csv.gz', index=False)
    assert not set(old.loc[role.eq('TRAIN'),'scaffold_group']) & set(old.loc[role.eq('VALID'),'scaffold_group'])
    assert not old.loc[role.isin(['TRAIN','VALID']),'has_deployment_old_drug_scaffold'].any()

    tasks = pd.read_parquet(DATA/'TRAIN.parquet')
    reg = pd.read_parquet(DATA/'TRAIN_REGRESSION.parquet')
    assert reg.regression_eligible.all() and reg.split.eq('train').all()
    assert np.isfinite(reg.median_p_activity_unique).all()
    assert not reg.duplicated(['pair_id','endpoint']).any()
    pooled, native, regress, totals = [], [], [], []
    for model, arm, chosen, endpoints in [
        ('A','kdki_inactive',['AFFINITY_KD_KI'],['Kd','Ki']),
        ('B','all_inactive',['AFFINITY_KD_KI','ACTIVITY_IC50','ACTIVITY_EC50'],['Kd','Ki','IC50','EC50'])]:
        train = pd.read_parquet(AB/f'{arm}_TRAIN.parquet')
        assert train.pair_id.is_unique and train.split.eq('train').all()
        pooled.extend(queries(train, model, 'current_pooled_binary_including_inactivity'))
        task_pool = tasks[tasks.pair_id.isin(train.pair_id) & tasks.task.isin(chosen)]
        assert task_pool.binary_label.isin([0, 1]).all()
        assert not task_pool.duplicated(['pair_id', 'task']).any()
        for task, g in task_pool.groupby('task'):
            native.extend(queries(g, model, task))
        same = reg[reg.pair_id.isin(train.pair_id) & reg.endpoint.isin(endpoints)]
        # Exact pair/feature identity join; auxiliaries may not silently expand training members.
        match = same.merge(train[['pair_id','drug_feature_index','target_feature_index','split_group']], on='pair_id', validate='many_to_one', suffixes=('', '_arm'))
        for col in ['drug_feature_index','target_feature_index','split_group']:
            assert match[col].equals(match[col+'_arm'])
        for endpoint in endpoints:
            g = same[same.endpoint.eq(endpoint)]
            regress.append(dict(model=model, endpoint=endpoint, exact_regression_rows=len(g), unique_pairs=g.pair_id.nunique()))
        totals.append(dict(model=model, existing_train_pairs=len(train), exact_regression_rows=len(same),
                           unique_pairs_with_exact_regression=same.pair_id.nunique(),
                           exact_regression_rows_outside_current_arm=int((reg.endpoint.isin(endpoints)&~reg.pair_id.isin(train.pair_id)).sum())))
    pd.DataFrame(pooled).to_csv(OUT/'POOLED_RANK_QUERY_AVAILABILITY.csv', index=False)
    pd.DataFrame(native).to_csv(OUT/'SAME_TASK_RANK_QUERY_AVAILABILITY.csv', index=False)
    pd.DataFrame(regress).to_csv(OUT/'EXACT_REGRESSION_AVAILABILITY.csv', index=False)
    pd.DataFrame(totals).to_csv(OUT/'AUXILIARY_POOL_TOTALS.csv', index=False)
    assert all(digest(ROOT/p)==sha for p,sha in hashes.items())
    summary = dict(status='COMPLETE_DATA_AUDIT_DESIGN_ONLY_NO_TRAINING', input_sha256=hashes, inputs_unchanged=True,
                   source_pairs=509172, capped_calibration_pairs=86674, per_target_per_class_cap=150,
                   source_label_counts=source.calibration_label.value_counts().to_dict(),
                   s5_role_counts=counts.to_dict('records'), auxiliary_pool_totals=totals,
                   caveats=['Pooled query counts include generic inactivity and cross-endpoint pooling; same-task counts are the proposed numeric ranking pools.',
                            'Same-task does not certify identical assay conditions; endpoint-specific or same-assay strength ranking requires stricter grouping.',
                            'Regression counts are endpoint rows within existing A/B training pair memberships, not unique independent experiments.',
                            'Outside-arm regression rows are not included in the proposed first controlled ablation; they require separate QC and feature availability checks.'],
                   artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv','.gz']},
                   source_sha256=digest(Path(__file__)))
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(counts.to_string(index=False))
    print(pd.DataFrame(regress).to_string(index=False))
    print(pd.DataFrame(totals).to_string(index=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Independent artifact, rank metric, feature, and selection contract checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_ab_ranking_selection_20260911'
AB = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    protocol = json.loads((OUT / 'SELECTION_PROTOCOL.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in protocol['input_hashes'].items())
    rankprotocol = json.loads((OUT / 'RANKING_PROTOCOL.json').read_text())
    predpath = ROOT / 'outputs/biomaster_old_production_comparison_20260911/TEST_PREDICTIONS.parquet'
    assert sha(predpath) == rankprotocol['test_predictions_sha256']
    query = pd.read_csv(OUT / 'QUERY_METRICS.csv.gz')
    pred = pd.read_parquet(predpath)
    for r in query.sample(40, random_state=20260911).itertuples():
        key = 'molecule_id' if r.direction == 'drug_to_target' else 'target_id'
        f = pred[pred.panel.eq(r.panel) & pred[key].eq(r.query)]
        if r.scope == 'old_train_pair_unseen':
            f = f[~f.old_train_pair_seen]
        column = ('old_production_head0' if key == 'molecule_id' else 'old_production_head1') if r.model == 'old_production' else f'{r.model}_{r.seed}_score'
        assert len(f) == r.pairs
        assert abs(average_precision_score(f.binary_label, f[column]) - r.ap) < 1e-12
    # A/B query AP must agree with the preceding endpoint experiment, not just aggregate plausibility.
    oldquery = pd.read_csv(AB / 'TEST_QUERY_METRICS.csv.gz')
    oldquery['direction'] = oldquery.direction.map({'within_target':'target_to_drug', 'within_drug':'drug_to_target'})
    a = query[query.model.ne('old_production') & query.scope.eq('all')].rename(columns={'model':'arm'})
    joined = a.merge(oldquery, on=['arm','seed','panel','direction','query'], suffixes=['_now','_previous'], validate='one_to_one')
    assert len(joined) == len(a)
    np.testing.assert_allclose(joined.ap_now, joined.ap_previous, rtol=1e-12, atol=1e-12)
    final = pd.read_csv(ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv')
    finalids = set(final.pair_id)
    slots = final.groupby('新靶点ChEMBL编号').size().to_dict()
    chosen = pd.read_csv(OUT / 'COUNTERFACTUAL_SELECTIONS.csv')
    summary = pd.read_csv(OUT / 'RETENTION_SUMMARY.csv')
    pool = pd.read_parquet(OUT / 'CORE276480_SELECTION_AUDIT.parquet')
    elig = set(pool.loc[pool.counterfactual_eligible,'pair_id'])
    checks = 0
    for (model, scenario), f in chosen.groupby(['model','scenario']):
        assert len(f) == f.pair_id.nunique() == 384 and set(f.pair_id) <= elig
        r = summary[summary.model.eq(model) & summary.scenario.eq(scenario)].iloc[0]
        assert int(r.retained) == len(set(f.pair_id) & finalids)
        if scenario.startswith('fixed112'):
            assert f.groupby('target_chembl_id').size().to_dict() == slots
            assert f.groupby('ligand_inchikey').size().max() <= 4 and f.ligand_inchikey.nunique() >= 128
            assert f.ligand_inchikey.str.split('-').str[0].value_counts().max() <= 4
        if scenario.endswith('latest_exact_excluded'):
            assert not f.latest_exact_evidence.any()
        checks += 1
    for name in ['kdki_inactive_consensus','all_inactive_consensus']:
        primary = chosen[chosen.model.eq(name) & chosen.scenario.eq('fixed112_frozen_evidence')]
        export_name = ('A' if name.startswith('kdki') else 'B') + '_COUNTERFACTUAL384_NOT_LAB_RELEASE.csv'
        export = pd.read_csv(OUT / export_name)
        assert set(export.pair_id) == set(primary.pair_id)
        detail = pd.read_csv(OUT / 'SPR384_MODEL_RANK_AND_RETENTION.csv')
        assert set(detail.loc[detail[name + '__fixed112_frozen_evidence__retained'],'pair_id']) == set(primary.pair_id) & finalids
    compact = pd.read_csv(OUT / 'SPR384_AB_REVIEW.csv')
    assert len(compact) == compact.pair_id.nunique() == 384 and set(compact.pair_id) == finalids
    assert compact['AB共同保留'].sum() == 20 and compact['AB至少一个保留'].sum() == 72
    for path in OUT.glob('*_DIRECTIONAL_LOGITS.npy'):
        value = np.load(path, mmap_mode='r')
        assert value.shape == (720,890,2) and np.isfinite(value).all()
    # Check catalog features against independently indexed training-bank identities.
    d = pd.read_csv(ROOT / 'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv')
    molecules = pd.read_parquet(ROOT / 'data/processed/biomaster_training_full_20260910_v1/MOLECULES.parquet').set_index('smiles')
    canonical = d.smiles.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True))
    indices = canonical.map(molecules.drug_feature_index)
    done = np.load(AB / 'features/DRUG_DONE.npy', mmap_mode='r')
    catids = np.flatnonzero(indices.notna())
    catids = catids[done[indices.iloc[catids].to_numpy(int)]]
    trainids = indices.iloc[catids].to_numpy(int)
    bundle = ROOT / 'outputs/biomaster_best_model_20260906/retargetmap_selected_v1/features'
    for source, target in [('DRUG_CLIP','drug_global'),('MORGAN','drug_global'),('GRAPH','drug_graph_mean'),('AVAILABLE','pretrained_available')]:
        x = np.load(AB / 'features' / (source+'.npy'), mmap_mode='r')[trainids]
        y = np.load(bundle / (target+'.npy'), mmap_mode='r')[catids]
        if source == 'DRUG_CLIP': y = y[:,:512]
        if source == 'MORGAN': y = y[:,512:]
        np.testing.assert_array_equal(x,y)
    assert len(catids) == 472
    log = (OUT / 'FOCUSED_TEST_RESULT.txt').read_text()
    assert '10 passed' in log
    result = dict(all_pass=True, unchanged_input_hashes=True, sampled_query_AP_independently_recomputed=40,
                  previous_ab_query_ap_reproduced=len(joined), selection_sets_verified=checks,
                  exact_fixed_slots_and_diversity_caps=True, retained_pair_ids_independently_recomputed=True,
                  shared_training_catalog_feature_rows_identical=len(catids), focused_tests='10 passed',
                  script_sha256=sha(Path(__file__)))
    (OUT / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

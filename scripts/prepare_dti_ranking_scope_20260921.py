#!/usr/bin/env python3
"""Freeze a score-independent balanced submatrix for phase1 rank agreement.

No prediction or experimental-label files are read. Protein metadata and existing
sequence-homology groups support sampling; this is not external preregistration.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_ranking_scope_20260921'
STORE = ROOT / 'data/research/dti_ranking_scope_20260921_v1'
PREP = ROOT / 'data/research/dti_reliability_20260920_v1'
TRAIN = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
CATALOG = ROOT / 'outputs/catalog_seven_models_20260916'
SEED = 20260921


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csv(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def priority(identifier):
    return hashlib.sha256(f'{SEED}|{identifier}'.encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True); STORE.mkdir(parents=True, exist_ok=True)
    sources = [TRAIN / 'TARGETS.parquet', PREP / 'TARGET_CLUSTERS.parquet',
               CATALOG / 'DRUGS.csv', CATALOG / 'TARGETS.csv']
    # Application submatrix: same 720 molecules, independently hashed 96-target subset.
    targets = pd.read_csv(CATALOG / 'TARGETS.csv')
    drugs = pd.read_csv(CATALOG / 'DRUGS.csv')
    meta_path = ROOT / 'outputs/biomaster_matrix_720x890_20260910/TARGET_INDEX.csv.gz'; sources.append(meta_path)
    meta = pd.read_csv(meta_path, usecols=['target_chembl_id', 'sequence_sha256', 'ot_project_assay_family',
        'target_class_l1', 'target_class_l2', 'structure_ready_strict'])
    targets = targets.merge(meta, left_on=['target_id', 'protein_sha256'],
        right_on=['target_chembl_id', 'sequence_sha256'], how='left', validate='one_to_one')
    assert targets.target_chembl_id.notna().all()
    training_targets = pd.read_parquet(TRAIN / 'TARGETS.parquet', columns=['sequence_sha256', 'target_feature_index'])
    clusters = pd.read_parquet(PREP / 'TARGET_CLUSTERS.parquet', columns=['target_feature_index', 'homology_cluster'])
    mapping = training_targets.merge(clusters, on='target_feature_index', validate='one_to_one')
    targets = targets.merge(mapping, on='sequence_sha256', how='left', validate='one_to_one')
    targets['cluster_known'] = targets.homology_cluster.notna()
    targets['sampling_cluster'] = targets.homology_cluster.fillna('UNMAPPED:')
    targets.loc[~targets.cluster_known, 'sampling_cluster'] = 'UNMAPPED:' + targets.loc[~targets.cluster_known, 'target_id']
    targets['sampling_family'] = targets.ot_project_assay_family.fillna('unclassified')
    targets['hash_priority'] = targets.target_id.map(priority)
    family_order = sorted(targets.sampling_family.unique())
    selected, used, selected_clusters = [], set(), set()
    for unique_cluster in [True, False]:
        while len(selected) < 96:
            changed = False
            for family in family_order:
                pool = targets.loc[targets.sampling_family.eq(family) & ~targets.target_id.isin(used)]
                if unique_cluster:
                    pool = pool.loc[~pool.sampling_cluster.isin(selected_clusters)]
                if pool.empty:
                    continue
                row = pool.sort_values('hash_priority').iloc[0]
                selected.append(row.target_id); used.add(row.target_id); selected_clusters.add(row.sampling_cluster)
                changed = True
                if len(selected) == 96:
                    break
            if not changed:
                break
    assert len(selected) == 96
    targets['selected_APP96'] = targets.target_id.isin(selected)
    csv('APPLICATION_TARGET_SELECTION_384.csv', targets.drop(columns='sequence'))
    chosen = targets.loc[targets.selected_APP96].sort_values('target_id')
    csv('APPLICATION_96_TARGETS.csv', chosen.drop(columns='sequence'))
    csv('APPLICATION_720_DRUGS.csv', drugs)
    family = targets.groupby('sampling_family').agg(full_targets=('target_id', 'size'),
        chosen_targets=('selected_APP96', 'sum')).reset_index()
    csv('APPLICATION_FAMILY_COUNTS.csv', family)
    pairs = drugs[['drug_id']].merge(chosen[['target_id']], how='cross')
    assert len(pairs) == 69120 and not pairs.duplicated().any()
    pairs.to_csv(STORE / 'APPLICATION_720X96_PAIRS.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    chosen[['target_id', 'uniprot_id', 'gene', 'sequence']].to_parquet(STORE / 'APPLICATION_TARGET_INPUTS.parquet', index=False)

    summary = dict(created_utc=stamp(), study_scope='Phase1 bidirectional ranking agreement and recommendation divergence only; correctness deferred to phase2; Davis excluded.',
        application=dict(drugs=720, targets=96, pairs=len(pairs), parent_targets=384, parent_pairs=276480,
            sampling='Round-robin molecular-function families, deterministic SHA256 seed20260921, prefer distinct known homology clusters. No scores, labels, SPR384 membership or runtime coverage used.',
            selected_known_homology_clusters=chosen.loc[chosen.cluster_known, 'homology_cluster'].nunique(),
            selected_unmapped_homology_targets=int((~chosen.cluster_known).sum()),
            inherited_bias='Parent720x384 was selected by the project; this subset is not an independent external dataset.',
            full_parent_matrix_role='Supplementary candidate-pool sensitivity; never compare ranks across different denominators.'),
        phase1_labels_used=False, davis_used=False, new_phase1_training_fits=0,
        interpretation='Disagreement is not evidence of incorrectness; agreement is not evidence of correctness.',
        no_predictions_read=True, new_training_started=False, performance_metrics_computed=False,
        new_independent_external_correctness_claim_ready=False,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        artifacts={str(p.relative_to(ROOT)): sha(p) for p in sorted(list(STORE.glob('*')) + list(OUT.glob('*.csv'))) if p.is_file()})
    save(OUT / 'SCOPE_SUMMARY.json', summary)
    save(OUT / 'VALIDATION.json', dict(status='PASS', score_independent_selection=True, application_unique_pairs=69120,
        target_sequence_mapping_exact=True, original_A_data_not_changed=True, no_experimental_labels_read=True, davis_not_read=True,
        hash_manifest_sha256=sha(OUT / 'SCOPE_SUMMARY.json'), producer_sha256=sha(Path(__file__))))
    print(family.to_string(index=False)); print('Phase1: no experimental labels, no Davis, no new fits.')


def stamp():
    return datetime.now(timezone.utc).isoformat()


if __name__ == '__main__':
    main()

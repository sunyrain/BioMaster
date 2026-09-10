#!/usr/bin/env python3
"""Read-only data alignment, module drift and structural-retention audit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig
from diagnose_biomaster_structural_contact_learning import paired_interval

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
DEST = OUT / 'system_audit_20260907'


def read(path):
    return json.loads(path.read_text())


def main():
    DEST.mkdir(exist_ok=True)
    data = ROOT / 'outputs/biomaster_best_model_20260906/data'
    source = ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
    train = pd.read_csv(data / 'roll_2020/TRAIN.csv.gz')
    validation = pd.read_csv(data / 'roll_2020/VALIDATION.csv.gz')
    old = pd.read_csv(source / 'OLD_DRUG_INDEX.csv')
    features = read(OUT / 'features/MANIFEST.json')
    targets = {r['target_feature_index'] for r in features['records']}
    available = np.load(OUT / 'features/MOLECULE_AVAILABLE.npy')
    tables = {}
    for name, frame in [('all_training', train), ('old_drug_training', train[train.is_project_old_drug]),
                        ('all_validation', validation), ('old_drug_validation', validation[validation.is_project_old_drug])]:
        local = available[frame.drug_feature_index.to_numpy()] & frame.target_feature_index.isin(targets).to_numpy()
        positive = frame[frame.binary_label.eq(1)]
        tables[name] = dict(rows=len(frame), molecules=frame.drug_feature_index.nunique(), targets=frame.target_feature_index.nunique(),
                            positives=len(positive), negatives=int(frame.binary_label.eq(0).sum()), positive_fraction=float(frame.binary_label.mean()),
                            local_available_rows=int(local.sum()), positive_drug_queries=positive.drug_feature_index.nunique(),
                            positive_target_queries=positive.target_feature_index.nunique())
    pd.DataFrame(tables).T.to_csv(DEST / 'DATA_COUNTS.csv')
    admitted = pd.read_parquet(OUT / 'structural_data/training_2020/ADMITTED.parquet')
    clusters = {}
    for split, frame in admitted.groupby('split'):
        counts = frame.cluster.value_counts()
        clusters[split] = dict(complexes=len(frame), clusters=len(counts), largest_cluster=str(counts.index[0]),
                               largest_cluster_complexes=int(counts.iloc[0]), largest_cluster_fraction=float(counts.iloc[0]/len(frame)))
    pocket_counts = pd.Series([r['target_feature_index'] for r in features['records']]).value_counts()
    cfg = PocketPrecisionConfig(**read(ROOT / 'configs/biomaster_pocket_precision_20260906.json')['model'])
    torch.set_num_threads(4); torch.manual_seed(20260921)
    initial_model = PocketPrecision(cfg); initial = initial_model.state_dict()
    structural_path = OUT / 'structural_pretraining/cutoff_2020/seed_20260921/STRUCTURAL_PRETRAINED.pt'
    epoch_path = OUT / 'downstream_diagnostics/epoch1_fp32_20260907/EPOCH1_SNAPSHOT.pt'
    pre = torch.load(structural_path, map_location='cpu', weights_only=False)['local_model']
    epoch = torch.load(epoch_path, map_location='cpu', weights_only=False)
    new = {k[len('local.'):]:v for k,v in epoch['model'].items() if k.startswith('local.')}
    drift = {}
    for prefix in ['blocks.', 'pool_attention.', 'local_output.', 'pocket_gate.', 'refine.', 'distance_head.', 'contact_head.']:
        keys = [k for k in pre if k.startswith(prefix)]
        def compare(first, second):
            delta = sum(float((first[k].float()-second[k].float()).square().sum()) for k in keys)**.5
            norm = sum(float(second[k].float().square().sum()) for k in keys)**.5
            return dict(exactly_unchanged=all(torch.equal(first[k],second[k]) for k in keys), l2_change=delta, relative_l2_change=delta/max(norm,1e-30))
        drift[prefix] = dict(pretraining_vs_initial=compare(pre,initial), epoch1_vs_pretraining=compare(new,pre))
    before_path = OUT / 'contact_diagnostics/completed_pretraining/COMPLEX_METRICS.csv'
    after_path = OUT / 'downstream_diagnostics/epoch1_contact_retention_20260907/COMPLEX_METRICS.csv'
    before, after = pd.read_csv(before_path), pd.read_csv(after_path)
    merged = after.merge(before, on=['system_id','cluster'], suffixes=('_after','_before'), validate='one_to_one')
    if len(merged) != len(before) or len(merged) != len(after):
        raise ValueError('structural retention must compare identical complexes and clusters')
    retention = {}
    for metric in ['native_contact_ap','all_regions_contact_ap','native_distance_mae_capped32_A']:
        retention[metric] = dict(before=float(merged[metric+'_before'].mean()), after=float(merged[metric+'_after'].mean()),
                                 after_minus_before=paired_interval(merged, metric+'_after', metric+'_before'))
    merged.to_csv(DEST / 'PAIRED_STRUCTURAL_RETENTION.csv', index=False)
    protocol = read(ROOT / 'configs/biomaster_pocket_precision_20260906.json')['training']
    steps = int(np.ceil(len(train)/protocol['effective_batch_size']))
    query_events = steps // protocol['retrieval_every_updates']
    pos = train[train.is_project_old_drug & train.binary_label.eq(1)]
    expected = {}
    for key in ['drug_feature_index','target_feature_index']:
        count = pos[key].nunique()
        expected[key] = dict(eligible_queries=count, draws_first_epoch=query_events,
                            expected_distinct_queries=count*(1-(1-1/count)**query_events),
                            interpretation='expectation under uniform random draws with replacement, not reconstructed actual sampled IDs')
    fp32 = read(OUT / 'downstream_diagnostics/epoch1_fp32_20260907/RESULT.json')
    program = read(OUT / 'PROGRAM_STATUS.json')
    for identity in program['source_identities']:
        if file_identity(identity['path']) != identity:
            raise ValueError('running program source changed')
    result = dict(status='COMPLETE_READ_ONLY_SYSTEM_AUDIT', utc=datetime.now(timezone.utc).isoformat(), data=tables,
                  old_drug_training_fraction=tables['old_drug_training']['rows']/len(train),
                  single_target_training_molecules=int(train.groupby('drug_feature_index').size().eq(1).sum()),
                  structural_clusters=clusters,
                  features=dict(targets=features['targets'], targets_with_pockets=len(targets), pockets=features['pockets'],
                                single_pocket_targets=int(pocket_counts.eq(1).sum()), old_drugs_with_molecule_features=int(available[old.drug_feature_index.to_numpy()].sum()),
                                old_drug_total=len(old)),
                  local_parameters=sum(p.numel() for p in initial_model.parameters()),
                  frozen_parent_parameters=sum(v.numel() for k,v in epoch['model'].items() if k.startswith('parent.') and v.is_floating_point()),
                  parameter_drift=drift, structural_retention=retention, retrieval_query_exposure=expected,
                  fp32_ranking=dict(metrics=fp32['metrics'], paired_query_ap_intervals=fp32['paired_query_ap_intervals']),
                  prepared_assay_data_not_consumed_by_current_trainer=read(data / 'ASSAY_MANIFEST.json')['counts']['2020'],
                  identities=[file_identity(p) for p in [Path(__file__), data/'roll_2020/TRAIN.csv.gz', data/'roll_2020/VALIDATION.csv.gz',
                              structural_path, epoch_path, before_path, after_path]],
                  running_training_sources_unchanged=True, training_modified=False, used_for_selection=False,
                  caveat='Observed degradation with unchanged structural heads establishes lost original readout performance, not complete erasure of all latent geometric information.')
    (DEST / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(data=tables, structural_retention=retention, retrieval_query_exposure=expected),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

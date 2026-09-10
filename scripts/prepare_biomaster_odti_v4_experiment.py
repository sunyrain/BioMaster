#!/usr/bin/env python3
"""Freeze R1 development protocol and full-target query panels before fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.odti_support_data_v3 import SupportStore
from build_biomaster_odti_v4_features import sha256, write_json

OUT = ROOT / 'outputs/biomaster_odti_v4_20260905'
DATA = ROOT / 'outputs/biomaster_v3_20260905/data'


def main():
    out = OUT / 'protocol'
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((DATA / 'DATA_MANIFEST_V3.json').read_text())
    path = DATA / 'RELATIONS_V3.csv.gz'
    assert sha256(path) == old['prepared_relations_sha256']
    frame = pd.read_csv(path)
    store = SupportStore.load(DATA / 'support_store')
    assert store.store_hash == old['store_hash']
    assert not set(store.entity_codes[frame.loc[frame.split.eq('train'), 'drug_feature_index']]) & set(store.entity_codes[frame.loc[~frame.split.eq('train'), 'drug_feature_index']])
    targets = np.sort(frame.target_feature_index.unique())
    protocol = {'status': 'PREPARING', 'stage': 'R1_AND_EXPLORATORY_SIGNED_EVIDENCE',
                'scope': 'historical V3 entity holdout; development regression only; legacy mixed-endpoint binary labels',
                'not_implemented': ['DrugCLIP geometry', 'atom-residue pair interaction', 'stereo adapter', 'endpoint/functional heads', 'backbone unfreezing'],
                'seeds': [20260905, 20260906, 20260907], 'max_epochs': 30, 'min_epochs': 8, 'patience': 5,
                'selection': 'validation full-panel macro known-positive retrieval AP; no test-based selection',
                'loss': 'measured BCE + equal-drug observed PN pairwise ranking; unknown labels excluded',
                'training_coverage': 'all 342031 measured training relations every epoch + equal-query stream',
                'initial_lr': 0.0002, 'weight_decay': 0.01, 'batch_size': 2048, 'query_batch': 64, 'query_max_items': 32,
                'signed_evidence': {'k_per_class': 16, 'cap_per_branch': 2., 'base_warmup_frozen_epochs': 2, 'branch_dropout': 0.2,
                                    'reference_delta': 'binary shared potential difference, no quantitative delta claim'},
                'candidate_targets': targets.tolist(), 'candidate_count': len(targets), 'data': old,
                'test_previously_used_in_v3': True, 'unknown_panel_entries_are_measured_negatives': False, 'panels': {}}
    write_json(out / 'PROTOCOL_V4.json', protocol)
    for split, size in [('validation', 512), ('test', 1024)]:
        part = frame.loc[frame.split.eq(split)]
        positive = part.loc[part.binary_label.eq(1)]
        eligible = positive.drug_feature_index.unique()
        salt = 'V3_DENSE' if split == 'test' else 'V4_VALIDATION_PANEL'
        queries = np.asarray(sorted(eligible, key=lambda d: hashlib.sha256(f'{salt}:{d}'.encode()).hexdigest())[:size])
        ds, ts = np.repeat(queries, len(targets)), np.tile(targets, len(queries))
        pairs = set(zip(positive.drug_feature_index, positive.target_feature_index))
        labels = np.array([(d,t) in pairs for d,t in zip(ds,ts)], dtype=bool).reshape(len(queries), len(targets))
        np.savez_compressed(out / f'{split}_panel.npz', drug_indices=queries, target_indices=targets, known_positive=labels)
        def progress(done, total):
            if done % 20000 < 1000 or done == total:
                print(json.dumps({'split': split, 'support_rows': done, 'total': total}), flush=True)
        evidence = store.cache_retrieve(out / f'{split}_support', ds, ts, k=16, progress=progress)
        assert evidence.indices.shape == (len(ds), 2, 16)
        protocol['panels'][split] = {'queries': len(queries), 'known_positive_pairs': int(labels.sum()),
            'selection': f'SHA256({salt}:drug_feature_index), eligible if at least one known positive',
            'panel_sha256': sha256(out / f'{split}_panel.npz'),
            'support': {n: str(getattr(evidence,n).filename) for n in ['indices', 'similarities', 'mask']}}
        write_json(out / 'PROTOCOL_V4.json', protocol)
    protocol['status'] = 'FROZEN'
    write_json(out / 'PROTOCOL_V4.json', protocol)
    print(json.dumps({'status': 'FROZEN', 'panel_sizes': {k:v['queries'] for k,v in protocol['panels'].items()}}), flush=True)


if __name__ == '__main__':
    main()

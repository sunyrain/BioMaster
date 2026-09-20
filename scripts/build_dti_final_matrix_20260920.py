#!/usr/bin/env python3
"""Export research membership and a read-only catalogue score snapshot.

No training, model downloads, worker changes, label-based selection or web deployment.
The ten-column score matrix has explicit missing values for unqualified additions.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_final_model_matrix_20260920'
CONFIG = ROOT / 'configs/dti_official_weights_20260920/MODEL_MATRIX_v3.json'
LIVE = ROOT / 'outputs/catalog_seven_models_20260916'
KEYS = ['drug_id', 'target_id']
SCORE_NAMES = {'drugclip': 'DrugCLIP', 'conplex': 'ConPLex', 'nesso': 'Nesso-1',
               'probematch': 'ProbeMatchDTI', 'dtbind': 'DTBind_occurrence'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = pd.DataFrame(json.loads(CONFIG.read_text())['models'])
    assert registry.id.is_unique
    current = registry.loc[registry.group.eq('CURRENT'), 'id'].tolist()
    additions = registry.loc[registry.group.eq('PRIORITY_ADD'), 'id'].tolist()
    intended = current + additions
    assert len(current) == 6 and len(additions) == 4
    snapshot_utc = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(f'file:{LIVE / "scores.sqlite"}?mode=ro', uri=True, timeout=15) as db:
        db.execute('BEGIN')
        predictions = pd.read_sql_query('SELECT * FROM predictions', db)
        states = pd.read_sql_query('SELECT * FROM target_status', db)
        db.rollback()
    assert not predictions.duplicated(['model'] + KEYS).any()
    completed = predictions.loc[predictions.status.eq('completed') & predictions.model.isin(SCORE_NAMES)]
    assert np.isfinite(completed.score).all()
    base_path = ROOT / 'outputs/model_agreement_20260917/SCORE_SNAPSHOT.csv.gz'
    a_path = ROOT / 'outputs/dtiam_a_catalog_20260917/DTIAM_A_720X384_SCORES.csv.gz'
    base = pd.read_csv(base_path, usecols=KEYS + ['drug_name', 'gene', 'drugclip', 'conplex'])
    release = pd.read_csv(a_path, usecols=['ligand_inchikey', 'target_chembl_id', 'dtiam_probability'])
    release = release.rename(columns={'ligand_inchikey': 'drug_id', 'target_chembl_id': 'target_id',
                                      'dtiam_probability': 'DTIAM_A'})
    frame = base.merge(release, on=KEYS, validate='one_to_one')
    live_scores = completed.pivot(index=KEYS, columns='model', values='score').reset_index()
    frame = frame.merge(live_scores, on=KEYS, how='left', validate='one_to_one').rename(columns=SCORE_NAMES)
    for name in additions:
        frame[name] = np.nan
    frame = frame[KEYS + ['drug_name', 'gene'] + intended].sort_values(KEYS).reset_index(drop=True)
    assert len(frame) == 720 * 384 and frame.drug_id.nunique() == 720 and frame.target_id.nunique() == 384
    assert not frame.duplicated(KEYS).any()
    assert frame.groupby('drug_id').size().eq(384).all() and frame.groupby('target_id').size().eq(720).all()
    assert np.isfinite(frame[intended].stack().dropna()).all()
    assert frame[additions].isna().all().all()
    matrix_name = 'CATALOG_720X384_SCORE_SNAPSHOT.csv.gz'
    frame.to_csv(OUT / matrix_name, index=False, compression={'method': 'gzip', 'mtime': 0})
    csv('TARGET_STATUS_SNAPSHOT.csv', states)
    csv('NONCOMPLETED_RECORDED_PAIRS.csv', predictions.loc[predictions.status.ne('completed')])
    coverage = []
    for model in intended:
        part = frame.loc[frame[model].notna()]
        counts = part.groupby('target_id').size()
        coverage.append(dict(model=model, scored_pairs=len(part), total_pairs=len(frame),
                             coverage_percent=100 * len(part) / len(frame),
                             targets_with_scores=len(counts), targets_720=int(counts.eq(720).sum()),
                             missing_pairs=len(frame) - len(part),
                             state='PENDING_ADAPTER_QUALIFICATION' if model in additions else
                             ('COMPLETE' if len(part) == len(frame) else 'PARTIAL_SEE_STATUS'),
                             snapshot_utc=snapshot_utc))
    coverage = pd.DataFrame(coverage)
    csv('MODEL_COVERAGE.csv', coverage)
    registry = registry.merge(coverage[['model', 'scored_pairs', 'coverage_percent', 'state']],
                              left_on='id', right_on='model', how='left', validate='one_to_one').drop(columns='model')
    csv('FINAL_MODEL_MATRIX.csv', registry)
    csv('PRIMARY_TEN_MODEL_MATRIX.csv', registry.loc[registry.id.isin(intended)])
    active = json.loads((ROOT / 'configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json').read_text())
    a_runs = pd.read_csv(ROOT / active['active_main_runs'])
    assert len(a_runs) == 45 and a_runs.arm.eq('A').all()
    replay = json.loads((OUT / 'DTIAM_USAGE_AUDIT.json').read_text())
    assert replay['status'] == 'PASS'
    frozen = json.loads((ROOT / 'outputs/dti_research_preparation_20260920/PACKAGE_MANIFEST.json').read_text())
    for name, metadata in frozen['files'].items():
        assert sha(ROOT / name) == metadata['sha256'], name
    common = frame[current].notna().all(axis=1)
    common_targets = frame.loc[common].groupby('target_id').size()
    manifest = dict(snapshot_utc=snapshot_utc, rows=len(frame), drugs=720, targets=384,
                    requested_model_columns=intended, current_models=current, pending_models=additions,
                    excluded_models=['ReTargetMap', 'DTIAM_B'], model_inventory_rows=len(registry),
                    current_six_common_scored_pairs=int(common.sum()),
                    current_six_targets_with_all_720=int(common_targets.eq(720).sum()),
                    new_training_started=False, new_matrix_inference_started=False,
                    dtiam_existing_predictor_smoke_test=True, comparison_metrics_computed=False,
                    existing_background_workers_changed=False,
                    missing_policy='Missing scores remain empty, never zero/negative. Pending additions excluded from current intersections.',
                    source_snapshots={str(p.relative_to(ROOT)): sha(p) for p in [base_path, a_path, CONFIG]},
                    live_sqlite='Read-only transactional snapshot; file hash intentionally not used for an actively written WAL database.',
                    artifacts={n: {'sha256': sha(OUT / n), 'bytes': (OUT / n).stat().st_size}
                               for n in [matrix_name, 'MODEL_COVERAGE.csv', 'TARGET_STATUS_SNAPSHOT.csv',
                                         'NONCOMPLETED_RECORDED_PAIRS.csv', 'FINAL_MODEL_MATRIX.csv',
                                         'PRIMARY_TEN_MODEL_MATRIX.csv', 'DTIAM_USAGE_AUDIT.json',
                                         'NEW_MODEL_SOURCE_AUDIT.json', 'TASK_WEIGHT_REMOTE_METADATA.json']},
                    limits=['Unlabelled catalogue is not a performance test.',
                            'Native scores differ in endpoints and units; rank within matched candidates.',
                            'Six current channels do not imply complete coverage.',
                            'Task checkpoint availability does not imply qualified inference.'])
    save(OUT / 'MANIFEST.json', manifest)
    save(OUT / 'VALIDATION.json', dict(status='PASS', frozen_v1_package_unchanged=True,
        unique_complete_catalogue_keys=True, excluded_model_columns_absent=True,
        pending_addition_scores_all_missing=True, finite_observed_scores=True,
        a_main_fits=45, new_b_fits=0, dtiam_replay_max_abs_difference=replay['max_abs_difference'],
        manifest_sha256=sha(OUT / 'MANIFEST.json'), producer_sha256=sha(Path(__file__))))
    print(coverage.to_string(index=False))
    print('Current six common scored pairs:', int(common.sum()))


if __name__ == '__main__':
    main()

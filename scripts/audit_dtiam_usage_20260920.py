#!/usr/bin/env python3
"""Read the author's actual entry points and replay a local frozen DTIAM predictor."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from dtiam_ab_common_20260912 import ROOT, SOURCE, FEATURE_COLUMNS, banks, table


def main():
    out = ROOT / 'outputs/dti_final_model_matrix_20260920'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'third_party/sota_dti_2026/DTIAM/code'
    code = (source / 'training_validation.py').read_text()
    assert 'TabularPredictor(label="y", eval_metric=eval_metric).fit(' in code
    assert 'training_validation.py dti yamanishi_08 warm_start' in (source / 'test.sh').read_text()
    assert 'list(comp_feat[str(cid)]) + list(prot_feat[pid])' in (source / 'utils.py').read_text()
    catalog = ROOT / 'outputs/dtiam_a_catalog_20260917'
    selection = json.loads((catalog / 'SELECTION.json').read_text())
    chosen = selection['selected']
    run = ROOT / 'outputs/biomaster_dtiam_ab_20260912' / f"kdki_inactive__seed_{chosen['seed']}"
    predictor_path = run / 'predictor'
    print('Loading the existing A predictor; no fitting.', flush=True)
    predictor = TabularPredictor.load(str(predictor_path), require_version_match=True)
    assert list(predictor.feature_metadata_in.get_features()) == FEATURE_COLUMNS
    validation = pd.read_parquet(SOURCE / 'COMMON_VALIDATION.parquet').drop_duplicates('pair_id').sort_values('pair_id')
    sample = validation.iloc[np.linspace(0, len(validation) - 1, 16, dtype=int)].copy()
    expected = pd.read_parquet(run / 'VALIDATION_PREDICTIONS.parquet').drop_duplicates('pair_id').set_index('pair_id').query_score
    values = predictor.predict_proba(table(sample, banks()), model=chosen['model'])[1].to_numpy()
    reference = expected.loc[sample.pair_id].to_numpy()
    delta = float(np.max(np.abs(values - reference)))
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert delta < 2e-6, delta
    pd.DataFrame({'pair_id': sample.pair_id.to_numpy(), 'frozen_score': reference,
                  'replayed_score': values}).to_csv(out / 'DTIAM_A_REPLAY.csv', index=False)
    result = {
        'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'author_test_sh_calls_training': True, 'author_task_predictor_calls_fit': True,
        'features': 'BerMol CLS768 + ESM2 t33 650M mean1280; first1022 residues, EOS included as in author code',
        'predictor': str(predictor_path.relative_to(ROOT)), 'selected': chosen,
        'positive_class': 1, 'feature_count': len(FEATURE_COLUMNS), 'replayed_pairs': len(sample),
        'max_abs_difference': delta, 'autogluon_version': importlib.metadata.version('autogluon.tabular'),
        'author_documented_autogluon_version': '0.5.2',
        'classification_not_numeric_affinity': True, 'new_training_started': False,
        'local_task_training_pairs': selection['training_pairs'],
        'catalog_rows_previously_scored': json.loads((catalog / 'MANIFEST.json').read_text())['rows'],
        'source_sha256': {str((source / n).relative_to(ROOT)): hashlib.sha256((source / n).read_bytes()).hexdigest()
                          for n in ['training_validation.py', 'test.sh', 'utils.py', 'data_process/extract_feature.py']},
        'selection_sha256': hashlib.sha256((catalog / 'SELECTION.json').read_bytes()).hexdigest(),
        'limitation': 'Locally fitted A task head, not author-released task weights or exact paper replication.',
    }
    (out / 'DTIAM_USAGE_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

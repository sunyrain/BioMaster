#!/usr/bin/env python3
"""Read completed validation artifacts and live progress without opening TEST."""
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from dtiam_ab_common_20260912 import OUT, SOURCE, basic, digest, probability_score, query_summary, write_json


def main():
    dest = OUT / 'interim_20260915'
    dest.mkdir(exist_ok=True)
    frame = pd.read_parquet(SOURCE / 'COMMON_VALIDATION.parquet')
    keys = ['pair_id', 'panel', 'binary_label']
    kd = frame.panel.eq('AFFINITY_KD_KI').to_numpy()
    inactive = frame.panel.eq('EXPLICIT_INACTIVE').to_numpy()
    part = frame[kd].reset_index(drop=True)
    sources = {str((SOURCE / 'COMMON_VALIDATION.parquet').relative_to(ROOT)):
               digest(SOURCE / 'COMMON_VALIDATION.parquet')}
    rows = []
    completed = []
    for path in sorted(OUT.glob('*__seed_*/SELECTION.json')):
        run = path.parent
        selection = json.loads(path.read_text())
        assert selection['status'] == 'COMPLETE_FIT_AND_VALIDATION_FROZEN'
        assert digest(run / 'CALIBRATION.json') == selection['calibration_sha256']
        assert digest(run / 'FIT_RETURNED.json') == selection['fit_sha256']
        pred = pd.read_parquet(run / 'VALIDATION_PREDICTIONS.parquet')
        assert frame[keys].equals(pred[keys]), run.name
        cal = json.loads((run / 'CALIBRATION.json').read_text())
        for view in ['native', 'query']:
            score = pred[view + '_score'].to_numpy(float)
            prob = pred[view + '_prob'].to_numpy(float)
            c = cal[view]
            assert np.allclose(prob, expit(c['slope'] * probability_score(score) + c['intercept']), atol=2e-7)
            m = basic(part.binary_label, score[kd], prob[kd], c['threshold'])
            q = query_summary(part, score[kd])
            selected = next(x for x in selection['validation_scores']
                            if x['model'] == selection['choices'][view])
            assert abs(m['ap'] - selected['ap']) < 1e-10
            assert abs(q['selection_score'] - selected['selection_score']) < 1e-10
            row = dict(family='DTIAM', arm=selection['arm'], seed=selection['seed'], view=view,
                       selected_model=selection['choices'][view], kdki_ap=m['ap'], kdki_auroc=m['auroc'],
                       kdki_recall=m['recall'], balanced_query_ap=q['selection_score'],
                       target_macro_ap=q['target']['macro_ap'], drug_macro_ap=q['drug']['macro_ap'],
                       target_p5=q['target']['p5'], drug_p5=q['drug']['p5'],
                       target_queries=q['target']['queries'], drug_queries=q['drug']['queries'],
                       explicit_inactive_rows=int(inactive.sum()),
                       explicit_inactive_fpr=float((prob[inactive] >= c['threshold']).mean()))
            for endpoint in ['IC50', 'EC50']:
                ix = frame.panel.eq('ACTIVITY_' + endpoint).to_numpy()
                row[endpoint.lower() + '_ap'] = basic(frame.loc[ix, 'binary_label'], score[ix], prob[ix], c['threshold'])['ap']
            rows.append(row)
        completed.append(run.name)
        for p in [path, run / 'CALIBRATION.json', run / 'VALIDATION_PREDICTIONS.parquet', run / 'FIT_RETURNED.json']:
            sources[str(p.relative_to(ROOT))] = digest(p)
    table = pd.DataFrame(rows)
    table.to_csv(dest / 'COMPLETED_VALIDATION_METRICS.csv', index=False)
    averaged = table.groupby(['arm', 'view'])[['kdki_ap', 'kdki_auroc', 'balanced_query_ap',
        'explicit_inactive_fpr', 'kdki_recall', 'ic50_ap', 'ec50_ap']].agg(['mean', 'std', 'count'])
    averaged.columns = ['_'.join(c) for c in averaged.columns]
    averaged.reset_index().to_csv(dest / 'SEED_MEAN_VALIDATION_METRICS.csv', index=False)

    status = json.loads((OUT / 'STATUS.json').read_text())
    child = json.loads((OUT / 'CURRENT_CHILD.json').read_text())
    current = OUT / status['current_suite']
    text = Path(child['log']).read_text()
    epochs = re.findall(r'Epoch (\d+) \(Update (\d+)\).*?Best Epoch: (\d+)', text)
    now = datetime.now(timezone.utc)
    elapsed = (now - datetime.fromisoformat(child['started_utc'])).total_seconds() / 3600
    stages = {}
    for seed in [20260921, 20260922]:
        run = OUT / f'all_inactive__seed_{seed}'
        stages[str(seed)] = {
            'torch_hours': json.loads((run / 'ISOLATED_CORE/MODEL_NeuralNetTorch.json').read_text())['seconds'] / 3600,
            'large_hours': json.loads((run / 'ISOLATED_CORE/MODEL_LightGBMLarge.json').read_text())['seconds'] / 3600,
            'fastai_ensemble_hours': json.loads((run / 'FASTAI_COMPLETE.json').read_text())['seconds'] / 3600}
    expected = {k: float(np.mean([x[k] for x in stages.values()])) for k in stages['20260921']}
    assert child['args'][-1] == 'NeuralNetTorch', 'ETA formula must be revised if stage changed'
    remaining = max(0, expected['torch_hours'] - elapsed) + expected['large_hours'] + expected['fastai_ensemble_hours'] + .05 + .5
    proc = Path(f'/proc/{child["pid"]}/status').read_text()
    rss = int(re.search(r'VmRSS:\s+(\d+)', proc)[1]) / 1024**2
    live = dict(utc=now.isoformat(), runner=status, completed_suites=completed, total_suites=6,
        queue_alive=Path(f'/proc/{status["pid"]}').exists(), child_pid=child['pid'],
        child_model=child['args'][-1], child_elapsed_hours=elapsed, child_RSS_GiB=rss,
        completed_current_learners=[p.stem.removeprefix('MODEL_') for p in sorted((current / 'ISOLATED_CORE').glob('MODEL_*.json'))],
        latest_epoch=int(epochs[-1][0]), latest_updates=int(epochs[-1][1]), best_epoch=int(epochs[-1][2]),
        log_modified_utc=datetime.fromtimestamp(Path(child['log']).stat().st_mtime, timezone.utc).isoformat(),
        cgroup_memory_GiB=int(Path('/sys/fs/cgroup/memory.current').read_text()) / 1024**3,
        cgroup_memory_limit_GiB=int(Path('/sys/fs/cgroup/memory.max').read_text()) / 1024**3,
        cgroup_memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
        root_free_GiB=shutil.disk_usage('/').free / 1024**3,
        data_free_GiB=shutil.disk_usage(ROOT).free / 1024**3,
        test_gate_open=(OUT / 'TEST_GATE.json').exists(),
        current_errors=[str(p.relative_to(OUT)) for p in [OUT / 'ERROR.json', current / 'ERROR.json'] if p.exists()],
        eta=dict(previous_B_stage_hours=stages, current_torch_expected_total_hours=expected['torch_hours'],
            central_remaining_hours=remaining, remaining_planning_hours=[3, 7],
            central_finish_utc=(now + timedelta(hours=remaining)).isoformat(),
            earliest_planning_finish_utc=(now + timedelta(hours=3)).isoformat(),
            latest_planning_finish_utc=(now + timedelta(hours=7)).isoformat(),
            common_validation_allowance_hours=.05, test_report_allowance_hours=.5,
            caveat='Stage durations extrapolated from two completed B seeds. TEST/report allowance unmeasured; early stopping and interruptions can change ETA.'),
        validation_only=True, test_predictions_read=False,
        validation_pairs=int(kd.sum()), validation_positive=int(part.binary_label.sum()),
        validation_negative=int((part.binary_label == 0).sum()), sources=sources)
    write_json(dest / 'PROGRESS.json', live)
    print(table[table.view.eq('query')].to_string(index=False))
    print(json.dumps({k: live[k] for k in ['utc', 'latest_epoch', 'best_epoch', 'test_gate_open', 'eta']}, indent=2))


if __name__ == '__main__':
    main()

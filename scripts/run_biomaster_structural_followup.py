#!/usr/bin/env python3
"""Supervise data preparation, structural pretraining and one development fit.

Existing workers are adopted by exact script/argument match. Failures are
reported explicitly; later stages never run on an incomplete dataset.
"""
import fcntl
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_biomaster_unified_interaction import write_json
from biomaster.odti_pockets_v3 import file_identity

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
DATA = OUT / 'structural_data'
LOGS = OUT / 'program_logs'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def active(script, extra=()):
    for p in psutil.process_iter(['pid', 'cmdline', 'status']):
        args = p.info['cmdline'] or []
        if len(args) >= 2 and args[0].endswith('python') and args[1] == script and all(x in args[2:] for x in extra):
            if p.info['status'] != psutil.STATUS_ZOMBIE:
                return p.info['pid']
    return None


def launch(script, args=(), log_name=None):
    log = (LOGS / (log_name or (Path(script).stem + '.log'))).open('a')
    child = subprocess.Popen([sys.executable, script, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log.close(); return child.pid


def run_blocking(script, args=()):
    log = LOGS / (Path(script).stem + '_' + (args[0] if args else 'run') + '.log')
    with log.open('a') as handle:
        result = subprocess.run([sys.executable, script, *args], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} failed with code {result.returncode}; see {log}')


def main():
    LOGS.mkdir(exist_ok=True)
    lock = (OUT / '.structural_program.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    fetch = 'scripts/fetch_biomaster_structural_complexes.py'
    mapper = 'scripts/prepare_biomaster_structural_training.py'
    encoder = 'scripts/encode_biomaster_structural_online.py'
    pretrainer = 'scripts/pretrain_biomaster_structural_interaction.py'
    downstream = 'scripts/train_biomaster_pocket_precision.py'
    status_path = OUT / 'PROGRAM_STATUS.json'
    identities = [file_identity(ROOT / p) for p in [fetch, mapper, encoder, pretrainer, downstream,
                   'biomaster/structural_complex.py', 'biomaster/structural_encoding.py', 'biomaster/structural_training.py',
                   'biomaster/pocket_precision.py', 'configs/biomaster_structural_pretraining_20260906.json']]
    def status(stage, **extra):
        for identity in identities:
            if file_identity(identity['path']) != identity:
                raise ValueError('program source changed while running')
        event = dict(status=stage, utc=datetime.now(timezone.utc).isoformat(), supervisor_pid=__import__('os').getpid(), elapsed_seconds=time.monotonic()-started,
                     source_identities=identities, downstream_ranking_complete=False, **extra)
        write_json(status_path, event)
        print(json.dumps({k: v for k, v in event.items() if k != 'source_identities'}), flush=True)
    try:
        retries = {fetch: 0, mapper: 0, encoder: 0}
        while True:
            fs = read(DATA / 'complexes_2020/STATUS.json')
            ms = read(DATA / 'training_2020/MAPPING_MANIFEST.json')
            es = read(DATA / 'training_2020/ONLINE_ENCODING_STATUS.json')
            complete = {fetch: fs.get('status') == 'COMPLETE', mapper: ms.get('status') == 'COMPLETE', encoder: es.get('status') == 'COMPLETE'}
            pids = {}
            for script, args in [(fetch, []), (mapper, ['map', '--watch-downloads']), (encoder, [])]:
                if complete[script]:
                    continue
                pid = active(script, ['map'] if script == mapper else [])
                if pid is None:
                    if retries[script] >= 3:
                        raise RuntimeError(f'{script} failed to finish after three supervised launches')
                    pid = launch(script, args); retries[script] += 1
                pids[Path(script).stem] = pid
            status('DATA_PREPARATION_RUNNING', pids=pids, downloads=fs, mapping_counts=read(DATA / 'training_2020/MAPPING_STATUS.json'),
                   encoded=es.get('completed'), data_ready=all(complete.values()))
            if all(complete.values()):
                break
            time.sleep(30)
        status('OVERLAP_AND_GEOMETRY_AUDIT_RUNNING')
        if not (DATA / 'training_2020/OVERLAP_AUDIT.json').exists():
            run_blocking(mapper, ['audit'])
        if not (DATA / 'training_2020/MANIFEST.json').exists():
            run_blocking(mapper, ['finalize-online'])
        manifest = read(DATA / 'training_2020/MANIFEST.json')
        if manifest.get('status') != 'STRUCTURAL_DATA_READY' or not manifest.get('all_encoded_records_verified'):
            raise ValueError('structural admission not complete')
        status('STRUCTURAL_PRETRAINING_RUNNING', counts=manifest['counts'])
        run_blocking(pretrainer)
        checkpoint = OUT / 'structural_pretraining/cutoff_2020/seed_20260921/STRUCTURAL_PRETRAINED.pt'
        if not checkpoint.is_file():
            raise ValueError('structural training did not produce a completed checkpoint')
        status('DOWNSTREAM_2020_DEVELOPMENT_RUNNING', checkpoint=file_identity(checkpoint))
        run_blocking(downstream, ['--cutoff', '2020', '--seed', '20260921', '--structural-checkpoint', str(checkpoint)])
        status('ONE_DEVELOPMENT_FIT_COMPLETE_NOT_SELECTED',
               downstream_fit='training/structurally_pretrained/cutoff_2020/seed_20260921',
               remaining='2018 structural/data training, additional seeds, paired controls, and regression evaluations')
    except Exception as error:
        write_json(status_path, dict(status='FAILED_REQUIRES_REVIEW', error=repr(error), elapsed_seconds=time.monotonic()-started,
                                     source_identities=identities))
        raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Attach read-only contact diagnostics to the running structural training job."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_biomaster_unified_interaction import write_json
from biomaster.odti_pockets_v3 import file_identity

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
TRAIN = OUT / 'structural_pretraining/cutoff_2020/seed_20260921'
DIAGNOSTICS = OUT / 'contact_diagnostics'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    DIAGNOSTICS.mkdir(exist_ok=True)
    lock = (DIAGNOSTICS / '.follow.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    phases = [('first_200_updates', TRAIN / 'LATEST.pt'), ('completed_pretraining', TRAIN / 'STRUCTURAL_PRETRAINED.pt')]
    started = time.monotonic()
    identity = file_identity(ROOT / 'scripts/diagnose_biomaster_structural_contact_learning.py')
    def status(value, **extra):
        event = dict(status=value, utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                     elapsed_seconds=time.monotonic()-started, diagnostic_script=identity,
                     used_for_selection=False, **extra)
        write_json(DIAGNOSTICS / 'STATUS.json', event)
        print(json.dumps(event), flush=True)
    try:
        for phase, checkpoint in phases:
            if (DIAGNOSTICS / phase / 'RESULT.json').exists():
                continue
            while True:
                program = read(OUT / 'PROGRAM_STATUS.json')
                training = read(TRAIN / 'STATUS.json')
                # A post-pretraining downstream failure does not invalidate the
                # already completed structural checkpoint's contact diagnostic.
                ready = checkpoint.exists() and (phase != 'first_200_updates' or training.get('updates', 0) >= 200)
                if ready:
                    break
                if program.get('status') == 'FAILED_REQUIRES_REVIEW':
                    raise RuntimeError(f'upstream program failed before {phase}: {program.get("error")}')
                status('WAITING_FOR_CHECKPOINT', phase=phase, program_status=program.get('status'),
                       structural_updates=training.get('updates', 0))
                time.sleep(30)
            if file_identity(identity['path']) != identity:
                raise ValueError('diagnostic source changed during follow-up')
            status('DIAGNOSTIC_RUNNING', phase=phase)
            folder = DIAGNOSTICS / phase
            folder.mkdir(exist_ok=True)
            command = [sys.executable, 'scripts/diagnose_biomaster_structural_contact_learning.py',
                       '--checkpoint', str(checkpoint), '--output', str(folder)]
            for attempt in range(3):
                with (folder / 'RUN.log').open('a') as handle:
                    result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    break
                if 'checkpoint changed during snapshot' not in (folder / 'RUN.log').read_text():
                    raise RuntimeError(f'contact diagnostic failed: {folder / "RUN.log"}')
                time.sleep(5)
            if not (folder / 'RESULT.json').exists():
                raise RuntimeError('checkpoint snapshot retries exhausted')
            status('DIAGNOSTIC_COMPLETE', phase=phase, result=str(folder / 'RESULT.json'))
        status('ALL_STRUCTURAL_CONTACT_DIAGNOSTICS_COMPLETE')
    except Exception as error:
        status('FAILED_REQUIRES_REVIEW', error=repr(error))
        raise


if __name__ == '__main__':
    main()

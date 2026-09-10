#!/usr/bin/env python3
"""Supervise the authorized data-to-training run with explicit readiness gates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_biomaster_context_full_20260908 import write_json, OUT as DATA
from biomaster.context_full_training import file_sha

OUT=ROOT/'outputs/biomaster_context_full_20260908'
RUN=OUT/'training/cutoff_2020/seed_20260921'


def main():
    interrupted=[];child=None
    signal.signal(signal.SIGTERM,lambda *_:interrupted.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:interrupted.append('SIGINT'))
    preparation=json.loads((OUT/'PREPARATION_PROCESS.json').read_text())
    gate=json.loads((OUT/'LAUNCH_GATE.json').read_text())
    if not gate['tests_passed'] or not gate['real_data_resume_verified']:
        raise ValueError('verified training launch gate required')
    for path,digest in gate['sources'].items():
        if file_sha(path)!=digest:raise ValueError('launch source/config changed: '+path)
    def write(state,**more):
        write_json(OUT/'STATUS.json',dict(status=state,utc=datetime.now(timezone.utc).isoformat(),
            supervisor_pid=os.getpid(),training_pid=child.pid if child else None,training_output=str(RUN),
            promotion_eligible=False,**more))
    while True:
        if interrupted:
            # Preparation owns a dedicated process group (Python workers and
            # P2Rank subprocesses); never leave that job running after stop.
            try:os.killpg(preparation['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            write('STOPPED_BY_SIGNAL',training_started=False);return
        data_state=json.loads((DATA/'STATUS.json').read_text()) if (DATA/'STATUS.json').exists() else {}
        if data_state.get('status')=='PREPARATION_FAILED':
            write('STOPPED_DATA_PREPARATION_FAILED',data=data_state,training_started=False);return
        if data_state.get('status')=='FULL_DATA_READY':break
        try:os.kill(preparation['pid'],0)
        except ProcessLookupError:
            write('STOPPED_PREPARATION_PROCESS_EXITED',data=data_state,training_started=False);return
        write('WAITING_FOR_COMPLETE_AUDITED_DATA',data=data_state,training_started=False)
        time.sleep(10)
    for path,digest in gate['sources'].items():
        if file_sha(path)!=digest:raise ValueError('source/config changed while preparing data: '+path)
    RUN.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u',str(ROOT/'scripts/train_biomaster_context_full_20260908.py'),'--output',str(RUN)]
    if (RUN/'LATEST.pt').exists():command.append('--resume')
    with (OUT/'TRAINING.log').open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        write_json(OUT/'TRAINING_PROCESS.json',dict(pid=child.pid,command=command,started_utc=datetime.now(timezone.utc).isoformat()))
        while child.poll() is None:
            if interrupted:child.send_signal(signal.SIGTERM)
            state=json.loads((RUN/'STATUS.json').read_text()) if (RUN/'STATUS.json').exists() else {}
            write('TRAINING_RUNNING',training_started=True,training=state);time.sleep(10)
        state=json.loads((RUN/'STATUS.json').read_text()) if (RUN/'STATUS.json').exists() else {}
        write('TRAINING_EXITED',training_started=True,exit_code=child.returncode,training=state)


if __name__=='__main__':main()

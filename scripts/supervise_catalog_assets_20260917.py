#!/usr/bin/env python3
"""Resume cached DTBind graph ranges after network failures, without duplicate fetches."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/frontier_dti_20260916'

def write_state(**value):
    p=OUT/'dtbind/FETCH_SUPERVISOR.tmp'
    p.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**value),indent=2))
    p.replace(OUT/'dtbind/FETCH_SUPERVISOR.json')

def main():
    try:
        pid=int((OUT/'fetch_dtbind.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'fetch_frontier_dti_assets_20260916.py' in cmd:
            raise RuntimeError(f'A downloader is already running: {pid}')
    except (OSError,ValueError):pass
    for attempt in range(1,25):
        write_state(state='starting',attempt=attempt)
        with (OUT/'fetch_dtbind.log').open('a') as log:
            log.write(f'\nNetwork recovery attempt {attempt} at {datetime.now(timezone.utc).isoformat()}\n');log.flush()
            child=subprocess.Popen([str(ROOT/'.venvs/frontier_dti/bin/python'),'-u','scripts/fetch_frontier_dti_assets_20260916.py','dtbind','--catalog'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (OUT/'fetch_dtbind.pid').write_text(str(child.pid))
            write_state(state='running',attempt=attempt,child_pid=child.pid)
            code=child.wait()
        if code==0:
            write_state(state='completed',attempt=attempt);return
        write_state(state='retry_wait',attempt=attempt,exit_code=code,retry_seconds=60)
        time.sleep(60)
    write_state(state='failed',attempts=24,reason='network retries exhausted')
if __name__=='__main__':main()

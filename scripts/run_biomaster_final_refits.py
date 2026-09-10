#!/usr/bin/env python3
"""Freeze the architecture, then refit the declared three seeds through 2022."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_biomaster_unified_interaction import write_json
from biomaster.odti_pockets_v3 import file_identity

OUT=ROOT/'outputs/biomaster_best_model_20260906'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-anchored',action='store_true');a=p.parse_args()
    lock=(OUT/'FINAL_REFIT_CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.wait_for_anchored:
        while True:
            dependency=OUT/'ANCHORED_STATUS.json'
            state=json.loads(dependency.read_text()) if dependency.exists() else {'status':'PENDING'}
            if state['status']=='COMPLETE':break
            if state['status']=='FAILED':raise RuntimeError('anchored ablation failed')
            write_json(OUT/'FINAL_REFIT_STATUS.json',dict(status='WAITING_FOR_ANCHORED',
                utc=datetime.now(timezone.utc).isoformat(),anchored_completed=state.get('completed',0)))
            time.sleep(10)
    with (OUT/'logs/final_selection.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/select_biomaster_final_architecture.py'),'--freeze'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    selected=json.loads((OUT/'FINAL_ARCHITECTURE_SELECTION.json').read_text())
    jobs=[('global_refit','global',s) for s in [20260921,20260922,20260923]]
    if selected['selected_variant']!='parent':
        jobs += [('refine',selected['selected_variant'],s) for s in [20260921,20260922,20260923]]
    files=[]
    for number,(mode,variant,seed) in enumerate(jobs,1):
        write_json(OUT/'FINAL_REFIT_STATUS.json',dict(status='RUNNING',job=number,jobs=len(jobs),mode=mode,variant=variant,
            seed=seed,cutoff=2022,utc=datetime.now(timezone.utc).isoformat()))
        with (OUT/'logs'/f'final_refit_{mode}_{variant}_{seed}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/fit_biomaster_selected_model.py'),
                '--cutoff','2022','--seed',str(seed),'--mode',mode,'--variant',variant],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        folder=OUT/'final_fit/cutoff_2022'/(variant if mode=='refine' else 'global_refit')/f'seed_{seed}'
        r=json.loads((folder/'RESULT.json').read_text())
        if r['train_max_year']!=2022 or r['test_labels_used'] or r['smoke_only']:raise ValueError('invalid temporal refit')
        if file_identity(Path(r['checkpoint']['path']))!=r['checkpoint']:raise ValueError('checkpoint changed')
        files.append(dict(mode=mode,variant=variant,seed=seed,checkpoint=r['checkpoint'],result=file_identity(folder/'RESULT.json')))
    write_json(OUT/'FINAL_REFIT_STATUS.json',dict(status='COMPLETE',jobs=len(jobs),files=files,
        final_selection=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'),driver=file_identity(Path(__file__))))


if __name__=='__main__':
    try:main()
    except BlockingIOError:raise
    except BaseException as error:
        write_json(OUT/'FINAL_REFIT_STATUS.json',dict(status='FAILED',error=str(error),
            utc=datetime.now(timezone.utc).isoformat(),driver=file_identity(Path(__file__))))
        raise

#!/usr/bin/env python3
"""Run selected-parent local comparisons with two bounded GPU workers."""
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
CONFIG=ROOT/'configs/biomaster_best_model_refinement_20260906.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-for-representations',action='store_true')
    args=parser.parse_args();config=json.loads(CONFIG.read_text())
    controller_lock=(OUT/'ANCHOR_CONTROLLER.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.wait_for_representations:
        while True:
            dependency=OUT/'MOLECULAR_CONTROL_STATUS.json'
            state=json.loads(dependency.read_text()) if dependency.exists() else {'status':'PENDING'}
            if state['status']=='COMPLETE':break
            if state['status']=='FAILED':raise RuntimeError('molecular control comparison failed')
            write_json(OUT/'ANCHORED_STATUS.json',dict(status='WAITING_FOR_REPRESENTATIONS',
                utc=datetime.now(timezone.utc).isoformat(),molecular_control_job=state.get('job'),
                note='waiting for the full 54-run global comparison including fingerprint controls'))
            time.sleep(10)
    for script,arguments in [('select_biomaster_global_parent.py',['--freeze-global']),('verify_biomaster_anchored_parent.py',[])]:
        with (OUT/'logs'/f'anchor_{script}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*arguments],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    for variant in ['global','geometry']:
        with (OUT/'logs'/f'anchor_smoke_{variant}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/fit_biomaster_selected_model.py'),
                '--cutoff','2018','--seed','20260921','--variant',variant,'--smoke-steps','5'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[dict(seed=seed,cutoff=cutoff,variant=variant) for seed in config['seeds']
          for cutoff in config['cutoffs'] for variant in config['variants']]
    active=[];finished=[];next_job=0
    def status(kind):
        write_json(OUT/'ANCHORED_STATUS.json',dict(status=kind,utc=datetime.now(timezone.utc).isoformat(),
            jobs=len(jobs),completed=len(finished),finished=finished,
            running=[dict(**item['job'],pid=item['process'].pid,log=item['name']) for item in active],
            driver=file_identity(Path(__file__)),configuration=file_identity(CONFIG)))
    try:
        while active or next_job<len(jobs):
            while next_job<len(jobs) and len(active)<config['maximum_concurrent_workers']:
                job=jobs[next_job];next_job+=1
                name=f"anchor_{job['cutoff']}_{job['variant']}_{job['seed']}.log"
                log=(OUT/'logs'/name).open('a')
                process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/fit_biomaster_selected_model.py'),
                    '--cutoff',str(job['cutoff']),'--seed',str(job['seed']),'--variant',job['variant']],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT)
                active.append(dict(job=job,process=process,log=log,name=name));status('RUNNING')
            changed=False
            for item in list(active):
                returncode=item['process'].poll()
                if returncode is None:continue
                item['log'].close();active.remove(item)
                if returncode:raise RuntimeError(f"fitting failed ({returncode}): {item['name']}")
                finished.append(item['job']);changed=True
            if changed:status('RUNNING')
            if active:time.sleep(1)
        status('COMPLETE')
    except BaseException as error:
        for item in active:
            item['process'].terminate();item['process'].wait();item['log'].close()
        write_json(OUT/'ANCHORED_STATUS.json',dict(status='FAILED',error=str(error),finished=finished,
            utc=datetime.now(timezone.utc).isoformat(),driver=file_identity(Path(__file__))))
        raise
    finally:controller_lock.close()


if __name__=='__main__':
    try:main()
    except BlockingIOError:
        # Another owner of the controller lock must keep its own status.
        raise
    except BaseException as error:
        current=json.loads((OUT/'ANCHORED_STATUS.json').read_text()) if (OUT/'ANCHORED_STATUS.json').exists() else {}
        if current.get('status')!='FAILED':
            write_json(OUT/'ANCHORED_STATUS.json',dict(status='FAILED',error=str(error),phase='dependency_or_preflight',
                utc=datetime.now(timezone.utc).isoformat(),driver=file_identity(Path(__file__))))
        raise

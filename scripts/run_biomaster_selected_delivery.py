#!/usr/bin/env python3
"""Serial GPU queue: frozen regression scoring, evaluation, full-fit deployment."""
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


def state(status,**kwargs):
    write_json(OUT/'DELIVERY_STATUS.json',dict(status=status,utc=datetime.now(timezone.utc).isoformat(),**kwargs))


def run(script,args,log):
    state('RUNNING',script=script,args=args,log=log)
    with (OUT/'logs'/log).open('a') as stream:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-refits',action='store_true');a=p.parse_args()
    lock=(OUT/'DELIVERY_CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.wait_for_refits:
        while True:
            path=OUT/'FINAL_REFIT_STATUS.json';prior=json.loads(path.read_text()) if path.exists() else {}
            if prior.get('status')=='COMPLETE':break
            if prior.get('status')=='FAILED':raise ValueError('temporal refit failed')
            state('WAITING_FOR_REFITS',refit_status=prior.get('status'));time.sleep(10)
    run('evaluate_biomaster_selected_regressions.py',['--phase','all'],'selected_regressions.log')
    run('evaluate_biomaster_selected_development_age.py',['--phase','final'],'selected_final_age.log')
    run('prepare_biomaster_selected_fullfit.py',[],'selected_fullfit_data.log')
    selected=json.loads((OUT/'FINAL_ARCHITECTURE_SELECTION.json').read_text());seed=selected['deployment_seed']
    run('fit_biomaster_selected_model.py',['--cutoff','2025','--seed',str(seed),'--mode','global_refit','--variant','global'],
        'selected_fullfit_global.log')
    variant=selected['selected_variant']
    if variant!='parent':
        run('fit_biomaster_selected_model.py',['--cutoff','2025','--seed',str(seed),'--variant',variant],'selected_fullfit_refine.log')
    folder=OUT/'final_fit/cutoff_2025'/('global_refit' if variant=='parent' else variant)/f'seed_{seed}'
    result=json.loads((folder/'RESULT.json').read_text())
    if result['train_max_year']!=2025 or result['smoke_only'] or result['training_role']!='fullfit_deployment':
        raise ValueError('not a valid selected full fit')
    if result['checkpoint']!=file_identity(folder/'BEST.pt'):raise ValueError('full fit checkpoint drift')
    write_json(OUT/'DEPLOYMENT_SELECTION.json',dict(status='SELECTED',checkpoint_sha256=result['checkpoint']['sha256'],
        checkpoint=result['checkpoint'],architecture_selection=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'),
        result=file_identity(folder/'RESULT.json'),training_data=file_identity(OUT/'data/fullfit_2025/MANIFEST.json'),
        regression_result=file_identity(OUT/'regression/RESULT.json'),seed=seed,downstream_cutoff=2025,
        fullfit_weights_have_no_independent_test_score=True,driver=file_identity(Path(__file__))))
    state('READY_FOR_EXPORT',checkpoint=result['checkpoint'],selected_seed=seed,final_package_delivered=False)


if __name__=='__main__':
    try:main()
    except BlockingIOError:raise
    except BaseException as e:
        state('FAILED',error=str(e));raise

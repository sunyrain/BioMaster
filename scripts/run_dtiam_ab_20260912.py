#!/usr/bin/env python3
"""Durable serial queue: exact features -> six native suites -> locked TEST -> report."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
import psutil

from dtiam_ab_common_20260912 import ROOT,OUT,FEATURES,ARMS,SEEDS,now,digest,write_json

AGPY=ROOT/'.venv_dtiam_compat/bin/python'


def status(stage,**extra):
    value=dict(stage=stage,updated_utc=now(),pid=os.getpid(),**extra)
    write_json(OUT/'STATUS.json',value);print(json.dumps(value),flush=True)


def command(args,log_path):
    with log_path.open('a') as log:
        process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        write_json(OUT/'CURRENT_CHILD.json',dict(pid=process.pid,args=list(map(str,args)),started_utc=now(),log=str(log_path)))
        while process.poll() is None:
            try:
                pp=psutil.Process(process.pid)
                resident=sum(p.memory_info().rss for p in [pp]+pp.children(recursive=True) if p.is_running())
                if resident>78*2**30:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError('Actual training RSS exceeded 78 GiB; run preserved for resource adjustment')
            except psutil.NoSuchProcess:pass
            time.sleep(2)
        result=process.returncode
    if result:
        raise RuntimeError(f'Child exited {result}; see {log_path}')


def verify_frozen():
    m=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for name,sha in {**m['inputs'],**m['code'],**m['frozen_inputs']}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Frozen file changed: '+name)
    assert digest(OUT/'PROTOCOL.json')==m['protocol_sha256']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_frozen()
    write_json(OUT/'PROCESS.json',dict(pid=os.getpid(),started_utc=now(),command='python -u scripts/run_dtiam_ab_20260912.py'))
    while not (FEATURES/'MANIFEST.json').exists():
        if (FEATURES/'ERROR.json').exists():raise RuntimeError('Feature extraction failed; inspect features/ERROR.json')
        feature_process=json.loads((OUT/'FEATURE_PROCESS.json').read_text())
        try:os.kill(feature_process['pid'],0)
        except ProcessLookupError:raise RuntimeError('Feature process exited before feature completeness gate')
        detail=json.loads((FEATURES/'STATUS.json').read_text()) if (FEATURES/'STATUS.json').exists() else {}
        status('FEATURE_PREPARATION',feature_progress=detail,total_suites=6,completed_suites=0)
        time.sleep(20)
    fm=json.loads((FEATURES/'MANIFEST.json').read_text());assert fm['status']=='COMPLETE'
    for name,sha in fm['files'].items():assert digest(ROOT/name)==sha
    verify_frozen()
    if not (OUT/'RESOURCE_PROFILE.json').exists():
        status('FULL_B_MEMORY_PROFILE',total_suites=6,completed_suites=0)
        command([str(AGPY),'-u','scripts/profile_dtiam_ab_resources_20260912.py'],OUT/'RESOURCE_PROFILE.log')
    assert json.loads((OUT/'RESOURCE_PROFILE.json').read_text())['status']=='PASS_FULL_B_PREPROCESS_ONLY'
    completed=0
    for seed in SEEDS:
        for arm in ARMS:
            name=f'{arm}__seed_{seed}';run=OUT/name;run.mkdir(exist_ok=True)
            if not (run/'SELECTION.json').exists():
                status('TRAINING',current_suite=name,completed_suites=completed,total_suites=6)
                base=[str(AGPY),'-u','scripts/train_dtiam_ab_20260912.py','--arm',arm,'--seed',str(seed)]
                command(base+['--core-only'],run/'RUN.log')
                command(base+['--fastai'],run/'FASTAI.log')
                command(base,run/'VALIDATION.log')
            selected=json.loads((run/'SELECTION.json').read_text())
            assert selected['status']=='COMPLETE_FIT_AND_VALIDATION_FROZEN'
            completed+=1
    verify_frozen()
    selections={str((OUT/f'{a}__seed_{s}'/'SELECTION.json').relative_to(ROOT)):
        digest(OUT/f'{a}__seed_{s}'/'SELECTION.json') for a in ARMS for s in SEEDS}
    write_json(OUT/'TEST_GATE.json',dict(status='ALL_SIX_FITS_AND_VALIDATION_CHOICES_FROZEN',opened_utc=now(),selection_hashes=selections))
    for arm in ARMS:
        for seed in SEEDS:
            status('GATED_TEST_INFERENCE',current_suite=f'{arm}__seed_{seed}',completed_suites=6,total_suites=6)
            command([str(AGPY),'-u','scripts/train_dtiam_ab_20260912.py','--test','--arm',arm,'--seed',str(seed)],
                OUT/f'{arm}__seed_{seed}'/'TEST.log')
    status('REPORTING',completed_suites=6,total_suites=6)
    command([sys.executable,'-u','scripts/report_dtiam_ab_20260912.py'],OUT/'REPORT.log')
    verify_frozen()
    summary=json.loads((OUT/'SUMMARY.json').read_text());assert summary['status']=='COMPLETE_6_NATIVE_DTIAM_SUITES'
    status('COMPLETE',completed_suites=6,total_suites=6,summary='SUMMARY.json')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write_json(OUT/'ERROR.json',dict(utc=now(),error=repr(error),traceback=traceback.format_exc()))
        status('FAILED_REQUIRES_INSPECTION',error=repr(error))
        raise

#!/usr/bin/env python3
"""Prioritize the handed-off SPR384, then labeled controls; isolate long proteins."""
import json
import os
import subprocess
import time
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/frontier_dti_20260916/nesso'

def save(value):
    p=OUT/'STATUS.tmp';p.write_text(json.dumps(value,indent=2));p.replace(OUT/'STATUS.json')

def main():
    data=pd.read_csv(OUT.parent/'UNIQUE_PAIRS.csv')
    checkpoint=ROOT/'.cache/nesso/huggingface/models--recursionpharma--nesso/snapshots/1896c84c7186c506c7efd79051480809d51098bf'
    commands=[str(ROOT/'.venvs/nesso/bin/nesso'),'predict',str(OUT/'inputs'),
        '--out_dir',str(OUT),'--checkpoint',str(checkpoint/'v1.0.0'),'--ccd',str(checkpoint/'ccd.pkl'),
        '--accelerator','gpu','--devices','1','--num_workers','2','--precision','bf16-mixed',
        '--recycling_steps','5','--no_kernels','--require_affinity','--seed','20260916']
    (OUT/'COMMAND.json').write_text(json.dumps(commands,indent=2))
    groups=[('SPR384',data[data.cohort.eq('SPR384')],None),('BINDINGDB_regular',data[data.cohort.eq('BINDINGDB479') & data.protein_length.le(2000)],None)]
    groups.extend((f'long_{r.pair_id}',data[data.pair_id.eq(r.pair_id)],900) for r in data[data.cohort.eq('BINDINGDB479') & data.protein_length.gt(2000)].itertuples())
    history=[];start=time.time()
    env=dict(os.environ,NESSO_CACHE=str(ROOT/'.cache/nesso'),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for stage,frame,timeout in groups:
        inputs=OUT/'queues'/stage;inputs.mkdir(parents=True,exist_ok=True)
        pending=0
        for row in frame.itertuples():
            if (OUT/'predictions'/row.pair_id/'affinity.json').exists():continue
            link=inputs/(row.pair_id+'.yaml')
            if not link.exists():link.symlink_to(OUT/'inputs'/link.name)
            pending+=1
        if not pending:continue
        command=commands.copy();command[2]=str(inputs)
        status=dict(state='inference',stage=stage,pending_in_stage=pending,elapsed_seconds=time.time()-start,pid=os.getpid(),long_pair_timeout_seconds=900)
        save(status);print(stage,pending,flush=True)
        with (OUT/'RUN.log').open('a') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            status['child_pid']=child.pid;save(status)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                code='timeout'
        history.append(dict(stage=stage,pending=pending,result=code,elapsed_seconds=time.time()-start))
        (OUT/'QUEUE_HISTORY.json').write_text(json.dumps(history,indent=2))
    count=len(list((OUT/'predictions').glob('*/affinity.json')))
    save(dict(state='completed' if count==len(data) else 'completed_with_missing',completed=count,total=len(data),elapsed_seconds=time.time()-start,pid=os.getpid()))

if __name__=='__main__':main()

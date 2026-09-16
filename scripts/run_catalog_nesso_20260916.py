#!/usr/bin/env python3
"""Queue full target-wise Nesso inference with live publication and bounded intermediates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.catalog_models import DIRECTORY,connect,save
OUT=ROOT/DIRECTORY/'nesso';OLD=ROOT/'outputs/frontier_dti_20260916/nesso'

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except OSError:return False

def status(**value):
    p=OUT/'STATUS.tmp';p.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),catalog_pairs=276480,**value),ensure_ascii=False,indent=2));p.replace(OUT/'STATUS.json')

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,default=0);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);db=connect(ROOT,write=True);started=time.time()
    drugs=pd.read_csv(OUT.parent/'DRUGS.csv');targets=pd.read_csv(OUT.parent/'TARGETS.csv')
    imported=set()
    def harvest(directory):
        records=[]
        for f in directory.glob('*/affinity.json'):
            if str(f) in imported:continue
            try:
                drug,target=f.parent.name.split('__');value=json.loads(f.read_text())['affinity_probability_binary']
                if drug not in set(drugs.drug_id) or target not in set(targets.target_id):continue
                records.append(dict(model='nesso',drug_id=drug,target_id=target,status='completed',score=float(value)));imported.add(str(f))
            except (ValueError,KeyError):continue
        if records:save(db,records)
    def count():return db.execute("SELECT COUNT(*) FROM predictions WHERE model='nesso' AND status='completed'").fetchone()[0]
    while args.wait_pid and alive(args.wait_pid):
        harvest(OLD/'predictions');status(state='waiting_gpu',waiting_for_pid=args.wait_pid,completed=count(),elapsed_seconds=time.time()-started)
        time.sleep(30)
    harvest(OLD/'predictions')
    ckpt=ROOT/'.cache/nesso/huggingface/models--recursionpharma--nesso/snapshots/1896c84c7186c506c7efd79051480809d51098bf'
    env=dict(os.environ,NESSO_CACHE=str(ROOT/'.cache/nesso'),HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command_base=[str(ROOT/'.venvs/nesso/bin/nesso'),'predict',None,'--out_dir',None,'--checkpoint',str(ckpt/'v1.0.0'),'--ccd',str(ckpt/'ccd.pkl'),'--accelerator','gpu','--devices','1','--num_workers','2','--precision','bf16-mixed','--recycling_steps','5','--no_kernels','--require_affinity','--seed','20260916']
    for t in targets.itertuples():
        done={r[0] for r in db.execute("SELECT drug_id FROM predictions WHERE model='nesso' AND target_id=? AND status IN ('completed','failed')",(t.target_id,))}
        pending=[d for d in drugs.itertuples() if d.drug_id not in done]
        if not pending:continue
        if shutil.disk_usage(ROOT).free<8*1024**3:
            status(state='paused_disk_space',target=t.gene,completed=count(),required_free_gib=8)
            while shutil.disk_usage(ROOT).free<8*1024**3:time.sleep(30)
        batch=OUT/'targets'/t.target_id;inputs=batch/'inputs';inputs.mkdir(parents=True,exist_ok=True)
        for d in pending:
            value={'sequences':[{'protein':{'id':'A','sequence':t.sequence}},{'ligand':{'id':'B','smiles':d.smiles}}],'properties':[{'affinity':{'binder':'B'}}]}
            (inputs/f'{d.drug_id}__{t.target_id}.yaml').write_text(yaml.safe_dump(value,sort_keys=False))
        # Identical sequence embeddings are reusable; all per-pair preprocessing stays official.
        embeddings=batch/'processed/esm_embeddings';embeddings.mkdir(parents=True,exist_ok=True)
        md5=hashlib.md5(t.sequence.encode()).hexdigest();previous=OLD/'processed/esm_embeddings'/f'{md5}.safetensors'
        if previous.exists() and not (embeddings/previous.name).exists():(embeddings/previous.name).symlink_to(previous)
        command=command_base.copy();command[2]=str(inputs);command[4]=str(batch)
        (batch/'COMMAND.json').write_text(json.dumps(command,indent=2));batch_start=time.time()
        with (batch/'RUN.log').open('a') as log:
            child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                harvest(batch/'predictions');status(state='inference',target=t.gene,target_id=t.target_id,child_pid=child.pid,completed=count(),elapsed_seconds=time.time()-started,batch_elapsed_seconds=time.time()-batch_start)
                # Avoid a hung entire target stalling all other targets indefinitely.
                if time.time()-batch_start>max(3600,len(pending)*120):
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    break
                time.sleep(15)
        harvest(batch/'predictions')
        completed={r[0] for r in db.execute("SELECT drug_id FROM predictions WHERE model='nesso' AND target_id=? AND status='completed'",(t.target_id,))}
        save(db,[dict(model='nesso',drug_id=d.drug_id,target_id=t.target_id,status='failed',reason=f'inference_missing_after_target_stage_exit_{child.returncode}') for d in pending if d.drug_id not in completed])
        # Retain raw affinity JSON/YAML/command/log, reclaim only reproducible intermediates.
        processed=batch/'processed'
        if processed.is_dir() and processed.parent.parent==OUT/'targets':shutil.rmtree(processed)
        print(t.gene,'completed',len(completed),'of',len(drugs),'elapsed',time.time()-batch_start,flush=True)
    status(state='completed' if count()==276480 else 'completed_with_missing',completed=count(),elapsed_seconds=time.time()-started)
    db.close()
if __name__=='__main__':main()

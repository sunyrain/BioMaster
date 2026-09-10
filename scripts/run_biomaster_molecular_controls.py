#!/usr/bin/env python3
"""Complete the fixed representation comparison with two fingerprint controls."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json,SOURCE

OUT=ROOT/'outputs/biomaster_best_model_20260906'
CONFIG=ROOT/'configs/biomaster_best_model_molecular_controls_20260906.json'


def worker(representation,cutoff,seed,smoke_steps=0):
    import train_biomaster_best_model as original
    from biomaster.molecular_controls import MolecularControlConfig,MolecularControlInteraction,MolecularControlBank
    p=json.loads(CONFIG.read_text())
    if representation not in p['representations'] or cutoff not in p['cutoffs'] or seed not in p['seeds']:
        raise ValueError('undeclared molecular control')
    base_identity=original.identity
    def identity(c,parent=None):
        return dict(training=base_identity(c,parent),representation=representation,protocol=file_identity(CONFIG),
            driver=file_identity(Path(__file__)),model=file_identity(ROOT/'biomaster/molecular_controls.py'),
            features=file_identity(OUT/'CHEMICAL_FEATURE_IDENTITY.json'))
    original.identity=identity;original.OUT=OUT/'representations'/representation
    original.UnifiedConfig=lambda **kw:MolecularControlConfig(**kw,drug_representation=representation)
    original.UnifiedInteraction=MolecularControlInteraction
    original.SupplementedFeatureBank=lambda base,supplement,local:MolecularControlBank(base,supplement,SOURCE,representation,local=local)
    original.train(cutoff,'global_ema',seed,smoke_steps=smoke_steps)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--wait-for-representations',action='store_true')
    parser.add_argument('--representation',choices=['morgan','drugclip_morgan'],default='morgan')
    parser.add_argument('--cutoff',type=int,choices=[2018,2020],default=2018)
    parser.add_argument('--seed',type=int,default=20260921);parser.add_argument('--smoke-steps',type=int,default=0)
    a=parser.parse_args()
    if a.worker:return worker(a.representation,a.cutoff,a.seed,a.smoke_steps)
    p=json.loads(CONFIG.read_text())
    if a.wait_for_representations:
        while True:
            state=json.loads((OUT/'REPRESENTATION_STATUS.json').read_text())
            if state['status']=='COMPLETE':break
            if state['status']=='FAILED':raise RuntimeError('representation run failed')
            write_json(OUT/'MOLECULAR_CONTROL_STATUS.json',dict(status='WAITING_FOR_REPRESENTATIONS',
                utc=datetime.now(timezone.utc).isoformat(),representation_job=state.get('job')))
            time.sleep(10)
    jobs=[(s,c,r) for s in p['seeds'] for c in p['cutoffs'] for r in p['representations']]
    for number,(seed,cutoff,representation) in enumerate(jobs,1):
        state=dict(status='RUNNING',job=number,jobs=len(jobs),seed=seed,cutoff=cutoff,representation=representation,
                   utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'MOLECULAR_CONTROL_STATUS.json',state)
        name=f'molecular_control_{cutoff}_{representation}_{seed}.log'
        with (OUT/'logs'/name).open('a') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--representation',representation,
                '--cutoff',str(cutoff),'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(OUT/'MOLECULAR_CONTROL_STATUS.json',{**state,'status':'FAILED','returncode':result.returncode,'log':name})
            raise SystemExit(result.returncode)
    write_json(OUT/'MOLECULAR_CONTROL_STATUS.json',dict(status='COMPLETE',jobs=len(jobs),driver=file_identity(Path(__file__))))


if __name__=='__main__':main()

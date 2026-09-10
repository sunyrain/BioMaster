#!/usr/bin/env python3
"""Compare frozen chemical representations with the same global EMA training."""
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
CONFIG=ROOT/'configs/biomaster_best_model_representation_20260906.json'


def prepare_identity():
    path=OUT/'CHEMICAL_FEATURE_IDENTITY.json'
    if path.exists():return
    write_json(path,dict(status='COMPLETE',files=[file_identity(SOURCE/n) for n in
        ['BERMOL.npy','BERMOL_DONE.npy','MORGAN.npy','MOLECULES.csv.gz']],
        parent=file_identity(SOURCE/'DATA_MANIFEST.json'),labels_used=False))


def worker(representation,cutoff,seed,smoke_steps=0):
    import train_biomaster_best_model as original
    from biomaster.chemical_representation import ChemicalConfig,ChemicalInteraction,ChemicalFeatureBank
    config=json.loads(CONFIG.read_text())
    if representation not in config['representations'] or cutoff not in config['cutoffs'] or seed not in config['seeds']:
        raise ValueError('undeclared chemical comparison')
    prepare_identity();base_identity=original.identity
    def identity(c,parent=None):
        return dict(training=base_identity(c,parent),representation=representation,
            config=file_identity(CONFIG),driver=file_identity(Path(__file__)),
            chemical_model=file_identity(ROOT/'biomaster/chemical_representation.py'),
            features=file_identity(OUT/'CHEMICAL_FEATURE_IDENTITY.json'))
    original.identity=identity
    # Only the representation and output namespace change. In particular DATA,
    # training protocol, both validation windows and selection remain identical.
    original.OUT=OUT/'representations'/representation
    original.UnifiedConfig=lambda **kw:ChemicalConfig(**kw,drug_representation=representation)
    original.UnifiedInteraction=ChemicalInteraction
    original.SupplementedFeatureBank=lambda base,supplement,local:ChemicalFeatureBank(base,supplement,SOURCE,representation,local=local)
    return original.train(cutoff,'global_ema',seed,smoke_steps=smoke_steps)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--wait-for-development',action='store_true')
    parser.add_argument('--representation',choices=['bermol','bermol_morgan'],default='bermol')
    parser.add_argument('--cutoff',type=int,choices=[2018,2020],default=2018)
    parser.add_argument('--seed',type=int,default=20260921);parser.add_argument('--smoke-steps',type=int,default=0)
    a=parser.parse_args()
    if a.worker:return worker(a.representation,a.cutoff,a.seed,a.smoke_steps)
    prepare_identity()
    if a.wait_for_development:
        while True:
            state=json.loads((OUT/'DEVELOPMENT_STATUS.json').read_text())
            if state['status']=='COMPLETE':break
            if state['status']=='FAILED':raise RuntimeError('resolve development failure before chemical runs')
            write_json(OUT/'REPRESENTATION_STATUS.json',dict(status='WAITING_FOR_DEVELOPMENT',
                utc=datetime.now(timezone.utc).isoformat(),development_job=state['job']))
            time.sleep(10)
    config=json.loads(CONFIG.read_text());jobs=[]
    for seed in config['seeds']:
        for cutoff in config['cutoffs']:
            for representation in config['representations']:jobs.append((seed,cutoff,representation))
    for number,(seed,cutoff,representation) in enumerate(jobs,1):
        status=dict(status='RUNNING',job=number,jobs=len(jobs),seed=seed,cutoff=cutoff,representation=representation,
                    utc=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'REPRESENTATION_STATUS.json',status)
        name=f'representation_{cutoff}_{representation}_{seed}.log'
        with (OUT/'logs'/name).open('a') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--worker','--representation',representation,
                '--cutoff',str(cutoff),'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(OUT/'REPRESENTATION_STATUS.json',{**status,'status':'FAILED','returncode':result.returncode,'log':name})
            raise SystemExit(result.returncode)
    write_json(OUT/'REPRESENTATION_STATUS.json',dict(status='COMPLETE',jobs=len(jobs),driver=file_identity(Path(__file__))))


if __name__=='__main__':main()

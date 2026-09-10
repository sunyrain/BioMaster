#!/usr/bin/env python3
"""Run the additional early baselines and predeclared optimization ablations."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json,SOURCE,OUTPUT

OUT=ROOT/'outputs/biomaster_best_model_20260906'


def baseline(seed,variant,smoke_steps=0):
    import train_biomaster_unified_interaction as original
    from biomaster.best_model_training import SupplementedFeatureBank,RollingStage
    base_identity=original.identity
    def current_identity():
        return dict(original=base_identity(),cutoff=2018,driver=file_identity(Path(__file__)),
            helper=file_identity(ROOT/'biomaster/best_model_training.py'),
            data=[file_identity(OUT/'data'/n) for n in ['ROLL_MANIFEST.json',
                  'supplemental_features/ATOM_MANIFEST.json','supplemental_features/SUPPLEMENT_PROVENANCE.json']])
    original.identity=current_identity
    original.OUT=OUT/'rolling_original'/'roll_2018';original.OUT.mkdir(parents=True,exist_ok=True)
    original.TemporalStage=lambda name: RollingStage(OUT/'data',SOURCE,2018)
    original.UnifiedFeatureBank=lambda path,local: SupplementedFeatureBank(OUTPUT,OUT/'data/supplemental_features',local=local)
    original.train('development',variant,seed,smoke_steps)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--variant',choices=['global','capacity'],default='global')
    parser.add_argument('--smoke-steps',type=int,default=0)
    args=parser.parse_args()
    if args.worker:return baseline(args.seed,args.variant,args.smoke_steps)
    config=json.loads((ROOT/'configs/biomaster_best_model_optimization_20260906.json').read_text())
    jobs=[]
    for seed in config['seeds']:
        for variant in ['global','capacity']:
            jobs.append((dict(phase='early_baseline',cutoff=2018,variant=variant,seed=seed),
                [sys.executable,'-u',str(Path(__file__)),'--worker','--variant',variant,'--seed',str(seed)]))
    for seed in config['seeds']:
        for cutoff in config['cutoffs']:
            for variant in config['variants']:
                jobs.append((dict(phase='optimization',cutoff=cutoff,variant=variant,seed=seed),
                    [sys.executable,'-u',str(ROOT/'scripts/train_biomaster_best_model.py'),
                     '--cutoff',str(cutoff),'--variant',variant,'--seed',str(seed)]))
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    for index,(event,command) in enumerate(jobs):
        status=dict(status='RUNNING',job=index+1,jobs=len(jobs),utc=datetime.now(timezone.utc).isoformat(),**event)
        write_json(OUT/'DEVELOPMENT_STATUS.json',status)
        name=f"{event['phase']}_{event['cutoff']}_{event['variant']}_{event['seed']}.log"
        with (OUT/'logs'/name).open('a') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(OUT/'DEVELOPMENT_STATUS.json',{**status,'status':'FAILED','returncode':result.returncode,'log':name})
            raise SystemExit(result.returncode)
    write_json(OUT/'DEVELOPMENT_STATUS.json',dict(status='COMPLETE',jobs=len(jobs),
        driver=file_identity(Path(__file__)),configuration=file_identity(ROOT/'configs/biomaster_best_model_optimization_20260906.json')))


if __name__=='__main__':main()

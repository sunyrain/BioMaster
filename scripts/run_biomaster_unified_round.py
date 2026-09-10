#!/usr/bin/env python3
"""Sequential, fail-fast first-round ablations with a persistent status file."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_biomaster_unified_interaction import OUTPUT,write_json


def main():
    out=OUTPUT.parent;p=json.loads((ROOT/'configs/biomaster_unified_interaction_20260906.json').read_text())
    def run(arguments,name):
        write_json(out/'ROUND_STATUS.json',dict(status='RUNNING',step=name,arguments=arguments))
        with (out/'logs'/(name+'.log')).open('a',buffering=1) as handle:
            subprocess.run([sys.executable,'-u',*arguments],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    try:
        if not (OUTPUT/'ATOM_MANIFEST.json').exists():
            write_json(out/'ROUND_STATUS.json',dict(status='WAITING_FOR_CONFORMERS'))
            while True:
                status=json.loads((OUTPUT/'CONFORMER_STATUS.json').read_text())
                if status['completed']==status['required']:break
                time.sleep(5)
            run(['scripts/prepare_biomaster_unified_interaction.py','atoms'],'atoms')
        run(['scripts/train_biomaster_unified_interaction.py','--variant','geometry','--smoke-steps','4'],'smoke')
        run(['scripts/audit_biomaster_unified_interaction.py'],'audit')
        for seed in p['seeds']:
            for variant in p['variants']:
                run(['scripts/train_biomaster_unified_interaction.py','--stage','development','--variant',variant,'--seed',str(seed)],f'development_{variant}_{seed}')
        # Use early validation ONLY to choose the local candidate for refitting.
        results=[]
        for variant in p['variants']:
            members=[json.loads((out/'development'/variant/f'seed_{s}'/'RESULT.json').read_text()) for s in p['seeds']]
            results.append(dict(variant=variant,selection=sum(r['validation']['selection'] for r in members)/len(members),members=members))
        local=max((r for r in results if r['variant'] in ['sequence','site','geometry']),key=lambda r:r['selection'])['variant']
        selected=['global','capacity',local]
        write_json(out/'SELECTION.json',dict(status='FROZEN_BEFORE_FINAL_REFIT_AND_REGRESSION',
                   final_variants=selected,development_results=results,test_labels_used_for_selection=False))
        for seed in p['seeds']:
            for variant in selected:
                run(['scripts/train_biomaster_unified_interaction.py','--stage','final','--variant',variant,'--seed',str(seed)],f'final_{variant}_{seed}')
        run(['scripts/evaluate_biomaster_unified_interaction.py'],'regression')
        run(['scripts/summarize_biomaster_unified_interaction.py'],'report')
        write_json(out/'ROUND_STATUS.json',dict(status='COMPLETE',selected_local=local,promoted=False))
    except Exception as error:
        write_json(out/'ROUND_FAILURE.json',dict(status='FAILED',error=repr(error)))
        raise


if __name__=='__main__':main()

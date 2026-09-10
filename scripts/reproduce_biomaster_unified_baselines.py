#!/usr/bin/env python3
"""Run the predeclared extra seeds without changing first-round source/artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_biomaster_unified_interaction import write_json
from biomaster.odti_pockets_v3 import file_identity

OUT = ROOT / 'outputs/biomaster_best_model_20260906'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--variant', choices=['global', 'capacity'], default='global')
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--stage', choices=['development', 'final'], default='development')
    args = parser.parse_args()
    if args.worker:
        import train_biomaster_unified_interaction as original
        # Only the output root changes. Features, configuration, source identity,
        # optimizer, data, exposure seeds and stopping rules are unchanged.
        original.OUT = OUT / 'reproduction'
        original.OUT.mkdir(parents=True, exist_ok=True)
        original.train(args.stage, args.variant, args.seed)
        return
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    for stage in ['development', 'final']:
        for seed in [20260922, 20260923]:
            for variant in ['global', 'capacity']:
                event = dict(status='RUNNING', phase='baseline_reproduction', stage=stage,
                             variant=variant, seed=seed, utc=datetime.now(timezone.utc).isoformat())
                write_json(OUT / 'REPRODUCTION_STATUS.json', event)
                name = f'reproduce_{stage}_{variant}_{seed}.log'
                with (OUT / 'logs' / name).open('a') as handle:
                    subprocess.run([sys.executable, '-u', str(Path(__file__)), '--worker',
                                    '--stage', stage, '--variant', variant, '--seed', str(seed)],
                                   cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    write_json(OUT / 'REPRODUCTION_STATUS.json', dict(status='COMPLETE', runs=8,
               driver=file_identity(Path(__file__)), original_training_code_unchanged=True))


if __name__ == '__main__':
    main()

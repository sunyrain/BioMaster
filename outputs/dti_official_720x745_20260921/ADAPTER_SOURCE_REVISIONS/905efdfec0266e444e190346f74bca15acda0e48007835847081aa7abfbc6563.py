#!/usr/bin/env python3
"""Resumable native Nesso CLI; preserve raw evidence, reclaim intermediates."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import numpy as np
import pandas as pd
import yaml
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, dump, directory, read_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, default=0)
    parser.add_argument('--wait-for-file', default='')
    args = parser.parse_args()
    name = 'Nesso-1'
    out = directory(name)
    drugs, targets = inputs()
    start = time.time()
    while args.wait_for_file and not (ROOT / args.wait_for_file).exists():
        status(name, 'QUEUED_BEHIND_GPU_FRONT_QUEUE', waiting_for_file=args.wait_for_file,
               scored_pairs=len(read_scores(name)))
        time.sleep(30)
    while args.wait_pid and (ROOT / f'/proc/{args.wait_pid}/stat').exists():
        # PID reuse guard: only wait on this study's official MAMMAL job.
        cmdline = (ROOT / f'/proc/{args.wait_pid}/cmdline').read_bytes()
        if b'run_dti_official_mammal_20260921.py' not in cmdline:
            break
        status(name, 'QUEUED_BEHIND_MAMMAL', waiting_for_pid=args.wait_pid, scored_pairs=len(read_scores(name)))
        time.sleep(30)
    snapshot = ROOT / '.cache/nesso/huggingface/models--recursionpharma--nesso/snapshots/1896c84c7186c506c7efd79051480809d51098bf'
    ckpt = snapshot / 'v1.0.0'
    env = dict(os.environ, NESSO_CACHE=str(ROOT / '.cache/nesso'), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    contract(name, checkpoint_sha256=sha(ckpt/'model.safetensors'), adapter_sha256=sha(__file__),
             native_source_revision='1896c84c7186c506c7efd79051480809d51098bf (HF checkpoint)',
             head='affinity_probability_binary; native regression also retained in raw JSON',
             precision='bf16-mixed', seed=20260916, recycling_steps=5,
             historical_reuse='exact old catalogue scores, same native CLI settings')
    frame = read_scores(name)
    count_initial = len(frame)
    failures_file = out/'INPUT_FAILURES.json'
    failures = json.loads(failures_file.read_text()) if failures_file.exists() else []
    inference_start = time.time()
    for target in targets.itertuples():
        done = set(frame.loc[frame.target_id.eq(target.target_id), 'drug_id'])
        pending = [drug for drug in drugs.itertuples() if drug.drug_id not in done]
        if not pending:
            continue
        if (out / 'targets' / target.target_id / 'FINISHED.json').exists():
            continue
        while shutil.disk_usage(ROOT).free < 8 * 1024**3:
            status(name, 'WAITING_DISK_RESERVE', scored_pairs=len(frame), required_free_gib=8)
            time.sleep(30)
        batch = out / 'targets' / target.target_id
        inp = batch / 'inputs'
        inp.mkdir(parents=True, exist_ok=True)
        for drug in pending:
            value = dict(sequences=[dict(protein=dict(id='A', sequence=target.sequence)), dict(ligand=dict(id='B', smiles=drug.smiles))],
                         properties=[dict(affinity=dict(binder='B'))])
            (inp / f'{drug.drug_id}__{target.target_id}.yaml').write_text(yaml.safe_dump(value, sort_keys=False))
        command = [str(ROOT/'.venvs/nesso/bin/nesso'), 'predict', str(inp), '--out_dir', str(batch),
                   '--checkpoint', str(ckpt), '--ccd', str(snapshot/'ccd.pkl'), '--accelerator', 'gpu', '--devices', '1',
                   '--num_workers', '2', '--precision', 'bf16-mixed', '--recycling_steps', '5', '--no_kernels',
                   '--require_affinity', '--seed', '20260916']
        dump(batch/'COMMAND.json', command)
        began = time.time()
        imported = set(done)

        def harvest():
            nonlocal frame
            rows = []
            for path in (batch/'predictions').glob('*/affinity.json'):
                drug_id, target_id = path.parent.name.split('__')
                if drug_id in imported:
                    continue
                try:
                    value = json.loads(path.read_text())
                    score = float(value['affinity_probability_binary'])
                    assert np.isfinite(score) and target_id == target.target_id and drug_id in set(drugs.drug_id)
                    rows.append(dict(drug_id=drug_id, target_id=target_id, score=score))
                    imported.add(drug_id)
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
            if rows:
                frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
                save_scores(name, frame)

        with (batch/'RUN.log').open('a') as log:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                harvest()
                generated = len(frame)-count_initial
                elapsed = time.time()-inference_start
                status(name, 'INFERENCE', child_pid=child.pid, target=target.gene, target_id=target.target_id,
                       scored_pairs=len(frame), elapsed_seconds=time.time()-start,
                       eta_seconds=(536400-len(frame))*elapsed/generated if generated else None)
                if time.time()-began > max(3600, len(pending)*120):
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                time.sleep(15)
        harvest()
        missing = [drug.drug_id for drug in pending if drug.drug_id not in imported]
        failures.extend(dict(target_id=target.target_id, drug_id=d, reason=f'MISSING_NATIVE_OUTPUT_EXIT_{child.returncode}') for d in missing)
        dump(out/'INPUT_FAILURES.json', failures)
        dump(batch/'FINISHED.json', dict(exit_code=child.returncode, missing_drugs=missing, elapsed_seconds=time.time()-began))
        # Exact raw predictions and YAML inputs are compressed before removal.
        archive = batch/'RAW_NATIVE_OUTPUTS.tar.gz'
        with tarfile.open(archive.with_suffix('.tmp'), 'w:gz') as tar:
            for part in ['inputs', 'predictions']:
                if (batch/part).exists():
                    tar.add(batch/part, arcname=part)
        archive.with_suffix('.tmp').replace(archive)
        verified = 0
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if member.isfile():
                    retained = tar.extractfile(member)
                    digest = hashlib.sha256()
                    for block in iter(lambda: retained.read(1024*1024), b''):
                        digest.update(block)
                    assert digest.hexdigest() == sha(batch/member.name), member.name
                    verified += 1
        dump(batch/'RAW_ARCHIVE.json', dict(path=archive.name, sha256=sha(archive), verified_files=verified))
        for part in ['processed', 'inputs', 'predictions']:
            if (batch/part).is_dir():
                shutil.rmtree(batch/part)
        print(target.gene, len(frame), 'scored', len(missing), 'missing', flush=True)
    status(name, 'COMPLETE' if len(frame)==536400 else 'COMPLETE_WITH_MISSING_OUTPUTS', scored_pairs=len(frame), elapsed_seconds=time.time()-start)


if __name__ == '__main__':
    main()

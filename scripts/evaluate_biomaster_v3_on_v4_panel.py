#!/usr/bin/env python3
"""Evaluate frozen historical V3 B checkpoints on the identical R1 test panel.

These are historical development controls, with their original feature banks,
six-epoch training and observed-validation checkpoint selection retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.odti_support_data_v3 import SupportBatch
from biomaster.odti_v3_panel_metrics import positive_retrieval_metrics
from biomaster.selectivity_training_v3 import observed_metrics
from train_biomaster_selectivity_v3 import Runtime
from train_biomaster_odti_v4 import subgroup_metrics
from build_biomaster_odti_v4_features import sha256, write_json


def main():
    torch.set_num_threads(4)
    out = ROOT / 'outputs/biomaster_odti_v4_20260905'
    protocol = json.loads((out / 'protocol/PROTOCOL_V4.json').read_text())
    old = protocol['data']
    assert sha256(old['prepared_relations_path']) == old['prepared_relations_sha256']
    data = pd.read_csv(old['prepared_relations_path'])
    support = SupportBatch(**{n:np.load(old[f'support_{n}_path'],mmap_mode='r') for n in ['indices','similarities','mask']})
    panelpath = out / 'protocol/test_panel.npz'
    assert sha256(panelpath) == protocol['panels']['test']['panel_sha256']
    with np.load(panelpath) as f:
        panel = {k:f[k] for k in f.files}
    panel['support'] = {n:np.load(p,mmap_mode='r') for n,p in protocol['panels']['test']['support'].items()}
    evidence = SupportBatch(**panel['support'])
    ds,ts = np.repeat(panel['drug_indices'],len(panel['target_indices'])),np.tile(panel['target_indices'],len(panel['drug_indices']))
    families = data.drop_duplicates('target_feature_index').set_index('target_feature_index').family_index.to_dict()
    family_ids = np.array([families[int(t)] for t in ts])
    test = np.flatnonzero(data.split.eq('test'))
    result = {}
    for seed in [20260905,20260906]:
        path = ROOT / f'outputs/biomaster_v3_20260905/experiments/B_support_pair/seed_{seed}/BEST_MODEL_V3.pt'
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['data_audit']['assignment_sha256'] == old['assignment_sha256']
        args = argparse.Namespace(**checkpoint['arguments'])
        assert args.local is False and args.support != 'none'
        args.eval_batch_size = 1024
        runtime = Runtime(args,data,checkpoint['data_audit'],None,support)
        runtime.restore(checkpoint)
        runtime.mode(False)
        measured,_ = runtime.predict(test)
        chunks = []
        with torch.inference_mode():
            for start in range(0,len(ds),args.eval_batch_size):
                ix = np.arange(start,min(start+args.eval_batch_size,len(ds)))
                with runtime.autocast():
                    prediction = runtime.model(**runtime.inputs(ds[ix],ts[ix],family_ids[ix],evidence,ix))
                chunks.append(prediction['final_logit'].float().cpu().numpy())
        matrix = np.concatenate(chunks).reshape(panel['known_positive'].shape)
        name = f'v3_B_support_{seed}'
        result[name] = {'checkpoint':str(path.relative_to(ROOT)), 'checkpoint_sha256':sha256(path),
            'selected_epoch':checkpoint['epoch'], 'training_protocol':'historical six-epoch V3 B; observed validation AP selection',
            'observed':observed_metrics(runtime.labels[test],measured,runtime.drugs[test],runtime.targets[test]),
            'panel':positive_retrieval_metrics(panel['known_positive'],matrix,panel['target_indices']),
            'similarity_subgroups':subgroup_metrics(panel,matrix)}
        np.savez_compressed(out / 'evaluation' / f'{name}_PREDICTIONS_V4.npz', observed_rows=test, observed_scores=measured,
            panel_scores=matrix, known_positive=panel['known_positive'], drug_indices=panel['drug_indices'], target_indices=panel['target_indices'])
        write_json(out / 'evaluation/HISTORICAL_V3_CONTROLS_V4.json',result)
        print(json.dumps({'model':name,'panel_ap':result[name]['panel']['macro_positive_retrieval_ap']}),flush=True)
        del runtime,checkpoint
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

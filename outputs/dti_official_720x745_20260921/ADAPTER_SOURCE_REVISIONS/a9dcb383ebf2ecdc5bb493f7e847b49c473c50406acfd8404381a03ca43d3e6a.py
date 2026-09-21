#!/usr/bin/env python3
"""Author CheMLT-F scaffold task checkpoint, native KIBA head (index12)."""
import ast
import json
import sys
import time
from typing import Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, RobertaPreTrainedModel, DebertaV2Model, DebertaTokenizerFast
from safetensors.torch import load_file
from rdkit import Chem
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, dump, directory


def main():
    start = time.time()
    name = 'CheMLT-F'
    status(name, 'LOADING_NATIVE_MODEL')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = ROOT / '.external/CheMLT-F/Training/Scaffold_CheMLT-F.ipynb'
    nb = json.loads(source.read_text())
    text = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree = ast.parse(text)
    names = {'MLTClassificationHead', 'RobertaMultiTaskModel', 'tokenization3', 'canonicalize_smiles'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional, Union=Union, Tuple=Tuple, List=List, AutoConfig=AutoConfig,
                     RobertaPreTrainedModel=RobertaPreTrainedModel, DebertaV2Model=DebertaV2Model, Chem=Chem)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    root = ROOT / 'data/research/dti_official_weights_20260921/CheMLT-F'
    mol_path = root / 'Pre-Training/PubchemModelDeberta'
    prot_path = root / 'Pre-Training/ProteinModelDeb'
    namespace['tokenizer'] = DebertaTokenizerFast.from_pretrained(mol_path, local_files_only=True)
    namespace['tokenizer2'] = DebertaTokenizerFast.from_pretrained(prot_path, local_files_only=True)
    counts = [1, 1, 1, 2, 12, 17, 27, 617, 1, 1, 1, 1, 1]
    model = namespace['RobertaMultiTaskModel'](str(mol_path), str(prot_path), counts,
               ['multi_label_classification']*8+['regression']*5)
    checkpoint = root / 'Models/Scaffold_CheMLT-F/model.safetensors'
    model.load_state_dict(load_file(checkpoint), strict=True)
    model.cuda().eval()
    drugs, targets = inputs()
    canonical = [namespace['canonicalize_smiles']({'compound_iso_smiles':s})['compound_iso_smiles'] for s in drugs.smiles]
    tokenize = namespace['tokenization3']
    # Cache native independent entity encoders, with the author's two512-token protein chunks.
    dvec, tvec = [], []
    for i, smi in enumerate(canonical):
        batch = tokenize(dict(compound_iso_smiles=[smi], target_sequence=[targets.iloc[0].sequence]))
        with torch.inference_mode():
            value = model.encoder1(input_ids=torch.tensor(batch['input_ids'], device='cuda'),
                     attention_mask=torch.tensor(batch['attention_mask'], device='cuda')).last_hidden_state[:, 0, :]
        dvec.append(value.cpu())
        if i%100==0:
            status(name, 'ENCODING_DRUGS', encoded=i, total=720, elapsed_seconds=time.time()-start)
    for i, target in enumerate(targets.itertuples()):
        batch = tokenize(dict(compound_iso_smiles=[canonical[0]], target_sequence=[target.sequence]))
        vectors = []
        with torch.inference_mode():
            for suffix in ['2', '3']:
                value = model.encoder2(input_ids=torch.tensor(batch['input_ids'+suffix], device='cuda'),
                         attention_mask=torch.tensor(batch['attention_mask'+suffix], device='cuda')).last_hidden_state[:, 0, :]
                vectors.append(value.cpu())
        tvec.append(torch.cat(vectors, dim=1))
        if i%100==0:
            status(name, 'ENCODING_TARGETS', encoded=i, total=745, elapsed_seconds=time.time()-start)
    dp, tp = torch.cat(dvec).cuda(), torch.cat(tvec).cuda()
    values = []
    with torch.inference_mode():
        for protein in tp:
            fused = torch.cat([dp, protein[None].expand(720, -1)], dim=1)
            values.append(model.classification_heads[12](F.gelu(model.dense(fused))).squeeze(1).cpu().numpy())
    scores = np.stack(values, axis=1)
    assert np.isfinite(scores).all()
    checks = []
    for di, ti in [(0,0),(360,372),(719,744)]:
        batch = tokenize(dict(compound_iso_smiles=[canonical[di]], target_sequence=[targets.iloc[ti].sequence]))
        with torch.inference_mode():
            expected = model(**{k: torch.tensor(v, device='cuda') for k,v in batch.items()}, task_index=12)['logits'].item()
        delta = abs(expected-float(scores[di, ti]))
        assert delta < 1e-4, delta
        checks.append(delta)
    frame = pd.DataFrame(dict(drug_id=np.repeat(drugs.drug_id,745), target_id=np.tile(targets.target_id,720), score=scores.reshape(-1)))
    contract(name, checkpoint_sha256=sha(checkpoint), adapter_sha256=sha(__file__), native_notebook_sha256=sha(source),
             head='KIBA regression head index12, scaffold multitask checkpoint; higher KIBA convention', precision='float32_cuda_tf32_off',
             preprocessing='Author canonical isomeric SMILES + tokenization3, 512 drug tokens / two512 protein token chunks',
             note='KIBA composite activity score, not calibrated Kd; no DAVIS evaluation or fitting')
    dump(directory(name)/'REPLAY_CHECK.json', dict(status='PASS', native_forward_max_abs_deltas=checks))
    save_scores(name, frame)
    status(name, 'COMPLETE', scored_pairs=len(frame), elapsed_seconds=time.time()-start)
    print(name, len(frame), flush=True)


if __name__ == '__main__':
    main()

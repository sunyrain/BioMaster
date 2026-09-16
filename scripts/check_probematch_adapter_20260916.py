#!/usr/bin/env python3
"""Compare the adapter with the released pack/predict contract using real cached inputs."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_probematch_frontier_20260916 import load_official
OUT=ROOT/'outputs/frontier_dti_20260916/probematch';REPO=ROOT/'.external/ProbeMatchDTI'
torch.set_num_threads(3)
Model,constants,atom_features=load_official()
tree=ast.parse((REPO/'main.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pack']
module={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REPO/'main.py'),'exec'),module)
model=Model(3,torch.device('cpu'),4).eval();model.load_state_dict(torch.load(REPO/'model/All_Model',map_location='cpu',weights_only=True),strict=True)
data=pd.read_csv(OUT.parent/'UNIQUE_PAIRS.csv');pred=pd.read_csv(OUT/'PREDICTIONS.csv').set_index('pair_id')
samples=pd.concat([data.head(3),data[data.protein_length.gt(1200)].head(2)]).drop_duplicates('pair_id')
checks=[]
for row in samples.itertuples():
    molecule=Chem.MolFromSmiles(row.smiles)
    atoms=torch.tensor(np.array([atom_features(a) for a in molecule.GetAtoms()]),dtype=torch.float32)
    adj=torch.tensor(Chem.rdmolops.GetAdjacencyMatrix(molecule),dtype=torch.float32)
    words=np.array([constants['CHAR_SMI_SET'][c] for c in row.smiles[:100]]+[0]*max(0,100-len(row.smiles)))
    protein=np.array([constants['protein_dict'][c] for c in row.sequence[:5000]])
    packed=module['pack']([torch.tensor(words)],[atoms],[adj],[torch.tensor(protein)],[row.sequence],[row.smiles],[0],{row.sequence:np.load(OUT/f'features/protein_{row.target_id}.npy')},{row.smiles:np.load(OUT/f'features/drug_{row.drug_id}.npy')},torch.device('cpu'))
    seed=int(hashlib.sha256(row.pair_id.encode()).hexdigest()[:8],16);torch.manual_seed(seed)
    with torch.inference_mode():value=float(model.predict(packed)[1][0,1])
    expected=float(pred.loc[row.pair_id,'score']);diff=abs(value-expected)
    assert diff<1e-6,(row.pair_id,value,expected)
    checks.append(dict(pair_id=row.pair_id,official_pack_predict=value,adapter_score=expected,absolute_difference=diff,protein_length=row.protein_length))
report=dict(status='PASS',checks=checks,contract='Official pack + predict against production adapter, including long-sequence packing',strict_checkpoint=True)
(OUT/'ADAPTER_CHECK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

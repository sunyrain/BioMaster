#!/usr/bin/env python3
"""Official ProbeMatchDTI All_Model inference, no fitting or weight changes."""
import argparse
import ast
import hashlib
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/frontier_dti_20260916/probematch'
REPO=ROOT/'.external/ProbeMatchDTI'

def atomic_json(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def load_official():
    sys.path.insert(0,str(REPO))
    # Upstream omitted networks/utils/masking.py. Released model instantiates
    # ProbAttention(mask_flag=None), so neither mask class is reached. Fail loudly
    # if a future model does need one; do not alter any active attention path.
    masking=types.ModuleType('networks.utils.masking')
    class UnusedMask:
        def __init__(self,*args,**kwargs):
            raise RuntimeError('Upstream omitted masking module; masked attention unsupported')
    masking.TriangularCausalMask=masking.ProbMask=UnusedMask
    sys.modules['networks.utils']=types.ModuleType('networks.utils')
    sys.modules['networks.utils.masking']=masking
    from networks.model import ProbeMatchDTI
    # Import pure preprocessing definitions without its top-level encoder loads.
    tree=ast.parse((REPO/'predata.py').read_text())
    names={'protein_dict','CHAR_SMI_SET'}
    selected=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)]
    constants={};exec(compile(ast.Module(body=selected,type_ignores=[]),str(REPO/'predata.py'),'exec'),constants)
    from graph_features import atom_features
    return ProbeMatchDTI,constants,atom_features

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=8);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'features';cache.mkdir(exist_ok=True)
    torch.set_num_threads(args.threads);RDLogger.DisableLog('rdApp.warning')
    torch.manual_seed(20260916);np.random.seed(20260916)
    data=pd.read_csv(OUT.parent/'UNIQUE_PAIRS.csv')
    started=time.time();atomic_json(OUT/'STATUS.json',dict(state='waiting_encoders',pid=__import__('os').getpid()))
    deadline=time.time()+3600
    while not (OUT.parent/'ENCODERS.json').exists():
        if time.time()>deadline:raise TimeoutError('Pinned encoder download did not complete')
        time.sleep(15)
    from transformers import AutoModel,AutoTokenizer
    errors={}
    for kind,column,key,resource,limit in [('protein','sequence','target_id','prot_bert_bfd',1200),('drug','smiles','drug_id','PubChem10M_SMILES_BPE_450k',100)]:
        directory=ROOT/'.cache/frontier_dti'/resource
        tokenizer=AutoTokenizer.from_pretrained(directory,do_lower_case=False,local_files_only=True)
        model=AutoModel.from_pretrained(directory,local_files_only=True).eval().to(args.device)
        unique=data.drop_duplicates(key)
        for i,row in enumerate(unique.itertuples(index=False),1):
            identity=getattr(row,key);path=cache/f'{kind}_{identity}.npy'
            if not path.exists():
                try:
                    value=getattr(row,column)
                    text=' '.join(value[:5000]) if kind=='protein' else value
                    tokenized=tokenizer.batch_encode_plus([text],add_special_tokens=True,padding=True,return_tensors='pt')
                    with torch.inference_mode():
                        result=model(input_ids=tokenized['input_ids'].to(args.device),attention_mask=tokenized['attention_mask'].to(args.device)).last_hidden_state[0,:limit].float().cpu().numpy()
                    np.save(path,result)
                except Exception as exc:
                    errors[identity]=f'{type(exc).__name__}: {exc}';print(identity,errors[identity],flush=True)
            atomic_json(OUT/'STATUS.json',dict(state=f'encoding_{kind}',completed=i,total=len(unique),elapsed_seconds=time.time()-started,pid=__import__('os').getpid()))
            if i%10==0:print(kind,i,len(unique),'elapsed',round(time.time()-started),flush=True)
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
    Model,constants,atom_features=load_official()
    model=Model(layer_gnn=3,device=torch.device(args.device),source_number=4).to(args.device).eval()
    checkpoint=REPO/'model/All_Model'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='bd7de877c1198f62b1b2038c25b2b96c588c9b49f565e713b0d14c3f4e6b84eb'
    model.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True),strict=True)
    predictions=[]
    for i,row in enumerate(data.itertuples(index=False),1):
        result=dict(pair_id=row.pair_id,status='failed',score=None)
        try:
            prot=np.load(cache/f'protein_{row.target_id}.npy');drug=np.load(cache/f'drug_{row.drug_id}.npy')
            # Faithfully retain upstream fixed 100/1200 input windows and special tokens.
            molecule=Chem.MolFromSmiles(row.smiles);atoms=np.array([atom_features(a) for a in molecule.GetAtoms()])
            adjacency=Chem.rdmolops.GetAdjacencyMatrix(molecule)+np.eye(len(atoms))
            mw=torch.zeros((1,100),device=args.device);pw=torch.zeros((1,1200),device=args.device)
            chars=[constants['CHAR_SMI_SET'][c] for c in row.smiles[:100]];mw[0,:len(chars)]=torch.tensor(chars,device=args.device)
            seq=[constants['protein_dict'][c] for c in row.sequence[:1200]];pw[0,:len(seq)]=torch.tensor(seq,device=args.device)
            plm=torch.zeros((1,1200,1024),device=args.device);dlm=torch.zeros((1,100,768),device=args.device)
            plm[0,:len(prot)]=torch.tensor(prot,device=args.device);dlm[0,:len(drug)]=torch.tensor(drug,device=args.device)
            inputs=(mw,torch.tensor(atoms,dtype=torch.float32,device=args.device)[None],torch.tensor(adjacency,dtype=torch.float32,device=args.device)[None],pw,plm,dlm)
            seed=int(hashlib.sha256(row.pair_id.encode()).hexdigest()[:8],16);torch.manual_seed(seed)
            with torch.inference_mode():
                output=model.forward(inputs)
                score=float(output[6].softmax(-1)[0,1]);semantic=float(output[1].softmax(-1)[0,1]);structure=float(output[2].softmax(-1)[0,1])
            if not np.isfinite(score):raise ValueError('nonfinite score')
            result.update(status='completed',score=score,semantic_score=semantic,structure_score=structure,protein_window_truncated=len(row.sequence)>1200,smiles_window_truncated=len(row.smiles)>100)
        except Exception as exc:
            result['reason']=f'{type(exc).__name__}: {exc}'
        predictions.append(result)
        if i%10==0 or i==len(data):
            temp=OUT/'PREDICTIONS.tmp';pd.DataFrame(predictions).to_csv(temp,index=False);temp.replace(OUT/'PREDICTIONS.csv')
            atomic_json(OUT/'STATUS.json',dict(state='completed' if i==len(data) else 'inference',completed=sum(x['status']=='completed' for x in predictions),attempted=i,total=len(data),elapsed_seconds=time.time()-started,pid=__import__('os').getpid()))
            print('inference',i,len(data),flush=True)
    atomic_json(OUT/'ADAPTER.json',dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_revision='2b9c0dcb40bcc4e5482d6ea98aa1419538ecdd12',strict_weight_load=True,device=args.device,missing_dependency='Inactive mask classes raise if reached; active forward unchanged',ranking_score='softmax of official 0.40 fused + 0.27 semantic + 0.33 structure logits',seed='sha256(pair_id) first 32 bits for stochastic ProbAttention',protein_context='first 5000 residues; official LM retains CLS/SEP; packed first 1200 tokens',drug_context='full SMILES; packed first 100 tokens',training='none',feature_errors=errors))

if __name__=='__main__': main()

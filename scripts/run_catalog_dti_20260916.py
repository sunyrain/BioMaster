#!/usr/bin/env python3
"""Resumable official ProbeMatch/DTBind scoring across complete directory targets."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from rdkit import Chem,RDLogger
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.catalog_models import DIRECTORY,connect,save
OUT=ROOT/DIRECTORY
OLD=ROOT/'outputs/frontier_dti_20260916'

def status(model,**value):
    directory=OUT/model;directory.mkdir(parents=True,exist_ok=True)
    p=directory/'STATUS.tmp';p.write_text(json.dumps(dict(pid=os.getpid(),updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),**value),ensure_ascii=False,indent=2));p.replace(directory/'STATUS.json')

class Probe:
    def __init__(self,drugs,targets,threads):
        from run_probematch_frontier_20260916 import load_official
        from transformers import AutoModel,AutoTokenizer
        self.cache=OLD/'probematch/features';self.errors={}
        for kind,frame,key,column,resource,limit in [('protein',targets,'target_id','sequence','prot_bert_bfd',1200),('drug',drugs,'drug_id','smiles','PubChem10M_SMILES_BPE_450k',100)]:
            missing=[r for r in frame.to_dict('records') if not (self.cache/f'{kind}_{r[key]}.npy').exists()]
            if not missing:continue
            path=ROOT/'.cache/frontier_dti'/resource
            tokenizer=AutoTokenizer.from_pretrained(path,do_lower_case=False,local_files_only=True)
            model=AutoModel.from_pretrained(path,local_files_only=True).eval()
            for i,r in enumerate(missing):
                status('probematch',state='encoding_'+kind,encoded=i,total_to_encode=len(missing),catalog_pairs=276480)
                try:
                    text=' '.join(r[column][:5000]) if kind=='protein' else r[column]
                    tokens=tokenizer.batch_encode_plus([text],add_special_tokens=True,padding=True,return_tensors='pt')
                    with torch.inference_mode():result=model(**{k:v for k,v in tokens.items() if k in ('input_ids','attention_mask')}).last_hidden_state[0,:limit].float().numpy()
                    np.save(self.cache/f'{kind}_{r[key]}.npy',result)
                except Exception as exc:self.errors[r[key]]=f'{type(exc).__name__}: {exc}'
            del model
        Model,self.constants,self.atom_features=load_official()
        self.model=Model(layer_gnn=3,device=torch.device('cpu'),source_number=4).eval()
        checkpoint=ROOT/'.external/ProbeMatchDTI/model/All_Model'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='bd7de877c1198f62b1b2038c25b2b96c588c9b49f565e713b0d14c3f4e6b84eb'
        self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.drugs={}
        for r in drugs.itertuples():
            try:
                mol=Chem.MolFromSmiles(r.smiles)
                atoms=np.array([self.atom_features(a) for a in mol.GetAtoms()]);adj=Chem.rdmolops.GetAdjacencyMatrix(mol)+np.eye(len(atoms))
                mw=torch.zeros((1,100));chars=[self.constants['CHAR_SMI_SET'][c] for c in r.smiles[:100]];mw[0,:len(chars)]=torch.tensor(chars)
                lm=np.load(self.cache/f'drug_{r.drug_id}.npy');dlm=torch.zeros((1,100,768));dlm[0,:len(lm)]=torch.tensor(lm)
                self.drugs[r.drug_id]=(mw,torch.tensor(atoms,dtype=torch.float32)[None],torch.tensor(adj,dtype=torch.float32)[None],dlm)
            except Exception as exc:self.errors[r.drug_id]=f'{type(exc).__name__}: {exc}'
    def prepare_target(self,t):
        if t.target_id in self.errors:raise ValueError(self.errors[t.target_id])
        lm=np.load(self.cache/f'protein_{t.target_id}.npy');self.plm=torch.zeros((1,1200,1024));self.plm[0,:len(lm)]=torch.tensor(lm)
        self.pw=torch.zeros((1,1200));seq=[self.constants['protein_dict'][c] for c in t.sequence[:1200]];self.pw[0,:len(seq)]=torch.tensor(seq)
    def score(self,t,d):
        if d.drug_id in self.errors:raise ValueError(self.errors[d.drug_id])
        mw,atoms,adj,dlm=self.drugs[d.drug_id]
        torch.manual_seed(int(hashlib.sha256(f'{d.drug_id}__{t.target_id}'.encode()).hexdigest()[:8],16))
        with torch.inference_mode():return float(self.model((mw,atoms,adj,self.pw,self.plm,dlm))[6].softmax(-1)[0,1])

class DTBind:
    def __init__(self,drugs,targets,threads):
        from Bio import SeqIO
        from torch_geometric.data import Batch
        self.Batch=Batch;repo=ROOT/'.external/DTBind'
        sys.path.insert(0,str(repo/'script/occurrence'))
        from dti_model import DTISite,params
        torch.autograd.set_detect_anomaly(False)
        self.model=DTISite(params).eval();self.model.load_state_dict(torch.load(repo/'models/occurrence_model.pth',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
        self.seqs={r.id:str(r.seq) for r in SeqIO.parse(repo/'Data/dti/biosnap_protein_seq.fasta','fasta')}
        path=repo/'data_process/graph_construction/drug_gra.py';tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if not isinstance(n,ast.Expr)];module={};exec(compile(tree,str(path),'exec'),module)
        self.drugs={};self.errors={};dest=OLD/'dtbind/drug_graph'
        for d in drugs.itertuples():
            try:
                p=dest/f'{d.drug_id}.pt'
                if not p.exists():module['Mol2Graph'](d.smiles,d.drug_id,str(dest))
                self.drugs[d.drug_id]=Batch.from_data_list([torch.load(p,map_location='cpu',weights_only=False)])
            except Exception as exc:self.errors[d.drug_id]=str(exc)
    def prepare_target(self,t):
        if t.uniprot_id not in self.seqs:raise ValueError('publisher_protein_graph_not_available')
        if self.seqs[t.uniprot_id]!=t.sequence:raise ValueError('publisher_sequence_mismatch')
        path=OLD/f'dtbind/protein_graph/{t.uniprot_id}.pt'
        if not path.exists():
            try:finished=json.loads((OLD/'dtbind/FETCH_STATUS.json').read_text()).get('status')=='completed'
            except (OSError,ValueError):finished=False
            if finished:raise ValueError('publisher_protein_graph_not_available')
            raise FileNotFoundError('feature_download_pending')
        g=torch.load(path,map_location='cpu',weights_only=False)
        assert g.x.shape==(len(t.sequence),1024) and g.surface_x.shape==(len(t.sequence),10)
        self.pg=self.Batch.from_data_list([g])
    def score(self,t,d):
        if d.drug_id in self.errors:raise ValueError(self.errors[d.drug_id])
        with torch.inference_mode():return float(self.model(self.pg,self.drugs[d.drug_id]).reshape(-1)[0])

def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['probematch','dtbind']);p.add_argument('--threads',type=int,default=3);args=p.parse_args()
    torch.set_num_threads(args.threads);RDLogger.DisableLog('rdApp.warning');torch.manual_seed(20260916)
    drugs=pd.read_csv(OUT/'DRUGS.csv');targets=pd.read_csv(OUT/'TARGETS.csv');db=connect(ROOT,write=True);started=time.time();computed=0
    status(args.model,state='preparing',catalog_pairs=len(drugs)*len(targets))
    adapter=(Probe if args.model=='probematch' else DTBind)(drugs,targets,args.threads)
    # Replay known project inputs before enabling new full-catalog predictions.
    checks=[]
    for t in targets.itertuples():
        known=db.execute("SELECT * FROM predictions WHERE model=? AND target_id=? AND status='completed' LIMIT 1",(args.model,t.target_id)).fetchone()
        if known is None:continue
        adapter.prepare_target(t);d=next(x for x in drugs.itertuples() if x.drug_id==known['drug_id']);score=adapter.score(t,d)
        delta=abs(score-known['score']);assert delta<1e-6,(t.target_id,delta)
        checks.append(dict(target=t.target_id,drug=d.drug_id,max_abs_difference=delta))
        if len(checks)==5:break
    (OUT/args.model/'REPLAY_CHECK.json').write_text(json.dumps(checks,indent=2))
    while True:
        waiting=0
        for t in targets.itertuples():
            done={r['drug_id'] for r in db.execute("SELECT drug_id FROM predictions WHERE model=? AND target_id=? AND status IN ('completed','failed')",(args.model,t.target_id))}
            if len(done)==len(drugs):continue
            try:adapter.prepare_target(t)
            except Exception as exc:
                pending=isinstance(exc,FileNotFoundError)
                db.execute('INSERT OR REPLACE INTO target_status VALUES (?,?,?,?)',(args.model,t.target_id,'pending' if pending else 'unavailable',str(exc)));db.commit()
                waiting+=int(pending);continue
            db.execute('INSERT OR REPLACE INTO target_status VALUES (?,?,?,?)',(args.model,t.target_id,'running',''));db.commit()
            batch=[]
            for d in drugs.itertuples():
                if d.drug_id in done:continue
                row=dict(model=args.model,drug_id=d.drug_id,target_id=t.target_id,status='completed')
                try:
                    row['score']=adapter.score(t,d)
                    if not np.isfinite(row['score']):raise ValueError('nonfinite output')
                except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
                batch.append(row);computed+=1
                if len(batch)>=25:
                    save(db,batch);batch=[]
                    n=db.execute("SELECT COUNT(*) FROM predictions WHERE model=? AND status='completed'",(args.model,)).fetchone()[0]
                    status(args.model,state='inference',target=t.gene,completed=n,catalog_pairs=276480,computed_this_run=computed,elapsed_seconds=time.time()-started)
            save(db,batch);db.execute('INSERT OR REPLACE INTO target_status VALUES (?,?,?,?)',(args.model,t.target_id,'completed',''));db.commit()
            print(args.model,t.gene,'target completed',flush=True)
        count=db.execute("SELECT COUNT(*) FROM predictions WHERE model=? AND status='completed'",(args.model,)).fetchone()[0]
        status(args.model,state='waiting_graphs' if waiting else 'completed' if count==276480 else 'completed_with_missing',completed=count,catalog_pairs=276480,waiting_targets=waiting,computed_this_run=computed,elapsed_seconds=time.time()-started)
        if not waiting:break
        time.sleep(30)
    db.close()
if __name__=='__main__':main()

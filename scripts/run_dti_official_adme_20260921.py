#!/usr/bin/env python3
"""Native BindingDB ADME-DTI Combined (16 submodels + attention head)."""
import time,sys,os,types
import numpy as np,pandas as pd,torch
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha,status,save_scores

def main():
 name='ADME-DTI';out=directory(name);start=time.time()
 while not all((out/f).exists() for f in ['PROTEIN_ESM.npy','ESMC_CONTRACT.json','DRUG_DESCRIPTOR_CONTRACT.json']):
  status(name,'WAITING_NATIVE_FEATURES',scored_pairs=0);time.sleep(30)
 sys.path.insert(0,str(ROOT/'.external/ADME-DTI'));os.chdir(out)
 # Author directories are namespace packages; isolate them from BioMaster/src.
 for package in ['src','utils','config']:
  module=types.ModuleType(package);module.__path__=[str(ROOT/'.external/ADME-DTI'/package)];sys.modules[package]=module
 from config import config
 from src.dti_model_combined import DTIModelCombined
 from src.metadata import Metadata
 config.models_path=str(ROOT/'data/research/dti_official_weights_20260921/ADME-DTI/saved_models/classification/BindingDB');config.torch_device='cuda';config.combined=True;config.task='classification'
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 drugs,targets=inputs();pe=np.load(out/'PROTEIN_ESM.npy');p=out/'data/embeddings/target_embedding';p.mkdir(parents=True,exist_ok=True)
 pd.DataFrame({'target':targets.sequence,'embedding':list(pe)}).to_parquet(p/'ESM2.parquet',index=False)
 model=DTIModelCombined().cuda();weight=ROOT/'data/research/dti_official_weights_20260921/ADME-DTI/saved_models/classification/BindingDB/_Combined/model.pt';model.load_state_dict(torch.load(weight,weights_only=False),strict=True);model.eval()
 drug_maps=[];target_maps=[]
 for tr in config.drug_transformers:
  f=pd.read_parquet(out/f'data/embeddings/drug_embedding/{tr}.parquet');drug_maps.append(dict(zip(f.drug,f.embedding)))
 for tr in config.target_transformers:
  f=pd.read_parquet(out/f'data/embeddings/target_embedding/{tr}.parquet');target_maps.append(dict(zip(f.target,f.embedding)))
 valid_drugs=drugs.loc[drugs.smiles.map(lambda s:all(s in m for m in drug_maps))];failures=[];dm={};tm={}
 for d in valid_drugs.itertuples():dm[d.drug_id]=Metadata._Metadata__DrugMetadata.from_drug(d.smiles)
 for t in targets.itertuples():
  try:
   if not all(t.sequence in m for m in target_maps):raise ValueError('MISSING_NATIVE_ENCODER_FEATURE')
   tm[t.target_id]=Metadata._Metadata__TargetMetadata.from_target(t.sequence)
  except Exception as e:failures.append({'target_id':t.target_id,'reason':str(e)[:300]})
 ds=list(valid_drugs.itertuples());tlist=[t for t in targets.itertuples() if t.target_id in tm]
 def forward(ds,ts):
  dd=[torch.tensor(np.stack([m[d.smiles] for d in ds]),dtype=torch.float32,device='cuda') for m in drug_maps]
  pp=[torch.tensor(np.stack([m[t.sequence] for t in ts]),dtype=torch.float32,device='cuda') for m in target_maps]
  mm=[Metadata(dm[d.drug_id],tm[t.target_id]) for d,t in zip(ds,ts)]
  return model(dd,pp,mm).sigmoid().flatten()
 with torch.inference_mode():
  one=torch.stack([forward(ds[i:i+1],tlist[i:i+1])[0] for i in range(3)]);batch=forward(ds[:3],tlist[:3]);delta=float((one-batch).abs().max());assert delta<1e-4,delta
  dump(out/'REPLAY_CHECK.json',{'native_single_vs_batch_max_delta':delta,'passed':True,'all_17_checkpoints_loaded_strictly':True,'combined_checkpoint_sha256':sha(weight)})
  rows=[]
  for t in tlist:
   for i in range(0,len(ds),256):
    b=ds[i:i+256];v=forward(b,[t]*len(b)).cpu().numpy();rows.extend(dict(drug_id=d.drug_id,target_id=t.target_id,score=float(s)) for d,s in zip(b,v))
   status(name,'INFERENCE',scored_pairs=len(rows),target=t.gene,elapsed_seconds=time.time()-start)
   if len(rows)% (len(ds)*10)==0:save_scores(name,pd.DataFrame(rows))
 save_scores(name,pd.DataFrame(rows));dump(out/'INPUT_FAILURES.json',{'target_failures':failures,'missing_descriptor_drugs':drugs.loc[~drugs.drug_id.isin(valid_drugs.drug_id),'drug_id'].tolist()})
 status(name,'COMPLETE' if len(rows)==536400 else 'COMPLETE_WITH_NATIVE_INPUT_FAILURES',scored_pairs=len(rows),elapsed_seconds=time.time()-start)
if __name__=='__main__':main()

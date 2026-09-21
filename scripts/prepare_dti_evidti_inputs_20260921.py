#!/usr/bin/env python3
"""Native geometry and ProtT5 feature restoration, with public-case replay."""
import argparse,sys,time,os,re,fcntl
import numpy as np,pandas as pd,torch
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha

def geometry():
 sys.path.insert(0,str(ROOT/'.external/PaddleHelix'));sys.path.insert(0,str(ROOT/'third_party/sota_dti_2026/EviDTI'))
 from rdkit import Chem,rdBase
 from compound_tools import mol_to_geognn_graph_data_MMFF3d
 rdBase.SeedRandomNumberGenerator(20260921);out=directory('EviDTI');drugs,_=inputs();folder=out/'geometry';folder.mkdir(exist_ok=True);failed=[]
 for d in drugs.itertuples():
  p=folder/f'{d.drug_id}.npy'
  if p.exists():continue
  try:
   value=mol_to_geognn_graph_data_MMFF3d(Chem.MolFromSmiles(d.smiles));assert value is not None;np.save(p,value,allow_pickle=True)
  except Exception as e:failed.append(dict(drug_id=d.drug_id,reason=str(e)))
 dump(out/'GEOMETRY_FAILURES.json',failed);dump(out/'GEOMETRY_READY.json',{'count':len(list(folder.glob('*.npy'))),'method':'Author MMFF3d 10 conformers, native >400 atom fallback; native geometry code','rdkit_seed':20260921,'compound_tools_sha256':sha(ROOT/'third_party/sota_dti_2026/EviDTI/compound_tools.py')})

def protein():
 out=directory('EviDTI');checkpoint=ROOT/'data/research/dti_native_encoders_20260921/prot_t5_xl_half_uniref50-enc'
 while not (ROOT/'outputs/dti_official_720x745_20260921/evidti_encoder_download/VERIFIED_FILES.json').exists():time.sleep(30)
 lock=(ROOT/'outputs/dti_official_720x745_20260921/FEATURE_GPU.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 from transformers import T5EncoderModel,T5Tokenizer
 model=T5EncoderModel.from_pretrained(checkpoint).eval().half().cuda();tok=T5Tokenizer.from_pretrained(checkpoint,do_lower_case=False);torch.set_num_threads(2)
 def encode(s):
  aa=list(re.sub('[UZOB]','X',''.join(s.split())));batch=tok.batch_encode_plus([aa],add_special_tokens=True,padding=True,is_split_into_words=True,return_tensors='pt')
  with torch.inference_mode():return model(input_ids=batch['input_ids'].cuda())[0][0,:-1].cpu().numpy()
 repo=ROOT/'third_party/sota_dti_2026/EviDTI/dataset/case_drugbank';case=pd.read_csv(repo/'drugbank_total.csv');expected=np.load(repo/'protein_1d_feature.npy',allow_pickle=True);replay=[]
 for i,t in case.drop_duplicates('uid').iterrows():
  value=encode(t.seq);assert value.shape==expected[i].shape
  replay.append({'uniprot':t.uid,'max_abs_delta':float(np.max(np.abs(value.astype('float32')-expected[i].astype('float32'))))})
 delta=max(r['max_abs_delta'] for r in replay);passed=delta<0.01
 dump(out/'PROTT5_REPLAY_CHECK.json',{'passed':passed,'max_abs_delta':delta,'tolerance':0.01,'records':replay,'weights_sha256':sha(checkpoint/'pytorch_model.bin'),'encoder_variant':'Rostlab official encoder-only half-precision conversion of ProtT5 XL UniRef50; public-feature replay required'})
 if not passed:raise RuntimeError('EviDTI ProtT5 feature replay failed')
 _,targets=inputs();folder=out/'protein';folder.mkdir(exist_ok=True);failed=[];start=time.time()
 for t in targets.itertuples():
  p=folder/f'{t.target_id}.npy'
  if p.exists():continue
  try:
   value=encode(t.sequence);assert value.shape==(len(t.sequence),1024) and np.isfinite(value).all();np.save(p,value)
  except (RuntimeError,ValueError) as e:failed.append(dict(target_id=t.target_id,reason=str(e)[:400]));torch.cuda.empty_cache()
  dump(out/'PROTT5_PROGRESS.json',{'completed_targets':len(list(folder.glob('*.npy'))),'failed_targets':len(failed),'elapsed_seconds':time.time()-start})
 dump(out/'PROTT5_FAILURES.json',failed);dump(out/'PROTEIN_READY.json',{'count':len(list(folder.glob('*.npy'))),'pooling':'none; full per-residue embeddings, UZOB→X, remove EOS','precision':'native FP16 then stored FP16'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['geometry','protein']);a=p.parse_args();globals()[a.kind]()

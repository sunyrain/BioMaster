#!/usr/bin/env python3
"""Restore official ESM1b/ESM2-3B entity features without replacing encoders."""
import argparse,time,sys,fcntl
import numpy as np
import torch
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha,status

def main():
 p=argparse.ArgumentParser();p.add_argument('model',choices=['GraphBAN','ADME-DTI']);a=p.parse_args()
 name='esm1b_t33_650M_UR50S' if a.model=='GraphBAN' else 'esm2_t36_3B_UR50D'
 layer=33 if a.model=='GraphBAN' else 36
 checkpoint=ROOT/'data/research/dti_native_encoders_20260921'/f'{name}.pt';out=directory(a.model)
 dest=out/'PROTEIN_ESM.npy'
 if dest.exists():return
 while not checkpoint.exists():
  status(a.model,'WAITING_OFFICIAL_ENCODER_DOWNLOAD',encoder=name,scored_pairs=0);time.sleep(30)
 import esm
 lock=(ROOT/'outputs/dti_official_720x745_20260921/FEATURE_GPU.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX)
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 status(a.model,'ENCODING_PROTEINS',encoder=name,completed_targets=0)
 data=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model,alphabet=esm.pretrained.load_model_and_alphabet_core(name,data,regression_data=None)
 del data
 model=model.eval().cuda();convert=alphabet.get_batch_converter(truncation_seq_length=1022)
 _,targets=inputs();values=[];start=time.time()
 with torch.inference_mode():
  for i,t in enumerate(targets.itertuples()):
   seq=t.sequence[:1022];_,_,tok=convert([(t.target_id,seq)])
   r=model(tok.cuda(),repr_layers=[layer],return_contacts=False)['representations'][layer][0,1:len(seq)+1].mean(0)
   values.append(r.cpu().numpy())
   if i%10==0:status(a.model,'ENCODING_PROTEINS',encoder=name,completed_targets=i+1,elapsed_seconds=time.time()-start)
 np.save(dest,np.stack(values));dump(out/'PROTEIN_ESM_CONTRACT.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'target_ids':targets.target_id.tolist(),'native_layer':layer,'truncate_residues':1022,'pool':'mean exclude BOS/EOS','precision':'FP32 CUDA, TF32 disabled','contact_regression':'unused; no contact outputs requested'})
 status(a.model,'PROTEIN_FEATURES_READY',scored_pairs=0,completed_targets=745)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""CPU native DGL entity encoding, then exact native traced BAN/MLP CUDA tail."""
import argparse,sys,time,subprocess,json
import numpy as np
import pandas as pd
import torch
from dti_official_runtime_20260921 import ROOT,OUT,inputs,directory,dump,sha,status,save_scores

class Tail(torch.nn.Module):
 def __init__(self,model):
  super().__init__();self.bcn=model.bcn;self.mlp=model.mlp_classifier
 def forward(self,d,p):
  f,_=self.bcn(d,p);return self.mlp(f).softmax(1)[:,1]

def gpu_tail(epoch):
 out=directory('GraphBAN');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 d=torch.from_numpy(np.load(out/'DRUG_FUSION.npy')).cuda();p=torch.from_numpy(np.load(out/'PROTEIN_FUSION.npy')).cuda()
 model=torch.jit.load(str(out/'NATIVE_TAIL.pt')).eval().cuda();ref=json.loads((out/'NATIVE_PARITY_INPUT.json').read_text())
 with torch.inference_mode():
  observed=model(d[:2],p[:2]).cpu().numpy();delta=float(np.max(np.abs(observed-ref['scores'])))
  dump(out/f'PARITY_EPOCH_{epoch}.json',{'native_cpu_vs_cached_cuda_tail_max_delta':delta,'passed':delta<1e-3})
  assert delta<1e-3,delta
  scores=np.zeros((745,720),np.float32);start=time.time()
  for j in range(745):
   for i in range(0,720,16):scores[j,i:i+16]=model(d[i:i+16],p[j:j+1].expand(min(16,720-i),-1,-1)).cpu().numpy()
   if j%5==0:status('GraphBAN','INFERENCE',epoch=epoch,checkpoint_scored_pairs=(j+1)*720,elapsed_seconds=time.time()-start)
 np.save(out/f'CHECKPOINT_{epoch}_SCORES.npy',scores)

def main():
 a=argparse.ArgumentParser();a.add_argument('--gpu-tail',type=int);args=a.parse_args()
 if args.gpu_tail:gpu_tail(args.gpu_tail);return
 out=directory('GraphBAN');start=time.time()
 while not all((out/f).exists() for f in ['PROTEIN_ESM.npy','DRUG_CHEMBERTA.npy']):
  status('GraphBAN','WAITING_NATIVE_FEATURES',scored_pairs=0);time.sleep(30)
 sys.path.insert(0,str(ROOT/'.external/GraphBAN/case_study'))
 import dgl
 from models import GraphBAN
 from dataloader import DTIDataset
 from configs import get_cfg_defaults
 from utils import integer_label_protein
 torch.set_num_threads(3);torch.manual_seed(12)
 cfg=get_cfg_defaults();cfg.merge_from_file(str(ROOT/'.external/GraphBAN/case_study/GraphBAN_DA.yaml'));cfg.freeze()
 model=GraphBAN(**cfg).eval();drugs,targets=inputs();de=np.load(out/'DRUG_CHEMBERTA.npy');pe=np.load(out/'PROTEIN_ESM.npy')
 # Native predict.py truncates the Protein column before BOTH encoders.
 chars=torch.tensor(np.stack([integer_label_protein(s[:1022]) for s in targets.sequence]))
 df=pd.DataFrame({'SMILES':drugs.smiles,'Protein':targets.sequence.iloc[0][:1022],'fcfp':list(de),'esm':[pe[0]]*720})
 ds=DTIDataset(df.index.values,df);graphs=[ds[i][0] for i in range(720)]
 checkpoint_times=[]
 for epoch in range(31,51):
  if (out/f'CHECKPOINT_{epoch}_SCORES.npy').exists():continue
  status('GraphBAN','NATIVE_ENTITY_ENCODING',epoch=epoch,completed_checkpoints=epoch-31)
  model.load_state_dict(torch.load(out/f'model_epoch_{epoch}.pth',map_location='cpu'),strict=True);model.eval();dvals=[];pvals=[]
  with torch.inference_mode():
   for i in range(0,720,16):
    g=dgl.batch([x.clone() for x in graphs[i:i+16]])
    v=model.mol_fusion(model.drug_extractor(g),model.molecule_FCFP(torch.from_numpy(de[i:i+16])))
    dvals.append(v.numpy())
   for i in range(0,745,16):
    v=model.pro_fusion(model.protein_extractor(chars[i:i+16]),model.protein_esm(torch.from_numpy(pe[i:i+16])))
    pvals.append(v.numpy())
   d=np.concatenate(dvals);p=np.concatenate(pvals)
   direct=model(dgl.batch([graphs[0].clone(),graphs[1].clone()]),torch.from_numpy(de[:2]),chars[:2],torch.from_numpy(pe[:2]),torch.device('cpu'),mode='eval')[2].softmax(1)[:,1]
   tail=Tail(model).eval();cached=tail(torch.from_numpy(d[:2]),torch.from_numpy(p[:2]));delta=float((direct-cached).abs().max());assert delta<2e-5,delta
   traced=torch.jit.trace(tail,(torch.from_numpy(d[:2]),torch.from_numpy(p[:2])),check_trace=True)
   traced.save(str(out/'NATIVE_TAIL.pt'))
  np.save(out/'DRUG_FUSION.npy',d);np.save(out/'PROTEIN_FUSION.npy',p)
  dump(out/'NATIVE_PARITY_INPUT.json',{'epoch':epoch,'scores':direct.numpy().tolist(),'cached_cpu_max_delta':delta})
  subprocess.run([str(ROOT/'.venvs/balm/bin/python'),str(ROOT/'scripts/run_dti_official_graphban_20260921.py'),'--gpu-tail',str(epoch)],check=True,cwd=ROOT)
 scores=np.stack([np.load(out/f'CHECKPOINT_{e}_SCORES.npy') for e in range(31,51)]).mean(0)
 frame=pd.DataFrame({'target_id':np.repeat(targets.target_id.to_numpy(),720),'drug_id':np.tile(drugs.drug_id.to_numpy(),745),'score':scores.ravel()});save_scores('GraphBAN',frame)
 status('GraphBAN','COMPLETE',scored_pairs=len(frame),elapsed_seconds=time.time()-start,checkpoint_count=20)
if __name__=='__main__':main()

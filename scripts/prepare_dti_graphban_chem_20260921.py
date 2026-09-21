#!/usr/bin/env python3
import torch,numpy as np
from transformers import AutoTokenizer,RobertaModel
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha
out=directory('GraphBAN');p=ROOT/'data/research/dti_native_encoders_20260921/ChemBERTa-77M-MTR'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
tok=AutoTokenizer.from_pretrained(p);m=RobertaModel.from_pretrained(p,num_labels=2,add_pooling_layer=True).eval().cuda();drugs,_=inputs();values=[]
with torch.inference_mode():
 for d in drugs.itertuples():
  v=tok(d.smiles,return_tensors='pt',padding='max_length',max_length=290,truncation=True).to('cuda');values.append(m(**v).last_hidden_state[0,0].cpu().numpy())
np.save(out/'DRUG_CHEMBERTA.npy',np.stack(values));dump(out/'DRUG_CHEMBERTA_CONTRACT.json',{'drug_ids':drugs.drug_id.tolist(),'checkpoint_sha256':sha(p/'pytorch_model.bin'),'pool':'native CLS hidden; unused newly initialized pooler is never read','precision':'FP32 CUDA'})

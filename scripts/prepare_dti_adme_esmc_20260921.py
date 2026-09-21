#!/usr/bin/env python3
"""Native ESM-C 600M full-sequence pooled features; no substitute windowing."""
import time,json,sys,fcntl
import numpy as np,pandas as pd,torch
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha
out=directory('ADME-DTI');weight=ROOT/'data/research/dti_native_encoders_20260921/esmc-600m-2024-12/data/weights/esmc_600m_2024_12_v0.pth'
while not weight.exists():time.sleep(30)
lock=(ROOT/'outputs/dti_official_720x745_20260921/FEATURE_GPU.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from esm.sdk.api import ESMProtein,LogitsConfig
torch.set_num_threads(2);model=ESMC(d_model=1152,n_heads=18,n_layers=36,tokenizer=get_esmc_model_tokenizers(),use_flash_attn=False).eval()
model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
model=model.cuda().to(torch.bfloat16);_,targets=inputs();values=[];failed=[];start=time.time()
with torch.inference_mode():
 for t in targets.itertuples():
  try:
   pt=model.encode(ESMProtein(sequence=t.sequence));v=model.logits(pt,LogitsConfig(return_embeddings=True)).embeddings.mean(dim=1)[0].float().cpu().numpy();assert np.isfinite(v).all();values.append(dict(target=t.sequence,embedding=v,target_id=t.target_id))
  except (RuntimeError,ValueError) as e:
   failed.append(dict(target_id=t.target_id,reason=type(e).__name__,detail=str(e)[:300]));torch.cuda.empty_cache()
  dump(out/'ESMC_PROGRESS.json',{'completed_targets':len(values),'failed_targets':len(failed),'last_target':t.target_id,'elapsed_seconds':time.time()-start})
path=out/'data/embeddings/target_embedding';path.mkdir(parents=True,exist_ok=True);pd.DataFrame(values).to_parquet(path/'ESMC.parquet',index=False)
dump(out/'ESMC_INPUT_FAILURES.json',failed);dump(out/'ESMC_CONTRACT.json',{'weights_sha256':sha(weight),'pool':'Native client.logits embeddings.mean(dim=1)[0], includes special tokens','length':'full sequence, failure retained; no truncation or window extension','precision':'native SDK bf16 on CUDA; native unfused attention','strict_checkpoint_load':True})

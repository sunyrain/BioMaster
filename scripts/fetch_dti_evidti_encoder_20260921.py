#!/usr/bin/env python3
import requests
from concurrent.futures import ThreadPoolExecutor
import download_dti_official_weights_20260921 as t
t.OUT=t.ROOT/'outputs/dti_official_720x745_20260921/evidti_encoder_download';t.OUT.mkdir(exist_ok=True);repo='Rostlab/prot_t5_xl_half_uniref50-enc'
r=requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true',timeout=60);r.raise_for_status();meta=r.json();rev=meta['sha'];rows=[]
for f in meta['siblings']:
 name=f['rfilename']
 if name.endswith(('.json','.model','.txt')) or name=='pytorch_model.bin':
  lfs=f.get('lfs',{});rows.append(dict(model='prot_t5_xl_half_uniref50-enc',name=name,bytes=f['size'],revision=rev,url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}?download=true',expected_sha256=lfs.get('sha256'),expected_git_blob=f.get('blobId') if not lfs else None,destination=f'data/research/dti_native_encoders_20260921/prot_t5_xl_half_uniref50-enc/{name}'))
t.save(t.OUT/'MANIFEST.json',{'files':rows})
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(t.download,rows))
t.save(t.OUT/'VERIFIED_FILES.json',results);assert all(r['status']=='VERIFIED' for r in results)

#!/usr/bin/env python3
"""Resume required official FAIR encoders and add native ESM-C checkpoint."""
import argparse,requests
from concurrent.futures import ThreadPoolExecutor
import download_dti_official_weights_20260921 as t

def main():
 p=argparse.ArgumentParser();p.add_argument('--esmc-only',action='store_true');a=p.parse_args()
 t.OUT=t.ROOT/'outputs/dti_official_720x745_20260921'/('esmc_download' if a.esmc_only else 'fair_esm_download');t.OUT.mkdir(exist_ok=True)
 t.STORE=t.ROOT/'data/research/dti_native_encoders_20260921';rows=[]
 if not a.esmc_only:
  for name,size in [('esm1b_t33_650M_UR50S',7828576466),('esm2_t36_3B_UR50D',5678116398)]:
   rows.append(dict(model=name,name=name+'.pt',bytes=size,url=f'https://dl.fbaipublicfiles.com/fair-esm/models/{name}.pt',expected_sha256=None,destination=str((t.STORE/(name+'.pt')).relative_to(t.ROOT))))
 else:
  repo='biohub/esmc-600m-2024-12';r=requests.get(f'https://huggingface.co/api/models/{repo}?blobs=true',timeout=60);r.raise_for_status();meta=r.json();rev=meta['sha']
  for f in meta['siblings']:
   if f['rfilename'].endswith('.pth'):
    rows.append(dict(model='esmc-600m-2024-12',name=f['rfilename'],bytes=f['size'],url=f"https://huggingface.co/{repo}/resolve/{rev}/{f['rfilename']}?download=true",expected_sha256=f['lfs']['sha256'],revision=rev,destination=str((t.STORE/'esmc-600m-2024-12'/f['rfilename']).relative_to(t.ROOT))))
 t.save(t.OUT/'MANIFEST.json',dict(files=rows))
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(t.download,rows))
 t.save(t.OUT/'VERIFIED_FILES.json',results)
 assert all(r['status']=='VERIFIED' for r in results)
if __name__=='__main__':main()

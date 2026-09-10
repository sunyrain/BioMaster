#!/usr/bin/env python3
"""Archive official affinity data with checksums; restart-safe downloads."""
from pathlib import Path
import requests,concurrent.futures,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/external/affinity_refresh_20260910'
def download(job):
 name,url=job;p=OUT/name
 for attempt in range(3):
  try:
   if not p.exists():
    with requests.get(url.replace('www.bindingdb','ww.bindingdb') if attempt==1 else url,stream=True,timeout=(20,90)) as r:
     r.raise_for_status()
     with p.with_suffix(p.suffix+'.part').open('wb') as f:
      for b in r.iter_content(4*1024*1024):f.write(b)
    p.with_suffix(p.suffix+'.part').replace(p)
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   result=dict(file=str(p.relative_to(ROOT)),url=url,bytes=p.stat().st_size,sha256=h.hexdigest(),status='downloaded')
   print(name,result['bytes'],flush=True);return result
  except Exception as e:print(name,attempt,str(e),flush=True)
 return dict(file=str(p.relative_to(ROOT)),url=url,status='failed',error='all 3 attempts failed')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 jobs=json.loads((OUT/'download_jobs.json').read_text())
 for i in range(3,8):
  jobs.append((f'RAF_MEK_MOESM{i}.xlsx',f'https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41589-026-02212-2/MediaObjects/41589_2026_2212_MOESM{i}_ESM.xlsx'))
 jobs.append(('GatorAffinity_README.md','https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-DB/raw/main/README.md'))
 # Metadata only; no multi-terabyte predicted structures.
 jobs.append(('GatorAffinity_index.csv','https://huggingface.co/datasets/AIDD-LiLab/GatorAffinity-DB/resolve/main/GatorAffnity-DB_index.csv'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(download,jobs))
 (OUT/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()

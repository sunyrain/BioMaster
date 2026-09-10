"""Pin public EC-KG revisions and verify downloaded shards against publisher SHA256."""
import requests,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
OUT=Path(__file__).resolve().parents[1]/'outputs/biomaster_disease_evidence_720x888_20260909/eckg';OUT.mkdir(exist_ok=True)
def get(url):
 r=requests.get(url,timeout=60);r.raise_for_status();return r
jobs=[];manifest={}
for name in ['drug-list','disease-list','kg-nodes','kg-edges']:
 meta=get(f'https://huggingface.co/api/datasets/everycure/{name}').json();rev=meta['sha']
 files=get(f'https://huggingface.co/api/datasets/everycure/{name}/tree/{rev}?recursive=true').json()
 manifest[name]={'revision':rev,'lastModified':meta.get('lastModified'),'files':[f for f in files if f['path'].endswith('.parquet')]}
 for f in manifest[name]['files']:jobs.append((name,rev,f))
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
def download(job):
 name,rev,f=job;p=OUT/name/Path(f['path']).name;p.parent.mkdir(exist_ok=True)
 expected=f['lfs']['oid']
 if p.exists() and p.stat().st_size==f['size']:return str(p)
 for attempt in range(3):
  try:
   r=requests.get(f'https://huggingface.co/datasets/everycure/{name}/resolve/{rev}/{f["path"]}?download=true',stream=True,timeout=120);r.raise_for_status();h=hashlib.sha256()
   with open(str(p)+'.part','wb') as w:
    for b in r.iter_content(1024*1024):w.write(b);h.update(b)
   assert h.hexdigest()==expected,(p,'checksum');Path(str(p)+'.part').rename(p);print(name,p.name,flush=True);return str(p)
  except Exception as e:
   print('retry',name,p.name,str(e)[:150],flush=True);time.sleep(1)
 return None
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(download,jobs))
(OUT/'DOWNLOAD_STATUS.json').write_text(json.dumps({'requested':len(jobs),'completed':sum(x is not None for x in results)},indent=2))

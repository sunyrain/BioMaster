#!/usr/bin/env python3
"""Fetch pinned encoders, or stream only required DTBind occurrence graphs."""
import argparse
import hashlib
import json
import io
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

class RangeStream(io.RawIOBase):
    """Ordered, bounded-memory HTTP ranges for a slow large gzip download."""
    def __init__(self,url,total=38706096444,chunk=4*1024*1024,workers=8):
        self.url,self.total,self.chunk,self.workers=url,total,chunk,workers
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.index=0;self.position=0;self.buffer=b'';self.jobs={}
        for i in range(workers):self.submit(i)
    def submit(self,i):
        if i*self.chunk<self.total:self.jobs[i]=self.pool.submit(self.fetch,i)
    def fetch(self,i):
        start=i*self.chunk;end=min(self.total-1,start+self.chunk-1)
        for attempt in range(4):
            try:
                r=requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,120))
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
                assert len(r.content)==end-start+1
                return r.content
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    def readable(self):return True
    def read(self,size=-1):
        if size<0:raise ValueError('unbounded reads prohibited')
        result=[]
        while size>0:
            if not self.buffer:
                if self.index not in self.jobs:break
                self.buffer=self.jobs.pop(self.index).result();self.submit(self.index+self.workers);self.index+=1
            piece=self.buffer[:size];self.buffer=self.buffer[len(piece):];size-=len(piece);self.position+=len(piece);result.append(piece)
        return b''.join(result)
    def close(self):
        for job in self.jobs.values():job.cancel()
        self.pool.shutdown(wait=False,cancel_futures=True)
        super().close()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/frontier_dti_20260916'

def encoders():
    specs = [
        ('Rostlab/prot_bert_bfd','6c5c8a55a52ff08a664dfd584aa1773f125a0487',['config.json','pytorch_model.bin','special_tokens_map.json','tokenizer_config.json','vocab.txt']),
        ('seyonec/PubChem10M_SMILES_BPE_450k','c18fccd09b3326bf2d4633412c256d7db872156d',['config.json','pytorch_model.bin','merges.txt','vocab.json','special_tokens_map.json','tokenizer_config.json'])]
    jobs=[(repo,rev,name) for repo,rev,names in specs for name in names]
    def fetch(job):
        repo,rev,name=job
        path=ROOT/'.cache/frontier_dti'/repo.split('/')[-1]/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
            with requests.get(url,stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                with path.with_suffix(path.suffix+'.part').open('wb') as f:
                    for c in r.iter_content(4*1024*1024): f.write(c)
            path.with_suffix(path.suffix+'.part').replace(path)
        info=dict(repo=repo,revision=rev,file=name,bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
        print(json.dumps(info),flush=True);return info
    with ThreadPoolExecutor(max_workers=3) as pool: items=list(pool.map(fetch,jobs))
    (OUT/'ENCODERS.json').write_text(json.dumps(items,indent=2))

def dtbind_graphs():
    dest=OUT/'dtbind/protein_graph';dest.mkdir(parents=True,exist_ok=True)
    targets=set(pd.read_csv(OUT/'UNIQUE_PAIRS.csv').uniprot_id)
    # Do not store/extract the 38.7 GB archive. Occurrence graphs are the first section.
    url='https://zenodo.org/records/17283638/files/DTBind_datasets.tar.gz?download=1'
    found=[];count=0;started=time.time(); entered=False
    with RangeStream(url) as response:
        with tarfile.open(fileobj=response,mode='r|gz') as archive:
            for member in archive:
                name=member.name
                if '/occurrence/protein_graph/' in name and member.isfile():
                    entered=True;count+=1
                    accession=Path(name).stem
                    if accession in targets:
                        payload=archive.extractfile(member).read()
                        (dest/(accession+'.pt')).write_bytes(payload)
                        found.append(dict(uniprot_id=accession,archive_member=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
                        print(f'kept {accession}: {len(found)}/{len(targets)}; scanned {count}',flush=True)
                    if count%10==0:
                        temp=OUT/'dtbind/FETCH_STATUS.tmp'
                        temp.write_text(json.dumps(dict(status='running',scanned=count,found=len(found),requested=len(targets),downloaded_bytes=response.position,elapsed_seconds=time.time()-started)))
                        temp.replace(OUT/'dtbind/FETCH_STATUS.json')
                elif entered and '/occurrence/protein_graph' not in name:
                    print('Completed protein graph archive section',name,flush=True);break
    (OUT/'dtbind/GRAPH_SOURCES.json').write_text(json.dumps(dict(source=url,graphs=found,requested=sorted(targets),missing=sorted(targets-{x['uniprot_id'] for x in found}),scanned=count,elapsed_seconds=time.time()-started),indent=2))
    (OUT/'dtbind/FETCH_STATUS.json').write_text(json.dumps(dict(status='completed',scanned=count,found=len(found),requested=len(targets),elapsed_seconds=time.time()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['encoders','dtbind']);args=p.parse_args()
    encoders() if args.action=='encoders' else dtbind_graphs()

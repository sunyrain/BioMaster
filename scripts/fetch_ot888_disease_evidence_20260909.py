"""Fetch every associated disease with source therapeutic areas, no top-N truncation."""
import json,time,gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
from fetch_final1000_opentargets_full_diseases import post_graphql,META_QUERY,TARGET_DISEASES_QUERY
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909';CACHE=OUT/'ot_cache';CACHE.mkdir(exist_ok=True)
Q=TARGET_DISEASES_QUERY.replace('name\n        }','name\n          therapeuticAreas { id name }\n        }')
def call(q,v):return post_graphql(q,v,60,3,1)
def fetch(r):
 eid=str(r.ot_id);path=CACHE/(eid+'.json.gz')
 if ';' in eid:
  parts=[]
  for single in eid.split(';'):
   status=fetch(r._replace(ot_id=single))
   if status['status']!='COMPLETE':return {**status,'ot_id':eid,'status':'MULTI_ID_INCOMPLETE'}
   with gzip.open(CACHE/(single+'.json.gz'),'rt') as f:parts.append(json.load(f))
  unique={}
  for part in parts:
   for row in part['rows']:
    key=row['disease']['id']
    if key not in unique or row['score']>unique[key]['score']:unique[key]={**row,'source_ensembl_id':part['target']['id']}
  merged={'target':{'id':eid,'approvedSymbol':r.gene_symbol},'count':len(unique),'rows':list(unique.values()),'count_semantics':'UNION_OF_SAME_UNIPROT_MULTI_ENSEMBL_DISEASES','individual_counts':{p['target']['id']:p['count'] for p in parts}}
  with gzip.open(path,'wt') as f:json.dump(merged,f)
  return {'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'ot_id':eid,'status':'COMPLETE_MULTI_ID_UNION','count':len(unique),'approved_symbol':r.gene_symbol}
 if not eid.startswith('ENSG'):return {'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'status':'MISSING_ENSEMBL','count':0}
 try:
  if path.exists():
   with gzip.open(path,'rt') as f:j=json.load(f)
  else:
   rows=[];page=0;count=None;target=None
   while count is None or len(rows)<count:
    data=call(Q,{'ensemblId':eid,'pageIndex':page,'pageSize':3000})['data']['target']
    if data is None:break
    target={k:data[k] for k in ['id','approvedSymbol','approvedName']};a=data['associatedDiseases'];count=a['count'];rows+=a['rows'];page+=1
    if not a['rows'] and len(rows)<count:raise ValueError('incomplete pagination')
   j={'target':target,'count':count,'rows':rows,'fetched_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
   if count is not None:
    unique={x['disease']['id']:x for x in rows}
    # Tied scores can repeat at page boundaries; independent page boundaries recover gaps.
    for size in [2999,2903,2501]:
     if len(unique)==count:break
     for pi in range((count+size-1)//size):
      a=call(Q,{'ensemblId':eid,'pageIndex':pi,'pageSize':size})['data']['target']['associatedDiseases']
      unique.update({x['disease']['id']:x for x in a['rows']})
    j['rows']=list(unique.values());j['unique_count']=len(unique)
   with gzip.open(path,'wt') as f:json.dump(j,f)
  return {'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'ot_id':eid,'status':('COMPLETE' if len(j['rows'])==j['count'] else 'PARTIAL_PAGINATION') if j['target'] else 'TARGET_NOT_FOUND','count':j['count'] or 0,'approved_symbol':(j['target'] or {}).get('approvedSymbol','')}
 except Exception as e:return {'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'ot_id':eid,'status':'ERROR','error':repr(e),'count':0}
def main():
 meta=call(META_QUERY,{});(OUT/'OT_META_START.json').write_text(json.dumps(meta,indent=2))
 t=pd.read_csv(OUT/'TARGET_REGISTRY_888.csv');result=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(fetch,r) for r in t.itertuples()]
  for f in as_completed(fs):
   result.append(f.result())
   if len(result)%25==0:print(len(result),pd.Series([r['status'] for r in result]).value_counts().to_dict(),flush=True)
 pd.DataFrame(result).to_csv(OUT/'OT_TARGET_COVERAGE_888.csv',index=False)
 (OUT/'OT_META_END.json').write_text(json.dumps(call(META_QUERY,{}),indent=2))
 print('finished',flush=True)
if __name__=='__main__':main()

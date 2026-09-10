"""Fetch public FDA labeling excerpts for origin review; no claim of a complete approval registry."""
import concurrent.futures,gzip,hashlib,json,re,time,threading
from pathlib import Path
import pandas as pd,requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/joint384_comprehensive_20260909';CACHE=OUT/'fda_label_cache'
lock=threading.Lock();last=[0.]
def query(name):
 p=CACHE/(hashlib.sha256(name.encode()).hexdigest()[:16]+'.json.gz')
 if p.exists():
  with gzip.open(p,'rt') as f:return json.load(f)
 url='https://api.fda.gov/drug/label.json';params={'search':f'openfda.generic_name:"{name}"','limit':30,'sort':'effective_time:desc'}
 j={};response_url='';status=0
 for attempt in range(3):
  with lock:
   delay=max(0,.38-(time.monotonic()-last[0]));time.sleep(delay);last[0]=time.monotonic()
  try:
   r=requests.get(url,params=params,timeout=40);status=r.status_code;response_url=r.url;j=r.json()
   if status in [200,404]:break
  except Exception as e:j={'error':str(e)}
  time.sleep(2+attempt*2)
 result={'drug_names':name,'url':response_url,'http_status':status,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'response':j}
 with gzip.open(p,'wt') as f:json.dump(result,f)
 return result

def main():
 CACHE.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(ROOT/'outputs/joint384_design_20260909/ALL_574_AGENT_REVIEW.csv');d=d[d.decision.isin(['KEEP_REVIEW','LOW_PRIORITY'])].copy();names=sorted(d.drug_names.unique());rows=[];index=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for n,res in enumerate(pool.map(query,names),1):
   name=res['drug_names'];response=res['response'];records=response.get('results',[]);eligible=0;combinations=0;snippets=[];links=[];ids=[]
   for rec in records:
    o=rec.get('openfda',{});generic=';'.join(o.get('generic_name',[]));applications=o.get('application_number',[]);text='\n'.join(rec.get('indications_and_usage',[]) or rec.get('purpose',[]) or rec.get('otc_purpose',[]));sid=rec.get('set_id','')
    match=bool(re.search(r'(?<![a-z])'+re.escape(name.lower())+r'(?![a-z])',generic.lower()))
    combo=bool(re.search(r'\bAND\b|;|,| / ',generic,flags=re.I))
    approved_record=any(re.match(r'^(NDA|ANDA|BLA)\d+',x) for x in applications)
    ok=match and approved_record and not combo and bool(text)
    row={'drug_names':name,'set_id':sid,'effective_time':rec.get('effective_time'),'generic_name':generic,'brand_names':';'.join(o.get('brand_name',[])),'application_numbers':';'.join(applications),'name_match':match,'combination_flag':combo,'approval_application_record_present':approved_record,'eligible_single_drug_label':ok,'indications_text':text,'pharm_class_epc':';'.join(o.get('pharm_class_epc',[])),'source_url':'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid='+sid,'api_url':res['url']}
    rows.append(row)
    if ok:
     eligible+=1
     if text not in snippets:snippets.append(text);links.append(row['source_url']);ids.append(sid)
    if combo:combinations+=1
   # All retrieved unique eligible texts retained in long table; compact review text capped by label count, not sentence truncation.
   index.append({'drug_names':name,'pair_count':int(d.drug_names.eq(name).sum()),'http_status':res['http_status'],'returned_records':len(records),'total_api_matches':response.get('meta',{}).get('results',{}).get('total',0),'eligible_single_drug_label_count':eligible,'unique_eligible_indication_texts':len(snippets),'combination_records':combinations,'indications_text':'\n\n[LABEL]\n'.join(snippets[:4]),'source_urls':';'.join(links[:4]),'all_eligible_set_ids':';'.join(ids),'label_status':'FDA_SINGLE_DRUG_LABEL_FOUND' if eligible else 'NO_USABLE_SINGLE_DRUG_LABEL','api_last_updated':response.get('meta',{}).get('last_updated','')})
   if n%20==0:print(n,'/',len(names),'labels',sum(x['eligible_single_drug_label_count']>0 for x in index),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'FDA_LABEL_RECORDS.csv.gz',index=False)
 frame=pd.DataFrame(index);frame.to_csv(OUT/'DRUG_ORIGIN_LABEL_REVIEW_INPUT.csv',index=False)
 for i in range(3):frame.iloc[i::3].to_csv(OUT/f'ORIGIN_REVIEW_PART_{i+1}.csv',index=False)
 (OUT/'LABEL_FETCH_SUMMARY.json').write_text(json.dumps({'drugs':len(frame),'with_usable_single_drug_label':int(frame.eligible_single_drug_label_count.gt(0).sum()),'http_statuses':frame.http_status.value_counts().to_dict(),'records':len(rows),'max_results_per_drug':30,'not_complete_approval_registry':True},indent=2));print((OUT/'LABEL_FETCH_SUMMARY.json').read_text(),flush=True)
if __name__=='__main__':main()

"""Restore frozen-universe disease evidence without training. Resumable identity audit."""
import json,re,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def norm(x):return re.sub('[^a-z0-9]','',str(x).lower())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def registries():
 rank=ROOT/'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz'
 d=pd.read_csv(rank,usecols=['ligand_inchikey','drug_names']).drop_duplicates()
 assert len(d)==720 and d.ligand_inchikey.is_unique
 f=pd.read_csv(ROOT/'outputs/final_design_space_2005_2026_v1/FINAL_FROZEN_FDA_DRUG_ENTITIES_2005_2026.csv')
 f['frozen_rows_for_structure']=f.groupby('model_inchikey').model_inchikey.transform('size')
 f=f.drop_duplicates('model_inchikey')
 d=d.merge(f,left_on='ligand_inchikey',right_on='model_inchikey',how='left',validate='one_to_one')
 d['exact_structure_in_fda_registry']=d.model_inchikey.notna()
 feats=pd.read_csv(ROOT/'outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1/OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz',usecols=['ligand_inchikey','ligand_smiles'])
 d=d.merge(feats,on='ligand_inchikey',validate='one_to_one')
 d.to_csv(OUT/'DRUG_REGISTRY_720.csv',index=False)
 t=pd.read_csv(ROOT/'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv');assert len(t)==888
 t.to_csv(OUT/'TARGET_REGISTRY_888.csv',index=False)
 n=pd.read_csv(ROOT/'data/raw/txgnn/node.csv',dtype=str);n=n[n.node_type.eq('drug')]
 def match(r):
  names=set(norm(s) for s in str(r.drug_names).split(';'))
  m=n[n.node_name.map(norm).isin(names)];rule='EXACT_NAME';path=OUT/'raw'/f'{r.ligand_inchikey}.json'
  if len(m)!=1 or not r.exact_structure_in_fda_registry or ';' in r.drug_names or norm(r.drug_names)=='fenofibrate':
   old=ROOT/'outputs/txgnn_biopathnet_comparison_20260909/raw'/path.name
   if not path.exists() and old.exists():path.write_bytes(old.read_bytes())
   if not path.exists():
    for attempt in range(3):
     try:
      resp=requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{r.ligand_inchikey}/synonyms/JSON',timeout=35)
      j={'status':resp.status_code,'body':resp.json(),'fetched_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};dump(path,j)
      if resp.status_code in [200,404]:break
     except Exception as e:
      if attempt==2:dump(path,{'status':'ERROR','error':str(e)})
     time.sleep(1)
   j=json.loads(path.read_text());syn=[]
   for a in j.get('body',{}).get('InformationList',{}).get('Information',[]):syn+=a.get('Synonym',[])
   m=n[n.node_name.map(norm).isin(set(map(norm,syn)))|n.node_id.isin(syn)];rule='FULL_INCHIKEY_PUBCHEM_SYNONYM'
  m=m.drop_duplicates('node_id');ok=len(m)==1
  return {'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'graph_drug_id':m.iloc[0].node_id if ok else '', 'graph_drug_name':m.iloc[0].node_name if ok else '', 'mapping_rule':rule,'mapping_status':'MAPPED_NAME_STRUCTURE_NOT_VERIFIED_IN_GRAPH' if ok else ('AMBIGUOUS_HOLD' if len(m)>1 else 'UNMAPPED'),'candidate_ids':';'.join(m.node_id),'identity_hold':r.ligand_inchikey=='LPAUOXUZGSBGDU-STDDISTJSA-N'}
 with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(match,d.itertuples()))
 c=pd.DataFrame(rows);dup=c.graph_drug_id.ne('') & c.graph_drug_id.duplicated(keep=False)
 c.loc[dup,'mapping_status']='MULTIPLE_PROJECT_STRUCTURES_TO_ONE_GRAPH_NODE_HOLD'
 c['inference_eligible']=c.mapping_status.eq('MAPPED_NAME_STRUCTURE_NOT_VERIFIED_IN_GRAPH') & ~c.identity_hold
 c.to_csv(OUT/'DRUG_COVERAGE_720.csv',index=False)
 print(c.mapping_status.value_counts().to_dict(),flush=True)
if __name__=='__main__':registries()

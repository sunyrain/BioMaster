import pandas as pd,re,requests,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
out=Path('outputs/txgnn_biopathnet_comparison_20260909/raw')
p=pd.read_csv('outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv');d=p[p.selection_role.ne('POSITIVE_CONTROL')].drop_duplicates('ligand_inchikey')
n=pd.read_csv('data/raw/txgnn/node.csv',dtype=str);n=n[n.node_type.eq('drug')]
norm=lambda x:re.sub('[^a-z0-9]','',str(x).lower());names=set(n.node_name.map(norm))
missing=d[~d.drug_names.map(norm).isin(names)]
def run(row):
 f=out/(row.ligand_inchikey+'.json');url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{row.ligand_inchikey}/synonyms/JSON'
 if f.exists():j=json.loads(f.read_text())
 else:
  try:
   r=requests.get(url,timeout=40);j={'url':url,'status':r.status_code,'body':r.json()};f.write_text(json.dumps(j))
  except Exception as e:return row.drug_names,str(e)
 hits=[]
 for a in j.get('body',{}).get('InformationList',{}).get('Information',[]):
  for s in a.get('Synonym',[]):
   if norm(s) in names or s in set(n.node_id):hits.append(s)
 return row.drug_names,hits
with ThreadPoolExecutor(max_workers=3) as ex:
 for r in ex.map(run,missing.itertuples()):print(r,flush=True)

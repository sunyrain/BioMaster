"""Audit pinned EC-KG coverage and project-relevant edge increments vs frozen PrimeKG.
New endpoint pairs are evidence increments, never an estimate of prediction improvement.
"""
import json,re,collections,time
from pathlib import Path
import pandas as pd,pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909';EC=OUT/'eckg'
def alias(x):
 x=str(x);x=x.replace('DRUGBANK:','DrugBank:').replace('UNIPROTKB:','UniProtKB:').replace('ENSEMBL:','ENSEMBL:').replace('NCBIGENE:','NCBIGene:')
 if x.startswith('CHEMBL.COMPOUND:'):x='CHEMBL:'+x.split(':',1)[1]
 return x

def main():
 d=pd.read_csv(OUT/'DRUG_REGISTRY_720.csv').fillna('');c=pd.read_csv(OUT/'DRUG_COVERAGE_720.csv').fillna('');t=pd.read_csv(OUT/'TARGET_REGISTRY_888.csv').fillna('')
 desired=collections.defaultdict(set);kind={}
 for r in d.itertuples():
  key='drug:'+r.ligand_inchikey;kind[key]='drug';desired['INCHIKEY:'+r.ligand_inchikey].add((key,'FULL_INCHIKEY'))
  # GSRS structure must exactly match this model entity before using its CID/ChEMBL aliases.
  if r.exact_structure_in_fda_registry:
   if r.gsrs_pubchem_cid and r.pubchem_to_model_structure_relation=='PUBCHEM_CID_EQUALS_MODEL_FULL_INCHIKEY':desired['PUBCHEM.COMPOUND:'+str(r.gsrs_pubchem_cid).removesuffix('.0')].add((key,'STRUCTURE_VERIFIED_PUBCHEM'))
   for x in str(r.chembl_full_match_ids).split(';'):
    if x.startswith('CHEMBL'):desired['CHEMBL:'+x.strip()].add((key,'FULL_INCHIKEY_CHEMBL'))
  p=OUT/'raw'/f'{r.ligand_inchikey}.json'
  if p.exists():
   j=json.loads(p.read_text())
   for item in j.get('body',{}).get('InformationList',{}).get('Information',[]):
    if item.get('CID'):desired['PUBCHEM.COMPOUND:'+str(item['CID'])].add((key,'FULL_INCHIKEY_PUBCHEM_LOOKUP'))
 for r in c.itertuples():
  if r.inference_eligible:desired['DrugBank:'+r.graph_drug_id].add(('drug:'+r.ligand_inchikey,'TXGNN_NAME_IDENTITY'))
 for r in t.itertuples():
  key='target:'+r.target_chembl_id;kind[key]='target';desired['UniProtKB:'+r.uniprot_accession].add((key,'EXACT_UNIPROT'))
  if r.ot_id:desired['ENSEMBL:'+r.ot_id].add((key,'REGISTRY_ENSEMBL'))
 nodes=[];matches=[];total=0;types=collections.Counter()
 for p in sorted((EC/'kg-nodes').glob('*.parquet')):
  for b in pq.ParquetFile(p).iter_batches(batch_size=20000,columns=['id','name','category','equivalent_identifiers','upstream_data_source']):
   for r in b.to_pylist():
    total+=1;types[r['category']]+=1;ids=set(map(alias,[r['id']]+(r['equivalent_identifiers'] or [])));hits=set()
    for x in ids:
     for key,rule in desired.get(x,[]):hits.add((key,rule,x))
    if hits or r['id'].startswith('MONDO:') or any(x.startswith('MONDO:') for x in ids):nodes.append(r)
    for key,rule,x in hits:matches.append({'project_key':key,'entity_type':kind[key],'ec_id':r['id'],'ec_name':r['name'],'ec_category':r['category'],'mapping_rule':rule,'matched_identifier':x,'equivalent_identifiers':json.dumps(r['equivalent_identifiers'])})
  print('nodes shard',p.name,flush=True)
 m=pd.DataFrame(matches).drop_duplicates();m.to_csv(OUT/'ECKG_PROJECT_NODE_MATCHES.csv.gz',index=False)
 pq.write_table(pa.Table.from_pylist(nodes),OUT/'ECKG_PROJECT_AND_DISEASE_NODES.parquet')
 summary={'total_nodes_scanned':total,'node_categories':dict(types),'mapped_project_drugs':int(m[m.entity_type.eq('drug')].project_key.nunique()),'mapped_project_targets':int(m[m.entity_type.eq('target')].project_key.nunique())}
 endpoints=set(m.ec_id);writer=None;edgecount=kept=0
 manifest=json.loads((EC/'SOURCE_MANIFEST.json').read_text())
 for file in manifest['kg-edges']['files']:
  p=EC/'kg-edges'/Path(file['path']).name
  deadline=time.time()+1800
  while not p.exists():
   if time.time()>deadline:raise TimeoutError(str(p))
   time.sleep(5)
  for b in pq.ParquetFile(p).iter_batches(batch_size=100000):
   edgecount+=len(b);tab=pa.Table.from_batches([b]);sel=pc.or_(pc.is_in(tab['subject'],value_set=pa.array(list(endpoints))),pc.is_in(tab['object'],value_set=pa.array(list(endpoints))))
   part=tab.filter(sel);kept+=len(part)
   if writer is None:writer=pq.ParquetWriter(OUT/'ECKG_PROJECT_INCIDENT_EDGES.parquet',part.schema,compression='zstd')
   writer.write_table(part)
  print('edges shard',p.name,'kept',kept,flush=True)
 if writer:writer.close()
 summary.update({'total_edges_scanned':edgecount,'project_incident_edge_rows':kept,'edge_shards_scanned':len(list((EC/'kg-edges').glob('*.parquet'))),'node_shards_scanned':len(list((EC/'kg-nodes').glob('*.parquet')))})
 (OUT/'ECKG_SCAN_SUMMARY.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()

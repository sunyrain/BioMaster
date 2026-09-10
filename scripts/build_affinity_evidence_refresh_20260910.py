#!/usr/bin/env python3
"""Reproducible source-aware evidence index. Frozen SPR candidates are read-only.
Full BindingDB archive is scanned; index retains 720-drug connectivity neighborhood
and extra control compounds. Experimental datasets retain all measured rows.
"""
from pathlib import Path
import csv,gzip,hashlib,io,json,re,sqlite3,zipfile,math
from functools import lru_cache
from collections import Counter,defaultdict
import pandas as pd
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/external/affinity_refresh_20260910'
SURVEY=ROOT/'outputs/recent_affinity_sources_20260910'
OUT=ROOT/'outputs/affinity_evidence_refresh_20260910'
DB=ROOT/'data/processed/biomaster_affinity_evidence_20260910.sqlite'
BASE=ROOT/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv'
DRUG=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv'
TARGET=ROOT/'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv'
CHEMBL=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
OLD=ROOT/'outputs/evidence_routing_compute_execution_20260808_v1/external_relation_audit_v7/raw/bindingdb_202607'
COLUMNS=['source','record_id','ligand_key','ligand_name','smiles','target_accessions','target_name','target_scope','organism','endpoint','relation','value_raw','unit_raw','value_nM','method','quality','doi','pmid','source_url','provenance_group','metadata_json']
def clean(v):
 if v is None:return ''
 if isinstance(v,float) and math.isnan(v):return ''
 return str(v).strip()
def dump(x):return json.dumps(x,ensure_ascii=False,default=str)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
@lru_cache(maxsize=150000)
def key(smiles):
 m=Chem.MolFromSmiles(clean(smiles))
 return Chem.MolToInchiKey(m) if m else ''
def concentration(value,unit,relation=''):
 s=clean(value).replace('−','-').replace('≤','<=').replace('≥','>=')
 m=re.fullmatch(r'\s*([<>~≈=]{0,2})\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*',s)
 if not m:return clean(relation),'',None
 rel=clean(relation) or m[1] or '='
 u=clean(unit).replace('μ','u').replace('µ','u').lower()
 factor={'m':1e9,'mm':1e6,'um':1e3,'nm':1,'pm':.001,'fm':.000001}.get(u)
 n=float(m[2]);return rel,m[2],n*factor if factor is not None and n>0 and math.isfinite(n) else None

def ziprows(path):
 with zipfile.ZipFile(path) as z:
  members=[x for x in z.namelist() if x.endswith(('.tsv','.txt'))]
  with io.TextIOWrapper(z.open(members[0]),encoding='utf-8-sig',errors='strict',newline='') as f:
   reader=csv.reader(f,delimiter='\t');header=next(reader)
   for line,row in enumerate(reader,2):
    yield line,header,row

def extract_bdb(path,blocks):
 cache=OUT/(path.stem+'.project.jsonl.gz');stat=cache.with_suffix('.stats.json')
 if cache.exists() and stat.exists():return cache,json.loads(stat.read_text())
 counts=Counter();selected=0
 with gzip.open(cache,'wt') as out:
  for line,header,row in ziprows(path):
   counts['rows_scanned']+=1
   if len(row)<40:
    counts['short_rows']+=1;continue
   k=row[3].strip()
   if k.split('-')[0] not in blocks:continue
   d={h:v.strip() for h,v in zip(header,row) if v.strip()}
   if len(row)>len(header):counts['overwide_selected_rows']+=1;d['_extra_fields']=row[len(header):]
   d['_line']=line;out.write(dump(d)+'\n');selected+=1
   if counts['rows_scanned']%500000==0:print(path.name,dict(counts),flush=True)
 counts['selected_rows']=selected;counts['source_sha256']=sha(path)
 stat.write_text(dump(counts));print(path.name,dict(counts),flush=True);return cache,dict(counts)

def bdb_identity(d):
 acc=sorted({v for c,v in d.items() if re.match(r'UniProt \((?:SwissProt|TrEMBL)\) Primary ID of Target Chain \d+$',c) and v})
 n=int(float(d.get('Number of Protein Chains in Target (>1 implies a multichain complex)','0')))
 name=d.get('Target Name','');scope='single_accession_construct_unverified'
 if len(acc)!=1 or n>1:scope='complex_or_unresolved'
 elif re.search(r'\bmutant\b|\bmutation\b|\b[A-Z]\d+[A-Z]\b',name):scope='mutant'
 return ';'.join(acc),scope

def bdb_fingerprint(d):
 a,s=bdb_identity(d)
 return hashlib.sha256(dump([d.get('Ligand InChI Key'),a,s,d.get('Target Name'),*[d.get(e+' (nM)','') for e in ['Kd','Ki','IC50','EC50']]]).encode()).hexdigest()

class Writer:
 def __init__(self,c):self.c=c;self.pending=[];self.counts=Counter()
 def add(self,source,rid,ligand_key='',ligand_name='',smiles='',target_accessions='',target_name='',target_scope='unresolved',organism='',endpoint='',value='',unit='',relation='',method='',quality='unreviewed',doi='',pmid='',url='',group='',meta=None):
  rel,_,nm=concentration(value,unit,relation)
  r=[source,str(rid),ligand_key,ligand_name,smiles,target_accessions,target_name,target_scope,organism,{'kd':'Kd','ki':'Ki','ic50':'IC50','ec50':'EC50'}.get(endpoint.lower(),endpoint),rel,clean(value),unit,nm,method,quality,doi,pmid,url,group,dump(meta or {})]
  self.pending.append(r);self.counts[source]+=1
  if len(self.pending)>=5000:self.flush()
 def flush(self):
  if self.pending:self.c.executemany('insert into measurements('+','.join(COLUMNS)+') values ('+','.join('?' for _ in COLUMNS)+')',self.pending);self.c.commit();self.pending=[]

def import_bdb(w,cache):
 for line in gzip.open(cache,'rt'):
  d=json.loads(line);a,s=bdb_identity(d);rid=d['BindingDB Reactant_set_id']
  for endpoint in ['Kd','Ki','IC50','EC50']:
   value=d.get(endpoint+' (nM)','')
   if not value:continue
   meta={k:v for k,v in d.items() if 'Sequence' not in k and 'Secondary ID' not in k and 'Alternative ID' not in k}
   w.add('BindingDB_202609',rid+':'+endpoint,d.get('Ligand InChI Key',''),d.get('BindingDB Ligand Name',''),d.get('Ligand SMILES',''),a,d.get('Target Name',''),s,d.get('Target Source Organism According to Curator or DataSource',''),endpoint,value,'nM',quality='source_reported_unreviewed',doi=d.get('Article DOI',''),pmid=d.get('PMID',''),url=d.get('Link to Ligand-Target Pair in BindingDB',''),group='BindingDB:'+rid,meta=meta)
 w.flush()

def import_hiq(w):
 d=pd.read_csv(SURVEY/'hiqbind_sm_metadata.csv').fillna('')
 for i,r in d.iterrows():
  a=sorted(set(x.strip() for x in r['Protein UniProtID'].split(',') if x.strip()))
  scope='single_accession_construct_unverified' if len(a)==1 else 'complex_or_unresolved'
  k=key(r['Ligand SMILES']);g='HiQBind:'+hashlib.sha256(dump([r['PDBID'],k,a,r['Binding Affinity Measurement'],r['Binding Affinity Value'],r['Binding Affinity Annotation']]).encode()).hexdigest()
  w.add('HiQBind_v3',i+2,k,r['Ligand Name'],r['Ligand SMILES'],';'.join(a),r['Protein UniProtName'],scope,'',r['Binding Affinity Measurement'],r['Binding Affinity Value'],r['Binding Affinity Unit'],r['Binding Affinity Sign'],quality='structure_attached_literature_label',url='https://www.rcsb.org/structure/'+r['PDBID'],group=g,meta=r.to_dict())
 w.flush()

def import_openbind(w):
 d=pd.read_csv(RAW/'OpenBind_all_affinity_data_release_v1.csv').fillna('')
 for i,r in d.iterrows():
  w.add('OpenBind_20260828',i+2,key(r.SMILES),r['OpenBind IDs'],r.SMILES,'','CVA16 2A protease surrogate for EV-A71','viral_construct','Coxsackievirus A16','Kd',r['KD (M)'],'M',method='GCI',quality='author_used_in_analysis' if r['Used in analysis'] is True or str(r['Used in analysis']).lower()=='true' else 'author_not_used_in_analysis',url='https://zenodo.org/records/22142262',group=f'OpenBind:{i+2}',meta=r.to_dict())
 w.flush()

def import_cache(w,targets):
 geneacc=dict(zip(targets.gene_symbol,targets.uniprot_accession))
 for round_ in [1,2]:
  p=RAW/f'CACHE_{round_}-All_data_CACHE3_Round{round_}.xlsx'
  d=pd.read_excel(p,f'Round {round_}_SPR data').fillna('')
  for i,r in d.iterrows():
   t=clean(r.get('Target ID'));a=geneacc.get('PARP14','') if t=='PARP14A' else ''
   classification=clean(r.get('Classification'))
   w.add('CACHE3',f'R{round_}:SPR:{i+2}',key(r['Smile']),r['Common Name'],r['Smile'],a,t,'domain_specific' if a else 'viral_or_unresolved_construct','Homo sapiens' if a else '', 'Kd',r['KD (M)'],'M',method='SPR',quality='author_fit_flags:'+classification if classification else 'fit_without_author_classification',url='https://cache-challenge.org/results-cache-challenge-3',group=f'CACHE3:R{round_}:SPR:{i+2}',meta=r.to_dict())
 d=pd.read_excel(RAW/'CACHE_all_data_with_structures_vs2.xlsx',header=1).fillna('')
 for i,r in d.iterrows():
  if not clean(r.get('Smiles')):continue
  t=clean(r.get('Target ID'));a=geneacc.get(t,'');raw=r.get('curated KD M') or r.get('KD (M)')
  w.add('CACHE1',i+3,key(r['Smiles']),clean(r['CACHE ID']),r['Smiles'],a,t+' WDR' if t=='LRRK2' else t+' PWWP' if t=='NSD2' else t,'domain_specific','Homo sapiens','Kd' if clean(raw) else 'SPR_single_concentration_response',raw if clean(raw) else r.get('Adjusted relative Run 1'),'M' if clean(raw) else 'relative_response',method='SPR',quality='requires_QC_review:'+clean(r.get('curation comment')),url='https://cache-challenge.org/results-cache-challenge-1',group=f'CACHE1:{i+3}',meta=r.to_dict())
 w.flush()

def import_super(w,targets):
 geneacc=dict(zip(targets.gene_symbol,targets.uniprot_accession))
 p=SURVEY/'SUPERCHARGE_Supplementary_Data6.xlsx'
 for s in ['5CL_SI-3_fit','5CL_TAK285_fit','2CL_TAK285_fit']:
  d=pd.read_excel(p,s).fillna('');drug='TAK-285' if 'TAK285' in s else 'Src Inhibitor 3'
  k='ZYQXEVJIFYIBHZ-UHFFFAOYSA-N' if drug=='TAK-285' else ''
  for i,r in d.iterrows():
   w.add('Supercharge_Data6',s+':'+str(i+2),k,drug,'',geneacc.get(r['Name(s)'],''),r['Name(s)'],'lysate_protein_or_group','Homo sapiens','Kd_app',r['apparent Kd'],'nM',method='lysate_kinobead_competition',quality='author_potential_target' if str(r['potTarget']) in ['1','1.0'] else 'not_author_potential_target',doi='10.1038/s41586-025-09763-9',url='https://www.nature.com/articles/s41586-025-09763-9',group='Supercharge:'+s+':'+str(i+2),meta=r.to_dict())
 w.flush()

def import_chembl(w,drugs,targets):
 """Existing ChEMBL37 comparator, exact registered entities and direct target IDs."""
 c=sqlite3.connect(f'file:{CHEMBL}?mode=ro',uri=True);c.row_factory=sqlite3.Row
 tids={r['chembl_id']:r['tid'] for r in c.execute('select tid,chembl_id from target_dictionary')};tm=targets.set_index('target_chembl_id')
 query='''select ac.*,a.description,a.assay_type,a.confidence_score,a.relationship_type,a.variant_id,a.assay_organism,td.chembl_id target_id,td.target_type,td.organism,td.pref_name target_name,a.chembl_id assay_chembl_id,cs.standard_inchi_key,cs.canonical_smiles,md.pref_name drug_name,d.doi,d.pubmed_id from activities ac join assays a using(assay_id) join target_dictionary td on td.tid=a.tid join compound_structures cs on cs.molregno=ac.molregno join molecule_dictionary md on md.molregno=ac.molregno left join docs d on d.doc_id=ac.doc_id where ac.molregno=? and ac.standard_type in ('Kd','Ki','IC50','EC50')'''
 for k in drugs.ligand_inchikey:
  for mol, in c.execute('select molregno from compound_structures where standard_inchi_key=?',(k,)):
   for row in c.execute(query,(mol,)):
    r=dict(row);t=r['target_id']
    if t not in tm.index:continue
    scope='single_accession_construct_unverified' if r['target_type']=='SINGLE PROTEIN' and r['confidence_score']==9 and r['relationship_type']=='D' and not r['variant_id'] else 'annotation_or_variant_review'
    w.add('ChEMBL37_existing',r['activity_id'],k,clean(r['drug_name']),r['canonical_smiles'],clean(tm.loc[t,'uniprot_accession']),r['target_name'],scope,clean(r['organism']),r['standard_type'],r['standard_value'],clean(r['standard_units']),clean(r['standard_relation']),method=clean(r['assay_type']),quality='validity_flag:'+clean(r['data_validity_comment']) if r['data_validity_comment'] else 'source_reported_unreviewed',doi=clean(r['doi']),pmid=clean(r['pubmed_id']),url='https://www.ebi.ac.uk/chembl/assay_report_card/'+r['assay_chembl_id']+'/',group='ChEMBL37:'+str(r['activity_id']),meta=r)
 c.close();w.flush()

def main():
 OUT.mkdir(parents=True,exist_ok=True);DB.parent.mkdir(parents=True,exist_ok=True)
 baseline_sha=sha(BASE);drugs=pd.read_csv(DRUG).fillna('');targets=pd.read_csv(TARGET).fillna('');base=pd.read_csv(BASE).fillna('')
 blocks=set(k.split('-')[0] for k in drugs.ligand_inchikey)
 controls=pd.read_csv(ROOT/'outputs/joint384_comprehensive_20260909/REFERENCE_CONTROLS_EXTRA.csv').fillna('')
 for col in controls:
  if 'inchikey' in col:blocks.update(k.split('-')[0] for k in controls[col] if k)
 stats={};caches={}
 for version in ['202607','202609']:
  for part in ['BindingDB_Articles','PubChem']:
   path=(OLD if version=='202607' else RAW)/f'BindingDB_{part}_{version}_tsv.zip'
   caches[version,part],stats[version+'_'+part]=extract_bdb(path,blocks)
 allcache,stats['202609_All']=extract_bdb(RAW/'BindingDB_All_202609_tsv.zip',blocks)
 supplemental=[]
 for part in ['BindingDB_Articles','PubChem','ChEMBL','Patents','PDSPKi']:
  path=RAW/f'BindingDB_{part}_202609_tsv.zip'
  if path.exists():
   cache,stats['202609_'+part]=extract_bdb(path,blocks);supplemental.append((part,cache))
 # Preserve source-specific discrepancies while avoiding re-importing identical rows.
 union=OUT/'BindingDB_202609_union.project.jsonl.gz';seen=set();ids=set();added=Counter();conflicts=Counter()
 with gzip.open(union,'wt') as f:
  for part,cache in [('All',allcache)]+supplemental:
   for line in gzip.open(cache,'rt'):
    d=json.loads(line);rid=d['BindingDB Reactant_set_id'];fp=(rid,bdb_fingerprint(d))
    if fp in seen:continue
    if rid in ids:conflicts[part]+=1
    seen.add(fp);ids.add(rid);d['_archive_subset']=part;f.write(dump(d)+'\n');added[part]+=1
 stats['union']={'unique_record_cores':len(seen),'rows_contributed_by_archive':dict(added),'same_id_different_core':dict(conflicts)}
 allcache=union
 temp=DB.with_suffix('.building.sqlite')
 if temp.exists():temp.unlink()
 c=sqlite3.connect(temp);c.execute('pragma journal_mode=OFF');c.execute('pragma synchronous=OFF')
 schema=','.join(f'{col} '+('REAL' if col=='value_nM' else 'TEXT') for col in COLUMNS)
 c.execute('create table measurements (id integer primary key,'+schema+')');w=Writer(c)
 for name,fn,args in [('BindingDB',import_bdb,(allcache,)),('HiQBind',import_hiq,()),('OpenBind',import_openbind,()),('CACHE',import_cache,(targets,)),('Supercharge',import_super,(targets,)),('ChEMBL37',import_chembl,(drugs,targets))]:
  print('IMPORT',name,flush=True);fn(w,*args);print(dict(w.counts),flush=True)
 c.execute('create index evidence_pair on measurements(ligand_key,target_accessions)');c.execute('create index evidence_source on measurements(source,record_id)')
 drugs[['ligand_inchikey','drug_names','ligand_smiles']].to_sql('project_drugs',c,index=False,if_exists='replace')
 targets[['target_chembl_id','uniprot_accession','gene_symbol','target_name','assay_lane']].to_sql('project_targets',c,index=False,if_exists='replace')
 # Only single-accession, non-mutant, exact full-key records enter conservative coverage.
 c.execute('''create view exact_project_evidence as select m.*,t.target_chembl_id,t.gene_symbol from measurements m join project_drugs d on d.ligand_inchikey=m.ligand_key join project_targets t on t.uniprot_accession=m.target_accessions where m.target_scope='single_accession_construct_unverified' and (m.organism in ('','Homo sapiens','Human'))''')
 c.execute('''create view comparable_numeric_evidence as select * from exact_project_evidence where value_nM is not null and endpoint in ('Kd','Ki','IC50','EC50') and quality not like 'validity_flag:%' ''')
 qcpath=RAW/'SOURCE_RECORD_QC_OVERRIDES_20260910.csv'
 if qcpath.exists():
  pd.read_csv(qcpath,dtype=str).to_sql('evidence_quality_reviews',c,if_exists='replace',index=False)
  c.execute("create view reviewed_comparable_numeric_evidence as select e.* from comparable_numeric_evidence e where not exists (select 1 from evidence_quality_reviews q where q.source=e.source and q.record_id=e.record_id)")
 c.commit();assert c.execute('pragma integrity_check').fetchone()[0]=='ok';c.close();temp.replace(DB)
 (OUT/'BUILD_MANIFEST.json').write_text(dump(dict(baseline_sha256=baseline_sha,database=str(DB.relative_to(ROOT)),source_scan_stats=stats,imported_measurement_rows=dict(w.counts),note='Rows are source observations, not independent experiments; exact identity is not construct or assay validation.')))
 assert sha(BASE)==baseline_sha
 print('DONE',DB,flush=True)
if __name__=='__main__':main()

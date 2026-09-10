"""Recheck all 384 pairs against unfiltered ChEMBL activity endpoints, read only."""
import sqlite3,json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/best_candidate_synthesis_20260909';DB=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db';SOURCE=ROOT/'outputs/joint384_design_20260909/CANDIDATES_384.csv'
def main():
 p=pd.read_csv(SOURCE);c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
 c.execute('create temp table project_pairs(pair_id text,ligand_inchikey text,target_chembl_id text)');c.executemany('insert into project_pairs values(?,?,?)',p[['pair_id','ligand_inchikey','target_chembl_id']].itertuples(index=False,name=None))
 c.execute('create temp table drug_mols(ligand_inchikey text,molregno integer,entity_scope text)');maps=[]
 for key in p.ligand_inchikey.unique():
  direct=[x[0] for x in c.execute('select molregno from compound_structures where standard_inchi_key=?',(key,))]
  mids={m:'EXACT_FULL_INCHIKEY' for m in direct}
  for m in direct:
   for (child,) in c.execute('select molregno from molecule_hierarchy where parent_molregno=?',(m,)):mids.setdefault(child,'SALT_OR_FORM_WITH_EXACT_STRUCTURE_PARENT')
  maps.extend((key,m,scope) for m,scope in mids.items())
 c.executemany('insert into drug_mols values(?,?,?)',maps);c.execute('create index temp.dm_mol on drug_mols(molregno)')
 q='''SELECT p.pair_id,p.ligand_inchikey,p.target_chembl_id,m.entity_scope,md.chembl_id molecule_chembl_id,
 a.activity_id,a.standard_type,a.standard_relation,a.standard_value,a.standard_units,a.pchembl_value,
 a.activity_comment,a.data_validity_comment,a.potential_duplicate,s.chembl_id assay_chembl_id,s.description assay_description,
 s.assay_type,s.assay_organism,s.assay_cell_type,s.assay_subcellular_fraction,s.confidence_score,s.relationship_type,
 d.chembl_id doc_chembl_id,d.title doc_title,d.doi,d.pubmed_id,d.year
 FROM drug_mols m JOIN project_pairs p ON p.ligand_inchikey=m.ligand_inchikey
 JOIN target_dictionary t ON t.chembl_id=p.target_chembl_id
 JOIN activities a ON a.molregno=m.molregno
 JOIN assays s ON s.assay_id=a.assay_id AND s.tid=t.tid
 JOIN molecule_dictionary md ON md.molregno=m.molregno
 LEFT JOIN docs d ON d.doc_id=a.doc_id'''
 rows=pd.read_sql_query(q,c);rows=rows.merge(p[['pair_id','drug_names','gene_symbol']],on='pair_id',validate='many_to_one')
 rows['source_url']='https://www.ebi.ac.uk/chembl/explore/activity/'+rows.activity_id.astype(str)
 rows.to_csv(OUT/'ALL384_RAW_CHEMBL_ACTIVITY_RECORDS.csv.gz',index=False)
 audit=p[['pair_id','drug_names','gene_symbol']].copy();counts=rows.groupby('pair_id').size();audit['raw_activity_rows']=audit.pair_id.map(counts).fillna(0).astype(int)
 strong=set(rows.loc[rows.confidence_score.ge(9)&rows.relationship_type.eq('D'),'pair_id'])
 audit['raw_high_confidence_direct_target_annotation']=audit.pair_id.isin(strong)
 audit['novelty_reaudit_status']=audit.raw_activity_rows.map(lambda n:'PRIOR_ACTIVITY_PRESENT_NOT_NEW_UNTESTED_PAIR' if n else 'NO_EXACT_TARGET_RECORD_IN_LOCAL_CHEMBL37_NOT_PROOF_OF_NOVELTY')
 audit.to_csv(OUT/'ALL384_RAW_ACTIVITY_NOVELTY_REAUDIT.csv',index=False)
 info={'source384_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'database_path':str(DB),'database_bytes':DB.stat().st_size,'database_mtime_ns':DB.stat().st_mtime_ns,'database_version':'ChEMBL37','database_modified':False,'sql':q,'candidate_pairs':len(audit),'pairs_with_raw_activities':int(audit.raw_activity_rows.gt(0).sum()),'pairs_high_confidence_direct_target_annotation':len(strong),'activity_records':len(rows),'meaning':'Prior endpoint records include percent inhibition, cell/function and negative records; do not interpret as binding-positive or direct measured affinity.'}
 (OUT/'RAW_ACTIVITY_AUDIT.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k!='sql'},indent=2),flush=True)
if __name__=='__main__':main()

"""Exact full-InChIKey overlap with public paper supplement; not pair binding validation."""
from pathlib import Path
import pandas as pd,json
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'outputs/proteintalks_relevance_20260910';T=P/'tables'
x=pd.read_csv(ROOT/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv');reg=pd.read_csv(ROOT/'outputs/biomaster_disease_evidence_720x888_20260909/DRUG_REGISTRY_720.csv')
def key(s):
 try:
  m=Chem.MolFromSmiles(str(s));return Chem.MolToInchiKey(m) if m else ''
 except Exception:return ''
a=pd.read_excel(T/'TableS8_PTPC_info_0703.xlsx',sheet_name='B_drug_info');meta=pd.read_excel(T/'TableS8_PTPC_info_0703.xlsx',sheet_name='A_metadata');a['full_inchikey']=a.SMILES.map(key)
a['has_experiment_metadata']=a.Pert_ID.astype(str).isin(meta.Pert_ID.astype(str));valid=a[a.full_inchikey.ne('')&a.has_experiment_metadata]
b=x.merge(valid[['Pert_ID','Compound_name','full_inchikey']],left_on='ligand_inchikey',right_on='full_inchikey',how='inner')
m=meta.groupby('Pert_ID').agg(cell_lines=('Cell_line',lambda z:'; '.join(sorted(set(z.astype(str))))),time_hours=('Pert_time_(hrs)',lambda z:'; '.join(sorted(set(z.astype(str))))),metadata_rows=('Sample_ID','size')).reset_index();m.Pert_ID=m.Pert_ID.astype(str);b.Pert_ID=b.Pert_ID.astype(str);b=b.merge(m,on='Pert_ID',how='left')
b[['candidate_id','modeled_entity_name','gene_symbol','ligand_inchikey','target_chembl_id','Pert_ID','Compound_name','cell_lines','time_hours','metadata_rows']].to_csv(P/'SPR384_PTPC_DRUG_OVERLAP.csv',index=False,encoding='utf-8-sig')
head=pd.read_excel(T/'TableS8_PTPC_info_0703.xlsx',sheet_name='C_protein_Matrix',nrows=0).columns
measured={str(k).split('_')[0] for k in head if '_' in str(k)}
g=x[['gene_symbol','uniprot_accession','target_chembl_id']].drop_duplicates().copy();g['in_PTPC_protein_matrix_header']=g.uniprot_accession.isin(measured);g.to_csv(P/'SPR112_PTPC_PROTEIN_COVERAGE.csv',index=False,encoding='utf-8-sig')
s={'PTPC_annotations':len(a),'PTPC_unique_names':a.Compound_name.nunique(),'PTPC_valid_full_keys':a.full_inchikey.ne('').sum(),'PTPC_metadata_matched_annotations':len(valid),'SPR384_matching_drugs':b.ligand_inchikey.nunique(),'SPR384_pairs_with_matching_drug':b.pair_id.nunique(),'registry720_matching_drugs':len(set(reg.ligand_inchikey)&set(valid.full_inchikey)),'SPR112_targets_in_protein_header':int(g.in_PTPC_protein_matrix_header.sum()),'scope':'Exact full InChIKey from supplied SMILES, no salt stripping or stereo relaxation. Drug perturbation overlap is not drug-target binding evidence; header presence is not detection in every sample.'}
(P/'OVERLAP_SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,default=int));print(json.dumps(s,ensure_ascii=False,default=int));print(b[['modeled_entity_name','gene_symbol']].drop_duplicates().head(15).to_string(index=False))

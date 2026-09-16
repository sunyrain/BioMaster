#!/usr/bin/env python3
"""Freeze the existing website's full 720x384 directory; import exact prior scores."""
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_data import PAIRS,TARGETS
from biomaster.catalog_models import DIRECTORY,connect,save
OUT=ROOT/DIRECTORY

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_csv(ROOT/PAIRS,usecols=['ligand_inchikey','target_chembl_id'])
    drugs=pd.read_csv(ROOT/'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv')
    drugs=drugs[drugs.drug_id.isin(pairs.ligand_inchikey)].sort_values('drug_id')
    targets=pd.read_csv(ROOT/TARGETS,low_memory=False).rename(columns={'target_chembl_id':'target_id','uniprot_accession':'uniprot_id','gene_symbol':'gene'})
    targets=targets[targets.target_id.isin(pairs.target_chembl_id)].copy()
    assert len(drugs)==720 and len(targets)==384 and len(pairs)==len(drugs)*len(targets)
    for r in drugs.itertuples():assert Chem.MolToInchiKey(Chem.MolFromSmiles(r.smiles))==r.drug_id
    targets['protein_length']=targets.sequence.str.len()
    targets['protein_sha256']=targets.sequence.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    targets['priority']=targets.gene.ne('AR').astype(int)
    targets=targets.sort_values(['priority','protein_length','target_id'])
    drugs[['drug_id','name','smiles']].to_csv(OUT/'DRUGS.csv',index=False)
    targets[['target_id','uniprot_id','gene','sequence','protein_length','protein_sha256']].to_csv(OUT/'TARGETS.csv',index=False)
    db=connect(ROOT,write=True)
    previous=ROOT/'outputs/frontier_dti_20260916'
    identities=pd.read_csv(previous/'UNIQUE_PAIRS.csv').set_index('pair_id')
    records=[]
    for model in ['probematch','dtbind']:
        for r in pd.read_csv(previous/model/'PREDICTIONS.csv').to_dict('records'):
            if r['status']!='completed':continue
            old=identities.loc[r['pair_id']];t=targets[targets.target_id.eq(old.target_id)]
            if len(t)!=1 or t.iloc[0].sequence!=old.sequence:continue
            if drugs.set_index('drug_id').loc[old.drug_id,'smiles']!=old.smiles:continue
            records.append(dict(model=model,drug_id=old.drug_id,target_id=old.target_id,status='completed',score=r['score']))
    for r in identities.itertuples():
        p=previous/'nesso/predictions'/r.Index/'affinity.json'
        if p.exists():
            score=json.loads(p.read_text()).get('affinity_probability_binary')
            if score is not None:records.append(dict(model='nesso',drug_id=r.drug_id,target_id=r.target_id,status='completed',score=float(score)))
    save(db,records);db.close()
    protocol=dict(drugs=720,targets=384,pairs=276480,priority='AR all 720 drugs, then complete targets ordered by length; no SPR-only filtering',catalog_source=PAIRS,
      input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'DRUGS.csv',OUT/'TARGETS.csv']},
      ranks='per entity over the full directory; missing scores remain null; partial score coverage explicitly displayed',
      biomaster_direction='drug query uses forward logit; target query uses reverse logit, unchanged from original rankings API',
      prior_scores_imported=len(records),weights='same pinned official checkpoints and preprocessing as frontier_dti_20260916; no training')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print(json.dumps(protocol,ensure_ascii=False))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Conservative historical old-drug sensitivity roster; no activity labels read."""
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_biomaster_odti_v4_features import sha256,write_json
OUT=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'


def main():
    if (OUT/'TEST_RELEASE.json').exists():raise RuntimeError('freeze this sensitivity before test release')
    old=pd.read_csv(OUT/'OLD_DRUG_INDEX.csv')
    db=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
    con=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    con.execute('CREATE TEMP TABLE old_inchi(inchi TEXT PRIMARY KEY)')
    con.executemany('INSERT INTO old_inchi VALUES (?)',[(str(s),) for s in old.ligand_inchikey])
    sql='''SELECT c.standard_inchi_key, m.chembl_id, m.first_approval, m.max_phase
           FROM old_inchi o JOIN compound_structures c ON c.standard_inchi_key=o.inchi
           JOIN molecule_dictionary m ON m.molregno=c.molregno'''
    matches=pd.read_sql_query(sql,con);con.close()
    dates=matches.groupby('standard_inchi_key').first_approval.min()
    old['chembl_first_approval']=old.ligand_inchikey.map(dates)
    old['confirmed_approved_by_2022']=old.chembl_first_approval.le(2022)
    path=OUT/'REGISTRY_APPROVAL_AUDIT.csv';old.to_csv(path,index=False)
    matches.to_csv(OUT/'REGISTRY_APPROVAL_SOURCE.csv',index=False)
    write_json(OUT/'REGISTRY_APPROVAL_PROTOCOL.json',{'status':'FROZEN_BEFORE_TEST','cutoff':2022,
        'method':'exact full InChIKey match to ChEMBL37 molecule_dictionary.first_approval; missing dates excluded from conservative sensitivity',
        'no_activity_labels_read':True,'primary_scope_unchanged':'current 720 registry',
        'sensitivity':'both query drugs and target-to-drug candidate drugs restricted to confirmed approval <=2022',
        'drugs':len(old),'exact_matches':int(old.ligand_inchikey.isin(matches.standard_inchi_key).sum()),
        'dated_drugs':int(old.chembl_first_approval.notna().sum()),'confirmed_approved_by_2022':int(old.confirmed_approved_by_2022.sum()),
        'approved_after_2022':int(old.chembl_first_approval.gt(2022).sum()),
        'files':{path.name:sha256(path),'REGISTRY_APPROVAL_SOURCE.csv':sha256(OUT/'REGISTRY_APPROVAL_SOURCE.csv')},
        'source_sha256':sha256(Path(__file__))})
    print(json.dumps({'dated':int(old.chembl_first_approval.notna().sum()),'approved_by_2022':int(old.confirmed_approved_by_2022.sum()),'after_2022':int(old.chembl_first_approval.gt(2022).sum())}),flush=True)


if __name__=='__main__':main()

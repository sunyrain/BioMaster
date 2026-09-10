#!/usr/bin/env python3
"""Extract annual raw-activity aggregates before assigning any temporal label."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_biomaster_odti_v4_features import sha256,write_json

OUT=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
DB=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
OLD=ROOT/'outputs/biomaster_old_drug_bidirectional_20260906'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'RAW_MANIFEST.json').exists():raise FileExistsError('raw extract is frozen')
    connection=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
    connection.execute('PRAGMA temp_store=MEMORY')
    targets=pd.read_csv(OLD/'TARGETS_384.csv.gz')
    mapping=[]
    for row in targets.itertuples():
        db=connection.execute('SELECT tid,target_type,tax_id FROM target_dictionary WHERE chembl_id=?',(row.target_chembl_id,)).fetchone()
        assert db is not None and db[1]=='SINGLE PROTEIN' and db[2]==9606,(row.target_chembl_id,db)
        mapping.append((int(row.target_feature_index),int(db[0])))
    connection.execute('CREATE TEMP TABLE target_scope (target_index INTEGER,tid INTEGER PRIMARY KEY)')
    connection.executemany('INSERT INTO target_scope VALUES (?,?)',mapping)
    sql="""
    SELECT t.target_index, COALESCE(mh.parent_molregno,a.molregno) AS parent_molregno,
           d.year AS document_year, COUNT(*) AS activity_rows,
           COUNT(DISTINCT COALESCE(a.doc_id,ass.doc_id)) AS document_count,
           COUNT(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS numeric_rows,
           SUM(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS pchembl_sum,
           MIN(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS min_pchembl,
           MAX(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS max_pchembl,
           MAX(CASE WHEN LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%inactive%'
                         OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%not active%'
                         OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%no activity%'
                    THEN 1 ELSE 0 END) AS any_explicit_inactive
    FROM target_scope t JOIN assays ass ON ass.tid=t.tid JOIN activities a ON a.assay_id=ass.assay_id
    LEFT JOIN molecule_hierarchy mh ON mh.molregno=a.molregno
    LEFT JOIN docs d ON d.doc_id=COALESCE(a.doc_id,ass.doc_id)
    WHERE ass.assay_type='B' AND ass.confidence_score>=9 AND ass.relationship_type='D' AND ass.variant_id IS NULL
      AND COALESCE(a.potential_duplicate,0)=0 AND COALESCE(a.data_validity_comment,'') IN ('','Manually validated')
      AND ((a.pchembl_value IS NOT NULL AND a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=')
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%inactive%'
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%not active%'
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%no activity%')
    GROUP BY t.target_index,COALESCE(mh.parent_molregno,a.molregno),d.year
    """
    (OUT/'EXTRACTION.sql').write_text(sql)
    started=time.monotonic()
    print(json.dumps({'stage':'raw_annual_query','targets':len(targets)}),flush=True)
    frame=pd.read_sql_query(sql,connection)
    frame.to_csv(OUT/'ANNUAL_RAW.csv.gz',index=False)
    connection.execute('CREATE TEMP TABLE parent_scope (parent_molregno INTEGER PRIMARY KEY)')
    connection.executemany('INSERT INTO parent_scope VALUES (?)',[(int(x),) for x in frame.parent_molregno.unique()])
    molecules=pd.read_sql_query('SELECT p.parent_molregno,c.canonical_smiles,c.standard_inchi_key,m.chembl_id,m.pref_name FROM parent_scope p LEFT JOIN compound_structures c ON c.molregno=p.parent_molregno LEFT JOIN molecule_dictionary m ON m.molregno=p.parent_molregno',connection)
    molecules.to_csv(OUT/'PARENT_MOLECULES.csv.gz',index=False)
    report={'status':'COMPLETE','annual_rows':len(frame),'activity_rows':int(frame.activity_rows.sum()),
        'parent_molecules':len(molecules),'missing_year_activity_rows':int(frame.loc[frame.document_year.isna(),'activity_rows'].sum()),
        'seconds':time.monotonic()-started,'database':str(DB.relative_to(ROOT)),'database_size':DB.stat().st_size,
        'database_mtime_ns':DB.stat().st_mtime_ns,'sql_sha256':sha256(OUT/'EXTRACTION.sql'),
        'label_rules':'numeric equality Ki/Kd/IC50; B, confidence9, direct human single protein, wildtype; explicit inactivity separate',
        'files':{str(p.relative_to(ROOT)):sha256(p) for p in [OUT/'ANNUAL_RAW.csv.gz',OUT/'PARENT_MOLECULES.csv.gz',OLD/'TARGETS_384.csv.gz']}}
    write_json(OUT/'RAW_MANIFEST.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

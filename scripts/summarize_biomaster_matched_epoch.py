#!/usr/bin/env python3
"""Post-selection comparison at the same sixth training epoch for every global model."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from select_biomaster_global_parent import global_runs,OUT
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json


def main():
    if not (OUT/'GLOBAL_PARENT_SELECTION.json').exists():raise ValueError('this is a post-selection exposure diagnostic')
    records=[];inputs=[]
    for run in global_runs():
        path=run['folder']/'HISTORY.json';history=json.loads(path.read_text())
        row=next(r for r in history if r['epoch']==6);m=row['validation']
        records.append(dict(model=run['name'],cutoff=run['cutoff'],seed=run['seed'],selection=m['selection'],
            d2t_ap=m['d2t']['ap'],t2d_ap=m['t2d']['ap'],coverage_rows=row['coverage_rows'],exposure_sha256=row['exposure_sha256']))
        inputs.append(file_identity(path))
    table=pd.DataFrame(records)
    for _,group in table.groupby(['seed','cutoff']):
        assert len(group)==9 and group.exposure_sha256.nunique()==1 and group.coverage_rows.nunique()==1
    assert len(table)==54
    table.to_csv(OUT/'MATCHED_EPOCH_6_RUNS.csv',index=False)
    summary=table.groupby('model')[['selection','d2t_ap','t2d_ap']].mean()
    summary.to_csv(OUT/'MATCHED_EPOCH_6_SUMMARY.csv')
    write_json(OUT/'MATCHED_EPOCH_6_AUDIT.json',dict(status='PASS',runs=len(table),epoch=6,
        identical_observed_and_retrieval_exposure_per_seed_window=True,
        extra_query_or_assay_supervision='varies only in the declared objective ablations',
        used_to_reselect_global_model=False,global_selection=file_identity(OUT/'GLOBAL_PARENT_SELECTION.json'),
        producer=file_identity(Path(__file__)),inputs=inputs))
    print(summary.to_string())


if __name__=='__main__':main()

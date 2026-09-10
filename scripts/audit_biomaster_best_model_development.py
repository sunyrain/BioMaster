#!/usr/bin/env python3
"""Audit frozen baselines, date boundaries and supplementary training data."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json

OUT=ROOT/'outputs/biomaster_best_model_20260906'


def verify(record):
    actual=file_identity(Path(record['path']))
    if actual['sha256']!=record['sha256']:raise ValueError(f"artifact drift: {record['path']}")


def main():
    first=json.loads((OUTPUT.parent/'RESULT_MANIFEST.json').read_text())
    for record in [*first['files'].values(),first['report']]:verify(record)
    original=json.loads((OUTPUT.parent/'development/global/seed_20260921/RESULT.json').read_text())['identity']
    for record in original['sources']+[original['config']]:verify(record)
    reproduced=[]
    for file in sorted((OUT/'reproduction').glob('*/[gc]*/seed_*/RESULT.json')):
        r=json.loads(file.read_text())
        if r['identity']!=original:raise ValueError('reproduction did not use identical source and configuration')
        verify(r['checkpoint'])
        if r['stage']=='final':
            development=json.loads((OUT/'reproduction/development'/r['variant']/f"seed_{r['seed']}"/'RESULT.json').read_text())
            if r['selected_epoch']!=development['selected_epoch']:raise ValueError('refit epoch differs from development selection')
        reproduced.append(file_identity(file))
    if len(reproduced)!=8:raise ValueError('baseline reproduction incomplete')
    data=OUT/'data';old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    old_map=old.set_index('drug_feature_index').old_drug_index
    first_obs=pd.read_csv(SOURCE/'FIRST_OBSERVATIONS.csv.gz')
    required=np.union1d(np.load(OUTPUT/'REQUIRED_MOLECULE_IDS.npy'),np.load(data/'SUPPLEMENTAL_MOLECULE_IDS.npy'))
    rolls=[]
    for cutoff in [2018,2020]:
        folder=data/f'roll_{cutoff}';tr=pd.read_csv(folder/'TRAIN.csv.gz');va=pd.read_csv(folder/'VALIDATION.csv.gz')
        assert tr.max_document_year.max()<=cutoff
        assert va.min_document_year.min()>cutoff and va.max_document_year.max()<=cutoff+2
        assert not va.has_undated.any()
        assert tr.merge(va,on=['drug_feature_index','target_feature_index']).empty
        assert np.isin(tr.drug_feature_index,required).all() and np.isin(va.drug_feature_index,required).all()
        risk=np.load(folder/'RISK.npy');expected=np.ones_like(risk)
        prior=first_obs[(first_obs.first_document_year.le(cutoff)|first_obs.has_undated)&first_obs.drug_feature_index.isin(old_map.index)]
        expected[prior.drug_feature_index.map(old_map).to_numpy(int),prior.target_feature_index.to_numpy(int)]=False
        assert np.array_equal(expected,risk)
        rolls.append(dict(cutoff=cutoff,train_rows=len(tr),validation_rows=len(va),
            prior_or_undated_old_pairs=int((~risk).sum()),all_training_and_validation_features_prepared=True,
            exact_date_risk_mask=True,pair_disjoint=True))
    assays=[]
    for cutoff in [2018,2020,2022]:
        f=pd.read_csv(data/f'ASSAY_CONTRASTS_{cutoff}.csv.gz')
        assert f.max_document_year.max()<=cutoff
        assert f.groupby('assay_group').binary_label.nunique().eq(2).all()
        for key in ['assay_id','standard_type','standard_units','target_feature_index']:
            assert f.groupby('assay_group')[key].nunique().eq(1).all()
        assert (f.loc[f.binary_label.eq(1),'min_pchembl']>=6).all()
        assert (f.loc[f.binary_label.eq(0),'max_pchembl']<=5).all()
        assert not f.duplicated(['assay_group','drug_feature_index']).any()
        path=data/f'roll_{cutoff}/TRAIN.csv.gz' if cutoff<2022 else SOURCE/'FINAL_TRAIN.csv.gz'
        tr=pd.read_csv(path,usecols=['drug_feature_index','target_feature_index','binary_label'])
        merged=f.merge(tr,on=['drug_feature_index','target_feature_index','binary_label'])
        assert len(merged)==len(f)
        assays.append(dict(cutoff=cutoff,groups=f.assay_group.nunique(),rows=len(f),
            exact_assay_endpoint_groups=True,replicate_threshold_gap=True,only_cutoff_observed_binary_pairs=True))
    supplement=data/'supplemental_features';ids=np.load(data/'SUPPLEMENTAL_MOLECULE_IDS.npy')
    assert np.array_equal(ids,np.load(supplement/'REQUIRED_MOLECULE_IDS.npy'))
    assert not np.isin(ids,np.load(OUTPUT/'REQUIRED_MOLECULE_IDS.npy')).any()
    assert np.load(supplement/'ATOM_DONE.npy')[ids].all()
    m=json.loads((supplement/'ATOM_MANIFEST.json').read_text())
    for record in [*m['files'].values(),*m['globals'].values()]:verify(record)
    portable=[]
    for path in sorted(OUT.glob('package_smoke_*_VALIDATION.json')):
        r=json.loads(path.read_text());assert r['status']=='PASS';verify(r['bundle']);verify(r['checkpoint'])
        portable.append(file_identity(path))
    report=dict(status='PASS',first_round_immutable_artifacts=len(first['files'])+1,
        exact_original_reproductions=reproduced,rolls=rolls,assay_supervision=assays,
        supplemental_molecules=len(ids),supplemental_features_hashes_verified=True,
        standalone_smoke_packages=portable,model_selection_complete=False,
        producer=file_identity(Path(__file__)))
    write_json(OUT/'DATA_AND_REPRODUCTION_AUDIT.json',report)
    print(json.dumps({k:report[k] for k in ['status','first_round_immutable_artifacts','rolls','assay_supervision','supplemental_molecules','model_selection_complete']}))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Validate a planned run and load aligned feature/label batches; never train on test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import jsonschema

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'outputs/dti_research_preparation_20260920'
CONFIG=ROOT/'configs/dti_reliability_20260920'
DATA=ROOT/'data/research/dti_reliability_20260920_v1'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_members(paths):
    members=pd.concat([pd.read_parquet(ROOT/p) for p in paths.split(';')],ignore_index=True)
    assert members.pair_index.is_unique,'Duplicated pair in training union'
    assert members.binary_label.isin([0,1]).all(),'Only measured, eligible binary labels permitted'
    assert members.loc[members.explicit_inactive,'binary_label'].eq(0).all(),'Inactivity label conflict'
    return members


class FeatureBatches:
    """Read common frozen features using original registry indices, without copying the banks."""
    def __init__(self,members,feature_names=('bermol','esm2_dtiam')):
        self.members=members.reset_index(drop=True)
        registry={r['name']:r for r in json.loads((REPORT/'FEATURE_REGISTRY.json').read_text())}
        self.features={name:np.load(ROOT/registry[name]['path'],mmap_mode='r') for name in feature_names}
        self.registry=registry

    def batch(self,positions):
        rows=self.members.iloc[positions]
        result={'pair_index':rows.pair_index.to_numpy(),'label':rows.binary_label.to_numpy(dtype=np.float32),
                'explicit_inactive':rows.explicit_inactive.to_numpy(dtype=bool)}
        for name,bank in self.features.items():
            key='target_feature_index' if name.startswith('esm2') else 'drug_feature_index'
            values=np.asarray(bank[rows[key].to_numpy()],dtype=np.float32)
            assert np.isfinite(values).all(),name
            result[name]=values
            mask=self.registry[name].get('availability_path')
            if mask:result[name+'_available']=np.load(ROOT/mask,mmap_mode='r')[rows[key].to_numpy()]
            else:assert (~np.all(values==0,axis=1)).all(),name
        return result

    def auxiliary(self,arm):
        indices=set(self.members.pair_index)
        regression=pd.read_parquet(DATA/'REGRESSION_EXACT.parquet')
        regression=regression[regression.pair_index.isin(indices)].copy()
        assay=pd.read_parquet(DATA/'ASSAY_AUXILIARY.parquet')
        assay=assay[assay.pair_index.isin(indices)].copy()
        if arm=='A':
            regression=regression[regression.endpoint.isin(['Kd','Ki'])]
            assay=assay[assay.endpoint.isin(['Kd','Ki'])]
        group_n=assay.groupby('assay_group').pair_index.nunique()
        assay=assay[assay.assay_group.isin(group_n[group_n.ge(5)].index)].copy()
        assert set(regression.pair_index).issubset(indices)
        assert set(assay.pair_index).issubset(indices)
        return regression,assay


def validate_partitions():
    """Independent re-read of all materialized members and split invariants."""
    canonical=pd.read_parquet(DATA/'CANONICAL_PAIRS.parquet').set_index('pair_index')
    partitions=pd.read_parquet(DATA/'PARTITIONS.parquet').set_index('pair_index')
    sources=pd.read_parquet(DATA/'PAIR_DOCUMENT_GROUPS.parquet')
    source_qc=pd.read_parquet(DATA/'PAIR_SOURCE_QC.parquet').set_index('pair_index')
    all_report=[]
    for arm in ['A','B']:
        for view in ['scaffold_replay','source_purged','cold_target','double_cold','temporal_conservative']:
            members={split:read_members(str((DATA/f'{arm}_{view}_{split}.parquet').relative_to(ROOT))) for split in ['train','validation','test']}
            for split,f in members.items():
                assert f.pair_index.isin(canonical.index).all()
                c=canonical.loc[f.pair_index]
                assert np.array_equal(c.drug_feature_index,f.drug_feature_index)
                assert np.array_equal(c.target_feature_index,f.target_feature_index)
                assert partitions.loc[f.pair_index,view].eq(split).all()
                if view=='temporal_conservative':
                    dates=source_qc.loc[f.pair_index]
                    assert not dates.missing_any_year.any()
                    if split=='train':assert dates.last_year.le(2022).all()
                    if split=='validation':assert (dates.first_year.ge(2023)&dates.last_year.le(2023)).all()
                    if split=='test':assert dates.first_year.ge(2024).all()
            for first,second in [('train','validation'),('train','test'),('validation','test')]:
                a,b=members[first].pair_index,members[second].pair_index
                assert set(a).isdisjoint(b)
                if view in ['scaffold_replay','source_purged','double_cold']:
                    assert set(canonical.loc[a].split_group).isdisjoint(canonical.loc[b].split_group)
                if view in ['cold_target','double_cold']:
                    assert set(partitions.loc[a].homology_cluster).isdisjoint(partitions.loc[b].homology_cluster)
                if view=='source_purged':
                    assert not source_qc.loc[a].missing_any_document.any()
                    assert not source_qc.loc[b].missing_any_document.any()
                    assert set(sources.loc[sources.pair_index.isin(a)].document_group).isdisjoint(sources.loc[sources.pair_index.isin(b)].document_group)
            all_report.append(dict(arm=arm,view=view,status='PASS',train_pairs=len(members['train']),
                validation_pairs=len(members['validation']),test_pairs=len(members['test'])))
    return all_report


def preflight(run_id=None,full=False):
    schema=json.loads((CONFIG/'PREDICTION_ROW.schema.json').read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    example=dict(run_id='schema_check',model_version='frozen',pair_id='synthetic_contract_example',
        query_direction='target_to_drug',score=.2,higher_is_better=True,score_semantics='classification_logit',
        status='OK',input_manifest_sha256='a'*64)
    jsonschema.validate(example,schema)
    invalid=[dict(example,score=None),dict(example,status='MISSING_FEATURE'),dict(example,status='MISSING_FEATURE',score=None)]
    for item in invalid:
        try:jsonschema.validate(item,schema)
        except jsonschema.ValidationError:pass
        else:raise AssertionError('Invalid prediction contract accepted')
    jsonschema.validate(dict(example,status='MISSING_FEATURE',score=None,reason='Unavailable'),schema)
    weights=pd.read_csv(REPORT/'WEIGHT_FILES.csv')
    upstream=json.loads((REPORT/'source_metadata/NESSO_HF_WEIGHTS.json').read_text())
    official=next(row['lfs']['oid'] for row in upstream if row['path'].endswith('model.safetensors'))
    assert weights.loc[weights.asset_id.eq('nesso_model.safetensors'),'sha256'].iloc[0]==official
    (REPORT/'CONTRACT_CHECKS.json').write_text(json.dumps(dict(status='PASS',prediction_schema=True,
        invalid_ok_without_score_rejected=True,missing_with_numeric_score_rejected=True,
        missing_without_reason_rejected=True,nesso_official_lfs_matches=True),indent=2)+'\n')
    protocol=json.loads((CONFIG/'PROTOCOL.json').read_text())
    for path,digest in protocol['frozen_artifacts'].items():assert sha(ROOT/path)==digest,path
    manifest=json.loads((REPORT/'DATA_MANIFEST.json').read_text())
    checks=[]
    if full:
        for path,record in manifest['prepared'].items():assert sha(ROOT/path)==record['sha256'],path
        for path,record in manifest['inputs'].items():assert sha(ROOT/path)==record['sha256'],path
        assert sha(ROOT/'scripts/prepare_dti_reliability_data_20260920.py')==manifest['producer_sha256']
        for feature in json.loads((REPORT/'FEATURE_REGISTRY.json').read_text()):
            assert sha(ROOT/feature['path'])==feature['sha256']
            if feature.get('availability_path'):assert sha(ROOT/feature['availability_path'])==feature['availability_sha256']
        checks=validate_partitions()
    runs=pd.read_csv(REPORT/'EXPERIMENT_MATRIX.csv').fillna('')
    for row in runs.itertuples():
        for path in (row.train_members+';'+row.validation_members+';'+row.test_members).split(';'):
            assert path in manifest['prepared'],path
            assert (ROOT/path).is_file(),path
    selected=runs if run_id is None else runs[runs.run_id.eq(run_id)]
    assert len(selected)>0,'Unknown run ID'
    reports=[]
    # One smoke batch per distinct input/member combination, not 87 repeated bank loads.
    for row in selected.drop_duplicates(['train_members','arm','architecture']).itertuples():
        train=read_members(row.train_members);val=read_members(row.validation_members);test=read_members(row.test_members)
        assert set(train.pair_index).isdisjoint(val.pair_index)
        assert set(train.pair_index).isdisjoint(test.pair_index)
        names=protocol['architectures'][row.architecture]['inputs']
        reader=FeatureBatches(train,names)
        batch=reader.batch(np.linspace(0,len(train)-1,min(128,len(train)),dtype=int))
        reports.append(dict(run_id=row.run_id,train_pairs=len(train),validation_pairs=len(val),test_pairs=len(test),
            features={name:list(batch[name].shape) for name in names},status='MEMBERS_AND_BATCH_PASS'))
    # Exercise the auxiliary route once, ensuring no held-out pair enters training losses.
    sample=selected.iloc[0];reader=FeatureBatches(read_members(sample.train_members))
    reg,assay=reader.auxiliary(sample.arm)
    result=dict(status='PASS',run_registry_rows=len(runs),partition_checks=checks,batch_checks=reports,
        training_only_auxiliary_check=dict(run_id=sample.run_id,exact_rows=len(reg),assay_rows_min5=len(assay),
            historical_assay_overlap_rows=int(assay.shares_validation_test_assay.sum())),
        all_data_and_feature_hashes_checked=full,trainer_invoked=False,output_contract='Run registry contains plans, not training outcomes.')
    suffix='PREFLIGHT_FULL.json' if full else 'PREFLIGHT_BATCH.json'
    (REPORT/suffix).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],runs=len(runs),partition_checks=len(checks),batch_checks=len(reports),trainer_invoked=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id');parser.add_argument('--full',action='store_true');args=parser.parse_args()
    preflight(args.run_id,args.full)

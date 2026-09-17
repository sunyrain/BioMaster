#!/usr/bin/env python3
"""Score the complete catalog with validation-selected September DTIAM A; no fit."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','6')
from pathlib import Path
import json
import time
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from autogluon.tabular import TabularPredictor
from dtiam_ab_common_20260912 import ROOT,OUT as TRAIN,DATA,SOURCE,FEATURES,FEATURE_COLUMNS,digest,write_json,now,table,banks,probability_score

OUT=ROOT/'outputs/dtiam_a_catalog_20260917'
STORE=ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1'
CATALOG=ROOT/'outputs/catalog_seven_models_20260916'
RELEASE_ID='dtiam_a_kdki_inactive_20260917'

def state(stage,**extra):
    value=dict(stage=stage,updated_utc=now(),pid=os.getpid(),**extra)
    write_json(OUT/'STATUS.json',value);print(json.dumps(value),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    candidates=[]
    for seed in [20260921,20260922,20260923]:
        run=TRAIN/f'kdki_inactive__seed_{seed}'
        selection=json.loads((run/'SELECTION.json').read_text())
        assert selection['status']=='COMPLETE_FIT_AND_VALIDATION_FROZEN'
        model=selection['choices']['query']
        score=next(r['selection_score'] for r in selection['validation_scores'] if r['model']==model)
        candidates.append(dict(seed=seed,model=model,validation_selection_score=score,selection_sha256=digest(run/'SELECTION.json')))
    selected=sorted(candidates,key=lambda x:(-x['validation_selection_score'],x['seed']))[0]
    run=TRAIN/f"kdki_inactive__seed_{selected['seed']}"
    model=selected['model']
    policy=dict(release_id=RELEASE_ID,created_utc=now(),selection_rule='Highest frozen COMMON_VALIDATION bidirectional macro AP among A query-selected views; smaller seed breaks exact ties. No TEST or catalog labels used.',
                selected=selected,candidates=candidates,training_pairs=337570,explicit_inactive_training_pairs=105368,
                representation='Frozen official BerMol CLS768 + ESM2-t33-650M1280, first1022 residues including EOS',
                score='Raw native positive probability; no test-set recalibration, no multi-seed averaging')
    write_json(OUT/'SELECTION.json',policy)
    state('VERIFYING_FEATURES',selected=selected)
    audit=json.loads((STORE/'DTIAM_DEPLOYMENT_FEATURE_STORE_AUDIT_V1.json').read_text())
    paths={
        'drug_features_sha256':STORE/'DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy',
        'target_features_sha256':STORE/'DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy',
        'drug_index_sha256':STORE/'DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz',
        'target_index_sha256':STORE/'DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz'}
    assert audit['status']=='PASS'
    for key,path in paths.items():assert digest(path)==audit['outputs'][key],key
    d=pd.read_csv(paths['drug_index_sha256']);t=pd.read_csv(paths['target_index_sha256'])
    drugs=pd.read_csv(CATALOG/'DRUGS.csv');targets=pd.read_csv(CATALOG/'TARGETS.csv')
    dm=drugs.merge(d,left_on='drug_id',right_on='ligand_inchikey',validate='one_to_one')
    tm=targets.merge(t,left_on='target_id',right_on='target_chembl_id',validate='one_to_one')
    assert len(dm)==720 and len(tm)==384 and dm.smiles.eq(dm.ligand_smiles).all()
    assert tm.sequence_x.eq(tm.sequence_y).all()
    bank=(np.load(paths['drug_features_sha256'],mmap_mode='r'),np.load(paths['target_features_sha256'],mmap_mode='r'))
    assert bank[0].shape==(720,768) and bank[1].shape==(384,1280)
    assert all(np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all() for x in bank)
    # Exact-text overlaps independently check compatibility with September banks.
    references=[('drug',DATA/'MOLECULES.parquet','smiles','drug_feature_index',dm,'smiles',0,'BERMOL'),
                ('target',DATA/'TARGETS.parquet','sequence','target_feature_index',tm,'sequence_x',1,'ESM2')]
    parity=[]
    for kind,path,key,index,frame,localkey,side,name in references:
        source=pd.read_parquet(path,columns=[key,index]).drop_duplicates(key)
        mapping=frame.merge(source,left_on=localkey,right_on=key,suffixes=('_catalog','_train'))
        done=np.load(FEATURES/f'{name}_DONE.npy',mmap_mode='r')
        mapping=mapping[done[mapping[index+'_train'].to_numpy(int)]]
        original=np.load(FEATURES/f'{name}.npy',mmap_mode='r')
        a=bank[side][mapping[index+'_catalog'].to_numpy(int)]
        b=original[mapping[index+'_train'].to_numpy(int)]
        delta=float(np.max(np.abs(a-b)))
        assert len(mapping)>0 and np.allclose(a,b,atol=3e-5,rtol=3e-5),(kind,delta)
        parity.append(dict(kind=kind,exact_identity_overlaps=len(mapping),max_abs_feature_error=delta))
    weights={str(p.relative_to(run)):digest(p) for p in sorted((run/'predictor').rglob('*')) if p.is_file()}
    write_json(OUT/'FEATURE_CHECK.json',dict(status='PASS',identity_matched_drugs=720,identity_matched_targets=384,
                target_truncated_to_1022=int(targets.protein_length.gt(1022).sum()),parity=parity,
                feature_files={str(p.relative_to(ROOT)):digest(p) for p in paths.values()}))
    state('LOADING_PREDICTOR',selected=selected)
    predictor=TabularPredictor.load(str(run/'predictor'),require_version_match=True)
    predictor.persist(models=[model],with_ancestors=True,max_memory=.3)
    assert list(predictor.feature_metadata_in.get_features())==FEATURE_COLUMNS
    # Reload selected model and reproduce saved validation outputs before catalog use.
    validation=pd.read_parquet(SOURCE/'COMMON_VALIDATION.parquet').drop_duplicates('pair_id').sort_values('pair_id')
    sample=validation.iloc[np.linspace(0,len(validation)-1,32,dtype=int)].copy()
    truth=pd.read_parquet(run/'VALIDATION_PREDICTIONS.parquet').drop_duplicates('pair_id').set_index('pair_id').query_score
    replay=np.asarray(predictor.predict_proba(table(sample,banks()),model=model,as_pandas=False,as_multiclass=False),np.float32)
    delta=float(np.max(np.abs(replay-truth.loc[sample.pair_id].to_numpy())))
    assert delta<2e-6,delta
    write_json(OUT/'REPLAY_CHECK.json',dict(status='PASS',pairs=32,max_abs_difference=delta,selected=selected,source='Frozen validation predictions; no refit'))
    pairs=dm[['drug_id','name','drug_feature_index']].merge(tm[['target_id','gene','target_feature_index']],how='cross').sort_values(['drug_id','target_id']).reset_index(drop=True)
    assert len(pairs)==276480 and not pairs.duplicated(['drug_id','target_id']).any()
    values=np.empty(len(pairs),np.float32);started=time.monotonic()
    for start in range(0,len(pairs),16384):
        part=pairs.iloc[start:start+16384]
        probability=np.asarray(predictor.predict_proba(table(part,bank),model=model,as_pandas=False,as_multiclass=False),np.float32)
        assert len(probability)==len(part) and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
        values[start:start+len(part)]=probability
        completed=start+len(part);elapsed=time.monotonic()-started
        state('SCORING',completed=completed,total=len(pairs),elapsed_seconds=elapsed,eta_seconds=elapsed*(len(pairs)-completed)/completed)
    output=pairs.rename(columns={'drug_id':'ligand_inchikey','target_id':'target_chembl_id','name':'drug_names','gene':'gene_symbol'})
    output['dtiam_probability']=values
    calibration=json.loads((run/'CALIBRATION.json').read_text())['query']
    output['dtiam_validation_calibrated_probability']=expit(calibration['slope']*probability_score(values)+calibration['intercept'])
    output['dtiam_model_version']=RELEASE_ID
    for direction,key in [('drug','ligand_inchikey'),('target','target_chembl_id')]:
        output[f'dtiam_{direction}_rank']=output.groupby(key).dtiam_probability.rank(ascending=False,method='first').astype(int)
    path=OUT/'DTIAM_A_720X384_SCORES.csv.gz';temp=OUT/'DTIAM_A_720X384_SCORES.tmp.gz'
    output.to_csv(temp,index=False);temp.replace(path)
    output[output.dtiam_target_rank.le(10)].to_csv(OUT/'DTIAM_A_TOP10_PER_TARGET.csv',index=False)
    output[output.dtiam_drug_rank.le(10)].to_csv(OUT/'DTIAM_A_TOP10_PER_DRUG.csv',index=False)
    manifest=dict(status='COMPLETE',release_id=RELEASE_ID,completed_utc=now(),rows=len(output),drugs=720,targets=384,
                  scores_sha256=digest(path),selected=selected,selection_rule=policy['selection_rule'],
                  predictor=str((run/'predictor').relative_to(ROOT)),predictor_files_sha256=weights,
                  calibration=calibration,score_column='dtiam_probability',training_pairs=337570,
                  note='Classification score for ranking, not measured affinity or verified wet-lab hit probability.',
                  inference_seconds=time.monotonic()-started)
    write_json(OUT/'MANIFEST.json',manifest)
    state('COMPLETE',completed=len(output),total=len(output),inference_seconds=manifest['inference_seconds'])

if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('FAILED',error=f'{type(exc).__name__}: {exc}')
        raise

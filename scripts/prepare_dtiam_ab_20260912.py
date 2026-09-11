#!/usr/bin/env python3
"""Freeze native DTIAM retraining on exactly the September A/B member pools."""
import importlib.metadata
import json
import platform
import shutil
import numpy as np
import pandas as pd
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from dtiam_ab_common_20260912 import ROOT, OUT, SOURCE, DATA, ARMS, SEEDS, digest, write_json, now


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'PROTOCOL.json'
    if protocol_path.exists():
        manifest=json.loads((OUT/'DATA_MANIFEST.json').read_text())
        for name,sha in {**manifest['inputs'],**manifest['code'],**manifest['frozen_inputs']}.items():
            if digest(ROOT/name)!=sha:
                raise ValueError('Frozen experiment input changed: '+name)
        assert digest(protocol_path)==manifest['protocol_sha256']
        print('Frozen DTIAM protocol verified',flush=True)
        return
    validation=pd.read_parquet(SOURCE/'COMMON_VALIDATION.parquet')
    test=pd.read_parquet(SOURCE/'COMMON_TEST.parquet')
    assert set(validation.pair_id).isdisjoint(test.pair_id)
    assert set(validation.split_group).isdisjoint(test.split_group)
    counts=[];train=[]
    for arm in ARMS:
        f=pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet')
        assert f.pair_id.is_unique and f.binary_label.isin([0,1]).all()
        assert not f.conflict.any() and not f.inactive_positive_conflict.any()
        for v in [validation,test]:
            assert set(f.pair_id).isdisjoint(v.pair_id)
            assert set(f.split_group).isdisjoint(v.split_group)
        counts.append(dict(arm=arm,pairs=len(f),positive=int(f.binary_label.sum()),
            negative=int((f.binary_label==0).sum()),explicit_inactive=int(f.explicit_inactive.sum()),
            molecules=f.molecule_id.nunique(),target_sequences=f.target_id.nunique(),
            train_sha256=digest(SOURCE/f'{arm}_TRAIN.parquet')))
        train.append(f)
    # Feature indices include exactly the required supervised train/validation/test pools.
    drugs=set(np.load(SOURCE/'REQUIRED_DRUG_IDS.npy').tolist())
    targets=set(np.load(SOURCE/'REQUIRED_TARGET_IDS.npy').tolist())
    for f in train+[validation,test]:
        assert set(f.drug_feature_index)<=drugs and set(f.target_feature_index)<=targets
    assert [r['pairs'] for r in counts]==[337570,1116270]
    pd.DataFrame(counts).to_csv(OUT/'ARM_DATA_COUNTS.csv',index=False)
    rows=[]
    for name,f in [('validation',validation),('test',test)]:
        for panel,g in f.groupby('panel'):
            rows.append(dict(split=name,panel=panel,pairs=len(g),positive=int(g.binary_label.sum()),
                negative=int((g.binary_label==0).sum()),molecules=g.molecule_id.nunique(),
                target_sequences=g.target_id.nunique(),random_ap=float(g.binary_label.mean())))
    pd.DataFrame(rows).to_csv(OUT/'PANEL_COUNTS.csv',index=False)
    hyperparameters=get_hyperparameter_config('default')
    write_json(OUT/'AUTOGLUON_HYPERPARAMETERS.json',hyperparameters)
    environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n)
        for n in ['autogluon.tabular','autogluon.core','torch','numpy','pandas','scikit-learn',
                  'lightgbm','catboost','xgboost','fastai']})
    write_json(OUT/'ENVIRONMENT.json',environment)
    protocol=dict(status='FROZEN_BEFORE_SUPERVISED_TRAINING',created_utc=now(),
        arms=counts,seeds=SEEDS,fit_suites=6,representation_dimensions=2048,
        representation='Frozen official BerMol CLS768 + ESM2 t33 650M mean1280 including EOS; first1022 residues',
        implementation='DTIAM official representation and native AutoGluon default; compatible AG1.4/Python3.10, not bitwise AG0.5.2 paper reproduction',
        official_sources=['third_party/sota_dti_2026/DTIAM/code/data_process/extract_feature.py',
                          'third_party/sota_dti_2026/DTIAM/code/training_validation.py'],
        outer_split='Exact frozen scaffold/connectivity train, common validation and common test; no per-target cap or subsampling',
        fit=dict(problem_type='binary',eval_metric='roc_auc',hyperparameters_file='AUTOGLUON_HYPERPARAMETERS.json',
            presets=None,excluded_model_types=[],time_limit=None,num_cpus=20,num_gpus=0,
            memory_limit_GB=68,fit_strategy='sequential',sample_weight=None,
            process_stages='Native core learners, fresh-process FASTAI, then native weighted ensemble over all 11 learners',
            fastai_resource_only_override={'ag.max_memory_usage_ratio':2},
            resource_gate='Full-B native preprocessing + loader must peak <=65 GiB; monitor actual child RSS <=78 GiB',
            learner_random_state='seed',random_initialization='native learner and model seeds; no old supervised weights',
            holdout='AutoGluon native internal holdout from training members only; save exact internal row IDs',
            weighted_ensemble='Fitted only on native internal training holdout; external common validation never supplied to fit'),
        stopping='Native early stopping or native fixed iteration/forest budgets, no overall wallclock cutoff; record actual fitted hyperparameters, do not assert universal convergence',
        completeness='All 11 default binary base learners and WeightedEnsemble must fit; any skipped/failed learner is INCOMPLETE, not a passed suite',
        selection_views=dict(native='AutoGluon internal holdout ROC-AUC winner',
            query='Choose trained base/ensemble by common validation KdKi mean(target macro AP, drug macro AP); tie-break AP then model name'),
        calibration='Same Platt protocol on common validation KdKi, max-F1 validation threshold; input logit(clipped native probability)',
        ranking='Score-based target and drug queries >=10 measured pairs and both classes; report macro AP, P5/P10/P20; query selection does not add a ranking training loss',
        test_gate='No TEST inference until all six suites and both validation selections are frozen; report all seeds and existing BioMaster baselines',
        panels=['AFFINITY_KD_KI','ACTIVITY_IC50','ACTIVITY_EC50','ALL_ENDPOINT_UNION','EXPLICIT_INACTIVE'],
        secondary_scopes='Exact shared strict document/raw-assay flags, target training coverage and SPR112-target subset',
        uncertainty='Paired query bootstrap of three-seed mean query metrics, 2000 resamples, fixed seed; conditional exploratory intervals',
        protected='Production model, website and SPR384/112 handoff remain frozen',
        limitations=['Native DTIAM unweighted AutoGluon differs from BioMaster class-balanced losses and feature encoders.',
                    'A/B mixed endpoint labels remain endpoint-specific measurements, not all physical affinity.',
                    'Retrospective scaffold holdout is not prospective SPR validation or protein homology cold start.',
                    'Frozen public encoder pretraining exposure is not certified.',
                    'Native inner holdout means base learner fitting rows are a subset of the identical training-member pool.'])
    write_json(protocol_path,protocol)
    frozen=json.loads((ROOT/'outputs/biomaster_assay_aware_20260911/DATA_MANIFEST.json').read_text())['frozen_inputs']
    for name,sha in frozen.items():
        assert digest(ROOT/name)==sha
    inputs=[SOURCE/f'{a}_TRAIN.parquet' for a in ARMS]+[SOURCE/n for n in
        ['COMMON_VALIDATION.parquet','COMMON_TEST.parquet','REQUIRED_DRUG_IDS.npy','REQUIRED_TARGET_IDS.npy']]+[
        DATA/'MOLECULES.parquet',DATA/'TARGETS.parquet',OUT/'AUTOGLUON_HYPERPARAMETERS.json',OUT/'ENVIRONMENT.json',
        ROOT/'outputs/biomaster_assay_aware_20260911/STRICT_SOURCE_ASSAY_FLAGS.parquet',
        ROOT/'outputs/biomaster_model_consolidation_20260911/RECOMMENDED_MODELS.json',
        ROOT/'outputs/biomaster_model_consolidation_20260911/MODEL_LEDGER.csv',
        ROOT/'outputs/biomaster_model_decision_audit_20260911/WETLAB112_TARGET_BENCHMARK.csv',
        ROOT/'outputs/biomaster_old_production_comparison_20260911/TEST_PREDICTIONS.parquet',
        ROOT/'outputs/biomaster_endpoint_multitask_20260911/TEST_PREDICTIONS.parquet']
    code=list((ROOT/'scripts').glob('*dtiam_ab*20260912.py'))
    write_json(OUT/'DATA_MANIFEST.json',dict(created_utc=now(),inputs={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        code={str(p.relative_to(ROOT)):digest(p) for p in code},frozen_inputs=frozen,protocol_sha256=digest(protocol_path)))
    print(json.dumps(dict(status='FROZEN',counts=counts,required_drugs=len(drugs),required_targets=len(targets),
                         free_disk_GiB=shutil.disk_usage(OUT).free/2**30)),flush=True)


if __name__=='__main__':
    main()

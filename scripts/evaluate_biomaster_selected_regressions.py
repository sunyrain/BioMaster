#!/usr/bin/env python3
"""Freeze selected temporal predictions, then evaluate already-opened regressions.

Primary inference matches the development bf16 protocol. FP32 is also scored
for the development-selected representative seed, before any regression labels
are read. Seed means below average metrics, never model predictions.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.anchored_interaction import AnchoredFeatureBank
from biomaster.model_registry import infer_family,build_model
from biomaster.odti_pockets_v3 import file_identity
from biomaster.ranking_audit import measured_ranking,risk_set_ranking,query_bootstrap
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json

OUT=ROOT/'outputs/biomaster_best_model_20260906'
REG=OUT/'regression'
SEEDS=[20260921,20260922,20260923]
PRIOR=ROOT/'outputs/biomaster_unified_interaction_20260906'


def selection():
    s=json.loads((OUT/'FINAL_ARCHITECTURE_SELECTION.json').read_text())
    if s['status']!='FROZEN_FINAL_ARCHITECTURE' or s['test_labels_used_for_selection']:
        raise ValueError('a development-only frozen architecture is required')
    if s['global_parent_selection']!=file_identity(OUT/'GLOBAL_PARENT_SELECTION.json'):
        raise ValueError('global selection drift')
    return s


def predict(model,bank,d,t,precision,batch_size=256):
    result=np.empty((len(d),2),np.float32)
    with torch.inference_mode():
        for lo in range(0,len(d),batch_size):
            hi=min(lo+batch_size,len(d))
            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
            with ctx:values=model(bank.batch(d[lo:hi],t[lo:hi],model.cfg.variant))
            result[lo:hi]=values.float().cpu().numpy()
    if not np.isfinite(result).all():raise ValueError('nonfinite predictions')
    return result


def score():
    s=selection();REG.mkdir(exist_ok=True);(REG/'scores').mkdir(exist_ok=True)
    status=json.loads((OUT/'FINAL_REFIT_STATUS.json').read_text())
    if status['status']!='COMPLETE':raise ValueError('all temporal refits must finish first')
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    target=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    pool=pd.read_csv(SOURCE/'QUERY_POOL.csv.gz')
    assert np.array_equal(old.old_drug_index,np.arange(len(old)))
    assert np.array_equal(target.target_feature_index,np.arange(len(target)))
    od=np.repeat(old.drug_feature_index.to_numpy(int),len(target));ot=np.tile(np.arange(len(target)),len(old))
    pd_=pool.drug_feature_index.to_numpy(int);pt=pool.target_feature_index.to_numpy(int)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    jobs=[]
    for name in ['selected','global','capacity']:
        for seed in SEEDS:
            if name=='selected':
                variant='global_refit' if s['selected_variant']=='parent' else s['selected_variant']
                folder=OUT/'final_fit/cutoff_2022'/variant/f'seed_{seed}'
            else:
                base=PRIOR if seed==SEEDS[0] else OUT/'reproduction'
                folder=base/'final'/name/f'seed_{seed}'
            jobs.append((name,seed,folder))
    products=[]
    for name,seed,folder in jobs:
        r=json.loads((folder/'RESULT.json').read_text());checkpoint=folder/'BEST.pt'
        if r['train_max_year']!=2022 or r['test_labels_used'] or r['smoke_only']:
            raise ValueError('not a valid temporal checkpoint')
        if file_identity(checkpoint)!=r['checkpoint']:raise ValueError('checkpoint drift')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['cutoff']!=2022 or state['seed']!=seed:raise ValueError('checkpoint cutoff/seed mismatch')
        if name=='selected':
            if state['identity']['final_selection']!=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'):
                raise ValueError('refit selected a different architecture')
            expected=s['refinement_epochs'] or s['fixed_refit_global_epochs']
            if state['epoch']!=expected:raise ValueError('refit epoch differs from frozen rule')
        identity=dict(checkpoint=r['checkpoint'],result=file_identity(folder/'RESULT.json'),
            selection=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'),producer=file_identity(Path(__file__)),
            source_axes=[file_identity(SOURCE/f) for f in ['OLD_DRUG_INDEX.csv','TARGET_INDEX.csv.gz','QUERY_POOL.csv.gz']],
            feature_manifests=[file_identity(OUTPUT/f) for f in ['ATOM_MANIFEST.json','TARGET_MANIFEST.json']],
            supplement=file_identity(OUT/'data/supplemental_features/ATOM_MANIFEST.json'))
        precision_modes=['bf16','fp32'] if name=='selected' and seed==s['deployment_seed'] else ['bf16']
        model=None;bank=None
        for precision in precision_modes:
            key=f'{name}_{seed}_{precision}';path=REG/'scores'/f'{key}.npz';metadata=path.with_suffix('.json')
            if metadata.exists():
                prior=json.loads(metadata.read_text())
                if prior['identity']!=identity or prior['prediction']!=file_identity(path):
                    raise ValueError('frozen predictions changed')
            else:
                if model is None:
                    model=build_model(infer_family(state),state['config']).cuda().eval()
                    model.load_state_dict(state['model'],strict=True)
                    bank=AnchoredFeatureBank(OUTPUT,OUT/'data/supplemental_features',SOURCE,
                        state['config'].get('drug_representation','drugclip'),local=model.is_local)
                measured=predict(model,bank,pd_,pt,precision)
                dense=predict(model,bank,od,ot,precision).reshape(len(old),len(target),2)
                np.savez_compressed(path,pool=measured,old=dense)
                prior=dict(identity=identity,prediction=file_identity(path),model=name,seed=seed,precision=precision,
                    selected_training_epochs=r['trained_epochs'],parameters=r['trainable_parameters'],
                    no_regression_label_read=True,downstream_cutoff=2022)
                write_json(metadata,prior)
            products.append(dict(key=key,metadata=file_identity(metadata),**{k:v for k,v in prior.items() if k!='identity'}))
            print(json.dumps(dict(event='predictions_frozen',key=key)),flush=True)
        del model,bank,state;gc.collect();torch.cuda.empty_cache()
    release=dict(status='PREDICTIONS_FROZEN_BEFORE_REGRESSION_LABEL_READ',products=products,
        selection=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'),producer=file_identity(Path(__file__)),
        no_prediction_ensemble=True,primary_precision='bf16 autocast; float32 stored logits',
        fp32_sensitivity_seed=s['deployment_seed'],downstream_cutoff=2022,
        pretraining_cutoff_certified=False,tests_previously_inspected=True,prospective_claim=False)
    path=REG/'PREDICTION_RELEASE.json'
    if path.exists() and json.loads(path.read_text())!=release:raise ValueError('release drift')
    write_json(path,release)


def evaluate():
    s=selection();release=json.loads((REG/'PREDICTION_RELEASE.json').read_text())
    if release['selection']!=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'):raise ValueError('selection drift')
    if release['producer']!=file_identity(Path(__file__)):raise ValueError('evaluator changed after score freeze')
    models={};descriptors={}
    for row in release['products']:
        path=Path(row['prediction']['path'])
        if file_identity(path)!=row['prediction']:raise ValueError('frozen score drift')
        values=np.load(path);models[row['key']]=(values['pool'],values['old'])
        descriptors[row['key']]={k:row[k] for k in ['model','seed','precision']}
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    target=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    pool=pd.read_csv(SOURCE/'QUERY_POOL.csv.gz');nd,nt=len(old),len(target)
    pm=json.loads((SOURCE/'final/scores/MANIFEST.json').read_text())
    for filename in ['POOL_SCORES.npz','OLD_SCORES.npz']:
        if file_identity(SOURCE/'final/scores'/filename)['sha256']!=pm['files'][filename]:raise ValueError('legacy scores drift')
    pp=np.load(SOURCE/'final/scores/POOL_SCORES.npz');po=np.load(SOURCE/'final/scores/OLD_SCORES.npz')
    for name in ['temporal_v3','temporal_J','positive_nearest']:
        models[name]=(pp[name],po[name].reshape(nd,nt,2))
        descriptors[name]=dict(model=name,seed=0,precision='archived')
    risk=np.load(SOURCE/'RISK_SETS.npz')['final']
    age=pd.read_csv(SOURCE/'REGISTRY_APPROVAL_AUDIT.csv').sort_values('old_drug_index')
    assert np.array_equal(age.drug_feature_index,old.drug_feature_index)
    approved=age.confirmed_approved_by_2022.to_numpy(bool)
    pocket=pd.read_csv(OUTPUT/'TARGET_COVERAGE.csv').pocket_union_residues.to_numpy()>0
    summaries=[];queries=[];pairs=[];label_sources=[]
    def collect(scope,key,direction,result,q,p=None):
        tags=dict(scope=scope,key=key,direction=direction,**descriptors[key])
        summaries.append(dict(**tags,**result))
        if len(q):queries.append(q.assign(**tags))
        if p is not None and len(p):pairs.append(p.assign(**tags))
    def measured(scope,frame,values):
        for key,scores in values.items():
            for head,direction in enumerate(['d2t','t2d']):
                result,q=measured_ranking(frame,scores[:,head],direction);collect(scope,key,direction,result,q)
    pool_index=pd.MultiIndex.from_frame(pool[['drug_feature_index','target_feature_index']])
    old_mapping=old.set_index('drug_feature_index').old_drug_index
    data_manifest=json.loads((SOURCE/'DATA_MANIFEST.json').read_text())
    for period in ['2023_2025','2023','2024','2025']:
        path=SOURCE/f'TEST_{period}_NEW.csv.gz';ident=file_identity(path)
        if ident['sha256']!=data_manifest['files'][str(path.relative_to(ROOT))]:raise ValueError('test data drift')
        label_sources.append(ident);table=pd.read_csv(path)
        assert table.min_document_year.min()>=2023 and table.max_document_year.max()<=2025
        rows=pool_index.get_indexer(pd.MultiIndex.from_frame(table[['drug_feature_index','target_feature_index']]))
        if (rows<0).any():raise ValueError('missing measured predictions')
        labels=np.zeros((nd,nt),bool)
        positive=table[table.binary_label.eq(1)&table.drug_feature_index.isin(old_mapping.index)]
        labels[positive.drug_feature_index.map(old_mapping).to_numpy(int),positive.target_feature_index.to_numpy(int)]=True
        scopes=[('old720',np.ones(nd,bool),np.ones(nt,bool)),('approved_pre2023',approved,np.ones(nt,bool))]
        if period=='2023_2025':scopes.append(('old720_pocket352',np.ones(nd,bool),pocket))
        for population,dm,tm in scopes:
            for key,(_,dense) in models.items():
                for head,direction in enumerate(['d2t','t2d']):
                    y=labels[np.ix_(dm,tm)];scores=dense[:,:,head][np.ix_(dm,tm)];r=risk[np.ix_(dm,tm)]
                    qi=old.ligand_inchikey.to_numpy()[dm];ci=target.target_chembl_id.to_numpy()[tm]
                    if head:y,scores,r,qi,ci=y.T,scores.T,r.T,ci,qi
                    result,q,p=risk_set_ranking(y,scores,r,qi,ci)
                    collect(f'{period}_dense_{population}',key,direction,result,q,p)
        if period=='2023_2025':
            frame=table.rename(columns={'drug_feature_index':'drug_id','target_feature_index':'target_id','binary_label':'label'})
            for population,mask in [('all_compounds',np.ones(len(table),bool)),('old720',table.drug_feature_index.isin(old_mapping.index))]:
                measured(f'{period}_measured_{population}',frame.loc[mask],{n:v[0][rows][mask] for n,v in models.items()})
        print(json.dumps(dict(event='temporal_metrics',period=period)),flush=True)
    kp=ROOT/'outputs/evidence_routing_compute_execution_20260808_v1/old_drug_innovation_v7/OLD_DRUG_KIRHUB_PAIR_BENCHMARK_V7.csv'
    kir=pd.read_csv(kp);label_sources.append(file_identity(kp))
    di=kir.ligand_inchikey.map(old.set_index('ligand_inchikey').old_drug_index).to_numpy(int)
    ti=kir.target_chembl_id.map(target.set_index('target_chembl_id').target_feature_index).to_numpy(int)
    kir['drug_id']=kir.ligand_inchikey;kir['target_id']=kir.target_chembl_id
    kir['label']=kir.kirhub_wt_min_residual_activity_pct_1uM.le(30).astype(int)
    masks={'kirhub_historical_strict_2823':kir.kirhub_frozen_unreported_scope.to_numpy(bool),
        'kirhub_single_construct_unseen_through2022':kir.kirhub_wt_construct_count.eq(1).to_numpy()&risk[di,ti]}
    audits={}
    for scope,mask in masks.items():
        measured(scope,kir.loc[mask],{n:v[1][di,ti][mask] for n,v in models.items()})
        audits[scope]=dict(rows=int(mask.sum()),positives=int(kir.label[mask].sum()),prior_overlap=int((~risk[di,ti]&mask).sum()))
    assert audits['kirhub_historical_strict_2823']==dict(rows=2823,positives=202,prior_overlap=0)
    assert audits['kirhub_single_construct_unseen_through2022']==dict(rows=3914,positives=322,prior_overlap=0)
    table=pd.DataFrame(summaries);q_all=pd.concat(queries,ignore_index=True)
    table.to_csv(REG/'METRICS_BY_SEED.csv',index=False)
    q_all.to_csv(REG/'QUERY_METRICS.csv.gz',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(REG/'POSITIVE_RANKS.csv.gz',index=False)
    primary=table[table.precision.ne('fp32')]
    metrics=['macro_ap','macro_recall_20','macro_ndcg_20']
    aggregate=primary.groupby(['scope','model','direction'])[metrics].agg(['mean','std','count'])
    aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate.reset_index().to_csv(REG/'METRICS_SEED_SUMMARY.csv',index=False)
    # Average each query's AP across seeds, so seeds do not inflate sample size.
    pq=q_all[q_all.precision.ne('fp32')].groupby(['scope','model','direction','query_id'],as_index=False).ap.mean()
    comparisons=[]
    for (scope,direction),group in pq.groupby(['scope','direction']):
        a=group[group.model.eq('selected')]
        for control in ['global','capacity','temporal_v3','temporal_J','positive_nearest']:
            b=group[group.model.eq(control)]
            if len(a) and len(b):comparisons.append(dict(scope=scope,direction=direction,control=control,
                **query_bootstrap(a,b),seed_handling='mean metric within query, then paired query bootstrap',
                multiple_testing_corrected=False,selection_use=False))
    write_json(REG/'PAIRED_COMPARISONS.json',comparisons)
    historical=pd.read_csv(SOURCE/'TEST_METRICS.csv')
    for name in ['temporal_v3','temporal_J','positive_nearest']:
        for direction in ['d2t','t2d']:
            a=table[(table.scope=='2023_2025_dense_old720')&(table.model==name)&(table.direction==direction)].iloc[0]
            b=historical[(historical.scope=='test_2023_2025_new_dense_old720')&(historical.model==name)&(historical.direction==direction)].iloc[0]
            for metric in ['macro_ap','macro_recall_20']:np.testing.assert_allclose(a[metric],b[metric],rtol=0,atol=1e-12)
    representative=table[(table.model=='selected')&(table.seed==s['deployment_seed'])]
    representative.to_csv(REG/'REPRESENTATIVE_PRECISION_METRICS.csv',index=False)
    write_json(REG/'RESULT.json',dict(status='COMPLETE',selection=file_identity(OUT/'FINAL_ARCHITECTURE_SELECTION.json'),
        release=file_identity(REG/'PREDICTION_RELEASE.json'),label_sources=label_sources,kirhub_scopes=audits,
        legacy_baseline_reproduction='PASS at absolute tolerance 1e-12',seeds=SEEDS,representative_seed=s['deployment_seed'],
        precision_sensitivity='bf16 and fp32 for preselected representative; no reselection',
        no_prediction_ensemble=True,tests_previously_inspected=True,tests_used_for_selection=False,
        pretraining_cutoff_certified=False,scope=[nd,nt],fullfit_checkpoint_evaluated=False,
        metrics=file_identity(REG/'METRICS_BY_SEED.csv'),producer=file_identity(Path(__file__))))
    print(aggregate.loc[['2023_2025_dense_old720','2023_2025_dense_approved_pre2023','kirhub_historical_strict_2823']].to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['score','evaluate','all'],default='all');a=p.parse_args()
    if a.phase in ['score','all']:score()
    if a.phase in ['evaluate','all']:evaluate()

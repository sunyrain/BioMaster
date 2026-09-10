#!/usr/bin/env python3
"""Freeze final predictions, then run previously inspected temporal/KIRHub regression."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
import torch
from biomaster.unified_features import UnifiedFeatureBank
from biomaster.unified_interaction import UnifiedConfig,UnifiedInteraction
from biomaster.odti_pockets_v3 import file_identity
from biomaster.ranking_audit import measured_ranking,risk_set_ranking,query_bootstrap
from train_biomaster_unified_interaction import OUT,OUTPUT,SOURCE,identity,predict,TemporalStage,validation
from prepare_biomaster_unified_interaction import write_json


def geometry_probe(p):
    """After selection: early-validation diagnostic, never used to pick epochs."""
    checkpoint=OUT/'development/geometry'/f'seed_{p["seeds"][0]}'/'BEST.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=UnifiedInteraction(UnifiedConfig(**state['config'])).cuda()
    model.load_state_dict(state['model'])
    bank=UnifiedFeatureBank(OUTPUT)
    stage=TemporalStage('development')
    original,_=validation(model,bank,stage,p)
    before=bank.ca.clone()
    for i in range(len(bank.ca)):
        ids=torch.where(bank.geom_mask[i])[0]
        bank.ca[i,ids]=before[i,ids.flip(0)]
    shuffled,_=validation(model,bank,stage,p)
    write_json(OUT/'GEOMETRY_PROBE.json',dict(checkpoint=file_identity(checkpoint),
               cutoff=2020,validation='2021-2022',original=original,shuffled=shuffled,
               perturbation='reverse coordinates among structure-bearing residues; keep sequence, site selection, masks and quality',
               used_for_selection=False,interpretation='sensitivity diagnostic, not proof of true atom-residue contacts'))
    del model,bank,state;torch.cuda.empty_cache()


def precision_probe(p):
    """Quantify ranking sensitivity to autocast on a frozen early-validation G."""
    stage=TemporalStage('development')
    checkpoint=OUT/'development/global'/f'seed_{p["seeds"][0]}'/'BEST.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=UnifiedInteraction(UnifiedConfig(**state['config'])).cuda().eval();model.load_state_dict(state['model'])
    bank=UnifiedFeatureBank(OUTPUT,local=False)
    d=np.repeat(stage.old.drug_feature_index.to_numpy(),stage.nt);t=np.tile(np.arange(stage.nt),len(stage.old))
    amp=predict(model,bank,d,t,p['evaluation_batch_size'])
    fp32=np.empty_like(amp)
    with torch.inference_mode():
        for lo in range(0,len(d),p['evaluation_batch_size']):
            hi=min(lo+p['evaluation_batch_size'],len(d))
            fp32[lo:hi]=model(bank.batch(d[lo:hi],t[lo:hi],'global')).cpu().numpy()
    results={}
    for mode,flat in [('bf16_autocast',amp),('fp32',fp32)]:
        dense=flat.reshape(len(stage.old),stage.nt,2);result={}
        for head,name in enumerate(['d2t','t2d']):
            y,s,r=stage.val_known,dense[:,:,head],stage.risk
            q,c=np.arange(len(stage.old)),np.arange(stage.nt)
            if head:y,s,r,q,c=y.T,s.T,r.T,c,q
            summary,_,_=risk_set_ranking(y,s,r,q,c)
            result[name]=dict(ap=summary['macro_ap'],r20=summary['macro_recall_20'])
        results[mode]=result
    write_json(OUT/'PRECISION_PROBE.json',dict(checkpoint=file_identity(checkpoint),validation='2021-2022 only',
               results=results,max_logit_difference=float(np.abs(amp-fp32).max()),
               inference_precision_in_primary_experiment='bf16 autocast, float32 stored scores',
               selected_epochs_changed=False,test_labels_used=False))
    del model,bank,state;torch.cuda.empty_cache()


def main():
    p=json.loads((ROOT/'configs/biomaster_unified_interaction_20260906.json').read_text())
    selection=json.loads((OUT/'SELECTION.json').read_text())
    assert selection['status']=='FROZEN_BEFORE_FINAL_REFIT_AND_REGRESSION'
    assert not selection['test_labels_used_for_selection']
    assert json.loads((OUT/'VALIDATION.json').read_text())['status']=='PASS'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ident=identity();old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv');target=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz')
    pool=pd.read_csv(SOURCE/'QUERY_POOL.csv.gz') # identities only
    nd,nt=len(old),len(target)
    old_d=np.repeat(old.drug_feature_index.to_numpy(),nt);old_t=np.tile(np.arange(nt),nd)
    scores_dir=OUT/'scores';scores_dir.mkdir(exist_ok=True)
    models={};checkpoints={}
    for variant in selection['final_variants']:
        predictions=[];old_predictions=[]
        for seed in p['seeds']:
            folder=OUT/'final'/variant/f'seed_{seed}'
            result=json.loads((folder/'RESULT.json').read_text())
            assert result['identity']==ident and result['train_max_year']==2022 and not result['test_labels_used']
            checkpoint=folder/'BEST.pt';assert file_identity(checkpoint)==result['checkpoint']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['identity']==ident and state['cutoff']==2022
            model=UnifiedInteraction(UnifiedConfig(**state['config'])).cuda();model.load_state_dict(state['model'])
            bank=UnifiedFeatureBank(OUTPUT,local=model.is_local)
            measured=predict(model,bank,pool.drug_feature_index.to_numpy(),pool.target_feature_index.to_numpy(),p['evaluation_batch_size'])
            dense=predict(model,bank,old_d,old_t,p['evaluation_batch_size']).reshape(nd,nt,2)
            np.savez_compressed(scores_dir/f'{variant}_{seed}.npz',pool=measured,old=dense)
            predictions.append(measured);old_predictions.append(dense)
            checkpoints[str(checkpoint.relative_to(ROOT))]=file_identity(checkpoint)
            del model,bank,state;torch.cuda.empty_cache()
        models[variant]=(np.mean(predictions,axis=0),np.mean(old_predictions,axis=0))
    release=dict(status='PREDICTIONS_FROZEN_BEFORE_REGRESSION_LABEL_READ',identity=ident,
                 selection=file_identity(OUT/'SELECTION.json'),checkpoints=checkpoints,
                 predictions={x.name:file_identity(x) for x in scores_dir.glob('*.npz')},
                 downstream_cutoff=2022,pretraining_cutoff_certified=False,prospective_claim=False)
    write_json(OUT/'TEST_RELEASE.json',release)
    prior_manifest=json.loads((SOURCE/'final/scores/MANIFEST.json').read_text())
    for name in ['POOL_SCORES.npz','OLD_SCORES.npz']:
        assert file_identity(SOURCE/'final/scores'/name)['sha256']==prior_manifest['files'][name]
    prior_pool=np.load(SOURCE/'final/scores/POOL_SCORES.npz');prior_old=np.load(SOURCE/'final/scores/OLD_SCORES.npz')
    for name in ['temporal_v3','temporal_J','positive_nearest']:
        models[name]=(prior_pool[name],prior_old[name].reshape(nd,nt,2))
    risk=np.load(SOURCE/'RISK_SETS.npz')['final']
    approved=pd.read_csv(SOURCE/'REGISTRY_APPROVAL_AUDIT.csv').confirmed_approved_by_2022.to_numpy(bool)
    pocket=pd.read_csv(OUTPUT/'TARGET_COVERAGE.csv').pocket_union_residues.to_numpy()>0
    summaries=[];queries=[];pairs=[];frames={}
    def collect(scope,name,direction,result,q):
        summaries.append(dict(scope=scope,model=name,direction=direction,**result))
        if len(q):
            q=q.copy();q['scope']=scope;q['model']=name;q['direction']=direction
            queries.append(q);frames[(scope,name,direction)]=q
    def measured(scope,frame,values):
        for name,scores in values.items():
            for head,direction in enumerate(['d2t','t2d']):
                result,q=measured_ranking(frame,scores[:,head],direction);collect(scope,name,direction,result,q)
    pool_index=pd.MultiIndex.from_frame(pool[['drug_feature_index','target_feature_index']])
    old_mapping=old.set_index('drug_feature_index').old_drug_index
    for period in ['2023_2025','2023','2024','2025']:
        path=SOURCE/f'TEST_{period}_NEW.csv.gz'
        data_manifest=json.loads((SOURCE/'DATA_MANIFEST.json').read_text())
        assert file_identity(path)['sha256']==data_manifest['files'][str(path.relative_to(ROOT))]
        table=pd.read_csv(path)
        assert table.min_document_year.min()>=2023 and table.max_document_year.max()<=2025
        rows=pool_index.get_indexer(pd.MultiIndex.from_frame(table[['drug_feature_index','target_feature_index']]))
        assert (rows>=0).all()
        labels=np.zeros((nd,nt),bool)
        positive=table[table.binary_label.eq(1)&table.drug_feature_index.isin(old_mapping.index)]
        labels[positive.drug_feature_index.map(old_mapping).to_numpy(int),positive.target_feature_index.to_numpy(int)]=True
        assert not (labels&~risk).any()
        scopes=[('old720',np.ones(nd,bool),np.ones(nt,bool)),('approved_pre2023',approved,np.ones(nt,bool))]
        if period=='2023_2025':scopes.append(('old720_pocket352',np.ones(nd,bool),pocket))
        for population,dm,tm in scopes:
            for name,(_,dense) in models.items():
                for head,direction in enumerate(['d2t','t2d']):
                    y=labels[np.ix_(dm,tm)];s=dense[:,:,head][np.ix_(dm,tm)];r=risk[np.ix_(dm,tm)]
                    q=old.ligand_inchikey.to_numpy()[dm];c=target.target_chembl_id.to_numpy()[tm]
                    if head:y,s,r,q,c=y.T,s.T,r.T,c,q
                    result,qf,pf=risk_set_ranking(y,s,r,q,c)
                    scope=f'{period}_dense_{population}';collect(scope,name,direction,result,qf)
                    pf['scope']=scope;pf['model']=name;pf['direction']=direction;pairs.append(pf)
        if period=='2023_2025':
            frame=table.rename(columns={'drug_feature_index':'drug_id','target_feature_index':'target_id','binary_label':'label'})
            for population,mask in [('all_compounds',np.ones(len(table),bool)),('old720',table.drug_feature_index.isin(old_mapping.index))]:
                measured(f'{period}_measured_{population}',frame.loc[mask],{n:v[0][rows][mask] for n,v in models.items()})
    # Same current KIRHub catalog mapping, with explicit overlap to <=2022 history.
    kirhub_path=ROOT/'outputs/evidence_routing_compute_execution_20260808_v1/old_drug_innovation_v7/OLD_DRUG_KIRHUB_PAIR_BENCHMARK_V7.csv'
    kir=pd.read_csv(kirhub_path)
    di=kir.ligand_inchikey.map(old.set_index('ligand_inchikey').old_drug_index).to_numpy(int)
    ti=kir.target_chembl_id.map(target.set_index('target_chembl_id').target_feature_index).to_numpy(int)
    kir['drug_id']=kir.ligand_inchikey;kir['target_id']=kir.target_chembl_id
    kir['label']=kir.kirhub_wt_min_residual_activity_pct_1uM.le(30).astype(int)
    masks={'kirhub_historical_strict_2823':kir.kirhub_frozen_unreported_scope.to_numpy(bool),
           'kirhub_single_construct_unseen_through2022':kir.kirhub_wt_construct_count.eq(1).to_numpy() & risk[di,ti]}
    assert masks['kirhub_historical_strict_2823'].sum()==2823
    assert kir.label[masks['kirhub_historical_strict_2823']].sum()==202
    kir_values={name:values[1][di,ti] for name,values in models.items()}
    audit={}
    for scope,mask in masks.items():
        measured(scope,kir.loc[mask],{n:s[mask] for n,s in kir_values.items()})
        audit[scope]=dict(rows=int(mask.sum()),positives=int(kir.label[mask].sum()),
                          previously_measured_through2022=int((~risk[di,ti]&mask).sum()))
    write_json(OUT/'KIRHUB_SCOPE_AUDIT.json',dict(source=file_identity(kirhub_path),scopes=audit,
                threshold='residual activity <=30 percent at 1uM',all_new_models_and_V3_J_NN_downstream_cutoff=2022,
                role='previously inspected retrospective regression, not new confirmation'))
    table=pd.DataFrame(summaries)
    historical=pd.read_csv(SOURCE/'TEST_METRICS.csv')
    reproduced=[]
    for name in ['temporal_v3','temporal_J','positive_nearest']:
        for direction in ['d2t','t2d']:
            current=table[(table.scope=='2023_2025_dense_old720') & (table.model==name) & (table.direction==direction)].iloc[0]
            prior=historical[(historical.scope=='test_2023_2025_new_dense_old720') & (historical.model==name) & (historical.direction==direction)].iloc[0]
            for metric in ['macro_ap','macro_recall_20']:
                np.testing.assert_allclose(current[metric],prior[metric],rtol=0,atol=1e-12)
            reproduced.append(dict(model=name,direction=direction,ap=float(current.macro_ap),r20=float(current.macro_recall_20)))
    write_json(OUT/'BASELINE_REPRODUCTION.json',dict(status='PASS',tolerance=1e-12,values=reproduced,
               source=file_identity(SOURCE/'TEST_METRICS.csv')))
    table.to_csv(OUT/'TEST_METRICS.csv',index=False)
    q_all=pd.concat(queries,ignore_index=True);q_all.to_csv(OUT/'TEST_QUERY_METRICS.csv.gz',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(OUT/'TEST_POSITIVE_RANKS.csv.gz',index=False)
    comparisons=[]
    for scope in sorted(table.scope.unique()):
        for direction in ['d2t','t2d']:
            for candidate in selection['final_variants']:
                for control in ['global','capacity','temporal_v3','temporal_J','positive_nearest']:
                    if candidate==control:continue
                    a=frames.get((scope,candidate,direction));b=frames.get((scope,control,direction))
                    if a is not None and b is not None:
                        comparisons.append(dict(scope=scope,candidate=candidate,control=control,direction=direction,**query_bootstrap(a,b)))
    write_json(OUT/'TEST_PAIRED_COMPARISONS.json',comparisons)
    geometry_probe(p)
    precision_probe(p)
    write_json(OUT/'TEST_RESULT.json',dict(status='COMPLETE',models=list(models),
               release=file_identity(OUT/'TEST_RELEASE.json'),metrics=file_identity(OUT/'TEST_METRICS.csv'),
               selection_used_test=False,single_seed=True,promoted=False))
    print(table[table.scope.isin(['2023_2025_dense_old720','2023_2025_dense_approved_pre2023','kirhub_historical_strict_2823'])][['scope','model','direction','macro_ap','macro_recall_20']].to_string(index=False))


if __name__=='__main__':main()

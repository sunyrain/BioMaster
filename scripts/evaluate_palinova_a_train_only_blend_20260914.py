#!/usr/bin/env python3
"""Freeze a training-only retrieval blend on validation, then evaluate diagnostics."""
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.endpoint_multitask import AuxiliaryInteraction
from biomaster.train_only_blend import TrainOnlyEvidence, apply_selection, validation_rank_selection

OUT = ROOT/'outputs/palinova_a_train_only_blend_20260914'
AB = ROOT/'outputs/biomaster_endpoint_ablation_20260911'
MT = ROOT/'outputs/biomaster_endpoint_multitask_20260911'
FEATURES = AB/'features'
PREV = ROOT/'outputs/palinova_ab_same_task_20260914'
KD = 'AFFINITY_KD_KI'
INACTIVE = 'EXPLICIT_INACTIVE'
REGISTRY = ROOT/'outputs/biomaster_model_consolidation_20260911/RECOMMENDED_MODELS.json'
MODEL = MT/'kdki_inactive__binary__seed_20260921/model.pt'
FEATURE_NAMES = ['neural_score','positive_neighbor','target_prior']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def log(stage,**kw):
    v=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kw)
    write('STATUS.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)


@torch.inference_mode()
def neural_validation(frame):
    arrays={k:np.load(FEATURES/(v+'.npy'),mmap_mode='r') for k,v in
        [('clip','DRUG_CLIP'),('fp','MORGAN'),('graph','GRAPH'),('available','AVAILABLE'),('target','TARGET')]}
    state=torch.load(MODEL,map_location='cpu',weights_only=True)
    model=AuxiliaryInteraction(state['config'],state['endpoints']).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    result=[]
    for start in range(0,len(frame),4096):
        f=frame.iloc[start:start+4096];d=f.drug_feature_index.to_numpy(int);t=f.target_feature_index.to_numpy(int)
        batch={'drug_global':torch.tensor(np.concatenate([arrays['clip'][d],arrays['fp'][d].astype(np.float32)],1),device='cuda'),
            'drug_graph_mean':torch.tensor(arrays['graph'][d],device='cuda'),
            'pretrained_available':torch.tensor(arrays['available'][d].astype(np.float32),device='cuda'),
            'target_global':torch.tensor(arrays['target'][t],device='cuda')}
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(batch)
        result.append(logits.float().mean(1).cpu().numpy())
    del model,state
    scores=np.concatenate(result);assert np.isfinite(scores).all()
    return scores


def metrics(frame,score):
    y=frame.binary_label.to_numpy(int);score=np.asarray(score,float)
    both=len(y)>0 and 0<y.sum()<len(y)
    r=dict(pairs=len(y),positive=int(y.sum()),negative=int(len(y)-y.sum()),
        random_ap_reference=float(y.mean()) if len(y) else None,
        ap=float(average_precision_score(y,score)) if both else None,
        auroc=float(roc_auc_score(y,score)) if both else None)
    queries=[]
    for direction,key in [('drug','molecule_id'),('target','target_id')]:
        for name,ids in frame.groupby(key,sort=True).indices.items():
            if not 0<y[ids].sum()<len(ids):continue
            order=np.argsort(-score[ids],kind='stable')
            queries.append(dict(direction=direction,query=name,pairs=len(ids),
                ap=float(average_precision_score(y[ids],score[ids])),
                p5=float(y[ids][order[:5]].mean()) if len(ids)>=5 else None,
                p20=float(y[ids][order[:20]].mean()) if len(ids)>=20 else None))
        for minimum in [2,10]:
            q=[x for x in queries if x['direction']==direction and x['pairs']>=minimum]
            r[f'{direction}_queries_ge{minimum}']=len(q)
            r[f'{direction}_macro_ap_ge{minimum}']=float(np.mean([x['ap'] for x in q])) if q else None
        q=[x for x in queries if x['direction']==direction and x['pairs']>=10]
        r[direction+'_p5_ge10']=float(np.mean([x['p5'] for x in q])) if q else None
        p20=[x['p20'] for x in q if x['p20'] is not None]
        r[direction+'_p20_ge20']=float(np.mean(p20)) if p20 else None
    return r,queries


def calibrate(y,s):
    y=np.asarray(y,int);s=np.asarray(s,float)
    model=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,random_state=20260914).fit(s[:,None],y)
    assert model.coef_[0,0]>0
    prob=model.predict_proba(s[:,None])[:,1]
    p,r,t=precision_recall_curve(y,prob);f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return dict(slope=float(model.coef_[0,0]),intercept=float(model.intercept_[0]),
        threshold=float(t[np.argmax(f)]),validation_f1=float(f.max()),fit_panel='VALIDATION_KD_KI_ONLY')


def threshold_metrics(frame,s,cal):
    y=frame.binary_label.to_numpy(int);positive=expit(cal['slope']*s+cal['intercept'])>=cal['threshold']
    tp=int((positive & (y==1)).sum());fp=int((positive & (y==0)).sum());n=int((y==0).sum());p=int(y.sum())
    return dict(recall=tp/p if p else None,false_positive_rate=fp/n if n else None,
        precision=tp/(tp+fp) if tp+fp else None,threshold=cal['threshold'])


def pair_bootstrap(frame,candidate,reference,seed=20260914):
    y=frame.binary_label.to_numpy(int)
    groups=list(frame.groupby('molecule_id',sort=True).indices.values());rng=np.random.default_rng(seed)
    values=[]
    if not len(y) or min(y.sum(),len(y)-y.sum())<10:return None
    for _ in range(1000):
        ix=np.concatenate([groups[k] for k in rng.integers(len(groups),size=len(groups))]);yy=y[ix]
        if yy.min()==yy.max():continue
        values.append([average_precision_score(yy,candidate[ix])-average_precision_score(yy,reference[ix]),
                       roc_auc_score(yy,candidate[ix])-roc_auc_score(yy,reference[ix])])
    q=np.quantile(values,[.025,.975],axis=0)
    return dict(drug_clusters=len(groups),replicates=len(values),
        ap_delta=float(average_precision_score(y,candidate)-average_precision_score(y,reference)),
        ap_delta_ci95_low=float(q[0,0]),ap_delta_ci95_high=float(q[1,0]),
        auroc_delta=float(roc_auc_score(y,candidate)-roc_auc_score(y,reference)),
        auroc_delta_ci95_low=float(q[0,1]),auroc_delta_ci95_high=float(q[1,1]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    registry=json.loads(REGISTRY.read_text())['roles']['binding']
    assert sha(MODEL)==registry['checkpoint_sha256']
    paths=[Path(__file__),ROOT/'biomaster/train_only_blend.py',REGISTRY,MODEL,AB/'kdki_inactive_TRAIN.parquet',
        AB/'COMMON_VALIDATION.parquet',AB/'COMMON_TEST.parquet',MT/'TEST_PREDICTIONS.parquet',
        PREV/'ALL_PANEL_PREDICTIONS_AND_OVERLAP.csv.gz',PREV/'CORE720x384_RECOMMENDED_SCORES.parquet',
        PREV/'DRUG_INPUT_AUDIT.csv',PREV/'INPUT_drug_global.npy',FEATURES/'MANIFEST.json',
        ROOT/'outputs/biomaster_matrix_720x890_20260910/TARGET_INDEX.csv.gz',
        ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in paths}
    grid=list(itertools.product([0.,.25,.5,1.,2.,2.5,4.],[0.,.125,.25,.5,1.,2.]))
    write('PROTOCOL.json',dict(status='FROZEN_BEFORE_COMPONENT_BUILD_AND_WEIGHT_SELECTION',input_hashes=frozen,
        neural_checkpoint=registry,training_reference='337570 A TRAIN pairs only; positive neighbor pools from188799 measured positives; all measured labels form prior',
        nearest='Maximum Morgan2048 radius2 Tanimoto at identical sequence; exclude query connectivity including self and stereoisomers; no positive pool ->0',
        prior='(target positives+10*global training prevalence)/(target measured pairs+10); unseen target ->global prevalence',
        feature_standardization='Mean and population std fitted on full common Kd/Ki VALIDATION, never test',
        weights='Neural coefficient fixed1; nonnegative neighbor and prior grid. Zero included in every applicable family.',
        grid=grid,selection='Same primary direction as historical drug ranker: mean drug AP across5 hashed drug folds minus fold std; all1097 mixed-label drug queries, >=2 measured pairs. Ties prefer lower weight sum.',
        ablations=['A_neural','A_plus_neighbor','A_plus_prior','A_joint_blend'],
        secondary_metrics='Global AP/AUROC; drug and target macroAP with>=2 and>=10; P5 with>=10; P20 with>=20; explicit inactive FPR paired with Kd/Ki recall',
        thresholds='Each blend calibrated and maximum-F1 threshold chosen solely on same validation Kd/Ki labels',
        evaluation='Existing common neural A TEST and prior S5/BindingDB panels only after SELECTION.json frozen; prior test results viewed, retrospective diagnostics',
        overlap='Retain exact same A/B-common-unseen masks from previous audit; neighbor excludes self even on overlapping full diagnostic panels',
        neural_retraining=False,website_modified=False,wetlab_reselection=False,dtiam_test_access=False))
    log('PREPARING_TRAINING_REFERENCE')
    train=pd.read_parquet(AB/'kdki_inactive_TRAIN.parquet')
    assert len(train)==337570 and int(train.binary_label.sum())==188799
    assert not train.pair_id.duplicated().any() and train.split.eq('train').all()
    fpbank=np.load(FEATURES/'MORGAN.npy',mmap_mode='r')
    evidence=TrainOnlyEvidence(train,fpbank,prior_strength=10.)
    evidence.counts.assign(smoothed_prior=evidence.priors).to_csv(OUT/'TRAIN_TARGET_PRIORS.csv')
    trainkeys=set(train.molecule_id.str.split('-').str[0]+'__'+train.target_id)
    reference_counts=dict(pairs=len(train),positive=int(train.binary_label.sum()),negative=int((train.binary_label==0).sum()),
        targets=train.target_id.nunique(),positive_pool_targets=len(evidence.pools),global_prior=evidence.global_prior,
        fingerprint_reference='Unmodified training feature cache; hash declared in feature manifest')
    write('TRAIN_REFERENCE.json',reference_counts)
    validation=pd.read_parquet(AB/'COMMON_VALIDATION.parquet')
    log('SCORING_VALIDATION_NEURAL',rows=len(validation))
    validation['neural_score']=neural_validation(validation)
    validation[['panel','pair_id','neural_score']].to_parquet(OUT/'VALIDATION_NEURAL_SCORES.parquet',index=False)
    val=validation[validation.panel.eq(KD)].copy().reset_index(drop=True)
    assert len(val)==34685
    assert not set(val.molecule_id.str.split('-').str[0]+'__'+val.target_id)&trainkeys
    ap=average_precision_score(val.binary_label,val.neural_score)
    assert abs(ap-registry['representative_validation_value'])<1e-10,(ap,registry['representative_validation_value'])
    log('BUILDING_VALIDATION_NEIGHBORS',pairs=len(val),verified_neural_ap=ap)
    torch.backends.cuda.matmul.allow_tf32=False
    components=evidence.transform(val.molecule_id,val.target_id,fpbank[val.drug_feature_index.to_numpy(int)],
        progress=lambda n,total:log('BUILDING_VALIDATION_NEIGHBORS',targets=n,total_targets=total))
    val=pd.concat([val,components],axis=1)
    assert val.same_connectivity_references_excluded.sum()==0
    log('SELECTING_VALIDATION_WEIGHTS',grid_candidates=len(grid))
    selection,search=validation_rank_selection(val,val[FEATURE_NAMES],grid)
    assert selection['validation_drug_queries']==1097
    scores=apply_selection(val[FEATURE_NAMES],selection)
    selection['calibrations']={name:calibrate(val.binary_label,s) for name,s in scores.items()}
    selection['input_hashes']=frozen
    selection['frozen_utc']=datetime.now(timezone.utc).isoformat()
    search.to_csv(OUT/'VALIDATION_WEIGHT_SEARCH.csv',index=False)
    write('SELECTION.json',selection)
    selection_hash=sha(OUT/'SELECTION.json')
    log('SELECTION_FROZEN',selection_sha256=selection_hash,
        selected_weights={k:[v['neighbor_weight'],v['prior_weight']] for k,v in selection['choices'].items()})

    rows=[];queryrows=[];cis=[]
    def evaluate(frame,scope,views,thresholds=False):
        for name,score in views.items():
            score=np.asarray(score,float)
            m,q=metrics(frame,score)
            row=dict(scope=scope,model=name,**m)
            if thresholds and name in selection['calibrations']:
                row.update(threshold_metrics(frame,score,selection['calibrations'][name]))
            rows.append(row)
            queryrows.extend(dict(scope=scope,model=name,**x) for x in q)
    valviews={**scores,'neighbor_only':val.positive_neighbor.to_numpy(),'prior_only':val.target_prior.to_numpy()}
    evaluate(val,'COMMON_VALIDATION_KD_KI',valviews,thresholds=True)
    for name,s in scores.items():val[name+'_score']=s
    val[['pair_id','molecule_id','target_id','binary_label',*FEATURE_NAMES,'nearest_training_positive',
         *[name+'_score' for name in scores]]].to_parquet(OUT/'VALIDATION_BLEND_SCORES.parquet',index=False)

    log('BUILDING_COMMON_TEST_COMPONENTS')
    test=pd.read_parquet(AB/'COMMON_TEST.parquet')
    test=test[test.panel.isin([KD,INACTIVE])].copy()
    assert not set(test.molecule_id.str.split('-').str[0]+'__'+test.target_id)&trainkeys
    saved=pd.read_parquet(MT/'TEST_PREDICTIONS.parquet',columns=['panel','pair_id','kdki_inactive__binary__20260921_score'])
    test=test.merge(saved,on=['panel','pair_id'],validate='one_to_one').rename(columns={'kdki_inactive__binary__20260921_score':'neural_score'})
    unique=test.drop_duplicates('pair_id').reset_index(drop=True)
    c=evidence.transform(unique.molecule_id,unique.target_id,fpbank[unique.drug_feature_index.to_numpy(int)],
        progress=lambda n,total:log('BUILDING_COMMON_TEST_COMPONENTS',targets=n,total_targets=total))
    assert c.same_connectivity_references_excluded.sum()==0
    c['pair_id']=unique.pair_id
    test=test.merge(c,on='pair_id',validate='many_to_one')
    for name,s in apply_selection(test[FEATURE_NAMES],selection).items():test[name+'_score']=s
    for panel,sub in test.groupby('panel',sort=True):
        evaluate(sub.reset_index(drop=True),'COMMON_TEST_'+panel,
            {**{name:sub[name+'_score'].to_numpy() for name in scores},
             'neighbor_only':sub.positive_neighbor.to_numpy(),'prior_only':sub.target_prior.to_numpy()},thresholds=True)
    test.to_parquet(OUT/'COMMON_TEST_BLEND_PREDICTIONS.parquet',index=False,compression='zstd')
    # Within a fixed target, adding only a target constant cannot change ranking.
    r=pd.DataFrame(rows)
    base=r[r.scope.eq('COMMON_TEST_'+KD)&r.model.eq('A_neural')].iloc[0]
    prior=r[r.scope.eq('COMMON_TEST_'+KD)&r.model.eq('A_plus_prior')].iloc[0]
    assert abs(base.target_macro_ap_ge10-prior.target_macro_ap_ge10)<1e-12
    assert abs(base.ap-registry['representative_test']['ap'])<1e-12
    q=pd.DataFrame(queryrows)
    for direction in ['drug','target']:
        scoped=q[q.scope.eq('COMMON_TEST_'+KD)&q.direction.eq(direction)&q.pairs.ge(10)]
        ref=scoped[scoped.model.eq('A_neural')].set_index('query')
        for model in ['A_plus_neighbor','A_plus_prior','A_joint_blend']:
            cand=scoped[scoped.model.eq(model)].set_index('query').reindex(ref.index)
            for key in ['ap','p5','p20']:
                diff=(cand[key]-ref[key]).dropna().to_numpy(float)
                if not len(diff):continue
                rng=np.random.default_rng(20260914)
                distribution=diff[rng.integers(len(diff),size=(1000,len(diff)))].mean(1)
                cis.append(dict(scope='COMMON_TEST_'+KD,model=model,reference='A_neural',
                    unit=direction+'_query',metric=key,queries=len(diff),delta=float(diff.mean()),
                    ci95_low=float(np.quantile(distribution,.025)),ci95_high=float(np.quantile(distribution,.975))))

    log('BUILDING_PREVIOUS_PANEL_COMPONENTS')
    old=pd.read_csv(PREV/'ALL_PANEL_PREDICTIONS_AND_OVERLAP.csv.gz',low_memory=False)
    old=old.rename(columns={'Palinova_A_binary_score':'neural_score'})
    drugs=pd.read_csv(PREV/'DRUG_INPUT_AUDIT.csv')
    lookup=pd.Series(np.arange(len(drugs)),index=drugs.drug_id)
    catalogfp=np.load(PREV/'INPUT_drug_global.npy')[:,512:].astype(np.uint8)
    uniq=old.drop_duplicates('normalized_pair_id').reset_index(drop=True)
    c=evidence.transform(uniq.molecule_id,uniq.target_id,catalogfp[uniq.molecule_id.map(lookup).to_numpy(int)])
    c['normalized_pair_id']=uniq.normalized_pair_id
    old=old.merge(c,on='normalized_pair_id',validate='many_to_one')
    for name,s in apply_selection(old[FEATURE_NAMES],selection).items():old[name+'_score']=s
    masks={'S5':old.cohort.eq('S5_HISTORICAL_TEST'),'MERGED488':old.cohort.eq('MERGED_REFRESH_488'),
        'BINDINGDB479':old.is_bindingdb479}
    for cohort,mask in masks.items():
        for subset,keep in [('all',np.ones(len(old),bool)),('AB_pair_unseen',old.both_AB_train_validation_pair_unseen),
                             ('AB_group_unseen',old.both_AB_train_validation_group_unseen)]:
            sub=old[mask&keep].reset_index(drop=True);scope=cohort+'__'+subset
            views={'original_SPR_ranker':sub.independent_validation_rank_score.to_numpy(),
                **{name:sub[name+'_score'].to_numpy() for name in scores},
                'neighbor_only':sub.positive_neighbor.to_numpy(),'prior_only':sub.target_prior.to_numpy()}
            evaluate(sub,scope,views)
            if subset=='AB_pair_unseen':
                for ref in ['original_SPR_ranker','A_neural']:
                    for name in ['A_plus_neighbor','A_plus_prior','A_joint_blend']:
                        ci=pair_bootstrap(sub,views[name],views[ref])
                        if ci:cis.append(dict(scope=scope,model=name,reference=ref,unit='drug_cluster',**ci))
        log('EVALUATED_PREVIOUS_PANEL',cohort=cohort)
    old.to_csv(OUT/'PREVIOUS_PANEL_BLEND_PREDICTIONS.csv.gz',index=False,compression='gzip')

    log('BUILDING_CATALOG_COMPONENTS',pairs=276480)
    core=pd.read_parquet(PREV/'CORE720x384_RECOMMENDED_SCORES.parquet')
    ti=pd.read_csv(ROOT/'outputs/biomaster_matrix_720x890_20260910/TARGET_INDEX.csv.gz',usecols=['target_chembl_id','sequence_sha256'])
    target_ids='SEQ:'+core.target_chembl_id.map(ti.set_index('target_chembl_id').sequence_sha256)
    c=evidence.transform(core.ligand_inchikey,target_ids,catalogfp[core.ligand_inchikey.map(lookup).to_numpy(int)],
        progress=lambda n,total:log('BUILDING_CATALOG_COMPONENTS',targets=n,total_targets=total))
    core=pd.concat([core.reset_index(drop=True),c],axis=1)
    core['neural_score']=core.Palinova_A_binary_score
    for name,s in apply_selection(core[FEATURE_NAMES],selection).items():
        core[name+'_score']=s
        core[name+'_rank384']=core.groupby('ligand_inchikey')[name+'_score'].rank(method='first',ascending=False).astype(int)
    assert np.array_equal(core.A_neural_rank384,core.Palinova_A_binary_rank384)
    core.to_parquet(OUT/'CATALOG_BLEND_COMPONENTS_AND_RANKS.parquet',index=False,compression='zstd')
    final=pd.read_csv(ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv')
    candidate=final.merge(core,on='pair_id',validate='one_to_one')
    assert len(candidate)==384
    outcols=['排序','小分子药物名称','新靶点名称','pair_id',*FEATURE_NAMES,'nearest_training_positive',
             *[name+'_rank384' for name in scores]]
    candidate[outcols].to_csv(OUT/'FROZEN384_BLEND_RANK_DIAGNOSTIC.csv',index=False,encoding='utf-8-sig')
    retention={name:dict(top10=int((candidate[name+'_rank384']<=10).sum()),
        top20=int((candidate[name+'_rank384']<=20).sum()),median_rank=float(candidate[name+'_rank384'].median())) for name in scores}
    pd.DataFrame(rows).to_csv(OUT/'METRICS.csv',index=False)
    pd.DataFrame(queryrows).to_csv(OUT/'QUERY_METRICS.csv.gz',index=False,compression='gzip')
    pd.DataFrame(cis).to_csv(OUT/'PAIRED_INTERVALS.csv',index=False)
    assert sha(OUT/'SELECTION.json')==selection_hash
    for path,h in frozen.items():assert sha(ROOT/path)==h,path
    write('SUMMARY.json',dict(status='COMPLETE_VALIDATION_SELECTED_TRAIN_ONLY_BLEND',
        neural_retraining=False,selection=selection,metrics=rows,original384_rank_diagnostic=retention,
        validations=dict(original_A_validation_ap_reproduced=float(ap),original_A_test_ap_reproduced=float(base.ap),
            A_plus_prior_preserves_within_target_ranking=True,validation_and_test_no_training_connectivity_pair_overlap=True,
            all_original_inputs_unchanged=True,selection_unchanged_during_test=True),
        limitations=['Existing validation reused for checkpoint and blend selection; not nested independent confirmation.',
            'Test and historical panels were inspected before this study; this is a new frozen diagnostic, not prospective evidence.',
            'Reference pools use A training only; full historical panels still include neural/prior training overlap.',
            'No source-document or public-encoder pretraining independence certification.',
            'AP and TopK in observed panels do not calibrate frozen384 experimental hit rates.',
            'Full blend permits zero weights; an ablation may select the baseline. No outcome-driven expansion of grid.',
            'Only preselected A binary seed20260921 used; seed robustness of this blend not established.'],
        website_modified=False,wetlab_reselection=False,dtiam_test_access=False))
    log('COMPLETE',selected_weights={k:[v['neighbor_weight'],v['prior_weight']] for k,v in selection['choices'].items()},
        original384_rank_diagnostic=retention)


if __name__=='__main__':
    main()

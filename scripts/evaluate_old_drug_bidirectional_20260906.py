#!/usr/bin/env python3
"""Score the frozen project old-drug universe and evaluate both ranking axes.

No fitting, checkpoint selection or ranking-weight tuning uses these labels.
Full-universe known-relationship recovery, measured-PN ranking, historical
test-drug subsets and the common DrugCLIP coverage are explicitly separated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.odti_v4 import GlobalPairV4,GlobalConfigV4,SignedEvidenceV4
from biomaster.odti_support_data_v3 import SupportBatch
from biomaster.selectivity_training_v3 import support_summary_features
from biomaster.old_drug_ranking import known_ranking,observed_ranking,directional_borda
from train_biomaster_odti_v4 import Runtime as RuntimeV4
from train_biomaster_selectivity_v3 import Runtime as RuntimeV3
from prepare_old_drug_bidirectional_20260906 import OUT,DEPLOY,V4,V3
from build_biomaster_odti_v4_features import sha256,write_json


def load_inputs():
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    assert protocol['status']=='FROZEN'
    for name,digest in protocol['input_sha256'].items():
        assert sha256(ROOT/name)==digest,name
    drugs=pd.read_csv(OUT/'OLD_DRUGS_720.csv')
    targets=pd.read_csv(OUT/'TARGETS_384.csv.gz')
    pairs=pd.read_csv(OUT/'CANONICAL_PAIRS_720X384.csv.gz',low_memory=False)
    evidence={n:np.load(p,mmap_mode='r') for n,p in protocol['support'].items()}
    return protocol,drugs,targets,pairs,evidence


def save_scores(name,d2t,t2d,metadata):
    if d2t.shape!=(720,384) or t2d.shape!=(720,384):
        raise ValueError('Every score artifact must retain the full named axes')
    path=OUT/'scores'/f'{name}.npz'
    np.savez_compressed(path,drug_to_target=d2t,target_to_drug=t2d)
    write_json(path.with_suffix('.json'),{'name':name,**metadata,'score_sha256':sha256(path)})


def neural_scores(protocol,drugs,targets,evidence):
    out=OUT/'scores'
    out.mkdir(exist_ok=True)
    ds=np.repeat(drugs.global_drug_index.to_numpy(np.int64),384)
    ts=np.tile(targets.global_target_index.to_numpy(np.int64),720)
    old_morgan=np.load(OUT/'features/OLD720_MORGAN2048.npy')
    old_bermol=np.load(OUT/'features/OLD720_BERMOL768.npy')
    extra_esm=np.load(OUT/'features/ADDED_FULL_ESM2_MEANS.npy')
    runtime=RuntimeV4()
    runtime.molecules['morgan']=torch.cat([runtime.molecules['morgan'],torch.from_numpy(old_morgan).cuda()])
    runtime.molecules['bermol']=torch.cat([runtime.molecules['bermol'],torch.from_numpy(old_bermol).cuda()])
    runtime.proteins=torch.cat([runtime.proteins,torch.from_numpy(extra_esm).cuda()])
    for variant in ['morgan','bermol','signed']:
        for seed in [20260905,20260906,20260907]:
            name=f'{variant}_{seed}'
            if (out/f'{name}.npz').exists():
                continue
            path=V4/'runs'/name/'BEST_MODEL_V4.pt'
            checkpoint=torch.load(path,map_location='cuda',weights_only=False)
            assert checkpoint['identity']==runtime.identity
            base=GlobalPairV4(GlobalConfigV4(**checkpoint['config'])).cuda()
            model=SignedEvidenceV4(base).cuda() if variant=='signed' else base
            model.load_state_dict(checkpoint['model'])
            kind='morgan' if variant=='morgan' else 'bermol'
            matrix=runtime.predict(model,kind,ds,ts,evidence)['logit'].reshape(720,384)
            assert np.isfinite(matrix).all()
            save_scores(name,matrix,matrix,{'kind':'new_neural_checkpoint','checkpoint':str(path.relative_to(ROOT)),
                'checkpoint_sha256':sha256(path),'selected_epoch':checkpoint['epoch'],
                'training_exposure':'original V3 train split; exact old-drug and pair exposure in PROTOCOL and label masks',
                'reverse':'same pair logit ranked across old drugs; no reverse head trained in R1'})
            print(json.dumps({'scored':name}),flush=True)
            del model,base,checkpoint
        member=[np.load(out/f'{variant}_{s}.npz')['drug_to_target'] for s in [20260905,20260906,20260907]]
        mean=np.mean(member,axis=0)
        save_scores(f'{variant}_ensemble',mean,mean,{'kind':'fixed_three_seed_mean_logits','members':[f'{variant}_{s}' for s in [20260905,20260906,20260907]],'training_exposure':'same as members'})
    data=runtime.data
    audit=protocol['v3_data_manifest']['audit']
    original_support=SupportBatch(**runtime.support)
    del runtime
    torch.cuda.empty_cache()
    selected=targets.global_target_index.ge(843).to_numpy()
    extra_prot=np.load(DEPLOY/'PROJECT384_PROTBERT1024_FLOAT32_V1.npy')[selected]
    extra_aux=np.load(ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy')[selected]
    families={name:i for i,name in enumerate(audit['families'])}
    fs=np.tile(targets.target_assay_family.map(families).to_numpy(),720)
    assert np.isfinite(fs).all()
    batch=SupportBatch(**evidence)
    for seed in [20260905,20260906]:
        name=f'v3_support_{seed}'
        if (out/f'{name}.npz').exists():
            continue
        path=ROOT/f'outputs/biomaster_v3_20260905/experiments/B_support_pair/seed_{seed}/BEST_MODEL_V3.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        args=argparse.Namespace(**checkpoint['arguments'])
        rt=RuntimeV3(args,data,audit,None,original_support)
        rt.restore(checkpoint)
        rt.drug_bank=torch.cat([rt.drug_bank,torch.from_numpy(old_morgan).cuda()])
        rt.target_bank=torch.cat([rt.target_bank,torch.from_numpy(extra_prot).cuda()])
        rt.aux_bank=torch.cat([rt.aux_bank,torch.from_numpy(extra_aux).cuda()])
        rt.mode(False)
        chunks=[]
        with torch.inference_mode():
            for start in range(0,len(ds),1024):
                ix=np.arange(start,min(start+1024,len(ds)))
                with rt.autocast():
                    prediction=rt.model(**rt.inputs(ds[ix],ts[ix],fs[ix],batch,ix))
                chunks.append(prediction['final_logit'].float().cpu().numpy())
        matrix=np.concatenate(chunks).reshape(720,384)
        assert np.isfinite(matrix).all()
        save_scores(name,matrix,matrix,{'kind':'historical_v3_checkpoint','checkpoint':str(path.relative_to(ROOT)),
            'checkpoint_sha256':sha256(path),'training_exposure':'same V3 training entities and support pool as R1'})
        print(json.dumps({'scored':name}),flush=True)
        del rt,checkpoint
        torch.cuda.empty_cache()
    matrix=np.mean([np.load(out/f'v3_support_{s}.npz')['drug_to_target'] for s in [20260905,20260906]],axis=0)
    save_scores('v3_support_ensemble',matrix,matrix,{'kind':'fixed_two_seed_mean_logits','training_exposure':'same as V3 members'})


def existing_scores(protocol,drugs,targets,pairs,evidence):
    core=ROOT/'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz'
    existing=pd.read_csv(core).set_index('pairId').loc[pairs.pairId]
    def matrix(column):
        return existing[column].to_numpy().reshape(720,384)
    save_scores('dtiam',matrix('dtiam_probability'),matrix('dtiam_probability'),
        {'kind':'DTIAM_official_representation_compatible_AutoML_S5_retrain','source':str(core.relative_to(ROOT)),
         'source_sha256':sha256(core),'training_exposure':'different historical S5 fitting split; no common unbiased claim'})
    a,b=matrix('biomaster_routed_stack_score'),matrix('old_drug_leakage_safe_score_v10')
    forward=directional_borda(a,b,axis=1)
    assert np.allclose(forward,matrix('biomaster_independent_borda_score'),atol=1e-7)
    save_scores('production_independent',forward,directional_borda(a,b,axis=0),
        {'kind':'existing_independent_system','source':str(core.relative_to(ROOT)),
         'source_sha256':sha256(core),'training_exposure':'historical training/graph knowledge may include known relations',
         'reverse':'additional deterministic target-wise Borda of the same raw components; not a previously validated reverse model'})
    v6path=ROOT/'outputs/biomaster_bidirectional_v6_720x384/BIDIRECTIONAL_V6_FULL_FIT_720X384_SCORES.csv.gz'
    v6=pd.read_csv(v6path,low_memory=False).set_index('pairId').loc[pairs.pairId]
    save_scores('v6_directional_full_fit',v6.ensemble_drug_to_target_logit.to_numpy().reshape(720,384),
        v6.ensemble_target_to_drug_logit.to_numpy().reshape(720,384),{'kind':'three_seed_direction_specific_heads',
            'source':str(v6path.relative_to(ROOT)),'source_sha256':sha256(v6path),'training_exposure':'FULL_FIT has used existing biochemical labels; recovery audit only'})
    conplex=pairs.conplex_score.to_numpy().reshape(720,384)
    save_scores('conplex',conplex,conplex,{'kind':'existing_ConPLex_scores','training_exposure':'public pretrained task overlap not audited here'})
    features=support_summary_features(evidence['similarities'],evidence['mask'])
    nearest=features[:,1].reshape(720,384)
    save_scores('positive_nearest',nearest,nearest,{'kind':'train_positive_max_Tanimoto','training_exposure':'same fixed V3 train support; query chemical entity excluded'})
    pn=json.loads((V4/'evaluation/PN_BASELINE_V4.json').read_text())
    logits=((features-np.array(pn['scaler_mean']))/np.array(pn['scaler_scale']))@np.array(pn['coefficient'])[0]+pn['intercept'][0]
    matrix_pn=logits.reshape(720,384)
    save_scores('pn_logistic',matrix_pn,matrix_pn,{'kind':'frozen_train_only_PN_logistic','training_exposure':'same original V3 train observations; coefficients not refitted here'})
    # DrugCLIP joins exact ligand SMILES and canonical sequence identity.
    dcpath=ROOT/'outputs/affinity_first_remote_discovery_v1/drugclip_inference_v1/DRUGCLIP_PROJECT722_X_307_MATRIX_V1.csv.gz'
    dc=pd.read_csv(dcpath,usecols=['model_ligand_smiles','sequence_key','drugclip_cosine_mean_v1'])
    # DrugCLIP used the older SEQ000xxx namespace, not the later SEQ37 hashes.
    # Recover its own input sequences, then join by full sequence SHA256.
    readiness_path=ROOT/'outputs/affinity_first_remote_discovery_v1/TARGET_MODEL_READINESS_463_V1.csv'
    universe=pd.read_csv(readiness_path,usecols=['sequence_key','sequence']).drop_duplicates()
    universe['sequence_sha256']=universe.sequence.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    sequence=universe.groupby('sequence_key').sequence_sha256.agg(lambda v:sorted(set(v))).to_dict()
    target_map=targets.set_index('sequence_sha256').target_feature_index.to_dict()
    key_map={k:target_map[v[0]] for k,v in sequence.items() if len(v)==1 and v[0] in target_map}
    drug_map=drugs.set_index('model_ligand_smiles').drug_feature_index.to_dict()
    dc['d']=dc.model_ligand_smiles.map(drug_map)
    dc['t']=dc.sequence_key.map(key_map)
    dc=dc.dropna(subset=['d','t'])
    assert not dc.duplicated(['d','t']).any()
    dc_matrix=np.full((720,384),np.nan,np.float32)
    dc_matrix[dc.d.to_numpy(int),dc.t.to_numpy(int)]=dc.drugclip_cosine_mean_v1
    qi=np.flatnonzero(np.isfinite(dc_matrix).any(1))
    ti=np.flatnonzero(np.isfinite(dc_matrix).any(0))
    assert len(qi)>0 and len(ti)>0 and np.isfinite(dc_matrix[np.ix_(qi,ti)]).all()
    save_scores('drugclip',dc_matrix,dc_matrix,{'kind':'original_six_fold_mean_cosine','source':str(dcpath.relative_to(ROOT)),
        'source_sha256':sha256(dcpath),'training_exposure':'public DrugCLIP pretraining overlap with known controls not excluded',
        'target_identity_source':str(readiness_path.relative_to(ROOT)),'target_identity_source_sha256':sha256(readiness_path),
        'score_scope':'exact-SMILES and exact-sequence mapped rectangular subset','old_drugs':len(qi),'targets':len(ti)})
    np.savez_compressed(OUT/'DRUGCLIP_COMMON_SCOPE.npz',drug_indices=qi,target_indices=ti)


def evaluate(protocol,drugs,targets):
    evaluation=OUT/'evaluation'
    evaluation.mkdir(exist_ok=True)
    with np.load(OUT/'LABELS_AND_SCOPES.npz') as f:
        known,observed=f['known_relationship'],f['observed_binary']
        historical_test=f['historical_test_drugs']
        absent=f['absent_drugs']
    common=np.load(OUT/'DRUGCLIP_COMMON_SCOPE.npz')
    models=['production_independent','v6_directional_full_fit','v3_support_ensemble','morgan_ensemble',
        'bermol_ensemble','signed_ensemble','dtiam','conplex','positive_nearest','pn_logistic','drugclip']
    # Historical-test-drug scope applies to V3/R1/support models only. Other
    # systems have different fitting boundaries, so they are omitted there.
    family_models={'v3_support_ensemble','morgan_ensemble','bermol_ensemble','signed_ensemble','positive_nearest','pn_logistic'}
    scopes={'project_720x384':(np.arange(720),np.arange(384)),
        'drugclip_common':(common['drug_indices'],common['target_indices']),
        'original_test_old_drugs':(np.flatnonzero(historical_test),np.arange(384)),
        'old_drugs_absent_from_binary_relations':(np.flatnonzero(absent),np.arange(384))}
    all_results,all_ranks,toplists=[],[],[]
    for scope,(di,ti) in scopes.items():
        scope_results={}
        scope_known=known[np.ix_(di,ti)]
        scope_observed=observed[np.ix_(di,ti)]
        for name in models:
            if name=='drugclip' and scope!='drugclip_common':
                continue
            if scope in ['original_test_old_drugs','old_drugs_absent_from_binary_relations'] and name not in family_models:
                continue
            prediction=np.load(OUT/'scores'/f'{name}.npz')
            forward=prediction['drug_to_target'][np.ix_(di,ti)]
            reverse=prediction['target_to_drug'][np.ix_(di,ti)]
            dnames=drugs.ligand_inchikey.to_numpy()[di]
            tnames=targets.target_chembl_id.to_numpy()[ti]
            result={}
            for direction,y,s,q,c,labels in [('drug_to_target',scope_known,forward,dnames,tnames,scope_observed),
                ('target_to_drug',scope_known.T,reverse.T,tnames,dnames,scope_observed.T)]:
                summary,queries,relations=known_ranking(y,s,q,c)
                measured,measured_queries=observed_ranking(labels,s,q,c)
                result[direction]={'known':summary,'observed':measured}
                if len(queries):
                    queries.to_csv(evaluation/f'{scope}__{name}__{direction}__queries.csv',index=False)
                if len(measured_queries):
                    measured_queries.to_csv(evaluation/f'{scope}__{name}__{direction}__observed_queries.csv',index=False)
                if len(relations):
                    relations.insert(0,'direction',direction)
                    relations.insert(0,'model',name)
                    relations.insert(0,'scope',scope)
                    all_ranks.append(relations)
                if scope=='project_720x384':
                    for query,row in zip(q,s):
                        order=np.lexsort((c,-row))[:20]
                        toplists.extend({'model':name,'direction':direction,'query_id':query,
                            'candidate_id':c[j],'rank':rank,'score':float(row[j])} for rank,j in enumerate(order,1))
                all_results.append({'scope':scope,'model':name,'direction':direction,
                    'known_ap':summary['macro_ap'],'known_mrr':summary['macro_mrr'],
                    'known_recall_5':summary['macro_recall_at_5'],'known_recall_10':summary['macro_recall_at_10'],
                    'known_recall_20':summary['macro_recall_at_20'],'known_ndcg_20':summary['macro_ndcg_at_20'],
                    'known_queries':summary['evaluated_queries'],'known_pairs':summary['known_relationships'],
                    'candidate_count':summary['candidates_per_query'],'observed_ap':measured['macro_ap'],
                    'observed_auroc':measured['macro_auroc'],'observed_two_class_queries':measured['two_class_queries']})
            scope_results[name]=result
        write_json(evaluation/f'{scope}__METRICS.json',scope_results)
        print(json.dumps({'evaluated_scope':scope,'drugs':len(di),'targets':len(ti),'known_pairs':int(scope_known.sum()),'models':len(scope_results)}),flush=True)
    summary=pd.DataFrame(all_results)
    summary.to_csv(evaluation/'SUMMARY.csv',index=False)
    pd.concat(all_ranks,ignore_index=True).to_csv(evaluation/'KNOWN_RELATION_RANKS.csv.gz',index=False)
    pd.DataFrame(toplists).to_csv(evaluation/'TOP20_BOTH_DIRECTIONS.csv.gz',index=False)
    write_json(OUT/'STATUS.json',{'status':'COMPLETE','task':'known old-drug/target bidirectional ranking',
        'scopes':{k:{'drugs':len(v[0]),'targets':len(v[1])} for k,v in scopes.items()},
        'models_evaluated':len(models),'new_training':False,'original_protocol_sha256':sha256(OUT/'PROTOCOL.json'),
        'claim':'known-relationship recovery; inherited model training exposure and retrospective label coverage explicitly reported'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    protocol,drugs,targets,pairs,evidence=load_inputs()
    if not args.evaluate_only:
        neural_scores(protocol,drugs,targets,evidence)
        existing_scores(protocol,drugs,targets,pairs,evidence)
    evaluate(protocol,drugs,targets)


if __name__=='__main__':
    main()

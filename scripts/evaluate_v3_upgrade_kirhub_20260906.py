#!/usr/bin/env python3
"""Evaluate the frozen J candidate on KiRHub; no fitting or model selection."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.ranking_audit import measured_ranking,query_bootstrap
from build_biomaster_odti_v4_features import sha256,write_json

OUT=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/kirhub'
OLD=ROOT/'outputs/biomaster_old_drug_bidirectional_20260906'
UPGRADE=ROOT/'outputs/biomaster_v3_incremental_20260906'
SOURCE=ROOT/'outputs/evidence_routing_compute_execution_20260808_v1/old_drug_innovation_v7/OLD_DRUG_KIRHUB_PAIR_BENCHMARK_V7.csv'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    candidate=json.loads((UPGRADE/'CANDIDATE.json').read_text())
    assert sha256(ROOT/candidate['current_scores'])==candidate['scores_sha256']
    benchmark=pd.read_csv(SOURCE)
    drugs=pd.read_csv(OLD/'OLD_DRUGS_720.csv')
    targets=pd.read_csv(OLD/'TARGETS_384.csv.gz')
    di=benchmark.ligand_inchikey.map(drugs.set_index('ligand_inchikey').drug_feature_index).to_numpy(int)
    ti=benchmark.target_chembl_id.map(targets.set_index('target_chembl_id').target_feature_index).to_numpy(int)
    assert len(benchmark)==8058 and not benchmark.pairId.duplicated().any()
    benchmark['drug_id']=benchmark.ligand_inchikey
    benchmark['target_id']=benchmark.target_chembl_id
    benchmark['label']=benchmark.kirhub_wt_min_residual_activity_pct_1uM.le(30).astype(int)
    known=np.load(OLD/'LABELS_AND_SCOPES.npz')
    binary_train=known['seen_training_pair'][di,ti]
    known_train=known['known_relationship'][di,ti]&drugs.training_role.to_numpy()[di].astype(str).__eq__('train')
    benchmark['seen_v3_binary_training_pair']=binary_train
    benchmark['seen_J_known_relation_training_pair']=known_train
    benchmark['old_drug_training_role']=drugs.training_role.to_numpy()[di]
    scopes={'historical_strict_2823':benchmark.kirhub_frozen_unreported_scope.to_numpy(bool),
            'single_construct_7505':benchmark.kirhub_wt_construct_count.eq(1).to_numpy(),
            'single_construct_unseen_by_J':benchmark.kirhub_wt_construct_count.eq(1).to_numpy()&~(binary_train|known_train),
            'all_mapped_8058':np.ones(len(benchmark),bool)}
    assert scopes['historical_strict_2823'].sum()==2823 and benchmark.label[scopes['historical_strict_2823']].sum()==202
    matrices={}
    for name in ['frozen_v3','J_old_relation_pretrained']:
        matrices[name]=np.load(UPGRADE/'scores'/f'{name}.npz')['old']
    for name in ['production_independent','v6_directional_full_fit','dtiam','positive_nearest','pn_logistic','conplex']:
        with np.load(OLD/'scores'/f'{name}.npz') as f:
            matrices[name]=np.stack([f['drug_to_target'],f['target_to_drug']],-1)
    score_sources={str((UPGRADE/'scores'/f'{n}.npz').relative_to(ROOT)):sha256(UPGRADE/'scores'/f'{n}.npz')
                   for n in ['frozen_v3','J_old_relation_pretrained']}
    score_sources.update({str((OLD/'scores'/f'{n}.npz').relative_to(ROOT)):sha256(OLD/'scores'/f'{n}.npz') for n in matrices if n not in ['frozen_v3','J_old_relation_pretrained']})
    protocol={'status':'FROZEN','candidate_sha256':sha256(UPGRADE/'CANDIDATE.json'),'labels':'residual activity <=30% at 1uM',
        'fitting':False,'claim':'retrospective external activity audit; dataset previously inspected; no KiRHub functional adapter fitted here',
        'source_sha256':{str(SOURCE.relative_to(ROOT)):sha256(SOURCE),**score_sources},
        'scopes':{name:{'rows':int(mask.sum()),'positives':int(benchmark.label[mask].sum()),
            'drugs':int(benchmark.drug_id[mask].nunique()),'targets':int(benchmark.target_id[mask].nunique()),
            'pairs_seen_v3_binary_train':int(binary_train[mask].sum()),'pairs_seen_J_known_train':int(known_train[mask].sum())} for name,mask in scopes.items()}}
    if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text())==protocol
    else:write_json(OUT/'PROTOCOL.json',protocol)
    summary,queries=[],[]
    for scope,mask in scopes.items():
        frame=benchmark.loc[mask]
        for name,matrix in matrices.items():
            for i,direction in enumerate(['d2t','t2d']):
                scores=matrix[di,ti,i]
                benchmark[f'{name}__{direction}']=scores
                result,q=measured_ranking(frame,scores[mask],direction)
                summary.append({'scope':scope,'model':name,'direction':direction,**result})
                q.insert(0,'direction',direction);q.insert(0,'model',name);q.insert(0,'scope',scope);queries.append(q)
    result=pd.DataFrame(summary)
    strict=result[(result.scope=='historical_strict_2823')&(result.direction=='d2t')].set_index('model')
    assert abs(strict.loc['dtiam','macro_ap']-.3867)<.0001
    assert abs(strict.loc['production_independent','macro_ap']-.4124)<.0001
    q=pd.concat(queries,ignore_index=True)
    intervals=[]
    for scope in scopes:
        for direction in ['d2t','t2d']:
            select=q[(q.scope==scope)&(q.direction==direction)]
            for base in ['frozen_v3','dtiam','production_independent','positive_nearest']:
                intervals.append({'scope':scope,'direction':direction,'candidate':'J_old_relation_pretrained','control':base,
                    **query_bootstrap(select[select.model=='J_old_relation_pretrained'],select[select.model==base])})
    result.to_csv(OUT/'METRICS.csv',index=False);q.to_csv(OUT/'QUERY_METRICS.csv.gz',index=False)
    benchmark.to_csv(OUT/'PREDICTIONS.csv.gz',index=False)
    write_json(OUT/'PAIRED_COMPARISONS.json',intervals)
    write_json(OUT/'STATUS.json',{'status':'COMPLETE','models':len(matrices),'scopes':len(scopes),'historical_DTIAM_and_production_reproduced':True})
    print(result[(result.scope=='historical_strict_2823')][['model','direction','macro_ap','macro_recall_20']].to_string(index=False))


if __name__=='__main__':main()

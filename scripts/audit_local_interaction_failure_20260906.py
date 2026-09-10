#!/usr/bin/env python3
"""Post-selection development-only diagnostics; never train or reselect weights."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.anchored_interaction import AnchoredFeatureBank
from biomaster.model_registry import build_model,infer_family
from biomaster.best_model_training import old_labels
from biomaster.ranking_audit import risk_set_ranking,query_bootstrap
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json
BASE=ROOT/'outputs/biomaster_best_model_20260906'
OUT=ROOT/'outputs/biomaster_local_failure_audit_20260906'


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    protocol=dict(role='post-selection diagnosis on previously used 2019-2020/2021-2022 development labels',
        conditions=dict(site=['full','local_off'],geometry=['full','local_off','geometry_off','coordinates_shuffled']),
        cutoffs=[2018,2020],seeds=[20260921,20260922,20260923],precision='bf16 autocast',batch_size=256,
        local_off='bypass the correction but retain the final jointly-trained global weights and shared readout',
        coordinates_shuffled='one deterministic permutation of CA rows within each target geometry mask; residue identities/masks/quality unchanged',
        no_training=True,changes_selection=False,uses_2023_2025_or_kirhub_labels=False,
        selected_bundle=file_identity(BASE/'retargetmap_selected_v1/MANIFEST.json'),producer=file_identity(Path(__file__)))
    pp=OUT/'PROTOCOL.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('diagnostic protocol drift')
    write_json(pp,protocol)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    target=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    coverage=pd.read_csv(OUTPUT/'TARGET_COVERAGE.csv').sort_values('target_feature_index')
    bank=AnchoredFeatureBank(OUTPUT,BASE/'data/supplemental_features',SOURCE,'drugclip_morgan',local=True)
    native_ca=bank.ca.clone();native_mask=bank.geom_mask.clone();shuffled=native_ca.clone()
    rng=np.random.default_rng(20260906)
    for ti in range(len(target)):
        ids=torch.where(native_mask[ti])[0];perm=torch.tensor(rng.permutation(len(ids)),device=ids.device)
        shuffled[ti,ids]=native_ca[ti,ids[perm]]
    nd,nt=len(old),len(target);drug_axis=old.drug_feature_index.to_numpy(int)
    stages={};coverage_rows=[]
    for cutoff in protocol['cutoffs']:
        val=pd.read_csv(BASE/'data'/f'roll_{cutoff}'/'VALIDATION.csv.gz');y=old_labels(val,old,nt)
        qd=np.flatnonzero(y.any(1));qt=np.flatnonzero(y.any(0));needed=np.zeros_like(y)
        needed[qd,:]=True;needed[:,qt]=True;di,ti=np.where(needed)
        stages[cutoff]=dict(labels=y,risk=np.load(BASE/'data'/f'roll_{cutoff}'/'RISK.npy'),di=di,ti=ti)
        positives=np.where(y)[1]
        coverage_rows.append(dict(cutoff=cutoff,positive_pairs=int(y.sum()),target_queries=len(qt),
            positive_pairs_with_geometry=int((coverage.selected_pocket_residues.to_numpy()[positives]>0).sum()),
            positive_pairs_in_truncated_pocket_unions=int((coverage.pocket_union_residues.to_numpy()[positives]>96).sum()),
            selected_pocket_fraction_on_positive_targets=float((coverage.selected_pocket_residues/coverage.selected_actual_residues).to_numpy()[positives].mean())))
    with torch.no_grad():
        distance=torch.cdist(native_ca,native_ca)
        valid=native_mask[:,:,None]&native_mask[:,None,:]&(distance>0)&(distance<=24)
        degree=valid.sum(-1)[native_mask].cpu().numpy()
    write_json(OUT/'FEATURE_DIAGNOSIS.json',dict(targets=nt,geometry_targets=int(native_mask.any(1).sum()),
        selected_residue_tokens=int(coverage.selected_actual_residues.sum()),pocket_residue_tokens=int(coverage.selected_pocket_residues.sum()),
        non_pocket_fill_tokens=int((coverage.selected_actual_residues-coverage.selected_pocket_residues).sum()),
        targets_with_union_over_96=int(coverage.pocket_union_residues.gt(96).sum()),
        mean_selected_pocket_fraction=float((coverage.selected_pocket_residues/coverage.selected_actual_residues).mean()),
        geometry_neighbor_degree_quantiles={str(q):float(np.quantile(degree,q)) for q in [0,.1,.5,.9,1]},
        old_validation=coverage_rows,site_member_feature_passed_to_network=False))
    summaries=[];frames=[];sensitivity=[];jobs=sum(len(x)*6 for x in protocol['conditions'].values());done=0
    for cutoff in protocol['cutoffs']:
        stage=stages[cutoff];di,ti=stage['di'],stage['ti']
        for seed in protocol['seeds']:
            for variant,conditions in protocol['conditions'].items():
                checkpoint=BASE/'anchored'/f'cutoff_{cutoff}'/variant/f'seed_{seed}'/'BEST.pt'
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model=build_model(infer_family(state),state['config']).cuda().eval();model.load_state_dict(state['model'],strict=True)
                identity=dict(checkpoint=file_identity(checkpoint),protocol=file_identity(pp))
                pred={}
                for condition in conditions:
                    folder=OUT/f'cutoff_{cutoff}'/variant/f'seed_{seed}'/condition;folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'RESULT.json').exists():
                        result=json.loads((folder/'RESULT.json').read_text())
                        if result['identity']!=identity:raise ValueError('diagnostic result drift')
                        values=np.load(folder/'SCORES.npy')
                    else:
                        bank.ca=shuffled if condition=='coordinates_shuffled' else native_ca
                        bank.geom_mask=torch.zeros_like(native_mask) if condition=='geometry_off' else native_mask
                        values=np.empty((len(di),2),np.float32);ratios=[];start=time.monotonic()
                        with torch.inference_mode():
                            for lo in range(0,len(di),256):
                                sl=slice(lo,min(lo+256,len(di)))
                                b=bank.batch(drug_axis[di[sl]],ti[sl],'global' if condition=='local_off' else variant)
                                with torch.autocast('cuda',dtype=torch.bfloat16):
                                    if condition=='local_off':
                                        d=model.drug_global(torch.cat([b['drug_global'],b['drug_graph_mean'],b['pretrained_available'][:,None]],-1))
                                        t=model.target_global(b['target_global']);g=model.fusion(model.global_fusion(torch.cat([d,t,d*t,(d-t).abs()],-1)))
                                        scores=model.readout(model.shared(g))
                                    else:
                                        captures={};handles=[]
                                        if condition=='full':
                                            handles=[model.fusion.register_forward_hook(lambda _m,_a,v:captures.__setitem__('g',v)),
                                                model.refine.register_forward_hook(lambda _m,_a,v:captures.__setitem__('delta',v))]
                                        scores=model(b)
                                        for h in handles:h.remove()
                                        if captures:
                                            ratios.extend((captures['delta'].float().norm(dim=1)/captures['g'].float().norm(dim=1).clamp_min(1e-8)).cpu().tolist())
                                values[sl]=scores.float().cpu().numpy()
                        if not np.isfinite(values).all():raise ValueError('nonfinite diagnostic scores')
                        dense=np.zeros((nd,nt,2),np.float32);dense[di,ti]=values;metrics={};qframes=[]
                        for head,direction in enumerate(['d2t','t2d']):
                            y=stage['labels'];risk=stage['risk'];s=dense[:,:,head]
                            qi=np.arange(nd);ci=np.arange(nt)
                            if head:y,risk,s,qi,ci=y.T,risk.T,s.T,ci,qi
                            m,q,_=risk_set_ranking(y,s,risk,qi,ci);metrics[direction]=m;q['direction']=direction;qframes.append(q)
                        comp=.35*(metrics['d2t']['macro_ap']+metrics['t2d']['macro_ap'])+.15*(metrics['d2t']['macro_recall_20']+metrics['t2d']['macro_recall_20'])
                        result=dict(identity=identity,cutoff=cutoff,seed=seed,variant=variant,condition=condition,
                            composite=comp,metrics=metrics,seconds=time.monotonic()-start,
                            correction_to_global_norm_mean=float(np.mean(ratios)) if ratios else None)
                        pd.concat(qframes).to_csv(folder/'QUERY_METRICS.csv',index=False)
                        np.save(folder/'SCORES.npy',values);write_json(folder/'RESULT.json',result)
                    pred[condition]=values
                    if condition=='full':
                        prior=json.loads(checkpoint.with_name('RESULT.json').read_text())['validation']
                        # Same precision, but packed evaluation changes batch grouping.
                        result['original_validation_composite']=prior['selection']
                    summaries.append(dict(cutoff=cutoff,seed=seed,variant=variant,condition=condition,composite=result['composite'],
                        d2t_ap=result['metrics']['d2t']['macro_ap'],t2d_ap=result['metrics']['t2d']['macro_ap'],
                        d2t_r20=result['metrics']['d2t']['macro_recall_20'],t2d_r20=result['metrics']['t2d']['macro_recall_20'],
                        correction_ratio=result['correction_to_global_norm_mean']))
                    q=pd.read_csv(folder/'QUERY_METRICS.csv',dtype={'query_id':str});q=q.assign(cutoff=cutoff,seed=seed,variant=variant,condition=condition);frames.append(q)
                    done+=1;write_json(OUT/'STATUS.json',dict(status='RUNNING',completed=done,jobs=jobs,
                        utc=datetime.now(timezone.utc).isoformat(),variant=variant,condition=condition,cutoff=cutoff,seed=seed))
                    print(json.dumps(summaries[-1]),flush=True)
                for condition in conditions[1:]:
                    difference=pred[condition]-pred['full']
                    sensitivity.append(dict(cutoff=cutoff,seed=seed,variant=variant,condition=condition,
                        mean_absolute_logit_change=float(np.abs(difference).mean()),max_absolute_logit_change=float(np.abs(difference).max())))
                del model,state;torch.cuda.empty_cache()
    table=pd.DataFrame(summaries);table.to_csv(OUT/'RUNS.csv',index=False)
    cols=['composite','d2t_ap','t2d_ap','d2t_r20','t2d_r20','correction_ratio']
    table.groupby(['variant','condition'])[cols].mean().to_csv(OUT/'SUMMARY.csv')
    pd.DataFrame(sensitivity).to_csv(OUT/'SCORE_SENSITIVITY.csv',index=False)
    qall=pd.concat(frames,ignore_index=True);comparisons=[]
    for (cutoff,variant),group in qall.groupby(['cutoff','variant']):
        for direction in ['d2t','t2d']:
            by=group[group.direction.eq(direction)].groupby(['condition','query_id'],as_index=False).ap.mean()
            for condition in protocol['conditions'][variant][1:]:
                comparisons.append(dict(cutoff=int(cutoff),variant=variant,direction=direction,condition=condition,
                    effect='condition minus full',**query_bootstrap(by[by.condition.eq(condition)],by[by.condition.eq('full')])))
    write_json(OUT/'PAIRED_QUERY_EFFECTS.json',comparisons)
    if file_identity(BASE/'retargetmap_selected_v1/MANIFEST.json')!=protocol['selected_bundle']:raise ValueError('selected release changed')
    write_json(OUT/'STATUS.json',dict(status='COMPLETE',completed=done,jobs=jobs,protocol=file_identity(pp),no_model_weights_changed=True))


if __name__=='__main__':main()

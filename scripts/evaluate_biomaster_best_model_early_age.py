#!/usr/bin/env python3
"""Historical approval sensitivity on development windows only, not selection."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.best_model_training import SupplementedFeatureBank,RollingStage
from biomaster.unified_interaction import UnifiedInteraction,UnifiedConfig
from biomaster.refined_interaction import RefinedInteraction,RefinedConfig
from biomaster.ranking_audit import risk_set_ranking
from biomaster.odti_pockets_v3 import file_identity
from train_biomaster_unified_interaction import predict
from prepare_biomaster_unified_interaction import write_json,SOURCE,OUTPUT
from summarize_biomaster_best_model_development import records,OUT


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    approval_path=SOURCE/'REGISTRY_APPROVAL_AUDIT.csv'
    approval=pd.read_csv(approval_path).sort_values('old_drug_index')
    stages={c:RollingStage(OUT/'data',SOURCE,c) for c in [2018,2020]}
    protocol=dict(status='FROZEN_BEFORE_AGE_SENSITIVITY',role='development sensitivity; not architecture selection',
        approval_definition='ChEMBL37 first_approval metadata matched by exact full InChIKey; not independently certified FDA dates',
        scope='both drug queries and target-to-drug candidate drugs approved by the training cutoff; missing dates excluded',
        source=file_identity(approval_path),cutoffs=[2018,2020],producer=file_identity(Path(__file__)))
    protocol_path=OUT/'EARLY_AGE_PROTOCOL.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('age protocol drift')
    else:write_json(protocol_path,protocol)
    counts=[]
    for cutoff,stage in stages.items():
        if not np.array_equal(approval.drug_feature_index,stage.old.drug_feature_index):raise ValueError('approval/feature axis mismatch')
        eligible=approval.chembl_first_approval.le(cutoff).to_numpy()
        counts.append(dict(cutoff=cutoff,approved_drugs=int(eligible.sum()),
            excluded_later=int(approval.chembl_first_approval.gt(cutoff).sum()),
            excluded_missing=int(approval.chembl_first_approval.isna().sum()),
            validation_positive_pairs=int(stage.val_known[eligible].sum()),
            positive_drug_queries=int(stage.val_known[eligible].any(1).sum()),
            positive_target_queries=int(stage.val_known[eligible].any(0).sum())))
    write_json(OUT/'EARLY_AGE_COVERAGE.json',dict(counts=counts,protocol=file_identity(protocol_path)))
    bank=None;bank_local=None;rows=[]
    for cutoff,variant,seed,folder in records():
        if not (folder/'RESULT.json').exists():continue
        destination=OUT/'age_sensitivity'/f'roll_{cutoff}'/variant/f'seed_{seed}'
        destination.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'BEST.pt'
        ident=dict(checkpoint=file_identity(checkpoint),protocol=file_identity(protocol_path))
        result_path=destination/'RESULT.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result['identity']!=ident:raise ValueError('age result checkpoint drift')
        else:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['cutoff']!=cutoff or state['seed']!=seed:raise ValueError('checkpoint stage mismatch')
            if state.get('refinement'):
                model=RefinedInteraction(RefinedConfig(**state['config']))
            else:model=UnifiedInteraction(UnifiedConfig(**state['config']))
            model.load_state_dict(state['model'],strict=True);model.cuda().eval()
            if bank_local!=model.is_local:
                del bank
                bank=SupplementedFeatureBank(OUTPUT,OUT/'data/supplemental_features',local=model.is_local)
                bank_local=model.is_local
            stage=stages[cutoff];eligible=np.flatnonzero(approval.chembl_first_approval.le(cutoff))
            known=stage.val_known[eligible];risk=stage.risk[eligible];metrics={};query_frames=[]
            for head,direction in enumerate(['d2t','t2d']):
                k=known if head==0 else known.T;r=risk if head==0 else risk.T
                q=np.flatnonzero(k.any(1));n=k.shape[1]
                if head==0:
                    d=np.repeat(stage.old.drug_feature_index.to_numpy()[eligible[q]],n);t=np.tile(np.arange(n),len(q))
                    query_ids=eligible[q];candidate_ids=np.arange(stage.nt)
                else:
                    d=np.tile(stage.old.drug_feature_index.to_numpy()[eligible],len(q));t=np.repeat(q,n)
                    query_ids=q;candidate_ids=eligible
                s=predict(model,bank,d,t,256)[:,head].reshape(len(q),n)
                m,queries,_=risk_set_ranking(k[q],s,r[q],query_ids,candidate_ids)
                metrics[direction]=m;queries['direction']=direction;query_frames.append(queries)
            result=dict(cutoff=cutoff,variant=variant,seed=seed,identity=ident,metrics=metrics,
                approved_candidate_drugs=len(eligible),architecture_selection_use=False)
            pd.concat(query_frames,ignore_index=True).to_csv(destination/'QUERY_METRICS.csv',index=False)
            write_json(result_path,result);del model
        for direction,m in result['metrics'].items():
            rows.append(dict(cutoff=cutoff,variant=variant,seed=seed,direction=direction,
                ap=m['macro_ap'],r20=m['macro_recall_20'],queries=m['positive_queries'],positive_pairs=m['positive_pairs'],
                approved_candidate_drugs=result['approved_candidate_drugs']))
    pd.DataFrame(rows).to_csv(OUT/'EARLY_AGE_METRICS.csv',index=False)
    print(json.dumps(dict(completed_runs=len(rows)//2,coverage=counts)),flush=True)


if __name__=='__main__':main()

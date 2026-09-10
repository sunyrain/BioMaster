#!/usr/bin/env python3
"""Approval-age sensitivity for the selected global/final development recipe."""
import argparse
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
from biomaster.best_model_training import RollingStage
from biomaster.ranking_audit import risk_set_ranking
from biomaster.odti_pockets_v3 import file_identity
from train_biomaster_unified_interaction import predict
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json

OUT=ROOT/'outputs/biomaster_best_model_20260906'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['global','final'],default='global');args=p.parse_args()
    selected=json.loads((OUT/'GLOBAL_PARENT_SELECTION.json').read_text())
    name=selected['selected']['name'];runs=selected['parents']
    if args.phase=='final':
        final=json.loads((OUT/'FINAL_ARCHITECTURE_SELECTION.json').read_text())
        if final['selected_variant']!='parent':
            name+='__'+final['selected_variant'];runs=[]
            for cutoff in [2018,2020]:
                for seed in [20260921,20260922,20260923]:
                    folder=OUT/'anchored'/f'cutoff_{cutoff}'/final['selected_variant']/f'seed_{seed}'
                    runs.append(dict(cutoff=cutoff,seed=seed,checkpoint=file_identity(folder/'BEST.pt')))
    approved=pd.read_csv(SOURCE/'REGISTRY_APPROVAL_AUDIT.csv').sort_values('old_drug_index')
    stages={c:RollingStage(OUT/'data',SOURCE,c) for c in [2018,2020]}
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    bank=None;local_state=None;rows=[]
    for run in runs:
        cutoff=run['cutoff'];seed=run['seed'];stage=stages[cutoff]
        path=Path(run['checkpoint']['path']);assert file_identity(path)==run['checkpoint']
        folder=OUT/'selected_development_age'/args.phase/f'cutoff_{cutoff}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        ident=dict(checkpoint=run['checkpoint'],approval_source=file_identity(SOURCE/'REGISTRY_APPROVAL_AUDIT.csv'),
            age_protocol=file_identity(OUT/'EARLY_AGE_PROTOCOL.json'),producer=file_identity(Path(__file__)))
        if (folder/'RESULT.json').exists():
            result=json.loads((folder/'RESULT.json').read_text())
            if result['identity']!=ident:raise ValueError('age result identity changed')
        else:
            state=torch.load(path,map_location='cpu',weights_only=False)
            model=build_model(infer_family(state),state['config']).cuda().eval();model.load_state_dict(state['model'],strict=True)
            if model.is_local!=local_state:
                del bank
                bank=AnchoredFeatureBank(OUTPUT,OUT/'data/supplemental_features',SOURCE,selected['selected']['representation'],local=model.is_local)
                local_state=model.is_local
            assert np.array_equal(approved.drug_feature_index,stage.old.drug_feature_index)
            eligible=np.flatnonzero(approved.chembl_first_approval.le(cutoff));known=stage.val_known[eligible];risk=stage.risk[eligible]
            metrics={};frames=[]
            for head,direction in enumerate(['d2t','t2d']):
                y=known if head==0 else known.T;r=risk if head==0 else risk.T
                q=np.flatnonzero(y.any(1));n=y.shape[1]
                if head==0:
                    d=np.repeat(stage.old.drug_feature_index.to_numpy()[eligible[q]],n);t=np.tile(np.arange(n),len(q))
                    query_ids=eligible[q];candidates=np.arange(stage.nt)
                else:
                    d=np.tile(stage.old.drug_feature_index.to_numpy()[eligible],len(q));t=np.repeat(q,n)
                    query_ids=q;candidates=eligible
                scores=predict(model,bank,d,t,128)[:,head].reshape(len(q),n)
                m,queries,_=risk_set_ranking(y[q],scores,r[q],query_ids,candidates)
                metrics[direction]=m;queries['direction']=direction;frames.append(queries)
            result=dict(identity=ident,model=name,cutoff=cutoff,seed=seed,approved_drugs=len(eligible),metrics=metrics,used_for_selection=False)
            pd.concat(frames,ignore_index=True).to_csv(folder/'QUERY_METRICS.csv',index=False)
            write_json(folder/'RESULT.json',result);del model
        for direction,m in result['metrics'].items():
            rows.append(dict(model=name,cutoff=cutoff,seed=seed,direction=direction,approved_drugs=result['approved_drugs'],
                ap=m['macro_ap'],r20=m['macro_recall_20'],queries=m['positive_queries'],positive_pairs=m['positive_pairs']))
    table=pd.DataFrame(rows);table.to_csv(OUT/f'SELECTED_{args.phase.upper()}_EARLY_AGE.csv',index=False)
    print(table.groupby(['cutoff','direction'])[['ap','r20']].mean().to_string())


if __name__=='__main__':main()

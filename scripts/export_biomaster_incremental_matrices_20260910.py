#!/usr/bin/env python3
"""Export all predeclared increment seeds as research matrices, never production."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.model_registry import build_model
from biomaster.portable_ranker_v2 import CatalogRanker,digest
from scripts.extend_biomaster_matrix_20260910 import BUNDLE,OUT as MATRIX,write_json
from scripts.retrain_biomaster_bindingdb_20260910 import OUT as TRAIN


def main():
    result=json.loads((TRAIN/'SUMMARY.json').read_text())
    assert result['status']=='COMPLETE_RESEARCH_INCREMENTAL_COMPARISON'
    protocol=json.loads((TRAIN/'PROTOCOL.json').read_text())
    inputs=json.loads((TRAIN/'FITTING_INPUTS.json').read_text())
    assert digest(TRAIN/'PROTOCOL.json')==inputs['protocol_sha256']
    assert digest(TRAIN/'FROZEN_SPLIT.csv')==inputs['split_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ranker=CatalogRanker(BUNDLE,device='cuda',verify=True)
    targets=pd.read_csv(MATRIX/'TARGET_INDEX.csv.gz',keep_default_na=False)
    features=torch.tensor(np.load(MATRIX/'TARGET_GLOBAL.npy'),device='cuda')
    base=np.load(MATRIX/'DIRECTIONAL_LOGITS.npy')
    drugs=ranker.drugs[['drug_id','name']]
    drug_features={k:v for k,v in ranker.features.items() if k!='target_global'}
    reports=[]
    for seed in protocol['seeds']:
        folder=TRAIN/f'bindingdb_increment_seed_{seed}'
        path=folder/'model.pt';state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['protocol_sha256']==inputs['protocol_sha256'] and state['split_sha256']==inputs['split_sha256']
        model=build_model(state['architecture'],state['config']);model.load_state_dict(state['model']);model.cuda().eval()
        scores=np.empty((720,890,2),np.float32)
        with torch.inference_mode():
            for i in range(890):
                scores[:,i]=model(dict(drug_features,target_global=features[i:i+1].expand(720,-1))).cpu().numpy()
        assert np.isfinite(scores).all()
        np.save(folder/'RESEARCH_DIRECTIONAL_LOGITS_720x890.npy',scores)
        ranks=pd.DataFrame(-scores[:,:,1]).rank(axis=0,method='min').to_numpy(int)
        old_ranks=pd.DataFrame(-base[:,:,1]).rank(axis=0,method='min').to_numpy(int)
        frames=[]
        for i,r in enumerate(targets.itertuples()):
            take=np.flatnonzero(ranks[:,i]<=20)
            f=drugs.iloc[take].copy(); f['target_id']=r.uniprot_accession;f['gene_symbol']=r.gene_symbol
            f['target_chembl_id']=r.target_chembl_id;f['core_model_target']=r.core_model_target
            f['research_drug_to_target_logit']=scores[take,i,0];f['research_target_to_drug_logit']=scores[take,i,1]
            f['research_rank_720']=ranks[take,i];f['parent_rank_720']=old_ranks[take,i]
            f['status']='RESEARCH_PREDICTION_NOT_NEW_WETLAB_RECOMMENDATION'
            frames.append(f)
        pd.concat(frames).sort_values(['gene_symbol','research_rank_720']).to_csv(folder/'RESEARCH_TARGET_TOP20.csv.gz',index=False)
        report=dict(seed=seed,model_sha256=digest(path),matrix_sha256=digest(folder/'RESEARCH_DIRECTIONAL_LOGITS_720x890.npy'),
                    drugs=720,targets=890,pairs=640800,axis_indexes=str(MATRIX.relative_to(ROOT)),
                    model_selected_by_holdout=False,production_deployed=False,
                    mean_absolute_logit_change=float(np.abs(scores-base).mean()))
        write_json(folder/'MATRIX_MANIFEST.json',report);reports.append(report)
        print(json.dumps(report),flush=True)
    write_json(TRAIN/'RESEARCH_MATRICES.json',reports)
    for p,h in json.loads((MATRIX/'FROZEN_INPUTS.json').read_text()).items(): assert digest(ROOT/p)==h


if __name__=='__main__':main()

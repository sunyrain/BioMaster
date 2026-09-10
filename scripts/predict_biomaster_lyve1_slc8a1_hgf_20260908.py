#!/usr/bin/env python3
"""Audited exploratory extension of the selected model to three human targets.

Does not extend or mutate the released catalog. Uses exact full-chain ESM2
features and the target-to-drug head over the same 720 old-drug candidates.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.portable_ranker_v2 import CatalogRanker,digest
from scripts.build_biomaster_odti_target_token_features_v1 import window_bounds

BUNDLE=ROOT/'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'
OUT=ROOT/'outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf'
SOURCE=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'


def main():
    started=time.monotonic();targets=json.loads((OUT/'TARGETS.json').read_text())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    core=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz')
    for item in targets:
        if hashlib.sha256(item['sequence'].encode()).hexdigest()!=item['sequence_sha256']:raise ValueError('sequence identity mismatch')
        if core.sequence_sha256.eq(item['sequence_sha256']).any():raise ValueError('query unexpectedly in core; use catalog interface')
    # A core target encoded with the same missing-sequence pipeline verifies
    # feature compatibility against the deployed full-chain mean cache.
    existing=ROOT/'outputs/biomaster_unified_interaction_20260906/features/missing_residues'
    choices=core[core.sequence_sha256.isin([p.stem for p in existing.glob('*.npy')])].copy()
    choices['n']=choices.sequence.str.len();control=choices.sort_values('n').iloc[0]
    to_encode=targets+[dict(gene_symbol='CONTROL_'+control.gene_symbol,sequence=control.sequence,
                            sequence_sha256=control.sequence_sha256)]
    import esm
    os.environ['TORCH_HOME']='/root/autodl-tmp/.cache/torch'
    encoder,alphabet=esm.pretrained.esm2_t33_650M_UR50D();encoder=encoder.cuda().eval()
    converter=alphabet.get_batch_converter();features=[]
    with torch.inference_mode():
        for item in to_encode:
            seq=item['sequence'];path=OUT/(item['gene_symbol']+'_ESM2_RESIDUES.npy')
            if path.exists():states=np.load(path)
            else:
                total=np.zeros((len(seq),1280),np.float32);counts=np.zeros((len(seq),1),np.float32)
                for lo,hi in window_bounds(len(seq),1022,128):
                    _,_,token=converter([(item['gene_symbol'],seq[lo:hi])])
                    with torch.autocast('cuda',dtype=torch.float16):
                        h=encoder(token.cuda(),repr_layers=[33],return_contacts=False)['representations'][33][0,1:hi-lo+1].float().cpu().numpy()
                    total[lo:hi]+=h;counts[lo:hi]+=1
                if not (counts>0).all():raise ValueError('incomplete sequence encoding')
                states=(total/counts).astype(np.float16);np.save(path,states)
            if states.shape!=(len(seq),1280) or not np.isfinite(states).all():raise ValueError('invalid residue features')
            features.append(states.astype(np.float32).mean(0))
            print(json.dumps(dict(encoded=item['gene_symbol'],residues=len(seq))),flush=True)
    del encoder;torch.cuda.empty_cache()
    ranker=CatalogRanker(BUNDLE,device='cpu',verify=True)
    control_index=int(control.target_feature_index)
    packaged_index=int(np.flatnonzero(ranker.targets.native_feature_index.to_numpy()==control_index)[0])
    reference=ranker.features['target_global'][packaged_index].numpy()
    difference=float(np.max(np.abs(features[-1]-reference)))
    if difference>2e-3:raise ValueError(f'new ESM2 inputs differ from catalog protocol: {difference}')
    np.save(OUT/'TARGET_GLOBAL.npy',np.stack(features[:3]))
    results=[];neighbors=[];scores=[]
    with torch.inference_mode():
        for item,feature in zip(targets,features[:3]):
            target=torch.tensor(feature,dtype=torch.float32)
            output=[]
            for i in range(len(ranker.drugs)):
                batch={k:ranker.features[k][i:i+1] for k in ['drug_global','drug_graph_mean','pretrained_available']}
                batch['target_global']=target[None]
                output.append(ranker.model(batch).numpy()[0])
            score=np.stack(output);scores.append(score)
            if not np.isfinite(score).all():raise ValueError('nonfinite ranking score')
            frame=ranker.drugs.copy();frame['gene_symbol']=item['gene_symbol']
            frame['drug_to_target_logit']=score[:,0];frame['target_to_drug_logit']=score[:,1]
            frame=frame.sort_values(['target_to_drug_logit','drug_id'],ascending=[False,True]).reset_index(drop=True)
            frame['rank_720']=np.arange(1,len(frame)+1);frame['evidence']='UNVALIDATED_COLD_TARGET_MODEL_HYPOTHESIS'
            frame.to_csv(OUT/(item['gene_symbol']+'_RANKED_720.csv'),index=False)
            frame.head(20).to_csv(OUT/(item['gene_symbol']+'_TOP20.csv'),index=False)
            results.append(frame)
            # Descriptive embedding-space neighbors, not sequence identity or
            # proof that these targets share a pharmacological binding site.
            cos=torch.nn.functional.cosine_similarity(ranker.features['target_global'],target[None],dim=-1).numpy()
            ids=np.argsort(-cos)[:10]
            for i in ids:neighbors.append(dict(query=item['gene_symbol'],core_gene=ranker.targets.gene_symbol.iloc[i],
                core_target=ranker.targets.target_id.iloc[i],esm2_mean_cosine=float(cos[i])))
    pd.concat(results).to_csv(OUT/'ALL_2160_PREDICTIONS.csv',index=False)
    pd.DataFrame(neighbors).to_csv(OUT/'EMBEDDING_NEIGHBORS.csv',index=False)
    score_matrix=np.stack(scores)[:,:,1]
    pair_diagnostics=[]
    for i in range(3):
        for j in range(i+1,3):
            a=pd.Series(score_matrix[i]).rank();b=pd.Series(score_matrix[j]).rank()
            topi=set(np.argsort(-score_matrix[i])[:20]);topj=set(np.argsort(-score_matrix[j])[:20])
            pair_diagnostics.append(dict(target_a=targets[i]['gene_symbol'],target_b=targets[j]['gene_symbol'],
                rank_spearman=float(a.corr(b)),top20_overlap=len(topi&topj)))
    report=dict(status='EXPLORATORY_COLD_TARGET_PREDICTIONS_COMPLETE',model='selected_global_fullfit_2025',
        model_sha256=digest(BUNDLE/'model.pt'),bundle_manifest_sha256=digest(BUNDLE/'MANIFEST.json'),
        candidate_count=720,targets=[{k:v for k,v in x.items() if k!='sequence'} for x in targets],
        feature_protocol='ESM2 t33 650M layer33; overlapping 1022/128 windows; FP16 residue cache; full-chain FP32 mean',
        control_gene=control.gene_symbol,control_feature_max_abs_difference=difference,
        scoring='CPU FP32, batch=1, directional target-to-drug logit; neither binding probability nor Kd',
        not_in_downstream_training_target_catalog=True,public_encoder_exposure_unknown=True,
        external_target_performance_validated=False,
        context_full_training_model_used=False,catalog_modified=False,rank_diagnostics=pair_diagnostics,
        feature_identities={p.name:digest(p) for p in OUT.glob('*ESM2_RESIDUES.npy')},
        target_input_sha256=digest(OUT/'TARGETS.json'),
        producer_sha256=digest(Path(__file__)),seconds=time.monotonic()-started)
    (OUT/'PREDICTION_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded diagnostics of deployed architectures; no training or production writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/model_architecture_audit_20260919'
SHARED = OUT / 'latest_results/SEVEN_SHARED_COMPLETE_TARGETS.csv.gz'
MODELS = ['biomaster', 'drugclip', 'dtiam', 'conplex', 'nesso', 'probematch', 'dtbind']


def diagnose():
    shared = pd.read_csv(SHARED, float_precision='round_trip')
    rows = []
    for r in shared.itertuples():
        name = f'{r.drug_id}__{r.target_id}'
        paths = [ROOT / f'outputs/catalog_seven_models_20260916/nesso/targets/{r.target_id}/predictions/{name}/affinity.json',
                 ROOT / f'outputs/frontier_dti_20260916/nesso/predictions/{name}/affinity.json']
        path = next(p for p in paths if p.is_file())
        value = json.loads(path.read_text())
        assert abs(float(value['affinity_probability_binary']) - r.nesso) < 1e-12
        rows.append(dict(drug_id=r.drug_id, target_id=r.target_id, gene=r.gene,
                         nesso=r.nesso, nesso_pic50=6-float(value['affinity_pred_value']),
                         entropy_crop_pl=value.get('entropy_crop_pl')))
    heads = pd.DataFrame(rows)
    heads.to_csv(OUT / 'NESSO_SAME_BACKBONE_HEADS.csv.gz', index=False)
    joined = shared.merge(heads[['drug_id','target_id','nesso_pic50']], on=['drug_id','target_id'], validate='one_to_one')
    compare = []
    for query in ['target_id', 'drug_id']:
        for entity, g in joined.groupby(query):
            for a,b in [('nesso','nesso_pic50'), ('biomaster','biomaster_reverse')]:
                compare.append(dict(query=query, entity=entity, head_a=a, head_b=b, n=len(g),
                                    spearman=spearmanr(g[a],g[b]).statistic))
    compare = pd.DataFrame(compare)
    compare.to_csv(OUT / 'SAME_BACKBONE_HEAD_AGREEMENT.csv',index=False)
    hs = compare.groupby(['query','head_a','head_b'], as_index=False).agg(
        queries=('entity','size'), mean_spearman=('spearman','mean'),
        median_spearman=('spearman','median'), negative_queries=('spearman',lambda x:int(x.lt(0).sum())))
    hs.to_csv(OUT / 'SAME_BACKBONE_HEAD_SUMMARY.csv',index=False)
    usable_heads = []
    for entity, g in heads[heads.entropy_crop_pl.gt(0)].groupby('target_id'):
        usable_heads.append(dict(target_id=entity, n=len(g),
                                 spearman=spearmanr(g.nesso,g.nesso_pic50).statistic))
    pd.DataFrame(usable_heads).to_csv(OUT/'NESSO_HEADS_NONZERO_CROP_ENTROPY.csv',index=False)
    # Descriptive two-way decomposition, without labels. Not a causal shortcut score.
    components = []
    for m in MODELS + ['biomaster_reverse']:
        arr = shared.pivot(index='drug_id',columns='target_id',values=m).to_numpy()
        for space in ['native_score','pooled_percentile_rank']:
            x = arr if space == 'native_score' else pd.Series(arr.ravel()).rank(method='average',pct=True).to_numpy().reshape(arr.shape)
            grand = x.mean()
            drug = x.mean(axis=1,keepdims=True)-grand
            target = x.mean(axis=0,keepdims=True)-grand
            residual = x-grand-drug-target
            total = np.mean((x-grand)**2)
            parts = [np.mean(drug**2)/total,np.mean(target**2)/total,np.mean(residual**2)/total]
            assert abs(sum(parts)-1)<1e-10
            components.append(dict(model=m,score_space=space,drug_main_fraction=parts[0],
                                   target_main_fraction=parts[1],pair_residual_fraction=parts[2]))
    pd.DataFrame(components).to_csv(OUT/'DESCRIPTIVE_SCORE_DECOMPOSITION.csv',index=False)
    targets = pd.read_csv(ROOT/'outputs/catalog_seven_models_20260916/TARGETS.csv')
    drugs = pd.read_csv(ROOT/'outputs/catalog_seven_models_20260916/DRUGS.csv')
    limits = dict(catalog_targets=len(targets), shared_targets=shared.target_id.nunique(),
                  shared_max_protein_length=int(targets[targets.target_id.isin(shared.target_id)].protein_length.max()),
                  dtiam_catalog_proteins_over_1022=int(targets.protein_length.gt(1022).sum()),
                  probe_catalog_proteins_over_1200=int(targets.protein_length.gt(1200).sum()),
                  probe_catalog_smiles_over_100_chars=int(drugs.smiles.str.len().gt(100).sum()),
                  note='SMILES character length is descriptive, not a truncation count: the fixed word tensor is used only for batch size by the active forward path. The full molecular graph is retained; the LM branch has a separate 100-token limit.')
    info = json.loads((ROOT/'outputs/biomaster_dtiam_ab_20260912/kdki_inactive__seed_20260923/AUTOGLUON_INFO.json').read_text())
    weights = info['model_info']['WeightedEnsemble_L2']['children_info']['S1F1']['model_weights']
    result = dict(checks='PASS', shared_pairs=len(shared), shared_targets=shared.target_id.nunique(),
                  limits=limits,dtiam_ensemble_weights=weights,
                  dtiam_tree_weight=sum(v for k,v in weights.items() if not k.startswith('NeuralNet')),
                  dtiam_neural_weight=sum(v for k,v in weights.items() if k.startswith('NeuralNet')),
                  nesso_crop_entropy_zero=int(heads.entropy_crop_pl.eq(0).sum()),
                  nesso_crop_entropy_missing=int(heads.entropy_crop_pl.isna().sum()),
                  nesso_heads_excluding_zero_crop_entropy=dict(
                      queries=len(usable_heads), pairs=sum(r['n'] for r in usable_heads),
                      mean_spearman=float(np.mean([r['spearman'] for r in usable_heads])),
                      negative_queries=int(sum(r['spearman']<0 for r in usable_heads))),
                  head_agreement=hs.to_dict('records'),
                  interpretation='Same-backbone different-head comparisons do not isolate training labels or objective. Matrix decomposition is descriptive and has no experimental truth.')
    (OUT/'ARCHITECTURE_DIAGNOSTICS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def probe_stability():
    import torch
    from rdkit import RDLogger
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_catalog_dti_20260916 import Probe
    torch.set_num_threads(3)  # Match the production adapter's CPU reduction order.
    RDLogger.DisableLog('rdApp.warning')
    shared = pd.read_csv(SHARED,float_precision='round_trip')
    drugs = pd.read_csv(ROOT/'outputs/catalog_seven_models_20260916/DRUGS.csv')
    targets = pd.read_csv(ROOT/'outputs/catalog_seven_models_20260916/TARGETS.csv')
    rng = np.random.default_rng(20260919)
    selected = rng.choice(sorted(shared.drug_id.unique()),size=32,replace=False)
    drugs = drugs[drugs.drug_id.isin(selected)]
    targets = targets[targets.gene.isin(['AR','MIF'])]
    assert len(drugs)==32 and len(targets)==2
    adapter = Probe(drugs,targets,2)
    rows=[]
    for t in targets.itertuples():
        adapter.prepare_target(t)
        for d in drugs.itertuples():
            inputs=adapter.drugs[d.drug_id]
            mw,atoms,adj,dlm=inputs
            model_input=(mw,atoms,adj,adapter.pw,adapter.plm,dlm)
            base_seed=int(hashlib.sha256(f'{d.drug_id}__{t.target_id}'.encode()).hexdigest()[:8],16)
            for offset in [0,1,2,3]:
                torch.manual_seed(base_seed+offset)
                with torch.inference_mode():score=float(adapter.model(model_input)[6].softmax(-1)[0,1])
                rows.append(dict(drug_id=d.drug_id,target_id=t.target_id,gene=t.gene,seed_offset=offset,score=score))
        print('probe stability target complete',t.gene,flush=True)
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'PROBEMATCH_INFERENCE_SEED_SCORES.csv',index=False)
    initial=raw[raw.seed_offset.eq(0)].merge(shared[['drug_id','target_id','probematch']],on=['drug_id','target_id'],validate='one_to_one')
    replay_delta=float((initial.score-initial.probematch).abs().max())
    assert replay_delta<2e-6
    summaries=[]
    for gene,g in raw.groupby('gene'):
        pivot=g.pivot(index='drug_id',columns='seed_offset',values='score')
        for offset in [1,2,3]:
            summaries.append(dict(gene=gene,n=len(pivot),seed_offset=offset,
                                 spearman=spearmanr(pivot[0],pivot[offset]).statistic,
                                 max_absolute_change=float((pivot[0]-pivot[offset]).abs().max()),
                                 top10_overlap=len(set(pivot[0].nlargest(10).index)&set(pivot[offset].nlargest(10).index))))
    pd.DataFrame(summaries).to_csv(OUT/'PROBEMATCH_INFERENCE_SEED_SUMMARY.csv',index=False)
    result=dict(status='PASS',pairs=64,repeats=4,targets=['AR','MIF'],random_drugs=32,
                weights_unchanged=True,model_eval=True,device='cpu',cpu_threads=3,replay_max_delta=replay_delta,
                summaries=summaries,limitation='Two targets and 32 sampled drugs; Top10 here is within this diagnostic subset, not the full catalog. No labels or training-seed test.')
    (OUT/'PROBEMATCH_INFERENCE_SEED_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def conplex_replay():
    import h5py
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    shared=pd.read_csv(SHARED,float_precision='round_trip')
    drugs=pd.read_csv(ROOT/'outputs/catalog_seven_models_20260916/DRUGS.csv')
    drugs=drugs[drugs.drug_id.isin(shared.drug_id)].sort_values('drug_id')
    targets=pd.read_csv(ROOT/'outputs/catalog_seven_models_20260916/TARGETS.csv')
    targets=targets[targets.target_id.isin(shared.target_id)].sort_values('target_id')
    folders=[ROOT/'outputs/strict_dta_720x338_v1/conplex_cache',ROOT/'outputs/recovered_dta_720x46_v1/conplex_cache']
    def read_features(rows,column,name):
        cache=[h5py.File(folder/f'{name}_features.h5','r') for folder in folders]
        try:
            values=[]
            for row in rows.itertuples():
                key=getattr(row,column).replace('/','|')
                source=next(c for c in cache if key in c)
                values.append(source[key][:])
            return torch.tensor(np.asarray(values),dtype=torch.float32)
        finally:
            for c in cache:c.close()
    d=read_features(drugs,'smiles','Morgan');t=read_features(targets,'sequence','ProtBert')
    assert bool(torch.isfinite(d).all()) and bool(torch.isfinite(t).all())
    checkpoint=ROOT/'third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    with torch.inference_mode():
        dl=F.relu(F.linear(d,state['drug_projector.0.weight'],state['drug_projector.0.bias']))
        tl=F.relu(F.linear(t,state['target_projector.0.weight'],state['target_projector.0.bias']))
        scores=F.cosine_similarity(dl[:,None,:],tl[None,:,:],dim=-1).numpy()
        active_overlap=((dl>0).float() @ (tl>0).float().T).numpy()
    stored=shared.pivot(index='drug_id',columns='target_id',values='conplex').loc[drugs.drug_id,targets.target_id].to_numpy()
    delta=float(np.max(np.abs(scores-stored)))
    assert delta<2e-6
    zeros=scores==0
    details=[]
    for i,row in enumerate(drugs.itertuples()):
        if zeros[i].all():
            details.append(dict(drug_id=row.drug_id,drug_name=row.name,
                                input_nonzero=int(torch.count_nonzero(d[i])),
                                latent_active_dimensions=int(torch.count_nonzero(dl[i])),
                                target_latents_all_nonzero=bool((tl.norm(dim=1)>0).all()),
                                overlap_max=int(active_overlap[i].max()),zero_queries=int(zeros[i].sum())))
    pd.DataFrame(details).to_csv(OUT/'CONPLEX_ZERO_MECHANISM.csv',index=False)
    report=dict(checks='PASS',pairs=int(scores.size),max_abs_delta=delta,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                zero_scores=int(zeros.sum()),zero_score_fraction=float(zeros.mean()),
                zeros_without_shared_positive_dimensions=int((zeros & (active_overlap==0)).sum()),
                all_zero_input_drugs=int((d.norm(dim=1)==0).sum()),all_zero_input_targets=int((t.norm(dim=1)==0).sum()),
                all_zero_latent_drugs=int((dl.norm(dim=1)==0).sum()),all_zero_latent_targets=int((tl.norm(dim=1)==0).sum()),
                constant_zero_drugs=details,
                interpretation='Frozen linear/ReLU/cosine replay explains ties, not all disagreements or their accuracy.')
    (OUT/'CONPLEX_REPLAY_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe-stability',action='store_true');p.add_argument('--conplex-replay',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.probe_stability:probe_stability()
    elif args.conplex_replay:conplex_replay()
    else:diagnose()

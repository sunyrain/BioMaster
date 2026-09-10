#!/usr/bin/env python3
"""Freeze the actual project 720-old-drug by 384-target ranking audit.

The primary labels are the frozen 807 known project/MoA relationships. Unknown
pairs stay unlabelled. Binary biochemical observations form a separate audit.
Feature extension is label-free; model training and its support pool are fixed.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.odti_support_data_v3 import SupportStore
from build_biomaster_comprehensive_training_v1 import _standardize_and_fingerprint
from build_biomaster_odti_v4_features import sha256, write_json

OUT = ROOT / 'outputs/biomaster_old_drug_bidirectional_20260906'
DEPLOY = ROOT / 'outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1'
V4 = ROOT / 'outputs/biomaster_odti_v4_20260905'
V3 = ROOT / 'outputs/biomaster_v3_20260905/data'


def full_esm_means(table):
    from build_biomaster_odti_target_token_features_v1 import window_bounds
    import esm
    os.environ['TORCH_HOME'] = '/root/autodl-tmp/.cache/torch'
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.cuda().eval()
    converter = alphabet.get_batch_converter()
    output = []
    with torch.inference_mode():
        for row in table.itertuples():
            seq = row.sequence
            total, count = np.zeros((len(seq),1280),np.float32), np.zeros(len(seq),np.float32)
            for left,right in window_bounds(len(seq),1022,128):
                _,_,tokens = converter([(str(row.target_chembl_id),seq[left:right])])
                with torch.autocast('cuda',dtype=torch.float16):
                    representation = model(tokens.cuda(),repr_layers=[33],return_contacts=False)['representations'][33]
                total[left:right] += representation[0,1:right-left+1].float().cpu().numpy()
                count[left:right] += 1
            assert (count>0).all() and np.isfinite(total).all()
            output.append((total/count[:,None]).astype(np.float16).mean(0,dtype=np.float32))
            print(json.dumps({'stage':'new_target_features','completed':len(output),'total':len(table)}),flush=True)
    del model
    torch.cuda.empty_cache()
    return np.asarray(output)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    if (OUT/'PROTOCOL.json').exists():
        raise FileExistsError('The protocol is already frozen; use its existing features and scoring entrypoint')
    features_dir = OUT/'features'
    features_dir.mkdir(exist_ok=True)
    drugs = pd.read_csv(DEPLOY/'OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz').sort_values('drug_feature_index')
    targets = pd.read_csv(DEPLOY/'PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz').sort_values('target_feature_index')
    pairs = pd.read_csv(DEPLOY/'OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz',low_memory=False).sort_values(['drug_feature_index','target_feature_index'])
    assert len(drugs)==720 and len(targets)==384 and len(pairs)==720*384
    assert pairs[['drug_feature_index','target_feature_index']].drop_duplicates().shape[0] == len(pairs)
    assert np.array_equal(pairs.drug_feature_index,np.repeat(np.arange(720),384))
    assert np.array_equal(pairs.target_feature_index,np.tile(np.arange(384),720))
    data = pd.read_csv(V3/'RELATIONS_V3.csv.gz')
    audit = json.loads((V3/'DATA_MANIFEST_V3.json').read_text())
    assert sha256(V3/'RELATIONS_V3.csv.gz') == audit['prepared_relations_sha256']
    raw = pd.read_csv(audit['identity']['relations_path'],low_memory=False)
    assets = pd.read_csv(ROOT/'outputs/biomaster_odti_v4_plan_20260905/CURRENT_MOLECULE_ASSETS_V4.csv.gz')
    target_assets = pd.read_csv(ROOT/'outputs/biomaster_odti_v4_plan_20260905/CURRENT_TARGET_ASSETS_V4.csv.gz')
    base_count, target_count = len(assets),len(target_assets)
    # Derive features under exactly the training molecule standardization.
    encoded = [_standardize_and_fingerprint(s) for s in drugs.ligand_smiles]
    assert all(v[0] for v in encoded)
    drugs['model_ligand_smiles'] = [v[0] for v in encoded]
    drugs['murcko_scaffold'] = [v[1] for v in encoded]
    morgan = np.stack([np.unpackbits(np.frombuffer(v[2],dtype=np.uint8)) for v in encoded])
    np.save(features_dir/'OLD720_MORGAN2048.npy',morgan)
    pre = ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1'
    old_index = pd.read_csv(pre/'DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz').set_index('ligand_inchikey')
    selected = old_index.loc[drugs.ligand_inchikey]
    assert np.array_equal(selected.ligand_smiles,drugs.model_ligand_smiles)
    bermol = np.load(pre/'DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy')[selected.drug_feature_index.to_numpy()]
    assert selected.dtiam_bermol_available.all() and np.isfinite(bermol).all() and (np.linalg.norm(bermol,axis=1)>0).all()
    np.save(features_dir/'OLD720_BERMOL768.npy',bermol)
    global_targets = target_assets.set_index('sequence_sha256').target_feature_index.to_dict()
    targets['sequence_sha256'] = targets.sequence.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    targets['global_target_index'] = targets.sequence_sha256.map(global_targets)
    missing = targets.global_target_index.isna()
    targets.loc[missing,'global_target_index'] = np.arange(target_count,target_count+missing.sum())
    targets.global_target_index = targets.global_target_index.astype(int)
    new_target_path = features_dir/'ADDED_FULL_ESM2_MEANS.npy'
    if new_target_path.exists():
        extra = np.load(new_target_path)
        assert extra.shape == (missing.sum(),1280)
    else:
        extra = full_esm_means(targets.loc[missing])
        np.save(new_target_path,extra)
    assert np.isfinite(extra).all() and (np.linalg.norm(extra,axis=1)>0).all()
    # Match the query's complete training entity, including aliases present in
    # source InChIs or in the exact standardized model SMILES.
    keys, smiles = {},{}
    for d,e in data[['drug_feature_index','entity_key']].drop_duplicates().itertuples(index=False):
        smiles.setdefault(assets.model_ligand_smiles.iloc[d],set()).add(e)
    for k,e in zip(raw.iloc[data.source_row].parent_standard_inchi_key,data.entity_key):
        keys.setdefault(k,set()).add(e)
    entity_split = data.drop_duplicates('entity_key').set_index('entity_key').split.to_dict()
    names=[]
    for row in drugs.itertuples():
        matches = keys.get(row.ligand_inchikey,set()) | smiles.get(row.model_ligand_smiles,set())
        if len(matches)>1:
            raise ValueError(f'ambiguous old-drug training entity: {row.ligand_inchikey}')
        names.append(next(iter(matches)) if matches else f'__old_query_{row.ligand_inchikey}')
    drugs['entity_key'] = names
    drugs['training_role'] = drugs.entity_key.map(entity_split).fillna('absent_from_binary_relations')
    drugs['global_drug_index'] = base_count + np.arange(len(drugs))
    drugs.to_csv(OUT/'OLD_DRUGS_720.csv',index=False)
    targets.to_csv(OUT/'TARGETS_384.csv.gz',index=False)
    fp_path = features_dir/'SUPPORT_MORGAN_EXTENDED.npy'
    x=np.lib.format.open_memmap(fp_path,mode='w+',dtype=np.uint8,shape=(base_count+720,2048))
    x[:base_count]=np.load(audit['identity']['features_path'],mmap_mode='r')
    x[base_count:]=morgan
    x.flush()
    entities=np.array([f'__unused_feature_{i}' for i in range(len(x))],dtype=object)
    scaffolds=np.full(len(x),'',dtype=object)
    unique=data.drop_duplicates('drug_feature_index')
    entities[unique.drug_feature_index]=unique.entity_key
    scaffolds[unique.drug_feature_index]=unique.murcko_scaffold.fillna('')
    entities[base_count:]=drugs.entity_key
    scaffolds[base_count:]=drugs.murcko_scaffold
    store=SupportStore(x,data.drug_feature_index,data.target_feature_index,data.binary_label,
        np.flatnonzero(data.split.eq('train')),entity_keys=entities,scaffold_keys=scaffolds,feature_path=fp_path)
    with np.load(V3/'support_store/support_index.npz') as original:
        for a,b in [('pool_drug_indices','drug_indices'),('pool_target_indices','target_indices'),('pool_labels','labels')]:
            assert np.array_equal(getattr(store,a),original[b])
    store.save(OUT/'support_store')
    ds=np.repeat(drugs.global_drug_index,384)
    ts=np.tile(targets.global_target_index,720)
    evidence=store.cache_retrieve(OUT/'support_cache',ds,ts,k=16,
        progress=lambda done,total:print(json.dumps({'stage':'support','completed':done,'total':total}),flush=True) if done%20000<1000 or done==total else None)
    observed=np.full((720,384),np.nan,np.float32)
    train_pairs=np.zeros((720,384),bool)
    old_by_entity=drugs.groupby('entity_key').drug_feature_index.apply(list).to_dict()
    old_by_target=targets.set_index('global_target_index').target_feature_index.to_dict()
    for row in data.itertuples():
        if row.entity_key in old_by_entity and row.target_feature_index in old_by_target:
            t=old_by_target[row.target_feature_index]
            for d in old_by_entity[row.entity_key]:
                assert np.isnan(observed[d,t]) or observed[d,t]==row.binary_label
                observed[d,t]=row.binary_label
                train_pairs[d,t]=row.split=='train'
    known=pairs.is_any_frozen_known_relationship.to_numpy(bool).reshape(720,384)
    assert int(known.sum())==807
    np.savez_compressed(OUT/'LABELS_AND_SCOPES.npz',known_relationship=known,
        observed_binary=observed,seen_training_pair=train_pairs,drug_indices=np.arange(720),target_indices=np.arange(384),
        historical_test_drugs=drugs.training_role.eq('test').to_numpy(),
        absent_drugs=drugs.training_role.eq('absent_from_binary_relations').to_numpy())
    pairs.to_csv(OUT/'CANONICAL_PAIRS_720X384.csv.gz',index=False)
    protocol={'status':'FROZEN','task':'known old-drug to target ranking AND target to known old-drug ranking',
        'old_drugs':720,'targets':384,'known_relationships':807,'known_drug_queries':int(known.any(1).sum()),
        'known_target_queries':int(known.any(0).sum()),'known_label_source':'frozen project relationships (791) plus ChEMBL37 MoA additions (16)',
        'unknown_is_negative':False,'main_claim':'known-relationship recovery, with training-exposure audit; not independent novel-target validation',
        'selection':'all project old drugs and all current 384 target-core entries; no score-based sampling or model selection',
        'directions':{'drug_to_target':'720 queries x 384 target candidates','target_to_drug':'384 queries x 720 old-drug candidates'},
        'public_features':{'bermol_old_drugs':720,'full_esm_existing_exact_sequences':int((~missing).sum()),'full_esm_new_sequences':int(missing.sum())},
        'old_drug_training_roles':drugs.training_role.value_counts().to_dict(),
        'known_pairs_seen_binary_training':int((known&train_pairs).sum()),
        'known_pairs_with_biochemical_positive':int((known&(observed==1)).sum()),
        'known_pairs_with_biochemical_negative':int((known&(observed==0)).sum()),
        'observed_binary_pairs':int(np.isfinite(observed).sum()),'observed_positive_pairs':int((observed==1).sum()),
        'observed_negative_pairs':int((observed==0).sum()),'v3_data_manifest':audit,
        'support_training_rows':int(store.metadata['training_rows']),'support_pool_unchanged_from_original':True,
        'support':{n:str(getattr(evidence,n).filename) for n in ['indices','similarities','mask']},
        'baseline_reverse_borda':'rerank each raw constituent within target before averaging; never transpose row-normalized scores as a calibrated score',
        'drugclip':'native six-fold mean cosine; compare every model again on identical rectangular coverage subset',
        'input_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [DEPLOY/'OLD_DRUG_FEATURE_INDEX_720_V1.csv.gz',DEPLOY/'PROJECT_TARGET_FEATURE_INDEX_384_V1.csv.gz',DEPLOY/'OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz',V3/'RELATIONS_V3.csv.gz']}}
    write_json(OUT/'PROTOCOL.json',protocol)
    feature_paths = [features_dir/'OLD720_MORGAN2048.npy', features_dir/'OLD720_BERMOL768.npy',
        new_target_path, fp_path, OUT/'OLD_DRUGS_720.csv', OUT/'TARGETS_384.csv.gz',
        OUT/'LABELS_AND_SCOPES.npz']
    write_json(OUT/'FEATURE_MANIFEST.json', {'status':'COMPLETE','label_free_encoder_extension':True,
        'files':{str(p.relative_to(ROOT)):sha256(p) for p in feature_paths}})
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['v3_data_manifest','support','input_sha256']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

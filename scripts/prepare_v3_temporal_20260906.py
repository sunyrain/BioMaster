#!/usr/bin/env python3
"""Build time-filtered labels, independent feature axes and cutoff-only support."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import dill
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.temporal_relations import aggregate_window,first_observations,new_relations
from biomaster.odti_support_data_v3 import SupportStore
from build_biomaster_comprehensive_training_v1 import _standardize_and_fingerprint
from build_biomaster_odti_v4_features import sha256,write_json
from extract_v3_temporal_raw_20260906 import OUT,OLD

CONFIG=ROOT/'configs/biomaster_v3_temporal_20260906.json'


def molecule_features(parents,old):
    path=OUT/'MOLECULES.csv.gz'
    mapping_path=OUT/'PARENT_MAPPING.csv.gz'
    if path.exists():return pd.read_csv(path),pd.read_csv(mapping_path)
    assets=pd.read_csv(ROOT/'outputs/biomaster_odti_v4_plan_20260905/CURRENT_MOLECULE_ASSETS_V4.csv.gz')
    by_input=assets.set_index('input_smiles').model_ligand_smiles.to_dict()
    by_input.update({s:s for s in assets.model_ligand_smiles})
    by_input.update({s:s for s in old.model_ligand_smiles})
    valid_inputs=parents.canonical_smiles.dropna().astype(str)
    missing=sorted(set(valid_inputs)-set(by_input))
    print(json.dumps({'stage':'standardization','new_inputs':len(missing)}),flush=True)
    with ProcessPoolExecutor(max_workers=16) as pool:
        encoded=list(pool.map(_standardize_and_fingerprint,missing,chunksize=128))
    for text,value in zip(missing,encoded):
        if value[0]:by_input[text]=value[0]
    parents['model_ligand_smiles']=parents.canonical_smiles.map(by_input)
    parents.loc[parents.model_ligand_smiles.isna()].to_csv(OUT/'UNRESOLVED_MOLECULES.csv.gz',index=False)
    smiles=sorted(set(parents.model_ligand_smiles.dropna())|set(old.model_ligand_smiles))
    molecules=pd.DataFrame({'drug_feature_index':np.arange(len(smiles)),'model_ligand_smiles':smiles})
    molecules['entity_key']=molecules.model_ligand_smiles.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    lookup=molecules.set_index('model_ligand_smiles').drug_feature_index
    parents['drug_feature_index']=parents.model_ligand_smiles.map(lookup)
    molecules['global_feature_index']=molecules.model_ligand_smiles.map(assets.set_index('model_ligand_smiles').drug_feature_index)
    old_lookup=old.set_index('model_ligand_smiles').drug_feature_index
    old_morgan=np.load(OLD/'features/OLD720_MORGAN2048.npy')
    original=np.load(ROOT/'outputs/retrain_20260901/comprehensive_training_v1/MORGAN2048_UINT8_COMPREHENSIVE_V1.npy',mmap_mode='r')
    fp=np.lib.format.open_memmap(OUT/'MORGAN.npy',mode='w+',dtype=np.uint8,shape=(len(molecules),2048))
    cached=molecules.global_feature_index.notna().to_numpy()
    fp[cached]=original[molecules.global_feature_index[cached].to_numpy(int)]
    encoded_by_smiles={v[0]:v for v in encoded if v[0]}
    scaffold=assets.set_index('model_ligand_smiles').murcko_scaffold.to_dict()
    scaffold.update(old.set_index('model_ligand_smiles').murcko_scaffold.to_dict())
    for i in np.flatnonzero(~cached):
        s=smiles[i]
        if s in old_lookup:
            fp[i]=old_morgan[int(old_lookup[s])]
        else:
            value=encoded_by_smiles.get(s)
            if value is None:value=_standardize_and_fingerprint(s)
            assert value[0]==s
            fp[i]=np.unpackbits(np.frombuffer(value[2],dtype=np.uint8));scaffold[s]=value[1]
    molecules['murcko_scaffold']=molecules.model_ligand_smiles.map(scaffold).fillna('')
    fp.flush()
    molecules.to_csv(path,index=False);parents.to_csv(mapping_path,index=False)
    return molecules,parents


def bermol_features(molecules,old):
    path=OUT/'BERMOL.npy'
    donepath=OUT/'BERMOL_DONE.npy'
    if path.exists():
        bank=np.load(path,mmap_mode='r+');done=np.load(donepath,mmap_mode='r+')
    else:
        bank=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(molecules),768))
        done=np.lib.format.open_memmap(donepath,mode='w+',dtype=bool,shape=(len(molecules),))
        bank[:]=0;done[:]=False
        existing=np.load(ROOT/'outputs/biomaster_odti_v4_20260905/features/BERMOL768_FLOAT32_V4.npy',mmap_mode='r')
        available=np.load(ROOT/'outputs/biomaster_odti_v4_20260905/features/BERMOL_AVAILABLE_V4.npy')
        ids=np.flatnonzero(molecules.global_feature_index.notna())
        global_ids=molecules.global_feature_index.iloc[ids].to_numpy(int)
        ids,global_ids=ids[available[global_ids]],global_ids[available[global_ids]]
        bank[ids]=existing[global_ids];done[ids]=True
        old_index=old.set_index('model_ligand_smiles').drug_feature_index.to_dict()
        old_bank=np.load(OLD/'features/OLD720_BERMOL768.npy')
        for i in np.flatnonzero(~done):
            if molecules.model_ligand_smiles.iloc[i] in old_index:
                bank[i]=old_bank[old_index[molecules.model_ligand_smiles.iloc[i]]];done[i]=True
        bank.flush();done.flush()
    missing=np.flatnonzero(~done)
    print(json.dumps({'stage':'bermol','new_molecules':len(missing)}),flush=True)
    if len(missing):
        sys.path.insert(0,str(ROOT/'third_party/sota_dti_2026/DTIAM/code/BerMol'))
        from bermol.tokenizer import BerMolTokenizer
        with (ROOT/'third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl').open('rb') as f:predictor=dill.load(f)
        predictor.model.cuda().eval();tokenizer=BerMolTokenizer(predictor.vocab)
        with torch.inference_mode():
            for start in range(0,len(missing),4096):
                ids=missing[start:start+4096]
                tokens=[(int(i),tokenizer.encode(molecules.model_ligand_smiles.iloc[i]).squeeze(0)) for i in ids]
                tokens.sort(key=lambda item:len(item[1]));cursor=0
                while cursor<len(tokens):
                    length=len(tokens[min(len(tokens)-1,cursor+63)][1])
                    count=min(64,max(1,1_000_000//length**2))
                    selected=tokens[cursor:cursor+count];width=max(len(t) for _,t in selected)
                    batch=torch.zeros((len(selected),width),dtype=torch.long,device='cuda')
                    attention=torch.full((len(selected),width,width),-10000.,device='cuda')
                    for j,(_,token) in enumerate(selected):batch[j,:len(token)]=token.cuda();attention[j,:,:len(token)]=0
                    _,pooled=predictor.model.encoder(batch,attention)
                    values=pooled.cpu().numpy();assert np.isfinite(values).all() and (np.linalg.norm(values,axis=1)>0).all()
                    bank[[i for i,_ in selected]]=values;cursor+=len(selected)
                done[ids]=True;bank.flush();done.flush()
                print(json.dumps({'stage':'bermol','new_completed':start+len(ids),'new_total':len(missing)}),flush=True)
        del predictor;torch.cuda.empty_cache()
    assert done.all()


def main():
    torch.set_num_threads(4)
    if (OUT/'DATA_MANIFEST.json').exists():raise FileExistsError('temporal data already frozen')
    config=json.loads(CONFIG.read_text())
    protocol={**config,'status':'FROZEN','config_sha256':sha256(CONFIG),'raw_manifest_sha256':sha256(OUT/'RAW_MANIFEST.json')}
    if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text())==protocol
    else:write_json(OUT/'PROTOCOL.json',protocol)
    parents=pd.read_csv(OUT/'PARENT_MOLECULES.csv.gz')
    old=pd.read_csv(OLD/'OLD_DRUGS_720.csv')
    targets=pd.read_csv(OLD/'TARGETS_384.csv.gz')
    molecules,mapping=molecule_features(parents,old)
    old=old.rename(columns={'drug_feature_index':'old_drug_index'})
    old['drug_feature_index']=old.model_ligand_smiles.map(molecules.set_index('model_ligand_smiles').drug_feature_index)
    assert old.drug_feature_index.notna().all() and old.drug_feature_index.is_unique
    old.to_csv(OUT/'OLD_DRUG_INDEX.csv',index=False)
    targets.to_csv(OUT/'TARGET_INDEX.csv.gz',index=False)
    annual=pd.read_csv(OUT/'ANNUAL_RAW.csv.gz')
    annual['drug_feature_index']=annual.parent_molregno.map(mapping.set_index('parent_molregno').drug_feature_index)
    unresolved=int(annual.loc[annual.drug_feature_index.isna(),'activity_rows'].sum())
    annual=annual.dropna(subset=['drug_feature_index']).rename(columns={'target_index':'target_feature_index'})
    annual.drug_feature_index=annual.drug_feature_index.astype(int)
    annual=annual.groupby(['drug_feature_index','target_feature_index','document_year'],dropna=False,sort=True).agg(
        activity_rows=('activity_rows','sum'),numeric_rows=('numeric_rows','sum'),pchembl_sum=('pchembl_sum','sum'),
        min_pchembl=('min_pchembl','min'),max_pchembl=('max_pchembl','max'),any_explicit_inactive=('any_explicit_inactive','max')).reset_index()
    annual.to_csv(OUT/'ANNUAL_STANDARDIZED.csv.gz',index=False)
    first=first_observations(annual);first.to_csv(OUT/'FIRST_OBSERVATIONS.csv.gz',index=False)
    tables={'development_train':aggregate_window(annual,end=2020),'final_train':aggregate_window(annual,end=2022),
        'development_validation_all':aggregate_window(annual,start=2021,end=2022),
        'test_2023_2025_all':aggregate_window(annual,start=2023,end=2025)}
    tables['development_validation_new']=new_relations(tables['development_validation_all'],first,2020)
    tables['test_2023_2025_new']=new_relations(tables['test_2023_2025_all'],first,2022)
    for year in [2023,2024,2025]:
        tables[f'test_{year}_new']=new_relations(aggregate_window(annual,start=year,end=year),first,year-1)
    counts={}
    for name,frame in tables.items():
        counts[name]={'all_pairs':len(frame),'conflicting':int(frame.conflicting.sum()),'grey_or_unresolved':int((frame.binary_label.isna()&~frame.conflicting).sum())}
        frame=frame.loc[frame.binary_label.notna()].copy();frame.binary_label=frame.binary_label.astype(int)
        frame['entity_key']=molecules.entity_key.to_numpy()[frame.drug_feature_index]
        frame['murcko_scaffold']=molecules.murcko_scaffold.to_numpy()[frame.drug_feature_index]
        frame['target_assay_family']=targets.target_assay_family.to_numpy()[frame.target_feature_index]
        frame['is_project_old_drug']=frame.drug_feature_index.isin(old.drug_feature_index)
        frame.to_csv(OUT/f'{name.upper()}.csv.gz',index=False);tables[name]=frame
        counts[name].update({'rows':len(frame),'positives':int(frame.binary_label.sum()),'drugs':int(frame.drug_feature_index.nunique()),
            'old_drug_rows':int(frame.is_project_old_drug.sum()),'old_drug_positives':int(frame.loc[frame.is_project_old_drug,'binary_label'].sum()),
            'min_used_year':int(frame.min_document_year.min()),'max_used_year':int(frame.max_document_year.max())})
    # Every first-seen test pair is disjoint from all pre-cutoff observations,
    # including observations too weak/ambiguous to train a binary classifier.
    for name,cutoff in [('development_validation_new',2020),('test_2023_2025_new',2022)]:
        historical=first[first.first_document_year.le(cutoff)|first.has_undated]
        overlap=tables[name].merge(historical,on=['drug_feature_index','target_feature_index'])
        assert overlap.empty
    pool=pd.concat([f[['drug_feature_index','target_feature_index']] for f in tables.values()]).drop_duplicates().sort_values(['drug_feature_index','target_feature_index']).reset_index(drop=True)
    pool['pool_row']=np.arange(len(pool));pool.to_csv(OUT/'QUERY_POOL.csv.gz',index=False)
    first_lookup=first[first.drug_feature_index.isin(old.drug_feature_index)]
    risk={}
    old_by_drug=old.set_index('drug_feature_index').old_drug_index.to_dict()
    for name,cutoff in [('development',2020),('final',2022)]:
        values=np.ones((720,384),bool)
        prior=first_lookup[first_lookup.first_document_year.le(cutoff)|first_lookup.has_undated]
        values[prior.drug_feature_index.map(old_by_drug).to_numpy(int),prior.target_feature_index.to_numpy(int)]=False
        risk[name]=values
    np.savez_compressed(OUT/'RISK_SETS.npz',**risk)
    # Label-free public representations are aligned by the current exact axes.
    deploy=ROOT/'outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1'
    np.save(OUT/'PROTBERT.npy',np.load(deploy/'PROJECT384_PROTBERT1024_FLOAT32_V1.npy'))
    np.save(OUT/'ESM2_LEGACY.npy',np.load(ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy'))
    complete=np.concatenate([np.load(ROOT/'outputs/biomaster_odti_v4_20260905/features/ESM2_FULL_MEAN1280_FLOAT32_V4.npy'),np.load(OLD/'features/ADDED_FULL_ESM2_MEANS.npy')])
    np.save(OUT/'ESM2_COMPLETE.npy',complete[targets.global_target_index.to_numpy(int)])
    bermol_features(molecules,pd.read_csv(OLD/'OLD_DRUGS_720.csv'))
    features=np.load(OUT/'MORGAN.npy',mmap_mode='r')
    supports={}
    for stage,table in [('development','development_train'),('final','final_train')]:
        train=tables[table]
        folder=OUT/stage;folder.mkdir(exist_ok=True)
        store=SupportStore(features,train.drug_feature_index,train.target_feature_index,train.binary_label,np.arange(len(train)),
            entity_keys=molecules.entity_key,scaffold_keys=molecules.murcko_scaffold.fillna(''),feature_path=OUT/'MORGAN.npy')
        store.save(folder/'support_store')
        last=[time.monotonic()]
        def progress(done,total):
            if time.monotonic()-last[0]>=20 or done==total:
                print(json.dumps({'stage':stage,'support_done':done,'total':total}),flush=True);last[0]=time.monotonic()
        main_support=store.cache_retrieve(folder/'pool_support',pool.drug_feature_index,pool.target_feature_index,k=16,progress=progress)
        old_support=store.cache_retrieve(folder/'old_support',np.repeat(old.drug_feature_index.to_numpy(int),384),np.tile(np.arange(384),720),k=16,progress=progress)
        supports[stage]={'store_hash':store.store_hash,'training_rows':len(train),'training_max_year':int(train.max_document_year.max()),
            'pool':{n:str(getattr(main_support,n).filename) for n in ['indices','similarities','mask']},
            'old':{n:str(getattr(old_support,n).filename) for n in ['indices','similarities','mask']}}
    file_paths=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.npy','.npz','.gz','.csv']]
    manifest={'status':'COMPLETE','protocol_sha256':sha256(OUT/'PROTOCOL.json'),'counts':counts,
        'molecules':len(molecules),'query_pairs':len(pool),'unresolved_structure_activity_rows':unresolved,
        'supports':supports,'files':{str(p.relative_to(ROOT)):sha256(p) for p in file_paths},
        'no_current_V3_or_J_checkpoint_reused':True,'time_filter_before_label_aggregation':True}
    write_json(OUT/'DATA_MANIFEST.json',manifest)
    print(json.dumps({'status':'COMPLETE','counts':counts}),flush=True)


if __name__=='__main__':main()

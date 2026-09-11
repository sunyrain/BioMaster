#!/usr/bin/env python3
"""Resume exact DTIAM representations; never read response labels for encoding."""
import argparse
import gc
import json
import os
import sys
import time

import dill
import numpy as np
import pandas as pd
import torch

from dtiam_ab_common_20260912 import ROOT, OUT, DATA, SOURCE, FEATURES, now, digest, write_json, check_values

BERMOL = ROOT/'third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl'
BERMOL_HASH = 'afd1929fdbef6da110b5f4d0688a0f4a5f88fc3fb660bfcc1fb56e9624de38b1'
ESM = ROOT.parent/'.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt'
ESM_HASH = 'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c'
OLD = ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1'


def state(stage, **kw):
    value = dict(stage=stage, updated_utc=now(), **kw)
    write_json(FEATURES/'STATUS.json', value)
    print(json.dumps(value), flush=True)


def assets():
    molecules = pd.read_parquet(DATA/'MOLECULES.parquet').sort_values('drug_feature_index')
    targets = pd.read_parquet(DATA/'TARGETS.parquet').sort_values('target_feature_index')
    for f, c in [(molecules,'drug_feature_index'),(targets,'target_feature_index')]:
        assert np.array_equal(f[c], np.arange(len(f)))
    return molecules, targets, np.load(SOURCE/'REQUIRED_DRUG_IDS.npy'), np.load(SOURCE/'REQUIRED_TARGET_IDS.npy')


def create_bank(name, n, width):
    path, donepath = FEATURES/(name+'.npy'), FEATURES/(name+'_DONE.npy')
    if path.exists():
        bank, done = np.load(path,mmap_mode='r+'), np.load(donepath,mmap_mode='r+')
        assert bank.shape==(n,width) and done.shape==(n,) and bank.dtype==np.float32
    else:
        bank = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n,width))
        done = np.lib.format.open_memmap(donepath, mode='w+', dtype=bool, shape=(n,))
        bank[:] = 0; done[:] = False; bank.flush(); done.flush()
    return bank, done


def fill_cache(bank, done, required, identities, index, array, key, source_id, available=None):
    old = pd.read_csv(index)
    if available:
        old = old.loc[np.load(available).astype(bool)].copy()
    lookup = old.drop_duplicates(key).set_index(key)[source_id]
    missing = required[~done[required]]
    mapped = identities.iloc[missing].map(lookup)
    use = mapped.notna().to_numpy(); ids = missing[use]
    positions = mapped[use].to_numpy(np.int64); values = np.load(array,mmap_mode='r')
    for start in range(0,len(ids),8192):
        v = values[positions[start:start+8192]]; check_values(v)
        bank[ids[start:start+8192]] = v
    bank.flush(); done[ids] = True; done.flush()
    sample = ids[np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int)].tolist() if len(ids) else []
    return dict(index=str(index.relative_to(ROOT)), array=str(array.relative_to(ROOT)),
                index_sha256=digest(index), array_sha256=digest(array), reused=len(ids), parity_ids=sample)


def initialize(molecules, targets, drugs, proteins):
    FEATURES.mkdir(parents=True,exist_ok=True)
    drugbank, drugdone = create_bank('BERMOL',len(molecules),768)
    protbank, protdone = create_bank('ESM2',len(targets),1280)
    path = FEATURES/'CACHE_PROVENANCE.json'
    if path.exists():
        return drugbank, drugdone, protbank, protdone
    state('INITIALIZING_EXACT_IDENTITY_CACHE')
    caches = []
    temporal = ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
    v4 = ROOT/'outputs/biomaster_odti_v4_20260905/features'
    for idx, arr, avail in [
        (temporal/'MOLECULES.csv.gz',temporal/'BERMOL.npy',temporal/'BERMOL_DONE.npy'),
        (ROOT/'outputs/biomaster_odti_v4_plan_20260905/CURRENT_MOLECULE_ASSETS_V4.csv.gz',
         v4/'BERMOL768_FLOAT32_V4.npy',v4/'BERMOL_AVAILABLE_V4.npy')]:
        caches.append(dict(kind='drug', **fill_cache(drugbank,drugdone,drugs,molecules.smiles,
            idx,arr,'model_ligand_smiles','drug_feature_index',avail)))
    for idx, arr, key in [
        (OLD/'dtiam_official_feature_store_v1/DTIAM_ESM2_TARGET_INDEX_V1.csv.gz',
         OLD/'dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy','protein_sequence'),
        (OLD/'dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz',
         OLD/'dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy','sequence')]:
        caches.append(dict(kind='protein', **fill_cache(protbank,protdone,proteins,targets.sequence,
            idx,arr,key,'target_feature_index')))
    write_json(path, dict(created_utc=now(), sources=caches, required_drugs=len(drugs),
        required_targets=len(proteins), drug_reused=int(drugdone[drugs].sum()),
        protein_reused=int(protdone[proteins].sum()),
        protein_truncated_to_1022=int(targets.sequence.iloc[proteins].str.len().gt(1022).sum()),
        asset_hashes={str(p.relative_to(ROOT)):digest(p) for p in
            [DATA/'MOLECULES.parquet',DATA/'TARGETS.parquet',SOURCE/'REQUIRED_DRUG_IDS.npy',SOURCE/'REQUIRED_TARGET_IDS.npy']}))
    return drugbank, drugdone, protbank, protdone


def bermol(molecules, required, bank, done, limit):
    if digest(BERMOL) != BERMOL_HASH:
        raise ValueError('BerMol checkpoint changed')
    sys.path.insert(0,str(BERMOL.parent/'BerMol'))
    from bermol.tokenizer import BerMolTokenizer
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    with BERMOL.open('rb') as f:
        predictor = dill.load(f)
    predictor.model.cuda().eval()
    tokenizer = BerMolTokenizer(predictor.vocab)
    parity = []
    with torch.inference_mode():
        for source in json.loads((FEATURES/'CACHE_PROVENANCE.json').read_text())['sources']:
            if source['kind'] != 'drug':
                continue
            for i in source['parity_ids']:
                fresh = predictor.transform(molecules.smiles.iloc[i],device='cuda')[1].cpu().numpy()[0]
                delta = float(np.max(np.abs(fresh-bank[i])))
                if not np.allclose(fresh,bank[i],atol=3e-5,rtol=3e-5):
                    raise ValueError(f'Cache incompatible with official BerMol at {i}: {delta}')
                parity.append(dict(drug_feature_index=i,max_abs_error=delta))
        write_json(FEATURES/'BERMOL_PARITY.json',dict(status='PASS',cache_vs_official=parity))
        missing = required[~done[required]]
        if limit:
            missing = missing[:limit]
        started = time.monotonic(); count = 0; max_tokens = 0; batch_parity = None
        for start in range(0,len(missing),2048):
            ids = missing[start:start+2048]
            tokens = [(int(i),tokenizer.encode(molecules.smiles.iloc[i]).squeeze(0)) for i in ids]
            tokens.sort(key=lambda x:len(x[1])); cursor = 0
            while cursor<len(tokens):
                maximum = len(tokens[min(len(tokens)-1,cursor+127)][1])
                batch_size = min(128,max(1,2_000_000//maximum**2))
                selected = tokens[cursor:cursor+batch_size]; length = max(len(t) for _,t in selected)
                batch = torch.zeros((len(selected),length),dtype=torch.long,device='cuda')
                mask = torch.full((len(selected),length,length),-10000.,device='cuda')
                for j,(_,token) in enumerate(selected):
                    batch[j,:len(token)] = token.cuda(); mask[j,:,:len(token)] = 0
                _, pooled = predictor.model.encoder(batch,mask)
                values = pooled.float().cpu().numpy(); check_values(values)
                if batch_parity is None:
                    errors = []
                    for j in [0,len(selected)-1]:
                        i = selected[j][0]
                        v = predictor.transform(molecules.smiles.iloc[i],device='cuda')[1].cpu().numpy()[0]
                        if not np.allclose(v,values[j],atol=3e-5,rtol=3e-5):
                            raise ValueError('Batched BerMol differs from official single molecule inference')
                        errors.append(float(np.max(np.abs(v-values[j]))))
                    batch_parity = errors
                    write_json(FEATURES/'BERMOL_PARITY.json',dict(status='PASS',cache_vs_official=parity,batch_vs_official_errors=errors))
                bank[[i for i,_ in selected]] = values
                cursor += len(selected); max_tokens=max(max_tokens,length)
            bank.flush(); done[ids] = True; done.flush(); count += len(ids)
            elapsed = time.monotonic()-started; remaining = int((~done[required]).sum())
            state('BERMOL_ENCODING',required=len(required),completed=int(done[required].sum()),
                  newly_encoded_this_process=count,seconds=round(elapsed,1),rows_per_second=count/elapsed,
                  remaining=remaining,estimated_remaining_seconds=remaining*elapsed/count,max_tokens_this_process=max_tokens)
    del predictor; gc.collect(); torch.cuda.empty_cache()


def esm2(targets, required, bank, done):
    if digest(ESM) != ESM_HASH:
        raise ValueError('ESM checkpoint changed')
    os.environ['TORCH_HOME'] = str(ESM.parents[2])
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.cuda().eval(); converter = alphabet.get_batch_converter()
    def encode(i):
        seq = targets.sequence.iloc[i][:1022]
        _,_,tokens = converter([(str(i),seq)])
        result = model(tokens.cuda(),repr_layers=[33],return_contacts=False)
        # Literal official DTIAM pooling, including EOS, excluding BOS.
        return result['representations'][33][0,1:].mean(0).float().cpu().numpy()
    parity=[]; started=time.monotonic()
    with torch.inference_mode():
        for source in json.loads((FEATURES/'CACHE_PROVENANCE.json').read_text())['sources']:
            if source['kind']!='protein':
                continue
            for i in source['parity_ids']:
                fresh=encode(i); delta=float(np.max(np.abs(fresh-bank[i])))
                if not np.allclose(fresh,bank[i],atol=3e-5,rtol=3e-5):
                    raise ValueError(f'Cache incompatible with official ESM2 at {i}: {delta}')
                parity.append(dict(target_feature_index=i,max_abs_error=delta))
        write_json(FEATURES/'ESM2_PARITY.json',dict(status='PASS',samples=parity))
        missing=required[~done[required]]
        for k,i in enumerate(missing):
            value=encode(i); check_values(value[None]); bank[i]=value
            # Durable progress is committed after the actual vector flush.
            bank.flush(); done[i]=True; done.flush()
            if (k+1)%25==0 or k+1==len(missing):
                elapsed=time.monotonic()-started
                state('ESM2_ENCODING',required=len(required),completed=int(done[required].sum()),
                      newly_encoded_this_process=k+1,seconds=round(elapsed,1),remaining=len(missing)-k-1,
                      estimated_remaining_seconds=(len(missing)-k-1)*elapsed/(k+1))
    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit-new-drugs',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for feature completion')
    # Fix floating point behavior for cache and official implementation parity.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    molecules,targets,drugs,proteins=assets()
    db,dd,pb,pd_=initialize(molecules,targets,drugs,proteins)
    bermol(molecules,drugs,db,dd,args.limit_new_drugs)
    if args.limit_new_drugs:
        return
    esm2(targets,proteins,pb,pd_)
    assert dd[drugs].all() and pd_[proteins].all()
    for bank,ids in [(db,drugs),(pb,proteins)]:
        for start in range(0,len(ids),8192):
            check_values(bank[ids[start:start+8192]])
    paths=[FEATURES/n for n in ['BERMOL.npy','ESM2.npy','BERMOL_DONE.npy','ESM2_DONE.npy',
        'CACHE_PROVENANCE.json','BERMOL_PARITY.json','ESM2_PARITY.json']]
    write_json(FEATURES/'MANIFEST.json',dict(status='COMPLETE',completed_utc=now(),
        required_molecules=len(drugs),required_targets=len(proteins),
        bermol='Official frozen BerMol CLS768 FP32, exact SMILES, no truncation',
        esm2='Official ESM2-t33-650M FP32, first 1022 residues, final layer mean including EOS',
        checkpoint_sha256={'bermol':BERMOL_HASH,'esm2':ESM_HASH},
        files={str(f.relative_to(ROOT)):digest(f) for f in paths},label_dependency='NONE'))
    state('COMPLETE',required_drugs=len(drugs),required_targets=len(proteins))


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import traceback
        write_json(FEATURES/'ERROR.json',dict(utc=now(),error=repr(error),traceback=traceback.format_exc()))
        raise

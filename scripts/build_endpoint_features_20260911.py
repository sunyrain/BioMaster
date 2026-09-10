#!/usr/bin/env python3
"""Full-scope, resumable global DrugCLIP/Morgan/ESM2 features; no atom-token cache."""
import argparse,json,multiprocessing as mp,os,pickle,sys,time
from collections import Counter
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import lmdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_endpoint_ablation_20260911 import DATA,OUT,FEATURES,write_json
from biomaster.portable_ranker_v2 import digest

BASE=ROOT/'outputs/biomaster_unified_interaction_20260906/features'
SUPP=ROOT/'outputs/biomaster_best_model_20260906/data/supplemental_features'
OLD=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'

def log(stage,**v):
    v=dict(stage=stage,**v);write_json(FEATURES/(stage+'_STATUS.json'),v);print(json.dumps(v),flush=True)

def bank(name,shape,dtype):
    p=FEATURES/(name+'.npy')
    if p.exists():
        a=np.load(p,mmap_mode='r+');assert a.shape==shape and a.dtype==np.dtype(dtype);return a
    return np.lib.format.open_memmap(p,mode='w+',shape=shape,dtype=dtype)

def prepare():
    FEATURES.mkdir(exist_ok=True)
    if (FEATURES/'INITIALIZED.json').exists():return
    drugs=pd.read_parquet(DATA/'MOLECULES.parquet');targets=pd.read_parquet(DATA/'TARGETS.parquet')
    ids=np.load(OUT/'REQUIRED_DRUG_IDS.npy');tids=np.load(OUT/'REQUIRED_TARGET_IDS.npy');n=len(drugs)
    for name,shape,dtype in [('DRUG_CLIP',(n,512),'float32'),('GRAPH',(n,40),'float32'),('MORGAN',(n,2048),'uint8'),
        ('AVAILABLE',(n,),'bool'),('DRUG_DONE',(n,),'bool'),('CHEM_DONE',(n,),'bool'),
        ('TARGET',(len(targets),1280),'float32'),('TARGET_DONE',(len(targets),),'bool')]:bank(name,shape,dtype).flush()
    old=pd.read_csv(OLD/'MOLECULES.csv.gz',usecols=['drug_feature_index','model_ligand_smiles'])
    old=old.drop_duplicates('model_ligand_smiles',keep='first').set_index('model_ligand_smiles').drug_feature_index
    matches=drugs.iloc[ids][['drug_feature_index','smiles']].copy();matches['old_id']=matches.smiles.map(old)
    matches=matches[matches.old_id.notna()].copy();matches.old_id=matches.old_id.astype(int)
    clip=np.load(FEATURES/'DRUG_CLIP.npy',mmap_mode='r+');graph=np.load(FEATURES/'GRAPH.npy',mmap_mode='r+')
    available=np.load(FEATURES/'AVAILABLE.npy',mmap_mode='r+');done=np.load(FEATURES/'DRUG_DONE.npy',mmap_mode='r+')
    chem=np.load(FEATURES/'CHEM_DONE.npy',mmap_mode='r+');fp=np.load(FEATURES/'MORGAN.npy',mmap_mode='r+')
    old_fp=np.load(OLD/'MORGAN.npy',mmap_mode='r');reused=[];inputs={}
    for directory in [BASE,SUPP]:
        olddone=np.load(directory/'ATOM_DONE.npy',mmap_mode='r');oc=np.load(directory/'MOLECULE_GLOBAL.npy',mmap_mode='r')
        oa=np.load(directory/'PRETRAINED_AVAILABLE.npy',mmap_mode='r');og=np.load(directory/'ATOM_INDEX.npz')['graph_mean']
        m=matches[olddone[matches.old_id.to_numpy()] & ~done[matches.drug_feature_index.to_numpy()]]
        d=m.drug_feature_index.to_numpy();o=m.old_id.to_numpy()
        clip[d]=oc[o];graph[d]=og[o];available[d]=oa[o];fp[d]=old_fp[o];done[d]=True;chem[d]=True
        reused.extend(dict(drug_feature_index=int(i),old_drug_feature_index=int(j),source=str(directory.relative_to(ROOT))) for i,j in zip(d,o))
        for name in ['ATOM_MANIFEST.json','MOLECULE_GLOBAL.npy','PRETRAINED_AVAILABLE.npy','ATOM_INDEX.npz']:
            p=directory/name;inputs[str(p.relative_to(ROOT))]=digest(p)
    for a in [clip,graph,available,done,chem,fp]:a.flush()
    pd.DataFrame(reused).to_csv(FEATURES/'REUSED_DRUG_FEATURES.csv.gz',index=False)
    matrix=ROOT/'outputs/biomaster_matrix_720x890_20260910'
    ti=pd.read_csv(matrix/'TARGET_INDEX.csv.gz');tf=np.load(matrix/'TARGET_GLOBAL.npy')
    lookup={h:i for i,h in enumerate(ti.sequence_sha256)}
    target=np.load(FEATURES/'TARGET.npy',mmap_mode='r+');td=np.load(FEATURES/'TARGET_DONE.npy',mmap_mode='r+')
    for i in tids:
        h=targets.sequence_sha256.iloc[i]
        if h in lookup:target[i]=tf[lookup[h]];td[i]=True
    target.flush();td.flush()
    for p in [OLD/'MOLECULES.csv.gz',OLD/'MORGAN.npy',matrix/'TARGET_INDEX.csv.gz',matrix/'TARGET_GLOBAL.npy']:
        inputs[str(p.relative_to(ROOT))]=digest(p)
    record=dict(status='COMPLETE',molecules=n,required_molecules=len(ids),required_targets=len(tids),
        exact_canonical_smiles_cached_molecules=int(done[ids].sum()),new_molecules=int((~done[ids]).sum()),
        cached_targets=int(td[tids].sum()),new_targets=int((~td[tids]).sum()),source_hashes=inputs,
        no_label_features=True,drug_reuse_identity='exact canonical SMILES',target_reuse_identity='exact sequence SHA256')
    write_json(FEATURES/'INITIALIZED.json',record);log('initialize',**record)

def make_molecule(job):
    from rdkit import Chem,RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from prepare_biomaster_unified_interaction import conformer,heavy_record
    RDLogger.DisableLog('rdApp.*')
    i,s=job
    try:
        _,r=conformer(job);r=heavy_record(r)
    except (ValueError,RuntimeError,OverflowError) as exc:
        from biomaster.odti_local_features_v3 import molecular_graph
        g=molecular_graph(s,max_atoms=100000)
        if not g['available']:raise
        m=Chem.RemoveHs(Chem.MolFromSmiles(s))
        r=dict(index=i,atoms=[a.GetSymbol() for a in m.GetAtoms()],graph_mean=g['atom_features'].astype(np.float16).astype(np.float32).mean(0),status='rdkit_conformer_exception_graph_fallback:'+type(exc).__name__)
    mol=Chem.MolFromSmiles(s)
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprintAsNumPy(mol).astype(np.uint8)
    keep={k:r[k] for k in ['index','atoms','graph_mean','status']}
    if 'coordinates' in r:keep['coordinates']=r['coordinates']
    return i,keep,fp

def chemistry(workers):
    prepare();drugs=pd.read_parquet(DATA/'MOLECULES.parquet');ids=np.load(OUT/'REQUIRED_DRUG_IDS.npy')
    done=np.load(FEATURES/'CHEM_DONE.npy',mmap_mode='r+');missing=ids[~done[ids]]
    graph=np.load(FEATURES/'GRAPH.npy',mmap_mode='r+');fp=np.load(FEATURES/'MORGAN.npy',mmap_mode='r+')
    env=lmdb.open(str(FEATURES/'CONFORMERS.lmdb'),subdir=False,map_size=16*1024**3,lock=True,readahead=False)
    start=time.monotonic();log('chemistry',required=len(ids),missing=len(missing),workers=workers)
    pending=[];counts=Counter()
    def flush():
        with env.begin(write=True) as txn:
            for i,r,f in pending:txn.put(str(i).encode(),pickle.dumps(r,protocol=5));graph[i]=r['graph_mean'];fp[i]=f
        graph.flush();fp.flush();done[[i for i,_,_ in pending]]=True;done.flush();pending.clear()
    with mp.get_context('spawn').Pool(workers,maxtasksperchild=3000) as pool:
        for j,result in enumerate(pool.imap_unordered(make_molecule,((int(i),drugs.smiles.iloc[i]) for i in missing),chunksize=4),1):
            pending.append(result);counts[result[1]['status']]+=1
            if len(pending)>=1024 or j==len(missing):
                flush();log('chemistry',completed=int(done[ids].sum()),required=len(ids),new_completed=j,new_total=len(missing),seconds=round(time.monotonic()-start,1),counts=dict(counts))
    if pending:flush()
    env.sync();env.close();assert done[ids].all()
    write_json(FEATURES/'CHEMISTRY_COMPLETE.json',dict(status='COMPLETE',required=len(ids),all_done=True,new_counts=dict(counts),seconds=time.monotonic()-start))

def proteins():
    import torch
    from build_biomaster_odti_target_token_features_v1 import window_bounds
    prepare();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    targets=pd.read_parquet(DATA/'TARGETS.parquet');ids=np.load(OUT/'REQUIRED_TARGET_IDS.npy')
    done=np.load(FEATURES/'TARGET_DONE.npy',mmap_mode='r+');bank_=np.load(FEATURES/'TARGET.npy',mmap_mode='r+')
    missing=ids[~done[ids]];start=time.monotonic();log('protein',required=len(ids),missing=len(missing))
    if len(missing):
        os.environ['TORCH_HOME']='/root/autodl-tmp/.cache/torch'
        import esm
        model,alphabet=esm.pretrained.esm2_t33_650M_UR50D();model=model.cuda().eval();convert=alphabet.get_batch_converter()
        with torch.inference_mode():
            for j,i in enumerate(missing):
                seq=targets.sequence.iloc[i];total=np.zeros((len(seq),1280),np.float32);counts=np.zeros((len(seq),1),np.float32)
                for lo,hi in window_bounds(len(seq),1022,128):
                    _,_,token=convert([(str(i),seq[lo:hi])])
                    with torch.autocast('cuda',dtype=torch.float16):h=model(token.cuda(),repr_layers=[33],return_contacts=False)['representations'][33]
                    total[lo:hi]+=h[0,1:hi-lo+1].float().cpu().numpy();counts[lo:hi]+=1
                assert (counts>0).all();v=(total/counts).astype(np.float16).astype(np.float32).mean(0)
                assert np.isfinite(v).all();bank_[i]=v;bank_.flush();done[i]=True;done.flush()
                if (j+1)%25==0 or j+1==len(missing):log('protein',completed=int(done[ids].sum()),required=len(ids),seconds=round(time.monotonic()-start,1))
    assert done[ids].all();write_json(FEATURES/'PROTEIN_COMPLETE.json',dict(status='COMPLETE',required=len(ids),full_sequences_no_truncation=True,seconds=time.monotonic()-start))

def encode():
    import torch
    from prepare_biomaster_unified_interaction import load_pretrained,pretrained_batch,pretrained_tokens,CHECKPOINT
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    ids=np.load(OUT/'REQUIRED_DRUG_IDS.npy');done=np.load(FEATURES/'DRUG_DONE.npy',mmap_mode='r+')
    chem=np.load(FEATURES/'CHEM_DONE.npy',mmap_mode='r')
    clip=np.load(FEATURES/'DRUG_CLIP.npy',mmap_mode='r+');available=np.load(FEATURES/'AVAILABLE.npy',mmap_mode='r+')
    sizes=pd.read_parquet(DATA/'MOLECULES.parquet',columns=['heavy_atoms']).heavy_atoms.to_numpy()
    start=time.monotonic();log('drugclip',required=len(ids),missing=int((~done[ids]).sum()),streaming=True)
    # Lock-aware short read transactions permit the chemistry writer to continue.
    env=lmdb.open(str(FEATURES/'CONFORMERS.lmdb'),subdir=False,readonly=True,lock=True,readahead=True,map_size=16*1024**3)
    task,model=load_pretrained();counts=Counter();processed=0;last_log=0
    with torch.inference_mode():
        while not done[ids].all():
            ready=ids[~done[ids] & chem[ids]]
            if not len(ready):
                time.sleep(2);continue
            order=ready[np.argsort(sizes[ready],kind='stable')]
            for start_i in range(0,len(order),128):
                chunk=order[start_i:start_i+128]
                with env.begin() as txn:
                    records=[pickle.loads(txn.get(str(i).encode())) for i in chunk]
                valid=[r for r in records if 'coordinates' in r]
                if valid:
                    h=pretrained_tokens(model,pretrained_batch(valid,task.dictionary)).float().cpu().numpy();assert np.isfinite(h).all()
                    for j,r in enumerate(valid):clip[r['index']]=h[j,0];available[r['index']]=True
                counts.update(r['status'] for r in records)
                clip.flush();available.flush();done[chunk]=True;done.flush();processed+=len(chunk)
                if processed-last_log>=4096:
                    log('drugclip',completed=int(done[ids].sum()),required=len(ids),seconds=round(time.monotonic()-start,1));last_log=processed
    env.close();assert done[ids].all()
    write_json(FEATURES/'DRUGCLIP_COMPLETE.json',dict(status='COMPLETE',required=len(ids),available=int(available[ids].sum()),graph_fallback=int((~available[ids]).sum()),
        checkpoint_sha256=digest(CHECKPOINT),new_status_counts=dict(counts),seconds=time.monotonic()-start,fp32_interface=True,streaming_committed_conformers=True))
    log('drugclip',completed=len(ids),required=len(ids),seconds=round(time.monotonic()-start,1))


def verify():
    ids=np.load(OUT/'REQUIRED_DRUG_IDS.npy');tids=np.load(OUT/'REQUIRED_TARGET_IDS.npy')
    for name in ['CHEMISTRY','PROTEIN','DRUGCLIP']:assert (FEATURES/(name+'_COMPLETE.json')).exists(),name
    for name,take in [('DRUG_CLIP',ids),('GRAPH',ids),('MORGAN',ids),('TARGET',tids)]:
        a=np.load(FEATURES/(name+'.npy'),mmap_mode='r');assert np.isfinite(a[take]).all(),name
        if name=='MORGAN':assert (a[take].sum(1)>0).all() and np.isin(a[take],[0,1]).all()
        if name in ['GRAPH','TARGET']:assert (np.abs(a[take]).sum(1)>0).all()
        if name=='DRUG_CLIP':
            available=np.load(FEATURES/'AVAILABLE.npy');assert (np.abs(a[ids[available[ids]]]).sum(1)>0).all()
    for name,take in [('DRUG_DONE',ids),('CHEM_DONE',ids),('TARGET_DONE',tids)]:assert np.load(FEATURES/(name+'.npy'))[take].all()
    write_json(FEATURES/'MANIFEST.json',dict(status='COMPLETE',required_molecules=len(ids),required_sequences=len(tids),
        all_required_rows_prepared=True,no_labels_encoded=True,files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(FEATURES.glob('*.npy'))},
        producer_sha256=digest(Path(__file__))))
    log('complete',required_molecules=len(ids),required_sequences=len(tids))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','chemistry','proteins','encode','verify']);p.add_argument('--workers',type=int,default=64);a=p.parse_args()
    if a.stage=='chemistry':chemistry(a.workers)
    else:globals()[a.stage]()

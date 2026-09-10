#!/usr/bin/env python3
"""Label-independent, resumable atom/residue/CA feature preparation.

Downstream temporal labels are NEVER inputs to representation generation.
The public DrugCLIP checkpoint is not certified to predate our time cutoffs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lmdb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from biomaster.odti_local_features_v3 import molecular_graph
from biomaster.odti_pockets_v3 import build_pocket_store, DEFAULT_POCKET_MASTER, THREE_TO_ONE, file_identity

SOURCE = ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
OUTPUT = ROOT / 'outputs/biomaster_unified_interaction_20260906/features'
CHECKPOINT = ROOT / 'third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds/fold_0.pt'


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def conformer(job):
    index, smiles = job
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'invalid canonical SMILES {index}')
    # All explicit hydrogens removed in the upstream pretrained interface.
    mol = Chem.RemoveHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    graph = molecular_graph(Chem.MolToSmiles(mol, canonical=False), max_atoms=100000)
    # Re-parse the SAME string for geometry, so graph and coordinates share order.
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=False))
    atom_features = graph['atom_features'].astype(np.float16)
    src, dst = np.nonzero(graph['bond_type'])
    record = dict(index=int(index), smiles=smiles, atoms=[a.GetSymbol() for a in mol.GetAtoms()],
                  atom_features=atom_features, edges=np.array([src, dst], np.int32),
                  bond_type=graph['bond_type'][src, dst].astype(np.uint8),
                  bond_stereo=graph['bond_stereo'][src, dst].astype(np.uint8),
                  graph_mean=atom_features.astype(np.float32).mean(0), status='oversized_global_graph_fallback')
    if len(atom_features) <= 128:
        hmol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int((20260906 + index) % 2147483647)
        params.numThreads = 1
        params.timeout = 8
        params.maxIterations = 100
        code = AllChem.EmbedMolecule(hmol, params)
        if code == 0:
            # Fixed finite optimization budget; its convergence flag is retained.
            try:
                if AllChem.MMFFHasAllMoleculeParams(hmol):
                    opt = AllChem.MMFFOptimizeMolecule(hmol, maxIters=50)
                    kind = 'mmff'
                else:
                    opt = AllChem.UFFOptimizeMolecule(hmol, maxIters=50)
                    kind = 'uff'
            except Exception:
                opt, kind = -1, 'none'
            xyz = np.asarray(hmol.GetConformer().GetPositions()[:len(atom_features)], np.float32)
            if not np.isfinite(xyz).all():
                raise FloatingPointError(f'coordinates {index}')
            record.update(coordinates=xyz, status=f'etkdg3d_{kind}_{opt}')
        else:
            record['status'] = 'embedding_failed_graph_fallback'
    return int(index), record


def prepare_conformers(out, workers):
    molecules = pd.read_csv(SOURCE / 'MOLECULES.csv.gz')
    assert np.array_equal(molecules.drug_feature_index, np.arange(len(molecules)))
    pool = pd.read_csv(SOURCE / 'QUERY_POOL.csv.gz')
    old = pd.read_csv(SOURCE / 'OLD_DRUG_INDEX.csv')
    ids = np.unique(np.r_[pool.drug_feature_index, old.drug_feature_index]).astype(int)
    np.save(out / 'REQUIRED_MOLECULE_IDS.npy', ids)
    env = lmdb.open(str(out / 'MOLECULES.lmdb'), subdir=False, map_size=24*1024**3,
                    lock=True, readahead=False)
    with env.begin() as txn:
        missing = [int(i) for i in ids if txn.get(str(i).encode()) is None]
    started = time.monotonic()
    print(json.dumps({'stage':'conformers', 'required':len(ids), 'missing':len(missing)}), flush=True)
    jobs = ((i, molecules.model_ligand_smiles.iloc[i]) for i in missing)
    # Spawn avoids inherited LMDB handles and CUDA contexts.
    with mp.get_context('spawn').Pool(workers, maxtasksperchild=4000) as workers_pool:
        pending = []
        for j, (i, record) in enumerate(workers_pool.imap_unordered(conformer, jobs, chunksize=16), 1):
            pending.append((str(i).encode(), pickle.dumps(record, protocol=5)))
            if len(pending) >= 512 or j == len(missing):
                with env.begin(write=True) as txn:
                    for key, value in pending:
                        txn.put(key, value, overwrite=False)
                pending.clear()
                status = dict(stage='conformers', completed=len(ids)-len(missing)+j,
                              required=len(ids), seconds=round(time.monotonic()-started,1))
                write_json(out / 'CONFORMER_STATUS.json', status)
                print(json.dumps(status), flush=True)
    env.sync()
    env.close()


def load_pretrained():
    import torch
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_drugclip_dtwg_subset_affinity import load_task_and_model
    task, model = load_task_and_model(torch.device('cuda'))
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    missing, unexpected = model.load_state_dict(state['model'], strict=False)
    unused_heads={f'{head}.linear{layer}.{kind}' for head in
                  ['mol_project_2','mol_project_fake','pocket_project_2','pocket_project_fake']
                  for layer in [1,2] for kind in ['weight','bias']}
    if missing or set(unexpected)!=unused_heads:
        raise ValueError(f'pretrained state mismatch: {missing}, {unexpected}')
    del state
    model.eval().requires_grad_(False)
    return task, model


def pretrained_batch(records, dictionary, device='cuda'):
    import torch
    n = max(len(r['atoms']) for r in records) + 2
    tokens = torch.full((len(records), n), dictionary.pad(), dtype=torch.long)
    distances = torch.zeros(len(records), n, n)
    edges = torch.zeros(len(records), n, n, dtype=torch.long)
    for j, r in enumerate(records):
        k = len(r['atoms'])
        t = torch.tensor([dictionary.bos()] + [dictionary.index(a) for a in r['atoms']] + [dictionary.eos()])
        tokens[j, :k+2] = t
        xyz = torch.from_numpy(r['coordinates'].copy())
        xyz -= xyz.mean(0, keepdim=True)
        # Upstream specials have ZERO distances, not distances to the centroid.
        distances[j, 1:k+1, 1:k+1] = torch.cdist(xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')
        edges[j, :k+2, :k+2] = t[:, None] * len(dictionary) + t[None, :]
    return tuple(x.to(device) for x in (tokens, distances, edges))


def pretrained_tokens(model, batch):
    st, dist, et = batch
    mol = model.mol_model
    bias = mol.gbf_proj(mol.gbf(dist, et)).permute(0,3,1,2).contiguous()
    return mol.encoder(mol.embed_tokens(st), padding_mask=st.eq(mol.padding_idx),
                       attn_mask=bias.view(-1, st.shape[1], st.shape[1]))[0]


def heavy_record(record):
    """Match upstream removal of ALL hydrogens, including explicit isotopes."""
    keep=np.array([atom!='H' for atom in record['atoms']],bool)
    if keep.all():return record
    out=dict(record)
    remap=np.full(len(keep),-1,int);remap[keep]=np.arange(keep.sum())
    edges=record['edges'];valid=keep[edges[0]]&keep[edges[1]]
    out.update(atoms=[a for a,k in zip(record['atoms'],keep) if k],
               atom_features=record['atom_features'][keep],edges=remap[edges[:,valid]],
               bond_type=record['bond_type'][valid],bond_stereo=record['bond_stereo'][valid])
    if 'coordinates' in record:out['coordinates']=record['coordinates'][keep]
    return out


def encode_atoms(out):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    ids = np.load(out / 'REQUIRED_MOLECULE_IDS.npy')
    n = len(pd.read_csv(SOURCE / 'MOLECULES.csv.gz', usecols=['drug_feature_index']))
    env = lmdb.open(str(out / 'MOLECULES.lmdb'), subdir=False, readonly=True, lock=False, readahead=True)
    indexpath = out / 'ATOM_INDEX.npz'
    if not indexpath.exists():
        lengths = np.zeros(n, np.int32)
        status_counts = Counter()
        graph_mean = np.zeros((n,40), np.float32)
        with env.begin() as txn:
            for i in ids:
                data = txn.get(str(i).encode())
                if data is None:
                    raise RuntimeError(f'conformer stage incomplete at {i}')
                r = heavy_record(pickle.loads(data))
                status_counts[r['status']] += 1
                graph_mean[i] = r['graph_mean']
                # Failed conformers retain FULL graphs and global graph summaries.
                lengths[i] = len(r['atoms']) if len(r['atoms']) <= 128 else 0
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        np.savez(indexpath, lengths=lengths, offsets=offsets, graph_mean=graph_mean)
        write_json(out / 'CONFORMER_AUDIT.json', dict(required=len(ids), counts=dict(status_counts),
                   local_atom_limit=128, no_truncation=True, coordinates_are_ligand_only=True))
    info = np.load(indexpath)
    lengths, offsets = info['lengths'], info['offsets']
    total = int(offsets[-1])
    def bank(name, shape, dtype):
        p = out / (name+'.npy')
        return np.load(p,mmap_mode='r+') if p.exists() else np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
    tokens = bank('ATOM_TOKENS',(total,512),np.float16)
    features = bank('ATOM_CHEMISTRY',(total,40),np.float16)
    # Compact per-atom neighbor indices, maximum explicitly checked, never truncated.
    neighbors = bank('ATOM_NEIGHBORS',(total,8),np.int16)
    bond = bank('ATOM_BOND',(total,8),np.uint8)
    stereo = bank('ATOM_STEREO',(total,8),np.uint8)
    global_bank = bank('MOLECULE_GLOBAL',(n,512),np.float32)
    available = bank('PRETRAINED_AVAILABLE',(n,),bool)
    done = bank('ATOM_DONE',(n,),bool)
    task, model = load_pretrained()
    selected = [int(i) for i in ids if not done[i]]
    selected.sort(key=lambda i:(int(lengths[i]),i))
    started = time.monotonic()
    with torch.inference_mode(), env.begin() as txn:
        for start in range(0,len(selected),128):
            chunk = selected[start:start+128]
            records = [heavy_record(pickle.loads(txn.get(str(i).encode()))) for i in chunk]
            valid = [r for r in records if 'coordinates' in r]
            if valid:
                # fp32 upstream interface; cached states are stored fp16 afterwards.
                h = pretrained_tokens(model, pretrained_batch(valid, task.dictionary)).float().cpu().numpy()
                if not np.isfinite(h).all():
                    raise FloatingPointError('nonfinite pretrained atom states')
                for j,r in enumerate(valid):
                    i = r['index']; a,b = offsets[i:i+2]
                    tokens[a:b] = h[j,1:len(r['atoms'])+1]
                    global_bank[i] = h[j,0]
                    available[i] = True
            for r in records:
                i = r['index']; a,b = offsets[i:i+2]
                if b > a:
                    features[a:b] = r['atom_features']
                    neighbors[a:b] = -1
                    for source in range(b-a):
                        take = np.flatnonzero(r['edges'][0] == source)
                        if len(take)>8:
                            raise ValueError(f'atom degree exceeds storage {i}:{source}')
                        neighbors[a+source,:len(take)] = r['edges'][1,take]
                        bond[a+source,:len(take)] = r['bond_type'][take]
                        stereo[a+source,:len(take)] = r['bond_stereo'][take]
            # Commit features before marking their rows complete.
            for arr in [tokens,features,neighbors,bond,stereo,global_bank,available]:
                arr.flush()
            done[chunk] = True
            done.flush()
            if start % 2048 == 0 or start+128>=len(selected):
                status=dict(stage='atom_tokens',completed=int(done[ids].sum()),required=len(ids),
                            seconds=round(time.monotonic()-started,1))
                write_json(out/'ATOM_STATUS.json',status);print(json.dumps(status),flush=True)
    assert done[ids].all()
    write_json(out/'ATOM_MANIFEST.json',dict(status='COMPLETE',checkpoint=file_identity(CHECKPOINT),
               molecule_index=file_identity(SOURCE/'MOLECULES.csv.gz'),atom_count=total,
               required_molecules=len(ids),pretrained_available=int(available[ids].sum()),
               source='DrugCLIP fold_0 mol_model:512 CLS and true heavy-atom states',
               precision='fp32 inference, fp16 atom cache',
               temporal_scope='Downstream label cutoffs only; public pretraining overlap/time NOT certified',
               files={p.name:file_identity(p) for p in out.glob('ATOM*.np*')},
               globals={name:file_identity(out/(name+'.npy')) for name in ['MOLECULE_GLOBAL','PRETRAINED_AVAILABLE']},
               producer=file_identity(Path(__file__)),
               labels_used=False))
    env.close()


def actual_indices(length, maximum):
    """Representative actual residues, never averaged bins."""
    return np.unique(np.linspace(0,length-1,min(length,maximum)).round().astype(int))


def prepare_targets(out, max_residues=96):
    import torch
    targets = pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz')
    pockets = build_pocket_store(DEFAULT_POCKET_MASTER,[SOURCE/'TARGET_INDEX.csv.gz'])
    write_json(out/'POCKETS.json',pockets)
    assets = pd.read_csv(ROOT/'outputs/biomaster_odti_v4_plan_20260905/CURRENT_TARGET_ASSETS_V4.csv.gz')
    residues = np.load(ROOT/'outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy',mmap_mode='r')
    sources = {}
    for r in assets.itertuples():
        if r.full_residue_available and int(r.token_length)==len(r.protein_sequence):
            sources[r.sequence_sha256]=(residues,int(r.token_offset),int(r.token_length))
    added = ROOT/'outputs/biomaster_odti_v4_20260905/features'
    added_array = np.load(added/'ESM2_ADDED_RESIDUES_FLOAT16_V4.npy',mmap_mode='r')
    for r in pd.read_csv(added/'ESM2_ADDED_RESIDUE_INDEX_V4.csv').itertuples():
        sources[r.sequence_sha256]=(added_array,int(r.token_offset),int(r.token_length))
    missing = [r for r in targets.itertuples() if r.sequence_sha256 not in sources]
    missing_dir=out/'missing_residues';missing_dir.mkdir(exist_ok=True)
    to_generate = [r for r in missing if not (missing_dir/(r.sequence_sha256+'.npy')).exists()]
    if to_generate:
        os.environ['TORCH_HOME']='/root/autodl-tmp/.cache/torch'
        import esm
        sys.path.insert(0,str(ROOT/'scripts'))
        from build_biomaster_odti_target_token_features_v1 import window_bounds
        torch.set_num_threads(4)
        model,alphabet=esm.pretrained.esm2_t33_650M_UR50D()
        model=model.cuda().eval()
        convert=alphabet.get_batch_converter()
        with torch.inference_mode():
            for i,r in enumerate(to_generate):
                seq=r.sequence
                summed=np.zeros((len(seq),1280),np.float32);counts=np.zeros((len(seq),1),np.float32)
                for lo,hi in window_bounds(len(seq),1022,128):
                    _,_,token=convert([('target',seq[lo:hi])])
                    with torch.autocast('cuda',dtype=torch.float16):
                        h=model(token.cuda(),repr_layers=[33],return_contacts=False)['representations'][33][0,1:hi-lo+1].float().cpu().numpy()
                    summed[lo:hi]+=h;counts[lo:hi]+=1
                assert (counts>0).all()
                np.save(missing_dir/(r.sequence_sha256+'.npy'),(summed/counts).astype(np.float16))
                print(json.dumps(dict(stage='missing_residues',completed=i+1,total=len(to_generate))),flush=True)
        del model;torch.cuda.empty_cache()
    for r in missing:
        array=np.load(missing_dir/(r.sequence_sha256+'.npy'))
        assert array.shape==(len(r.sequence),1280)
        sources[r.sequence_sha256]=(array,0,len(array))
    n=len(targets);k=max_residues
    arrays=dict(global_mean=np.zeros((n,1280),np.float32))
    for mode in ['sequence','site']:
        arrays[mode+'_tokens']=np.zeros((n,k,1280),np.float16)
        arrays[mode+'_indices']=np.full((n,k),-1,np.int32)
    arrays.update(ca=np.zeros((n,k,3),np.float32),quality=np.zeros((n,k),np.float32),
                  geometry_mask=np.zeros((n,k),bool),site_member=np.zeros((n,k),bool))
    audit=[]
    for r in targets.itertuples():
        assert hashlib.sha256(r.sequence.encode()).hexdigest()==r.sequence_sha256
        array,offset,length=sources[r.sequence_sha256]
        assert length==len(r.sequence)
        full=np.asarray(array[offset:offset+length],np.float32)
        if not np.isfinite(full).all():raise FloatingPointError('ESM2 full residue bank')
        i=int(r.target_feature_index);arrays['global_mean'][i]=full.mean(0)
        sequence_ids=actual_indices(length,k)
        item=pockets['targets'].get(r.sequence_sha256)
        union=sorted({j for p in item['pockets'] for j in p['residue_indices']}) if item else []
        if union:
            # Give all accepted sites representation; if the union is too large,
            # sample real residues evenly across their ordered canonical indices.
            site_ids=np.array(union)[actual_indices(len(union),k)]
            if len(site_ids)<k:
                extra=[j for j in sequence_ids if j not in set(site_ids)]
                site_ids=np.sort(np.r_[site_ids,extra[:k-len(site_ids)]]).astype(int)
        else:
            site_ids=sequence_ids
        for mode,indices in [('sequence',sequence_ids),('site',site_ids)]:
            arrays[mode+'_tokens'][i,:len(indices)]=full[indices]
            arrays[mode+'_indices'][i,:len(indices)]=indices
        if item:
            ca={}
            with Path(item['sources']['receptor']['path']).open() as handle:
                for line in handle:
                    if line.startswith('ENDMDL'):break
                    if not (line.startswith('ATOM') and line[12:16].strip()=='CA' and line[16:17] in [' ','A'] and line[21:22]=='A' and line[26:27]==' '):continue
                    j=int(line[22:26])-1
                    aa=THREE_TO_ONE.get(line[17:20].strip(),'X')
                    if 0<=j<length and aa==r.sequence[j]:
                        ca[j]=(np.array([float(line[a:a+8]) for a in [30,38,46]],np.float32),float(line[60:66])/100)
            for column,j in enumerate(site_ids):
                if j in union:
                    if j not in ca:raise ValueError('validated pocket lost CA mapping')
                    xyz,quality=ca[j]
                    arrays['ca'][i,column]=xyz
                    arrays['quality'][i,column]=np.clip(quality,0,1)
                    arrays['geometry_mask'][i,column]=True
                    arrays['site_member'][i,column]=True
        audit.append(dict(target_feature_index=i,sequence_sha256=r.sequence_sha256,length=length,
                          selected_actual_residues=len(site_ids),pocket_union_residues=len(union),
                          selected_pocket_residues=int(arrays['site_member'][i].sum()),full_mean_coverage=length))
    for name,values in arrays.items():np.save(out/('TARGET_'+name.upper()+'.npy'),values)
    pd.DataFrame(audit).to_csv(out/'TARGET_COVERAGE.csv',index=False)
    write_json(out/'TARGET_MANIFEST.json',dict(status='COMPLETE',target_index=file_identity(SOURCE/'TARGET_INDEX.csv.gz'),
                targets=n,full_residue_targets=n,new_full_residue_targets=len(missing),max_local_residues=k,
                selection='actual deterministic residue subsample + exact full-sequence mean; no token binning',
                geometry='mapped AlphaFold CA distances; predicted P2Rank sites; no cross ligand-protein distances',
                pocket_coverage=pockets['coverage'],labels_used=False,
                residue_sources=[file_identity(ROOT/name) for name in [
                    'outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_INDEX_COMBINED_V1.csv.gz',
                    'outputs/biomaster_odti_v4_20260905/features/ESM2_ADDED_RESIDUE_INDEX_V4.csv']],
                files={p.name:file_identity(p) for p in out.glob('TARGET_*.npy')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['conformers','atoms','targets','all'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=48)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ['conformers','all']:prepare_conformers(args.output,args.workers)
    if args.stage in ['targets','all']:prepare_targets(args.output)
    if args.stage in ['atoms','all']:encode_atoms(args.output)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Official sixfold DrugCLIP, preserved historical pockets + audited AF extensions."""
import pickle
import time
import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, dump, directory, historical


def write_lmdb(path, records):
    if path.exists():
        path.unlink()
    env = lmdb.open(str(path), subdir=False, map_size=2*1024**3)
    with env.begin(write=True) as transaction:
        for i, record in enumerate(records):
            transaction.put(f'{i:06d}'.encode(), pickle.dumps(record, protocol=4))
    env.sync()
    env.close()


def main():
    name = 'DrugCLIP'
    start = time.time()
    status(name, 'PREPARING_NATIVE_POCKETS')
    out = directory(name)
    drugs, targets = inputs()
    from prepare_drugclip_project_inputs_v1 import pocket_record
    strict = ROOT / 'outputs/strict_dta_720x338_v1/inputs'
    recovered = ROOT / 'outputs/recovered_dta_720x46_v1/inputs'
    records, metadata = {}, []
    for folder, manifest, lmdb_name, route in [
        (strict, 'STRICT338_DRUGCLIP_POCKET_MANIFEST.csv', 'strict338_experimental_pockets.lmdb', 'HISTORICAL_EXPERIMENTAL_POCKET'),
        (recovered, 'RECOVERED44_DRUGCLIP_POCKET_MANIFEST.csv', 'recovered44_predicted_pockets.lmdb', 'HISTORICAL_PREDICTED_POCKET')]:
        frame = pd.read_csv(folder/manifest)
        env = lmdb.open(str(folder/lmdb_name), subdir=False, readonly=True, lock=False)
        with env.begin() as tx:
            for row in frame.to_dict('records'):
                if pd.isna(row.get('lmdb_index')) or row.get('status') != 'OK':
                    continue
                target = row['target_chembl_id']
                record = pickle.loads(tx.get(str(int(row['lmdb_index'])).encode()))
                record['pocket'] = target
                records[target] = record
                metadata.append(dict(target_id=target, route=route, source=str((folder/manifest).relative_to(ROOT)),
                                     atom_count=len(record['pocket_atoms'])))
        env.close()
    assert len(records)==382
    registry = pd.read_csv(ROOT/'outputs/retrain_20260901/target_registry_745_feature_store_v1/TARGET_REGISTRY_745_FEATURE_INDEX_V1.csv.gz').set_index('target_chembl_id')
    failures = []
    parser = PDBParser(QUIET=True)
    for target in targets.itertuples():
        if target.target_id in records:
            continue
        row = registry.loc[target.target_id]
        assert row.sequence == target.sequence
        try:
            pdb = ROOT / str(row.af_pdb_path)
            structure = parser.get_structure(target.target_id, pdb)[0]
            chains = list(structure.get_chains())
            assert len(chains)==1 and ''.join(seq1(r.resname) for r in chains[0] if r.id[0]==' ') == target.sequence, 'AF_SEQUENCE_MISMATCH'
            record, reason, residues, atoms = pocket_record(pd.Series(dict(pdb_path=str(pdb),
                top_pocket_residue_ids=row.p2rank_residue_ids, sequence_key=target.target_id)))
            if record is None:
                raise ValueError(reason)
            records[target.target_id] = record
            metadata.append(dict(target_id=target.target_id, route='EXACT_SEQUENCE_AF_V6_P2RANK_TOP1',
                                 source=str(pdb), structure_sha256=sha(pdb), atom_count=atoms, residues=residues,
                                 p2rank_probability=float(row.p2rank_top_probability)))
        except Exception as exc:
            failures.append(dict(target_id=target.target_id, reason=f'{type(exc).__name__}: {exc}'))
    target_order = [t for t in targets.target_id if t in records]
    write_lmdb(out/'pockets.lmdb', [records[t] for t in target_order])
    old_m = pd.read_csv(strict/'STRICT720_DRUGCLIP_LIGAND_MANIFEST.csv').set_index('ligand_inchikey')
    mol_records = []
    env = lmdb.open(str(strict/'strict720_ligands.lmdb'), subdir=False, readonly=True, lock=False)
    with env.begin() as tx:
        for drug in drugs.itertuples():
            row = old_m.loc[drug.drug_id]
            record = pickle.loads(tx.get(str(int(row.lmdb_index)).encode()))
            assert Chem.MolToInchiKey(Chem.MolFromSmiles(record.get('smi',record.get('smiles')))) == drug.drug_id
            mol_records.append(record)
    env.close()
    write_lmdb(out/'molecules.lmdb', mol_records)
    pd.DataFrame(metadata).to_csv(out/'POCKET_AUDIT.csv', index=False)
    dump(out/'INPUT_FAILURES.json', failures)
    from run_drugclip_dtwg_subset_affinity import load_task_and_model
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    task, model = load_task_and_model(torch.device('cuda'))
    task.args.max_pocket_atoms = 256
    molset = task.load_mols_dataset_dtwg(str(out/'molecules.lmdb'), 'atoms', 'coordinates', dataset_type=1)
    pktset = task.load_pockets_dataset(str(out/'pockets.lmdb'))
    weight_dir = ROOT/'third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds'
    unused = {f'{h}.linear{n}.{k}' for h in ['mol_project_2','mol_project_fake','pocket_project_2','pocket_project_fake'] for n in [1,2] for k in ['weight','bias']}
    mf, pf, checks = [], [], []
    with torch.inference_mode():
        for fold in range(6):
            ckpt = weight_dir/f'fold_{fold}.pt'
            state = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
            missing, unexpected = model.load_state_dict(state['model'], strict=False)
            assert not missing and set(unexpected)==unused
            del state
            model.eval()
            arrays = []
            for dataset, batch_size, kind in [(molset,32,'mol'),(pktset,4,'pocket')]:
                encoder = model.mol_model if kind=='mol' else model.pocket_model
                project = model.mol_project if kind=='mol' else model.pocket_project
                values, names = [], []
                for sample in DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collater, num_workers=0, shuffle=False):
                    net = sample['net_input']
                    token = net[kind+'_src_tokens'].cuda()
                    dist = net[kind+'_src_distance'].cuda()
                    edges = net[kind+'_src_edge_type'].cuda()
                    bias = encoder.gbf_proj(encoder.gbf(dist,edges)).permute(0,3,1,2).contiguous()
                    hidden = encoder.encoder(encoder.embed_tokens(token), padding_mask=token.eq(encoder.padding_idx),
                             attn_mask=bias.view(-1,token.shape[1],token.shape[1]))[0][:,0]
                    values.append(torch.nn.functional.normalize(project(hidden),dim=-1).float().cpu().numpy())
                    if kind=='pocket':
                        names.extend(sample['pocket_name'])
                if kind=='pocket':
                    assert names==target_order
                arrays.append(np.concatenate(values))
            mf.append(arrays[0])
            pf.append(arrays[1])
            checks.append(dict(fold=fold,sha256=sha(ckpt),missing_keys=missing,unused_auxiliary_keys=sorted(unexpected)))
            status(name,'ENCODING_NATIVE_FOLDS',completed_folds=fold+1,pockets=len(target_order),elapsed_seconds=time.time()-start)
            print('fold',fold,'complete',flush=True)
    mol,pocket = np.stack(mf,axis=1),np.stack(pf,axis=1)
    np.save(out/'MOL_EMBEDDINGS.npy',mol)
    np.save(out/'POCKET_EMBEDDINGS.npy',pocket)
    scores = np.einsum('dfh,tfh->dtf',mol,pocket,optimize=True).mean(axis=2)
    frame = pd.DataFrame(dict(drug_id=np.repeat(drugs.drug_id,len(target_order)),target_id=np.tile(target_order,720),score=scores.reshape(-1)))
    old = frame.merge(historical(name),on=['drug_id','target_id'],suffixes=('','_historical'))
    dump(out/'HISTORICAL_REPLAY.json',dict(pairs=len(old),max_abs_delta=float((old.score-old.score_historical).abs().max()),
         spearman=float(old[['score','score_historical']].corr(method='spearman').iloc[0,1]),
         note='All current scores uniformly recomputed FP32; historical catalogue used different encoding precision/batching.'))
    contract(name,checkpoints=checks,adapter_sha256=sha(__file__),head='Mean sixfold native normalized embedding cosine',precision='float32_cuda_tf32_off',
             pocket_policy='Preserve338 historical experimental +44 predicted pockets; remaining targets exact-sequence AFv6/P2Rank top1, no score-based pocket selection.',
             pocket_manifest_sha256=sha(out/'POCKET_AUDIT.csv'),pockets=len(target_order),atom_cap=256,
             note='Mixed structure-source strata explicitly recorded; not a sequence-only architectural comparison.')
    save_scores(name,frame)
    status(name,'COMPLETE' if len(frame)==536400 else 'COMPLETE_WITH_MISSING_POCKETS',scored_pairs=len(frame),elapsed_seconds=time.time()-start)


if __name__=='__main__':
    main()

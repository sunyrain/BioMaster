#!/usr/bin/env python3
"""Auditable DTIAM/DrugCLIP predictions for the existing three-target query.

Run esm/prepare/drugclip with the project Python, dtiam with .venv_dtiam_compat.
No fitting, model selection, or changes to the released catalog are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd

QUERY = ROOT / 'outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf'
OUT = QUERY / 'comparators'
BUNDLE = ROOT / 'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'
DTI = ROOT / 'outputs/old_drug_target_sota_v1/public_retrained_v1'
DFEAT = DTI / 'dtiam_deployment_feature_store_v1'
RUN = DTI / 'dtiam_same_data_compatible_v1/S5_OLD_DRUG_ENTITY_COLD__fold_-1__OFFICIAL_DEFAULT_COMPAT_V1'
CLIP = ROOT / 'third_party/sota_dti_2026/Drug-The-Whole-Genome'
GENES = ['LYVE1', 'SLC8A1', 'HGF']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def targets():
    rows = json.loads((QUERY / 'TARGETS.json').read_text())
    assert [x['gene_symbol'] for x in rows] == GENES
    return rows


def drugs():
    d = pd.read_csv(BUNDLE / 'drugs.csv.gz')
    assert len(d) == 720 and d.drug_id.is_unique
    return d


def esm_features():
    import torch
    import esm
    os.environ['TORCH_HOME'] = '/root/autodl-tmp/.cache/torch'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.cuda().eval()
    convert = alphabet.get_batch_converter()
    control_index = pd.read_csv(DFEAT / 'DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz')
    control = control_index[control_index.sequence.str.len().between(150, 300)].iloc[0]
    values = []
    with torch.inference_mode():
        for r in targets() + [dict(gene_symbol='CONTROL', sequence=control.sequence)]:
            seq = r['sequence'][:1022]
            _, _, tok = convert([(r['gene_symbol'], seq)])
            h = model(tok.cuda(), repr_layers=[33], return_contacts=False)['representations'][33]
            # Preserve official DTIAM definition: exclude BOS, INCLUDE EOS.
            v = h[0, 1:].mean(0).float().cpu().numpy()
            values.append(v)
            print('ESM', r['gene_symbol'], len(seq), flush=True)
    expected = np.load(DFEAT / 'DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy')[int(control.target_feature_index)]
    delta = float(np.max(np.abs(values[-1] - expected)))
    if delta > 2e-5: raise ValueError(('DTIAM ESM compatibility control failed', delta))
    np.save(OUT / 'DTIAM_TARGET_ESM2.npy', np.stack(values[:3]))
    save('DTIAM_ESM_AUDIT.json', dict(status='PASS', control_target_index=int(control.target_feature_index),
         control_max_abs_diff=delta, pooling='FP32 final layer mean residues plus EOS; BOS excluded',
         truncate_at=1022, actual_truncated_queries=0,
         targets_sha256=sha(QUERY / 'TARGETS.json'), feature_sha256=sha(OUT / 'DTIAM_TARGET_ESM2.npy')))


def write_lmdb(path, rows):
    import lmdb
    if path.exists(): raise FileExistsError(path)
    env = lmdb.open(str(path), subdir=False, map_size=1024**3)
    with env.begin(write=True) as tx:
        # Fixed width keys make LMDB lexical order equal catalog order.
        for i, r in enumerate(rows): tx.put(f'{i:06d}'.encode(), pickle.dumps(r, protocol=4))
    env.sync(); env.close()


def prepare():
    import lmdb
    import requests
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    from rdkit import Chem
    from rdkit.Chem import AllChem
    pocket_rows, pocket_meta, input_files, coverage = [], [], {}, {}
    java = ROOT / '.conda_envs/pocket_tools/lib/jvm/bin/java'
    prank = ROOT / 'tools/p2rank_2.5.1'
    for r in targets():
        gene, acc = r['gene_symbol'], r['uniprot_accession']
        pdb = OUT / f'{gene}_AF.pdb'
        if not pdb.exists():
            response = requests.get(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}', timeout=45)
            response.raise_for_status(); entries = response.json(); save(f'{gene}_AF_API.json', entries)
            entry = next(x for x in entries if x['uniprotAccession'] == acc and x['uniprotStart'] == 1 and x['uniprotEnd'] == r['length'] and x['entryId'] == f'AF-{acc}-F1')
            response = requests.get(entry['pdbUrl'], timeout=45); response.raise_for_status()
            pdb.write_bytes(response.content)
        structure = PDBParser(QUIET=True).get_structure(gene, str(pdb))[0]
        residues = [x for x in structure.get_residues() if x.id[0] == ' ']
        assert ''.join(seq1(x.resname) for x in residues) == r['sequence'], gene
        result = OUT / f'{gene.lower()}_p2rank'
        csv = result / (pdb.name + '_predictions.csv')
        marker = result / 'QUERY_COMPLETE.json'
        if not marker.exists():
            cmd = [str(java), '-Xmx2048m', '-cp', str(prank/'bin/p2rank.jar')+':'+str(prank/'bin/lib/*'),
                   'cz.siret.prank.program.Main', 'predict', '-f', str(pdb), '-o', str(result),
                   '-c', 'alphafold', '-threads', '1', '-visualizations', '0']
            with (OUT/f'{gene}_P2RANK.log').open('w') as log:
                subprocess.run(cmd, check=True, timeout=300, stdout=log, stderr=subprocess.STDOUT)
            marker.write_text(json.dumps(dict(pdb_sha256=sha(pdb), csv_sha256=sha(csv))))
        signed = json.loads(marker.read_text())
        assert signed == dict(pdb_sha256=sha(pdb), csv_sha256=sha(csv))
        frame = pd.read_csv(csv); frame.columns = frame.columns.str.strip()
        retained = 0
        for row in frame.to_dict('records'):
            if float(row['probability']) < .2: continue
            ids = {tuple(x.rsplit('_', 1)) for x in str(row['residue_ids']).split()}
            selected = [x for x in residues if (x.parent.id, str(x.id[1])) in ids]
            assert len(selected) == len(ids)
            atoms = [a for x in selected for a in x.get_atoms() if a.element not in {'H', 'D'}]
            assert 5 <= len(selected) and 0 < len(atoms) <= 512
            name = gene + '_P2RANK_' + str(int(row['rank']))
            pocket_rows.append(dict(pocket=name, pocket_atoms=[a.element for a in atoms],
                                    pocket_coordinates=np.array([a.coord for a in atoms], np.float32)))
            pocket_meta.append(dict(gene_symbol=gene, pocket=name, uniprot=acc,
                p2rank_probability=float(row['probability']), residue_ids=' '.join(sorted('_'.join(x) for x in ids)),
                residues=len(selected), atoms=len(atoms), mean_residue_plddt=float(np.mean([x['CA'].bfactor for x in selected])),
                encoder_atom_cap=256, cropping_required=len(atoms)>256, source='AlphaFold_v6_P2Rank_2.5.1'))
            retained += 1
        coverage[gene] = dict(retained=retained, predicted=len(frame), max_probability=float(frame.probability.max()) if len(frame) else None)
        input_files[str(pdb)] = sha(pdb); input_files[str(csv)] = sha(csv)
        print('POCKETS', gene, retained, flush=True)
    if not (OUT/'pockets.lmdb').exists(): write_lmdb(OUT/'pockets.lmdb', pocket_rows)
    pd.DataFrame(pocket_meta).to_csv(OUT/'P2RANK_POCKET_MANIFEST.csv', index=False)
    library, molecule_rows, molmeta = drugs(), [], []
    env = lmdb.open(str(ROOT/'outputs/biomaster_unified_interaction_20260906/features/MOLECULES.lmdb'), subdir=False, readonly=True, lock=False)
    with env.begin() as tx:
        for r in library.itertuples(index=False):
            raw = tx.get(str(r.native_feature_index).encode()); old = pickle.loads(raw)
            canonical = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True)
            assert canonical(old['smiles']) == canonical(r.smiles)
            route = 'existing_independent_ETKDG'
            if 'coordinates' in old:
                x = dict(atoms=old['atoms'], coordinates=[old['coordinates']], smi=r.smiles, name=r.drug_id)
            elif r.name == 'quinidine' and (OUT/'QUINIDINE_PUBCHEM_3D.sdf').exists():
                sdf = OUT/'QUINIDINE_PUBCHEM_3D.sdf'
                mol = Chem.MolFromMolBlock(sdf.read_text().split('$$$$')[0], removeHs=True)
                assert mol is not None and Chem.MolToInchiKey(mol) == r.drug_id
                assert mol.GetNumAtoms() == len(old['atoms']) and mol.GetConformer().Is3D()
                x = dict(atoms=[a.GetSymbol() for a in mol.GetAtoms()], coordinates=[np.asarray(mol.GetConformer().GetPositions(),np.float32)], smi=r.smiles, name=r.drug_id)
                route = 'PubChem3D_exact_InChIKey_after_4_failed_ETKDG_attempts'
                input_files[str(sdf)] = sha(sdf)
            else:
                mol = Chem.AddHs(Chem.MolFromSmiles(r.smiles))
                status = -1
                for attempt in range(4):
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 20260908 + int(r.native_feature_index) + attempt
                    params.numThreads = 1; params.timeout = 15; params.maxIterations = 100
                    params.useRandomCoords = True
                    status = AllChem.EmbedMolecule(mol, params)
                    if status == 0: break
                if status != 0: raise ValueError((r.name, 'bounded conformer generation failed'))
                opt = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                mol = Chem.RemoveHs(mol)
                x = dict(atoms=[a.GetSymbol() for a in mol.GetAtoms()], coordinates=[np.asarray(mol.GetConformer().GetPositions(),np.float32)], smi=r.smiles, name=r.drug_id)
                route = f'fresh_ETKDGv3_randomcoords_attempt{attempt}_MMFF500_status{opt}'
            x['smiles'] = r.smiles
            molecule_rows.append(x)
            molmeta.append(dict(drug_id=r.drug_id, name=r.name, smiles=r.smiles, native_feature_index=int(r.native_feature_index),
                lmdb_key=f'{len(molmeta):06d}', geometry_source=route, source_record_sha256=hashlib.sha256(raw).hexdigest()))
    env.close()
    if not (OUT/'molecules.lmdb').exists(): write_lmdb(OUT/'molecules.lmdb', molecule_rows)
    pd.DataFrame(molmeta).to_csv(OUT/'DRUGCLIP_MOLECULE_MANIFEST.csv', index=False)
    save('PREPARATION_AUDIT.json', dict(status='PASS', drugs=720, pocket_counts=coverage,
        sequence_checks='all AlphaFold chains equal queried full human sequence', inputs=input_files,
        pocket_policy='all P2Rank probability >= 0.2; no ligand or score-dependent selection',
        inputs_sha256={p:sha(OUT/p) for p in ['pockets.lmdb','molecules.lmdb','P2RANK_POCKET_MANIFEST.csv','DRUGCLIP_MOLECULE_MANIFEST.csv']},
        p2rank_jar_sha256=sha(prank/'bin/p2rank.jar'), p2rank_config_sha256=sha(prank/'config/alphafold.groovy')))


def dtiam():
    from autogluon.tabular import TabularPredictor
    from threadpoolctl import threadpool_limits
    library = drugs()
    index = pd.read_csv(DFEAT/'DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz').set_index('ligand_inchikey').loc[library.drug_id]
    assert (index.ligand_smiles.to_numpy() == library.smiles.to_numpy()).all()
    d = np.load(DFEAT/'DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy')[index.drug_feature_index.to_numpy(int)]
    t = np.load(OUT/'DTIAM_TARGET_ESM2.npy')
    cols = [f'bermol_{i:04d}' for i in range(768)] + [f'esm2_{i:04d}' for i in range(1280)]
    summary = json.loads((RUN/'RUN_SUMMARY_V1.json').read_text())
    assert summary['status'] == 'PASS'
    # Sign all serialized model dependencies, including ensemble members.
    hashes = {str(p.relative_to(RUN/'predictor')):sha(p) for p in sorted((RUN/'predictor').rglob('*'))
              if p.is_file() and 'utils/data' not in str(p) and '/logs/' not in str(p)}
    predictor = TabularPredictor.load(str(RUN/'predictor'), require_version_match=True)
    assert predictor.model_best == summary['best_model']
    with threadpool_limits(limits=2):
        for i, gene in enumerate(GENES):
            x = pd.DataFrame(np.concatenate([d, np.repeat(t[i:i+1], 720, axis=0)], axis=1), columns=cols)
            y = predictor.predict_proba(x)[1].to_numpy(np.float32)
            assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
            frame = library.copy(); frame['gene_symbol']=gene; frame['dtiam_probability']=y
            frame = frame.sort_values(['dtiam_probability','drug_id'],ascending=[False,True],kind='stable')
            frame['dtiam_rank'] = np.arange(1,721)
            frame.to_csv(OUT/f'{gene}_DTIAM_720.csv', index=False)
            print('DTIAM SCORED', gene, flush=True)
    save('DTIAM_MANIFEST.json',dict(status='PASS', model='DTIAM_OFFICIAL_REPRESENTATION_COMPATIBLE_RETRAIN_S5',
        best_model=predictor.model_best, train_rows=summary['counts']['train_rows'], python=sys.version,
        predictor_sha256=hashes, run_summary_sha256=sha(RUN/'RUN_SUMMARY_V1.json'),
        inputs={str(p):sha(p) for p in [BUNDLE/'drugs.csv.gz', DFEAT/'DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy',
            DFEAT/'DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz', OUT/'DTIAM_TARGET_ESM2.npy']},
        caveat='AutoGluon1.4 compatible project-data retrain, not original paper downstream weights; outputs uncalibrated for these queries'))


def drugclip():
    import torch
    from torch.utils.data import DataLoader
    from run_drugclip_dtwg_subset_affinity import load_task_and_model
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    task, model = load_task_and_model(torch.device('cuda'))
    task.args.max_pocket_atoms = 256
    molset = task.load_mols_dataset_dtwg(str(OUT/'molecules.lmdb'), 'atoms', 'coordinates', dataset_type=1)
    pktset = task.load_pockets_dataset(str(OUT/'pockets.lmdb'))
    loaders = [(molset,32,'mol'), (pktset,4,'pocket')]
    all_m, all_p, checks = [], [], []
    expected_unused={f'{h}.linear{n}.{k}' for h in ['mol_project_2','mol_project_fake','pocket_project_2','pocket_project_fake']
                     for n in [1,2] for k in ['weight','bias']}
    signatures = json.loads((CLIP/'data/model_weights/6_folds/DRUGCLIP_SIXFOLD_WEIGHTS_V1.json').read_text())
    pocket_names = None
    with torch.inference_mode():
        for fold in range(6):
            ckpt = CLIP/f'data/model_weights/6_folds/fold_{fold}.pt'
            digest=sha(ckpt); assert digest == signatures['folds'][fold]['sha256']
            state = torch.load(ckpt,map_location='cpu',weights_only=False,mmap=True)
            missing, unexpected = model.load_state_dict(state['model'],strict=False)
            assert not missing and set(unexpected)==expected_unused
            del state
            model.eval(); embeddings=[]
            for dataset, bsz, kind in loaders:
                encoder = model.mol_model if kind=='mol' else model.pocket_model
                project = model.mol_project if kind=='mol' else model.pocket_project
                arr, names = [], []
                for sample in DataLoader(dataset,batch_size=bsz,collate_fn=dataset.collater,num_workers=0,shuffle=False):
                    net=sample['net_input']; st=net[kind+'_src_tokens'].cuda()
                    dist=net[kind+'_src_distance'].cuda(); et=net[kind+'_src_edge_type'].cuda()
                    bias=encoder.gbf_proj(encoder.gbf(dist,et)).permute(0,3,1,2).contiguous()
                    h=encoder.encoder(encoder.embed_tokens(st),padding_mask=st.eq(encoder.padding_idx),
                                      attn_mask=bias.view(-1,st.shape[1],st.shape[1]))[0][:,0]
                    v=torch.nn.functional.normalize(project(h),dim=-1)
                    arr.append(v.float().cpu().numpy())
                    if kind=='pocket':names.extend(sample['pocket_name'])
                embeddings.append(np.concatenate(arr))
                if kind=='pocket':
                    if pocket_names is None: pocket_names=names
                    assert names==pocket_names
            all_m.append(embeddings[0]);all_p.append(embeddings[1])
            checks.append(dict(fold=fold,sha256=digest,missing_keys=missing,unused_auxiliary_keys=sorted(unexpected)))
            print('DRUGCLIP FOLD COMPLETE',fold,flush=True)
    m=np.stack(all_m,axis=1);p=np.stack(all_p,axis=1)
    assert m.shape==(720,6,128) and np.isfinite(m).all() and np.isfinite(p).all()
    np.save(OUT/'DRUGCLIP_MOLECULE_EMBEDDINGS.npy',m)
    np.save(OUT/'DRUGCLIP_P2RANK_EMBEDDINGS.npy',p)
    save('DRUGCLIP_ENCODED_POCKET_NAMES.json',pocket_names)
    meta=pd.read_csv(OUT/'P2RANK_POCKET_MANIFEST.csv')
    assert pocket_names==meta.pocket.tolist()
    # Official retrieval order: average six fold cosines, median/MAD per
    # pocket over the candidate library, then max across pockets.
    score_pockets(m,p,meta,'P2RANK')
    base=CLIP/'data_downloads/benchmark_throughput'
    names=np.load(base/'dtwg_af_names_.npy',allow_pickle=True)
    raw=np.load(base/'dtwg_af_embeddings.npy',mmap_mode='r')
    selected=[];records=[]
    for r in targets():
        for i,n in enumerate(names):
            match=re.search(r'AF-([A-Z0-9]+)-F1-',str(n))
            if match and match[1]==r['uniprot_accession']:
                selected.append(i);records.append(dict(gene_symbol=r['gene_symbol'],pocket=str(n),official_row=i))
    dtmeta=pd.DataFrame(records);dtmeta.to_csv(OUT/'OFFICIAL_DTWG_POCKETS.csv',index=False)
    dt=np.asarray(raw[selected]).reshape(-1,6,128)
    score_pockets(m,dt,dtmeta,'DTWG')
    save('DRUGCLIP_MANIFEST.json',dict(status='PASS',weights='Official Drug-The-Whole-Genome six folds 0..5',
        folds=checks, precision='FP32, TF32 disabled', mol_shape=list(m.shape),pocket_shape=list(p.shape),
        primary='AFv6 + ligand-blind P2Rank >=0.2 where eligible; LYVE1 uses official DTWG because no P2Rank pocket qualifies',
        scoring='mean six fold cosines; per-pocket 0.6745*(score-median)/(MAD+1e-6) across 720; max pockets',
        secondary='official preencoded DTWG AFv4 pockets for covered targets; SLC8A1 absent',
        dtwg_norm_range=[float(np.linalg.norm(dt,axis=-1).min()),float(np.linalg.norm(dt,axis=-1).max())],
        source_hashes={str(x.relative_to(ROOT)):sha(x) for x in [CLIP/'unimol/tasks/drugclip.py',CLIP/'unimol/models/drugclip.py',
            base/'dtwg_af_names_.npy',base/'dtwg_af_embeddings.npy',OUT/'molecules.lmdb',OUT/'pockets.lmdb']},
        caution='Retrieval scores, not calibrated affinity; public training overlap unknown'))


def score_pockets(m,p,meta,route):
    library=drugs()
    raw=np.einsum('dfh,pfh->pdf',m,p,optimize=True)
    cosine=raw.mean(axis=2)
    median=np.median(cosine,axis=1,keepdims=True)
    mad=np.median(np.abs(cosine-median),axis=1,keepdims=True)
    mean=.6745*(cosine-median)/(mad+1e-6)
    fold_median=np.median(raw,axis=1,keepdims=True)
    fold_mad=np.median(np.abs(raw-fold_median),axis=1,keepdims=True)
    z=.6745*(raw-fold_median)/(fold_mad+1e-6)
    np.savez_compressed(OUT/f'DRUGCLIP_{route}_POCKET_SCORES.npz',fold_cosines=raw,fold_zscores=z,ensemble_pocket_zscores=mean)
    for gene in GENES:
        ix=np.where(meta.gene_symbol.to_numpy()==gene)[0]
        if not len(ix):continue
        y=mean[ix].max(axis=0);win=ix[mean[ix].argmax(axis=0)]
        table=library.copy();table['gene_symbol']=gene;table['drugclip_score']=y
        table['drugclip_max_mean_cosine']=cosine[ix].max(axis=0)
        table['winning_pocket']=meta.pocket.to_numpy()[win]
        table['winning_pocket_fold_cosine_std']=raw[win,np.arange(720)].std(axis=1)
        for f in range(6):
            table[f'fold_{f}_rank']=pd.Series(z[ix,:,f].max(axis=0)).rank(ascending=False,method='min').to_numpy(int)
        table['fold_top20_count']=(table[[f'fold_{f}_rank' for f in range(6)]]<=20).sum(axis=1)
        table['drugclip_cosine_rank']=table.drugclip_max_mean_cosine.rank(ascending=False,method='min').astype(int)
        table=table.sort_values(['drugclip_score','drug_id'],ascending=[False,True],kind='stable')
        table['drugclip_rank']=np.arange(1,721)
        table.to_csv(OUT/f'{gene}_DRUGCLIP_{route}_720.csv',index=False)
        print('DRUGCLIP SCORED',route,gene,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['esm','prepare','dtiam','drugclip'])
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    source=Path(__file__).read_bytes()
    (OUT/(args.stage+'_source.py')).write_bytes(source)
    {'esm':esm_features,'prepare':prepare,'dtiam':dtiam,'drugclip':drugclip}[args.stage]()
    assert Path(__file__).read_bytes()==source, 'Producer changed while executing'
    save(args.stage.upper()+'_COMPLETE.json',dict(utc=datetime.now(timezone.utc).isoformat(),producer_sha256=sha(__file__),status='PASS'))


if __name__=='__main__': main()

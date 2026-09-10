#!/usr/bin/env python3
"""Map, audit and encode real-complex structural supervision, resumably."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.structural_complex import map_complex
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json

DATA = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_data'
SOURCE = DATA / 'complexes_2020'
OUT = DATA / 'training_2020'


def map_one(annotation):
    RDLogger.DisableLog('rdApp.*')
    sid = annotation['system_id']; folder = SOURCE / 'systems' / sid
    path = OUT / 'mapped' / (sid + '.pkl')
    report = path.with_suffix('.json')
    try:
        download = json.loads((folder / 'DOWNLOAD.json').read_text())
        for item in download['files']:
            if file_identity(folder / item['name'])['sha256'] != item['sha256']:
                raise ValueError('download_file_identity_changed')
        mapped = map_complex(folder, annotation)
        raw = pickle.dumps(mapped, protocol=5)
        tmp = path.with_suffix('.tmp'); tmp.write_bytes(raw); tmp.replace(path)
        record = dict(system_id=sid, status='MAPPED', sha256=hashlib.sha256(raw).hexdigest(),
                      chains=len(mapped['sequences']), regions=len(mapped['regions']),
                      contacts=sum(int((d < 4.5).sum()) for d in mapped['distance_labels']),
                      producer=file_identity(ROOT / 'biomaster/structural_complex.py'))
    except Exception as error:
        record = dict(system_id=sid, status='REJECTED', reason=str(error)[:250],
                      producer=file_identity(ROOT / 'biomaster/structural_complex.py'))
    write_json(report, record)
    return record


def mapping(workers, watch):
    (OUT / 'mapped').mkdir(parents=True, exist_ok=True)
    expected = pd.read_parquet(SOURCE / 'SELECTED.parquet')
    records = {}
    producer = file_identity(ROOT / 'biomaster/structural_complex.py')
    for path in (OUT / 'mapped').glob('*.json'):
        record = json.loads(path.read_text())
        if record['producer'] != producer:
            raise ValueError('mapping code changed; prepare a separately versioned mapping cache')
        records[record['system_id']] = record
    started = time.monotonic()
    while True:
        jobs = [r for r in expected.to_dict('records') if r['system_id'] not in records
                and (SOURCE / 'systems' / r['system_id'] / 'DOWNLOAD.json').is_file()]
        if jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                futures = [pool.submit(map_one, r) for r in jobs]
                for i, f in enumerate(as_completed(futures), 1):
                    r = f.result(); records[r['system_id']] = r
                    if i % 50 == 0 or i == len(futures):
                        event = dict(stage='mapping', processed=len(records), expected=len(expected),
                                     counts=dict(Counter(r['status'] for r in records.values())), seconds=time.monotonic()-started)
                        write_json(OUT / 'MAPPING_STATUS.json', event); print(json.dumps(event), flush=True)
        if len(records) == len(expected):
            break
        download = json.loads((SOURCE / 'STATUS.json').read_text())
        if not watch or download['status'] != 'DOWNLOADING':
            break
        time.sleep(10)
    report = dict(status='COMPLETE' if len(records) == len(expected) else 'WAITING_FOR_DOWNLOADS',
                  records=list(records.values()), counts=dict(Counter(r['status'] for r in records.values())), expected=len(expected),
                  selection=file_identity(SOURCE / 'SELECTED.parquet'), producer=producer)
    write_json(OUT / 'MAPPING_MANIFEST.json', report)


def overlap_audit():
    manifest = json.loads((OUT / 'MAPPING_MANIFEST.json').read_text())
    if manifest['status'] != 'COMPLETE':
        raise ValueError('all download/mapping outcomes must be known before dataset admission')
    records = [r for r in manifest['records'] if r['status'] == 'MAPPED']
    sequences, molecules, mapped = {}, {}, {}
    for record in records:
        path = OUT / 'mapped' / (record['system_id'] + '.pkl')
        if file_identity(path)['sha256'] != record['sha256']:
            raise ValueError('mapped complex changed')
        item = pickle.loads(path.read_bytes()); mapped[record['system_id']] = item
        for seq in item['sequences'].values():
            sequences[hashlib.sha256(seq.encode()).hexdigest()] = seq
        molecules[item['ligand']['inchikey']] = item['ligand']['smiles']
    fasta = OUT / 'CHAINS.fasta'
    fasta.write_text(''.join(f'>{k}\n{v}\n' for k, v in sorted(sequences.items())))
    source = ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
    targets = pd.read_csv(source / 'TARGET_INDEX.csv.gz')
    target_fasta = OUT / 'PROJECT_TARGETS.fasta'
    target_fasta.write_text(''.join(f'>{r.sequence_sha256}\n{r.sequence}\n' for r in targets.itertuples()))
    matches = OUT / 'PROJECT_SEQUENCE_HITS.tsv'
    command = ['mmseqs', 'easy-search', str(fasta), str(target_fasta), str(matches), str(OUT / 'mmseqs_tmp'),
               '--min-seq-id', '0.3', '-c', '0.5', '--cov-mode', '0', '-s', '7.5', '--max-seqs', '1000',
               '--threads', '16', '--format-output', 'query,target,fident,alnlen,qcov,tcov,evalue']
    with (OUT / 'MMSEQS.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    hits = pd.read_csv(matches, sep='\t', names=['query', 'target', 'identity', 'length', 'qcov', 'tcov', 'evalue']) if matches.stat().st_size else pd.DataFrame(columns=['query'])
    homologs = set(hits['query'])
    RDLogger.DisableLog('rdApp.*')
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    old = pd.read_csv(source / 'OLD_DRUG_INDEX.csv')
    old_fp = [fingerprint.GetFingerprint(Chem.MolFromSmiles(s)) for s in old.model_ligand_smiles]
    similarity = {}
    for key, smiles in molecules.items():
        fp = fingerprint.GetFingerprint(Chem.MolFromSmiles(smiles))
        similarity[key] = max(DataStructs.BulkTanimotoSimilarity(fp, old_fp))
    rows, rejected = [], Counter()
    for sid, item in mapped.items():
        # SDF identity alone is insufficient: check declared chiral centers
        # against the experimental coordinates, not just against SDF tags.
        declared = Chem.MolFromMolFile(item['ligand']['sdf_path'], removeHs=True)
        geometric = Chem.Mol(declared)
        Chem.AssignStereochemistryFrom3D(geometric, replaceExistingTags=True)
        Chem.AssignStereochemistry(declared, cleanIt=True, force=True)
        Chem.AssignStereochemistry(geometric, cleanIt=True, force=True)
        mismatch = any(a.HasProp('_CIPCode') and (not b.HasProp('_CIPCode') or a.GetProp('_CIPCode') != b.GetProp('_CIPCode'))
                       for a, b in zip(declared.GetAtoms(), geometric.GetAtoms()))
        if mismatch:
            rejected['experimental_coordinate_chirality_mismatch'] += 1; continue
        seqs = {hashlib.sha256(s.encode()).hexdigest() for s in item['sequences'].values()}
        if seqs & homologs:
            rejected['project_protein_sequence_similarity'] += 1; continue
        sim = similarity[item['ligand']['inchikey']]
        if sim >= 0.8:
            rejected['project_morgan_tanimoto_at_least_0_8'] += 1; continue
        a = item['annotation']
        rows.append(dict(system_id=sid, split=a['internal_split'], cluster=a['cluster_for_val_split'],
                         release_year=int(pd.Timestamp(a['entry_release_date']).year),
                         max_project_drug_tanimoto=sim, regions=len(item['regions']),
                         atoms=len(item['ligand']['atoms']), max_residues=max(len(r['ca']) for r in item['regions']),
                         mapped_sha256=file_identity(OUT / 'mapped' / (sid + '.pkl'))['sha256']))
    admitted = pd.DataFrame(rows)
    if set(admitted[admitted.split.eq('train')].cluster) & set(admitted[admitted.split.eq('validation')].cluster):
        raise ValueError('cluster split overlap')
    admitted.to_parquet(OUT / 'ADMITTED.parquet', index=False)
    used_seq = {h for sid in admitted.system_id for region in mapped[sid]['regions'] for h in region['sequence_hashes']}
    write_json(OUT / 'SEQUENCES.json', {h: sequences[h] for h in sorted(used_seq)})
    write_json(OUT / 'OVERLAP_AUDIT.json', dict(status='PASS_DEFINED_CRITERIA', admitted=len(admitted),
               splits=admitted.split.value_counts().to_dict(), exclusions=dict(rejected),
               project_sequence_policy='exclude any chain with MMseqs identity>=30%, alignment covers >=50% of BOTH sequences',
               project_ligand_policy='exclude connectivity match and Morgan radius2 2048-bit Tanimoto>=0.8',
               not_certified='remote homologs/local pockets below thresholds and public pretrained overlap',
               mmseqs_command=command, mmseqs_hits=file_identity(matches), official_train_split_only=True,
               internal_clusters_disjoint=True, max_release_year=int(admitted.release_year.max()),
               experimental_complexes_only=True, mapping=file_identity(OUT / 'MAPPING_MANIFEST.json'),
               index=file_identity(OUT / 'ADMITTED.parquet')))
    print(json.dumps(dict(stage='audit', admitted=len(admitted), splits=admitted.split.value_counts().to_dict(), rejected=dict(rejected))), flush=True)


def encode_sequences():
    import torch
    import esm
    from build_biomaster_odti_target_token_features_v1 import window_bounds
    directory = OUT / 'esm2'; directory.mkdir(exist_ok=True)
    sequences = json.loads((OUT / 'SEQUENCES.json').read_text())
    missing = [(h, s) for h, s in sequences.items() if not (directory / (h + '.npy')).is_file()]
    if not missing:
        return
    os.environ['TORCH_HOME'] = '/root/autodl-tmp/.cache/torch'
    torch.set_num_threads(4)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D(); model = model.cuda().eval()
    convert = alphabet.get_batch_converter(); started = time.monotonic()
    with torch.inference_mode():
        for j, (digest, seq) in enumerate(missing, 1):
            total = np.zeros((len(seq), 1280), np.float32); counts = np.zeros((len(seq), 1), np.float32)
            for lo, hi in window_bounds(len(seq), 1022, 128):
                _, _, token = convert([(digest, seq[lo:hi])])
                with torch.autocast('cuda', dtype=torch.float16):
                    h = model(token.cuda(), repr_layers=[33], return_contacts=False)['representations'][33][0, 1:hi-lo+1].float().cpu().numpy()
                total[lo:hi] += h; counts[lo:hi] += 1
            if not (counts > 0).all() or not np.isfinite(total).all():
                raise ValueError('incomplete/nonfinite protein encoding')
            path = directory / (digest + '.npy'); tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as handle:
                np.save(handle, (total/counts).astype(np.float16))
            tmp.replace(path)
            if j % 20 == 0 or j == len(missing):
                event = dict(stage='ESM2', completed=len(sequences)-len(missing)+j, total=len(sequences), seconds=time.monotonic()-started)
                write_json(OUT / 'ENCODING_STATUS.json', event); print(json.dumps(event), flush=True)
    del model; torch.cuda.empty_cache()


def encode_complexes():
    import torch
    from prepare_biomaster_unified_interaction import load_pretrained, pretrained_batch, pretrained_tokens
    from prepare_biomaster_pocket_precision import encode_pocket, geometry_features
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    directory = OUT / 'encoded'; directory.mkdir(exist_ok=True)
    index = pd.read_parquet(OUT / 'ADMITTED.parquet')
    task, model = load_pretrained(); started = time.monotonic(); records = []
    with torch.inference_mode():
        for i, row in enumerate(index.itertuples(), 1):
            path = directory / (row.system_id + '.pt')
            source = OUT / 'mapped' / (row.system_id + '.pkl')
            if not path.exists():
                item = pickle.loads(source.read_bytes()); ligand = item['ligand']
                h = pretrained_tokens(model, pretrained_batch([ligand], task.dictionary))[0]
                aligned = torch.nn.functional.normalize(model.mol_project(h[None, 0]), dim=-1)[0].cpu().numpy()
                n = len(ligand['atoms'])
                atom_tokens = h[1:n+1].cpu().numpy().astype(np.float16)
                atom_distance = np.linalg.norm(ligand['coordinates'][:, None] - ligand['coordinates'][None], axis=-1).astype(np.float32)
                edges = np.concatenate([np.eye(6, dtype=np.float32)[ligand['bond']], np.eye(7, dtype=np.float32)[ligand['stereo']]], -1)
                pockets = []
                for region, truth in zip(item['regions'], item['distance_labels']):
                    tokens, pocket_aligned = encode_pocket(model, task, region)
                    rid = region['atom_residue']; nr = len(region['ca'])
                    pooled = np.stack([tokens[rid == j].mean(0) for j in range(nr)]).astype(np.float16)
                    residues = np.stack([np.load(OUT / 'esm2' / (digest + '.npy'), mmap_mode='r')[position]
                                         for digest, position in zip(region['sequence_hashes'], region['residue_indices'])])
                    distance, geometry, quality = geometry_features(region)
                    pockets.append(dict(residue_tokens=residues, pocket_residue_tokens=pooled,
                                        residue_distance=distance, residue_geometry=geometry, pocket_aligned=pocket_aligned,
                                        pocket_metadata=np.array([1., quality.mean()], np.float32), distance_labels=truth))
                payload = dict(system_id=row.system_id, atom_tokens=atom_tokens, atom_chemistry=ligand['atom_chemistry'],
                               atom_distance=atom_distance, atom_edges=edges, drug_aligned=aligned, pockets=pockets,
                               source_sha256=row.mapped_sha256, input_ligand_geometry='INDEPENDENT_ETKDG',
                               label_geometry='EXPERIMENTAL_BOUND_POSE', native_pocket_index=0)
                tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)
            records.append(dict(system_id=row.system_id, split=row.split, cluster=row.cluster,
                                release_year=row.release_year, file=str(path.relative_to(OUT)), identity=file_identity(path)))
            if i % 50 == 0 or i == len(index):
                event = dict(stage='PAIRED_ENCODING', completed=i, total=len(index), seconds=time.monotonic()-started)
                write_json(OUT / 'ENCODING_STATUS.json', event); print(json.dumps(event), flush=True)
    audit = json.loads((OUT / 'OVERLAP_AUDIT.json').read_text())
    write_json(OUT / 'MANIFEST.json', dict(status='STRUCTURAL_DATA_READY', records=records,
               counts=index.split.value_counts().to_dict(), max_release_year=int(index.release_year.max()),
               project_overlap_audit_pass=True, project_overlap_criteria=audit,
               experimental_complexes_only=True, official_train_split_only=True,
               input_ligand_geometry='INDEPENDENT_ETKDG', label_geometry='EXPERIMENTAL_BOUND_POSE',
               unknown_pairs_used_as_contact_negatives=False,
               sources=[file_identity(ROOT / p) for p in ['biomaster/structural_complex.py', 'scripts/prepare_biomaster_structural_training.py']],
               audit=file_identity(OUT / 'OVERLAP_AUDIT.json')))


def finalize_online_features():
    """Admit verified online encodings after all mapping and overlap outcomes."""
    import torch
    from prepare_biomaster_unified_interaction import CHECKPOINT
    index = pd.read_parquet(OUT / 'ADMITTED.parquet')
    audit = json.loads((OUT / 'OVERLAP_AUDIT.json').read_text())
    records = []
    for row in index.itertuples():
        path = OUT / 'encoded' / (row.system_id + '.pt')
        data = torch.load(path, map_location='cpu', weights_only=False)
        if data['source_sha256'] != row.mapped_sha256 or data['input_ligand_geometry'] != 'INDEPENDENT_ETKDG':
            raise ValueError('online encoded identity/input geometry mismatch')
        raw = pickle.loads((OUT / 'mapped' / (row.system_id + '.pkl')).read_bytes())
        free = raw['ligand']['coordinates']
        expected_distance = np.linalg.norm(free[:, None] - free[None], axis=-1)
        if not np.allclose(data['atom_distance'], expected_distance, atol=1e-5):
            raise ValueError('input is not the independent ligand conformer')
        if len(data['pockets']) != len(raw['distance_labels']):
            raise ValueError('region count mismatch')
        for pocket, truth in zip(data['pockets'], raw['distance_labels']):
            if not np.array_equal(pocket['distance_labels'], truth):
                raise ValueError('observed distance labels changed')
            if pocket['residue_tokens'].shape != (truth.shape[1], 1280):
                raise ValueError('residue encoding/label shape mismatch')
            for key in ['residue_tokens', 'pocket_residue_tokens', 'residue_geometry', 'distance_labels']:
                if not np.isfinite(pocket[key]).all():
                    raise ValueError('nonfinite structural feature')
        records.append(dict(system_id=row.system_id, split=row.split, cluster=row.cluster,
                            release_year=row.release_year, file=str(path.relative_to(OUT)), identity=file_identity(path)))
    write_json(OUT / 'MANIFEST.json', dict(status='STRUCTURAL_DATA_READY', records=records,
               counts=index.split.value_counts().to_dict(), max_release_year=int(index.release_year.max()),
               project_overlap_audit_pass=True, project_overlap_criteria=audit,
               experimental_complexes_only=True, official_train_split_only=True,
               input_ligand_geometry='INDEPENDENT_ETKDG', label_geometry='EXPERIMENTAL_BOUND_POSE',
               unknown_pairs_used_as_contact_negatives=False, all_encoded_records_verified=True,
               public_pretraining_overlap='not certified', drugclip_checkpoint=file_identity(CHECKPOINT),
               sources=[file_identity(ROOT / p) for p in ['biomaster/structural_complex.py', 'biomaster/structural_encoding.py',
                                                        'scripts/prepare_biomaster_structural_training.py']],
               audit=file_identity(OUT / 'OVERLAP_AUDIT.json')))
    print(json.dumps(dict(stage='finalized', counts=index.split.value_counts().to_dict())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['map', 'audit', 'encode', 'finalize-online', 'all'])
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--watch-downloads', action='store_true')
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ['map', 'all']:
        mapping(args.workers, args.watch_downloads)
    if args.stage in ['audit', 'all']:
        overlap_audit()
    if args.stage in ['encode', 'all']:
        encode_sequences(); encode_complexes()
    if args.stage == 'finalize-online':
        finalize_online_features()


if __name__ == '__main__':
    main()

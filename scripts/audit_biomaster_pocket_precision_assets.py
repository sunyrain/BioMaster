#!/usr/bin/env python3
"""Verify exact pocket assets and paired-vector agreement with native DrugCLIP."""
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.pocket_precision_features import PocketPrecisionBank
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import load_pretrained, pretrained_batch, heavy_record, write_json
from prepare_biomaster_pocket_precision import BASE, SUPPLEMENT, SOURCE, OUTPUT, encode_pocket


def main():
    bank = PocketPrecisionBank(OUTPUT, BASE, SUPPLEMENT)
    checked, max_residues, max_atoms = 0, 0, 0
    for record in bank.manifest['records']:
        path = OUTPUT / record['file']
        if file_identity(path)['sha256'] != record['file_sha256']:
            raise ValueError('pocket content hash mismatch')
        x = bank.pocket(record['file'])
        nr, na = record['residues'], record['heavy_atoms']
        if x['residue_tokens'].shape != (nr, 1280) or x['atom_tokens'].shape != (na, 512):
            raise ValueError('token alignment shape mismatch')
        if len(np.unique(x['residue_indices'])) != nr or set(x['atom_residue']) != set(range(nr)):
            raise ValueError('residue mapping mismatch')
        if not np.allclose(x['residue_distance'], x['residue_distance'].T, atol=1e-6):
            raise ValueError('internal geometry is not symmetric')
        if not np.isfinite(x['atom_tokens']).all() or not np.isfinite(x['residue_tokens']).all():
            raise ValueError('nonfinite pretrained states')
        max_residues = max(max_residues, nr); max_atoms = max(max_atoms, na); checked += 1
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    task, model = load_pretrained()
    candidates = sorted(i for i in bank.molecule_source if bank.available[i])[:3]
    selected = sorted(bank.manifest['records'], key=lambda r: r['heavy_atoms'])
    selected = [selected[0], selected[len(selected)//2], selected[-1]]
    differences, rigid_errors = [], []
    rng = np.random.default_rng(51)
    with torch.inference_mode():
        for i, record in zip(candidates, selected):
            _, _, env = bank.sources[bank.molecule_source[i]]
            with env.begin() as txn:
                mol = heavy_record(pickle.loads(txn.get(str(i).encode())))
            pocket = bank.pocket(record['file'])
            mb = pretrained_batch([mol], task.dictionary)
            pb = pretrained_batch([pocket], task.pocket_dictionary)
            native, scale = model(mol_src_tokens=mb[0], mol_src_distance=mb[1], mol_src_edge_type=mb[2],
                                  pocket_src_tokens=pb[0], pocket_src_distance=pb[1], pocket_src_edge_type=pb[2],
                                  smi_list=[str(i)], pocket_list=[record['file']])
            cosine = float((native / scale)[0, 0])
            cached = float(np.dot(bank.aligned[i], pocket['pocket_aligned']))
            if abs(cosine - cached) > 2e-4:
                raise ValueError(f'native/cached paired space disagreement: {cosine} {cached}')
            differences.append(abs(cosine - cached))
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            moved = {**pocket, 'coordinates': (pocket['coordinates'] @ q + np.array([7, -5, 13])).astype(np.float32)}
            _, encoded = encode_pocket(model, task, moved)
            error = float(np.max(np.abs(encoded - pocket['pocket_aligned'])))
            if error > 2e-4:
                raise ValueError('pocket encoder rigid-motion check failed')
            rigid_errors.append(error)
    result = dict(status='PASS', files_checked=checked, max_residues_per_pocket=max_residues,
                  max_heavy_atoms_per_pocket=max_atoms, native_drugclip_pairs=3,
                  max_native_cached_cosine_error=max(differences), max_rigid_motion_embedding_error=max(rigid_errors),
                  feature_manifest=file_identity(OUTPUT / 'MANIFEST.json'), script=file_identity(Path(__file__)))
    write_json(OUTPUT.parent / 'ASSET_AUDIT.json', result)
    print(json.dumps(result), flush=True); bank.close()


if __name__ == '__main__':
    main()

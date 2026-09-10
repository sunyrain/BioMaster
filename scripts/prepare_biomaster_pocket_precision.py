#!/usr/bin/env python3
"""Prepare separate, untruncated heavy-atom pockets and aligned DrugCLIP states."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.odti_pockets_v3 import build_pocket_store, DEFAULT_POCKET_MASTER, THREE_TO_ONE, file_identity
from prepare_biomaster_unified_interaction import load_pretrained, pretrained_batch, write_json, CHECKPOINT

BASE = ROOT / 'outputs/biomaster_unified_interaction_20260906/features'
SUPPLEMENT = ROOT / 'outputs/biomaster_best_model_20260906/data/supplemental_features'
SOURCE = ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
OUTPUT = ROOT / 'outputs/biomaster_pocket_precision_20260906/features'


def residue_sources():
    assets = pd.read_csv(ROOT / 'outputs/biomaster_odti_v4_plan_20260905/CURRENT_TARGET_ASSETS_V4.csv.gz')
    array = np.load(ROOT / 'outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy', mmap_mode='r')
    result = {}
    for row in assets.itertuples():
        if row.full_residue_available and row.token_length == len(row.protein_sequence):
            result[row.sequence_sha256] = (array, int(row.token_offset), int(row.token_length))
    added = ROOT / 'outputs/biomaster_odti_v4_20260905/features'
    array = np.load(added / 'ESM2_ADDED_RESIDUES_FLOAT16_V4.npy', mmap_mode='r')
    for row in pd.read_csv(added / 'ESM2_ADDED_RESIDUE_INDEX_V4.csv').itertuples():
        result[row.sequence_sha256] = (array, int(row.token_offset), int(row.token_length))
    for path in (BASE / 'missing_residues').glob('*.npy'):
        array = np.load(path, mmap_mode='r')
        result[path.stem] = (array, 0, len(array))
    return result


def read_heavy_atoms(path, sequence):
    """Exact canonical mapping, first model, consistent per-residue altloc.

    Supported assets are exact-sequence AlphaFold PDBs, chain A. Experimental
    structures require a separately audited chain/sequence map.
    """
    candidates = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('ENDMDL'):
                break
            if not line.startswith('ATOM') or line[21] != 'A' or line[26] != ' ':
                continue
            if line[16] not in [' ', 'A']:
                continue
            j = int(line[22:26]) - 1
            aa = THREE_TO_ONE.get(line[17:20].strip(), 'X')
            if not 0 <= j < len(sequence) or sequence[j] != aa:
                raise ValueError(f'canonical residue mismatch: {path}:{j}')
            name = line[12:16].strip()
            element = line[76:78].strip() or name.lstrip('0123456789')[0]
            if element.upper() in ['H', 'D']:
                continue
            xyz = np.array([float(line[a:a+8]) for a in [30, 38, 46]], np.float32)
            occupancy = float(line[54:60])
            if not np.isfinite(xyz).all() or occupancy <= 0:
                continue
            key = (j, name)
            if key in candidates:
                raise ValueError(f'ambiguous atom mapping: {path}:{key}')
            candidates[key] = (element, xyz, np.clip(float(line[60:66]) / 100, 0, 1))
    return candidates


def pocket_arrays(atoms, indices):
    symbols, xyz, residue, names, quality = [], [], [], [], []
    ca, frame, frame_mask = [], [], []
    for i, j in enumerate(indices):
        selected = sorted((key, value) for key, value in atoms.items() if key[0] == j)
        if (j, 'CA') not in atoms:
            raise ValueError(f'missing CA: {j}')
        for (_, name), (element, position, q) in selected:
            symbols.append(element); xyz.append(position); residue.append(i); names.append(name); quality.append(q)
        origin = atoms[j, 'CA'][1]
        ca.append(origin)
        valid = (j, 'N') in atoms and (j, 'C') in atoms
        if valid:
            x = atoms[j, 'C'][1] - origin
            u = atoms[j, 'N'][1] - origin
            x = x / max(np.linalg.norm(x), 1e-8)
            y = u - np.dot(u, x) * x
            norm = np.linalg.norm(y)
            valid = norm > 1e-6
            y = y / max(norm, 1e-8)
            basis = np.stack([x, y, np.cross(x, y)], axis=1)
        frame.append(basis if valid else np.zeros((3, 3), np.float32)); frame_mask.append(valid)
    return dict(atoms=symbols, coordinates=np.array(xyz, np.float32), atom_residue=np.array(residue, np.int32),
                atom_names=np.array(names), atom_quality=np.array(quality, np.float32),
                ca=np.array(ca, np.float32), frames=np.array(frame, np.float32),
                frame_mask=np.array(frame_mask, bool), residue_indices=np.array(indices, np.int32))


def geometry_features(record):
    xyz, ca, frames = record['coordinates'], record['ca'], record['frames']
    vector = ca[None, :, :] - ca[:, None, :]
    distance = np.linalg.norm(vector, axis=-1)
    direction = vector / np.maximum(distance[..., None], 1e-8)
    local_i = np.einsum('ijc,icd->ijd', direction, frames)
    local_j = np.einsum('ijc,jcd->ijd', -direction, frames)
    rotation = np.einsum('ica,jcb->ijab', frames, frames).reshape(len(ca), len(ca), 9)
    frame_valid = record['frame_mask'][:, None] & record['frame_mask'][None, :]
    orientation = np.concatenate([local_i, local_j, rotation], -1) * frame_valid[..., None]
    # Minimum heavy-atom distance, preserving side chains rather than only CA.
    minimum = np.empty_like(distance)
    rid = record['atom_residue']
    for i in range(len(ca)):
        delta = xyz[rid == i, None, :] - xyz[None, :, :]
        d = np.linalg.norm(delta, axis=-1).min(0)
        for j in range(len(ca)):
            minimum[i, j] = d[rid == j].min()
    quality = np.array([record['atom_quality'][rid == i].mean() for i in range(len(ca))], np.float32)
    extra = np.concatenate([minimum[..., None] / 32, orientation,
                            (quality[:, None] * quality[None, :])[..., None]], -1)
    assert extra.shape[-1] == 17
    return distance.astype(np.float32), extra.astype(np.float32), quality


def encode_pocket(model, task, record):
    tokens, distance, edges = pretrained_batch([record], task.pocket_dictionary)
    encoder = model.pocket_model
    bias = encoder.gbf_proj(encoder.gbf(distance, edges)).permute(0, 3, 1, 2).contiguous()
    output = encoder.encoder(encoder.embed_tokens(tokens), padding_mask=tokens.eq(encoder.padding_idx),
                             attn_mask=bias.reshape(-1, tokens.shape[1], tokens.shape[1]))[0][0]
    aligned = torch.nn.functional.normalize(model.pocket_project(output[None, 0]), dim=-1)[0]
    return output[1:len(record['atoms'])+1].float().cpu().numpy(), aligned.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--max-encoder-atoms', type=int, default=2048,
                        help='Fail with an explicit record if exceeded; never truncate atoms')
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True); (out / 'pockets').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest_path = out / 'MANIFEST.json'
    if manifest_path.exists():
        raise RuntimeError('completed feature store exists; use a new output path')
    started = time.monotonic()
    pockets = build_pocket_store(DEFAULT_POCKET_MASTER, [SOURCE / 'TARGET_INDEX.csv.gz'], max_pockets=100000)
    write_json(out / 'POCKET_SELECTION.json', pockets)
    sources = residue_sources()
    targets = pd.read_csv(SOURCE / 'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    task, model = load_pretrained()
    checkpoint_identity = file_identity(CHECKPOINT)
    records, rejected = [], []
    with torch.inference_mode():
        for row in targets.itertuples():
            item = pockets['targets'].get(row.sequence_sha256)
            if item is None:
                continue
            full, start, length = sources[row.sequence_sha256]
            assert length == len(row.sequence)
            atoms = read_heavy_atoms(item['sources']['receptor']['path'], row.sequence)
            for pocket in item['pockets']:
                name = f't{row.target_feature_index}_{pocket["pocket_id"]}'
                path = out / 'pockets' / (name + '.npz')
                r = pocket_arrays(atoms, pocket['residue_indices'])
                if len(r['atoms']) > args.max_encoder_atoms:
                    rejected.append(dict(name=name, reason='encoder_atom_budget_exceeded_not_truncated', atoms=len(r['atoms'])))
                    continue
                # Always rebuild incomplete stores: stale partial files cannot
                # silently mix encoders or extraction versions.
                h, aligned = encode_pocket(model, task, r)
                pooled = np.stack([h[r['atom_residue'] == i].mean(0) for i in range(len(r['ca']))])
                distance, geometry, quality = geometry_features(r)
                esm = np.asarray(full[start + r['residue_indices']], np.float16)
                np.savez_compressed(path, **r, atom_tokens=h.astype(np.float16), residue_tokens=esm,
                                    pocket_residue_tokens=pooled.astype(np.float16), pocket_aligned=aligned,
                                    residue_distance=distance, residue_geometry=geometry, quality=quality)
                records.append(dict(target_feature_index=int(row.target_feature_index), sequence_sha256=row.sequence_sha256,
                                    pocket_id=pocket['pocket_id'], probability=pocket['probability'],
                                    residues=len(r['ca']), heavy_atoms=len(r['atoms']), file=str(path.relative_to(out)),
                                    file_sha256=file_identity(path)['sha256'], source=item['sources']))
            if len(records) % 25 < len(item['pockets']):
                write_json(out / 'STATUS.json', dict(status='ENCODING_POCKETS', pockets=len(records), seconds=time.monotonic()-started))
                print(json.dumps(dict(pockets=len(records), target=int(row.target_feature_index), seconds=round(time.monotonic()-started, 1))), flush=True)
        # Restore the original aligned 128D molecule space using the SAME fold's
        # projection head and cached raw CLS features; do not randomly reproject.
        global_states = np.load(BASE / 'MOLECULE_GLOBAL.npy', mmap_mode='r')
        aligned = np.zeros((len(global_states), 128), np.float32)
        available = np.load(BASE / 'PRETRAINED_AVAILABLE.npy').copy()
        for root in [BASE, SUPPLEMENT]:
            ids = np.load(root / 'REQUIRED_MOLECULE_IDS.npy')
            values = np.load(root / 'MOLECULE_GLOBAL.npy', mmap_mode='r')
            ok = np.load(root / 'PRETRAINED_AVAILABLE.npy')
            available[ids] = ok[ids]
            for begin in range(0, len(ids), 4096):
                take = ids[begin:begin+4096]
                x = torch.from_numpy(values[take].copy()).cuda()
                y = torch.nn.functional.normalize(model.mol_project(x), dim=-1).cpu().numpy()
                aligned[take] = y * ok[take, None]
        np.save(out / 'MOLECULE_ALIGNED.npy', aligned)
        np.save(out / 'MOLECULE_AVAILABLE.npy', available)
    manifest = dict(format='POCKET_PRECISION_FEATURES_V1', status='COMPLETE' if not rejected else 'INCOMPLETE_REQUIRES_LARGE_POCKET_HANDLING',
                    labels_used=False, source_kind='PREDICTED_P2RANK_ON_EXACT_ALPHAFOLD',
                    targets=len(targets), targets_with_pockets=len({r['target_feature_index'] for r in records}),
                    pockets=len(records), heavy_atoms=sum(r['heavy_atoms'] for r in records),
                    residues=sum(r['residues'] for r in records), no_residue_or_atom_truncation=True,
                    no_full_sequence_padding=True, all_valid_diverse_pockets=True,
                    experimental_contacts_available=False, rejected=rejected, records=records,
                    checkpoint=checkpoint_identity, producer=file_identity(Path(__file__)),
                    target_index=file_identity(SOURCE / 'TARGET_INDEX.csv.gz'),
                    molecule_projection='original mol_project, normalize; same fold as original pocket_project',
                    encoder_precision='FP32, TF32 disabled', public_pretraining_temporal_overlap='not_certified',
                    seconds=time.monotonic()-started)
    write_json(manifest_path, manifest)
    write_json(out / 'STATUS.json', {k: v for k, v in manifest.items() if k not in ['records']})
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['records']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

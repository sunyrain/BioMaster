#!/usr/bin/env python3
"""Complete label-free R1 features; reuse exact-SMILES caches and full residues.

No molecule or sequence is truncated. Completed chunks can be resumed; the
COMPLETE manifest is written only after all rows pass finite/nonzero checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import dill
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'outputs/biomaster_odti_v4_plan_20260905'
OUT = ROOT / 'outputs/biomaster_odti_v4_20260905/features'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def bermol(out, limit=0):
    table = pd.read_csv(PLAN / 'CURRENT_MOLECULE_ASSETS_V4.csv.gz').sort_values('drug_feature_index')
    assert np.array_equal(table.drug_feature_index, np.arange(len(table)))
    path, donepath = out / 'BERMOL768_FLOAT32_V4.npy', out / 'BERMOL_DONE_V4.npy'
    if path.exists():
        bank = np.load(path, mmap_mode='r+')
        done = np.load(donepath, mmap_mode='r+')
        assert bank.shape == (len(table), 768) and done.shape == (len(table),)
    else:
        bank = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(table), 768))
        done = np.lib.format.open_memmap(donepath, mode='w+', dtype=bool, shape=(len(table),))
        bank[:] = 0
        done[:] = False
        for name, part in table.loc[table.bermol_available].groupby('bermol_array'):
            old = np.load(ROOT / name, mmap_mode='r')
            ids = part.drug_feature_index.to_numpy()
            values = old[part.bermol_row.to_numpy(dtype=np.int64)]
            if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                raise ValueError(f'invalid old BerMol features: {name}')
            bank[ids] = values
            bank.flush()
            done[ids] = True
        done.flush()
    errorpath = out / 'BERMOL_UNAVAILABLE_V4.json'
    errors = json.loads(errorpath.read_text()) if errorpath.exists() else {}
    missing = np.flatnonzero(~done)
    if limit:
        missing = missing[:limit]
    if len(missing):
        sys.path.insert(0, str(ROOT / 'third_party/sota_dti_2026/DTIAM/code/BerMol'))
        from bermol.tokenizer import BerMolTokenizer
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.warning')
        with (ROOT / 'third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl').open('rb') as f:
            predictor = dill.load(f)
        predictor.model.cuda().eval()
        tokenizer = BerMolTokenizer(predictor.vocab)
        started = time.monotonic()
        with torch.inference_mode():
            for chunkstart in range(0, len(missing), 4096):
                ids = missing[chunkstart:chunkstart + 4096]
                tokens = []
                for i in ids:
                    try:
                        tokens.append((int(i), tokenizer.encode(table.model_ligand_smiles.iloc[i]).squeeze(0)))
                    except Exception as error:
                        errors[str(i)] = {'smiles': table.model_ligand_smiles.iloc[i], 'error': str(error)}
                        write_json(errorpath, errors)
                tokens.sort(key=lambda item: len(item[1]))
                cursor = 0
                while cursor < len(tokens):
                    maximum = len(tokens[min(len(tokens) - 1, cursor + 63)][1])
                    count = min(64, max(1, 1_000_000 // maximum**2))
                    selected = tokens[cursor:cursor + count]
                    length = max(len(t) for _, t in selected)
                    batch = torch.zeros((len(selected), length), dtype=torch.long, device='cuda')
                    attention = torch.full((len(selected), length, length), -10000., device='cuda')
                    for j, (_, token) in enumerate(selected):
                        batch[j, :len(token)] = token.cuda()
                        attention[j, :, :len(token)] = 0
                    _, pooled = predictor.model.encoder(batch, attention)
                    values = pooled.float().cpu().numpy()
                    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                        raise FloatingPointError('nonfinite or empty BerMol representation')
                    bank[[i for i, _ in selected]] = values
                    cursor += len(selected)
                bank.flush()
                done[ids] = True
                done.flush()
                status = {'stage': 'bermol', 'completed': int(done.sum()), 'total': len(done),
                          'new_rows': min(chunkstart + len(ids), len(missing)),
                          'seconds': round(time.monotonic() - started, 1)}
                write_json(out / 'FEATURE_STATUS_V4.json', status)
                print(json.dumps(status), flush=True)
        del predictor
        torch.cuda.empty_cache()
    if not errorpath.exists():
        write_json(errorpath, errors)
    available = np.array(done)
    available[[int(i) for i in errors]] = False
    np.save(out / 'BERMOL_AVAILABLE_V4.npy', available)
    return bool(done.all())


def esm2(out):
    path = out / 'ESM2_FULL_MEAN1280_FLOAT32_V4.npy'
    if path.exists():
        values = np.load(path)
        if values.shape == (843, 1280) and np.isfinite(values).all() and (np.linalg.norm(values, axis=1) > 0).all():
            return
        raise ValueError('incomplete ESM mean file; inspect before rebuilding')
    table = pd.read_csv(PLAN / 'CURRENT_TARGET_ASSETS_V4.csv.gz').sort_values('target_feature_index')
    assert np.array_equal(table.target_feature_index, np.arange(len(table)))
    residues = np.load(ROOT / 'outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy', mmap_mode='r')
    means = np.zeros((len(table), 1280), np.float32)
    for row in table.loc[table.full_residue_available].itertuples():
        assert int(row.token_length) == int(row.sequence_length)
        block = residues[int(row.token_offset):int(row.token_offset + row.token_length)]
        means[row.target_feature_index] = block.mean(axis=0, dtype=np.float32)
    os.environ['TORCH_HOME'] = '/root/autodl-tmp/.cache/torch'
    import esm
    from build_biomaster_odti_target_token_features_v1 import window_bounds
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.cuda().eval()
    converter = alphabet.get_batch_converter()
    appended, index = [], []
    offset = 0
    with torch.inference_mode():
        for row in table.loc[~table.full_residue_available].itertuples():
            seq = row.protein_sequence
            total = np.zeros((len(seq), 1280), np.float32)
            count = np.zeros(len(seq), np.float32)
            for left, right in window_bounds(len(seq), 1022, 128):
                _, _, tokens = converter([(str(row.target_feature_index), seq[left:right])])
                with torch.autocast('cuda', dtype=torch.float16):
                    result = model(tokens.cuda(), repr_layers=[33], return_contacts=False)
                total[left:right] += result['representations'][33][0, 1:right-left+1].float().cpu().numpy()
                count[left:right] += 1
            if (count == 0).any() or not np.isfinite(total).all():
                raise ValueError('ESM residue coverage failure')
            stitched = (total / count[:, None]).astype(np.float16)
            means[row.target_feature_index] = stitched.mean(axis=0, dtype=np.float32)
            appended.append(stitched)
            index.append({'target_feature_index': row.target_feature_index, 'sequence_sha256': row.sequence_sha256,
                          'token_offset': offset, 'token_length': len(seq)})
            offset += len(seq)
            print(json.dumps({'stage': 'esm2', 'new_targets': len(index), 'target_index': row.target_feature_index}), flush=True)
    del model
    torch.cuda.empty_cache()
    if not np.isfinite(means).all() or not (np.linalg.norm(means, axis=1) > 0).all():
        raise ValueError('invalid complete ESM means')
    np.save(out / 'ESM2_ADDED_RESIDUES_FLOAT16_V4.npy', np.concatenate(appended))
    pd.DataFrame(index).to_csv(out / 'ESM2_ADDED_RESIDUE_INDEX_V4.csv', index=False)
    np.save(path, means)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--limit-new', type=int, default=0, help='profiling only; never writes a COMPLETE manifest')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    complete = bermol(args.out, args.limit_new)
    if not complete or args.limit_new:
        return
    esm2(args.out)
    available = np.load(args.out / 'BERMOL_AVAILABLE_V4.npy')
    bank = np.load(args.out / 'BERMOL768_FLOAT32_V4.npy', mmap_mode='r')
    if not np.isfinite(bank[available]).all() or not (np.linalg.norm(bank[available],axis=1)>0).all():
        raise ValueError('invalid available BerMol rows')
    files = list(args.out.glob('*.npy')) + list(args.out.glob('*.csv')) + [args.out / 'BERMOL_UNAVAILABLE_V4.json']
    manifest = {'status': 'COMPLETE', 'molecule_asset_rows': len(available), 'available_molecules': int(available.sum()),
                'unavailable_molecules': int((~available).sum()), 'targets': 843, 'label_dependency': 'NONE',
                'bermol_inference': 'official tokenizer/encoder, FP32, no truncation; exact-SMILES old cache reuse',
                'protein_pooling': 'mean of all residue states; 1022-residue windows with 128 overlap; no BOS/EOS',
                'inputs': {n: sha256(PLAN / n) for n in ['CURRENT_MOLECULE_ASSETS_V4.csv.gz', 'CURRENT_TARGET_ASSETS_V4.csv.gz']},
                'files': {str(f.relative_to(ROOT)): sha256(f) for f in files}}
    write_json(args.out / 'FEATURE_MANIFEST_V4.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()

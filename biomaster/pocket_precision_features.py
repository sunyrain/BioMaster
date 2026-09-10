"""Ragged independent-pocket batches, with immutable atom order and provenance."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import torch


class PocketPrecisionBank:
    def __init__(self, path, base, supplement, device='cuda'):
        self.path, self.device = Path(path), torch.device(device)
        self.manifest = json.loads((self.path / 'MANIFEST.json').read_text())
        if self.manifest['status'] != 'COMPLETE' or self.manifest['labels_used']:
            raise ValueError('complete label-independent pockets required')
        self.pockets = defaultdict(list)
        for r in self.manifest['records']:
            self.pockets[r['target_feature_index']].append(r)
        self.aligned = np.load(self.path / 'MOLECULE_ALIGNED.npy', mmap_mode='r')
        self.available = np.load(self.path / 'MOLECULE_AVAILABLE.npy')
        self.molecule_source = {}
        self.sources = []
        for source in [Path(base), Path(supplement)]:
            manifest = json.loads((source / 'ATOM_MANIFEST.json').read_text())
            if manifest['status'] != 'COMPLETE' or manifest['labels_used']:
                raise ValueError('invalid molecule features')
            index = np.load(source / 'ATOM_INDEX.npz')
            arrays = {key: np.load(source / (name + '.npy'), mmap_mode='r') for key, name in [
                ('tokens', 'ATOM_TOKENS'), ('chemistry', 'ATOM_CHEMISTRY'),
                ('neighbors', 'ATOM_NEIGHBORS'), ('bond', 'ATOM_BOND'), ('stereo', 'ATOM_STEREO')]}
            env = lmdb.open(str(source / 'MOLECULES.lmdb'), subdir=False, readonly=True, lock=False)
            item = (index, arrays, env)
            self.sources.append(item)
            for i in np.load(source / 'REQUIRED_MOLECULE_IDS.npy'):
                if int(i) in self.molecule_source:
                    raise ValueError('molecule source collision')
                self.molecule_source[int(i)] = len(self.sources) - 1

    @lru_cache(maxsize=2048)
    def molecule(self, i):
        if i not in self.molecule_source:
            raise ValueError(f'unprepared molecule: {i}')
        index, arrays, env = self.sources[self.molecule_source[i]]
        lo, hi = index['offsets'][i:i+2]
        with env.begin() as txn:
            raw = txn.get(str(i).encode())
        if raw is None:
            raise ValueError(f'missing conformer record: {i}')
        record = pickle.loads(raw)
        keep = np.array([a != 'H' for a in record['atoms']])
        if not self.available[i] or hi == lo or 'coordinates' not in record:
            return None
        xyz = record['coordinates'][keep]
        if len(xyz) != hi - lo:
            raise ValueError(f'atom-order/count mismatch: {i}')
        n = len(xyz)
        distance = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1).astype(np.float32)
        bond, stereo = np.zeros((n, n), int), np.zeros((n, n), int)
        neighbor = arrays['neighbors'][lo:hi]
        rows, cols = np.nonzero(neighbor >= 0)
        dst = neighbor[rows, cols]
        if (dst >= n).any():
            raise ValueError('bond index outside molecular atom order')
        bond[rows, dst] = arrays['bond'][lo:hi][rows, cols]
        stereo[rows, dst] = arrays['stereo'][lo:hi][rows, cols]
        edge = np.concatenate([np.eye(6, dtype=np.float32)[bond], np.eye(7, dtype=np.float32)[stereo]], -1)
        return dict(atom_tokens=arrays['tokens'][lo:hi], atom_chemistry=arrays['chemistry'][lo:hi],
                    atom_distance=distance, atom_edges=edge, coordinates=xyz)

    @lru_cache(maxsize=2048)
    def pocket(self, relative):
        with np.load(self.path / relative) as data:
            return {k: data[k] for k in data.files}

    def batch(self, drug_ids, target_ids):
        drugs, targets = np.asarray(drug_ids), np.asarray(target_ids)
        if drugs.ndim != 1 or targets.shape != drugs.shape or not len(drugs):
            raise ValueError('aligned nonempty 1D IDs required')
        if drugs.dtype.kind not in 'iu' or targets.dtype.kind not in 'iu':
            raise ValueError('integer IDs required')
        if targets.min() < 0 or targets.max() >= self.manifest['targets']:
            raise ValueError('target outside registry')
        rows = []
        for owner, (d, t) in enumerate(zip(drugs, targets)):
            mol = self.molecule(int(d))
            selected = self.pockets[int(t)] if mol is not None else []
            if selected:
                for record in selected:
                    rows.append((owner, d, mol, self.pocket(record['file']), record))
            else:
                rows.append((owner, d, None, None, None))
        na = max([len(r[2]['atom_tokens']) for r in rows if r[2] is not None] + [1])
        nr = max([len(r[3]['residue_tokens']) for r in rows if r[3] is not None] + [1])
        n = len(rows)
        shapes = dict(atom_tokens=(n, na, 512), atom_chemistry=(n, na, 40), atom_mask=(n, na),
                      atom_distance=(n, na, na), atom_edges=(n, na, na, 13),
                      residue_tokens=(n, nr, 1280), pocket_residue_tokens=(n, nr, 512), residue_mask=(n, nr),
                      residue_distance=(n, nr, nr), residue_geometry=(n, nr, nr, 17),
                      drug_aligned=(n, 128), pocket_aligned=(n, 128), pocket_metadata=(n, 2), owner=(n,))
        out = {k: np.zeros(s, dtype=bool if k.endswith('_mask') else np.int64 if k == 'owner' else np.float32)
               for k, s in shapes.items()}
        for row, (owner, d, mol, pocket, record) in enumerate(rows):
            out['owner'][row] = owner
            if mol is None:
                continue
            a, r = len(mol['atom_tokens']), len(pocket['residue_tokens'])
            out['atom_mask'][row, :a] = True; out['residue_mask'][row, :r] = True
            for key in ['atom_tokens', 'atom_chemistry']:
                out[key][row, :a] = mol[key]
            for key in ['atom_distance', 'atom_edges']:
                out[key][row, :a, :a] = mol[key]
            for key in ['residue_tokens', 'pocket_residue_tokens']:
                out[key][row, :r] = pocket[key]
            for key in ['residue_distance', 'residue_geometry']:
                out[key][row, :r, :r] = pocket[key]
            out['drug_aligned'][row] = self.aligned[d]
            out['pocket_aligned'][row] = pocket['pocket_aligned']
            out['pocket_metadata'][row] = [record['probability'], pocket['quality'].mean()]
        return {k: torch.from_numpy(v).to(self.device) for k, v in out.items()}

    def close(self):
        self.molecule.cache_clear(); self.pocket.cache_clear()
        for _, _, env in self.sources:
            env.close()

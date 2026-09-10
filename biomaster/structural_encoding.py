"""Shared paired encoder for full-chain sequence and experimental pocket inputs."""
import hashlib
import os
from pathlib import Path
import pickle

import numpy as np
import torch


class StructuralEncoder:
    def __init__(self, output):
        import esm
        from scripts.prepare_biomaster_unified_interaction import load_pretrained
        self.output = Path(output)
        for name in ['esm2', 'encoded']:
            (self.output / name).mkdir(exist_ok=True)
        os.environ['TORCH_HOME'] = '/root/autodl-tmp/.cache/torch'
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        self.esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm = self.esm.cuda().eval()
        self.convert = alphabet.get_batch_converter()
        self.task, self.clip = load_pretrained()

    @torch.inference_mode()
    def sequence(self, sequence):
        from scripts.build_biomaster_odti_target_token_features_v1 import window_bounds
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        path = self.output / 'esm2' / (digest + '.npy')
        if not path.exists():
            total = np.zeros((len(sequence), 1280), np.float32)
            counts = np.zeros((len(sequence), 1), np.float32)
            for lo, hi in window_bounds(len(sequence), 1022, 128):
                _, _, token = self.convert([(digest, sequence[lo:hi])])
                with torch.autocast('cuda', dtype=torch.float16):
                    h = self.esm(token.cuda(), repr_layers=[33], return_contacts=False)['representations'][33][0, 1:hi-lo+1].float().cpu().numpy()
                total[lo:hi] += h; counts[lo:hi] += 1
            if not (counts > 0).all() or not np.isfinite(total).all():
                raise ValueError('incomplete/nonfinite full-chain sequence states')
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as handle:
                np.save(handle, (total / counts).astype(np.float16))
            tmp.replace(path)
        return digest, np.load(path, mmap_mode='r')

    @torch.inference_mode()
    def complex(self, record):
        from scripts.prepare_biomaster_unified_interaction import pretrained_batch, pretrained_tokens
        from scripts.prepare_biomaster_pocket_precision import encode_pocket, geometry_features
        sid = record['system_id']; source = self.output / 'mapped' / (sid + '.pkl')
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('mapped record hash mismatch')
        path = self.output / 'encoded' / (sid + '.pt')
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['source_sha256'] != record['sha256']:
                raise ValueError('existing encoded record has different source')
            return
        item = pickle.loads(raw)
        sequence_states = dict(self.sequence(s) for s in item['sequences'].values())
        ligand = item['ligand']; n = len(ligand['atoms'])
        h = pretrained_tokens(self.clip, pretrained_batch([ligand], self.task.dictionary))[0]
        aligned = torch.nn.functional.normalize(self.clip.mol_project(h[None, 0]), dim=-1)[0].cpu().numpy()
        atom_tokens = h[1:n+1].cpu().numpy().astype(np.float16)
        atom_distance = np.linalg.norm(ligand['coordinates'][:, None] - ligand['coordinates'][None], axis=-1).astype(np.float32)
        edges = np.concatenate([np.eye(6, dtype=np.float32)[ligand['bond']], np.eye(7, dtype=np.float32)[ligand['stereo']]], -1)
        pockets = []
        for region, truth in zip(item['regions'], item['distance_labels']):
            tokens, pocket_aligned = encode_pocket(self.clip, self.task, region)
            rid = region['atom_residue']; nr = len(region['ca'])
            pooled = np.stack([tokens[rid == j].mean(0) for j in range(nr)]).astype(np.float16)
            residues = np.stack([sequence_states[digest][position] for digest, position in zip(region['sequence_hashes'], region['residue_indices'])])
            distance, geometry, quality = geometry_features(region)
            pockets.append(dict(residue_tokens=residues, pocket_residue_tokens=pooled,
                                residue_distance=distance, residue_geometry=geometry, pocket_aligned=pocket_aligned,
                                pocket_metadata=np.array([1., quality.mean()], np.float32), distance_labels=truth))
        payload = dict(system_id=sid, atom_tokens=atom_tokens, atom_chemistry=ligand['atom_chemistry'],
                       atom_distance=atom_distance, atom_edges=edges, drug_aligned=aligned, pockets=pockets,
                       source_sha256=record['sha256'], input_ligand_geometry='INDEPENDENT_ETKDG',
                       label_geometry='EXPERIMENTAL_BOUND_POSE', native_pocket_index=0)
        tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)

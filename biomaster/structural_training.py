"""Batches and supervised losses for mapped experimental complexes."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


class StructuralDataset:
    def __init__(self, root, records, device='cuda'):
        self.root, self.records, self.device = Path(root), records, device

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=128)
    def load(self, index):
        item = torch.load(self.root / self.records[index]['file'], map_location='cpu', weights_only=False)
        if item['input_ligand_geometry'] != 'INDEPENDENT_ETKDG' or item['label_geometry'] != 'EXPERIMENTAL_BOUND_POSE':
            raise ValueError('invalid geometry provenance')
        return item

    def batch(self, indices):
        rows, native = [], []
        for owner, index in enumerate(indices):
            item = self.load(int(index)); native.append(len(rows) + item['native_pocket_index'])
            rows.extend((owner, item, p) for p in item['pockets'])
        n = len(rows)
        na = max(len(m['atom_tokens']) for _, m, _ in rows)
        nr = max(len(p['residue_tokens']) for _, _, p in rows)
        shapes = dict(atom_tokens=(n, na, 512), atom_chemistry=(n, na, 40), atom_mask=(n, na),
                      atom_distance=(n, na, na), atom_edges=(n, na, na, 13),
                      residue_tokens=(n, nr, 1280), pocket_residue_tokens=(n, nr, 512), residue_mask=(n, nr),
                      residue_distance=(n, nr, nr), residue_geometry=(n, nr, nr, 17),
                      drug_aligned=(n, 128), pocket_aligned=(n, 128), pocket_metadata=(n, 2), owner=(n,))
        batch = {k: np.zeros(s, dtype=bool if k.endswith('_mask') else np.int64 if k == 'owner' else np.float32)
                 for k, s in shapes.items()}
        truth = np.zeros((n, na, nr), np.float32)
        for row, (owner, m, p) in enumerate(rows):
            a, r = len(m['atom_tokens']), len(p['residue_tokens'])
            batch['owner'][row] = owner
            batch['atom_mask'][row, :a] = True; batch['residue_mask'][row, :r] = True
            for k in ['atom_tokens', 'atom_chemistry']:
                batch[k][row, :a] = m[k]
            for k in ['atom_distance', 'atom_edges']:
                batch[k][row, :a, :a] = m[k]
            for k in ['residue_tokens', 'pocket_residue_tokens']:
                batch[k][row, :r] = p[k]
            for k in ['residue_distance', 'residue_geometry']:
                batch[k][row, :r, :r] = p[k]
            batch['drug_aligned'][row] = m['drug_aligned']
            batch['pocket_aligned'][row] = p['pocket_aligned']
            batch['pocket_metadata'][row] = p['pocket_metadata']
            truth[row, :a, :r] = p['distance_labels']
        result = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
        labels = dict(distance=torch.from_numpy(truth).to(self.device), native=torch.tensor(native, device=self.device))
        return result, labels


def balanced_mean(values, positive):
    terms = []
    if positive.any():
        terms.append(values[positive].mean())
    if (~positive).any():
        terms.append(values[~positive].mean())
    return torch.stack(terms).mean()


def complex_losses(aux, batch, labels, cfg):
    """Equal-complex weighting, near/far distance and contact class balancing."""
    terms = []
    for owner in range(len(labels['native'])):
        selected = batch['owner'] == owner
        valid = aux['pair_mask'][selected]
        truth = labels['distance'][selected][valid]
        if not torch.isfinite(truth).all() or (truth < 0).any() or not len(truth):
            raise ValueError('invalid observed complex labels')
        bins = (truth / cfg.distance_max * (cfg.distance_bins - 1)).long().clamp_max(cfg.distance_bins - 1)
        ce = F.cross_entropy(aux['distance_logits'][selected][valid].float(), bins, reduction='none')
        contact = truth < 4.5
        bce = F.binary_cross_entropy_with_logits(aux['contact_logits'][selected][valid].float(), contact.float(), reduction='none')
        distance_loss = balanced_mean(ce, truth < 16)
        contact_loss = balanced_mean(bce, contact)
        gates = aux['pocket_gate'][selected].float()
        rows = torch.nonzero(selected, as_tuple=True)[0]
        native = torch.nonzero(rows == labels['native'][owner], as_tuple=True)[0]
        if len(native) != 1:
            raise ValueError('native pocket identity mismatch')
        site_loss = F.cross_entropy(gates[None], native) if len(gates) > 1 else gates.sum() * 0
        terms.append(torch.stack([distance_loss, contact_loss, site_loss]))
    return torch.stack(terms).mean(0)

"""Formal joint model: real global inputs in both ranking and structure tasks."""
import torch
from torch import nn
from torch.nn import functional as F

from biomaster.context_pocket import ContextPocketRanker
from biomaster.structural_training import balanced_mean


SOURCES = ('experimental_native', 'experimental_p2rank', 'alphafold_p2rank')


def source_features(batch, source):
    if source not in SOURCES:
        raise ValueError('explicit receptor and pocket origins required')
    b = dict(batch)
    valid = b['residue_mask'][:, :, None] & b['residue_mask'][:, None, :]
    b['residue_geometry'] = b['residue_geometry'].clone()
    b['residue_geometry'][..., -1] = valid.to(b['residue_geometry'].dtype)
    # occupancy, pLDDT, probability, experimental receptor, predicted receptor,
    # predicted pocket. Occupancy is never substituted for pLDDT.
    q = b['pocket_metadata'].new_zeros((len(b['owner']), 6))
    predicted_receptor = source == 'alphafold_p2rank'
    q[:, 1 if predicted_receptor else 0] = b['pocket_metadata'][:, 1]
    q[:, 3] = not predicted_receptor
    q[:, 4] = predicted_receptor
    if source != 'experimental_native':
        q[:, 2] = b['pocket_metadata'][:, 0]
        q[:, 5] = 1
    available = b['atom_mask'].any(1) & b['residue_mask'].any(1)
    b['quality_fields'] = q * available[:, None]
    return b


class FullContextPocketRanker(ContextPocketRanker):
    def __init__(self, parent_checkpoint, structural_checkpoint=None, cfg=None):
        super().__init__(parent_checkpoint, structural_checkpoint, cfg)
        c = self.cfg
        self.pocket_gate = nn.Sequential(nn.LayerNorm(c.width+c.parent_width+7),
            nn.Linear(c.width+c.parent_width+7, 128), nn.GELU(), nn.Linear(128, 1))
        self.ranking_head = nn.Sequential(nn.LayerNorm(c.parent_width+c.width+128+1+6+1),
            nn.Linear(c.parent_width+c.width+128+1+6+1, c.width), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(c.width, c.parent_width), nn.GELU(), nn.Linear(c.parent_width, 2))

    def global_context(self, batch):
        if batch is None:
            raise ValueError('real global features are required; zero-context fallback is disabled')
        widths = dict(drug_global=2560, drug_graph_mean=40, target_global=1280)
        for key, width in widths.items():
            if key not in batch or batch[key].ndim != 2 or batch[key].shape[1] != width:
                raise ValueError('invalid real global input: '+key)
        return super().global_context(batch)

    def forward(self, global_batch, pocket_batch, return_aux=False):
        g = self.global_context(global_batch)
        evidence, aux = self.interaction(source_features(pocket_batch, 'alphafold_p2rank'), g, return_aux)
        scores = self.ranking_head(evidence)
        return (scores, aux) if return_aux else scores

    def structural_forward(self, batch, global_batch, *, source):
        g = self.global_context(global_batch)
        _, aux = self.interaction(source_features(batch, source), g, True)
        return aux


def observed_structure_losses(aux, batch, labels, cfg):
    """Observed contacts/distances; multi-positive site supervision, no oracle crop.

    A predicted pocket with no observed contacts is a negative *site in this
    complex*, never an experimentally inactive drug/target relation. Missed
    sites have no gate target and remain in the coverage evaluation denominator.
    """
    terms = []
    for owner in torch.unique(batch['owner']):
        selected = batch['owner'] == owner
        valid = aux['pair_mask'][selected]
        if not valid.any():
            continue
        truth = labels['distance'][selected][valid].float()
        if not torch.isfinite(truth).all() or (truth < 0).any():
            raise ValueError('invalid observed structure labels')
        bins = (truth/cfg.distance_max*(cfg.distance_bins-1)).long().clamp_max(cfg.distance_bins-1)
        distance = F.cross_entropy(aux['distance_logits'][selected][valid].float(), bins, reduction='none')
        contact = truth < 4.5
        bce = F.binary_cross_entropy_with_logits(aux['contact_logits'][selected][valid].float(),
                                                contact.float(), reduction='none')
        site_positive = ((labels['distance'][selected] < 4.5) & valid).flatten(1).any(1)
        gate = aux['pocket_gate'][selected].float()
        site = (-torch.logsumexp(gate.log_softmax(0)[site_positive], 0)
                if site_positive.any() else gate.sum()*0)
        terms.append(torch.stack([balanced_mean(distance, truth < 16), balanced_mean(bce, contact), site]))
    if not terms:
        return aux['contact_logits'].sum()*0 + torch.zeros(3, device=batch['owner'].device)
    return torch.stack(terms).mean(0)

"""Global-conditioned pocket interaction with a trainable joint ranking head.

Reuses the audited six-block structural trunk, not the stopped downstream
weights. The old global network provides initialization only. Its task readout
and hidden-state residual bridge are not part of this model.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from biomaster.molecular_controls import MolecularControlConfig, MolecularControlInteraction
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig, mean_masked, safe_softmax


def sourced_pockets(batch, source):
    """Separate occupancy, pLDDT and pocket probability; do not alias quality.

The last legacy pair-geometry channel mixed occupancy with pLDDT. Replace it
with a constant valid-pair indicator, then expose the distinct quality fields
to the new gate/readout. Existing cached assets remain immutable. This changes
the input protocol, so the migrated structural baseline must be measured anew.
"""
    if source not in ('experimental', 'predicted'):
        raise ValueError('explicit structural source required')
    b = dict(batch)
    valid = b['residue_mask'][:, :, None] & b['residue_mask'][:, None, :]
    geometry = b['residue_geometry'].clone()
    geometry[..., -1] = valid.to(geometry.dtype)
    b['residue_geometry'] = geometry
    q = b['pocket_metadata'].new_zeros((len(b['owner']), 4))
    if source == 'experimental':
        q[:, 0] = b['pocket_metadata'][:, 1]  # occupancy, not confidence
    else:
        q[:, 1] = b['pocket_metadata'][:, 1]  # pLDDT / 100
        q[:, 2] = b['pocket_metadata'][:, 0]  # P2Rank probability
        q[:, 3] = 1  # explicit predicted-source indicator
    available = b['atom_mask'].any(1) & b['residue_mask'].any(1)
    b['quality_fields'] = q * available[:, None]
    return b


class ContextPocketRanker(nn.Module):
    def __init__(self, parent_checkpoint, structural_checkpoint=None, cfg=None):
        super().__init__()
        self.cfg = cfg or PocketPrecisionConfig()
        c = self.cfg
        parent = MolecularControlInteraction(MolecularControlConfig(**parent_checkpoint['config']))
        parent.load_state_dict(parent_checkpoint['model'], strict=True)
        if parent.cfg.width != c.parent_width:
            raise ValueError('global context width mismatch')
        self.drug_global = deepcopy(parent.drug_global)
        self.target_global = deepcopy(parent.target_global)
        self.global_fusion = deepcopy(parent.global_fusion)
        self.global_refine = deepcopy(parent.fusion)
        trunk = PocketPrecision(c)
        if structural_checkpoint is not None:
            if structural_checkpoint['config'] != c.to_dict():
                raise ValueError('structural trunk architecture mismatch')
            trunk.load_state_dict(structural_checkpoint['local_model'], strict=True)
        # Deliberately exclude legacy refine and pocket_gate.
        names = ['atom_input', 'residue_input', 'atom_bias', 'residue_bias', 'atom_pair',
                 'residue_pair', 'blocks', 'pool_attention', 'local_output', 'contact_head', 'distance_head']
        for name in names:
            setattr(self, name, getattr(trunk, name))
        self._structural_names = names
        self.context = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c.parent_width), nn.Linear(c.parent_width, 2*c.width+c.pair_width))
            for _ in range(c.blocks)])
        for layer in self.context:
            nn.init.normal_(layer[-1].weight, std=1e-3)
            nn.init.zeros_(layer[-1].bias)
        self.pocket_gate = nn.Sequential(nn.LayerNorm(c.width+c.parent_width+5),
                                         nn.Linear(c.width+c.parent_width+5, 128), nn.GELU(), nn.Linear(128, 1))
        self.ranking_head = nn.Sequential(
            nn.LayerNorm(c.parent_width+c.width+128+1+4+1),
            nn.Linear(c.parent_width+c.width+128+1+4+1, c.width), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(c.width, c.parent_width), nn.GELU(), nn.Linear(c.parent_width, 2))
        self.phase = 'joint'

    def set_phase(self, phase):
        if phase not in ('adaptation', 'joint'):
            raise ValueError('unknown training phase')
        self.phase = phase
        for name in self._structural_names:
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(phase == 'joint')
        # Global context, gate, context projections and new scoring always learn.
        for name in ['drug_global', 'target_global', 'global_fusion', 'global_refine',
                     'context', 'pocket_gate', 'ranking_head']:
            getattr(self, name).requires_grad_(True)

    def global_context(self, batch):
        drug = torch.cat([batch['drug_global'], batch['drug_graph_mean'],
                          batch['pretrained_available'][:, None]], -1)
        d, t = self.drug_global(drug), self.target_global(batch['target_global'])
        return self.global_refine(self.global_fusion(torch.cat([d, t, d*t, (d-t).abs()], -1)))

    def interaction(self, batch, context, return_aux=False):
        c = self.cfg
        am, rm = batch['atom_mask'], batch['residue_mask']
        pm = am[:, :, None] & rm[:, None, :]
        a = self.atom_input(torch.cat([batch['atom_tokens'], batch['atom_chemistry']], -1)) * am[..., None]
        r = self.residue_input(torch.cat([batch['residue_tokens'], batch['pocket_residue_tokens']], -1)) * rm[..., None]
        ab = self.atom_bias(batch['atom_distance'], batch['atom_edges'])
        rb = self.residue_bias(batch['residue_distance'], batch['residue_geometry'])
        z = (self.atom_pair(a)[:, :, None] + self.residue_pair(r)[:, None]) * pm[..., None]
        owner = batch['owner']
        g = context[owner]
        for block, conditioning in zip(self.blocks, self.context):
            ac, rc, pc = conditioning(g).split([c.width, c.width, c.pair_width], -1)
            a = (a+ac[:, None]) * am[..., None]
            r = (r+rc[:, None]) * rm[..., None]
            z = (z+pc[:, None, None]) * pm[..., None]
            if self.training and c.checkpoint_blocks:
                a, r, z = checkpoint(block, a, r, z, am, rm, ab, rb, use_reentrant=False)
            else:
                a, r, z = block(a, r, z, am, rm, ab, rb)
        pw = safe_softmax(self.pool_attention(z).squeeze(-1).flatten(1), pm.flatten(1))
        pair_pool = (pw[..., None]*z.flatten(1, 2)).sum(1)
        local = self.local_output(torch.cat([mean_masked(a, am, 1), mean_masked(r, rm, 1),
                                            pair_pool, batch['drug_aligned'], batch['pocket_aligned']], -1))
        aligned = batch['drug_aligned']*batch['pocket_aligned']
        cosine = aligned.sum(-1, keepdim=True)
        quality = batch['quality_fields']
        gate = self.pocket_gate(torch.cat([local, g, cosine, quality], -1)).squeeze(-1)
        available = am.any(1) & rm.any(1)
        # Keep global/local/matching evidence in separate channels until readout.
        pocket_evidence = torch.cat([local, aligned, cosine, quality], -1)
        pooled, exists, weights = [], [], torch.zeros_like(gate)
        for i in range(len(context)):
            take = owner == i
            w = safe_softmax(gate[take], available[take])
            pooled.append((w[:, None]*pocket_evidence[take]).sum(0))
            exists.append(available[take].any())
            weights[take] = w
        pooled = torch.stack(pooled)
        exists = torch.stack(exists).to(pooled.dtype)[:, None]
        evidence = torch.cat([context, pooled, exists], -1)
        aux = dict(pocket_gate=gate, pocket_weights=weights, pair_mask=pm,
                   relation_available=exists.squeeze(-1).bool(), evidence=evidence)
        if return_aux:
            aux.update(contact_logits=self.contact_head(z).squeeze(-1), distance_logits=self.distance_head(z))
        return evidence, aux

    def forward(self, global_batch, pocket_batch, return_aux=False):
        g = self.global_context(global_batch)
        evidence, aux = self.interaction(sourced_pockets(pocket_batch, 'predicted'), g, return_aux)
        scores = self.ranking_head(evidence)
        return (scores, aux) if return_aux else scores

    def structural_forward(self, batch):
        n = int(batch['owner'].max())+1
        context = batch['atom_tokens'].new_zeros((n, self.cfg.parent_width))
        _, aux = self.interaction(sourced_pockets(batch, 'experimental'), context, True)
        return aux

    def parameter_groups(self, global_lr=1e-5, structural_lr=1e-6, new_lr=1e-4):
        groups = {'global': [], 'structural': [], 'new': []}
        for name, p in self.named_parameters():
            prefix = name.split('.')[0]
            key = ('global' if prefix in ['drug_global', 'target_global', 'global_fusion', 'global_refine']
                   else 'structural' if prefix in self._structural_names else 'new')
            groups[key].append(p)
        return [dict(params=values, lr={'global': global_lr, 'structural': structural_lr, 'new': new_lr}[key],
                     group_name=key) for key, values in groups.items()]


def structural_consistency(student, teacher):
    """Teacher outputs are soft consistency targets, not observed contacts."""
    mask = student['pair_mask'] & teacher['pair_mask']
    if not mask.any():
        return student['contact_logits'].sum()*0
    s, t = student['contact_logits'][mask].float(), teacher['contact_logits'][mask].float().detach()
    contact = F.binary_cross_entropy_with_logits(s, t.sigmoid()) - F.binary_cross_entropy_with_logits(t, t.sigmoid())
    sd, td = student['distance_logits'][mask].float(), teacher['distance_logits'][mask].float().detach()
    distance = F.kl_div(sd.log_softmax(-1), td.softmax(-1), reduction='batchmean')
    return contact + distance

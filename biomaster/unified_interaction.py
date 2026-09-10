"""One representation for global, atom-residue and intra-pocket interactions.

No task-specific frozen score, retrieval-support score, or coordinate difference
between independently generated ligand and receptor frames enters this model.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class UnifiedConfig:
    variant: str = 'geometry'  # global, capacity, sequence, site, geometry
    width: int = 192
    pair_width: int = 32
    blocks: int = 2
    dropout: float = 0.1
    capacity_hidden: int = 768

    def to_dict(self):
        return asdict(self)


def masked_mean(x, mask, dim):
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1)


def masked_softmax(x, mask, dim):
    y = torch.softmax(x.float().masked_fill(~mask, -1e9), dim=dim) * mask
    return (y / y.sum(dim=dim, keepdim=True).clamp_min(1e-8)).to(x.dtype)


class ResidualMLP(nn.Module):
    def __init__(self, width, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden,width))

    def forward(self, x):
        return x + self.net(x)


class BondMessage(nn.Module):
    """Neighbor states conditioned by the bond to THAT neighbor, including stereo."""
    def __init__(self, width):
        super().__init__()
        self.value = nn.Linear(width,width,bias=False)
        self.bond = nn.Embedding(6,width,padding_idx=0)
        self.stereo = nn.Embedding(7,width,padding_idx=0)
        self.out = nn.Linear(width,width,bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, atoms, mask, neighbors, bond, stereo):
        valid = (neighbors >= 0) & mask[:,:,None]
        batch = torch.arange(atoms.shape[0],device=atoms.device)[:,None,None]
        indices = neighbors.clamp_min(0).long()
        valid = valid & mask[batch,indices]
        values = self.value(self.norm(atoms))[batch,indices]
        edge = self.bond(bond.long()) + self.stereo(stereo.long())
        messages = (values * torch.sigmoid(edge) + edge) * valid.unsqueeze(-1)
        aggregate = messages.sum(2) / valid.sum(2).clamp_min(1).unsqueeze(-1)
        return (atoms + self.out(aggregate)) * mask.unsqueeze(-1)


class PocketGeometry(nn.Module):
    """Rotation/translation invariant receptor-only CA neighborhoods.

    Invalid or unavailable geometry contributes exactly zero. The presence mask
    is not itself a learned scoring feature. Quality affects geometric messages.
    """
    def __init__(self, width):
        super().__init__()
        self.register_buffer('centers',torch.linspace(2,24,16))
        self.kernel = nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,1))
        self.value = nn.Linear(width,width,bias=False)
        self.out = nn.Linear(width,width,bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, residues, ca, quality, geometry_mask):
        distance = torch.cdist(ca.float(),ca.float(),compute_mode='donot_use_mm_for_euclid_dist')
        valid = geometry_mask[:,:,None] & geometry_mask[:,None,:] & (distance > 0) & (distance <= 24)
        rbf = torch.exp(-0.5*((distance.unsqueeze(-1)-self.centers)/1.5)**2)
        affinity = F.softplus(self.kernel(rbf).squeeze(-1)) * valid
        affinity = affinity * quality[:,:,None] * quality[:,None,:]
        weight = affinity / affinity.sum(-1,keepdim=True).clamp_min(1e-6)
        message = torch.bmm(weight.to(residues.dtype),self.value(self.norm(residues)))
        return residues + self.out(message) * quality.unsqueeze(-1) * geometry_mask.unsqueeze(-1)


class InteractionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w,p=cfg.width,cfg.pair_width
        self.atom_norm=nn.LayerNorm(w);self.residue_norm=nn.LayerNorm(w)
        self.atom_pair=nn.Linear(w,p);self.residue_pair=nn.Linear(w,p)
        self.global_pair=nn.Linear(w,p)
        self.pair_update=ResidualMLP(p,2*p,cfg.dropout)
        self.attention=nn.Linear(p,1,bias=False)
        self.atom_context=nn.Linear(w,w,bias=False);self.residue_context=nn.Linear(w,w,bias=False)
        self.atom_update=ResidualMLP(w,2*w,cfg.dropout)
        self.residue_update=ResidualMLP(w,2*w,cfg.dropout)
        self.condition=nn.Linear(w,2*w)
        self.global_update=nn.Sequential(nn.Linear(w+p,w),nn.GELU(),nn.Linear(w,w))
        self.bonds=BondMessage(w)
        self.geometry=PocketGeometry(w) if cfg.variant=='geometry' else None

    def forward(self, atoms,residues,pair,global_state,batch):
        am,rm=batch['atom_mask'],batch['residue_mask']
        atoms=self.bonds(atoms,am,batch['neighbors'],batch['bond'],batch['stereo'])
        if self.geometry is not None:
            residues=self.geometry(residues,batch['ca'],batch['quality'],batch['geometry_mask'] & rm)
        ac,rc=self.condition(global_state).chunk(2,-1)
        atoms=self.atom_norm(atoms+ac[:,None]) * am[:,:,None]
        residues=self.residue_norm(residues+rc[:,None]) * rm[:,:,None]
        a,r=self.atom_pair(atoms),self.residue_pair(residues)
        pair=self.pair_update(pair+a[:,:,None]+r[:,None,:]+a[:,:,None]*r[:,None,:]
                              +self.global_pair(global_state)[:,None,None])
        pm=am[:,:,None] & rm[:,None,:]
        pair=pair*pm.unsqueeze(-1)
        attention=self.attention(pair).squeeze(-1)
        a_context=torch.bmm(masked_softmax(attention,pm,2),self.atom_context(residues))
        r_context=torch.bmm(masked_softmax(attention,pm,1).transpose(1,2),self.residue_context(atoms))
        atoms=self.atom_update(atoms+a_context)*am[:,:,None]
        residues=self.residue_update(residues+r_context)*rm[:,:,None]
        pooled=masked_mean(pair.flatten(1,2),pm.flatten(1,2),1)
        local_available=pm.flatten(1).any(1).unsqueeze(-1)
        global_state=global_state+self.global_update(torch.cat([global_state,pooled],-1))*local_available
        return atoms,residues,pair,global_state


class UnifiedInteraction(nn.Module):
    def __init__(self,cfg=UnifiedConfig()):
        super().__init__()
        if cfg.variant not in ['global','capacity','sequence','site','geometry']:
            raise ValueError('unknown unified interaction variant')
        self.cfg=cfg
        w,p=cfg.width,cfg.pair_width
        self.drug_global=nn.Sequential(nn.LayerNorm(553),nn.Linear(553,w),nn.GELU())
        self.target_global=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,w),nn.GELU())
        self.global_fusion=nn.Sequential(nn.Linear(4*w,w),nn.GELU(),ResidualMLP(w,2*w,cfg.dropout))
        self.is_local=cfg.variant in ['sequence','site','geometry']
        if self.is_local:
            self.atom_input=nn.Sequential(nn.LayerNorm(552),nn.Linear(552,w))
            self.residue_input=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,w))
            self.position=nn.Linear(4,w,bias=False)
            self.blocks=nn.ModuleList([InteractionBlock(cfg) for _ in range(cfg.blocks)])
            self.final_pair_attention=nn.Linear(p,1,bias=False)
            self.fusion=nn.Sequential(nn.Linear(3*w+p,w),nn.GELU(),ResidualMLP(w,2*w,cfg.dropout))
        else:
            hidden=cfg.capacity_hidden if cfg.variant=='capacity' else 2*w
            count=cfg.blocks if cfg.variant=='capacity' else 1
            self.fusion=nn.Sequential(*[ResidualMLP(w,hidden,cfg.dropout) for _ in range(count)])
        self.shared=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.GELU(),nn.Dropout(cfg.dropout))
        self.readout=nn.Linear(w,2)

    def forward(self,batch,return_representation=False):
        dg=torch.cat([batch['drug_global'],batch['drug_graph_mean'],batch['pretrained_available'][:,None]],-1)
        d,t=self.drug_global(dg),self.target_global(batch['target_global'])
        g=self.global_fusion(torch.cat([d,t,d*t,(d-t).abs()],-1))
        if self.is_local:
            am,rm=batch['atom_mask'],batch['residue_mask']
            a=self.atom_input(torch.cat([batch['atom_tokens'],batch['atom_chemistry']],-1))*am[:,:,None]
            pos=batch['residue_position']
            pos=torch.stack([pos,torch.sin(pos*6.2831853),torch.cos(pos*6.2831853),pos.square()],-1)
            r=(self.residue_input(batch['residue_tokens'])+self.position(pos))*rm[:,:,None]
            pair=a.new_zeros((len(a),a.shape[1],r.shape[1],self.cfg.pair_width))
            for block in self.blocks:a,r,pair,g=block(a,r,pair,g,batch)
            pm=am[:,:,None]&rm[:,None,:]
            flat=pair.flatten(1,2)
            weight=masked_softmax(self.final_pair_attention(flat).squeeze(-1),pm.flatten(1,2),1)
            pooled=(flat*weight.unsqueeze(-1)).sum(1)
            g=self.fusion(torch.cat([g,masked_mean(a,am,1),masked_mean(r,rm,1),pooled],-1))
        else:
            g=self.fusion(g)
        representation=self.shared(g)
        scores=self.readout(representation)
        return (scores,representation) if return_representation else scores


def capacity_matched_config(reference=None):
    """Match actual trainable parameter count, before looking at any outcomes."""
    reference=reference or UnifiedConfig(variant='geometry')
    target=sum(p.numel() for p in UnifiedInteraction(reference).parameters())
    values=reference.to_dict();values['variant']='capacity'
    # Parameter count is affine in hidden width. Solve with two evaluations.
    counts=[]
    for hidden in [1,2]:
        values['capacity_hidden']=hidden
        counts.append(sum(p.numel() for p in UnifiedInteraction(UnifiedConfig(**values)).parameters()))
    hidden=max(1,round(1+(target-counts[0])/(counts[1]-counts[0])))
    values['capacity_hidden']=hidden
    return UnifiedConfig(**values)

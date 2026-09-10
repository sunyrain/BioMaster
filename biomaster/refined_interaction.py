"""Warm-started refinement of one global representation, with one shared readout."""
from dataclasses import asdict,dataclass
import torch
from torch import nn
from biomaster.unified_interaction import (UnifiedInteraction,UnifiedConfig,InteractionBlock,
                                         ResidualMLP,masked_mean,masked_softmax)


@dataclass
class RefinedConfig:
    variant:str='site'
    width:int=192
    pair_width:int=32
    blocks:int=1
    dropout:float=.1
    capacity_hidden:int=1


class RefinedInteraction(UnifiedInteraction):
    def __init__(self,cfg):
        if cfg.variant not in ['global','capacity','site','geometry']:
            raise ValueError('invalid refinement')
        # All global parameters keep the baseline's names and shapes. A single
        # representation is refined BEFORE its shared head; no legacy logits.
        super().__init__(UnifiedConfig(variant='global',width=cfg.width,dropout=cfg.dropout))
        self.cfg=cfg;self.is_local=cfg.variant in ['site','geometry']
        w,p=cfg.width,cfg.pair_width
        if self.is_local:
            self.atom_input=nn.Sequential(nn.LayerNorm(552),nn.Linear(552,w))
            self.residue_input=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,w))
            self.position=nn.Linear(4,w,bias=False)
            self.interactions=nn.ModuleList([InteractionBlock(cfg) for _ in range(cfg.blocks)])
            self.pair_attention=nn.Linear(p,1,bias=False)
            self.refine=nn.Sequential(nn.LayerNorm(3*w+p),nn.Linear(3*w+p,w),nn.GELU(),nn.Linear(w,w))
            nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
        elif cfg.variant=='capacity':
            self.refine=ResidualMLP(w,cfg.capacity_hidden,cfg.dropout)
            nn.init.zeros_(self.refine.net[-1].weight);nn.init.zeros_(self.refine.net[-1].bias)

    def warm_start(self, global_state):
        baseline=UnifiedInteraction(UnifiedConfig(variant='global',width=self.cfg.width,dropout=self.cfg.dropout))
        baseline.load_state_dict(global_state,strict=True)
        current=self.state_dict()
        for name,value in global_state.items():current[name]=value
        self.load_state_dict(current,strict=True)

    def forward(self,batch,return_representation=False):
        dg=torch.cat([batch['drug_global'],batch['drug_graph_mean'],batch['pretrained_available'][:,None]],-1)
        d,t=self.drug_global(dg),self.target_global(batch['target_global'])
        g=self.fusion(self.global_fusion(torch.cat([d,t,d*t,(d-t).abs()],-1)))
        if self.is_local:
            am,rm=batch['atom_mask'],batch['residue_mask']
            a=self.atom_input(torch.cat([batch['atom_tokens'],batch['atom_chemistry']],-1))*am[:,:,None]
            pos=batch['residue_position']
            pos=torch.stack([pos,torch.sin(pos*6.2831853),torch.cos(pos*6.2831853),pos.square()],-1)
            r=(self.residue_input(batch['residue_tokens'])+self.position(pos))*rm[:,:,None]
            z=a.new_zeros((len(a),a.shape[1],r.shape[1],self.cfg.pair_width))
            context=g
            for block in self.interactions:a,r,z,context=block(a,r,z,context,batch)
            mask=am[:,:,None]&rm[:,None,:];flat=z.flatten(1,2)
            weight=masked_softmax(self.pair_attention(flat).squeeze(-1),mask.flatten(1,2),1)
            local=torch.cat([context-g,masked_mean(a,am,1),masked_mean(r,rm,1),(flat*weight[:,:,None]).sum(1)],-1)
            g=g+self.refine(local)*mask.flatten(1).any(1)[:,None]
        elif self.cfg.variant=='capacity':g=self.refine(g)
        representation=self.shared(g);scores=self.readout(representation)
        return (scores,representation) if return_representation else scores


def matched_refinement_config(reference):
    target=sum(p.numel() for p in RefinedInteraction(reference).parameters())
    values=asdict(reference);values['variant']='capacity';counts=[]
    for hidden in [1,2]:
        values['capacity_hidden']=hidden
        counts.append(sum(p.numel() for p in RefinedInteraction(RefinedConfig(**values)).parameters()))
    values['capacity_hidden']=max(1,round(1+(target-counts[0])/(counts[1]-counts[0])))
    return RefinedConfig(**values)

"""A single shared interaction model anchored to the selected global encoder."""
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from biomaster.unified_interaction import UnifiedConfig,UnifiedInteraction,ResidualMLP
from biomaster.refined_interaction import RefinedConfig,RefinedInteraction
from biomaster.best_model_training import SupplementedFeatureBank


@dataclass
class AnchoredConfig(RefinedConfig):
    parent_variant:str='global'
    parent_blocks:int=2
    parent_capacity_hidden:int=768
    drug_representation:str='drugclip'


def chemical_width(representation):
    if representation=='drugclip':return 512
    if representation=='bermol':return 768
    if representation=='bermol_morgan':return 2816
    if representation=='morgan':return 2048
    if representation=='drugclip_morgan':return 2560
    raise ValueError('unknown chemical representation')


def replace_input(model,representation,width):
    if representation!='drugclip':
        size=chemical_width(representation)+41
        model.drug_global=nn.Sequential(nn.LayerNorm(size),nn.Linear(size,width),nn.GELU())


def parent_model(cfg):
    if cfg.parent_variant not in ['global','capacity']:raise ValueError('parent must be a global interaction model')
    base=UnifiedInteraction(UnifiedConfig(variant=cfg.parent_variant,width=cfg.width,dropout=cfg.dropout,
        blocks=cfg.parent_blocks,capacity_hidden=cfg.parent_capacity_hidden))
    replace_input(base,cfg.drug_representation,cfg.width)
    return base


class AnchoredInteraction(RefinedInteraction):
    """Retain the exact parent representation and readout; learn one local update."""
    def __init__(self,cfg):
        super().__init__(cfg)
        if cfg.parent_variant=='capacity':
            self.fusion=nn.Sequential(*[ResidualMLP(cfg.width,cfg.parent_capacity_hidden,cfg.dropout)
                                       for _ in range(cfg.parent_blocks)])
        elif cfg.parent_variant!='global':raise ValueError('invalid parent architecture')
        replace_input(self,cfg.drug_representation,cfg.width)

    def warm_start(self,state):
        parent=parent_model(self.cfg);parent.load_state_dict(state,strict=True)
        values=self.state_dict()
        for name,value in state.items():
            if values[name].shape!=value.shape:raise ValueError(f'parent shape mismatch: {name}')
            values[name]=value
        self.load_state_dict(values,strict=True)


def anchored_config(parent_config,variant):
    return AnchoredConfig(variant=variant,width=parent_config['width'],dropout=parent_config['dropout'],
        pair_width=32,blocks=1,parent_variant=parent_config['variant'],
        parent_blocks=parent_config.get('blocks',2),parent_capacity_hidden=parent_config.get('capacity_hidden',768),
        drug_representation=parent_config.get('drug_representation','drugclip'))


def matched_anchored_config(reference):
    target=sum(p.numel() for p in AnchoredInteraction(reference).parameters())
    values=asdict(reference);values['variant']='capacity';counts=[]
    for hidden in [1,2]:
        values['capacity_hidden']=hidden
        counts.append(sum(p.numel() for p in AnchoredInteraction(AnchoredConfig(**values)).parameters()))
    values['capacity_hidden']=max(1,round(1+(target-counts[0])/(counts[1]-counts[0])))
    return AnchoredConfig(**values)


class AnchoredFeatureBank(SupplementedFeatureBank):
    def __init__(self,base,supplement,source,representation,device='cuda',local=True):
        chemical_width(representation)
        super().__init__(base,supplement,device=device,local=local)
        self.representation=representation;source=Path(source)
        if representation=='drugclip':return
        if representation in ['bermol','bermol_morgan']:
            required=self.required.cpu().numpy()
            if not np.load(source/'BERMOL_DONE.npy')[required].all():raise ValueError('incomplete BerMol bank')
            values=np.load(source/'BERMOL.npy')
            if not np.isfinite(values[required]).all():raise ValueError('invalid BerMol representation')
            self.bermol=torch.tensor(values,device=self.device)
        if representation in ['morgan','drugclip_morgan','bermol_morgan']:
            self.morgan=torch.tensor(np.load(source/'MORGAN.npy'),device=self.device)

    def batch(self,drug_ids,target_ids,variant):
        batch=super().batch(drug_ids,target_ids,variant)
        if self.representation!='drugclip':
            d=torch.tensor(drug_ids,dtype=torch.long,device=self.device)
            if self.representation in ['bermol','bermol_morgan']:
                batch['drug_global']=self.bermol[d];batch['pretrained_available']=torch.ones(len(d),device=self.device)
            if self.representation=='morgan':
                batch['drug_global']=self.morgan[d].float();batch['pretrained_available']=torch.ones(len(d),device=self.device)
            elif self.representation.endswith('_morgan'):
                batch['drug_global']=torch.cat([batch['drug_global'],self.morgan[d].float()],-1)
        return batch

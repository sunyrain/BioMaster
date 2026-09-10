"""Controlled chemical input replacement, preserving the global interaction trunk."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from biomaster.unified_interaction import UnifiedConfig,UnifiedInteraction
from biomaster.best_model_training import SupplementedFeatureBank


@dataclass
class ChemicalConfig(UnifiedConfig):
    drug_representation:str='bermol'


class ChemicalInteraction(UnifiedInteraction):
    def __init__(self,cfg):
        if cfg.variant!='global' or cfg.drug_representation not in ['bermol','bermol_morgan']:
            raise ValueError('only declared global chemical replacements are supported')
        super().__init__(cfg)
        self.chemical_width=768+(2048 if cfg.drug_representation=='bermol_morgan' else 0)
        size=self.chemical_width+41
        self.drug_global=nn.Sequential(nn.LayerNorm(size),nn.Linear(size,cfg.width),nn.GELU())


class ChemicalFeatureBank(SupplementedFeatureBank):
    def __init__(self,base,supplement,source,representation,device='cuda',local=False):
        if local or representation not in ['bermol','bermol_morgan']:raise ValueError('invalid chemical feature route')
        super().__init__(base,supplement,device=device,local=False)
        self.representation=representation;source=Path(source)
        done=np.load(source/'BERMOL_DONE.npy')
        required=self.required.cpu().numpy()
        if not done[required].all():raise ValueError('missing BerMol representation')
        features=np.load(source/'BERMOL.npy')
        if not np.isfinite(features[required]).all() or not (np.linalg.norm(features[required],axis=1)>0).all():
            raise ValueError('invalid BerMol feature')
        self.bermol=torch.tensor(features,device=self.device)
        if representation=='bermol_morgan':
            morgan=np.load(source/'MORGAN.npy')
            if morgan.shape!=(len(done),2048) or not np.isin(morgan,[0,1]).all():
                raise ValueError('invalid Morgan binary feature bank')
            self.morgan=torch.tensor(morgan,device=self.device)

    def batch(self,drug_ids,target_ids,variant):
        batch=super().batch(drug_ids,target_ids,variant)
        d=torch.tensor(drug_ids,dtype=torch.long,device=self.device)
        chemical=self.bermol[d]
        if self.representation=='bermol_morgan':chemical=torch.cat([chemical,self.morgan[d].float()],-1)
        batch['drug_global']=chemical
        # BerMol is complete on the declared feature axis. DrugCLIP conformer
        # success is not an input when DrugCLIP is replaced.
        batch['pretrained_available']=torch.ones(len(d),device=self.device)
        return batch

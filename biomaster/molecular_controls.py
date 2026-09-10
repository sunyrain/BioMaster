"""Fingerprint-only and DrugCLIP-plus-fingerprint controls for chemical inputs."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from biomaster.unified_interaction import UnifiedConfig,UnifiedInteraction
from biomaster.best_model_training import SupplementedFeatureBank


@dataclass
class MolecularControlConfig(UnifiedConfig):
    drug_representation:str='morgan'


class MolecularControlInteraction(UnifiedInteraction):
    def __init__(self,cfg):
        if cfg.variant!='global' or cfg.drug_representation not in ['morgan','drugclip_morgan']:
            raise ValueError('invalid molecular control')
        super().__init__(cfg)
        width=2048+(512 if cfg.drug_representation=='drugclip_morgan' else 0)+41
        self.drug_global=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,cfg.width),nn.GELU())


class MolecularControlBank(SupplementedFeatureBank):
    def __init__(self,base,supplement,source,representation,device='cuda',local=False):
        if local or representation not in ['morgan','drugclip_morgan']:raise ValueError('invalid control route')
        super().__init__(base,supplement,device=device,local=False)
        self.representation=representation
        fp=np.load(Path(source)/'MORGAN.npy')
        if fp.shape!=(len(self.required),2048) or not np.isin(fp,[0,1]).all():raise ValueError('invalid fingerprint bank')
        self.morgan=torch.tensor(fp,device=self.device)

    def batch(self,drug_ids,target_ids,variant):
        batch=super().batch(drug_ids,target_ids,variant)
        d=torch.tensor(drug_ids,dtype=torch.long,device=self.device);fp=self.morgan[d].float()
        if self.representation=='morgan':
            batch['drug_global']=fp;batch['pretrained_available']=torch.ones(len(d),device=self.device)
        else:batch['drug_global']=torch.cat([batch['drug_global'],fp],-1)
        return batch

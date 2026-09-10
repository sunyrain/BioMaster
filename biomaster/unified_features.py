"""Immutable feature banks for UnifiedInteraction; no labels or support scores."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


class UnifiedFeatureBank:
    def __init__(self,path,device='cuda',local=True):
        self.path=Path(path);self.device=torch.device(device)
        for name in ['ATOM','TARGET']:
            manifest=json.loads((self.path/(name+'_MANIFEST.json')).read_text())
            if manifest['status']!='COMPLETE' or manifest['labels_used']:
                raise ValueError('features must be complete and label independent')
        index=np.load(self.path/'ATOM_INDEX.npz')
        self.lengths=torch.as_tensor(index['lengths'].copy(),device=device)
        self.offsets=torch.as_tensor(index['offsets'][:-1].copy(),device=device)
        self.graph_mean=torch.as_tensor(index['graph_mean'].copy(),device=device)
        self.required=torch.zeros(len(self.lengths),dtype=torch.bool,device=device)
        self.required[np.load(self.path/'REQUIRED_MOLECULE_IDS.npy')]=True
        def load(name):
            return torch.from_numpy(np.load(self.path/(name+'.npy'))).to(device)
        self.drug_global=load('MOLECULE_GLOBAL')
        self.available=load('PRETRAINED_AVAILABLE').float()
        self.target_global=load('TARGET_GLOBAL_MEAN')
        self.local=local
        if local:
            self.atom_tokens=load('ATOM_TOKENS')
            self.atom_chemistry=load('ATOM_CHEMISTRY')
            self.neighbors=load('ATOM_NEIGHBORS')
            self.bond=load('ATOM_BOND');self.stereo=load('ATOM_STEREO')
            self.residues={mode:load('TARGET_'+mode.upper()+'_TOKENS') for mode in ['sequence','site']}
            self.indices={mode:load('TARGET_'+mode.upper()+'_INDICES') for mode in ['sequence','site']}
            self.ca=load('TARGET_CA');self.quality=load('TARGET_QUALITY');self.geom_mask=load('TARGET_GEOMETRY_MASK')
            lengths=pd.read_csv(self.path/'TARGET_COVERAGE.csv').sort_values('target_feature_index').length.to_numpy()
            self.protein_lengths=torch.tensor(lengths,device=device)

    def batch(self,drug_ids,target_ids,variant):
        d=torch.as_tensor(drug_ids,dtype=torch.long,device=self.device)
        t=torch.as_tensor(target_ids,dtype=torch.long,device=self.device)
        if not self.required[d].all():raise ValueError('molecule outside prepared feature bank')
        batch=dict(drug_global=self.drug_global[d],drug_graph_mean=self.graph_mean[d],
                   pretrained_available=self.available[d],target_global=self.target_global[t])
        if variant in ['global','capacity']:return batch
        if not self.local:raise ValueError('local features were not loaded')
        length=self.lengths[d];maximum=max(1,int(length.max()))
        am=torch.arange(maximum,device=self.device)[None,:]<length[:,None]
        take=(self.offsets[d,None]+torch.arange(maximum,device=self.device)[None,:]).clamp_max(len(self.atom_tokens)-1)
        mode='sequence' if variant=='sequence' else 'site'
        indices=self.indices[mode][t];rm=indices>=0
        positions=indices.clamp_min(0)/self.protein_lengths[t,None].clamp_min(1)
        batch.update(atom_tokens=self.atom_tokens[take].float()*am[:,:,None],
                     atom_chemistry=self.atom_chemistry[take].float()*am[:,:,None],
                     atom_mask=am,neighbors=torch.where(am[:,:,None],self.neighbors[take],-1),bond=self.bond[take],stereo=self.stereo[take],
                     residue_tokens=self.residues[mode][t].float(),residue_mask=rm,residue_position=positions,
                     ca=self.ca[t],quality=self.quality[t],geometry_mask=self.geom_mask[t])
        return batch

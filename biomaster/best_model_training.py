"""Dated query supervision and feature supplements for controlled model selection."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from biomaster.unified_features import UnifiedFeatureBank


class SupplementedFeatureBank(UnifiedFeatureBank):
    """Overlay additional label-free molecules without mutating the base cache."""
    def __init__(self, base, supplement, device='cuda', local=True):
        super().__init__(base, device=device, local=local)
        supplement = Path(supplement)
        manifest = json.loads((supplement / 'ATOM_MANIFEST.json').read_text())
        if manifest['status'] != 'COMPLETE' or manifest['labels_used']:
            raise ValueError('supplement is not complete and label independent')
        ids_np = np.load(supplement / 'REQUIRED_MOLECULE_IDS.npy')
        ids = torch.tensor(ids_np, device=self.device, dtype=torch.long)
        if self.required[ids].any():
            raise ValueError('supplement overwrites frozen base molecules')
        if not np.load(supplement / 'ATOM_DONE.npy')[ids_np].all():
            raise ValueError('incomplete supplemental features')
        index = np.load(supplement / 'ATOM_INDEX.npz')
        def selected(name):
            return torch.tensor(np.load(supplement / (name+'.npy'), mmap_mode='r')[ids_np].copy(), device=self.device)
        self.drug_global[ids] = selected('MOLECULE_GLOBAL')
        self.available[ids] = selected('PRETRAINED_AVAILABLE').float()
        self.graph_mean[ids] = torch.tensor(index['graph_mean'][ids_np], device=self.device)
        self.lengths[ids] = torch.tensor(index['lengths'][ids_np], device=self.device)
        if local:
            offset = len(self.atom_tokens)
            self.offsets[ids] = torch.tensor(index['offsets'][ids_np]+offset, device=self.device)
            for field, name in [('atom_tokens','ATOM_TOKENS'),('atom_chemistry','ATOM_CHEMISTRY'),
                                ('neighbors','ATOM_NEIGHBORS'),('bond','ATOM_BOND'),('stereo','ATOM_STEREO')]:
                extra = torch.tensor(np.load(supplement / (name+'.npy')), device=self.device)
                setattr(self, field, torch.cat([getattr(self, field), extra], dim=0))
        self.required[ids] = True

    def batch(self, drug_ids, target_ids, variant):
        d, t = np.asarray(drug_ids), np.asarray(target_ids)
        if d.ndim != 1 or t.ndim != 1 or len(d) != len(t) or not len(d):
            raise ValueError('nonempty equal length 1D entity IDs required')
        if d.dtype.kind not in 'iu' or t.dtype.kind not in 'iu':
            raise ValueError('entity IDs must be integers')
        if d.min() < 0 or d.max() >= len(self.lengths) or t.min() < 0 or t.max() >= len(self.target_global):
            raise ValueError('entity ID outside feature registry')
        return super().batch(d,t,variant)


def old_labels(frame, old, nt):
    matrix = np.zeros((len(old),nt),bool)
    index = old.set_index('drug_feature_index').old_drug_index
    rows = frame[frame.binary_label.eq(1) & frame.drug_feature_index.isin(index.index)]
    matrix[rows.drug_feature_index.map(index).to_numpy(int), rows.target_feature_index.to_numpy(int)] = True
    return matrix


class RollingStage:
    def __init__(self, data, source, cutoff):
        if cutoff not in [2018,2020]:
            raise ValueError('development cutoff must be 2018 or 2020')
        folder = Path(data) / f'roll_{cutoff}'
        source = Path(source)
        self.name=f'roll_{cutoff}'; self.cutoff=cutoff
        self.train=pd.read_csv(folder / 'TRAIN.csv.gz')
        self.val=pd.read_csv(folder / 'VALIDATION.csv.gz')
        if self.train.max_document_year.max()>cutoff or self.val.min_document_year.min()<=cutoff:
            raise ValueError('temporal boundary violation')
        if self.val.max_document_year.max()>cutoff+2:
            raise ValueError('validation after permitted development window')
        if not self.train.binary_label.isin([0,1]).all() or self.train.duplicated(['drug_feature_index','target_feature_index']).any():
            raise ValueError('training requires unique observed binary relations')
        self.d=self.train.drug_feature_index.to_numpy(int)
        self.t=self.train.target_feature_index.to_numpy(int)
        self.y=self.train.binary_label.to_numpy(np.float32)
        self.old=pd.read_csv(source / 'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
        self.nt=len(pd.read_csv(source / 'TARGET_INDEX.csv.gz'))
        self.known=old_labels(self.train,self.old,self.nt)
        self.val_known=old_labels(self.val,self.old,self.nt)
        self.risk=np.load(folder / 'RISK.npy')
        if self.train.merge(self.val,on=['drug_feature_index','target_feature_index']).shape[0]:
            raise ValueError('training/validation relation overlap')
        if (self.val_known & ~self.risk).any():
            raise ValueError('validation positive outside risk set')
        self.positive_queries=[np.flatnonzero(self.known.any(1)),np.flatnonzero(self.known.any(0))]

    def retrieval(self, rng, p):
        ds,ts,ys,groups=[],[],[],[]; start=0
        for head in [0,1]:
            matrix=self.known if head==0 else self.known.T
            queries=rng.choice(self.positive_queries[head],p['retrieval_queries_per_direction'],replace=False)
            for query in queries:
                positive=np.flatnonzero(matrix[query]); other=np.flatnonzero(~matrix[query])
                count=min(len(other),max(1,p['retrieval_candidates']-len(positive)))
                candidates=np.r_[positive,rng.choice(other,count,replace=False)];rng.shuffle(candidates)
                if head==0:
                    d=np.full(len(candidates),self.old.drug_feature_index.iloc[query]);t=candidates
                else:
                    d=self.old.drug_feature_index.to_numpy()[candidates];t=np.full(len(candidates),query)
                ds.extend(d);ts.extend(t);ys.extend(matrix[query,candidates]);groups.append((start,start+len(candidates),head))
                start+=len(candidates)
        return np.array(ds,int),np.array(ts,int),np.array(ys,bool),groups


class ObservedQueries:
    """Uniform query sampling; only measured positives and measured negatives."""
    def __init__(self, frame, group_column, head):
        if not frame.binary_label.isin([0,1]).all():
            raise ValueError('query labels must be observed binary labels')
        self.d=frame.drug_feature_index.to_numpy(int);self.t=frame.target_feature_index.to_numpy(int)
        self.y=frame.binary_label.to_numpy(bool);self.head=head;self.groups=[]
        for ids in frame.groupby(group_column,sort=True).indices.values():
            pos=ids[self.y[ids]];neg=ids[~self.y[ids]]
            if len(pos) and len(neg):self.groups.append((pos,neg))
        if not self.groups:raise ValueError('no observed two-class queries')

    def sample(self,rng,queries=2,per_class=16):
        out=[]
        for q in rng.choice(len(self.groups),min(queries,len(self.groups)),replace=False):
            pos,neg=self.groups[q]
            ids=np.r_[rng.choice(pos,min(per_class,len(pos)),replace=False),
                      rng.choice(neg,min(per_class,len(neg)),replace=False)]
            out.append((self.d[ids],self.t[ids],self.y[ids],self.head))
        return out


def measured_pair_loss(scores, labels, temperature=1.0):
    """Equal weight to an observed positive/negative ordering within one query."""
    if temperature<=0:raise ValueError('positive temperature required')
    labels=labels.bool()
    if not labels.any() or labels.all():raise ValueError('both observed classes required')
    return F.softplus((scores[~labels][None,:]-scores[labels][:,None])/temperature).mean()


class WeightAverage:
    """One EMA model; the exported checkpoint does not require an ensemble."""
    def __init__(self,model,decay=.995):
        if not 0<=decay<1:raise ValueError('EMA decay outside [0,1)')
        self.model=copy.deepcopy(model).eval();self.decay=decay;self.updates=0
        for p in self.model.parameters():p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates+=1
        # Correct initialization bias without retaining a randomly initialized
        # model for hundreds of early optimization steps.
        decay=min(self.decay,(1+self.updates)/(10+self.updates))
        for dest,src in zip(self.model.parameters(),model.parameters(),strict=True):
            dest.lerp_(src.detach(),1-decay)
        for dest,src in zip(self.model.buffers(),model.buffers(),strict=True):dest.copy_(src)

"""Portable inference for one selected global or anchored catalog model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .model_registry import build_model


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''):value.update(chunk)
    return value.hexdigest()


class CatalogRanker:
    """Rank the exact packaged catalog. Scores are logits, not probabilities."""
    def __init__(self,directory,device='cpu',verify=True):
        self.directory=Path(directory)
        if verify:
            manifest=json.loads((self.directory/'MANIFEST.json').read_text())
            for name,record in manifest['files'].items():
                path=self.directory/name
                if not path.exists() and record.get('optional_geometry'):continue
                if not path.is_file() or digest(path)!=record['sha256']:
                    raise ValueError(f'bundle integrity check failed: {name}')
        self.metadata=json.loads((self.directory/'metadata.json').read_text())
        if self.metadata['format_version']!=2:raise ValueError('unsupported bundle format')
        self.drugs=pd.read_csv(self.directory/'drugs.csv.gz',keep_default_na=False)
        self.targets=pd.read_csv(self.directory/'targets.csv.gz',keep_default_na=False)
        if self.drugs.drug_id.duplicated().any() or self.targets.target_id.duplicated().any():
            raise ValueError('duplicate catalog identifiers')
        if len(self.drugs)!=self.metadata['drugs'] or len(self.targets)!=self.metadata['targets']:
            raise ValueError('catalog size mismatch')
        self._drugs={str(v):i for i,v in enumerate(self.drugs.drug_id)}
        self._targets={str(v):i for i,v in enumerate(self.targets.target_id)}
        self.device=torch.device(device)
        state=torch.load(self.directory/'model.pt',map_location='cpu',weights_only=True)
        self.model=build_model(state['architecture'],state['config'])
        self.model.load_state_dict(state['model'],strict=True);self.model.to(self.device).eval()
        self.features={}
        def load(name):
            arr=np.load(self.directory/'features'/(name+'.npy'),allow_pickle=False)
            if arr.dtype.kind=='f' and not np.isfinite(arr).all():raise ValueError(f'nonfinite feature {name}')
            value=torch.tensor(arr,device=self.device);self.features[name]=value
            return value
        for name in ['drug_global','drug_graph_mean','pretrained_available','target_global']:load(name)
        nd,nt=len(self.drugs),len(self.targets)
        representation=getattr(self.model.cfg,'drug_representation','drugclip')
        width={'drugclip':512,'bermol':768,'bermol_morgan':2816,'morgan':2048,'drugclip_morgan':2560}[representation]
        expected={'drug_global':(nd,width),'drug_graph_mean':(nd,40),'pretrained_available':(nd,),'target_global':(nt,1280)}
        for name,shape in expected.items():
            if tuple(self.features[name].shape)!=shape:raise ValueError(f'feature shape mismatch {name}')
        self.geometry_available=False
        if self.model.is_local:
            for name in ['atom_lengths','atom_offsets','atom_tokens','atom_chemistry','neighbors','bond','stereo',
                         'residue_tokens','residue_indices','protein_lengths']:load(name)
            f=self.features
            if tuple(f['atom_lengths'].shape)!=(nd,) or tuple(f['atom_offsets'].shape)!=(nd+1,):
                raise ValueError('invalid compact atom index')
            if f['atom_offsets'][0]!=0 or not torch.equal(f['atom_offsets'][1:]-f['atom_offsets'][:-1],f['atom_lengths']):
                raise ValueError('atom lengths/offsets mismatch')
            if int(f['atom_offsets'][-1])!=len(f['atom_tokens']):raise ValueError('atom cache length mismatch')
            if tuple(f['residue_tokens'].shape[:2])!=tuple(f['residue_indices'].shape) or len(f['residue_indices'])!=nt:
                raise ValueError('residue cache axes mismatch')
            geom_names=['ca','quality','geometry_mask']
            if all((self.directory/'features'/(n+'.npy')).exists() for n in geom_names):
                for name in geom_names:load(name)
                self.geometry_available=bool(f['geometry_mask'].any())
            else:
                shape=f['residue_indices'].shape
                f['ca']=torch.zeros((*shape,3),device=self.device)
                f['quality']=torch.zeros(shape,device=self.device)
                f['geometry_mask']=torch.zeros(shape,device=self.device,dtype=torch.bool)

    def _indices(self,values,lookup,kind):
        if isinstance(values,(str,bytes)) or not hasattr(values,'__len__') or len(values)==0:
            raise ValueError(f'{kind} IDs must be a nonempty sequence of catalog strings')
        if any(not isinstance(x,str) for x in values):raise ValueError(f'{kind} IDs must be strings')
        missing=[v for v in values if v not in lookup]
        if missing:raise ValueError(f'unknown {kind} catalog ID: {missing[0]}')
        return np.array([lookup[v] for v in values],dtype=np.int64)

    def _batch(self,d,t,use_geometry=True):
        f=self.features;d=torch.tensor(d,device=self.device);t=torch.tensor(t,device=self.device)
        batch={name:f[name][d if name!='target_global' else t] for name in
               ['drug_global','drug_graph_mean','pretrained_available','target_global']}
        if not self.model.is_local:return batch
        length=f['atom_lengths'][d];maximum=max(1,int(length.max()))
        am=torch.arange(maximum,device=self.device)[None,:]<length[:,None]
        take=(f['atom_offsets'][d,None]+torch.arange(maximum,device=self.device)[None,:]).clamp_max(len(f['atom_tokens'])-1)
        indices=f['residue_indices'][t];rm=indices>=0
        positions=indices.clamp_min(0)/f['protein_lengths'][t,None].clamp_min(1)
        batch.update(atom_tokens=f['atom_tokens'][take].float()*am[:,:,None],
            atom_chemistry=f['atom_chemistry'][take].float()*am[:,:,None],atom_mask=am,
            neighbors=torch.where(am[:,:,None],f['neighbors'][take],-1),bond=f['bond'][take],stereo=f['stereo'][take],
            residue_tokens=f['residue_tokens'][t].float(),residue_mask=rm,residue_position=positions,
            ca=f['ca'][t],quality=f['quality'][t],
            geometry_mask=f['geometry_mask'][t] if use_geometry else torch.zeros_like(f['geometry_mask'][t]))
        return batch

    @torch.inference_mode()
    def score_pairs(self,drug_ids,target_ids,batch_size=128,use_geometry=True):
        d=self._indices(drug_ids,self._drugs,'drug');t=self._indices(target_ids,self._targets,'target')
        if len(d)!=len(t):raise ValueError('one target ID required per drug ID')
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:raise ValueError('positive integer batch_size required')
        out=np.empty((len(d),2),np.float32)
        for start in range(0,len(d),batch_size):
            stop=min(start+batch_size,len(d))
            out[start:stop]=self.model(self._batch(d[start:stop],t[start:stop],use_geometry)).float().cpu().numpy()
        if not np.isfinite(out).all():raise FloatingPointError('nonfinite model score')
        return pd.DataFrame(dict(drug_id=list(drug_ids),target_id=list(target_ids),
            drug_to_target_score=out[:,0],target_to_drug_score=out[:,1]))

    def rank_targets(self,drug_id,candidates=None,top_k=20,**kwargs):
        if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k<1:raise ValueError('positive integer top_k required')
        self._indices([drug_id],self._drugs,'drug')
        candidates=list(self.targets.target_id) if candidates is None else candidates
        self._indices(candidates,self._targets,'target')
        if len(set(candidates))!=len(candidates):raise ValueError('ranking candidates must be unique')
        frame=self.score_pairs([drug_id]*len(candidates),candidates,**kwargs)
        frame['gene_symbol']=frame.target_id.map(self.targets.set_index('target_id').gene_symbol)
        return self._rank(frame,'target_id','drug_to_target_score',top_k)

    def rank_drugs(self,target_id,candidates=None,top_k=20,**kwargs):
        if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k<1:raise ValueError('positive integer top_k required')
        self._indices([target_id],self._targets,'target')
        candidates=list(self.drugs.drug_id) if candidates is None else candidates
        self._indices(candidates,self._drugs,'drug')
        if len(set(candidates))!=len(candidates):raise ValueError('ranking candidates must be unique')
        frame=self.score_pairs(candidates,[target_id]*len(candidates),**kwargs)
        frame['drug_name']=frame.drug_id.map(self.drugs.set_index('drug_id').name)
        return self._rank(frame,'drug_id','target_to_drug_score',top_k)

    @staticmethod
    def _rank(frame,key,score,top_k):
        if frame[key].duplicated().any():raise ValueError('ranking candidates must be unique')
        if not isinstance(top_k,int) or isinstance(top_k,bool) or top_k<1:raise ValueError('positive integer top_k required')
        frame=frame.sort_values([score,key],ascending=[False,True]).reset_index(drop=True)
        frame['rank']=np.arange(1,len(frame)+1);frame['candidate_count']=len(frame)
        return frame.head(top_k)

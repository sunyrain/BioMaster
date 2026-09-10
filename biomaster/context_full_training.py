"""Complete validation axes, real structure context, and resumable samplers."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from biomaster.structural_training import StructuralDataset
from biomaster.ranking_audit import risk_set_ranking, query_bootstrap


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


class RealContextStructures(StructuralDataset):
    def __init__(self, root, records, *, predicted=False, device='cuda'):
        self.overlay=Path(root)
        manifest=json.loads((self.overlay/'MANIFEST.json').read_text())
        super().__init__(Path(manifest['base']),records,device)
        self.predicted=predicted

    @lru_cache(maxsize=64)
    def load(self, index):
        record=self.records[index]
        path=self.overlay/record['predicted_file'] if self.predicted else self.root/record['file']
        expected=record['predicted_sha256'] if self.predicted else record['identity']['sha256']
        if file_sha(path)!=expected:raise ValueError('structural feature identity changed')
        item=torch.load(path,map_location='cpu',weights_only=False)
        if item['input_ligand_geometry']!='INDEPENDENT_ETKDG':raise ValueError('bound pose used as input')
        return item

    @lru_cache(maxsize=256)
    def context(self,index):
        record=self.records[index];path=self.overlay/record['global_file']
        if file_sha(path)!=record['global_sha256']:raise ValueError('global context identity changed')
        with np.load(path) as f:values={k:f[k].copy() for k in f.files}
        for key in ['drug_global','target_global']:
            if not np.isfinite(values[key]).all() or not np.any(values[key]):
                raise ValueError('missing or nonfinite real global context')
        return values

    def real_batch(self,indices):
        if any(not self.load(int(i))['pockets'] for i in indices):
            if len(indices)!=1:raise ValueError('handle missing predictions per complex')
            return None
        b,labels=self.batch(indices)
        contexts=[self.context(int(i)) for i in indices]
        g={k:torch.tensor(np.stack([v[k] for v in contexts]),device=self.device,dtype=torch.float32)
           for k in contexts[0]}
        return b,labels,g


class PermutationStream:
    """Consume every observed row before reshuffling; state is checkpointed."""
    def __init__(self,values,rng):
        self.values=np.asarray(values);self.order=rng.permutation(self.values);self.cursor=0;self.epochs=0

    def take(self,count,rng):
        pieces=[]
        while count:
            n=min(count,len(self.order)-self.cursor)
            pieces.append(self.order[self.cursor:self.cursor+n]);self.cursor+=n;count-=n
            if self.cursor==len(self.order):
                self.epochs+=1;self.order=rng.permutation(self.values);self.cursor=0
        return np.concatenate(pieces)

    def state_dict(self):
        return dict(order=self.order,cursor=self.cursor,epochs=self.epochs)

    def load_state_dict(self,state):
        if not np.array_equal(np.sort(state['order']),np.sort(self.values)):raise ValueError('sampler universe changed')
        self.order=state['order'];self.cursor=state['cursor'];self.epochs=state['epochs']


def query_cycle(queries,rng):
    orders=[rng.permutation(q) for q in queries]
    return np.array([(h,int(orders[h][i])) for i in range(max(map(len,orders))) for h in [0,1]
                     if i<len(orders[h])],dtype=np.int64)


def validation_pairs(stage):
    """Union of ALL candidates of ALL positive development queries (both heads)."""
    dq=np.flatnonzero(stage.val_known.any(1));tq=np.flatnonzero(stage.val_known.any(0))
    needed=np.zeros_like(stage.val_known,dtype=bool);needed[dq,:]=True;needed[:,tq]=True
    d,t=np.nonzero(needed)
    return dq,tq,d,t


def evaluate_matrices(stage,scores,reference=None):
    dq,tq,_,_=validation_pairs(stage);results={};frames={};positives={}
    for h,name in enumerate(['d2t','t2d']):
        queries=dq if h==0 else tq
        labels=stage.val_known if h==0 else stage.val_known.T
        risk=stage.risk if h==0 else stage.risk.T
        s=scores[:,:,h] if h==0 else scores[:,:,h].T
        metrics,frame,pairs=risk_set_ranking(labels[queries],s[queries],risk[queries],queries,np.arange(labels.shape[1]))
        if len(frame)!=len(queries):raise ValueError('incomplete validation queries')
        results[name]=metrics;frames[name]=frame;positives[name]=pairs
        if reference is not None:results[name]['paired_ap_vs_parent']=query_bootstrap(frame,reference[name])
    return results,frames,positives


def capture_rng(rng):
    return dict(numpy=rng.bit_generator.state,torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state,rng):
    rng.bit_generator.state=state['numpy'];torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

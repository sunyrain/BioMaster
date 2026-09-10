"""Pair-level endpoint ablations with shared explicit inactivity and conflict audit."""
from __future__ import annotations
import numpy as np
import pandas as pd

ARMS={'kdki_inactive':['AFFINITY_KD_KI'],
      'all_inactive':['AFFINITY_KD_KI','ACTIVITY_IC50','ACTIVITY_EC50']}


def combine(labels, explicit_inactive_pairs, tasks):
    """One vote per pair; inactivity cannot override a positive observation."""
    f=labels.copy()
    f['explicit_inactive']=f.pair_id.isin(explicit_inactive_pairs)
    f['positive_any_task']=f.has_positive | f.has_conflict
    anypos=f.groupby('pair_id').positive_any_task.max()
    f=f[f.task.isin(tasks) | f.explicit_inactive].copy()
    selected=f.task.isin(tasks)
    f['positive']=f.has_positive & selected
    f['negative']=(f.has_negative & selected) | f.explicit_inactive
    f['conflict']=f.has_conflict & selected
    result=f.groupby('pair_id',sort=True).agg(
        drug_feature_index=('drug_feature_index','first'),target_feature_index=('target_feature_index','first'),
        molecule_id=('molecule_id','first'),target_id=('target_id','first'),split=('split','first'),split_group=('split_group','first'),
        positive=('positive','max'),negative=('negative','max'),conflict=('conflict','max'),
        explicit_inactive=('explicit_inactive','max'),seen_by_parent=('seen_by_frozen_parent_connectivity_pair','max')).reset_index()
    result['inactive_positive_conflict']=result.explicit_inactive & result.pair_id.map(anypos)
    result['conflict'] |= (result.positive & result.negative) | result.inactive_positive_conflict
    result['eligible']=(result.positive | result.negative) & ~result.conflict
    result['binary_label']=result.positive.astype('int8')
    result['inactivity_added_without_numeric_negative']=result.explicit_inactive & ~result.pair_id.isin(
        labels.loc[labels.task.isin(tasks) & labels.has_negative,'pair_id'])
    return result


class CyclingRows:
    """Complete shuffled passes before repeating; checkpointable independent RNG."""
    def __init__(self,n,seed):
        self.n=n;self.rng=np.random.default_rng(seed);self.order=self.rng.permutation(n);self.offset=0;self.passes=0
    def take(self,size):
        chunks=[]
        while size:
            available=min(size,self.n-self.offset)
            chunks.append(self.order[self.offset:self.offset+available]);self.offset+=available;size-=available
            if self.offset==self.n:
                self.passes+=1;self.order=self.rng.permutation(self.n);self.offset=0
        return np.concatenate(chunks)

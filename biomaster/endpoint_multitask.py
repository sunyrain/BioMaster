"""Measured-query and endpoint-specific auxiliary supervision for A/B fits."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from biomaster.model_registry import build_model

VARIANTS = {'binary': (False, False), 'rank': (True, False),
            'regression': (False, True), 'joint': (True, True)}
ENDPOINTS = {'kdki_inactive': ['Kd', 'Ki'],
             'all_inactive': ['Kd', 'Ki', 'IC50', 'EC50']}


class AuxiliaryInteraction(nn.Module):
    def __init__(self, config, endpoints):
        super().__init__()
        self.base = build_model('molecular_control', config)
        self.regression = nn.Linear(self.base.readout.in_features, len(endpoints))

    def forward(self, batch):
        logits, representation = self.base(batch, return_representation=True)
        return logits, self.regression(representation)


class MeasuredRankPool:
    """Sample only existing task rows, uniformly over queries with both classes."""
    def __init__(self, frame, group_column):
        if not frame.binary_label.isin([0, 1]).all():
            raise ValueError('Only observed binary labels are allowed')
        self.d = frame.drug_feature_index.to_numpy(np.int64)
        self.t = frame.target_feature_index.to_numpy(np.int64)
        y = frame.binary_label.to_numpy(bool)
        self.groups = []
        for indices in frame.groupby(group_column, sort=True).indices.values():
            pos, neg = indices[y[indices]], indices[~y[indices]]
            if len(pos) and len(neg):
                self.groups.append((pos, neg))
        if not self.groups:
            raise ValueError('No measured two-class query')

    def sample(self, rng, queries=2, per_class=16):
        rows = []
        for q in rng.choice(len(self.groups), min(queries, len(self.groups)), replace=False):
            pos, neg = self.groups[q]
            # Repetition is within a measured class, never from unknown pairs.
            rows.append(np.stack([rng.choice(pos, per_class, replace=len(pos) < per_class),
                                  rng.choice(neg, per_class, replace=len(neg) < per_class)]))
        ids = np.stack(rows)
        return self.d[ids], self.t[ids]


def query_rank_loss(scores, temperature=1.):
    """scores: query x (positive, negative) x candidates; equal query weights."""
    if temperature <= 0 or scores.ndim != 3 or scores.shape[1] != 2:
        raise ValueError('Positive temperature and query x 2 x candidates required')
    pos, neg = scores[:, 0], scores[:, 1]
    return F.softplus((neg[:, None, :] - pos[:, :, None]) / temperature).mean()


def endpoint_huber(predictions, values, endpoint_ids, means, scales):
    """Only selected exact observations enter the loss; unused heads get no target."""
    selected = predictions[torch.arange(len(values), device=predictions.device), endpoint_ids]
    if not torch.isfinite(values).all() or not torch.isfinite(selected).all():
        raise ValueError('Observed values and selected predictions must be finite')
    if not (scales > 0).all():
        raise ValueError('Positive train-only endpoint scales required')
    truth = (values - means[endpoint_ids]) / scales[endpoint_ids]
    row_loss = F.huber_loss(selected, truth, reduction='none', delta=1.)
    return torch.stack([row_loss[endpoint_ids == e].mean() for e in endpoint_ids.unique()]).mean()


class ValidationPlateau:
    """Validation-selected checkpoint; meaningful plateau drives LR drops and stop."""
    def __init__(self, lr=3e-4, minimum_lr=5e-6, delta=2e-4,
                 reduce_every=4, patience=12, min_passes=10):
        self.lr, self.minimum_lr, self.delta = lr, minimum_lr, delta
        self.reduce_every, self.patience, self.min_passes = reduce_every, patience, min_passes
        self.best = -float('inf')
        self.anchor = -float('inf')
        self.stale = 0
        self.observations = 0
        self.reductions_since_improvement = 0

    def observe(self, score, passes):
        if not np.isfinite(score):
            raise ValueError('Finite validation selection score required')
        self.observations += 1
        best = score > self.best
        self.best = max(self.best, score)
        if score > self.anchor + self.delta:
            self.anchor = score
            self.stale = 0
            self.reductions_since_improvement = 0
        else:
            self.stale += 1
        stop = (passes >= self.min_passes and self.stale >= self.patience
                and (self.reductions_since_improvement >= 2 or self.lr <= self.minimum_lr))
        reduced = False
        if not stop and self.stale and self.stale % self.reduce_every == 0:
            next_lr = max(self.minimum_lr, self.lr * .5)
            reduced = next_lr < self.lr
            self.lr = next_lr
            self.reductions_since_improvement += int(reduced)
        return dict(best=best, reduced_lr=reduced, converged=stop)


def stream_state(stream):
    return dict(order=stream.order.copy(), offset=stream.offset, passes=stream.passes,
                rng=copy.deepcopy(stream.rng.bit_generator.state))


def restore_stream(stream, state):
    stream.order = state['order'].copy()
    stream.offset, stream.passes = state['offset'], state['passes']
    stream.rng.bit_generator.state = copy.deepcopy(state['rng'])

"""Measured-assay supervision and auditable target/class exposure controls."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from biomaster.endpoint_ablation import CyclingRows


def balanced_target_weights(frame, smoothing=20., maximum_gain=5.):
    """Equal class mass; inverse-root target/class exposure, bounded relative gain."""
    if maximum_gain < 1 or smoothing < 0 or set(frame.binary_label.unique()) != {0, 1}:
        raise ValueError('Two observed classes, nonnegative smoothing and gain >= 1 required')
    counts = frame.groupby(['target_id', 'binary_label']).binary_label.transform('size').to_numpy()
    raw = (counts + smoothing) ** -.5
    labels = frame.binary_label.to_numpy()
    weights = np.empty(len(frame), np.float64)
    for label in [0, 1]:
        mask = labels == label
        lo, hi = 0., maximum_gain / raw[mask].min()
        for _ in range(70):
            mid = (lo + hi) / 2
            if np.minimum(mid * raw[mask], maximum_gain).mean() < 1:
                lo = mid
            else:
                hi = mid
        gain = np.minimum((lo + hi) / 2 * raw[mask], maximum_gain)
        weights[mask] = .5 / mask.mean() * gain
        assert np.isclose(weights[mask].sum(), len(frame) / 2)
    return weights.astype(np.float32)


class RotatingTargetClassRows:
    """Rotate through every observed row, taking up to cap per target/class/round."""
    def __init__(self, frame, seed, cap=150):
        if cap < 1 or frame.empty:
            raise ValueError('Positive cap and nonempty frame required')
        self.n = len(frame); self.cap = cap; self.rng = np.random.default_rng(seed)
        self.groups = [np.asarray(g, np.int64) for g in frame.groupby(['target_id', 'binary_label'], sort=True).indices.values()]
        self.group_streams = [CyclingRows(len(g), seed + 1000 + i) for i, g in enumerate(self.groups)]
        self.order = np.empty(0, np.int64); self.offset = 0; self.passes = 0
        self.round_size = sum(min(cap, len(g)) for g in self.groups)
        labels = frame.binary_label.to_numpy()
        self.positive_fraction = sum(min(cap, len(g)) * int(labels[g[0]]) for g in self.groups) / self.round_size
        if not 0 < self.positive_fraction < 1:
            raise ValueError('Both observed classes required')

    def take(self, size):
        chunks = []
        while size:
            if self.offset == len(self.order):
                self.order = np.concatenate([g[self._take_group(s, min(self.cap, len(g)))] for g, s in zip(self.groups, self.group_streams)])
                self.rng.shuffle(self.order); self.offset = 0
            count = min(size, len(self.order) - self.offset)
            chunks.append(self.order[self.offset:self.offset + count]); self.offset += count; size -= count
            if self.offset == len(self.order):
                self.passes += 1
        return np.concatenate(chunks)

    @staticmethod
    def _take_group(stream, count):
        remaining = stream.n - stream.offset
        if remaining >= count:
            return stream.take(count)
        # On a cycle boundary, keep the just-seen tail out of the new head.
        # Every round still contains distinct compounds, including uneven groups.
        tail = stream.take(remaining)
        repeated = np.isin(stream.order, tail)
        stream.order = np.concatenate([stream.order[~repeated], stream.order[repeated]])
        return np.concatenate([tail, stream.take(count - remaining)])


def row_stream_state(stream):
    from biomaster.endpoint_multitask import stream_state
    state = stream_state(stream)
    if isinstance(stream, RotatingTargetClassRows):
        state['group_streams'] = [stream_state(s) for s in stream.group_streams]
    return state


def restore_row_stream(stream, state):
    from biomaster.endpoint_multitask import restore_stream
    restore_stream(stream, state)
    if isinstance(stream, RotatingTargetClassRows):
        for child, value in zip(stream.group_streams, state['group_streams'], strict=True):
            restore_stream(child, value)


class AssayPairs:
    """Pairs stay within source assay + exact target sequence + endpoint."""
    def __init__(self, frame, minimum_difference=.3):
        self.frame = frame.reset_index(drop=True)
        self.minimum_difference = minimum_difference
        self.values = self.frame.p_activity.to_numpy(np.float32)
        self.d = self.frame.drug_feature_index.to_numpy(np.int64)
        self.t = self.frame.target_feature_index.to_numpy(np.int64)
        self.groups = []
        for _, ids in self.frame.groupby('assay_group', sort=True).indices.items():
            group = self.frame.iloc[ids]
            if group.target_feature_index.nunique() != 1 or group.endpoint.nunique() != 1:
                raise ValueError('Mixed target or endpoint inside assay group')
            if group.pair_id.nunique() != len(group):
                raise ValueError('Repeated compound inside assay group')
            if len(ids) >= 2 and np.ptp(self.values[ids]) >= minimum_difference:
                self.groups.append(ids)
        if not self.groups:
            raise ValueError('No comparable assay pairs')

    def sample(self, rng, count):
        pairs = []
        for gi in rng.integers(len(self.groups), size=count):
            ids = self.groups[gi]
            first = rng.choice(ids)
            other = ids[np.abs(self.values[ids] - self.values[first]) >= self.minimum_difference]
            if len(other):
                second = rng.choice(other)
            else:
                first, second = ids[np.argmin(self.values[ids])], ids[np.argmax(self.values[ids])]
            pairs.append([first, second])
        indices = np.asarray(pairs, np.int64)
        return self.d[indices], self.t[indices], self.values[indices]


def assay_difference_huber(predicted_standardized_pairs, true_pairs, endpoint_scale):
    """True and predicted differences are compared in physical p-activity units."""
    if predicted_standardized_pairs.ndim != 2 or predicted_standardized_pairs.shape[1] != 2:
        raise ValueError('Expected pair x 2 values')
    if true_pairs.shape != predicted_standardized_pairs.shape or float(endpoint_scale) <= 0:
        raise ValueError('Matching pair values and positive scale required')
    pred = (predicted_standardized_pairs[:, 0] - predicted_standardized_pairs[:, 1]) * endpoint_scale
    truth = true_pairs[:, 0] - true_pairs[:, 1]
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError('Finite exact observations required')
    return F.huber_loss(pred, truth, delta=1.)

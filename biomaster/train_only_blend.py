"""Training-only chemical retrieval and validation-only linear rank selection."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score


class TrainOnlyEvidence:
    """Exact-sequence positive neighbors; smoothed measured training prevalence.

    Query labels are deliberately not accepted. Fingerprint arrays contain only
    binary Morgan bits. Same-connectivity queries cannot retrieve themselves or
    their stereoisomers as a positive neighbor.
    """
    def __init__(self, train, fingerprint_bank, device='cuda', prior_strength=10.):
        if not train.binary_label.isin([0, 1]).all():
            raise ValueError('Training must contain measured binary labels only')
        if train.pair_id.duplicated().any():
            raise ValueError('One measured label per training pair required')
        if 'split' in train and not train.split.eq('train').all():
            raise ValueError('Only training members can populate the evidence pool')
        if prior_strength <= 0:
            raise ValueError('Prior smoothing must be positive')
        self.train = train.reset_index(drop=True)
        self.fps = fingerprint_bank
        self.device = device
        self.global_prior = float(train.binary_label.mean())
        self.counts = train.groupby('target_id').binary_label.agg(['sum', 'count'])
        self.priors = ((self.counts['sum'] + prior_strength*self.global_prior) /
                       (self.counts['count'] + prior_strength))
        self.pools = {target: f.drop_duplicates('molecule_id').reset_index(drop=True)
                      for target, f in train[train.binary_label.eq(1)].groupby('target_id', sort=False)}

    @torch.inference_mode()
    def transform(self, molecules, targets, query_fingerprints, progress=None):
        molecules = np.asarray(molecules, str)
        targets = np.asarray(targets, str)
        qfp = np.asarray(query_fingerprints)
        if len(molecules) != len(targets) or len(qfp) != len(targets):
            raise ValueError('Query identity and fingerprint rows must align')
        if not np.isin(qfp, [0, 1]).all():
            raise ValueError('Expected binary fingerprints')
        qconn = np.array([m.split('-')[0] for m in molecules])
        result = np.zeros(len(targets), np.float32)
        nearest = np.full(len(targets), '', dtype=object)
        available = np.zeros(len(targets), np.int32)
        excluded = np.zeros(len(targets), np.int32)
        groups = pd.Series(targets).groupby(pd.Series(targets), sort=True).indices
        for number, (target, positions) in enumerate(groups.items(), 1):
            pool = self.pools.get(target)
            if pool is None:
                continue
            refs = pool.molecule_id.to_numpy(str)
            rconn = np.array([m.split('-')[0] for m in refs])
            refbits = np.asarray(self.fps[pool.drug_feature_index.to_numpy(int)], np.float32)
            if not np.isin(refbits, [0, 1]).all():
                raise ValueError('Reference fingerprints must be binary')
            rb = torch.as_tensor(refbits, device=self.device)
            rsum = rb.sum(1)
            connections = {k: i for i, k in enumerate(sorted(set(rconn) | set(qconn[positions])))}
            rc = torch.tensor([connections[k] for k in rconn], device=self.device)
            for start in range(0, len(positions), 256):
                ix = positions[start:start+256]
                qb = torch.tensor(qfp[ix], device=self.device, dtype=torch.float32)
                qc = torch.tensor([connections[k] for k in qconn[ix]], device=self.device)
                same = qc[:, None] == rc[None, :]
                intersection = qb @ rb.T
                union = qb.sum(1)[:, None] + rsum[None, :] - intersection
                sim = torch.where(union > 0, intersection / union.clamp_min(1), 0.)
                sim.masked_fill_(same, -1.)
                best, arg = sim.max(1)
                best = best.cpu().numpy(); arg = arg.cpu().numpy()
                excluded[ix] = same.sum(1).cpu().numpy()
                available[ix] = len(refs) - excluded[ix]
                result[ix] = np.maximum(best, 0)
                supported = best > 0
                nearest[ix[supported]] = refs[arg[supported]]
            if progress and (number % 200 == 0 or number == len(groups)):
                progress(number, len(groups))
        prior = pd.Series(targets).map(self.priors).fillna(self.global_prior).to_numpy(float)
        if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
            raise ValueError('Invalid Tanimoto output')
        return pd.DataFrame(dict(positive_neighbor=result, target_prior=prior,
            nearest_training_positive=nearest, eligible_positive_neighbors=available,
            same_connectivity_references_excluded=excluded))


def validation_rank_selection(frame, features, grid):
    """Mean drug AP minus five-group instability; labels only from validation."""
    if 'split' in frame and not frame.split.eq('validation').all():
        raise ValueError('Weights must be selected exclusively on validation')
    features = np.asarray(features, float)
    if features.shape != (len(frame), 3) or not np.isfinite(features).all():
        raise ValueError('Expected finite neural / neighbor / prior features')
    mean, scale = features.mean(0), features.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (features-mean)/scale
    y = frame.binary_label.to_numpy(int)
    groups = []
    for key, ix in frame.groupby('molecule_id', sort=True).indices.items():
        if 0 < y[ix].sum() < len(ix):
            fold = int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16) % 5
            groups.append((ix, fold))
    if len({fold for _, fold in groups}) != 5:
        raise ValueError('All five validation drug groups must contain mixed-label queries')
    records = []
    for neighbor_weight, prior_weight in grid:
        score = z @ np.array([1., neighbor_weight, prior_weight])
        folds = [[] for _ in range(5)]
        for ix, fold in groups:
            folds[fold].append(average_precision_score(y[ix], score[ix]))
        means = np.array([np.mean(x) for x in folds])
        records.append(dict(neighbor_weight=float(neighbor_weight),prior_weight=float(prior_weight),
            robust_objective=float(means.mean()-means.std()),fold_mean=float(means.mean()),
            fold_std=float(means.std()),drug_macro_ap=float(np.mean([a for xs in folds for a in xs])),
            **{f'fold_{i}_ap':float(value) for i,value in enumerate(means)}))
    table = pd.DataFrame(records)
    def best(rows):
        return max(rows.to_dict('records'), key=lambda r: (r['robust_objective'],r['fold_mean'],
            -r['fold_std'],-r['neighbor_weight']-r['prior_weight']))
    choices = {'A_neural':best(table[table.neighbor_weight.eq(0)&table.prior_weight.eq(0)]),
        'A_plus_neighbor':best(table[table.prior_weight.eq(0)]),
        'A_plus_prior':best(table[table.neighbor_weight.eq(0)]),
        'A_joint_blend':best(table)}
    return dict(mean=mean.tolist(),scale=scale.tolist(),choices=choices,
                validation_drug_queries=len(groups),fold_query_counts=[sum(f==i for _,f in groups) for i in range(5)]), table


def apply_selection(features, selection):
    z = (np.asarray(features,float)-selection['mean'])/selection['scale']
    return {name:z @ np.array([1.,r['neighbor_weight'],r['prior_weight']])
            for name,r in selection['choices'].items()}

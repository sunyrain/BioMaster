"""Known pharmacological relationship retrieval, distinct from activity BCE."""
import torch
from torch.nn import functional as F


def known_relation_loss(scores, known, temperature=2.0):
    """Average log probability of all known relations, then average queries.

    Unlabelled candidates form a retrieval denominator; this loss does not
    assign a biochemical inactive label. Queries without positives are skipped.
    Competing known positives are all retained, with total query weight one.
    """
    if scores.shape != known.shape or scores.ndim != 2:
        raise ValueError('aligned query-by-candidate score and known-label matrices required')
    if temperature <= 0 or not torch.isfinite(scores).all():
        raise ValueError('finite scores and a positive temperature are required')
    known=known.bool()
    eligible=known.any(1)
    if not eligible.any():
        return scores.sum()*0
    log_probability=F.log_softmax(scores[eligible]/temperature,dim=1)
    positives=known[eligible]
    return -((log_probability*positives).sum(1)/positives.sum(1)).mean()

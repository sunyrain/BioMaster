"""Protected global backbone and full-candidate query gradients."""
import torch
from torch import nn
from torch.nn import functional as F

from biomaster.molecular_controls import MolecularControlConfig, MolecularControlInteraction
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig


class ProtectedPocketRanker(nn.Module):
    def __init__(self, parent_checkpoint, cfg=None):
        super().__init__()
        self.parent = MolecularControlInteraction(MolecularControlConfig(**parent_checkpoint['config']))
        self.parent.load_state_dict(parent_checkpoint['model'], strict=True)
        self.parent.requires_grad_(False).eval()
        self.local = PocketPrecision(cfg or PocketPrecisionConfig())
        if self.parent.cfg.width != self.local.cfg.parent_width:
            raise ValueError('incompatible global representation width')

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def parent_state(self, batch):
        p = self.parent
        dg = torch.cat([batch['drug_global'], batch['drug_graph_mean'], batch['pretrained_available'][:, None]], -1)
        d, t = p.drug_global(dg), p.target_global(batch['target_global'])
        g = p.global_fusion(torch.cat([d, t, d * t, (d - t).abs()], -1))
        return p.fusion(g)

    def forward(self, global_batch, pocket_batch, return_aux=False):
        with torch.no_grad():
            g = self.parent_state(global_batch)
            parent_scores = self.parent.readout(self.parent.shared(g))
        if return_aux:
            g, aux = self.local(pocket_batch, g, return_aux=True)
            aux['parent_scores'] = parent_scores
        else:
            g = self.local(pocket_batch, g)
        scores = self.parent.readout(self.parent.shared(g))
        return (scores, aux) if return_aux else scores


def multi_positive_ranking_loss(scores, positive, temperature=1.0):
    if scores.ndim != 1 or positive.shape != scores.shape or not positive.any():
        raise ValueError('one full candidate query with at least one known positive required')
    if temperature <= 0:
        raise ValueError('positive temperature required')
    # Uniform target mass across known positives. Unobserved candidates are
    # retrieval background; this is not a biochemical inactivity label.
    return -F.log_softmax(scores.float() / temperature, dim=0)[positive].mean()


def cached_query_backward(forward, drugs, targets, positive, head, model, *, batch_size=1, weight=1.0):
    """Exact full-query gradient with bounded activation memory.

    Both passes use deterministic eval-mode forward (grad enabled in pass two).
    This disables dropout for this objective instead of using inconsistent masks.
    No top-k candidate truncation or detached surrogate gradient is introduced.
    """
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            all_scores = torch.cat([forward(drugs[s:s+batch_size], targets[s:s+batch_size])[:, head]
                                    for s in range(0, len(drugs), batch_size)])
        leaf = all_scores.detach().float().requires_grad_(True)
        loss = multi_positive_ranking_loss(leaf, positive.to(leaf.device)) * weight
        gradient, = torch.autograd.grad(loss, leaf)
        for s in range(0, len(drugs), batch_size):
            score = forward(drugs[s:s+batch_size], targets[s:s+batch_size])[:, head]
            (score.float() * gradient[s:s+batch_size]).sum().backward()
        return float(loss.detach())
    finally:
        model.train(training)

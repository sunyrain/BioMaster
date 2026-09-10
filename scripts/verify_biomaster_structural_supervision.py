#!/usr/bin/env python3
"""Actual experimental-label/full-width optimizer check; weights are discarded."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig
from biomaster.structural_training import StructuralDataset, complex_losses
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json
from prepare_biomaster_structural_training import OUT


def main():
    records = [dict(file=str(p.relative_to(OUT)), system_id=p.stem) for p in sorted((OUT / 'encoded').glob('*.pt'))[:12]]
    if len(records) < 12:
        raise ValueError('12 real encoded structures required')
    torch.set_num_threads(4); torch.manual_seed(57); torch.backends.cuda.matmul.allow_tf32 = False
    cfg = PocketPrecisionConfig(); model = PocketPrecision(cfg).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = StructuralDataset(OUT, records); losses, first_gradients, seconds = [], {}, []
    for begin in range(0, 12, 2):
        started = time.monotonic(); batch, labels = dataset.batch([begin, begin+1])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, aux = model(batch, torch.zeros(2, 192, device='cuda'), return_aux=True)
            terms = complex_losses(aux, batch, labels, cfg)
            loss = (terms * torch.tensor([1., 1., .2], device='cuda')).sum()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if begin == 0:
            for name in ['distance_head.1.weight', 'contact_head.1.weight', 'atom_input.1.weight',
                         'residue_input.1.weight', 'residue_bias.net.0.weight', 'blocks.5.outer_out.weight']:
                grad = dict(model.named_parameters())[name].grad
                if grad is None or not torch.isfinite(grad).all() or grad.norm() == 0:
                    raise ValueError('invalid experimental-label gradient: ' + name)
                first_gradients[name] = float(grad.norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        losses.append(terms.detach().float().cpu().tolist()); seconds.append(time.monotonic()-started)
    result = dict(status='PASS', full_model_parameters=sum(p.numel() for p in model.parameters()),
                  real_complexes=12, optimizer_updates=6, losses_distance_contact_site=losses,
                  first_step_gradient_norms=first_gradients, median_update_seconds=float(np.median(seconds)),
                  weights_discarded=True, qualification='engineering only; sampled before final overlap admission, never used to initialize formal training',
                  downstream_ranking_evaluated=False, script=file_identity(Path(__file__)))
    write_json(OUT.parent.parent / 'STRUCTURAL_ENGINEERING_RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

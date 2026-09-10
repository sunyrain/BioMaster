#!/usr/bin/env python3
"""Full-width real-data GPU integration check. Does not report ranking quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.molecular_controls import MolecularControlBank
from biomaster.pocket_precision import PocketPrecisionConfig
from biomaster.pocket_precision_features import PocketPrecisionBank
from biomaster.pocket_precision_training import ProtectedPocketRanker
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json
from prepare_biomaster_pocket_precision import BASE, SUPPLEMENT, SOURCE, OUTPUT


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    config_path = ROOT / 'configs/biomaster_pocket_precision_20260906.json'
    cfg = PocketPrecisionConfig(**json.loads(config_path.read_text())['model'])
    selection = json.loads((ROOT / 'outputs/biomaster_best_model_20260906/GLOBAL_PARENT_SELECTION.json').read_text())
    source = next(r for r in selection['parents'] if r['cutoff'] == 2020 and r['seed'] == 20260921)
    if file_identity(source['checkpoint']['path']) != source['checkpoint']:
        raise ValueError('frozen parent identity changed')
    ckpt = torch.load(source['checkpoint']['path'], map_location='cpu', weights_only=False)
    model = ProtectedPocketRanker(ckpt, cfg).cuda()
    parent_before = state_hash(model.parent)
    bank = PocketPrecisionBank(OUTPUT, BASE, SUPPLEMENT)
    globals = MolecularControlBank(BASE, SUPPLEMENT, SOURCE, 'drugclip_morgan')
    data = pd.read_csv(ROOT / 'outputs/biomaster_best_model_20260906/data/roll_2020/TRAIN.csv.gz')
    assert data.max_document_year.max() <= 2020
    # Engineering coverage includes observed positives and measured negatives,
    # with different target pockets; this is not a sampled evaluation panel.
    examples = pd.concat([data[data.binary_label.eq(y) & data.target_feature_index.isin(bank.pockets)
                               & data.drug_feature_index.isin(bank.molecule_source)].drop_duplicates('target_feature_index').head(args.steps)
                          for y in [0, 1]]).sample(frac=1, random_state=13).head(args.steps)
    optimizer = torch.optim.AdamW(model.local.parameters(), lr=1e-4, weight_decay=0.01)
    losses, gradients, seconds, shapes = [], {}, [], []
    model.train()
    for step, row in enumerate(examples.itertuples()):
        started = time.monotonic()
        d, t = np.array([row.drug_feature_index]), np.array([row.target_feature_index])
        g, b = globals.batch(d, t, 'global'), bank.batch(d, t)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores, aux = model(g, b, return_aux=True)
            loss = F.binary_cross_entropy_with_logits(scores.float(), torch.full_like(scores.float(), float(row.binary_label)))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if step == 0:
            for name in ['atom_input.1.weight', 'residue_input.1.weight', 'atom_bias.net.0.weight', 'residue_bias.net.0.weight',
                         'blocks.0.outer_out.weight', 'blocks.5.pair_bias.weight']:
                grad = dict(model.local.named_parameters())[name].grad
                if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                    raise ValueError(f'missing/invalid first-step gradient: {name}')
                gradients[name] = float(grad.norm())
        torch.nn.utils.clip_grad_norm_(model.local.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        losses.append(float(loss)); seconds.append(time.monotonic()-started)
        shapes.append(dict(pockets=len(b['owner']), atoms=b['atom_mask'].shape[1], residues=b['residue_mask'].shape[1]))
        print(json.dumps(dict(step=step+1, loss=losses[-1], seconds=round(seconds[-1], 3), shape=shapes[-1])), flush=True)
    model.eval()
    with torch.no_grad():
        expected = model.parent(g)
        manual = model.parent.readout(model.parent.shared(model.parent_state(g)))
        torch.testing.assert_close(expected, manual, atol=0, rtol=0)
        missing = {k: v.clone() for k, v in b.items()}
        missing['residue_mask'].zero_()
        torch.testing.assert_close(model(g, missing), expected, atol=0, rtol=0)
        scores = model(g, b)
        altered = {k: v.clone() for k, v in b.items()}
        altered['pocket_residue_tokens'].zero_(); altered['residue_geometry'].zero_(); altered['residue_distance'].zero_()
        geometric_change = float((model(g, altered) - scores).abs().max())
        # Restore full trained architecture including parent and verify FP32 reload.
        out = OUTPUT.parent / 'engineering'; out.mkdir(exist_ok=True)
        path = out / 'ENGINEERING_ONLY.pt'
        torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), status='ENGINEERING_ONLY_NOT_SELECTED', parent=source['checkpoint']), path)
        restored = ProtectedPocketRanker(ckpt, cfg).cuda().eval()
        restored.load_state_dict(torch.load(path, weights_only=False)['model'], strict=True)
        torch.testing.assert_close(restored(g, b), scores, atol=0, rtol=0)
    parent_after = state_hash(model.parent)
    assert parent_before == parent_after
    result = dict(status='PASS', purpose='full-size real-data optimizer/invariance/reload integration; NOT model-quality evaluation',
                  config=file_identity(config_path), model_source=file_identity(ROOT / 'biomaster/pocket_precision.py'),
                  feature_manifest=file_identity(OUTPUT / 'MANIFEST.json'), parent=source['checkpoint'],
                  local_parameters=sum(p.numel() for p in model.local.parameters()),
                  frozen_global_parameters=sum(p.numel() for p in model.parent.parameters()), steps=len(losses),
                  measured_training_year_max=2020, losses=losses, first_step_gradient_norms=gradients,
                  median_step_seconds=float(np.median(seconds)), batch_shapes=shapes,
                  peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                  frozen_parent_unchanged=True, parent_exact_forward=True, missing_pocket_exact_parent=True,
                  full_checkpoint_exact_reload=True, last_example_geometry_and_pocket_token_score_change=geometric_change,
                  experimental_contact_training_performed=False, ranking_quality_measured=False)
    write_json(out / 'RESULT.json', result)
    bank.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

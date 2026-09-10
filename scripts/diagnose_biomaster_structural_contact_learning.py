#!/usr/bin/env python3
"""Read-only structural contact diagnostics, separate from checkpoint selection.

Reports contacts inside the native pocket separately from native/remote-region
classification. Atom-permuted-label controls ask whether predictions distinguish
atoms within the same molecule. All controls use the same frozen predictions;
the constant score's AP is the positive prevalence. These are structural metrics,
not old-drug/target retrieval AP.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig
from biomaster.structural_training import StructuralDataset
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
DATA = OUT / 'structural_data/training_2020'


def contact_metrics(truth, score, seed):
    """Each row is a ligand atom and each column a native-pocket residue."""
    positive = np.asarray(truth) < 4.5
    score = np.asarray(score, np.float64)
    prevalence = float(positive.mean())
    ap = float(average_precision_score(positive.ravel(), score.ravel()))
    residue_only = np.broadcast_to(score.mean(0, keepdims=True), score.shape)
    residue_ap = float(average_precision_score(positive.ravel(), residue_only.ravel()))
    rng = np.random.default_rng(seed)
    permutations = []
    for _ in range(5):
        perm = rng.permutation(len(score))
        permutations.append(float(average_precision_score(positive.ravel(), score[perm].ravel())))
    atom_queries, residue_queries = [], []
    for y, s in zip(positive, score):
        if y.any() and not y.all():
            atom_queries.append(float(average_precision_score(y, s)))
    for y, s in zip(positive.T, score.T):
        if y.any() and not y.all():
            residue_queries.append(float(average_precision_score(y, s)))
    return dict(native_contact_ap=ap, native_contact_prevalence=prevalence,
                native_residue_only_score_ap=residue_ap,
                native_atom_permuted_score_ap=float(np.mean(permutations)),
                native_atom_to_residue_macro_ap=float(np.mean(atom_queries)) if atom_queries else None,
                native_residue_to_atom_macro_ap=float(np.mean(residue_queries)) if residue_queries else None,
                native_contact_pairs=int(positive.sum()), native_pair_candidates=int(positive.size))


def paired_interval(frame, first, second, cluster_column='cluster'):
    delta = frame[first] - frame[second]
    rng = np.random.default_rng(20260907)
    clusters = frame.assign(delta=delta).groupby(cluster_column).delta.agg(['sum', 'count'])
    if not len(clusters):
        raise ValueError('empty diagnostic cluster set')
    selected = rng.integers(0, len(clusters), size=(5000, len(clusters)))
    values = clusters['sum'].to_numpy()[selected].sum(1) / clusters['count'].to_numpy()[selected].sum(1)
    return dict(mean_difference=float(delta.mean()), ci95=np.quantile(values, [0.025, 0.975]).tolist(),
                clusters=len(clusters), resampling_unit='whole internal structural cluster', iterations=5000)


@torch.inference_mode()
def evaluate(model, dataset):
    rows = []
    cfg = model.cfg
    centers = (torch.arange(cfg.distance_bins, device='cuda') + 0.5) * cfg.distance_max / (cfg.distance_bins - 1)
    centers[-1] = cfg.distance_max
    for index, record in enumerate(dataset.records):
        batch, labels = dataset.batch([index])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, aux = model(batch, torch.zeros(1, cfg.parent_width, device='cuda'), return_aux=True)
        native = int(labels['native'][0])
        na = int(batch['atom_mask'][native].sum()); nr = int(batch['residue_mask'][native].sum())
        truth = labels['distance'][native, :na, :nr].cpu().numpy()
        scores = aux['contact_logits'][native, :na, :nr].float().cpu().numpy()
        seed = int(hashlib.sha256(record['system_id'].encode()).hexdigest()[:8], 16)
        row = dict(system_id=record['system_id'], cluster=record['cluster'], **contact_metrics(truth, scores, seed))
        mask = aux['pair_mask']
        all_truth = labels['distance'][mask].cpu().numpy() < 4.5
        all_score = aux['contact_logits'][mask].float().cpu().numpy()
        row['all_regions_contact_ap'] = float(average_precision_score(all_truth, all_score))
        row['all_regions_contact_prevalence'] = float(all_truth.mean())
        logits = aux['distance_logits'][native, :na, :nr].float()
        prediction = (logits.softmax(-1) * centers).sum(-1).cpu().numpy()
        row['native_distance_mae_capped32_A'] = float(np.abs(prediction - truth.clip(max=cfg.distance_max)).mean())
        row['native_region_selected'] = bool(int(aux['pocket_gate'].argmax()) == native)
        row['regions'] = len(batch['owner'])
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(evaluated=index+1, total=len(dataset))), flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output / 'RESULT.json').exists():
        raise ValueError('diagnostic output already completed; use a new output directory')
    manifest = json.loads((DATA / 'MANIFEST.json').read_text())
    if manifest['status'] != 'STRUCTURAL_DATA_READY':
        raise ValueError('admitted structural data required')
    before = args.checkpoint.stat()
    raw = args.checkpoint.read_bytes()
    after = args.checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('checkpoint changed during snapshot; retry after the save completes')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if payload['identity']['data'] != file_identity(DATA / 'MANIFEST.json'):
        raise ValueError('checkpoint/data mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep a local immutable snapshot of exactly the weights evaluated.
    checkpoint_hash = hashlib.sha256(raw).hexdigest()
    cfg = PocketPrecisionConfig(**payload['config'])
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    model = PocketPrecision(cfg).cuda().eval(); model.load_state_dict(payload['local_model'], strict=True)
    torch.save(dict(local_model=payload['local_model'], config=payload['config'],
                    source_checkpoint_sha256=checkpoint_hash, source_identity=payload['identity']), args.output / 'SNAPSHOT.pt')
    records = [r for r in manifest['records'] if r['split'] == 'validation']
    dataset = StructuralDataset(DATA, records)
    started = time.monotonic()
    frame = evaluate(model, dataset)
    frame.to_csv(args.output / 'COMPLEX_METRICS.csv', index=False)
    means = frame.select_dtypes(include=['number', 'bool']).mean().to_dict()
    result = dict(status='COMPLETE_READ_ONLY_DIAGNOSTIC', scope='cluster-held-out experimental structural contacts; NOT old-drug ranking',
                  source_checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                  checkpoint_status=payload.get('status'), progress=payload.get('progress', {}).get('updates'),
                  data=file_identity(DATA / 'MANIFEST.json'), script=file_identity(Path(__file__)),
                  complexes=len(frame), means=means, seconds=time.monotonic()-started,
                  native_vs_atom_permuted=paired_interval(frame, 'native_contact_ap', 'native_atom_permuted_score_ap'),
                  native_vs_residue_only=paired_interval(frame, 'native_contact_ap', 'native_residue_only_score_ap'),
                  native_vs_constant=paired_interval(frame, 'native_contact_ap', 'native_contact_prevalence'),
                  used_for_checkpoint_selection=False, fitted_or_modified_model=False)
    write_json(args.output / 'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

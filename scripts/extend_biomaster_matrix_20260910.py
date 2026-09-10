#!/usr/bin/env python3
"""Versioned 720 x 890 inference; preserve the released model and wet-lab list."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.portable_ranker_v2 import CatalogRanker, digest
from scripts.build_biomaster_odti_target_token_features_v1 import window_bounds

BUNDLE = ROOT / 'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'
OUT = ROOT / 'outputs/biomaster_matrix_720x890_20260910'
BASE = ROOT / 'outputs/biomaster_unified_interaction_20260906/features'
QUERY = ROOT / 'outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf'
REGISTRY = ROOT / 'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv'
FROZEN = ROOT / 'outputs/spr384_final_experiment_table_20260910'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    OUT.mkdir(exist_ok=True)
    started = time.monotonic()
    identities = {str(p.relative_to(ROOT)): digest(p) for p in
                  [BUNDLE/'model.pt', BUNDLE/'MANIFEST.json', REGISTRY,
                   FROZEN/'SPR384_FINAL_EXPERIMENT_TABLE.csv', FROZEN/'SPR384_FINAL_DETAILED.csv']}
    frozen = OUT/'FROZEN_INPUTS.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identities:
        raise ValueError('frozen input changed')
    write_json(frozen, identities)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ranker = CatalogRanker(BUNDLE, device='cpu', verify=True)
    targets = pd.read_csv(REGISTRY)
    targets['catalog_scope'] = 'official_888'
    extras = []
    for item in json.loads((QUERY/'TARGETS.json').read_text()):
        if item['uniprot_accession'] in set(targets.uniprot_accession):
            continue
        extras.append(dict(item, target_chembl_id='CHEMBL4076' if item['gene_symbol']=='SLC8A1' else '',
                           target_name=item['protein_name'], sequence_length=len(item['sequence']),
                           catalog_scope='requested_extension'))
    targets = pd.concat([targets, pd.DataFrame(extras)], ignore_index=True).fillna('')
    assert len(targets) == 890 and targets.uniprot_accession.nunique() == 890
    for row in targets.itertuples():
        import hashlib
        assert hashlib.sha256(row.sequence.encode()).hexdigest() == row.sequence_sha256
    targets.insert(0, 'matrix_target_index', np.arange(len(targets)))
    targets['core_model_target'] = targets.uniprot_accession.isin(ranker.targets.target_id)
    targets.to_csv(OUT/'TARGET_INDEX.csv.gz', index=False)
    ranker.drugs.to_csv(OUT/'DRUG_INDEX.csv', index=False)
    sources = {}
    asset_dir = ROOT/'outputs/biomaster_bindingdb_target_token_feature_package_v1'
    raw = np.load(asset_dir/'ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy', mmap_mode='r')
    assets = pd.read_csv(ROOT/'outputs/biomaster_odti_v4_plan_20260905/CURRENT_TARGET_ASSETS_V4.csv.gz')
    for r in assets.itertuples():
        if r.full_residue_available and int(r.token_length) == len(r.protein_sequence):
            sources[r.sequence_sha256] = (raw, int(r.token_offset), int(r.token_length), 'full_residue_bank')
    added = ROOT/'outputs/biomaster_odti_v4_20260905/features'
    array = np.load(added/'ESM2_ADDED_RESIDUES_FLOAT16_V4.npy', mmap_mode='r')
    for r in pd.read_csv(added/'ESM2_ADDED_RESIDUE_INDEX_V4.csv').itertuples():
        sources[r.sequence_sha256] = (array, int(r.token_offset), int(r.token_length), 'added_residue_bank')
    for p in (BASE/'missing_residues').glob('*.npy'):
        a = np.load(p, mmap_mode='r'); sources[p.stem] = (a, 0, len(a), 'core_residue_cache')
    for item in json.loads((QUERY/'TARGETS.json').read_text()):
        a = np.load(QUERY/(item['gene_symbol']+'_ESM2_RESIDUES.npy'), mmap_mode='r')
        sources[item['sequence_sha256']] = (a, 0, len(a), 'requested_target_cache')
    core = ranker.targets.set_index('target_id')
    cache = OUT/'target_means'; cache.mkdir(exist_ok=True)
    missing = [r for r in targets.itertuples() if r.sequence_sha256 not in sources
               and not r.core_model_target and not (cache/(r.sequence_sha256+'.npy')).exists()]
    write_json(OUT/'STATUS.json', dict(stage='features', missing=len(missing), total=890))
    print(json.dumps(dict(stage='features', new_encodings=len(missing))), flush=True)
    if missing:
        os.environ['TORCH_HOME'] = '/root/autodl-tmp/.cache/torch'
        import esm
        encoder, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        encoder = encoder.cuda().eval(); convert = alphabet.get_batch_converter()
        with torch.inference_mode():
            for i, r in enumerate(missing):
                total = np.zeros((len(r.sequence), 1280), np.float32)
                counts = np.zeros((len(r.sequence), 1), np.float32)
                for lo, hi in window_bounds(len(r.sequence), 1022, 128):
                    _, _, token = convert([(r.gene_symbol, r.sequence[lo:hi])])
                    with torch.autocast('cuda', dtype=torch.float16):
                        h = encoder(token.cuda(), repr_layers=[33], return_contacts=False)['representations'][33]
                    total[lo:hi] += h[0, 1:hi-lo+1].float().cpu().numpy(); counts[lo:hi] += 1
                assert (counts > 0).all()
                # Match the selected model: round per-residue features to FP16 before the FP32 mean.
                mean = (total/counts).astype(np.float16).astype(np.float32).mean(0)
                assert np.isfinite(mean).all()
                path = cache/(r.sequence_sha256+'.npy')
                with path.with_suffix('.tmp').open('wb') as f: np.save(f, mean)
                path.with_suffix('.tmp').replace(path)
                status = dict(stage='features', completed=i+1, required=len(missing), gene=r.gene_symbol,
                              seconds=round(time.monotonic()-started, 1))
                write_json(OUT/'STATUS.json', status)
                if (i+1)%20 == 0 or i+1 == len(missing): print(json.dumps(status), flush=True)
        del encoder; torch.cuda.empty_cache()
    features = []; audit = []; compatibility = []
    for r in targets.itertuples():
        mean = None
        if r.sequence_sha256 in sources:
            a, offset, length, route = sources[r.sequence_sha256]
            assert length == len(r.sequence)
            mean = a[offset:offset+length].astype(np.float32).mean(0)
        if r.core_model_target:
            i = ranker._targets[r.uniprot_accession]
            reference = ranker.features['target_global'][i].numpy()
            if mean is not None: compatibility.append(float(np.max(np.abs(mean-reference))))
            mean = reference; route = 'frozen_bundle_exact'
        elif mean is None:
            mean = np.load(cache/(r.sequence_sha256+'.npy')); route = 'new_full_chain_esm2'
        assert mean.shape == (1280,) and np.isfinite(mean).all()
        features.append(mean)
        audit.append(dict(matrix_target_index=r.matrix_target_index, gene_symbol=r.gene_symbol,
                          sequence_sha256=r.sequence_sha256, sequence_length=len(r.sequence), feature_route=route))
    assert max(compatibility, default=0) < 2e-3
    features = np.stack(features); np.save(OUT/'TARGET_GLOBAL.npy', features)
    pd.DataFrame(audit).to_csv(OUT/'FEATURE_AUDIT.csv', index=False)
    ranker.model.cuda(); drug_features = {k: v.cuda() for k,v in ranker.features.items() if k != 'target_global'}
    score = np.empty((720, 890, 2), np.float32)
    with torch.inference_mode():
        for i, mean in enumerate(features):
            batch = dict(drug_features, target_global=torch.tensor(mean, device='cuda')[None].expand(720,-1))
            score[:, i] = ranker.model(batch).cpu().numpy()
    assert np.isfinite(score).all()
    # Verify all core pairs against the immutable catalog interface (different batch/device).
    ranker.model.cpu(); core_ids = np.flatnonzero(targets.core_model_target.to_numpy())
    maximum = 0.0
    with torch.inference_mode():
        for i in core_ids:
            d = np.arange(720); t = np.full(720, ranker._targets[targets.uniprot_accession.iloc[i]])
            reference = ranker.model(ranker._batch(d,t)).numpy()
            maximum = max(maximum, float(np.max(np.abs(reference-score[:,i]))))
    assert maximum < 2e-4, maximum
    np.save(OUT/'DIRECTIONAL_LOGITS.npy', score)
    frames = []
    drug_rank = pd.DataFrame(-score[:,:,1]).rank(axis=0, method='min').to_numpy(int)
    target_rank = pd.DataFrame(-score[:,:,0]).rank(axis=1, method='min').to_numpy(int)
    for i, r in enumerate(targets.itertuples()):
        f = ranker.drugs[['drug_id','name']].copy()
        f['target_id'] = r.uniprot_accession; f['target_chembl_id'] = r.target_chembl_id; f['gene_symbol'] = r.gene_symbol
        f['drug_to_target_logit'] = score[:,i,0]; f['target_to_drug_logit'] = score[:,i,1]
        f['drug_rank_within_target_720'] = drug_rank[:,i]; f['target_rank_within_drug_890'] = target_rank[:,i]
        f['core_model_target'] = r.core_model_target
        f['evidence_status'] = 'MODEL_PREDICTION_NOT_EXPERIMENTAL_BINDING'
        frames.append(f)
    full = pd.concat(frames, ignore_index=True)
    assert len(full)==640800 and not full.duplicated(['drug_id','target_id']).any()
    full.to_csv(OUT/'ALL_640800_PREDICTIONS.csv.gz', index=False)
    full[full.drug_rank_within_target_720<=20].sort_values(['gene_symbol','drug_rank_within_target_720']).to_csv(OUT/'TARGET_TOP20.csv.gz', index=False)
    for p,h in identities.items(): assert digest(ROOT/p)==h, p
    summary = dict(status='COMPLETE', drugs=720, original_targets=888, requested_extra_targets=['LYVE1','SLC8A1'],
                   targets=890, pairs=len(full), original_core_targets=384, extended_targets=506,
                   additional_pairs=720*506, model_sha256=digest(BUNDLE/'model.pt'),
                   core_feature_max_abs_difference=max(compatibility, default=0),
                   core_score_max_abs_difference=maximum, frozen_inputs_unchanged=True,
                   feature_routes=pd.Series([x['feature_route'] for x in audit]).value_counts().to_dict(),
                   feature_protocol='ESM2 t33 650M layer33; 1022/128 windows; FP16 residue rounding; FP32 full-chain mean',
                   scores='Two directional logits, neither probabilities nor Kd; ranks use 720 drugs / 890 targets',
                   extended_target_accuracy_validated=False, production_bundle_modified=False,
                   seconds=round(time.monotonic()-started,1), producer_sha256=digest(Path(__file__)))
    write_json(OUT/'SUMMARY.json', summary); write_json(OUT/'STATUS.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()

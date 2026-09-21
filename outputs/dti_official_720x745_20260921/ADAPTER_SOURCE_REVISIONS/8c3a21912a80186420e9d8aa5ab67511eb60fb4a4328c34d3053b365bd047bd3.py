#!/usr/bin/env python3
"""SCOPE Total five-checkpoint ensemble on every sequence accepted by its encoder."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import time
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, dump, directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    start = time.time()
    name = 'ScopeDTI_light'
    status(name, 'PREPARING_NATIVE_CONFORMERS')
    torch.set_num_threads(3)
    torch.manual_seed(20260813)
    RDLogger.DisableLog('rdApp.*')
    sys.path.insert(0, str(ROOT / '.external/scope_dti_lightweight'))
    from run_scope_dti_old_drug_entity_cold_v1 import (deterministic_sdf, sdf_to_graphs, integer_label_protein,
         ScopePairDataset, custom_collate_fn, ScopeModel, get_cfg_defaults, CHECKPOINT_DIR)
    drugs, targets = inputs()
    folder = directory(name)
    graph_path = folder / 'DRUG_GRAPHS.pt'
    if graph_path.exists():
        graphs = torch.load(graph_path, weights_only=False, map_location='cpu')
    else:
        def build(drug):
            try:
                sdf, route = deterministic_sdf(drug.smiles)
                return drug.drug_id, sdf_to_graphs(sdf) if sdf else None, route
            except Exception as exc:
                return drug.drug_id, None, f'{type(exc).__name__}: {exc}'
        with ThreadPoolExecutor(max_workers=4) as pool:
            built = list(pool.map(build, drugs.itertuples()))
        graphs = {key: graph for key, graph, _ in built if graph is not None}
        dump(folder / 'CONFORMER_AUDIT.json', [dict(drug_id=key, available=graph is not None, route=route) for key, graph, route in built])
        torch.save(graphs, graph_path)
    protein = {}
    failures = []
    for row in targets.itertuples():
        try:
            protein[row.target_id] = torch.tensor(integer_label_protein(row.sequence), dtype=torch.float32)
        except Exception as exc:
            failures.append(dict(target_id=row.target_id, reason=str(exc)))
    vocab = pd.read_parquet(ROOT / '.external/scope_dti_lightweight/protein_targets/Total_predict.parquet')
    vocab_seqs = set(vocab.sequence.astype(str))
    targets[['target_id', 'sequence']].assign(in_author_cli_vocabulary=targets.sequence.isin(vocab_seqs)).drop(columns='sequence').to_csv(folder/'AUTHOR_VOCABULARY_COVERAGE.csv', index=False)
    frame = drugs.loc[drugs.drug_id.isin(graphs), ['drug_id']].merge(targets.loc[targets.target_id.isin(protein), ['target_id']], how='cross')
    frame['model_ligand_smiles'] = frame.drug_id  # dataset key only; graphs built from exact SMILES above
    frame['query_accession'] = frame.target_id
    frame['binary_label'] = 0.0  # ignored by inference forward; no labels are supplied to the model
    checkpoints = sorted(CHECKPOINT_DIR.glob('*.pth'))
    assert len(checkpoints) == 5
    contract(name, checkpoints={p.name: sha(p) for p in checkpoints}, adapter_sha256=sha(__file__),
             head='Mean sigmoid probability of five official Total checkpoints', precision='float32_cuda',
             preprocessing='Native 3D conformer graph and integer_label_protein(first1200); arbitrary sequence extension beyond author GUI catalogue disclosed',
             native_source='run_scope_dti_old_drug_entity_cold_v1.py',
             author_vocabulary_members=int(targets.sequence.isin(vocab_seqs).sum()))
    dump(folder/'INPUT_FAILURES.json', failures)
    if args.smoke:
        frame = frame.iloc[:64].copy()
        checkpoints = checkpoints[:1]
    loader = DataLoader(ScopePairDataset(frame, graphs, protein), batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    arrays = []
    for ci, ckpt in enumerate(checkpoints):
        path = folder / f'CHECKPOINT_{ci+1}_SCORES.npy'
        if path.exists() and not args.smoke:
            array = np.load(path)
            assert len(array) == len(frame)
            arrays.append(array)
            continue
        model = ScopeModel(**get_cfg_defaults())
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
        model.cuda().eval()
        values = []
        with torch.inference_mode():
            for bi, (drug, seq, _) in enumerate(loader):
                _, _, logits, _ = model(drug.cuda(), seq.cuda(), mode='eval')
                values.append(logits.sigmoid().squeeze(1).cpu().numpy())
                if bi % 100 == 0:
                    status(name, 'INFERENCE', checkpoint=ci+1, checkpoint_scored_pairs=sum(len(x) for x in values),
                           per_checkpoint_pairs=len(frame), elapsed_seconds=time.time()-start)
        array = np.concatenate(values)
        assert np.isfinite(array).all()
        arrays.append(array)
        if not args.smoke:
            np.save(path, array)
        del model
        torch.cuda.empty_cache()
    if args.smoke:
        dump(folder/'SMOKE.json', dict(status='PASS', pairs=len(frame), score_min=float(arrays[0].min()), score_max=float(arrays[0].max())))
        status(name, 'SMOKE_PASS', elapsed_seconds=time.time()-start)
        return
    frame['score'] = np.stack(arrays).mean(0)
    save_scores(name, frame[['drug_id','target_id','score']])
    status(name, 'COMPLETE' if len(frame)==536400 else 'COMPLETE_WITH_MISSING_INPUTS', scored_pairs=len(frame), elapsed_seconds=time.time()-start)


if __name__ == '__main__':
    main()

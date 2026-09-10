#!/usr/bin/env python3
"""Read-only matched-batch parent evaluation for the first downstream report."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.best_model_training import RollingStage
from biomaster.molecular_controls import MolecularControlBank
from biomaster.odti_pockets_v3 import file_identity
from biomaster.pocket_precision_training import ProtectedPocketRanker
from biomaster.ranking_audit import risk_set_ranking
from prepare_biomaster_pocket_precision import BASE, SUPPLEMENT, SOURCE

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
DEST = OUT / 'downstream_diagnostics/parent_precision_20260907'


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    selection = json.loads((ROOT / 'outputs/biomaster_best_model_20260906/GLOBAL_PARENT_SELECTION.json').read_text())
    parent = next(r for r in selection['parents'] if r['cutoff'] == 2020 and r['seed'] == 20260921)
    for key in ['checkpoint', 'result']:
        if file_identity(parent[key]['path']) != parent[key]:
            raise ValueError('frozen parent identity changed')
    model = ProtectedPocketRanker(torch.load(parent['checkpoint']['path'], map_location='cpu', weights_only=False)).cuda().eval()
    bank = MolecularControlBank(BASE, SUPPLEMENT, SOURCE, 'drugclip_morgan')
    data = ROOT / 'outputs/biomaster_best_model_20260906/data'
    stage = RollingStage(data, SOURCE, 2020)
    results, query_frames = {}, []
    for head, direction in enumerate(['d2t', 't2d']):
        known = stage.val_known if head == 0 else stage.val_known.T
        risk = stage.risk if head == 0 else stage.risk.T
        ids = np.flatnonzero(known.any(1)); candidates = known.shape[1]
        if head == 0:
            d = np.repeat(stage.old.drug_feature_index.to_numpy()[ids], candidates)
            t = np.tile(np.arange(candidates), len(ids))
        else:
            d = np.tile(stage.old.drug_feature_index.to_numpy(), len(ids))
            t = np.repeat(ids, candidates)
        scores = np.empty(len(d), np.float32)
        for i in range(len(d)):
            batch = bank.batch(d[i:i+1], t[i:i+1], 'global')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                g = model.parent_state(batch)
                score = model.parent.readout(model.parent.shared(g))
            scores[i] = score[0, head].float().item()
            if (i+1) % 10000 == 0:
                print(json.dumps(dict(direction=direction, pairs=i+1, seconds=time.monotonic()-started)), flush=True)
        metrics, queries, _ = risk_set_ranking(known[ids], scores.reshape(len(ids), candidates), risk[ids], query_ids=ids, candidate_ids=np.arange(candidates))
        results[direction] = metrics
        queries['direction'] = direction; query_frames.append(queries)
        print(json.dumps(dict(direction=direction, metrics=metrics)), flush=True)
    history_path = OUT / 'training/structurally_pretrained/cutoff_2020/seed_20260921/HISTORY.json'
    first = json.loads(history_path.read_text())[0]
    differences = {d:{key:first['metrics'][d][key]-results[d][key] for key in ['macro_ap', 'macro_recall_20']} for d in ['d2t', 't2d']}
    result = dict(status='COMPLETE_READ_ONLY_PARENT_PRECISION_AUDIT', utc=datetime.now(timezone.utc).isoformat(),
                  scope='cutoff 2020, validation 2021-2022, seed 20260921; first downstream epoch only',
                  parent=parent, evaluation_precision='BF16 autocast, microbatch=1, TF32 disabled; exact frozen-parent path of current ranker',
                  parent_metrics=results, downstream_first_epoch=first, downstream_minus_matched_parent=differences,
                  parent_composite=sum(.35*results[d]['macro_ap']+.15*results[d]['macro_recall_20'] for d in results),
                  validation_identity=file_identity(data / 'roll_2020/VALIDATION.csv.gz'), risk_identity=file_identity(data / 'roll_2020/RISK.npy'),
                  source_identity=file_identity(Path(__file__)), seconds=time.monotonic()-started,
                  used_for_selection=False, training_modified=False, new_model_rescored=False,
                  uncertainty='No paired query confidence interval: current trainer stores aggregate validation only.')
    DEST.mkdir(parents=True, exist_ok=True)
    pd.concat(query_frames, ignore_index=True).to_csv(DEST / 'PARENT_QUERY_METRICS.csv', index=False)
    (DEST / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

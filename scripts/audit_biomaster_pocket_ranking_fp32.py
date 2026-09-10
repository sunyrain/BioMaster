#!/usr/bin/env python3
"""Freeze epoch-one weights and compare full candidate rankings in FP32."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
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
from biomaster.pocket_precision import PocketPrecisionConfig
from biomaster.pocket_precision_features import PocketPrecisionBank
from biomaster.pocket_precision_training import ProtectedPocketRanker
from biomaster.ranking_audit import query_bootstrap, risk_set_ranking
from prepare_biomaster_pocket_precision import BASE, SUPPLEMENT, SOURCE, OUTPUT

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
TRAIN = OUT / 'training/structurally_pretrained/cutoff_2020/seed_20260921'
DEST = OUT / 'downstream_diagnostics/epoch1_fp32_20260907'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


@torch.inference_mode()
def main():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = (DEST / '.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (DEST / 'RESULT.json').exists():
        return
    started = time.monotonic()
    def status(phase, **values):
        event = dict(status=phase, utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                     seconds=time.monotonic()-started, used_for_selection=False, **values)
        write(DEST / 'STATUS.json', event); print(json.dumps(event), flush=True)
    try:
        status('FREEZING_FIRST_EPOCH_CHECKPOINT')
        checkpoint = TRAIN / 'BEST.pt'; snapshot = DEST / 'EPOCH1_SNAPSHOT.pt'
        if not snapshot.exists():
            before = file_identity(checkpoint)
            tmp = snapshot.with_suffix('.tmp'); shutil.copyfile(checkpoint, tmp)
            if file_identity(checkpoint) != before or file_identity(tmp)['sha256'] != before['sha256']:
                tmp.unlink(); raise ValueError('checkpoint changed during snapshot')
            tmp.replace(snapshot)
        saved = torch.load(snapshot, map_location='cpu', weights_only=False)
        first = json.loads((TRAIN / 'HISTORY.json').read_text())[0]
        if saved['metrics'] != first['metrics'] or first['epoch'] != 1:
            raise ValueError('snapshot is not the first-epoch validation checkpoint')
        identity = saved['identity']
        for key in ['parent', 'validation', 'risk', 'features']:
            if file_identity(identity[key]['path']) != identity[key]:
                raise ValueError(f'{key} identity changed')
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        parent = torch.load(identity['parent']['path'], map_location='cpu', weights_only=False)
        model = ProtectedPocketRanker(parent, PocketPrecisionConfig(**saved['config']))
        model.load_state_dict(saved['model'], strict=True)
        if any(not torch.equal(model.parent.state_dict()[k], v) for k,v in parent['model'].items()):
            raise ValueError('embedded parent differs from frozen source')
        model.float().cuda().eval()
        bank = MolecularControlBank(BASE, SUPPLEMENT, SOURCE, 'drugclip_morgan')
        pockets = PocketPrecisionBank(OUTPUT, BASE, SUPPLEMENT)
        stage = RollingStage(ROOT / 'outputs/biomaster_best_model_20260906/data', SOURCE, 2020)
        results, contrasts = {}, {}
        total_pairs = int(stage.val_known.any(1).sum()*stage.val_known.shape[1] + stage.val_known.any(0).sum()*stage.val_known.shape[0])
        completed = 0
        for head, direction in enumerate(['d2t', 't2d']):
            known = stage.val_known if head == 0 else stage.val_known.T
            risk = stage.risk if head == 0 else stage.risk.T
            ids = np.flatnonzero(known.any(1)); candidates = known.shape[1]
            if head == 0:
                drugs = np.repeat(stage.old.drug_feature_index.to_numpy()[ids], candidates)
                targets = np.tile(np.arange(candidates), len(ids))
            else:
                drugs = np.tile(stage.old.drug_feature_index.to_numpy(), len(ids))
                targets = np.repeat(ids, candidates)
            scores = {key:np.empty(len(drugs), np.float32) for key in ['parent', 'new']}
            for i in range(len(drugs)):
                d, t = drugs[i:i+1], targets[i:i+1]
                global_batch, pocket_batch = bank.batch(d, t, 'global'), pockets.batch(d, t)
                g = model.parent_state(global_batch)
                ps = model.parent.readout(model.parent.shared(g))
                state = model.local(pocket_batch, g)
                ns = model.parent.readout(model.parent.shared(state))
                scores['parent'][i] = ps[0, head].item(); scores['new'][i] = ns[0, head].item()
                completed += 1
                if completed % 256 == 0:
                    status('FP32_RANKING_RUNNING', direction=direction, completed_pairs=completed, total_pairs=total_pairs)
            frames = {}; results[direction] = {}
            for key, values in scores.items():
                metrics, queries, pairs = risk_set_ranking(known[ids], values.reshape(len(ids), candidates), risk[ids], query_ids=ids, candidate_ids=np.arange(candidates))
                results[direction][key] = metrics; frames[key] = queries
                queries.to_csv(DEST / f'{direction}_{key}_QUERY_METRICS.csv', index=False)
                pairs.to_csv(DEST / f'{direction}_{key}_POSITIVE_RANKS.csv', index=False)
            contrasts[direction] = query_bootstrap(frames['new'], frames['parent'], iterations=10000)
            np.savez_compressed(DEST / f'{direction}_SCORES.npz', parent=scores['parent'].reshape(len(ids), candidates),
                                new=scores['new'].reshape(len(ids), candidates), labels=known[ids], risk=risk[ids],
                                query_ids=ids, candidate_ids=np.arange(candidates))
            status('DIRECTION_COMPLETE', direction=direction, metrics=results[direction], contrast=contrasts[direction],
                   completed_pairs=completed, total_pairs=total_pairs)
        pockets.close()
        result = dict(status='COMPLETE_READ_ONLY_FP32_RANKING_AUDIT', utc=datetime.now(timezone.utc).isoformat(),
                      snapshot=file_identity(snapshot), parent=identity['parent'], validation=identity['validation'], risk=identity['risk'],
                      source=file_identity(Path(__file__)), evaluation='FP32 weights/inputs/operations, TF32 disabled, microbatch=1, fixed epoch 1',
                      scope='cutoff 2020, validation 2021-2022, seed 20260921; all risk-set candidates, unknown background not inactivity',
                      metrics=results, paired_query_ap_intervals=contrasts, seconds=time.monotonic()-started,
                      used_for_selection=False, promotion_eligible=False, training_modified=False,
                      uncertainty='Query bootstrap on one development window and seed; not multi-seed or external confirmation.')
        write(DEST / 'RESULT.json', result)
        status('COMPLETE_READ_ONLY_FP32_RANKING_AUDIT', result=str(DEST / 'RESULT.json'), completed_pairs=completed, total_pairs=total_pairs)
    except Exception as error:
        status('FAILED_REQUIRES_REVIEW', error=repr(error))
        raise


if __name__ == '__main__':
    main()

"""Shared, resumable output contract for the independent 720 x 745 study."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_official_720x745_20260921'
INPUTS = ROOT / 'data/research/dti_ranking_scope_20260921_v4'
DRUG_FILE = ROOT / 'outputs/dti_ranking_720x745_20260921/DRUGS_720.csv'
TARGET_FILE = INPUTS / 'TARGET_INPUTS_745.parquet'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inputs():
    drugs = pd.read_csv(DRUG_FILE)
    targets = pd.read_parquet(TARGET_FILE)
    assert len(drugs) == drugs.drug_id.nunique() == 720
    assert len(targets) == targets.target_id.nunique() == 745
    assert targets.sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest()).eq(targets.protein_sha256).all()
    return drugs, targets


def directory(model):
    path = OUT / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def status(model, state, **kwargs):
    dump(directory(model) / 'STATUS.json', dict(model=model, state=state, pid=os.getpid(),
         updated_utc=datetime.now(timezone.utc).isoformat(), requested_pairs=536400, **kwargs))


def contract(model, **kwargs):
    dump(directory(model) / 'CONTRACT.json', dict(model=model, drug_input_sha256=sha(DRUG_FILE),
         target_input_sha256=sha(TARGET_FILE), new_training=False, native_higher_is_better=True, **kwargs))


def historical(model):
    frame = pd.read_parquet(INPUTS / 'PARTIAL_720X745_SCORE_SNAPSHOT.parquet')
    if model not in frame:
        return pd.DataFrame(columns=['drug_id', 'target_id', 'score'])
    return frame[['drug_id', 'target_id', model]].dropna().rename(columns={model: 'score'})


def save_scores(model, frame, name='SCORES.parquet'):
    assert not frame[['drug_id', 'target_id']].duplicated().any()
    assert np.isfinite(frame.score).all()
    drugs, targets = inputs()
    assert set(frame.drug_id).issubset(drugs.drug_id) and set(frame.target_id).issubset(targets.target_id)
    path = directory(model) / name
    tmp = path.with_suffix('.tmp.parquet')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_scores(model):
    path = directory(model) / 'SCORES.parquet'
    return pd.read_parquet(path) if path.exists() else historical(model)

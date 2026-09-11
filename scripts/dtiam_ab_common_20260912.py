"""Shared paths and validation metrics for the frozen September DTIAM comparison."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_dtiam_ab_20260912'
SOURCE = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'
DATA = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
FEATURES = OUT / 'features'
ARMS = ['kdki_inactive', 'all_inactive']
SEEDS = [20260921, 20260922, 20260923]
FEATURE_COLUMNS = [f'bermol_{i:04d}' for i in range(768)] + [f'esm2_{i:04d}' for i in range(1280)]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def check_values(values):
    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
        raise ValueError('Missing, zero or nonfinite required feature vector')


def table(frame, bank, labels=False):
    x = np.empty((len(frame), 2048), np.float32)
    for start in range(0, len(frame), 16384):
        f = frame.iloc[start:start+16384]
        x[start:start+len(f), :768] = bank[0][f.drug_feature_index.to_numpy(np.int64)]
        x[start:start+len(f), 768:] = bank[1][f.target_feature_index.to_numpy(np.int64)]
    result = pd.DataFrame(x, columns=FEATURE_COLUMNS, copy=False)
    if labels:
        result['y'] = frame.binary_label.to_numpy(np.int8)
    return result


def banks():
    m = json.loads((FEATURES/'MANIFEST.json').read_text())
    if m['status'] != 'COMPLETE':
        raise ValueError('Feature extraction is incomplete')
    return tuple(np.load(FEATURES/n, mmap_mode='r') for n in ['BERMOL.npy', 'ESM2.npy'])


def probability_score(p):
    return logit(np.clip(np.asarray(p, float), 1e-7, 1-1e-7))


def calibrate(frame, scores):
    use = frame.panel.eq('AFFINITY_KD_KI').to_numpy()
    y = frame.loc[use, 'binary_label'].to_numpy(int)
    model = LogisticRegression(C=1e6, max_iter=1000, random_state=20260911).fit(scores[use,None], y)
    if model.coef_[0,0] <= 0:
        raise ValueError('Nonpositive validation calibration slope')
    p = model.predict_proba(scores[use,None])[:,1]
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    k = int(np.argmax(f1))
    return dict(slope=float(model.coef_[0,0]), intercept=float(model.intercept_[0]),
                threshold=float(thresholds[k]), validation_f1=float(f1[k]),
                fit_panel='VALIDATION_AFFINITY_KD_KI', input='clipped native probability logit')


def basic(y, scores, probabilities, threshold):
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    p = np.asarray(probabilities, float); pred = p >= threshold
    positive = int(y.sum()); negative = len(y)-positive
    tp = int((pred & (y==1)).sum()); fp = int((pred & (y==0)).sum())
    both = positive > 0 and negative > 0
    result = dict(pairs=len(y), positive=positive, negative=negative,
        prevalence=float(y.mean()), random_ap=float(y.mean()) if both else None,
        auroc=float(roc_auc_score(y, scores)) if both else None,
        ap=float(average_precision_score(y, scores)) if both else None,
        brier=float(brier_score_loss(y, p)), precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/positive if positive else None, false_positive_rate=fp/negative if negative else None,
        threshold=float(threshold), mean_probability=float(p.mean()))
    order = np.argsort(-scores, kind='stable')
    for fraction, name in [(.01,'top1pct'), (.05,'top5pct'), (.1,'top10pct')]:
        n = max(1, math.ceil(len(y)*fraction)); precision = float(y[order[:n]].mean())
        result.update({name+'_n':n, name+'_precision':precision,
                       name+'_enrichment':precision/y.mean() if positive else None})
    return result


def query_rows(frame, scores, column):
    y = frame.binary_label.to_numpy(int); rows = []
    for key, ids in frame.groupby(column, sort=True).indices.items():
        if len(ids) < 10 or not 0 < y[ids].sum() < len(ids):
            continue
        yy = y[ids]; s = np.asarray(scores)[ids]
        order = np.argsort(-s, kind='stable')
        r = dict(query=key, pairs=len(ids), positive=int(yy.sum()),
                 ap=float(average_precision_score(yy,s)), prevalence=float(yy.mean()))
        for k in [5,10,20]:
            r['p'+str(k)] = float(yy[order[:k]].mean()) if len(ids)>=k else None
        rows.append(r)
    return rows


def query_summary(frame, scores):
    result = {}
    for name, col in [('target','target_id'), ('drug','molecule_id')]:
        q = pd.DataFrame(query_rows(frame, scores, col))
        if q.empty:
            raise ValueError('No eligible query groups')
        result[name] = dict(queries=len(q), macro_ap=float(q.ap.mean()),
                            **{c:float(q[c].mean()) for c in ['p5','p10','p20']})
    result['selection_score'] = (result['target']['macro_ap']+result['drug']['macro_ap'])/2
    return result

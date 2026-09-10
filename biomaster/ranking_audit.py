"""Measured candidate ranking: explicit labels, separate query directions."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def measured_ranking(frame, scores, direction):
    scores=np.asarray(scores)
    if len(frame)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('one finite score is required for each measured pair')
    query,candidate=('drug_id','target_id') if direction=='d2t' else ('target_id','drug_id')
    if direction not in ['d2t','t2d']:
        raise ValueError('unknown direction')
    data=frame[[query,candidate,'label']].copy()
    if data.duplicated([query,candidate]).any() or not data.label.isin([0,1]).all():
        raise ValueError('unique measured binary pairs required')
    data['score']=scores
    rows=[]
    for key,part in data.groupby(query,sort=False):
        y=part.label.to_numpy(int)
        if np.unique(y).size!=2:continue
        s=part.score.to_numpy()
        ranked=y[np.lexsort((part[candidate].astype(str).to_numpy(),-s))]
        row={'query_id':str(key),'candidates':len(y),'positives':int(y.sum()),
             'ap':float(average_precision_score(y,s)),'auroc':float(roc_auc_score(y,s))}
        for k in [5,10,20]:
            weights=1/np.log2(np.arange(min(k,len(y)))+2)
            row[f'recall_{k}']=float(ranked[:k].sum()/y.sum())
            row[f'ndcg_{k}']=float((ranked[:k]*weights).sum()/weights[:min(k,int(y.sum()))].sum())
        rows.append(row)
    queries=pd.DataFrame(rows)
    result={'rows':len(data),'positive_rows':int(data.label.sum()),'all_queries':data[query].nunique(),
        'two_class_queries':len(rows),'candidate_scope':'only observed candidates, unknown pairs excluded'}
    for metric in ['ap','auroc','recall_5','recall_10','recall_20','ndcg_20']:
        result['macro_'+metric]=float(queries[metric].mean()) if len(queries) else None
    result['median_candidates']=float(queries.candidates.median()) if len(queries) else None
    return result,queries


def query_bootstrap(first,second,level=.95,iterations=10000):
    delta=first.set_index('query_id').ap-second.set_index('query_id').ap
    if delta.isna().any() or not len(delta):raise ValueError('identical nonempty query sets required')
    values=delta.to_numpy()
    rng=np.random.default_rng(20260906)
    samples=np.mean(values[rng.integers(len(values),size=(iterations,len(values)))],1)
    lo,hi=np.quantile(samples,[(1-level)/2,1-(1-level)/2])
    return {'queries':len(values),'delta_ap':float(values.mean()),'ci_low':float(lo),'ci_high':float(hi),'confidence':level}


def risk_set_ranking(labels, scores, risk, query_ids, candidate_ids):
    """Rank newly published positives among candidates unknown at the cutoff.

    Unknown candidates are retrieval background, not experimental negatives.
    Removing historical measurements must never silently remove a test positive.
    """
    labels, scores, risk = np.asarray(labels), np.asarray(scores), np.asarray(risk)
    query_ids, candidate_ids = np.asarray(query_ids), np.asarray(candidate_ids)
    shape = (len(query_ids), len(candidate_ids))
    if any(x.shape != shape for x in [labels, scores, risk]):
        raise ValueError('query/candidate axes must align')
    if not np.isin(labels, [0, 1]).all() or not np.isin(risk, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('binary labels/mask and finite scores required')
    if len(np.unique(query_ids)) != len(query_ids) or len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError('unique axis IDs required')
    labels, risk = labels.astype(bool), risk.astype(bool)
    if (labels & ~risk).any():
        raise ValueError('test positives cannot be removed by historical risk mask')
    rows, pairs = [], []
    for key, y, s, allowed in zip(query_ids, labels, scores, risk):
        if not y.any():
            continue
        ids = candidate_ids[allowed].astype(str)
        order = np.lexsort((ids, -s[allowed]))
        ranked = y[allowed][order]
        ranks = np.flatnonzero(ranked) + 1
        row = {'query_id': str(key), 'positives': len(ranks), 'candidates': len(order),
               'ap': float(np.mean(np.arange(1, len(ranks)+1)/ranks)), 'mrr': float(1/ranks[0])}
        for k in [5, 10, 20]:
            discount = 1/np.log2(np.arange(min(k, len(order)))+2)
            row[f'recall_{k}'] = float(ranked[:k].sum()/len(ranks))
            row[f'ndcg_{k}'] = float((ranked[:k]*discount).sum()/discount[:min(k,len(ranks))].sum())
        rows.append(row)
        for rank in ranks:
            pairs.append({'query_id': str(key), 'candidate_id': ids[order[rank-1]],
                          'rank': int(rank), 'candidates': len(order)})
    queries = pd.DataFrame(rows)
    result = {'all_queries': len(query_ids), 'positive_queries': len(rows), 'positive_pairs': int(labels.sum()),
              'candidate_scope': 'all registry candidates minus prior/undated measurements; unknown background, not inactive'}
    for metric in ['ap', 'mrr', 'recall_5', 'recall_10', 'recall_20', 'ndcg_20']:
        result['macro_'+metric] = float(queries[metric].mean()) if len(queries) else None
    for name, fun in [('min', np.min), ('median', np.median), ('max', np.max)]:
        result[name+'_candidates'] = float(fun(queries.candidates)) if len(queries) else None
    return result, queries, pd.DataFrame(pairs)

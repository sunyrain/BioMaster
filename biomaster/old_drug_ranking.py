"""Direction-explicit known-relationship recovery and observed-PN metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def known_ranking(labels, scores, query_ids, candidate_ids):
    """Score only queries with known positives; unknowns are retrieval background.

    AP uses deterministic rank ties (ascending candidate ID). No claim of
    complete biological ground truth or measured negatives is made here.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    query_ids, candidate_ids = np.asarray(query_ids), np.asarray(candidate_ids)
    if labels.shape != scores.shape or scores.shape != (len(query_ids),len(candidate_ids)):
        raise ValueError('query/candidate axes must match scores and labels')
    if not len(candidate_ids) or not np.isfinite(scores).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('finite scores, nonempty candidates and binary known labels required')
    if len(np.unique(query_ids)) != len(query_ids) or len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError('IDs must be unique within each axis')
    rows, pairs = [], []
    for query,y,s in zip(query_ids,labels.astype(bool),scores):
        if not y.any():
            continue
        order=np.lexsort((candidate_ids,-s))
        positions=np.flatnonzero(y[order])+1
        n=len(positions)
        row={'query_id':str(query),'known_positives':n,'candidate_count':len(s),
             'ap':float(np.mean(np.arange(1,n+1)/positions)),'mrr':float(1/positions[0]),
             'best_known_rank':int(positions[0])}
        for k in [1,5,10,20]:
            ranked=y[order][:k]
            discount=1/np.log2(np.arange(1,min(k,len(s))+1)+1)
            row[f'recall_at_{k}']=float(ranked.sum()/n)
            row[f'hit_at_{k}']=float(ranked.any())
            row[f'ndcg_at_{k}']=float((ranked*discount).sum()/discount[:min(n,k)].sum())
        rows.append(row)
        for rank in positions:
            index=order[rank-1]
            pairs.append({'query_id':str(query),'candidate_id':str(candidate_ids[index]),
                          'rank':int(rank),'candidate_count':len(s),'score':float(s[index])})
    frame=pd.DataFrame(rows)
    keys=['ap','mrr']+[f'{name}_at_{k}' for k in [1,5,10,20] for name in ['recall','hit','ndcg']]
    summary={'all_queries':len(query_ids),'evaluated_queries':len(rows),'queries_without_known_positive':len(query_ids)-len(rows),
             'candidates_per_query':len(candidate_ids),'known_relationships':int(labels.sum()),
             'metric_scope':'known-relationship recovery, unknown relationships are not measured negatives',
             'tie_break':'descending score, ascending candidate ID'}
    summary.update({f'macro_{k}':float(frame[k].mean()) if len(frame) else None for k in keys})
    summary['median_known_pair_rank']=float(np.median([r['rank'] for r in pairs])) if pairs else None
    return summary,frame,pd.DataFrame(pairs)


def observed_ranking(labels,scores,query_ids,candidate_ids):
    labels,scores=np.asarray(labels),np.asarray(scores)
    if labels.shape!=scores.shape or labels.shape!=(len(query_ids),len(candidate_ids)):
        raise ValueError('observed matrix axes do not match')
    if not np.isfinite(scores).all():
        raise ValueError('scores must be finite')
    rows=[]
    for query,y,s in zip(query_ids,labels,scores):
        valid=np.isfinite(y)
        if not np.isin(y[valid],[0,1]).all():
            raise ValueError('observed labels must be 0/1, unknowns NaN')
        yy,ss=y[valid],s[valid]
        if len(np.unique(yy))!=2:
            continue
        rows.append({'query_id':str(query),'measured_candidates':len(yy),'positives':int(yy.sum()),
            'ap':float(average_precision_score(yy,ss)),'auroc':float(roc_auc_score(yy,ss))})
    frame=pd.DataFrame(rows)
    return {'all_queries':len(query_ids),'two_class_queries':len(rows),'observed_pairs':int(np.isfinite(labels).sum()),
        'observed_positive_pairs':int((labels==1).sum()),'observed_negative_pairs':int((labels==0).sum()),
        'macro_ap':float(frame.ap.mean()) if len(frame) else None,
        'macro_auroc':float(frame.auroc.mean()) if len(frame) else None,
        'median_measured_candidates_in_two_class_queries':float(frame.measured_candidates.median()) if len(frame) else None,
        'two_class_queries_with_more_than_20_candidates':int(frame.measured_candidates.gt(20).sum()) if len(frame) else 0,
        'metric_scope':'measured positive/negative observations only, no unknown labels used'},frame


def directional_borda(first,second,axis):
    """Normalize raw components over the candidates for the requested direction.

    axis=1 ranks targets within drug; axis=0 ranks drugs within target. A
    drug-normalized Borda matrix is never silently reused as target normalization.
    """
    if axis not in (0,1) or np.shape(first)!=np.shape(second):
        raise ValueError('two aligned matrices and axis 0/1 required')
    count=np.shape(first)[axis]
    if count==1:
        return np.ones_like(first,dtype=np.float64)
    a=(pd.DataFrame(first).rank(axis=axis,method='average').to_numpy()-1)/max(count-1,1)
    b=(pd.DataFrame(second).rank(axis=axis,method='average').to_numpy()-1)/max(count-1,1)
    return (a+b)/2

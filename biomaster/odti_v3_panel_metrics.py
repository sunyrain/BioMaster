"""Known-positive retrieval over a supplied candidate panel.

Zeros in the label matrix are retrieval background: this function never
asserts that an unreported drug/target pair is a measured biochemical negative.
These AP/recall values are not observed-PN classification metrics or estimates
of recall for every true biological target of a drug.
"""

from __future__ import annotations

import numpy as np


def positive_retrieval_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    target_ids: np.ndarray,
) -> dict:
    """Macro-average known-positive ranks, skipping queries without positives.

    ``labels`` is an ``[N, T]`` boolean (or strictly binary) known-positive
    indicator; ``scores`` is a finite ``[N, T]`` array. ``target_ids`` contains
    ``T`` unique identifiers. Ties are broken in ascending target-ID order,
    making scores independent of the input column order. AP is mean precision
    at the known-positive ranks under that deterministic ordering; it is not
    scikit-learn's threshold-based tie treatment for observed-PN AP.

    Returned retrieval metrics are ``None`` when no query has a known positive.
    The caller defines the candidate panel and any prior-target exclusions;
    this function does not infer them from scores or labels.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    target_ids = np.asarray(target_ids)
    if labels.ndim != 2 or scores.ndim != 2 or labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same [N, T] shape")
    if target_ids.ndim != 1 or len(target_ids) != labels.shape[1]:
        raise ValueError("target_ids must be [T] and match the candidate columns")
    if not len(target_ids):
        raise ValueError("the candidate target panel must be nonempty")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite, including retrieval-background entries")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be boolean or strictly binary known-positive indicators")
    if target_ids.dtype.kind in "fciub":
        if target_ids.dtype.kind == "c" or not np.isfinite(target_ids).all():
            raise ValueError("numeric target IDs must be finite and real")
    else:
        if any(value is None or str(value).strip() in {"", "nan", "None"} for value in target_ids):
            raise ValueError("target IDs must be nonempty and nonmissing")
    try:
        if len(np.unique(target_ids)) != len(target_ids):
            raise ValueError("target IDs must be unique")
        # Check a well-defined identifier order even if no query is evaluated.
        np.argsort(target_ids, kind="stable")
    except TypeError as error:
        raise ValueError("target IDs must have a consistent sortable type") from error

    positives = labels.astype(bool, copy=False)
    counts = positives.sum(axis=1)
    eligible = np.flatnonzero(counts > 0)
    values = {
        "macro_positive_retrieval_ap": [],
        "macro_mrr": [],
        "macro_recall_at_5": [],
        "macro_recall_at_20": [],
        "macro_ndcg_at_20": [],
    }
    top20 = min(20, len(target_ids))
    discounts = 1.0 / np.log2(np.arange(1, top20 + 1) + 1)
    for query in eligible:
        order = np.lexsort((target_ids, -scores[query]))
        ranked = positives[query, order]
        ranks = np.flatnonzero(ranked) + 1
        positive_count = int(counts[query])
        values["macro_positive_retrieval_ap"].append(
            float(np.mean(np.arange(1, positive_count + 1) / ranks))
        )
        values["macro_mrr"].append(float(1.0 / ranks[0]))
        values["macro_recall_at_5"].append(float(ranked[:5].sum() / positive_count))
        values["macro_recall_at_20"].append(float(ranked[:20].sum() / positive_count))
        ideal_dcg = discounts[:min(positive_count, top20)].sum()
        values["macro_ndcg_at_20"].append(float(np.dot(ranked[:top20], discounts) / ideal_dcg))

    result = {
        "total_queries": int(len(labels)),
        "queries": int(len(eligible)),
        "skipped_queries_without_known_positive": int(len(labels) - len(eligible)),
        "candidate_targets": int(len(target_ids)),
        "known_positive_pairs": int(counts.sum()),
        "candidate_scope": "entire supplied target panel; non-positive entries are retrieval background",
        "unknown_background_is_measured_negative": False,
        "metric_scope": "known-positive retrieval only; not observed-PN AP or complete biological-target recall",
        "tie_break": "ascending target ID after descending score",
    }
    result.update({name: float(np.mean(items)) if items else None for name, items in values.items()})
    return result


__all__ = ["positive_retrieval_metrics"]

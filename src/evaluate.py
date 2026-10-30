"""Evaluation metrics (Precision@k, Recall@k, MRR, NDCG@k) and bootstrap uncertainty.

Conventions (documented in the paper):
  * Graded relevance 0..3. A dataset counts as *relevant* for P@k, R@k and MRR when its grade is
    >= ``rel_threshold`` (default 2 = "relevant"). NDCG uses the full graded scale.
  * Judgments are sparse: any (query, dataset) pair that is not listed has grade 0.
  * Queries with no relevant dataset (under the threshold) are undefined for P/R/MRR/NDCG and are
    skipped (returned as NaN) rather than silently scored 0.
  * Recall@k = (relevant items in top-k) / (all relevant items for the query). When a query has
    more than k relevant items its maximum attainable Recall@k is below 1.
  * NDCG@k uses exponential gain (2^grade - 1) and a log2(rank + 1) discount; the ideal ranking is
    built from all judged grades of that query.
  * MRR = mean over queries of 1 / (rank of the first relevant item) within the supplied ranking
    (the experiments supply the top-``mrr_depth`` items; 0 if none appears).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def _relevant(rel: Mapping[int, int], threshold: int) -> set[int]:
    return {d for d, g in rel.items() if g >= threshold}


def precision_at_k(ranked: Sequence[int], rel: Mapping[int, int], k: int = 5, threshold: int = 2) -> float:
    if not _relevant(rel, threshold):
        return float("nan")
    top = list(ranked)[:k]
    relevant = _relevant(rel, threshold)
    return sum(1 for d in top if d in relevant) / k


def recall_at_k(ranked: Sequence[int], rel: Mapping[int, int], k: int = 5, threshold: int = 2) -> float:
    relevant = _relevant(rel, threshold)
    if not relevant:
        return float("nan")
    return sum(1 for d in list(ranked)[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[int], rel: Mapping[int, int], threshold: int = 2) -> float:
    relevant = _relevant(rel, threshold)
    if not relevant:
        return float("nan")
    for pos, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / pos
    return 0.0


def dcg(grades: Sequence[float]) -> float:
    return sum((2.0 ** g - 1.0) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: Sequence[int], rel: Mapping[int, int], k: int = 5) -> float:
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0.0:
        return float("nan")
    gains = [rel.get(d, 0) for d in list(ranked)[:k]]
    return dcg(gains) / idcg


def evaluate_query(ranked: Sequence[int], rel: Mapping[int, int], k: int = 5, threshold: int = 2) -> dict[str, float]:
    return {
        f"P@{k}": precision_at_k(ranked, rel, k, threshold),
        f"R@{k}": recall_at_k(ranked, rel, k, threshold),
        "MRR": reciprocal_rank(ranked, rel, threshold),
        f"NDCG@{k}": ndcg_at_k(ranked, rel, k),
    }


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    keys = per_query[0].keys() if per_query else []
    return {k: float(np.nanmean([q[k] for q in per_query])) if per_query else float("nan") for k in keys}


# --------------------------------------------------------------------------- uncertainty

def cluster_bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 42,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling whole *groups* (topics) with replacement.

    Queries from the same topic are correlated, so resampling queries independently would give
    over-optimistic intervals. NaN values are dropped first.
    """
    mask = ~np.isnan(values)
    values, groups = values[mask], groups[mask]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(groups)
    members = {g: values[groups == g] for g in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means[b] = np.concatenate([members[g] for g in pick]).mean()
    return float(values.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_cluster_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 42,
                             alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean paired difference (a - b) with a cluster-bootstrap percentile CI."""
    return cluster_bootstrap_ci(a - b, groups, n_boot, seed, alpha)

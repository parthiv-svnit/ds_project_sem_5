"""Metric tests with hand-computed expected values."""
import math

import numpy as np
import pytest

from evaluate import (cluster_bootstrap_ci, dcg, evaluate_query, ndcg_at_k, precision_at_k,
                      recall_at_k, reciprocal_rank)

REL = {1: 3, 2: 2, 3: 1, 4: 0, 5: 2}   # relevant (grade>=2): ids 1, 2, 5


def test_precision_at_k():
    assert precision_at_k([1, 4, 2, 9, 8], REL, k=5) == pytest.approx(2 / 5)
    assert precision_at_k([9, 8, 7, 6, 4], REL, k=5) == 0.0


def test_recall_at_k():
    assert recall_at_k([1, 4, 2, 9, 8], REL, k=5) == pytest.approx(2 / 3)
    assert recall_at_k([1, 2, 5, 9, 8], REL, k=5) == pytest.approx(1.0)


def test_recall_denominator_is_all_relevant_not_k():
    many = {i: 3 for i in range(1, 11)}  # 10 relevant
    assert recall_at_k(list(range(1, 6)), many, k=5) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert reciprocal_rank([9, 8, 2, 1], REL) == pytest.approx(1 / 3)
    assert reciprocal_rank([9, 8, 7], REL) == 0.0
    assert reciprocal_rank([1], REL) == 1.0


def test_ndcg_perfect_and_hand_computed():
    ideal = [1, 2, 3, 5, 4]  # grades 3,2,1,2,0 -> ideal order is grades 3,2,2,1,0
    # perfect ranking for NDCG: ids with grades 3,2,2,1,0
    perfect = [1, 2, 5, 3, 4]
    assert ndcg_at_k(perfect, REL, k=5) == pytest.approx(1.0)
    got = [2, 1, 4, 3, 5]  # grades 2,3,0,1,2
    expected_dcg = (2**2 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3) + 0 + (2**1 - 1) / math.log2(5) + (2**2 - 1) / math.log2(6)
    expected_idcg = dcg([3, 2, 2, 1, 0])
    assert ndcg_at_k(got, REL, k=5) == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_uses_graded_scale_not_binary():
    # swapping a grade-3 for a grade-2 item at rank 1 must lower NDCG
    a = ndcg_at_k([1, 2, 5, 3, 4], REL, 5)
    b = ndcg_at_k([2, 1, 5, 3, 4], REL, 5)
    assert b < a


def test_queries_without_relevant_items_are_nan_not_zero():
    empty = {1: 1, 2: 0}  # no grade >= 2
    assert math.isnan(precision_at_k([1, 2], empty))
    assert math.isnan(recall_at_k([1, 2], empty))
    assert math.isnan(reciprocal_rank([1, 2], empty))
    all_zero = {1: 0}
    assert math.isnan(ndcg_at_k([1], all_zero))


def test_evaluate_query_keys():
    out = evaluate_query([1, 2, 3, 4, 5], REL)
    assert set(out) == {"P@5", "R@5", "MRR", "NDCG@5"}


def test_cluster_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = np.array([0.2, 0.4, 0.6, 0.8, 0.5, 0.3])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    m1, lo1, hi1 = cluster_bootstrap_ci(vals, groups, n_boot=300, seed=1)
    m2, lo2, hi2 = cluster_bootstrap_ci(vals, groups, n_boot=300, seed=1)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 <= m1 <= hi1
    assert m1 == pytest.approx(vals.mean())

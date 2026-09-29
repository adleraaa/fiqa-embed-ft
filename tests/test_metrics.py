import math

import pytest

from fiqa_ft.metrics import (
    mean_scores,
    mrr_at_k,
    ndcg_at_k,
    per_query_scores,
    recall_at_k,
)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # relevant docs at ranks 2 and 4, two relevant in total
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y", "b"], {"a": 1, "b": 1}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_respects_cutoff():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_ndcg_graded_relevance_uses_linear_gain():
    # trec_eval ndcg_cut uses gain = rel, not 2^rel - 1
    got = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10)
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert got == pytest.approx(dcg / idcg)


def test_recall_divides_by_all_relevant_not_k():
    rels = {"a": 1, "b": 1, "c": 1, "d": 1}
    assert recall_at_k(["a", "x", "b"], rels, 10) == pytest.approx(0.5)
    assert recall_at_k(["a", "x", "b"], rels, 1) == pytest.approx(0.25)


def test_mrr_first_hit_only():
    assert mrr_at_k(["x", "a", "b"], {"a": 1, "b": 1}, 10) == pytest.approx(0.5)
    assert mrr_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_missing_query_scores_zero_instead_of_being_skipped():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["a"]}  # q2 missing, e.g. dropped by a bug
    scores = per_query_scores(run, qrels, ["ndcg@10"])
    assert scores["ndcg@10"] == {"q1": 1.0, "q2": 0.0}
    assert mean_scores(scores)["ndcg@10"] == pytest.approx(0.5)


def test_queries_without_relevant_docs_are_excluded():
    scores = per_query_scores({"q1": ["a"]}, {"q1": {"a": 1}, "q2": {}}, ["recall@10"])
    assert list(scores["recall@10"]) == ["q1"]

import random

from fiqa_ft.mining import MiningConfig, MiningStats, mine_triplets, select_negatives

CANDS = [("p1", 0.95), ("n1", 0.93), ("n2", 0.80), ("n3", 0.70), ("n4", 0.60), ("n5", 0.50)]


def test_labelled_positives_are_never_negatives():
    cfg = MiningConfig(n_negatives=5, pool_size=10)
    assert "p1" not in select_negatives(CANDS, {"p1"}, cfg)


def test_skip_top_counts_raw_ranks():
    cfg = MiningConfig(n_negatives=2, pool_size=10, skip_top=3)
    # ranks 0-2 (p1, n1, n2) are skipped
    assert select_negatives(CANDS, {"p1"}, cfg) == ["n3", "n4"]


def test_margin_drops_candidates_too_close_to_positive():
    cfg = MiningConfig(n_negatives=10, pool_size=10, margin=0.9)
    stats = MiningStats()
    # threshold = 0.9 * 0.9 = 0.81 -> n1 (0.93) is a likely false negative
    got = select_negatives(CANDS, {"p1"}, cfg, positive_score=0.9, stats=stats)
    assert got == ["n2", "n3", "n4", "n5"]
    assert stats.dropped_margin == 1
    assert stats.dropped_positive == 1


def test_margin_ignored_without_positive_score():
    cfg = MiningConfig(n_negatives=1, pool_size=10, margin=0.5)
    assert select_negatives(CANDS, {"p1"}, cfg, positive_score=None) == ["n1"]


def test_pool_size_bounds_sampling():
    cfg = MiningConfig(n_negatives=1, pool_size=2)
    rng = random.Random(0)
    picks = {select_negatives(CANDS, {"p1"}, cfg, rng=rng)[0] for _ in range(50)}
    assert picks == {"n1", "n2"}


def test_mine_triplets_falls_back_to_random_non_positive_docs():
    qrels = {"q": {"p1": 1, "p2": 1}}
    cfg = MiningConfig(n_negatives=2, pool_size=5)
    triplets, stats = mine_triplets(
        pairs=[("q", "p1")],
        qrels=qrels,
        candidates={"q": [("p2", 0.9)]},  # only candidate is another positive
        all_doc_ids=["p1", "p2", "d1", "d2", "d3"],
        config=cfg,
    )
    negs = [t[2] for t in triplets]
    assert len(negs) == 2 and len(set(negs)) == 2
    assert not {"p1", "p2"} & set(negs)
    assert stats.random_fallback == 2
    assert stats.pairs == 1


def test_mine_triplets_is_deterministic_for_a_seed():
    qrels = {"q": {"p1": 1}}
    kwargs = dict(
        pairs=[("q", "p1")] * 20,
        qrels=qrels,
        candidates={"q": CANDS},
        all_doc_ids=["n1"],
        config=MiningConfig(pool_size=5),
    )
    assert mine_triplets(seed=3, **kwargs)[0] == mine_triplets(seed=3, **kwargs)[0]

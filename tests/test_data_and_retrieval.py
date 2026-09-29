import torch

from fiqa_ft.data import build_qrels, doc_text, id_key, train_pairs
from fiqa_ft.retrieval import bm25_search, ranked_ids, to_results, top_k


def test_build_qrels_casts_ids_to_str_and_drops_nonpositive():
    rows = [
        {"query-id": 8, "corpus-id": 566392, "score": 1},
        {"query-id": 8, "corpus-id": 65404, "score": 1},
        {"query-id": 15, "corpus-id": 325273, "score": 0},
    ]
    assert build_qrels(rows) == {"8": {"566392": 1, "65404": 1}}


def test_doc_text_handles_empty_title():
    assert doc_text("", "  body ") == "body"
    assert doc_text(None, "body") == "body"
    assert doc_text("Title", "body") == "Title body"


def test_train_pairs_numeric_order():
    qrels = {"10": {"b": 1, "a": 1}, "9": {"c": 1}}
    assert train_pairs(qrels) == [("9", "c"), ("10", "a"), ("10", "b")]


def test_id_key_sorts_numeric_ids_numerically_and_does_not_crash_on_others():
    # sorted(ids, key=int) would raise on "q7"; mining and training both sort ids.
    assert sorted(["10", "q7", "9", "a"], key=id_key) == ["9", "10", "a", "q7"]


def test_top_k_matches_full_sort_across_chunks():
    g = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(7, 16, generator=g), dim=1)
    d = torch.nn.functional.normalize(torch.randn(50, 16, generator=g), dim=1)
    scores, idx = top_k(q, d, k=5, chunk_size=3)  # chunk smaller than n_queries
    expected = torch.argsort(q @ d.T, dim=1, descending=True)[:, :5]
    assert torch.equal(idx, expected)
    assert torch.all(scores[:, :-1] >= scores[:, 1:])


def test_top_k_clamps_k_to_corpus_size():
    scores, idx = top_k(torch.eye(2), torch.eye(2), k=10)
    assert idx.shape == (2, 2)


def test_to_results_maps_indices_back_to_ids():
    res = to_results(["q1"], ["a", "b", "c"], torch.tensor([[0.9, 0.1]]), torch.tensor([[2, 0]]))
    assert ranked_ids(res) == {"q1": ["c", "a"]}
    assert res["q1"][0][1] == torch.tensor(0.9).item()


def test_bm25_ranks_lexical_match_first():
    corpus = {
        "d1": "How to file income tax returns for a small business",
        "d2": "Best index funds for long term retirement investing",
        "d3": "Credit card interest rates explained",
    }
    res = bm25_search({"q": "retirement index fund"}, corpus, k=2)
    assert ranked_ids(res)["q"][0] == "d2"

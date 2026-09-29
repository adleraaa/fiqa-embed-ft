"""Exact dense retrieval (brute-force cosine) and BM25 retrieval.

57k documents x 384 dims fits in a few hundred MB, so exact search is cheap and
avoids approximate-nearest-neighbour recall loss muddying the comparison.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

Results = dict[str, list[tuple[str, float]]]  # query id -> [(doc id, score)] best first


def top_k(
    query_emb: torch.Tensor, doc_emb: torch.Tensor, k: int, chunk_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k inner-product search, chunked over queries to bound memory.

    With L2-normalised embeddings the inner product equals cosine similarity.
    Returns (scores, indices), both shaped [n_queries, k].
    """
    k = min(k, doc_emb.shape[0])
    scores, indices = [], []
    for start in range(0, query_emb.shape[0], chunk_size):
        sims = query_emb[start : start + chunk_size] @ doc_emb.T
        s, i = torch.topk(sims, k, dim=1)
        scores.append(s)
        indices.append(i)
    return torch.cat(scores), torch.cat(indices)


def to_results(
    query_ids: Sequence[str], doc_ids: Sequence[str], scores: torch.Tensor, indices: torch.Tensor
) -> Results:
    scores_l, indices_l = scores.float().cpu().tolist(), indices.cpu().tolist()
    return {
        qid: [(doc_ids[j], s) for j, s in zip(idx_row, s_row, strict=True)]
        for qid, idx_row, s_row in zip(query_ids, indices_l, scores_l, strict=True)
    }


def ranked_ids(results: Results) -> dict[str, list[str]]:
    return {qid: [d for d, _ in hits] for qid, hits in results.items()}


def encode(model, texts: Sequence[str], prompt: str | None = None, batch_size: int = 128):
    """Encode with L2 normalisation; returns a torch tensor on the model's device."""
    return model.encode(
        list(texts),
        prompt=prompt,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def dense_search(
    model,
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    k: int = 100,
    query_prompt: str | None = None,
    batch_size: int = 128,
) -> Results:
    query_ids, doc_ids = list(queries), list(corpus)
    q = encode(model, [queries[i] for i in query_ids], prompt=query_prompt, batch_size=batch_size)
    d = encode(model, [corpus[i] for i in doc_ids], batch_size=batch_size)
    scores, indices = top_k(q, d, k)
    return to_results(query_ids, doc_ids, scores, indices)


def bm25_search(queries: Mapping[str, str], corpus: Mapping[str, str], k: int = 100) -> Results:
    """BM25 (Lucene variant, k1=1.5, b=0.75) over English-stemmed, stopword-free tokens."""
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    query_ids, doc_ids = list(queries), list(corpus)
    corpus_tokens = bm25s.tokenize(
        [corpus[i] for i in doc_ids], stopwords="en", stemmer=stemmer, show_progress=False
    )
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(
        [queries[i] for i in query_ids], stopwords="en", stemmer=stemmer, show_progress=False
    )
    indices, scores = retriever.retrieve(query_tokens, k=min(k, len(doc_ids)), show_progress=False)
    return to_results(query_ids, doc_ids, torch.as_tensor(scores), torch.as_tensor(indices))

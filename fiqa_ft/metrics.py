"""Per-query IR metrics with the same definitions as pytrec_eval / BEIR.

- nDCG@k uses linear gain (rel) and log2(rank + 1) discount, like trec_eval's ndcg_cut.
- Recall@k is |relevant in top k| / |relevant|.
- MRR@k is 1 / rank of the first relevant document in the top k, else 0.

Only queries that have at least one relevant document are scored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from fiqa_ft.data import Qrels

Run = Mapping[str, Sequence[str]]  # query id -> ranked doc ids (best first)

DEFAULT_METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr@10")


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


_FUNCS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k}


def per_query_scores(
    run: Run, qrels: Qrels, metrics: Sequence[str] = DEFAULT_METRICS
) -> dict[str, dict[str, float]]:
    """{metric: {qid: score}} for every query in qrels.

    A query missing from the run counts as an empty ranking (score 0), so a retrieval
    bug that drops queries lowers the score instead of being silently ignored.
    """
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        name, k = metric.split("@")
        fn = _FUNCS[name]
        out[metric] = {
            qid: fn(run.get(qid, ()), rels, int(k)) for qid, rels in qrels.items() if rels
        }
    return out


def mean_scores(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {m: sum(v.values()) / len(v) if v else 0.0 for m, v in per_query.items()}

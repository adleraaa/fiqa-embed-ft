"""Glue between retrieval, metrics and the JSON files under results/."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fiqa_ft.data import QUERY_INSTRUCTION, Qrels
from fiqa_ft.metrics import mean_scores, per_query_scores
from fiqa_ft.retrieval import Results, dense_search, ranked_ids


def score_results(results: Results, qrels: Qrels) -> dict[str, Any]:
    per_query = per_query_scores(ranked_ids(results), qrels)
    return {"mean": mean_scores(per_query), "per_query": per_query}


def evaluate_dense(
    model,
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels: Qrels,
    use_instruction: bool = True,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Retrieve top-100 from the full corpus for every query in qrels and score it."""
    t0 = time.perf_counter()
    split_queries = {qid: queries[qid] for qid in qrels}
    results = dense_search(
        model,
        split_queries,
        corpus,
        k=100,
        query_prompt=QUERY_INSTRUCTION if use_instruction else None,
        batch_size=batch_size,
    )
    out = score_results(results, qrels)
    out["seconds"] = time.perf_counter() - t0
    return out


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

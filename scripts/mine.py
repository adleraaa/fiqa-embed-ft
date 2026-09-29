"""Mine one hard negative per (query, positive) training pair.

Writes artifacts/negatives_<name>.jsonl (gitignored, regenerated in ~1-2 min) and
results/mining/<name>.json (config + filter statistics).

Examples:
  python scripts/mine.py --method bm25 --name bm25
  python scripts/mine.py --method dense --name dense_nofilter
  python scripts/mine.py --method dense --margin 0.95 --name dense_margin95
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from fiqa_ft.data import QUERY_INSTRUCTION, load_fiqa, train_pairs
from fiqa_ft.evaluation import write_json
from fiqa_ft.mining import MiningConfig, mine_triplets
from fiqa_ft.retrieval import bm25_search, encode, to_results, top_k

# Candidates retrieved per query before filtering. Deep enough that the margin rule
# (which removes everything scoring close to the positive) still leaves a pool.
DEPTH = 1000


def dense_candidates(fiqa, qids):
    """Top-DEPTH candidates from the *base* model plus cos(query, positive) per pair."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    doc_ids = list(fiqa.corpus)
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    with torch.autocast("cuda", dtype=torch.float16):
        q = encode(model, [fiqa.queries[i] for i in qids], prompt=QUERY_INSTRUCTION)
        d = encode(model, [fiqa.corpus[i] for i in doc_ids])
    q, d = q.float(), d.float()
    scores, indices = top_k(q, d, DEPTH)
    candidates = to_results(qids, doc_ids, scores, indices)

    qpos = {qid: i for i, qid in enumerate(qids)}
    positive_scores = {
        (qid, did): float(q[qpos[qid]] @ d[doc_index[did]])
        for qid, rels in fiqa.qrels["train"].items()
        for did in rels
    }
    return candidates, positive_scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["bm25", "dense"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--skip-top", type=int, default=0)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.method == "bm25" and args.margin is not None:
        raise SystemExit("--margin needs cosine scores; BM25 scores are unbounded")

    fiqa = load_fiqa()
    qrels = fiqa.qrels["train"]
    qids = sorted(qrels, key=int)
    t0 = time.perf_counter()
    if args.method == "bm25":
        candidates = bm25_search({q: fiqa.queries[q] for q in qids}, fiqa.corpus, k=DEPTH)
        positive_scores = None
    else:
        candidates, positive_scores = dense_candidates(fiqa, qids)

    config = MiningConfig(
        n_negatives=1, pool_size=args.pool_size, skip_top=args.skip_top, margin=args.margin
    )
    triplets, stats = mine_triplets(
        train_pairs(qrels),
        qrels,
        candidates,
        list(fiqa.corpus),
        config,
        positive_scores=positive_scores,
        seed=args.seed,
    )
    seconds = time.perf_counter() - t0

    out = Path(f"artifacts/negatives_{args.name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    summary = {
        "name": args.name,
        "method": args.method,
        "depth": DEPTH,
        "seed": args.seed,
        "seconds": seconds,
        "triplets": len(triplets),
        **asdict(stats),
    }
    write_json(f"results/mining/{args.name}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}))


if __name__ == "__main__":
    main()

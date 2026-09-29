"""Mine one hard negative per (query, positive) training pair.

Candidates are retrieved once; the random choice among eligible candidates is then
made once per seed, so each training seed also gets its own draw of negatives.
Writes artifacts/negatives_<name>_seed<k>.jsonl (gitignored, regenerated in ~1-2 min)
and results/mining/<name>.json (config + filter statistics per seed).

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

import torch

from fiqa_ft.data import (
    BASE_MODEL,
    MODEL_REVISIONS,
    QUERY_INSTRUCTION,
    id_key,
    load_fiqa,
    train_pairs,
)
from fiqa_ft.evaluation import write_json
from fiqa_ft.mining import MiningConfig, mine_triplets
from fiqa_ft.retrieval import bm25_search, encode, to_results, top_k
from fiqa_ft.training import negatives_path, require_cuda

# Candidates retrieved per query before filtering. Deep enough that the margin rule
# (which removes everything scoring close to the positive) still leaves a pool.
DEPTH = 1000


def dense_candidates(fiqa, qids):
    """Top-DEPTH candidates from the *base* model plus cos(query, positive) per pair."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BASE_MODEL, device="cuda", revision=MODEL_REVISIONS[BASE_MODEL])
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
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    if args.method == "bm25" and args.margin is not None:
        raise SystemExit("--margin needs cosine scores; BM25 scores are unbounded")

    if args.method == "dense":
        require_cuda()

    fiqa = load_fiqa()
    qrels = fiqa.qrels["train"]
    qids = sorted(qrels, key=id_key)
    t0 = time.perf_counter()
    if args.method == "bm25":
        candidates = bm25_search({q: fiqa.queries[q] for q in qids}, fiqa.corpus, k=DEPTH)
        positive_scores = None
    else:
        candidates, positive_scores = dense_candidates(fiqa, qids)

    retrieval_seconds = time.perf_counter() - t0

    config = MiningConfig(
        n_negatives=1, pool_size=args.pool_size, skip_top=args.skip_top, margin=args.margin
    )
    per_seed = {}
    for seed in args.seeds:
        triplets, stats = mine_triplets(
            train_pairs(qrels),
            qrels,
            candidates,
            list(fiqa.corpus),
            config,
            positive_scores=positive_scores,
            seed=seed,
        )
        out = negatives_path(args.name, seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for t in triplets:
                f.write(json.dumps(t) + "\n")
        stats_dict = asdict(stats)
        stats_dict.pop("config")
        per_seed[str(seed)] = {"triplets": len(triplets), **stats_dict}

    summary = {
        "name": args.name,
        "method": args.method,
        "depth": DEPTH,
        "retrieval_seconds": retrieval_seconds,
        "config": asdict(config),
        # Filter counts are the same for every seed (they depend only on the candidates);
        # the seed changes which eligible candidate or random fallback document is drawn.
        "per_seed": per_seed,
    }
    write_json(f"results/mining/{args.name}.json", summary)
    print(json.dumps(summary["per_seed"]))


if __name__ == "__main__":
    main()

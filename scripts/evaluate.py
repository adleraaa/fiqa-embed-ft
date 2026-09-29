"""Zero-shot evaluation on the FiQA test split (dense model or BM25).

Examples:
  python scripts/evaluate.py --model BAAI/bge-small-en-v1.5 --name baseline_bge-small
  python scripts/evaluate.py --model BAAI/bge-small-en-v1.5 --no-instruction \
      --name baseline_bge-small_noinstr
  python scripts/evaluate.py --bm25 --name bm25
"""

from __future__ import annotations

import argparse
import time

import torch

from fiqa_ft.data import load_fiqa
from fiqa_ft.evaluation import evaluate_dense, score_results, write_json
from fiqa_ft.retrieval import bm25_search


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--bm25", action="store_true")
    p.add_argument("--no-instruction", action="store_true")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    args = p.parse_args()

    fiqa = load_fiqa()
    qrels = fiqa.qrels[args.split]
    record: dict = {"name": args.name, "split": args.split}

    if args.bm25:
        t0 = time.perf_counter()
        results = bm25_search({q: fiqa.queries[q] for q in qrels}, fiqa.corpus, k=100)
        record.update(score_results(results, qrels), seconds=time.perf_counter() - t0)
        record["system"] = "bm25s (Lucene BM25, k1=1.5, b=0.75, English stemmer + stopwords)"
    else:
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(args.model, device=device)
        record["system"] = args.model
        record["use_instruction"] = not args.no_instruction
        record["max_seq_length"] = model.max_seq_length
        record.update(
            evaluate_dense(
                model,
                fiqa.queries,
                fiqa.corpus,
                qrels,
                use_instruction=not args.no_instruction,
            )
        )

    write_json(f"results/runs/{args.name}.json", record)
    print(args.name, {k: round(v, 4) for k, v in record["mean"].items()})


if __name__ == "__main__":
    main()

"""Load BEIR FiQA-2018 into plain dicts.

Sources (Hugging Face Hub):
  corpus + queries: https://huggingface.co/datasets/BeIR/fiqa
  qrels:            https://huggingface.co/datasets/BeIR/fiqa-qrels
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# bge-*-en-v1.5 was trained with this prefix on retrieval queries (documents get no prefix).
# We use it everywhere (baseline, training anchors, evaluation) so that all arms are comparable.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

Qrels = dict[str, dict[str, int]]  # query id -> {doc id: relevance}

# BEIR names the dev split "validation" on the Hub.
_SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "test"}


@dataclass
class FiQA:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, Qrels]  # split name ("train" / "dev" / "test") -> qrels


def doc_text(title: str | None, text: str) -> str:
    """Concatenate title and body the way BEIR does (FiQA titles are empty)."""
    title = (title or "").strip()
    return f"{title} {text.strip()}" if title else text.strip()


def build_qrels(rows: Iterable[Mapping]) -> Qrels:
    """Turn qrels rows into {qid: {docid: rel}}.

    Ids are cast to str because the Hub stores them as ints in qrels but as strings
    in corpus/queries; mixing the two silently yields zero matches.
    Rows with relevance <= 0 are dropped (they are not positives).
    """
    qrels: Qrels = {}
    for row in rows:
        rel = int(row["score"])
        if rel <= 0:
            continue
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = rel
    return qrels


def train_pairs(qrels: Qrels) -> list[tuple[str, str]]:
    """All (query id, positive doc id) pairs in a deterministic order."""
    return [(qid, did) for qid in sorted(qrels, key=_id_key) for did in sorted(qrels[qid])]


def sample_queries(qrels: Qrels, n: int | None, seed: int = 0) -> Qrels:
    """Subsample queries (used to keep per-epoch dev evaluation cheap). n=None keeps all."""
    if n is None or n >= len(qrels):
        return qrels
    keep = random.Random(seed).sample(sorted(qrels, key=_id_key), n)
    return {qid: qrels[qid] for qid in keep}


def _id_key(x: str) -> tuple[int, str]:
    # Numeric ids sort numerically, anything else lexicographically after them.
    return (int(x), "") if x.isdigit() else (1 << 62, x)


def load_fiqa() -> FiQA:
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    query_ds = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/fiqa-qrels")

    corpus = {str(r["_id"]): doc_text(r["title"], r["text"]) for r in corpus_ds}
    queries = {str(r["_id"]): r["text"].strip() for r in query_ds}
    qrels = {name: build_qrels(qrels_ds[hub]) for name, hub in _SPLIT_NAMES.items()}
    return FiQA(corpus=corpus, queries=queries, qrels=qrels)

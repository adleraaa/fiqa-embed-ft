"""Hard-negative selection from a retriever's ranked candidates.

FiQA has on average ~2.6 labelled answers per query and many unlabelled but correct
answers in the corpus. The documents a strong retriever ranks highest are therefore
disproportionately *false* negatives. Two filters address this:

- skip_top: ignore the first N ranks outright.
- margin: drop any candidate scoring above margin * score(query, positive)
  ("TopK-PercPos" from NV-Retriever, Moreira et al. 2024, arXiv:2407.15831).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class MiningConfig:
    n_negatives: int = 1
    pool_size: int = 30  # sample negatives from the first pool_size eligible candidates
    skip_top: int = 0
    margin: float | None = None  # relative to the positive's score; None disables


@dataclass
class MiningStats:
    pairs: int = 0
    dropped_positive: int = 0  # candidates removed because they are labelled positives
    dropped_skip_top: int = 0
    dropped_margin: int = 0
    random_fallback: int = 0  # pairs with too few eligible candidates
    config: dict = field(default_factory=dict)


def select_negatives(
    candidates: Sequence[tuple[str, float]],
    positives: Set[str],
    config: MiningConfig,
    positive_score: float | None = None,
    rng: random.Random | None = None,
    stats: MiningStats | None = None,
) -> list[str]:
    """Pick up to config.n_negatives doc ids from ranked (doc id, score) candidates.

    Candidates must be sorted best first. The margin rule needs positive_score; it is
    skipped when either is None (e.g. BM25 scores are not comparable to cosine).
    """
    max_score = None
    if config.margin is not None and positive_score is not None:
        max_score = config.margin * positive_score

    eligible: list[str] = []
    for rank, (doc_id, score) in enumerate(candidates):
        if doc_id in positives:
            if stats is not None:
                stats.dropped_positive += 1
            continue
        if rank < config.skip_top:
            if stats is not None:
                stats.dropped_skip_top += 1
            continue
        if max_score is not None and score > max_score:
            if stats is not None:
                stats.dropped_margin += 1
            continue
        eligible.append(doc_id)
        if len(eligible) >= config.pool_size:
            break

    n = min(config.n_negatives, len(eligible))
    return rng.sample(eligible, n) if rng else eligible[:n]


def mine_triplets(
    pairs: Sequence[tuple[str, str]],
    qrels: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, Sequence[tuple[str, float]]],
    all_doc_ids: Sequence[str],
    config: MiningConfig,
    positive_scores: Mapping[tuple[str, str], float] | None = None,
    seed: int = 0,
) -> tuple[list[tuple[str, str, str]], MiningStats]:
    """(query id, positive id, negative id) triplets, one per negative per pair.

    If a pair has fewer eligible candidates than requested, the remainder is filled
    with random non-positive corpus documents so every arm trains on the same pairs.
    """
    rng = random.Random(seed)
    stats = MiningStats(config=asdict(config))
    triplets: list[tuple[str, str, str]] = []
    for qid, pos_id in pairs:
        positives = set(qrels[qid])
        pos_score = positive_scores.get((qid, pos_id)) if positive_scores else None
        negs = select_negatives(
            candidates.get(qid, ()), positives, config, pos_score, rng=rng, stats=stats
        )
        while len(negs) < config.n_negatives:
            doc_id = rng.choice(all_doc_ids)
            if doc_id not in positives and doc_id not in negs:
                negs.append(doc_id)
                stats.random_fallback += 1
        triplets.extend((qid, pos_id, neg) for neg in negs)
        stats.pairs += 1
    return triplets, stats

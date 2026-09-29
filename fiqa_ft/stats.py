"""Query-level bootstrap confidence intervals and paired significance tests.

Queries are the unit of resampling: the test set is a sample of 648 queries, and that
sampling is the main source of uncertainty in a single nDCG@10 number.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def seed_average(per_query_runs: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average per-query scores over several seeds of the same system.

    All runs must cover the same queries; a mismatch means the runs were scored on
    different splits and must not be averaged.
    """
    if not per_query_runs:
        raise ValueError("need at least one run")
    qids = set(per_query_runs[0])
    if any(set(r) != qids for r in per_query_runs[1:]):
        raise ValueError("runs cover different queries")
    return {q: float(np.mean([r[q] for r in per_query_runs])) for q in sorted(qids)}


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lower, upper) percentile bootstrap CI of the mean."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_delta(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI of mean(b - a) where a[i] and b[i] are scores for the same query."""
    a_, b_ = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_.shape != b_.shape:
        raise ValueError("paired inputs must have the same length")
    return bootstrap_ci(b_ - a_, n_resamples=n_resamples, alpha=alpha, seed=seed)


def paired_randomization_test(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> float:
    """Two-sided sign-flip permutation test for mean(b - a) == 0.

    Under the null the two systems are exchangeable per query, so each difference
    is equally likely to have either sign. The +1 terms keep p > 0 (Phipson & Smyth 2010).
    """
    a_, b_ = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_.shape != b_.shape:
        raise ValueError("paired inputs must have the same length")
    d = b_ - a_
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.count_nonzero(null >= observed - 1e-12) + 1) / (n_resamples + 1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order.

    Controls the family-wise error rate over all tests in the family: an adjusted
    p below 0.05 means the difference survives correction for every comparison made.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        # the smallest p is multiplied by m, the next by m - 1, ...; the running max
        # keeps adjusted p-values in the same order as the raw ones
        running_max = max(running_max, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running_max
    return adjusted

"""The glue in scripts/summarize.py: which comparisons are made and how they are paired."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

_spec = importlib.util.spec_from_file_location(
    "summarize", Path(__file__).parents[1] / "scripts" / "summarize.py"
)
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


def _run(scores: list[float], **extra) -> dict:
    pq = {str(i): s for i, s in enumerate(scores)}
    return {"per_query": {"ndcg@10": pq, "recall@100": pq}, **extra}


def test_paired_refuses_different_query_sets():
    with pytest.raises(ValueError, match="different queries"):
        summarize.paired({"1": 0.5, "2": 0.5}, {"1": 0.5, "3": 0.5})


def test_paired_pairs_by_query_id_not_by_position():
    a = {"1": 0.0, "2": 1.0}
    b = {"2": 1.0, "1": 0.0}  # same scores, different dict order
    assert summarize.paired(a, b)["delta"] == 0.0


def test_compare_all_makes_expected_pairs_and_holm_adjusts():
    trained = {"train_seconds": 1.0}
    systems = {
        summarize.BASELINE: [_run([0.4] * 20)],
        summarize.INBATCH: [_run([0.4] * 20, **trained)],
        "dense_nofilter/full": [_run([0.4] * 20, **trained)],
        "dense_margin95/full": [_run([0.5] * 20, **trained)],
    }
    comps = summarize.compare_all(systems)
    pairs = {(c["system"], c["reference"]) for c in comps}
    assert pairs == {
        (summarize.INBATCH, summarize.BASELINE),
        ("dense_nofilter/full", summarize.BASELINE),
        ("dense_nofilter/full", summarize.INBATCH),
        ("dense_margin95/full", summarize.BASELINE),
        ("dense_margin95/full", summarize.INBATCH),
        ("dense_margin95/full", "dense_nofilter/full"),  # the direct filter test
    }
    assert len(comps) == 2 * len(pairs)  # nDCG@10 and R@100 each
    assert all(c["p_holm"] >= c["p"] for c in comps)

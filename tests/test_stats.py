import numpy as np
import pytest

from fiqa_ft.stats import (
    bootstrap_ci,
    holm_adjust,
    paired_bootstrap_delta,
    paired_randomization_test,
    seed_average,
)


def test_bootstrap_ci_contains_mean_and_is_ordered():
    x = np.random.default_rng(0).normal(0.4, 0.2, size=500)
    mean, lo, hi = bootstrap_ci(x, n_resamples=2000)
    assert lo < mean < hi
    # the standard error is about 0.2 / sqrt(500) ~ 0.009, so the 95% CI is ~0.035 wide
    assert 0.02 < hi - lo < 0.05


def test_paired_delta_sign():
    a = np.full(100, 0.3)
    b = a + 0.1
    delta, lo, hi = paired_bootstrap_delta(a, b, n_resamples=500)
    assert delta == pytest.approx(0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_randomization_test_detects_consistent_improvement():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 300)
    b = a + 0.05 + rng.normal(0, 0.05, 300)
    assert paired_randomization_test(a, b, n_resamples=2000) < 0.01


def test_randomization_test_null_is_not_significant():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 300)
    b = a + rng.normal(0, 0.1, 300)  # zero-mean noise
    assert paired_randomization_test(a, b, n_resamples=2000) > 0.05


def test_identical_systems_give_p_one():
    a = [0.1, 0.5, 0.9]
    assert paired_randomization_test(a, a, n_resamples=100) == pytest.approx(1.0)


def test_seed_average():
    runs = [{"q1": 0.2, "q2": 1.0}, {"q1": 0.4, "q2": 0.0}]
    assert seed_average(runs) == {"q1": pytest.approx(0.3), "q2": pytest.approx(0.5)}


def test_seed_average_rejects_different_query_sets():
    with pytest.raises(ValueError):
        seed_average([{"q1": 0.2}, {"q2": 0.4}])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_randomization_test([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        paired_bootstrap_delta([1, 2], [1, 2, 3])


def test_holm_adjust_matches_hand_computation():
    # sorted: 0.01 x 3 = 0.03, 0.02 x 2 = 0.04, 0.04 x 1 = 0.04 (then back to input order)
    assert holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert holm_adjust([]) == []

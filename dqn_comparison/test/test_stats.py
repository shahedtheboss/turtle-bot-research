"""Unit tests for the planned-contrast statistics in analyze.py (added 2026-11-17).

Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/test_stats.py -q
Every expected value below can be worked out by hand (comments show how).
"""

import pytest

from dqn_comparison.analyze import (exact_perm_test_paired, exact_perm_test_unpaired, holm,
                                    vargha_delaney_a)


def test_a_complete_separation():
    # every X (sooner) beats every Y: 25/25 pairs
    assert vargha_delaney_a([60, 60, 80, 80, 100], [220] * 5) == 1.0


def test_a_identical_groups_is_half():
    assert vargha_delaney_a([80, 100], [80, 100]) == 0.5


def test_a_ties_count_half():
    # pairs: (80,80) tie=0.5, (80,100) win=1  -> 1.5 / 2
    assert vargha_delaney_a([80], [80, 100]) == 0.75


def test_unpaired_complete_separation_is_1_over_252():
    # Only 1 of C(10,5)=252 splits puts all 5 smallest values in X.
    a, p1, p2 = exact_perm_test_unpaired([100, 120, 120, 100, 80], [220] * 5)
    assert a == 1.0
    assert p1 == pytest.approx(1 / 252)
    # two-sided: the mirror split (all 220s in X, A=0) is equally extreme -> 2/252
    assert p2 == pytest.approx(2 / 252)


def test_unpaired_wrong_direction_gives_large_p():
    # X is SLOWER than Y: the one-sided test for "X sooner" must not be significant
    _, p1, _ = exact_perm_test_unpaired([220] * 5, [60, 60, 80, 80, 100])
    assert p1 == pytest.approx(1.0)


def test_unpaired_no_difference():
    a, p1, p2 = exact_perm_test_unpaired([80, 100], [80, 100])
    assert a == 0.5
    assert p2 == pytest.approx(1.0)


def test_paired_all_positive_is_1_over_32():
    # Y - X = +20 for all 5 seeds; only the all-+ pattern of 2^5=32 reaches mean +20
    m, p1, p2 = exact_perm_test_paired([80, 60, 100, 80, 100], [100, 80, 120, 100, 120])
    assert m == pytest.approx(20.0)
    assert p1 == pytest.approx(1 / 32)
    assert p2 == pytest.approx(2 / 32)


def test_paired_zero_differences():
    # all differences 0: every sign pattern ties the observed mean -> p = 1
    _, p1, p2 = exact_perm_test_paired([80, 80], [80, 80])
    assert p1 == pytest.approx(1.0) and p2 == pytest.approx(1.0)


def test_holm_known_example():
    # sorted p: 0.01, 0.02, 0.03, 0.04 -> x4, x3, x2, x1 = 0.04, 0.06, 0.06 (monotone), 0.06
    assert holm([0.01, 0.04, 0.03, 0.02]) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])

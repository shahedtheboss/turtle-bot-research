"""Tests for the v2 E* metric. (The cross-check against the v1 results is on the version-1 branch.)"""

from dqn_comparison.metrics_v2 import e_star


def test_first_confirmed_check():
    checks = [(20, 0), (40, 3), (60, 2), (80, 4), (100, 3), (120, 4)]
    assert e_star(checks, 200, 20) == (80, False)        # 40 passes but 60 fails -> not confirmed


def test_threshold_is_three_of_four_by_default():
    assert e_star([(20, 3), (40, 3)], 200, 20) == (20, False)
    assert e_star([(20, 2), (40, 4)], 200, 20) == (220, True)


def test_last_check_cannot_be_confirmed_so_run_is_censored():
    checks = [(e, 0) for e in range(20, 200, 20)] + [(200, 4)]
    assert e_star(checks, 200, 20) == (220, True)
    assert e_star([(180, 4), (200, 4)], 200, 20) == (180, False)  # 180 is the latest learnable E*


def test_missing_check_breaks_confirmation():
    assert e_star([(20, 4), (60, 4)], 200, 20) == (220, True)     # 40 missing: not consecutive


def test_order_and_types_do_not_matter():
    assert e_star([('60', '4'), ('40', '3'), ('20', '0')], 200, 20) == (40, False)


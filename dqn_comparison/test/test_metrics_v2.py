"""Tests for the v2 E* metric, including a check that it reproduces the v1 values on the v1 data."""
import glob
import os
import statistics

import pytest

from dqn_comparison.metrics_v2 import e_star, e_star_from_csv

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


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


V1_RUNS = sorted(glob.glob(os.path.join(REPO, 'results', '*_s[1-5]', '*_evals.csv')))


@pytest.mark.skipif(len(V1_RUNS) != 30, reason='v1 results not present')
def test_reproduces_v1_medians_with_v1_rule():
    """With 5 check episodes and threshold 4 (the v1 rule) the function must give the published v1 medians."""
    by_arm = {}
    for path in V1_RUNS:
        arm = os.path.basename(os.path.dirname(path)).rsplit('_s', 1)[0]
        by_arm.setdefault(arm, []).append(e_star_from_csv(path, 200, 20, threshold=4)[0])
    medians = {arm: statistics.median(v) for arm, v in by_arm.items()}
    assert medians == {'dqn': 100, 'ddqn': 80, 'd3qn': 60, 'd3qn_g': 120, 'rainbow': 220, 'rainbow_g': 100}
    assert sorted(by_arm['dqn']) == [80, 100, 100, 120, 140]       # per-seed values, paper Fig. 4

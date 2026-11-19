"""Tests for the v2 final evaluation setup (no Gazebo needed)."""
import os

import pytest

from dqn_comparison import evaluate_v2 as E
from dqn_comparison.evaluate import load_greedy_agent

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.mark.parametrize('world', [2, 3, '4s', 4, 6])
def test_twenty_combinations_times_repeats(world):
    c = E.combinations(world, 2)
    assert len(c) == 40
    assert len({(pi, gi) for pi, _, gi, _, _ in c}) == 20          # 5 poses x 4 goals
    assert c[0][0] == 0 and c[0][2] == 0                           # starts with the start pose, goal 1


def test_parse_args():
    a = E.parse_args(['p', '--world', '4s', '--policy', 'heuristic', '--out-csv', 'x.csv'])
    assert (a.world, a.policy, a.repeats) == ('4s', 'heuristic', 2)
    with pytest.raises(SystemExit):
        E.parse_args(['p', '--world', '4', '--out-csv', 'x.csv'])    # greedy needs --model


SMOKE_RAINBOW = os.path.join(REPO, 'results_smoke', 'a_changes', 'w2_rainbow_s9', 'w2_rainbow_s9_model_final.pth')


@pytest.mark.skipif(not os.path.isfile(SMOKE_RAINBOW), reason='v2 smoke checkpoint not present')
def test_loader_keeps_scaled_rainbow_support():
    agent = load_greedy_agent(SMOKE_RAINBOW)
    assert (agent.v_min, agent.v_max) == (pytest.approx(-0.75), pytest.approx(1.25))

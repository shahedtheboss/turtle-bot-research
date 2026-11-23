"""Tests for the v2 final evaluation setup (no Gazebo needed)."""
import os

import pytest

from dqn_comparison import evaluate_v2 as E
from dqn_comparison.learning import load_greedy_agent

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


# ---- deviation 4: baselines H and H_noisy ----
import inspect
import random

from dqn_comparison import learning
from dqn_comparison.common import GUIDED_NOISE, GUIDED_NOISE_WEIGHTS, heuristic_action


def _states(n=50, seed=0):
    r = random.Random(seed)
    return [[r.random() for _ in range(50)] + [r.random(), r.uniform(-1, 1)] for _ in range(n)]


def test_heuristic_choice_without_noise_is_the_heuristic():
    rng = random.Random(1)
    for s in _states():
        assert E.heuristic_choice(s, 3.5, rng, 0.0) == heuristic_action(s, 3.5)


def test_noisy_heuristic_is_seeded_and_forward_biased():
    s = _states(1)[0]
    a = [E.heuristic_choice(s, 3.5, random.Random(7), 1.0) for _ in range(1)]
    b = [E.heuristic_choice(s, 3.5, random.Random(7), 1.0) for _ in range(1)]
    assert a == b                                                   # same seed, same actions
    rng = random.Random(3)
    draws = [E.heuristic_choice(s, 3.5, rng, 1.0) for _ in range(20000)]
    assert abs(draws.count(0) / len(draws) - 4 / 10) < 0.02         # forward has weight 4 of 10


def test_noise_weights_match_training_guide():
    assert GUIDED_NOISE == 0.15
    assert f"weights={GUIDED_NOISE_WEIGHTS}" in inspect.getsource(learning.run_episode)


def test_parse_args_noise():
    a = E.parse_args(['p', '--world', '2', '--policy', 'heuristic', '--guide-noise', '0.15', '--seed', '3',
                      '--out-csv', 'x.csv'])
    assert (a.guide_noise, a.seed) == (0.15, 3)
    with pytest.raises(SystemExit):
        E.parse_args(['p', '--world', '2', '--model', 'm.pth', '--guide-noise', '0.15', '--out-csv', 'x.csv'])

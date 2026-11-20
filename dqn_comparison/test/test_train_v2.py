"""Tests for the v2 training script's setup (no Gazebo needed)."""
import random

import pytest

from dqn_comparison import train_v2 as T


def test_parse_args_world_and_arms():
    a = T.parse_args(['prog', '--world', '4s', '--arm', 'rainbow_g', '--seed', '3'])
    assert (a.world, a.algo, a.guide, a.seed) == ('4s', 'rainbow', True, 3)
    assert T.parse_args(['prog', '--world', '6', '--arm', 'dqn']).world == 6
    with pytest.raises(SystemExit):
        T.parse_args(['prog', '--world', '1', '--arm', 'dqn'])        # stage 1 is not a v2 world
    with pytest.raises(SystemExit):
        T.parse_args(['prog', '--world', '4', '--arm', 'd3qn_g'])     # guided D3QN dropped in v2


def test_goal_sequence_depends_only_on_seed():
    assert T.goal_sequence(1, 200) == T.goal_sequence(1, 200)        # same for every method
    assert T.goal_sequence(1, 200) != T.goal_sequence(2, 200)
    assert T.goal_sequence(1, 50) == T.goal_sequence(1, 200)[:50]    # prefix-stable: budget does not change early goals


def test_goal_sequence_does_not_touch_training_rng():
    random.seed(123)
    before = random.random()
    random.seed(123)
    T.goal_sequence(5, 400)
    assert random.random() == before


def test_goal_sequence_exactly_balanced_in_blocks():
    for s in range(1, 6):
        seq = T.goal_sequence(s, 200)
        assert [seq.count(g) for g in range(4)] == [50, 50, 50, 50]
        assert all(sorted(seq[i:i + 4]) == [0, 1, 2, 3] for i in range(0, 200, 4))
    assert T.goal_sequence(1, 8)[:4] != T.goal_sequence(1, 8)[4:]       # order is shuffled, not fixed


def test_untracked_code_files_returns_a_list():
    files = T.untracked_code_files()
    assert isinstance(files, list) and all(not f.startswith("results") for f in files)


# ---- v2 algorithm changes: reward scaling and PER beta (2026-11-19) ----
import numpy as np                                           # noqa: E402
from dqn_comparison.agent_dqn import DQNAgent                # noqa: E402


def test_reward_scale_applies_to_stored_transitions_dqn():
    agent = T.apply_reward_scale(DQNAgent(52, 7), 0.01)          # the option still works
    s = np.zeros(52, dtype=np.float32)
    agent.remember(s, 0, 100.0, s, True)
    agent.remember(s, 1, -25.0, s, True, truncated=False)
    assert [m[2] for m in agent.memory] == [pytest.approx(1.0), pytest.approx(-0.25)]


def test_reward_scale_applies_to_rainbow_and_support_is_scaled():
    from dqn_comparison.agent_rainbow import build_rainbow
    agent = T.apply_reward_scale(build_rainbow(52, 7, v_min=-0.75, v_max=1.25), 0.01)
    assert (agent.v_min, agent.v_max) == (pytest.approx(-0.75), pytest.approx(1.25))
    assert agent.delta_z == pytest.approx(0.04)
    seen = []
    agent.memory.add = lambda *a: seen.append(a[2])           # what reaches the replay buffer
    agent.remember(np.zeros(52), 0, 100.0, np.zeros(52), True)
    assert seen == [pytest.approx(1.0)]


def test_v1_rainbow_defaults_unchanged():
    from dqn_comparison.agent_rainbow import build_rainbow
    a = build_rainbow()
    assert (a.v_min, a.v_max) == (-75.0, 125.0) and a.memory.beta_override is None


def test_per_beta_schedule():
    assert T.per_beta(0, 200) == pytest.approx(0.4)
    assert T.per_beta(100, 200) == pytest.approx(0.7)
    assert T.per_beta(200, 200) == pytest.approx(1.0)
    assert T.per_beta(250, 200) == pytest.approx(1.0)


def test_replay_buffer_uses_beta_override_and_keeps_v1_schedule_without_it():
    agent = T.AGENTS_V2['rainbow']()
    s = np.zeros(52, dtype=np.float32)
    for i in range(300):
        agent.memory.add(s, i % 7, 0.01, s, i % 10 == 9)
    agent.memory.sample()
    assert agent.memory.last_beta == pytest.approx(0.4, abs=1e-5)  # v1 schedule: 1 sample of 1e6
    agent.memory.beta_override = 0.9
    agent.memory.sample()
    assert agent.memory.last_beta == 0.9


def test_diagnostic_flags_default_off():
    a = T.parse_args(['p', '--world', '4', '--arm', 'd3qn'])
    assert a.diag_goal is None and a.reward_scale == T.REWARD_SCALE
    b = T.parse_args(['p', '--world', '4s', '--arm', 'd3qn', '--diag-goal', '2,-2', '--reward-scale', '1'])
    assert b.diag_goal == '2,-2' and b.reward_scale == 1.0


def test_default_is_v1_rewards_after_diagnosis():
    assert T.REWARD_SCALE == 1.0
    a = T.AGENTS_V2['rainbow']()
    assert (a.v_min, a.v_max) == (-75.0, 125.0)                   # v1 support again
    assert T.parse_args(['p', '--world', '4', '--arm', 'd3qn']).stop_when_learned is False

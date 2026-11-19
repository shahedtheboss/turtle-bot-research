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

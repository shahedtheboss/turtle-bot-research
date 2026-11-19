"""Tests for the v2 world geometry, obstacle motion and layouts (no Gazebo needed)."""
import math
import os
import sys

import pytest

from dqn_comparison import worlds_v2 as W

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
import v2_world_layout as L  # noqa: E402


def test_keyframes_match_plugin_sources():
    L.check_keyframes()                       # raises AssertionError on any difference


@pytest.mark.parametrize('t,expected', [(0, (0.0, 0.0)), (10, (-0.5, -1.0)), (30, (-2.0, -1.0)),
                                        (160, (0.0, 0.0)), (190, (-2.0, -1.0))])   # 190 = 30 after one loop
def test_obstacle1_interpolates_and_loops(t, expected):
    x, y = W.anim_position('obstacle1', t)
    assert x == pytest.approx(expected[0]) and y == pytest.approx(expected[1])


def test_stage3_group_rotates_quarter_turn_in_10_s():
    before = W.obstacles_at(3, 0)
    after = W.obstacles_at(3, 10)              # 40 s per turn -> 90 deg
    for (x0, y0, _), (x1, y1, _) in zip(before, after):
        assert (x1, y1) == pytest.approx((-y0, x0))


def test_wall_distance():
    assert W.wall_distance(1, 0.0, 0.0) == pytest.approx(W.ARENA_HALF)
    assert W.wall_distance(1, 2.0, 0.0) == pytest.approx(W.ARENA_HALF - 2.0)
    assert W.wall_distance(4, 1.0, -1.0) == 0.0          # inside an inner wall


@pytest.mark.parametrize('stage', [1, 2, 3, 4, 6])
def test_layouts_satisfy_rules(stage):
    lay = W.LAYOUTS[stage]
    poses = [lay['start']] + lay['tests']
    assert all(L.pose_ok(stage, x, y, lay['W']) for x, y, _ in poses)
    assert all(L.goal_ok(stage, gx, gy, lay['start']) for gx, gy in lay['goals'])
    assert L.smallest_w(stage, poses) == lay['W']        # W is the smallest valid warm-up
    assert len(lay['tests']) == 4 and len(lay['goals']) == 4

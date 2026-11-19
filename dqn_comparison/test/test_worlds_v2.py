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


def test_stage3_rotation_is_recorded_eased_and_periodic():
    # recorded in Gazebo (B4b): eased, not constant speed; half a turn at 20 s; 40 s period
    assert math.degrees(W.recorded_group_yaw(5.0)) == pytest.approx(32.3, abs=0.2)
    assert abs(math.degrees(W.recorded_group_yaw(20.0))) == pytest.approx(180.0, abs=0.2)
    assert W.recorded_group_yaw(45.0) == pytest.approx(W.recorded_group_yaw(5.0))


def test_recorded_cylinders_start_at_their_offsets_and_match_lidar_check():
    # t = 0: cylinders at the nested <include> offsets, not at (0, 0)
    assert [o[:2] for o in W.obstacles_at(4, 0.0)] == [pytest.approx((2.0, 2.0), abs=0.01),
                                                       pytest.approx((-2.0, -2.0), abs=0.01)]
    # positions confirmed by the LiDAR truth check (scripts/v2_obstacle_truth_check.py)
    assert [o[:2] for o in W.obstacles_at(4, 15.3)] == [pytest.approx((1.21, 0.97), abs=0.02),
                                                        pytest.approx((-1.00, -1.36), abs=0.02)]
    # stage 6 uses the same plugins for obstacles 1 and 2 -> identical paths
    assert W.obstacles_at(6, 77.7)[:2] == W.obstacles_at(4, 77.7)


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

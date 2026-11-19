"""Tests for the v2 episode start (fixed obstacle phase) with a fake Gazebo: no simulator needed."""
import pytest
import rclpy

import dqn_comparison.episode_v2 as ep
from dqn_comparison.environment import RobotEnvironment


@pytest.fixture(scope='module', autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


class Reply:
    success = True


def fake_gazebo(monkeypatch, env, reset_works=True):
    calls = []

    def spin_once(node, timeout_sec=None):            # 1 ms of simulated time and one sensor message per poll
        env.sim_now += 0.001
        env.sensor_update_count += 1

    def call(env_, client, request, what):
        calls.append((what, env.sim_now))
        if what == '/reset_simulation' and reset_works:
            env.sim_now = 0.0                          # the simulated clock restarts
        return Reply()

    monkeypatch.setattr(ep.rclpy, 'spin_once', spin_once)
    monkeypatch.setattr(ep, '_call', call)
    monkeypatch.setattr(env, 'begin_episode', lambda start, settle_sec: calls.append(('begin', start)))
    return calls


def make_env(monkeypatch, sim=True):
    monkeypatch.setenv('DQN_SIM_TIME_STEP', '1' if sim else '0')
    env = RobotEnvironment()
    env.sim_now = 123.4                                # time before the reset
    return env


@pytest.mark.parametrize('stage', [1, 4, 6])
def test_episode_starts_at_w_plus_settle(monkeypatch, stage):
    env = make_env(monkeypatch)
    calls = fake_gazebo(monkeypatch, env)
    t = ep.prepare_episode_v2(env, stage, (0.0, 0.0, 0.0), (1.75, -1.75))
    w = ep.LAYOUTS[stage]['W']
    assert t == pytest.approx(w + ep.SETTLE_SIM, abs=0.005)
    assert [c[0] for c in calls] == ['/reset_simulation', '/set_entity_state', 'begin']
    assert calls[1][1] == pytest.approx(w, abs=0.002)  # teleport happens at time W, after the warm-up
    assert (env.goal_x, env.goal_y) == (1.75, -1.75)
    env.destroy_node()


def test_requires_sim_time_stepping(monkeypatch):
    env = make_env(monkeypatch, sim=False)
    fake_gazebo(monkeypatch, env)
    with pytest.raises(RuntimeError, match='DQN_SIM_TIME_STEP'):
        ep.prepare_episode_v2(env, 4, (0.0, 0.0, 0.0), (1.75, -1.75))
    env.destroy_node()


def test_reset_that_keeps_the_clock_is_detected(monkeypatch):
    env = make_env(monkeypatch)
    fake_gazebo(monkeypatch, env, reset_works=False)  # e.g. /reset_world: clock not reset
    monkeypatch.setattr(ep, 'WALL_TIMEOUT', 0.2)
    with pytest.raises(RuntimeError, match='clock going back'):
        ep.prepare_episode_v2(env, 4, (0.0, 0.0, 0.0), (1.75, -1.75))
    env.destroy_node()

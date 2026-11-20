"""Tests for simulated-time stepping in RobotEnvironment (v2). No Gazebo needed:
rclpy.spin_once is replaced by a fake that advances the simulated clock."""
import pytest
import rclpy

import dqn_comparison.environment as envmod
from dqn_comparison.environment import RobotEnvironment


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def make_env(monkeypatch, sim):
    monkeypatch.setenv("DQN_SIM_TIME_STEP", "1" if sim else "0")
    env = RobotEnvironment()
    sent = []
    monkeypatch.setattr(env.cmd_vel_pub, "publish", lambda m: sent.append((m.linear.x, m.angular.z)))
    return env, sent


def fake_clock(monkeypatch, env, tick):
    """Each spin_once advances the simulated clock by `tick` seconds (tick=0: frozen clock)."""
    def spin_once(node, timeout_sec=None):
        env.sim_now = (env.sim_now if env.sim_now is not None else 100.0) + tick
    monkeypatch.setattr(envmod.rclpy, "spin_once", spin_once)


def test_default_is_wall_clock_mode(monkeypatch):
    env, _ = make_env(monkeypatch, sim=False)
    assert env.sim_time_stepping is False
    assert not hasattr(env, "clock_sub")          # v1 behaviour: no /clock subscription
    env.destroy_node()


def test_action_lasts_half_a_simulated_second(monkeypatch):
    env, sent = make_env(monkeypatch, sim=True)
    fake_clock(monkeypatch, env, tick=0.001)     # 1000 Hz clock
    env.execute_action_for_duration(0)           # forward 0.15 m/s
    assert 0.5 <= env.last_action_sim_duration < 0.503
    assert sent[0] == (0.15, 0.0)                # action first ...
    assert sent[-1] == (0.0, 0.0)                # ... then stop
    env.destroy_node()


def test_frozen_clock_raises_and_stops_robot(monkeypatch):
    env, sent = make_env(monkeypatch, sim=True)
    env.sim_now = 100.0
    env.sim_wait_timeout = 0.05
    fake_clock(monkeypatch, env, tick=0.0)       # /clock never advances
    with pytest.raises(RuntimeError):
        env.execute_action_for_duration(0)
    assert sent[-1] == (0.0, 0.0)                # robot is still stopped
    env.destroy_node()


def test_poll_interval_default_and_override(monkeypatch):
    env, _ = make_env(monkeypatch, sim=True)
    assert env.sim_poll == 0.001                       # original behaviour
    env.destroy_node()
    monkeypatch.setenv('DQN_SIM_POLL_S', '0.005')
    env, sent = make_env(monkeypatch, sim=True)
    assert env.sim_poll == 0.005
    fake_clock(monkeypatch, env, tick=0.005)           # 200 Hz clock
    env.execute_action_for_duration(0)
    assert 0.5 <= env.last_action_sim_duration < 0.506 and sent[-1] == (0.0, 0.0)
    env.destroy_node()

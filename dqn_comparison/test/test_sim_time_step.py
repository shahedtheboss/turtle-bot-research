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


def make_env(monkeypatch, sim, pause=False):
    monkeypatch.setenv("DQN_SIM_TIME_STEP", "1" if sim else "0")
    monkeypatch.setenv("DQN_SIM_PAUSE", "1" if pause else "0")
    env = RobotEnvironment()
    sent = []
    monkeypatch.setattr(env.cmd_vel_pub, "publish", lambda m: sent.append((m.linear.x, m.angular.z)))
    return env, sent


def fake_clock(monkeypatch, env, tick):
    """Each spin_once advances the simulated clock by `tick` seconds (tick=0: frozen clock)."""
    def spin_once(node, timeout_sec=None):
        env.sim_now = (env.sim_now if env.sim_now is not None else 100.0) + tick
        env.clock_count += 1
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


class FakeGazebo:
    """Simulated clock that keeps running while the agent learns (nothing spins), unless paused.
    spin_once delivers the current time as a /clock message; learn(wall_s) lets wall time pass at speed 3."""
    def __init__(self, env, tick=0.005, speed=3.0):
        self.env, self.tick, self.speed, self.real, self.paused = env, tick, speed, 100.0, False
        self.moves = []                                   # (sim time at move command, sim time at stop)

    def spin_once(self, node, timeout_sec=None):
        if not self.paused:
            self.real += self.tick
            self.env.sim_now = self.real
            self.env.clock_count += 1

    def learn(self, wall_s):
        if not self.paused:
            self.real += wall_s * self.speed              # Gazebo runs on; the stale sim_now is not updated

    def publish(self, msg):
        if msg.linear.x or msg.angular.z:
            self.moves.append([self.real, None])
        elif self.moves and self.moves[-1][1] is None:
            self.moves[-1][1] = self.real


def install(monkeypatch, env, gz):
    monkeypatch.setattr(envmod.rclpy, "spin_once", gz.spin_once)
    monkeypatch.setattr(env.cmd_vel_pub, "publish", gz.publish)


def test_action_keeps_full_length_after_slow_learning_without_pause(monkeypatch):
    """The 2026-11-21 bug: after 1 s of learning the next action ended at once. Fixed by the fresh clock."""
    env, _ = make_env(monkeypatch, sim=True, pause=False)
    gz = FakeGazebo(env)
    install(monkeypatch, env, gz)
    env.execute_action_for_duration(0)
    gz.learn(1.0)                                         # 1 s wall of learning = 3 s simulated
    env.execute_action_for_duration(0)
    start, stop = gz.moves[-1]
    assert stop - start >= 0.5                            # the robot really moved for 0.5 s of sim time
    assert 0.5 <= env.last_action_sim_duration < 0.51
    env.destroy_node()


def test_pause_stops_simulated_time_between_steps(monkeypatch):
    env, _ = make_env(monkeypatch, sim=True, pause=True)
    gz = FakeGazebo(env)
    install(monkeypatch, env, gz)
    calls = []

    def fake_pause(paused):
        if env.physics_paused != paused:
            calls.append(paused)
            gz.paused = env.physics_paused = paused
    monkeypatch.setattr(env, "set_physics_paused", fake_pause)
    monkeypatch.setattr(env, "get_state", lambda: [1.0] * 50 + [1.0, 0.0])
    monkeypatch.setattr(env, "calculate_reward", lambda d: (0.0, False, ""))
    env.best_goal_distance = 0.0
    for _ in range(3):
        env.step(0)
        t_after_step = gz.real
        gz.learn(2.0)                                     # long learning: must not advance simulated time
        assert gz.real == t_after_step
    assert calls[:2] == [False, True] and env.physics_paused is True
    lengths = [b - a for a, b in gz.moves]
    assert all(0.5 <= x < 0.51 for x in lengths)          # every action lasted 0.5 s of sim time
    env.destroy_node()

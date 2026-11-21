"""Study v2: start an episode at the SAME obstacle phase every time (option b).

    t_start = prepare_episode_v2(env, stage=4, pose=(0.0, 0.0, 0.0), goal=(1.75, -1.75))

Sequence (all waits on the SIMULATED clock; needs DQN_SIM_TIME_STEP=1 and /clock at 1000 Hz):
  1. /reset_simulation (no fallback: /reset_world keeps the clock, so the obstacle phase would differ),
     then confirm the reset by seeing the simulated clock jump back.
  2. Wait until simulated time >= W (worlds_v2.LAYOUTS[stage]['W']); obstacle animations restart at 0
     on reset, so every episode meets the obstacles at the same place. The robot is kept stopped.
  3. Teleport the robot to `pose` (upright, zero velocity), after the warm-up, because the layout rules
     only guarantee the poses are clear at time W. (Corrected 2026-11-19: the cylinders do NOT start at
     (0, 0); they start at their nested offsets, see worlds_v2. The teleport still resets any drift.)
  4. Wait SETTLE_SIM seconds of simulated time and for a fresh LiDAR/odometry message, set the goal and
     anchor the odometry (env.begin_episode without its wall-clock wait).
Returns the simulated time at which the episode starts (logged, to verify the phase).
v1 code (common.prepare_episode, evaluate.prepare_episode) is unchanged.
"""
import math
import time

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from dqn_comparison.worlds_v2 import LAYOUTS

ROBOT_NAME = 'turtlebot3_burger'
SETTLE_SIM = 0.2          # simulated seconds after the teleport, for fresh sensor data
WALL_TIMEOUT = 30.0       # wall-clock safety limit for each wait


def _call(env, client, request, what):
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f'{what}: service unavailable')
    future = client.call_async(request)
    deadline = time.monotonic() + 5.0
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(env, timeout_sec=0.01)
    if not future.done() or future.result() is None:
        raise RuntimeError(f'{what}: no reply')
    return future.result()


def _spin_until(env, condition, what):
    deadline = time.monotonic() + WALL_TIMEOUT
    while not condition():
        if time.monotonic() >= deadline or not rclpy.ok():
            raise RuntimeError(f'timed out waiting for {what}')
        rclpy.spin_once(env, timeout_sec=env.sim_poll)


def reset_to_phase(env, w):
    """Reset Gazebo and wait until simulated time >= w. Returns nothing; raises on failure."""
    if not env.sim_time_stepping:
        raise RuntimeError('prepare_episode_v2 needs DQN_SIM_TIME_STEP=1 (simulated-time stepping)')
    if not hasattr(env, '_v2_reset_client'):
        env._v2_reset_client = env.create_client(Empty, '/reset_simulation')
        env._v2_state_client = env.create_client(SetEntityState, '/set_entity_state')
    env.set_physics_paused(False)                         # 2026-11-21: physics is paused between episodes
    env._wait_for_clock()
    t_before = env.sim_now
    env.publish_stop()
    _call(env, env._v2_reset_client, Empty.Request(), '/reset_simulation')
    if t_before > 0.05:                                   # confirm the clock jumped back
        _spin_until(env, lambda: env.sim_now < t_before - 0.01, 'the reset (simulated clock going back)')
    _spin_until(env, lambda: env.sim_now >= w, f'simulated time {w} s')
    env.publish_stop()


def teleport(env, x, y, yaw):
    req = SetEntityState.Request()
    req.state = EntityState()
    req.state.name = ROBOT_NAME
    req.state.reference_frame = 'world'
    req.state.pose.position.x = float(x)
    req.state.pose.position.y = float(y)
    req.state.pose.position.z = 0.01                      # upright, just above the ground
    req.state.pose.orientation.z = math.sin(yaw / 2.0)
    req.state.pose.orientation.w = math.cos(yaw / 2.0)
    req.state.twist = Twist()                             # no leftover velocity
    if not _call(env, env._v2_state_client, req, '/set_entity_state').success:
        raise RuntimeError('Gazebo refused to move the robot')


def prepare_episode_v2(env, stage, pose, goal):
    """Reset to the stage's fixed obstacle phase, place the robot, set the goal. Returns start sim time."""
    reset_to_phase(env, LAYOUTS[stage]['W'])
    teleport(env, *pose)
    t_tp, count = env.sim_now, env.sensor_update_count
    _spin_until(env, lambda: env.sim_now - t_tp >= SETTLE_SIM and env.sensor_update_count > count,
                'sensor data after the teleport')
    env.goal_x, env.goal_y = float(goal[0]), float(goal[1])
    env.begin_episode(start=pose, settle_sec=0.0)
    env.set_physics_paused(True)                          # 2026-11-21: the first action unpauses
    return env.sim_now

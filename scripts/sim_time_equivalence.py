"""Study v2, step B3: check that simulated-time stepping at 3x speed gives the same robot motion
as the v1 real-time, wall-clock setup (analysis only; trains nothing).

Plays one fixed random action sequence from P0 several times and logs, after every action, the
robot's world pose, the simulated time elapsed and the action's measured simulated length.
Run inside a Gazebo started with turtlebot3_drl_v2.launch.py (see scripts/run_sim_time_equivalence.sh).

  python3 scripts/sim_time_equivalence.py --mode x3_sim --out results_smoke/b3/x3_sim.csv
"""
import argparse
import csv
import math
import os
import random
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist

from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.common import wait_for_sensors

# v1 episode start (reset, teleport, settle), moved unchanged from v1 evaluate.py on 2026-11-21 so this
# check keeps comparing against exactly the v1 setup.
ROBOT_NAME = "turtlebot3_burger"
GOAL_X, GOAL_Y = 2.0, -2.0
SETTLE_SEC = 2.0


def reset_simulation(node):
    """Ask Gazebo to reset the world. Returns True only if a reset call succeeded."""
    for service_name in ("/reset_simulation", "/reset_world"):
        client = node.create_client(Empty, service_name)
        if not client.wait_for_service(timeout_sec=1.0):
            continue
        future = client.call_async(Empty.Request())
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if future.done() and future.result() is not None:
            return True
    return False

def teleport_robot(node, x, y, yaw):
    """Move the robot model in Gazebo. Raises if the service is missing or refuses."""
    client = node.create_client(SetEntityState, "/set_entity_state")
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("Gazebo /set_entity_state service unavailable")

    request = SetEntityState.Request()
    request.state = EntityState()
    request.state.name = ROBOT_NAME
    request.state.reference_frame = "world"
    request.state.pose.position.x = float(x)
    request.state.pose.position.y = float(y)
    request.state.pose.position.z = 0.01            # keep the base just above the ground
    request.state.pose.orientation.z = math.sin(yaw / 2.0)
    request.state.pose.orientation.w = math.cos(yaw / 2.0)
    request.state.twist = Twist()                   # no leftover velocity after the move

    future = client.call_async(request)
    deadline = time.monotonic() + 5.0
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not future.done():
        raise TimeoutError("Timed out repositioning the robot")
    response = future.result()
    if response is None or not response.success:
        raise RuntimeError("Gazebo refused the reposition request")

def settle(node, seconds):
    deadline = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

def prepare_episode(env, pose):
    """Reset Gazebo, teleport to the start pose, and re-anchor the pose tracking there."""
    if not reset_simulation(env):
        raise RuntimeError("Gazebo reset failed")
    x, y, yaw = pose
    teleport_robot(env, x, y, yaw)
    settle(env, SETTLE_SEC)
    wait_for_sensors(env)
    env.goal_x, env.goal_y = GOAL_X, GOAL_Y
    env.begin_episode(start=(x, y, yaw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--actions', type=int, default=30)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    actions = random.Random(args.seed).choices(range(7), k=args.actions)   # same sequence every run
    rclpy.init()
    env = RobotEnvironment()
    clock = {'t': float('nan')}

    def on_clock(msg):
        clock['t'] = msg.clock.sec + msg.clock.nanosec * 1e-9
    env.create_subscription(Clock, '/clock', on_clock,
                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
    rows = []
    try:
        wait_for_sensors(env)
        for rep in range(args.repeats):
            prepare_episode(env, (0.0, 0.0, 0.0))
            wall0, sim0 = time.monotonic(), clock['t']
            for i, a in enumerate(actions):
                env.step(a)
                x, y, yaw = env.get_world_pose()
                rows.append({'mode': args.mode, 'repeat': rep, 'step': i, 'action': a,
                             'x': round(x, 5), 'y': round(y, 5), 'yaw': round(yaw, 5),
                             'action_sim_s': round(env.last_action_sim_duration, 5),
                             'sim_elapsed_s': round(clock['t'] - sim0, 4),
                             'wall_elapsed_s': round(time.monotonic() - wall0, 4)})
            env.get_logger().info(f'{args.mode} repeat {rep}: end pose ({x:.3f}, {y:.3f}), '
                                  f'wall {rows[-1]["wall_elapsed_s"]:.1f} s, sim {rows[-1]["sim_elapsed_s"]:.1f} s')
    finally:
        env.publish_stop()
        env.destroy_node()
        rclpy.shutdown()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()

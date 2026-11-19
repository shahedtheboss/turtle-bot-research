"""Study v2, step B3: check that simulated-time stepping at 3x speed gives the same robot motion
as the v1 real-time, wall-clock setup (analysis only; trains nothing).

Plays one fixed random action sequence from P0 several times and logs, after every action, the
robot's world pose, the simulated time elapsed and the action's measured simulated length.
Run inside a Gazebo started with turtlebot3_drl_v2.launch.py (see scripts/run_sim_time_equivalence.sh).

  python3 scripts/sim_time_equivalence.py --mode x3_sim --out results_smoke/b3/x3_sim.csv
"""
import argparse
import csv
import os
import random
import time

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.evaluate import prepare_episode, wait_for_sensors


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

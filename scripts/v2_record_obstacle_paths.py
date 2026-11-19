"""Study v2, step B4b: record the REAL obstacle motion from Gazebo after a reset (analysis only).

Stages 4 and 6: every animated obstacle publishes nav_msgs/Odometry on /obstacle/odom (gazebo_ros_p3d,
50 Hz, world frame, simulated-time stamp, child frame link_obstacleN). Stage 3: the rotating group has
no odometry, so its yaw is read with /get_entity_state, bracketed by the simulated clock before and
after each call (t = midpoint, uncertainty = half the bracket).
Writes CSV rows: stage, name, t (sim s since reset), x, y, yaw, t_err.

  DQN_SIM_TIME_STEP=1 python3 scripts/v2_record_obstacle_paths.py --stage 4 --seconds 400 --out FILE.csv
"""
import argparse
import csv
import math

import rclpy
from gazebo_msgs.srv import GetEntityState
from nav_msgs.msg import Odometry

from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.episode_v2 import reset_to_phase
from dqn_comparison.evaluate import wait_for_sensors


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def stage_arg(v):
    return v if v == '4s' else int(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=stage_arg, required=True)
    ap.add_argument('--seconds', type=float, default=400.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    rclpy.init()
    env = RobotEnvironment()
    rows = []
    t_reset = {'t': None}

    def on_odom(msg):
        if t_reset['t'] is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose
        rows.append({'stage': a.stage, 'name': msg.child_frame_id.split('::')[-1].replace('link_', ''), 't': round(t, 4),
                     'x': round(p.position.x, 4), 'y': round(p.position.y, 4),
                     'yaw': round(yaw_of(p.orientation), 4), 't_err': 0.0})

    env.create_subscription(Odometry, '/obstacle/odom', on_odom, 100)
    cli = env.create_client(GetEntityState, '/get_entity_state')
    try:
        wait_for_sensors(env)
        reset_to_phase(env, 0.0)
        t_reset['t'] = 0.0
        while env.sim_now < a.seconds:
            if a.stage == 3:
                t0 = env.sim_now
                req = GetEntityState.Request()
                req.name = 'turtlebot3_drl_obstacles'
                f = cli.call_async(req)
                rclpy.spin_until_future_complete(env, f, timeout_sec=5)
                t1 = env.sim_now
                q = f.result().state.pose.orientation
                rows.append({'stage': 3, 'name': 'group', 't': round((t0 + t1) / 2, 4), 'x': 0.0, 'y': 0.0,
                             'yaw': round(yaw_of(q), 4), 't_err': round((t1 - t0) / 2, 4)})
            else:
                rclpy.spin_once(env, timeout_sec=0.01)
    finally:
        env.destroy_node()
        rclpy.shutdown()
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['stage', 'name', 't', 'x', 'y', 'yaw', 't_err'])
            w.writeheader()
            w.writerows(rows)
        print('written', a.out, len(rows), 'rows')


if __name__ == '__main__':
    main()

"""Study v2, step B5: check prepare_episode_v2 in a running Gazebo (analysis only).

For each pose of the stage (start + 4 test poses) and each repeat: prepare the episode, then compare
  - robot pose in Gazebo (ground truth) with the requested pose, and its tilt (must be upright);
  - every obstacle's REAL position (/obstacle/odom; stage 3: group yaw) with worlds_v2.obstacles_at(stage, t),
    the recorded paths (phase check), and the real clearance between the robot and every obstacle;
  - nearest LiDAR range at the start (must be above the 0.15 m collision distance).
  DQN_SIM_TIME_STEP=1 python3 scripts/v2_reset_check.py --stage 4 --repeats 3 --out results_smoke/b5/stage4.csv
"""
import argparse
import csv
import math
import os
import re

import rclpy
from gazebo_msgs.srv import GetEntityState
from nav_msgs.msg import Odometry

from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.episode_v2 import prepare_episode_v2, _spin_until
from dqn_comparison.evaluate import wait_for_sensors
from dqn_comparison import worlds_v2 as W

WORLD_DIR = '/home/shahed/turtle-bot-research/install/turtlebot3_gazebo/share/turtlebot3_gazebo/worlds'


def entity(env, client, name):
    req = GetEntityState.Request()
    req.name = name
    f = client.call_async(req)
    rclpy.spin_until_future_complete(env, f, timeout_sec=10)
    s = f.result().state
    q = s.pose.orientation
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return s.pose.position.x, s.pose.position.y, yaw, tilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lay = W.LAYOUTS[a.stage]
    world = open(f'{WORLD_DIR}/turtlebot3_drl_stage{a.stage}/burger.model').read()
    movers = re.findall(r'<model name="([^"]+)">\s*<plugin', world)

    rclpy.init()
    env = RobotEnvironment()
    cli = env.create_client(GetEntityState, '/get_entity_state')
    names = W.WORLDS[a.stage]['anim']
    odom = {}

    def on_odom(m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        odom[m.child_frame_id.split('::')[-1].replace('link_', '')] = ((m.pose.pose.position.x, m.pose.pose.position.y), t)
    env.create_subscription(Odometry, '/obstacle/odom', on_odom, 100)
    rows = []
    try:
        wait_for_sensors(env)
        for rep in range(a.repeats):
            for k, pose in enumerate([lay['start']] + lay['tests']):
                t0 = prepare_episode_v2(env, a.stage, pose, lay['goals'][0])
                x, y, yaw, tilt = entity(env, cli, 'turtlebot3_burger')
                t_check = env.sim_now
                pred = W.obstacles_at(a.stage, t_check)
                if a.stage == 3:                        # rotating group: compare its yaw with the recording
                    _, _, gyaw, _ = entity(env, cli, movers[0])
                    obs_err = abs((math.degrees(gyaw - W.recorded_group_yaw(t_check)) + 180) % 360 - 180)
                    real = pred
                elif movers:                            # real cylinders from /obstacle/odom (latest message)
                    _spin_until(env, lambda: all(odom.get(n, (None, -1))[1] >= t_check for n in names),
                                'fresh obstacle odometry')
                    real = [(odom[n][0][0], odom[n][0][1], W.ANIM_RADIUS) for n in names]
                    pred = [W.obstacles_at(a.stage, odom[n][1])[i] for i, n in enumerate(names)]
                    obs_err = max(math.hypot(r[0] - p[0], r[1] - p[1]) for r, p in zip(real, pred))
                else:
                    obs_err, real = float('nan'), pred
                clearance = min((math.hypot(ox - pose[0], oy - pose[1]) - r for ox, oy, r in real), default=float('inf'))
                state = env.get_state()
                rows.append({'stage': a.stage, 'repeat': rep, 'pose': k, 't_start': round(t0, 4),
                             'pos_err_m': round(math.hypot(x - pose[0], y - pose[1]), 4),
                             'yaw_err_deg': round(abs((math.degrees(yaw - pose[2]) + 180) % 360 - 180), 2),
                             'tilt_deg': round(tilt, 2), 'obstacle_err': round(obs_err, 4), 'clearance_m': round(clearance, 3),
                             'min_lidar_m': round(min(state[:50]), 3), 'goal_dist_m': round(state[50], 3)})
                env.get_logger().info(str(rows[-1]))
    finally:
        env.publish_stop()
        env.destroy_node()
        rclpy.shutdown()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()

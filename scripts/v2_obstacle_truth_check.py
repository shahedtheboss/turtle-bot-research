"""Study v2, B4b: which obstacle position does the robot's LiDAR actually see? (analysis only)

Two Gazebo sources disagree by the <include> offset: /get_entity_state (model pose) and /obstacle/odom
(gazebo_ros_p3d link pose). For several times after a stage-4 reset, the robot is teleported to
fixed points; for each candidate obstacle position we predict the LiDAR range in that direction and
compare it with the measured scan. The source whose prediction matches the scan is the truth.
  DQN_SIM_TIME_STEP=1 python3 scripts/v2_obstacle_truth_check.py
"""
import math

import rclpy
from gazebo_msgs.srv import GetEntityState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.episode_v2 import reset_to_phase, teleport, _spin_until
from dqn_comparison.evaluate import wait_for_sensors

R_OBS = 0.16


def main():
    rclpy.init()
    env = RobotEnvironment()
    scan = {'m': None}
    odom = {}
    env.create_subscription(LaserScan, '/scan', lambda m: scan.__setitem__('m', m), qos_profile_sensor_data)
    env.create_subscription(Odometry, '/obstacle/odom',
                            lambda m: odom.__setitem__(m.child_frame_id.split('::')[-1].replace('link_', ''),
                                                       (m.pose.pose.position.x, m.pose.pose.position.y)), 100)
    cli = env.create_client(GetEntityState, '/get_entity_state')

    def model_pos(name):
        req = GetEntityState.Request()
        req.name = name
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(env, f, timeout_sec=5)
        p = f.result().state.pose.position
        return p.x, p.y

    def predicted_range(rx, ry, ox, oy):
        d = math.hypot(ox - rx, oy - ry)
        return d - R_OBS, math.atan2(oy - ry, ox - rx)      # surface distance, bearing (robot yaw = 0)

    def measured(bearing):
        m = scan['m']
        i = round((bearing - m.angle_min) / m.angle_increment) % len(m.ranges)
        window = [m.ranges[(i + k) % len(m.ranges)] for k in (-1, 0, 1)]
        return min(r for r in window if math.isfinite(r)) if any(math.isfinite(r) for r in window) else float('inf')

    try:
        wait_for_sensors(env)
        reset_to_phase(env, 0.0)
        for t_target, (rx, ry) in [(15.0, (0.0, 0.0)), (30.0, (-1.0, 0.5)), (60.0, (0.0, 0.0)), (100.0, (1.0, 0.5))]:
            _spin_until(env, lambda: env.sim_now >= t_target, f't={t_target}')
            teleport(env, rx, ry, 0.0)
            t_tp = env.sim_now
            _spin_until(env, lambda: env.sim_now - t_tp >= 0.3, 'settle')
            for name in ('obstacle1', 'obstacle2'):
                mp = model_pos(f'turtlebot3_drl_{name}')
                op = odom.get(name)
                for label, (ox, oy) in (('get_entity_state', mp), ('obstacle/odom', op)):
                    rng, brg = predicted_range(rx, ry, ox, oy)
                    meas = measured(brg)
                    print(f't={env.sim_now:6.1f} robot ({rx:4.1f},{ry:4.1f}) {name} {label:16s} at ({ox:5.2f},{oy:5.2f}): '
                          f'predicted range {rng:5.2f} m, LiDAR in that direction {meas:5.2f} m')
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

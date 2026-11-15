import math
import time
import rclpy
import random
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState

class RobotEnvironment(Node):


    NUM_SECTORS = 50
    MAX_LIDAR = 3.5

    def __init__(self):
        super().__init__('robot_environment')

        self.sensor_update_count = 0

        self.goal_x = 2.0
        self.goal_y = -2.0

        #Initial termination thresholds. 
        self.goal_tolerance = 0.20
        self.collision_distance = 0.15

        self.goal_reward = 100.0
        self.collision_penalty = -25.0
        self.step_penalty = -0.1
        self.progress_gain = 5.0
        self.heading_penalty = 0.05
        self.warning_distance = 0.30

        self.action_duration = 0.5

        self.start_world = (0.0, 0.0, 0.0)
        self.odom_origin = None

        #No-progress cutoff: episode ends not a true terminal
        self.best_goal_distance = None
        self.steps_since_progress = 0
        # Terminate if no progress for 60 steps
        self.max_no_progress_steps = 60
        self.min_progress = 0.05

        #Diagnostic for training log
        self.last_goal_distance = float('nan')
        self.last_min_lidar = float('nan')

        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')

        # Define the actions available to our future DQN agent.
        self.actions = {
            0: {'name': 'forward',      'linear_x': 0.15, 'angular_z':  0.0},
            1: {'name': 'soft_forward_left', 'linear_x': 0.11, 'angular_z':  0.4},
            2: {'name': 'soft_forward_right','linear_x': 0.11, 'angular_z': -0.4},
            3: {'name': 'forward_left', 'linear_x': 0.08, 'angular_z':  0.8},
            4: {'name': 'forward_right','linear_x': 0.08, 'angular_z': -0.8},
            5: {'name': 'hard_left',    'linear_x': 0.02, 'angular_z':  1.0},
            6: {'name': 'hard_right',   'linear_x': 0.02, 'angular_z': -1.0},
            
        }

        self.latest_scan = None
        self.latest_odom = None

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.get_logger().info(
            f'Environment started. Fixed goal: '
            f'({self.goal_x:.2f}, {self.goal_y:.2f})'
        )

    def scan_callback(self, msg):
        self.latest_scan = self.process_scan(msg)
        self.sensor_update_count += 1

    def odom_callback(self, msg):
        self.sensor_update_count += 1
        self.latest_odom = msg

    def process_scan(self, msg):

        num_sectors = self.NUM_SECTORS

        # ---------------- CHANGED: LaserScan field is range_max; fallback uses 3.5 m ----------------
        max_range = msg.range_max

        if not math.isfinite(max_range):

            self.get_logger().error(
                'LaserScan range_max is not finite. Using default value of 3.5 meters.'
            )

            max_range = self.MAX_LIDAR
        # ---------------- END CHANGED ----------------

        sector_width = (2.0 * math.pi) / num_sectors

        sector_readings = [[] for _ in range(num_sectors)
        ]

        for i, distance in enumerate(msg.ranges):

            angle = msg.angle_min + i * msg.angle_increment

            angle = angle  % (2.0 * math.pi)

            sector_index = int(angle / sector_width)

            sector_index = min(sector_index, num_sectors - 1)

            #Skip invalid readings
            if math.isfinite(distance) and distance > 0:
                distance = min(distance, max_range)
                sector_readings[sector_index].append(min(distance, max_range))

        return [min(r) if r else max_range for r in sector_readings]

    @staticmethod
    def _yaw(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def get_world_pose(self):

        p = self.latest_odom.pose.pose
        ox, oy, oyaw = p.position.x, p.position.y, self._yaw(p.orientation)

        if self.odom_origin is None:
            self.odom_origin = (ox, oy, oyaw)
        x0, y0, yaw0 = self.odom_origin

        # Motion since episode start, expressed in the starting frame.
        dx, dy = ox - x0, oy - y0
        c, s = math.cos(-yaw0), math.sin(-yaw0)
        rx, ry = c * dx - s * dy, s * dx + c * dy
 
        # Place that motion in the world frame, from the true start pose.
        sx, sy, syaw = self.start_world
        c, s = math.cos(syaw), math.sin(syaw)
        wx = sx + c * rx - s * ry
        wy = sy + s * rx + c * ry
        yaw = oyaw - yaw0 + syaw
        return wx, wy, math.atan2(math.sin(yaw), math.cos(yaw))

    def get_goal_information(self):
        x, y, yaw = self.get_world_pose()
        dx, dy = self.goal_x - x, self.goal_y - y
        bearing = math.atan2(dy, dx)
        relative_angle = math.atan2(
            math.sin(bearing - yaw), math.cos(bearing - yaw)
        )
        return math.hypot(dx, dy), relative_angle

    def begin_episode(self, start=(0.0, 0.0, 0.0), settle_sec=0.5):

        end = time.monotonic() + settle_sec
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
 
        self.start_world = tuple(float(v) for v in start)
        p = self.latest_odom.pose.pose
        self.odom_origin = (p.position.x, p.position.y, self._yaw(p.orientation))
        self.best_goal_distance = None
        self.steps_since_progress = 0

    def get_state(self):
        """Build a 52-feature state: 52 LiDAR sectors + goal distance/angle."""

        if self.latest_scan is None or self.latest_odom is None:
            return None

        goal_distance, relative_angle = self.get_goal_information()

        if len(self.latest_scan) != self.NUM_SECTORS:
            raise ValueError(
                f"Expected {self.NUM_SECTORS} LiDAR sectors, got {len(self.latest_scan)}"
            )

        state = self.latest_scan + [goal_distance, relative_angle]

        assert len(state) == 52
        return state

    def calculate_reward(self, previous_goal_distance):

        state = self.get_state()

        if state is None:
            return None, False, "waiting for sensor data"

        lidar = state[:50]
        goal_distance = state[50]
        relative_angle = state[51]
        nearest = min(lidar)

        if goal_distance <= self.goal_tolerance:
            return self.goal_reward, True, "goal_reached"
        if nearest <= self.collision_distance:
            return self.collision_penalty, True, "collision_warning"
 
        progress = previous_goal_distance - goal_distance
        reward = self.step_penalty
        reward += self.progress_gain * progress
        reward -= self.heading_penalty * abs(relative_angle)
 
        # Early warning before a collision (0 at 0.30 m, about -1 at 0.15 m).
        if nearest < self.warning_distance:
            reward -= (
                (self.warning_distance - nearest)
                / (self.warning_distance - self.collision_distance)
            )
        return reward, False, "continue"

    def publish_stop(self):
        """Publish zero velocity to stop the robot."""
        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0
        self.cmd_vel_pub.publish(command)

    def execute_action_for_duration(self, action_id, duration=None):
        
        if duration is None:
            duration = self.action_duration

        # Validate the requested action.
        if action_id not in self.actions:
            raise ValueError(f"Invalid action ID: {action_id}")

        if duration <= 0.0:
            raise ValueError("Duration must be positive.")

        action = self.actions[action_id]

        command = Twist()
        command.linear.x = float(action['linear_x'])
        command.angular.z = float(action['angular_z'])

        self.cmd_vel_pub.publish(command)

        end_time = time.monotonic() + duration

        try:
            # Continue processing LiDAR and odometry callbacks
            # while the action is being executed.
            while time.monotonic() < end_time and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)

        finally:
            #  Stop the robot when the action ends.
            # Do not publish the movement command again afterward.
            self.publish_stop()

    def normalize(self, s):
        lidar = [min(d, self.MAX_LIDAR) / self.MAX_LIDAR for d in s[:50]]
        goal_dist = min(s[50] / 6.0, 1.0)     # 6.0 = rough max distance in the arena
        return lidar + [goal_dist, s[51] / math.pi]

    def step(self, action_id):
        """Execute an action and return the new state, reward, and done flag."""

        state  = self.get_state()
        if state is None:
            raise RuntimeError("State is not available yet. Waiting for sensor data.")
        
        previous_goal_distance = state[50]

        #Record the current sensor message count before moving
        previous_sensor_count = self.sensor_update_count

        self.execute_action_for_duration(action_id)

        deadline = time.monotonic() + 0.1

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

            if self.sensor_update_count > previous_sensor_count:
                break

        next_state = self.get_state()

        if next_state is None:
            raise RuntimeError("State is not available after action execution. Waiting for sensor data.")

        reward, terminated, reason = self.calculate_reward(previous_goal_distance)

        goal_distance = next_state[50]
        self.last_goal_distance = goal_distance
        self.last_min_lidar = min(next_state[:50])

        # No-progress cut-off (the training script treats this as truncation).
        if not terminated:

            if (self.best_goal_distance is None or goal_distance < self.best_goal_distance - self.min_progress):
                self.best_goal_distance = goal_distance
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1  
                if self.steps_since_progress >= self.max_no_progress_steps:
                    terminated, reason = True, "no_progress"

        return self.normalize(next_state), reward, terminated, reason


    # ---------------- CHANGED: Gazebo entity is 'turtlebot3_burger' (verified with GetEntityState) ----------------
    def set_start(self, x, y, yaw, model_name='turtlebot3_burger'):
    # ---------------- END CHANGED ----------------

        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('SetEntityState service not available.')
            return False
        req = SetEntityState.Request()
        req.state.name = model_name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.set_state_client.call_async(req)
        deadline = time.monotonic() + 3.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        # ---------------- CHANGED: success needs a reply with success=True, not just a finished call ----------------
        ok = future.done() and future.result() is not None and future.result().success
        if ok:
            self.begin_episode(start=(x, y, yaw))
        else:
            self.get_logger().error(f'set_start failed for entity {model_name!r}.')

        return ok
        # ---------------- END CHANGED ----------------

def main(args=None):
    rclpy.init(args=args)
    node = RobotEnvironment()

    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and node.get_state() is None:
            if time.monotonic() >= deadline:
                node.get_logger().error("Sensor timeout. Cancelling movement.")
                return
            rclpy.spin_once(node, timeout_sec=0.1)

        node.begin_episode(start=(0.0, 0.0, 0.0))
        previous = node.get_state()[50]
        node.get_logger().info(f"Start goal distance: {previous:.3f} m")

        for i in range(5):
            _, reward, terminated, reason = node.step(0)
            current = node.last_goal_distance
            node.get_logger().info(
                f"step {i + 1}: goal_dist={current:.3f} "
                f"(change {previous - current:+.3f} m) reward={reward:.2f} "
                f"min_lidar={node.last_min_lidar:.2f} {reason}"
            )
            previous = current
            if terminated:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

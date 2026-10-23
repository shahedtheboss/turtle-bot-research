import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class RobotEnvironment(Node):

    def __init__(self):
        super().__init__('robot_environment')

        self.sensor_update_count = 0

        self.goal_x = 2.0
        self.goal_y = -2.0

        # Initial termination thresholds.
        self.goal_tolerance = 0.20       # Goal reached within 20 cm
        self.collision_distance = 0.20   # LiDAR-based obstacle warning

        # Define the actions available to our future DQN agent.
        # Format: action_id -> (linear velocity, angular velocity)
        # These values define commands only; they do not move the robot yet.

        self.actions = {
            0: {'name': 'forward',      'linear_x': 0.15, 'angular_z':  0.0},
            1: {'name': 'forward_left', 'linear_x': 0.15, 'angular_z':  0.4},
            2: {'name': 'forward_right','linear_x': 0.15, 'angular_z': -0.4},
            3: {'name': 'hard_left',    'linear_x': 0.05, 'angular_z':  0.7},
            4: {'name': 'hard_right',   'linear_x': 0.05, 'angular_z': -0.7},
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
        self.print_observation()

    def odom_callback(self, msg):
        self.sensor_update_count += 1
        self.latest_odom = msg

    def process_scan(self, msg):
        directions = [
            0.0,                 # Front
            math.pi / 4,         # Front-left
            -math.pi / 4,        # Front-right
            math.pi / 2,         # Left
            -math.pi / 2         # Right
        ]

        features = []

        for target_angle in directions:
            readings = []

            for i, distance in enumerate(msg.ranges):
                angle = msg.angle_min + i * msg.angle_increment

                difference = abs(math.atan2(
                    math.sin(angle - target_angle),
                    math.cos(angle - target_angle)
                ))

                if difference <= math.radians(22.5):
                    if not math.isfinite(distance):
                        distance = msg.range_max

                    distance = max(msg.range_min, distance)
                    distance = min(msg.range_max, distance)
                    readings.append(distance)

            features.append(
                min(readings) if readings else msg.range_max
            )

        return features

    def get_goal_information(self):

        odom = self.latest_odom
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

        x = position.x
        y = position.y

        qx = orientation.x
        qy = orientation.y
        qz = orientation.z
        qw = orientation.w

        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(sin_yaw, cos_yaw)

        dx = self.goal_x - x
        dy = self.goal_y - y

        goal_distance = math.hypot(dx, dy)

        goal_bearing = math.atan2(dy, dx)

        relative_angle = math.atan2(
            math.sin(goal_bearing - yaw),
            math.cos(goal_bearing - yaw)
        )

        return goal_distance, relative_angle


    def get_state(self):
        """Build the seven-feature state for the future DQN."""

        if self.latest_scan is None or self.latest_odom is None:
            return None

        goal_distance, relative_angle = self.get_goal_information()

        # State order:
        # front, front-left, front-right, left, right,
        # goal distance, goal relative angle.
        state = self.latest_scan + [
            goal_distance,
            relative_angle
        ]

        return state

    def calculate_reward(self, previous_goal_distance):
        """
        Calculate reward after an action has been executed.

        Args:
            previous_goal_distance: Distance to the goal before
                                    the action was executed.

        Returns:
            reward: Numerical feedback for the agent.
            terminated: Whether the episode should end.
            reason: Why the episode ended, or 'continue'.
        """

        state = self.get_state()

        if state is None:
            return None, False, "waiting for sensor data"

        # My state layout:
        # [front, front-left, front-right, left, right,
        #  goal_distance, relative_goal_angle]

        obstacle_distances = state[:5]
        current_goal_distance = state[5]

        # First, check for a possible collision.
        # This is a LiDAR-based warning, not definitive
        # confirmation of a physical collision.
        if min(obstacle_distances) <= self.collision_distance:
            return -30.0, True, "collision_warning"

        # Next, check whether the goal has been reached.
        if current_goal_distance <= self.goal_tolerance:
            return 100.0, True, "goal_reached"

        # Give a small penalty for taking another step.
        reward = -0.1

        # Positive progress means the robot got closer.
        # Negative progress means it moved farther away.
        progress = previous_goal_distance - current_goal_distance

        reward += 7.0 * progress

        return reward, False, "continue"


    
    def print_observation(self):
        """Print the seven-feature state and robot motion."""

        # Build the state. If either sensor hasn't provided data,
        # get_state() returns None.
        state = self.get_state()

        if state is None:
            return

        odom = self.latest_odom
        position = odom.pose.pose.position
        twist = odom.twist.twist

        self.get_logger().info(
            f'State [5 LiDAR, goal distance, goal angle] = '
            f'{[round(v, 2) for v in state]} | '
            f'position=({position.x:.2f}, {position.y:.2f}) | '
            f'velocity=({twist.linear.x:.3f} m/s, '
            f'{twist.angular.z:.3f} rad/s)',
            throttle_duration_sec=2.0
        )


    def execute_action(self, action_id):
        """Send a velocity command to the robot based on the action ID."""

        if action_id not in self.actions:
            self.get_logger().error(f'Invalid action ID: {action_id}')
            return

        action = self.actions[action_id]

        command = Twist()
        command.linear.x = float(action['linear_x'])
        command.angular.z = float(action['angular_z'])

        self.cmd_vel_pub.publish(command)


    def publish_stop(self):
        """Publish zero velocity to stop the robot."""
        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0
        self.cmd_vel_pub.publish(command)



    def execute_action_for_duration(self, action_id, duration=1):
        """
        Execute one discrete action for a short duration,
        then publish a stop command.

        Call this from the main control loop, not from a
        sensor callback or while another spin() is running.
        """

        if action_id not in self.actions:
            raise ValueError(f"Invalid action ID: {action_id}")

        if duration <= 0.0:
            raise ValueError("Duration must be positive.")

        action = self.actions[action_id]

        command = Twist()
        command.linear.x = float(action['linear_x'])
        command.angular.z = float(action['angular_z'])

        self.get_logger().info(
            f"Executing action {action_id}: "
            f"{action['name']} for {duration:.2f} seconds"
        )

        self.cmd_vel_pub.publish(command)

        end_time = time.monotonic() + duration    #time.monotonic() measures elapsed time reliably.

        try:
            while time.monotonic() < end_time and rclpy.ok():
                # Keep processing LiDAR and odometry callbacks.
                rclpy.spin_once(self, timeout_sec=0.05)
        finally:
            # Stop even if the action loop exits unexpectedly.
            self.publish_stop()

        self.get_logger().info("Stop command published.")

        

    ##################

    def step(self, action_id):
        state = self.get_state()

        if state is None:
            raise RuntimeError("Sensor data is not ready.")

        previous_goal_distance = state[5]

        # Record the current sensor message count before moving.
        previous_update_count = self.sensor_update_count

        # Execute the selected action.
        self.execute_action_for_duration(action_id, duration=1.0)

        # Wait for new sensor data after the action.
        deadline = time.monotonic() + 2.0

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.sensor_update_count > previous_update_count:
                break

        # Get the latest observation.
        next_state = self.get_state()

        if next_state is None:
            raise RuntimeError("Could not obtain the next state.")

        reward, terminated, reason = self.calculate_reward(
            previous_goal_distance
        )

        return next_state, reward, terminated, reason
    ##################
  




def main(args=None):
    rclpy.init(args=args)
    node = RobotEnvironment()

    try:
        # Wait until both LiDAR and odometry are available.
        deadline = time.monotonic() + 5.0

        while rclpy.ok() and node.get_state() is None:
            if time.monotonic() >= deadline:
                node.get_logger().error(
                    "Sensor timeout. Cancelling movement."
                )
                return

            rclpy.spin_once(node, timeout_sec=0.1)

        before = node.latest_odom.pose.pose.position
        start_x, start_y = before.x, before.y

        node.get_logger().info(
            f"Before: x={start_x:.3f}, y={start_y:.3f}"
        )

        # One short forward action, followed by an automatic stop.
        node.execute_action_for_duration(0, duration=0.3)

        # Allow odometry to update.
        deadline = time.monotonic() + 1.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        after = node.latest_odom.pose.pose.position

        node.get_logger().info(
            f"After: x={after.x:.3f}, y={after.y:.3f}"
        )
        node.get_logger().info(
            f"Change: dx={after.x - start_x:.3f} m, "
            f"dy={after.y - start_y:.3f} m"
        )

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
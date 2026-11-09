import math
import time
import rclpy
import random
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState


class RobotEnvironment(Node):

    NUM_SECTORS = 50  # Number of LiDAR sectors for state representation
    MAX_LIDAR = 3.5  # Maximum LiDAR range in meters

    def __init__(self):
        super().__init__('robot_environment')

        self.sensor_update_count = 0

        self.goal_x = 2.0
        self.goal_y = -2.0

        # Initial termination thresholds.
        self.goal_tolerance = 0.20       # Goal reached within 20 cm
        self.collision_distance = 0.15   # LiDAR-based obstacle warning

        self.goal_reward = 100.0
        self.collision_penalty = -25.0
        self.step_penalty = -0.1    
        self.progress_gain = 5.0
        self.heading_penalty = 0.05
        self.warning_distance = 0.30

        self.action_duration = 0.5  # Duration for each action in seconds

        self.start_world = (0.0, 0.0, 0.0)  # Default starting position (x, y, yaw)
        self.odom_origin = None  # To store the initial odometry position for relative calculations

        #No-progress cutoff: episode ends not a true terminal
        self.best_goal_distance = None
        self.steps_since_progress = 0
        self.max_no_progress_steps = 60  # Terminate if no progress for 60 steps
        self.min_progress = 0.05

        #Diadnostic for training log
        self.last_goal_distance = float('nan')
        self.last_min_lidar = float('nan')

        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state')

        # Define the actions available to our future DQN agent.
        # Format: action_id -> (linear velocity, angular velocity)
        # These values define commands only; they do not move the robot yet.

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
        # self.print_observation()

    def odom_callback(self, msg):
        self.sensor_update_count += 1
        self.latest_odom = msg
    """
    # def process_scan(self, msg):
    #     directions = [
    #         0.0,                 # Front
    #         math.pi / 4,         # Front-left
    #         -math.pi / 4,        # Front-right
    #         math.pi / 2,         # Left
    #         -math.pi / 2         # Right
    #     ]

    #     features = []

    #     for target_angle in directions:
    #         readings = []

    #         for i, distance in enumerate(msg.ranges):
    #             angle = msg.angle_min + i * msg.angle_increment

    #             difference = abs(math.atan2(
    #                 math.sin(angle - target_angle),
    #                 math.cos(angle - target_angle)
    #             ))

    #             if difference <= math.radians(22.5):
    #                 if not math.isfinite(distance):
    #                     distance = msg.range_max

    #                 distance = max(msg.range_min, distance)
    #                 distance = min(msg.range_max, distance)
    #                 readings.append(distance)

    #         features.append(
    #             min(readings) if readings else msg.range_max
    #         )

    #     return features
    """

    
    def process_scan(self, msg):
        """
        Convert the raw LiDAR scan into exactly 20 features.

        INPUT:
            msg: A ROS 2 LaserScan message containing:
                - ranges: distance measurements from the LiDAR
                - angle_min: angle of the first measurement
                - angle_increment: angular difference between measurements
                - range_max: maximum measurement range

        OUTPUT:
            A Python list containing 20 values.
            Each value represents the minimum valid distance
            detected within one angular sector.

        WHY DO THIS?
            The raw LiDAR may provide hundreds of measurements.
            Feeding all of them directly into our DQN would require
            a much larger input layer.

            Instead, we compress the scan into 20 features so that
            our DQN receives a fixed-size representation of the
            obstacles surrounding the robot.
        """
        """
        # ---------------------------------------------------------
        # SECTION 1: Define the number of LiDAR features
        # ---------------------------------------------------------

        # We want to divide the LiDAR scan into 20 angular sectors.
        # Each sector will eventually contribute ONE number to our state.
        #
        # If the scan covers a complete 360-degree circle:
        #     360 degrees / 20 sectors = 18 degrees per sector.
        #
        # This means our LiDAR representation will contain 20 values,
        # regardless of how many raw measurements the sensor provides.
        """
        num_sectors = self.NUM_SECTORS  

        # Obtain the maximum range reported by this LiDAR message.
        # For example, if range_max is 3.5, the sensor's usable
        # maximum reported distance is 3.5 metres.
        #
        # We use this value when a sector has no valid measurements,
        # treating it as having no detected obstacle within range.
        max_range = msg.range_max

        """
        # ---------------------------------------------------------
        # SECTION 2: Validate the maximum range
        # ---------------------------------------------------------

        # math.isfinite(x) returns False if x is NaN, +infinity,
        # or -infinity. We also reject zero and negative values.
        #
        # Why check this?
        # A valid positive maximum range is necessary to represent
        # empty sectors meaningfully.
        #
        # Without this check, an invalid range could propagate into
        # the state and produce misleading sensor features.
        """
        if not math.isfinite(max_range) or max_range <= 0:

            # Log an error so we can diagnose a malformed sensor
            # configuration or message while debugging.
            self.get_logger().error(
                f"Invalid LiDAR maximum range: {max_range}"
            )

            # Return a list with exactly 20 elements to preserve
            # the expected output length.
            #
            # IMPORTANT:
            # These zeros are NOT genuine measurements. They can make
            # every sector appear dangerously close to an obstacle.
            # A more robust implementation should flag this scan as
            # invalid and prevent these values from being used as a
            # normal observation during training.
            return [0.0] * num_sectors

        """
        # ---------------------------------------------------------
        # SECTION 3: Calculate the angular width of each sector
        # ---------------------------------------------------------

        # math.pi represents pi, approximately 3.14159 radians.
        #
        # A complete circle is 2*pi radians, or 360 degrees.
        #
        # Dividing the complete circle by 20 gives the angular
        # width assigned to each sector.
        #
        # Numerically:
        #     sector_width = 2*pi / 20
        #                  = pi/10 radians
        #                  = 18 degrees
        #
        # NOTE:
        # This calculation assumes we want 20 equal sectors around
        # a complete circle. It does not verify that the actual
        # LiDAR message covers the full 360 degrees.
        """
        sector_width = (2.0 * math.pi) / num_sectors

        """
        # Create 20 separate empty lists.
        #
        # Example structure:
        #     [[], [], [], ..., []]
        #
        # Each inner list will collect the raw LiDAR measurements
        # that fall inside its corresponding angular sector.
        #
        # We collect readings first because a sector may contain
        # multiple measurements. After processing the entire scan,
        # we will select the minimum distance from each sector.
        #
        # The list comprehension runs once for every sector.
        """
        sector_readings = [[] for _ in range(num_sectors)]


        # ---------------------------------------------------------
        # SECTION 4: Process every raw LiDAR measurement
        # ---------------------------------------------------------

        # msg.ranges contains the raw distance measurements.
        #
        # enumerate() gives us two things for every measurement:
        #     i        -> its index: 0, 1, 2, 3, ...
        #     distance -> the measured distance at that index
        #
        # We need the index to calculate the measurement's angle.
        for i, distance in enumerate(msg.ranges):

            # Calculate the angle corresponding to this measurement.
            #
            # Formula:
            #     angle = angle_min + i * angle_increment
            #
            # angle_min is the angle of the first measurement.
            # angle_increment is the angular separation between
            # consecutive measurements.
            #
            # Example:
            #     angle_min = -180 degrees
            #     angle_increment = 1 degree
            #     i = 90
            #     angle = -180 + 90*1 = -90 degrees
            #
            # ROS LaserScan angles are expressed in radians.
            angle = msg.angle_min + i * msg.angle_increment


            # -----------------------------------------------------
            # SECTION 5: Normalize the angle
            # -----------------------------------------------------

            # The modulo operator (%) wraps the angle into the
            # interval [0, 2*pi).
            #
            # Examples:
            #     -90 degrees  -> 270 degrees
            #       0 degrees ->   0 degrees
            #     450 degrees ->  90 degrees
            #
            # Directions that differ by a complete rotation refer
            # to the same physical direction.
            #
            # Normalization makes it easier to assign angles to
            # sector indices using a consistent 0-to-360-degree
            # representation.
            angle = angle % (2.0 * math.pi)


            # -----------------------------------------------------
            # SECTION 6: Determine the sector for this measurement
            # -----------------------------------------------------

            # Divide the normalized angle by the sector width.
            #
            # Example:
            #     angle = 35 degrees
            #     sector_width = 18 degrees
            #     35 / 18 = 1.94
            #
            # int(1.94) gives 1, so this measurement belongs to
            # sector 1. Sector numbering starts at zero.
            sector_index = int(angle / sector_width)

            # Valid sector indices are 0 through 19.
            #
            # In theory, an angle strictly below 360 degrees should
            # produce an index no greater than 19. However, floating-
            # point calculations can introduce tiny rounding errors.
            #
            # min() ensures the index cannot exceed the last sector.
            sector_index = min(sector_index, num_sectors - 1)


            # -----------------------------------------------------
            # SECTION 7: Validate and store the distance
            # -----------------------------------------------------

            # Keep only finite, positive distance measurements.
            #
            # math.isfinite(distance) rejects NaN and infinity.
            # distance > 0 rejects zero and negative readings.
            #
            # Invalid readings are skipped instead of being stored
            # as if they were genuine obstacle distances.
            if math.isfinite(distance) and distance > 0:

                # Cap any reading above the declared maximum range.
                #
                # Example:
                #     distance = 4.0 metres
                #     max_range = 3.5 metres
                #     resulting distance = 3.5 metres
                #
                # This prevents a reading from exceeding the
                # configured maximum used by our representation.
                distance = min(distance, max_range)

                # Add this measurement to its assigned sector.
                #
                # We use append() because multiple raw measurements
                # can belong to the same sector.
                #
                # For example:
                #     sector_readings[1] = [0.8, 0.9, 1.1]
                #
                # We will choose the minimum only after all readings
                # have been processed.
                sector_readings[sector_index].append(min(distance, max_range))

        """
        # ---------------------------------------------------------
        # SECTION 8: Reduce each sector to one feature
        # ---------------------------------------------------------

        # We now have 20 lists of raw readings.
        # The DQN needs one value per sector, not every raw reading.
        #
        # For each sector:
        #
        #     If readings exist:
        #         min(readings) returns the nearest detected obstacle.
        #
        #     If no valid readings exist:
        #         use max_range as a fallback, representing that no
        #         obstacle was detected within the usable range.
        #
        # Example:
        #     [1.8, 0.7, 1.2, 0.9] -> 0.7
        #
        # The list comprehension repeats this operation for all
        # 20 sectors, producing exactly 20 features.
        """
        # features = [
        #     min(readings) if readings else max_range
        #     for readings in sector_readings
        # ]

        """
        # ---------------------------------------------------------
        # SECTION 9: Return the processed LiDAR features
        # ---------------------------------------------------------

        # Return the 20 sector-minimum distances to the caller.
        #
        # The scan_callback() stores this result in self.latest_scan.
        # Then get_state() appends:
        #     - goal distance
        #     - relative goal angle
        #
        # Final state size:
        #     20 LiDAR features + 2 goal features = 22 values
        #
        # This function returns only the LiDAR features, not the
        # complete DQN state.
        """
        # return features
        return [min(r) if r else max_range for r in sector_readings]

    @staticmethod
    def _yaw(q):
        """""
        Convert a quaternion to a yaw angle in radians.
        """
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def get_world_pose(self):
        """
        Get the robot's current position and orientation in the world frame.

        Returns:
            (x, y, yaw): Tuple containing the robot's x and y coordinates
                         and its yaw angle in radians.
        """
        p = self.latest_odom.pose.pose
        ox, oy, oyaw = p.position.x, p.position.y, self._yaw(p.orientation)

        if self.odom_origin is None:
            self.odom_origin = (ox, oy, oyaw)
        x0 , y0, yaw0 = self.odom_origin

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
    
    """
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
    """
    def begin_episode(self, start=(0.0, 0.0, 0.0), settle_sec=0.5):
        """
        Call right after every Gazebo reset or teleport.
        `start` is the TRUE world pose (x, y, yaw) of the robot now.
        """
        end = time.monotonic() + settle_sec
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
 
        self.start_world = tuple(float(v) for v in start)
        p = self.latest_odom.pose.pose
        self.odom_origin = (p.position.x, p.position.y, self._yaw(p.orientation))
        self.best_goal_distance = None
        self.steps_since_progress = 0
    """
    # def get_state(self):
    #     #Build the seven-feature state for the future DQN.#

    #     if self.latest_scan is None or self.latest_odom is None:
    #         return None

    #     goal_distance, relative_angle = self.get_goal_information()

    #     # State order:
    #     # front, front-left, front-right, left, right,
    #     # goal distance, goal relative angle.
    #     state = self.latest_scan + [
    #         goal_distance,
    #         relative_angle
    #     ]

    #     return state
    """

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
    """
    def calculate_reward(self, previous_goal_distance):
        
        # Calculate reward after an action has been executed.

        # Args:
        #     previous_goal_distance: Distance to the goal before
        #                             the action was executed.

        # Returns:
        #     reward: Numerical feedback for the agent.
        #     terminated: Whether the episode should end.
        #     reason: Why the episode ended, or 'continue'.
        #

        state = self.get_state()

        if state is None:
            return None, False, "waiting for sensor data"

        # My state layout:
        # [front, front-left, front-right, left, right,
        #  goal_distance, relative_goal_angle]

        obstacle_distances = state[:20]
        current_goal_distance = state[22]

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
    """


    def calculate_reward(self, previous_goal_distance):
        """
        Calculate the reward after the robot executes an action.

        State layout:
            state[0:20] -> 20 LiDAR sector distances
            state[20]   -> goal distance
            state[21]   -> relative goal angle
        """

        state = self.get_state()

        # Sensor data may not be ready yet.
        if state is None:
            return None, False, "waiting for sensor data"

        # Make sure the environment produces the expected state.
        # if len(state) != 52:
        #     raise ValueError(
        #         f"Expected 52 state values, got {len(state)}"
        #     )
        """ 
        # Extract the 20 LiDAR distances.
        obstacle_distances = state[:50]

        # Goal distance is the 21st value (index 20).
        current_goal_distance = state[50]

        # First, check for a LiDAR-based collision warning.
        # Check whether the robot reached the goal first.
        if current_goal_distance <= self.goal_tolerance:
            return 100.0, True, "goal_reached"

        # Otherwise, check for a LiDAR-based collision warning.
        if min(obstacle_distances) <= self.collision_distance:
            return -50.0, True, "collision_warning"

        # # Penalize each additional step slightly.
        reward = -0.1

        # # Reward progress toward the goal; penalize moving away.
        # progress = previous_goal_distance - current_goal_distance
        # if progress > 0:
        #     reward += 7.0 * progress
        # else:
        #     reward += 3.0 * progress
        
        # reward += 7.0 * progress

        progress = previous_goal_distance - current_goal_distance
        reward += 5.0 * progress

        # Small heading term: prefer facing the goal.
        reward -= 0.05 * abs(state[51])

        # Early warning before collision: 0 at 0.30 m, about -1 near 0.15 m.
        nearest = min(obstacle_distances)
        if nearest < 0.30:
            reward -= (0.30 - nearest) / 0.15

        self.get_logger().info(
            f"Goal distance: {current_goal_distance:.3f} m, "
            f"Relative angle: {state[21]:.3f} rad, "
            f"Min LiDAR distance: {min(obstacle_distances):.3f} m"
        )       
        """
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

        # self.get_logger().info(
        #     f'State [22 LiDAR, goal distance, goal angle] = '
        #     f'{[round(v, 2) for v in state]} | '
        #     f'position=({position.x:.2f}, {position.y:.2f}) | '
        #     f'velocity=({twist.linear.x:.3f} m/s, '
        #     f'{twist.angular.z:.3f} rad/s)',
        #     throttle_duration_sec=2.0
        # )

    """
    # def execute_action(self, action_id):
    #     #Send a velocity command to the robot based on the action ID.

    #     if action_id not in self.actions:
    #         self.get_logger().error(f'Invalid action ID: {action_id}')
    #         return

    #     action = self.actions[action_id]

    #     command = Twist()
    #     command.linear.x = float(action['linear_x'])
    #     command.angular.z = float(action['angular_z'])

    #     self.cmd_vel_pub.publish(command)
    """


    def publish_stop(self):
        """Publish zero velocity to stop the robot."""
        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0
        self.cmd_vel_pub.publish(command)



    def execute_action_for_duration(self, action_id, duration=None):
        """
        Execute a discrete action for a specified duration,
        process sensor callbacks while moving, and then stop.
        """

        if duration is None:
            duration = self.action_duration

        # SECTION 1: Validate the requested action.
        if action_id not in self.actions:
            raise ValueError(f"Invalid action ID: {action_id}")

        if duration <= 0.0:
            raise ValueError("Duration must be positive.")

        # SECTION 2: Build the velocity command for this action.
        action = self.actions[action_id]

        command = Twist()
        command.linear.x = float(action['linear_x'])
        command.angular.z = float(action['angular_z'])

        # self.get_logger().info(
        #     f"Executing action {action_id}: "
        #     f"{action['name']} for {duration:.2f} seconds"
        # )

        # SECTION 3: Publish the movement command.
        self.cmd_vel_pub.publish(command)

        # Use a monotonic clock to measure elapsed time.
        end_time = time.monotonic() + duration

        try:
            # Continue processing LiDAR and odometry callbacks
            # while the action is being executed.
            while time.monotonic() < end_time and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)

        finally:
            # SECTION 4: Stop the robot when the action ends.
            # Do not publish the movement command again afterward.
            self.publish_stop()

        # self.get_logger().info("Stop command published.")

    def normalize(self, s):
        lidar = [min(d, self.MAX_LIDAR) / self.MAX_LIDAR for d in s[:50]]
        goal_dist = min(s[50] / 6.0, 1.0)     # 6.0 = rough max distance in the arena
        return lidar + [goal_dist, s[51] / math.pi]

        

    ##################

    def step(self, action_id):
        state = self.get_state()

        if state is None:
            raise RuntimeError("Sensor data is not ready.")

        previous_goal_distance = state[50]

        # Record the current sensor message count before moving.
        previous_update_count = self.sensor_update_count

        # Execute the selected action.
        self.execute_action_for_duration(action_id)

        # Wait for new sensor data after the action.
        deadline = time.monotonic() + 0.1

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

            if self.sensor_update_count > previous_update_count:
                break

        # Get the latest observation.
        next_state = self.get_state()

        if next_state is None:
            raise RuntimeError("Could not obtain the next state.")

        reward, terminated, reason = self.calculate_reward(previous_goal_distance)

        # Update diagnostic information for training logs.
        goal_distance = next_state[50]
        self.last_goal_distance = goal_distance
        self.last_min_lidar = min(next_state[:50])
 
        # No-progress cut-off (the training script treats this as truncation).
        if not terminated:
            # Check if the robot has made sufficient progress toward the goal.
            if (self.best_goal_distance is None
                    or goal_distance < self.best_goal_distance - self.min_progress):
                self.best_goal_distance = goal_distance
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1
                if self.steps_since_progress >= self.max_no_progress_steps:
                    terminated, reason = True, "no_progress"

        return self.normalize(next_state), reward, terminated, reason
    ##################

    def set_start(self, x, y, yaw, model_name='burger'):
        """Teleport the robot (eval only)."""
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/set_entity_state not available')
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

        # Check if the service call completed successfully.
        ok = future.done()
        if ok:
            self.begin_episode(start=(x, y, yaw))

        return future.done()






def main(args=None):
    """Quick sanity test: five forward steps; goal distance should fall."""
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

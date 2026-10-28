
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import rclpy
from std_srvs.srv import Empty
from mini_dqn_nav.environment import RobotEnvironment
from mini_dqn_nav.agent import DQNAgent
from collections import Counter
from mini_dqn_nav.train import reset_simulation 
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist

EVAL_STARTS = [(-0.6, 0.9, 0.0)]
TRAIN_SPAWN_X, TRAIN_SPAWN_Y = 0.0, 0.0
GOAL_X, GOAL_Y = 2.0, -2.0

MODEL_PATH = os.path.expanduser(
    "~/ros2_ws/src/mini_dqn_nav/mini_dqn_nav/models/dqn_model_7actions_20_lidars.pth"
)
NUM_EPISODES = 30
MAX_STEPS = 100


 
def set_robot_start_position(env, x=-0.6, y=0.9, yaw=0.0):
    # Create a ROS 2 client for Gazebo's SetEntityState service.
    # A service request lets us set the simulated model's pose and twist.
    client = env.create_client(
        SetEntityState, "/set_entity_state"
    )
 
    # Wait at most 5 seconds for Gazebo to offer this service.
    # If it is unavailable, stop rather than pretending the robot was moved.
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("Gazebo set_entity_state service unavailable")
 
    # Build a message describing the desired state of the robot model.
    state = EntityState()
 
    # This name must match the entity name in the running Gazebo world.
    state.name = "turtlebot3_burger"
 
    # Express the requested pose relative to the world coordinate frame.
    state.reference_frame = "world"
 
    # Set the robot's desired world position in metres.
    state.pose.position.x = x
    state.pose.position.y = y
 
    # Keep the robot slightly above z=0 to avoid placing its base below the
    # simulated ground. The appropriate value depends on the robot/world.
    state.pose.position.z = 0.01
 
    # Convert the requested planar yaw angle into the z and w components of
    # a quaternion. For rotation about the vertical axis:
    #   qz = sin(yaw/2)
    #   qw = cos(yaw/2)
    # x and y quaternion components remain zero for this planar rotation.
    state.pose.orientation.z = np.sin(yaw / 2.0)
    state.pose.orientation.w = np.cos(yaw / 2.0)
 
    # Set linear and angular velocities to zero so the robot does not retain
    # a previous motion command when teleported.
    state.twist = Twist()
 
    # Put the desired robot state into a service request.
    request = SetEntityState.Request()
    request.state = state
 
    # Send the request asynchronously. The call returns a future, which will
    # become complete when the service response arrives.
    future = client.call_async(request)
 
    # Set a deadline so this function cannot wait forever for the response.
    deadline = time.monotonic() + 5.0
 
    # Keep processing ROS callbacks until the request completes or times out.
    while rclpy.ok() and not future.done():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out repositioning robot")
 
        # Process ROS events for up to 0.1 seconds, allowing the service
        # response and other callbacks to be handled.
        rclpy.spin_once(env, timeout_sec=0.1)
 
    # Retrieve the service response after the future completes.
    response = future.result()
 
    # A completed request is not necessarily a successful request, so check
    # both that a response exists and that Gazebo reports success.
    if response is None or not response.success:
        raise RuntimeError("Failed to reposition robot")
 
    # Log the requested position for debugging and experiment records.
    env.get_logger().info(
        f"Robot repositioned to ({x:.2f}, {y:.2f})"
    )
 
 

def reset_simulation(node):
    for service_name in ("/reset_simulation", "/reset_world"):
        client = node.create_client(Empty, service_name)
        if not client.wait_for_service(timeout_sec=1.0):
            continue
        future = client.call_async(Empty.Request())
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        if future.done() and future.result() is not None:
            node.get_logger().info(f"Gazebo reset using {service_name}.")
            return True
    node.get_logger().warning("Could not reset Gazebo; check that it is running.")
    return False





def wait_for_state(env, timeout=10.0):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and env.get_state() is None:
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for LiDAR/odometry data.")
        rclpy.spin_once(env, timeout_sec=0.1)


def align_goal_to_start(env, x, y, yaw):
    odom = env.latest_odom.pose.pose.position

    # If odom follows the world pose, the goal in the odom frame is unchanged
    if np.hypot(odom.x - x, odom.y - y) < np.hypot(odom.x, odom.y):
        env.goal_x, env.goal_y = GOAL_X, GOAL_Y
        return

    # Otherwise odom restarted at (0, 0, 0) at the teleport pose, so express
    # the training goal (world position) in that new odom frame
    dx = (TRAIN_SPAWN_X + GOAL_X) - x
    dy = (TRAIN_SPAWN_Y + GOAL_Y) - y
    env.goal_x = float(np.cos(yaw) * dx + np.sin(yaw) * dy)
    env.goal_y = float(-np.sin(yaw) * dx + np.cos(yaw) * dy)
    env.get_logger().info(
        f"Goal in odom frame for this start: ({env.goal_x:.2f}, {env.goal_y:.2f})"
    )


def load_policy(agent, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        weights = checkpoint["policy_net"]
    elif isinstance(checkpoint, dict) and "policy_net_state_dict" in checkpoint:
        weights = checkpoint["policy_net_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    else:
        weights = checkpoint
    agent.policy_net.load_state_dict(weights)
    agent.policy_net.eval()
    agent.epsilon = 0.0


def plot_results(results):
    episodes = [r["episode"] for r in results]
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]
    labels = ["Success", "Collision", "Timeout", "Other"]
    counts = {label: 0 for label in labels}
    cumulative = {label: [] for label in labels}

    for result in results:
        reason = result["outcome"].lower()
        if "goal" in reason or "success" in reason:
            counts["Success"] += 1
        elif "collision" in reason or "tumble" in reason:
            counts["Collision"] += 1
        elif "timeout" in reason or "max_steps" in reason:
            counts["Timeout"] += 1
        else:
            counts["Other"] += 1
        for label in labels:
            cumulative[label].append(counts[label])

    window = min(10, len(rewards))
    moving_avg = [
        float(np.mean(rewards[max(0, i - window + 1):i + 1]))
        for i in range(len(rewards))
    ]

    # DQN has one Q-network and a TD loss, not separate actor/critic losses.
    # Since this is evaluation only, loss is not calculated or fabricated.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    success_count = sum(
        1 for r in results
        if "goal" in r["outcome"].lower()
        or "success" in r["outcome"].lower()
    )
    success_rate = 100.0 * success_count / len(results) if results else 0.0

    fig.suptitle(
        f"Evaluation Accuracy (Success Rate): {success_rate:.1f}%"
        f" | TurtleBot3 DQN — Greedy Evaluation",
        fontsize=15
    )

    ax = axes[0, 0]
    for label in labels:
        if counts[label] > 0 or label == "Success":
            ax.plot(episodes, cumulative[label], label=label)
    ax.set_title("Cumulative evaluation outcomes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodes")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(episodes, moving_avg, label=f"Moving average (window={window})")
    ax.set_title("Average reward over evaluation episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(episodes, steps, label="Steps per episode")
    ax.set_title("Steps per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    bars = ax.bar(labels, [counts[label] for label in labels])
    ax.set_title("Evaluation outcome counts")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Episodes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars)
    fig.tight_layout()
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    env = None
    results = []
    try:
        env = RobotEnvironment()

        if not reset_simulation(env):
            raise RuntimeError("Initial Gazebo reset failed.")

        # 3. Allow sensor callbacks to update after reset
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(env, timeout_sec=0.1)

        # 4. Wait until LiDAR/odometry state is available
        wait_for_state(env)
            
        
        agent = DQNAgent(state_size=52, action_size=7)
        wait_for_state(env)
        load_policy(agent, MODEL_PATH)
        env.get_logger().info(f"Loaded checkpoint: {MODEL_PATH}")
        env.get_logger().info(
            f"Evaluating greedily for up to {NUM_EPISODES} episodes; no training/saving."
        )


        if not reset_simulation(env):
            raise RuntimeError("Gazebo reset failed")
        set_robot_start_position(
            env, x=EVAL_STARTS[0][0], y=EVAL_STARTS[0][1], yaw=EVAL_STARTS[0][2]
        )
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(env, timeout_sec=0.1)
        wait_for_state(env)
        align_goal_to_start(env, *EVAL_STARTS[0])



        for episode in range(1, NUM_EPISODES + 1):
            state = env.get_state()
            if state is None:
                wait_for_state(env)
                state = env.get_state()
            state = env.normalize(state)       # after the if block, runs every episode
            total_reward = 0.0
            steps_taken = 0
            outcome = "Timeout"

            for step in range(MAX_STEPS):
                steps_taken = step + 1
                action = agent.select_greedy_action(state)
                next_state, reward, terminated, reason = env.step(action)
                total_reward += float(reward)
                state = next_state
                if terminated:
                    outcome = str(reason)
                    break

            results.append({
                "episode": episode,
                "outcome": outcome,
                "reward": total_reward,
                "steps": steps_taken,
            })
            env.get_logger().info(
                f"Evaluation {episode}/{NUM_EPISODES} | outcome={outcome} "
                f"| reward={total_reward:.2f} | steps={steps_taken}"
            )

            if episode < NUM_EPISODES:
                if not reset_simulation(env):
                    env.get_logger().warning("Stopping early: Gazebo reset failed.")
                    break

                # INSERT 1: teleport to the next eval pose (right after the reset, before the settle loop)
                start_x, start_y, start_yaw = EVAL_STARTS[episode % len(EVAL_STARTS)]
                set_robot_start_position(env, x=start_x, y=start_y, yaw=start_yaw)

                deadline = time.monotonic() + 2.0
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(env, timeout_sec=0.1)
                wait_for_state(env)

                # INSERT 2: keep the fixed goal at the same world spot (after wait_for_state)
                align_goal_to_start(env, start_x, start_y, start_yaw)

        if results:
            success = sum(1 for r in results if "goal" in r["outcome"].lower()
                          or "success" in r["outcome"].lower())
            collision = sum(1 for r in results if "collision" in r["outcome"].lower()
                            or "tumble" in r["outcome"].lower())
            timeout = sum(1 for r in results if "timeout" in r["outcome"].lower()
                          or "max_steps" in r["outcome"].lower())
            print("\n===== GREEDY EVALUATION SUMMARY =====")
            print(f"Episodes completed: {len(results)}/{NUM_EPISODES}")
            print(f"Goal successes: {success}")
            print(f"Collisions: {collision}")
            print(f"Timeouts: {timeout}")
            print(f"Success rate: {100 * success / len(results):.1f}%")
            print(f"Average reward: {np.mean([r['reward'] for r in results]):.2f}")
            print(f"Average steps: {np.mean([r['steps'] for r in results]):.2f}")
            print("No CSV/JSON files were created.")
            plot_results(results)
    except KeyboardInterrupt:
        if results:
            print("\nEvaluation interrupted; plotting completed episodes so far.")
            plot_results(results)
    finally:
        if env is not None:
            try:
                env.publish_stop()
                env.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import rclpy
from std_srvs.srv import Empty
from mini_dqn_nav.environment import RobotEnvironment
from mini_dqn_nav.agent import DQNAgent
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist


# CONFIGURATION


# Expand "~" to your home directory.
# For example:
#   ~/ros2_ws/...  ->  /home/<your-user>/ros2_ws/...
# This is the saved, trained DQN checkpoint that evaluation will load.
# Evaluation uses the weights from this file; it does not train a new model.
MODEL_PATH = os.path.expanduser(
    "~/ros2_ws/src/mini_dqn_nav/mini_dqn_nav/models/dqn_model_7actions_20_lidars.pth"
)

# Number of evaluation episodes to run.
# An episode is one attempt by the robot to reach the goal or end for another
# reason, such as a collision warning or reaching the step limit.
NUM_EPISODES = 20

# Maximum number of actions/steps allowed in each episode.
# If the environment never reports termination within this many steps, the
# episode keeps its initial outcome label, "Timeout".
MAX_STEPS = 100



# TELEPORT THE ROBOT TO A KNOWN STARTING POSE


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


# RESET THE GAZEBO SIMULATION/WORLD

def reset_simulation(node):
    # Try both common Gazebo reset services. The first available and
    # successfully completed service ends this function with True.
    for service_name in ("/reset_simulation", "/reset_world"):

        # Both reset services use std_srvs/Empty: they take no request fields.
        client = node.create_client(Empty, service_name)

        # If this reset service is not available quickly, try the next one.
        if not client.wait_for_service(timeout_sec=1.0):
            continue

        # Request a reset without blocking on the service call itself.
        future = client.call_async(Empty.Request())

        # Give the reset response up to 5 seconds to arrive.
        deadline = time.monotonic() + 5.0

        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                # Stop waiting for this service and allow the loop to try
                # the other reset service.
                break

            rclpy.spin_once(node, timeout_sec=0.1)

        # If a response arrived, log the service name and report success.
        # Note: Empty responses have no explicit "success" field.
        if future.done() and future.result() is not None:
            node.get_logger().info(f"Gazebo reset using {service_name}.")
            return True

    # Neither service produced a completed response we could use.
    node.get_logger().warning("Could not reset Gazebo; check that it is running.")
    return False


# WAIT UNTIL THE ROBOT'S STATE IS AVAILABLE

def wait_for_state(env, timeout=10.0):
    # Set a maximum waiting time for sensor callbacks to populate the state.
    deadline = time.monotonic() + timeout

    # Keep waiting while ROS is running and get_state() has no usable state.
    while rclpy.ok() and env.get_state() is None:
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for LiDAR/odometry data.")

        # Process callbacks so LiDAR and odometry messages can arrive and
        # update the environment's stored sensor information.
        rclpy.spin_once(env, timeout_sec=0.1)


# LOAD THE TRAINED DQN POLICY


def load_policy(agent, path):
    # Fail clearly if the checkpoint path is wrong or the file is missing.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load the checkpoint onto the same device as the agent (CPU or GPU).
    # weights_only=False allows general checkpoint objects to be unpickled;
    # only load checkpoints you trust.
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)

    # Different training scripts save weights under different dictionary keys.
    # Handle the common formats used by this script.
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        weights = checkpoint["policy_net"]
    elif isinstance(checkpoint, dict) and "policy_net_state_dict" in checkpoint:
        weights = checkpoint["policy_net_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    else:
        # If no recognized wrapper key exists, assume the loaded object itself
        # is the model's state_dict.
        weights = checkpoint

    # Copy the saved parameters into the agent's policy network.
    # This will raise an error if the saved architecture/parameter shapes do
    # not match the DQNAgent policy network.
    agent.policy_net.load_state_dict(weights)

    # Put the network in evaluation mode. This matters for layers such as
    # dropout or batch normalization if the architecture uses them.
    agent.policy_net.eval()

    # Disable epsilon exploration if other agent methods consult this value.
    # This script also calls select_greedy_action() explicitly in the loop.
    agent.epsilon = 0.0


# ---------------------------------------------------------------------------
# PLOT THE COMPLETED EVALUATION EPISODES
# ---------------------------------------------------------------------------

def plot_results(results):
    # Extract values from the list of per-episode result dictionaries.
    episodes = [r["episode"] for r in results]
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]

    # These are the outcome categories displayed in the plots.
    labels = ["Success", "Collision", "Timeout", "Other"]

    # Count how many episodes fall into each category.
    counts = {label: 0 for label in labels}

    # Store cumulative category counts after each episode so we can plot
    # how the totals grow throughout the evaluation.
    cumulative = {label: [] for label in labels}

    for result in results:
        # Normalize the outcome text to lowercase for case-insensitive checks.
        reason = result["outcome"].lower()

        # Classify outcomes using keywords in the environment's reason string.
        if "goal" in reason or "success" in reason:
            counts["Success"] += 1
        elif "collision" in reason or "tumble" in reason:
            counts["Collision"] += 1
        elif "timeout" in reason or "max_steps" in reason:
            counts["Timeout"] += 1
        else:
            counts["Other"] += 1

        # After each episode, save the current cumulative count for every
        # category. This creates one cumulative curve per outcome category.
        for label in labels:
            cumulative[label].append(counts[label])

    # Compute a trailing moving average of episode rewards.
    # For each episode, average its reward and up to the previous 9 rewards.
    # Near the beginning, the window contains fewer than 10 values.
    window = min(10, len(rewards))
    moving_avg = [
        float(np.mean(rewards[max(0, i - window + 1):i + 1]))
        for i in range(len(rewards))
    ]

    # DQN uses a Q-network and normally a temporal-difference (TD) loss during
    # training. This script only evaluates a saved policy, so it does not
    # calculate a training loss and must not invent one for the plot.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Count successful episodes independently for the success-rate title.
    success_count = sum(
        1 for r in results
        if "goal" in r["outcome"].lower()
        or "success" in r["outcome"].lower()
    )

    # Convert the number of successes into a percentage.
    # Guard against division by zero if results is empty.
    success_rate = 100.0 * success_count / len(results) if results else 0.0

    # Add an overall title describing this as greedy (non-exploratory)
    # evaluation and displaying the measured success rate.
    fig.suptitle(
        f"Evaluation Accuracy (Success Rate): {success_rate:.1f}%"
        f" | TurtleBot3 DQN — Greedy Evaluation",
        fontsize=15
    )

    # Plot 1: cumulative number of episodes in each outcome category.
    ax = axes[0, 0]
    for label in labels:
        # Always show Success, even if it remains at zero; show other
        # categories only if at least one episode belongs to them.
        if counts[label] > 0 or label == "Success":
            ax.plot(episodes, cumulative[label], label=label)
    ax.set_title("Cumulative evaluation outcomes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodes")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: trailing moving average of the total reward per episode.
    ax = axes[0, 1]
    ax.plot(episodes, moving_avg, label=f"Moving average (window={window})")
    ax.set_title("Average reward over evaluation episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: number of environment steps used in each episode.
    ax = axes[1, 0]
    ax.plot(episodes, steps, label="Steps per episode")
    ax.set_title("Steps per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: final count of episodes in each outcome category.
    ax = axes[1, 1]
    bars = ax.bar(labels, [counts[label] for label in labels])
    ax.set_title("Evaluation outcome counts")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Episodes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars)

    # Adjust spacing so titles and labels fit, then display the figure.
    fig.tight_layout()
    plt.show()



# MAIN EVALUATION PROCEDURE


def main(args=None):
    # Initialize the ROS 2 Python client library before creating ROS nodes.
    rclpy.init(args=args)

    # Initialize these variables so cleanup/summary logic can safely check
    # whether a node exists and which episode results have been collected.
    env = None
    results = []

    try:
        # Create the robot environment node. It is expected to subscribe to
        # sensors and expose methods such as get_state(), step(), and
        # publish_stop().
        env = RobotEnvironment()

        # Create a DQN agent with 7 input features and 7 discrete actions.
        # These dimensions MUST match the model architecture and environment.
        agent = DQNAgent(state_size=22, action_size=7)

        # Wait for valid sensor observations before attempting evaluation.
        wait_for_state(env)

        # Load the previously trained policy weights and disable exploration.
        load_policy(agent, MODEL_PATH)

        # Log the checkpoint path so the run can be traced to its model file.
        env.get_logger().info(f"Loaded checkpoint: {MODEL_PATH}")

        # Explain the evaluation configuration in the ROS log.
        env.get_logger().info(
            f"Evaluating greedily for up to {NUM_EPISODES} episodes; no training/saving."
        )

        # Run episode numbers 1 through NUM_EPISODES inclusive.
        for episode in range(1, NUM_EPISODES + 1):

            # Start each episode using the current state already available
            # from the environment. The first episode uses the current
            # simulation pose; later episodes are reset below.
            state = env.get_state()

            # If the state is unavailable, wait for sensor data and try again.
            if state is None:
                wait_for_state(env)
                state = env.get_state()

            # Accumulate the rewards produced during this episode.
            total_reward = 0.0

            # Count how many actions/steps are executed in this episode.
            steps_taken = 0

            # Default outcome if no terminal reason is received before the
            # loop reaches MAX_STEPS. The environment may supply another
            # outcome by returning terminated=True.
            outcome = "Timeout"

            # Execute at most MAX_STEPS interactions with the environment.
            for step in range(MAX_STEPS):

                # Convert the zero-based loop index into a human-readable
                # count: the first executed step is 1, not 0.
                steps_taken = step + 1

                # Choose the action with the highest predicted Q-value.
                # This is greedy selection: no random exploratory action is
                # intentionally requested here.
                action = agent.select_greedy_action(state)

                # Apply the action to the environment and receive:
                #   next_state: observation after the action
                #   reward: scalar feedback for the action
                #   terminated: whether the episode should end
                #   reason: textual explanation for termination/continuation
                next_state, reward, terminated, reason = env.step(action)

                # Add this step's reward to the episode's cumulative reward.
                total_reward += float(reward)

                # Use the newly observed state for the next decision.
                state = next_state

                # End the episode early if the environment signals a terminal
                # condition such as goal reached or collision warning.
                if terminated:
                    outcome = str(reason)
                    break

            # Store a compact summary of this episode for later statistics
            # and plotting. Each result is a dictionary with consistent keys.
            results.append({
                "episode": episode,
                "outcome": outcome,
                "reward": total_reward,
                "steps": steps_taken,
            })

            # Log the result of this episode in the ROS 2 terminal output.
            env.get_logger().info(
                f"Evaluation {episode}/{NUM_EPISODES} | outcome={outcome} "
                f"| reward={total_reward:.2f} | steps={steps_taken}"
            )

            # Reset and reposition the robot before the next episode.
            # No reset is needed after the final episode.
            if episode < NUM_EPISODES:

                # Reset the Gazebo simulation/world using an available service.
                if not reset_simulation(env):
                    raise RuntimeError("Gazebo reset failed")

                # After resetting, teleport the robot to the chosen start pose.
                set_robot_start_position(env, x=-0.6, y=0.9, yaw=0.0)

                # Allow time for physics and sensor streams to settle after
                # the reset/teleport operation.
                deadline = time.monotonic() + 2.0
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(env, timeout_sec=0.1)

                # Ensure the robot's state is available before the next run.
                wait_for_state(env)

        # Print a summary only if at least one episode was recorded.
        if results:
            # Count success outcomes using the same keyword-based convention
            # as the plotting function.
            success = sum(1 for r in results if "goal" in r["outcome"].lower()
                          or "success" in r["outcome"].lower())

            # Count collision/tumble labels. This relies on the environment's
            # reason strings and does not independently verify physical contact.
            collision = sum(1 for r in results if "collision" in r["outcome"].lower()
                            or "tumble" in r["outcome"].lower())

            # Count timeout outcomes, including explicit max-step labels.
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

            # This message documents that the script only prints and plots
            # results; it does not save CSV or JSON files.
            print("No CSV/JSON files were created.")

            # Display the four evaluation plots.
            plot_results(results)

    except KeyboardInterrupt:
        # If the user presses Ctrl+C, retain the results already collected
        # and plot those completed episodes rather than discarding them.
        if results:
            print("\nEvaluation interrupted; plotting completed episodes so far.")
            plot_results(results)

    finally:
        # Cleanup is attempted whether evaluation finishes or is interrupted
        # or an exception occurs.
        if env is not None:
            try:
                # Send a zero-velocity stop command before destroying the node.
                env.publish_stop()

                # Destroy the ROS node to release its resources.
                env.destroy_node()
            except Exception:
                # Ignore cleanup exceptions so shutdown can still proceed.
                pass

        # Shut down ROS 2 only if it is still initialized/running.
        if rclpy.ok():
            rclpy.shutdown()


# Python sets __name__ to "__main__" when this file is run directly.
# This guard prevents main() from automatically running if the file is imported
# as a module by another Python script.
if __name__ == "__main__":
    main()

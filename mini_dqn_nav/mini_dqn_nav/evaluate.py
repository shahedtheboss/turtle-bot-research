
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import rclpy
from std_srvs.srv import Empty
from mini_dqn_nav.environment import RobotEnvironment
from mini_dqn_nav.agent import DQNAgent

MODEL_PATH = os.path.expanduser(
    "~/ros2_ws/src/mini_dqn_nav/mini_dqn_nav/models/dqn_model_5actions.pth"
)
NUM_EPISODES = 20
MAX_STEPS = 100


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
        agent = DQNAgent(state_size=7, action_size=5)
        wait_for_state(env)
        load_policy(agent, MODEL_PATH)
        env.get_logger().info(f"Loaded checkpoint: {MODEL_PATH}")
        env.get_logger().info(
            f"Evaluating greedily for up to {NUM_EPISODES} episodes; no training/saving."
        )

        for episode in range(1, NUM_EPISODES + 1):
            state = env.get_state()
            if state is None:
                wait_for_state(env)
                state = env.get_state()
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
                deadline = time.monotonic() + 2.0
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(env, timeout_sec=0.1)
                wait_for_state(env)

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

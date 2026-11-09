"""Train Rainbow DQN with the mini_dqn_nav RobotEnvironment."""

import os
import math
import time
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import rclpy
from std_srvs.srv import Empty

from mini_dqn_nav.environment import RobotEnvironment
from mini_dqn_nav.agent_rainbow import RainbowDQNAgent


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
SEED = 42
EPISODES = 300
MAX_STEPS = 250
UPDATES_PER_STEP = 2        # gradient updates per environment step (sim is the bottleneck)
EVAL_EVERY = 25             # greedy evaluation every N training episodes (0 = off)
EVAL_EPISODES = 3
START_POSE = (0.0, 0.0, 0.0)  # TRUE world pose of the robot after a Gazebo reset
                              # (check your launch file's x_pose / y_pose)

# Optional curriculum: easier goals first. Only enable if those goal points are
# free space in your world (otherwise the goal may sit inside an obstacle).
CURRICULUM = False
CURRICULUM_GOALS = [(60, (1.0, -1.0)), (120, (1.5, -1.5))]  # (until_episode, goal)
FINAL_GOAL = (2.0, -2.0)

# Guided exploration. An untrained NoisyNet acts almost deterministically and 6 of
# the 7 actions turn, so early episodes wander near the start. For the first
# GUIDED_EPISODES episodes a simple obstacle-aware goal-seeker picks the action with
# probability decaying from GUIDED_START to GUIDED_END; every transition is stored.
GUIDED_START = 0.8
GUIDED_END = 0.1
GUIDED_EPISODES = 100
GUIDED_NOISE = 0.15           # chance the guided action is replaced by a forward-biased random one


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("Random seed:", seed)


def reset_simulation(node):
    """Reset Gazebo between episodes."""
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
    node.get_logger().warning("No usable Gazebo reset service found.")
    return False


def prepare_episode(env, first):
    """Reset Gazebo (except before the very first episode) and re-anchor the pose."""
    if not first:
        if not reset_simulation(env):
            return False
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(env, timeout_sec=0.1)
    if env.get_state() is None:
        raise RuntimeError("Sensor data unavailable after reset.")
    env.begin_episode(start=START_POSE)
    return True


def guided_probability(episode):
    if episode >= GUIDED_EPISODES:
        return GUIDED_END
    frac = (episode - 1) / max(1, GUIDED_EPISODES - 1)
    return GUIDED_START + (GUIDED_END - GUIDED_START) * frac


def heuristic_action(state, max_lidar):
    """
    Simple obstacle-aware goal seeker on the NORMALIZED 52-value state.
    Sector 0 starts at the robot's front and angles increase to the LEFT, so the
    front arc is sectors 47-49 and 0-2, left is 3-12 and right is 38-47.
    Actions: 0 forward, 1/2 soft left/right, 3/4 forward left/right, 5/6 hard left/right.
    """
    lidar = [d * max_lidar for d in state[:50]]
    goal_angle = state[51] * math.pi          # > 0: goal is to the left

    if min(lidar[47:50] + lidar[0:3]) < 0.45:  # obstacle ahead: turn to the freer side
        return 5 if min(lidar[3:13]) >= min(lidar[38:48]) else 6
    if abs(goal_angle) < 0.2:
        return 0
    if goal_angle > 0:
        return 1 if goal_angle < 0.6 else (3 if goal_angle < 1.2 else 5)
    return 2 if goal_angle > -0.6 else (4 if goal_angle > -1.2 else 6)


def goal_for_episode(episode):
    if not CURRICULUM:
        return FINAL_GOAL
    for until_episode, goal in CURRICULUM_GOALS:
        if episode <= until_episode:
            return goal
    return FINAL_GOAL


def run_episode(env, agent, greedy, episode=0):
    """
    Run one episode. greedy=False trains (NoisyNet exploration, replay, updates);
    greedy=True evaluates with the deterministic mean-weight policy.
    """
    state = env.normalize(env.get_state())
    total_reward, losses = 0.0, []
    action_counts = [0] * 7
    reason, steps_taken, terminated = "timeout", 0, False
    p_guided = 0.0 if greedy else guided_probability(episode)

    for step in range(MAX_STEPS):
        steps_taken = step + 1
        if greedy:
            action = agent.select_greedy_action(state)
        elif random.random() < p_guided:
            if random.random() < GUIDED_NOISE:
                action = random.choices(range(7), weights=[4, 1, 1, 1, 1, 1, 1])[0]
            else:
                action = heuristic_action(state, env.MAX_LIDAR)
        else:
            action = agent.select_action(state)
        action_counts[action] += 1
        next_state, reward, terminated, reason = env.step(action)

        if not greedy:
            # Only goal / collision are true terminals. A time limit or the
            # no-progress cut-off truncates the episode but must still bootstrap.
            time_limit = (step == MAX_STEPS - 1) and not terminated
            truncated = time_limit or (terminated and reason == "no_progress")
            terminal = terminated and reason != "no_progress"
            agent.remember(state, action, reward, next_state, terminal, truncated)
            for _ in range(UPDATES_PER_STEP):
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

        state = next_state
        total_reward += float(reward)

        if terminated:
            break

    outcome = reason if terminated else "timeout"
    return {
        "reward": total_reward,
        "steps": steps_taken,
        "outcome": outcome,
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "actions": action_counts,
    }


def main(args=None):
    rclpy.init(args=args)
    env = None
    agent = None
    model_path = os.path.expanduser(
        "~/ros2_ws/src/mini_dqn_nav/mini_dqn_nav/models/"
        "rainbow_dqn_52state_7actions.pth"
    )
    best_path = model_path.replace(".pth", "_best.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    set_seed(SEED)

    rewards_history, losses_history = [], []
    steps_history, outcomes = [], []
    eval_history = []            # (episode, success_rate)
    best_success = -1.0

    try:
        env = RobotEnvironment()
        agent = RainbowDQNAgent(
            state_size=52, action_size=7,
            learning_rate=1e-4, gamma=0.99,
            batch_size=128, memory_size=50000,
            n_step=3, num_atoms=51,
            v_min=-75.0, v_max=125.0,
            target_update_every=250,
            warmup_transitions=1000
        )
        env.get_logger().info(f"Rainbow DQN device: {agent.device}")

        deadline = time.monotonic() + 10.0
        while rclpy.ok() and env.get_state() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for initial LiDAR/odometry.")
            rclpy.spin_once(env, timeout_sec=0.1)

        first = True
        for episode in range(1, EPISODES + 1):
            env.goal_x, env.goal_y = goal_for_episode(episode)
            if not prepare_episode(env, first):
                env.get_logger().warning("Stopping: Gazebo reset failed.")
                break
            first = False

            result = run_episode(env, agent, greedy=False, episode=episode)

            rewards_history.append(result["reward"])
            losses_history.append(result["loss"])
            steps_history.append(result["steps"])
            outcomes.append(result["outcome"])

            env.get_logger().info(
                f"Episode {episode}/{EPISODES} | {result['outcome']} | "
                f"reward={result['reward']:.2f} | steps={result['steps']} | "
                f"loss={result['loss']:.4f} | replay={len(agent.memory)} | "
                f"guided={guided_probability(episode):.2f} | "
                f"goal_dist={env.last_goal_distance:.2f} | "
                f"min_lidar={env.last_min_lidar:.2f} | actions={result['actions']}"
            )
            agent.save(model_path)

            # Periodic greedy evaluation; keep the best checkpoint.
            if EVAL_EVERY and episode % EVAL_EVERY == 0 and episode < EPISODES:
                wins = 0
                env.goal_x, env.goal_y = FINAL_GOAL
                for _ in range(EVAL_EPISODES):
                    if not prepare_episode(env, first=False):
                        break
                    if run_episode(env, agent, greedy=True)["outcome"] == "goal_reached":
                        wins += 1
                rate = wins / EVAL_EPISODES
                eval_history.append((episode, rate))
                env.get_logger().info(
                    f"[EVAL] episode {episode}: greedy success "
                    f"{wins}/{EVAL_EPISODES}"
                )
                if rate > best_success:
                    best_success = rate
                    agent.save(best_path)

        counts = Counter(outcomes)
        print("\n--- Episode Outcomes (training) ---")
        for name, count in counts.items():
            print(f"{name}: {count}")
        total = len(outcomes)
        success = counts.get("goal_reached", 0)
        print(
            f"Training success rate: {100 * success / total:.1f}% ({success}/{total})"
            if total else "No episodes completed."
        )
        print("Greedy evaluation history:", eval_history)
        print("Rainbow checkpoint:", model_path)
        print("Best-by-eval checkpoint:", best_path)

        x = np.arange(1, len(rewards_history) + 1)
        window = min(10, len(rewards_history))
        moving = [
            np.mean(rewards_history[max(0, i - window + 1):i + 1])
            for i in range(len(rewards_history))
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("TurtleBot3 Rainbow DQN - Training")
        axes[0, 0].plot(x, rewards_history, label="Episode reward")
        axes[0, 0].plot(x, moving, label=f"Moving average ({window})")
        axes[0, 0].set_title("Episode reward")
        axes[0, 0].legend()
        axes[0, 1].plot(x, losses_history)
        axes[0, 1].set_title("Mean distributional loss")
        axes[1, 0].plot(x, steps_history)
        axes[1, 0].set_title("Steps per episode")
        axes[1, 1].bar(list(counts.keys()), list(counts.values()))
        axes[1, 1].set_title("Training outcomes")
        axes[1, 1].tick_params(axis="x", rotation=20)
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("rainbow_training_performance.png", dpi=150)
        plt.show()

    except KeyboardInterrupt:
        if env is not None:
            env.get_logger().info("Training interrupted.")
    except Exception as error:
        if env is not None:
            env.get_logger().error(f"Training failed: {error}")
        raise
    finally:
        if agent is not None:
            agent.save(model_path)
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
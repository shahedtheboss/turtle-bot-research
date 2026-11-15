"""Evaluate a Rainbow DQN checkpoint, or the scripted heuristic, in TurtleBot3 Gazebo.

Each episode starts from one of the poses in --starts (rotated in order), runs
greedily for at most MAX_STEPS actions, and is logged to a CSV file. Greedy
evaluation uses the NoisyNet mean weights, so it is deterministic for a fixed
checkpoint and simulator state.

# ---------------- CHANGED: examples match train.py's results/<run>/<run>_* layout ----------------
Examples (run from the repo root after sourcing ROS, install/setup.bash and the venv):
  python3 -m rainbow_ablation.evaluate --model results/G_s1/G_s1_model_final.pth --episodes 50 --label G_s1 --out-csv results/G_s1/G_s1_final_eval.csv
  python3 -m rainbow_ablation.evaluate --policy heuristic --episodes 50 --label H --out-csv results/H/H_final_eval.csv
# ---------------- END CHANGED ----------------
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")  # save figures to disk; never open a blocking window
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import torch
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Empty
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist

from rainbow_ablation.agent import RainbowDQNAgent
from rainbow_ablation.environment import RobotEnvironment
from rainbow_ablation.train import heuristic_action

MAX_STEPS = 250                     # same episode cap as training
ROBOT_NAME = "turtlebot3_burger"    # entity name in the running Gazebo world
GOAL_X, GOAL_Y = 2.0, -2.0
SETTLE_SEC = 2.0                    # wait after reset/teleport so sensors refresh
# ---------------- CHANGED: study evaluation poses (was the pilot's single pose (0.2, 0.2, 0)) ----------------
# Pose 0 is the training start. Poses 1-4 were chosen from the stage-4 world file:
# each is >= 0.5 m from every wall and away from both cylinders' first 15 s of motion.
# With 50 episodes, each pose gets 10 (episode e uses pose (e-1) mod 5).
DEFAULT_STARTS = [
    (0.0, 0.0, 0.0),            # P0 training start; wall inner_wall_3 blocks the straight line
    (-1.5, -0.8, 0.0),          # P1 far west, behind inner_wall_7, 3.7 m from goal
    (0.0, 1.0, -math.pi / 2),   # P2 north, facing south, 3.6 m from goal
    (1.8, -0.2, -math.pi / 2),  # P3 east, facing the goal, 1.8 m (easy pose)
    (0.3, -1.8, 0.0),           # P4 south corridor, facing the goal, 1.7 m (easy pose)
]
# ---------------- END CHANGED ----------------
# Actions the training guide's noise draws from (same weights as train.py).
NOISE_WEIGHTS = [4, 1, 1, 1, 1, 1, 1]
OUTCOMES = ("goal_reached", "collision_warning", "no_progress", "timeout")


def parse_starts(text):
    """Parse 'x,y,yaw;x,y,yaw;...' into a list of float triples."""
    poses = []
    for item in text.split(";"):
        x, y, yaw = (float(v) for v in item.split(","))
        poses.append((x, y, yaw))
    if not poses:
        raise argparse.ArgumentTypeError("--starts needs at least one pose")
    return poses


def parse_args(argv):
    p = argparse.ArgumentParser(description="Evaluate a Rainbow checkpoint or the heuristic.")
    p.add_argument("--policy", choices=["greedy", "heuristic"], default="greedy")
    p.add_argument("--model", help="checkpoint path (required for --policy greedy)")
    p.add_argument("--episodes", type=int, default=50, help="total episodes, rotated over the poses")
    p.add_argument("--starts", type=parse_starts, default=DEFAULT_STARTS,
                   help="poses as 'x,y,yaw;x,y,yaw'; episode e uses pose (e-1) mod count")
    p.add_argument("--guide-noise", type=float, default=0.0,
                   help="heuristic only: chance an action is replaced by a random one (0 = pure heuristic)")
    p.add_argument("--seed", type=int, default=0, help="seeds Python's random (used by --guide-noise)")
    p.add_argument("--label", default="eval", help="name written into the CSV and figure")
    p.add_argument("--out-csv", help="write one row per episode to this CSV file")
    p.add_argument("--plot", help="save a summary figure to this PNG path")
    # Strip ROS 2 remapping arguments (e.g. --ros-args ...) before parsing ours.
    return p.parse_args(remove_ros_args(argv)[1:])


def reset_simulation(node):
    """Ask Gazebo to reset the world. Returns True only if a reset call succeeded."""
    for service_name in ("/reset_simulation", "/reset_world"):
        client = node.create_client(Empty, service_name)
        if not client.wait_for_service(timeout_sec=1.0):
            continue
        future = client.call_async(Empty.Request())
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if future.done() and future.result() is not None:
            return True
    return False


def teleport_robot(node, x, y, yaw):
    """Move the robot model in Gazebo. Raises if the service is missing or refuses."""
    client = node.create_client(SetEntityState, "/set_entity_state")
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError("Gazebo /set_entity_state service unavailable")

    request = SetEntityState.Request()
    request.state = EntityState()
    request.state.name = ROBOT_NAME
    request.state.reference_frame = "world"
    request.state.pose.position.x = float(x)
    request.state.pose.position.y = float(y)
    request.state.pose.position.z = 0.01            # keep the base just above the ground
    request.state.pose.orientation.z = math.sin(yaw / 2.0)
    request.state.pose.orientation.w = math.cos(yaw / 2.0)
    request.state.twist = Twist()                   # no leftover velocity after the move

    future = client.call_async(request)
    deadline = time.monotonic() + 5.0
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not future.done():
        raise TimeoutError("Timed out repositioning the robot")
    response = future.result()
    if response is None or not response.success:
        raise RuntimeError("Gazebo refused the reposition request")


def settle(node, seconds):
    deadline = time.monotonic() + seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def wait_for_sensors(node, timeout=10.0):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and node.get_state() is None:
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for LiDAR/odometry data.")
        rclpy.spin_once(node, timeout_sec=0.1)


def prepare_episode(env, pose):
    """Reset Gazebo, teleport to the start pose, and re-anchor the pose tracking there."""
    if not reset_simulation(env):
        raise RuntimeError("Gazebo reset failed")
    x, y, yaw = pose
    teleport_robot(env, x, y, yaw)
    settle(env, SETTLE_SEC)
    wait_for_sensors(env)
    env.goal_x, env.goal_y = GOAL_X, GOAL_Y
    env.begin_episode(start=(x, y, yaw))


def load_greedy_agent(path, device=None):
    """Build the agent stored in the checkpoint (Rainbow, or DQN/DDQN/D3QN for Paper 1)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=True: the checkpoint holds only tensors and plain numbers, so
    # loading it cannot run arbitrary pickled code (the old script used False).
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=True)
    # ---------------- NEW CODE ADDED: Paper 1 checkpoints (DQN / DDQN / D3QN) ----------------
    # Reason (item 4.1): rebuild the right network from the checkpoint's "algo" field.
    # Old checkpoints have no "algo" and are Rainbow, so they keep loading as before.
    algo = ckpt.get("algo", "rainbow")
    if algo in ("dqn", "ddqn", "d3qn"):
        from rainbow_ablation.paper1.agent_dqn import DQNAgent
        from rainbow_ablation.paper1.agent_ddqn import DDQNAgent
        from rainbow_ablation.paper1.agent_d3qn import D3QNAgent
        cls = {"dqn": DQNAgent, "ddqn": DDQNAgent, "d3qn": D3QNAgent}[algo]
        agent = cls(ckpt["state_size"], ckpt["action_size"], **ckpt.get("cfg", {}))   # review fix 6
        agent.policy_net.load_state_dict(ckpt["policy_net"])
        agent.policy_net.eval()
        agent.epsilon = 0.0          # greedy evaluation never explores
        return agent
    if algo != "rainbow":                      # review fix 7: fail clearly on unknown checkpoints
        raise ValueError(f"Unknown algo {algo!r} in checkpoint {path}")
    # ---------------- END NEW CODE ----------------
    agent = RainbowDQNAgent(
        state_size=ckpt["state_size"], action_size=ckpt["action_size"],
        num_atoms=ckpt["num_atoms"], v_min=ckpt["v_min"], v_max=ckpt["v_max"],
    )
    agent.policy_net.load_state_dict(ckpt["policy_net"])
    agent.policy_net.eval()
    return agent


def run_episode(env, choose, rng, guide_noise):
    """Run one episode. choose(state) -> action. Returns (outcome, reward, steps)."""
    state = env.normalize(env.get_state())
    total_reward, steps, outcome = 0.0, 0, "timeout"
    for step in range(MAX_STEPS):
        steps = step + 1
        action = choose(state, rng, guide_noise)
        next_state, reward, terminated, reason = env.step(action)
        total_reward += float(reward)
        state = next_state
        if terminated:
            outcome = str(reason)
            break
    return outcome, total_reward, steps


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_text(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.3f}, {hi:.3f}]"


def save_summary_plot(results, path, label):
    successes = [r for r in results if r["outcome"] == "goal_reached"]
    rate = 100.0 * len(successes) / len(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{label}: greedy success {rate:.1f}% ({len(successes)}/{len(results)})")
    counts = [sum(r["outcome"] == o for r in results) for o in OUTCOMES]
    axes[0].bar(OUTCOMES, counts)
    axes[0].set_title("Outcomes")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].plot([r["episode"] for r in results], [r["steps"] for r in results], marker="o")
    axes[1].set_title("Steps per episode")
    axes[1].set_xlabel("Episode")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results, starts, label):
    n = len(results)
    k = sum(r["outcome"] == "goal_reached" for r in results)
    print(f"\n===== EVALUATION SUMMARY: {label} =====")
    print(f"Episodes: {n}")
    for outcome in OUTCOMES:
        print(f"{outcome}: {sum(r['outcome'] == outcome for r in results)}")
    print(f"Success: {k}/{n} = {100.0 * k / n:.1f}%  Wilson 95% {wilson_text(k, n)}")
    for i, pose in enumerate(starts):
        group = [r for r in results if r["pose_index"] == i]
        ki = sum(r["outcome"] == "goal_reached" for r in group)
        print(f"  pose {i} {pose}: {ki}/{len(group)}")
    won = [r["steps"] for r in results if r["outcome"] == "goal_reached"]
    if won:
        print(f"Mean steps on successful episodes: {np.mean(won):.1f}")


def main(argv=None):
    args = parse_args(sys.argv if argv is None else argv)
    if args.policy == "greedy" and not args.model:
        raise SystemExit("--model is required for --policy greedy")
    if not 0.0 <= args.guide_noise <= 1.0:
        raise SystemExit("--guide-noise must be between 0 and 1")
    random.seed(args.seed)

    rclpy.init(args=argv)
    env = None
    results = []
    try:
        env = RobotEnvironment()
        wait_for_sensors(env)

        if args.policy == "greedy":
            agent = load_greedy_agent(args.model, device=None)
            env.get_logger().info(f"Loaded {type(agent).__name__} checkpoint {args.model} (greedy, no exploration)")   # review fix 7

            def choose(state, rng, noise):
                return agent.select_greedy_action(state)
        else:
            def choose(state, rng, noise):
                if rng.random() < noise:
                    return rng.choices(range(7), weights=NOISE_WEIGHTS)[0]
                return heuristic_action(state, env.MAX_LIDAR)

        for episode in range(1, args.episodes + 1):
            pose_index = (episode - 1) % len(args.starts)
            pose = args.starts[pose_index]
            prepare_episode(env, pose)
            outcome, reward, steps = run_episode(env, choose, random, args.guide_noise)
            end_x, end_y, _ = env.get_world_pose()
            row = {
                "label": args.label, "episode": episode, "pose_index": pose_index,
                "start_x": pose[0], "start_y": pose[1], "start_yaw": pose[2],
                "outcome": outcome, "steps": steps, "reward": round(reward, 4),
                "end_x": round(end_x, 3), "end_y": round(end_y, 3),
                "goal_distance_end": round(env.last_goal_distance, 3),
                "min_lidar_end": round(env.last_min_lidar, 3),
            }
            results.append(row)
            env.get_logger().info(
                f"Eval {episode}/{args.episodes} pose {pose_index} | {outcome} | "
                f"steps={steps} | reward={reward:.2f}"
            )

        print_summary(results, args.starts, args.label)

        if args.out_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"CSV written: {args.out_csv}")
        if args.plot:
            save_summary_plot(results, args.plot, args.label)
            print(f"Figure written: {args.plot}")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(results)} episodes; nothing was saved.")
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

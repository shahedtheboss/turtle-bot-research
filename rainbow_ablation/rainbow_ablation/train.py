"""Train Rainbow DQN with the rainbow_ablation RobotEnvironment."""

import os
import math
import time
import random
from collections import Counter
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
import torch
import rclpy
from std_srvs.srv import Empty
from rclpy.utilities import remove_ros_args
# ---------------- CHANGED: sim clock for the real-time factor ----------------
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
# ---------------- END CHANGED ----------------

from rainbow_ablation.environment import RobotEnvironment
from rainbow_ablation.agent import RainbowDQNAgent


# (study protocol: 200 episodes, greedy check every 20 episodes with 5 tests)
SEED = 1
EPISODES = 200
EVAL_EVERY = 20             # greedy evaluation every N training episodes (0 = off)
EVAL_EPISODES = 5

MAX_STEPS = 250
UPDATES_PER_STEP = 2        # gradient updates per environment step (sim is the bottleneck)
START_POSE = (0.0, 0.0, 0.0)  # TRUE world pose of the robot after a Gazebo reset
                              # (check your launch file's x_pose / y_pose)

# Optional curriculum: easier goals first. Only enable if those goal points are
# free space in your world (otherwise the goal may sit inside an obstacle).
CURRICULUM = False
CURRICULUM_GOALS = [(60, (1.0, -1.0)), (120, (1.5, -1.5))]  # (until_episode, goal)
FINAL_GOAL = (2.0, -2.0)

""" 
  Guided exploration. An untrained NoisyNet acts almost deterministically and 6 of
  the 7 actions turn, so early episodes wander near the start. For the first
  GUIDED_EPISODES episodes a simple obstacle-aware goal-seeker picks the action with
  probability decaying from GUIDED_START to GUIDED_END; every transition is stored.
"""  
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


# ---------------- Arm switch. N = no guide at any episode ----------------
def guided_probability(episode, arm="G"):
    if arm == "N":
        return 0.0
    # ---- CHANGED: arm R uses exactly G's schedule; only the action source differs ----

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


def run_episode(env, agent, greedy, episode=0, arm="G"):
    """
    Run one episode. greedy=False trains (NoisyNet exploration, replay, updates);
    greedy=True evaluates with the deterministic mean-weight policy.
    """
    state = env.normalize(env.get_state())
    total_reward, losses = 0.0, []
    action_counts = [0] * 7
    reason, steps_taken, terminated = "timeout", 0, False
    # ---------------- Arm-aware guide; counters of guide-chosen steps ----------------
    p_guided = 0.0 if greedy else guided_probability(episode, arm)
    guided_steps, guided_noise_steps = 0, 0

    for step in range(MAX_STEPS):
        steps_taken = step + 1
        if greedy:
            action = agent.select_greedy_action(state)
        elif random.random() < p_guided:
            guided_steps += 1                     
            # ---------------- CHANGED: arm R = matched random control ----------------
            # R replaces every guide-chosen step with a forward-biased random action
            # (the same weights as the guide's noise). Same schedule, same rate, no
            # task knowledge. All R steps count as guided_steps and guided_noise_steps.
            if arm == "R" or random.random() < GUIDED_NOISE:
                guided_noise_steps += 1
                action = random.choices(range(7), weights=[4, 1, 1, 1, 1, 1, 1])[0]
            # ---------------- END CHANGED ----------------
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
        # ---------------- Guide counters returned for the CSV ----------------
        "p_guided": p_guided,
        "guided_steps": guided_steps,
        "guided_noise_steps": guided_noise_steps,
    }



EPISODE_FIELDS = [
    "episode", "outcome", "reward", "steps", "env_steps_total", "loss",
    "p_guided", "guided_steps", "guided_noise_steps", "replay_size",
    "learning_steps", "goal_dist", "min_lidar", "wall_time_s",
    "sim_time_s", "rtf",       # ---- CHANGED: episode sim time and real-time factor ----
    "a0", "a1", "a2", "a3", "a4", "a5", "a6",
]
EVAL_FIELDS = ["episode", "env_steps_total", "wins", "n", "success_rate", "outcomes"]


def parse_args(argv):
    p = argparse.ArgumentParser(description="Train Rainbow DQN for the guide ablation.")
    p.add_argument("--arm", choices=["G", "N", "R"], required=True,   # ---- CHANGED: + R ----
                   help="G = decaying scripted guide, N = no guide, "
                        "R = random actions at the guide's rate (matched control)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--run-name", help="default: <arm>_s<seed>, e.g. N_s1")
    p.add_argument("--results-dir", default="results",
                   help="parent folder; each run gets results/<run-name>/")
    p.add_argument("--overwrite", action="store_true",
                   help="allow reusing an existing run folder (otherwise refuse)")
    return p.parse_args(remove_ros_args(argv)[1:])


def git_info():
    """Commit hash and whether the working tree had uncommitted changes."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=here, text=True).strip())
        return commit, dirty
    except Exception:
        return "unknown", None


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_plot(path, run_name, episode_rows, eval_rows):
    """Four panels: reward, loss, steps + guided steps, greedy success."""
    x = [r["episode"] for r in episode_rows]
    rewards = [r["reward"] for r in episode_rows]
    window = min(10, len(rewards))
    moving = [np.mean(rewards[max(0, i - window + 1):i + 1]) for i in range(len(rewards))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Rainbow DQN training - {run_name}")
    axes[0, 0].plot(x, rewards, label="Episode reward")
    axes[0, 0].plot(x, moving, label=f"Moving average ({window})")
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].legend()
    axes[0, 1].plot(x, [r["loss"] for r in episode_rows])
    axes[0, 1].set_title("Mean distributional loss")
    axes[1, 0].plot(x, [r["steps"] for r in episode_rows], label="steps")
    axes[1, 0].plot(x, [r["guided_steps"] for r in episode_rows], label="guide-chosen steps")
    axes[1, 0].set_title("Steps per episode")
    axes[1, 0].legend()
    if eval_rows:
        axes[1, 1].plot([r["episode"] for r in eval_rows],
                        [r["success_rate"] for r in eval_rows], marker="o")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title("Greedy success (guide off)")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training episode")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(sys.argv if argv is None else argv)
    run_name = args.run_name or f"{args.arm}_s{args.seed}"

    # Every file of this run lives in results/<run_name>/ and carries the run name.
    run_dir = os.path.abspath(os.path.join(args.results_dir, run_name))
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.overwrite:
        sys.exit(f"{run_dir} already has files; refusing to overwrite (use --overwrite).")
    os.makedirs(run_dir, exist_ok=True)

    def out(suffix):
        return os.path.join(run_dir, f"{run_name}_{suffix}")

    model_path, best_path = out("model_final.pth"), out("model_best.pth")
    episodes_csv, evals_csv = out("episodes.csv"), out("evals.csv")
    meta_path, plot_path = out("run_meta.json"), out("training.png")

    rclpy.init(args=argv if argv is not None else sys.argv)
    env = None
    agent = None
    set_seed(args.seed)

    commit, dirty = git_info()
    meta = {
        "run_name": run_name, "arm": args.arm, "seed": args.seed,
        "episodes": args.episodes, "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes, "max_steps": MAX_STEPS,
        "updates_per_step": UPDATES_PER_STEP, "start_pose": START_POSE,
        "goal": FINAL_GOAL,
        "guide": {"start": GUIDED_START, "end": GUIDED_END,
                  "episodes": GUIDED_EPISODES, "noise": GUIDED_NOISE,
                  "active": args.arm in ("G", "R"),          # ---- CHANGED: R uses the schedule ----
                  "action_source": {"G": "heuristic + 15% random", "R": "random only",
                                    "N": "none"}[args.arm]},
        # ---------------- CHANGED: parallel-slot isolation, for checking it as a confound ----------------
        "slot": os.environ.get("RAINBOW_SLOT"),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "gazebo_master_uri": os.environ.get("GAZEBO_MASTER_URI", "default"),
        "parallel_runs": os.environ.get("RAINBOW_PARALLEL"),
        # ---------------- END CHANGED ----------------
        "git_commit": commit, "git_dirty": dirty,
        "torch_version": torch.__version__,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "end_time": None, "status": "running",
    }
    write_json(meta_path, meta)

    episode_rows, eval_rows = [], []
    rtf_values = []          # ---- CHANGED: per-episode real-time factors ----
    best_success = -1.0
    env_steps_total = 0
    t0 = time.monotonic()
    ep_file = open(episodes_csv, "w", newline="")
    ev_file = open(evals_csv, "w", newline="")
    ep_writer = csv.DictWriter(ep_file, fieldnames=EPISODE_FIELDS)
    ev_writer = csv.DictWriter(ev_file, fieldnames=EVAL_FIELDS)
    ep_writer.writeheader()
    ev_writer.writeheader()

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
        meta["device"] = str(agent.device)
        # ---------------- CHANGED: track Gazebo sim time (best-effort QoS, as /clock is published) ----------------
        sim_clock = {"t": None}
        def on_clock(msg):
            sim_clock["t"] = msg.clock.sec + msg.clock.nanosec * 1e-9
        env.create_subscription(
            Clock, "/clock", on_clock,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # ---------------- END CHANGED ----------------
        write_json(meta_path, meta)
        env.get_logger().info(f"Run {run_name} | arm={args.arm} | device={agent.device}")
        env.get_logger().info(f"Writing to {run_dir}")

        deadline = time.monotonic() + 10.0
        while rclpy.ok() and env.get_state() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for initial LiDAR/odometry.")
            rclpy.spin_once(env, timeout_sec=0.1)

        first = True
        for episode in range(1, args.episodes + 1):
            env.goal_x, env.goal_y = goal_for_episode(episode)
            if not prepare_episode(env, first):
                env.get_logger().warning("Stopping: Gazebo reset failed.")
                meta["status"] = f"stopped: reset failed at episode {episode}"
                break
            first = False

            sim0, wall0 = sim_clock["t"], time.monotonic()   # ---- CHANGED: RTF start ----
            result = run_episode(env, agent, greedy=False, episode=episode, arm=args.arm)
            # ---------------- CHANGED: real-time factor = sim seconds / wall seconds ----------------
            sim1, wall1 = sim_clock["t"], time.monotonic()
            sim_dt = (sim1 - sim0) if (sim0 is not None and sim1 is not None) else float("nan")
            rtf = sim_dt / (wall1 - wall0) if wall1 > wall0 and sim_dt == sim_dt else float("nan")
            if rtf == rtf:
                rtf_values.append(rtf)
            # ---------------- END CHANGED ----------------
            env_steps_total += result["steps"]

            row = {
                "episode": episode, "outcome": result["outcome"],
                "reward": round(result["reward"], 4), "steps": result["steps"],
                "env_steps_total": env_steps_total, "loss": result["loss"],
                "p_guided": round(result["p_guided"], 4),
                "guided_steps": result["guided_steps"],
                "guided_noise_steps": result["guided_noise_steps"],
                "replay_size": len(agent.memory),
                "learning_steps": agent.learning_steps,
                "goal_dist": round(env.last_goal_distance, 4),
                "min_lidar": round(env.last_min_lidar, 4),
                "wall_time_s": round(time.monotonic() - t0, 1),
                "sim_time_s": round(sim_dt, 2), "rtf": round(rtf, 3),   # ---- CHANGED ----
            }
            row.update({f"a{i}": c for i, c in enumerate(result["actions"])})
            episode_rows.append(row)
            ep_writer.writerow(row)
            ep_file.flush()        # data survives a crash or Ctrl+C

            env.get_logger().info(
                f"[{run_name}] Episode {episode}/{args.episodes} | {result['outcome']} | "
                f"reward={result['reward']:.2f} | steps={result['steps']} | "
                f"loss={result['loss']:.4f} | replay={len(agent.memory)} | "
                f"guided={result['guided_steps']}/{result['steps']} | rtf={rtf:.2f} | "   # ---- CHANGED ----
                f"goal_dist={env.last_goal_distance:.2f} | actions={result['actions']}"
            )
            agent.save(model_path)

            # Greedy evaluation (guide off for all arms). Now also runs at the
            # last episode, so E* can be found at episode 200.
            if args.eval_every and episode % args.eval_every == 0:
                outcomes = []
                env.goal_x, env.goal_y = FINAL_GOAL
                for _ in range(args.eval_episodes):
                    if not prepare_episode(env, first=False):
                        break
                    outcomes.append(run_episode(env, agent, greedy=True)["outcome"])
                wins = outcomes.count("goal_reached")
                n = len(outcomes)
                rate = wins / n if n else float("nan")
                ev = {"episode": episode, "env_steps_total": env_steps_total,
                      "wins": wins, "n": n, "success_rate": rate,
                      "outcomes": ";".join(outcomes)}
                eval_rows.append(ev)
                ev_writer.writerow(ev)
                ev_file.flush()
                env.get_logger().info(f"[{run_name}] [EVAL] episode {episode}: {wins}/{n}")
                if n and rate > best_success:
                    best_success = rate
                    agent.save(best_path)
        else:
            meta["status"] = "completed"

        counts = Counter(r["outcome"] for r in episode_rows)
        print(f"\n--- {run_name}: training outcomes ---")
        for name, count in counts.items():
            print(f"{name}: {count}")
        print("Greedy evaluations:", [(r["episode"], r["wins"]) for r in eval_rows])

    except KeyboardInterrupt:
        meta["status"] = f"interrupted after {len(episode_rows)} episodes"
        if env is not None:
            env.get_logger().info("Training interrupted.")
    except Exception as error:
        meta["status"] = f"failed: {error}"
        if env is not None:
            env.get_logger().error(f"Training failed: {error}")
        raise
    finally:
        ep_file.close()
        ev_file.close()
        if agent is not None:
            agent.save(model_path)
            meta["learning_steps"] = agent.learning_steps
        meta["episodes_completed"] = len(episode_rows)
        meta["env_steps_total"] = env_steps_total
        meta["wall_time_h"] = round((time.monotonic() - t0) / 3600, 3)
        # ---------------- CHANGED: RTF summary (the parallel-run confound check) ----------------
        rtfs = rtf_values
        meta["rtf_mean"] = round(float(np.mean(rtfs)), 3) if rtfs else None
        meta["rtf_min"] = round(float(np.min(rtfs)), 3) if rtfs else None
        # ---------------- END CHANGED ----------------
        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        write_json(meta_path, meta)
        if episode_rows:
            save_plot(plot_path, run_name, episode_rows, eval_rows)
            print("Figure:", plot_path)
        print("Run folder:", run_dir)
        if env is not None:
            try:
                env.publish_stop()
                env.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()
# ---------------- END CHANGED ----------------


if __name__ == "__main__":
    main()

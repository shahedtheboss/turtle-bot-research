"""Paper 1 shared training loop: DQN, DDQN, D3QN and Rainbow, each with or without the guide.

One loop for every method (paper1_change_plan.md item 1-T), so logging, budget, resets and
evaluation are identical and only the algorithm (and guide) differs.

    python3 -m rainbow_ablation.paper1.train --arm d3qn --seed 1
    python3 -m rainbow_ablation.paper1.train --algo rainbow --guide on --seed 2

Arms (shorthand for --algo/--guide):
    dqn, ddqn, d3qn, rainbow      -> guide off
    d3qn_g, rainbow_g             -> guide on (0.8 -> 0.1 over 100 episodes, 15% noise)

Output: <results-dir>/<arm>_s<seed>/<run>_{episodes.csv, evals.csv, training.png,
run_meta.json, model_final.pth, model_best.pth}.

Based on rainbow_ablation/train.py (the ablation study's loop). Helpers that are already
tested there are imported, not copied, so the two studies cannot silently diverge.
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import rclpy
from rclpy.utilities import remove_ros_args
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

# ---------------- NEW CODE ADDED: reuse the tested helpers from the ablation loop ----------------
# Reason: same reset, start pose, guide, heuristic and logging helpers for both papers.
from rainbow_ablation.train import (
    set_seed, prepare_episode, heuristic_action, goal_for_episode, git_info, write_json,
    MAX_STEPS, UPDATES_PER_STEP, START_POSE, FINAL_GOAL,
    GUIDED_START, GUIDED_END, GUIDED_EPISODES, GUIDED_NOISE,
)
import matplotlib.pyplot as plt     # backend already set to Agg by rainbow_ablation.train
# ---------------- END NEW CODE ----------------
from rainbow_ablation.environment import RobotEnvironment
from rainbow_ablation.paper1.agent_dqn import DQNAgent
from rainbow_ablation.paper1.agent_ddqn import DDQNAgent
from rainbow_ablation.paper1.agent_d3qn import D3QNAgent
from rainbow_ablation.paper1.agent_rainbow import build_rainbow


SEED = 1
EPISODES = 200            # item 3.3 (was train_DDDQN.py:69 = 300, train_ini.py:52 = 120)
EVAL_EVERY = 20           # item 3.6 (none in train_DDDQN.py)
EVAL_EPISODES = 5

# ---------------- NEW CODE ADDED: method registry and arm shorthands ----------------
# Reason (item 3.1, 5.1): one loop for all methods; *_g = guided.
AGENTS = {
    "dqn": lambda: DQNAgent(52, 7),
    "ddqn": lambda: DDQNAgent(52, 7),
    "d3qn": lambda: D3QNAgent(52, 7),
    "rainbow": lambda: build_rainbow(52, 7),
}
ARMS = {
    "dqn": ("dqn", False), "ddqn": ("ddqn", False), "d3qn": ("d3qn", False),
    "rainbow": ("rainbow", False), "d3qn_g": ("d3qn", True), "rainbow_g": ("rainbow", True),
}
# ---------------- END NEW CODE ----------------

EPISODE_FIELDS = [
    "episode", "outcome", "reward", "steps", "env_steps_total", "loss",
    "epsilon", "random_steps",                       # NEW (item 3.8): epsilon-random steps
    "p_guided", "guided_steps", "guided_noise_steps",
    "replay_size", "learning_steps", "goal_dist", "min_lidar",
    "wall_time_s", "sim_time_s", "rtf",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6",
]
EVAL_FIELDS = [
    "episode", "env_steps_total", "wins", "n", "success_rate", "outcomes",
    # NEW (item 5.2): overestimation check, averaged over the eval episodes
    "q_start_mean", "disc_return_mean", "overestimation_mean",
    # (review fix 1) overestimation is averaged only over eval episodes that ended in a
    # true terminal (goal/collision); truncated episodes have no Monte Carlo return.
    "overestimation_n",
    # (review fix 1, plan 5.2) mean max-Q over the fixed 300 probe states
    "q_probe_mean",
]

# ---------------- NEW CODE ADDED: fixed probe states for the action-collapse check ----------------
# Reason (item 5.3): the same 300 synthetic states for every run (own RNG, does not touch
# the training seed): LiDAR 0.1-1 (normalised), goal distance 0.3-0.8, goal angle -1..1.
def probe_states(n=300, seed=12345):
    rng = np.random.default_rng(seed)
    return np.array([
        np.concatenate([rng.uniform(0.1, 1.0, 50), [rng.uniform(0.3, 0.8)], [rng.uniform(-1, 1)]])
        for _ in range(n)
    ], dtype=np.float32)


def action_histogram(actions, k=7):
    counts = Counter(int(a) for a in actions)
    return [counts.get(i, 0) for i in range(k)]
# ---------------- END NEW CODE ----------------


def guided_probability(episode, guide):
    """Same schedule as rainbow_ablation/train.py:104-113, switched by the guide flag."""
    # ---------------- CHANGED CODE (was rainbow_ablation/train.py:104-106, keyed on arm "N"): ----------------
    # Reason: the guide is now an on/off flag usable with any algorithm (item 5.1).
    if not guide:
        return 0.0
    # ---------------- END CHANGED CODE ----------------
    if episode >= GUIDED_EPISODES:
        return GUIDED_END
    frac = (episode - 1) / max(1, GUIDED_EPISODES - 1)
    return GUIDED_START + (GUIDED_END - GUIDED_START) * frac


def run_episode(env, agent, greedy, episode=0, guide=False, env_steps_before=0, gamma=0.99):
    """One episode (copied from rainbow_ablation/train.py:143-208, changes marked)."""
    state = env.normalize(env.get_state())
    total_reward, losses = 0.0, []
    action_counts = [0] * 7
    reason, steps_taken, terminated = "timeout", 0, False
    p_guided = 0.0 if greedy else guided_probability(episode, guide)
    guided_steps, guided_noise_steps, random_steps = 0, 0, 0
    # ---------------- NEW CODE ADDED: overestimation bookkeeping (item 5.2) ----------------
    q_start = float(np.max(agent.estimate_q(state))) if greedy else float("nan")
    disc_return, discount = 0.0, 1.0
    # ---------------- END NEW CODE ----------------

    for step in range(MAX_STEPS):
        steps_taken = step + 1
        if greedy:
            action = agent.select_greedy_action(state)
        elif random.random() < p_guided:
            guided_steps += 1
            if random.random() < GUIDED_NOISE:
                guided_noise_steps += 1
                action = random.choices(range(7), weights=[4, 1, 1, 1, 1, 1, 1])[0]
            else:
                action = heuristic_action(state, env.MAX_LIDAR)
        else:
            # ---------------- CHANGED CODE (was rainbow_ablation/train.py:173, action = agent.select_action(state)): ----------------
            # Reason (items 2.8, 3.8): epsilon follows environment steps; count epsilon-random steps.
            agent.update_epsilon(env_steps_before + step)
            action, was_random = agent.select_action(state)
            random_steps += int(was_random)
            # ---------------- END CHANGED CODE ----------------
        action_counts[action] += 1
        next_state, reward, terminated, reason = env.step(action)

        if not greedy:
            # Only goal / collision are true terminals (item 2.9). A time limit or the
            # no-progress cut-off truncates the episode but must still bootstrap.
            time_limit = (step == MAX_STEPS - 1) and not terminated
            truncated = time_limit or (terminated and reason == "no_progress")
            terminal = terminated and reason != "no_progress"
            agent.remember(state, action, reward, next_state, terminal, truncated)
            for _ in range(UPDATES_PER_STEP):          # item 3.2: 2 updates per step for all
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

        disc_return += discount * float(reward)          # NEW (item 5.2)
        discount *= gamma
        state = next_state
        total_reward += float(reward)
        if terminated:
            break

    return {
        "reward": total_reward, "steps": steps_taken,
        "outcome": reason if terminated else "timeout",
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "actions": action_counts, "p_guided": p_guided,
        "guided_steps": guided_steps, "guided_noise_steps": guided_noise_steps,
        "random_steps": random_steps,
        "q_start": q_start, "disc_return": disc_return,
        "true_terminal": bool(terminated and reason in ("goal_reached", "collision_warning")),
    }


def parse_args(argv):
    p = argparse.ArgumentParser(description="Paper 1: train DQN / DDQN / D3QN / Rainbow.")
    p.add_argument("--arm", choices=sorted(ARMS), help="shorthand for --algo + --guide")
    p.add_argument("--algo", choices=sorted(AGENTS))
    p.add_argument("--guide", choices=["on", "off"], default="off")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--run-name", help="default: <arm>_s<seed>, e.g. d3qn_g_s1")
    p.add_argument("--results-dir", default="results_paper1")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(remove_ros_args(argv)[1:])
    if args.arm:
        args.algo, guide = ARMS[args.arm]
        args.guide = "on" if guide else "off"
    elif not args.algo:
        p.error("give --arm, or --algo (and optionally --guide)")
    args.arm = args.algo + ("_g" if args.guide == "on" else "")
    return args


def save_plot(path, run_name, episode_rows, eval_rows):
    x = [r["episode"] for r in episode_rows]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Paper 1 training - {run_name}")
    rewards = [r["reward"] for r in episode_rows]
    w = min(10, len(rewards))
    axes[0, 0].plot(x, rewards, label="reward")
    axes[0, 0].plot(x, [np.mean(rewards[max(0, i - w + 1):i + 1]) for i in range(len(rewards))],
                    label=f"moving avg ({w})")
    axes[0, 0].legend(); axes[0, 0].set_title("Episode reward")
    axes[0, 1].plot(x, [r["loss"] for r in episode_rows]); axes[0, 1].set_title("Mean loss")
    axes[0, 2].plot(x, [r["epsilon"] for r in episode_rows], label="epsilon (end of episode)")
    axes[0, 2].plot(x, [r["p_guided"] for r in episode_rows], label="guide probability")
    axes[0, 2].legend(); axes[0, 2].set_title("Exploration")
    axes[1, 0].plot(x, [r["steps"] for r in episode_rows], label="steps")
    axes[1, 0].plot(x, [r["guided_steps"] for r in episode_rows], label="guide steps")
    axes[1, 0].plot(x, [r["random_steps"] for r in episode_rows], label="epsilon-random steps")
    axes[1, 0].legend(); axes[1, 0].set_title("Steps per episode")
    if eval_rows:
        ex = [r["episode"] for r in eval_rows]
        axes[1, 1].plot(ex, [r["success_rate"] for r in eval_rows], marker="o")
        axes[1, 2].plot(ex, [r["q_start_mean"] for r in eval_rows], marker="o", label="predicted max Q(s0)")
        axes[1, 2].plot(ex, [r["disc_return_mean"] for r in eval_rows], marker="s", label="actual discounted return")
        axes[1, 2].legend()
    axes[1, 1].set_ylim(-0.05, 1.05); axes[1, 1].set_title("Greedy success (guide off)")
    axes[1, 2].set_title("Overestimation check")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3); ax.set_xlabel("Training episode")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    args = parse_args(sys.argv if argv is None else argv)
    run_name = args.run_name or f"{args.arm}_s{args.seed}"
    run_dir = os.path.abspath(os.path.join(args.results_dir, run_name))
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.overwrite:
        sys.exit(f"{run_dir} already has files; refusing to overwrite (use --overwrite).")
    os.makedirs(run_dir, exist_ok=True)

    def out(suffix):
        return os.path.join(run_dir, f"{run_name}_{suffix}")

    model_path, best_path = out("model_final.pth"), out("model_best.pth")
    meta_path, plot_path = out("run_meta.json"), out("training.png")
    guide = args.guide == "on"

    rclpy.init(args=argv if argv is not None else sys.argv)
    env = agent = None
    set_seed(args.seed)                         # before the agent: same seed -> same init per algo

    commit, dirty = git_info()
    meta = {
        "study": "paper1", "run_name": run_name, "arm": args.arm, "algo": args.algo,
        "guide": guide, "seed": args.seed, "episodes": args.episodes,
        "eval_every": args.eval_every, "eval_episodes": args.eval_episodes,
        "max_steps": MAX_STEPS, "updates_per_step": UPDATES_PER_STEP,
        "start_pose": START_POSE, "goal": FINAL_GOAL,
        "guide_schedule": {"start": GUIDED_START, "end": GUIDED_END,
                           "episodes": GUIDED_EPISODES, "noise": GUIDED_NOISE},
        "git_commit": commit, "git_dirty": dirty, "torch_version": torch.__version__,
        "slot": os.environ.get("RAINBOW_SLOT"),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "gazebo_master_uri": os.environ.get("GAZEBO_MASTER_URI", "default"),
        "parallel_runs": os.environ.get("RAINBOW_PARALLEL"),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "end_time": None, "status": "running",
    }
    write_json(meta_path, meta)

    episode_rows, eval_rows, rtf_values = [], [], []
    best_success, env_steps_total = -1.0, 0
    t0 = time.monotonic()
    ep_file = open(out("episodes.csv"), "w", newline="")
    ev_file = open(out("evals.csv"), "w", newline="")
    ep_writer = csv.DictWriter(ep_file, fieldnames=EPISODE_FIELDS)
    ev_writer = csv.DictWriter(ev_file, fieldnames=EVAL_FIELDS)
    ep_writer.writeheader(); ev_writer.writeheader()

    try:
        env = RobotEnvironment()
        agent = AGENTS[args.algo]()
        meta["device"] = str(agent.device)
        # ---------------- NEW CODE ADDED: hyperparameters recorded per run (item 5.9) ----------------
        # (review fix 8) read Rainbow's values from the built agent, not a second hard-coded copy
        meta["hyperparameters"] = (
            dict(agent.cfg) if hasattr(agent, "cfg") else
            {"learning_rate": agent.optimizer.param_groups[0]["lr"], "gamma": agent.gamma,
             "batch_size": agent.batch_size, "memory_size": agent.memory.capacity,
             "n_step": agent.memory.n_step, "num_atoms": agent.num_atoms,
             "v_min": agent.v_min, "v_max": agent.v_max,
             "target_update_every": agent.target_update_every,
             "warmup_transitions": agent.warmup_transitions, "grad_clip": 10.0,
             "per_alpha": agent.memory.alpha, "per_beta_start": agent.memory.beta_start,
             "per_beta_frames": agent.memory.beta_frames, "noisy_sigma0": 0.5})
        # ---------------- END NEW CODE ----------------

        # ---------------- NEW CODE ADDED: initial action-collapse check (item 5.3) ----------------
        probes = probe_states()
        # (review fix 4) save/restore every RNG state around the probe, so the check does
        # not consume random numbers that training would otherwise use
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        greedy_hist = action_histogram(agent.select_greedy_action(s) for s in probes)
        explore_hist = action_histogram(agent.select_action(s)[0] for s in probes)
        meta["initial_greedy_action_hist"] = greedy_hist
        meta["initial_explore_action_hist"] = explore_hist
        meta["initial_greedy_top_share"] = round(max(greedy_hist) / len(probes), 3)
        random.setstate(rng_state[0]); np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2])
        if rng_state[3] is not None:
            torch.cuda.set_rng_state_all(rng_state[3])
        # ---------------- END NEW CODE ----------------
        write_json(meta_path, meta)
        env.get_logger().info(
            f"Run {run_name} | algo={args.algo} guide={guide} | device={agent.device} | "
            f"initial greedy actions {greedy_hist}")

        sim_clock = {"t": None}

        def on_clock(msg):
            sim_clock["t"] = msg.clock.sec + msg.clock.nanosec * 1e-9
        env.create_subscription(Clock, "/clock", on_clock,
                                QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        deadline = time.monotonic() + 10.0
        while rclpy.ok() and env.get_state() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for initial LiDAR/odometry.")
            rclpy.spin_once(env, timeout_sec=0.1)

        first = True
        for episode in range(1, args.episodes + 1):
            env.goal_x, env.goal_y = goal_for_episode(episode)
            if not prepare_episode(env, first):          # item 3.5: reset + begin_episode(START_POSE)
                meta["status"] = f"stopped: reset failed at episode {episode}"
                break
            first = False

            sim0, wall0 = sim_clock["t"], time.monotonic()
            result = run_episode(env, agent, greedy=False, episode=episode, guide=guide,
                                 env_steps_before=env_steps_total)
            sim1, wall1 = sim_clock["t"], time.monotonic()
            sim_dt = (sim1 - sim0) if (sim0 is not None and sim1 is not None) else float("nan")
            rtf = sim_dt / (wall1 - wall0) if wall1 > wall0 and sim_dt == sim_dt else float("nan")
            if rtf == rtf:
                rtf_values.append(rtf)
            env_steps_total += result["steps"]
            agent.update_epsilon(env_steps_total)   # (review fix 11b) epsilon column = value at episode end

            row = {
                "episode": episode, "outcome": result["outcome"],
                "reward": round(result["reward"], 4), "steps": result["steps"],
                "env_steps_total": env_steps_total, "loss": result["loss"],
                "epsilon": round(float(agent.epsilon), 4), "random_steps": result["random_steps"],
                "p_guided": round(result["p_guided"], 4),
                "guided_steps": result["guided_steps"],
                "guided_noise_steps": result["guided_noise_steps"],
                "replay_size": len(agent.memory), "learning_steps": agent.learning_steps,
                "goal_dist": round(env.last_goal_distance, 4),
                "min_lidar": round(env.last_min_lidar, 4),
                "wall_time_s": round(time.monotonic() - t0, 1),
                "sim_time_s": round(sim_dt, 2), "rtf": round(rtf, 3),
            }
            row.update({f"a{i}": c for i, c in enumerate(result["actions"])})
            episode_rows.append(row)
            ep_writer.writerow(row); ep_file.flush()
            env.get_logger().info(
                f"[{run_name}] Ep {episode}/{args.episodes} | {result['outcome']} | "
                f"R={result['reward']:.1f} | steps={result['steps']} | eps={agent.epsilon:.2f} | "
                f"guide={result['guided_steps']} rand={result['random_steps']} | rtf={rtf:.2f}")
            agent.save(model_path)

            if args.eval_every and episode % args.eval_every == 0:
                outcomes, qs, gs = [], [], []
                q_probe = float(np.mean(np.max(agent.estimate_q(probes), axis=1)))
                env.goal_x, env.goal_y = FINAL_GOAL
                for _ in range(args.eval_episodes):
                    if not prepare_episode(env, first=False):
                        break
                    r = run_episode(env, agent, greedy=True)
                    outcomes.append(r["outcome"])
                    if r["true_terminal"]:
                        qs.append(r["q_start"]); gs.append(r["disc_return"])
                wins, n = outcomes.count("goal_reached"), len(outcomes)
                rate = wins / n if n else float("nan")
                ev = {"episode": episode, "env_steps_total": env_steps_total, "wins": wins, "n": n,
                      "success_rate": rate, "outcomes": ";".join(outcomes),
                      "q_start_mean": round(float(np.mean(qs)), 3) if qs else float("nan"),
                      "disc_return_mean": round(float(np.mean(gs)), 3) if gs else float("nan"),
                      "overestimation_mean": round(float(np.mean(qs) - np.mean(gs)), 3) if qs else float("nan"),
                      "overestimation_n": len(qs), "q_probe_mean": round(q_probe, 3)}
                eval_rows.append(ev); ev_writer.writerow(ev); ev_file.flush()
                env.get_logger().info(f"[{run_name}] [EVAL] ep {episode}: {wins}/{n} | "
                                      f"Q(s0)={ev['q_start_mean']} vs G={ev['disc_return_mean']}")
                if n and rate > best_success:
                    best_success = rate
                    agent.save(best_path)
        else:
            meta["status"] = "completed"

    except KeyboardInterrupt:
        meta["status"] = f"interrupted after {len(episode_rows)} episodes"
    except Exception as error:
        meta["status"] = f"failed: {error}"
        raise
    finally:
        ep_file.close(); ev_file.close()
        if agent is not None:
            agent.save(model_path)
            meta["learning_steps"] = agent.learning_steps
        meta["episodes_completed"] = len(episode_rows)
        meta["env_steps_total"] = env_steps_total
        meta["wall_time_h"] = round((time.monotonic() - t0) / 3600, 3)
        meta["rtf_mean"] = round(float(np.mean(rtf_values)), 3) if rtf_values else None
        meta["rtf_min"] = round(float(np.min(rtf_values)), 3) if rtf_values else None
        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
        write_json(meta_path, meta)
        if episode_rows:
            save_plot(plot_path, run_name, episode_rows, eval_rows)
        print("Run folder:", run_dir)
        if env is not None:
            try:
                env.publish_stop(); env.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

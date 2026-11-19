"""Study v2 training loop: five worlds, moving obstacles, a random goal (of 4) per episode.

Reuses the v1 learning code UNCHANGED (train.run_episode, agents, guide schedule, probe states, plot);
train.py itself is not modified, so v1 stays reproducible from this branch too. Differences from v1:
  - --world 2 | 3 | 4s | 4 | 6 (worlds_v2.STUDY_STAGES); Gazebo must be started with
    turtlebot3_drl_v2.launch.py stage:=<world>; needs DQN_SIM_TIME_STEP=1 (actions timed on /clock)
  - every episode (also the first) starts with episode_v2.prepare_episode_v2: /reset_simulation,
    warm-up W (same obstacle phase every episode), teleport to the start pose
  - training goal: one of the world's 4 goals in shuffled blocks (each goal once per 4 episodes), drawn
    by a SEPARATE RNG seeded from --seed, so for a
    given seed every method trains on the same goal sequence and goal draws do not use training randomness
  - greedy checks: one greedy episode per goal from the start pose (4 per check)
  - arms: dqn, ddqn, d3qn, rainbow, rainbow_g (d3qn_g dropped in v2); results in results_v2/
Approved v2 algorithm changes (2026-11-19), applied identically to every method:
  - REWARD SCALING: every reward the agent LEARNS from is multiplied by REWARD_SCALE = 0.01 (environment
    and logged rewards stay in v1 units). Positive scaling leaves the optimal policy unchanged; with
    errors mostly < 1 the Huber loss acts like squared error, so the +1 goal target is fitted (v1 probe:
    Q ~ 12 one step before a +100 goal). Rainbow's C51 support is scaled the same way: [-0.75, 1.25].
  - PER BETA (Rainbow): 0.4 -> 1.0 linearly over the episode budget (v1: <= 0.43 reached).
Network, optimiser, epsilon schedule, guide and replay sizes are unchanged from v1.

  DQN_SIM_TIME_STEP=1 python3 -m dqn_comparison.train_v2 --world 4 --arm d3qn --seed 1
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
import torch
from rclpy.utilities import remove_ros_args

from dqn_comparison.common import (set_seed, git_info, write_json, MAX_STEPS, UPDATES_PER_STEP,
                                   GUIDED_START, GUIDED_END, GUIDED_EPISODES, GUIDED_NOISE)
from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.episode_v2 import prepare_episode_v2
from dqn_comparison.train import (AGENTS, EPISODE_FIELDS, probe_states, action_histogram,
                                  run_episode, save_plot)
from dqn_comparison.agent_rainbow import build_rainbow
from dqn_comparison import worlds_v2 as W

EPISODES = 200            # placeholder; the v2 budget is set by the pilot and fixed before the main runs
EVAL_EVERY = 20
GOAL_RNG_OFFSET = 10_000  # goal RNG seed = seed + offset (independent of the training RNG)
ARMS_V2 = {"dqn": ("dqn", False), "ddqn": ("ddqn", False), "d3qn": ("d3qn", False),
           "rainbow": ("rainbow", False), "rainbow_g": ("rainbow", True)}

REWARD_SCALE = 0.01
RAINBOW_SUPPORT = (-75.0 * REWARD_SCALE, 125.0 * REWARD_SCALE)   # v1 support, scaled like the rewards
PER_BETA_START = 0.4
AGENTS_V2 = {"dqn": AGENTS["dqn"], "ddqn": AGENTS["ddqn"], "d3qn": AGENTS["d3qn"],
             "rainbow": lambda: build_rainbow(52, 7, v_min=RAINBOW_SUPPORT[0], v_max=RAINBOW_SUPPORT[1])}

EPISODE_FIELDS_V2 = ["world", "goal_index", "goal_x", "goal_y", "t_start", "per_beta"] + EPISODE_FIELDS
EVAL_FIELDS_V2 = ["episode", "env_steps_total", "wins", "n", "success_rate", "outcomes",
                  "steps", "q_start_mean", "disc_return_mean", "q_probe_mean"]


def apply_reward_scale(agent, scale=REWARD_SCALE):
    """Every transition the agent stores (and so learns from) carries reward * scale."""
    store = agent.remember

    def remember(state, action, reward, next_state, terminal, truncated=False):
        return store(state, action, reward * scale, next_state, terminal, truncated)
    agent.remember = remember
    return agent


def per_beta(episode, episodes, start=PER_BETA_START):
    """PER importance-sampling exponent for this episode: start at episode 0, 1.0 at the last episode."""
    return start + (1.0 - start) * min(1.0, episode / episodes)


def world_arg(v):
    v = v if v == "4s" else int(v)
    if v not in W.STUDY_STAGES:
        raise argparse.ArgumentTypeError(f"world must be one of {W.STUDY_STAGES}")
    return v


def goal_sequence(seed, episodes, n_goals=4):
    """Goal index for every training episode: depends only on the seed (same for every method).

    Shuffled blocks: every block of n_goals episodes contains each goal once, in random order, so the
    goals are exactly balanced (an i.i.d. draw gave e.g. 35 vs 55 of 200 for seed 1) while the next goal
    stays unpredictable within a block."""
    rng = random.Random(seed + GOAL_RNG_OFFSET)
    seq = []
    while len(seq) < episodes:
        block = list(range(n_goals))
        rng.shuffle(block)
        seq += block
    return seq[:episodes]


def untracked_code_files():
    """Untracked files in the CODE folders (not results): common.git_info ignores untracked files on
    purpose (run outputs), so a new uncommitted code file would otherwise go unnoticed."""
    here = os.path.dirname(os.path.realpath(__file__))
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=here, text=True).strip()
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--",
                                       "dqn_comparison", "scripts", "turtlebot3_simulations"], cwd=top, text=True)
        return [line[3:] for line in out.splitlines() if line.startswith("??")]
    except Exception:
        return None


def parse_args(argv):
    p = argparse.ArgumentParser(description="Study v2: train DQN / DDQN / D3QN / Rainbow in one world.")
    p.add_argument("--world", type=world_arg, required=True)
    p.add_argument("--arm", choices=sorted(ARMS_V2), required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--run-name", help="default: w<world>_<arm>_s<seed>, e.g. w4_d3qn_s1")
    p.add_argument("--results-dir", default="results_v2")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(remove_ros_args(argv)[1:])
    args.algo, args.guide = ARMS_V2[args.arm]
    return args


def main(argv=None):
    args = parse_args(sys.argv if argv is None else argv)
    lay = W.LAYOUTS[args.world]
    run_name = args.run_name or f"w{args.world}_{args.arm}_s{args.seed}"
    run_dir = os.path.abspath(os.path.join(args.results_dir, run_name))
    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.overwrite:
        sys.exit(f"{run_dir} already has files; refusing to overwrite (use --overwrite).")
    os.makedirs(run_dir, exist_ok=True)

    def out(suffix):
        return os.path.join(run_dir, f"{run_name}_{suffix}")

    model_path, best_path = out("model_final.pth"), out("model_best.pth")
    meta_path, plot_path = out("run_meta.json"), out("training.png")
    goals = goal_sequence(args.seed, args.episodes)

    rclpy.init(args=argv if argv is not None else sys.argv)
    env = agent = None
    set_seed(args.seed)                          # before the agent: same seed -> same init per algo
    commit, dirty = git_info()
    meta = {
        "study": "v2", "run_name": run_name, "world": args.world, "arm": args.arm, "algo": args.algo,
        "guide": args.guide, "seed": args.seed, "episodes": args.episodes, "eval_every": args.eval_every,
        "max_steps": MAX_STEPS, "updates_per_step": UPDATES_PER_STEP,
        "layout": {"W": lay["W"], "start": lay["start"], "goals": lay["goals"], "tests": lay["tests"]},
        "goal_rng_seed": args.seed + GOAL_RNG_OFFSET,
        "goal_counts": [goals.count(i) for i in range(4)],
        "sim_time_stepping": os.environ.get("DQN_SIM_TIME_STEP") == "1",
        "rtf_cap": os.environ.get("V2_RTF_CAP"),
        "reward_scale": REWARD_SCALE, "rainbow_support": RAINBOW_SUPPORT,
        "per_beta_schedule": f"{PER_BETA_START} + {1 - PER_BETA_START} * episode / episodes (Rainbow only)",
        "q_logging": "q_start_mean and q_probe_mean are divided by reward_scale (environment reward units)",
        "guide_schedule": {"start": GUIDED_START, "end": GUIDED_END,
                           "episodes": GUIDED_EPISODES, "noise": GUIDED_NOISE},
        "git_commit": commit, "git_dirty": dirty, "git_untracked_code": untracked_code_files(),
        "torch_version": torch.__version__,
        "slot": os.environ.get("RAINBOW_SLOT"), "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "gazebo_master_uri": os.environ.get("GAZEBO_MASTER_URI", "default"),
        "parallel_runs": os.environ.get("RAINBOW_PARALLEL"),
        "start_time": datetime.now().isoformat(timespec="seconds"), "end_time": None, "status": "running",
    }
    write_json(meta_path, meta)

    episode_rows, eval_rows, rtf_values = [], [], []
    best_success, env_steps_total = -1.0, 0
    t0 = time.monotonic()
    ep_file = open(out("episodes.csv"), "w", newline="")
    ev_file = open(out("evals.csv"), "w", newline="")
    ep_writer = csv.DictWriter(ep_file, fieldnames=EPISODE_FIELDS_V2)
    ev_writer = csv.DictWriter(ev_file, fieldnames=EVAL_FIELDS_V2)
    ep_writer.writeheader(); ev_writer.writeheader()

    try:
        env = RobotEnvironment()
        if not env.sim_time_stepping:
            raise RuntimeError("train_v2 needs DQN_SIM_TIME_STEP=1 (actions timed on simulated time)")
        agent = apply_reward_scale(AGENTS_V2[args.algo]())
        meta["device"] = str(agent.device)
        meta["hyperparameters"] = (dict(agent.cfg) if hasattr(agent, "cfg") else
                                   {"learning_rate": agent.optimizer.param_groups[0]["lr"], "gamma": agent.gamma,
                                    "batch_size": agent.batch_size, "n_step": agent.memory.n_step,
                                    "v_min": agent.v_min, "v_max": agent.v_max,
                                    "per_beta_frames": agent.memory.beta_frames})
        probes = probe_states()
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        greedy_hist = action_histogram(agent.select_greedy_action(s) for s in probes)
        meta["initial_greedy_action_hist"] = greedy_hist
        random.setstate(rng_state[0]); np.random.set_state(rng_state[1]); torch.set_rng_state(rng_state[2])
        write_json(meta_path, meta)

        deadline = time.monotonic() + 10.0
        while rclpy.ok() and env.get_state() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for initial LiDAR/odometry.")
            rclpy.spin_once(env, timeout_sec=0.1)

        for episode in range(1, args.episodes + 1):
            gi = goals[episode - 1]
            goal = lay["goals"][gi]
            t_start = prepare_episode_v2(env, args.world, lay["start"], goal)   # reset, warm-up W, teleport
            beta = per_beta(episode, args.episodes) if hasattr(agent.memory, "beta_override") else float("nan")
            if beta == beta:
                agent.memory.beta_override = beta

            wall0 = time.monotonic()
            result = run_episode(env, agent, greedy=False, episode=episode, guide=args.guide,
                                 env_steps_before=env_steps_total)
            sim_dt, wall_dt = env.sim_now - t_start, time.monotonic() - wall0
            rtf = sim_dt / wall_dt if wall_dt > 0 else float("nan")
            rtf_values.append(rtf)
            env_steps_total += result["steps"]
            agent.update_epsilon(env_steps_total)

            row = {"world": args.world, "goal_index": gi, "goal_x": goal[0], "goal_y": goal[1],
                   "t_start": round(t_start, 4), "per_beta": round(beta, 4),
                   "episode": episode, "outcome": result["outcome"], "reward": round(result["reward"], 4),
                   "steps": result["steps"], "env_steps_total": env_steps_total, "loss": result["loss"],
                   "epsilon": round(float(agent.epsilon), 4), "random_steps": result["random_steps"],
                   "p_guided": round(result["p_guided"], 4), "guided_steps": result["guided_steps"],
                   "guided_noise_steps": result["guided_noise_steps"], "replay_size": len(agent.memory),
                   "learning_steps": agent.learning_steps, "goal_dist": round(env.last_goal_distance, 4),
                   "min_lidar": round(env.last_min_lidar, 4), "wall_time_s": round(time.monotonic() - t0, 1),
                   "sim_time_s": round(sim_dt, 2), "rtf": round(rtf, 3)}
            row.update({f"a{i}": c for i, c in enumerate(result["actions"])})
            episode_rows.append(row)
            ep_writer.writerow(row); ep_file.flush()
            env.get_logger().info(
                f"[{run_name}] Ep {episode}/{args.episodes} goal G{gi + 1} | {result['outcome']} | "
                f"R={result['reward']:.1f} | steps={result['steps']} | eps={agent.epsilon:.2f} | rtf={rtf:.2f}")
            agent.save(model_path)

            if args.eval_every and episode % args.eval_every == 0:
                outcomes, steps, qs, gs = [], [], [], []
                q_probe = float(np.mean(np.max(agent.estimate_q(probes), axis=1))) / REWARD_SCALE
                for g in lay["goals"]:                     # one greedy episode per goal, from the start pose
                    prepare_episode_v2(env, args.world, lay["start"], g)
                    r = run_episode(env, agent, greedy=True)
                    outcomes.append(r["outcome"]); steps.append(r["steps"])
                    if r["true_terminal"]:
                        qs.append(r["q_start"] / REWARD_SCALE); gs.append(r["disc_return"])
                wins, n = outcomes.count("goal_reached"), len(outcomes)
                ev = {"episode": episode, "env_steps_total": env_steps_total, "wins": wins, "n": n,
                      "success_rate": wins / n, "outcomes": ";".join(outcomes),
                      "steps": ";".join(map(str, steps)),
                      "q_start_mean": round(float(np.mean(qs)), 3) if qs else float("nan"),
                      "disc_return_mean": round(float(np.mean(gs)), 3) if gs else float("nan"),
                      "q_probe_mean": round(q_probe, 3)}
                eval_rows.append(ev); ev_writer.writerow(ev); ev_file.flush()
                env.get_logger().info(f"[{run_name}] [EVAL] ep {episode}: {wins}/{n} goals ({ev['outcomes']})")
                if wins / n > best_success:
                    best_success = wins / n
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
        meta["rtf_mean"] = round(float(np.nanmean(rtf_values)), 3) if rtf_values else None
        meta["rtf_min"] = round(float(np.nanmin(rtf_values)), 3) if rtf_values else None
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

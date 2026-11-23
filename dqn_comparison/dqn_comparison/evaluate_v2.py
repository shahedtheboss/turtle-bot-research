"""Study v2 final evaluation: every (pose, goal) combination of a world, greedy, fixed obstacle phase.

Poses = the world's start + 4 test poses (worlds_v2.LAYOUTS), goals = its 4 goals -> 20 combinations,
each repeated --repeats times (default 2 -> 40 episodes). Every episode starts with prepare_episode_v2
(reset, warm-up W, teleport), so the obstacle phase is identical. Greedy: no epsilon, no noise, no guide.
--policy heuristic evaluates the scripted guide alone (baseline H), as in v1. With --guide-noise 0.15 (H_noisy,
deviation 4) each action is replaced, with that probability, by the forward-biased random action used in training;
--seed makes that noise reproducible.
Needs DQN_SIM_TIME_STEP=1 and Gazebo from turtlebot3_drl_v2.launch.py stage:=<world>.

  DQN_SIM_TIME_STEP=1 python3 -m dqn_comparison.evaluate_v2 --world 4 \
      --model results_v2/w4_d3qn_s1/w4_d3qn_s1_model_final.pth --out-csv results_v2/w4_d3qn_s1/w4_d3qn_s1_final_eval.csv
"""
import argparse
import csv
import os
import random
import sys

import rclpy
from rclpy.utilities import remove_ros_args

from dqn_comparison.common import GUIDED_NOISE_WEIGHTS, heuristic_action, wait_for_sensors, MAX_STEPS
from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.episode_v2 import prepare_episode_v2
from dqn_comparison.learning import load_greedy_agent
from dqn_comparison.train_v2 import world_arg
from dqn_comparison import worlds_v2 as W

FIELDS = ["world", "label", "pose_index", "pose_x", "pose_y", "pose_yaw", "goal_index", "goal_x", "goal_y",
          "repeat", "outcome", "steps", "reward", "t_start", "t_end", "end_x", "end_y"]


def greedy_rollout(env, choose, rng, guide_noise):
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


def heuristic_choice(state, max_lidar, rng, noise):
    """Baseline H (noise 0) or H_noisy: with probability noise, the training guide's forward-biased random action."""
    if noise > 0 and rng.random() < noise:
        return rng.choices(range(7), weights=GUIDED_NOISE_WEIGHTS)[0]
    return heuristic_action(state, max_lidar)


def combinations(world, repeats):
    """(pose_index, pose, goal_index, goal, repeat) in a fixed order: pose-major, then goal, then repeat."""
    lay = W.LAYOUTS[world]
    poses = [lay["start"]] + lay["tests"]
    return [(pi, p, gi, g, r) for pi, p in enumerate(poses) for gi, g in enumerate(lay["goals"])
            for r in range(repeats)]


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Study v2 final evaluation")
    ap.add_argument("--world", type=world_arg, required=True)
    ap.add_argument("--policy", choices=["greedy", "heuristic"], default="greedy")
    ap.add_argument("--model", help="checkpoint (required for --policy greedy)")
    ap.add_argument("--guide-noise", type=float, default=0.0, help="heuristic only: noise probability (0 = H)")
    ap.add_argument("--seed", type=int, default=1, help="heuristic only: RNG seed of the noise")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args(remove_ros_args(argv)[1:])
    if a.policy == "greedy" and not a.model:
        ap.error("--model is required for --policy greedy")
    if a.policy == "greedy" and a.guide_noise != 0:
        ap.error("--guide-noise is only for --policy heuristic")
    return a


def main(argv=None):
    a = parse_args(sys.argv if argv is None else argv)
    rclpy.init(args=argv if argv is not None else sys.argv)
    env, rows = None, []
    try:
        env = RobotEnvironment()
        if not env.sim_time_stepping:
            raise RuntimeError("evaluate_v2 needs DQN_SIM_TIME_STEP=1")
        env.set_physics_paused(False)        # 2026-11-21: training may have left Gazebo paused
        wait_for_sensors(env)
        if a.policy == "greedy":
            agent = load_greedy_agent(a.model)

            def choose(state, rng, noise):
                return agent.select_greedy_action(state)
        else:
            def choose(state, rng, noise):
                return heuristic_choice(state, env.MAX_LIDAR, rng, noise)

        rng = random.Random(a.seed)
        for pi, pose, gi, goal, rep in combinations(a.world, a.repeats):
            t_start = prepare_episode_v2(env, a.world, pose, goal)
            outcome, reward, steps = greedy_rollout(env, choose, rng, a.guide_noise)
            x, y, _ = env.get_world_pose()
            rows.append({"world": a.world, "label": a.label, "pose_index": pi, "pose_x": pose[0],
                         "pose_y": pose[1], "pose_yaw": pose[2], "goal_index": gi, "goal_x": goal[0],
                         "goal_y": goal[1], "repeat": rep, "outcome": outcome, "steps": steps,
                         "reward": round(reward, 4), "t_start": round(t_start, 4), "t_end": round(env.sim_now, 4),
                         "end_x": round(x, 4), "end_y": round(y, 4)})
            env.get_logger().info(f"[{a.label}] pose {pi} goal G{gi + 1} rep {rep}: {outcome} ({steps} steps)")
        wins = sum(r["outcome"] == "goal_reached" for r in rows)
        print(f"===== {a.label or a.policy} world {a.world}: {wins}/{len(rows)} goals reached =====")
    finally:
        if rows:
            os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
            with open(a.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            print("CSV written:", a.out_csv)
        if env is not None:
            env.publish_stop()
            env.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

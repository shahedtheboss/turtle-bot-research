"""Post hoc diagnostic (analysis only; does not change any project code or results).

For each final epsilon-greedy checkpoint, run ONE greedy episode from the training start P0
(greedy evaluation from a fixed pose is near-deterministic) and log, at every step, the
network's Q-values, the action, the reward, goal distance and min LiDAR. Afterwards, for
successful episodes, compare max_a Q(s_t, a) with the realised discounted return-to-go
G_t = sum_k gamma^k r_{t+k}. If Q tracks G near the goal but not far from it, value
propagation is slow; if Q stays far below G even one step before the goal, the terminal
+100 is not being fitted (e.g. Huber loss + unclipped reward).

Run with scripts/run_q_probe_slot.sh, which starts an isolated headless Gazebo slot (same
isolation as scripts/run_eval_slot.sh) and writes results/q_trajectory/q_trajectory.csv.
"""
import argparse
import csv
import os

import numpy as np
import rclpy

from dqn_comparison.environment import RobotEnvironment
from dqn_comparison.evaluate import (MAX_STEPS, prepare_episode, wait_for_sensors,
                                     load_greedy_agent, DEFAULT_STARTS)

ARMS = ["dqn", "ddqn", "d3qn", "d3qn_g"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rclpy.init()
    env = RobotEnvironment()
    rows = []
    try:
        wait_for_sensors(env)
        for arm in ARMS:
            for seed in range(1, 6):
                run = f"{arm}_s{seed}"
                agent = load_greedy_agent(os.path.join(args.results_dir, run, f"{run}_model_final.pth"))
                prepare_episode(env, DEFAULT_STARTS[0])
                state = env.normalize(env.get_state())
                outcome = "timeout"
                for t in range(MAX_STEPS):
                    q = agent.estimate_q(state)[0]                      # shape [7]
                    action = int(np.argmax(q))                         # same as select_greedy_action
                    next_state, reward, terminated, reason = env.step(action)
                    rows.append({"run": run, "t": t, "action": action, "reward": float(reward),
                                 "max_q": float(q.max()), "q": ";".join(f"{v:.3f}" for v in q),
                                 "goal_dist": state[50] * 6.0, "min_lidar": min(state[:50]) * 3.5,
                                 "outcome": ""})
                    state = next_state
                    if terminated:
                        outcome = str(reason)
                        break
                for r in rows:
                    if r["run"] == run:
                        r["outcome"] = outcome
                env.get_logger().info(f"{run}: {outcome} after {t + 1} steps")
    finally:
        env.publish_stop()
        env.destroy_node()
        rclpy.shutdown()
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("written", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()

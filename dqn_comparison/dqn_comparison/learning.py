"""Shared learning code for study v2: method registry, one training/greedy episode, probe states, plot.

Moved UNCHANGED from the v1 training loop (train.py, version-1 branch, paper1-v1.0) during the
2026-11-21 cleanup, so v2 keeps exactly the learning logic of v1. The "item x.y" notes refer to the v1
change plan.
"""
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from dqn_comparison.common import (heuristic_action, MAX_STEPS, UPDATES_PER_STEP,
                                   GUIDED_START, GUIDED_END, GUIDED_EPISODES, GUIDED_NOISE)
from dqn_comparison.agent_dqn import DQNAgent
from dqn_comparison.agent_ddqn import DDQNAgent
from dqn_comparison.agent_d3qn import D3QNAgent
from dqn_comparison.agent_rainbow import build_rainbow
from dqn_comparison.rainbow_core import RainbowDQNAgent

AGENTS = {
    "dqn": lambda: DQNAgent(52, 7),
    "ddqn": lambda: DDQNAgent(52, 7),
    "d3qn": lambda: D3QNAgent(52, 7),
    "rainbow": lambda: build_rainbow(52, 7),
}

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


def save_plot(path, run_name, episode_rows, eval_rows):
    x = [r["episode"] for r in episode_rows]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Training - {run_name}")
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
        from dqn_comparison.agent_dqn import DQNAgent
        from dqn_comparison.agent_ddqn import DDQNAgent
        from dqn_comparison.agent_d3qn import D3QNAgent
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

"""Helpers shared by the DQN comparison (train.py) and the guided ablation (guided_ablation/train.py).

# ---------------- CHANGED CODE (was rainbow_ablation/train.py:38-140 and 237-252): ----------------
# Reason (2026-11-15 rename to dqn_comparison): these helpers used to live inside the ablation
# training loop and the comparison loop imported them from there. They now live here so both
# studies import the same, single copy. Logic unchanged.
"""

import json
import math
import os
import random
import subprocess
import time

import numpy as np
import rclpy
import torch
from std_srvs.srv import Empty

MAX_STEPS = 250
UPDATES_PER_STEP = 2        # gradient updates per environment step (sim is the bottleneck)
START_POSE = (0.0, 0.0, 0.0)  # TRUE world pose of the robot after a Gazebo reset
                              # (check your launch file's x_pose / y_pose)

# Optional curriculum: easier goals first. Only enable if those goal points are
# free space in your world (otherwise the goal may sit inside an obstacle).
CURRICULUM = False
CURRICULUM_GOALS = [(60, (1.0, -1.0)), (120, (1.5, -1.5))]  # (until_episode, goal)
FINAL_GOAL = (2.0, -2.0)

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


def guide_schedule(episode):
    """Guide probability for a guided run: 0.8 at episode 1, linear to 0.1 at episode 100, then 0.1."""
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


def git_info():
    """Commit hash and whether the working tree had uncommitted changes."""
    # realpath: with --symlink-install this file is reached through build/, so resolve
    # the link to run git inside the repository.
    here = os.path.dirname(os.path.realpath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, text=True).strip())   # CHANGED (2026-11-16): untracked run outputs (results/) are not code changes
        return commit, dirty
    except Exception:
        return "unknown", None


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

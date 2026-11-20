"""Helpers shared by the v2 modules: constants, seeding, guide heuristic, sensor wait, git info, JSON.

Cleaned 2026-11-21: the v1-only parts (fixed start pose/goal, curriculum, v1 reset/prepare_episode,
unused guide_schedule) were removed; they remain on the version-1 branch."""

import json
import math
import os
import random
import subprocess
import time

import numpy as np
import rclpy
import torch

MAX_STEPS = 250
UPDATES_PER_STEP = 2        # gradient updates per environment step (sim is the bottleneck)
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


def wait_for_sensors(node, timeout=10.0):
    deadline = time.monotonic() + timeout
    while rclpy.ok() and node.get_state() is None:
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for LiDAR/odometry data.")
        rclpy.spin_once(node, timeout_sec=0.1)

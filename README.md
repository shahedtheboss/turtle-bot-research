# turtle-bot-research

Deep reinforcement learning for **mapless goal navigation** of a TurtleBot3 Burger, built on
**ROS 2 Humble**, **Gazebo Classic 11** and **PyTorch**.

The robot sees only a 2D LiDAR scan and the distance and angle to its goal. It has no map and no planner,
and it learns to reach goals while avoiding static and moving obstacles.

> This is an active research project. The code is public; the study design, results and paper will be
> released when the work is published.

## Highlights

- **Value-based DRL agents:** DQN-family agents, implemented from scratch in PyTorch, sharing one
  environment and one training loop.
- **Several Gazebo worlds,** from simple walls to moving obstacles. Every episode starts in the same
  state, so runs can be reproduced.
- **Faster than real time without changing behaviour.** Simulated-time stepping lets several isolated
  headless Gazebo instances train in parallel. A check confirms the robot moves the same way as in
  real-time control.
- **Reproducible runs.** Seeds, git commit, machine and device are recorded in every run's metadata.
- **Evaluation and statistics tools:** greedy evaluation, Wilson intervals, IQM, bootstrap confidence
  intervals, effect sizes and exact permutation tests.

## How it works

| Part | Description |
|---|---|
| Observation | 52 values: 50 LiDAR sectors, goal distance, goal angle |
| Actions | 7 discrete velocity commands, each held for 0.5 s of simulated time |
| Reward | Progress toward the goal, plus a bonus for reaching it and a penalty for collisions |
| Episode start | Simulation reset, a fixed warm-up, then the robot is placed at the start |

## Repository layout

ROS 2 package `dqn_comparison/`; the Python code is in `dqn_comparison/dqn_comparison/`.

| File | Contents |
|---|---|
| `environment.py` | Gazebo environment: observations, actions, reward, simulated-time stepping |
| `agent_*.py`, `rainbow_core.py` | Learning agents |
| `learning.py` | Agent registry, episode loop, checkpoint loading |
| `train_v2.py`, `evaluate_v2.py` | Training and evaluation entry points |
| `worlds_v2.py`, `episode_v2.py` | World layouts and episode start |
| `metrics_v2.py`, `stats.py` | Metrics and statistics |
| `scripts/` | Launchers for isolated parallel Gazebo runs, plus checks |
| `dqn_comparison/test/` | Unit tests |

## Getting started

Requirements: Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11, the TurtleBot3 packages, and Python with
PyTorch (a CUDA GPU is recommended).

```bash
git clone <this repo> ~/turtle-bot-research && cd ~/turtle-bot-research
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run the tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/ -q \
  --ignore=dqn_comparison/test/test_flake8.py --ignore=dqn_comparison/test/test_pep257.py
```

Run a short training run in a headless Gazebo slot (slot 1, world 4, agent D3QN, seed 9, 3 episodes):

```bash
EVAL_REPEATS=1 scripts/run_slot_v2.sh 1 4 d3qn 9 --episodes 3 --eval-every 3 \
  --results-dir results_smoke/try --overwrite
```

## Earlier work

The first study (one world, one fixed goal) is on the [`version-1`](../../tree/version-1) branch, with its
code, results and paper. Its archived release is on Zenodo:
[10.5281/zenodo.24313470](https://doi.org/10.5281/zenodo.24313470).

## License

MIT. See `LICENSE`.

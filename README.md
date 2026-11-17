# turtle-bot-research

Value-based deep reinforcement learning for mapless LiDAR goal navigation of a TurtleBot3 Burger in ROS 2 Humble and Gazebo Classic. The stage-4 DRL world has static walls and two cylinders. The world file animates the cylinders through Gazebo plugins, but those plugins were not on `GAZEBO_PLUGIN_PATH` in this setup, so in every Paper 1 run the cylinders stayed static at (2, 2) and (-2, -2).

## Studies in this repository

| Study | Question | Code | Status |
|---|---|---|---|
| **Paper 1 (main): DQN comparison** | How do DQN, Double DQN, Dueling Double DQN (D3QN) and Rainbow DQN compare when the environment, state, actions, reward, budget, seeds and evaluation are identical? And how much does a decaying scripted guide change the result? | `dqn_comparison/dqn_comparison/` | All 30 runs done (6 configurations × seeds 1–5); paper draft in preparation |
| **Paper 2 (later): guided ablation** | A controlled ablation of the guide for Rainbow: guided (G) vs unguided (N) vs a matched random-action control (R) | `dqn_comparison/dqn_comparison/guided_ablation/` | Parked |

The original exploratory code and results (DQN → DDQN → D3QN → Rainbow pilot) are on the [`pilot_runs`](../../tree/pilot_runs) branch.

## Layout

ROS 2 package: `dqn_comparison/`. Its Python code is in `dqn_comparison/dqn_comparison/`.

| File | Contents |
|---|---|
| `environment.py` | Gazebo environment: 52-value state (50 LiDAR sectors + goal distance and angle), 7 actions, reward |
| `common.py` | Shared helpers: Gazebo reset, start pose, the scripted guide and its schedule, seeding, git info |
| `agent_dqn.py`, `agent_ddqn.py`, `agent_d3qn.py`, `agent_rainbow.py` | One file per method; DDQN and D3QN override only their target or network |
| `rainbow_core.py` | Rainbow DQN (NoisyNet, dueling C51, prioritized 3-step replay, Double DQN target) |
| `train.py` | One training loop for all arms (`dqn`, `ddqn`, `d3qn`, `d3qn_g`, `rainbow`, `rainbow_g`) |
| `evaluate.py` | Greedy or heuristic evaluation from 5 fixed start poses, with Wilson intervals |
| `analyze.py` | Summary tables and figures |
| `paper_figures.py` | Publication figures, LaTeX tables and post hoc analyses (analysis only) |
| `HYPERPARAMETERS.md` | All settings |
| `guided_ablation/train.py` | Paper 2 (G / N / R arms), parked |

Other folders:

| Path | Contents |
|---|---|
| `dqn_comparison/test/test_agents.py` | Unit tests |
| `scripts/run_slot.sh` | One training run in an isolated headless Gazebo "slot" (2–3 can run in parallel) |
| `turtlebot3_simulations/`, `turtlebot3_msgs/` | Gazebo worlds, launch files and messages |
| `commands.md` | Every command, in order |
| `docs/previous_project/` | Notes from the earlier project |

## Paper 1 results and reproducibility

| Path or tag | Contents |
|---|---|
| `results/<arm>_s<seed>/` | Per run: episode CSV, greedy-check CSV, final-evaluation CSV, `run_meta.json` (git commit, dirty flag, slot, hyperparameters), Gazebo logs, plots |
| `results/H/`, `results/H_eta/` | Heuristic baselines (guide alone, and with 15% action noise) |
| `results/q_trajectory/` | Post hoc probe: Q-values and rewards along one greedy episode from P0 per final epsilon-greedy checkpoint (`scripts/run_q_probe_slot.sh`) |
| tag `comparison-prereg-v1` | Code, protocol and primary metric fixed before training (15 Nov 2026) |
| tag `comparison-prereg-v1.1` | Seeds raised from three to five before any final result (16 Nov 2026) |
| tag `comparison-analysis-v1.2` | Statistical test code frozen (17 Nov 2026) |

Model checkpoints (`*.pth`) are not in git; the final ones are attached to the paper release.

**Dirty-tree flag.** Eight runs record `git_dirty = true` in `run_meta.json`. The flag was caused by untracked output files in the working tree; the training code was identical across all commits used by the 30 runs.

## Build

The repo root is the colcon workspace. Clone it **outside** any other ROS workspace, then:

    cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && colcon build --symlink-install

Source `install/setup.bash` from this repo only. Don't also source another workspace that has `turtlebot3_*` packages.

## Quick start

    source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/test_agents.py -q
    scripts/run_slot.sh 1 d3qn 1

See `commands.md` for the smoke test, the batch schedule, evaluation and analysis.

## License

MIT, see `LICENSE`.

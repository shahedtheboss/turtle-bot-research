# turtle-bot-research

Value-based deep reinforcement learning for mapless goal navigation of a TurtleBot3 Burger in ROS 2 Humble
and Gazebo Classic 11. This branch (`main`) holds **study v2**: DQN, Double DQN, Dueling Double DQN (D3QN),
Rainbow DQN and Rainbow with a decaying scripted guide, each tuned with an equal budget, compared in five
Gazebo worlds (static and moving obstacles) with four goals drawn at random per episode.

**Study v1** (one world, one fixed goal; paper release `paper1-v1.0`, Zenodo DOI 10.5281/zenodo.24313470) lives on
the [`version-1`](../../tree/version-1) branch with its code, results and paper. v2 reuses the v1 learning code
unchanged (now in `learning.py`).

## Design (summary)

| Item | Setting |
|---|---|
| Worlds | DRL stages 2, 3, 4s (stage 4, obstacles static), 4 (moving), 6; launch `turtlebot3_drl_v2.launch.py stage:=<world>` |
| Episode start | `/reset_simulation`, warm-up W s of simulated time (same obstacle phase every episode), teleport to start |
| Goals | 4 per world; shuffled blocks, one RNG per seed, so all methods see the same goal sequence |
| State / actions | 52 values (50 LiDAR sectors, goal distance, goal angle) / 7 discrete actions, each 0.5 s of simulated time |
| Speed | simulated-time stepping, real-time factor capped at 3, /clock 200 Hz, 5 ms poll (checked equal to v1 real-time motion) |
| Budget | 500 episodes; greedy check (4 goals) every 20 episodes |
| Primary metric | E* = first check with >= 3/4 goals, confirmed by the next check; censored at 520 |
| Final evaluation | 5 poses × 4 goals × 2 repeats, greedy, final checkpoint |
| Tuning | learning rate per algorithm, equal budget, separate seeds (`docs/v2_tuning_protocol.md`) |
| Main runs | 5 arms × 5 worlds × seeds 1-5 = 125 runs (`scripts/main_runs_v2.csv`) |

## Layout

ROS 2 package `dqn_comparison/`, Python code in `dqn_comparison/dqn_comparison/`:

| File | Contents |
|---|---|
| `environment.py` | Gazebo environment: state, actions, reward, simulated-time action timing |
| `agent_dqn.py`, `agent_ddqn.py`, `agent_d3qn.py`, `agent_rainbow.py`, `rainbow_core.py` | The four algorithms |
| `learning.py` | Method registry, one training/greedy episode, probe states, plot, checkpoint loading (from v1) |
| `common.py` | Constants, seeding, guide heuristic, sensor wait, git info |
| `worlds_v2.py` | Per-world layouts (start, goals, test poses, warm-up W) and recorded obstacle paths |
| `episode_v2.py` | Episode start (reset, warm-up, teleport) |
| `train_v2.py` | Training run (one world, arm, seed); tuned learning rates in `TUNED_LR` |
| `evaluate_v2.py` | Final evaluation over all pose × goal combinations |
| `metrics_v2.py`, `stats.py` | E*; Wilson, IQM, bootstrap, Vargha-Delaney A, exact permutation tests, Holm |
| `data/v2_obstacle_paths.csv` | Recorded moving-obstacle paths |

Scripts (`scripts/`):

| Script | Purpose |
|---|---|
| `run_slot_v2.sh` | One run (train + final evaluation) in an isolated headless Gazebo slot |
| `run_main_v2.sh` | The 125 main runs from `main_runs_v2.csv`, N slots in parallel, per machine, with take-over |
| `run_tuning_v2.sh` | The tuning runs |
| `run_sim_time_equivalence.sh`, `sim_time_equivalence.py` | Check: simulated-time stepping moves the robot like v1 real time |
| `v2_*.py` | World layout, obstacle-path recording and checks, reset check |

Docs: `docs/v2_tuning_protocol.md` (tuning protocol and result), `docs/setup_second_machine.md` (lab PC).

## Build

The repo root is the colcon workspace. Clone it outside any other ROS workspace:

    cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && colcon build --symlink-install

## Quick start

    source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/ -q --ignore=dqn_comparison/test/test_flake8.py --ignore=dqn_comparison/test/test_pep257.py
    EVAL_REPEATS=1 scripts/run_slot_v2.sh 1 4 d3qn 9 --episodes 3 --eval-every 3 --results-dir results_smoke/try --overwrite
    scripts/run_main_v2.sh laptop          # main runs planned for this machine (see the script header)

## License

MIT, see `LICENSE`.

# turtle-bot-research

Value-based deep reinforcement learning for mapless LiDAR goal navigation of a TurtleBot3 Burger in ROS 2 Humble and Gazebo Classic. The stage-4 DRL world has static walls and two moving cylinders.

## Studies in this repository

| Study | Question | Code | Status |
|---|---|---|---|
| **Paper 1** | How do DQN, Double DQN, Dueling Double DQN (D3QN) and Rainbow DQN compare when the environment, state, actions, reward, budget, seeds and evaluation are identical? And how much does a decaying scripted guide change the result? | `rainbow_ablation/rainbow_ablation/paper1/` | Code ready; runs pending |
| **Paper 2** | A controlled ablation of the guide for Rainbow: guided (G) vs unguided (N) vs a matched random-action control (R) | `rainbow_ablation/rainbow_ablation/{train,agent,evaluate}.py` | Paused |

The original exploratory code and results (DQN → DDQN → D3QN → Rainbow pilot) are on the [`pilot_runs`](../../tree/pilot_runs) branch.

## Layout

| Path | Contents |
|---|---|
| `rainbow_ablation/rainbow_ablation/environment.py` | Gazebo environment: 52-value state (50 LiDAR sectors + goal distance and angle), 7 actions, reward |
| `rainbow_ablation/rainbow_ablation/agent.py` | Rainbow DQN (NoisyNet, dueling C51, prioritized 3-step replay, Double DQN target) |
| `rainbow_ablation/rainbow_ablation/paper1/` | Paper 1 agents (`agent_dqn.py`, `agent_ddqn.py`, `agent_d3qn.py`, `agent_rainbow.py`), the shared `train.py`, `analyze.py`, `HYPERPARAMETERS.md` |
| `rainbow_ablation/rainbow_ablation/evaluate.py` | Greedy or heuristic evaluation from 5 fixed start poses, with Wilson intervals |
| `rainbow_ablation/test/` | Unit tests (`test_paper1_agents.py`) |
| `scripts/run_slot.sh` | Runs one training run in an isolated headless Gazebo "slot", so 2–3 runs can share the machine |
| `turtlebot3_simulations/`, `turtlebot3_msgs/` | Gazebo worlds, launch files and messages |
| `commands.md` | Every command needed, in order |
| `docs/previous_project/` | Notes from the earlier project |

## Build

The repo root is the colcon workspace. Clone it **outside** any other ROS workspace, then:

    cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && colcon build --symlink-install

Source `install/setup.bash` from this repo only. Don't also source another workspace that has `turtlebot3_*` packages.

## Quick start (Paper 1)

    source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest rainbow_ablation/test/test_paper1_agents.py -q
    TRAIN_MODULE=rainbow_ablation.paper1.train scripts/run_slot.sh 1 d3qn 1

See `commands.md` for the smoke test, the batch schedule, evaluation and analysis.

## License

MIT, see `LICENSE`.

# TurtleBot3 Deep RL Navigation

Reinforcement-learning navigation for a simulated TurtleBot3 Burger in Gazebo, built on ROS 2 Humble and PyTorch. The robot must reach a fixed goal at (2, −2) from a LiDAR and odometry state, using discrete velocity actions.

The project goes from DQN to Double DQN, Dueling DDQN and Rainbow DQN. The next step is DDPG, which needs continuous actions.

| | |
|---|---|
| **Platform** | ROS 2 Humble, Gazebo, TurtleBot3 Burger, Python, PyTorch |
| **Current stage** | Rainbow DQN trained and evaluated. DDPG next. |
| **Full experiment log** | [`mini_dqn_nav/logs.md`](mini_dqn_nav/logs.md) (tables). The original prose log is [`mini_dqn_nav/old_logs.md`](mini_dqn_nav/old_logs.md). |

---

## Contents

1. [Algorithms](#algorithms)
2. [Environment](#environment)
3. [Results](#results)
4. [Setup and running](#setup-and-running)
5. [Repository layout](#repository-layout)
6. [Known discrepancies](#known-discrepancies)
7. [Roadmap](#roadmap)

---

## Algorithms

| Algorithm | Key ideas | Status |
|---|---|---|
| DQN | Q-network, experience replay, target network, ε-greedy exploration | Baseline (E0). Superseded. |
| Double DQN | Policy network picks the next action, target network scores it | Implemented (E1, E2) |
| Dueling DDQN | Separate value and advantage streams | Implemented (E3) |
| Rainbow DQN | Double, Dueling, NoisyNet, C51 distributional, PER, 3-step returns | Implemented and evaluated (E4) |
| DDPG | Continuous actions, actor and critic | Planned |

Rainbow's components are in [`mini_dqn_nav/agent_rainbow.py`](mini_dqn_nav/agent_rainbow.py). The DQN family lives in the archived `inital_codes/*_DDDQN.py` files.

---

## Environment

**State.** 52 values: 50 LiDAR sectors, goal distance and goal angle. Rainbow normalises them before the network sees them.

**Actions.** Seven discrete velocity commands:

| ID | Name | Linear (m/s) | Angular (rad/s) |
|---|---|---|---|
| 0 | forward | 0.15 | 0.0 |
| 1 | soft_forward_left | 0.11 | +0.4 |
| 2 | soft_forward_right | 0.11 | −0.4 |
| 3 | forward_left | 0.08 | +0.8 |
| 4 | forward_right | 0.08 | −0.8 |
| 5 | hard_left | 0.02 | +1.0 |
| 6 | hard_right | 0.02 | −1.0 |

**Reward (current, Rainbow).**

| Term | Value |
|---|---|
| Goal reached (within 0.20 m) | +100, episode ends |
| Collision (LiDAR within 0.15 m) | −25, episode ends |
| Per step | −0.1 |
| Progress toward goal | +5.0 × distance gained |
| Heading | −0.05 × \|goal angle\| |
| Wall warning (below 0.30 m) | ramps to about −1 |

A 60-step stall with no progress of 0.05 m ends the episode as a truncation, not a terminal. Each action lasts 0.5 s.

---

## Results

### Greedy evaluation

| Run | Model | Episodes | Success | Collisions | Timeouts | Success % | Start pose |
|---|---|---|---|---|---|---|---|
| R1 | Dueling DDQN | 30 | 30 | 0 | 0 | 100.0 | (0, 0, 0) |
| R2 | DDQN, fixed reward | 30 | 30 | 0 | 0 | 100.0 | (0, 0, 0) |
| R3 | DDQN, fixed reward | 30 | 28 | 2 | 0 | 93.3 | (0.2, 0.2, 0) |
| R4 | DDQN, new reward | 20 | 7 | 13 | 0 | 35.0 | Not stated |
| R5 | DQN (not stated) | 20 | 20 | 0 | 0 | 100.0 | Not stated |
| R6 | DQN (not stated) | 20 | 20 | 0 | 0 | 100.0 | Not stated |
| R7 | Rainbow, final | 50 | 50 | 0 | 0 | 100.0 | (0.2, 0.2, 0) |

R1–R6 are read from the evaluation graphs in [`Graphs/`](Graphs). R7 is from the terminal `SUMMARY` on 10 Nov 2026: average reward 107.05, average steps 47.84, plot in [`Graphs/rainbow_0.5s_eval.png`](Graphs/rainbow_0.5s_eval.png).

All runs are greedy and use one seed. Most use a fixed start, so the episodes are close to repeats. A 100% result shows the policy works from that start, not that it generalises.

### Training

| Run | Episodes | Goal | Collision | No progress | Goal % | Source |
|---|---|---|---|---|---|---|
| E1 DDQN, fixed reward | 300 | 134 | – | – | 44.7 | Log (training or evaluation not stated) |
| E3 Dueling DDQN | 300 | 170 | – | – | 56.7 | Log and `Dueling_dqn_train_300ep.png` |
| E2 DDQN, new reward | 200 | not read | not read | not read | not read | `new_reward_200ep_pen_DDQN_22state_train.png` |
| E4 Rainbow DQN | 300 | ≈225 | ≈65 | ≈10 | ≈75 | `rainbow_0.5s_train.png`, read from bars (approximate) |

The Rainbow training figure includes exploration and the first 100 scripted episodes, so it is not the greedy success rate.

### Rainbow evaluation

| Setting | Value |
|---|---|
| Checkpoint | `models/rainbow_dqn_52state_7actions.pth` (final, 27,470 learning steps) |
| Start pose | Held-out (0.2, 0.2, 0) |
| Goal | (2, −2) |
| Episodes | 50, greedy |
| Result | 50 goals, 0 collisions, 0 timeouts |
| Interpretation | Every episode reached the goal in about 47.8 steps. The straight-line minimum is about 38 forward steps. |
| Comparability | Not comparable with the (0, 0, 0) runs. The start pose, environment and reward have all changed. |

Best-by-eval checkpoint (`rainbow_dqn_52state_7actions_best.pth`, 8,704 learning steps): not yet evaluated.

---

## Setup and running

These commands are from [`commands.txt`](commands.txt). They assume the workspace is at `~/ros2_ws`.

**1. Start Gazebo (terminal 1)**

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

**2. Build the package**

```bash
cd ~/ros2_ws
colcon build --packages-select mini_dqn_nav --symlink-install
source ~/ros2_ws/install/setup.bash
```

**3. Train Rainbow (terminal 2)**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/pytorch-env/bin/activate
export TURTLEBOT3_MODEL=burger
export DRLNAV_BASE_PATH="$HOME/ros2_ws"
python -m mini_dqn_nav.train_rainbow
```

**4. Evaluate Rainbow (terminal 2)**

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/pytorch-env/bin/activate
export TURTLEBOT3_MODEL=burger
cd ~/ros2_ws/src/mini_dqn_nav
python -m mini_dqn_nav.rainbow_evaluate
```

The evaluation prints a `SUMMARY` block and then opens a plot window. On a machine without a display, the plot step fails after the summary has printed.

The archived DQN, DDQN and Dueling scripts are in [`inital_codes/`](inital_codes) as `*_DDDQN.py`. They no longer run as modules.

---

## Repository layout

```
.
├── CLAUDE.md                     working rules for this project
├── commands.txt                  run commands
├── Graphs/                       training and evaluation plots
├── inital_codes/                 archived DQN, DDQN and Dueling scripts
├── mini_dqn_nav/
│   ├── environment.py            ROS 2 environment: LiDAR, odometry, actions, reward
│   ├── agent_rainbow.py          Rainbow DQN agent
│   ├── train_rainbow.py          Rainbow training loop
│   ├── rainbow_evaluate.py       Rainbow greedy evaluation
│   ├── logs.md                   experiment log (tables)
│   ├── old_logs.md               original experiment log (prose)
│   └── models/                   checkpoints
├── package.xml, setup.py         ROS 2 package
└── test/                         ament lint tests
```

The `test/` folder only holds ament lint checks. There are no unit tests for the algorithms yet.

---

## Known discrepancies

These are recorded in [`mini_dqn_nav/logs.md`](mini_dqn_nav/logs.md), section 8.

| # | Conflict | Action |
|---|---|---|
| D1 | The log says the network is 256 wide. The live Dueling network uses 128. | Confirm which network each checkpoint used |
| D2 | The log says 120 episodes for the new-reward run. The graph shows 200. | Check the training config |
| D3 | It is not stated whether the E1 and E3 goal counts are training or evaluation. | Check each source |
| D4 | The Rainbow change log says "untested in Gazebo". Training and evaluation have both run. | Superseded |
| D5 | Training assumes the robot starts at (0, 0) after each reset. Not verified. | Check the spawn pose in the launch file |

---

## Roadmap

1. Evaluate the best-by-eval Rainbow checkpoint.
2. Evaluate Rainbow from the training start (0, 0, 0) for comparison with the DQN runs.
3. Verify the spawn pose (D5).
4. Add offline tests for the C51 projection and n-step aggregation.
5. Resolve D1–D3.
6. DDPG: continuous velocity actions, actor and critic networks.

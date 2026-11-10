
# Deep Reinforcement Learning for TurtleBot3 Autonomous Navigation

### Learning to Navigate. Learning to Avoid. Building Toward Autonomous Robotics. (A Research Project Pathway)

A Deep Q-Network (DQN)-based autonomous navigation system for TurtleBot3, built using PyTorch, ROS 2, and Gazebo. The project explores how a mobile robot can learn goal-directed navigation and obstacle avoidance through interaction with a simulated environment and reward-driven learning.

**Current stage:** Intermediate working implementation | Simulation-based experimentation | Research-oriented development

---

## Overview

Ever wondered how a robot can learn to navigate toward a destination without being explicitly programmed for every possible situation?

This project explores that question using Deep Reinforcement Learning (DRL).

Instead of relying entirely on hand-crafted navigation rules, a DQN agent learns to associate observations of its environment with actions that maximize cumulative reward. The robot receives LiDAR-based observations, considers its relative position to the goal, chooses a movement action, and learns from the consequences.

The objective is to develop a foundation for autonomous navigation and obstacle avoidance that can be systematically evaluated, improved, and eventually extended into more advanced reinforcement learning research.

This is an independently developed intermediate implementation informed by an existing open-source DQN navigation project. The reference provided a foundation for understanding the theoretical structure and working infrastructure; I then implemented and integrated my own version, debugged the learning pipeline, and developed a roadmap for further experimentation.

## What the Robot Learns

The agent interacts with its environment through a reinforcement learning loop:

1. Observe the environment using LiDAR and goal-relative information.
2. Select a movement action using an epsilon-greedy policy.
3. Execute the action through ROS 2 in Gazebo.
4. Receive a reward based on navigation progress and environment events.
5. Store the transition in replay memory.
6. Update the policy network using sampled experiences.
7. Repeat and gradually refine the navigation policy.

The central idea is simple: learn a useful navigation policy through experience rather than hard-coding every decision.

<img width="1656" height="1026" alt="image" src="https://github.com/user-attachments/assets/520e12a2-90d9-4dd8-8e3f-09cfa8858e72" />

## System Architecture

```text
                 Gazebo Simulation
                        |
                        v
                  TurtleBot3
                        |
             +----------+----------+
             |                     |
             v                     v
          LiDAR                  Odometry
             |                     |
             +----------+----------+
                        |
                        v
                State Representation
          [50 LiDAR readings + 2 goal-relative/
                 odometry-related features]
                        |
                        v
              Dueling Double DQN Network
                        |
                        v
               Epsilon-Greedy Action
                        |
                        v
                  ROS 2 /cmd_vel
                        |
                        v
                Robot Movement
                        |
                        v
                 Reward & Next State
                        |
                        v
                  Replay Memory
                        |
                        v
              DQN Optimization Step
                        |
                        v
                 Updated Policy
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Reinforcement learning | Double DQN and Dueling Double DQN |
| Robotics middleware | ROS 2 Humble |
| Simulation | Gazebo |
| Robot platform | TurtleBot3 Burger |
| Obstacle sensing | LiDAR |
| Optimization | Gradient descent with Adam |
| Experience learning | Replay memory and target network; PER planned next |
| Visualization | Matplotlib |

## State, Actions, and Rewards

### 1. State Representation

The current agent uses a **52-dimensional observation**:

- **50 LiDAR readings** representing nearby obstacle distances.
- **2 goal-relative/odometry features** describing the robot's relationship to the target.

This representation provides obstacle and goal information without requiring raw camera images or a large visual perception model.

### 2. Discrete Action Space

The agent chooses among **seven discrete movement actions**. Each action maps to a configured linear and angular velocity command sent to the robot through ROS 2.

| Action space | Description |
|---|---|
| 7 discrete actions | Different configured movement and turning commands for navigation |

The exact velocity assigned to each action is defined in the environment implementation.

### 3. Reward Design

The reward function encourages progress toward the goal while penalizing undesirable outcomes.

The current implementation includes:

- Positive reward for reaching the goal.
- Negative reward for collisions.
- Step penalties to discourage inefficient behavior.
- Progress-based reward encouraging movement toward the target.

The reward structure is an important part of the experiment: changing it can influence whether the agent learns efficient navigation, gets stuck, or develops undesirable movement patterns.

## The DQN Implementation

The learning agent uses a neural network to estimate action values, or Q-values.

<img width="1292" height="910" alt="Screenshot from 2026-10-21 05-04-30" src="https://github.com/user-attachments/assets/e4370aee-802d-4d37-b116-7712978f90fc" />

The current improved agent uses a **Dueling Double DQN** setup:

```text
Input: 52 observation features
          |
          v
     Shared feature layers
          |
          v
     +----------+-----------+
     |                      |
     v                      v
  Value stream        Advantage stream
     |                      |
     +----------+-----------+
                |
                v
   Combine value and advantages
                |
                v
      Output: 7 action values
```

The implementation builds on core DQN components and adds two improvements:

- **Policy network:** Estimates the Q-value of each available action.
- **Target network:** Provides more stable targets during learning.
- **Double DQN target calculation:** The policy network selects the best next action, while the target network evaluates that action, helping reduce Q-value overestimation.
- **Dueling architecture:** Separately estimates state value and the relative advantages of available actions before combining them into action values.
- **Experience replay:** Stores transitions and samples mini-batches to reduce correlation between consecutive experiences.
- **Epsilon-greedy exploration:** Balances exploration with selecting the highest-valued action.
- **Smooth L1 loss, gradient clipping, and Adam:** Support policy-network optimization.

The goal is not merely to run a neural network, but to understand and integrate the complete reinforcement learning pipeline.

## Training and Evaluation

Training and evaluation are treated as separate stages.

### Training

During training, the agent explores using an epsilon-greedy policy, stores experiences in replay memory, and updates the policy network.

Training visualizations track:

- Episode reward and moving-average reward.
- Mean DQN loss.
- Exploration rate (epsilon).
- Steps per episode.

These metrics help diagnose learning behavior, but none should be interpreted in isolation.

#### Training Results

| Algorithm | Training episodes | Goals reached | Training success rate |
|---|---:|---:|---:|
| Double DQN | 300 | 134 | 44.7% |
| Dueling Double DQN | 300 | 170 | 56.7% |

In these recorded runs, Dueling Double DQN reached the goal in 36 more training episodes, a 12-percentage-point increase in observed training success rate. This is a comparison of the recorded runs, not a multi-seed statistical evaluation.

#### Double DQN Training Graphs

<!-- Replace the placeholders below with the actual GitHub image URLs for your Double DQN graphs. -->

**Episode Reward and Moving Average**

![Double DQN Episode Reward]<img width="1300" height="900" alt="image" src="https://github.com/user-attachments/assets/3a412340-188d-4964-bced-8e6253af2031" />





#### Dueling Double DQN Training Graphs

<img width="1300" height="900" alt="image" src="https://github.com/user-attachments/assets/c84430a2-878f-4a2f-a8dd-e9b331f8aeb0" />




### Evaluation

During evaluation, the trained policy selects greedy actions without intentional epsilon-based exploration.**Also different initial position (0.2,0.2) were used for increasing challenge during navigation.**

#### Recorded Evaluation Results

| Algorithm | Successful episodes | Evaluation success rate |
|---|---:|---:|
| Double DQN | 28/30 | 93.3% |
| Dueling Double DQN | 30/30 | 100% |

**Double DQN**
<img width="1400" height="900" alt="image" src="https://github.com/user-attachments/assets/6cba30ff-ce58-4541-aa10-4411bdc88b09" />

**Dueling Double DQN**
<img width="1400" height="900" alt="image" src="https://github.com/user-attachments/assets/bdbd8469-ede3-4862-b095-6180700212b2" />

The recorded 30/30 Dueling Double DQN result was obtained at the tested start position `(0.2, 0.2)`. The robot's initial position was also changed in an additional test to make evaluation more challenging; no separate numerical result is reported here for that changed-position test.

These are preliminary simulation results from a limited number of episodes. They do not establish robust generalization across all starting positions or environments.


The evaluation workflow tracks:

- Goal-reaching success rate.
- Collision count.
- Timeout count.
- Average episode reward.
- Average steps per episode.

## Running the Project

The implementation uses a ROS 2 workspace and a Python virtual environment containing PyTorch.

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo and the required TurtleBot3 simulation packages
- Python virtual environment with PyTorch and the project dependencies
- A configured ROS 2 workspace containing the project package


## My Contributions

This project represents my hands-on work in translating reinforcement learning theory into a functioning robotics simulation.

### Implementation and Integration

- Developed and integrated a DQN-based navigation implementation using Python and PyTorch.
- Updated the observation representation to 52 features: 50 LiDAR readings and 2 goal-relative/odometry features.
- Worked with a seven-action discrete movement space.
- Integrated the learning agent with ROS 2 and TurtleBot3 simulation in Gazebo.
- Implemented and benchmarked Double DQN and Dueling Double DQN.
- Integrated training, model checkpointing, and a separate greedy evaluation workflow.

### Parameter Experimentation and Environment Design

A key part of this project was exploring how the environment and learning configuration influence navigation behavior.

- **Action-space design:** Worked with the discrete action space, including forward movement, curved turns, and sharper turns, to give the agent different ways to maneuver.
- **Reward and penalty tuning:** Worked with goal-reaching rewards, collision penalties, step costs, and progress-based rewards to shape the navigation objective.
- **Goal-position changes:** Changed the target coordinates to explore navigation toward different locations and make the task more challenging.
- **Difficulty and behavior:** Used the simulation to investigate how changes in navigation goals and movement choices affect the agent's behavior.

These experiments helped me understand that a DQN agent's performance depends not only on its neural network, but also on the action space, reward function, and environment in which it learns.

### Debugging and Experimentation

Building a working RL system required more than implementing the algorithm.

During development, I encountered and investigated issues including:

- Training crashes caused by an uninitialized loss variable when an episode ended before a loss was recorded.
- Incorrect loss-logging order, which caused the training plot to display misleading zero-loss values.
- Environment and software integration issues involving PyTorch, ROS 2, and Gazebo.
- The need to distinguish improving reward trends from verified navigation success.

These experiences reinforced an important lesson: a working training loop is not sufficient evidence of a successful learning system. Evaluation, debugging, and reproducible measurements are essential.

### Build

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

colcon build --packages-select mini_dqn_nav

source ~/ros2_ws/install/setup.bash
```

### Configure the Environment

```bash
source ~/pytorch-env/bin/activate

export TURTLEBOT3_MODEL=burger
export DRLNAV_BASE_PATH="$HOME/ros2_ws"
```

Start the configured Gazebo world and ensure the simulation is running before launching training or evaluation.

### Train the Agent

```bash
python -m mini_dqn_nav.train
```

Training saves model checkpoints according to the configured checkpoint path.

### Evaluate the Trained Policy

Run the evaluation entry point with the existing trained checkpoint configured in the evaluation script. The evaluation should use the same network architecture and state/action dimensions as training.

The exact evaluation command depends on the filename and Python entry point in your implementation.

## Research and Future Development

This implementation is a starting point, not the final objective.

My longer-term goal is to investigate how reinforcement learning methods can improve autonomous robot navigation in terms of reliability, efficiency, and generalization.

### Planned Directions

**1. Reliable Baseline Evaluation**

Establish reproducible evaluation across multiple episodes and report success rate, collision rate, episode reward, and navigation efficiency.

**2. Reward-Function Investigation**

Study how progress rewards, collision penalties, and time penalties affect learned behavior. Investigate reward designs that discourage oscillation and unnecessary movement.

**3. Hyperparameter Experiments**

Evaluate the effects of learning rate, discount factor, replay batch size, exploration decay, and target-network update frequency.

**4. DQN Improvements**

Double DQN and the Dueling network architecture have been implemented and tested in the current experimental workflow. Next, implement Prioritized Experience Replay (PER), then explore Rainbow DQN, which combines several DQN improvements.

**5. Generalization**

Continue testing at changed initial robot positions and different obstacle configurations. Compare algorithms under matched conditions and report episode counts and outcomes for each scenario.

**6. Comparative Experiments**

Compare the baseline DQN against improved variants using the same environment, training budget, and evaluation protocol.

**7. Advanced Navigation Research**

Investigate more advanced DRL approaches, potentially including PPO and other algorithms, depending on the limitations observed in the baseline.

These are planned research directions, not features or results claimed to have already been implemented.

## Why This Project Matters to Me

This project has been an opportunity to move beyond studying machine learning concepts in isolation and work through the practical challenges of connecting a learning algorithm to a simulated robot.

It has helped me explore the intersection of reinforcement learning, neural networks, robotics middleware, simulation, and experimental evaluation.

I intend to continue improving the implementation, documenting the experiments, and using the project as a foundation for deeper work in deep reinforcement learning and autonomous robotics.

## Acknowledgments

Thanks to the authors of the open-source [DQN Navigation and Obstacle Avoidance](https://github.com/suhasnagaraj99/DQN-Navigation-and-Obstacle-Avoidance) project for providing a useful reference for learning and implementation.

## Connect

I'm interested in reinforcement learning, autonomous robotics, intelligent control, and research-oriented engineering projects.

Feedback, suggestions, and discussions about improving the navigation policy are welcome.

---

**Topics:** Deep Reinforcement Learning · DQN · Double DQN · Dueling DQN · TurtleBot3 · ROS 2 · Gazebo · PyTorch · Obstacle Avoidance · Robotics Research

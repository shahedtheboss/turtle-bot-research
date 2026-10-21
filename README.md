# Deep Reinforcement Learning for TurtleBot3 Autonomous Navigation

### Learning to Navigate. Learning to Avoid. Building Toward Autonomous Robotics.

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
             [5 LiDAR sectors, distance,
                 relative goal angle]
                        |
                        v
                 DQN Policy Network
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
| Reinforcement learning | Deep Q-Network (DQN) |
| Robotics middleware | ROS 2 Humble |
| Simulation | Gazebo |
| Robot platform | TurtleBot3 Burger |
| Obstacle sensing | LiDAR |
| Optimization | Gradient descent with Adam |
| Experience learning | Replay memory and target network |
| Visualization | Matplotlib |

## State, Actions, and Rewards

### 1. State representation

The agent receives a seven-dimensional state:

- Five LiDAR sectors representing nearby obstacle distances.
- Distance to the target.
- Relative angle between the robot's heading and the target.

This compact representation provides information about immediate obstacles and the robot's goal without requiring raw camera images or a large visual perception model.

### 2. Discrete action space

The agent chooses among five movement actions:

| Action | Behavior |
|---|---|
| 0 | Move forward |
| 1 | Move forward while turning left |
| 2 | Move forward while turning right |
| 3 | Turn left sharply |
| 4 | Turn right sharply |

Each action maps to a linear and angular velocity command sent to the robot through ROS 2.

### 3. Reward design

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


The current network architecture is:

```text
Input: 7 state features
          |
          v
Fully Connected Layer: 128 neurons
          |
          v
        ReLU
          |
          v
Fully Connected Layer: 128 neurons
          |
          v
        ReLU
          |
          v
Output: 5 action values
```

The implementation includes several core DQN components:

- **Policy network:** Estimates the Q-value of each available action.
- **Target network:** Provides more stable targets during learning.
- **Experience replay:** Stores transitions and samples random mini-batches to reduce correlation between consecutive experiences.
- **Epsilon-greedy exploration:** Balances random exploration with choosing the highest-valued action.
- **Bellman target:** Estimates the return from the immediate reward and discounted future value.
- **Smooth L1 loss:** Measures the discrepancy between predicted Q-values and target values.
- **Gradient clipping:** Limits excessively large gradient norms during optimization.
- **Adam optimizer:** Updates the policy network parameters.

The goal is not merely to run a neural network, but to understand and integrate the complete reinforcement learning pipeline.

## My Contributions

This project represents my hands-on work in translating reinforcement learning theory into a functioning robotics simulation.

### Implementation and integration

- Developed an intermediate DQN navigation implementation using Python and PyTorch.
- Implemented the seven-feature state representation and five-action movement space.
- Integrated the learning agent with a ROS 2 environment and TurtleBot3 simulation in Gazebo.
- Worked with reward-based learning, experience replay, target-network updates, and epsilon-greedy exploration.
- Integrated training, model checkpointing, and a separate greedy evaluation workflow.

### Parameter experimentation and environment design

A key part of this project was exploring how the environment and learning configuration influence navigation behavior.

- **Action-space design:** Worked with the discrete action space, including forward movement, curved turns, and sharper turns, to give the agent different ways to maneuver.
- **Reward and penalty tuning:** Worked with goal-reaching rewards, collision penalties, step costs, and progress-based rewards to shape the navigation objective.
- **Goal-position changes:** Changed the target coordinates to explore navigation toward different locations and make the task more challenging.
- **Difficulty and behavior:** Used the simulation to investigate how changes in navigation goals and movement choices affect the agent's behavior.

These experiments helped me understand that a DQN agent's performance depends not only on its neural network, but also on the action space, reward function, and environment in which it learns.

### Debugging and experimentation

Building a working RL system required more than implementing the algorithm.

During development, I encountered and investigated issues including:

- Training crashes caused by an uninitialized loss variable when an episode ended before a loss was recorded.
- Incorrect loss-logging order, which caused the training plot to display misleading zero-loss values.
- Environment and software integration issues involving PyTorch, ROS 2, and Gazebo.
- The need to distinguish improving reward trends from verified navigation success.

These experiences reinforced an important lesson: a working training loop is not sufficient evidence of a successful learning system. Evaluation, debugging, and reproducible measurements are essential.

## Open-Source Acknowledgment

This project was informed by:

[DQN Navigation and Obstacle Avoidance — Suhas Nagaraj](https://github.com/suhasnagaraj99/DQN-Navigation-and-Obstacle-Avoidance)

I used the project as a reference for the theoretical structure and working infrastructure while developing my own intermediate implementation and working through its integration and debugging challenges.

The reference is acknowledged here to distinguish the original work from my implementation and planned extensions.

## Training and Evaluation

Training and evaluation are treated as separate stages.

### Training

During training, the agent explores using an epsilon-greedy policy, stores experiences in replay memory, and updates the policy network.

Training visualizations track:

- Episode reward.
- Moving-average reward.
- Mean DQN loss.
- Exploration rate (epsilon).
- Steps per episode.

These metrics help diagnose learning behavior, but none should be interpreted in isolation.

### Evaluation

During evaluation, the trained policy selects greedy actions without intentional epsilon-based exploration.

The evaluation workflow is designed to measure:

- Goal-reaching success rate.
- Collision count.
- Timeout count.
- Average episode reward.
- Average steps per episode.

The current evaluation is an initial simulation experiment, not a claim of robust or deployment-ready autonomous navigation.

**Experimental results:** Numerical results will be added after validating the trained checkpoint and completing the evaluation runs.

## Running the Project

The implementation uses a ROS 2 workspace and a Python virtual environment containing PyTorch.

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo and the required TurtleBot3 simulation packages
- Python virtual environment with PyTorch and the project dependencies
- A configured ROS 2 workspace containing the project package

### Build

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash

colcon build --packages-select mini_dqn_nav

source ~/ros2_ws/install/setup.bash
```

### Configure the environment

```bash
source ~/pytorch-env/bin/activate

export TURTLEBOT3_MODEL=burger
export DRLNAV_BASE_PATH="$HOME/ros2_ws"
```

Start the configured Gazebo world and ensure the simulation is running before launching training or evaluation.

### Train the agent

```bash
python -m mini_dqn_nav.train
```

Training saves model checkpoints according to the configured checkpoint path.

### Evaluate the trained policy

Run the evaluation entry point with the existing trained checkpoint configured in the evaluation script. The evaluation should use the same network architecture and state/action dimensions as training.

The exact evaluation command depends on the filename and Python entry point in your implementation.

## Research and Future Development

This implementation is a starting point, not the final objective.

My longer-term goal is to investigate how reinforcement learning methods can improve autonomous robot navigation in terms of reliability, efficiency, and generalization.

### Planned directions

**1. Reliable baseline evaluation**

Establish reproducible evaluation across multiple episodes and report success rate, collision rate, episode reward, and navigation efficiency.

**2. Reward-function investigation**

Study how progress rewards, collision penalties, and time penalties affect learned behavior. Investigate reward designs that discourage oscillation and unnecessary movement.

**3. Hyperparameter experiments**

Evaluate the effects of learning rate, discount factor, replay batch size, exploration decay, and target-network update frequency.

**4. DQN improvements**

Explore Double DQN to reduce Q-value overestimation, and investigate prioritized experience replay or dueling network architectures where appropriate.

**5. Generalization**

Test whether a policy learned in one environment can navigate different obstacle configurations and goal locations without relying on a single fixed scenario.

**6. Comparative experiments**

Compare the baseline DQN against improved variants using the same environment, training budget, and evaluation protocol.

**7. Advanced navigation research**

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

**Topics:** Deep Reinforcement Learning · DQN · Autonomous Navigation · TurtleBot3 · ROS 2 · Gazebo · PyTorch · Obstacle Avoidance · Robotics Research

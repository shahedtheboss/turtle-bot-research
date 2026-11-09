# TurtleBot3 DQN Navigation — Project Change Log

**Project:** Deep Reinforcement Learning for TurtleBot3 Autonomous Navigation  
**Platform:** ROS 2 Humble, Gazebo, TurtleBot3 Burger, Python, PyTorch  
**Research goal:** Develop and evaluate an improved DQN-based mobile robot navigation system.

## 1. Initial implementation and first evaluation

I developed an independent ROS 2 navigation implementation using the original DQN navigation project as a reference.

The initial implementation included:

- A custom robot environment communicating through `/scan`, `/odom`, and `/cmd_vel`.
- A seven-dimensional observation consisting of five LiDAR sectors, goal distance, and relative goal angle.
- A fixed navigation goal at `(2.0, -2.0)`.
- Five discrete actions for forward movement and turning.
- A DQN agent with a fully connected neural network containing two hidden layers of 128 neurons each.
- Experience replay, a policy network, a target network, epsilon-greedy exploration, and a training loop.
- Separate training and evaluation scripts.

The initial model was evaluated using greedy action selection, with exploration disabled during evaluation. The evaluation script recorded episode outcomes, rewards, and episode lengths and generated plots.

**Baseline configuration**

- Network: `7 → 128 → 128 → 5`
- Replay buffer capacity: 10,000 transitions
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Discount factor: 0.99
- Loss: Smooth L1 (Huber) loss
- Target network: hard update every two episodes
- Initial epsilon: 1.0
- Minimum epsilon: 0.05
- Epsilon decay: 0.99 per episode

The original five-action model and its checkpoint were retained as the baseline for subsequent comparisons.

 ### Bug = Bot was taking repeated reverse actions and finishing episodes wiht step expiring during evaluation --- Fixed by removing reverse intially

## 2. Environment and reward design

The custom environment uses five LiDAR sectors: front, front-left, front-right, left, and right. The nearest reading within each sector is used to represent nearby obstacles.

The reward and termination configuration includes:

- Goal-reaching reward: `+100`
- Obstacle-proximity termination penalty: `-30` (changed from -100 to fix goals to reach in obstucted paths)
- Per-step penalty: `-0.1`
- Progress reward: `+7 × distance progress` 
## Bug-- repeatedly bumping into corners fixed by changing intial +5 to +7
- Goal tolerance: approximately `0.20 m`
- LiDAR collision-warning threshold: changed from `0.20 m` to approximately `0.08 m`

The collision-warning threshold is a LiDAR-based termination rule; it does not actively prevent the robot from approaching an obstacle. It should not be described as a physical collision detector or a safety controller.

The goal remains fixed for the current experiments. Variable goals have not been included in the present scope.
Train Results:
![alt text](image.png)

Eval:
![alt text](image-1.png)

## 3. Recent DQN upgrades

After establishing the initial baseline, I began modifying the learning algorithm and model configuration.

### 3.1 Neural network architecture

The original network used two hidden layers of 128 neurons each. The network architecture was subsequently modified to increase model capacity.

The revised configuration was intended to use two hidden layers of 256 neurons each:

`7 → 256 → 256 → 7`

The output layer has seven neurons because the action space was expanded to seven discrete actions.

*Note: The exact architecture should be verified against the current `agent.py` before recording this as the final implemented configuration.*

### 3.2 Expanded action space

The action space was expanded from five actions to seven actions to give the robot additional turning flexibility.

The proposed seven-action set includes:

1. Forward movement.
2. Forward-left movement.
3. Forward-right movement.
4. Slow forward-left movement.
5. Slow forward-right movement.
6. Rotation in place to the left.
7. Rotation in place to the right.

The new action set changes the number of Q-values predicted by the network from five to seven. The environment's action mapping and the agent's output layer must remain consistent.

A new checkpoint must be trained for the seven-action configuration; the original five-action checkpoint is not directly compatible with the new output layer.

### 3.3 Batch size

The batch size was increased from 64 to 128 as part of the revised training configuration.

This means each learning update samples 128 transitions from replay memory rather than 64. The change may affect gradient estimates, memory usage, and learning dynamics, so its effect should be measured experimentally.

Results:
Train ![alt text](image-2.png)
Evaluate ![alt text](../eval_v2.png)


### 3.4 Double DQN

The learning target was identified for modification from standard DQN to Double DQN.

Standard DQN selects and evaluates the next action using the target network:

`target_q = rewards + gamma × max(target_net(next_states))`

Double DQN separates action selection and action evaluation:

1. The policy network selects the highest-valued next action.
2. The target network evaluates the selected action.
3. The resulting value is used in the Bellman target.

Conceptually:

`next_actions = argmax(policy_net(next_states))`

`next_q = target_net(next_states).gather(next_actions)`

`target_q = rewards + gamma × next_q × (1 - dones)`

This modification is intended to reduce overestimation of action values. It does not guarantee improved navigation performance.

**Implementation status:** The Double DQN code change should be verified in the current `agent.py` before the experiment is labeled as completed.

## 3.5 Updated learn() function 

## 4. Recommended experimental comparison

The experiments should distinguish the effects of individual changes:

| Experiment | Configuration | Purpose |
|---|---|---|
| E1 | Original five-action DQN | Baseline |
| E2 | Double DQN, five actions | Isolate the algorithm change |
| E3 | Double DQN, seven actions | Evaluate the expanded action space |
| E4 | Seven-action Double DQN with revised network and batch size | Evaluate the revised training configuration |
| E5 | Seven-action Double DQN with a safety filter | Investigate the proposed safety contribution |

The original model should remain available for comparison. Training conditions, environment, goal, and evaluation protocol should be held constant wherever possible.

## 8. Evaluation metrics

The evaluation should report:

- Goal-reaching success rate.
- Collision rate and LiDAR-warning termination rate separately.
- Average episode reward.
- Path length.
- Time to goal.
- Minimum obstacle clearance.
- Training stability across multiple random seeds.
- Safety-filter intervention rate, if a filter is implemented.

A high success rate on one fixed environment does not by itself demonstrate generalization or research novelty.

## 8.Wider LiDAR scan 
Use wider LiDAR scans around 20 distance scans and 2 goal relative information(odometry)
    NOTE:
        - This calculation assumes we want 20 equal sectors around
        - a complete circle. It does not verify that the actual
        - LiDAR message covers the full 360 degrees.
        sector_width = (2.0 * math.pi) / num_sectors


        - Create 20 separate empty lists.
        -
        - Example structure:
        -     [[], [], [], ..., []]
        -
        - Each inner list will collect the raw LiDAR measurements
        - that fall inside its corresponding angular sector.
        -
        - We collect readings first because a sector may contain
        - multiple measurements. After processing the entire scan,
        - we will select the minimum distance from each sector.
        -
        - The list comprehension runs once for every sector.

    Results
    Train
    ![alt text](image-3.png)

    Evaluate
    ![alt text](../train_DDQN_22scans.png)

## 9. calculate_reward() — there's a priority issue
    Look at your current order:
    if min(obstacle_distances) <= self.collision_distance:    return -30.0, True, "collision_warning"if current_goal_distance <= self.goal_tolerance:    return 100.0, True, "goal_reached"



    The collision warning is checked first. If the robot is within the goal tolerance while any LiDAR sector is at or below the collision threshold, the episode ends as collision_warning, even if the goal has been reached.

    I'd change the ordering so that the goal condition is checked first, provided your intended task considers reaching the goal successful regardless of that proximity warning.
    # Check whether the robot reached the goal first.if current_goal_distance <= self.goal_tolerance:    return 100.0, True, "goal_reached"# Otherwise, check for a LiDAR-based collision warning.if min(obstacle_distances) <= self.collision_distance:    return -30.0, True, "collision_warning"



    This fixes the priority issue, but it won't necessarily fix the 0% success rate if the robot isn't actually reaching the goal.

    if distance postive -> 7 times reward
    negative ->3 times penalty

    ALL IN 120 EPISODES AND 100 STEPS

    Results
    Train:
    ![alt text](../new_reward_pen_DDQN_22state_train.png)
    Evaluate:
    ![alt text](../new_reward_pen_DDQN_22state_eval.png)

## 10. Inrease LiDAR and action state
    -LiDAR sttae 50
    -Input state 52
    -action 11

## 10. Change epsilon
    -Change epsilon values from 
    -Decay per steop instead of per episode

## Crucial Fixes and Changes
environment.py
    Added normalize(), which scales LiDAR by 3.5, goal distance by 6.0 and angle by π.
    step() now returns self.normalize(next_state) instead of the raw state.
    Added a set_start() teleport helper using SetEntityState.
    Rewrote the non-terminal reward: -0.1 step cost, symmetric 5.0 * progress (replacing the 7×/3× asymmetric version), -0.05 * |angle| heading term, and a proximity penalty below 0.30 m.
    Raised the collision terminal penalty from -30 to -50.
    Reduced the action set from 11 actions to 7 (removed the reverse and extra slow actions).
train.py
    DQNAgent(state_size=52, action_size=11) changed to action_size=7.
    Moved agent.decay_epsilon() into the step loop, called only when loss is not None.
    Commented out the old per-episode agent.decay_epsilon().
    Increased episodes from 100 to about 300 (inferred from your plot, which shows 300 episodes).
    agent.py
    epsilon_decay changed from 0.99 to 0.9995.
    Default action_size in DQNAgent.__init__ changed from 11 to 7.
    Added a 2000-transition warmup in the active learn() (inferred: epsilon stays at 1.0 until about episode 45 in your plot).
evaluate.py
    Added the constants EVAL_STARTS, TRAIN_SPAWN_X/Y and GOAL_X/Y.
    Added the align_goal_to_start() function.
    DQNAgent(state_size=22, action_size=7) changed to state_size=52.
    Added a reset, teleport and goal alignment before the loop so episode 1 starts from an eval pose.
    Added state = env.normalize(state) after the if state is None: block (first version was misplaced inside the block).
    Replaced the hard-coded (-0.6, 0.9, 0.0) teleport with EVAL_STARTS[episode % len(EVAL_STARTS)], and added align_goal_to_start(...) after wait_for_state.
    Changed EVAL_STARTS to [(0.0, 0.0, 0.0)] to test from the training start.
Build and run
    Rebuilt the package with colcon build --packages-select mini_dqn_nav --symlink-install after the stale-build error (inferred from the fact that eval ran afterwards).
Not confirmed in the files you sent

    These were still missing in the last versions I saw, so check each one:

    state[20] changed to state[50] in step(). Your plot suggests this was fixed afterwards, but I haven’t seen the file.
    state[21] changed to state[51] in the reward logging line.
    env.normalize(env.get_state()) for the first state of each episode in train.py.
    Storing terminated instead of done in agent.remember(...).
    Step-based target updates (learn_steps, update every 500 steps), and removing the episode-based update.
    Lower learning rate (0.0003) and best-checkpoint saving.
Results: train ![alt text](image-4.png)
         test ![alt text](../Graphs/eval_300ep_DDQN_fixed_reward.png) ![alt text](../Graphs/eval_300ep_DDQN_fixed_reward(0.2,0.2).png)

Metric	    
Episodes	    300
Goals reached	134
Success rate	44.7%
    
## Implemented Dueling Double DQN
 -128 neuron 2 layer Linear
 -128 to 64 neuron value stream
 -64 to 1 advantage stream
Results- train ![alt text](../Graphs/Dueling_dqn_train_300ep.png)

Metric
Episodes		300
Goals reached	170
Success rate	56.7%

## 4. Changes that remain experimental or planned

The following are not assumed to be implemented merely because they were discussed:

    - AdamW or RMSprop optimizer comparisons.
    - A learning-rate change from `0.001` to `0.0003`.
    - Prioritized Experience Replay.
    - Dueling DQN.
    - A separate LiDAR-based safety filter.
    - Variable goal positions.
    - Evaluation across unseen obstacle layouts.

    These should be introduced as separate experiments rather than combined into the same run.

        Phase 3 — Improve navigation generalization.
    - Begin with relatively simple obstacle configurations and progressively increase difficulty.
    - Randomize starting poses after the agent demonstrates basic competence.
    - Evaluate on fixed, held-out starting positions so you can tell whether the robot has learned general navigation rather than memorized one route.
    Phase 4 — Compare architectural and return improvements.
    - Test Dueling DQN.
    - Then test N-step returns.
    - Change one major component at a time so you know which improvement actually helps.


## 10.Phase 2 — Improve the learning signal.
    -(Done) Normalized the 20 LiDAR distances, goal distance and relative angle consistently.
    - Added PER, including importance-sampling weights and priority updates.
## 7. Research direction

The project is evolving from a basic DQN navigation implementation toward a controlled investigation of Double DQN, action-space design, and potentially safety-aware navigation.

The research contribution must be established through a literature review and comparative experiments. The existing DQN framework and the initial navigation implementation should be properly credited; the contribution should be described in terms of the modifications, methodology, experiments, and findings actually produced.

## 8. Immediate next steps

1. Verify the current network architecture, action mapping, batch size, and Double DQN target calculation in the source files.
2. Preserve the original baseline checkpoint.
3. Train a fresh model for the revised configuration.
4. Evaluate it under the same conditions as the baseline.
5. Save the configuration and results for every experiment.
6. Add a safety filter only after the revised baseline is reproducible.




 ## CHANGE LOG: Rainbow TurtleBot3 navigation (untested in Gazebo)

agent_rainbow.py
 1. Timeout counted as a terminal state, which zeroed the value bootstrap.
    -> Added a `truncated` flag. The n-step queue is flushed without cutting bootstrapping.
 2. `policy_net.reset_noise()` inside `no_grad` (likely autograd in-place error; unverified).
    -> Removed.
 3. C51 support too wide (-100..200), slow target updates, long warmup.
    -> v_min/v_max = -75/125, target update 250, warmup 1000.

environment.py
 4. Goal computed from odom, which a teleport doesn't update (probable cause of the 0%
    result from shifted starts; unverified).
    -> World-frame pose tracking and `begin_episode()`. `set_start()` calls it.
 5. Log printed state[21] as the goal angle (wrong index, it's state[51]); INFO spam each step.
    -> Fixed and removed.
 6. 1.0 s open-loop steps (0.15 m per decision, little warning before a wall).
    -> 0.5 s steps.
 7. No cut-off for stalled episodes.
    -> `no_progress` after 60 steps without 0.05 m of progress (treated as truncation).
 8. Collision penalty -50, stale 22/20-feature comments.
    -> Collision -25, comments cleaned up.
 9. Sensor-wait loop exited after one spin.
    -> Fixed 0.1 s settle (optional; your environment_1.py still has the old loop).
10. Hard-coded 52 and 50.
    -> `NUM_SECTORS + 2` suggested (optional).

Your environment_1.py (found by the check)
11. `warning_penalty` set but `warning_distance` used -> AttributeError. Renamed.
12. `min_progress` never defined -> next AttributeError. Added `self.min_progress = 0.05`.
13. `set_start` default entity 'burger'. Should be 'turtlebot3_burger'.

train_rainbow.py
14. Timeout / no_progress passed as terminal -> now `terminal` and `truncated` are separate.
15. 1 gradient update per step -> 2. Per-episode log now shows outcome, goal distance,
    min LiDAR, action counts. Greedy eval every 25 episodes with best checkpoint saved.
    `"\\n"` print bug fixed. Optional goal curriculum (off by default).
16. Warmup wandered near the start (untrained NoisyNet acts near-deterministically; 6 of
    7 actions turn).
    -> Guided exploration: obstacle-aware goal seeker, probability 0.8 -> 0.1 over 100 episodes.

rainbow_evaluate.py (from evaluate.py)
17. Rainbow agent import, removed the stray train import, new model path, MAX_STEPS 250.
    `align_goal_to_start` now keeps the world-frame goal and calls `begin_episode`.
    C51 support loaded from the checkpoint. `no_progress` counted as Timeout.

Checked and NOT bugs (GPT claims): state-size mismatch (comments only), "100-step limit"
(MAX_STEPS is 250).

Still unverified: items 2 and 4 (my reasoning, not run), and whether the robot spawns
at the world origin (START_POSE).


Your agent is a Rainbow DQN: a base DQN with six add-ons. The parameter values below are the ones now in `agent_rainbow.py` and `train_rainbow.py`.

## 1. Base DQN
A neural network estimates the value of each action. The agent trains on random minibatches from a replay buffer and bootstraps from a slowly updated target network.
- **Network:** 52 inputs → 256 → 256 (ReLU) shared layers, then two heads (see Dueling).
- **Optimiser:** Adam with `lr=1e-4`, gradient clip 10.0, `gamma=0.99`.
- **Batch and buffer:** `batch_size=128` and `memory_size=50,000`.
- **Warmup:** `warmup_transitions=1000`, so no learning happens before 1000 stored transitions.

## 2. Double DQN
Plain DQN picks and evaluates the next action with the same network, which overestimates values. Here the online network picks the next action and the target network evaluates it.
- **Key parameter:** `target_update_every=250`, a hard copy every 250 learning steps. With 2 updates per environment step, that is about every 125 steps.
- **Effect:** smaller values make the target track faster but less stably. Larger values are steadier but slower.

## 3. Dueling architecture
The network splits into a state-value stream V(s) and an advantage stream A(s,a), combined as `V + A − mean(A)`. This helps when many actions have similar value, which is common in navigation.
- **Sizes:** value head 256→128→51 and advantage head 256→128→(7×51).

## 4. NoisyNet (exploration)
The head layers use learnable Gaussian noise on their weights instead of epsilon-greedy. The network learns how much to explore per state.
- **Key parameter:** `sigma0=0.5`, the initial noise scale. Noise is resampled every action and every learning step.
- **Training vs evaluation:** training uses noisy weights, while `select_greedy_action` switches to the mean weights.
- **Weakness here:** an untrained NoisyNet acts almost deterministically, which caused the wandering during warmup. That is why guided exploration (section 8) was added.

## 5. Distributional RL (C51)
The network predicts a probability distribution over returns instead of one number. Q is the expectation of that distribution.
- **Key parameters:** `num_atoms=51`, `v_min=-75`, `v_max=125`. The atoms are 200/50 = 4.0 apart.
- **Support must cover the real return range.** Returns outside it are clipped. If it is too wide, small value differences get blurred, because rewards per step are much smaller than 4.0.
- **Loss:** cross-entropy between the predicted distribution and the projected target.

## 6. Prioritized Experience Replay (PER)
Transitions with larger errors are sampled more often. The importance-sampling weights correct the bias this introduces.
- **Key parameters:** `alpha=0.6` controls how strongly priority matters. `beta_start=0.4` controls the bias correction. `beta_frames=1,000,000` is how long beta takes to anneal to 1.
- **Caveat:** with only a few tens of thousands of updates, beta barely rises above 0.4, so the bias correction stays weak. For a short run, `beta_frames` of about 30,000–50,000 matches your budget better. I haven't changed this.
- **Priority:** the per-sample cross-entropy loss, with `eps=1e-6`. New transitions get the current maximum priority.

## 7. n-step returns
Rewards are summed over `n_step=3` steps before bootstrapping, with discount `gamma^3` on the bootstrap. This propagates the goal reward backward faster. Larger n is faster but more biased with off-policy data.
- **Truncation:** episodes cut off by a timeout or no-progress still bootstrap, while goal and collision do not.

## 8. Additions that are not part of Rainbow
- **Guided exploration:** `GUIDED_START=0.8` decaying to `GUIDED_END=0.1` over `GUIDED_EPISODES=100`, with `GUIDED_NOISE=0.15`. A scripted obstacle-aware goal seeker fills the buffer with goal-directed data early.
- **Two updates per step:** `UPDATES_PER_STEP=2`.
- **Greedy evaluation:** every `EVAL_EVERY=25` episodes, `EVAL_EPISODES=3`, with the best checkpoint saved.
- **No-progress cut-off:** 60 steps without 0.05 m of progress ends the episode as a truncation.

## 9. Reward design (environment)
- **Goal:** +100 within 0.20 m of the goal.
- **Collision:** −25 at 0.15 m or closer.
- **Per step:** −0.1, plus 5.0 × progress toward the goal, minus 0.05 × |goal angle|.
- **Wall warning:** a penalty of up to about −1 when closer than 0.30 m to an obstacle.
- **Step duration:** 0.5 s per action, about 0.075 m for forward.

## What to tune first
1. **Reward and `v_min`/`v_max` together:** the support must still cover the return range.
2. **Guided exploration** and `UPDATES_PER_STEP`: these have the biggest effect on speed.
3. **`target_update_every` and `beta_frames`.**
4. **`n_step`, `alpha`, `sigma0`:** leave these at their defaults until the first three work.

## Rainbow DQN: greedy evaluation (pending results)

**Archive:** the Dueling Double DQN `agent.py`, `train.py` and `evaluate.py` were moved to `inital_codes/agent_DDDQN.py`, `inital_codes/train_DDDQN.py` and `inital_codes/evaluate_DDDQN.py` (commit `071345d`). The old `python -m mini_dqn_nav.train` and `.evaluate` commands no longer run.

**Change:** `rainbow_evaluate.py` `NUM_EPISODES` changed from 30 to 50, to narrow the confidence interval. A 30-episode run at 75% success has a Wilson 95% interval of about 57–87%.

**Setup:**
- Checkpoint: `rainbow_dqn_52state_7actions.pth` (final), `learning_steps = 27,470`
- Start pose: held-out (0.2, 0.2, 0), goal (2, −2), 0.5 s per action, `MAX_STEPS = 250`
- Greedy (mean-weight) actions, one seed, one start pose

**Not comparable** with the (0, 0, 0) DQN/DDQN results, because the start pose, environment and reward differ.

**Results:** to be filled in after the run. The best-by-eval checkpoint (`learning_steps = 8,704`) will be reported separately.
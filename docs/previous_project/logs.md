# TurtleBot3 DQN Navigation: Experiment Log

**Project:** Deep RL for TurtleBot3 autonomous navigation
**Platform:** ROS 2 Humble, Gazebo, TurtleBot3 Burger, Python, PyTorch
**Goal:** Compare DQN variants and Rainbow DQN on a fixed-goal navigation task, then move to DDPG.

The previous prose version of this log is in git history (commit `719670d`). Numbers below come from the logged text and from the graphs in `Graphs/`. Each row says where its value came from.

---

## 1. Experiment register

| ID | Algorithm | Actions | State | Episodes | Step time | Status | Notes |
|---|---|---|---|---|---|---|---|
| E0 | DQN (baseline) | 5 | 7 (5 LiDAR + goal) | 100 | 1.0 s | Done, superseded | Bug fixed: repeated reverse actions |
| E1 | DDQN, fixed reward | 7 | 22 (20 LiDAR + goal) | 300 | 1.0 s | Done | Log: 134/300 goals |
| E2 | DDQN, new reward, penalty | 7 | 22 | 200 (log says 120) | 1.0 s | Done | See discrepancy D2 |
| E3 | Dueling DDQN | 7 | 22 | 300 | 1.0 s | Done | Log: 170/300 goals |
| E4 | Rainbow DQN | 7 | 52 (50 LiDAR + goal distance and angle) | 300 | 0.5 s | Done, evaluated | 225 goals in training; 50/50 greedy eval |

---

## 2. Greedy evaluation results

Each row is one evaluation run. Success rate is over the episodes listed. Confidence intervals are Wilson 95%.

| Run | Source | Model | Episodes | Success | Collisions | Timeouts | Success % | 95% CI | Start pose | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | Graph `Dueling_dqn_eval_300ep.png` | Dueling DDQN | 30 | 30 | 0 | 0 | 100.0 | 88–100 | (0.2, 0.2, 0), per original README | Steps 20–24 |
| R2 | Graph `eval_300ep_DDQN_fixed_reward.png` | DDQN, fixed reward | 30 | 30 | 0 | 0 | 100.0 | 88–100 | Not stated | Steps 23–33 |
| R3 | Graph `eval_300ep_DDQN_fixed_reward(0.2,0.2).png` | DDQN, fixed reward | 30 | 28 | 2 | 0 | 93.3 | 79–98 | (0.2, 0.2, 0) | Held-out start |
| R4 | Graph `new_reward_pen_DDQN_22state_eval.png` | DDQN, new reward | 20 | 7 | 13 | 0 | 35.0 | 18–57 | Not stated | Steps 18–36 |
| R5 | Graph `eval_v2.png` | DQN (model not stated) | 20 | 20 | 0 | 0 | 100.0 | 84–100 | Not stated | Steps 21–23 |
| R6 | Graph `Evaluate.png` | DQN (model not stated) | 20 | 20 | 0 | 0 | 100.0 | 84–100 | Not stated | Steps 21–22 |
| R7 | Terminal `SUMMARY`, 10 Nov 2026 | Rainbow, final checkpoint | 50 | 50 | 0 | 0 | 100.0 | 93–100 | Held-out (0.2, 0.2, 0) | Avg reward 107.05, avg steps 47.84. Graph `Graphs/rainbow_0.5s_eval.png` |

**Reading these rows.** Every row is greedy. The fixed start means episodes are close to repeats, so a 100% row shows the policy works from that start, not that it generalises. R1, R3 and R7 use the held-out start (0.2, 0.2, 0), per the original README and the file names. The start of R2, R4, R5 and R6 is not stated in the sources.

---

## 3. Training results

| Run | Source | Episodes | Goal | Collision | No progress | Timeout | Goal % | Notes |
|---|---|---|---|---|---|---|---|---|
| E1 DDQN, fixed reward | Log | 300 | 134 | not logged | not logged | not logged | 44.7 | Log does not say whether this is training or evaluation |
| E3 Dueling DDQN | Log, graph `Dueling_dqn_train_300ep.png` | 300 | 170 | not logged | not logged | not logged | 56.7 | Reward flat at about −50 until episode 65, then jumps to about +108 with repeated collisions |
| E2 DDQN, new reward | Graph `new_reward_200ep_pen_DDQN_22state_train.png` | 200 | not read | not read | not read | not read | not read | Reward flat at about −25 until episode 55, then about +115 with repeated drops |
| E4 Rainbow DQN | Graph `Graphs/rainbow_0.5s_train.png` | 300 | ≈225 | ≈65 | ≈10 | 0 | ≈75 | Training uses NoisyNet exploration, and the first 100 episodes use a scripted guide. Not the greedy rate. Bar values read from the graph, so treat as approximate. |

---

## 4. Algorithm and architecture history

| Stage | Change | Network | Result | Source |
|---|---|---|---|---|
| E0 | DQN baseline | 7 → 128 → 128 → 5 | Repeated reverse actions; fixed by removing reverse | Log §1 |
| E0 | Hyperparameters | Adam lr 1e-3, batch 64, buffer 10k, γ 0.99, hard target update every 2 episodes, ε 1.0 → 0.05 per episode | Baseline | Log §1 |
| E1 | Network widened, 7 actions, batch 128 | Log says 7 → 256 → 256 → 7, not verified. Code `agent.py` uses a Dueling network with 128-wide features | Not verified | Log §3, discrepancy D1 |
| E1 | Double DQN target | Policy net picks the action, target net scores it | Implemented | Log §3.4 |
| E3 | Dueling architecture | Features 128 → 128, value head 128 → 64 → 1, advantage head 128 → 64 → actions | 56.7% training goals | Log §3, `agent.py` |
| E4 | Rainbow DQN | Features 52 → 256 → 256; value head 256 → 128 → 51; advantage head 256 → 128 → 7 × 51 | See section 3 | `agent_rainbow.py` |
| E4 | Rainbow components | Double DQN, Dueling, NoisyNet (σ₀ 0.5), C51 (51 atoms, −75 … 125), PER (α 0.6, β₀ 0.4), 3-step returns | Implemented | Code |

---

## 5. Environment and reward history

| Version | Actions | State | Goal reward | Collision | Per-step | Progress | Other |
|---|---|---|---|---|---|---|---|
| Initial | 5 | 7 | +100 | −30 (changed from −100) | −0.1 | +7 per m | Goal tolerance 0.20 m, collision threshold 0.20 → 0.08 m |
| Asymmetric | 7 | 22 | +100 | −30 | −0.1 | +7 per m forward, +3 per m penalty backward | Log §9 |
| Symmetric | 7 | 22 | +100 | −50 | −0.1 | +5 per m | −0.05 × \|angle\| heading term, proximity penalty below 0.30 m |
| Rainbow | 7 | 52 | +100 | −25 | −0.1 | +5 per m | −0.05 × \|angle\|, warning ramp below 0.30 m, 0.5 s steps, no-progress cut-off at 60 steps |

Tables above reflect the code as it stands. Earlier rows are from the log text.

---

## 6. Bug fixes and changes (chronological)

| # | Area | Problem | Fix | Status |
|---|---|---|---|---|
| 1 | DQN policy | Repeated reverse actions, episodes ending on timeout | Removed reverse action | Fixed |
| 2 | Reward | Robot repeatedly bumped corners | Progress reward raised from +5 to +7 | Fixed, later replaced by symmetric +5 |
| 3 | Reward order | Collision check ran before goal check, so reaching the goal inside the warning zone counted as a collision | Goal checked first | Fixed in code |
| 4 | Epsilon | Decayed per episode (×0.99) | Decay per step, ×0.9995 | Fixed |
| 5 | Rainbow | Timeout treated as terminal, cutting bootstrap | `truncated` flag, bootstrap kept | Fixed |
| 6 | Rainbow | `reset_noise()` inside no-grad (possible autograd error) | Removed | Unverified |
| 7 | Rainbow | Goal computed from odometry, which teleports do not update | World-frame pose tracking and `begin_episode()` | Fixed |
| 8 | Rainbow | Log printed the wrong state index | Fixed | Fixed |
| 9 | Rainbow | 1.0 s steps gave little warning before a wall | 0.5 s steps | Fixed |
| 10 | Rainbow | Stalled episodes never ended | No-progress cut-off after 60 steps | Fixed |
| 11 | Rainbow | Early warmup wandered near the start | Guided exploration, probability 0.8 → 0.1 over 100 episodes | Fixed |
| 12 | Rainbow | Sensor-wait loop exited after one spin | 0.1 s settle | Fixed |
| 13 | Rainbow | `warning_penalty` vs `warning_distance` naming, missing `min_progress`, wrong default model name | Renamed or added | Fixed |

---

## 7. Rainbow evaluation (detail)

| Item | Value |
|---|---|
| Checkpoint | `models/rainbow_dqn_52state_7actions.pth` (final), `learning_steps = 27,470` |
| Best-by-eval checkpoint | `models/rainbow_dqn_52state_7actions_best.pth`, `learning_steps = 8,704`, chosen on 3 greedy episodes, **not yet evaluated** |
| Start pose | Held-out (0.2, 0.2, 0) |
| Goal | (2, −2) |
| Step time | 0.5 s per action |
| Episode cap | 250 steps |
| Mode | Greedy (mean-weight) actions |
| Episodes | 50 (changed from 30 to narrow the interval) |
| Result | 50 goals, 0 collisions, 0 timeouts, 100%, avg reward 107.05, avg steps 47.84 |
| Interpretation | Every episode reached the goal, with steps close to the roughly 38-step straight-line minimum. The fixed start makes episodes near-repeats, so this is evidence from one start only. |
| Comparability | Same start as R1 and R3, but episode counts, seeds, and the environment and reward versions differ. The 100% rows cannot separate the algorithms. |

---

## 8. Discrepancies to resolve

| # | Where | Conflict | What to do |
|---|---|---|---|
| D1 | Log §3.1 vs `agent.py` | Log says the net is 256 wide. The Dueling net in `agent.py` uses 128. `QNetwork` (1024 wide) is defined but not used by the live code. | Confirm which network produced each checkpoint |
| D2 | Log §9 vs `Graphs/new_reward_200ep_pen_DDQN_22state_train.png` | Log says 120 episodes; the graph shows 200 | Check the training config used for the run |
| D3 | Log E1 and E3 metric blocks | It is not stated whether the goal counts are from training or evaluation | Check the source for each run |
| D4 | Rainbow change log | Says "untested in Gazebo", but training and evaluation have both run | Superseded by section 7 |
| D5 | Spawn pose | Training assumes the robot starts at (0, 0) after each reset. Not verified. | Check the spawn pose in the launch file |

---

## 9. Open items

| # | Item | Status |
|---|---|---|
| 1 | Evaluate best-by-eval Rainbow checkpoint (`_best.pth`) | Not started |
| 2 | Evaluate Rainbow at training start (0, 0, 0) for comparison with the DQN rows | Not started |
| 2a | Correct R1 start pose in this table (earlier draft said (0, 0, 0); corrected to (0.2, 0.2, 0) per original README) | Done |
| 3 | Verify spawn pose (D5) | Not started |
| 4 | Offline tests for C51 projection and n-step aggregation | Not started |
| 5 | Resolve D1, D2, D3 | Not started |
| 6 | DDPG: continuous velocity actions, actor and critic | Not started |

# Paper 1 hyperparameters (item 5.9)

These settings are **shared, not tuned per method**. Every run also records its exact values in `run_meta.json` under `hyperparameters`.

## Environment and protocol (all methods)
| Setting | Value |
|---|---|
| Robot / world | TurtleBot3 Burger, Gazebo Classic, stage-4 DRL world (2 moving cylinders) |
| State | 52: 50 LiDAR sectors (min range per 7.2° sector, ÷3.5 m) + goal distance (÷6 m) + goal angle (÷π) |
| Actions | 7 discrete (linear, angular): (0.15,0), (0.11,±0.4), (0.08,±0.8), (0.02,±1.0), each held for 0.5 s |
| Reward | +100 goal (≤0.20 m); −25 collision (≤0.15 m); −0.1 per step; +5·Δdistance; −0.05·\|goal angle\|; proximity ramp below 0.30 m |
| Episode end | goal / collision (terminal); no progress for 60 steps or 250 steps (truncation, bootstrapped) |
| Training | 200 episodes, start (0,0,0), goal (2,−2), 2 gradient updates per environment step |
| Seeds | 1, 2, 3; `set_seed` runs before the agent is built |
| In-training check | every 20 episodes: 5 greedy episodes from (0,0,0) |
| Final evaluation | 50 greedy episodes, 5 poses × 10, 250-step cap |

## Learning settings
| Setting | DQN | DDQN | D3QN | Rainbow |
|---|---|---|---|---|
| Network trunk | 52→256→256 | same | same | same |
| Head | 256→128→7 | same | V: 256→128→1, A: 256→128→7 | Noisy V: 256→128→51, Noisy A: 256→128→7×51 |
| Target | max Q_target | online argmax, target value | as DDQN | distributional C51 with a Double DQN target |
| Returns | 1-step | 1-step | 1-step | 3-step |
| Replay | uniform 50k | uniform 50k | uniform 50k | prioritized 50k (α 0.6; β starts at 0.4 and is annealed over 1M samples, so it only reaches ≈0.46 in a 200-episode run) |
| Exploration | ε 1.0→0.05 linear over 10k env steps | same | same | NoisyNet (σ₀ 0.5) |
| Loss | Huber | Huber | Huber | cross-entropy on C51 |
| Optimiser | Adam 1e-4 | same | same | same |
| Batch / γ | 128 / 0.99 | same | same | same |
| Target update | hard, every 250 learning steps | same | same | same |
| Warm-up | 1,000 transitions | same | same | same |
| Gradient clip | 10 | 10 | 10 | 10 |
| C51 support | – | – | – | 51 atoms on [−75, 125] |

## Guide (`*_g` configs)
- **Schedule:** the probability of a guide step falls linearly from 0.8 to 0.1 over episodes 1–100, then stays at 0.1.
- **Guide actions:** a guide step uses the obstacle-aware goal-seeking heuristic. With probability 0.15 it uses a forward-biased random action instead (weights 4:1:1:1:1:1:1).
- **What gets stored:** all transitions go into replay, including guide steps.
- **Greedy checks and final evaluation:** never use the guide.

## Design notes (from the Opus review)
- **ε and the guide:** in guided runs (`*_g`), ε decays with **all** environment steps, guide steps included. So the schedule is the same in time for every config, and a guided D3QN does less of its own ε-exploration early on. The `epsilon` column is the value at the end of each episode.
- **Paired seeds:** the same seed gives DQN and DDQN identical initial weights and identical Python random streams (common random numbers). A paired comparison is valid for those two.
- **Overestimation metric:** max Q(s0) minus the discounted return, averaged only over greedy check episodes that ended in a goal or a collision (`overestimation_n` says how many). Truncated episodes have no Monte Carlo return. `q_probe_mean` is the mean max-Q over 300 fixed probe states.
- **Rainbow's untrained C51 head** predicts about 25 (the middle of [−75, 125]), so its early Q values are not comparable. Compare the later checks.
- **The robot keeps moving during `learn()`:** Rainbow's update takes about 3 ms longer per step than DQN's (measured offline), roughly 1% of a 0.5 s action. Negligible, but not zero.

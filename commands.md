# Commands

Every command assumes the repo is at `~/turtle-bot-research` (branch `main`). Run each block in a fresh terminal unless it says otherwise.

- **ROS 2 package:** `dqn_comparison`.
- **Main study:** the DQN comparison (DQN, DDQN, D3QN, Rainbow, with and without the guide).
- **Paper 2:** the guided ablation, parked; it's in the appendix.

Arms of the DQN comparison:

| Arm | Algorithm | Guide |
|---|---|---|
| `dqn` | DQN | off |
| `ddqn` | Double DQN | off |
| `d3qn` | Dueling Double DQN | off |
| `d3qn_g` | Dueling Double DQN | on |
| `rainbow` | Rainbow DQN | off |
| `rainbow_g` | Rainbow DQN | on |

## 0. One-time setup (once per machine)

The repo root is the colcon workspace: `dqn_comparison`, `turtlebot3_gazebo`, `turtlebot3_msgs` and the rest. `build/`, `install/` and `log/` are git-ignored.

Rebuild only after changing a package.xml, setup.py, launch file or world. Python edits take effect without a rebuild, thanks to `--symlink-install`.
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && colcon build --symlink-install --allow-overriding turtlebot3_msgs
```
Never source `~/ros2_ws/install/setup.bash` in the same terminal: it has packages with the same names, and whichever is sourced last wins.

Python packages, installed into the PyTorch venv:
```bash
source ~/pytorch-env/bin/activate && pip install -r ~/turtle-bot-research/requirements.txt
```

## 1. Environment for any python command (each new terminal)
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
```

## 2. Gazebo with the GUI (single runs, final evaluation, checks)
Keep this terminal open while you use it.
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```
Add `gui:=false` for no window.

## 3. Checks (with the section 2 Gazebo running, in a section 1 terminal)
The robot spawns at (0, 0, 0):
```bash
ros2 service call /get_entity_state gazebo_msgs/srv/GetEntityState "{name: turtlebot3_burger, reference_frame: world}"
```
The reset services exist:
```bash
ros2 service list | grep -E "reset_simulation|reset_world|set_entity_state"
```

## 4. Unit tests (no Gazebo)
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/test_agents.py -q
```

## 5. 3-slot smoke test (about 10 min)
Close any GUI Gazebo first. Each slot (1–6) starts its own headless Gazebo, with ROS domain 30 + slot and Gazebo port 11345 + slot, so the slots can't interfere. Use one terminal per slot; a plain terminal is fine, because the script sources everything itself.
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 dqn 1 --episodes 5 --eval-every 5 --run-name smoke_dqn --results-dir results_smoke
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 2 d3qn_g 1 --episodes 5 --eval-every 5 --run-name smoke_d3qn_g --results-dir results_smoke
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 3 rainbow_g 1 --episodes 5 --eval-every 5 --run-name smoke_rainbow_g --results-dir results_smoke
```
What to check:
- each `results_smoke/*/*_run_meta.json` says `"status": "completed"`;
- `rtf` (real-time factor) in `*_episodes.csv` stays at or above 0.95;
- `random_steps` > 0 for `dqn`;
- `guided_steps` > 0 for the `_g` arms.

## 6. Commit and tag before the real runs (pre-registration)
```bash
git add -A && git commit -m "DQN comparison: final code before study runs" && git tag comparison-prereg-v1 && git push && git push --tags
```

## 7. Study batches (6 arms × 5 seeds = 30 runs)
Batch 1 ran 3 at a time. From batch 2 on, run **6 at a time**: one seed of all 6 arms, each in its own terminal. Measured load is about 0.4 CPU core and 1.65 GB RAM per run, so 6 fit. Output goes to `results/<arm>_s<seed>/`.

| Batch | Parallel | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 |
|---|---|---|---|---|---|---|---|
| 1 (done/running) | 3 | dqn 1 | ddqn 1 | d3qn 1 | – | – | – |
| 2 | 6 | dqn 2 | ddqn 2 | d3qn 2 | d3qn_g 2 | rainbow 2 | rainbow_g 2 |
| 3 | 6 | rainbow_g 3 | rainbow 3 | d3qn_g 3 | d3qn 3 | ddqn 3 | dqn 3 |
| 4 | 3 | d3qn_g 1 | rainbow 1 | rainbow_g 1 | – | – | – |
| 5 | 6 | ddqn 4 | dqn 4 | rainbow_g 4 | rainbow 4 | d3qn_g 4 | d3qn 4 |
| 6 | 6 | d3qn_g 5 | rainbow_g 5 | dqn 5 | ddqn 5 | d3qn 5 | rainbow 5 |

Seeds 4–5 (batches 5–6) were added by the pre-registration amendment `comparison-prereg-v1.1` (2026-11-16), before any final result was seen. 5 seeds per arm, 30 runs in total.

Batch 2, all six commands (one terminal each):
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=6 scripts/run_slot.sh 1 dqn 2
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=6 scripts/run_slot.sh 2 ddqn 2
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=6 scripts/run_slot.sh 3 d3qn 2
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=6 scripts/run_slot.sh 4 d3qn_g 2
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=6 scripts/run_slot.sh 5 rainbow 2
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=6 scripts/run_slot.sh 6 rainbow_g 2
```
About 10 minutes into a 6-run batch, check:
- `rtf` ≥ 0.95 in every `*_episodes.csv`;
- free RAM > 2 GB (`free -g`).

If either fails, stop slots 4–6 (Ctrl+C) and run them later. Interrupted runs keep their folders; re-run them with `--run-name <arm>_s<seed>_rerun`.

**Unattended, recommended: slot pool** (`scripts/run_pool.sh`, in use since 2026-11-16 15:02). It keeps up to 6 runs going and starts the **next pre-registered run as soon as any slot frees up** and RAM allows, so a slow run (e.g. unguided Rainbow) doesn't leave 5 slots idle. It blocks suspend, idle sleep and the lid switch, skips runs that already have results, and leaves hand-started runs alone. Start it detached:
```bash
cd ~/turtle-bot-research && setsid nohup scripts/run_pool.sh > log/queue/pool_nohup.out 2>&1 < /dev/null &
```
Progress: `tail -f ~/turtle-bot-research/log/queue/pool.log`. Preview: `DRY_RUN=1 scripts/run_pool.sh`. **Stop it** with `pkill -f "bash scripts/run_pool.sh"`. (Careful: a bare `pkill -f run_pool.sh` inside a longer shell command also kills that shell.)

**Older alternative, batch by batch** (`scripts/run_queue.sh`) It blocks suspend, idle sleep and the lid switch while it runs. It waits for any running batch to finish, checks there's enough free RAM, then runs batches 3–6 one after another, skipping runs whose folder already exists. Start it detached, so it survives closing the terminal:
```bash
cd ~/turtle-bot-research && setsid nohup scripts/run_queue.sh > log/queue/nohup.out 2>&1 < /dev/null &
```
Follow progress, and confirm the sleep block is active:
```bash
tail -f ~/turtle-bot-research/log/queue/queue.log
```
```bash
systemd-inhibit --list | grep dqn_comparison
```
Preview without starting anything: `DRY_RUN=1 scripts/run_queue.sh`. Stop everything: `pkill -f run_queue.sh; pkill -f run_slot.sh`.

If a run crashes, keep its folder (it's data, so note it in `reports/logs.md`) and re-run it under a new name:
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 dqn 1 --run-name dqn_s1_rerun
```
Single run without slots (section 2 Gazebo plus a section 1 terminal):
```bash
python3 -m dqn_comparison.train --arm d3qn --seed 1
```

## 8. Final evaluation (section 2 Gazebo plus a section 1 terminal)
All 30 runs, 50 greedy episodes each (5 poses × 10):
```bash
for S in 1 2 3 4 5; do for A in dqn ddqn d3qn d3qn_g rainbow rainbow_g; do R=${A}_s${S}; python3 -m dqn_comparison.evaluate --model results/$R/${R}_model_final.pth --episodes 50 --label $R --out-csv results/$R/${R}_final_eval.csv --plot results/$R/${R}_final_eval.png; done; done
```
Heuristic baselines:
```bash
mkdir -p results/H && python3 -m dqn_comparison.evaluate --policy heuristic --episodes 50 --label H --out-csv results/H/H_final_eval.csv
```
```bash
mkdir -p results/H_eta && python3 -m dqn_comparison.evaluate --policy heuristic --guide-noise 0.15 --seed 1 --episodes 50 --label H_eta --out-csv results/H_eta/H_eta_final_eval.csv
```

## 9. Analysis (tables and figures in `results/summary/`)
```bash
python3 -m dqn_comparison.analyze --results-dir results
```

## 10. Monitoring and troubleshooting
Last 5 episodes of a running run:
```bash
tail -n 5 ~/turtle-bot-research/results/dqn_s1/dqn_s1_episodes.csv
```
GPU and CPU load:
```bash
nvidia-smi
```
```bash
htop
```
Stop leftover processes before relaunching:
```bash
pkill -f gzserver; pkill -f gzclient; pkill -f dqn_comparison; pkill -f run_slot.sh
```

## 11. Backups and git
After each training day, back up everything (the `.pth` files are git-ignored):
```bash
rsync -a ~/turtle-bot-research/results/ ~/rainbow_backups/results_$(date +%F)/
```
Commit the CSVs, figures and metadata (the models stay local):
```bash
git add results && git commit -m "Add study runs $(date +%F)" && git push
```

---

# Appendix: Paper 2, guided ablation of Rainbow (parked)

Arms:
- `G`: guide on.
- `N`: no guide.
- `R`: random actions at the guide's rate.

Code is `dqn_comparison/dqn_comparison/guided_ablation/train.py`. Output goes to `results_ablation/<arm>_s<seed>/`.

Parallel slot:
```bash
cd ~/turtle-bot-research && TRAIN_MODULE=dqn_comparison.guided_ablation.train RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 N 1
```
Single run (section 2 Gazebo plus a section 1 terminal):
```bash
python3 -m dqn_comparison.guided_ablation.train --arm G --seed 1
```
Evaluate a run:
```bash
python3 -m dqn_comparison.evaluate --model results_ablation/G_s1/G_s1_model_final.pth --episodes 50 --label G_s1 --out-csv results_ablation/G_s1/G_s1_final_eval.csv
```
Plan and design: `reports/experiment_plan.md` and `reports/logs.md`.

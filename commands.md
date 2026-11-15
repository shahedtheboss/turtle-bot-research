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
Close any GUI Gazebo first. Each slot starts its own headless Gazebo, with ROS domain 30 + slot and Gazebo port 11345 + slot, so the slots can't interfere. Use one terminal per slot; a plain terminal is fine, because the script sources everything itself.
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

## 7. Study batches (6 arms × 3 seeds = 18 runs = 6 batches of 3, about 2 h each)
Run each batch's 3 commands in 3 terminals. Output goes to `results/<arm>_s<seed>/`.

| Batch | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|
| 1 | dqn 1 | ddqn 1 | d3qn 1 |
| 2 | d3qn_g 1 | rainbow 1 | rainbow_g 1 |
| 3 | ddqn 2 | d3qn 2 | dqn 2 |
| 4 | rainbow 2 | rainbow_g 2 | d3qn_g 2 |
| 5 | d3qn 3 | dqn 3 | ddqn 3 |
| 6 | rainbow_g 3 | d3qn_g 3 | rainbow 3 |

Example (batch 1, slot 1):
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 dqn 1
```
If a run crashes, keep its folder (it's data, so note it in `reports/logs.md`) and re-run it under a new name:
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 dqn 1 --run-name dqn_s1_rerun
```
Single run without slots (section 2 Gazebo plus a section 1 terminal):
```bash
python3 -m dqn_comparison.train --arm d3qn --seed 1
```

## 8. Final evaluation (section 2 Gazebo plus a section 1 terminal)
All 18 runs, 50 greedy episodes each (5 poses × 10):
```bash
for R in dqn_s1 ddqn_s1 d3qn_s1 d3qn_g_s1 rainbow_s1 rainbow_g_s1 dqn_s2 ddqn_s2 d3qn_s2 d3qn_g_s2 rainbow_s2 rainbow_g_s2 dqn_s3 ddqn_s3 d3qn_s3 d3qn_g_s3 rainbow_s3 rainbow_g_s3; do python3 -m dqn_comparison.evaluate --model results/$R/${R}_model_final.pth --episodes 50 --label $R --out-csv results/$R/${R}_final_eval.csv --plot results/$R/${R}_final_eval.png; done
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

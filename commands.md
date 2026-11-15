# Commands

Every command assumes the repo is at `~/turtle-bot-research`. Run each block in a fresh terminal unless it says otherwise.

## 0. One-time setup (only once per machine)

The repo root is the colcon workspace. It holds rainbow_ablation, turtlebot3_gazebo, turtlebot3_msgs and the rest. Build in place; `build/`, `install/` and `log/` are git-ignored. Rebuild only when you change a package.xml, setup.py, a launch file or a world. Python edits take effect without a rebuild, thanks to `--symlink-install`.
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && colcon build --symlink-install
```
Never source `~/ros2_ws/install/setup.bash` in the same terminal. It has packages with the same names, and whichever is sourced last wins.
Python packages, installed into the existing PyTorch venv:
```bash
source ~/pytorch-env/bin/activate && pip install -r ~/turtle-bot-research/requirements.txt
```

## 1. Terminal 1: start Gazebo

Keep this terminal open for the whole run.
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && export TURTLEBOT3_MODEL=burger && ros2 launch turtlebot3_gazebo turtlebot3_drl_stage4.launch.py
```

## 2. Terminal 2: environment for training and evaluation

Run this once in every new terminal before any `python3 -m rainbow_ablation...` command.
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
```

## 3. Checks before the study

Check that the robot spawns at (0, 0, 0). The pose should be close to 0, 0, 0:
```bash
ros2 topic echo /odom --once --field pose.pose
```
Check the Gazebo reset service exists:
```bash
ros2 service list | grep -E "reset_simulation|reset_world|set_entity_state"
```
Smoke test, one per arm. These write to `results_smoke/` and never mix with real runs. Delete the folder afterwards.
```bash
python3 -m rainbow_ablation.train --arm N --episodes 5 --eval-every 5 --run-name smoke_N --results-dir results_smoke
```
```bash
python3 -m rainbow_ablation.train --arm G --episodes 5 --eval-every 5 --run-name smoke_G --results-dir results_smoke
```
What to check in each smoke folder:
- In `*_episodes.csv`, `guided_steps` is all 0 for N and mostly non-zero for G.
- `*_evals.csv` has 1 row.
- `*_training.png` opens.
- `*_run_meta.json` says `"status": "completed"`.

## 4. Before the first real run: commit and tag

The runs record the git commit. Commit everything first, so `git_dirty` is `false`.
```bash
git add -A && git commit -m "Pre-registration: final code and plan before study runs"
```
```bash
git tag prereg-v1 && git push && git push --tags
```

## 5. Training runs (Stage 1: 5 seeds x 3 arms, in parallel slots)

Each run gets its own **slot**: its own headless Gazebo, its own ROS domain (30 + slot) and its own Gazebo port (11345 + slot). Runs in different slots can't see each other. You don't need Terminal 1's Gazebo for this; close it first to free RAM.

Use one terminal per slot. `run_slot.sh` sources everything itself, so a plain terminal is enough. Set `RAINBOW_PARALLEL` to how many slots run at once; it's recorded in `run_meta.json`.

**Step 5a: 2-slot smoke test (do once, about 10 min).** Both must finish, and `rtf` in the CSVs should stay at or above 0.95.
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=2 scripts/run_slot.sh 1 N 1 --episodes 5 --eval-every 5 --run-name par_N_slot1 --results-dir results_smoke
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=2 scripts/run_slot.sh 2 R 1 --episodes 5 --eval-every 5 --run-name par_R_slot2 --results-dir results_smoke
```
Check the real-time factor (column `rtf`):
```bash
cut -d, -f1,15,16 results_smoke/par_N_slot1/par_N_slot1_episodes.csv results_smoke/par_R_slot2/par_R_slot2_episodes.csv
```
If you want 3 slots, repeat with a third slot (`3 G 1`) and `RAINBOW_PARALLEL=3`, and check that `rtf` stays at or above 0.95. If it doesn't, use 2 slots.

**Step 5b: study batches.** One batch is one seed of every arm, run at the same time. The slots rotate each batch, so no arm always gets the same slot. With 3 slots, run each batch's three commands in three terminals:

| Batch | Slot 1 | Slot 2 | Slot 3 |
|---|---|---|---|
| 1 | N 1 | G 1 | R 1 |
| 2 | G 2 | R 2 | N 2 |
| 3 | R 3 | N 3 | G 3 |
| 4 | N 4 | G 4 | R 4 |
| 5 | G 5 | R 5 | N 5 |

Batch 1, for example:
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 N 1
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 2 G 1
```
```bash
cd ~/turtle-bot-research && RAINBOW_PARALLEL=3 scripts/run_slot.sh 3 R 1
```
With 2 slots, keep the same order but run 2 at a time: N1+G1, then R1+G2, then R2+N2, and so on.

Each run writes to `results/<arm>_s<seed>/`, including `<run>_gazebo.log`. A batch takes as long as its slowest run, usually N at about 1.5 to 2 h.

If a run crashes, keep the folder: it's data, so note it in `reports/logs.md`. Re-run it under a new name:
```bash
cd ~/turtle-bot-research && scripts/run_slot.sh 1 N 1 --run-name N_s1_rerun
```
Single run without slots (uses Terminal 1's Gazebo and terminal 2's environment):
```bash
python3 -m rainbow_ablation.train --arm N --seed 1
```

## 6. Final evaluation (after training)

Greedy test of one finished run. Change `N_s1` to the run you want:
```bash
R=N_s1 && python3 -m rainbow_ablation.evaluate --model results/$R/${R}_model_final.pth --episodes 50 --label $R --out-csv results/$R/${R}_final_eval.csv --plot results/$R/${R}_final_eval.png
```
Evaluate all 15 runs in one go:
```bash
for R in N_s1 G_s1 R_s1 N_s2 G_s2 R_s2 N_s3 G_s3 R_s3 N_s4 G_s4 R_s4 N_s5 G_s5 R_s5; do python3 -m rainbow_ablation.evaluate --model results/$R/${R}_model_final.pth --episodes 50 --label $R --out-csv results/$R/${R}_final_eval.csv --plot results/$R/${R}_final_eval.png; done
```
Heuristic baselines (H and H_eta):
```bash
mkdir -p results/H && python3 -m rainbow_ablation.evaluate --policy heuristic --episodes 50 --label H --out-csv results/H/H_final_eval.csv --plot results/H/H_final_eval.png
```
```bash
mkdir -p results/H_eta && python3 -m rainbow_ablation.evaluate --policy heuristic --guide-noise 0.15 --seed 1 --episodes 50 --label H_eta --out-csv results/H_eta/H_eta_final_eval.csv --plot results/H_eta/H_eta_final_eval.png
```

## 7. Monitoring and troubleshooting

Follow a running run's progress: it shows the last 5 episodes.
```bash
tail -n 5 ~/turtle-bot-research/results/N_s1/N_s1_episodes.csv
```
Check Gazebo's real-time factor. It should stay near 1.0:
```bash
source /opt/ros/humble/setup.bash && gz stats
```
GPU or CPU load:
```bash
nvidia-smi
```
```bash
htop
```
Gazebo frozen or leftover processes from an old session. Stop them before relaunching:
```bash
pkill -f gzserver; pkill -f gzclient; pkill -f "rainbow_ablation.train"; pkill -f run_slot.sh
```

## 8. Backups and git

Back up the results after each training day. The `.pth` files are git-ignored, so this is their only copy.
```bash
rsync -a ~/turtle-bot-research/results/ ~/rainbow_backups/results_$(date +%F)/
```
Commit the CSVs, figures and metadata after each day. The models stay local.
```bash
git add results && git commit -m "Add study runs $(date +%F)" && git push
```
Check what has changed:
```bash
git status && git log --oneline -5
```

---

# Paper 1: DQN vs DDQN vs D3QN vs Rainbow (branch `paper1-comparison`)

Code: `rainbow_ablation/rainbow_ablation/paper1/`. Results: `results_paper1/<arm>_s<seed>/`. Settings: `paper1/HYPERPARAMETERS.md`.

Arms: `dqn`, `ddqn`, `d3qn`, `d3qn_g`, `rainbow`, `rainbow_g` (`_g` means the decaying guide is on).

## P0. Unit tests (no Gazebo)
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest rainbow_ablation/test/test_paper1_agents.py -q
```

## P1. 3-slot smoke test (about 10 min, close any GUI Gazebo first)
```bash
cd ~/turtle-bot-research && TRAIN_MODULE=rainbow_ablation.paper1.train RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 dqn 1 --episodes 5 --eval-every 5 --run-name smoke_dqn --results-dir results_smoke
```
```bash
cd ~/turtle-bot-research && TRAIN_MODULE=rainbow_ablation.paper1.train RAINBOW_PARALLEL=3 scripts/run_slot.sh 2 d3qn_g 1 --episodes 5 --eval-every 5 --run-name smoke_d3qn_g --results-dir results_smoke
```
```bash
cd ~/turtle-bot-research && TRAIN_MODULE=rainbow_ablation.paper1.train RAINBOW_PARALLEL=3 scripts/run_slot.sh 3 rainbow_g 1 --episodes 5 --eval-every 5 --run-name smoke_rainbow_g --results-dir results_smoke
```
What to check:
- each `*_run_meta.json` says `completed`;
- `rtf` in `*_episodes.csv` stays at or above 0.95;
- `random_steps` > 0 for `dqn`;
- `guided_steps` > 0 for the `_g` arms.

## P2. Commit and tag before the real runs (pre-registration)
```bash
git add -A && git commit -m "Paper 1: final code before study runs" && git tag paper1-prereg-v1
```

## P3. Study batches (6 configs × 3 seeds = 18 runs = 6 batches of 3, about 2 h each)
In each batch, run the 3 commands in 3 terminals, all with `TRAIN_MODULE=rainbow_ablation.paper1.train RAINBOW_PARALLEL=3`:

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
cd ~/turtle-bot-research && TRAIN_MODULE=rainbow_ablation.paper1.train RAINBOW_PARALLEL=3 scripts/run_slot.sh 1 dqn 1
```

## P4. Final evaluation (one Gazebo with the GUI, terminal 2 environment)
```bash
for R in dqn_s1 ddqn_s1 d3qn_s1 d3qn_g_s1 rainbow_s1 rainbow_g_s1 dqn_s2 ddqn_s2 d3qn_s2 d3qn_g_s2 rainbow_s2 rainbow_g_s2 dqn_s3 ddqn_s3 d3qn_s3 d3qn_g_s3 rainbow_s3 rainbow_g_s3; do python3 -m rainbow_ablation.evaluate --model results_paper1/$R/${R}_model_final.pth --episodes 50 --label $R --out-csv results_paper1/$R/${R}_final_eval.csv --plot results_paper1/$R/${R}_final_eval.png; done
```
Heuristic baselines:
```bash
mkdir -p results_paper1/H && python3 -m rainbow_ablation.evaluate --policy heuristic --episodes 50 --label H --out-csv results_paper1/H/H_final_eval.csv
```
```bash
mkdir -p results_paper1/H_eta && python3 -m rainbow_ablation.evaluate --policy heuristic --guide-noise 0.15 --seed 1 --episodes 50 --label H_eta --out-csv results_paper1/H_eta/H_eta_final_eval.csv
```

## P5. Analysis (tables and figures in `results_paper1/summary/`)
```bash
python3 -m rainbow_ablation.paper1.analyze --results-dir results_paper1
```

# Running study v2 on a second computer (e.g. a lab PC)

Goal: an identical software setup, so runs from both machines are comparable.
Reference machine (2026-11-19): Ubuntu 22.04.5, ROS 2 Humble, Gazebo 11.10.2, Python 3.10.12,
torch 2.14.1, numpy 2.2.6, matplotlib 3.10.9 (see `requirements.txt`).

## 0. Requirements
- **Ubuntu 22.04** (native, not a VM if possible: Gazebo is CPU-heavy). ROS 2 Humble does not support other versions.
- At least 8 CPU threads and 16 GB RAM for 4 parallel runs (fewer slots on smaller machines).
- An NVIDIA GPU is NOT needed (networks are tiny); CPU-only PyTorch is fine.
- Admin (sudo) rights for the installs.

## 1. ROS 2 Humble + Gazebo 11 (about 20-30 min)
```bash
sudo apt update && sudo apt install -y software-properties-common curl git python3-venv
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3 ros-humble-turtlebot3-msgs ros-humble-dynamixel-sdk python3-colcon-common-extensions lm-sensors
```
Check: `gzserver --version` should print 11.10.x.

## 2. Project + Python environment (about 10 min)
```bash
cd ~ && git clone https://github.com/shahedtheboss/turtle-bot-research.git
cd ~/turtle-bot-research && git switch main
python3 -m venv ~/pytorch-env && source ~/pytorch-env/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.2.6 matplotlib==3.10.9 pytest==9.1.1
source /opt/ros/humble/setup.bash && colcon build --symlink-install
```
Note: `results_v2/`, the recorded obstacle paths and all code come from git; nothing else must be copied.

## 3. Verify BEFORE any real run (about 30 min)
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/ -q --ignore=dqn_comparison/test/test_flake8.py --ignore=dqn_comparison/test/test_pep257.py
MODES="rt_wall x3_sim" REPEATS=10 TAG=_r10 V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005 scripts/run_sim_time_equivalence.sh
V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005 EVAL_REPEATS=1 scripts/run_slot_v2.sh 1 4 d3qn 9 --episodes 3 --eval-every 3 --results-dir results_smoke/second_machine --overwrite
```
Send back `results_smoke/b3/*_r10.csv` and the smoke folder: the end poses must match the reference machine's
(same mean within ~1 cm) before this machine runs study data.

## 4. Measure how many runs fit (about 30 min)
Run 4 slots, then 6, then 8, each with a short run, and keep the highest count where the per-run speed
(steps per minute, from each run's episodes CSV and run_meta.json times) stays close to the 4-slot value
and the CPU stays below ~95 C (`sensors`). Rough guide: about 2.3 CPU threads per run.

## 5. Running the main runs
The run list `scripts/main_runs_v2.csv` assigns whole seeds to machines (pre-registered): lab PC seeds 1-3
(75 runs), laptop seeds 4-5 (50 runs).
```bash
PARALLEL=6 scripts/run_main_v2.sh labpc                 # this machine's runs
DRY_RUN=1 scripts/run_main_v2.sh labpc                  # what is left, without running
```
The launcher can be stopped and restarted at any time: finished runs are skipped, interrupted ones start again.
Each run records the machine (hostname), git commit and untracked-code list in `run_meta.json`.
Copy finished run folders back (USB or `rsync -a results_v2/ <laptop>:~/turtle-bot-research/results_v2/`).

**If the lab PC is not available long enough (fallback):** copy its finished run folders into the laptop's
`results_v2/`, then on the laptop run `scripts/run_main_v2.sh laptop --take-over`: it does its own runs first,
then every remaining run of the list. The same works the other way round (`labpc --take-over`).

## Caveats
- Different CPUs give different simulation speeds; with simulated-time stepping this changes waiting time,
  not the task (B3 test). Machine is recorded per run and can be checked in the analysis.
- Keep both machines on the SAME git commit (`git rev-parse HEAD` must match).

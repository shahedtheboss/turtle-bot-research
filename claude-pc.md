# Lab PC handover: study v2 main runs (seeds 1–3)

This file has two parts:
- **Part A** is the context for a Claude Code session on the lab PC: give it this file first.
- **Part B** is the step-by-step guide for the person setting the PC up.

Written 2026-11-25 on the laptop. Repo commit: `37230b8`; pre-registration tag: `v2-prereg-1.3`.

---

## Part A — Context for Claude on the lab PC

### Who the user is and how to work with them
- The user is an engineering undergraduate learning ML, RL, robotics, ROS 2 and Gazebo, and doing research.
  **Follow `CLAUDE.md` in the repo root.**
  - Explain before implementing.
  - Plan nontrivial changes and wait for approval.
  - Never claim something works without checking it.
  - Report failures honestly.
- **Ask before any significant change.**
  - Never push unless the user asks.
  - Never commit `reports/` (it is kept local only).
  - Never edit a script that is currently running; change scripts only between runs.
- **Don't** enter passwords, change system or security settings, or approve your own permissions.
  The user does those themselves.
- Commit messages: the user's convention is **no** Co-Authored-By line.

### The study (v2, target journal: Robotica)
- **125 main runs:** 5 learning arms × 5 worlds × 5 seeds.
  - **Arms:** `dqn`, `ddqn`, `d3qn`, `rainbow`, `rainbow_g` (Rainbow with a heuristic guide).
  - **Worlds:** 2, 3, 4s, 4, 6.
- **Each run:**
  - 500 episodes, with no early stop;
  - a greedy check every 20 episodes;
  - then a final evaluation.
- **Main metric:** E\*, the first episode at which the greedy check reaches ≥3/4 goals, confirmed by the
  next check. Runs that never reach it are censored at 520.
- **Machine split (pre-registered, whole seeds per machine):**
  - **lab PC:** seeds 1–3, 75 runs;
  - **laptop:** seeds 4–5, 50 runs.
  - Assignments are in `scripts/main_runs_v2.csv`, in the `machine` column.
- **Pre-registration:** `docs/v2_preregistration.md` (tags `v2-prereg-1.0` … `1.3`), with Deviations 1–4.
  Read it before touching anything. Do **not** re-tune; the settings are fixed in `TUNED_EXTRA` in
  `dqn_comparison/dqn_comparison/train_v2.py`.
- **Heuristic baselines** (H and H_noisy) are already done and committed. Don't rerun them.

### Laptop status (as of 2026-11-25 19:40)
- 33 of 50 laptop runs were done and the rest were in progress, with zero failures.
- At about 4 runs in parallel, one run takes 1.25–2 h. Rainbow and worlds 4/6 are the slowest.

### Technical facts that matter on this PC
- **Software:** Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11, Python 3.10.
  - PyTorch **2.14.1+cu130** in `~/pytorch-env`; `scripts/run_slot_v2.sh` sources that exact path.
- **CUDA is required.** `run_slot_v2.sh` sets `DQN_DEVICE=cuda`, and the run fails if CUDA is missing,
  so no run can silently fall back to the CPU.
  - The lab PC needs an NVIDIA GPU, a working driver (≥580 for CUDA 13.0) and the CUDA PyTorch wheel.
  - **`docs/setup_second_machine.md` is out of date on this point:** it says CPU PyTorch is fine.
    It is not, since Deviation 1 (2026-11-21).
- **Simulated-time stepping:** physics is paused between actions; one step is 0.512 s of simulated time.
  - Machine speed changes only the waiting time, not the results.
  - `V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005` is fixed for every run; the launcher sets it.
- **Launcher:** `scripts/run_main_v2.sh labpc`.
  - **Restartable:** finished runs are skipped; interrupted ones start again from scratch.
  - **Parallel slots:** `PARALLEL=N`, default 4, max 8. Each slot is an isolated headless Gazebo.
  - **Logging:** one log per run in `log/slots/<run>.log`; each run's `run_meta.json` records the machine,
    git commit and device.
  - **Done** means `results_v2/<run>/<run>_run_meta.json` has `"status": "completed"` **and**
    `<run>_final_eval.csv` exists.
- **Known problems:**
  - Occasionally Gazebo hangs at start (GLX BadDrawable), and its `gzserver` blocks the slot.
    - **Symptom:** a slot never starts its next run, and its log stops early.
    - **Fix:** wait until no other run is starting, then stop that slot's `gzserver` and rerun.
  - The launcher writes "exit 0" even for a failed run, so always check the run's folder, not the log line.
- **Stopping without losing work:** kill only the launcher's `run_main_v2.sh` processes. Running
  `run_slot_v2.sh` runs then finish normally, and no new ones start. Resume by starting the launcher again.
- **Temperatures:** watch them with `sensors`. The CPU should stay below about 95 °C.

### What Claude should do on the lab PC, in order
1. Check the environment: `nvidia-smi`; `torch.cuda.is_available()` is True; `git rev-parse HEAD` matches
   the laptop's (`37230b8` or newer, if the user has pushed since).
2. Run the unit tests (expect 89 passed, 1 skipped).
3. Run the sim-time equivalence check and the smoke run (Part B, step 6). Check that the smoke run's
   `run_meta.json` shows device `cuda` and about 0.51 simulated seconds per step.
4. With the user's approval, choose `PARALLEL`. Use 4 if the CPU has fewer than about 14 threads; up to 6
   on a 16+ thread CPU if temperatures allow.
5. Start the main runs with `nohup`, so they survive an AnyDesk/SSH disconnect (Part B, step 7).
6. On "status": count finished runs (`DRY_RUN=1 scripts/run_main_v2.sh labpc`), show the running ones, check
   the number of `gzserver` processes, temperatures and memory, and look for failures.
7. At the end: copy `results_v2/` from the lab PC to the laptop (Part B, step 9). Don't commit or push
   results unless the user asks.

---

## Part B — Step-by-step guide (for you, at the lab PC)

### Step 0 — Check the lab PC is suitable (5 min)
In a terminal on the lab PC:
```bash
lsb_release -d        # must say Ubuntu 22.04
nproc                 # CPU threads: 8 or more is needed for 4 parallel runs
free -h               # 16 GB RAM or more
nvidia-smi            # must show an NVIDIA GPU and driver 580 or newer
echo $XDG_SESSION_TYPE   # x11 is best for AnyDesk; "wayland" means see step 1
```
If `nvidia-smi` fails or shows no NVIDIA GPU, **stop**: the runs need CUDA. Tell me or your professor.

### Step 1 — Remote access (AnyDesk)
1. Install AnyDesk on the lab PC from **anydesk.com** (the official Linux `.deb`), and on your laptop.
2. If step 0 printed `wayland`, switch the lab PC to Xorg; AnyDesk can't be controlled properly on Wayland.
   Do **one** of these:
   - **Simple:** log out, click the gear icon on the login screen, choose **Ubuntu on Xorg**, and log in.
   - **Permanent (needs sudo):** run `sudo nano /etc/gdm3/custom.conf`, remove the `#` in front of
     `WaylandEnable=false`, save, and reboot.
3. In AnyDesk on the lab PC, open Settings → Security and enable **unattended access** with a strong
   password (one only you know).
4. Settings → Power on the lab PC:
   - set **Automatic suspend: Off** and **Blank screen: Never**;
   - turn **off** automatic logout if your university set it;
   - if the PC belongs to the lab, ask before changing these.
5. Note the lab PC's AnyDesk ID. Test connecting from your laptop before you leave the lab.

### Step 2 — Install ROS 2 Humble and Gazebo 11 (20–30 min; skip if already installed)
Check first:
```bash
ls /opt/ros/humble && gzserver --version
```
If either is missing, install them:
```bash
sudo apt update && sudo apt install -y software-properties-common curl git python3-venv
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-gazebo-ros-pkgs ros-humble-turtlebot3 ros-humble-turtlebot3-msgs ros-humble-dynamixel-sdk python3-colcon-common-extensions lm-sensors
```
`gzserver --version` should then print 11.10.x.

### Step 3 — Get the code (5 min)
```bash
cd ~ && git clone https://github.com/shahedtheboss/turtle-bot-research.git
cd ~/turtle-bot-research && git switch main && git pull
git rev-parse --short HEAD     # must match the laptop (37230b8 or newer)
```
If the repo is already there, just `cd ~/turtle-bot-research && git pull`.

### Step 4 — Python environment with CUDA PyTorch (10 min)
```bash
python3 -m venv ~/pytorch-env
source ~/pytorch-env/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==2.2.6 matplotlib==3.10.9 pytest==9.1.1
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print 2.14.1+cu130 True
```
The venv **must** be at `~/pytorch-env`, because the run scripts look for it there.

### Step 5 — Build and run the unit tests (5 min)
```bash
cd ~/turtle-bot-research
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest dqn_comparison/test/ -q \
  --ignore=dqn_comparison/test/test_flake8.py --ignore=dqn_comparison/test/test_pep257.py
```
Expected: **89 passed, 1 skipped**.

### Step 6 — Verify the simulation before real runs (about 30 min)
```bash
cd ~/turtle-bot-research && source /opt/ros/humble/setup.bash && source install/setup.bash && source ~/pytorch-env/bin/activate
MODES="rt_wall x3_sim" REPEATS=10 TAG=_r10 V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005 scripts/run_sim_time_equivalence.sh
V2_CLOCK_HZ=200 DQN_SIM_POLL_S=0.005 EVAL_REPEATS=1 scripts/run_slot_v2.sh 1 4 d3qn 9 --episodes 3 \
  --eval-every 3 --results-dir results_smoke/second_machine --overwrite
```
Check:
- the smoke run finished without errors;
- `grep device results_smoke/second_machine/*/*run_meta.json` shows `cuda`;
- the end poses in `results_smoke/b3/*_r10.csv` match the laptop's (same mean within about 1 cm).
  Bring the files to the laptop, or ask Claude to compare them.

### Step 7 — Start the 75 lab-PC runs
```bash
cd ~/turtle-bot-research
DRY_RUN=1 scripts/run_main_v2.sh labpc | tail -1      # should say: 75 runs to do on labpc
mkdir -p log
nohup scripts/run_main_v2.sh labpc >> log/main_labpc.log 2>&1 &
```
- `nohup … &` keeps the runs going after you close the terminal or AnyDesk.
- For 6 parallel runs on a strong PC, put `PARALLEL=6` in front of `nohup`.
  Only do this if the CPU has 16 or more threads.
- **Expected time:** about 28–30 h with 4 parallel; less with 6.

After 2–3 minutes, check that it started:
```bash
tail -5 log/main_labpc.log        # 4 "start" lines
pgrep -c gzserver                 # 4 (or your PARALLEL value)
nvidia-smi                        # python processes using the GPU
```

### Step 8 — Check on the runs (any time, via AnyDesk)
```bash
cd ~/turtle-bot-research
DRY_RUN=1 scripts/run_main_v2.sh labpc | tail -1                 # runs left
tail -8 log/main_labpc.log                                       # latest starts and ends
ps -eo etime,args | grep -o "run-name [a-z0-9_]\+" | sort -u      # what is running
sensors | grep Package; free -h                                  # temperature and memory
```
**Pause safely** (running runs finish; nothing new starts):
```bash
pkill -f "run_main_v2.sh labpc"
```
**Resume:** repeat the `nohup …` line from step 7. Finished runs are skipped.

Don't reboot or shut down while `run_slot_v2.sh` runs are active. An interrupted run restarts from zero.

### Step 9 — When all 75 runs are done
Check that nothing is left:
```bash
DRY_RUN=1 scripts/run_main_v2.sh labpc | tail -1     # "0 runs to do"
```
Then copy the results to the laptop with a USB stick, by copying the `results_v2/w*_s1`, `w*_s2` and `w*_s3`
folders into the laptop's `~/turtle-bot-research/results_v2/`. Or, on the same network:
```bash
rsync -a ~/turtle-bot-research/results_v2/ <laptop-user>@<laptop-ip>:~/turtle-bot-research/results_v2/
```
Don't copy the folders whose names end in `_s4` or `_s5` over the laptop's own; rsync only adds or updates.

### If something goes wrong
- **A slot stops starting new runs, or `pgrep -c gzserver` shows more than `PARALLEL`:** Gazebo probably hung.
  Don't kill things at random: open Claude on the lab PC with this file and describe what you see.
- **A run's folder has no `_final_eval.csv` after its "end" line:** that run failed. Restarting the launcher
  reruns it.
- **The CPU stays above about 95 °C:** pause (step 8), and use `PARALLEL=3` or improve the airflow.
- **The lab PC is unavailable for a long time:** the pre-registered fallback is to copy its finished folders to
  the laptop and run `scripts/run_main_v2.sh laptop --take-over` there. Decide this with Claude, not alone.

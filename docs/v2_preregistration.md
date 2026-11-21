# Study v2: pre-registration (frozen by git tag `v2-prereg-1.0`, 2026-11-21, before any main run)

Everything below is fixed before the first main run. Changes after the tag are reported as deviations.

## Design
- **Arms (5):** DQN, DDQN, D3QN, Rainbow, Rainbow + decaying scripted guide (`rainbow_g`).
- **Worlds (5):** DRL stages 2, 3, 4s (stage 4 with static obstacles), 4 (moving obstacles), 6.
- **Seeds:** 1-5 for every arm x world, so 125 runs (`scripts/main_runs_v2.csv`). Tuning seeds 101-102 are never reported as main results.
- **Task:** 4 goals per world, drawn in shuffled blocks from a per-seed RNG (identical goal sequence for every arm with the same seed); same obstacle phase every episode; 52-value state, 7 actions.
- **Budget:** 500 training episodes per run, no early stopping; greedy check (one episode per goal) every 20 episodes.
- **Hyperparameters:** shared v1 settings, except the learning rate tuned per algorithm (`docs/v2_tuning_protocol.md`): DQN 2.5e-4, DDQN 2.5e-4, D3QN 1e-4, Rainbow 1e-4 (also `rainbow_g`).
- **Simulation:** simulated-time stepping, real-time factor cap 3, /clock 200 Hz, 5 ms poll.

## Metrics
- **Primary:** E* = first greedy check with >= 3 of 4 goals reached, confirmed by the next check; runs that never meet it are censored at 520.
- **Secondary:** final success rate (final checkpoint, 5 poses x 4 goals x 2 repeats = 40 episodes); collision rate in that evaluation.

## Analysis plan (statistics code: `stats.py`, `metrics_v2.py`)
- Per world: median and IQM of E* per arm over the 5 seeds; final success with Wilson intervals.
- Comparisons: pairwise exact permutation tests on E* between arms within each world (paired by seed, since seeds share goal sequences), Holm correction over the comparisons of a world; effect size Vargha-Delaney A.
- Across worlds: arm ranks per world and the median of each arm's E* over worlds; no pooled p-value across worlds.
- Censored runs enter the tests with the censoring value 520 (rank-based effect sizes are unaffected by its exact value).
- With 5 seeds the tests have low power; non-significant differences are reported as inconclusive, not as equivalence.

## Machines
- Planned split: lab PC seeds 1-3 (75 runs), laptop seeds 4-5 (50 runs). Whole seeds per machine, so each arm and world is equally affected by machine differences.
- Fallback: if a machine is not available long enough, the other runs the remaining runs (`scripts/run_main_v2.sh <machine> --take-over`). The machine of every run is recorded (hostname in `run_meta.json`) and reported; a machine effect is checked descriptively.
- A run that fails for a technical reason (crash, Gazebo not starting) is rerun with the same settings and reported.

## Deviation 1 (2026-11-21): simulated-time stepping bug fixed; tuning redone; main runs restarted
**Found** in the first 4 main runs (laptop, world 2, seed 4) about 1.5 h after the tag: the environment read the
simulated clock only while processing messages, so after each learning update the next action started from a
stale time and ended early by (learning wall time x speed). When learning took longer than ~0.17 s wall time the
actions shrank to a few milliseconds and the robot stopped moving (DDQN, D3QN, Rainbow, Rainbow+guide ended
episodes within 3-10 cm of the start; 1.7-4.8 s of simulated time per step instead of 0.5). In the earlier tuning
and diagnosis runs the same bug shortened actions by a load- and algorithm-dependent amount (not measurable
afterwards).
**Fix:** Gazebo physics is paused between actions (simulated time advances only while an action runs) and the
clock is refreshed before each action. Checks: a unit test that fails on the old code and passes on the new;
in Gazebo, 30 fixed actions end at (1.062, -0.200) with 0.5 s of artificial learning delay per step vs
(1.066, -0.195) for the v1 real-time reference; training at 4 parallel runs gives 0.51 s of simulated time per
step for every algorithm.
**Consequences:** the 4 main runs are discarded (results_invalid_v2_stale_clock/, not analysed); the tuning is
rerun with the same protocol on the fixed code; the main runs restart from scratch after a new tag
(v2-prereg-1.1). No main-study result had been looked at beyond these 4 invalid runs.
(The tuning rerun planned here was replaced by deviation 2.)

## Deviation 2 (2026-11-21): learning rates kept from the first tuning; learning device fixed to CUDA
**Cause found:** at 12:37:49 the NVIDIA driver was changed (595 -> 580-open, `apt install`), which broke CUDA until
a reboot. The code chose CUDA "if available, else CPU" without warning, so the first tuning ran on the GPU and the
4 discarded main runs (except DQN, started 13 s before the change) and the tuning rerun on the CPU. CPU learning with
4 parallel runs was slow, which is why the timing bug (deviation 1) stopped the robot in the main runs but only
shortened actions slightly in the first tuning (GPU learning is fast; the shortening cannot be measured afterwards).
**Decision (author):** the tuning rerun was stopped after ~1 h (partial, CPU; results_tune_v2_rerun_partial_cpu/,
not used) and the learning rates selected by the FIRST tuning (results_tune_v2_stale_clock/) are kept:
DQN 2.5e-4, DDQN 2.5e-4, D3QN 1e-4, Rainbow 1e-4. Limitation, to be reported: that tuning ran on the pre-fix code,
where actions were shortened by the (short, GPU) learning time, so the rates were chosen on a slightly different
task; in it, 1e-4 and 2.5e-4 were within seed-to-seed noise for every algorithm.
**Device:** all main runs use CUDA, required (`DQN_DEVICE=cuda` in run_slot_v2.sh: a run fails instead of falling
back to the CPU); the device is recorded in run_meta.json. Laptop and lab PC both run CUDA.
